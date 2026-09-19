"""Label test for the hitting-time-CDF proposal: is *offline-supported graph hitting time*
T_G worth learning at all? Scores it -- with no model trained -- against the privileged
recovery-time oracle T of the Experiment-2 audit (outputs/pusht/experiments/viability_exp2/raw_offset{N}.npz),
with exactly the metrics viability_eval_audit.py reports for the critic, so the three columns
line up: latent L2 (LeWM) | trained critic V | graph label T_G.

Graph: every cached frame is a node. Transition edges z_t -> z_{t+1} are DIRECTED with cost
1 env step (finer than the proposal's 5-step blocks, strictly more informative; block
distance is just ceil(T_G/5)). Optional identification edges are the repo's validated
k-capped FAISS construction (common.graph_lib.build_identification_edges, k=4, symmetric,
cost id_weight=1 -- the id_weight=0 design was found wrong in this repo). The audit tasks'
own episodes are encoded and added to the node set, otherwise a temporal-only graph could
never reach any goal (goals are frames of those episodes).

For each (task, candidate) the query latent (z_true = encoded executed endpoint, or z_pred =
what the planner scores) is snapped to its nearest graph node(s) and T_G = directed shortest
path to the task's goal node. Disconnected = a large finite value (rank metrics only).

Reports, per offset / graph variant / query input:
    spearman(T_G, T) per task, mean            (== eval_audit's spearman(V, -T))
    selection regret  T[argmin T_G] - min T    (vs L2's and V's)
    oracle-best percentile under T_G           (0 = ranked first)
    auroc of -T_G vs 1[T <= h_natural]
    disconnected fraction, NN distance stats, slot-0 (expert continuation) sanity check

Run:
    python scripts/viability_graph_label_test.py --cache outputs/pusht/critic_training/cache_1000_s0.npz \
        --audit outputs/pusht/experiments/viability_exp2 --goal-offsets 25 50
"""

import argparse
import json
import sys
import time
from pathlib import Path

import hdf5plugin  # noqa: F401
import h5py
import numpy as np
from scipy import stats as sps
from scipy.sparse.csgraph import dijkstra

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.envs import ENV_MECHANICS
from common.graph_lib import (build_identification_edges, build_weighted_graph,
                              find_transition_edges)
from common.lewm_loader import load_lewm
from common.log_util import log
from planning_cost_gate import DEV, make_encode_frame

DISCONNECTED = 10_000.0


# ------------------------------------------------------------------
# nodes: cache episodes + the audit tasks' own episodes
# ------------------------------------------------------------------

def load_audits(audit_dir, offsets):
    out = {}
    for off in offsets:
        p = Path(audit_dir) / f"raw_offset{off}.npz"
        if p.exists():
            out[off] = dict(np.load(p))
        else:
            log(f"[skip] {p} not found")
    return out


def build_nodes(cache_path, audits, root, extra_cache):
    """Returns z (N,D), ep_idx (h5 episode id per node), step_idx, and row2node: dict
    h5_episode -> node offset of its first frame (for mapping goal_row to a node)."""
    c = np.load(cache_path)
    mech = ENV_MECHANICS["pusht"]
    with h5py.File(mech.h5_path(root), "r", swmr=True) as f:
        h5_off = f["ep_offset"][:].astype(np.int64)
        h5_len = f["ep_len"][:].astype(np.int64)
    have = set(int(e) for e in c["episode_id"])
    need = set()
    for d in audits.values():
        for r in d["start_row"]:
            need.add(int(np.searchsorted(h5_off, int(r), side="right") - 1))
    missing = sorted(need - have)
    log(f"[nodes] cache: {len(have)} episodes / {len(c['z'])} frames; audit tasks touch "
        f"{len(need)} episodes, {len(missing)} not in cache -> encoding them")

    z_parts, ep_parts, step_parts = [c["z"]], [], []
    ep_parts.append(np.repeat(c["episode_id"].astype(np.int64), c["ep_len"]))
    step_parts.append(np.concatenate([np.arange(l) for l in c["ep_len"]]))
    if missing:
        extra = Path(extra_cache)
        if extra.exists() and set(np.load(extra)["episode_id"].tolist()) >= set(missing):
            e = np.load(extra)
            keep = np.isin(e["episode_id"], missing)
            sel = np.repeat(keep, e["ep_len"])
            z_m = e["z"][sel]
            ep_m = np.repeat(e["episode_id"], e["ep_len"])[sel]
            st_m = np.concatenate([np.arange(l) for l in e["ep_len"]])[sel]
            log(f"[nodes] reused {extra}")
        else:
            model = load_lewm(Path(mech.ckpt_dir(root)), device=DEV)
            encode = make_encode_frame(model, batch=64)
            zs, eps, sts = [], [], []
            t0 = time.time()
            with h5py.File(mech.h5_path(root), "r", swmr=True, rdcc_nbytes=256 * 2**20) as f:
                for i, e in enumerate(missing):
                    lo, hi = int(h5_off[e]), int(h5_off[e] + h5_len[e])
                    zs.append(encode(f["pixels"][lo:hi]))
                    eps.append(np.full(hi - lo, e, np.int64))
                    sts.append(np.arange(hi - lo))
                    if i % 20 == 0:
                        log(f"[encode] {i + 1}/{len(missing)} ({time.time() - t0:.0f}s)")
            z_m, ep_m, st_m = np.concatenate(zs), np.concatenate(eps), np.concatenate(sts)
            np.savez_compressed(extra, z=z_m, episode_id=np.array(missing),
                                ep_len=h5_len[missing])
            log(f"[nodes] wrote {extra}")
        z_parts.append(z_m); ep_parts.append(ep_m); step_parts.append(st_m)
    z = np.concatenate(z_parts).astype(np.float32)
    ep_idx, step_idx = np.concatenate(ep_parts), np.concatenate(step_parts)
    first = {}
    for i in np.flatnonzero(step_idx == 0):
        first[int(ep_idx[i])] = int(i)
    log(f"[nodes] total {len(z)} nodes / {len(first)} episodes")
    return z, ep_idx, step_idx, first, h5_off


def goal_nodes(d, first, h5_off):
    g = []
    for r in d["goal_row"]:
        e = int(np.searchsorted(h5_off, int(r), side="right") - 1)
        g.append(first[e] + int(r) - int(h5_off[e]))
    return np.array(g)


# ------------------------------------------------------------------
# graph variants
# ------------------------------------------------------------------

def build_variants(z, ep_idx, step_idx, k, ids):
    n = len(z)
    ti, tj = find_transition_edges(ep_idx, step_idx)
    out = {"temporal": (build_weighted_graph(n, ti, tj, np.array([], int), np.array([], int), 1.0),
                        dict(n_trans=int(len(ti)), n_id=0, n_id_cross=0, eps2=None))}
    for name, eps2_override in ids.items():
        id_i, id_j, rho, eps2 = build_identification_edges(z, ep_idx, step_idx, k=k, eps2_override=eps2_override)
        cross = int((ep_idx[id_i] != ep_idx[id_j]).sum())
        out[name] = (build_weighted_graph(n, ti, tj, id_i, id_j, 1.0),
                     dict(n_trans=int(len(ti)), n_id=int(len(id_i)), n_id_cross=cross,
                          eps2=float(eps2), rho_hat=float(rho)))
        log(f"[graph] {name}: {len(id_i)} id edges ({cross} cross-episode), eps2={eps2:.3f}")
    return out


def adjacent_sqdisp_quantile(z, ep_idx, step_idx, q):
    """Empirical quantile of ||z_{t+1} - z_t||^2 -- the eps2_override style noted in
    build_identification_edges' docstring."""
    ti, tj = find_transition_edges(ep_idx, step_idx)
    d2 = np.empty(len(ti), np.float32)
    for lo in range(0, len(ti), 20000):                      # chunked: this machine fails
        hi = min(lo + 20000, len(ti))                        # ~100MB contiguous allocs
        d2[lo:hi] = ((z[tj[lo:hi]] - z[ti[lo:hi]]) ** 2).sum(1)
    return float(np.quantile(d2, q))


def dist_to_goals(graph, goals, limit):
    """(n_tasks, N) directed shortest-path length INTO each goal node (Dijkstra on the
    transposed graph), inf beyond `limit`."""
    gt = graph.T.tocsr()
    return dijkstra(gt, directed=True, indices=goals, limit=limit)


# ------------------------------------------------------------------
# scoring
# ------------------------------------------------------------------

def knn(z_nodes, q, k):
    import faiss
    index = faiss.IndexFlatL2(z_nodes.shape[1])
    index.add(np.ascontiguousarray(z_nodes))
    D, I = index.search(np.ascontiguousarray(q.astype(np.float32)), k)
    return np.sqrt(np.maximum(D, 0)), I


def score(TG, T, cap, h_nat, c_pred, nn_dist):
    """TG, T (n, K). Mirrors viability_eval_audit.evaluate_offset's ranking block."""
    n, K = T.shape
    rhos, regret, best_rank = [], [], []
    for i in range(n):
        if np.ptp(T[i]) > 0:
            r = sps.spearmanr(TG[i], T[i]).statistic
            if np.isfinite(r):
                rhos.append(r)
        pick = np.lexsort((nn_dist[i], TG[i]))[0]           # ties in T_G -> closer node wins
        regret.append(T[i][pick] - T[i].min())
        best = int(np.argmin(T[i]))
        best_rank.append(float((TG[i] < TG[i][best]).mean()))
    y = (T <= h_nat).astype(float).ravel()
    from common.viability import auroc
    return dict(
        spearman_TG_vs_T_mean=float(np.mean(rhos)) if rhos else float("nan"),
        n_informative_tasks=len(rhos),
        regret_steps_TG=float(np.mean(regret)),
        oracle_best_percentile_under_TG=float(np.mean(best_rank)),
        auroc_at_natural_h=auroc((-TG).ravel(), y) if 0 < y.mean() < 1 else float("nan"),
        disconnected_frac=float((TG >= DISCONNECTED).mean()),
        nn_dist_median=float(np.median(nn_dist)), nn_dist_p90=float(np.percentile(nn_dist, 90)),
        slot0_TG_median=float(np.median(TG[:, 0])), slot0_T_median=float(np.median(T[:, 0])),
    )


def evaluate(graphs, z_nodes, d, offset, first, h5_off, ks, limit):
    T, cap = d["oracle_min_steps"].astype(np.float64), float(d["oracle_cap"])
    n, K = T.shape
    natural = int(d["natural_budget"]) if "natural_budget" in d else max(0, offset - 25)
    goals = goal_nodes(d, first, h5_off)
    out = dict(n_tasks=n, n_candidates=K, oracle_cap=cap, natural_budget=natural,
               censored_frac=float((T > cap).mean()), graphs={})
    # reference columns, recomputed here so the table is self-contained
    l2 = d["c_pred"]
    l2_rhos = [sps.spearmanr(l2[i], T[i]).statistic for i in range(n) if np.ptp(T[i]) > 0]
    out["lewm_l2"] = dict(spearman=float(np.nanmean(l2_rhos)),
                          regret=float(np.mean([T[i][int(np.argmin(l2[i]))] - T[i].min() for i in range(n)])))
    for gname, (graph, ginfo) in graphs.items():
        t0 = time.time()
        D = dist_to_goals(graph, goals, limit)                     # (n, N)
        D = np.where(np.isfinite(D), D, DISCONNECTED)
        res = dict(info=ginfo, dijkstra_secs=time.time() - t0, by_input={})
        for name in ("z_true", "z_pred"):
            q = d[name].reshape(-1, z_nodes.shape[1])
            dist, idx = knn(z_nodes, q, max(ks))
            res["by_input"][name] = {}
            task_of_q = np.repeat(np.arange(n), K)[:, None]                  # (nK, 1)
            for k in ks:
                Dk = D[task_of_q, idx[:, :k]]                                # (nK, k) gather, no row copies
                TG = np.median(Dk, axis=1).reshape(n, K) if k > 1 else Dk[:, 0].reshape(n, K)
                res["by_input"][name][f"k{k}"] = score(TG, T, cap, natural, l2, dist[:, 0].reshape(n, K))
        out["graphs"][gname] = res
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=str(Path(__file__).resolve().parent.parent))
    p.add_argument("--cache", default="outputs/pusht/critic_training/cache_1000_s0.npz")
    p.add_argument("--audit", default="outputs/pusht/experiments/viability_exp2")
    p.add_argument("--goal-offsets", type=int, nargs="+", default=[25, 50])
    p.add_argument("--id-k", type=int, default=4)
    p.add_argument("--id-quantiles", type=float, nargs="+", default=[0.5, 0.99],
                   help="eps2 overrides as empirical quantiles of adjacent-pair sq. displacement "
                        "(besides the calibrated default)")
    p.add_argument("--knn", type=int, nargs="+", default=[1, 5])
    p.add_argument("--limit", type=float, default=500.0, help="Dijkstra cutoff (env steps)")
    p.add_argument("--critic-eval", default="outputs/pusht/experiments/viability_exp2/eval_audit_critic_full_s0.json")
    p.add_argument("--out", default="outputs/pusht/critic_training/graph_label_test.json")
    args = p.parse_args()
    root = Path(args.root)

    audits = load_audits(root / args.audit, args.goal_offsets)
    z, ep_idx, step_idx, first, h5_off = build_nodes(
        root / args.cache, audits, root, root / "outputs" / "pusht" / "critic_training" / "cache_audit_episodes.npz")

    ids = {"id_calibrated": None}
    for q in args.id_quantiles:
        ids[f"id_q{q}"] = adjacent_sqdisp_quantile(z, ep_idx, step_idx, q)
    graphs = build_variants(z, ep_idx, step_idx, args.id_k, ids)

    critic = None
    if Path(root / args.critic_eval).exists():
        critic = json.loads(Path(root / args.critic_eval).read_text())["offsets"]

    results = dict(args=vars(args), n_nodes=int(len(z)), offsets={})
    for off, d in audits.items():
        r = evaluate(graphs, z, d, off, first, h5_off, args.knn, args.limit)
        results["offsets"][str(off)] = r
        log(f"\n[offset={off}] n_tasks={r['n_tasks']} natural h={r['natural_budget']} "
            f"censored={r['censored_frac']:.2f} | LeWM L2: spearman={r['lewm_l2']['spearman']:.3f} "
            f"regret={r['lewm_l2']['regret']:.1f}")
        if critic and str(off) in critic:
            for name in ("z_pred", "z_true"):
                c = critic[str(off)]["by_input"][name]["at_natural_h"]
                log(f"[offset={off}] critic V ({name}): spearman={c['spearman_v_vs_negT_mean']:.3f} "
                    f"regret={c['regret_steps_V']:.1f} best-pct={c['oracle_best_percentile_under_V']:.2f}")
        for gname, res in r["graphs"].items():
            info = res["info"]
            log(f"[offset={off}] graph={gname} trans={info['n_trans']} id={info['n_id']} "
                f"(cross-ep {info['n_id_cross']}) dijkstra {res['dijkstra_secs']:.1f}s")
            for name, byk in res["by_input"].items():
                for k, s in byk.items():
                    log(f"[offset={off}]   {name} {k}: spearman={s['spearman_TG_vs_T_mean']:.3f} "
                        f"(n={s['n_informative_tasks']}) regret={s['regret_steps_TG']:.1f} "
                        f"best-pct={s['oracle_best_percentile_under_TG']:.2f} auroc@h{r['natural_budget']}="
                        f"{s['auroc_at_natural_h']:.2f} disconnected={s['disconnected_frac']:.2f} "
                        f"nn_dist med/p90={s['nn_dist_median']:.2f}/{s['nn_dist_p90']:.2f} "
                        f"slot0 T_G/T median={s['slot0_TG_median']:.0f}/{s['slot0_T_median']:.0f}")
    out = root / args.out
    out.write_text(json.dumps(results, indent=2))
    log(f"[done] wrote {out}")


if __name__ == "__main__":
    main()
