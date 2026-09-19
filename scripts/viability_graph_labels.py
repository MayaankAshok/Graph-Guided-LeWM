"""Offline-supported hitting-time labels for the revised critic (viability-value.tex, sec.
"Revised Critic"): build the directed dataset graph over a latent cache and write a bank of
(start, goal, T_G) training pairs, where T_G is the directed shortest-path length in ENV
STEPS from the start frame to the goal frame.

Graph (common.graph_lib): every cached frame is a node; transition edges z_t -> z_{t+1} are
directed, cost 1; identification edges (k-capped FAISS kNN, symmetric, cost id_weight=1)
stitch episodes together. Stitching is ON by default -- viability_graph_label_test.py showed
a temporal-only graph leaves 70-80% of planner endpoints with no path to the goal -- at an
eps^2 taken from the empirical distribution of adjacent-frame squared displacement
(--id-eps2-quantile; the calibrated q=1e-3 threshold is ~10x too tight to stitch anything).

Labels are exact for finite paths and CENSORED ('T > B_max blocks') beyond the Dijkstra
bound 5*B_max: the trainer gives those rows weight --w-disconnected (0.25), since a missing
path may be missing coverage rather than physical impossibility. For every pair the bank
also stores T_G from the frames 5, 10, ..., 25 steps after the start, so that a predictor
rollout under the LOGGED actions from that start can be labelled at each of its imagined
frames with the graph label of the frame it is meant to represent.

Episode split (train / val) is identical to viability_train.py's (same seed/val_frac ->
LatentCache.split_episodes). Goals and starts of a pair always come from the same split.
Per goal, starts are drawn from the same episode before the goal (offsets 1..--max-offset),
the goal itself (identity pairs, T_G = 0, so bin 0 / V(z, z, 0) is trained), the same
episode after it (backward pairs), and random frames of other episodes in the split.

Usage:
    python scripts/viability_graph_labels.py --cache outputs/pusht/critic_training/cache_1000_s0.npz \
        --out outputs/pusht/critic_training/labels_1000_s0.npz
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.sparse.csgraph import dijkstra

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.graph_lib import build_identification_edges, build_weighted_graph, find_transition_edges
from common.log_util import log
from common.viability import HISTORY, SKIP

N_FWD = 5   # imagined frames per rollout: start + 5k, k = 1..5


def adjacent_sqdisp_quantile(z, ti, tj, q, chunk=20000):
    d2 = np.empty(len(ti), np.float32)
    for lo in range(0, len(ti), chunk):
        hi = min(lo + chunk, len(ti))
        d2[lo:hi] = ((z[tj[lo:hi]] - z[ti[lo:hi]]) ** 2).sum(1)
    return float(np.quantile(d2, q))


def build_graph(z, ep_idx, step_idx, args):
    n = len(z)
    ti, tj = find_transition_edges(ep_idx, step_idx)
    info = dict(n_nodes=int(n), n_trans=int(len(ti)), n_id=0, n_id_cross=0, eps2=None)
    if args.no_id_edges:
        g = build_weighted_graph(n, ti, tj, np.array([], int), np.array([], int), 1.0)
        return g, info
    eps2 = args.id_eps2 if args.id_eps2 is not None else adjacent_sqdisp_quantile(z, ti, tj, args.id_eps2_quantile)
    id_i, id_j, rho, eps2 = build_identification_edges(z, ep_idx, step_idx, k=args.id_k, eps2_override=eps2)
    info.update(n_id=int(len(id_i)), n_id_cross=int((ep_idx[id_i] != ep_idx[id_j]).sum()),
                eps2=float(eps2), rho_hat=float(rho), id_k=args.id_k)
    return build_weighted_graph(n, ti, tj, id_i, id_j, args.id_weight), info


def sample_pairs_for_goal(rng, g, ep_idx, ep_offset, ep_len, split_frames_of_other_eps, args):
    """Start frames for one goal node g: same-episode earlier, same-episode later, cross."""
    e = ep_idx[g]
    lo, hi = int(ep_offset[e]), int(ep_offset[e] + ep_len[e])
    starts = [np.full(args.n_same_zero, g)]          # identity pairs, T_G = 0 (bin 0)
    if g - lo > 0:
        off = rng.integers(1, min(args.max_offset, g - lo) + 1, size=args.n_same_fwd)
        starts.append(g - off)
    if hi - 1 - g > 0:
        off = rng.integers(1, min(args.max_offset, hi - 1 - g) + 1, size=args.n_same_bwd)
        starts.append(g + off)
    if args.n_cross > 0 and len(split_frames_of_other_eps):
        starts.append(rng.choice(split_frames_of_other_eps, size=args.n_cross, replace=False))
    return np.concatenate(starts) if starts else np.array([], np.int64)


_G = None   # per-process graph + layout, set in main() before the pool forks


def _label_job(job):
    """(split_idx, goal chunk, seed) -> dict of per-pair arrays for those goals."""
    si, gs, seed = job
    G = _G
    ep_idx, ep_offset, ep_len, args = G["ep_idx"], G["ep_offset"], G["ep_len"], G["args"]
    frames = G[f"frames{si}"]
    rng = np.random.default_rng(seed)
    D = dijkstra(G["graph_t"], directed=True, indices=gs, limit=G["limit"])      # (chunk, N)
    out = {k: [] for k in ("start", "goal", "tg", "tg_fwd", "split", "kind")}
    for r, g in enumerate(gs):
        others = frames[ep_idx[frames] != ep_idx[g]]
        starts = sample_pairs_for_goal(rng, int(g), ep_idx, ep_offset, ep_len, others, args)
        if len(starts) == 0:
            continue
        d = D[r]
        tg = np.where(np.isfinite(d[starts]), d[starts], -1.0).astype(np.float32)
        # forward frames start + 5k inside the start's episode (-2 = no such frame)
        fwd = starts[:, None] + SKIP * np.arange(1, N_FWD + 1)[None]
        e_s = ep_idx[starts]
        inside = fwd < (ep_offset[e_s] + ep_len[e_s])[:, None]
        fwd_c = np.minimum(fwd, G["n"] - 1)
        tgf = np.where(np.isfinite(d[fwd_c]), d[fwd_c], -1.0).astype(np.float32)
        tgf = np.where(inside, tgf, -2.0)
        same = e_s == ep_idx[g]
        kind = np.where(~same, 2, np.where(starts <= g, 0, 1)).astype(np.int8)  # 0 fwd/zero, 1 bwd, 2 cross
        out["start"].append(starts); out["goal"].append(np.full(len(starts), g))
        out["tg"].append(tg); out["tg_fwd"].append(tgf)
        out["split"].append(np.full(len(starts), si, np.int8)); out["kind"].append(kind)
    empty = dict(start=np.array([], np.int64), goal=np.array([], np.int64), tg=np.array([], np.float32),
                 tg_fwd=np.zeros((0, N_FWD), np.float32), split=np.array([], np.int8), kind=np.array([], np.int8))
    return {k: (np.concatenate(v) if v else empty[k]) for k, v in out.items()}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--val-frac", type=float, default=0.1)
    # graph
    p.add_argument("--no-id-edges", action="store_true", help="temporal edges only (ablation)")
    p.add_argument("--id-k", type=int, default=4)
    p.add_argument("--id-eps2-quantile", type=float, default=0.99,
                   help="eps^2 = this quantile of adjacent-frame squared displacement")
    p.add_argument("--id-eps2", type=float, default=None, help="explicit eps^2 (overrides the quantile)")
    p.add_argument("--id-weight", type=float, default=1.0)
    # labels
    p.add_argument("--b-max", type=int, default=45, help="classes 0..B_max blocks, then '> B_max'")
    p.add_argument("--goals-train", type=int, default=4000)
    p.add_argument("--goals-val", type=int, default=500)
    p.add_argument("--n-same-zero", type=int, default=2, help="identity pairs (start = goal) per goal")
    p.add_argument("--n-same-fwd", type=int, default=40)
    p.add_argument("--n-same-bwd", type=int, default=10)
    p.add_argument("--n-cross", type=int, default=50)
    p.add_argument("--max-offset", type=int, default=250)
    p.add_argument("--dijkstra-chunk", type=int, default=50)
    p.add_argument("--workers", type=int, default=1, help="fork()ed Dijkstra workers (Linux); 1 = serial")
    args = p.parse_args()

    t0 = time.time()
    c = np.load(args.cache)
    z = c["z"].astype(np.float32)
    ep_offset, ep_len = c["ep_offset"].astype(np.int64), c["ep_len"].astype(np.int64)
    n_ep = len(ep_len)
    ep_idx = np.repeat(np.arange(n_ep), ep_len)
    step_idx = np.concatenate([np.arange(l) for l in ep_len])
    log(f"[setup] cache {args.cache}: {len(z)} frames / {n_ep} episodes")

    graph, ginfo = build_graph(z, ep_idx, step_idx, args)
    graph_t = graph.T.tocsr()                                # reversed: distances INTO a goal
    limit = SKIP * args.b_max + 0.5
    log(f"[graph] {ginfo}  limit={limit:.0f} env steps")

    # same episode split as viability_train.py
    rng_split = np.random.default_rng(args.seed)
    perm = rng_split.permutation(n_ep)
    n_val = max(1, int(round(args.val_frac * n_ep)))
    splits = {"train": np.sort(perm[n_val:]), "val": np.sort(perm[:n_val])}
    rng = np.random.default_rng(args.seed + 1)              # goal sampling only
    hist = (HISTORY - 1) * SKIP

    # one job = one Dijkstra chunk of goals + the pair sampling for them; jobs run in a
    # fork()ed pool (the graph is inherited copy-on-write, nothing large crosses a pipe)
    global _G
    _G = dict(graph_t=graph_t, limit=limit, ep_idx=ep_idx, ep_offset=ep_offset, ep_len=ep_len, n=len(z), args=args)
    jobs = []
    for si, (name, eps) in enumerate(splits.items()):
        frames = np.flatnonzero(np.isin(ep_idx, eps))
        n_goals = args.goals_train if name == "train" else args.goals_val
        goals = rng.choice(frames, size=min(n_goals, len(frames)), replace=False)
        log(f"[{name}] {len(eps)} episodes / {len(frames)} frames -> {len(goals)} goals")
        _G[f"frames{si}"] = frames
        for j, lo in enumerate(range(0, len(goals), args.dijkstra_chunk)):
            jobs.append((si, goals[lo:lo + args.dijkstra_chunk], args.seed + 1000 * si + j))
    out = {k: [] for k in ("start", "goal", "tg", "tg_fwd", "split", "kind")}
    done = 0
    if args.workers > 1:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(args.workers) as pool:
            for res in pool.imap_unordered(_label_job, jobs, chunksize=1):
                for k in out:
                    out[k].append(res[k])
                done += 1
                if done % 20 == 0:
                    log(f"[labels] jobs {done}/{len(jobs)} ({time.time() - t0:.0f}s)")
    else:
        for job in jobs:
            res = _label_job(job)
            for k in out:
                out[k].append(res[k])
            done += 1
            if done % 20 == 0:
                log(f"[labels] jobs {done}/{len(jobs)} ({time.time() - t0:.0f}s)")

    arrs = {k: np.concatenate(v) for k, v in out.items()}
    # rollout eligibility: enough real history before the start and 25 logged actions after
    s = arrs["start"]; e_s = ep_idx[s]
    arrs["hist_ok"] = (step_idx[s] >= hist) & (step_idx[s] + SKIP * N_FWD <= ep_len[e_s] - 1)
    tg = arrs["tg"]
    for si, name in enumerate(splits):
        m = arrs["split"] == si
        fin = tg[m] >= 0
        log(f"[{name}] pairs={m.sum()} exact={fin.mean():.2f} (median T_G={np.median(tg[m][fin]):.0f} steps) "
            f"beyond-bound={1 - fin.mean():.2f} | by kind fwd/bwd/cross exact frac: "
            + " ".join(f"{(tg[m & (arrs['kind'] == k)] >= 0).mean():.2f}" for k in (0, 1, 2))
            + f" | rollout-eligible {arrs['hist_ok'][m].mean():.2f}")
    meta = dict(cache=str(Path(args.cache).resolve()), seed=args.seed, val_frac=args.val_frac,
                b_max=args.b_max, limit=limit, graph=ginfo, args=vars(args),
                train_episodes=splits["train"].tolist(), val_episodes=splits["val"].tolist())
    np.savez_compressed(args.out, meta=json.dumps(meta), **arrs)
    log(f"[done] wrote {args.out} ({len(tg)} pairs, {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
