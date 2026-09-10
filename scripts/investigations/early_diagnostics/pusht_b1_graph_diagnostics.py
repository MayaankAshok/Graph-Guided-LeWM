"""B1: graph diagnostics for the latent-graph idea (Push-T, real data/checkpoint).

Adapted from tworoom_b1_graph_diagnostics.py, but built on the unified common.graph_lib/
common.envs layer (with the compute_calibration fix -- see CLAUDE.md/graph_lib.py) instead
of Two-Room-specific code, for three reasons this isn't just a port:

  1. Two-Room's B1 used the exact O(n^2) brute-force identification-edge scan at every
     scale tested. That's exactly the thing B1.1 (landmark-count scaling) is meant to test
     the infeasibility of -- and Push-T's production config already uses the k-capped
     construction (config/graph/env/pusht.yaml: id_edge_method=capped) specifically because
     brute force is a wall here. So this script tests the CAPPED construction's scaling
     behavior (what's actually deployed), not brute force's.
  2. Two-Room's B1 used its own eps2_of(q, rho_hat) = 2*(1-rho_hat)*chi2.ppf(q, d) formula.
     That formula is the one that goes negative on Push-T (rho_hat statistically >= 1) --
     using it here would just reproduce the bug this fix addressed. Uses
     common.graph_lib.compute_calibration instead (always positive by construction).
  3. Push-T has no rooms -- the same_room/cross_room split doesn't apply. Split by
     same-episode/cross-episode instead, matching graph_lib.evaluate_phi_quality's
     convention and this environment's actual structure (cross-episode is where the
     identification-edge graph either stitches or fragments -- see [[pusht-...]] memories).

Four questions, same shape as Two-Room's B1 (docs/graph-proposal/main.tex, B1 row):
  1. Landmark-count scaling (50/300/1000 episodes): does quality hold, and how does
     edge count/avg degree grow, as the landmark budget grows?
  2. Successor-consistency filter: for an identification edge (i,j) with SIMILAR real
     actions a_i~=a_j and both having a real next frame, do the real successors
     (i+1, j+1) also pass the same identification test? If not, drop the edge. Does
     this reduce the measured false-edge rate without hurting graph quality?
  3. Connectivity/fragmentation vs q: does the graph fragment into disconnected
     components as q tightens? (Expected to be severe here -- see B0's own 16/4000
     result at q=1e-3; this sweeps q to see where fragmentation actually sets in.)
  4. Degree distribution: are there pathological hub nodes?
"""

import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
from scipy import stats as sps
from scipy.sparse.csgraph import connected_components, dijkstra

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from common.envs import ENV_MECHANICS  # noqa: E402
from common.graph_lib import (  # noqa: E402
    build_id_edges_faiss_capped, build_weighted_graph, compute_calibration,
    find_transition_edges, log,
)
from graph_gate import load_landmarks  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent.parent
OUT_DIR = ROOT / "outputs" / "b1_pusht"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MECH = ENV_MECHANICS["pusht"]
H5_PATH = MECH.h5_path(ROOT)
CKPT_DIR = MECH.ckpt_dir(ROOT)
LANDMARK_DIR = ROOT / "outputs" / "b0_pusht"
Q_DEFAULT = 1e-3
K_DEFAULT = 4  # matches config/graph/env/pusht.yaml's id_edge_k
rng = np.random.default_rng(0)


def get_landmarks(n_episodes, seed=0):
    return load_landmarks(MECH, H5_PATH, LANDMARK_DIR, n_episodes, CKPT_DIR, seed=seed)


def lean_evaluate(z, state, graph, true_dist_oracle, ep_idx, n_sources=100, n_targets=40, seed=0):
    """Single-setting version of graph_gate.evaluate(): one graph in, Spearman numbers out,
    split by same-episode vs cross-episode (Push-T's analogue of Two-Room's same/cross-room
    split -- there are no rooms here, but same-vs-cross-episode is exactly where
    identification edges either stitch or (per B0) fragment)."""
    n = z.shape[0]
    local_rng = np.random.default_rng(seed)
    src_idx = local_rng.choice(n, size=n_sources, replace=False)
    D_graph_full = dijkstra(graph, indices=src_idx, directed=True)

    rows = []
    for si, s in enumerate(src_idx):
        tgt = local_rng.choice(n, size=n_targets, replace=False)
        tgt = tgt[tgt != s]
        true_d = true_dist_oracle(state[s: s + 1], state[tgt])[0]
        graph_d = D_graph_full[si, tgt]
        eucl_d = np.linalg.norm(z[s: s + 1] - z[tgt], axis=1)
        cross = ep_idx[tgt] != ep_idx[s]
        for k in range(len(tgt)):
            rows.append((true_d[k], eucl_d[k], graph_d[k], bool(cross[k])))

    true_d = np.array([r[0] for r in rows]); eucl_d = np.array([r[1] for r in rows])
    graph_d = np.array([r[2] for r in rows]); cross = np.array([r[3] for r in rows])
    finite = np.isfinite(true_d) & np.isfinite(graph_d) & np.isfinite(eucl_d)
    n_dropped = int((~finite).sum())
    true_d, eucl_d, graph_d, cross = true_d[finite], eucl_d[finite], graph_d[finite], cross[finite]

    def sp(mask):
        if mask.sum() < 10:
            return None, None
        re, _ = sps.spearmanr(true_d[mask], eucl_d[mask])
        rg, _ = sps.spearmanr(true_d[mask], graph_d[mask])
        return float(re), float(rg)

    re_o, rg_o = sp(np.ones_like(cross, dtype=bool))
    re_s, rg_s = sp(~cross)
    re_c, rg_c = sp(cross)
    return dict(
        n_dropped_nonfinite=n_dropped, n_pairs=int(finite.sum()),
        overall=dict(euclidean=re_o, graph=rg_o),
        same_episode=dict(euclidean=re_s, graph=rg_s),
        cross_episode=dict(euclidean=re_c, graph=rg_c),
    )


# ============================================================
# B1.1 landmark-count scaling (capped construction, corrected calibration)
# ============================================================

def b1_scaling():
    log("\n" + "=" * 60 + "\nB1.1: landmark-count scaling (50 / 300 / 1000 episodes)\n" + "=" * 60)
    results = {}
    for n_ep in [50, 300, 1000]:
        log(f"\n--- n_episodes={n_ep} ---")
        t0 = time.time()
        z, ep_idx, step_idx, state = get_landmarks(n_ep)
        true_dist_oracle = MECH.build_true_distance_oracle(state)
        n = z.shape[0]
        rho_hat, eps2 = compute_calibration(z, ep_idx, step_idx, q_calib=Q_DEFAULT)

        trans_i, trans_j = find_transition_edges(ep_idx, step_idx)
        t_edges0 = time.time()
        id_i, id_j = build_id_edges_faiss_capped(z, ep_idx, step_idx, eps2, K_DEFAULT)
        t_edges = time.time() - t_edges0

        graph = build_weighted_graph(n, trans_i, trans_j, id_i, id_j, id_weight=1.0)
        t_eval0 = time.time()
        ev = lean_evaluate(z, state, graph, true_dist_oracle, ep_idx)
        t_eval = time.time() - t_eval0

        avg_deg = 2 * len(id_i) / n
        log(f"n_landmarks={n} rho_hat={rho_hat:.4f} eps2={eps2:.4f} "
            f"n_id_edges={len(id_i)} avg_degree={avg_deg:.2f} "
            f"edge_scan_time={t_edges:.1f}s eval_time={t_eval:.1f}s total_time={time.time()-t0:.1f}s")
        log(f"  overall: eucl={ev['overall']['euclidean']} graph={ev['overall']['graph']} "
            f"(n={ev['n_pairs']}, {ev['n_dropped_nonfinite']} dropped unreachable)")
        log(f"  same-episode: eucl={ev['same_episode']['euclidean']} graph={ev['same_episode']['graph']}")
        log(f"  cross-episode: eucl={ev['cross_episode']['euclidean']} graph={ev['cross_episode']['graph']}")

        results[n_ep] = dict(
            n_landmarks=int(n), rho_hat=rho_hat, eps2=eps2,
            n_id_edges=int(len(id_i)), avg_degree=avg_deg,
            edge_scan_time_s=t_edges, eval_time_s=t_eval,
            evaluation=ev,
        )

    ns = np.array([results[k]["n_landmarks"] for k in [50, 300, 1000]], dtype=float)
    es = np.array([results[k]["n_id_edges"] for k in [50, 300, 1000]], dtype=float)
    slope, intercept = np.polyfit(np.log(ns), np.log(np.maximum(es, 1)), 1)
    log(f"\n[scaling] n_id_edges ~ n_landmarks^{slope:.2f} (fit on 50/300/1000, capped "
        f"construction -- unlike Two-Room's uncapped ~n^2, this is expected close to linear "
        f"by design, since each node's out-degree is capped at k={K_DEFAULT})")
    results["scaling_fit"] = dict(power=float(slope))
    return results


# ============================================================
# B1.2 successor-consistency filter
# ============================================================

def b1_successor_consistency():
    log("\n" + "=" * 60 + "\nB1.2: successor-consistency filter\n" + "=" * 60)
    z, ep_idx, step_idx, state = get_landmarks(300)
    true_dist_oracle = MECH.build_true_distance_oracle(state)
    n = z.shape[0]
    rho_hat, eps2 = compute_calibration(z, ep_idx, step_idx, q_calib=Q_DEFAULT)

    f = h5py.File(H5_PATH, "r", swmr=True)
    ep_offset = f["ep_offset"][:]
    ep_len = f["ep_len"][:]
    global_idx = ep_offset[ep_idx] + step_idx
    action = f["action"][:][global_idx]
    has_successor = step_idx < (ep_len[ep_idx] - 1)
    key = ep_idx.astype(np.int64) * 1_000_000 + step_idx.astype(np.int64)
    key_to_row = {int(k): i for i, k in enumerate(key)}
    succ_row = np.full(n, -1, dtype=np.int64)
    for i in range(n):
        if has_successor[i]:
            k = int(ep_idx[i]) * 1_000_000 + int(step_idx[i]) + 1
            succ_row[i] = key_to_row.get(k, -1)
    f.close()

    trans_i, trans_j = find_transition_edges(ep_idx, step_idx)
    id_i, id_j = build_id_edges_faiss_capped(z, ep_idx, step_idx, eps2, K_DEFAULT)
    log(f"[filter] starting identification edges: {len(id_i)}")

    a_i, a_j = action[id_i], action[id_j]
    an_i = a_i / (np.linalg.norm(a_i, axis=1, keepdims=True) + 1e-8)
    an_j = a_j / (np.linalg.norm(a_j, axis=1, keepdims=True) + 1e-8)
    action_cos = np.sum(an_i * an_j, axis=1)
    n_nan_action = int(np.isnan(action_cos).sum())
    log(f"[filter] action-cosine over identification edges ({n_nan_action} NaN, excluded):")
    log(f"  mean={np.nanmean(action_cos):.3f} median={np.nanmedian(action_cos):.3f} "
        f"p10={np.nanpercentile(action_cos,10):.3f} p90={np.nanpercentile(action_cos,90):.3f}")

    ACTION_COS_THRESH = 0.7
    both_succ = (succ_row[id_i] >= 0) & (succ_row[id_j] >= 0)
    similar_action = action_cos > ACTION_COS_THRESH
    testable = both_succ & similar_action
    log(f"[filter] testable edges (both have a sampled successor AND action_cos>{ACTION_COS_THRESH}): "
        f"{testable.sum()}/{len(id_i)} ({100*testable.sum()/max(len(id_i),1):.1f}%)")

    si, sj = succ_row[id_i[testable]], succ_row[id_j[testable]]
    succ_sqd = np.sum((z[si] - z[sj]) ** 2, axis=1)
    passes = succ_sqd < eps2
    log(f"[filter] of testable edges, successor test passes: {passes.sum()}/{max(testable.sum(),1)} "
        f"({100*passes.mean() if testable.sum() else 0:.1f}%) -- fails (dropped): {(~passes).sum()}")

    drop_mask = np.zeros(len(id_i), dtype=bool)
    testable_idx = np.nonzero(testable)[0]
    drop_mask[testable_idx[~passes]] = True
    keep_mask = ~drop_mask
    id_i_f, id_j_f = id_i[keep_mask], id_j[keep_mask]
    log(f"[filter] identification edges after filtering: {len(id_i_f)} "
        f"({100*len(id_i_f)/max(len(id_i),1):.1f}% kept, {drop_mask.sum()} dropped)")

    def measure_false_edge_rate(ei, ej, n_sample=3000, dist_thresh=0.2444):
        # 0.2444 = the 95th-pct adjacent-frame true-distance threshold (tau) used
        # throughout this session's B0/diagnostic work, not an arbitrary pixel count --
        # Push-T's state is z-scored, not pixels, unlike Two-Room's 12px threshold.
        n_avail = len(ei)
        if n_avail == 0:
            return float("nan"), 0
        sample = rng.choice(n_avail, size=min(n_sample, n_avail), replace=False)
        td = np.array([
            true_dist_oracle(state[ei[k]:ei[k]+1], state[ej[k]:ej[k]+1])[0, 0]
            for k in sample
        ])
        return float(np.mean(td > dist_thresh)), len(sample)

    false_rate_before, n_before = measure_false_edge_rate(id_i, id_j)
    false_rate_after, n_after = measure_false_edge_rate(id_i_f, id_j_f)
    log(f"[filter] measured false-edge rate (true dist > tau): "
        f"before={false_rate_before*100:.1f}% (n={n_before})  after={false_rate_after*100:.1f}% (n={n_after})")

    graph_before = build_weighted_graph(n, trans_i, trans_j, id_i, id_j, id_weight=1.0)
    graph_after = build_weighted_graph(n, trans_i, trans_j, id_i_f, id_j_f, id_weight=1.0)
    ev_before = lean_evaluate(z, state, graph_before, true_dist_oracle, ep_idx)
    ev_after = lean_evaluate(z, state, graph_after, true_dist_oracle, ep_idx)
    log(f"[filter] Spearman before: overall={ev_before['overall']['graph']} "
        f"same-ep={ev_before['same_episode']['graph']} cross-ep={ev_before['cross_episode']['graph']}")
    log(f"[filter] Spearman after:  overall={ev_after['overall']['graph']} "
        f"same-ep={ev_after['same_episode']['graph']} cross-ep={ev_after['cross_episode']['graph']}")

    return dict(
        action_cos_threshold=ACTION_COS_THRESH,
        n_id_edges_before=int(len(id_i)), n_id_edges_after=int(len(id_i_f)),
        n_testable=int(testable.sum()), n_dropped=int(drop_mask.sum()),
        false_edge_rate_before=false_rate_before, false_edge_rate_after=false_rate_after,
        evaluation_before=ev_before, evaluation_after=ev_after,
    )


# ============================================================
# B1.3 connectivity / fragmentation vs q
# ============================================================

def b1_connectivity():
    log("\n" + "=" * 60 + "\nB1.3: connectivity / fragmentation vs q\n" + "=" * 60)
    z, ep_idx, step_idx, state = get_landmarks(300)
    n = z.shape[0]

    trans_i, trans_j = find_transition_edges(ep_idx, step_idx)

    # cross-episode query pairs, fixed across q, for a reachability check (Push-T's
    # analogue of Two-Room's left/right-room reachability check)
    cross_pairs = []
    for _ in range(500):
        i = rng.integers(0, n)
        same_ep_others = np.nonzero(ep_idx != ep_idx[i])[0]
        j = rng.choice(same_ep_others)
        cross_pairs.append((i, j))

    results = {}
    for q in [0.5, 0.2, 0.05, 0.01, 1e-3, 1e-4, 1e-6, 1e-9]:
        _, eps2 = compute_calibration(z, ep_idx, step_idx, q_calib=q)
        id_i, id_j = build_id_edges_faiss_capped(z, ep_idx, step_idx, eps2, K_DEFAULT)
        graph = build_weighted_graph(n, trans_i, trans_j, id_i, id_j, id_weight=1.0)

        n_comp, labels = connected_components(graph, directed=False)
        sizes = np.bincount(labels)
        largest_frac = sizes.max() / n

        reachable = np.mean([labels[i] == labels[j] for i, j in cross_pairs])

        log(f"q={q:g}: eps2={eps2:.4f} n_id_edges={len(id_i)} n_components={n_comp} "
            f"largest_component={largest_frac*100:.2f}% cross_episode_reachable={reachable*100:.1f}%")
        results[str(q)] = dict(
            eps2=eps2, n_id_edges=int(len(id_i)), n_components=int(n_comp),
            largest_component_frac=float(largest_frac), cross_episode_reachable_frac=float(reachable),
        )
    return results


# ============================================================
# B1.4 degree distribution
# ============================================================

def b1_degree_distribution():
    log("\n" + "=" * 60 + "\nB1.4: identification-edge degree distribution\n" + "=" * 60)
    z, ep_idx, step_idx, state = get_landmarks(300)
    n = z.shape[0]
    _, eps2 = compute_calibration(z, ep_idx, step_idx, q_calib=Q_DEFAULT)
    id_i, id_j = build_id_edges_faiss_capped(z, ep_idx, step_idx, eps2, K_DEFAULT)

    deg = np.zeros(n, dtype=np.int64)
    np.add.at(deg, id_i, 1)
    np.add.at(deg, id_j, 1)

    pct = {p: float(np.percentile(deg, p)) for p in [50, 90, 99, 99.9]}
    log(f"[degree] mean={deg.mean():.1f} p50={pct[50]:.0f} p90={pct[90]:.0f} "
        f"p99={pct[99]:.0f} p99.9={pct[99.9]:.0f} max={deg.max()}  "
        f"(k-capped construction: max degree is structurally bounded by ~2*{K_DEFAULT}, "
        f"unlike Two-Room's uncapped B1 which could see real hub outliers)")

    hub_thresh = max(pct[99.9] * 1.5, pct[99] * 2, 2 * K_DEFAULT + 1)
    hubs = np.nonzero(deg > hub_thresh)[0]
    log(f"[degree] outlier hub threshold={hub_thresh:.0f}: {len(hubs)} nodes flagged")
    hub_info = []
    for h in hubs[:10]:
        log(f"  hub node {h}: degree={deg[h]}, state[:2]=({state[h,0]:.1f},{state[h,1]:.1f}), "
            f"ep={ep_idx[h]}, step={step_idx[h]}")
        hub_info.append(dict(node=int(h), degree=int(deg[h]),
                              state_xy=[float(state[h, 0]), float(state[h, 1])]))

    return dict(mean=float(deg.mean()), percentiles=pct, max=int(deg.max()),
                n_hubs=int(len(hubs)), hub_threshold=float(hub_thresh), hubs=hub_info)


def main():
    log(f"env=pusht")

    all_results = {}
    all_results["scaling"] = b1_scaling()
    all_results["successor_consistency"] = b1_successor_consistency()
    all_results["connectivity"] = b1_connectivity()
    all_results["degree_distribution"] = b1_degree_distribution()

    out_path = OUT_DIR / "b1_results.json"
    out_path.write_text(json.dumps(all_results, indent=2, default=str))
    log(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
