"""B1: graph diagnostics for the latent-graph idea (Two-Room, real data/checkpoint).

Four questions, each with a kill criterion (docs/graph-proposal/main.tex, B1 row):

  1. Landmark-count scaling (50/300/1000 episodes): does quality hold, and how does
     edge count grow, as the landmark budget grows? Directly informs Push-T feasibility.
  2. Successor-consistency filter: for an identification edge (i,j) with SIMILAR real
     actions a_i~=a_j and both having a real next frame, do the real successors
     (i+1, j+1) also pass the same identification test? If not, drop the edge. Does
     this reduce the measured false-edge rate without hurting graph quality?
  3. Connectivity/fragmentation vs q: does the graph fragment into disconnected
     components as q tightens? Does cross-room reachability survive?
  4. Degree distribution: are there pathological hub nodes?

Context fact checked first (not assumed): 446/500 sampled episodes already cross the
wall on their own (agent visits both x<112 and x>112 within one real trajectory), so
transition edges alone already carry substantial cross-room connectivity in Two-Room.
Identification edges are then mainly stitching *across* episodes for a globally
consistent metric, not the only way to physically connect the two rooms.
"""

import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from scipy import stats as sps
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from tworoom_lewm_loader import load_tworoom_lewm
from tworoom_b0_graph_gate import (
    DEV, H5_PATH, IMAGENET_MEAN, IMAGENET_STD, WALL_CENTER_X,
    build_true_distance_oracle, build_weighted_graph, find_graph_edges, log, rng,
)

ROOT = Path(__file__).resolve().parent.parent
LANDMARK_DIR = ROOT / "outputs" / "b0_tworoom"
OUT_DIR = ROOT / "outputs" / "b1_tworoom"
OUT_DIR.mkdir(parents=True, exist_ok=True)

Q_DEFAULT = 1e-3
D = 192


def eps2_of(q, rho_hat):
    return 2 * (1 - rho_hat) * sps.chi2.ppf(q, D)


# ============================================================
# shared: encode N episodes -> landmarks (generalizes b0's load_landmarks)
# ============================================================

def load_landmarks(n_episodes, seed=0):
    cache = LANDMARK_DIR / f"landmarks_ep{n_episodes}.npz"
    if cache.exists():
        log(f"[encode] loading cached landmarks from {cache}")
        d = np.load(cache)
        return d["z"], d["ep_idx"], d["step_idx"], d["proprio"]

    f = h5py.File(H5_PATH, "r", swmr=True, rdcc_nbytes=256 * 1024 * 1024)
    ep_offset = f["ep_offset"][:]
    ep_len = f["ep_len"][:]
    n_ep_total = len(ep_offset)

    local_rng = np.random.default_rng(seed)
    chosen_ep = local_rng.choice(n_ep_total, size=n_episodes, replace=False)
    chosen_ep.sort()

    model = load_tworoom_lewm(device=DEV)
    mean, std = IMAGENET_MEAN.to(DEV), IMAGENET_STD.to(DEV)

    all_z, all_ep, all_step, all_pos = [], [], [], []
    t0 = time.time()
    BS = 128
    for ei, e in enumerate(chosen_ep):
        s, L = int(ep_offset[e]), int(ep_len[e])
        pix = f["pixels"][s : s + L]
        pos = f["proprio"][s : s + L]
        steps = np.arange(L)

        z_chunks = []
        for b0 in range(0, L, BS):
            chunk = pix[b0 : b0 + BS]
            t = torch.from_numpy(chunk).to(DEV).permute(0, 3, 1, 2).float() / 255.0
            t = (t - mean) / std
            with torch.no_grad():
                out = model.encode({"pixels": t.unsqueeze(1)})
                z = out["emb"][:, 0]
            z_chunks.append(z.cpu().numpy())
        z_ep = np.concatenate(z_chunks, axis=0)

        all_z.append(z_ep)
        all_ep.append(np.full(L, e, dtype=np.int64))
        all_step.append(steps)
        all_pos.append(pos)

        if ei % 100 == 0:
            log(f"[encode] episode {ei}/{n_episodes} (ep_id={e}, len={L}) "
                f"cum_frames={sum(len(x) for x in all_ep)} elapsed={time.time()-t0:.1f}s")

    f.close()
    z = np.concatenate(all_z, axis=0).astype(np.float32)
    ep_idx = np.concatenate(all_ep, axis=0)
    step_idx = np.concatenate(all_step, axis=0)
    proprio = np.concatenate(all_pos, axis=0).astype(np.float32)

    log(f"[encode] done: {z.shape[0]} frames encoded in {time.time()-t0:.1f}s")
    np.savez(cache, z=z, ep_idx=ep_idx, step_idx=step_idx, proprio=proprio)
    return z, ep_idx, step_idx, proprio


def calibrate_rho(z, ep_idx, step_idx):
    order = np.lexsort((step_idx, ep_idx))
    ep_o, step_o = ep_idx[order], step_idx[order]
    adj = (ep_o[1:] == ep_o[:-1]) & (step_o[1:] == step_o[:-1] + 1)
    z_o = z[order]
    z_t, z_t1 = z_o[:-1][adj], z_o[1:][adj]
    rho_hat = float(np.mean(np.sum(z_t * z_t1, axis=1)) / D)
    return rho_hat


def lean_evaluate(z, proprio, graph, true_dist_oracle, n_sources=100, n_targets=40, seed=0):
    """Single-setting version of b0's evaluate(): one graph in, Spearman numbers out."""
    n = z.shape[0]
    local_rng = np.random.default_rng(seed)
    src_idx = local_rng.choice(n, size=n_sources, replace=False)
    D_graph_full = dijkstra(graph, indices=src_idx, directed=False)

    rows = []
    for si, s in enumerate(src_idx):
        tgt = local_rng.choice(n, size=n_targets, replace=False)
        tgt = tgt[tgt != s]
        true_d = true_dist_oracle(proprio[s : s + 1], proprio[tgt])[0]
        graph_d = D_graph_full[si, tgt]
        eucl_d = np.linalg.norm(z[s : s + 1] - z[tgt], axis=1)
        src_room = proprio[s, 0] < WALL_CENTER_X
        tgt_room = proprio[tgt, 0] < WALL_CENTER_X
        cross = src_room != tgt_room
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
        same_room=dict(euclidean=re_s, graph=rg_s),
        cross_room=dict(euclidean=re_c, graph=rg_c),
    )


# ============================================================
# B1.1 landmark-count scaling
# ============================================================

def b1_scaling():
    log("\n" + "=" * 60 + "\nB1.1: landmark-count scaling (50 / 300 / 1000 episodes)\n" + "=" * 60)
    true_dist_oracle = build_true_distance_oracle()
    results = {}
    for n_ep in [50, 300, 1000]:
        log(f"\n--- n_episodes={n_ep} ---")
        t0 = time.time()
        z, ep_idx, step_idx, proprio = load_landmarks(n_ep)
        n = z.shape[0]
        rho_hat = calibrate_rho(z, ep_idx, step_idx)
        eps2 = eps2_of(Q_DEFAULT, rho_hat)

        t_edges0 = time.time()
        trans_i, trans_j, id_i, id_j = find_graph_edges(z, ep_idx, step_idx, eps2)
        t_edges = time.time() - t_edges0

        graph = build_weighted_graph(n, trans_i, trans_j, id_i, id_j, id_weight=1.0)
        t_eval0 = time.time()
        ev = lean_evaluate(z, proprio, graph, true_dist_oracle)
        t_eval = time.time() - t_eval0

        avg_deg = 2 * len(id_i) / n
        log(f"n_landmarks={n} rho_hat={rho_hat:.4f} eps2={eps2:.2f} "
            f"n_id_edges={len(id_i)} avg_degree={avg_deg:.2f} "
            f"edge_scan_time={t_edges:.1f}s eval_time={t_eval:.1f}s total_time={time.time()-t0:.1f}s")
        log(f"  overall: eucl={ev['overall']['euclidean']:.4f} graph={ev['overall']['graph']:.4f}")
        log(f"  same-room: eucl={ev['same_room']['euclidean']:.4f} graph={ev['same_room']['graph']:.4f}")
        log(f"  cross-room: eucl={ev['cross_room']['euclidean']:.4f} graph={ev['cross_room']['graph']:.4f}")

        results[n_ep] = dict(
            n_landmarks=int(n), rho_hat=rho_hat, eps2=eps2,
            n_id_edges=int(len(id_i)), avg_degree=avg_deg,
            edge_scan_time_s=t_edges, eval_time_s=t_eval,
            evaluation=ev,
        )

    # extrapolate to Push-T scale (2.3M frames) assuming the observed n_id_edges-vs-n
    # power-law trend holds (it need not -- Push-T's physical state density differs;
    # flagged explicitly, this is a Two-Room-only extrapolation)
    ns = np.array([results[k]["n_landmarks"] for k in [50, 300, 1000]], dtype=float)
    es = np.array([results[k]["n_id_edges"] for k in [50, 300, 1000]], dtype=float)
    es_safe = np.maximum(es, 1)
    slope, intercept = np.polyfit(np.log(ns), np.log(es_safe), 1)
    pusht_n = 2_300_000
    pusht_edges_pred = float(np.exp(intercept) * pusht_n ** slope)
    log(f"\n[scaling] n_id_edges ~ n_landmarks^{slope:.2f} (fit on 50/300/1000)")
    log(f"[scaling] extrapolated identification edges at Push-T's {pusht_n:,} frames: "
        f"~{pusht_edges_pred:,.3g} (Two-Room density assumption -- see note above)")

    results["scaling_fit"] = dict(power=float(slope), pusht_frame_count=pusht_n,
                                   pusht_extrapolated_edges=pusht_edges_pred)
    return results


# ============================================================
# B1.2 successor-consistency filter
# ============================================================

def b1_successor_consistency():
    log("\n" + "=" * 60 + "\nB1.2: successor-consistency filter\n" + "=" * 60)
    true_dist_oracle = build_true_distance_oracle()
    z, ep_idx, step_idx, proprio = load_landmarks(300)
    n = z.shape[0]
    rho_hat = calibrate_rho(z, ep_idx, step_idx)
    eps2 = eps2_of(Q_DEFAULT, rho_hat)

    # pull real recorded actions for every landmark frame directly from the h5
    # (cheap: no re-encoding, just indexing a stored column)
    f = h5py.File(H5_PATH, "r", swmr=True)
    ep_offset = f["ep_offset"][:]
    ep_len = f["ep_len"][:]
    global_idx = ep_offset[ep_idx] + step_idx
    action = f["action"][:][global_idx]  # (n, 2)
    has_successor = step_idx < (ep_len[ep_idx] - 1)
    # successor's global row, valid only where has_successor
    succ_local = np.where(has_successor, np.arange(n), -1)  # placeholder, filled below
    # build a lookup: (ep_idx, step_idx) -> local landmark row index
    key = ep_idx.astype(np.int64) * 100000 + step_idx.astype(np.int64)
    key_to_row = {int(k): i for i, k in enumerate(key)}
    succ_row = np.full(n, -1, dtype=np.int64)
    for i in range(n):
        if has_successor[i]:
            k = int(ep_idx[i]) * 100000 + int(step_idx[i]) + 1
            succ_row[i] = key_to_row.get(k, -1)  # -1 if successor wasn't sampled as a landmark
    f.close()

    trans_i, trans_j, id_i, id_j = find_graph_edges(z, ep_idx, step_idx, eps2)
    log(f"[filter] starting identification edges: {len(id_i)}")

    # action similarity per identification edge
    a_i, a_j = action[id_i], action[id_j]
    an_i = a_i / (np.linalg.norm(a_i, axis=1, keepdims=True) + 1e-8)
    an_j = a_j / (np.linalg.norm(a_j, axis=1, keepdims=True) + 1e-8)
    action_cos = np.sum(an_i * an_j, axis=1)
    n_nan_action = int(np.isnan(action_cos).sum())
    log(f"[filter] action-cosine over identification edges ({n_nan_action} NaN -- raw "
        f"'action' column has NaNs at episode boundaries, per jepa.py's own nan_to_num "
        f"convention; excluded from the stats below and, since NaN>thresh is False in "
        f"numpy, correctly excluded from 'similar_action' i.e. treated as untestable):")
    log(f"  mean={np.nanmean(action_cos):.3f} median={np.nanmedian(action_cos):.3f} "
        f"p10={np.nanpercentile(action_cos,10):.3f} p90={np.nanpercentile(action_cos,90):.3f}")

    ACTION_COS_THRESH = 0.7
    both_succ = (succ_row[id_i] >= 0) & (succ_row[id_j] >= 0)
    similar_action = action_cos > ACTION_COS_THRESH
    testable = both_succ & similar_action
    log(f"[filter] testable edges (both have a sampled successor AND action_cos>{ACTION_COS_THRESH}): "
        f"{testable.sum()}/{len(id_i)} ({100*testable.sum()/len(id_i):.1f}%)")

    si, sj = succ_row[id_i[testable]], succ_row[id_j[testable]]
    succ_sqd = np.sum((z[si] - z[sj]) ** 2, axis=1)
    passes = succ_sqd < eps2
    log(f"[filter] of testable edges, successor test passes: {passes.sum()}/{testable.sum()} "
        f"({100*passes.mean():.1f}%) -- fails (dropped): {(~passes).sum()}")

    drop_mask = np.zeros(len(id_i), dtype=bool)
    testable_idx = np.nonzero(testable)[0]
    drop_mask[testable_idx[~passes]] = True
    keep_mask = ~drop_mask

    id_i_f, id_j_f = id_i[keep_mask], id_j[keep_mask]
    log(f"[filter] identification edges after filtering: {len(id_i_f)} "
        f"({100*len(id_i_f)/len(id_i):.1f}% kept, {drop_mask.sum()} dropped)")

    def measure_false_edge_rate(ei, ej, n_sample=3000, dist_thresh_px=12.0):
        n_avail = len(ei)
        if n_avail == 0:
            return float("nan"), 0
        sample = rng.choice(n_avail, size=min(n_sample, n_avail), replace=False)
        td = np.array([
            true_dist_oracle(proprio[ei[k]:ei[k]+1], proprio[ej[k]:ej[k]+1])[0, 0]
            for k in sample
        ])
        return float(np.mean(td > dist_thresh_px)), len(sample)

    false_rate_before, n_before = measure_false_edge_rate(id_i, id_j)
    false_rate_after, n_after = measure_false_edge_rate(id_i_f, id_j_f)
    log(f"[filter] measured false-edge rate (true dist > 12px): "
        f"before={false_rate_before*100:.1f}% (n={n_before})  after={false_rate_after*100:.1f}% (n={n_after})")

    graph_before = build_weighted_graph(n, trans_i, trans_j, id_i, id_j, id_weight=1.0)
    graph_after = build_weighted_graph(n, trans_i, trans_j, id_i_f, id_j_f, id_weight=1.0)
    ev_before = lean_evaluate(z, proprio, graph_before, true_dist_oracle)
    ev_after = lean_evaluate(z, proprio, graph_after, true_dist_oracle)
    log(f"[filter] Spearman before: overall={ev_before['overall']['graph']:.4f} "
        f"same={ev_before['same_room']['graph']:.4f} cross={ev_before['cross_room']['graph']:.4f}")
    log(f"[filter] Spearman after:  overall={ev_after['overall']['graph']:.4f} "
        f"same={ev_after['same_room']['graph']:.4f} cross={ev_after['cross_room']['graph']:.4f}")

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
    z, ep_idx, step_idx, proprio = load_landmarks(300)
    n = z.shape[0]
    rho_hat = calibrate_rho(z, ep_idx, step_idx)

    order = np.lexsort((step_idx, ep_idx))
    ep_o, step_o = ep_idx[order], step_idx[order]
    tm = (ep_o[1:] == ep_o[:-1]) & (step_o[1:] == step_o[:-1] + 1)
    trans_i, trans_j = order[:-1][tm], order[1:][tm]

    # cross-room query pairs, fixed across q, for a reachability check
    cross_pairs = []
    left = np.nonzero(proprio[:, 0] < WALL_CENTER_X)[0]
    right = np.nonzero(proprio[:, 0] >= WALL_CENTER_X)[0]
    for _ in range(500):
        i = rng.choice(left); j = rng.choice(right)
        cross_pairs.append((i, j))

    results = {}
    for q in [0.5, 0.2, 0.05, 0.01, 1e-3, 1e-4, 1e-6, 1e-9]:
        eps2 = eps2_of(q, rho_hat)
        _, _, id_i, id_j = find_graph_edges(z, ep_idx, step_idx, eps2)
        graph = build_weighted_graph(n, trans_i, trans_j, id_i, id_j, id_weight=1.0)

        n_comp, labels = connected_components(graph, directed=False)
        sizes = np.bincount(labels)
        largest_frac = sizes.max() / n

        reachable = np.mean([labels[i] == labels[j] for i, j in cross_pairs])

        log(f"q={q:g}: n_id_edges={len(id_i)} n_components={n_comp} "
            f"largest_component={largest_frac*100:.2f}% cross_room_reachable={reachable*100:.1f}%")
        results[str(q)] = dict(
            eps2=eps2, n_id_edges=int(len(id_i)), n_components=int(n_comp),
            largest_component_frac=float(largest_frac), cross_room_reachable_frac=float(reachable),
        )
    return results


# ============================================================
# B1.4 degree distribution
# ============================================================

def b1_degree_distribution():
    log("\n" + "=" * 60 + "\nB1.4: identification-edge degree distribution\n" + "=" * 60)
    z, ep_idx, step_idx, proprio = load_landmarks(300)
    n = z.shape[0]
    rho_hat = calibrate_rho(z, ep_idx, step_idx)
    eps2 = eps2_of(Q_DEFAULT, rho_hat)
    _, _, id_i, id_j = find_graph_edges(z, ep_idx, step_idx, eps2)

    deg = np.zeros(n, dtype=np.int64)
    np.add.at(deg, id_i, 1)
    np.add.at(deg, id_j, 1)

    pct = {p: float(np.percentile(deg, p)) for p in [50, 90, 99, 99.9]}
    log(f"[degree] mean={deg.mean():.1f} p50={pct[50]:.0f} p90={pct[90]:.0f} "
        f"p99={pct[99]:.0f} p99.9={pct[99.9]:.0f} max={deg.max()}")

    hub_thresh = max(pct[99.9] * 5, pct[99] * 10)
    hubs = np.nonzero(deg > hub_thresh)[0]
    log(f"[degree] outlier hub threshold={hub_thresh:.0f}: {len(hubs)} nodes flagged")
    hub_info = []
    for h in hubs[:10]:
        log(f"  hub node {h}: degree={deg[h]}, proprio=({proprio[h,0]:.1f},{proprio[h,1]:.1f}), "
            f"ep={ep_idx[h]}, step={step_idx[h]}")
        hub_info.append(dict(node=int(h), degree=int(deg[h]),
                              proprio=[float(proprio[h,0]), float(proprio[h,1])]))

    return dict(mean=float(deg.mean()), percentiles=pct, max=int(deg.max()),
                n_hubs=int(len(hubs)), hub_threshold=float(hub_thresh), hubs=hub_info)


def main():
    log(f"device={DEV}")

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
