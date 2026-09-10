"""B0 gate check for the latent-graph idea (Two-Room, real data, real pretrained checkpoint).

Question: does Dijkstra over a graph built from LeWM's Two-Room latents
(transition edges from real trajectories + identification edges from a
calibrated one-step-displacement threshold) track the TRUE wall-respecting
navigation distance better than raw latent Euclidean distance?

Everything here is measured, not assumed:

  - Checkpoint health: verified separately (scripts/tworoom_lewm_loader.py +
    an off-diagonal cosine-similarity check) before this script was written —
    the pretrained quentinll/lewm-tworooms checkpoint is NOT collapsed
    (off-diag cosine ~0.03, varying norms), unlike the local epoch-1 checkpoint.

  - Wall/door geometry: NOT taken from env source defaults. Verified directly
    from real proprio trajectories in tworoom.h5 (all 10,000 episodes):
    the door/wall config in the `observation` column is bit-identical across
    every episode sampled (first/mid/last step) -> a single fixed geometry
    for the whole dataset. That geometry's exact extent (wall x-band,
    door y-gap) was then read off empirically from where the agent's real
    (x, y) trace does / doesn't go, not from the env's init_value constants:
        wall band (impassable except through door): x in [100, 124]
        door gap (passable within the wall band):    y in [33.25, 64.75]
        room bounds:                                  x, y in [14, 209]

  - Calibration threshold for identification edges: rho_hat estimated from
    real consecutive-frame latent pairs (same construction as v(s,a)'s
    conditional-displacement null in docs/paper/main.tex sec:v), not assumed.

Ground truth distance: Dijkstra shortest path on a fine free-space grid built
from the geometry above, evaluated on real recorded (x, y) positions. This is
independent of the latent entirely.
"""

import json
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from scipy import stats as sps
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.graph_lib import DEV, build_weighted_graph, calibrate, find_graph_edges  # noqa: F401
from common.log_util import log  # noqa: F401
from tworoom_lewm_loader import load_tworoom_lewm

ROOT = Path(__file__).resolve().parent.parent
# LEWM_H5_PATH overrides the dataset location -- unset by default, so local runs are
# unchanged. Set this on a cluster to point at a node-local scratch copy (e.g. Ada's
# /ssd_scratch) instead of a shared/network filesystem, since concurrent array-job tasks
# on different nodes each need their OWN copy path, not one shared location.
H5_PATH = Path(os.environ.get("LEWM_H5_PATH", str(ROOT / "data" / "hf_dl" / "tworoom_extracted" / "tworoom.h5")))
OUT_DIR = ROOT / "outputs" / "b0_tworoom"
OUT_DIR.mkdir(parents=True, exist_ok=True)

torch.manual_seed(0)
np.random.seed(0)
rng = np.random.default_rng(0)

# ---- verified geometry (see module docstring) ----
WALL_X = (100.0, 124.0)
DOOR_Y = (33.25, 64.75)
ROOM_LO, ROOM_HI = 14.0, 209.0
WALL_CENTER_X = 112.0

import os
N_EPISODES = int(os.environ.get("B0_N_EPISODES", 300))  # landmark episodes (full trajectories encoded)
GRID_RES = 1.0            # pixels per grid cell for the true-distance oracle
N_QUERY_SOURCES = 100      # Dijkstra source landmarks on the latent graph
N_TARGETS_PER_SOURCE = 40  # sampled targets per source for the correlation set
QUANTILES = [0.5, 0.01, 1e-3, 1e-6]  # left-tail quantile of the adjacent-pair null; see calibrate()

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


# ============================================================
# Stage 1: ground-truth wall-respecting distance oracle
# ============================================================

def build_true_distance_oracle():
    xs = np.arange(ROOM_LO, ROOM_HI, GRID_RES)
    ys = np.arange(ROOM_LO, ROOM_HI, GRID_RES)
    nx, ny = len(xs), len(ys)

    def is_free(x, y):
        in_wall_band = (x >= WALL_X[0]) & (x <= WALL_X[1])
        in_door = (y >= DOOR_Y[0]) & (y <= DOOR_Y[1])
        return ~(in_wall_band & ~in_door)

    XX, YY = np.meshgrid(xs, ys, indexing="ij")  # (nx, ny)
    free = is_free(XX, YY)
    log(f"[true-dist] grid {nx}x{ny}, free cells {free.sum()}/{free.size}")

    node_id = -np.ones((nx, ny), dtype=np.int64)
    node_id[free] = np.arange(free.sum())
    n_nodes = free.sum()

    rows, cols, weights = [], [], []
    neighbors = [(1, 0), (0, 1), (1, 1), (1, -1)]  # undirected, add both dirs
    for dx, dy in neighbors:
        i0, i1 = max(0, -dx), nx - max(0, dx)
        j0, j1 = max(0, -dy), ny - max(0, dy)
        a = node_id[i0:i1, j0:j1]
        b = node_id[i0 + dx : i1 + dx, j0 + dy : j1 + dy]
        both_free = (a >= 0) & (b >= 0)
        w = GRID_RES * np.hypot(dx, dy)
        rows.append(a[both_free]); cols.append(b[both_free]); weights.append(np.full(both_free.sum(), w))
        rows.append(b[both_free]); cols.append(a[both_free]); weights.append(np.full(both_free.sum(), w))

    rows = np.concatenate(rows); cols = np.concatenate(cols); weights = np.concatenate(weights)
    graph = coo_matrix((weights, (rows, cols)), shape=(n_nodes, n_nodes)).tocsr()
    log(f"[true-dist] grid graph built: {n_nodes} nodes, {graph.nnz} directed edges")

    def snap(pos_xy):
        # pos_xy: (N,2) real proprio positions -> nearest free grid node id
        ix = np.clip(np.round((pos_xy[:, 0] - ROOM_LO) / GRID_RES).astype(int), 0, nx - 1)
        iy = np.clip(np.round((pos_xy[:, 1] - ROOM_LO) / GRID_RES).astype(int), 0, ny - 1)
        nid = node_id[ix, iy]
        # if the exact snapped cell is blocked (agent radius / near-wall rounding), search a small ring
        bad = np.nonzero(nid < 0)[0]
        for k in bad:
            found = False
            for r in range(1, 6):
                for ddx in range(-r, r + 1):
                    for ddy in range(-r, r + 1):
                        xi, yi = ix[k] + ddx, iy[k] + ddy
                        if 0 <= xi < nx and 0 <= yi < ny and node_id[xi, yi] >= 0:
                            nid[k] = node_id[xi, yi]
                            found = True
                            break
                    if found:
                        break
                if found:
                    break
        return nid

    def true_dist_from_sources(source_pos_xy, target_pos_xy):
        """source_pos_xy: (S,2), target_pos_xy: (T,2) -> (S,T) distance matrix."""
        src_nodes = snap(source_pos_xy)
        tgt_nodes = snap(target_pos_xy)
        D = dijkstra(graph, indices=src_nodes, directed=False)
        return D[:, tgt_nodes]

    return true_dist_from_sources


# ============================================================
# Stage 2: load episodes + encode with the frozen pretrained model
# ============================================================

def load_landmarks():
    cache = OUT_DIR / f"landmarks_ep{N_EPISODES}.npz"
    if cache.exists():
        log(f"[encode] loading cached landmarks from {cache}")
        d = np.load(cache)
        return d["z"], d["ep_idx"], d["step_idx"], d["proprio"]

    f = h5py.File(H5_PATH, "r", swmr=True, rdcc_nbytes=256 * 1024 * 1024)
    ep_offset = f["ep_offset"][:]
    ep_len = f["ep_len"][:]
    n_ep_total = len(ep_offset)

    chosen_ep = rng.choice(n_ep_total, size=N_EPISODES, replace=False)
    chosen_ep.sort()

    model = load_tworoom_lewm(device=DEV)
    mean, std = IMAGENET_MEAN.to(DEV), IMAGENET_STD.to(DEV)

    all_z, all_ep, all_step, all_pos = [], [], [], []
    t0 = time.time()
    BS = 128
    for ei, e in enumerate(chosen_ep):
        s, L = int(ep_offset[e]), int(ep_len[e])
        pix = f["pixels"][s : s + L]           # (L,224,224,3) uint8
        pos = f["proprio"][s : s + L]          # (L,2)
        steps = np.arange(L)

        z_chunks = []
        for b0 in range(0, L, BS):
            chunk = pix[b0 : b0 + BS]
            t = torch.from_numpy(chunk).to(DEV).permute(0, 3, 1, 2).float() / 255.0
            t = (t - mean) / std
            with torch.no_grad():
                out = model.encode({"pixels": t.unsqueeze(1)})
                z = out["emb"][:, 0]  # (b,192)
            z_chunks.append(z.cpu().numpy())
        z_ep = np.concatenate(z_chunks, axis=0)

        all_z.append(z_ep)
        all_ep.append(np.full(L, e, dtype=np.int64))
        all_step.append(steps)
        all_pos.append(pos)

        if ei % 50 == 0:
            log(f"[encode] episode {ei}/{N_EPISODES} (ep_id={e}, len={L}) "
                f"cum_frames={sum(len(x) for x in all_ep)} elapsed={time.time()-t0:.1f}s")

    f.close()
    z = np.concatenate(all_z, axis=0).astype(np.float32)
    ep_idx = np.concatenate(all_ep, axis=0)
    step_idx = np.concatenate(all_step, axis=0)
    proprio = np.concatenate(all_pos, axis=0).astype(np.float32)

    log(f"[encode] done: {z.shape[0]} frames encoded in {time.time()-t0:.1f}s")
    np.savez(cache, z=z, ep_idx=ep_idx, step_idx=step_idx, proprio=proprio)
    return z, ep_idx, step_idx, proprio


# ============================================================
# Stage 3/4: calibration + graph construction -- moved to common/graph_lib.py
# (calibrate, find_graph_edges, build_weighted_graph imported at module top)
# ============================================================

# ============================================================
# Stage 5: evaluation — Spearman(true, raw-euclidean) vs Spearman(true, graph-geodesic)
# ============================================================

ID_WEIGHT_SWEEP = [0.0, 0.5, 1.0, 2.0, 4.0]


def evaluate(z, ep_idx, step_idx, proprio, trans_i, trans_j, id_i, id_j, true_dist_oracle):
    n = z.shape[0]
    src_idx = rng.choice(n, size=N_QUERY_SOURCES, replace=False)

    tgt_by_src = {}
    for s in src_idx:
        tgt = rng.choice(n, size=N_TARGETS_PER_SOURCE, replace=False)
        tgt_by_src[s] = tgt[tgt != s]

    # ground truth + raw euclidean don't depend on id_weight -> compute once
    true_by_src, eucl_by_src, cross_by_src = {}, {}, {}
    for s in src_idx:
        tgt_idx = tgt_by_src[s]
        true_by_src[s] = true_dist_oracle(proprio[s : s + 1], proprio[tgt_idx])[0]
        eucl_by_src[s] = np.linalg.norm(z[s : s + 1] - z[tgt_idx], axis=1)
        src_room = proprio[s, 0] < WALL_CENTER_X
        tgt_room = proprio[tgt_idx, 0] < WALL_CENTER_X
        cross_by_src[s] = src_room != tgt_room

    all_results = {}
    for id_weight in ID_WEIGHT_SWEEP:
        graph = build_weighted_graph(n, trans_i, trans_j, id_i, id_j, id_weight)
        D_graph_full = dijkstra(graph, indices=src_idx, directed=False)  # (S, n)

        rows = []
        for si, s in enumerate(src_idx):
            tgt_idx = tgt_by_src[s]
            graph_d = D_graph_full[si, tgt_idx]
            for k in range(len(tgt_idx)):
                rows.append((true_by_src[s][k], eucl_by_src[s][k], graph_d[k], bool(cross_by_src[s][k])))

        true_d = np.array([r[0] for r in rows])
        eucl_d = np.array([r[1] for r in rows])
        graph_d = np.array([r[2] for r in rows])
        cross_room = np.array([r[3] for r in rows])

        finite = np.isfinite(true_d) & np.isfinite(graph_d) & np.isfinite(eucl_d)
        n_dropped = int((~finite).sum())
        true_d, eucl_d, graph_d, cross_room = true_d[finite], eucl_d[finite], graph_d[finite], cross_room[finite]

        def report(mask, name):
            if mask.sum() < 10:
                log(f"[eval id_weight={id_weight}] {name}: too few pairs ({mask.sum()}), skipping")
                return None
            rho_eucl, p_eucl = sps.spearmanr(true_d[mask], eucl_d[mask])
            rho_graph, p_graph = sps.spearmanr(true_d[mask], graph_d[mask])
            log(f"[eval id_weight={id_weight}] {name} (n={mask.sum()}): "
                f"Spearman(true, raw-euclidean)={rho_eucl:.4f} (p={p_eucl:.1e})  "
                f"Spearman(true, graph-geodesic)={rho_graph:.4f} (p={p_graph:.1e})")
            return dict(n=int(mask.sum()), spearman_euclidean=float(rho_eucl), spearman_graph=float(rho_graph))

        results = dict(n_dropped_nonfinite=n_dropped)
        results["overall"] = report(np.ones_like(cross_room, dtype=bool), "overall")
        results["same_room"] = report(~cross_room, "same-room")
        results["cross_room"] = report(cross_room, "cross-room (must cross the door)")
        all_results[str(id_weight)] = results

    return all_results


def main():
    log(f"device={DEV}")
    log("=== stage 1: true-distance oracle ===")
    true_dist_oracle = build_true_distance_oracle()

    log("=== stage 2: encode landmarks ===")
    z, ep_idx, step_idx, proprio = load_landmarks()
    log(f"landmarks: {z.shape[0]} frames from {len(np.unique(ep_idx))} episodes")

    log("=== stage 3: calibration ===")
    rho_hat, thresholds, adj_sqdisp, unrel_sqdisp = calibrate(z, ep_idx, step_idx, quantiles=QUANTILES, rng=rng)

    q_use = 1e-3
    eps2 = thresholds[q_use]
    log(f"=== stage 4: find latent graph edges (q={q_use:g}, eps^2={eps2:.2f}) ===")
    trans_i, trans_j, id_i, id_j = find_graph_edges(z, ep_idx, step_idx, eps2)

    log("=== stage 5: evaluate (sweeping identification-edge weight) ===")
    results = evaluate(z, ep_idx, step_idx, proprio, trans_i, trans_j, id_i, id_j, true_dist_oracle)

    summary = dict(
        n_episodes=N_EPISODES, n_landmarks=int(z.shape[0]),
        rho_hat=rho_hat, quantile_used=q_use, eps2_used=eps2,
        n_transition_edges=int(len(trans_i)), n_identification_edges=int(len(id_i)),
        thresholds={str(k): v for k, v in thresholds.items()},
        id_weight_sweep=results,
    )
    out_path = OUT_DIR / "b0_results.json"
    out_path.write_text(json.dumps(summary, indent=2))
    log(f"\nwrote {out_path}")
    log(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
