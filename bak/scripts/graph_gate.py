"""Unified B0-style gate check: does Dijkstra over a latent graph (transition edges from
real trajectories + identification edges) track the TRUE oracle distance better than raw
latent Euclidean distance? Config-driven consolidation of tworoom_b0_graph_gate.py/
pusht_b0_graph_gate.py -- see docs/graph-proposal/main.tex for the methodology writeup.

Run as:
    python scripts/graph_gate.py env=tworoom
    python scripts/graph_gate.py env=pusht
    python scripts/graph_gate.py env=pusht env.n_episodes=50   # override any env field

Each environment's identification-edge construction method (env.id_edge_method:
"bruteforce" or "capped") is a config choice, not hardcoded -- see config/graph/env/*.yaml
for why the two environments default to different methods (Two-Room's calibration always
succeeds, so its historical numbers used exact brute force; Push-T's calibration fails
outright, and brute force is a scaling wall at this size regardless).

NOTE: transition edges are now directed (common.graph_lib.build_weighted_graph), so
Two-Room's historical b0_results.json numbers (0.9515333078449236/0.43939877995042154) are
no longer bit-exactly reproducible -- Two-Room's transitions are only "roughly" reversible,
so the numbers move slightly, not qualitatively.
"""

import json
import sys
import time
from pathlib import Path

import h5py
import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from scipy import stats as sps
from scipy.sparse.csgraph import dijkstra

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.envs import ENV_MECHANICS
from common.graph_lib import DEV, build_graph_edges_for_env, build_weighted_graph, calibrate
from common.lewm_loader import load_lewm
from common.log_util import log

ROOT = Path(__file__).resolve().parent.parent
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def load_landmarks(mech, h5_path, out_dir, n_episodes, ckpt_dir, seed=0):
    cache = out_dir / f"landmarks_ep{n_episodes}_seed{seed}.npz"
    legacy_cache = out_dir / f"landmarks_ep{n_episodes}.npz"  # Two-Room's original naming (no
    # seed suffix -- it only ever used seed 0), kept as a fallback lookup so already-computed
    # landmarks are reused rather than re-encoded.
    for candidate in ([cache, legacy_cache] if seed == 0 else [cache]):
        if candidate.exists():
            log(f"[encode] loading cached landmarks from {candidate}")
            d = np.load(candidate)
            state_key = mech.state_h5_key if mech.state_h5_key in d else "state"
            return d["z"], d["ep_idx"], d["step_idx"], d[state_key]

    f = h5py.File(h5_path, "r", swmr=True, rdcc_nbytes=256 * 1024 * 1024)
    ep_offset = f["ep_offset"][:]
    ep_len = f["ep_len"][:]
    n_ep_total = len(ep_offset)

    rng = np.random.default_rng(seed)
    chosen_ep = rng.choice(n_ep_total, size=n_episodes, replace=False)
    chosen_ep.sort()

    model = load_lewm(ckpt_dir=ckpt_dir, device=DEV)
    mean, std = IMAGENET_MEAN.to(DEV), IMAGENET_STD.to(DEV)

    all_z, all_ep, all_step, all_state = [], [], [], []
    t0 = time.time()
    BS = 128
    for ei, e in enumerate(chosen_ep):
        s, L = int(ep_offset[e]), int(ep_len[e])
        pix = f["pixels"][s: s + L]                 # (L,224,224,3) uint8
        st = mech.read_state_slice(f, s, L)          # (L, state_dim)
        steps = np.arange(L)

        z_chunks = []
        for b0 in range(0, L, BS):
            chunk = pix[b0: b0 + BS]
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
        all_state.append(st)

        if ei % 50 == 0:
            log(f"[encode] episode {ei}/{n_episodes} (ep_id={e}, len={L}) "
                f"cum_frames={sum(len(x) for x in all_ep)} elapsed={time.time()-t0:.1f}s")

    f.close()
    z = np.concatenate(all_z, axis=0).astype(np.float32)
    ep_idx = np.concatenate(all_ep, axis=0)
    step_idx = np.concatenate(all_step, axis=0)
    state = np.concatenate(all_state, axis=0).astype(np.float32)

    log(f"[encode] done: {z.shape[0]} frames encoded in {time.time()-t0:.1f}s")
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, z=z, ep_idx=ep_idx, step_idx=step_idx, **{mech.state_h5_key: state})
    return z, ep_idx, step_idx, state


def evaluate(z, state, trans_i, trans_j, id_i, id_j, true_dist_oracle, cfg, rng):
    n = z.shape[0]
    src_idx = rng.choice(n, size=cfg.n_query_sources, replace=False)

    tgt_by_src = {}
    for s in src_idx:
        tgt = rng.choice(n, size=cfg.n_targets_per_source, replace=False)
        tgt_by_src[s] = tgt[tgt != s]

    true_by_src, eucl_by_src = {}, {}
    for s in src_idx:
        tgt_idx = tgt_by_src[s]
        true_by_src[s] = true_dist_oracle(state[s: s + 1], state[tgt_idx])[0]
        eucl_by_src[s] = np.linalg.norm(z[s: s + 1] - z[tgt_idx], axis=1)

    all_results = {}
    for id_weight in cfg.id_weight_sweep:
        graph = build_weighted_graph(n, trans_i, trans_j, id_i, id_j, id_weight)
        D_graph_full = dijkstra(graph, indices=src_idx, directed=True)  # (S, n)

        rows = []
        for si, s in enumerate(src_idx):
            tgt_idx = tgt_by_src[s]
            graph_d = D_graph_full[si, tgt_idx]
            for k in range(len(tgt_idx)):
                rows.append((true_by_src[s][k], eucl_by_src[s][k], graph_d[k]))

        true_d = np.array([r[0] for r in rows])
        eucl_d = np.array([r[1] for r in rows])
        graph_d = np.array([r[2] for r in rows])

        finite = np.isfinite(true_d) & np.isfinite(graph_d) & np.isfinite(eucl_d)
        n_dropped = int((~finite).sum())
        true_d, eucl_d, graph_d = true_d[finite], eucl_d[finite], graph_d[finite]

        rho_eucl, p_eucl = sps.spearmanr(true_d, eucl_d)
        rho_graph, p_graph = sps.spearmanr(true_d, graph_d)
        log(f"[eval id_weight={id_weight}] (n={len(true_d)}): "
            f"Spearman(true, raw-euclidean)={rho_eucl:.4f} (p={p_eucl:.1e})  "
            f"Spearman(true, graph-geodesic)={rho_graph:.4f} (p={p_graph:.1e})")
        all_results[str(id_weight)] = dict(
            n=len(true_d), n_dropped_nonfinite=n_dropped,
            spearman_euclidean=float(rho_eucl), spearman_graph=float(rho_graph),
        )

    return all_results


@hydra.main(version_base=None, config_path="../config/graph", config_name="graph_gate")
def main(cfg: DictConfig):
    mech = ENV_MECHANICS[cfg.env.name]
    out_dir = ROOT / "outputs" / f"b0_{cfg.env.output_prefix}"
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(0)
    np.random.seed(0)
    rng = np.random.default_rng(0)

    log(f"device={DEV} env={cfg.env.name}")
    log("=== stage 2: encode landmarks ===")
    z, ep_idx, step_idx, state = load_landmarks(
        mech, mech.h5_path(ROOT), out_dir, cfg.env.n_episodes, mech.ckpt_dir(ROOT),
    )
    log(f"landmarks: {z.shape[0]} frames from {len(np.unique(ep_idx))} episodes")

    log("=== stage 1: true-distance oracle ===")
    true_dist_oracle = mech.build_true_distance_oracle(state)

    log("=== stage 3: calibration ===")
    calib_rng = rng if cfg.env.share_calibration_rng else None
    rho_hat, thresholds, adj_sqdisp, unrel_sqdisp = calibrate(
        z, ep_idx, step_idx, quantiles=(0.5, 0.01, 1e-3, 1e-6), rng=calib_rng,
    )
    q_use = float(cfg.q_calibration)
    eps2 = thresholds[q_use]

    log(f"=== stage 4: identification edges (method={cfg.env.id_edge_method}) ===")
    # Re-derives rho_hat/eps2 internally (cheap -- pure numpy on already-encoded landmarks)
    # rather than reusing stage 3's -- keeps this dispatch identical to what datatiers.py
    # calls, and stage 3's own calibrate() call above is what actually needs to have
    # consumed `rng` (for tworoom's share_calibration_rng) before evaluate() runs; this
    # second internal call always uses its own fresh rng, so it doesn't perturb that state.
    trans_i, trans_j, id_i, id_j, rho_hat, eps2 = build_graph_edges_for_env(
        cfg.env, z, ep_idx, step_idx, q=q_use,
    )

    log("=== stage 5: evaluate (sweeping identification-edge weight) ===")
    results = evaluate(z, state, trans_i, trans_j, id_i, id_j, true_dist_oracle, cfg, rng)

    summary = dict(
        env=cfg.env.name, n_episodes=cfg.env.n_episodes, n_landmarks=int(z.shape[0]),
        id_edge_method=cfg.env.id_edge_method, rho_hat=rho_hat, quantile_used=q_use,
        eps2_used=eps2,
        n_transition_edges=int(len(trans_i)), n_identification_edges=int(len(id_i)),
        thresholds={str(k): v for k, v in thresholds.items()},
        id_weight_sweep=results,
    )
    out_path = out_dir / "b0_results.json"
    out_path.write_text(json.dumps(summary, indent=2))
    log(f"\nwrote {out_path}")
    log(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
