"""Splice-test candidate-pool-sparsity check: does the splice test's conclusion change when
the candidate pool comes from a smaller, expert_1000-sized (1000-episode) landmark set,
while still scoring with the ALREADY-VALIDATED full-scale IDM ensemble (87.7% held-out
action-recovery reduction, trained on all 16,685 training episodes -- see
idm_fullscale_ensemble.pt) rather than retraining a new, weaker IDM on only 1000 episodes?

This isolates candidate-pool density from IDM quality: the full-scale splice test used a
235,400-row validation pool and found candidates naturally sit close (median 0.887, near
real single-step scale) with no working splice signal (IDM-vs-random 1.018x). A 1000-episode
pool has roughly half that many rows (~125,000) -- if that's still not sparse enough to
reproduce the earlier bad candidate-distance regime (seen at 30/300-episode local scales),
this isolates pool sparsity as the deciding factor (or rules it out, if candidates are still
close and the splice test still fails).

Uses episodes 0..999 (the same on-disk order convention as expert_N tiers elsewhere in this
pipeline) for BOTH the candidate pool and the "real reference" transitions, so everything is
scored within one consistent, expert_1000-sized pool.

Run as:
    python ada_pusht_splice_expert1000.py
"""

import json
from pathlib import Path

import h5py
import hydra
import numpy as np
import torch
from torch import nn

DEV = "cuda" if torch.cuda.is_available() else "cpu"
EMB_H5 = "/ssd_scratch/mayaank.ashok/lewm_pusht_emb/pusht_expert_train_emb.h5"
PRED1_DIR = Path("/home2/mayaank.ashok/lewm_research/data/checkpoints/lewm_pusht_1step_predictor")
IDM_CKPT = Path("/home2/mayaank.ashok/lewm_research/idm_fullscale_ensemble.pt")
N_POOL_EPISODES = 1000


class MLP(nn.Module):
    def __init__(self, in_dim, hidden, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def build_transition_pairs(ep_offset, ep_len, episode_ids):
    i_list, j_list = [], []
    for e in episode_ids:
        s, L = int(ep_offset[e]), int(ep_len[e])
        if L < 2:
            continue
        rows = np.arange(s, s + L)
        i_list.append(rows[:-1])
        j_list.append(rows[1:])
    return np.concatenate(i_list), np.concatenate(j_list)


def ensemble_predict(models, emb, i_idx, j_idx, chunk=50_000):
    n = len(i_idx)
    out = np.empty((len(models), n, models[0].net[-1].out_features), dtype=np.float32)
    for lo in range(0, n, chunk):
        hi = min(lo + chunk, n)
        zi = torch.from_numpy(emb[i_idx[lo:hi]]).float().to(DEV)
        zj = torch.from_numpy(emb[j_idx[lo:hi]]).float().to(DEV)
        x = torch.cat([zi, zj], dim=-1)
        with torch.no_grad():
            preds = torch.stack([m(x) for m in models], dim=0)
        out[:, lo:hi] = preds.cpu().numpy()
    return out


def raw_topk_cross_episode_pairs(z, ep_idx, k=20, M=32):
    import faiss
    z32 = np.ascontiguousarray(z.astype(np.float32))
    n, d = z32.shape
    index = faiss.IndexHNSWFlat(d, M)
    index.hnsw.efConstruction = 200
    index.add(z32)
    index.hnsw.efSearch = max(64, 2 * k)
    _, I = index.search(z32, k + 1)
    rows, cols = [], []
    for i in range(n):
        for rank in range(1, k + 1):
            j = int(I[i, rank])
            if j >= 0 and ep_idx[j] != ep_idx[i]:
                a, b = (i, j) if i < j else (j, i)
                rows.append(a)
                cols.append(b)
    pairs = np.unique(np.stack([np.array(rows), np.array(cols)], axis=1), axis=0)
    return pairs[:, 0], pairs[:, 1]


def load_pred1():
    cfg = json.loads((PRED1_DIR / "config.json").read_text())
    model = hydra.utils.instantiate(cfg)
    sd = torch.load(PRED1_DIR / "weights_epoch_20.pt", map_location="cpu")
    model.load_state_dict(sd, strict=True)
    model = model.to(DEV).eval()
    model.requires_grad_(False)
    return model


def main():
    print(f"device={DEV}", flush=True)
    f = h5py.File(EMB_H5, "r", swmr=True, rdcc_nbytes=256 * 1024 * 1024)
    ep_offset = f["ep_offset"][:]
    ep_len = f["ep_len"][:]
    emb = f["emb"][:]
    action = f["action"][:]
    f.close()
    print("loaded emb/action arrays", flush=True)

    action_dim = action.shape[1]
    print(f"loading cached full-scale IDM ensemble from {IDM_CKPT}", flush=True)
    idm_models = []
    for sd in torch.load(IDM_CKPT, map_location=DEV):
        m = MLP(in_dim=384, hidden=256, out_dim=action_dim).to(DEV)
        m.load_state_dict(sd)
        idm_models.append(m.eval())

    print("loading pred1...", flush=True)
    model = load_pred1()

    def single_action_error(zi_idx, zj_idx, a1, chunk=50_000):
        n = len(zi_idx)
        errs = np.empty(n, dtype=np.float32)
        for lo in range(0, n, chunk):
            hi = min(lo + chunk, n)
            zi_t = torch.from_numpy(emb[zi_idx[lo:hi]]).float().to(DEV).unsqueeze(1)
            act_t = torch.from_numpy(a1[lo:hi]).float().to(DEV).unsqueeze(1)
            with torch.no_grad():
                act_emb = model.action_encoder(act_t)
                pred = model.predict(zi_t, act_emb)[:, -1].cpu().numpy()
            errs[lo:hi] = np.linalg.norm(pred - emb[zj_idx[lo:hi]], axis=1)
        return errs

    # --- expert_1000 pool: episodes 0..999, matching the expert_N tier convention ----------
    pool_episodes = np.arange(N_POOL_EPISODES)
    pool_rows = np.concatenate([
        np.arange(int(ep_offset[e]), int(ep_offset[e]) + int(ep_len[e])) for e in pool_episodes
    ])
    pool_ep_idx = np.concatenate([
        np.full(int(ep_len[e]), e) for e in pool_episodes
    ])
    print(f"expert_1000 pool: {len(pool_rows)} rows from {len(pool_episodes)} episodes", flush=True)

    real_i, real_j = build_transition_pairs(ep_offset, ep_len, pool_episodes)
    print(f"real transitions in pool: {len(real_i)}", flush=True)

    local_cand_i, local_cand_j = raw_topk_cross_episode_pairs(emb[pool_rows], pool_ep_idx, k=20)
    cand_i, cand_j = pool_rows[local_cand_i], pool_rows[local_cand_j]
    print(f"[candidates] {len(cand_i)} cross-episode pairs from top-20 kNN", flush=True)
    cand_dist = np.linalg.norm(emb[cand_i] - emb[cand_j], axis=1)
    print(f"[candidates] gap: mean={cand_dist.mean():.3f} median={np.median(cand_dist):.3f}",
          flush=True)

    n_sub = min(100_000, len(cand_i))
    sub = np.random.default_rng(0).choice(len(cand_i), size=n_sub, replace=False)
    cand_i, cand_j, cand_dist = cand_i[sub], cand_j[sub], cand_dist[sub]
    print(f"[candidates] subsampled to {len(cand_i)} pairs for scoring", flush=True)

    rng = np.random.default_rng(0)
    a_idm = ensemble_predict(idm_models, emb, cand_i, cand_j).mean(axis=0).astype(np.float32)
    rand_a1 = action[real_i[rng.integers(0, len(real_i), size=len(cand_i))]]

    hybrid_err = single_action_error(cand_i, cand_j, a_idm)
    random_err = single_action_error(cand_i, cand_j, rand_a1)
    hybrid_ratio = hybrid_err / np.maximum(cand_dist, 1e-8)
    random_ratio = random_err / np.maximum(cand_dist, 1e-8)

    real_err = single_action_error(real_i, real_j, action[real_i])
    real_dist = np.linalg.norm(emb[real_i] - emb[real_j], axis=1)
    real_ratio = real_err / np.maximum(real_dist, 1e-8)

    print(f"[splice] IDM error: mean={hybrid_err.mean():.3f} median={np.median(hybrid_err):.3f} "
          f"ratio median={np.median(hybrid_ratio):.3f} frac<1={np.mean(hybrid_ratio < 1):.3f}",
          flush=True)
    print(f"[splice] random error: mean={random_err.mean():.3f} median={np.median(random_err):.3f} "
          f"ratio median={np.median(random_ratio):.3f} frac<1={np.mean(random_ratio < 1):.3f}",
          flush=True)
    print(f"[splice] real reference: mean={real_err.mean():.3f} median={np.median(real_err):.3f} "
          f"ratio median={np.median(real_ratio):.3f} frac<1={np.mean(real_ratio < 1):.3f}",
          flush=True)
    print(f"[RESULT expert_1000, full-scale IDM] IDM-vs-random gap: "
          f"{random_err.mean()/hybrid_err.mean():.3f}x (mean error ratio); "
          f"frac<1 gap: {np.mean(hybrid_ratio<1):.3f} vs {np.mean(random_ratio<1):.3f}",
          flush=True)


if __name__ == "__main__":
    main()
