"""Full-scale IDM + splice test, self-contained for Ada.

Trains the IDM ensemble on the full ~16.7k-episode training pool (matching
ada_pusht_idm_scale.py's --n-train-episodes 16685 result: 87.7% held-out action-recovery
MSE reduction, vs. -6.2%/71.5% at the tiny local scales this whole investigation started
from), then runs the SAME cross-episode splice test used throughout this investigation:
build a wide top-20 kNN candidate pool from the held-out 2000-episode validation set, infer
an action for each candidate pair with the IDM, and check whether pred1 (the genuine 1-step
predictor, loaded from data/checkpoints/lewm_pusht_1step_predictor) predicts that action
lands closer to the candidate's partner than a random action would.

Every earlier splice-test failure (see docs/graph-proposal/latent-diagnostics-pusht.tex Sec.
5, and this session's expert_30 probe) was measured against a badly-underfit IDM. This is
the first time the test is run against an IDM that is actually good at its own task.

Run as:
    python ada_pusht_splice_fullscale.py --n-train-episodes 16685
    python ada_pusht_splice_fullscale.py --n-train-episodes 1000   # expert_1000
"""

import argparse
import time
from pathlib import Path

import h5py
import hydra
import json
import numpy as np
import torch
from torch import nn

DEV = "cuda" if torch.cuda.is_available() else "cpu"
EMB_H5 = "/ssd_scratch/mayaank.ashok/lewm_pusht_emb/pusht_expert_train_emb.h5"
PRED1_DIR = Path("/home2/mayaank.ashok/lewm_research/data/checkpoints/lewm_pusht_1step_predictor")
N_VAL_EPISODES = 2000


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


def train_idm_ensemble(emb, action, i_idx, j_idx, n_members, hidden, n_steps, batch_size, lr, seed):
    n = len(i_idx)
    zi = torch.from_numpy(emb[i_idx]).float().to(DEV)
    zj = torch.from_numpy(emb[j_idx]).float().to(DEV)
    a = torch.from_numpy(action[i_idx]).float().to(DEV)
    x = torch.cat([zi, zj], dim=-1)
    action_dim = a.shape[1]

    models = []
    for m in range(n_members):
        rng = np.random.default_rng(seed + m)
        boot = torch.from_numpy(rng.integers(0, n, size=n)).long().to(DEV)

        torch.manual_seed(seed + m)
        model = MLP(in_dim=x.shape[1], hidden=hidden, out_dim=action_dim).to(DEV)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        t0 = time.time()
        for step in range(n_steps):
            idx = boot[torch.randint(0, n, (batch_size,), device=DEV)]
            pred = model(x[idx])
            loss = ((pred - a[idx]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            if (step + 1) % 10000 == 0:
                print(f"  [member {m}] step {step+1}/{n_steps} loss={loss.item():.5f} "
                      f"elapsed={time.time()-t0:.1f}s", flush=True)
        models.append(model.eval())
    return models


def ensemble_predict(models, emb, i_idx, j_idx, chunk=50_000):
    # Chunked: a single giant forward pass over millions of pairs (e.g. the 2.57M-pair
    # candidate pool) allocates multi-GB intermediate activations per layer and OOMs on an
    # 11GB GPU (confirmed in practice) even though the model itself is tiny -- the input
    # batch size is the problem, not model size.
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train-episodes", type=int, default=16685)
    args = ap.parse_args()
    n_train_episodes = args.n_train_episodes

    print(f"device={DEV}", flush=True)
    f = h5py.File(EMB_H5, "r", swmr=True, rdcc_nbytes=256 * 1024 * 1024)
    ep_offset = f["ep_offset"][:]
    ep_len = f["ep_len"][:]
    n_ep_total = len(ep_offset)

    val_episodes = np.arange(n_ep_total - N_VAL_EPISODES, n_ep_total)
    train_pool = np.arange(0, n_ep_total - N_VAL_EPISODES)
    train_episodes = train_pool[:n_train_episodes]
    print(f"train episodes: {len(train_episodes)}, val episodes: {len(val_episodes)}", flush=True)

    tr_i, tr_j = build_transition_pairs(ep_offset, ep_len, train_episodes)
    te_i, te_j = build_transition_pairs(ep_offset, ep_len, val_episodes)
    print(f"train pairs: {len(tr_i)}, val pairs: {len(te_i)}", flush=True)

    t0 = time.time()
    emb = f["emb"][:]
    action = f["action"][:]
    f.close()
    print(f"loaded full emb/action arrays into RAM in {time.time()-t0:.1f}s", flush=True)

    # --- train the full-scale IDM ensemble (same config as the 87.7%-reduction run) --------
    # Cache to disk: this takes ~11 min and is deterministic (fixed seed) -- reuse it if a
    # later stage of this script crashes (as the unbatched ensemble_predict/single_action_error
    # calls did the first time, from a plain CUDA OOM on the full 2.57M-pair candidate batch)
    # rather than eating another 11 min retraining an identical ensemble.
    ckpt_path = Path(
        f"/home2/mayaank.ashok/lewm_research/idm_ensemble_ep{n_train_episodes}.pt")
    action_dim = action.shape[1]
    if ckpt_path.exists():
        print(f"loading cached IDM ensemble from {ckpt_path}", flush=True)
        idm_models = []
        for sd in torch.load(ckpt_path, map_location=DEV):
            m = MLP(in_dim=384, hidden=256, out_dim=action_dim).to(DEV)
            m.load_state_dict(sd)
            idm_models.append(m.eval())
    else:
        t0 = time.time()
        idm_models = train_idm_ensemble(emb, action, tr_i, tr_j, n_members=6, hidden=256,
                                         n_steps=30000, batch_size=1024, lr=1e-3, seed=0)
        print(f"trained IDM ensemble in {time.time()-t0:.1f}s", flush=True)
        torch.save([m.state_dict() for m in idm_models], ckpt_path)
        print(f"saved IDM ensemble to {ckpt_path}", flush=True)

    marginal_action = action[tr_i].mean(axis=0, keepdims=True)
    mse_marginal = float(((marginal_action - action[te_i]) ** 2).mean())
    pred_te = ensemble_predict(idm_models, emb, te_i, te_j).mean(axis=0)
    mse_model = float(((pred_te - action[te_i]) ** 2).mean())
    print(f"[sanity] held-out REAL action-recovery MSE={mse_model:.5f} vs marginal="
          f"{mse_marginal:.5f} (reduction={100*(1-mse_model/mse_marginal):.1f}%)", flush=True)

    # --- build candidate cross-episode pairs from the held-out validation episodes ---------
    val_rows = np.concatenate([
        np.arange(int(ep_offset[e]), int(ep_offset[e]) + int(ep_len[e])) for e in val_episodes
    ])
    val_ep_idx = np.concatenate([
        np.full(int(ep_len[e]), e) for e in val_episodes
    ])
    print(f"validation pool: {len(val_rows)} rows from {len(val_episodes)} episodes", flush=True)

    t0 = time.time()
    local_cand_i, local_cand_j = raw_topk_cross_episode_pairs(emb[val_rows], val_ep_idx, k=20)
    cand_i, cand_j = val_rows[local_cand_i], val_rows[local_cand_j]
    print(f"[candidates] {len(cand_i)} cross-episode pairs from top-20 kNN in "
          f"{time.time()-t0:.1f}s", flush=True)
    cand_dist = np.linalg.norm(emb[cand_i] - emb[cand_j], axis=1)
    print(f"[candidates] gap: mean={cand_dist.mean():.3f} median={np.median(cand_dist):.3f}",
          flush=True)

    # subsample for the splice test itself -- 2.5M+ pairs isn't needed for a robust estimate,
    # and keeps the chunked ensemble/predictor scoring below fast
    n_sub = min(100_000, len(cand_i))
    sub = np.random.default_rng(0).choice(len(cand_i), size=n_sub, replace=False)
    cand_i, cand_j, cand_dist = cand_i[sub], cand_j[sub], cand_dist[sub]
    print(f"[candidates] subsampled to {len(cand_i)} pairs for scoring", flush=True)

    # --- score with pred1 -------------------------------------------------------------------
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

    rng = np.random.default_rng(0)
    a_idm = ensemble_predict(idm_models, emb, cand_i, cand_j).mean(axis=0).astype(np.float32)
    rand_a1 = action[te_i[rng.integers(0, len(te_i), size=len(cand_i))]]

    hybrid_err = single_action_error(cand_i, cand_j, a_idm)
    random_err = single_action_error(cand_i, cand_j, rand_a1)
    hybrid_ratio = hybrid_err / np.maximum(cand_dist, 1e-8)
    random_ratio = random_err / np.maximum(cand_dist, 1e-8)

    real_err = single_action_error(te_i, te_j, action[te_i])
    real_dist = np.linalg.norm(emb[te_i] - emb[te_j], axis=1)
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
    print(f"[RESULT n_train_episodes={n_train_episodes}] IDM-vs-random gap: "
          f"{random_err.mean()/hybrid_err.mean():.3f}x (mean error ratio); "
          f"frac<1 gap: {np.mean(hybrid_ratio<1):.3f} vs {np.mean(random_ratio<1):.3f}",
          flush=True)


if __name__ == "__main__":
    main()
