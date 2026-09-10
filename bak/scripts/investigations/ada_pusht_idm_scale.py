"""Ada-only, self-contained (no repo deps beyond numpy/torch/h5py) IDM data-scale check.

Follow-up to the local expert_30 vs expert_300 finding: the IDM ensemble's held-out REAL
action-recovery MSE reduction vs. a marginal baseline went from -6.2% (expert_30, 3,241 train
pairs) to +69.8% (expert_300, 29,461 train pairs) with NO change in model size -- i.e. the
earlier failure was data scarcity, not model capacity (bigger MLPs at expert_30 made things
WORSE: -25.1% at hidden=512, -20.1% at hidden=1024). This checks whether the trend keeps
improving at 2-3k episodes and the full ~18.7k-episode scale, using the precomputed
frozen-encoder embeddings (pusht_expert_train_emb.h5, already computed for pred1's own
training) instead of re-running the ViT encoder.

A FIXED validation set (the last 2000 episodes on-disk) is used at every scale so results are
directly comparable across stages -- only the training-episode count changes.

Run as:
    python ada_pusht_idm_scale.py --n-train-episodes 2500
    python ada_pusht_idm_scale.py --n-train-episodes 16685   # full scale (all non-val episodes)
"""

import argparse
import time

import h5py
import numpy as np
import torch
from torch import nn

DEV = "cuda" if torch.cuda.is_available() else "cpu"
EMB_H5 = "/ssd_scratch/mayaank.ashok/lewm_pusht_emb/pusht_expert_train_emb.h5"
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
    """Real adjacent (frame-skip=1) transition pairs within each given episode."""
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
        # Index per-batch through the bootstrap map instead of materializing x[boot]/a[boot]
        # as full-size copies -- at ~2M rows that's ~3.2GB per member and doesn't fit six
        # times over even with the previous member's copy freed between iterations (hit a
        # real CUDA OOM at the full 16685-episode scale, fine at 2500 episodes where the
        # copy is only ~450MB). `boot` itself is tiny (n int64 indices); only per-step
        # mini-batches of the big tensors are ever materialized.
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
            if (step + 1) % 5000 == 0:
                print(f"  [member {m}] step {step+1}/{n_steps} loss={loss.item():.5f} "
                      f"elapsed={time.time()-t0:.1f}s", flush=True)
        models.append(model.eval())
    return models


def ensemble_predict(models, emb, i_idx, j_idx):
    zi = torch.from_numpy(emb[i_idx]).float().to(DEV)
    zj = torch.from_numpy(emb[j_idx]).float().to(DEV)
    x = torch.cat([zi, zj], dim=-1)
    with torch.no_grad():
        preds = torch.stack([m(x) for m in models], dim=0)
    return preds.cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train-episodes", type=int, required=True)
    ap.add_argument("--n-members", type=int, default=6)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--n-steps", type=int, default=30000)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print(f"device={DEV}", flush=True)
    f = h5py.File(EMB_H5, "r", swmr=True, rdcc_nbytes=256 * 1024 * 1024)
    ep_offset = f["ep_offset"][:]
    ep_len = f["ep_len"][:]
    n_ep_total = len(ep_offset)
    print(f"total episodes={n_ep_total}, total frames={ep_offset[-1] + ep_len[-1]}", flush=True)

    val_episodes = np.arange(n_ep_total - N_VAL_EPISODES, n_ep_total)
    train_pool = np.arange(0, n_ep_total - N_VAL_EPISODES)
    assert args.n_train_episodes <= len(train_pool), \
        f"only {len(train_pool)} non-validation episodes available"
    train_episodes = train_pool[: args.n_train_episodes]
    print(f"train episodes: {len(train_episodes)} (ids {train_episodes[0]}..{train_episodes[-1]}), "
          f"val episodes: {len(val_episodes)} (ids {val_episodes[0]}..{val_episodes[-1]})", flush=True)

    tr_i, tr_j = build_transition_pairs(ep_offset, ep_len, train_episodes)
    te_i, te_j = build_transition_pairs(ep_offset, ep_len, val_episodes)
    print(f"train pairs: {len(tr_i)}, val pairs: {len(te_i)}", flush=True)

    # load only the rows we actually need into RAM (full emb array is ~1.8GB, fine to load
    # whole thing once and index in-memory rather than repeated random h5 reads)
    t0 = time.time()
    emb = f["emb"][:]
    action = f["action"][:]
    f.close()
    print(f"loaded full emb/action arrays into RAM in {time.time()-t0:.1f}s", flush=True)

    marginal_action = action[tr_i].mean(axis=0, keepdims=True)
    mse_marginal = float(((marginal_action - action[te_i]) ** 2).mean())
    print(f"marginal-baseline (predict mean training action) held-out MSE={mse_marginal:.5f}", flush=True)

    t0 = time.time()
    models = train_idm_ensemble(emb, action, tr_i, tr_j, args.n_members, args.hidden,
                                 args.n_steps, args.batch_size, args.lr, args.seed)
    print(f"trained {args.n_members} members in {time.time()-t0:.1f}s", flush=True)

    pred_te = ensemble_predict(models, emb, te_i, te_j).mean(axis=0)
    mse_model = float(((pred_te - action[te_i]) ** 2).mean())
    ensemble_var = ensemble_predict(models, emb, te_i, te_j).var(axis=0).mean()
    print(f"[RESULT n_train_episodes={args.n_train_episodes}] held-out REAL action-recovery "
          f"MSE={mse_model:.5f} vs. marginal={mse_marginal:.5f} "
          f"(reduction={100*(1-mse_model/mse_marginal):.1f}%)  "
          f"ensemble_var(held-out real)={ensemble_var:.5f}", flush=True)


if __name__ == "__main__":
    main()
