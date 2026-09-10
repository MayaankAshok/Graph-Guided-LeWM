"""Push-T Diagnostic & Environment Verification Probe.

Investigates why the IDM + 1-step predictor failed on Push-T by evaluating:
1. Physical state geometry of latent kNN candidate pairs (agent pos, block pos, angle).
2. Ground-Truth simulator stepping with IDM actions vs. random actions vs. zero action.
3. Bidirectional testing (i -> j vs. j -> i).
4. Physical improvement vs. latent predictor (pred1) prediction error.
5. Real adjacent transition baseline.
6. Generates diagnostic plots for latent-diagnostics-pusht.tex.
"""

import argparse
import json
import time
from pathlib import Path

import h5py
import hydra
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

DEV = "cuda" if torch.cuda.is_available() else "cpu"
EMB_H5 = Path("/ssd_scratch/mayaank.ashok/lewm_pusht_emb/pusht_expert_train_emb.h5")
PUSHT_H5 = Path("/ssd_scratch/mayaank.ashok/lewm_data/datasets/pusht_expert_train.h5")
IDM_CKPT = Path("/home2/mayaank.ashok/lewm_research/idm_fullscale_ensemble.pt")
PRED1_DIR = Path("/home2/mayaank.ashok/lewm_research/data/checkpoints/lewm_pusht_1step_predictor")
FIG_DIR = Path("/home2/mayaank.ashok/lewm_research/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)
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


def load_idm_models():
    print(f"Loading cached full-scale IDM ensemble from {IDM_CKPT}...", flush=True)
    models = []
    for sd in torch.load(IDM_CKPT, map_location=DEV):
        m = MLP(in_dim=384, hidden=256, out_dim=2).to(DEV)
        m.load_state_dict(sd)
        models.append(m.eval())
    return models


def load_pred1():
    print(f"Loading pred1 from {PRED1_DIR}...", flush=True)
    cfg = json.loads((PRED1_DIR / "config.json").read_text())
    model = hydra.utils.instantiate(cfg)
    sd = torch.load(PRED1_DIR / "weights_epoch_20.pt", map_location="cpu")
    model.load_state_dict(sd, strict=True)
    model = model.to(DEV).eval()
    model.requires_grad_(False)
    return model


def predict_idm(models, emb, i_idx, j_idx):
    zi = torch.from_numpy(emb[i_idx]).float().to(DEV)
    zj = torch.from_numpy(emb[j_idx]).float().to(DEV)
    x = torch.cat([zi, zj], dim=-1)
    with torch.no_grad():
        preds = torch.stack([m(x) for m in models], dim=0)
    mean_a = preds.mean(dim=0).cpu().numpy()
    var_a = preds.var(dim=0).sum(dim=-1).cpu().numpy()
    return mean_a, var_a


def predict_pred1(pred1_model, emb, i_idx, actions):
    zi_t = torch.from_numpy(emb[i_idx]).float().to(DEV).unsqueeze(1)
    act_t = torch.from_numpy(actions).float().to(DEV).unsqueeze(1)
    with torch.no_grad():
        act_emb = pred1_model.action_encoder(act_t)
        pred = pred1_model.predict(zi_t, act_emb)[:, -1].cpu().numpy()
    return pred


def get_candidates(emb, ep_idx, k=20, max_cands=100_000, seed=0):
    import faiss
    print("Building FAISS index on validation embeddings...", flush=True)
    z32 = np.ascontiguousarray(emb.astype(np.float32))
    n, d = z32.shape
    index = faiss.IndexHNSWFlat(d, 32)
    index.hnsw.efConstruction = 200
    index.add(z32)
    index.hnsw.efSearch = max(64, 2 * k)
    print(f"Searching top-{k} neighbors...", flush=True)
    D, I = index.search(z32, k + 1)
    rows, cols = [], []
    for i in range(n):
        for rank in range(1, k + 1):
            j = int(I[i, rank])
            if j >= 0 and ep_idx[j] != ep_idx[i]:
                rows.append(i)
                cols.append(j)
    rows, cols = np.array(rows), np.array(cols)
    print(f"Found {len(rows)} raw cross-episode pairs", flush=True)
    if len(rows) > max_cands:
        rng = np.random.default_rng(seed)
        sub = rng.choice(len(rows), size=max_cands, replace=False)
        rows, cols = rows[sub], cols[sub]
    return rows, cols


def oracle_distance(s1, s2):
    def proj(s):
        return np.stack([
            s[:, 0], s[:, 1],
            s[:, 2], s[:, 3],
            np.cos(s[:, 4]), np.sin(s[:, 4])
        ], axis=1)
    p1 = proj(s1)
    p2 = proj(s2)
    agent_d = np.linalg.norm(p1[:, :2] - p2[:, :2], axis=1)
    block_d = np.linalg.norm(p1[:, 2:4] - p2[:, 2:4], axis=1)
    angle_d = np.linalg.norm(p1[:, 4:6] - p2[:, 4:6], axis=1)
    return agent_d, block_d, angle_d


def step_env_batch(env, states_i, actions):
    next_states = np.empty_like(states_i)
    for idx in range(len(states_i)):
        env.reset(options={"state": states_i[idx]})
        obs, reward, terminated, truncated, info = env.step(actions[idx])
        next_states[idx] = obs["state"]
    return next_states


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-eval", type=int, default=1000)
    args = parser.parse_args()

    print(f"=== Push-T Diagnostic & Environment Verification ===", flush=True)
    print(f"Device: {DEV}", flush=True)

    # 1. Load data
    f_emb = h5py.File(EMB_H5, "r", swmr=True)
    ep_offset = f_emb["ep_offset"][:]
    ep_len = f_emb["ep_len"][:]
    n_ep_total = len(ep_offset)
    val_episodes = np.arange(n_ep_total - N_VAL_EPISODES, n_ep_total)

    val_start = int(ep_offset[val_episodes[0]])
    val_count = int(ep_offset[-1] + ep_len[-1] - val_start)
    val_emb = f_emb["emb"][val_start: val_start + val_count]
    f_emb.close()

    f_pusht = h5py.File(PUSHT_H5, "r", swmr=True)
    val_state = f_pusht["state"][val_start: val_start + val_count]
    val_action = f_pusht["action"][val_start: val_start + val_count]
    val_ep_idx = f_pusht["episode_idx"][val_start: val_start + val_count]
    f_pusht.close()

    print(f"Loaded {len(val_emb)} validation frames into memory.", flush=True)

    # 2. Load models
    idm_models = load_idm_models()
    pred1_model = load_pred1()

    # 3. Build candidate pairs
    cand_i_local, cand_j_local = get_candidates(val_emb, val_ep_idx, k=20, max_cands=50_000, seed=42)
    cand_latent_d = np.linalg.norm(val_emb[cand_i_local] - val_emb[cand_j_local], axis=1)

    rng = np.random.default_rng(0)
    near_mask = cand_latent_d < 1.0
    med_mask = (cand_latent_d >= 1.0) & (cand_latent_d < 2.0)
    far_mask = cand_latent_d >= 2.0

    n_per_bucket = args.n_eval // 3
    idx_near = rng.choice(np.where(near_mask)[0], size=min(n_per_bucket, near_mask.sum()), replace=False)
    idx_med = rng.choice(np.where(med_mask)[0], size=min(n_per_bucket, med_mask.sum()), replace=False)
    idx_far = rng.choice(np.where(far_mask)[0], size=min(n_per_bucket, far_mask.sum()), replace=False)
    sample_indices = np.concatenate([idx_near, idx_med, idx_far])
    rng.shuffle(sample_indices)

    ci = cand_i_local[sample_indices]
    cj = cand_j_local[sample_indices]
    c_lat_dist = cand_latent_d[sample_indices]

    # Initialize environment
    from stable_worldmodel.envs.pusht.env import PushT
    env = PushT()

    s_i = val_state[ci]
    s_j = val_state[cj]
    agent_d, block_d, angle_d = oracle_distance(s_i, s_j)
    tot_before = agent_d + block_d

    a_idm, var_idm = predict_idm(idm_models, val_emb, ci, cj)
    a_rand = val_action[rng.choice(len(val_action), size=len(ci))]
    a_zero = np.zeros_like(a_idm)

    next_s_idm = step_env_batch(env, s_i, a_idm)
    next_s_rand = step_env_batch(env, s_i, a_rand)
    next_s_zero = step_env_batch(env, s_i, a_zero)

    ag_after_idm, bl_after_idm, _ = oracle_distance(next_s_idm, s_j)
    ag_after_rand, bl_after_rand, _ = oracle_distance(next_s_rand, s_j)
    ag_after_zero, bl_after_zero, _ = oracle_distance(next_s_zero, s_j)

    tot_after_idm = ag_after_idm + bl_after_idm
    tot_after_rand = ag_after_rand + bl_after_rand
    tot_after_zero = ag_after_zero + bl_after_zero

    pred_z_idm = predict_pred1(pred1_model, val_emb, ci, a_idm)
    target_z = val_emb[cj]
    pred_err_idm = np.linalg.norm(pred_z_idm - target_z, axis=1)

    print("Generating figures...", flush=True)

    # Plot 1: Performance by Latent Distance Regime
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    regimes = ["Near-dup\n(d < 1.0)", "Step-scale\n(1.0 <= d < 2.0)", "Distant\n(d >= 2.0)"]
    masks = [c_lat_dist < 1.0, (c_lat_dist >= 1.0) & (c_lat_dist < 2.0), c_lat_dist >= 2.0]

    med_before = [np.median(tot_before[m]) for m in masks]
    med_idm = [np.median(tot_after_idm[m]) for m in masks]
    med_rand = [np.median(tot_after_rand[m]) for m in masks]

    x = np.arange(len(regimes))
    w = 0.25
    ax1.bar(x - w, med_before, width=w, label="Initial distance", color="#6c757d", alpha=0.8)
    ax1.bar(x, med_idm, width=w, label="After IDM action", color="#1f77b4", alpha=0.85)
    ax1.bar(x + w, med_rand, width=w, label="After Random action", color="#d62728", alpha=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(regimes)
    ax1.set_ylabel("Median Physical Distance to Target s_j (px)")
    ax1.set_title("Physical Distance Before & After Stepping")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.3, axis="y")

    rates_improve = [np.mean(tot_after_idm[m] < tot_before[m]) * 100 for m in masks]
    rates_beats_rand = [np.mean(tot_after_idm[m] < tot_after_rand[m]) * 100 for m in masks]
    rates_agent_imp = [np.mean(ag_after_idm[m] < agent_d[m]) * 100 for m in masks]

    ax2.bar(x - w, rates_improve, width=w, label="Distance Reduced (%)", color="#2ca02c", alpha=0.85)
    ax2.bar(x, rates_beats_rand, width=w, label="IDM Beats Random (%)", color="#1f77b4", alpha=0.85)
    ax2.bar(x + w, rates_agent_imp, width=w, label="Agent Pos Improved (%)", color="#ff7f0e", alpha=0.85)
    ax2.set_xticks(x)
    ax2.set_xticklabels(regimes)
    ax2.set_ylabel("Percentage of Pairs (%)")
    ax2.set_title("Physical Improvement & Win Rates by Regime")
    ax2.set_ylim(0, 100)
    ax2.axhline(50, color="gray", linestyle=":", linewidth=1)
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "pusht_env_verify_regimes.png", dpi=150)
    plt.close(fig)
    print(f"Saved {FIG_DIR / 'pusht_env_verify_regimes.png'}", flush=True)

    # Plot 2: Latent Predictor Error vs True Physical Distance
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    corr = np.corrcoef(pred_err_idm, tot_after_idm)[0, 1]
    ax.scatter(pred_err_idm, tot_after_idm, alpha=0.35, color="#1f77b4", s=18, edgecolors="none")
    m, b = np.polyfit(pred_err_idm, tot_after_idm, 1)
    xs = np.linspace(pred_err_idm.min(), pred_err_idm.max(), 100)
    ax.plot(xs, m * xs + b, color="#d62728", linewidth=2, label=f"Fit (r = {corr:.3f})")
    ax.set_xlabel("Latent pred1 Error ||pred1(z_i, a_idm) - z_j||")
    ax.set_ylabel("Physical State Distance to s_j after Step (px)")
    ax.set_title("Latent Predictor Error vs. Ground-Truth Physical Error")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "pusht_env_verify_scatter.png", dpi=150)
    plt.close(fig)
    print(f"Saved {FIG_DIR / 'pusht_env_verify_scatter.png'}", flush=True)

    print("=== All Complete ===", flush=True)


if __name__ == "__main__":
    main()
