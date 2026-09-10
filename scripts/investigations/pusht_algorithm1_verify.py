"""Verify Algorithm 1 (docs/graph-proposal/latent-diagnostics-pusht.pdf, Section 5.3:
"Self-Calibrating Latent Graph Construction") end-to-end on Push-T, against ground truth.

Nothing in the doc ever runs the algorithm's own auto-calibrated thresholds (tau_id,
tau_reach, K*, tau_var, tau_rel) and checks the resulting edge set against physical reality.
Section 4 (the ground-truth simulator experiment) used HAND-PICKED distance buckets (1.0,
2.0) to explain why the latent-only splice test was blind, and Section 5's algorithm was
designed from that diagnosis but never itself re-checked. This script closes that gap: run
the full latent-only calibration + graph-construction pipeline exactly as Algorithm 1
specifies, then use the real PushT simulator (offline diagnostic oracle only, per the
doc's own strictly-latent-only mandate for the production pipeline) to check whether the
edges it decides to add are actually physically valid.

Reuses the already-trained/cached full-scale IDM ensemble and pred1 predictor from
pusht_diagnostic_env_verify.py's investigation -- no retraining needed, this is purely a
calibration + construction + verification pass.

Run as (on an Ada compute node, /ssd_scratch staged with the Push-T dataset + embeddings):
    python pusht_algorithm1_verify.py --n-eval-per-tier 300
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

try:
    _cuda_ok = torch.cuda.is_available()
except RuntimeError:
    _cuda_ok = False  # this node's CUDA runtime can be broken (cudaGetDeviceCount failing
                       # with "invalid device ordinal") independent of GPU allocation --
                       # torch.cuda.is_available() itself raises in that case rather than
                       # returning False, so it must be caught explicitly. All models here
                       # (tiny IDM MLPs, a depth-2/hidden-96 ARPredictor) are small enough to
                       # run on CPU without being a real bottleneck at this script's scale.
DEV = "cuda" if _cuda_ok else "cpu"
EMB_H5 = Path("/ssd_scratch/mayaank.ashok/lewm_pusht_emb/pusht_expert_train_emb.h5")
PUSHT_H5 = Path("/ssd_scratch/mayaank.ashok/lewm_data/datasets/pusht_expert_train.h5")
IDM_CKPT = Path("/home2/mayaank.ashok/lewm_research/idm_fullscale_ensemble.pt")
PRED1_DIR = Path("/home2/mayaank.ashok/lewm_research/data/checkpoints/lewm_pusht_1step_predictor")
FIG_DIR = Path("/home2/mayaank.ashok/lewm_research/figures")
OUT_DIR = Path("/home2/mayaank.ashok/lewm_research")
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
    """IDM ensemble's inferred single action connecting z_i -> z_j (Algorithm 1's
    abar_{u->v}), plus per-pair ensemble variance (sigma^2_I)."""
    zi = torch.from_numpy(emb[i_idx]).float().to(DEV)
    zj = torch.from_numpy(emb[j_idx]).float().to(DEV)
    x = torch.cat([zi, zj], dim=-1)
    with torch.no_grad():
        preds = torch.stack([m(x) for m in models], dim=0)
    mean_a = preds.mean(dim=0).cpu().numpy()
    var_a = preds.var(dim=0).sum(dim=-1).cpu().numpy()
    return mean_a, var_a


def predict_pred1(pred1_model, emb, i_idx, actions):
    """One pred1 forward call: P(z_i, action) -> z_hat."""
    zi_t = torch.from_numpy(emb[i_idx]).float().to(DEV).unsqueeze(1)
    act_t = torch.from_numpy(actions).float().to(DEV).unsqueeze(1)
    with torch.no_grad():
        act_emb = pred1_model.action_encoder(act_t)
        pred = pred1_model.predict(zi_t, act_emb)[:, -1].cpu().numpy()
    return pred


def _pred1_step_from_z(pred1_model, z_batch, actions):
    """Same as predict_pred1 but takes a z ARRAY directly (not indices into a stored emb
    table) -- needed for autoregressive chaining, where intermediate z's are never stored."""
    zi_t = torch.from_numpy(z_batch).float().to(DEV).unsqueeze(1)
    act_t = torch.from_numpy(actions).float().to(DEV).unsqueeze(1)
    with torch.no_grad():
        act_emb = pred1_model.action_encoder(act_t)
        pred = pred1_model.predict(zi_t, act_emb)[:, -1].cpu().numpy()
    return pred


def pred1_chain_from_start(pred1_model, z_start, action_chain):
    """action_chain: (n, K, action_dim), z_start: (n, D). Chains pred1 K times."""
    n, K, _ = action_chain.shape
    z_cur = z_start.astype(np.float32).copy()
    for k in range(K):
        z_cur = _pred1_step_from_z(pred1_model, z_cur, action_chain[:, k])
    return z_cur


def get_candidates(emb, ep_idx, k=20, max_cands=200_000, seed=0):
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
    """Physical (agent_dist, block_dist, angle_dist) between two Push-T state arrays --
    same projection PushTMechanics.build_true_distance_oracle uses."""
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


# ============================================================
# Algorithm 1, Phase 1: autonomous latent-only calibration
# ============================================================

def sample_in_traj_pairs(ep_idx, step_idx, gap, n, seed):
    """n random (t, t+gap) same-episode index pairs with enough room left in the episode."""
    rng = np.random.default_rng(seed)
    order = np.lexsort((step_idx, ep_idx))
    ep_o, step_o = ep_idx[order], step_idx[order]
    pos_in_order = np.empty(len(ep_idx), dtype=np.int64)
    pos_in_order[order] = np.arange(len(order))

    ep_len = np.zeros(int(ep_idx.max()) + 1, dtype=np.int64)
    for e in np.unique(ep_idx):
        ep_len[e] = int((ep_idx == e).sum())

    valid = np.nonzero(step_idx + gap < ep_len[ep_idx])[0]
    if len(valid) > n:
        valid = rng.choice(valid, size=n, replace=False)
    i_idx = valid
    j_idx = order[pos_in_order[valid] + gap]
    return i_idx, j_idx, pos_in_order, order


def calibrate_algorithm1(emb, ep_idx, step_idx, action, idm_models, pred1_model,
                          n_calib=10000, seed=0):
    """Phase 1 of Algorithm 1: derive K*, tau_id, tau_reach, tau_var, tau_rel purely from
    latent statistics on the offline dataset -- no ground truth used anywhere in this
    function. Returns a dict of calibrated values plus the raw distributions (for plotting)."""
    print("=== Phase 1: autonomous calibration ===", flush=True)

    # --- SNR(1) and K* selection --------------------------------------------------------
    i1, j1, pos_in_order, order = sample_in_traj_pairs(ep_idx, step_idx, gap=1, n=n_calib, seed=seed)
    d_step = np.linalg.norm(emb[j1] - emb[i1], axis=1)
    r1 = np.linalg.norm(predict_pred1(pred1_model, emb, i1, action[i1]) - emb[j1], axis=1)
    snr1 = float(np.median(d_step) / np.median(r1))
    print(f"[calib] SNR(1) = median(D_step)/median(R_1) = {np.median(d_step):.3f}/"
          f"{np.median(r1):.3f} = {snr1:.3f}", flush=True)

    k_star = 1
    snr_by_k = {1: snr1}
    d_kstep = d_step
    if snr1 < 2.0:
        print("[calib] low-SNR regime -- sweeping K in {2,4,5,8,16} for K*", flush=True)
        for K in (2, 4, 5, 8, 16):
            ik, jk, _, _ = sample_in_traj_pairs(ep_idx, step_idx, gap=K, n=n_calib, seed=seed + K)
            d_k = np.linalg.norm(emb[jk] - emb[ik], axis=1)
            # action chain: K real actions starting at each ik
            pos_i = pos_in_order[ik]
            chain_rows = np.stack([order[pos_i + s] for s in range(K)], axis=1)  # (n,K)
            action_chain = action[chain_rows]  # (n,K,action_dim)
            z_hat_k = pred1_chain_from_start(pred1_model, emb[ik], action_chain)
            r_k = np.linalg.norm(z_hat_k - emb[jk], axis=1)
            snr_k = float(np.median(d_k) / np.median(r_k))
            snr_by_k[K] = snr_k
            print(f"[calib]   K={K:>2}: median(D_K)={np.median(d_k):.3f} "
                  f"median(R_K)={np.median(r_k):.3f} SNR(K)={snr_k:.3f}", flush=True)
            if snr_k >= 2.0 and k_star == 1:
                k_star = K
                d_kstep = d_k
        if k_star == 1:
            k_star = max(snr_by_k, key=snr_by_k.get)
            print(f"[calib] WARNING: no K reached SNR>=2.0; falling back to best-SNR "
                  f"K*={k_star} (SNR={snr_by_k[k_star]:.3f})", flush=True)
            ik, jk, _, _ = sample_in_traj_pairs(ep_idx, step_idx, gap=k_star, n=n_calib, seed=seed + k_star)
            d_kstep = np.linalg.norm(emb[jk] - emb[ik], axis=1)
        print(f"[calib] K* = {k_star}", flush=True)

    # --- tau_id, tau_reach ---------------------------------------------------------------
    tau_id = float(np.quantile(d_step, 0.15))
    tau_reach = float(np.quantile(d_kstep, 0.95))
    print(f"[calib] tau_id = Q0.15(D_step) = {tau_id:.4f}", flush=True)
    print(f"[calib] tau_reach = Q0.95(D_K*-step) = {tau_reach:.4f}", flush=True)

    # --- tau_var: IDM ensemble variance on real in-trajectory single-step pairs ----------
    _, var_in = predict_idm(idm_models, emb, i1, j1)
    tau_var = float(np.quantile(var_in, 0.95))
    print(f"[calib] tau_var = Q0.95(V_in) = {tau_var:.5f} "
          f"(V_in median={np.median(var_in):.5f})", flush=True)

    # --- tau_rel: predictor relative-consistency ratio on real transitions ---------------
    pred_real = predict_pred1(pred1_model, emb, i1, action[i1])
    err_real = np.linalg.norm(pred_real - emb[j1], axis=1)
    a_null = np.zeros_like(action[i1])
    pred_null = predict_pred1(pred1_model, emb, i1, a_null)
    err_null = np.linalg.norm(pred_null - emb[j1], axis=1)
    q_rel = err_real / np.maximum(err_null, 1e-8)
    tau_rel = float(min(0.90, np.quantile(q_rel, 0.85)))
    print(f"[calib] tau_rel = min(0.90, Q0.85(Q_rel)) = min(0.90, {np.quantile(q_rel, 0.85):.4f}) "
          f"= {tau_rel:.4f}  (Q_rel median={np.median(q_rel):.4f})", flush=True)

    return dict(
        k_star=k_star, snr_by_k=snr_by_k, tau_id=tau_id, tau_reach=tau_reach,
        tau_var=tau_var, tau_rel=tau_rel,
        d_step=d_step, d_kstep=d_kstep, var_in=var_in, q_rel=q_rel,
    )


# ============================================================
# Algorithm 1, Phase 3: cross-episode edge construction (latent-only)
# ============================================================

def construct_edges(emb, ep_idx, cand_i, cand_j, calib, idm_models, pred1_model):
    """Classify every candidate cross-episode pair per Algorithm 1's Phase 3, bidirectionally.
    Returns per-pair-per-direction records: category in {"id_edge" (tier 1, both directions
    accepted together), "accepted" (tier 2, this direction passed both gates), "rejected_gate"
    (in reach but failed variance/relative gate), "beyond_reach" (never queried)}."""
    d = np.linalg.norm(emb[cand_i] - emb[cand_j], axis=1)
    is_id = d <= calib["tau_id"]
    in_band = (d > calib["tau_id"]) & (d <= calib["tau_reach"])
    beyond = d > calib["tau_reach"]

    print(f"[construct] {len(cand_i)} candidates: {is_id.sum()} tier-1 (id, d<=tau_id), "
          f"{in_band.sum()} tier-2 band (tau_id<d<=tau_reach), {beyond.sum()} beyond tau_reach",
          flush=True)

    band_i, band_j = cand_i[in_band], cand_j[in_band]
    records = []
    for (u_arr, v_arr, dirname) in [(band_i, band_j, "fwd"), (band_j, band_i, "rev")]:
        a_bar, var_a = predict_idm(idm_models, emb, u_arr, v_arr)
        pass_var = var_a <= calib["tau_var"]
        err_idm = np.full(len(u_arr), np.nan, dtype=np.float32)
        err_null = np.full(len(u_arr), np.nan, dtype=np.float32)
        ratio = np.full(len(u_arr), np.nan, dtype=np.float32)
        idx_pass_var = np.nonzero(pass_var)[0]
        if len(idx_pass_var):
            a_null = np.zeros_like(a_bar[idx_pass_var])
            pred_idm = predict_pred1(pred1_model, emb, u_arr[idx_pass_var], a_bar[idx_pass_var])
            pred_null = predict_pred1(pred1_model, emb, u_arr[idx_pass_var], a_null)
            err_idm[idx_pass_var] = np.linalg.norm(pred_idm - emb[v_arr[idx_pass_var]], axis=1)
            err_null[idx_pass_var] = np.linalg.norm(pred_null - emb[v_arr[idx_pass_var]], axis=1)
            ratio[idx_pass_var] = err_idm[idx_pass_var] / np.maximum(err_null[idx_pass_var], 1e-8)
        accepted = pass_var & (ratio <= calib["tau_rel"])
        for k in range(len(u_arr)):
            cat = "accepted" if accepted[k] else "rejected_gate"
            records.append(dict(u=int(u_arr[k]), v=int(v_arr[k]), direction=dirname,
                                 latent_d=float(np.linalg.norm(emb[u_arr[k]] - emb[v_arr[k]])),
                                 var=float(var_a[k]), ratio=float(ratio[k]) if pass_var[k] else None,
                                 action=a_bar[k].tolist(), category=cat))
        print(f"[construct]   direction={dirname}: {int(pass_var.sum())}/{len(u_arr)} pass "
              f"variance gate, {int(accepted.sum())}/{len(u_arr)} accepted overall", flush=True)

    return dict(
        id_i=cand_i[is_id], id_j=cand_j[is_id],
        beyond_i=cand_i[beyond], beyond_j=cand_j[beyond],
        band_records=records,
    )


# ============================================================
# Ground-truth verification
# ============================================================

def verify_category(env, state, i_arr, j_arr, actions, label, n_eval, rng):
    """Steps the REAL simulator from state[i] under `actions`, and under a random real
    action control, and reports physical distance to state[j] before/after -- exactly
    Section 4's ground-truth methodology, applied to THIS algorithm's own edge decisions
    rather than hand-picked distance buckets."""
    n = len(i_arr)
    if n == 0:
        print(f"[verify:{label}] n=0, skipping", flush=True)
        return None
    sub = rng.choice(n, size=min(n_eval, n), replace=False)
    si, sj, a = state[i_arr[sub]], state[j_arr[sub]], actions[sub]

    agent_d, block_d, _ = oracle_distance(si, sj)
    tot_before = agent_d + block_d

    next_s = step_env_batch(env, si, a)
    ag_after, bl_after, _ = oracle_distance(next_s, sj)
    tot_after = ag_after + bl_after

    # random-action control: a genuinely real action (from elsewhere in this same sampled
    # batch), mismatched to this pair -- matches Section 4's "random real action" control
    # rather than an out-of-distribution synthetic vector.
    a_rand = a[np.random.default_rng(1).permutation(len(a))]
    next_s_rand = step_env_batch(env, si, a_rand)
    ag_rand, bl_rand, _ = oracle_distance(next_s_rand, sj)
    tot_rand = ag_rand + bl_rand

    result = dict(
        label=label, n=int(len(sub)),
        median_before=float(np.median(tot_before)),
        median_after=float(np.median(tot_after)),
        median_after_random=float(np.median(tot_rand)),
        frac_improved=float(np.mean(tot_after < tot_before)),
        frac_beats_random=float(np.mean(tot_after < tot_rand)),
    )
    print(f"[verify:{label}] n={result['n']} median_dist before={result['median_before']:.2f}px "
          f"after={result['median_after']:.2f}px after_random={result['median_after_random']:.2f}px "
          f"  reduced={100*result['frac_improved']:.1f}%  beats_random={100*result['frac_beats_random']:.1f}%",
          flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-calib", type=int, default=10000)
    parser.add_argument("--n-eval-per-tier", type=int, default=300)
    parser.add_argument("--k-cand", type=int, default=20)
    args = parser.parse_args()

    print("=== Algorithm 1 (self-calibrating latent graph construction) verification ===",
          flush=True)
    print(f"Device: {DEV}", flush=True)

    f_emb = h5py.File(EMB_H5, "r", swmr=True)
    ep_offset = f_emb["ep_offset"][:]
    ep_len = f_emb["ep_len"][:]
    n_ep_total = len(ep_offset)
    val_episodes = np.arange(n_ep_total - N_VAL_EPISODES, n_ep_total)
    val_start = int(ep_offset[val_episodes[0]])
    val_count = int(ep_offset[-1] + ep_len[-1] - val_start)
    val_emb = f_emb["emb"][val_start: val_start + val_count]
    val_action_from_emb = f_emb["action"][val_start: val_start + val_count]
    f_emb.close()

    f_pusht = h5py.File(PUSHT_H5, "r", swmr=True)
    val_state = f_pusht["state"][val_start: val_start + val_count]
    val_ep_idx_raw = f_pusht["episode_idx"][val_start: val_start + val_count]
    f_pusht.close()

    # local (0-based within this val slice) ep_idx/step_idx, matching landmarks conventions
    uniq_eps, val_ep_idx = np.unique(val_ep_idx_raw, return_inverse=True)
    val_step_idx = np.empty(len(val_ep_idx), dtype=np.int64)
    for e in np.unique(val_ep_idx):
        m = val_ep_idx == e
        val_step_idx[m] = np.arange(int(m.sum()))

    print(f"Loaded {len(val_emb)} validation frames from {len(uniq_eps)} episodes.", flush=True)

    idm_models = load_idm_models()
    pred1_model = load_pred1()

    calib = calibrate_algorithm1(val_emb, val_ep_idx, val_step_idx, val_action_from_emb,
                                  idm_models, pred1_model, n_calib=args.n_calib)

    cand_i, cand_j = get_candidates(val_emb, val_ep_idx, k=args.k_cand, max_cands=200_000, seed=42)
    edges = construct_edges(val_emb, val_ep_idx, cand_i, cand_j, calib, idm_models, pred1_model)

    print("\n=== Ground-truth verification (real PushT simulator) ===", flush=True)
    from stable_worldmodel.envs.pusht.env import PushT
    env = PushT()
    rng = np.random.default_rng(7)

    results = {}

    # Tier 1: identification edges -- verify the "already same state" premise (zero action)
    if len(edges["id_i"]):
        zero_a = np.zeros((len(edges["id_i"]), 2), dtype=np.float32)
        results["tier1_identification"] = verify_category(
            env, val_state, edges["id_i"], edges["id_j"], zero_a,
            "tier1_identification (zero action)", args.n_eval_per_tier, rng)

    # Tier 2: accepted vs rejected-by-gate, within the active band
    band = edges["band_records"]
    for cat in ("accepted", "rejected_gate"):
        rows = [r for r in band if r["category"] == cat]
        if not rows:
            print(f"[verify] no rows in category={cat}", flush=True)
            continue
        u = np.array([r["u"] for r in rows])
        v = np.array([r["v"] for r in rows])
        a = np.array([r["action"] for r in rows], dtype=np.float32)
        results[f"tier2_{cat}"] = verify_category(
            env, val_state, u, v, a, f"tier2_{cat}", args.n_eval_per_tier, rng)

    # Beyond reach: sanity check these are genuinely far apart physically too
    if len(edges["beyond_i"]):
        zero_a = np.zeros((len(edges["beyond_i"]), 2), dtype=np.float32)
        results["beyond_reach"] = verify_category(
            env, val_state, edges["beyond_i"], edges["beyond_j"], zero_a,
            "beyond_reach (zero action, sanity check)", args.n_eval_per_tier, rng)

    summary = dict(
        k_star=calib["k_star"], snr_by_k=calib["snr_by_k"],
        tau_id=calib["tau_id"], tau_reach=calib["tau_reach"],
        tau_var=calib["tau_var"], tau_rel=calib["tau_rel"],
        n_candidates=len(cand_i), n_id_edges=len(edges["id_i"]),
        n_band=len(edges["band_records"]) // 1, n_beyond_reach=len(edges["beyond_i"]),
        n_accepted=sum(1 for r in band if r["category"] == "accepted"),
        n_rejected_gate=sum(1 for r in band if r["category"] == "rejected_gate"),
        verification=results,
    )
    out_path = OUT_DIR / "algorithm1_verify_results.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {out_path}", flush=True)
    print(json.dumps(summary, indent=2), flush=True)

    # --- figure: median physical distance before/after by algorithm-decided category -----
    cats = [c for c in ["tier1_identification", "tier2_accepted", "tier2_rejected_gate",
                         "beyond_reach"] if c in results]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(cats))
    w = 0.25
    before = [results[c]["median_before"] for c in cats]
    after = [results[c]["median_after"] for c in cats]
    after_rand = [results[c]["median_after_random"] for c in cats]
    ax.bar(x - w, before, width=w, label="before", color="#6c757d")
    ax.bar(x, after, width=w, label="after algorithm's action", color="#1f77b4")
    ax.bar(x + w, after_rand, width=w, label="after random action", color="#d62728")
    ax.set_xticks(x)
    ax.set_xticklabels(cats, rotation=15)
    ax.set_ylabel("median physical distance to target (px)")
    ax.set_title(f"Algorithm 1 edge categories vs. ground truth (K*={calib['k_star']})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "algorithm1_verify_by_category.png", dpi=150)
    print(f"Saved {FIG_DIR / 'algorithm1_verify_by_category.png'}", flush=True)

    print("\n=== All Complete ===", flush=True)


if __name__ == "__main__":
    main()
