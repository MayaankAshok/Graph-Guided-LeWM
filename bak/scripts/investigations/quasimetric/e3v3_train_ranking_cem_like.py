"""E3-v3 -- ranking loss over CEM-like candidate action sequences, with REAL environment
ground truth. Tests the revised hypothesis from E3-v2/E4's postmortem: a fixed-target
regression on predictor rollouts (E3-v2) flattened the discriminative signal CEM needs,
because it never taught the head to RANK different candidates against each other -- only to
match one scalar on average. This replaces that term with a pairwise ranking loss over K
CEM-like candidate action sequences per start state, supervised by GROUND-TRUTH outcome
distance from actually stepping the real Push-T physics simulator (not just the predictor) --
there is no dataset row that tells you what a random, non-expert action sequence achieves, so
real env stepping is the only way to get a genuine label for these candidates.

Pipeline per training example, precomputed ONCE (env stepping is CPU-bound and far too slow
to redo every training step):
  1. Sample a start row from a train episode (real state + real cached z0).
  2. Fix the goal as the SAME row's real 25-real-step-later continuation in the same episode
     (matching E4's exact live-rollout goal_offset=25 protocol -- the deployment distribution
     this is meant to help).
  3. Sample K candidate action sequences, each 25 real actions ~ N(0, var_scale=1) per real
     action dim in Z-SCORED units -- matching stable_worldmodel's CEMSolver var_scale=1.0
     default EXACTLY (config/eval/solver/cem.yaml), i.e. what CEM's *early* (least-narrowed,
     most representative of what the head must not be confused by) samples look like.
  4. De-normalize to raw units (mean/std from the real action column, matching
     WorldModelPolicy's `process['action'].inverse_transform` -- verified by reading
     stable_worldmodel/policy.py directly) and step the REAL env (mech.make_env(), reset to
     the start row's real state) 25 times to get the candidate's true resulting state.
  5. True outcome distance = the same projected/normalized Euclidean oracle
     (common.envs.PushTMechanics.build_true_distance_oracle) used throughout this project,
     from the candidate's true resulting state to the fixed goal's real state.
  6. Roll the SAME z-scored candidate actions (reshaped into 5 predictor blocks of 10-dim
     each) through the model's actual predictor from the real cached z0
     (e3v2_train_predictor_aware.predictor_rollout_from_z0) to get the PREDICTED terminal
     embedding z_hat.

Loss: within each (start, goal, K candidates) group, a pairwise margin-ranking loss requires
d_Q(z_hat_i, z_g) < d_Q(z_hat_j, z_g) whenever candidate i's true outcome distance is smaller
than candidate j's by more than a noise-floor margin -- directly supervising the RELATIVE
ordering CEM's top-k elite selection consumes, unlike E3-v2's fixed-scalar-target term.

Scale is deliberately modest (real physics stepping, not GPU tensor ops, is the bottleneck):
a few hundred start states x K candidates, precomputed once and cached. This is a smaller,
noisier pool than E0-E3's dataset-scale pairs, stated plainly rather than hidden.

Run:
    python scripts/investigations/quasimetric/e3v3_train_ranking_cem_like.py env=pusht n_episodes=18685
"""

import json
import os
import sys
import time
from pathlib import Path

import h5py
import hydra
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from scipy import stats as sps

ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.envs import ENV_MECHANICS
from common.graph_lib import DEV
from common.lewm_loader import load_lewm
from common.log_util import log
from common.quasimetric import LatentQuasimetric
from probe_gate import Arm, _chunked, make_pairs
from e1_directed_asymmetry import find_active_pushing_pairs
from e3_train_quasimetric import episode_split, gap1_pairs
from e3v2_train_predictor_aware import build_block_chain_lookup, predictor_rollout_from_z0

N_STEPS = int(os.environ.get("E3_N_STEPS", 20000))
BATCH_SIZE = int(os.environ.get("E3_BATCH_SIZE", 512))
LOCAL_BATCH = 128
ASYM_BATCH = 128
LR = 1e-3
W_RANK = 1.0
W_LOCAL = 2.0
W_ASYM = 0.5
W_CEM_RANK = 1.0
ASYM_MARGIN = 6.0

N_CANDIDATE_GROUPS = int(os.environ.get("E3V3_N_GROUPS", 400))   # start states, precompute pool
K_CANDIDATES = int(os.environ.get("E3V3_K", 16))                 # candidates per start state
CEM_REAL_STEPS = 25       # = horizon(5) * action_block(5), matches config/eval/pusht.yaml
CEM_VAR_SCALE = 1.0       # matches config/eval/solver/cem.yaml's var_scale exactly
RANK_MARGIN_NOISE_FLOOR = 0.15   # candidates whose true-distance differs by less than this
                                  # (normalized-state units) are treated as a tie, not ranked
GROUP_BATCH = int(os.environ.get("E3V3_GROUP_BATCH", 16))   # candidate GROUPS per training step
                                                             # (x K_CANDIDATES rows each)


def precompute_cem_candidate_pool(mech, cache, ep_idx, step_idx, train_eps, root, seed=0):
    """Real-env-stepping precompute, cached to disk. Returns a dict of numpy arrays:
    start_row, group_id, z_hat (predicted terminal emb), true_dist (to that group's goal),
    goal_row (one per group, repeated per candidate for convenience)."""
    cache_path = root / "outputs" / "quasimetric" / \
        f"e3v3_cem_pool_g{N_CANDIDATE_GROUPS}_k{K_CANDIDATES}.npz"
    if cache_path.exists():
        log(f"[cem-pool] loading cached pool from {cache_path}")
        d = np.load(cache_path)
        return {k: d[k] for k in d.files}

    rng = np.random.default_rng(seed)
    action_dim = mech.action_dim
    state = np.asarray(cache["state"])
    z_all = np.asarray(cache["z"])

    # global true-distance oracle, fit once on a representative state sample (matches
    # graph_gate.py's convention: fit on the full landmark state array)
    true_dist_oracle = mech.build_true_distance_oracle(state[::7])

    # real action column stats, for de-normalizing CEM-like z-scored candidates -- the exact
    # inverse of WorldModelPolicy's process['action'] StandardScaler (verified in policy.py)
    with h5py.File(str(mech.h5_path(root)), "r", swmr=True) as f5:
        action_col = f5["action"][:len(ep_idx)]
    act_mean, act_std = action_col.mean(0), action_col.std(0)
    act_std[act_std < 1e-6] = 1.0
    log(f"[cem-pool] action stats: mean={act_mean} std={act_std}")

    # next_row/next_action_block/chain_ok give us, for a start row, both the real 25-step
    # continuation (the goal, matching E4's protocol) and a fast validity check
    next_row, next_action_block, chain_ok = build_block_chain_lookup(ep_idx, step_idx, action_col, skip=5)
    tr_pool = np.nonzero(chain_ok & np.isin(ep_idx, list(train_eps)))[0]
    starts = rng.choice(tr_pool, size=N_CANDIDATE_GROUPS, replace=False)
    goals = starts.copy()
    for _ in range(5):   # walk the real 5-block (25-step) chain to the goal row
        goals = next_row[goals]

    model = load_lewm(ckpt_dir=mech.ckpt_dir(root), device=DEV)
    model.interpolate_pos_encoding = True

    env = mech.make_env()

    all_z_hat, all_true_dist, all_group_id, all_start_row, all_goal_row = [], [], [], [], []
    t0 = time.time()
    for gi, (s_row, g_row) in enumerate(zip(starts, goals)):
        start_state = state[s_row].astype(np.float64)
        goal_state = state[g_row].astype(np.float64)

        # K CEM-like candidates: z-scored per-real-action Gaussian noise, var_scale=1.0
        cand_z = rng.standard_normal((K_CANDIDATES, CEM_REAL_STEPS, action_dim)).astype(np.float32) * np.sqrt(CEM_VAR_SCALE)
        cand_raw = cand_z * act_std[None, None, :] + act_mean[None, None, :]

        true_d = np.empty(K_CANDIDATES, dtype=np.float32)
        for k in range(K_CANDIDATES):
            env.reset(options={"state": start_state.copy(), "goal_state": goal_state.copy()})
            obs = None
            for t in range(CEM_REAL_STEPS):
                obs, reward, terminated, truncated, info = env.step(cand_raw[k, t])
                if terminated or truncated:
                    break
            final_state = np.asarray(obs["state"], dtype=np.float64)
            true_d[k] = true_dist_oracle(final_state[None, :], goal_state[None, :])[0, 0]

        # predictor rollout for the SAME z-scored candidates, from the real cached z0
        with torch.no_grad():
            z0 = torch.from_numpy(np.tile(z_all[s_row], (K_CANDIDATES, 1))).float().to(DEV)
            blocks = torch.from_numpy(cand_z.reshape(K_CANDIDATES, 5, 5 * action_dim)).float().to(DEV)
            z_hat = predictor_rollout_from_z0(model, z0, blocks).cpu().numpy()

        all_z_hat.append(z_hat)
        all_true_dist.append(true_d)
        all_group_id.append(np.full(K_CANDIDATES, gi, dtype=np.int64))
        all_start_row.append(np.full(K_CANDIDATES, s_row, dtype=np.int64))
        all_goal_row.append(np.full(K_CANDIDATES, g_row, dtype=np.int64))

        if gi % 20 == 0:
            el = time.time() - t0
            log(f"[cem-pool] group {gi}/{N_CANDIDATE_GROUPS} elapsed={el:.1f}s "
                f"eta={el/(gi+1)*(N_CANDIDATE_GROUPS-gi-1)/60:.1f}min")

    env.close()
    pool = dict(
        z_hat=np.concatenate(all_z_hat).astype(np.float32),
        true_dist=np.concatenate(all_true_dist).astype(np.float32),
        group_id=np.concatenate(all_group_id),
        start_row=np.concatenate(all_start_row),
        goal_row=np.concatenate(all_goal_row),
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, **pool)
    log(f"[cem-pool] wrote {cache_path} ({len(pool['true_dist'])} rows, {time.time()-t0:.1f}s)")
    return pool


def sample_ranking_batch(pool, n_groups, rng):
    """Sample n_groups candidate-groups and build all valid (i,j) comparison pairs within
    each group whose true-distance gap exceeds the noise floor -- these are what the margin
    ranking loss trains on."""
    all_group_ids = np.unique(pool["group_id"])
    chosen = rng.choice(all_group_ids, size=min(n_groups, len(all_group_ids)), replace=False)
    rows_a, rows_b, sign = [], [], []
    for g in chosen:
        idx = np.nonzero(pool["group_id"] == g)[0]
        td = pool["true_dist"][idx]
        K = len(idx)
        for a in range(K):
            for b in range(a + 1, K):
                gap = td[b] - td[a]
                if abs(gap) < RANK_MARGIN_NOISE_FLOOR:
                    continue
                if gap > 0:
                    rows_a.append(idx[a]); rows_b.append(idx[b])
                else:
                    rows_a.append(idx[b]); rows_b.append(idx[a])
    return np.array(rows_a), np.array(rows_b)   # rows_a should rank CLOSER (smaller d_Q) than rows_b


@hydra.main(version_base=None, config_path=str(ROOT / "config" / "graph"), config_name="probe_gate")
def main(cfg: DictConfig):
    t0 = time.time()
    mech = ENV_MECHANICS[cfg.env.name]
    res_dir = ROOT / "outputs" / "quasimetric"
    res_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(os.environ.get("PROBE_CACHE_DIR",
                                    str(ROOT / "outputs" / f"probe_gate_{cfg.env.output_prefix}" / "cache")))

    log(f"=== E3-v3 train ranking-over-CEM-like-candidates: {cfg.env.name} (device={DEV}) ===")

    tag = f"ep{cfg.n_episodes}_seed{cfg.seed}_g{cfg.pixel_grid}"
    need = ["z", "ep_idx", "step_idx", "state"]
    cache = {k: np.load(cache_dir / f"{k}_{tag}.npy", mmap_mode="r") for k in need}
    ep_idx, step_idx = np.asarray(cache["ep_idx"]), np.asarray(cache["step_idx"])
    state = np.asarray(cache["state"])
    n = len(ep_idx)
    log(f"cache: {n} frames, {len(np.unique(ep_idx))} episodes")

    train_eps, test_eps, rows_tr, rows_te = episode_split(ep_idx, cfg.train_frac, cfg.seed)
    log(f"split: {len(train_eps)} train / {len(test_eps)} test episodes")

    arm = Arm("lewm", cache["z"], cache["z"].shape[1])
    arm.fit_norm(rows_tr, cfg.seed)

    tr = make_pairs(ep_idx, step_idx, train_eps, cfg.pairs_per_episode, cfg.seed, cfg.max_gap)
    te = make_pairs(ep_idx, step_idx, test_eps, cfg.pairs_per_episode, cfg.seed + 1, cfg.max_gap)
    tr_s, tr_g, tr_gap = tr
    te_s, te_g, te_gap = te
    gtr = torch.from_numpy(np.log1p(tr_gap)).float().to(DEV)

    loc_s, loc_g = gap1_pairs(ep_idx, step_idx, train_eps, n_per_episode=20, seed=cfg.seed + 2)

    A_all, B_all = find_active_pushing_pairs(state, ep_idx, step_idx)
    is_train_pair = np.isin(ep_idx[A_all], list(train_eps))
    asym_tr_A, asym_tr_B = A_all[is_train_pair], B_all[is_train_pair]

    log("precomputing CEM-like candidate pool (real env stepping -- this is the slow part)...")
    pool = precompute_cem_candidate_pool(mech, cache, ep_idx, step_idx, train_eps, ROOT, seed=cfg.seed)
    n_groups = len(np.unique(pool["group_id"]))
    log(f"pool: {n_groups} groups x up to {K_CANDIDATES} candidates = {len(pool['true_dist'])} rows")

    z_hat_all_n = np.clip((pool["z_hat"] - arm.mu[0]) / arm.sd[0], -10.0, 10.0)
    z_goal_all_n = np.clip((np.asarray(cache["z"][pool["goal_row"]]) - arm.mu[0]) / arm.sd[0], -10.0, 10.0)

    quasi_model = LatentQuasimetric(latent_dim=arm.dim, hidden_dim=256, proj_dim=64).to(DEV)
    opt = torch.optim.AdamW(quasi_model.parameters(), lr=LR, weight_decay=1e-4)
    rng = np.random.default_rng(cfg.seed)

    log(f"training {N_STEPS} steps, batch={BATCH_SIZE} rank / {LOCAL_BATCH} local / "
        f"{ASYM_BATCH} asym / {GROUP_BATCH} CEM-candidate groups per step")
    for step in range(N_STEPS):
        rb = rng.integers(0, len(tr_gap), BATCH_SIZE)
        d_fwd = quasi_model(arm.get(tr_s[rb]), arm.get(tr_g[rb]))
        loss_rank = F.smooth_l1_loss(d_fwd, gtr[rb])

        lb = rng.integers(0, len(loc_s), LOCAL_BATCH)
        d_local = quasi_model(arm.get(loc_s[lb]), arm.get(loc_g[lb]))
        loss_local = F.relu(d_local - 1.0).pow(2).mean()

        ab = rng.integers(0, len(asym_tr_A), ASYM_BATCH)
        d_af = quasi_model(arm.get(asym_tr_A[ab]), arm.get(asym_tr_B[ab]))
        d_ab = quasi_model(arm.get(asym_tr_B[ab]), arm.get(asym_tr_A[ab]))
        loss_asym = F.relu(d_af.detach() + ASYM_MARGIN - d_ab).mean()

        ra, rb_ = sample_ranking_batch(pool, GROUP_BATCH, rng)
        if len(ra) > 0:
            za = torch.from_numpy(z_hat_all_n[ra]).float().to(DEV)
            zb = torch.from_numpy(z_hat_all_n[rb_]).float().to(DEV)
            ga = torch.from_numpy(z_goal_all_n[ra]).float().to(DEV)
            gb = torch.from_numpy(z_goal_all_n[rb_]).float().to(DEV)
            d_a = quasi_model(za, ga)   # should rank CLOSER (smaller)
            d_b = quasi_model(zb, gb)   # should rank FARTHER (larger)
            loss_cem_rank = F.relu(1.0 + d_a - d_b).mean()   # margin-ranking hinge
        else:
            loss_cem_rank = torch.zeros((), device=DEV)

        loss = (W_RANK * loss_rank + W_LOCAL * loss_local + W_ASYM * loss_asym
               + W_CEM_RANK * loss_cem_rank)
        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % max(1, N_STEPS // 10) == 0:
            log(f"  step {step}/{N_STEPS} loss={loss.item():.4f} "
                f"(rank={loss_rank.item():.4f} local={loss_local.item():.4f} "
                f"asym={loss_asym.item():.4f} cem_rank={loss_cem_rank.item():.4f} "
                f"n_pairs={len(ra)}) ({time.time()-t0:.0f}s)")

    quasi_model.eval()

    with torch.no_grad():
        def d_q_fwd(s_rows, g_rows):
            return _chunked(lambda ix: quasi_model(arm.get(s_rows[ix]), arm.get(g_rows[ix]))
                            .cpu().numpy(), len(s_rows))
        dq_te = d_q_fwd(te_s, te_g)
    z_mm = cache["z"]
    l2_te = np.linalg.norm(np.asarray(z_mm[te_s]) - np.asarray(z_mm[te_g]), axis=1)
    rho_q = float(sps.spearmanr(dq_te, te_gap).statistic)
    rho_l2 = float(sps.spearmanr(l2_te, te_gap).statistic)
    log(f"\n=== E3 gate (real pairs, unchanged eval): Spearman(d_Q)={rho_q:.4f} vs raw L2={rho_l2:.4f} "
        f"-> {'PASSED' if rho_q > rho_l2 else 'FAILED'}")

    # -- held-out-style check on the SAME (train-only) CEM pool: within-group rank accuracy
    # (no separate held-out pool was precomputed, given the cost of real env stepping --
    # stated as a scope limitation, not hidden)
    with torch.no_grad():
        d_pool = _chunked(lambda ix: quasi_model(
            torch.from_numpy(z_hat_all_n[ix]).float().to(DEV),
            torch.from_numpy(z_goal_all_n[ix]).float().to(DEV)).cpu().numpy(), len(pool["true_dist"]))
    rho_within = []
    for g in np.unique(pool["group_id"]):
        idx = np.nonzero(pool["group_id"] == g)[0]
        if len(idx) > 2 and np.ptp(pool["true_dist"][idx]) > 1e-6:
            r = sps.spearmanr(pool["true_dist"][idx], d_pool[idx]).statistic
            if np.isfinite(r):
                rho_within.append(r)
    mean_within_rho = float(np.mean(rho_within)) if rho_within else float("nan")
    log(f"within-group Spearman(true CEM-candidate distance, d_Q), mean over {len(rho_within)} groups: "
        f"{mean_within_rho:.4f}")

    res = dict(
        env=cfg.env.name, n_steps=N_STEPS, n_groups=n_groups, k_candidates=K_CANDIDATES,
        gate=dict(spearman_quasimetric=rho_q, spearman_raw_l2=rho_l2, passed=bool(rho_q > rho_l2)),
        cem_candidate_ranking=dict(mean_within_group_spearman=mean_within_rho,
                                   n_groups_scored=len(rho_within)),
        secs=time.time() - t0,
    )
    p = res_dir / f"e3v3_train_ranking_cem_like_{cfg.env.name}.json"
    p.write_text(json.dumps(res, indent=2))
    torch.save(quasi_model.state_dict(), res_dir / f"e3v3_quasimetric_head_{cfg.env.name}.pt")
    log(f"wrote {p} ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
