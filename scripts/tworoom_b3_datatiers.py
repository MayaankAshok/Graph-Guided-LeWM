"""B3 across data-quality tiers (a dataset-size sweep + a genuine mixed-quality tier),
GCIQL only for now (GCIVL is a separate, later pass). Same eval methodology as
tworoom_b3_convergence.py -- train/test split, fixed precomputed eval sets, eval every 50
steps, peak-based comparison -- UNCHANGED, per instruction. This script adds: multiple data
tiers, crash-safe resume, and (superseding an earlier, rejected approach) a mixed tier built
from genuine live rollouts rather than post-hoc action-label corruption.

Data tiers:

  expert_N (N in DATASET_SIZES): the real expert dataset at N episodes, for several N --
    a size sweep, testing whether stitching (and therefore shaping) matters more when
    there's less data to learn from, per the proposal's own B3 risk paragraph.

  mixed: a genuine blend of MIXED_N_EXPERT real expert episodes (from the h5 dataset) and
    MIXED_N_NOISY freshly-collected episodes rolled out live in the actual TwoRoomEnv with
    ExpertPolicy(action_noise=MIXED_NOISE, action_repeat_prob=MIXED_REPEAT_PROB) -- i.e.
    noisy ACTIONS that actually perturb the resulting STATE trajectory (validated in
    tworoom_rollout_collector.py: mean episode length grows monotonically with noise,
    25.3 -> 34.5 steps from noise=0 to noise=0.7, confirming the noise genuinely degrades
    behavior rather than just relabeling). This replaces an earlier version of this script
    that corrupted the *action label* on top of otherwise-real recorded transitions without
    the state/reward actually reflecting that worse action -- flagged as too hacky and
    removed; not used anywhere in this repo's results.

Resume: every (tier, condition, seed) run is checkpointed to its own file after every eval
step (model weights for all 4 networks, optimizer state, RNG state, and the history list
so far). Checkpoints are written atomically (temp file + os.replace) so a kill mid-write
can't corrupt the file. There is exactly ONE checkpoint per run, always overwritten with
the latest state -- never a separate "best" checkpoint -- so re-running this script always
resumes each run from wherever it last got to, or skips it entirely if it already finished.
"""

import json
import os
import sys
import time
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401
import numpy as np
import torch
from scipy import stats as sps
from scipy.sparse.csgraph import dijkstra

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.checkpoint_io import load_checkpoint, save_checkpoint  # noqa: F401
from tworoom_b0_graph_gate import DEV, H5_PATH, build_true_distance_oracle, build_weighted_graph, find_graph_edges
from tworoom_rollout_collector import collect_rollouts, encode_pixels
from investigations.early_diagnostics.tworoom_b1_graph_diagnostics import load_landmarks
from tworoom_b3_gciql_shaping import MLP, ema_update, build_her_tuples, log
from tworoom_b3_convergence import precompute_eval_set, check_convergence

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "outputs" / "b3_tworoom"
CKPT_DIR = OUT_DIR / "checkpoints"
TIER_CACHE_DIR = OUT_DIR / "tier_cache"
for d_ in (OUT_DIR, CKPT_DIR, TIER_CACHE_DIR):
    d_.mkdir(parents=True, exist_ok=True)

TRAIN_FRAC = 0.8
Q_CALIB = 1e-3
GAMMA = 0.99
TAU_EXPECTILE = 0.9
EMA_TAU = 0.005
HIDDEN = 256
BATCH_SIZE = 256
N_STEPS = 50000
EVAL_EVERY = 50
N_EVAL_PAIRS = 1000
GOALS_PER_TRANSITION = 4
GOAL_GAMMA = 0.9
SEEDS = [0, 1, 2]

DATASET_SIZES = [10, 25, 50, 100]  # episode counts for the size sweep, all real expert data
MIXED_N_EXPERT = 50    # real expert episodes contributed to the mixed tier
MIXED_N_NOISY = 50     # freshly-rolled-out noisy episodes contributed to the mixed tier
MIXED_NOISE = 0.4      # ExpertPolicy action_noise -- validated to reliably lengthen episodes
MIXED_REPEAT_PROB = 0.15
MIXED_SEED = 7
NOISY_EP_ID_OFFSET = 1_000_000  # keeps freshly-collected episode ids disjoint from real ones (0-9999)

TIERS = [f"expert_{n}" for n in DATASET_SIZES] + ["mixed"]


def _load_real_tier_arrays(n_episodes):
    """Real expert data, n_episodes worth, in the (z, ep_idx, step_idx, proprio, action)
    bundle format every tier setup is built from."""
    z, ep_idx, step_idx, proprio = load_landmarks(n_episodes)
    f = h5py.File(H5_PATH, "r", swmr=True, rdcc_nbytes=256 * 1024 * 1024)
    ep_offset = f["ep_offset"][:]
    global_idx = ep_offset[ep_idx] + step_idx
    action = f["action"][:][global_idx].astype(np.float32)
    f.close()
    return z, ep_idx, step_idx, proprio, action


def _build_mixed_arrays():
    """Real expert episodes + freshly-collected noisy-policy episodes, genuinely rolled out
    in the live env (not a post-hoc action-label swap -- see module docstring)."""
    ez, e_ep, e_step, e_proprio, e_action = _load_real_tier_arrays(MIXED_N_EXPERT)

    rollout = collect_rollouts(
        n_episodes=MIXED_N_NOISY, action_noise=MIXED_NOISE, action_repeat_prob=MIXED_REPEAT_PROB,
        seed=MIXED_SEED, cache_name=f"mixed_noisy_n{MIXED_N_NOISY}_noise{MIXED_NOISE}",
    )
    log(f"[mixed] encoding {len(rollout['pixels'])} freshly-collected frames with the frozen LeWM encoder...")
    nz = encode_pixels(rollout["pixels"])
    n_ep_idx = rollout["ep_idx"] + NOISY_EP_ID_OFFSET  # disjoint from real episode ids (0-9999)

    z = np.concatenate([ez, nz], axis=0)
    ep_idx = np.concatenate([e_ep, n_ep_idx], axis=0)
    step_idx = np.concatenate([e_step, rollout["step_idx"]], axis=0)
    proprio = np.concatenate([e_proprio, rollout["proprio"]], axis=0)
    action = np.concatenate([e_action, rollout["action"]], axis=0)
    log(f"[mixed] combined: {len(ez)} real-expert frames ({MIXED_N_EXPERT} eps) + "
        f"{len(nz)} noisy-rollout frames ({MIXED_N_NOISY} eps) = {len(z)} total")
    return z, ep_idx, step_idx, proprio, action


# ============================================================
# Tier setup: landmarks, episode split, graph, phi_dist, eval sets
# ============================================================

def _build_setup(tier_key, z, ep_idx, step_idx, proprio, action):
    """tier_key is the cache key for this tier's precomputed phi_dist matrix."""
    cache_path = TIER_CACHE_DIR / f"{tier_key}_phi_dist.npy"
    n, d = z.shape

    uniq_eps = np.unique(ep_idx)
    rng = np.random.default_rng(0)
    shuffled = rng.permutation(uniq_eps)
    n_train = int(len(shuffled) * TRAIN_FRAC)
    train_eps, test_eps = shuffled[:n_train], shuffled[n_train:]
    train_rows = np.nonzero(np.isin(ep_idx, train_eps))[0]
    test_rows = np.nonzero(np.isin(ep_idx, test_eps))[0]
    log(f"[{tier_key}] {n} landmarks, {len(uniq_eps)} episodes "
        f"({len(train_eps)} train / {len(test_eps)} test episodes, "
        f"{len(train_rows)} / {len(test_rows)} rows)")

    order = np.lexsort((step_idx, ep_idx))
    ep_o, step_o = ep_idx[order], step_idx[order]
    adj = (ep_o[1:] == ep_o[:-1]) & (step_o[1:] == step_o[:-1] + 1)
    z_o = z[order]
    rho_hat = float(np.mean(np.sum(z_o[:-1][adj] * z_o[1:][adj], axis=1)) / d)
    eps2 = 2 * (1 - rho_hat) * sps.chi2.ppf(Q_CALIB, d)

    trans_i, trans_j, id_i, id_j = find_graph_edges(z, ep_idx, step_idx, eps2)
    graph = build_weighted_graph(n, trans_i, trans_j, id_i, id_j, id_weight=1.0)
    log(f"[{tier_key}] graph: {len(trans_i)} transition edges, {len(id_i)} identification edges "
        f"(rho_hat={rho_hat:.4f} eps2={eps2:.2f})")

    if cache_path.exists():
        log(f"[{tier_key}] loading cached phi_dist")
        phi_dist = np.load(cache_path)
    else:
        log(f"[{tier_key}] precomputing full pairwise graph-distance matrix...")
        t0 = time.time()
        phi_dist = dijkstra(graph, indices=np.arange(n), directed=False)
        finite_max = phi_dist[np.isfinite(phi_dist)].max()
        phi_dist = np.where(np.isfinite(phi_dist), phi_dist, finite_max * 2)
        log(f"[{tier_key}] done in {time.time()-t0:.1f}s")
        np.save(cache_path, phi_dist)

    true_dist_oracle = build_true_distance_oracle()
    train_eval = precompute_eval_set(train_rows, proprio, true_dist_oracle, N_EVAL_PAIRS, seed=100)
    test_eval = precompute_eval_set(test_rows, proprio, true_dist_oracle, N_EVAL_PAIRS, seed=101)
    log(f"[{tier_key}] eval pairs: {len(train_eval[0])} train, {len(test_eval[0])} test")

    return dict(z=z, ep_idx=ep_idx, step_idx=step_idx, proprio=proprio, action=action,
                train_eps=train_eps, test_eps=test_eps, phi_dist=phi_dist,
                train_eval=train_eval, test_eval=test_eval, d=d)


def get_true_dist_matrix(tier_key, proprio, chunk=1000, cache_dir=None):
    """Full NxN ground-truth (pixel-space, wall-respecting) distance matrix for a tier's
    landmarks -- the noise-free ceiling condition for the B4 distance-source ablation
    (graph vs euclidean vs this oracle, all fed through the same auxiliary-regression
    mechanism). Same true_dist_oracle used for eval-set true distances, just queried for
    every pair instead of a sampled few thousand. Chunked over sources to avoid a
    (N, 33925)-shaped intermediate float64 array -- for mixed_large's ~10k landmarks that
    intermediate would be ~2.8GB if done in one shot; chunking keeps peak memory bounded to
    one chunk's slice while the cached *result* is still the full dense NxN (same size
    class as phi_dist, e.g. ~864MB for mixed_large)."""
    cache_path = (cache_dir or TIER_CACHE_DIR) / f"{tier_key}_truedist.npy"
    if cache_path.exists():
        log(f"[{tier_key}] loading cached true_dist matrix")
        return np.load(cache_path)

    n = len(proprio)
    true_dist_oracle = build_true_distance_oracle()
    log(f"[{tier_key}] precomputing full pairwise TRUE-distance matrix ({n}x{n}, chunk={chunk})...")
    t0 = time.time()
    out = np.empty((n, n), dtype=np.float32)
    for i0 in range(0, n, chunk):
        i1 = min(n, i0 + chunk)
        D = true_dist_oracle(proprio[i0:i1], proprio)  # (chunk, n)
        finite = np.isfinite(D)
        if not finite.all():
            D = np.where(finite, D, np.nanmax(np.where(finite, D, np.nan)) * 2 if finite.any() else 0.0)
        out[i0:i1] = D.astype(np.float32)
        if (i0 // chunk) % 5 == 0:
            log(f"[{tier_key}] true_dist rows {i1}/{n} elapsed={time.time()-t0:.1f}s")
    log(f"[{tier_key}] true_dist matrix done in {time.time()-t0:.1f}s")
    np.save(cache_path, out)
    return out


def build_tier_setups(tiers=None):
    """Returns {tier_name: setup_dict}, built lazily only for the requested tiers so a
    partial rerun (e.g. just the mixed tier) doesn't pay for the whole sweep's setup cost."""
    tiers = tiers if tiers is not None else TIERS
    setups = {}
    for tier in tiers:
        if tier.startswith("expert_"):
            n = int(tier.split("_")[1])
            arrays = _load_real_tier_arrays(n)
            setups[tier] = _build_setup(tier, *arrays)
        elif tier == "mixed":
            arrays = _build_mixed_arrays()
            setups[tier] = _build_setup("mixed", *arrays)
        else:
            raise ValueError(f"unknown tier '{tier}'")
    return setups


# ============================================================
# Checkpointing -- save_checkpoint/load_checkpoint moved to common/checkpoint_io.py
# (imported at module top); ckpt_path stays local since CKPT_DIR is this script's own.
# ============================================================

def ckpt_path(tier, name, seed):
    return CKPT_DIR / f"{tier}__{name}__s{seed}.pt"


# ============================================================
# Training, resumable
# ============================================================

def run_condition_resumable(tier, name, use_shaping, setup, seed):
    path = ckpt_path(tier, name, seed)
    z, action, phi_dist = setup["z"], setup["action"], setup["phi_dist"]
    d = setup["d"]

    # always load to CPU first: map_location=DEV would move the saved CUDA RNG state tensor
    # onto the GPU (and break its required ByteTensor-on-CPU dtype), which
    # torch.cuda.set_rng_state then rejects. Model weights don't need the checkpoint's own
    # device -- load_state_dict copies onto whatever device the already-constructed
    # (DEV-resident) module is on, regardless of source device. load_checkpoint also falls
    # back to a .bak generation if the primary file is missing/corrupt.
    ckpt = load_checkpoint(path)
    if ckpt is not None and ckpt.get("done"):
        log(f"[{tier}/{name}/s{seed}] already complete ({ckpt['step']} steps) -- skipping")
        return ckpt["history"]

    torch.manual_seed(seed)
    z_t = torch.from_numpy(z).to(DEV)
    act_t = torch.from_numpy(action).float().to(DEV)

    v_net = MLP(2 * d, HIDDEN).to(DEV)
    v_target = MLP(2 * d, HIDDEN).to(DEV)
    q_net = MLP(2 * d + act_t.shape[1], HIDDEN).to(DEV)
    q_target = MLP(2 * d + act_t.shape[1], HIDDEN).to(DEV)
    v_target.load_state_dict(v_net.state_dict())
    q_target.load_state_dict(q_net.state_dict())

    from stable_worldmodel.wm.gcrl.module import ExpectileLoss
    expectile_loss = ExpectileLoss(tau=TAU_EXPECTILE)
    opt = torch.optim.Adam(list(v_net.parameters()) + list(q_net.parameters()), lr=3e-4)

    s_idx, next_idx, goal_idx, act_idx, done_arr = build_her_tuples(
        setup["ep_idx"], setup["step_idx"], action, seed=seed, allowed_episode_ids=setup["train_eps"]
    )
    n_tuples = len(s_idx)

    train_ei, train_ej, train_true_d = setup["train_eval"]
    test_ei, test_ej, test_true_d = setup["test_eval"]
    train_zsg = torch.cat([z_t[train_ei], z_t[train_ej]], dim=-1)
    test_zsg = torch.cat([z_t[test_ei], z_t[test_ej]], dim=-1)

    rng = np.random.default_rng(seed + 1)
    history = []
    start_step = 1

    if ckpt is not None:
        v_net.load_state_dict(ckpt["v_net"]); v_target.load_state_dict(ckpt["v_target"])
        q_net.load_state_dict(ckpt["q_net"]); q_target.load_state_dict(ckpt["q_target"])
        opt.load_state_dict(ckpt["opt"])
        history = ckpt["history"]
        start_step = ckpt["step"] + 1
        rng.bit_generator.state = ckpt["numpy_rng"]
        torch.set_rng_state(ckpt["torch_rng"])  # CPU ByteTensor, matches torch.get_rng_state()
        if torch.cuda.is_available() and ckpt.get("torch_cuda_rng") is not None:
            torch.cuda.set_rng_state(ckpt["torch_cuda_rng"])  # also CPU ByteTensor by contract
        log(f"[{tier}/{name}/s{seed}] resuming from step {start_step}/{N_STEPS} "
            f"({len(history)} eval points so far)")
    else:
        log(f"[{tier}/{name}/s{seed}] starting fresh, n_tuples={n_tuples}")

    if start_step > N_STEPS:
        return history

    t0 = time.time()
    for step in range(start_step, N_STEPS + 1):
        batch = rng.integers(0, n_tuples, BATCH_SIZE)
        s, nx, g, a_i, dn = s_idx[batch], next_idx[batch], goal_idx[batch], act_idx[batch], done_arr[batch]

        zs, zn, zg = z_t[s], z_t[nx], z_t[g]
        a = act_t[a_i]
        mask = (~torch.from_numpy(dn).to(DEV)).float().unsqueeze(-1)
        reward = -mask

        if use_shaping:
            phi_s = torch.from_numpy(-phi_dist[s, g]).float().to(DEV).unsqueeze(-1)
            phi_n = torch.from_numpy(-phi_dist[nx, g]).float().to(DEV).unsqueeze(-1)
            reward = reward + GAMMA * phi_n - phi_s

        sg = torch.cat([zs, zg], dim=-1)
        ng = torch.cat([zn, zg], dim=-1)
        sag = torch.cat([zs, a, zg], dim=-1)

        with torch.no_grad():
            q = q_target(sag)
        v = v_net(sg)
        value_loss = expectile_loss(v, q.detach())

        with torch.no_grad():
            next_v = v_target(ng)
            q_tgt = reward + GAMMA * mask * next_v
        q_pred = q_net(sag)
        critic_loss = ((q_pred - q_tgt) ** 2).mean()

        loss = value_loss + critic_loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        ema_update(v_target, v_net, EMA_TAU)
        ema_update(q_target, q_net, EMA_TAU)

        if step % EVAL_EVERY == 0 or step == 1:
            with torch.no_grad():
                v_train = v_net(train_zsg).squeeze(-1).cpu().numpy()
                v_test = v_net(test_zsg).squeeze(-1).cpu().numpy()
            rho_train, _ = sps.spearmanr(train_true_d, -v_train)
            rho_test, _ = sps.spearmanr(test_true_d, -v_test)
            history.append(dict(step=step, value_loss=float(value_loss.item()),
                                 critic_loss=float(critic_loss.item()),
                                 spearman_train=float(rho_train), spearman_test=float(rho_test)))
            is_done = step == N_STEPS
            save_checkpoint(path, step, v_net, v_target, q_net, q_target, opt, history, rng, is_done)
            if step % (EVAL_EVERY * 20) == 0 or step == 1:
                log(f"[{tier}/{name}/s{seed}] step={step} vloss={value_loss.item():.3f} "
                    f"closs={critic_loss.item():.3f} train_rho={rho_train:.4f} test_rho={rho_test:.4f} "
                    f"elapsed={time.time()-t0:.1f}s")

    return history


def peak_stats(history):
    i = max(range(len(history)), key=lambda k: history[k]["spearman_test"])
    return history[i]


def main():
    log(f"device={DEV}")
    setups = build_tier_setups()

    all_results = {}
    for tier in TIERS:
        setup = setups[tier]
        all_results[tier] = {}
        for use_shaping, name in [(False, "baseline"), (True, "shaped")]:
            all_results[tier][name] = []
            for seed in SEEDS:
                history = run_condition_resumable(tier, name, use_shaping, setup, seed)
                all_results[tier][name].append(dict(seed=seed, history=history))

        out_path = OUT_DIR / f"b3_datatiers_{tier}_results.json"
        out_path.write_text(json.dumps(all_results[tier], indent=2))
        log(f"wrote {out_path}")

    log("\n" + "=" * 70 + "\nSUMMARY: peak held-out-test Spearman by tier\n" + "=" * 70)
    for tier in TIERS:
        log(f"\n--- {tier} ---")
        for name in ["baseline", "shaped"]:
            peaks = []
            for r in all_results[tier][name]:
                p = peak_stats(r["history"])
                peaks.append(p["spearman_test"])
                log(f"  {name} seed={r['seed']}: peak={p['spearman_test']:.4f} @ step {p['step']}")
            log(f"  {name} mean peak = {np.mean(peaks):.4f} +- {np.std(peaks):.4f}")

    full_out = OUT_DIR / "b3_datatiers_all_results.json"
    full_out.write_text(json.dumps(all_results, indent=2))
    log(f"\nwrote {full_out}")


if __name__ == "__main__":
    main()
