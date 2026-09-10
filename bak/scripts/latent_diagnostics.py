"""Latent-space diagnostic figures for docs/graph-proposal/predictor-stitching.tex.

Generates histograms answering: how far apart are consecutive states vs. random states vs.
what a naive Gaussian would predict; how accurate is the frozen predictor at each of its
supported history lengths; how many effective dimensions does the embedding actually use;
does ensemble variance separate real transitions from the wide candidate pool it failed to
filter in E2; does cosine similarity behave the way the calibration formula assumes.

Run as:
    python scripts/latent_diagnostics.py env=tworoom
"""

import sys
from pathlib import Path

import hydra
import matplotlib
import numpy as np
import torch
from omegaconf import DictConfig

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.envs import ENV_MECHANICS
from common.graph_lib import (
    DEV, find_transition_edges, raw_topk_cross_episode_pairs, radius_cross_episode_pairs,
)
from common.idm_ensemble import ensemble_predict, train_idm_ensemble
from common.lewm_loader import load_lewm, load_local_jepa_checkpoint
from common.log_util import log
from idm_gate import load_landmarks_and_actions

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "docs" / "graph-proposal" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def build_block_history_chains(ep_idx, step_idx, H, skip, max_samples, seed):
    """Real (history-frames, action-blocks, target) chains matching the checkpoint's ACTUAL
    training convention (docs/lewm.txt App. D): one predictor "step" spans `skip` real
    environment steps, and the action fed for that step is the concatenation of the `skip`
    real actions taken in between (skip=5, action_dim=2 -> exactly 10 dims, matching
    action_encoder's input_dim with no padding needed at all -- see
    [[lewm-action-encoder-padding]] for why the earlier zero-padded single-action
    convention this replaces was wrong).

    H is the number of such `skip`-sized blocks of real history the predictor is given
    (frames spaced `skip' apart); the target is the real frame `H*skip` steps after the
    window start. Assumes each episode's landmark rows are every real step with no gaps
    (true for load_landmarks, which encodes every frame of each chosen episode)."""
    order = np.lexsort((step_idx, ep_idx))
    ep_o = ep_idx[order]
    boundaries = np.nonzero(np.diff(ep_o))[0] + 1
    starts_ep = np.concatenate(([0], boundaries))
    ends_ep = np.concatenate((boundaries, [len(order)]))

    frame_rows, action_block_rows, target_rows = [], [], []
    needed = H * skip
    for s, e in zip(starts_ep, ends_ep):
        ep_rows = order[s:e]
        L = len(ep_rows)
        for t in range(L - needed):
            frame_rows.append([ep_rows[t + k * skip] for k in range(H)])
            action_block_rows.append([ep_rows[t + k * skip: t + k * skip + skip] for k in range(H)])
            target_rows.append(ep_rows[t + needed])
    frame_rows = np.array(frame_rows)              # (N, H)
    action_block_rows = np.array(action_block_rows)  # (N, H, skip)
    target_rows = np.array(target_rows)            # (N,)

    if len(target_rows) > max_samples:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(target_rows), size=max_samples, replace=False)
        frame_rows, action_block_rows, target_rows = frame_rows[idx], action_block_rows[idx], target_rows[idx]
    return frame_rows, action_block_rows, target_rows


def predictor_error_at_block_history(model, z, action, frame_rows, action_block_rows,
                                      target_rows, chunk=2000):
    """Returns (errs, ratio): errs is the raw Euclidean prediction error; ratio is
    dist(pred, target) / dist(z_before, target), where z_before is the most recent history
    frame (the real state `skip` steps before the target) -- the natural "did the action
    block help at all" reference. ratio < 1 means the predicted block landed closer to the
    target than assuming no motion; ratio ~= 1 means the action block added ~no information
    beyond that."""
    N, H, skip = action_block_rows.shape
    errs = np.empty(N, dtype=np.float32)
    z_before = z[frame_rows[:, -1]]
    dist_before = np.linalg.norm(z_before - z[target_rows], axis=1)
    for lo in range(0, N, chunk):
        hi = min(lo + chunk, N)
        zi = torch.from_numpy(z[frame_rows[lo:hi]]).float().to(DEV)          # (b,H,D)
        a = action[action_block_rows[lo:hi]].reshape(hi - lo, H, -1)         # (b,H,skip*A) = (b,H,10)
        act = torch.from_numpy(a).float().to(DEV)
        with torch.no_grad():
            act_emb = model.action_encoder(act)
            pred = model.predict(zi, act_emb)[:, -1].cpu().numpy()
        errs[lo:hi] = np.linalg.norm(pred - z[target_rows[lo:hi]], axis=1)
    ratio = errs / np.maximum(dist_before, 1e-8)
    return errs, ratio


@hydra.main(version_base=None, config_path="../config/graph", config_name="idm_gate")
def main(cfg: DictConfig):
    mech = ENV_MECHANICS[cfg.env.name]
    predictor_kind = cfg.get("predictor", "block5")
    prefix = cfg.env.output_prefix  # keeps figures from different envs from overwriting each
                                     # other -- Two-Room's existing unprefixed files are left
                                     # as-is (predate this) rather than renamed
    if predictor_kind == "1step":
        assert cfg.env.name == "pusht", "the 1step checkpoint is Push-T only"
        prefix = f"{prefix}1step"
    log(f"=== latent-space diagnostics (env={cfg.env.name}, predictor={predictor_kind}) ===")
    z, ep_idx, step_idx, state, action, act_stats = load_landmarks_and_actions(mech, cfg)
    d = z.shape[1]
    trans_i, trans_j = find_transition_edges(ep_idx, step_idx)
    rng = np.random.default_rng(0)

    # --- 1. consecutive-state distances -------------------------------------------------
    adjacent_dist = np.linalg.norm(z[trans_i] - z[trans_j], axis=1)

    # --- 2. random-pair distances + theoretical Gaussian reference ----------------------
    n_rand = 50_000
    ri = rng.integers(0, len(z), n_rand)
    rj = rng.integers(0, len(z), n_rand)
    keep = ri != rj
    random_dist = np.linalg.norm(z[ri[keep]] - z[rj[keep]], axis=1)

    per_dim_std = z.std(axis=0)
    g1 = rng.normal(size=(n_rand, d)) * per_dim_std
    g2 = rng.normal(size=(n_rand, d)) * per_dim_std
    theoretical_dist = np.linalg.norm(g1 - g2, axis=1)

    fig, ax = plt.subplots(figsize=(6, 4))
    bins = np.linspace(0, max(random_dist.max(), theoretical_dist.max()), 100)
    ax.hist(random_dist, bins=bins, alpha=0.55, density=True, label="real random pairs")
    ax.hist(theoretical_dist, bins=bins, alpha=0.55, density=True,
            label="theoretical: iid Gaussian,\nmatched per-dim variance")
    ax.set_xlabel("Euclidean distance"); ax.set_ylabel("density")
    ax.set_title(f"Random-pair latent distance vs. theoretical Gaussian ({cfg.env.name})")
    ax.legend()
    fig.tight_layout(); fig.savefig(FIG_DIR / f"{prefix}_random_vs_theoretical.png", dpi=150)
    plt.close(fig)
    log(f"[fig] random_vs_theoretical.png -- real mean={random_dist.mean():.2f} "
        f"theoretical mean={theoretical_dist.mean():.2f}")

    fig, ax = plt.subplots(figsize=(6, 4))
    bins = np.linspace(0, max(adjacent_dist.max(), random_dist.max()), 120)
    ax.hist(adjacent_dist, bins=bins, alpha=0.6, density=True, label="consecutive (adjacent) states")
    ax.hist(random_dist, bins=bins, alpha=0.5, density=True, label="random pairs")
    ax.set_xlabel("Euclidean distance"); ax.set_ylabel("density")
    ax.set_title(f"Consecutive vs. random-pair latent distance ({cfg.env.name})")
    ax.legend()
    fig.tight_layout(); fig.savefig(FIG_DIR / f"{prefix}_adjacent_vs_random.png", dpi=150)
    plt.close(fig)
    log(f"[fig] adjacent_vs_random.png -- adjacent mean={adjacent_dist.mean():.2f} "
        f"median={np.median(adjacent_dist):.2f}")

    # --- 3. predictor error at history length 1, 2, 3 ------------------------------------
    log("loading predictor...")
    if predictor_kind == "1step":
        # genuinely-trained 1-step predictor: frozen quentinll/lewm-pusht encoder (same
        # encoder mech.ckpt_dir() landmarks were embedded with, so z is identical) + a small
        # ARPredictor/action_encoder/pred_proj trained from scratch on real (z_t, a_t) ->
        # z_{t+1} triples. num_frames=1 -- it only ever supports single-frame context, so no
        # H=2/3 multi-frame sweep (that was only ever a workaround for the block5 checkpoint's
        # frame-skip-5 convention, which this model was never trained under).
        model = load_local_jepa_checkpoint(
            ckpt_dir=ROOT / "data" / "checkpoints" / "lewm_pusht_1step_predictor",
            weights_file="weights_epoch_20.pt", device=DEV)
        SKIP = 1
        h_values = (1,)
    else:
        model = load_lewm(ckpt_dir=mech.ckpt_dir(ROOT), device=DEV)
        SKIP = 5  # real env steps per predictor "step" -- the checkpoint's actual training
                  # convention (docs/lewm.txt App. D), not 1
        h_values = (1, 2, 3)

    errs_by_h, ratio_by_h = {}, {}
    for H in h_values:
        frame_rows, action_block_rows, target_rows = build_block_history_chains(
            ep_idx, step_idx, H, skip=SKIP, max_samples=5000, seed=H)
        errs, ratio = predictor_error_at_block_history(
            model, z, action, frame_rows, action_block_rows, target_rows)
        errs_by_h[H] = errs
        ratio_by_h[H] = ratio
        log(f"[predictor H={H} blocks, {H * SKIP} real steps] n={len(errs)} "
            f"mean={errs.mean():.3f} median={np.median(errs):.3f} q90={np.quantile(errs, 0.9):.3f}  |  "
            f"ratio dist(pred,tgt)/dist(before,tgt): median={np.median(ratio):.3f} "
            f"frac<1={np.mean(ratio < 1):.3f} frac<0.5={np.mean(ratio < 0.5):.3f} "
            f"frac>1.5={np.mean(ratio > 1.5):.3f}")

    # SKIP=1 (pred1): there is exactly one H value (a bare single real action), so call it
    # that rather than "H=1 blocks (1 real steps)". SKIP>1 (block5): unchanged wording.
    def _series_label(H):
        if SKIP == 1:
            return "single action"
        return f"H={H} blocks ({H * SKIP} real steps)"

    fig, ax = plt.subplots(figsize=(6, 4))
    bins = np.linspace(0, max(e.max() for e in errs_by_h.values()), 100)
    for H, errs in errs_by_h.items():
        ax.hist(errs, bins=bins, alpha=0.5, density=True,
                label=f"{_series_label(H)}, mean={errs.mean():.2f}")
    ax.set_xlabel("prediction error, Euclidean distance ||pred - true target||")
    ax.set_ylabel("density")
    hist_title = "single action" if SKIP == 1 else f"{SKIP}-step blocks, by history length"
    ax.set_title(f"Predictor error, {hist_title} ({cfg.env.name})")
    ax.legend()
    fig.tight_layout(); fig.savefig(FIG_DIR / f"{prefix}_predictor_error_by_history.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    bins = np.linspace(0, 4, 100)
    for H, ratio in ratio_by_h.items():
        ax.hist(np.clip(ratio, 0, 4), bins=bins, alpha=0.5, density=True,
                label=f"{_series_label(H)} (median={np.median(ratio):.2f}, frac<1={np.mean(ratio < 1):.2f})")
    ax.axvline(1.0, color="black", linestyle="--", linewidth=1,
               label="ratio=1 (no better than no motion)")
    ax.set_xlabel("dist(pred, target) / dist(state before action, target)")
    ax.set_ylabel("density")
    ax.set_title(f"Predictor error relative to real movement, {hist_title} ({cfg.env.name})")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG_DIR / f"{prefix}_predictor_ratio_by_history.png", dpi=150)
    plt.close(fig)

    # --- bonus F: train vs. held-out episode generalization (pred1 only) -----------------
    # Does pred1's real single-step accuracy hold up on episodes it never trained on? Uses
    # the SAME split convention pred1 was actually trained under
    # (config/train/lewm_pusht_1step.yaml: split_mode=contiguous, train_split=0.8 -- first
    # 80% of the dataset in on-disk order is train, the rest held out) -- not the random
    # episode permutation used elsewhere in this script for the IDM ensemble's split.
    if predictor_kind == "1step":
        uniq_eps_sorted = np.sort(np.unique(ep_idx))
        n_train_eps = int(len(uniq_eps_sorted) * 0.8)
        contig_train_eps = uniq_eps_sorted[:n_train_eps]
        contig_test_eps = uniq_eps_sorted[n_train_eps:]

        all_frame_rows, all_action_block_rows, all_target_rows = build_block_history_chains(
            ep_idx, step_idx, H=1, skip=1, max_samples=200_000, seed=7)
        ep_of_sample = ep_idx[all_frame_rows[:, 0]]
        is_train = np.isin(ep_of_sample, contig_train_eps)
        is_test = np.isin(ep_of_sample, contig_test_eps)

        all_errs, _ = predictor_error_at_block_history(
            model, z, action, all_frame_rows, all_action_block_rows, all_target_rows)
        train_errs, test_errs = all_errs[is_train], all_errs[is_test]
        log(f"[train-vs-test] n_train={len(train_errs)} mean={train_errs.mean():.3f} "
            f"median={np.median(train_errs):.3f}  |  n_test={len(test_errs)} "
            f"mean={test_errs.mean():.3f} median={np.median(test_errs):.3f}")

        fig, ax = plt.subplots(figsize=(6, 4))
        bins = np.linspace(0, max(train_errs.max(), test_errs.max()), 100)
        ax.hist(train_errs, bins=bins, alpha=0.55, density=True,
                label=f"training episodes, first 80% (n={len(train_errs)}): "
                      f"mean={train_errs.mean():.2f}")
        ax.hist(test_errs, bins=bins, alpha=0.55, density=True,
                label=f"held-out episodes, last 20% (n={len(test_errs)}): "
                      f"mean={test_errs.mean():.2f}")
        ax.set_xlabel("prediction error, Euclidean distance ||pred - true target||")
        ax.set_ylabel("density")
        ax.set_title(f"Single-step prediction error: train vs. held-out episodes ({cfg.env.name})")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"{prefix}_predictor_error_train_vs_test.png", dpi=150)
        plt.close(fig)

    # --- bonus D: is the predictor at least pointing the right way? ----------------------
    # Even if the predicted endpoint is off by several units, does the predicted DISPLACEMENT
    # direction agree with the real displacement direction? Tested on a real one-block
    # (H=1, skip=5 real steps -- the checkpoint's actual training convention) transition,
    # AND on a CONTROL where the first action is replaced by a RANDOM real action (drawn
    # from elsewhere in the dataset) while the remaining 4 real continuation actions
    # (a2..a5) are kept exactly as they really occurred. This isolates whether the cosine
    # signal genuinely depends on the first action being correct, or whether the last 4
    # real actions alone would produce similar-looking agreement regardless of a1 -- i.e.
    # whether the earlier positive result was actually picking up action-sensitivity, or
    # just "4 of 5 real actions carry the whole signal no matter what a1 is."
    dir_frame_rows, dir_action_rows, dir_target_rows = build_block_history_chains(
        ep_idx, step_idx, H=1, skip=SKIP, max_samples=5000, seed=99)
    di = dir_frame_rows[:, 0]
    dj = dir_target_rows
    cont_actions = action[dir_action_rows[:, 0, 1:]]      # (n, SKIP-1, A): real a2..a5
    real_a1 = action[dir_action_rows[:, 0, 0]]            # (n, A): real a1
    # sample from trans_i (real transition SOURCES) rather than all landmark rows -- a
    # landmark can be the LAST frame of its episode, whose stored action is NaN (padding),
    # and trans_i is guaranteed to exclude those (every trans_i has a real next step)
    rand_a1 = action[trans_i[rng.integers(0, len(trans_i), size=len(di))]]  # (n, A)

    def cos_for_block(a1):
        block = np.concatenate([a1[:, None, :], cont_actions], axis=1).reshape(len(di), -1)
        zi_t = torch.from_numpy(z[di]).float().to(DEV).unsqueeze(1)
        act_t = torch.from_numpy(block).float().to(DEV).unsqueeze(1)
        with torch.no_grad():
            act_emb = model.action_encoder(act_t)
            pred = model.predict(zi_t, act_emb)[:, -1].cpu().numpy()
        v_pred = pred - z[di]
        v_real = z[dj] - z[di]
        return np.sum(v_pred * v_real, axis=1) / (
            np.linalg.norm(v_pred, axis=1) * np.linalg.norm(v_real, axis=1) + 1e-8)

    cos_dir_real = cos_for_block(real_a1)
    cos_dir_rand = cos_for_block(rand_a1)
    # SKIP=1 (pred1): the "block" is a single real action, full stop -- no a2..a5 continuation
    # exists, so don't label it as if there were one. SKIP>1 (block5): unchanged wording.
    cont_clause = f", real $a_2..a_{SKIP}$" if SKIP > 1 else ""
    log(f"[direction] real a1: mean={cos_dir_real.mean():.3f} median={np.median(cos_dir_real):.3f} "
        f"frac>0.5={np.mean(cos_dir_real > 0.5):.3f}")
    log(f"[direction] RANDOM a1 (control){', real a2..a' + str(SKIP) if SKIP > 1 else ' (single action)'}: "
        f"mean={cos_dir_rand.mean():.3f} median={np.median(cos_dir_rand):.3f} frac>0.5={np.mean(cos_dir_rand > 0.5):.3f}")

    fig, ax = plt.subplots(figsize=(6, 4))
    bins = np.linspace(-1, 1, 100)
    ax.hist(cos_dir_real, bins=bins, density=True, alpha=0.55,
            label=f"real $a_1$ (median={np.median(cos_dir_real):.2f})")
    ax.hist(cos_dir_rand, bins=bins, density=True, alpha=0.55,
            label=f"random $a_1${cont_clause} (median={np.median(cos_dir_rand):.2f})")
    ax.axvline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel(r"$\cos(\mathrm{pred}(\mathbf{z}_1,\mathrm{block}) - \mathbf{z}_1,\ \ \mathbf{z}_2 - \mathbf{z}_1)$")
    ax.set_ylabel("density")
    title_kind = "single action" if SKIP == 1 else "block, real vs. random $a_1$"
    ax.set_title(f"Predictor direction agreement, {title_kind} ({cfg.env.name})")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG_DIR / f"{prefix}_predictor_direction_agreement.png", dpi=150)
    plt.close(fig)

    # --- bonus E: cross-episode splice verification ---------------------------------------
    # For a wide-kNN candidate pair (i, j) from DIFFERENT episodes, with an IDM-inferred
    # single connecting action a: build the hybrid action block (a, a_j, a_{j+1}, a_{j+2},
    # a_{j+3}) -- the IDM's guess in the first slot, j's own REAL next 4 actions in the
    # rest -- and predict from z_i. If a genuinely identifies i with j, continuing with j's
    # own real actions should land you where j's own rollout actually goes: z_{j+4}. Compare
    # this hybrid-block error against the REAL in-distribution block error just measured
    # above (errs_by_h[1]) -- that is the correct reference scale, not an arbitrary
    # threshold, since both are exactly one predictor call on a `SKIP`-sized action block.
    log("=== cross-episode splice verification (IDM action + real continuation) ===")
    cand_i, cand_j = raw_topk_cross_episode_pairs(z, ep_idx, k=20)
    # need j to have >= SKIP-1 real steps remaining in its own episode
    order = np.lexsort((step_idx, ep_idx))
    ep_o, step_o = ep_idx[order], step_idx[order]
    pos_in_order = np.empty(len(z), dtype=np.int64)
    pos_in_order[order] = np.arange(len(order))
    ep_len = np.zeros(ep_idx.max() + 1, dtype=np.int64)
    for e in np.unique(ep_idx):
        ep_len[e] = (ep_idx == e).sum()
    valid = (step_idx[cand_j] + (SKIP - 1)) < ep_len[ep_idx[cand_j]]
    cand_i, cand_j = cand_i[valid], cand_j[valid]
    sub = rng.choice(len(cand_i), size=min(5000, len(cand_i)), replace=False)
    ci, cj = cand_i[sub], cand_j[sub]
    # j's own real continuation rows: j, j+1, j+2, j+3 (via order/pos_in_order, same episode)
    cj_pos = pos_in_order[cj]
    if SKIP > 1:
        cont_rows = np.stack([order[cj_pos + k] for k in range(SKIP - 1)], axis=1)  # (n, SKIP-1)
    else:
        # native 1-step predictor: no continuation needed, the splice test degenerates to
        # "does pred(z_i, a) land near z_j itself" -- exactly the identification-edge check.
        cont_rows = np.empty((len(cj), 0), dtype=np.int64)
    target4 = order[cj_pos + (SKIP - 1)]  # z_{j+4}: real state after j's own next SKIP-1 actions

    uniq_eps = np.unique(ep_idx)
    shuffled = np.random.default_rng(0).permutation(uniq_eps)
    n_train = int(len(shuffled) * 0.8)
    train_eps = shuffled[:n_train]
    trans_ep = ep_idx[trans_i]
    tr_i, tr_j = trans_i[np.isin(trans_ep, train_eps)], trans_j[np.isin(trans_ep, train_eps)]
    idm_models = train_idm_ensemble(z, action, tr_i, tr_j, action_dim=mech.action_dim,
                                     n_members=6, hidden=256, n_steps=3000, batch_size=256,
                                     lr=1e-3, seed=0)
    a_idm = ensemble_predict(idm_models, z, ci, cj).mean(axis=0).astype(np.float32)  # (n, action_dim)

    def splice_error(a1):
        block = np.concatenate([a1[:, None, :], action[cont_rows]], axis=1).reshape(len(ci), -1)
        zi_h = torch.from_numpy(z[ci]).float().to(DEV).unsqueeze(1)
        act_h = torch.from_numpy(block).float().to(DEV).unsqueeze(1)
        with torch.no_grad():
            act_emb_h = model.action_encoder(act_h)
            pred_h = model.predict(zi_h, act_emb_h)[:, -1].cpu().numpy()
        return np.linalg.norm(pred_h - z[target4], axis=1)

    hybrid_err = splice_error(a_idm)
    # control: a RANDOM real action instead of the IDM's inferred one, same real continuation
    # -- isolates whether the IDM's SPECIFIC inference matters, or whether starting from a
    # nearby candidate i plus j's real continuation would land near target4 regardless of a1
    rand_a1_splice = action[trans_i[rng.integers(0, len(trans_i), size=len(ci))]]
    random_err = splice_error(rand_a1_splice)

    hybrid_before = np.linalg.norm(z[ci] - z[target4], axis=1)
    hybrid_ratio = hybrid_err / np.maximum(hybrid_before, 1e-8)
    random_ratio = random_err / np.maximum(hybrid_before, 1e-8)

    ref_errs = errs_by_h[1]  # real, in-distribution single-block error, same SKIP
    # SKIP=1 (pred1): the "hybrid block" is just the single inferred/random action -- there is
    # no continuation to name. SKIP>1 (block5): unchanged wording (continuation is real).
    step_noun = "step" if SKIP == 1 else "block"
    cont_suffix = "" if SKIP == 1 else " + real continuation"
    log(f"[splice] n={len(hybrid_err)} hybrid-{step_noun} error: mean={hybrid_err.mean():.3f} "
        f"median={np.median(hybrid_err):.3f}  |  reference (real {step_noun}) error: "
        f"mean={ref_errs.mean():.3f} median={np.median(ref_errs):.3f}")
    log(f"[splice] RANDOM a1 control error: mean={random_err.mean():.3f} "
        f"median={np.median(random_err):.3f}")
    log(f"[splice] hybrid ratio: median={np.median(hybrid_ratio):.3f} frac<1={np.mean(hybrid_ratio < 1):.3f}  |  "
        f"random-a1 ratio: median={np.median(random_ratio):.3f} frac<1={np.mean(random_ratio < 1):.3f}")

    # SKIP=1 (pred1): plot IDM vs. random alone -- the real-reference comparison now has its
    # own dedicated (and more informative) train-vs-held-out figure above, so repeating a
    # single aggregate reference line here would just be redundant/less precise information.
    fig, ax = plt.subplots(figsize=(6, 4))
    if SKIP == 1:
        bins = np.linspace(0, max(hybrid_err.max(), random_err.max()), 100)
    else:
        bins = np.linspace(0, max(ref_errs.max(), hybrid_err.max(), random_err.max()), 100)
        ax.hist(ref_errs, bins=bins, alpha=0.5, density=True,
                label=f"real {step_noun} (ground truth): mean={ref_errs.mean():.2f}")
    ax.hist(hybrid_err, bins=bins, alpha=0.5, density=True,
            label=f"IDM action{cont_suffix}: mean={hybrid_err.mean():.2f}")
    ax.hist(random_err, bins=bins, alpha=0.5, density=True,
            label=f"random $a_1${cont_suffix}: mean={random_err.mean():.2f}")
    ax.set_xlabel("prediction error, Euclidean distance ||pred - target||")
    ax.set_ylabel("density")
    title_kind = "IDM vs. random, single action" if SKIP == 1 else "IDM vs. random vs. real"
    ax.set_title(f"Cross-episode splice error, {title_kind} ({cfg.env.name})")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG_DIR / f"{prefix}_splice_vs_ground_truth.png", dpi=150)
    plt.close(fig)

    # --- bonus G: distance-filtered splice test (pred1 only) -----------------------------
    # The wide top-20 pool above sits ~4.3 units apart on average -- far beyond pred1's real
    # single-step reach (~1.15-unit median move), so no action can close that gap regardless
    # of quality. Rebuild the candidate pool restricted to a distance pred1 can actually
    # cover: radius = 95th percentile of the REAL adjacent-step distance (calibrated from
    # data, not an arbitrary constant), then rerun the identical IDM-vs-random test on it.
    if predictor_kind == "1step":
        radius = float(np.quantile(adjacent_dist, 0.95))
        log(f"=== distance-filtered splice test (radius={radius:.3f}, pred1's reach scale) ===")
        # Filter the SAME bounded top-20 pool used above by distance, rather than an
        # unbounded radius search: Push-T's landmark set has dense clusters (e.g. similar
        # start states shared across many of the 300 episodes) where an unbounded
        # range_search blows up combinatorially (confirmed: hung/OOM'd in practice). Capping
        # at k=20 neighbors per point first keeps this bounded and cheap.
        d_all = np.linalg.norm(z[cand_i] - z[cand_j], axis=1)
        within_radius = d_all <= radius
        dcand_i, dcand_j = cand_i[within_radius], cand_j[within_radius]
        log(f"[distance-filtered] {len(dcand_i)} candidate pairs within radius={radius:.3f} "
            f"(out of the wide top-20 pool's {len(cand_i)} candidates, "
            f"mean dist {d_all.mean():.3f})")
        dsub = rng.choice(len(dcand_i), size=min(5000, len(dcand_i)), replace=False)
        dci, dcj = dcand_i[dsub], dcand_j[dsub]
        d_before = np.linalg.norm(z[dci] - z[dcj], axis=1)
        log(f"[distance-filtered] candidate gap: mean={d_before.mean():.3f} "
            f"median={np.median(d_before):.3f}")

        a_idm_d = ensemble_predict(idm_models, z, dci, dcj).mean(axis=0).astype(np.float32)
        rand_a1_d = action[trans_i[rng.integers(0, len(trans_i), size=len(dci))]]

        def single_action_error(zi_idx, zj_idx, a1):
            zi_t = torch.from_numpy(z[zi_idx]).float().to(DEV).unsqueeze(1)
            act_t = torch.from_numpy(a1).float().to(DEV).unsqueeze(1)
            with torch.no_grad():
                act_emb = model.action_encoder(act_t)
                pred = model.predict(zi_t, act_emb)[:, -1].cpu().numpy()
            return np.linalg.norm(pred - z[zj_idx], axis=1)

        d_hybrid_err = single_action_error(dci, dcj, a_idm_d)
        d_random_err = single_action_error(dci, dcj, rand_a1_d)
        d_hybrid_ratio = d_hybrid_err / np.maximum(d_before, 1e-8)
        d_random_ratio = d_random_err / np.maximum(d_before, 1e-8)
        log(f"[distance-filtered] IDM error: mean={d_hybrid_err.mean():.3f} "
            f"median={np.median(d_hybrid_err):.3f} ratio median={np.median(d_hybrid_ratio):.3f} "
            f"frac<1={np.mean(d_hybrid_ratio < 1):.3f}")
        log(f"[distance-filtered] random error: mean={d_random_err.mean():.3f} "
            f"median={np.median(d_random_err):.3f} ratio median={np.median(d_random_ratio):.3f} "
            f"frac<1={np.mean(d_random_ratio < 1):.3f}")

        # overlay the real single-step reference (same distance scale now, so this is a fair
        # comparison for the first time) -- makes visible whether distance-matched candidates
        # behave like real transitions once the scale confound is removed, or not.
        fig, ax = plt.subplots(figsize=(6, 4))
        bins = np.linspace(0, max(ref_errs.max(), d_hybrid_err.max(), d_random_err.max()), 100)
        ax.hist(ref_errs, bins=bins, alpha=0.5, density=True,
                label=f"real step (ground truth): mean={ref_errs.mean():.2f}, "
                      f"ratio frac$<$1={np.mean(ratio_by_h[1] < 1):.2f}")
        ax.hist(d_hybrid_err, bins=bins, alpha=0.5, density=True,
                label=f"IDM action: mean={d_hybrid_err.mean():.2f}, "
                      f"ratio frac$<$1={np.mean(d_hybrid_ratio < 1):.2f}")
        ax.hist(d_random_err, bins=bins, alpha=0.5, density=True,
                label=f"random $a_1$: mean={d_random_err.mean():.2f}, "
                      f"ratio frac$<$1={np.mean(d_random_ratio < 1):.2f}")
        ax.set_xlabel("prediction error, Euclidean distance ||pred - target||")
        ax.set_ylabel("density")
        ax.set_title(f"Distance-matched splice error: IDM vs. random vs. real ({cfg.env.name})")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"{prefix}_splice_distance_filtered.png", dpi=150)
        plt.close(fig)

    # --- bonus A: PCA / effective-dimensionality spectrum --------------------------------
    zc = z - z.mean(axis=0, keepdims=True)
    cov = (zc.T @ zc) / len(zc)
    eigvals = np.linalg.eigvalsh(cov)[::-1]
    eigvals = np.clip(eigvals, 0, None)
    participation_ratio = (eigvals.sum() ** 2) / (eigvals ** 2).sum()

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(np.arange(1, d + 1), np.cumsum(eigvals) / eigvals.sum())
    ax.axhline(0.95, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("number of principal components")
    ax.set_ylabel("cumulative variance explained")
    ax.set_title(f"Embedding covariance spectrum ({cfg.env.name}) -- "
                 f"participation ratio {participation_ratio:.1f}/{d}")
    fig.tight_layout(); fig.savefig(FIG_DIR / f"{prefix}_pca_spectrum.png", dpi=150)
    plt.close(fig)
    log(f"[fig] pca_spectrum.png -- participation ratio {participation_ratio:.2f} / {d}")

    # --- bonus B: IDM ensemble variance, real held-out pairs vs. wide candidate pool ------
    log("training IDM ensemble for variance comparison...")
    uniq_eps = np.unique(ep_idx)
    shuffled = np.random.default_rng(0).permutation(uniq_eps)
    n_train = int(len(shuffled) * 0.8)
    train_eps, test_eps = shuffled[:n_train], shuffled[n_train:]
    trans_ep = ep_idx[trans_i]
    tr_i, tr_j = trans_i[np.isin(trans_ep, train_eps)], trans_j[np.isin(trans_ep, train_eps)]
    te_i, te_j = trans_i[np.isin(trans_ep, test_eps)], trans_j[np.isin(trans_ep, test_eps)]
    models = train_idm_ensemble(z, action, tr_i, tr_j, action_dim=mech.action_dim,
                                 n_members=6, hidden=256, n_steps=3000, batch_size=256,
                                 lr=1e-3, seed=0)
    real_var = ensemble_predict(models, z, te_i, te_j).var(axis=0).mean(axis=-1)

    cand_i, cand_j = raw_topk_cross_episode_pairs(z, ep_idx, k=20)
    sub = np.random.default_rng(2).choice(len(cand_i), size=min(20000, len(cand_i)), replace=False)
    cand_var = ensemble_predict(models, z, cand_i[sub], cand_j[sub]).var(axis=0).mean(axis=-1)

    # third population: genuinely random, unrelated state pairs -- essentially zero chance of
    # a real one-step connecting action existing, unlike the top-20 pool (which stage 2's
    # ground-truth env check already showed is mostly made of real, stitchable connections).
    # If variance carries any information at all, THIS is where it should be visibly higher.
    n_rand_var = 20_000
    rand_i = np.random.default_rng(3).integers(0, len(z), n_rand_var)
    rand_j = np.random.default_rng(4).integers(0, len(z), n_rand_var)
    keep_r = rand_i != rand_j
    rand_var = ensemble_predict(models, z, rand_i[keep_r], rand_j[keep_r]).var(axis=0).mean(axis=-1)
    log(f"[idm-var] real={real_var.mean():.5f}  candidate={cand_var.mean():.5f}  "
        f"random={rand_var.mean():.5f}")

    fig, ax = plt.subplots(figsize=(6, 4))
    bins = np.linspace(0, np.quantile(np.concatenate([real_var, cand_var, rand_var]), 0.99), 100)
    ax.hist(real_var, bins=bins, alpha=0.55, density=True, label="real held-out adjacent pairs")
    ax.hist(cand_var, bins=bins, alpha=0.45, density=True, label="wide top-20 candidate pool")
    ax.hist(rand_var, bins=bins, alpha=0.45, density=True, label="genuinely random pairs")
    ax.set_xlabel("IDM ensemble Var(action)"); ax.set_ylabel("density")
    ax.set_title(f"Ensemble variance: real vs. candidate pool vs. random ({cfg.env.name})")
    ax.legend()
    fig.tight_layout(); fig.savefig(FIG_DIR / f"{prefix}_idm_variance_comparison.png", dpi=150)
    plt.close(fig)

    # --- bonus C: cosine similarity, adjacent vs. random ----------------------------------
    def cos(a, b):
        return np.sum(a * b, axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-8)

    cos_adjacent = cos(z[trans_i], z[trans_j])
    cos_random = cos(z[ri[keep]], z[rj[keep]])

    fig, ax = plt.subplots(figsize=(6, 4))
    bins = np.linspace(min(cos_adjacent.min(), cos_random.min()),
                        max(cos_adjacent.max(), cos_random.max()), 100)
    ax.hist(cos_adjacent, bins=bins, alpha=0.6, density=True, label="consecutive states")
    ax.hist(cos_random, bins=bins, alpha=0.5, density=True, label="random pairs")
    ax.set_xlabel("cosine similarity"); ax.set_ylabel("density")
    ax.set_title(f"Cosine similarity: consecutive vs. random pairs ({cfg.env.name})")
    ax.legend()
    fig.tight_layout(); fig.savefig(FIG_DIR / f"{prefix}_cosine_similarity.png", dpi=150)
    plt.close(fig)
    log(f"[fig] cosine_similarity.png -- adjacent mean={cos_adjacent.mean():.4f} "
        f"random mean={cos_random.mean():.4f}")

    log(f"\nall figures written to {FIG_DIR}")


if __name__ == "__main__":
    main()
