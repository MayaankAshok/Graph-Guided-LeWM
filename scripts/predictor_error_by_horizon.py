"""Diagnostic: how does the frozen LeWM predictor's latent rollout error grow with horizon?

For each horizon H in --horizons (multiples of 5, the predictor's block size), samples
(episode, start_step) pairs from the training-only cache (`outputs/<env>/cache_train.npz`,
built by `gas_mpc_prepare.py encode` -- the same frames used for TDR/graph construction, so
this never touches the held-out task200u evaluation episodes), and:

  1. rolls the predictor forward AUTOREGRESSIVELY (its own imagined latents feed the next
     block, exactly as CEM's own cost function sees it -- `common.viability.imagine`) under
     the REAL logged actions for that stretch of the episode, z-scored per CLAUDE.md's
     "Action normalization" convention (raw actions are NEVER fed to the predictor directly);
  2. compares the predicted latent at t+H against the REAL encoded latent at t+H (already
     cached, since cache_train.npz encodes every frame with the same frozen encoder);
  3. also draws, once, a "random pair" reference distribution: the latent distance between
     two uniformly random training frames (unrelated in time) -- the scale a completely
     uninformative predictor would produce, referenced in this project's failure-mode
     narrative as the point compounding predictor error approaches.

Writes one histogram (one panel per horizon, with the random-pair reference overlaid) and a
JSON of the raw per-sample errors + summary stats.

Run (Push-T; needs outputs/pusht/cache_train.npz -- run `gas_mpc_prepare.py encode` first):
    python scripts/predictor_error_by_horizon.py --horizons 5,25,50,100 --n-samples 500
    GAS_MPC_ENV=reacher python scripts/predictor_error_by_horizon.py --horizons 5,25,50
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.envs import ENV_MECHANICS
from common.heldout_tasks import fixed_episode_split
from common.lewm_loader import load_lewm
from common.log_util import log
from common.viability import HISTORY, SKIP, imagine

HELDOUT_FRAC = 0.02   # gas_mpc_prepare.py's --heldout-frac default

ROOT = Path(__file__).resolve().parent.parent
ENV = os.environ.get("GAS_MPC_ENV", "pusht")
MECH = ENV_MECHANICS[ENV]
OUT = Path(os.environ.get("GAS_MPC_OUT", str(ROOT / "outputs" / ENV)))
LOOKBACK = (HISTORY - 1) * SKIP   # env steps of real history needed before t (10, for HISTORY=3)


def load_cache():
    train_path = OUT / "cache_train.npz"
    if train_path.exists():
        with np.load(train_path) as d:
            return {k: d[k] for k in ("z", "action", "ep_offset", "ep_len", "act_mean", "act_std")}
    full_path = OUT / "cache_full.npz"
    if not full_path.exists():
        raise FileNotFoundError(
            f"neither {train_path} nor {full_path} exists -- run "
            f"`python scripts/gas_mpc_prepare.py encode` for env={ENV} first")
    log(f"[load_cache] {train_path} missing -- falling back to {full_path}, filtered to the "
        f"same fixed training-episode split gas_mpc_prepare.py encode would produce "
        f"(heldout_frac={HELDOUT_FRAC}, no held-out task200u frames touched)")
    with np.load(full_path) as d:
        full = {k: d[k] for k in ("z", "action", "ep_offset", "ep_len", "episode_id", "act_mean", "act_std")}
    selected, _held = fixed_episode_split(full["episode_id"], HELDOUT_FRAC)
    selected = np.asarray(selected)
    ep_pos = {eid: i for i, eid in enumerate(full["episode_id"])}
    rows = [np.arange(full["ep_offset"][ep_pos[e]], full["ep_offset"][ep_pos[e]] + full["ep_len"][ep_pos[e]])
            for e in selected]
    idx = np.concatenate(rows)
    new_len = np.array([len(r) for r in rows], dtype=np.int64)
    new_offset = np.concatenate([[0], np.cumsum(new_len)[:-1]]).astype(np.int64)
    return {"z": full["z"][idx], "action": full["action"][idx], "ep_offset": new_offset,
            "ep_len": new_len, "act_mean": full["act_mean"], "act_std": full["act_std"]}


def sample_starts(cache, horizon, n_samples, rng):
    """(episode, t) pairs (global row index t) with LOOKBACK real steps before t and
    `horizon` real steps (actions + true endpoint) after t, all inside one episode."""
    ep_offset, ep_len = cache["ep_offset"], cache["ep_len"]
    valid_len = ep_len - LOOKBACK - horizon    # usable start positions per episode, relative
    eps = np.flatnonzero(valid_len > 0)
    if eps.size == 0:
        return np.empty(0, dtype=np.int64)
    weights = valid_len[eps].astype(np.float64)
    ep_pick = rng.choice(eps, size=n_samples, replace=True, p=weights / weights.sum())
    offs = rng.integers(0, valid_len[ep_pick])              # 0 .. valid_len-1 within episode
    return ep_offset[ep_pick] + LOOKBACK + offs             # global row index t


@torch.no_grad()
def predictor_rollout_error(model, cache, horizon, starts, act_mean, act_std, device, batch=256):
    """-> (len(starts),) L2 distance between the predictor's autoregressive prediction of
    z[t+horizon] (fed real logged actions) and the true cached z[t+horizon]."""
    z, action = cache["z"], cache["action"]
    n_blocks = horizon // SKIP
    A = action.shape[1]
    errs = np.empty(len(starts), dtype=np.float32)
    for lo in range(0, len(starts), batch):
        t = starts[lo:lo + batch]
        b = len(t)
        # history: real latents at t-10, t-5, t
        z_hist = torch.from_numpy(np.stack([z[t - LOOKBACK], z[t - LOOKBACK + SKIP], z[t]], axis=1)).to(device)
        # a_past: the two real action blocks that produced that history (at t-10 and t-5)
        def raw_block(t0):
            idx = t0[:, None] + np.arange(SKIP)[None, :]        # (b, SKIP)
            return action[idx].reshape(b, SKIP * A)              # (b, SKIP*A)
        a_past = np.stack([raw_block(t - LOOKBACK), raw_block(t - LOOKBACK + SKIP)], axis=1)
        a_future = np.stack([raw_block(t + k * SKIP) for k in range(n_blocks)], axis=1)
        mean_tiled, std_tiled = np.tile(act_mean, SKIP), np.tile(act_std, SKIP)
        a_past = (a_past - mean_tiled) / std_tiled
        a_future = (a_future - mean_tiled) / std_tiled
        a_past = torch.from_numpy(a_past.astype(np.float32)).to(device)
        a_future = torch.from_numpy(a_future.astype(np.float32)).to(device)
        z_img, _, _ = imagine(model, z_hist, a_past, a_future)
        z_pred = z_img[:, -1].cpu().numpy()
        z_true = z[t + horizon]
        errs[lo:lo + b] = np.linalg.norm(z_pred - z_true, axis=-1)
    return errs


def random_pair_distances(cache, n_samples, rng):
    """Reference distribution: L2 distance between two uniformly random, unrelated training
    frames -- the scale a maximally uninformative ('predictor forgot everything') endpoint
    guess would land at."""
    n = len(cache["z"])
    i = rng.integers(0, n, size=n_samples)
    j = rng.integers(0, n, size=n_samples)
    return np.linalg.norm(cache["z"][i] - cache["z"][j], axis=-1)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--horizons", default="5,25,50,100", help="comma-separated env-step horizons, multiples of 5")
    p.add_argument("--n-samples", type=int, default=500, help="rollouts sampled per horizon")
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out-dir", default=None)
    args = p.parse_args()

    horizons = [int(h) for h in args.horizons.split(",")]
    for h in horizons:
        assert h % SKIP == 0, f"horizon {h} must be a multiple of SKIP={SKIP}"
    rng = np.random.default_rng(args.seed)
    out_dir = Path(args.out_dir) if args.out_dir else OUT / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)

    log(f"[predictor_error_by_horizon] env={ENV} horizons={horizons} n_samples={args.n_samples} "
        f"seed={args.seed} device={args.device}")
    cache = load_cache()
    act_mean, act_std = cache["act_mean"].astype(np.float32), cache["act_std"].astype(np.float32)
    model = load_lewm(MECH.ckpt_dir(ROOT), device=args.device)

    random_ref = random_pair_distances(cache, args.n_samples, rng)
    results = {"env": ENV, "seed": args.seed, "n_samples": args.n_samples,
               "random_pair_distance": {"values": random_ref.tolist(),
                                         "mean": float(random_ref.mean()), "median": float(np.median(random_ref)),
                                         "std": float(random_ref.std())}}

    t0 = time.time()
    per_h = {}
    for h in horizons:
        starts = sample_starts(cache, h, args.n_samples, rng)
        errs = predictor_rollout_error(model, cache, h, starts, act_mean, act_std, args.device, args.batch)
        per_h[h] = errs
        results[str(h)] = {"values": errs.tolist(), "mean": float(errs.mean()), "median": float(np.median(errs)),
                           "std": float(errs.std()), "n": int(len(errs))}
        log(f"[h={h:>3}] n={len(errs)} mean={errs.mean():.3f} median={np.median(errs):.3f} "
            f"std={errs.std():.3f} elapsed={time.time() - t0:.0f}s")

    json_path = out_dir / f"predictor_error_by_horizon_s{args.seed}.json"
    json_path.write_text(json.dumps(results, indent=1))

    bins = np.linspace(0, max(random_ref.max(), max(e.max() for e in per_h.values())), 40)

    fig, axes = plt.subplots(1, len(horizons), figsize=(4.2 * len(horizons), 3.6), sharey=False)
    axes = np.atleast_1d(axes)
    for ax, h in zip(axes, horizons):
        errs = per_h[h]
        ax.hist(errs, bins=bins, color="tab:blue", alpha=0.75, label="predictor error")
        ax.hist(random_ref, bins=bins, color="gray", alpha=0.4, label="random pair (ref.)")
        ax.axvline(np.median(errs), color="tab:blue", linestyle="--", linewidth=1)
        ax.set_title(f"H = {h} steps\nmedian={np.median(errs):.1f}")
        ax.set_xlabel("latent L2 error")
        if ax is axes[0]:
            ax.set_ylabel("count")
            ax.legend(fontsize=8)
    fig.suptitle(f"{ENV}: predictor rollout error vs. horizon (n={args.n_samples}/horizon, seed {args.seed})")
    fig.tight_layout()
    png_path = out_dir / f"predictor_error_by_horizon_s{args.seed}.png"
    fig.savefig(png_path, dpi=150)
    plt.close(fig)

    per_h_paths = []
    for h in horizons:
        errs = per_h[h]
        f2, ax = plt.subplots(figsize=(5, 4))
        ax.hist(errs, bins=bins, color="tab:blue", alpha=0.75, label="predictor error")
        ax.hist(random_ref, bins=bins, color="gray", alpha=0.4, label="random pair (ref.)")
        ax.axvline(np.median(errs), color="tab:blue", linestyle="--", linewidth=1,
                    label=f"median={np.median(errs):.2f}")
        ax.set_title(f"{ENV}: predictor rollout error, H = {h} steps (n={len(errs)}, seed {args.seed})")
        ax.set_xlabel("latent L2 error")
        ax.set_ylabel("count")
        ax.legend(fontsize=8)
        f2.tight_layout()
        h_path = out_dir / f"predictor_error_h{h}_s{args.seed}.png"
        f2.savefig(h_path, dpi=150)
        plt.close(f2)
        per_h_paths.append(h_path)

    log(f"[predictor_error_by_horizon] wrote {json_path}, {png_path}, and "
        f"{len(per_h_paths)} per-horizon PNGs ({time.time() - t0:.0f}s total)")


if __name__ == "__main__":
    main()
