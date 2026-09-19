"""Encode a subset of Push-T episodes once with the frozen LeWM encoder, for the viability
critic (viability_train.py). Everything downstream is latent-only, so this is the only step
that reads pixels or runs the ViT.

Stores, for N randomly chosen episodes (reproducible via --seed): the CLS latent of every
frame, plus the raw action, the 7-dim simulator state (used ONLY to build hindsight labels
via the env's own success predicate -- see viability_train.py's doubts on this) and the
episode layout re-indexed into the cache. The action mean/std are computed from the FULL
dataset action column, not the subset (CLAUDE.md, "Action normalization").

Usage:
    python scripts/viability_cache.py --episodes 1000 --out outputs/pusht/critic_training/cache_1000.npz
Needs PUSHT_H5_PATH (or data/datasets/pusht_expert_train.h5) and the pretrained checkpoint.
"""

import argparse
import sys
import time
from pathlib import Path

import hdf5plugin  # noqa: F401 -- registers the Blosc filter the h5's `pixels` column uses
import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.envs import ENV_MECHANICS
from common.lewm_loader import load_lewm
from common.log_util import log
from planning_cost_gate import DEV, make_encode_frame


def run(args):
    mech = ENV_MECHANICS["pusht"]
    root = Path(args.root).resolve()
    h5_path, ckpt_dir = mech.h5_path(root), mech.ckpt_dir(root)
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    log(f"[setup] h5={h5_path} ckpt={ckpt_dir} device={DEV} episodes={args.episodes} seed={args.seed}")

    model = load_lewm(Path(ckpt_dir), device=DEV)
    encode = make_encode_frame(model, batch=args.batch)

    with h5py.File(h5_path, "r", swmr=True, rdcc_nbytes=512 * 1024 * 1024) as f:
        ep_offset = f["ep_offset"][:].astype(np.int64)
        ep_len = f["ep_len"][:].astype(np.int64)
        act_all = f["action"][:].astype(np.float64)
        act_all = act_all[~np.isnan(act_all).any(axis=1)]
        act_mean, act_std = act_all.mean(0).astype(np.float32), act_all.std(0).astype(np.float32)
        del act_all

        rng = np.random.default_rng(args.seed)
        n_ep = len(ep_len)
        chosen = np.sort(rng.choice(n_ep, size=min(args.episodes, n_ep), replace=False))
        total = int(ep_len[chosen].sum())
        log(f"[setup] {len(chosen)} episodes, {total} frames, act_mean={act_mean} act_std={act_std}")

        z = np.empty((total, 192), np.float32)
        state = np.empty((total, mech.state_dim), np.float32)
        action = np.empty((total, mech.action_dim), np.float32)
        new_offset = np.empty(len(chosen), np.int64)
        new_len = ep_len[chosen].copy()
        pos, t0 = 0, time.time()
        for i, e in enumerate(chosen):
            lo, hi = int(ep_offset[e]), int(ep_offset[e] + ep_len[e])
            z[pos:pos + hi - lo] = encode(f["pixels"][lo:hi])
            state[pos:pos + hi - lo] = f["state"][lo:hi]
            action[pos:pos + hi - lo] = f["action"][lo:hi]
            new_offset[i] = pos
            pos += hi - lo
            if i % 50 == 0 or i == len(chosen) - 1:
                log(f"[encode] episode {i + 1}/{len(chosen)} frames={pos}/{total} "
                    f"elapsed={time.time() - t0:.0f}s ({pos / max(time.time() - t0, 1e-6):.0f} fps)")

    np.savez_compressed(out, z=z, state=state, action=action, ep_offset=new_offset, ep_len=new_len,
                        episode_id=chosen, act_mean=act_mean, act_std=act_std, seed=args.seed,
                        n_source_episodes=n_ep, h5_path=str(h5_path), ckpt_dir=str(ckpt_dir))
    log(f"[done] wrote {out} ({out.stat().st_size / 1e6:.0f} MB)")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=str(Path(__file__).resolve().parent.parent))
    p.add_argument("--out", default=None)
    p.add_argument("--episodes", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch", type=int, default=64)
    args = p.parse_args()
    args.out = args.out or str(Path(args.root) / "outputs" / "pusht" / "critic_training" / f"cache_{args.episodes}_s{args.seed}.npz")
    run(args)


if __name__ == "__main__":
    main()
