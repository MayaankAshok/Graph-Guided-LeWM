"""Diagnostic: for the fixed task200u pool actually used by gas_mpc_eval.py, what is the
latent L2 distance between each task's start and goal frame, per protocol?

Companion to Appendix F's predictor-rollout-error-by-horizon diagnostic
(predictor_error_by_horizon.py): that table shows how far the predictor's own imagined
endpoint drifts from the true endpoint as a function of horizon; this one shows how far the
true start is from the true goal for the tasks CEM is actually asked to solve, so the two
scales (predictor drift vs. task difficulty) can be compared directly, both against the same
random-pair reference (unrelated frames, ~19.0 on Push-T).

Reads pairs_{tag}_{POOL}.json from gas_mpc_eval.PAIRS_DIR (one file per protocol, written by
gas_mpc_make_tasks.py) and encodes each task's start_row/goal_row with the frozen encoder
(same encode_frame as encode_rows_standalone.py / planning_cost_gate.py), never a cached z --
task200u pool rows can be held out of cache_train.npz and cache_full.npz's row alignment is
not guaranteed to match cheaply, so encoding from raw pixels is the only assumption-free path.

    python scripts/task_start_goal_distance.py
    GAS_MPC_ENV=reacher python scripts/task_start_goal_distance.py
"""
import argparse
import json
import sys
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401 -- registers the Blosc filter the h5's `pixels` column uses
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.gas import DEV
from common.lewm_loader import load_lewm
from common.log_util import log
from gas_mpc_eval import ENV, MECH, PAIRS_DIR, POOL, PROTOCOLS, ROOT
from planning_cost_gate import make_encode_frame


def pair_tag(protocol):
    proto = PROTOCOLS[protocol]
    return "cross_episode" if proto["pairing"] == "cross_episode" else f"same_episode_off{proto['offset']}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--protocols", default="same25,same50,same100,cross")
    ap.add_argument("--random-ref-json", default=None,
                    help="predictor_error_by_horizon.py output to reuse its random_pair_distance "
                         "reference instead of recomputing one; default: outputs/<env>/diagnostics/"
                         "predictor_error_by_horizon_s0.json if present")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--batch", type=int, default=256)
    args = ap.parse_args()

    protocols = args.protocols.split(",")
    for p in protocols:
        if p not in PROTOCOLS:
            raise ValueError(f"unknown protocol {p!r}, choose from {list(PROTOCOLS)}")

    out_dir = Path(args.out_dir) if args.out_dir else ROOT / "outputs" / ENV / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)

    h5_path, ckpt_dir = MECH.h5_path(ROOT), MECH.ckpt_dir(ROOT)
    model = load_lewm(Path(ckpt_dir), device=DEV)
    encode = make_encode_frame(model, batch=args.batch)

    ref_path = Path(args.random_ref_json) if args.random_ref_json else out_dir / "predictor_error_by_horizon_s0.json"
    random_ref = None
    if ref_path.exists():
        ref_stats = json.loads(ref_path.read_text())["random_pair_distance"]
        random_ref = np.asarray(ref_stats["values"], dtype=np.float32)
        log(f"[task_start_goal_distance] reusing random-pair reference from {ref_path} "
            f"(mean={ref_stats['mean']:.2f}, n={len(random_ref)})")
    else:
        log(f"[task_start_goal_distance] {ref_path} not found -- plotting without the random-pair reference")

    results = {"env": ENV, "pool": POOL}
    per_proto = {}
    with h5py.File(h5_path, "r", swmr=True, rdcc_nbytes=512 * 1024 * 1024) as f:
        for protocol in protocols:
            tag = pair_tag(protocol)
            path = PAIRS_DIR / f"pairs_{tag}_{POOL}.json"
            pairs = json.loads(path.read_text())
            start_row = np.asarray(pairs["start_row"], dtype=np.int64)
            goal_row = np.asarray(pairs["goal_row"], dtype=np.int64)
            n = len(start_row)

            all_rows = np.concatenate([start_row, goal_row])
            order = np.argsort(all_rows)
            pixels = f["pixels"][all_rows[order]]
            z = encode(pixels)
            z_all = np.empty_like(z)
            z_all[order] = z
            z_start, z_goal = z_all[:n], z_all[n:]

            dist = np.linalg.norm(z_start - z_goal, axis=-1)
            per_proto[protocol] = dist
            results[protocol] = {"pool_path": str(path), "n": int(n),
                                 "mean": float(dist.mean()), "median": float(np.median(dist)),
                                 "std": float(dist.std()), "values": dist.tolist()}
            log(f"[{protocol:>7}] n={n} mean={dist.mean():.3f} median={np.median(dist):.3f} std={dist.std():.3f}")

    json_path = out_dir / "task_start_goal_distance.json"
    json_path.write_text(json.dumps(results, indent=1))

    max_x = max(d.max() for d in per_proto.values())
    if random_ref is not None:
        max_x = max(max_x, random_ref.max())
    bins = np.linspace(0, max_x, 40)

    fig, axes = plt.subplots(1, len(protocols), figsize=(4.2 * len(protocols), 3.6), sharey=False)
    axes = np.atleast_1d(axes)
    for ax, protocol in zip(axes, protocols):
        dist = per_proto[protocol]
        ax.hist(dist, bins=bins, density=True, color="tab:orange", alpha=0.8, label="start->goal")
        if random_ref is not None:
            ax.hist(random_ref, bins=bins, density=True, color="gray", alpha=0.4, label="random pair (ref.)")
        ax.axvline(np.mean(dist), color="tab:orange", linestyle="--", linewidth=1)
        ax.set_title(f"{protocol}\nmean={dist.mean():.1f}, std={dist.std():.1f}")
        ax.set_xlabel("latent L2 distance")
        if ax is axes[0]:
            ax.set_ylabel("density")
            ax.legend(fontsize=8)
    fig.suptitle(f"{ENV}: start->goal latent distance per task ({POOL} pool, n={per_proto[protocols[0]].shape[0]}/protocol)")
    fig.tight_layout()
    png_path = out_dir / "task_start_goal_distance.png"
    fig.savefig(png_path, dpi=150)
    plt.close(fig)

    log(f"[task_start_goal_distance] wrote {json_path} and {png_path}")


if __name__ == "__main__":
    main()
