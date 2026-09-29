"""Diagnostic: for the fixed task200u pool actually used by gas_mpc_eval.py, what is the
latent L2 distance between each task's start and goal frame, per protocol?

Companion to Appendix F's predictor-rollout-error-by-horizon diagnostic
(predictor_error_by_horizon.py): that table shows how far the predictor's own imagined
endpoint drifts from the true endpoint as a function of horizon; this one shows how far the
true start is from the true goal for the tasks CEM is actually asked to solve, so the two
scales (predictor drift vs. task difficulty) can be compared directly, both against the same
random-pair reference (unrelated frames, ~19.0 on Push-T).

Reads pairs_{tag}_{POOL}.json and the corresponding held-out `cache_test.npz` embeddings.

    python scripts/paper/diagnostics/task_start_goal_distance.py
    GAS_MPC_ENV=reacher python scripts/paper/diagnostics/task_start_goal_distance.py
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.heldout_tasks import cache_rows, validate_test_cache
from common.log_util import log
from gas_mpc.gas_mpc_eval import ENV, OUT, PAIRS_DIR, POOL, PROTOCOLS, ROOT


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
    args = ap.parse_args()

    protocols = args.protocols.split(",")
    for p in protocols:
        if p not in PROTOCOLS:
            raise ValueError(f"unknown protocol {p!r}, choose from {list(PROTOCOLS)}")

    out_dir = Path(args.out_dir) if args.out_dir else ROOT / "outputs" / ENV / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)

    with np.load(OUT / "cache_test.npz") as cache:
        validate_test_cache({k: cache[k] for k in (
            "heldout_only", "n_source_episodes", "heldout_frac", "episode_id", "training_episode_ids")})
        z = cache["z"]
        offsets, lengths, source_offsets = cache["ep_offset"], cache["ep_len"], cache["source_ep_offset"]
    test_index = dict(ep_offset=offsets, ep_len=lengths, source_ep_offset=source_offsets)

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
    for protocol in protocols:
            tag = pair_tag(protocol)
            path = PAIRS_DIR / f"pairs_{tag}_{POOL}.json"
            pairs = json.loads(path.read_text())
            start_row = np.asarray(pairs["start_row"], dtype=np.int64)
            goal_row = np.asarray(pairs["goal_row"], dtype=np.int64)
            n = len(start_row)

            z_start, z_goal = z[cache_rows(test_index, start_row)], z[cache_rows(test_index, goal_row)]

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
