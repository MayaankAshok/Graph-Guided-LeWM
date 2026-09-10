"""Unified sweep aggregation -- config-driven consolidation of aggregate_actor_sweep.py
(which only ever pointed at outputs/b3_tworoom, needing an ad hoc copy for Push-T each time
this session). Safe to run at any point -- only reports combos with a result file so far,
so it doubles as a progress check mid-sweep.

Run as:
    python scripts/aggregate_sweep.py --env pusht --mode actor       # live-rollout success rate
    python scripts/aggregate_sweep.py --env pusht --mode datatiers   # peak Spearman
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def peak_stats(history):
    i = max(range(len(history)), key=lambda k: history[k]["spearman_test"])
    return history[i]


def aggregate_actor(env, tiers, variants, selects, protocol):
    """Result files written before the protocol field existed carry no `eval_protocol`
    key -- they were all measured under the original cross_episode convention."""
    eval_dir = ROOT / "outputs" / f"b3_{env}" / "actor" / "rollout_eval"
    rows = defaultdict(dict)
    n_seen = 0
    for f in eval_dir.glob("*.json"):
        d = json.loads(f.read_text())
        n_seen += 1
        if d.get("eval_protocol", "cross_episode") != protocol:
            continue
        tier, variant, select = d["tier"], d["variant"], d.get("select") or "-"
        key = (tier, variant, select)
        rows[key].setdefault("success_rates", []).append(d["success_rate"])
        rows[key].setdefault("seeds", []).append(d["seed"])
        rows[key].setdefault("peak_steps", []).append(d["peak_step"])
        rows[key].setdefault("n_episodes", []).append(d["n_episodes"])

    print(f"protocol={protocol} ({n_seen} result files scanned in {eval_dir})")
    print(f"{'tier':<13}{'variant':<10}{'select':<9}{'n_seeds':>8}{'success_rate':>16}"
          f"{'eps/seed':>10}{'peak_step (range)':>22}")
    for tier in tiers:
        for variant in [*variants, "random"]:
            for select in (selects if variant != "random" else ["-"]):
                key = (tier, variant, select)
                if key not in rows:
                    continue
                r = rows[key]
                sr = np.array(r["success_rates"])
                steps = [s for s in r["peak_steps"] if s is not None]
                step_range = f"{min(steps)}-{max(steps)}" if steps else "-"
                print(f"{tier:<13}{variant:<10}{select:<9}{len(sr):>8}"
                      f"{sr.mean():>10.3f} +- {sr.std():<5.3f}{int(np.mean(r['n_episodes'])):>10}{step_range:>22}")
    if not rows:
        print(f"no results for protocol={protocol} yet in {eval_dir}")


def aggregate_datatiers(env, tiers, variants):
    ckpt_dir = ROOT / "outputs" / f"b3_{env}" / "checkpoints"
    print(f"{'tier':<13}{'variant':<10}{'n_seeds':>8}{'peak_spearman':>18}")
    for tier in tiers:
        for variant in variants:
            peaks = []
            for f in ckpt_dir.glob(f"{tier}__{variant}__s*.pt"):
                import torch
                ckpt = torch.load(f, map_location="cpu", weights_only=False)
                if not ckpt.get("done"):
                    continue
                peaks.append(peak_stats(ckpt["history"])["spearman_test"])
            if not peaks:
                continue
            peaks = np.array(peaks)
            print(f"{tier:<13}{variant:<10}{len(peaks):>8}{peaks.mean():>12.4f} +- {peaks.std():<6.4f}")
    if not any(ckpt_dir.glob("*.pt")):
        print(f"no results yet in {ckpt_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", required=True, choices=["tworoom", "pusht", "reacher"])
    ap.add_argument("--mode", required=True, choices=["datatiers", "actor"])
    ap.add_argument("--tiers", nargs="+",
                     default=["expert_10", "expert_25", "expert_50", "expert_100", "mixed", "mixed_large"])
    ap.add_argument("--variants", nargs="+", default=["baseline", "auxphi"])
    ap.add_argument("--selects", nargs="+", default=["rho", "success"])
    ap.add_argument("--protocol", default="same_episode", choices=["same_episode", "cross_episode"],
                    help="actor mode: which live-rollout protocol's results to report")
    args = ap.parse_args()

    if args.mode == "actor":
        aggregate_actor(args.env, args.tiers, args.variants, args.selects, args.protocol)
    else:
        aggregate_datatiers(args.env, args.tiers, args.variants)


if __name__ == "__main__":
    main()
