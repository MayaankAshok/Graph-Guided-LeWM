"""Experiment 2 Run C: re-query the recovery-time oracle on a stratified subset with a
stronger CEM (larger population, more iterations, more trials).

The subset is chosen by scripts/viability_metric_vs_dynamics.py (`--select-recheck`), which
writes recheck_subset_offset{N}.json listing (task, candidate, group) triples for three groups:

    low_error_inversion   the closer-but-slower member of an inverted pair whose model error
                          is in the lowest global quartile
    high_error_inversion  same, highest quartile
    non_inversion         candidates that take part in no inverted pair under either cost

Only the SLOW member of an inverted pair is rechecked: a stronger oracle can only lower a
recovery time, and lowering the fast member's time cannot remove the inversion. If the
stronger oracle finds a materially faster recovery for the slow member, the inversion was
an artifact of weak search, not of the latent metric. The recheck reuses the same endpoint
state, goal state, cap, and success predicate as Run A; only the search budget changes.

    python scripts/viability_oracle_recheck.py --out outputs/pusht/experiments/viability_exp2 \
        --goal-offsets 25 50 --oracle-trials 4 --oracle-npop 128 --oracle-niter 8 \
        > outputs/pusht/experiments/viability_exp2/run_c.log 2>&1

Writes recheck_offset{N}.npz with the subset indices, the Run A recovery times, and the
stronger oracle's; the analysis script picks these up automatically.
"""

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.log_util import log
from viability_inversion_audit import _init_oracle_worker, oracle_endpoint_job, oracle_seed


def run(args):
    out_dir = Path(args.out).resolve()
    if args.oracle_workers > 1:
        pool = ProcessPoolExecutor(max_workers=args.oracle_workers, initializer=_init_oracle_worker,
                                   mp_context=get_context("spawn"))
    else:
        _init_oracle_worker()
        pool = None
    for offset in args.goal_offsets:
        raw = np.load(out_dir / f"raw_offset{offset}.npz")
        subset = json.loads((out_dir / f"recheck_subset_offset{offset}.json").read_text())
        cap = int(raw["oracle_cap"]) if "oracle_cap" in raw.files else args.oracle_max_steps
        base_seed = int(subset["seed"])
        items = subset["items"]
        log(f"[offset={offset}] rechecking {len(items)} endpoints with "
            f"{args.oracle_trials}x{args.oracle_npop}x{args.oracle_niter} (cap={cap}) -- groups: "
            + ", ".join(f"{g}={sum(1 for it in items if it['group'] == g)}" for g in subset["groups"]))
        jobs = [
            (raw["final_states"][it["task"], it["cand"]], raw["goal_state"][it["task"]],
             # different seed stream from Run A so the recheck is an independent search
             oracle_seed(base_seed, offset, it["task"], it["cand"]) + 977_101,
             cap, args.oracle_trials, args.oracle_npop, args.oracle_niter, args.oracle_elite_frac)
            for it in items
        ]
        t0 = time.time()
        new_steps, new_dist = np.empty(len(items), np.int16), np.empty(len(items), np.float32)
        results = pool.map(oracle_endpoint_job, jobs) if pool else map(oracle_endpoint_job, jobs)
        for j, (steps, best) in enumerate(results):
            new_steps[j], new_dist[j] = steps, best
            if (j + 1) % 10 == 0 or j + 1 == len(items):
                log(f"[offset={offset}] {j + 1}/{len(items)} elapsed={time.time() - t0:.1f}s")
        task = np.array([it["task"] for it in items], np.int64)
        cand = np.array([it["cand"] for it in items], np.int64)
        old_steps = raw["oracle_min_steps"][task, cand]
        improved = new_steps < old_steps
        log(f"[offset={offset}] improved={improved.mean():.3f} "
            f"uncensored={((old_steps > cap) & (new_steps <= cap)).mean():.3f} "
            f"mean_drop={(old_steps - new_steps)[improved].mean() if improved.any() else 0.0:.1f}")
        np.savez_compressed(
            out_dir / f"recheck_offset{offset}.npz", task=task, cand=cand,
            group=np.array([it["group"] for it in items]), old_steps=old_steps,
            new_steps=new_steps, new_best_dist=new_dist, oracle_cap=cap,
            oracle_trials=args.oracle_trials, oracle_npop=args.oracle_npop,
            oracle_niter=args.oracle_niter,
        )
    if pool:
        pool.shutdown()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True, help="Run A output dir (raw_offset*.npz + recheck_subset_*.json)")
    p.add_argument("--goal-offsets", type=int, nargs="+", default=[25, 50])
    p.add_argument("--oracle-max-steps", type=int, default=100, help="only used if the raw file lacks oracle_cap")
    p.add_argument("--oracle-trials", type=int, default=4)
    p.add_argument("--oracle-npop", type=int, default=128)
    p.add_argument("--oracle-niter", type=int, default=8)
    p.add_argument("--oracle-elite-frac", type=float, default=0.15)
    p.add_argument("--oracle-workers", type=int, default=8)
    run(p.parse_args())


if __name__ == "__main__":
    main()
