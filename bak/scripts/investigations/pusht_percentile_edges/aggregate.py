"""Collect condition 1 (baseline, already run) + conditions 2-6 (run.py/run_pct_sweep.sh)
into one table: mean +- std live-rollout success_rate and peak_rho over 3 seeds, Push-T
expert_1000, goal=25/50 protocol.

Run (Ada, after run_pct_sweep.sh finishes):
    python scripts/investigations/pusht_percentile_edges/aggregate.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.log_util import log

ROOT = Path(__file__).resolve().parents[3]
EVAL_DIR = ROOT / "outputs" / "b3_pusht" / "actor" / "rollout_eval"
SEEDS = (0, 1, 2)

ROWS = [
    (1, "baseline (no graph)", "baseline_defaulteps"),
    (2, "auxphi, transonly (no id edges)", "auxphi_pct2"),
    (3, "auxphi, cross-traj <q10, weight 0", "auxphi_pct3"),
    (4, "auxphi, cross-traj <q10, weight 1", "auxphi_pct4"),
    (5, "auxphi, cross-traj <q90, weight 1", "auxphi_pct5"),
    (6, "auxphi, <q10 w=0 + [q10,q90) w=1", "auxphi_pct6"),
]


def mean_std(vals):
    import numpy as np
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None
    return float(np.mean(vals)), float(np.std(vals))


def load_condition(variant_tag):
    success, rho = [], []
    missing = []
    for seed in SEEDS:
        path = EVAL_DIR / f"expert_1000__{variant_tag}__s{seed}__sel-success__same_episode_goal25.json"
        if not path.exists():
            missing.append(str(path.name))
            continue
        d = json.loads(path.read_text())
        success.append(d["success_rate"])
        rho.append(d.get("peak_rho"))
    return success, rho, missing


def main():
    print(f"{'#':<3} {'condition':<38} {'success_rate (mean+-std)':<28} {'peak_rho (mean+-std)':<24} n_seeds")
    print("-" * 110)
    for num, desc, tag in ROWS:
        success, rho, missing = load_condition(tag)
        s_mean, s_std = mean_std(success)
        r_mean, r_std = mean_std(rho)
        s_str = f"{s_mean:.3f} +- {s_std:.3f}" if s_mean is not None else "MISSING"
        r_str = f"{r_mean:.4f} +- {r_std:.4f}" if r_mean is not None else "MISSING"
        print(f"{num:<3} {desc:<38} {s_str:<28} {r_str:<24} {len(success)}/{len(SEEDS)}")
        if missing:
            log(f"  missing: {missing}")


if __name__ == "__main__":
    main()
