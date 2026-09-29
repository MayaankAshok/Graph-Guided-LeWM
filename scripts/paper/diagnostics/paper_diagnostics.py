"""Recompute paper budget windows and Push-T success tolerances from evaluator outputs.

    python scripts/paper/diagnostics/paper_diagnostics.py --eval-dir outputs/pusht/eval
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np


def threshold_success(traj, goal, pixels, degrees):
    goal = goal[:, None, :]
    pos = np.linalg.norm(traj[..., :4] - goal[..., :4], axis=-1)
    angle = np.abs(traj[..., 4] - goal[..., 4])
    angle = np.minimum(angle, 2 * np.pi - angle)
    return np.any((pos < pixels) & (angle < np.deg2rad(degrees)), axis=1)


def summarize(result_path):
    record = json.loads(result_path.read_text())
    n = record["n"]
    hits = np.asarray(record["first_hit_step"])
    if len(hits) != n:
        raise ValueError(f"bad task count: {result_path}")
    budget = int(record["budget"])
    cumulative = {step: float(np.mean((hits >= 0) & (hits <= step)) * 100)
                  for step in range(50, budget + 1, 50)}
    chunks = sorted(result_path.parent.glob(result_path.stem + "__c*_traj.npz"))
    if not chunks:
        raise FileNotFoundError(f"missing trajectories for {result_path}")
    trajectories = []
    goals = []
    for path in chunks:
        with np.load(path) as data:
            trajectories.append(data["traj"])
            goals.append(data["goal_state"])
    traj, goal = np.concatenate(trajectories), np.concatenate(goals)
    if len(traj) != n:
        raise ValueError(f"trajectory count mismatch for {result_path}")
    tolerances = {tol: float(threshold_success(traj, goal, tol, tol).mean() * 100)
                  for tol in (20, 25, 30)}
    return record, cumulative, tolerances


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-dir", type=Path, default=Path("outputs/pusht/eval"))
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        traj = np.array([[[0, 0, 0, 0, 0], [21, 0, 0, 0, 0]],
                         [[24, 0, 0, 0, 0], [30, 0, 0, 0, 0]]])
        goal = np.zeros((2, 5))
        assert threshold_success(traj, goal, 20, 20).tolist() == [True, False]
        assert threshold_success(traj, goal, 25, 25).tolist() == [True, True]
        return
    rows = {}
    for path in sorted(args.eval_dir.glob("*.json")):
        if "__c" in path.stem or not re.search(r"_cfg[0-9a-f]{12}__", path.stem):
            continue
        record, cumulative, tolerances = summarize(path)
        if record.get("pool") != "task200u" or record["n"] != 50:
            continue
        method = re.sub(r"_cfg[0-9a-f]{12}$", "", record["method"])
        key = (method, record["protocol"])
        if record["seed"] in rows.setdefault(key, {}):
            raise ValueError(f"duplicate result for {key} seed {record['seed']}: {path}")
        rows[key][record["seed"]] = cumulative, tolerances
    for (method, protocol), seeds in sorted(rows.items()):
        print(f"{method} / {protocol}: {len(seeds)} seeds")
        for name, index in (("cumulative", 0), ("tolerance", 1)):
            keys = sorted({k for value in seeds.values() for k in value[index]})
            print(f"  {name}: " + ", ".join(
                f"{key}={np.mean([value[index][key] for value in seeds.values()]):.1f}%"
                for key in keys))
        means = {step: np.mean([value[0][step] for value in seeds.values()])
                 for step in sorted(next(iter(seeds.values()))[0])}
        previous = 0.0
        hazards = []
        for step, current in means.items():
            rate = 100 * (current - previous) / (100 - previous) if previous < 100 else 0.0
            hazards.append(f"{step}={rate:.1f}%")
            previous = current
        print("  hazard: " + ", ".join(hazards))


if __name__ == "__main__":
    main()
