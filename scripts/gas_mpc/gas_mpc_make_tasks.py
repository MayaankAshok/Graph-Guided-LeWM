"""Create the fixed task200u sets from held-out test episodes for every protocol."""
import argparse
import sys
from pathlib import Path

import numpy as np
import stable_worldmodel as swm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.envs import ENV_MECHANICS
from common.log_util import log
from gas_mpc.gas_mpc_eval import ENV, MECH, PAIRS_DIR, POOL, PROTOCOLS, ROOT, TASK_SEED, TASKS
from baselines.viability_cross_episode_baseline import load_or_make_pairs
from gas_mpc.gas_mpc_prepare import task_heldout_episodes


def presolved(mech, pairs):
    s, g = np.array(pairs["start_state"], dtype=np.float64), np.array(pairs["goal_state"], dtype=np.float64)
    return np.asarray(mech.goal_reached(s, g), dtype=bool)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", type=int, default=TASKS, help="must match the fixed pool size (2 in smoke mode)")
    args = ap.parse_args()
    if args.tasks != TASKS:
        ap.error(f"the active pool has exactly {TASKS} tasks")
    mech = MECH
    PAIRS_DIR.mkdir(parents=True, exist_ok=True)
    dataset = swm.data.HDF5Dataset(path=str(mech.h5_path(ROOT)), keys_to_cache=["action"])
    col = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    ep_col, step_col, state_col = dataset.get_col_data(col), dataset.get_col_data("step_idx"), mech.state_column(dataset)
    pool = POOL
    heldout_eps = task_heldout_episodes(ep_col)
    for name, proto in PROTOCOLS.items():
        pair_tag = "cross_episode" if proto["pairing"] == "cross_episode" else f"same_episode_off{proto['offset']}"
        pairs = load_or_make_pairs(PAIRS_DIR / f"pairs_{pair_tag}_{pool}.json", ep_col=ep_col, step_col=step_col,
                                   state_col=state_col, n=TASKS, seed=TASK_SEED, pairing=proto["pairing"],
                                   offset=proto["offset"] or 25, reject=mech.goal_reached,
                                   heldout_eps=heldout_eps)
        pre = presolved(mech, pairs)
        assert not pre.any(), f"{name}: {pre.sum()} pre-solved tasks survived the reject filter"
        pos, ang = mech.goal_errors(np.array(pairs["start_state"])[:, None], pairs["goal_state"])
        pos, ang = pos[:, 0], ang[:, 0]
        if ENV == "pusht":
            near = (pos < 60) & (np.degrees(ang) < 30)
            log(f"[{name}] {pool}: {TASKS} tasks, 0 pre-solved; initial agent+block err median {np.median(pos):.0f}px "
                f"(first 50: {np.median(pos[:50]):.0f}px); block-near {near.sum()} (first 50: {near[:50].sum()})")
        else:
            log(f"[{name}] {pool}: {TASKS} tasks, 0 pre-solved; initial primary error median {np.median(pos):.3f} "
                f"(first 50: {np.median(pos[:50]):.3f}), p90 {np.percentile(pos, 90):.3f}")


if __name__ == "__main__":
    main()
