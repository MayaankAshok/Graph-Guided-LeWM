"""Create (or verify) the fixed task sets shared by every seed: pairs_{protocol}_{POOL}.json in
gas_mpc_eval.PAIRS_DIR (outputs/pusht/pairs/ for Push-T, outputs/<env>/pairs/
otherwise -- GAS_MPC_ENV selects the environment), using TASK_SEED and the saved TDR split.

Pool conventions (gas_mpc_eval.POOL):
    task200u (current): both endpoints are held out; trivial tasks are
              rejected before reserving intervals. Within each protocol, same-episode
              intervals [start, goal] cannot overlap, including shared endpoints.
              Cross-episode tasks reserve their two endpoint frames. Protocol pools
              are independent and can reuse episodes/frames across protocols.
    Earlier task200u full-dataset draws are archived as *.full_dataset.json on migration.
    task200   the unfiltered draw every result before 2026-09-18 used; kept on disk, never
              rewritten (`--audit-old` counts its pre-solved tasks per protocol)

    python scripts/gas_mpc_make_tasks.py --tasks 200
    GAS_MPC_ENV=reacher python scripts/gas_mpc_make_tasks.py --tasks 200 --audit-old
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import stable_worldmodel as swm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.envs import ENV_MECHANICS
from common.log_util import log
from gas_mpc_eval import ENV, MECH, PAIRS_DIR, POOL, PROTOCOLS, ROOT, TASK_SEED
from viability_cross_episode_baseline import load_or_make_pairs
from gas_mpc_prepare import task_heldout_episodes


def presolved(mech, pairs):
    s, g = np.array(pairs["start_state"], dtype=np.float64), np.array(pairs["goal_state"], dtype=np.float64)
    return np.asarray(mech.goal_reached(s, g), dtype=bool)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", type=int, default=200)
    ap.add_argument("--asset-seed", type=int, default=0,
                    help="TDR asset whose saved held-out split defines the task episodes")
    ap.add_argument("--audit-old", action="store_true",
                    help="also count pre-solved tasks in the archived unfiltered task{T} pools")
    args = ap.parse_args()
    mech = MECH
    PAIRS_DIR.mkdir(parents=True, exist_ok=True)
    dataset = swm.data.HDF5Dataset(path=str(mech.h5_path(ROOT)), keys_to_cache=["action"])
    col = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    ep_col, step_col, state_col = dataset.get_col_data(col), dataset.get_col_data("step_idx"), mech.state_column(dataset)
    pool = POOL if args.tasks == 200 else f"task{args.tasks}u"
    heldout_eps = task_heldout_episodes(ep_col, args.asset_seed)
    for name, proto in PROTOCOLS.items():
        pair_tag = "cross_episode" if proto["pairing"] == "cross_episode" else f"same_episode_off{proto['offset']}"
        pairs = load_or_make_pairs(PAIRS_DIR / f"pairs_{pair_tag}_{pool}.json", ep_col=ep_col, step_col=step_col,
                                   state_col=state_col, n=args.tasks, seed=TASK_SEED, pairing=proto["pairing"],
                                   offset=proto["offset"] or 25, reject=mech.goal_reached,
                                   heldout_eps=heldout_eps)
        pre = presolved(mech, pairs)
        assert not pre.any(), f"{name}: {pre.sum()} pre-solved tasks survived the reject filter"
        pos, ang = mech.goal_errors(np.array(pairs["start_state"])[:, None], pairs["goal_state"])
        pos, ang = pos[:, 0], ang[:, 0]
        old = PAIRS_DIR / f"pairs_{pair_tag}_task{args.tasks}.json"
        old_note = ""
        if args.audit_old and old.exists():
            pre_old = presolved(mech, json.loads(old.read_text()))
            old_note = (f"; archived unfiltered pool: {pre_old[:50].sum()}/50 pre-solved in the first 50, "
                        f"{pre_old.sum()}/{len(pre_old)} overall")
        if ENV == "pusht":
            near = (pos < 60) & (np.degrees(ang) < 30)
            log(f"[{name}] {pool}: {args.tasks} tasks, 0 pre-solved; initial agent+block err median {np.median(pos):.0f}px "
                f"(first 50: {np.median(pos[:50]):.0f}px); block-near {near.sum()} (first 50: {near[:50].sum()}){old_note}")
        else:
            log(f"[{name}] {pool}: {args.tasks} tasks, 0 pre-solved; initial primary error median {np.median(pos):.3f} "
                f"(first 50: {np.median(pos[:50]):.3f}), p90 {np.percentile(pos, 90):.3f}{old_note}")


if __name__ == "__main__":
    main()
