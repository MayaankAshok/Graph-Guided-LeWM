"""Post-hoc diagnostic: does the task admit multiple, qualitatively different successful
strategies (measured as cross-CEM-seed trajectory diversity on plain OUR, no retrieval), and
does that predict whether demonstration retrieval helps?

Pure analysis over EXISTING eval outputs (5 CEM seeds x same-ep-25, OUR config, no retrieval)
-- no new CEM solves, no GPU. For each env, for each task solved by ALL 5 seeds, compute the
mean pairwise trajectory distance across the 5 seeds' state trajectories (10 pairs), in the
env's own state units, restricted to the first `min` common horizon. Average over qualifying
tasks -> one "solution multiplicity" number per env. Compare against known retrieval outcome:
Cube (retrieval helps, +20-28pts), Push-T (retrieval hurts, -1.6 to -3.2pts).
"""
import glob
import json
import os

import numpy as np

ROOT = "/home2/mayaank.ashok/lewm_research/outputs"

ENVS = {
    "pusht": dict(
        eval_dir=f"{ROOT}/pusht/eval",
        # OUR headline config, no retrieval, same-ep 25, all 5 seeds
        pattern="subgoal_tdr_la13.7_th8_htd8_te0.9_ft13.7_fl2_crit1_ccet__same25__task200u__s{seed}__n50__heldout_disjoint__trainonly_assets",
        state_slice=slice(0, 5),   # agent xy, block xy, block angle (angle needs care, see below)
        angle_dim=4,
        success_tol=20.0,          # px, PushTMechanics goal_errors' own success threshold
    ),
    "cube": dict(
        eval_dir=f"{ROOT}/cube/eval",
        # OUR headline config has a per-seed pipe hash -- glob it instead of a fixed pattern
        pattern=None,
        state_slice=slice(0, 6),   # block xyz, effector xyz
        angle_dim=None,
        success_tol=0.04,          # m, CubeMechanics.SUCCESS_DIST
    ),
    "reacher": dict(
        eval_dir=f"{ROOT}/reacher/eval",
        pattern=None,
        state_slice=slice(0, 2),   # qpos: shoulder (wraps), wrist (range-limited)
        angle_dim="reacher_shoulder",
        success_tol=0.05,          # rad, ReacherMechanics.QPOS_THRESHOLD (max-abs per joint)
    ),
}


def find_agg_json(cfg, seed):
    if cfg["pattern"] is not None:
        path = os.path.join(cfg["eval_dir"], cfg["pattern"].format(seed=seed) + ".json")
        return path if os.path.exists(path) else None
    # cube/reacher: OUR tag has a per-seed pipe hash -- glob it instead of a fixed pattern
    stem = {
        "cube": "subgoal_tdr_la13.63_th7.25_htd7.25_te0.9_ft13.63_fl2_crit1_ccet",
        "reacher": "subgoal_tdr_la5.83_th3.72_htd3.72_te0.9_ft5.83_fl2_crit1_ccet",
    }[cfg["_env"]]
    hits = glob.glob(os.path.join(
        cfg["eval_dir"],
        f"{stem}_pipe_*__same25__task200u__s{seed}__n50__heldout_disjoint__trainonly_assets.json"))
    return hits[0] if hits else None


def load_seed(cfg, seed):
    agg_path = find_agg_json(cfg, seed)
    if agg_path is None:
        print(f"  [skip] seed {seed}: no aggregate json found")
        return None
    base = agg_path[:-len(".json")]
    chunk_jsons = sorted(glob.glob(base + "__c*.json"))
    if not chunk_jsons:
        print(f"  [skip] seed {seed}: no chunk jsons alongside {agg_path}")
        return None
    pair_idx_all, traj_all, first_hit_all = [], [], []
    for cj in chunk_jsons:
        rec = json.load(open(cj))
        traj_path = cj[:-len(".json")] + "_traj.npz"
        if not os.path.exists(traj_path):
            print(f"  [skip] seed {seed}: missing {traj_path}")
            return None
        tz = np.load(traj_path)
        pair_idx_all.extend(rec["pair_idx"])
        first_hit_all.extend(rec["first_hit_step"])
        traj_all.append(tz["traj"])
    traj_all = np.concatenate(traj_all, axis=0)
    return dict(pair_idx=np.array(pair_idx_all), first_hit=np.array(first_hit_all), traj=traj_all)


def angle_diff(a, b):
    d = np.abs(a - b)
    return np.minimum(d, 2 * np.pi - d)


def traj_distance(cfg, t_a, t_b):
    """Mean per-step Euclidean distance over the common horizon, in the env's raw state
    units (px for Push-T positions, m for Cube). Push-T's angle dim is wrapped separately and
    folded in with a a small fixed scale (pi rad ~ half the block's own diameter's worth of
    visual rotation, so this is a rough combination, not a calibrated metric -- fine for a
    same-units-across-env-instances relative comparison, not for cross-env absolute values)."""
    T = min(t_a.shape[0], t_b.shape[0])
    a, b = t_a[:T], t_b[:T]
    if cfg["angle_dim"] == 4:  # Push-T: agent+block xy position diversity (angle excluded, diff units)
        lin = np.linalg.norm(a[:, :4] - b[:, :4], axis=-1)
        return float(lin.mean())
    if cfg["angle_dim"] == "reacher_shoulder":  # max-abs joint diff, matching the success predicate
        d0 = angle_diff(a[:, 0], b[:, 0])
        d1 = np.abs(a[:, 1] - b[:, 1])
        return float(np.maximum(d0, d1).mean())
    return float(np.linalg.norm(a[:, :cfg["state_slice"].stop] - b[:, :cfg["state_slice"].stop], axis=-1).mean())


def main():
    for env, cfg in ENVS.items():
        cfg["_env"] = env
        print(f"\n=== {env} ===")
        seeds = {}
        for seed in range(5):
            d = load_seed(cfg, seed)
            if d is not None:
                seeds[seed] = d
        if len(seeds) < 2:
            print("  not enough seeds loaded, skipping")
            continue
        seed_ids = sorted(seeds)
        # common task set: solved (first_hit >= 0) in every loaded seed
        common_idx = None
        per_seed_lookup = {}
        for s in seed_ids:
            d = seeds[s]
            solved = set(int(pi) for pi, fh in zip(d["pair_idx"], d["first_hit"]) if fh >= 0)
            common_idx = solved if common_idx is None else (common_idx & solved)
            per_seed_lookup[s] = {int(pi): i for i, pi in enumerate(d["pair_idx"])}
        common_idx = sorted(common_idx)
        print(f"  seeds loaded: {seed_ids}, tasks solved by ALL: {len(common_idx)}/{len(per_seed_lookup[seed_ids[0]])}")
        if not common_idx:
            continue
        per_task_div = []
        for pi in common_idx:
            trajs = [seeds[s]["traj"][per_seed_lookup[s][pi]] for s in seed_ids]
            dists = []
            for i in range(len(trajs)):
                for j in range(i + 1, len(trajs)):
                    dists.append(traj_distance(cfg, trajs[i], trajs[j]))
            per_task_div.append(np.mean(dists))
        per_task_div = np.array(per_task_div)
        tol = cfg["success_tol"]
        print(f"  cross-seed trajectory diversity (mean pairwise dist, env units): "
              f"mean={per_task_div.mean():.3f} median={np.median(per_task_div):.3f} "
              f"p25={np.percentile(per_task_div,25):.3f} p75={np.percentile(per_task_div,75):.3f}")
        unit = {"pusht": "px", "cube": "m", "reacher": "rad"}[env]
        print(f"  same, in units of the env's own success tolerance ({tol} {unit}): "
              f"mean={per_task_div.mean()/tol:.2f}x median={np.median(per_task_div)/tol:.2f}x")


if __name__ == "__main__":
    main()
