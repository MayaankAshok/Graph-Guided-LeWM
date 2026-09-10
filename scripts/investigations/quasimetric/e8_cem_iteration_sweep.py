"""E8 -- does iterative CEM narrowing specifically hurt the quasimetric cost?

E6 showed the quasimetric head's SINGLE-SHOT best-of-K pick (no iteration at all) is closer
to the true goal than raw L2's best pick, on the same broad candidate pool (1.31 vs 1.42 mean
true distance, n=2000 groups). E7's variance-shrinkage traces looked nearly identical between
costs, arguing against a "search fails to converge" story. Yet E4/E5's FULL 30-iteration CEM
result favors L2 by a wide margin. The one thing that changes between "best-of-K, no
iteration" and "full CEM" is the ITERATIVE narrowing itself: each of CEM's 30 rounds refits
its sampling distribution to its OWN previous elite set, which can amplify an early, small
systematic bias rather than just find-and-stop at a single evaluation. If narrowing is the
culprit, live outcome under d_Q should be BEST at few iterations (closer to the single-shot
best-of-K result, which favors d_Q) and get WORSE as n_steps increases toward 30 -- the
opposite trend from L2, which should stay flat or improve with more iterations.

This sweeps `cfg.solver.n_steps` in {1, 3, 5, 10, 20, 30} for l2 and quasimetric_v3, using
the SAME one-shot single-solve methodology as E5 (eval_budget == horizon*action_block, so
each n_steps value gets exactly one full solve+execution against real env-stepped ground
truth), on the same 50 (start, goal) pairs at every n_steps for a fair sweep.

Run:
    python scripts/investigations/quasimetric/e8_cem_iteration_sweep.py
"""

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import json
import sys
import time
from pathlib import Path

import hydra
import numpy as np
import stable_worldmodel as swm
import torch
from omegaconf import DictConfig, OmegaConf
from scipy import stats as sps
from sklearn import preprocessing

ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.envs import ENV_MECHANICS
from common.lewm_loader import load_lewm
from common.log_util import log
from common.quasimetric import LatentQuasimetric
from e4_live_rollout import (get_episodes_length, img_transform,
                             make_quasimetric_criterion, recompute_quasimetric_norm_stats)

N_STEPS_SWEEP = [int(x) for x in os.environ.get("E8_N_STEPS_SWEEP", "1,3,5,10,20,30").split(",")]
CONDITIONS = {"l2": None, "quasimetric_v3": "e3v3_quasimetric_head_pusht.pt"}


@hydra.main(version_base=None, config_path=str(ROOT / "config" / "eval"), config_name="pusht")
def main(cfg: DictConfig):
    t0 = time.time()
    mech = ENV_MECHANICS["pusht"]
    res_dir = ROOT / "outputs" / "quasimetric"
    res_dir.mkdir(parents=True, exist_ok=True)
    eval_budget_one_shot = cfg.plan_config.horizon * cfg.plan_config.action_block

    log(f"=== E8 CEM iteration sweep (num_eval={cfg.eval.num_eval}, steps={N_STEPS_SWEEP}) ===")

    h5_path = str(mech.h5_path(ROOT))
    dataset = swm.data.HDF5Dataset(path=h5_path, keys_to_cache=list(cfg.dataset.keys_to_cache))
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"

    process = {}
    for col in cfg.dataset.keys_to_cache:
        if col == "pixels":
            continue
        processor = preprocessing.StandardScaler()
        col_data = dataset.get_col_data(col)
        col_data = col_data[~np.isnan(col_data).any(axis=1)]
        processor.fit(col_data)
        process[col] = processor
        if col != "action":
            process[f"goal_{col}"] = process[col]
    transform = {"pixels": img_transform(cfg), "goal": img_transform(cfg)}

    ep_indices, _ = np.unique(dataset.get_col_data(col_name), return_index=True)
    episode_len = get_episodes_length(dataset, ep_indices)
    max_start_idx = episode_len - cfg.eval.goal_offset_steps - 1
    max_start_idx_dict = {ep_id: max_start_idx[i] for i, ep_id in enumerate(ep_indices)}
    max_start_per_row = np.array([max_start_idx_dict[e] for e in dataset.get_col_data(col_name)])
    valid_mask = dataset.get_col_data("step_idx") <= max_start_per_row
    valid_indices = np.nonzero(valid_mask)[0]
    g = np.random.default_rng(cfg.seed)
    random_episode_indices = g.choice(len(valid_indices) - 1, size=cfg.eval.num_eval, replace=False)
    random_episode_indices = np.sort(valid_indices[random_episode_indices])
    eval_episodes = dataset.get_row_data(random_episode_indices)[col_name]
    eval_start_idx = dataset.get_row_data(random_episode_indices)["step_idx"]

    import h5py
    with h5py.File(h5_path, "r", swmr=True) as f5:
        ep_offset = f5["ep_offset"][:]
        state_col = f5["state"]
        ep_ids = np.asarray(eval_episodes).astype(np.int64)
        goal_rows = ep_offset[ep_ids] + np.asarray(eval_start_idx).astype(np.int64) + cfg.eval.goal_offset_steps
        order = np.argsort(goal_rows)
        goal_states = np.empty((len(goal_rows), state_col.shape[1]), dtype=state_col.dtype)
        goal_states[order] = state_col[goal_rows[order]]
        true_dist_oracle = mech.build_true_distance_oracle(np.asarray(state_col[::37]))

    ckpt_dir = mech.ckpt_dir(ROOT)
    quasi_mu, quasi_sd = recompute_quasimetric_norm_stats(
        Path(os.environ.get("PROBE_CACHE_DIR", str(ROOT / "outputs" / "probe_gate_pusht" / "cache"))),
        n_episodes=18685, seed=0, train_frac=0.8,
    )

    results = {cond: {} for cond in CONDITIONS}
    for cond, head_name in CONDITIONS.items():
        for n_steps in N_STEPS_SWEEP:
            log(f"\n--- condition={cond}  n_steps={n_steps} ---")
            model = load_lewm(ckpt_dir=ckpt_dir, device="cuda")
            model.interpolate_pos_encoding = True
            if head_name is not None:
                quasi_model = LatentQuasimetric(latent_dim=192, hidden_dim=256, proj_dim=64)
                quasi_model.load_state_dict(torch.load(res_dir / head_name, map_location="cpu"))
                quasi_model = quasi_model.to("cuda").eval()
                quasi_model.requires_grad_(False)
                model.criterion = make_quasimetric_criterion(quasi_model, quasi_mu, quasi_sd)

            solver = swm.solver.CEMSolver(
                model=model, batch_size=cfg.solver.batch_size, num_samples=cfg.solver.num_samples,
                var_scale=cfg.solver.var_scale, n_steps=n_steps, topk=cfg.solver.topk,
                device=cfg.solver.device, seed=cfg.seed,
            )
            config = swm.PlanConfig(**cfg.plan_config)
            policy = swm.policy.WorldModelPolicy(
                solver=solver, config=config, process=process, transform=transform)
            world = swm.World(env_name=cfg.world.env_name, num_envs=cfg.eval.num_eval,
                              max_episode_steps=2 * eval_budget_one_shot, image_shape=(224, 224))
            world.set_policy(policy)

            t_run = time.time()
            metrics = world.evaluate(
                dataset=dataset, start_steps=eval_start_idx.tolist(),
                goal_offset=cfg.eval.goal_offset_steps, eval_budget=eval_budget_one_shot,
                episodes_idx=eval_episodes.tolist(),
                callables=OmegaConf.to_container(cfg.eval.get("callables"), resolve=True),
                video=None,
            )
            dt = time.time() - t_run

            final_state = np.asarray(world.infos["state"])
            if final_state.ndim == 3:
                final_state = final_state[:, -1]
            true_dist = true_dist_oracle(final_state, goal_states)
            true_dist = np.diagonal(true_dist) if true_dist.ndim == 2 else true_dist

            log(f"[{cond} n_steps={n_steps}] success_rate={metrics.get('success_rate')}  "
                f"true_dist mean={true_dist.mean():.3f}+-{true_dist.std():.3f}  ({dt:.1f}s)")
            results[cond][n_steps] = dict(
                success_rate=metrics.get("success_rate"),
                true_dist_mean=float(true_dist.mean()), true_dist_std=float(true_dist.std()),
                secs=dt,
            )
            world.close()

    log("\n=== SUMMARY: true_dist mean (lower=better) and success_rate vs. n_steps ===")
    log(f"{'n_steps':>8s}  {'l2 success':>10s}  {'l2 true_dist':>12s}  "
        f"{'d_Q success':>11s}  {'d_Q true_dist':>13s}")
    for n_steps in N_STEPS_SWEEP:
        rl2, rq = results["l2"][n_steps], results["quasimetric_v3"][n_steps]
        log(f"{n_steps:>8d}  {rl2['success_rate']:>10.1f}  {rl2['true_dist_mean']:>12.3f}  "
            f"{rq['success_rate']:>11.1f}  {rq['true_dist_mean']:>13.3f}")

    out = dict(num_eval=cfg.eval.num_eval, seed=cfg.seed, n_steps_sweep=N_STEPS_SWEEP,
              results=results, secs=time.time() - t0)
    steps_tag = "-".join(str(x) for x in N_STEPS_SWEEP)
    p = res_dir / f"e8_cem_iteration_sweep_pusht_n{cfg.eval.num_eval}_steps{steps_tag}.json"
    p.write_text(json.dumps(out, indent=2))
    log(f"wrote {p} ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
