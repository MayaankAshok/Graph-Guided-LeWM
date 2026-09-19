"""Diagnose the exact failed tasks in the paper-protocol Push-T LeWM evaluation.

Runs the unmodified seed-42 task bank, records the actions and predicted endpoint selected
at each MPC replan, replays failed tasks in real physics, and compares the first selected
plan with the known-successful dataset continuation.  A privileged real-physics CEM then
checks directed recovery from the selected endpoints in both directions.
"""

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import h5py
import hydra
import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from common.envs import ENV_MECHANICS
from common.lewm_loader import load_lewm
from common.log_util import log
from planning_cost_gate import make_encode_frame
from viability_inversion_audit import _init_oracle_worker, oracle_endpoint_job


def img_transform(cfg):
    return transforms.Compose([
        transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(**spt.data.dataset_stats.ImageNet),
        transforms.Resize(size=cfg.eval.img_size),
    ])


def read_rows(col, rows):
    rows = np.asarray(rows, dtype=np.int64)
    order = np.argsort(rows)
    out = np.empty((len(rows),) + col.shape[1:], dtype=col.dtype)
    out[order] = col[rows[order]]
    return out


def sample_eval_tasks(dataset, cfg):
    ep_col = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    ep_ids = np.unique(dataset.get_col_data(ep_col))
    all_ep = dataset.get_col_data(ep_col)
    all_step = dataset.get_col_data("step_idx")
    lengths = np.array([all_step[all_ep == ep].max() + 1 for ep in ep_ids])
    max_start = dict(zip(ep_ids, lengths - cfg.eval.goal_offset_steps - 1))
    valid = np.nonzero(all_step <= np.array([max_start[ep] for ep in all_ep]))[0]
    # This deliberately preserves eval.py's slightly odd len(valid)-1 sampling bound.
    picked = np.random.default_rng(cfg.seed).choice(
        len(valid) - 1, size=cfg.eval.num_eval, replace=False)
    rows = np.sort(valid[picked])
    data = dataset.get_row_data(rows)
    return rows, np.asarray(data[ep_col]), np.asarray(data["step_idx"]), ep_col


class TracePolicy(swm.policy.WorldModelPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.raw_actions = []

    def get_action(self, info_dict, **kwargs):
        action = super().get_action(info_dict, **kwargs)
        self.raw_actions.append(np.asarray(action).copy())
        return action


def state_error(state, goal):
    angle = abs(float(state[4] - goal[4]))
    angle = min(angle, 2 * np.pi - angle)
    return {
        "agent_pos": float(np.linalg.norm(state[:2] - goal[:2])),
        "block_pos": float(np.linalg.norm(state[2:4] - goal[2:4])),
        "angle_rad": angle,
        "official_pos4": float(np.linalg.norm(state[:4] - goal[:4])),
        "state_l2": float(np.linalg.norm(state - goal)),
    }


def eval_candidates(model, info, candidates):
    """Evaluate one candidate per env in batch-size-1 chunks, matching CEMSolver.

    LeWM's published criterion relies on broadcasting that is only valid at the configured
    solver batch_size=1, so evaluating all 50 environments together would change its shape
    semantics rather than merely trace it.
    """
    n_samples = candidates.shape[1]
    costs, preds, goals, starts = [], [], [], []
    for i in range(candidates.shape[0]):
        copied = {}
        for key, value in info.items():
            if torch.is_tensor(value):
                one = value[i:i + 1].to("cuda")
                copied[key] = one.unsqueeze(1).expand(-1, n_samples, *one.shape[1:])
            elif isinstance(value, np.ndarray):
                copied[key] = np.repeat(value[i:i + 1, None], n_samples, axis=1)
            else:
                copied[key] = value
        with torch.inference_mode():
            cost = model.get_cost(copied, candidates[i:i + 1].to("cuda"))
        costs.append(cost[0, 0].item())
        preds.append(copied["predicted_emb"][0, 0, -1].float().cpu().numpy())
        goals.append(copied["goal_emb"][0, -1].float().cpu().numpy())
        history = info["pixels"].shape[1]
        starts.append(copied["predicted_emb"][0, 0, history - 1].float().cpu().numpy())
    return np.asarray(costs), np.stack(preds), np.stack(goals), np.stack(starts)


@hydra.main(version_base=None, config_path=str(ROOT / "config" / "eval"), config_name="pusht")
def main(cfg: DictConfig):
    if cfg.eval.goal_offset_steps != 25 or cfg.eval.eval_budget != 50:
        raise ValueError("This diagnostic is defined for the paper protocol: offset=25, budget=50")
    t0 = time.time()
    mech = ENV_MECHANICS["pusht"]
    h5_path = mech.h5_path(ROOT)
    dataset = swm.data.HDF5Dataset(
        path=str(h5_path), keys_to_cache=list(cfg.dataset.keys_to_cache))
    rows, ep_ids, start_steps, ep_col = sample_eval_tasks(dataset, cfg)

    process = {}
    for col in cfg.dataset.keys_to_cache:
        if col == "pixels":
            continue
        values = dataset.get_col_data(col)
        values = values[~np.isnan(values).any(axis=1)]
        process[col] = preprocessing.StandardScaler().fit(values)
        if col != "action":
            process[f"goal_{col}"] = process[col]

    model = load_lewm(ckpt_dir=mech.ckpt_dir(ROOT), device="cuda")
    model.interpolate_pos_encoding = True
    solver = swm.solver.CEMSolver(
        model=model, batch_size=cfg.solver.batch_size, num_samples=cfg.solver.num_samples,
        var_scale=cfg.solver.var_scale, n_steps=cfg.solver.n_steps, topk=cfg.solver.topk,
        device=cfg.solver.device, seed=cfg.seed)

    solve_records = []
    real_solve = solver.solve

    def traced_solve(info, init_action=None):
        out = real_solve(info, init_action=init_action)
        selected = out["actions"].unsqueeze(1)
        cost, pred, goal, start = eval_candidates(model, info, selected)
        solve_records.append({"info": dict(info), "actions_z": selected.cpu(),
                              "cost": cost, "pred": pred, "goal": goal, "start": start})
        return out

    solver.solve = traced_solve
    policy = TracePolicy(
        solver=solver, config=swm.PlanConfig(**cfg.plan_config), process=process,
        transform={"pixels": img_transform(cfg), "goal": img_transform(cfg)})
    world = swm.World(env_name=cfg.world.env_name, num_envs=cfg.eval.num_eval,
                      max_episode_steps=2 * cfg.eval.eval_budget, image_shape=(224, 224))
    world.set_policy(policy)
    metrics = world.evaluate(
        dataset=dataset, start_steps=start_steps.tolist(), goal_offset=25,
        eval_budget=50, episodes_idx=ep_ids.tolist(),
        callables=OmegaConf.to_container(cfg.eval.callables, resolve=True), video=None)
    world.close()
    failed = np.nonzero(~np.asarray(metrics["episode_successes"], dtype=bool))[0]
    raw_actions = np.stack(policy.raw_actions)  # (50 steps, 50 envs, 2)
    log(f"baseline={metrics['success_rate']}% failed_slots={failed.tolist()}")

    with h5py.File(h5_path, "r", swmr=True) as f:
        offsets = f["ep_offset"][:]
        start_rows = offsets[ep_ids.astype(np.int64)] + start_steps.astype(np.int64)
        goal_rows = start_rows + 25
        start_states = mech.read_state_rows(f, start_rows)
        goal_states = mech.read_state_rows(f, goal_rows)
        goal_pixels = read_rows(f["pixels"], goal_rows)
        expert_raw = np.stack([f["action"][r:r + 25] for r in start_rows]).astype(np.float32)

    expert_z = process["action"].transform(expert_raw.reshape(-1, 2)).reshape(
        len(rows), 1, 5, 10)
    expert_cost, expert_pred, _, _ = eval_candidates(
        model, solve_records[0]["info"], torch.from_numpy(expert_z).float())
    selected_pred = solve_records[0]["pred"]
    selected_cost = solve_records[0]["cost"]
    encode = make_encode_frame(model)
    z_goal_render = encode(goal_pixels)

    env = mech.make_env()
    details, endpoint_jobs, endpoint_keys = [], [], []
    for slot in failed:
        traces = {}
        for name, actions in (("selected", raw_actions[:, slot]),
                              ("expert", expert_raw[slot])):
            obs, _ = env.reset(seed=cfg.seed,
                               options=mech.reset_options(start_states[slot], goal_states[slot]))
            success_step, snapshots = None, {}
            for step, action in enumerate(actions, start=1):
                obs, reward, terminated, truncated, _ = env.step(action)
                if step in (25, 50):
                    snapshots[step] = (np.asarray(obs["state"]).copy(), env.render().copy())
                if terminated and success_step is None:
                    success_step = step
                    break
            if 25 not in snapshots:
                snapshots[25] = (np.asarray(obs["state"]).copy(), env.render().copy())
            traces[name] = (success_step, snapshots)

        selected_state25, selected_pix25 = traces["selected"][1][25]
        selected_state50, selected_pix50 = traces["selected"][1][50]
        expert_state25, expert_pix25 = traces["expert"][1][25]
        z_exec = encode(np.stack([selected_pix25, expert_pix25, selected_pix50]))
        goal = goal_states[slot]
        goal_z = z_goal_render[slot]
        z0 = solve_records[0]["start"][slot]
        goal_dir = goal_z - z0
        goal_dir /= np.linalg.norm(goal_dir) + 1e-12

        d = {
            "slot": int(slot), "dataset_row": int(rows[slot]), "episode": int(ep_ids[slot]),
            "start_step": int(start_steps[slot]),
            "selected": {
                "pred_goal_cost": float(selected_cost[slot]),
                "exec_goal_cost": float(np.sum((z_exec[0] - goal_z) ** 2)),
                "model_endpoint_error": float(np.linalg.norm(selected_pred[slot] - z_exec[0])),
                "predicted_goal_progress": float((selected_pred[slot] - z0) @ goal_dir),
                "executed_goal_progress": float((z_exec[0] - z0) @ goal_dir),
                "step25_error": state_error(selected_state25, goal),
                "step50_error": state_error(selected_state50, goal),
            },
            "expert_control": {
                "success_step": traces["expert"][0],
                "pred_goal_cost": float(expert_cost[slot]),
                "exec_goal_cost": float(np.sum((z_exec[1] - goal_z) ** 2)),
                "model_endpoint_error": float(np.linalg.norm(expert_pred[slot] - z_exec[1])),
                "step25_error": state_error(expert_state25, goal),
            },
            "l2_symmetry_abs_error": float(abs(
                np.linalg.norm(z_exec[0] - goal_z) - np.linalg.norm(goal_z - z_exec[0]))),
        }
        second_i = int(np.nonzero(failed == slot)[0][0])
        second = solve_records[1]
        second_dir = second["goal"][second_i] - second["start"][second_i]
        second_dir /= np.linalg.norm(second_dir) + 1e-12
        d["selected_second_replan"] = {
            "pred_goal_cost": float(second["cost"][second_i]),
            "exec_goal_cost": float(np.sum((z_exec[2] - second["goal"][second_i]) ** 2)),
            "model_endpoint_error": float(np.linalg.norm(second["pred"][second_i] - z_exec[2])),
            "predicted_goal_progress": float(
                (second["pred"][second_i] - second["start"][second_i]) @ second_dir),
            "executed_goal_progress": float(
                (z_exec[2] - second["start"][second_i]) @ second_dir),
        }
        details.append(d)

        for key, state, target, horizon in (
            ("step25_to_goal_h25", selected_state25, goal, 25),
            ("goal_to_step25_h25", goal, selected_state25, 25),
            ("step50_to_goal_h50", selected_state50, goal, 50),
            ("goal_to_step50_h50", goal, selected_state50, 50),
        ):
            endpoint_keys.append((len(details) - 1, key, horizon))
            endpoint_jobs.append((state, target, 91_000 + slot * 100 + horizon, horizon,
                                  4, 128, 8, 0.15))
    env.close()

    if endpoint_jobs:
        with ProcessPoolExecutor(max_workers=min(len(endpoint_jobs), 12),
                                 mp_context=get_context("spawn"),
                                 initializer=_init_oracle_worker) as pool:
            oracle_results = list(pool.map(oracle_endpoint_job, endpoint_jobs))
        for (detail_i, key, horizon), (steps, best_dist) in zip(endpoint_keys, oracle_results):
            details[detail_i].setdefault("directed_oracle", {})[key] = {
                "steps": None if steps == horizon + 1 else int(steps),
                "censored": bool(steps == horizon + 1), "best_state_dist": float(best_dist)}

    out = {
        "config": {"seed": int(cfg.seed), "num_eval": int(cfg.eval.num_eval),
                   "goal_offset": 25, "eval_budget": 50},
        "baseline_success_rate": float(metrics["success_rate"]),
        "failed_slots": failed.tolist(), "failures": details,
        "seconds": time.time() - t0,
    }
    out_path = ROOT / "outputs" / "pusht" / "diagnostics" / "pusht_lewm_failures_seed42.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    log(f"wrote {out_path} ({out['seconds']:.1f}s)")


if __name__ == "__main__":
    main()
