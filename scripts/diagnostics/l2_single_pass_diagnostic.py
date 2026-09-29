"""Diagnostic: LeWM's own L2 CEM planner reaching a goal 25 env steps ahead in the same
episode, but capped at a 25-step budget -- exactly one CEM solve (horizon 5 blocks x
action_block 5 = 25 env steps), executed open-loop with NO replanning. This isolates a
single CEM pass from the receding-horizon protocol used everywhere else in this project
(`same25` in gas_mpc_eval.py uses budget=50, allowing one replan at step 25).

For every task, records:
  - predicted: the CEM cost it optimized (LeWM.criterion = sum((z_pred_T - z_goal)**2), the
    exact quantity `outputs['costs']` holds after solver.solve -- see
    stable_worldmodel/solver/cem.py:263,270 and wm/lewm/lewm.py's `criterion`). Since all
    envs' action buffers are empty at t=0 and the budget never lets a second replan happen,
    solver.solve is called exactly once per chunk, covering every env in that chunk in its
    original order (verified via WorldModelPolicy.get_action: replan_idx == range(n_envs)
    on the first call).
  - realized latent distance: the SAME quantity (sum of squared differences over the 192-d
    CLS latent), but computed AFTER actually simulating the chosen 25 actions in the real
    env -- the true final frame is re-encoded through the same frozen LeWM encoder and
    compared to the goal latent, instead of relying on the predictor's own imagined
    rollout. This is the model-based planning error made visible: CEM only ever sees its
    own predictor's rollout, never the real dynamics.
  - realized physical error: final-state position/angle error against the goal after the
    25-step open-loop rollout (same `goal_errors` used by every other eval script here),
    plus whether the env's own `terminated` fired within the budget (success, PushT's
    `eval_state`: pos err < 20px and angle err < pi/9). Kept as supplementary context.

Uses the SAME fixed 200-task set (`pairs_same_episode_off25_task200.json`, drawn from
gas_mpc_eval.TASK_SEED) as every other GAS-MPC evaluation in this project, so this is
directly comparable to the `same25` row elsewhere -- the only thing that changes is the
budget (25 vs 50) and the extra `predicted` field.

Run (single seed = CEM's own RNG seed; task set itself is independent of this seed):
    python scripts/diagnostics/l2_single_pass_diagnostic.py eval.num_eval=200 +diag.seed=0
    python scripts/diagnostics/l2_single_pass_diagnostic.py eval.num_eval=200 +diag.seed=0 +diag.chunk=25
"""

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import json
import sys
import time
from pathlib import Path

from copy import deepcopy

import hydra
import numpy as np
import stable_worldmodel as swm
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn import preprocessing
from stable_worldmodel.world.world import _apply_callables

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from common.envs import ENV_MECHANICS
from common.lewm_loader import load_lewm
from common.log_util import log
from gas_mpc.gas_mpc_eval import PAIRS_DIR, TASK_SEED
from baselines.viability_cross_episode_baseline import extract_rows, goal_errors, img_transform, load_or_make_pairs

OUT_DIR = ROOT / "outputs" / "pusht" / "pairs"
BUDGET = 25          # == plan_config.horizon * plan_config.action_block: exactly one CEM solve
OFFSET = 25
TASKS = 200


def attach_cost_hook(solver, storage):
    """Wrap solver.solve to record the CEM cost it returns for each env, in call order."""
    orig = solver.solve

    def solve(info_dict, init_action=None):
        out = orig(info_dict, init_action=init_action)
        storage.extend(out["costs"])
        return out

    solver.solve = solve


def evaluate_pairs_capture_final_pixels(world, dataset, pairs, budget, callables):
    """Mirror of viability_cross_episode_baseline.evaluate_pairs, but also returns the raw
    (pre-transform, multi-frame-history) 'pixels'/'goal' info at the final step, so the
    realized rollout's true final frame can be re-encoded through the same frozen LeWM
    encoder used for planning (rather than relying on the predictor's own imagined
    rollout). Overwritten every step so it always holds whatever the last on_step call
    saw, even if envs finish (freeze) before `budget` steps."""
    n = len(pairs["start_row"])
    assert n == world.num_envs
    init_state = extract_rows(dataset, pairs["start_ep"], pairs["start_step"])
    goal_raw = extract_rows(dataset, pairs["goal_ep"], pairs["goal_step"])
    goal_state = {("goal" if k == "pixels" else f"goal_{k}"): v for k, v in goal_raw.items()}
    assert np.allclose(init_state["state"], np.array(pairs["start_state"], dtype=np.float32))
    assert np.allclose(goal_state["goal_state"], np.array(pairs["goal_state"], dtype=np.float32))

    world.reset(seed=None)
    merged = {**init_state, **goal_state}
    for i in range(n):
        _apply_callables(world.envs.envs[i].unwrapped, callables, {k: v[i] for k, v in merged.items()})

    shape_prefix = world.infos["pixels"].shape[:2]
    for src in (init_state, goal_state):
        for k, v in src.items():
            if k in world.infos or k in goal_state:
                world.infos[k] = np.broadcast_to(v[:, None, ...], shape_prefix + v.shape[1:]).copy()
    goal_snapshot = {k: world.infos[k].copy() for k in goal_state}

    first_hit = np.full(n, -1, dtype=np.int64)
    states = [world.infos["state"][:, -1].copy()]
    clock = [0]
    final_raw = {"pixels": world.infos["pixels"].copy(), "goal": world.infos["goal"].copy()}

    def on_step(w):
        w.infos.update(deepcopy(goal_snapshot))
        clock[0] += 1
        newly = (first_hit < 0) & w.terminateds
        first_hit[newly] = clock[0]
        states.append(w.infos["state"][:, -1].copy())
        final_raw["pixels"] = w.infos["pixels"].copy()
        final_raw["goal"] = w.infos["goal"].copy()

    world._run(max_steps=budget, mode="wait", on_step=on_step)
    traj = np.stack(states, axis=1)                                   # (n, T+1, 7)
    if traj.shape[1] < budget + 1:                                     # every env done early
        pad = np.repeat(traj[:, -1:], budget + 1 - traj.shape[1], axis=1)
        traj = np.concatenate([traj, pad], axis=1)
    return first_hit, traj, final_raw


@hydra.main(version_base=None, config_path=str(ROOT / "config" / "eval"), config_name="pusht")
def main(cfg: DictConfig):
    t0 = time.time()
    dcfg = OmegaConf.merge(OmegaConf.create(dict(seed=0, chunk=25, force=False)), cfg.get("diag", {}))
    n = int(cfg.eval.num_eval)
    assert n <= TASKS, f"this diagnostic is defined over the fixed {TASKS}-task set; got eval.num_eval={n}"
    seed = int(dcfg.seed)
    chunk = int(dcfg.chunk)
    mech = ENV_MECHANICS["pusht"]
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    run_id = f"l2_single_pass_off{OFFSET}_B{BUDGET}__task{TASKS}__s{seed}__n{n}__v2"
    final_path = OUT_DIR / f"{run_id}.json"
    if final_path.exists() and not dcfg.force:
        r = json.loads(final_path.read_text())
        log(f"[done] {final_path.name} exists: success={r['success_rate']:.1f}% -- skipping")
        return

    log(f"=== l2_single_pass_diagnostic: offset={OFFSET} budget={BUDGET} (single CEM pass, "
        f"no replan) tasks={TASKS} seed={seed} chunk={chunk} "
        f"plan={OmegaConf.to_container(cfg.plan_config)} ===")
    assert cfg.plan_config.horizon * cfg.plan_config.action_block == BUDGET, (
        "budget must equal horizon*action_block for this to be a single CEM pass")

    h5_path = str(mech.h5_path(ROOT))
    dataset = swm.data.HDF5Dataset(path=h5_path, keys_to_cache=list(cfg.dataset.keys_to_cache))
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    ep_col, step_col = dataset.get_col_data(col_name), dataset.get_col_data("step_idx")
    state_col = dataset.get_col_data("state")
    log(f"dataset: {h5_path} ({len(ep_col)} rows, {len(np.unique(ep_col))} episodes)")

    pairs_path = PAIRS_DIR / f"pairs_same_episode_off{OFFSET}_task{TASKS}.json"
    pairs = load_or_make_pairs(pairs_path, ep_col=ep_col, step_col=step_col, state_col=state_col,
                               n=TASKS, seed=TASK_SEED, pairing="same_episode", offset=OFFSET)
    pairs = {k: (v[:n] if isinstance(v, list) else v) for k, v in pairs.items()}   # the first n tasks
    pairs["n"] = n

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
    callables = OmegaConf.to_container(cfg.eval.get("callables"), resolve=True)
    ckpt_dir = mech.ckpt_dir(ROOT)

    chunks = [(c, list(range(c, min(c + chunk, n)))) for c in range(0, n, chunk)]
    results = {}
    for ci, idx in chunks:
        cpath = OUT_DIR / f"{run_id}__c{ci}.json"
        if cpath.exists():
            results[ci] = json.loads(cpath.read_text())
            log(f"[chunk {ci}] cached: success={results[ci]['success_rate']:.1f}%")
            continue
        sub = {k: ([v[i] for i in idx] if isinstance(v, list) else v) for k, v in pairs.items()}
        sub["n"] = len(idx)
        cfg.seed = seed * 1000 + ci                 # CEM solver seed
        model = load_lewm(ckpt_dir=ckpt_dir, device="cuda")
        model.interpolate_pos_encoding = True
        config = swm.PlanConfig(**cfg.plan_config)
        solver = hydra.utils.instantiate(cfg.solver, model=model)
        predicted_costs = []
        attach_cost_hook(solver, predicted_costs)
        policy = swm.policy.WorldModelPolicy(solver=solver, config=config, process=process, transform=transform)
        world = swm.World(env_name=cfg.world.env_name, num_envs=len(idx), max_episode_steps=2 * BUDGET,
                          image_shape=(224, 224))
        world.set_policy(policy)
        t_run = time.time()
        first_hit, traj, final_raw = evaluate_pairs_capture_final_pixels(world, dataset, sub, BUDGET, callables)
        dt = time.time() - t_run

        assert len(predicted_costs) == len(idx), (
            f"expected exactly one CEM solve covering all {len(idx)} envs in this chunk "
            f"(single pass, no replan); got {len(predicted_costs)} cost entries -- some env "
            f"replanned, budget/horizon assumption violated")

        with torch.no_grad():
            dev = next(model.parameters()).device
            prepared = policy._prepare_info({"pixels": final_raw["pixels"], "goal": final_raw["goal"]})
            z_final = model.encode({"pixels": prepared["pixels"].to(dev)})["emb"][:, -1]
            z_goal = model.encode({"pixels": prepared["goal"].to(dev)})["emb"][:, -1]
            realized_latent_dist = ((z_final - z_goal) ** 2).sum(dim=-1).cpu().numpy()

        world.close()
        del model, solver, policy, world
        torch.cuda.empty_cache()

        pos, ang = goal_errors(traj, sub["goal_state"])
        pos0, ang0 = goal_errors(np.array(sub["start_state"])[:, None], sub["goal_state"])
        rec = dict(pair_idx=idx, first_hit_step=first_hit.tolist(),
                   success_rate=float(np.mean(first_hit >= 0) * 100), secs=dt,
                   predicted_cost=[float(c) for c in predicted_costs],
                   realized_latent_dist=realized_latent_dist.tolist(),
                   initial_pos_err=pos0[:, 0].tolist(), initial_ang_err=ang0[:, 0].tolist(),
                   final_pos_err=pos[:, -1].tolist(), final_ang_err=ang[:, -1].tolist())
        cpath.write_text(json.dumps(rec, indent=1))
        results[ci] = rec
        log(f"[chunk {ci}] success={rec['success_rate']:.1f}% ({dt:.0f}s) "
            f"predicted cost median {np.median(predicted_costs):.4f} "
            f"realized latent dist median {np.median(realized_latent_dist):.4f}")

    fh = np.concatenate([np.array(results[ci]["first_hit_step"]) for ci, _ in chunks])
    out = dict(offset=OFFSET, budget=BUDGET, tasks=TASKS, seed=seed, n=n,
               pairs_file=str(pairs_path),
               success_rate=float(np.mean(fh >= 0) * 100), n_success=int((fh >= 0).sum()),
               first_hit_step=fh.tolist(),
               predicted_cost=sum((results[ci]["predicted_cost"] for ci, _ in chunks), []),
               realized_latent_dist=sum((results[ci]["realized_latent_dist"] for ci, _ in chunks), []),
               initial_pos_err=sum((results[ci]["initial_pos_err"] for ci, _ in chunks), []),
               initial_ang_err=sum((results[ci]["initial_ang_err"] for ci, _ in chunks), []),
               final_pos_err=sum((results[ci]["final_pos_err"] for ci, _ in chunks), []),
               final_ang_err=sum((results[ci]["final_ang_err"] for ci, _ in chunks), []),
               solver=OmegaConf.to_container(cfg.solver), plan_config=OmegaConf.to_container(cfg.plan_config),
               secs=sum(results[ci]["secs"] for ci, _ in chunks), wall=time.time() - t0)
    final_path.write_text(json.dumps(out, indent=1))
    log(f"=== l2_single_pass: success={out['success_rate']:.1f}% ({out['n_success']}/{n}) "
        f"median final pos err {np.median(out['final_pos_err']):.0f}px; "
        f"median predicted cost {np.median(out['predicted_cost']):.4f}; "
        f"median realized latent dist {np.median(out['realized_latent_dist']):.4f}; "
        f"wrote {final_path.name} ({time.time()-t0:.0f}s) ===")


if __name__ == "__main__":
    main()
