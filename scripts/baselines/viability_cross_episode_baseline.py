"""The problem statement, measured: how often does LeWM's own CEM planner (latent-L2 cost,
the paper's planning config) reach a goal that is an ARBITRARY dataset state -- a random
frame from a DIFFERENT episode than the start -- as a function of the env-step budget?

Every closed-loop number in this repo so far uses the paper's protocol (goal = the frame
25/50 steps later in the SAME trajectory, so the goal is always a short expert continuation
away). This script drops that crutch: start and goal are two independent uniform-random
rows of the h5 with distinct episode ids, no distance filter, no reachability filter.

Deterministic pair sets: `outputs/pusht/pairs/pairs_cross_episode_n{N}_s{SEED}.json`
is generated from `np.random.default_rng(SEED)` alone and re-used verbatim on every later
run (any future planner/critic is scored on exactly these (start, goal) pairs). The CEM
solver is seeded with the same SEED.

Budgets: the L2 planner is budget-agnostic (nothing it computes depends on how many steps
remain) and stable_worldmodel's dataset-eval protocol freezes an env the moment it
terminates. So one rollout at `max_budget` with the FIRST success step recorded per pair
gives the success rate at every smaller budget exactly (success at budget B <=> first hit
<= B), not approximately: the first B steps of a budget-B run and of a budget-max run are
the same computation. `run_mode=separate` runs each budget as its own rollout instead
(needed later for any budget-AWARE cost, e.g. the critic with h_mode=remaining; here it is
a consistency check).

Plumbing check: `+cross.pairing=same_episode` re-implements the paper's protocol through
this script's own pair evaluator (goal = start + `cross.offset` in the same episode); it
must reproduce the ~94% eval.py/viability_live_rollout.py number at offset 25 / budget 50.

Run:
    python scripts/baselines/viability_cross_episode_baseline.py eval.num_eval=50 +cross.seeds=[0,1,2]
    python scripts/baselines/viability_cross_episode_baseline.py eval.num_eval=50 +cross.seeds=[0] \
        +cross.pairing=same_episode +cross.max_budget=50            # ~94% plumbing check
    python scripts/baselines/viability_cross_episode_baseline.py eval.num_eval=50 +cross.seeds=[0] \
        +cross.run_mode=separate +cross.budgets=[50,100]           # per-budget rollouts
    python scripts/baselines/viability_cross_episode_baseline.py eval.num_eval=50 +cross.seeds=[0,1,2,3,4] \
        +cross.aggregate=true      # no rollouts: combine per-seed result files (Ada runs one
                                   # one seed per GPU worker)
"""

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import json
import sys
import time
from copy import deepcopy
from pathlib import Path

import hydra
import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from omegaconf import DictConfig, ListConfig, OmegaConf
from sklearn import preprocessing
from stable_worldmodel.world.world import _apply_callables
from torchvision.transforms import v2 as transforms

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from common.envs import ENV_MECHANICS
from common.lewm_loader import load_lewm
from common.log_util import log


def img_transform(cfg):
    return transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(**spt.data.dataset_stats.ImageNet),
        transforms.Resize(size=cfg.eval.img_size),
    ])


def _as_int_list(v, default):
    if v is None:
        return list(default)
    if isinstance(v, (list, tuple, ListConfig)):
        return [int(x) for x in v]
    return [int(x) for x in str(v).split(",") if x.strip()]


# ----------------------------------------------------------------------------------------
# pair sets
# ----------------------------------------------------------------------------------------

def sample_pairs(ep_col, step_col, state_col, n, seed, pairing, offset, reject=None,
                 heldout_eps=None):
    """Deterministic (start, goal) rows. cross_episode: two independent uniform rows of the
    dataset, goal redrawn until its episode differs from the start's. same_episode: the
    paper's protocol (goal = start + offset in the same trajectory), for validating this
    script's evaluator against the published number.

    `reject(start_state, goal_state) -> bool array` (2026-09-18): pairs it flags are dropped
    at sampling. Passed the env's own success predicate (EnvMechanics.goal_reached) it removes
    tasks that are solved at t=0 -- on Cube 38% of all offset-25 windows, since the block only
    moves while carried; Push-T/Reacher have (almost) none."""
    rng = np.random.default_rng(seed)
    n_rows = len(ep_col)
    solved = (lambda a, b: np.asarray(reject(np.asarray(state_col[a], dtype=np.float64),
                                            np.asarray(state_col[b], dtype=np.float64)))) if reject else None
    if heldout_eps is not None:
        from common.heldout_tasks import sample_disjoint_rows
        start, goal = sample_disjoint_rows(ep_col, step_col, state_col, n, seed,
                                           pairing, offset, heldout_eps, reject)
    elif pairing == "cross_episode":
        start = rng.choice(n_rows, size=n, replace=False)
        goal = rng.choice(n_rows, size=n, replace=False)
        for i in range(n):
            while ep_col[goal[i]] == ep_col[start[i]] or (solved is not None and solved(start[i], goal[i])):
                goal[i] = rng.integers(n_rows)
    elif pairing == "same_episode":
        ep_ids, first = np.unique(ep_col, return_index=True)
        ep_len = np.diff(np.append(first, n_rows))
        max_start = ep_len[np.searchsorted(ep_ids, ep_col)] - offset - 1
        valid = np.nonzero(step_col <= max_start)[0]
        if solved is None:
            start = valid[rng.choice(len(valid) - 1, size=n, replace=False)]
        else:
            # walk a random permutation of the valid rows and keep the first n unsolved ones
            # (still a uniform draw over the unsolved windows); the pool is then sorted like the
            # unfiltered one, so "the first 50 tasks" is a row-ordered prefix in both conventions
            order, start = rng.permutation(len(valid) - 1), []
            for k in range(0, len(order), 4 * n):
                cand = valid[order[k: k + 4 * n]]
                start.extend(cand[~solved(cand, cand + offset)].tolist())
                if len(start) >= n:
                    break
            assert len(start) >= n, f"only {len(start)} unsolved windows at offset {offset}"
            start = np.array(start[:n])
        start = np.sort(start)
        goal = start + offset
        assert np.all(ep_col[goal] == ep_col[start])
    else:
        raise ValueError(pairing)
    if solved is not None:
        assert not solved(start, goal).any()
    result = dict(
        seed=int(seed), n=int(n), pairing=pairing, reject_solved=bool(reject is not None),
        offset=int(offset) if pairing == "same_episode" else None,
        start_row=start.tolist(), start_ep=ep_col[start].tolist(), start_step=step_col[start].tolist(),
        goal_row=goal.tolist(), goal_ep=ep_col[goal].tolist(), goal_step=step_col[goal].tolist(),
        start_state=state_col[start].tolist(), goal_state=state_col[goal].tolist(),
    )
    if heldout_eps is not None:
        result.update(heldout_eps=np.asarray(heldout_eps).tolist(), nonoverlap=True,
                      overlap_scope="within_protocol", interval_endpoints="inclusive")
    return result


def load_or_make_pairs(path, **kw):
    if path.exists():
        pairs = json.loads(path.read_text())
        fresh = sample_pairs(**kw)
        if kw.get("heldout_eps") is not None and not pairs.get("nonoverlap"):
            archive = path.with_name(path.stem + ".full_dataset.json")
            if archive.exists():
                raise FileExistsError(f"Refusing to overwrite archived task pool {archive}")
            path.rename(archive)
            path.write_text(json.dumps(fresh, indent=1))
            log(f"pairs: archived {archive}; wrote held-out pool {path}")
            return fresh
        for k in ("start_row", "goal_row"):
            assert pairs[k] == fresh[k], f"{path} does not match regeneration from its seed"
        log(f"pairs: loaded {path}")
        return pairs
    pairs = sample_pairs(**kw)
    path.write_text(json.dumps(pairs, indent=1))
    log(f"pairs: wrote {path}")
    return pairs


# ----------------------------------------------------------------------------------------
# evaluator: World._evaluate_from_dataset with the goal drawn from an arbitrary row
# ----------------------------------------------------------------------------------------

def extract_rows(dataset, eps, steps):
    """One dataset row per env, in the layout World._extract_init_goal produces."""
    eps, steps = np.asarray(eps), np.asarray(steps)
    out = {}
    for ep in dataset.load_chunk(eps, steps, steps + 1):
        for col in dataset.column_names:
            if col.startswith("goal"):
                continue
            val = ep[col]
            if not isinstance(val, (torch.Tensor, np.ndarray)):
                continue
            if col.startswith("pixels"):
                val = val.permute(0, 2, 3, 1)
            arr = val.numpy() if torch.is_tensor(val) else val
            out.setdefault(col, []).append(arr[0])
    return {k: np.stack(v) for k, v in out.items()}


def evaluate_pairs(world, dataset, pairs, budget, callables, mech=None, seed=None):
    """Mirror of stable_worldmodel's World._evaluate_from_dataset (mode='wait'), except
    init and goal come from two independently chosen rows. Returns the first env step at
    which each pair's env reported `terminated` (-1 = never within budget) and the state
    trajectory (n, budget+1, state_dim). `mech` (EnvMechanics, default Push-T) says how the
    state vector is read from dataset rows and live infos (Push-T: `state`; Reacher:
    qpos|qvel).

    `seed=None` (the historical default) leaves PushT.reset()'s `self.rng` and its
    variation-space resampling of agent/block start position+angle seeded from OS entropy;
    those three fields are immediately overwritten below via `_apply_callables`, so this
    was not the dominant source of the reproducibility gap measured 2026-09-21 (same seed,
    same tasks: identical success_rate/success_by_budget but 20-24% of individual tasks'
    final_pos_err differed by tens to ~240px) -- but it is still unseeded global state with
    no guarantee nothing else in the env/render path consults it, so callers doing a
    reproducibility-sensitive run should pass an explicit seed here regardless."""
    mech = mech or ENV_MECHANICS["pusht"]
    n = len(pairs["start_row"])
    assert n == world.num_envs
    init_state = extract_rows(dataset, pairs["start_ep"], pairs["start_step"])
    goal_raw = extract_rows(dataset, pairs["goal_ep"], pairs["goal_step"])
    goal_state = {("goal" if k == "pixels" else f"goal_{k}"): v for k, v in goal_raw.items()}
    assert np.allclose(mech.state_from_row(init_state), np.array(pairs["start_state"], dtype=np.float32))
    assert np.allclose(mech.state_from_row(goal_raw), np.array(pairs["goal_state"], dtype=np.float32))

    world.reset(seed=seed)
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
    states = [mech.state_from_infos(world.infos)]
    clock = [0]

    def on_step(w):
        w.infos.update(deepcopy(goal_snapshot))
        clock[0] += 1
        newly = (first_hit < 0) & w.terminateds
        first_hit[newly] = clock[0]
        states.append(mech.state_from_infos(w.infos))

    world._run(max_steps=budget, mode="wait", on_step=on_step)
    traj = np.stack(states, axis=1)                                   # (n, T+1, 7)
    if traj.shape[1] < budget + 1:                                     # every env done early
        pad = np.repeat(traj[:, -1:], budget + 1 - traj.shape[1], axis=1)
        traj = np.concatenate([traj, pad], axis=1)
    return first_hit, traj


def goal_errors(traj, goal):
    """Per-step components of PushT.eval_state against the goal: agent+block position error
    (px, success needs < 20) and block angle error (rad, success needs < pi/9)."""
    g = np.asarray(goal, dtype=np.float64)[:, None, :]
    pos = np.linalg.norm(traj[..., :4] - g[..., :4], axis=-1)
    ang = np.abs(traj[..., 4] - g[..., 4])
    ang = np.minimum(ang, 2 * np.pi - ang)
    return pos, ang


# ----------------------------------------------------------------------------------------

@hydra.main(version_base=None, config_path=str(ROOT / "config" / "eval"), config_name="pusht")
def main(cfg: DictConfig):
    t0 = time.time()
    mech = ENV_MECHANICS["pusht"]
    res_dir = ROOT / "outputs" / "pusht" / "pairs"
    res_dir.mkdir(parents=True, exist_ok=True)
    ccfg = cfg.get("cross", {})
    seeds = _as_int_list(ccfg.get("seeds"), [0])
    max_budget = int(ccfg.get("max_budget", 250))
    budgets = _as_int_list(ccfg.get("budgets"), range(25, max_budget + 1, 25))
    pairing = str(ccfg.get("pairing", "cross_episode"))
    offset = int(ccfg.get("offset", 25))
    run_mode = str(ccfg.get("run_mode", "curve"))
    run_tag = str(ccfg.get("tag", ""))
    n = int(cfg.eval.num_eval)
    assert run_mode in ("curve", "separate"), run_mode
    assert max(budgets) <= max_budget
    assert cfg.plan_config.horizon * cfg.plan_config.action_block <= min(budgets)

    log(f"=== LeWM CEM (L2) on {pairing} pairs: n={n} seeds={seeds} run_mode={run_mode} "
        f"max_budget={max_budget} budgets={budgets} CEM={OmegaConf.to_container(cfg.solver)} "
        f"plan={OmegaConf.to_container(cfg.plan_config)} ===")

    aggregate = bool(ccfg.get("aggregate", False))
    if not aggregate:
        h5_path = str(mech.h5_path(ROOT))
        dataset = swm.data.HDF5Dataset(path=h5_path, keys_to_cache=list(cfg.dataset.keys_to_cache))
        col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
        ep_col, step_col = dataset.get_col_data(col_name), dataset.get_col_data("step_idx")
        state_col = dataset.get_col_data("state")
        log(f"dataset: {h5_path} ({len(ep_col)} rows, {len(np.unique(ep_col))} episodes)")

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

    pair_tag = pairing if pairing == "cross_episode" else f"same_episode_off{offset}"
    all_runs = {}
    if aggregate:
        # combine per-seed result files written by earlier (e.g. one-seed-per-GPU) runs
        for seed in seeds:
            src = res_dir / f"lewm_cem_{pair_tag}_n{n}_s{seed}_B{max_budget}_{run_mode}{run_tag}.json"
            all_runs[seed] = json.loads(src.read_text())["runs"][str(seed)]
            all_runs[seed]["rollouts"] = {int(k): v for k, v in all_runs[seed]["rollouts"].items()}
            log(f"aggregate: loaded {src}")
        seeds_to_run = []
    else:
        seeds_to_run = seeds
    for seed in seeds_to_run:
        pairs_path = res_dir / f"pairs_{pair_tag}_n{n}_s{seed}.json"
        pairs = load_or_make_pairs(
            pairs_path, ep_col=ep_col, step_col=step_col, state_col=state_col, n=n, seed=seed,
            pairing=pairing, offset=offset)
        pos0, ang0 = goal_errors(np.array(pairs["start_state"])[:, None], pairs["goal_state"])
        pos0, ang0 = pos0[:, 0], ang0[:, 0]
        log(f"\n--- seed {seed}: {n} pairs, initial pos err median {np.median(pos0):.0f} px "
            f"(min {pos0.min():.0f}, max {pos0.max():.0f}), angle err median "
            f"{np.degrees(np.median(ang0)):.0f} deg; already-at-goal: "
            f"{int(((pos0 < 20) & (ang0 < np.pi / 9)).sum())} ---")

        cfg.seed = int(seed)                       # CEM solver seed follows the pair seed
        run_budgets = [max_budget] if run_mode == "curve" else budgets
        seed_out = dict(pairs_file=str(pairs_path), rollouts={})
        for budget in run_budgets:
            model = load_lewm(ckpt_dir=ckpt_dir, device="cuda")
            model.interpolate_pos_encoding = True
            config = swm.PlanConfig(**cfg.plan_config)
            solver = hydra.utils.instantiate(cfg.solver, model=model)
            policy = swm.policy.WorldModelPolicy(
                solver=solver, config=config, process=process, transform=transform)
            world = swm.World(env_name=cfg.world.env_name, num_envs=n,
                              max_episode_steps=2 * budget, image_shape=(224, 224))
            world.set_policy(policy)
            t_run = time.time()
            first_hit, traj = evaluate_pairs(world, dataset, pairs, budget, callables)
            dt = time.time() - t_run
            world.close()
            del model, solver, policy, world
            torch.cuda.empty_cache()

            pos, ang = goal_errors(traj, pairs["goal_state"])
            success_at = {b: float(np.mean((first_hit >= 0) & (first_hit <= b)) * 100)
                          for b in budgets if b <= budget}
            log(f"[seed {seed} budget {budget}] success={success_at[budget]:.1f}%  "
                f"({dt:.0f}s)  first-hit steps: {sorted(first_hit[first_hit >= 0].tolist())}")
            log("    success by budget: " + "  ".join(f"{b}:{v:.0f}" for b, v in success_at.items()))
            log(f"    min pos err over episode: median {np.median(pos.min(1)):.0f} px "
                f"(start {np.median(pos0):.0f}); final pos err median {np.median(pos[:, -1]):.0f}")
            seed_out["rollouts"][budget] = dict(
                secs=dt, first_hit_step=first_hit.tolist(), success_by_budget=success_at,
                initial_pos_err=pos0.tolist(), initial_ang_err=ang0.tolist(),
                min_pos_err=pos.min(1).tolist(), final_pos_err=pos[:, -1].tolist(),
                final_ang_err=ang[:, -1].tolist())
            np.savez_compressed(
                res_dir / f"traj_{pair_tag}_n{n}_s{seed}_B{budget}{run_tag}.npz",
                traj=traj.astype(np.float32), first_hit=first_hit,
                goal_state=np.array(pairs["goal_state"], dtype=np.float32))
        all_runs[seed] = seed_out

    log("\n=== SUMMARY: success rate (%) by env-step budget ===")
    log("budget  " + "".join(f"seed{s:<6d}" for s in seeds) + "mean   std    pooled")
    summary = {}
    for b in budgets:
        vals, hits = [], 0
        for seed in seeds:
            key = max_budget if run_mode == "curve" else b
            fh = np.array(all_runs[seed]["rollouts"][key]["first_hit_step"])
            ok = (fh >= 0) & (fh <= b)
            vals.append(float(ok.mean() * 100))
            hits += int(ok.sum())
        pooled = 100 * hits / (n * len(seeds))
        summary[b] = dict(per_seed=vals, mean=float(np.mean(vals)), std=float(np.std(vals)), pooled=pooled)
        log(f"{b:<8d}" + "".join(f"{v:<10.1f}" for v in vals)
            + f"{np.mean(vals):<7.1f}{np.std(vals):<7.1f}{pooled:.1f}")

    out = dict(pairing=pairing, offset=offset if pairing == "same_episode" else None, n=n,
               seeds=seeds, run_mode=run_mode, max_budget=max_budget, budgets=budgets,
               solver=OmegaConf.to_container(cfg.solver),
               plan_config=OmegaConf.to_container(cfg.plan_config),
               summary=summary, runs=all_runs, secs=time.time() - t0)
    p = res_dir / (f"lewm_cem_{pair_tag}_n{n}_s{'-'.join(map(str, seeds))}"
                   f"_B{max_budget}_{run_mode}{run_tag}.json")
    p.write_text(json.dumps(out, indent=1))
    log(f"wrote {p} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
