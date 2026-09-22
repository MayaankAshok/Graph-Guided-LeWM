"""Trace OUR's actual CEM optimization on its FIRST plan call per task, and validate its
candidate ranking against the privileged real-physics oracle at select iterations.

For `--n` held-out task200u tasks (same25 / same50, seed s0), runs OUR (subgoal_tdr +
phase-8 final switch + ET critic, exactly `scripts/ada_gas_mpc_phase8.sh` config B) for only
the FIRST CEM solve (25 env steps -- no replanning, no env execution beyond one CEM solve).
At every logged iteration, records the full population (300 candidates), OUR's cost, LeWM's
own z-space L2 cost on the SAME candidates (computed for free -- same predicted rollout,
different criterion), and the sampling mean/variance. At iterations 1, 10, 20 (1-indexed),
additionally: takes the 30 elite candidates (by OUR's own ranking), executes each one for
real in the simulator from the task's true start state, and runs the privileged real-physics
CEM oracle (scripts/viability_inversion_audit.py) from each elite's true endpoint to the
task's goal -- giving a ground-truth recovery-time ranking to compare both costs against.

Implementation notes:
  - stable_worldmodel's CEMSolver (batch_size=1 in this project's config) calls its
    Callback.reset() once per solve() call and start_batch() once per task within a solve;
    subclassing that lets a callback detect "this is the SECOND solve" and abort by raising,
    which is how this script stops after exactly one CEM solve per task without touching the
    library or running a full closed-loop rollout.
  - model.criterion is wrapped so OUR's own criterion (elite selection) and the untouched LeWM
    L2 criterion are both evaluated on the identical predicted rollout every iteration --  no
    extra predictor calls, so the L2 score is a free byproduct of the same solve.

    python scripts/gas_mpc_cem_trace_audit.py mpc.protocol=same25
    python scripts/gas_mpc_cem_trace_audit.py mpc.protocol=same50
"""
import json
import pickle
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
from stable_worldmodel.solver.callbacks import Callback

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from common.lewm_loader import load_lewm
from common.log_util import log
from gas_mpc_eval import (ENV, MECH, OUT, PAIRS_DIR, POOL, PROTOCOLS, TASK_SEED, TASKS,
                          GraphOracle, img_transform, make_criterion)
from gas_mpc_prepare import graph_path, load_tdr, task_heldout_episodes
from viability_cross_episode_baseline import load_or_make_pairs
from viability_inversion_audit import _init_oracle_worker, oracle_endpoint_job, oracle_seed

DEV = "cuda" if torch.cuda.is_available() else "cpu"
LOG_ITERS_0IDX = [0, 4, 9, 14, 19, 24, 29]     # iterations 1,5,10,15,20,25,30 (1-indexed)
N_ELITE = 30


class _StopAfterFirstSolve(Exception):
    pass


class TraceCallback(Callback):
    """See module docstring. Captures the first solve only; raises to abort the second."""

    def __init__(self, log_iters, oracle_iters, l2_log):
        super().__init__()
        self.log_iters, self.oracle_iters = set(log_iters), set(oracle_iters)
        self.l2_log = l2_log
        self.solve_idx = -1
        self.task_idx = -1
        self.trace = []          # per (task, logged iteration): candidates/costs/mean/var
        self.elites = []         # per (task, oracle iteration): elite candidates + both costs

    def reset(self):
        self.solve_idx += 1
        if self.solve_idx > 0:
            raise _StopAfterFirstSolve()
        self.task_idx = -1

    def start_batch(self):
        if self._current:
            pass
        self.task_idx += 1

    def end_solve(self):
        pass

    def __call__(self, **state):
        step = state["step"]
        if step not in self.log_iters and step not in self.oracle_iters:
            return
        l2_cost = self.l2_log[-1][0].numpy()          # (300,) matches this exact criterion call
        our_cost = state["costs"][0].detach().cpu().numpy()
        if step in self.log_iters:
            self.trace.append(dict(
                task_idx=self.task_idx, step=int(step),
                our_cost=our_cost, l2_cost=l2_cost,
                mean=state["prev_mean"][0].detach().cpu().numpy(),
                var=state["prev_var"][0].detach().cpu().numpy(),
                new_mean=state["mean"][0].detach().cpu().numpy(),
                new_var=state["var"][0].detach().cpu().numpy(),
            ))
        if step in self.oracle_iters:
            topk_inds = state["topk_inds"][0].detach().cpu().numpy()          # (30,)
            topk_candidates = state["topk_candidates"][0].detach().cpu().numpy()  # (30, H, A)
            self.elites.append(dict(
                task_idx=self.task_idx, step=int(step),
                candidates=topk_candidates,                # z-scored, (30, horizon, action_dim)
                our_cost=our_cost[topk_inds], l2_cost=l2_cost[topk_inds],
            ))


def per_task_spearman(cost, T):
    if len(cost) < 3 or np.ptp(T) == 0:
        return float("nan")
    return float(sps.spearmanr(cost, T).statistic)


def mean_abs_rank_mismatch(cost, T):
    """Mean |rank(cost) - rank(T)| over the elite set -- 0 = perfect agreement."""
    rc, rt = sps.rankdata(cost), sps.rankdata(T)
    return float(np.mean(np.abs(rc - rt)))


def run_oracle_on_elites(mech, env_pool, actions_zscored, scaler, action_low, action_high,
                         start_state, goal_state, seed_base, oracle_cap, oracle_trials,
                         oracle_npop, oracle_niter, oracle_elite_frac):
    """actions_zscored: (30, horizon, action_dim) -> real env execution -> oracle T per elite."""
    from common.envs import ENV_MECHANICS
    n_elite, horizon, adim = actions_zscored.shape
    raw = actions_zscored.reshape(n_elite, -1, mech.action_dim)          # (30, 25, 2), z-scored
    raw = scaler.inverse_transform(raw.reshape(-1, mech.action_dim)).reshape(n_elite, -1, mech.action_dim)
    raw = np.clip(raw, action_low, action_high).astype(np.float32)

    env = ENV_MECHANICS["pusht"].make_env()
    final_states = []
    for k in range(n_elite):
        env.reset(seed=int(seed_base + k), options=mech.reset_options(start_state, goal_state))
        obs = None
        for a in raw[k]:
            obs, *_ = env.step(a)
        final_states.append(np.asarray(obs["state"], dtype=np.float32))
    env.close()
    final_states = np.stack(final_states)

    jobs = [(final_states[k], goal_state, oracle_seed(seed_base, 0, 0, k), oracle_cap,
            oracle_trials, oracle_npop, oracle_niter, oracle_elite_frac) for k in range(n_elite)]
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=8, initializer=_init_oracle_worker) as pool:
        results = list(pool.map(oracle_endpoint_job, jobs))
    T = np.array([r[0] for r in results], dtype=np.float64)
    best_dist = np.array([r[1] for r in results], dtype=np.float64)
    return T, best_dist, final_states, raw


@hydra.main(version_base=None, config_path=str(ROOT / "config" / "eval"), config_name=MECH.eval_config_name)
def main(cfg: DictConfig):
    t0 = time.time()
    mech = MECH
    m = OmegaConf.create(dict(protocol="same25", seed=0, graph_seed=0, n=25,
                              h_td=8.0, te=0.9, lookahead=13.7, final_thresh=13.7,
                              final_metric="l2", critic_beta=1.0, critic_cost="et",
                              critic=str(OUT / "critic_s0_tdr_holdout" / "critic.pt"),
                              oracle_cap=100, oracle_trials=2, oracle_npop=64, oracle_niter=5,
                              oracle_elite_frac=0.15, oracle_iters="1,10,20"))
    m = OmegaConf.merge(m, cfg.get("mpc", {}))
    proto = PROTOCOLS[m.protocol]
    n, seed = int(m.n), int(m.seed)
    oracle_iters_0idx = [int(x) - 1 for x in str(m.oracle_iters).split(",") if x.strip()]
    cfg.plan_config.receding_horizon = 5

    h5_path, ckpt_dir = str(mech.h5_path(ROOT)), mech.ckpt_dir(ROOT)
    dataset = swm.data.HDF5Dataset(path=h5_path, keys_to_cache=list(cfg.dataset.keys_to_cache))
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    ep_col, step_col = dataset.get_col_data(col_name), dataset.get_col_data("step_idx")
    state_col = mech.state_column(dataset)
    pair_tag = f"same_episode_off{proto['offset']}"
    pairs_path = PAIRS_DIR / f"pairs_{pair_tag}_{POOL}.json"
    heldout_eps = task_heldout_episodes(ep_col, int(m.graph_seed))
    pairs = load_or_make_pairs(pairs_path, ep_col=ep_col, step_col=step_col, state_col=state_col, n=TASKS,
                               seed=TASK_SEED, pairing=proto["pairing"], offset=proto["offset"],
                               reject=mech.goal_reached, heldout_eps=heldout_eps)
    pairs = {k: (v[:n] if k.startswith(("start_", "goal_")) and isinstance(v, list) else v)
             for k, v in pairs.items()}
    log(f"=== trace audit protocol={m.protocol} n={n} seed={seed} tasks 0..{n - 1} of {POOL} ===")

    training_rows = ~np.isin(ep_col, heldout_eps)
    with np.load(OUT / "cache_train.npz") as stats:
        action_mean, action_std = stats["act_mean"], stats["act_std"]
    process = {}
    for col in cfg.dataset.keys_to_cache:
        if col == "pixels":
            continue
        processor = preprocessing.StandardScaler()
        if col == "action":
            processor.mean_ = action_mean
            processor.scale_ = np.where(action_std > 0, action_std, 1.0)
            processor.var_ = action_std ** 2
            processor.n_features_in_ = len(action_mean)
            processor.n_samples_seen_ = int(training_rows.sum())
        else:
            col_data = dataset.get_col_data(col)[training_rows]
            col_data = col_data[~np.isnan(col_data).any(axis=1)]
            processor.fit(col_data)
        process[col] = processor
        if col != "action":
            process[f"goal_{col}"] = process[col]
    transform = {"pixels": img_transform(cfg), "goal": img_transform(cfg)}
    callables = OmegaConf.to_container(cfg.eval.get("callables"), resolve=True)
    action_scaler = process["action"]

    tdr, ck = load_tdr(int(m.graph_seed))
    tdr_hist = ck["history"][-1]
    with open(graph_path(int(m.graph_seed), m.h_td, m.te), "rb") as fh:
        g = pickle.load(fh)
    if np.isin(ep_col[g["kept_rows"]], heldout_eps).any():
        raise ValueError("Evaluation tasks overlap graph-building frames")
    step_units = tdr_hist["median_by_gap"].get(5, tdr_hist["median_by_gap"].get("5"))
    oracle = GraphOracle(g, tdr, m.h_td, m.h_td, float(m.lookahead), step_units,
                         n_waypoints=int(cfg.plan_config.horizon) - 3 + 1, final_thresh=float(m.final_thresh))
    log(f"[tdr] {tdr_hist}; step_units={step_units}")

    from viability_eval_audit import load_critic
    critic, cargs = load_critic(str(m.critic))
    log(f"[critic] {m.critic} h_max={cargs['h_max']}")

    from gas_mpc_eval import PlanClock
    model = load_lewm(ckpt_dir=ckpt_dir, device="cuda")
    model.interpolate_pos_encoding = True
    budget = int(proto["budget"])
    clock = PlanClock(budget, int(cfg.plan_config.receding_horizon) * int(cfg.plan_config.action_block),
                      int(cfg.plan_config.horizon) * int(cfg.plan_config.action_block), cargs["h_max"])

    l2_fn = model.criterion               # raw LeWM z-space L2 (kept for comparison, unused by OUR)
    our_criterion = make_criterion("subgoal_tdr", oracle, 0.0, int(cfg.plan_config.horizon), l2_fn,
                                   critic=critic, critic_beta=float(m.critic_beta), clock=clock,
                                   final_metric=str(m.final_metric), critic_cost=str(m.critic_cost))
    l2_log = []

    def wrapped_criterion(info):
        our_cost = our_criterion(info)
        with torch.no_grad():
            l2_log.append(l2_fn(info).detach().cpu())
        return our_cost

    model.criterion = wrapped_criterion

    config = swm.PlanConfig(**cfg.plan_config)
    solver = hydra.utils.instantiate(cfg.solver, model=model)
    cb = TraceCallback(LOG_ITERS_0IDX, oracle_iters_0idx, l2_log)
    solver.callbacks = [cb]
    policy = swm.policy.WorldModelPolicy(solver=solver, config=config, process=process, transform=transform)
    world = swm.World(env_name=cfg.world.env_name, num_envs=n, max_episode_steps=2 * budget, image_shape=(224, 224))
    world.set_policy(policy)

    from viability_cross_episode_baseline import extract_rows
    init_state = extract_rows(dataset, pairs["start_ep"], pairs["start_step"])
    goal_raw = extract_rows(dataset, pairs["goal_ep"], pairs["goal_step"])
    goal_state = {("goal" if k == "pixels" else f"goal_{k}"): v for k, v in goal_raw.items()}
    from stable_worldmodel.world.world import _apply_callables
    world.reset(seed=None)
    merged = {**init_state, **goal_state}
    for i in range(n):
        _apply_callables(world.envs.envs[i].unwrapped, callables, {k: v[i] for k, v in merged.items()})
    shape_prefix = world.infos["pixels"].shape[:2]
    for src in (init_state, goal_state):
        for k, v in src.items():
            if k in world.infos or k in goal_state:
                world.infos[k] = np.broadcast_to(v[:, None, ...], shape_prefix + v.shape[1:]).copy()

    log(f"[cem] running first solve for {n} tasks (this raises after the 2nd solve begins, by design)")
    try:
        world._run(max_steps=budget, mode="wait", on_step=lambda w: None)
    except _StopAfterFirstSolve:
        log("[cem] first solve captured; stopped before the second replan")
    else:
        log("[cem] WARNING: episode terminated inside the first solve's 25 steps for every env "
            "-- no second solve occurred naturally, trace is still valid")

    log(f"[cem] trace: {len(cb.trace)} (task,iter) log points, {len(cb.elites)} elite sets")

    env_low = np.asarray(mech.make_env().action_space.low, dtype=np.float32)
    env_high = np.asarray(mech.make_env().action_space.high, dtype=np.float32)

    oracle_rows = []
    n_elites = len(cb.elites)
    for ei, e in enumerate(cb.elites):
        ti = e["task_idx"]
        start_state = np.array(pairs["start_state"][ti], dtype=np.float32)
        goal_state_ti = np.array(pairs["goal_state"][ti], dtype=np.float32)
        seed_base = oracle_seed(seed, {0: 1, 9: 2, 19: 3}[e["step"]], ti, 0)
        T, best_dist, final_states, raw_actions = run_oracle_on_elites(
            mech, None, e["candidates"], action_scaler, env_low, env_high,
            start_state, goal_state_ti, seed_base, int(m.oracle_cap), int(m.oracle_trials),
            int(m.oracle_npop), int(m.oracle_niter), float(m.oracle_elite_frac))
        rho_our = per_task_spearman(e["our_cost"], T)
        rho_l2 = per_task_spearman(e["l2_cost"], T)
        mismatch_our = mean_abs_rank_mismatch(e["our_cost"], T)
        mismatch_l2 = mean_abs_rank_mismatch(e["l2_cost"], T)
        oracle_rows.append(dict(task_idx=ti, step=e["step"], iteration=e["step"] + 1,
                                our_cost=e["our_cost"].tolist(), l2_cost=e["l2_cost"].tolist(),
                                oracle_min_steps=T.tolist(), oracle_best_dist=best_dist.tolist(),
                                rho_our=rho_our, rho_l2=rho_l2,
                                mismatch_our=mismatch_our, mismatch_l2=mismatch_l2))
        log(f"[oracle {ei + 1}/{n_elites}] task={ti} iter={e['step'] + 1}: "
            f"rho_our={rho_our:.3f} rho_l2={rho_l2:.3f} mismatch_our={mismatch_our:.2f} "
            f"mismatch_l2={mismatch_l2:.2f} T_range=[{T.min():.0f},{T.max():.0f}] "
            f"elapsed={time.time() - t0:.0f}s")

    out_dir = OUT / "experiments" / "cem_trace_audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    trace_path = out_dir / f"trace_{m.protocol}_s{seed}_n{n}.npz"
    np.savez_compressed(trace_path, trace=np.array(cb.trace, dtype=object), allow_pickle=True)

    summary = dict(protocol=m.protocol, seed=seed, n=n, mpc=OmegaConf.to_container(m),
                   log_iters=[i + 1 for i in LOG_ITERS_0IDX], oracle_iters=[i + 1 for i in oracle_iters_0idx],
                   oracle_rows=oracle_rows)
    by_iter = {}
    for it in [i + 1 for i in oracle_iters_0idx]:
        rows = [r for r in oracle_rows if r["iteration"] == it]
        ro = np.array([r["rho_our"] for r in rows]); rl = np.array([r["rho_l2"] for r in rows])
        mo = np.array([r["mismatch_our"] for r in rows]); ml = np.array([r["mismatch_l2"] for r in rows])
        by_iter[it] = dict(n_tasks=len(rows),
                           rho_our_mean=float(np.nanmean(ro)), rho_l2_mean=float(np.nanmean(rl)),
                           mismatch_our_mean=float(np.nanmean(mo)), mismatch_l2_mean=float(np.nanmean(ml)))
        log(f"[summary iter={it}] n={len(rows)} rho_our={by_iter[it]['rho_our_mean']:.3f} "
            f"rho_l2={by_iter[it]['rho_l2_mean']:.3f} mismatch_our={by_iter[it]['mismatch_our_mean']:.2f} "
            f"mismatch_l2={by_iter[it]['mismatch_l2_mean']:.2f}")
    summary["by_iter"] = by_iter

    out_path = out_dir / f"summary_{m.protocol}_s{seed}_n{n}.json"
    out_path.write_text(json.dumps(summary, indent=1))
    log(f"wrote {out_path} and {trace_path} ({time.time() - t0:.0f}s total)")


if __name__ == "__main__":
    main()
