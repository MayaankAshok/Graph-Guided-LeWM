"""Experiment 1 / Experiment 2 Run A: audit whether LeWM latent proximity disagrees with
finite-budget recovery, and save enough per-candidate state to decompose WHY.

For each same-trajectory Push-T start/goal problem, generate one fixed candidate pool, score
its predicted endpoints with LeWM terminal L2, execute the raw candidates in the simulator,
and estimate the minimum recovery time from every true endpoint with privileged real-physics
CEM. The output is a pilot artifact intended to decide whether critic training is warranted;
it does not train a value model.

Experiment 2 (docs/viability-proposal/experiment2-plan.tex) reuses this exact pipeline and
adds the artifact-complete save: for candidate i it stores

    z_pred_i  = predicted terminal embedding            (LeWM rollout)
    z_true_i  = E(o_i^true), the encoded executed endpoint
    c_pred_i  = ||z_pred_i - z_g||^2                    LeWM's deployed terminal cost
    c_true_i  = ||z_true_i - z_g||^2                    same metric, correct endpoint
    e_model_i = ||z_pred_i - z_true_i||                 latent endpoint (model) error
    T_i       = oracle minimum recovery time, right-censored at --oracle-max-steps
                (stored as max_steps + 1, with a separate `censored` mask)

The candidate population and oracle output are computed ONCE per task and shared by both
costs, so a difference between c_pred and c_true results is attributable purely to endpoint
prediction error. Analysis (Run B), figures, and the stronger-oracle subset selection live
in scripts/viability_metric_vs_dynamics.py; the stronger-oracle rerun (Run C) is
scripts/viability_oracle_recheck.py.

Run A (Ada, inside an interactive allocation with PUSHT_H5_PATH set):

    python scripts/viability_inversion_audit.py --n-problems 50 --n-candidates 32 \
        --goal-offsets 25 50 --oracle-max-steps 100 --out outputs/pusht/experiments/viability_exp2 \
        > outputs/pusht/experiments/viability_exp2/run_a.log 2>&1

The run checkpoints after every task (raw_offset{N}.partial.npz) and resumes from that file
if restarted with the same config; the candidate RNG is replayed so a resumed run is
bit-identical to an uninterrupted one.
"""

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import h5py
import numpy as np
from scipy import stats as sps

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.envs import ENV_MECHANICS
from common.lewm_loader import load_lewm
from common.log_util import log
from planning_cost_gate import (DEV, HISTORY, HORIZON, SKIP, latent_rollout, make_candidates,
                                 make_encode_frame, sample_problems)


def sample_problems_from_pool(h5_path, pool_path, n):
    """Same output schema as planning_cost_gate.sample_problems, but takes the first `n`
    (start_row, goal_row) pairs from an existing task200u pool file (gas_mpc_make_tasks.py's
    output, the exact pool gas_mpc_eval.py scores) instead of drawing a fresh random sample.
    Lets this audit's Spearman-vs-oracle numbers be checked against the tasks actually used
    for the paper's headline results, not a separate ad hoc draw. Returns (probs, goal_offset)
    -- goal_offset is read from the pool file itself, not assumed."""
    pool = json.loads(Path(pool_path).read_text())
    if pool["pairing"] != "same_episode":
        raise ValueError(f"{pool_path} is not a same-episode pool ({pool['pairing']!r})")
    if n > pool["n"]:
        raise ValueError(f"{pool_path} only has {pool['n']} tasks, requested {n}")
    goal_offset = int(pool["offset"])
    start_row = np.asarray(pool["start_row"][:n], dtype=np.int64)
    goal_row = np.asarray(pool["goal_row"][:n], dtype=np.int64)
    start_state = np.asarray(pool["start_state"][:n], dtype=np.float32)
    goal_state = np.asarray(pool["goal_state"][:n], dtype=np.float32)

    with h5py.File(h5_path, "r", swmr=True, rdcc_nbytes=256 * 1024 * 1024) as f:
        _a = f["action"][:].astype(np.float64)
        _a = _a[~np.isnan(_a).any(axis=1)]
        act_mean, act_std = _a.mean(0).astype(np.float32), _a.std(0).astype(np.float32)
        del _a

        hist_rows = np.stack([start_row - k * SKIP for k in range(HISTORY - 1, -1, -1)], axis=1)
        want = np.unique(np.concatenate([hist_rows.ravel(), goal_row]))
        pix = f["pixels"][want]
        row_to_i = {int(r): i for i, r in enumerate(want)}

        past_rows = np.stack([np.arange(r - (HISTORY - 1) * SKIP, r) for r in start_row])
        fut_rows = np.stack([np.arange(r, r + HORIZON * SKIP) for r in start_row])
        act_all = f["action"]
        past_act = np.stack([act_all[r[0]:r[-1] + 1] for r in past_rows]).astype(np.float32)
        real_act = np.stack([act_all[r[0]:r[-1] + 1] for r in fut_rows]).astype(np.float32)

    hist_pix = np.stack([[pix[row_to_i[int(r)]] for r in row] for row in hist_rows])
    goal_pix = np.stack([pix[row_to_i[int(r)]] for r in goal_row])
    probs = dict(start_row=start_row, goal_row=goal_row,
                 start_state=start_state, goal_state=goal_state,
                 hist_pixels=hist_pix, goal_pixels=goal_pix,
                 past_action=past_act, real_action=real_act,
                 act_mean=act_mean, act_std=act_std)
    return probs, goal_offset

_ORACLE_ENV = None
EMB_DIM = 192


def _init_oracle_worker():
    global _ORACLE_ENV
    _ORACLE_ENV = ENV_MECHANICS["pusht"].make_env()


def rollout(env, reset_opts, seed, actions):
    """Return first success step (zero if already successful) and best real distance."""
    obs, _ = env.reset(seed=seed, options=reset_opts)
    already_solved, initial_dist = env.eval_state(env.goal_state, obs["state"])
    if already_solved:
        return 0, float(initial_dist)
    best_dist = np.inf
    for step, action in enumerate(actions, start=1):
        obs, reward, terminated, truncated, info = env.step(np.asarray(action, dtype=np.float32))
        best_dist = min(best_dist, float(-reward))
        if terminated:
            return step, best_dist
    return None, best_dist


def oracle_trial(env, reset_opts, seed, max_steps, npop, niter, elite_frac, rng):
    """One privileged CEM attempt; returns its shortest found recovery time.

    The environment state and goal are visible to the CEM through the simulator reward and
    termination predicate only. No LeWM embedding or latent cost enters the oracle.
    """
    topk = max(2, int(round(npop * elite_frac)))
    mu = np.zeros((max_steps, 2), dtype=np.float64)
    sigma = np.ones((max_steps, 2), dtype=np.float64)
    best_steps = None
    best_dist = np.inf
    for _ in range(niter):
        candidates = np.clip(mu + sigma * rng.standard_normal((npop, max_steps, 2)), -1.0, 1.0)
        costs = np.empty(npop, dtype=np.float64)
        for j, candidate in enumerate(candidates):
            success_step, distance = rollout(env, reset_opts, seed, candidate)
            best_dist = min(best_dist, distance)
            if success_step is None:
                costs[j] = max_steps + distance
            else:
                best_steps = success_step if best_steps is None else min(best_steps, success_step)
                costs[j] = success_step
        elite = candidates[np.argsort(costs)[:topk]]
        mu = elite.mean(axis=0)
        sigma = np.maximum(elite.std(axis=0), 0.05)
    return best_steps, best_dist


def oracle_endpoint_job(job):
    """Pickle-friendly endpoint query so a CPU worker owns its own non-thread-safe simulator.

    Returns (min_steps, best_dist); min_steps == max_steps + 1 means right-censored (no plan
    found within the cap), NOT proof that recovery needs more than max_steps.
    """
    state, goal, seed_base, max_steps, trials, npop, niter, elite_frac = job
    steps, distances = [], []
    opts = ENV_MECHANICS["pusht"].reset_options(state, goal)
    for trial in range(trials):
        seed = int(seed_base + trial)
        found_steps, distance = oracle_trial(_ORACLE_ENV, opts, seed, max_steps, npop, niter, elite_frac,
                                             np.random.default_rng(seed + 1))
        if found_steps is not None:
            steps.append(found_steps)
        distances.append(distance)
    return int(min(steps)) if steps else int(max_steps + 1), float(min(distances))


def oracle_seed(base_seed, offset, task, cand):
    return int(base_seed * 3_000_017 + offset * 20_011 + task * 211 + cand * 7)


def inversion_summary(cost, recovery_steps, drift, quantiles=4):
    """Pairwise lower-cost/slower-recovery inversions, including a low-drift control.

    `cost` is any (n_tasks, n_candidates) terminal cost (c_pred or c_true); the recovery
    times and drift are shared, so the two costs' summaries are directly paired.
    """
    n_tasks, n_candidates = cost.shape
    per_task, severity, low_error, spearman = [], [], [], []
    q = np.quantile(drift.ravel(), 1.0 / quantiles)
    for i in range(n_tasks):
        left = cost[i][:, None] < cost[i][None, :]
        worse = recovery_steps[i][:, None] > recovery_steps[i][None, :]
        pairs = left & worse
        denom = np.maximum(left.sum(), 1)
        per_task.append(float(pairs.sum() / denom))
        if pairs.any():
            severity.extend((recovery_steps[i][:, None] - recovery_steps[i][None, :])[pairs].tolist())
        low = (drift[i] <= q)
        low_pairs = pairs & low[:, None] & low[None, :]
        low_denom = max((left & low[:, None] & low[None, :]).sum(), 1)
        low_error.append(float(low_pairs.sum() / low_denom))
        if np.ptp(recovery_steps[i]) > 0:
            spearman.append(float(sps.spearmanr(cost[i], recovery_steps[i]).statistic))
    return dict(
        inversion_rate_mean=float(np.mean(per_task)),
        inversion_rate_per_task=per_task,
        inversion_rate_lowest_drift_quartile=float(np.mean(low_error)),
        mean_recovery_step_gap=float(np.mean(severity)) if severity else 0.0,
        n_inverted_pairs=int(sum(int(x > 0) for x in per_task)),
        drift_low_quartile=float(q),
        recovery_steps_spearman_mean=float(np.nanmean(spearman)) if spearman else float("nan"),
        n_informative_tasks=len(spearman),
    )


def _partial_path(out_dir, offset):
    return out_dir / f"raw_offset{offset}.partial.npz"


def run(args):
    global _ORACLE_ENV
    mech = ENV_MECHANICS["pusht"]
    root = Path(args.root).resolve()
    h5_path, ckpt_dir = mech.h5_path(root), mech.ckpt_dir(root)
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    log(f"[setup] h5={h5_path} checkpoint={ckpt_dir} device={DEV}")
    log(f"[setup] problems={args.n_problems} candidates={args.n_candidates} "
        f"offsets={args.goal_offsets} oracle_max_steps={args.oracle_max_steps} "
        f"oracle={args.oracle_trials}x{args.oracle_npop}x{args.oracle_niter}")

    model = load_lewm(Path(ckpt_dir), device=DEV)
    encode = make_encode_frame(model)
    env = mech.make_env()
    low, high = np.asarray(env.action_space.low), np.asarray(env.action_space.high)
    result = dict(config=vars(args), offsets={})

    pool = (ProcessPoolExecutor(max_workers=args.oracle_workers, initializer=_init_oracle_worker,
                                mp_context=get_context("spawn"))
            if args.oracle_workers > 1 else None)
    if pool is None:
        _ORACLE_ENV = env
    K, cap = args.n_candidates, args.oracle_max_steps
    for offset in args.goal_offsets:
        if args.task_pool:
            pair_path = Path(root) / "outputs" / "pusht" / "pairs" / f"pairs_same_episode_off{offset}_{args.task_pool}.json"
            probs, pool_offset = sample_problems_from_pool(h5_path, pair_path, args.n_problems)
            assert pool_offset == offset, f"{pair_path} has offset={pool_offset}, expected {offset}"
            log(f"[offset={offset}] using {args.n_problems} tasks from {pair_path.name} "
                f"(pool n={json.loads(pair_path.read_text())['n']})")
        else:
            probs = sample_problems(h5_path, mech, offset, args.n_problems, args.seed + offset)
        rng = np.random.default_rng(args.seed + 1009 * offset)
        n = len(probs["start_row"])
        arrays = dict(
            c_pred=np.empty((n, K), np.float32), c_true=np.empty((n, K), np.float32),
            e_model=np.empty((n, K), np.float32), endpoint_dist=np.empty((n, K), np.float32),
            oracle_min_steps=np.empty((n, K), np.int16), oracle_best_dist=np.empty((n, K), np.float32),
            z_pred=np.empty((n, K, EMB_DIM), np.float32), z_true=np.empty((n, K, EMB_DIM), np.float32),
            z_goal=np.empty((n, EMB_DIM), np.float32),
            candidate_actions=np.empty((n, K, HORIZON * 5, mech.action_dim), np.float32),
            final_states=np.empty((n, K, mech.state_dim), np.float32),
        )
        start = 0
        partial = _partial_path(out_dir, offset)
        if args.resume and partial.exists():
            with np.load(partial) as saved:  # close the handle so the file can be unlinked later
                if int(saved["n_done"]) > 0 and saved["c_pred"].shape == (n, K):
                    start = int(saved["n_done"])
                    for key in arrays:
                        arrays[key][:start] = saved[key][:start]
            for i in range(start):  # replay the candidate RNG so the stream stays identical
                make_candidates(probs["real_action"][i], low, high, K, rng)
            if start:
                log(f"[offset={offset}] resuming from task {start}/{n} ({partial.name})")
        t0 = time.time()

        for i in range(start, n):
            candidates = make_candidates(probs["real_action"][i], low, high, K, rng)
            arrays["candidate_actions"][i] = candidates
            z_hist = encode(probs["hist_pixels"][i])
            z_goal = encode(probs["goal_pixels"][i:i + 1])[0]
            mean, std = probs["act_mean"], probs["act_std"]
            past = ((probs["past_action"][i] - mean) / std).reshape(2, 10)
            norm_candidates = ((candidates - mean) / std).reshape(K, HORIZON, 10)
            history = np.concatenate([np.repeat(past[None], K, axis=0), norm_candidates[:, :1]], axis=1)
            predicted = latent_rollout(model, np.repeat(z_hist[None], K, axis=0), history,
                                       norm_candidates[:, 1:])[:, -1]

            final_state, final_pixels = [], []
            for k, actions in enumerate(candidates):
                seed = int(args.seed * 1_000_003 + offset * 10_007 + i * 101 + k)
                env.reset(seed=seed, options=mech.reset_options(probs["start_state"][i], probs["goal_state"][i]))
                obs = None
                for action in actions:
                    obs, reward, terminated, truncated, info = env.step(action)
                    if terminated:
                        break
                final_state.append(np.asarray(obs["state"], dtype=np.float32))
                final_pixels.append(env.render())
            final_state, final_pixels = np.stack(final_state), np.stack(final_pixels)
            z_true = encode(final_pixels)

            arrays["z_goal"][i] = z_goal
            arrays["z_pred"][i], arrays["z_true"][i] = predicted, z_true
            arrays["c_pred"][i] = ((predicted - z_goal) ** 2).sum(axis=1)
            arrays["c_true"][i] = ((z_true - z_goal) ** 2).sum(axis=1)
            arrays["e_model"][i] = np.linalg.norm(predicted - z_true, axis=1)
            arrays["final_states"][i] = final_state
            arrays["endpoint_dist"][i] = np.linalg.norm(final_state - probs["goal_state"][i], axis=1)

            jobs = [
                (state, probs["goal_state"][i], oracle_seed(args.seed, offset, i, k),
                 cap, args.oracle_trials, args.oracle_npop, args.oracle_niter, args.oracle_elite_frac)
                for k, state in enumerate(final_state)
            ]
            values = pool.map(oracle_endpoint_job, jobs) if pool else map(oracle_endpoint_job, jobs)
            for k, (steps, best) in enumerate(values):
                arrays["oracle_min_steps"][i, k], arrays["oracle_best_dist"][i, k] = steps, best
            np.savez_compressed(partial, n_done=i + 1, **arrays)
            log(f"[offset={offset}] problem {i + 1}/{n} elapsed={time.time() - t0:.1f}s "
                f"censored={(arrays['oracle_min_steps'][i] > cap).mean():.2f}")

        T, drift = arrays["oracle_min_steps"], arrays["e_model"]
        censored = T > cap
        natural_budget = max(0, offset - HORIZON * 5)
        viability = T <= natural_budget
        summary = dict(
            c_pred=inversion_summary(arrays["c_pred"], T, drift),
            c_true=inversion_summary(arrays["c_true"], T, drift),
            natural_recovery_budget=natural_budget,
            viable_candidate_fraction=float(viability.mean()),
            median_oracle_min_steps=float(np.median(T)),
            unreached_fraction=float(censored.mean()),
            # candidate slot 0 is the dataset's own continuation, viable by construction
            # whenever offset >= candidate horizon: an oracle that misses it is too weak
            slot0_recovered_within_natural_budget=float((T[:, 0] <= natural_budget).mean()),
            median_model_error=float(np.median(drift)),
            median_endpoint_state_distance=float(np.median(arrays["endpoint_dist"])),
        )
        result["offsets"][str(offset)] = summary
        np.savez_compressed(out_dir / f"raw_offset{offset}.npz", **arrays,
                            l2=arrays["c_pred"], drift=drift,  # Experiment-1 names, kept for old readers
                            censored=censored, viability=viability, oracle_cap=cap,
                            start_row=probs["start_row"], goal_row=probs["goal_row"],
                            start_state=probs["start_state"], goal_state=probs["goal_state"],
                            act_mean=probs["act_mean"], act_std=probs["act_std"])
        partial.unlink(missing_ok=True)
        log(f"[offset={offset}] c_pred spearman={summary['c_pred']['recovery_steps_spearman_mean']:.3f} "
            f"inv={summary['c_pred']['inversion_rate_mean']:.3f} | "
            f"c_true spearman={summary['c_true']['recovery_steps_spearman_mean']:.3f} "
            f"inv={summary['c_true']['inversion_rate_mean']:.3f} | censored={censored.mean():.3f} "
            f"slot0_ok={summary['slot0_recovered_within_natural_budget']:.2f}")

    if pool:
        pool.shutdown()
    env.close()
    (out_dir / "results.json").write_text(json.dumps(result, indent=2))
    log(f"[done] wrote {out_dir / 'results.json'}")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=str(Path(__file__).resolve().parent.parent))
    p.add_argument("--out", default=None)
    p.add_argument("--n-problems", type=int, default=50)
    p.add_argument("--n-candidates", type=int, default=32)
    p.add_argument("--goal-offsets", type=int, nargs="+", default=[25, 50])
    p.add_argument("--oracle-max-steps", type=int, default=100)
    p.add_argument("--oracle-trials", type=int, default=2)
    p.add_argument("--oracle-npop", type=int, default=64)
    p.add_argument("--oracle-niter", type=int, default=5)
    p.add_argument("--oracle-elite-frac", type=float, default=0.15)
    p.add_argument("--oracle-workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--task-pool", default=None,
                   help="e.g. task200u: take the first --n-problems (start,goal) pairs from "
                        "outputs/pusht/pairs/pairs_same_episode_off{offset}_<task-pool>.json "
                        "instead of drawing a fresh random sample, so this audit's numbers can "
                        "be checked against the exact tasks gas_mpc_eval.py scores")
    p.add_argument("--no-resume", dest="resume", action="store_false",
                   help="ignore an existing raw_offset*.partial.npz and start the offset over")
    args = p.parse_args()
    args.out = args.out or str(Path(args.root) / "outputs" / "pusht" / "experiments" / "viability_inversion_audit")
    run(args)


if __name__ == "__main__":
    main()
