"""E10 -- Oracle-CEM: ground-truth reachability via a large CEM search against the REAL
PushT simulator (true pymunk physics), not any learned model.

E5 (e5_cost_hacking_diagnostic.py) already asks "does the learned CEM's predicted cost track
the outcome it actually produces" -- but the "ground truth" it correlates against is
`PushTMechanics.build_true_distance_oracle` (envs.py:262), which is just z-scored Euclidean
distance in privileged state space. It is NOT a reachability oracle: it doesn't know what's
actually achievable by ANY controller in H actions, only how geometrically far two states are.
This script supplies the real thing, in the sense docs/viability-value.md:869-877 proposes:
"run a privileged simulator planner with access to the true state and true dynamics ... a much
larger action search ... from the true state."

Algorithm, per (start_state, goal_state) pair (same pairs E4/E5/E7 sample -- identical seed,
same dataset, same goal_offset -- so results here join with theirs by pair index):
  1. Replay baseline: step the REAL recorded expert actions (the h5 "action" column, the exact
     rows between this pair's start and start+horizon) through the sim from start_state. Since
     `goal_state` is BY CONSTRUCTION the state that trajectory reaches at start+goal_offset,
     this replay should succeed at or near step `goal_offset_steps` -- a near-free sanity check
     that state read/replay/eval_state wiring is correct, logged alongside the search result.
  2. CEM search: `nrestarts` independent restarts, each `niter` iterations of `npop` candidate
     H-length action sequences, real physics rollout per candidate (env.reset to start_state,
     step through the candidate sequence, using the env's own `eval_state`/`terminated` check
     and its `reward = -distance` as the fitness signal for candidates that don't reach goal).
     Restart 0 seeds its CEM mean at the recorded expert actions (a known near-solution, per
     the replay above) with a tight init sigma, so search starts from a working plan instead of
     from scratch; later restarts start from zero-mean/full sigma to look for a genuinely
     shorter alternative the expert didn't take.
  3. J^oracle(s_i, g) = min, over every candidate ever evaluated across all restarts, of the
     number of actions to first satisfy env.terminated -- None (unbounded/unreached) if no
     candidate ever succeeded.

Cost of a NON-successful candidate = horizon + (best `-reward` distance it ever reached during
its rollout) -- always ranks worse than every successful candidate (whose cost is its own step
count, <= horizon), while still ordering failures by how close they got, so CEM has a gradient
to follow even before any candidate first succeeds.

No learned model/GPU is used anywhere in this script -- pure CPU pymunk stepping -- so it does
NOT use plan_config/solver/world from config/eval/pusht.yaml, only eval/dataset/seed (for
identical pair sampling to E4/E5/E7).

Cost note: with the defaults below (horizon=goal_offset_steps, npop=512, niter=25, nrestarts=3)
a single pair is roughly 38k full-physics rollouts. On a quick local timing check this ran at
~Xms/env.step() -- expect single-digit minutes per pair on one core; ALWAYS time a smoke test
before committing to a large sweep. There is no in-process vectorization (pymunk doesn't
parallelize across candidates within one Python process without real subprocess-level
parallelism); get parallelism instead by sharding PAIRS across OS processes/cores -- see
ORACLE_SHARD/ORACLE_N_SHARDS below, matching this repo's existing direct-SSH ada_*.py sweep
convention (N_PARALLEL processes, each single-threaded, run inside one interactive allocation).

Run (single process, all sampled pairs):
    python scripts/investigations/quasimetric/e10_oracle_cem.py

Smoke test (fast, tiny budget -- always run this first to calibrate wall-clock/pair):
    ORACLE_NPOP=32 ORACLE_NITER=3 ORACLE_NRESTARTS=1 \
        python scripts/investigations/quasimetric/e10_oracle_cem.py eval.num_eval=3

Sharded parallel run on Ada (inside an existing `sinteractive` allocation, one process per
core, e.g. 16 shards on a -c16 node):
    for i in $(seq 0 15); do
        ORACLE_SHARD=$i ORACLE_N_SHARDS=16 \
            python scripts/investigations/quasimetric/e10_oracle_cem.py \
            > outputs/quasimetric/e10_shard${i}.log 2>&1 &
    done
    wait

Resumable: each shard writes one JSON line per completed pair to its own .jsonl file,
flushed immediately -- rerunning the same shard skips pairs already present in that file.
"""

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import json
import sys
import time
from pathlib import Path

import h5py
import hydra
import numpy as np
import stable_worldmodel as swm
from omegaconf import DictConfig

ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.envs import ENV_MECHANICS
from common.log_util import log
from e4_live_rollout import get_episodes_length

REPLAY_INIT_SIGMA = 0.25
EXPLORE_INIT_SIGMA = 1.0


def _read_rows(h5_col, rows):
    """Scattered-row read via h5py fancy indexing, which needs sorted indices -- sort, read,
    unsort (same idiom as e5_cost_hacking_diagnostic.py's goal_states read)."""
    rows = np.asarray(rows).astype(np.int64)
    order = np.argsort(rows)
    out = np.empty((len(rows),) + h5_col.shape[1:], dtype=h5_col.dtype)
    out[order] = h5_col[rows[order]]
    return out


def _read_action_blocks(action_col, start_rows, horizon):
    """Per-pair CONTIGUOUS action slice [start_row, start_row+horizon) -- always safe as a
    direct h5py slice, unlike the scattered single-row reads above."""
    return np.stack([
        np.asarray(action_col[s: s + horizon]) for s in start_rows.astype(np.int64)
    ])


def rollout(env, reset_opts, seed, actions):
    """Step `actions` (T,2) through the real sim from reset_opts's start state. Returns
    (success_step or None, min_dist_ever_seen) -- success_step is 1-indexed action count."""
    env.reset(seed=seed, options=reset_opts)
    min_dist = np.inf
    for t in range(len(actions)):
        obs, reward, terminated, truncated, info = env.step(
            np.asarray(actions[t], dtype=np.float32))
        dist = -reward
        if dist < min_dist:
            min_dist = dist
        if terminated:
            return t + 1, float(min_dist)
    return None, float(min_dist)


def oracle_cem_search(env, reset_opts, seed, horizon, npop, niter, nrestarts, topk_frac,
                       min_sigma, rng, replay_actions=None):
    """Ground-truth min-actions-to-goal via CEM directly against real physics. Returns
    dict(steps, best_dist, n_rollouts, secs). `steps` is None if nothing ever succeeded."""
    topk = max(2, int(round(npop * topk_frac)))
    best_steps, best_dist, n_rollouts = None, np.inf, 0
    t0 = time.time()

    for restart in range(nrestarts):
        if restart == 0 and replay_actions is not None:
            mu = np.clip(replay_actions[:horizon].astype(np.float64), -1.0, 1.0)
            sigma = np.full((horizon, 2), REPLAY_INIT_SIGMA)
        else:
            mu = np.zeros((horizon, 2))
            sigma = np.full((horizon, 2), EXPLORE_INIT_SIGMA)

        for _ in range(niter):
            noise = rng.standard_normal((npop, horizon, 2))
            candidates = np.clip(mu[None] + sigma[None] * noise, -1.0, 1.0)

            costs = np.empty(npop)
            for i in range(npop):
                success_step, min_dist = rollout(env, reset_opts, seed, candidates[i])
                n_rollouts += 1
                if min_dist < best_dist:
                    best_dist = min_dist
                if success_step is not None:
                    costs[i] = success_step
                    if best_steps is None or success_step < best_steps:
                        best_steps = success_step
                else:
                    costs[i] = horizon + min_dist

            elite_idx = np.argsort(costs)[:topk]
            elites = candidates[elite_idx]
            mu = elites.mean(axis=0)
            sigma = np.maximum(elites.std(axis=0), min_sigma)

    return dict(steps=best_steps, best_dist=float(best_dist), n_rollouts=n_rollouts,
                secs=time.time() - t0)


@hydra.main(version_base=None, config_path=str(ROOT / "config" / "eval"), config_name="pusht")
def main(cfg: DictConfig):
    t0 = time.time()
    mech = ENV_MECHANICS["pusht"]
    res_dir = ROOT / "outputs" / "quasimetric"
    res_dir.mkdir(parents=True, exist_ok=True)

    horizon = int(os.environ.get("ORACLE_HORIZON", cfg.eval.goal_offset_steps))
    npop = int(os.environ.get("ORACLE_NPOP", 512))
    niter = int(os.environ.get("ORACLE_NITER", 25))
    nrestarts = int(os.environ.get("ORACLE_NRESTARTS", 3))
    topk_frac = float(os.environ.get("ORACLE_TOPK_FRAC", 0.1))
    min_sigma = float(os.environ.get("ORACLE_MIN_SIGMA", 0.05))
    shard = int(os.environ.get("ORACLE_SHARD", 0))
    n_shards = int(os.environ.get("ORACLE_N_SHARDS", 1))

    log(f"=== E10 Oracle-CEM (num_eval={cfg.eval.num_eval}, horizon={horizon}, npop={npop}, "
        f"niter={niter}, nrestarts={nrestarts}, shard={shard}/{n_shards}) ===")

    h5_path = str(mech.h5_path(ROOT))
    dataset = swm.data.HDF5Dataset(path=h5_path, keys_to_cache=list(cfg.dataset.keys_to_cache))
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"

    # -- identical eval-pair sampling to e4_live_rollout.py/e5_cost_hacking_diagnostic.py
    # (same seed/logic) so pairs here line up by index with theirs for later comparison.
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
    eval_episodes = np.asarray(dataset.get_row_data(random_episode_indices)[col_name]).astype(np.int64)
    eval_start_idx = np.asarray(dataset.get_row_data(random_episode_indices)["step_idx"]).astype(np.int64)

    with h5py.File(h5_path, "r", swmr=True) as f5:
        ep_offset = f5["ep_offset"][:]
        state_col = f5["state"]
        action_col = f5["action"]
        start_rows = ep_offset[eval_episodes] + eval_start_idx
        goal_rows = start_rows + cfg.eval.goal_offset_steps
        start_states = _read_rows(state_col, start_rows)
        goal_states = _read_rows(state_col, goal_rows)
        recorded_actions = _read_action_blocks(action_col, start_rows, horizon)
        # same broad-sample convention as graph_gate.py/E3-v3/E5, for a directly comparable
        # cheap-proxy distance alongside the ground-truth J^oracle computed below
        true_dist_oracle = mech.build_true_distance_oracle(np.asarray(state_col[::37]))

    euclid_dist = np.diagonal(true_dist_oracle(start_states, goal_states))

    pair_ids = np.arange(cfg.eval.num_eval)[shard::n_shards]
    out_path = res_dir / f"e10_oracle_cem_pusht_n{cfg.eval.num_eval}_shard{shard}of{n_shards}.jsonl"
    done_ids = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            if line.strip():
                done_ids.add(json.loads(line)["pair_id"])
    pair_ids = [p for p in pair_ids if int(p) not in done_ids]
    log(f"shard has {len(pair_ids)} pairs left to run ({len(done_ids)} already done in {out_path.name})")

    env = mech.make_env()
    times = []
    with open(out_path, "a") as f_out:
        for n_done, pair_id in enumerate(pair_ids):
            pair_id = int(pair_id)
            reset_opts = mech.reset_options(start_states[pair_id], goal_states[pair_id])
            episode_seed = int(cfg.seed) * 100_003 + pair_id  # fixed per pair: keeps any
            # env variation-space randomness (shape/scale/color) identical across every
            # reset() in this pair's search -- required for a valid CEM comparison across
            # candidates, and matches actor_rollout_utils.py's fixed-ep_seed convention.

            t_pair = time.time()
            replay_step, replay_dist = rollout(
                env, reset_opts, episode_seed, recorded_actions[pair_id])

            rng = np.random.default_rng(episode_seed + 1)
            result = oracle_cem_search(
                env, reset_opts, episode_seed, horizon, npop, niter, nrestarts, topk_frac,
                min_sigma, rng, replay_actions=recorded_actions[pair_id])
            dt_pair = time.time() - t_pair
            times.append(dt_pair)

            row = dict(
                pair_id=pair_id, episode=int(eval_episodes[pair_id]),
                start_idx=int(eval_start_idx[pair_id]), horizon=horizon,
                euclid_proxy_dist=float(euclid_dist[pair_id]),
                replay_success_step=replay_step, replay_min_dist=replay_dist,
                oracle_steps=result["steps"], oracle_best_dist=result["best_dist"],
                n_rollouts=result["n_rollouts"], secs=dt_pair,
            )
            f_out.write(json.dumps(row) + "\n")
            f_out.flush()

            eta = np.mean(times) * (len(pair_ids) - n_done - 1)
            log(f"[pair {pair_id}] replay_step={replay_step} oracle_steps={result['steps']} "
                f"euclid_proxy={euclid_dist[pair_id]:.2f} ({dt_pair:.1f}s, "
                f"{n_done + 1}/{len(pair_ids)} done, ETA {eta / 60:.1f}min)")

    env.close()
    log(f"=== shard {shard}/{n_shards} done: {len(pair_ids)} pairs, wrote {out_path} "
        f"({time.time() - t0:.1f}s total) ===")


if __name__ == "__main__":
    main()
