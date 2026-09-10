"""Per-environment mechanics -- the one part of the pipeline that genuinely cannot be pure
config data, because it's control flow (how do you reset this env to a specific state, how
do you read success off its step() return, how do you collect a noisy rollout), not values.
Everything else (which tier, how many steps, k) is a Hydra config value; this is the code a
new environment actually has to write.

Each method's body is a direct lift of logic that already existed in the tworoom_*.py/
pusht_*.py pairs -- nothing new is being invented, it's moved behind one shared interface so
graph_gate.py/actor_rollout_utils.py/rollout_collector.py/etc. can call
ENV_MECHANICS[cfg.env].foo(...) once instead of each new environment needing its own copy of
the surrounding function.
"""

import time
from pathlib import Path

import numpy as np
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

from common.log_util import log


class EnvMechanics:
    name: str
    state_dim: int
    action_dim: int
    max_episode_steps: int
    h5_path_env_var: str
    h5_path_default: str
    ckpt_repo: str

    def h5_path(self, root):
        import os
        return Path(os.environ.get(self.h5_path_env_var, str(root / self.h5_path_default)))

    @property
    def ckpt_dir_env_var(self):
        # LEWM_CKPT_DIR for tworoom (matches common.lewm_loader's own built-in default var
        # name), PUSHT_CKPT_DIR for pusht (matches the existing pusht_b0_graph_gate.py
        # convention) -- derived from name rather than a separate class attr since both
        # already follow this exact pattern.
        return "LEWM_CKPT_DIR" if self.name == "tworoom" else f"{self.name.upper()}_CKPT_DIR"

    def ckpt_dir(self, root):
        import os
        default = root / "data" / "checkpoints" / f"models--{self.ckpt_repo.replace('/', '--')}"
        return Path(os.environ.get(self.ckpt_dir_env_var, str(default)))

    def read_state_slice(self, f, s, L):
        if hasattr(self, "state_h5_key") and self.state_h5_key in f:
            return f[self.state_h5_key][s: s + L]
        if "qpos" in f and "qvel" in f:
            return np.concatenate([f["qpos"][s: s + L], f["qvel"][s: s + L]], axis=-1).astype(np.float32)
        if "state" in f:
            return f["state"][s: s + L]
        raise KeyError(f"Neither {getattr(self, 'state_h5_key', None)} nor (qpos, qvel) found in H5")

    def read_state_rows(self, f, rows):
        if hasattr(self, "state_h5_key") and self.state_h5_key in f:
            return f[self.state_h5_key][rows].astype(np.float32)
        if "qpos" in f and "qvel" in f:
            return np.concatenate([f["qpos"][rows], f["qvel"][rows]], axis=-1).astype(np.float32)
        if "state" in f:
            return f["state"][rows].astype(np.float32)
        raise KeyError(f"Neither {getattr(self, 'state_h5_key', None)} nor (qpos, qvel) found in H5")

    def build_true_distance_oracle(self, reference_state):
        """reference_state: the full landmark state/proprio array (only used by Push-T, to
        fit z-scoring stats; Two-Room's oracle is a fixed wall geometry, independent of the
        data). Returns closure(source_state_array, target_state_array) -> (S,T) dist matrix."""
        raise NotImplementedError

    def make_env(self):
        """A single (non-vectorized) live env instance, render_mode='rgb_array'."""
        raise NotImplementedError

    def reset_options(self, start_state, goal_state):
        """The env.reset(options=...) dict for forcing a specific (start, goal)."""
        raise NotImplementedError

    def step_result(self, obs, reward, terminated, truncated, info):
        """Returns (success: bool, final_dist: float) from one env.step() return."""
        raise NotImplementedError

    def collect_noisy_rollout(self, n_episodes, seed, max_steps=None, **policy_kwargs):
        """Roll out this environment's noisy/weak collection policy for n_episodes,
        recording every real env step. Returns dict(pixels, state, action, ep_idx, step_idx).
        policy_kwargs are passed straight to the environment's own policy constructor (the
        parameter names genuinely differ per environment -- action_noise/action_repeat_prob
        vs dist_constraint -- forcing a common naming scheme here would be artificial)."""
        raise NotImplementedError


# ============================================================
# Two-Room
# ============================================================

class TwoRoomMechanics(EnvMechanics):
    name = "tworoom"
    state_dim = 2
    action_dim = 2
    max_episode_steps = 150
    h5_path_env_var = "LEWM_H5_PATH"
    h5_path_default = "data/hf_dl/tworoom_extracted/tworoom.h5"
    ckpt_repo = "quentinll/lewm-tworooms"
    state_h5_key = "proprio"  # the h5 column holding this env's true-distance-oracle input

    # verified geometry (Two-Room's wall/door config is bit-identical across the whole
    # dataset -- read off empirically from real agent traces, not env source defaults)
    WALL_X = (100.0, 124.0)
    DOOR_Y = (33.25, 64.75)
    ROOM_LO, ROOM_HI = 14.0, 209.0
    GRID_RES = 1.0

    def build_true_distance_oracle(self, reference_state=None):
        xs = np.arange(self.ROOM_LO, self.ROOM_HI, self.GRID_RES)
        ys = np.arange(self.ROOM_LO, self.ROOM_HI, self.GRID_RES)
        nx, ny = len(xs), len(ys)

        def is_free(x, y):
            in_wall_band = (x >= self.WALL_X[0]) & (x <= self.WALL_X[1])
            in_door = (y >= self.DOOR_Y[0]) & (y <= self.DOOR_Y[1])
            return ~(in_wall_band & ~in_door)

        XX, YY = np.meshgrid(xs, ys, indexing="ij")
        free = is_free(XX, YY)
        log(f"[true-dist] grid {nx}x{ny}, free cells {free.sum()}/{free.size}")

        node_id = -np.ones((nx, ny), dtype=np.int64)
        node_id[free] = np.arange(free.sum())
        n_nodes = free.sum()

        rows, cols, weights = [], [], []
        neighbors = [(1, 0), (0, 1), (1, 1), (1, -1)]
        for dx, dy in neighbors:
            i0, i1 = max(0, -dx), nx - max(0, dx)
            j0, j1 = max(0, -dy), ny - max(0, dy)
            a = node_id[i0:i1, j0:j1]
            b = node_id[i0 + dx: i1 + dx, j0 + dy: j1 + dy]
            both_free = (a >= 0) & (b >= 0)
            w = self.GRID_RES * np.hypot(dx, dy)
            rows.append(a[both_free]); cols.append(b[both_free]); weights.append(np.full(both_free.sum(), w))
            rows.append(b[both_free]); cols.append(a[both_free]); weights.append(np.full(both_free.sum(), w))

        rows = np.concatenate(rows); cols = np.concatenate(cols); weights = np.concatenate(weights)
        graph = coo_matrix((weights, (rows, cols)), shape=(n_nodes, n_nodes)).tocsr()
        log(f"[true-dist] grid graph built: {n_nodes} nodes, {graph.nnz} directed edges")

        def snap(pos_xy):
            ix = np.clip(np.round((pos_xy[:, 0] - self.ROOM_LO) / self.GRID_RES).astype(int), 0, nx - 1)
            iy = np.clip(np.round((pos_xy[:, 1] - self.ROOM_LO) / self.GRID_RES).astype(int), 0, ny - 1)
            nid = node_id[ix, iy]
            bad = np.nonzero(nid < 0)[0]
            for k in bad:
                found = False
                for r in range(1, 6):
                    for ddx in range(-r, r + 1):
                        for ddy in range(-r, r + 1):
                            xi, yi = ix[k] + ddx, iy[k] + ddy
                            if 0 <= xi < nx and 0 <= yi < ny and node_id[xi, yi] >= 0:
                                nid[k] = node_id[xi, yi]
                                found = True
                                break
                        if found:
                            break
                    if found:
                        break
            return nid

        def true_dist_from_sources(source_pos_xy, target_pos_xy):
            src_nodes = snap(source_pos_xy)
            tgt_nodes = snap(target_pos_xy)
            D = dijkstra(graph, indices=src_nodes, directed=False)
            return D[:, tgt_nodes]

        return true_dist_from_sources

    def make_env(self):
        from stable_worldmodel.envs.two_room.env import TwoRoomEnv
        return TwoRoomEnv(render_mode="rgb_array")

    def reset_options(self, start_state, goal_state):
        return {"state": start_state, "target_state": goal_state}

    def step_result(self, obs, reward, terminated, truncated, info):
        return bool(terminated), float(info["distance_to_target"])

    def collect_noisy_rollout(self, n_episodes, seed, max_steps=None, action_noise=0.4,
                               action_repeat_prob=0.15):
        max_steps = max_steps or self.max_episode_steps
        from stable_worldmodel.envs.two_room.env import TwoRoomEnv
        from stable_worldmodel.envs.two_room.expert_policy import ExpertPolicy

        env = TwoRoomEnv(render_mode="rgb_array")
        policy = ExpertPolicy(action_noise=action_noise, action_repeat_prob=action_repeat_prob, seed=seed)
        policy.set_env(env)

        all_pixels, all_state, all_action, all_ep, all_step = [], [], [], [], []
        ep_lens, n_terminated = [], 0
        t0 = time.time()
        for ep in range(n_episodes):
            obs, info = env.reset(seed=seed * 100_000 + ep)
            policy._last_action = None
            for t in range(max_steps):
                pixels = env.render()
                state = info["proprio"].copy()
                action = policy.get_action(info)
                obs, reward, terminated, truncated, info = env.step(action)

                all_pixels.append(pixels)
                all_state.append(state)
                all_action.append(action.astype(np.float32))
                all_ep.append(ep)
                all_step.append(t)

                if terminated or truncated:
                    break
            ep_lens.append(t + 1)
            n_terminated += int(terminated)

            if ep % 20 == 0:
                log(f"[rollout] episode {ep}/{n_episodes} len={t+1} "
                    f"terminated={terminated} elapsed={time.time()-t0:.1f}s")

        log(f"[rollout] done: {len(all_ep)} steps, {n_episodes} episodes, "
            f"mean_len={np.mean(ep_lens):.1f}, terminated={n_terminated}/{n_episodes} "
            f"({100*n_terminated/n_episodes:.0f}%), noise={action_noise}, repeat_prob={action_repeat_prob}")

        return dict(
            pixels=np.stack(all_pixels).astype(np.uint8),
            state=np.stack(all_state).astype(np.float32),
            action=np.stack(all_action).astype(np.float32),
            ep_idx=np.array(all_ep, dtype=np.int64),
            step_idx=np.array(all_step, dtype=np.int64),
        )


# ============================================================
# Push-T
# ============================================================

class PushTMechanics(EnvMechanics):
    name = "pusht"
    state_dim = 7
    action_dim = 2
    max_episode_steps = 250
    h5_path_env_var = "PUSHT_H5_PATH"
    h5_path_default = "data/datasets/pusht_expert_train.h5"
    ckpt_repo = "quentinll/lewm-pusht"
    state_h5_key = "state"

    @staticmethod
    def _project_state(state):
        return np.stack([
            state[:, 0], state[:, 1],                       # agent x, y
            state[:, 2], state[:, 3],                       # block x, y
            np.cos(state[:, 4]), np.sin(state[:, 4]),        # block angle, wrap-free
        ], axis=1).astype(np.float64)

    def build_true_distance_oracle(self, reference_state):
        proj = self._project_state
        mean, std = proj(reference_state).mean(0, keepdims=True), proj(reference_state).std(0, keepdims=True)
        std[std < 1e-6] = 1.0

        def true_dist_from_states(source_state, target_state):
            a = (proj(source_state) - mean) / std
            b = (proj(target_state) - mean) / std
            return np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)

        return true_dist_from_states

    def make_env(self):
        from stable_worldmodel.envs.pusht.env import PushT
        return PushT(render_mode="rgb_array")

    def reset_options(self, start_state, goal_state):
        return {"state": start_state, "goal_state": goal_state}

    def step_result(self, obs, reward, terminated, truncated, info):
        return bool(terminated), float(-reward)

    def collect_noisy_rollout(self, n_episodes, seed, max_steps=None, dist_constraint=100):
        import gymnasium

        max_steps = max_steps or self.max_episode_steps
        from stable_worldmodel.envs.pusht.expert_policy import WeakPolicy

        # WeakPolicy.get_action indexes self.env.envs and writes into an (n_envs, act_dim)
        # array -- written for a vectorized env, unlike Two-Room's ExpertPolicy. Wrap in a
        # single-env SyncVectorEnv to satisfy that contract.
        vec_env = gymnasium.vector.SyncVectorEnv(
            [lambda: gymnasium.make("swm/PushT-v1", render_mode="rgb_array")]
        )
        policy = WeakPolicy(dist_constraint=dist_constraint, seed=seed)
        policy.set_env(vec_env)
        raw_env = vec_env.envs[0].unwrapped

        # Preallocated and filled in-place rather than list-append + np.stack -- at scale
        # (up to n_episodes*max_steps pixel frames) stacking needs the list AND the new
        # array in memory at once, which hits ArrayMemoryError on some machines well within
        # nominal free RAM. Upper-bounded by n_episodes*max_steps (WeakPolicy runs every
        # episode to max_steps, 0% success observed), truncated to the actual total at the end.
        cap = n_episodes * max_steps
        pix_buf = np.empty((cap, 224, 224, 3), dtype=np.uint8)
        state_buf = np.empty((cap, self.state_dim), dtype=np.float32)
        action_buf = np.empty((cap, self.action_dim), dtype=np.float32)
        ep_buf = np.empty(cap, dtype=np.int64)
        step_buf = np.empty(cap, dtype=np.int64)
        n_written = 0
        ep_lens, n_terminated = [], 0
        t0 = time.time()
        for ep in range(n_episodes):
            obs, info = vec_env.reset(seed=seed * 100_000 + ep)
            for t in range(max_steps):
                pix_buf[n_written] = raw_env.render()
                state_buf[n_written] = obs["state"][0]
                action = policy.get_action(info)
                action_buf[n_written] = action[0]
                ep_buf[n_written] = ep
                step_buf[n_written] = t
                n_written += 1
                obs, reward, terminated, truncated, info = vec_env.step(action)

                if terminated[0] or truncated[0]:
                    break
            ep_lens.append(t + 1)
            n_terminated += int(terminated[0])

            if ep % 20 == 0:
                log(f"[rollout] episode {ep}/{n_episodes} len={t+1} "
                    f"terminated={terminated} elapsed={time.time()-t0:.1f}s")

        log(f"[rollout] done: {n_written} steps, {n_episodes} episodes, "
            f"mean_len={np.mean(ep_lens):.1f}, terminated={n_terminated}/{n_episodes} "
            f"({100*n_terminated/n_episodes:.0f}%), dist_constraint={dist_constraint}")

        return dict(
            pixels=pix_buf[:n_written], state=state_buf[:n_written], action=action_buf[:n_written],
            ep_idx=ep_buf[:n_written], step_idx=step_buf[:n_written],
        )


# ============================================================
# Reacher (dm_control)
# ============================================================

class _ReacherEnv:
    """Wraps ReacherDMControlWrapper (task='qpos_match') to expose target_qpos in info --
    upstream's own `info` property adds target_pos/finger_pos (the *visual* reward target
    from the original dm_control reward, unused here) but not the qpos_match task's actual
    goal (self.env.task.target_qpos), which step_result() needs to score distance-to-goal."""

    def __new__(cls, *args, **kwargs):
        import os
        os.environ.setdefault("MUJOCO_GL", "egl")
        from stable_worldmodel.envs.dmcontrol.reacher import ReacherDMControlWrapper

        class _Impl(ReacherDMControlWrapper):
            @property
            def info(self):
                info = super().info
                tq = getattr(self.env.task, "target_qpos", None)
                info["target_qpos"] = tq.copy() if tq is not None else None
                return info

            def reset(self, seed=None, options=None):
                obs, info = super().reset(seed=seed, options=options)
                tq = getattr(self.env.task, "target_qpos", None)
                info["target_qpos"] = tq.copy() if tq is not None else None
                return obs, info

        return _Impl(*args, **kwargs)


class ReacherMechanics(EnvMechanics):
    name = "reacher"
    # dm_control reacher: nq=2 (shoulder, wrist), nv=2 -- confirmed live (physics.model.nq/nv)
    # rather than assumed; the env's own "state" reset option requires exactly nq+nv.
    state_dim = 4
    action_dim = 2
    max_episode_steps = 150
    h5_path_env_var = "REACHER_H5_PATH"
    h5_path_default = "data/datasets/reacher.h5"
    ckpt_repo = "quentinll/lewm-reacher"
    state_h5_key = "state"

    NQ = 2  # qpos-only prefix of the (nq+nv,) state vector -- qvel is irrelevant to the
            # qpos_match task's termination check, same reasoning as Push-T dropping fields
            # its oracle doesn't need.

    @staticmethod
    def _project_qpos(qpos):
        """qpos: (..., 2) = [shoulder, wrist]. Shoulder is an unlimited hinge (jnt_limited=
        False, observed range spans the full circle) so raw angle wraps at +/-pi and needs
        sin/cos treatment, exactly like Push-T's block angle. Wrist is range-limited
        (jnt_limited=True, +/-2.79 rad) so the raw value is safe to use directly."""
        qpos = np.asarray(qpos)
        return np.stack([
            np.cos(qpos[..., 0]), np.sin(qpos[..., 0]),
            qpos[..., 1],
        ], axis=-1).astype(np.float64)

    def build_true_distance_oracle(self, reference_state):
        def proj(state):
            return self._project_qpos(np.asarray(state)[:, :self.NQ])

        ref = proj(reference_state)
        mean, std = ref.mean(0, keepdims=True), ref.std(0, keepdims=True)
        std[std < 1e-6] = 1.0

        def true_dist_from_states(source_state, target_state):
            a = (proj(source_state) - mean) / std
            b = (proj(target_state) - mean) / std
            return np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)

        return true_dist_from_states

    def make_env(self):
        return _ReacherEnv(task="qpos_match", render_mode="rgb_array")

    def reset_options(self, start_state, goal_state):
        return {"state": np.asarray(start_state), "target_qpos": np.asarray(goal_state)[:self.NQ]}

    def step_result(self, obs, reward, terminated, truncated, info):
        target_qpos = info.get("target_qpos")
        if target_qpos is None:
            return bool(terminated), float("nan")
        a = self._project_qpos(info["qpos"][:self.NQ])
        b = self._project_qpos(np.asarray(target_qpos))
        return bool(terminated), float(np.linalg.norm(a - b))

    def collect_noisy_rollout(self, n_episodes, seed, max_steps=None, **policy_kwargs):
        raise NotImplementedError(
            "Reacher noisy-rollout collection (mixed/mixed_large tiers) is not implemented. "
            "dm_control's reacher expert policy (stable_worldmodel.envs.dmcontrol."
            "expert_policy.ExpertPolicy) requires a separately-trained SB3 SAC checkpoint + "
            "VecNormalize stats that LEWM does not ship, unlike Two-Room/Push-T's "
            "hand-written heuristic policies. Only expert_N tiers are supported for now."
        )


ENV_MECHANICS = {
    "tworoom": TwoRoomMechanics(),
    "pusht": PushTMechanics(),
    "reacher": ReacherMechanics(),
}
