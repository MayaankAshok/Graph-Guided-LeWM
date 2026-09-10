"""Shared live-rollout mechanics for the trained-actor evaluation used by both
tworoom_b3_actor_train.py (periodic in-training checks) and
tworoom_b3_actor_rollout_eval.py (the final, larger-sample standalone evaluation). Split
into its own module -- with no dependency on either of those two -- specifically to avoid
a circular import between them (actor_train needs these to run periodic checks;
rollout_eval needs actor_train's ckpt_path/get_setup/HIDDEN; if rollout_eval also owned
these functions, actor_train importing from it would import back into a
partially-initialized actor_train).
"""

import time

import numpy as np
import torch

from tworoom_b0_graph_gate import DEV, IMAGENET_MEAN, IMAGENET_STD, log
from tworoom_lewm_loader import load_tworoom_lewm

MAX_STEPS = 150
SUCCESS_DIST = 16.0  # matches TwoRoomEnv's own terminated = dist < 16.0


def make_env():
    from stable_worldmodel.envs.two_room.env import TwoRoomEnv
    return TwoRoomEnv(render_mode="rgb_array")


def make_encoder():
    # Load the frozen encoder ONCE, reused across every episode/call -- encode_pixels() in
    # tworoom_rollout_collector.py reloads the model from disk on every call, which is fine
    # for its own one-shot batch-encoding use case but catastrophic here (one call per env
    # step, up to 150/episode): an early smoke test using it per-step never got past a
    # handful of steps in two minutes. This mirrors encode_pixels' preprocessing exactly,
    # just with the model loaded once.
    encoder = load_tworoom_lewm(device=DEV)
    mean, std = IMAGENET_MEAN.to(DEV), IMAGENET_STD.to(DEV)

    def encode_frame(pixels_hwc):
        t = torch.from_numpy(pixels_hwc).to(DEV).permute(0, 3, 1, 2).float() / 255.0
        t = (t - mean) / std
        with torch.no_grad():
            out = encoder.encode({"pixels": t.unsqueeze(1)})
        return out["emb"][:, 0]  # (1,d) tensor, stays on DEV

    return encode_frame


def run_episodes(actor_net, env, encode_frame, setup, pair_idx, seed, log_progress=False):
    """pair_idx: indices into setup['test_eval']'s (ei,ej) arrays to use as (start,goal)."""
    test_ei, test_ej, _ = setup["test_eval"]
    z, proprio = setup["z"], setup["proprio"]
    episodes = []
    t0 = time.time()
    for k, i in enumerate(pair_idx):
        start_row, goal_row = int(test_ei[i]), int(test_ej[i])
        start_pos = proprio[start_row]
        goal_pos = proprio[goal_row]
        z_g = torch.from_numpy(z[goal_row]).float().to(DEV).unsqueeze(0)

        obs, info = env.reset(seed=seed * 100_000 + k, options={"state": start_pos, "target_state": goal_pos})
        success, t = False, 0
        for t in range(MAX_STEPS):
            pixels = env.render()[None]  # (1,H,W,3)
            z_s = encode_frame(pixels)  # (1,d)
            with torch.no_grad():
                a = torch.tanh(actor_net(torch.cat([z_s, z_g], dim=-1))).cpu().numpy()[0]
            obs, reward, terminated, truncated, info = env.step(a)
            if terminated:
                success = True
                break

        episodes.append(dict(start_row=start_row, goal_row=goal_row, success=bool(success),
                              steps=t + 1, final_dist=float(info["distance_to_target"])))
        if log_progress and k % 10 == 0:
            log(f"  episode {k}/{len(pair_idx)}: success={success} steps={t+1} "
                f"final_dist={info['distance_to_target']:.1f} elapsed={time.time()-t0:.1f}s")
    return episodes


def summarize(episodes):
    successes = [e["success"] for e in episodes]
    success_steps = [e["steps"] for e in episodes if e["success"]]
    fail_dists = [e["final_dist"] for e in episodes if not e["success"]]
    return dict(
        n_episodes=len(episodes),
        success_rate=float(np.mean(successes)),
        mean_steps_to_goal=float(np.mean(success_steps)) if success_steps else None,
        mean_final_dist_on_failure=float(np.mean(fail_dists)) if fail_dists else None,
        episodes=episodes,
    )


def rollout_eval(actor_net, setup, n_episodes, seed, env=None, encode_frame=None):
    """Standalone-use entrypoint (builds env/encoder if not given). For repeated calls in
    one process (e.g. periodic checks during training), build them once with make_env()/
    make_encoder() and pass them in instead."""
    if env is None:
        env = make_env()
    if encode_frame is None:
        encode_frame = make_encoder()
    rng = np.random.default_rng(seed)
    test_ei, _, _ = setup["test_eval"]
    idx = rng.choice(len(test_ei), size=min(n_episodes, len(test_ei)), replace=False)
    episodes = run_episodes(actor_net, env, encode_frame, setup, idx, seed, log_progress=True)
    return summarize(episodes)
