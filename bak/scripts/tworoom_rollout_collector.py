"""Live rollout collection for Two-Room, using the real env + ExpertPolicy at a tunable
noise level. Produces genuine (pixels, proprio, action) trajectories where a noisy action
actually changes the resulting state -- not a post-hoc label swap on top of real recorded
transitions (that was the earlier, rejected "hacky" approach).

Output arrays match the same convention load_landmarks()/tworoom_b0_graph_gate.py use for
the real dataset (pixels: (N,224,224,3) uint8, proprio: (N,2) float32, action: (N,2)
float32, ep_idx/step_idx: (N,) int), so everything downstream (encoding, graph building)
is unchanged -- only the source of the frames differs.
"""

import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "outputs" / "b3_tworoom" / "rollout_cache"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_STEPS = 150  # generous vs. the real dataset's observed max episode length (~101)


def log(*a):
    print(*a, flush=True)


def encode_pixels(pixels, batch_size=128):
    """Encode a (N,224,224,3) uint8 array with the frozen pretrained LeWM encoder.
    Same preprocessing/model as everywhere else in B0-B3 (tworoom_b2_encoder_swap.py's
    make_lewm_encoder), just applied to in-memory pixels rather than an h5 file."""
    from tworoom_b0_graph_gate import DEV, IMAGENET_MEAN, IMAGENET_STD
    from tworoom_lewm_loader import load_tworoom_lewm

    model = load_tworoom_lewm(device=DEV)
    mean, std = IMAGENET_MEAN.to(DEV), IMAGENET_STD.to(DEV)

    chunks = []
    for i in range(0, len(pixels), batch_size):
        chunk = pixels[i:i + batch_size]
        t = torch.from_numpy(chunk).to(DEV).permute(0, 3, 1, 2).float() / 255.0
        t = (t - mean) / std
        with torch.no_grad():
            out = model.encode({"pixels": t.unsqueeze(1)})
        chunks.append(out["emb"][:, 0].cpu().numpy())
    return np.concatenate(chunks, axis=0).astype(np.float32)


def collect_rollouts(n_episodes, action_noise, action_repeat_prob, seed, max_steps=MAX_STEPS,
                      cache_name=None):
    """Roll out ExpertPolicy(action_noise, action_repeat_prob) for n_episodes, recording
    every real env step. Returns dict(pixels, proprio, action, ep_idx, step_idx,
    terminated_frac, mean_ep_len)."""
    if cache_name is not None:
        cache_path = OUT_DIR / f"{cache_name}.npz"
        if cache_path.exists():
            log(f"[rollout] loading cached '{cache_name}'")
            d = np.load(cache_path)
            return {k: d[k] for k in d.files}

    from stable_worldmodel.envs.two_room.env import TwoRoomEnv
    from stable_worldmodel.envs.two_room.expert_policy import ExpertPolicy

    env = TwoRoomEnv(render_mode="rgb_array")
    policy = ExpertPolicy(action_noise=action_noise, action_repeat_prob=action_repeat_prob, seed=seed)
    policy.set_env(env)

    all_pixels, all_proprio, all_action, all_ep, all_step = [], [], [], [], []
    ep_lens, n_terminated = [], 0
    t0 = time.time()
    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed * 100_000 + ep)
        policy._last_action = None
        for t in range(max_steps):
            pixels = env.render()  # HWC uint8
            proprio = info["proprio"].copy()
            action = policy.get_action(info)
            obs, reward, terminated, truncated, info = env.step(action)

            all_pixels.append(pixels)
            all_proprio.append(proprio)
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

    result = dict(
        pixels=np.stack(all_pixels).astype(np.uint8),
        proprio=np.stack(all_proprio).astype(np.float32),
        action=np.stack(all_action).astype(np.float32),
        ep_idx=np.array(all_ep, dtype=np.int64),
        step_idx=np.array(all_step, dtype=np.int64),
    )
    log(f"[rollout] done: {len(all_ep)} steps, {n_episodes} episodes, "
        f"mean_len={np.mean(ep_lens):.1f}, terminated={n_terminated}/{n_episodes} "
        f"({100*n_terminated/n_episodes:.0f}%), noise={action_noise}, repeat_prob={action_repeat_prob}")

    if cache_name is not None:
        np.savez(OUT_DIR / f"{cache_name}.npz", **result)
        log(f"[rollout] cached to {cache_name}.npz")
    return result


if __name__ == "__main__":
    # Validation: noise=0 should closely resemble the real expert data (short episodes,
    # near-100% termination); higher noise should show longer episodes and more failures
    # to reach the goal within max_steps -- if that trend doesn't hold, something is wrong.
    log("=== validating rollout collector across noise levels ===")
    for noise, repeat_p in [(0.0, 0.0), (0.15, 0.05), (0.4, 0.15), (0.7, 0.3)]:
        r = collect_rollouts(n_episodes=30, action_noise=noise, action_repeat_prob=repeat_p, seed=0)
        ep_lens = np.bincount(r["ep_idx"])
        log(f"noise={noise} repeat_prob={repeat_p}: mean_ep_len={ep_lens.mean():.1f} "
            f"(min={ep_lens.min()}, max={ep_lens.max()})")
