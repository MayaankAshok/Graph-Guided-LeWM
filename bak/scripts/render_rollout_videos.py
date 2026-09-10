"""Render live-rollout videos for a trained actor checkpoint (actor_train.py), under the
same protocol/pair-sampling as actor_rollout_eval.py -- picks the first n_successes
successful and n_failures failed episodes out of a candidate pool and writes each as an mp4.

Never needs the graph/phi_dist: a trained actor's weights already encode whatever the
training-time graph taught it, and playback only calls the actor forward + steps the env, so
this always builds its setup with need_graph=false regardless of which variant trained the
checkpoint being played back -- keeps this runnable on a small/no-GPU machine with just the
h5 dataset and the frozen LeWM encoder, no Ada needed.

goal_mode=fixed replaces the usual per-episode "+25 steps in the same trajectory" goal with
ONE constant goal state, reused for every rendered episode: one real episode's own final
recorded state (not PushT's cosmetic env.goal_pose, which is a decorative block-only overlay
never read by eval_state(), and not a mean across many episode-ends either -- that was tried
first and gave 0/200 successes for both variants here, because averaging Cartesian
agent/block positions and angles across ~100 differently-configured episodes can produce a
combination that was never a real, physically-achieved pose in the first place). Start states
are still drawn the normal way (build_eval_pairs); only the goal is overridden.

Run as:
    python scripts/render_rollout_videos.py env=pusht tier=expert_100 variant=auxphi seed=0 \
        select=rho n_successes=5 n_failures=5 out_dir=/path/to/videos
    python scripts/render_rollout_videos.py env=pusht tier=expert_100 variant=auxphi seed=0 \
        goal_mode=fixed out_dir=/path/to/videos
"""

import sys
from pathlib import Path

import cv2
import hydra
import numpy as np
from omegaconf import DictConfig, open_dict

sys.path.insert(0, str(Path(__file__).resolve().parent))
from actor_rollout_utils import build_eval_pairs, eval_max_steps, make_encoder, make_env
from actor_train import ckpt_path, get_setup, run_tag
from common.checkpoint_io import load_checkpoint
from common.envs import ENV_MECHANICS
from common.graph_lib import DEV
from common.log_util import log
from common.training import HIDDEN, MLP

ROOT = Path(__file__).resolve().parent.parent


def pick_real_final_goal(setup, seed=0):
    """One real episode's own final recorded state, chosen reproducibly -- a genuinely
    achieved pose, unlike averaging many episodes' end states together (tried first: gave
    0/200 successes for both variants, because the mean of many different agent/block
    configurations can land on a combination no real episode ever actually occupied)."""
    ep_idx, step_idx, state = setup["ep_idx"], setup["step_idx"], setup["proprio"]
    order = np.lexsort((step_idx, ep_idx))
    ep_o = ep_idx[order]
    boundaries = np.nonzero(np.diff(ep_o))[0] + 1
    end_rows = order[np.concatenate((boundaries, [len(order)])) - 1]
    rng = np.random.default_rng(seed)
    row = int(rng.choice(end_rows))
    goal = state[row].astype(np.float32)
    log(f"[fixed-goal] real episode end (row {row}, ep={ep_idx[row]}): agent_xy={goal[:2]}, "
        f"block_xy={goal[2:4]}, angle={goal[4]:.3f}")
    return goal


def make_goal_ghost_renderer(env_name):
    """A second, throwaway env instance used ONLY to render the true goal pose for display
    overlays -- kept fully separate from whichever env is driving the actual rollout, so
    this can never change what gets fed to encode_frame() (which must stay byte-identical to
    what produced any already-reported success-rate numbers).

    Also disables the env's own decorative goal_pose overlay (the fixed [256,256]/pi/4
    highlight PushT draws in every frame regardless of our actual goal_state -- see env.py:
    self.goal_pose comes from a variation-space entry DEFAULT_VARIATIONS never touches, so it
    never reflects whatever goal we actually pass in) for THIS renderer only, so the ghost we
    extract is exactly agent+block at the true goal, nothing else."""
    overlay_env = make_env(env_name)
    if hasattr(overlay_env, "with_target"):
        overlay_env.with_target = False

    def render_ghost(mech, goal_state):
        _, info = overlay_env.reset(seed=0, options=mech.reset_options(goal_state, goal_state))
        goal_frame = np.asarray(info["goal"]) if "goal" in info else overlay_env.render()
        mask = np.abs(goal_frame.astype(np.int16) - 255).sum(axis=-1) > 30  # non-background
        return goal_frame, mask

    return render_ghost


def overlay_goal_ghost(frame, goal_frame, mask, alpha=0.55, outline_color=(255, 0, 255)):
    """Alpha-blend the goal frame's own agent/block colors (faded, not a flat tint --
    keeps blue-agent vs gray-block distinguishable) into `frame` at `mask`, plus a bright
    outline so the ghost stays legible even over a similarly-colored background."""
    out = frame.copy()
    out[mask] = ((1 - alpha) * out[mask].astype(np.float32)
                 + alpha * goal_frame[mask].astype(np.float32)).astype(np.uint8)
    contours, _ = cv2.findContours(mask.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, outline_color, 1)
    return out


def select_actor_state(cfg, ckpt, tag):
    if cfg.select == "rho":
        state_key, step_key = "peak_actor_state", "peak_step"
    elif cfg.select == "success":
        state_key, step_key = "peak_success_actor_state", "peak_success_step"
    else:
        state_key, step_key = "actor_net", "step"
    if ckpt.get(state_key) is None:
        raise SystemExit(f"[{tag}] {cfg.tier}/{cfg.variant}/s{cfg.seed} has no {state_key} yet")
    return ckpt[state_key], ckpt[step_key]


def write_video(path, frames, fps, goal_frame=None, hold_secs=1.0, label=None):
    """frames: list of HxWx3 uint8 RGB arrays. Prepends `goal_frame` held for `hold_secs`
    (viewer sees the target before the rollout plays), and stamps a small top-left label."""
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    seq = ([goal_frame] * max(1, int(fps * hold_secs)) if goal_frame is not None else []) + list(frames)
    for i, frame in enumerate(seq):
        bgr = cv2.cvtColor(np.ascontiguousarray(frame), cv2.COLOR_RGB2BGR)
        is_goal_hold = goal_frame is not None and i < max(1, int(fps * hold_secs))
        text = "GOAL" if is_goal_hold else label
        if text:
            cv2.putText(bgr, text, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 200, 0) if text != "GOAL" else (0, 165, 255), 2, cv2.LINE_AA)
        writer.write(bgr)
    writer.release()


@hydra.main(version_base=None, config_path="../config/graph", config_name="actor")
def main(cfg: DictConfig):
    with open_dict(cfg):
        cfg.need_graph = False  # playback never needs it -- see module docstring
    mech = ENV_MECHANICS[cfg.env.name]
    with open_dict(cfg.env):
        cfg.env.ckpt_dir_resolved = str(mech.ckpt_dir(ROOT))
    variant_name = f"{cfg.variant}{run_tag(cfg)}"
    tag = f"{cfg.env.name}/{cfg.tier}/{variant_name}/s{cfg.seed}"

    path = ckpt_path(cfg, cfg.tier, cfg.variant, cfg.seed)
    ckpt = load_checkpoint(path)
    if ckpt is None:
        raise SystemExit(f"[{tag}] no checkpoint at {path}")
    actor_state, step = select_actor_state(cfg, ckpt, tag)
    log(f"[{tag}] loaded checkpoint (select={cfg.select} -> step {step})")

    setup = get_setup(cfg, cfg.tier)
    d, act_dim = setup["d"], setup["action"].shape[1]
    actor_net = MLP(2 * d, HIDDEN, out_dim=act_dim).to(DEV)
    actor_net.load_state_dict(actor_state)
    actor_net.eval()

    env = make_env(cfg.env.name)
    render_ghost = make_goal_ghost_renderer(cfg.env.name)
    encode_frame = make_encoder(cfg.env.ckpt_dir_resolved)
    # A goal_mode=fixed target is some unrelated real episode's endpoint, not a nearby
    # same-trajectory state -- the paper protocol's 50-step budget (calibrated for a goal
    # only ~25 steps away) is too tight for that, so use the env's full episode length.
    max_steps = mech.max_episode_steps if cfg.goal_mode == "fixed" else eval_max_steps(cfg)
    fps = env.metadata.get("render_fps", 10) if hasattr(env, "metadata") else 10

    n_want = cfg.n_successes + cfg.n_failures
    pool_size = cfg.pool_size or max(4 * n_want, 40)
    pairs = build_eval_pairs(cfg, setup, pool_size, cfg.eval_seed, encode_frame)
    if cfg.goal_mode == "fixed":
        # Keep the varied start states build_eval_pairs already sampled; replace only the
        # goal with one constant value shared by every rendered episode.
        canon_goal = pick_real_final_goal(setup, seed=cfg.eval_seed)
        _, canon_info = env.reset(seed=0, options=mech.reset_options(canon_goal, canon_goal))
        canon_z_g = encode_frame(np.asarray(canon_info["goal"])[None]) if "goal" in canon_info \
            else encode_frame(env.render()[None])
        for p in pairs:
            p["goal_state"], p["z_g"] = canon_goal, canon_z_g
    log(f"[{tag}] goal_mode={cfg.goal_mode}, candidate pool: {len(pairs)} pairs, need "
        f"{cfg.n_successes} success + {cfg.n_failures} failure videos")

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_succ = n_fail = 0

    for k, p in enumerate(pairs):
        if n_succ >= cfg.n_successes and n_fail >= cfg.n_failures:
            break
        _, info = env.reset(seed=cfg.eval_seed * 100_000 + k,
                             options=mech.reset_options(p["start_state"], p["goal_state"]))
        z_g = encode_frame(np.asarray(info["goal"])[None]) if "goal" in info else p["z_g"]
        goal_frame = np.asarray(info["goal"]) if "goal" in info else None
        # Rendered from a SEPARATE env (make_goal_ghost_renderer) with the decorative
        # goal_pose overlay off -- composited only onto the DISPLAYED frames below, never
        # touching `pixels` (which drives z_s/the actor's action, must stay byte-identical
        # to whatever produced this checkpoint's already-reported success rate).
        ghost_frame, ghost_mask = render_ghost(mech, p["goal_state"])

        frames, success = [], False
        for t in range(max_steps):
            pixels = env.render()
            frames.append(overlay_goal_ghost(pixels, ghost_frame, ghost_mask))
            z_s = encode_frame(pixels[None])
            import torch
            with torch.no_grad():
                a = torch.tanh(actor_net(torch.cat([z_s, z_g], dim=-1))).cpu().numpy()[0]
            obs, reward, terminated, truncated, info = env.step(a)
            success, final_dist = mech.step_result(obs, reward, terminated, truncated, info)
            if success:
                frames.append(overlay_goal_ghost(env.render(), ghost_frame, ghost_mask))
                break

        if success and n_succ < cfg.n_successes:
            n_succ += 1
            out_path = out_dir / f"{cfg.env.name}_{cfg.tier}_{variant_name}_success_{n_succ}.mp4"
            write_video(out_path, frames, fps, goal_frame, label=f"SUCCESS (t={len(frames)})")
            log(f"[{tag}] pair {k}: SUCCESS in {len(frames)} steps -> {out_path}")
        elif not success and n_fail < cfg.n_failures:
            n_fail += 1
            out_path = out_dir / f"{cfg.env.name}_{cfg.tier}_{variant_name}_failure_{n_fail}.mp4"
            write_video(out_path, frames, fps, goal_frame, label="FAILED")
            log(f"[{tag}] pair {k}: failed (final_dist={final_dist:.1f}) -> {out_path}")

    if n_succ < cfg.n_successes or n_fail < cfg.n_failures:
        log(f"[{tag}] WARNING: only found {n_succ}/{cfg.n_successes} successes and "
            f"{n_fail}/{cfg.n_failures} failures in a pool of {len(pairs)} -- widen pool_size")
    log(f"[{tag}] done: {n_succ} success + {n_fail} failure videos written to {out_dir}")


if __name__ == "__main__":
    main()
