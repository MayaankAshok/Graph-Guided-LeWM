"""Side-by-side comparison videos: baseline (left) vs auxphi (right), same tier/seed/start
state/goal, picking episodes where the two DISAGREE on outcome (one succeeds, one fails).
No goal-preview hold -- starts directly on the rollout.

Reuses render_rollout_videos.py's goal construction (goal_mode=fixed: one real episode's
own end state, not an average across episodes -- see that script's docstring for why) and
pair sampling, but drives both actors through the identical (start, goal) pairs in lockstep
so the comparison is exact, not just same-tier-different-runs.

Run as:
    python scripts/render_compare_videos.py env=pusht tier=expert_100 seed=0 goal_mode=fixed \
        eval_seed=7 pool_size=80 n_compare=5 out_dir=/path/to/videos
"""

import sys
from pathlib import Path

import cv2
import hydra
import numpy as np
import torch
from omegaconf import DictConfig, open_dict

sys.path.insert(0, str(Path(__file__).resolve().parent))
from actor_rollout_utils import build_eval_pairs, make_encoder, make_env
from actor_train import ckpt_path, get_setup
from common.checkpoint_io import load_checkpoint
from common.envs import ENV_MECHANICS
from common.graph_lib import DEV
from common.log_util import log
from common.training import HIDDEN, MLP
from render_rollout_videos import make_goal_ghost_renderer, overlay_goal_ghost, pick_real_final_goal, select_actor_state

ROOT = Path(__file__).resolve().parent.parent


def load_actor(cfg, variant, d, act_dim, tag):
    path = ckpt_path(cfg, cfg.tier, variant, cfg.seed)
    ckpt = load_checkpoint(path)
    if ckpt is None:
        raise SystemExit(f"[{tag}] no checkpoint at {path}")
    state, step = select_actor_state(cfg, ckpt, tag)
    net = MLP(2 * d, HIDDEN, out_dim=act_dim).to(DEV)
    net.load_state_dict(state)
    net.eval()
    log(f"[{tag}] loaded {variant} (select={cfg.select} -> step {step})")
    return net


def run_one(env, mech, actor_net, encode_frame, start_state, goal_state, z_g, seed, max_steps,
            ghost_frame, ghost_mask):
    """ghost_frame/ghost_mask: the true-goal overlay (see render_rollout_videos.
    make_goal_ghost_renderer) -- composited only onto the DISPLAYED frames, never onto
    `pixels`, which drives z_s/the actor's action and must match the numbers already
    reported (e.g. the 8%/9% full-pool success rates)."""
    env.reset(seed=seed, options=mech.reset_options(start_state, goal_state))
    frames, success = [], False
    for t in range(max_steps):
        pixels = env.render()
        frames.append(overlay_goal_ghost(pixels, ghost_frame, ghost_mask))
        z_s = encode_frame(pixels[None])
        with torch.no_grad():
            a = torch.tanh(actor_net(torch.cat([z_s, z_g], dim=-1))).cpu().numpy()[0]
        obs, reward, terminated, truncated, info = env.step(a)
        success, final_dist = mech.step_result(obs, reward, terminated, truncated, info)
        if success:
            frames.append(overlay_goal_ghost(env.render(), ghost_frame, ghost_mask))
            break
    return frames, success


def write_compare_video(path, frames_a, frames_b, fps, label_a, label_b):
    """No goal-preview hold. Shorter side is padded by freezing its last frame so both
    panels stay in sync for the full video length."""
    h, w = frames_a[0].shape[:2]
    n = max(len(frames_a), len(frames_b))
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (2 * w + 8, h))
    divider = np.full((h, 8, 3), 255, dtype=np.uint8)
    for i in range(n):
        fa = frames_a[min(i, len(frames_a) - 1)]
        fb = frames_b[min(i, len(frames_b) - 1)]
        canvas = np.concatenate([fa, divider, fb], axis=1)
        bgr = cv2.cvtColor(np.ascontiguousarray(canvas), cv2.COLOR_RGB2BGR)
        for text, x, color in ((label_a, 6, (0, 200, 0) if "SUCCESS" in label_a else (0, 0, 220)),
                                (label_b, w + 8 + 6, (0, 200, 0) if "SUCCESS" in label_b else (0, 0, 220))):
            cv2.putText(bgr, text, (x, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        writer.write(bgr)
    writer.release()


@hydra.main(version_base=None, config_path="../config/graph", config_name="actor")
def main(cfg: DictConfig):
    with open_dict(cfg):
        cfg.need_graph = False  # playback only -- see render_rollout_videos.py
    mech = ENV_MECHANICS[cfg.env.name]
    with open_dict(cfg.env):
        cfg.env.ckpt_dir_resolved = str(mech.ckpt_dir(ROOT))
    tag = f"{cfg.env.name}/{cfg.tier}/s{cfg.seed}"

    setup = get_setup(cfg, cfg.tier)
    d, act_dim = setup["d"], setup["action"].shape[1]
    baseline_net = load_actor(cfg, "baseline", d, act_dim, tag)
    auxphi_net = load_actor(cfg, "auxphi", d, act_dim, tag)

    env = make_env(cfg.env.name)
    render_ghost = make_goal_ghost_renderer(cfg.env.name)
    encode_frame = make_encoder(cfg.env.ckpt_dir_resolved)
    max_steps = mech.max_episode_steps if cfg.goal_mode == "fixed" else mech.max_episode_steps
    fps = env.metadata.get("render_fps", 10) if hasattr(env, "metadata") else 10

    pairs = build_eval_pairs(cfg, setup, cfg.pool_size or 80, cfg.eval_seed, encode_frame)
    if cfg.goal_mode == "fixed":
        canon_goal = pick_real_final_goal(setup, seed=cfg.eval_seed)
        _, canon_info = env.reset(seed=0, options=mech.reset_options(canon_goal, canon_goal))
        canon_z_g = encode_frame(np.asarray(canon_info["goal"])[None]) if "goal" in canon_info \
            else encode_frame(env.render()[None])
        for p in pairs:
            p["goal_state"], p["z_g"] = canon_goal, canon_z_g
    log(f"[{tag}] {len(pairs)} candidate pairs -- IDENTICAL (start_state, goal_state) fed to "
        f"both baseline and auxphi below, same env seed per pair, so any outcome difference "
        f"is the actor, not the task. Looking for {cfg.n_compare} divergent outcomes; scoring "
        f"success rate over the full pool at max_steps={max_steps} (goal_mode={cfg.goal_mode}).")

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_found = 0
    n_base_succ = n_aux_succ = 0

    # goal_mode=fixed means one constant goal for every pair -- compute its ghost once
    # rather than re-rendering it per pair.
    cached_ghost = render_ghost(mech, pairs[0]["goal_state"]) if (cfg.goal_mode == "fixed" and pairs) else None

    for k, p in enumerate(pairs):
        if k >= cfg.n_first and n_found >= cfg.n_compare:
            break  # nothing left this pair could contribute (past the "first" range and
                   # the divergence quota is already filled) -- stop paying for more rollouts
        seed = cfg.eval_seed * 100_000 + k
        ghost_frame, ghost_mask = cached_ghost if cached_ghost is not None else render_ghost(mech, p["goal_state"])
        frames_base, succ_base = run_one(env, mech, baseline_net, encode_frame,
                                          p["start_state"], p["goal_state"], p["z_g"], seed, max_steps,
                                          ghost_frame, ghost_mask)
        frames_aux, succ_aux = run_one(env, mech, auxphi_net, encode_frame,
                                        p["start_state"], p["goal_state"], p["z_g"], seed, max_steps,
                                        ghost_frame, ghost_mask)
        n_base_succ += int(succ_base)
        n_aux_succ += int(succ_aux)

        label_a = f"baseline: {'SUCCESS' if succ_base else 'FAILED'} (t={len(frames_base)})"
        label_b = f"auxphi: {'SUCCESS' if succ_aux else 'FAILED'} (t={len(frames_aux)})"

        if k < cfg.n_first:
            out_path = out_dir / f"{cfg.env.name}_{cfg.tier}_first_{k+1}.mp4"
            write_compare_video(out_path, frames_base, frames_aux, fps, label_a, label_b)
            log(f"[{tag}] pair {k}: (first-{cfg.n_first} set) {label_a} | {label_b} -> {out_path}")

        if succ_base == succ_aux:
            log(f"[{tag}] pair {k}: agree (baseline={succ_base}, auxphi={succ_aux})")
            continue
        if n_found < cfg.n_compare:
            n_found += 1
            out_path = out_dir / f"{cfg.env.name}_{cfg.tier}_compare_{n_found}.mp4"
            write_compare_video(out_path, frames_base, frames_aux, fps, label_a, label_b)
            log(f"[{tag}] pair {k}: DIVERGE -- {label_a} | {label_b} -> {out_path}")
        else:
            log(f"[{tag}] pair {k}: DIVERGE (baseline={succ_base}, auxphi={succ_aux}) -- "
                f"video quota already filled, counted for stats only")

    n = len(pairs)
    log(f"\n[{tag}] === success rate over {n} pairs, max_steps={max_steps}, "
        f"goal_mode={cfg.goal_mode}, IDENTICAL pairs for both ===")
    log(f"[{tag}]   baseline: {n_base_succ}/{n} = {100*n_base_succ/n:.1f}%")
    log(f"[{tag}]   auxphi  : {n_aux_succ}/{n} = {100*n_aux_succ/n:.1f}%")
    if n_found < cfg.n_compare:
        log(f"[{tag}] WARNING: only found {n_found}/{cfg.n_compare} divergent pairs in this pool")
    log(f"[{tag}] done: {n_found} comparison videos written to {out_dir}")


if __name__ == "__main__":
    main()
