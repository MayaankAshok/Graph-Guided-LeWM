"""Shared live-rollout mechanics for the trained-actor evaluation -- config-driven
consolidation of tworoom_actor_rollout_utils.py/pusht_actor_rollout_utils.py. The genuinely
environment-specific parts (env construction, reset-to-a-specific-state, and reading
success/distance off one step()) are common.envs.ENV_MECHANICS methods; this module is the
identical surrounding loop both environments already had.

Two live-rollout evaluation protocols are supported, selected by cfg.eval_protocol:

  same_episode -- the LeWM paper's protocol (App. F.1; stable-worldmodel scripts/plan/
      eval_ff.py + World._evaluate_from_dataset), which is what every policy number in the
      paper's Fig. 6 (GCBC 75%, GCIVL 33%, GCIQL 20%, Random 2% on Push-T) was measured
      under. The start is a random dataset state; the goal is the state exactly
      `env.paper_goal_offset` timesteps LATER IN THE SAME TRAJECTORY (so it is reachable
      and consistent with the dataset dynamics); the policy gets `env.paper_eval_budget`
      env steps (Push-T: offset 25 / budget 50, Two-Room: 100 / 150). The goal embedding is
      the encoding of the dataset frame at the goal row -- exactly what
      World._evaluate_from_dataset hands the policy as `goal` (a dataset frame, not an
      env-rendered one). Success = the env's own `terminated` at any step within the
      budget, i.e. the same env success criterion the paper uses. cfg.eval_pool picks where
      the (start, goal) pairs come from: `dataset` = the full h5 minus the tier's training
      episodes (the paper samples from the whole dataset; excluding the tier's train
      episodes just keeps the eval held-out, and with ~18.7k episodes vs <=100 per tier it
      barely changes the pool), `test_episodes` = only the tier's own held-out episodes
      (much smaller pool -- 2 episodes for expert_10).

  cross_episode -- the original convention: arbitrary held-out (s, g) landmark pairs from
      setup['test_eval'] (the very pairs the Spearman proxy is scored on), budget = the
      env's own max_episode_steps (Push-T: 250). Far harder than the paper's protocol on
      Push-T: the goal is a random state from a DIFFERENT trajectory, which for a
      contact-manipulation task means an arbitrary block pose that expert data never
      demonstrates reaching from this start. Every Push-T actor scored 0.000 under it (see
      [[pusht-b5-b3-rl-training]]), which is why the paper's protocol was added.
"""

import time
from pathlib import Path

import h5py
import numpy as np
import torch

from common.envs import ENV_MECHANICS
from common.graph_lib import DEV
from common.lewm_loader import load_lewm
from common.log_util import log

ROOT = Path(__file__).resolve().parent.parent

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def make_env(env_name):
    return ENV_MECHANICS[env_name].make_env()


def make_encoder(ckpt_dir):
    # Loaded once, reused across every episode/step -- per-step reloading (as in
    # rollout_collector.encode_pixels' own batch use case) is catastrophic here.
    encoder = load_lewm(ckpt_dir=Path(ckpt_dir), device=DEV)
    mean, std = IMAGENET_MEAN.to(DEV), IMAGENET_STD.to(DEV)

    def encode_frame(pixels_hwc):
        t = torch.from_numpy(np.ascontiguousarray(pixels_hwc)).to(DEV).permute(0, 3, 1, 2).float() / 255.0
        t = (t - mean) / std
        with torch.no_grad():
            out = encoder.encode({"pixels": t.unsqueeze(1)})
        return out["emb"][:, 0]  # (n,d) tensor, stays on DEV

    return encode_frame


# ------------------------------------------------------------------
# (start, goal) pair sampling
# ------------------------------------------------------------------

def eval_max_steps(cfg):
    """Per-episode step budget for the configured protocol."""
    if cfg.eval_protocol == "same_episode":
        return int(cfg.env.paper_eval_budget)
    return ENV_MECHANICS[cfg.env.name].max_episode_steps


def sample_same_episode_pairs_from_h5(h5_path, state_key, goal_offset, n, seed, exclude_eps=None, mech=None):
    """The paper's start sampling (eval_ff.py): every dataset row whose step_idx <=
    ep_len - goal_offset - 1 is a valid start; pick n of them uniformly without
    replacement; the goal is the row goal_offset steps later in the same episode.
    Episode ids here are positions in ep_offset (the same convention graph_gate.
    load_landmarks uses for setup['ep_idx']), so exclude_eps can be setup['train_eps']."""
    f = h5py.File(h5_path, "r", swmr=True, rdcc_nbytes=256 * 1024 * 1024)
    ep_offset = f["ep_offset"][:].astype(np.int64)
    ep_len = f["ep_len"][:].astype(np.int64)
    eps = np.arange(len(ep_offset))
    max_start = ep_len - goal_offset - 1
    keep = max_start >= 0
    if exclude_eps is not None:
        keep &= ~np.isin(eps, np.asarray(exclude_eps))
    valid_eps = eps[keep]
    counts = max_start[valid_eps] + 1
    total = int(counts.sum())
    rng = np.random.default_rng(seed)
    pick = np.sort(rng.choice(total, size=min(n, total), replace=False))
    cum = np.cumsum(counts)
    ep_pos = np.searchsorted(cum, pick, side="right")
    prev = np.where(ep_pos > 0, cum[np.maximum(ep_pos - 1, 0)], 0)
    step = pick - prev
    e = valid_eps[ep_pos]
    start_rows = ep_offset[e] + step           # strictly increasing -> h5 fancy-index safe
    goal_rows = start_rows + goal_offset
    if mech is not None:
        start_state = mech.read_state_rows(f, start_rows)
        goal_state = mech.read_state_rows(f, goal_rows)
    else:
        start_state = f[state_key][start_rows].astype(np.float32)
        goal_state = f[state_key][goal_rows].astype(np.float32)
    goal_pixels = f["pixels"][goal_rows]        # (n,224,224,3) uint8
    f.close()
    return dict(ep=e, start_step=step, start_row=start_rows, goal_row=goal_rows,
                start_state=start_state, goal_state=goal_state, goal_pixels=goal_pixels)


def sample_same_episode_pairs_from_setup(setup, goal_offset, n, seed):
    """Same protocol, restricted to the tier's own held-out episodes (rows already encoded)."""
    ep_idx, step_idx = setup["ep_idx"], setup["step_idx"]
    test_eps = set(int(x) for x in setup["test_eps"])
    order = np.lexsort((step_idx, ep_idx))
    ep_o = ep_idx[order]
    boundaries = np.nonzero(np.diff(ep_o))[0] + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [len(order)]))
    cand_s, cand_g = [], []
    for s, e in zip(starts, ends):
        rows = order[s:e]
        if int(ep_idx[rows[0]]) not in test_eps:
            continue
        L = len(rows)
        if L - goal_offset - 1 < 0:
            continue
        t = np.arange(0, L - goal_offset)
        cand_s.append(rows[t]); cand_g.append(rows[t + goal_offset])
    cand_s, cand_g = np.concatenate(cand_s), np.concatenate(cand_g)
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(cand_s), size=min(n, len(cand_s)), replace=False)
    return dict(start_row=cand_s[pick], goal_row=cand_g[pick],
                start_state=setup["proprio"][cand_s[pick]], goal_state=setup["proprio"][cand_g[pick]],
                z_g=setup["z"][cand_g[pick]])


def build_eval_pairs(cfg, setup, n, seed, encode_frame):
    """Returns a list of dict(start_state, goal_state, z_g (1,d) tensor on DEV, start_row,
    goal_row) for the configured protocol/pool -- the only thing run_episodes consumes."""
    mech = ENV_MECHANICS[cfg.env.name]
    if cfg.eval_protocol == "cross_episode":
        test_ei, test_ej, _ = setup["test_eval"]
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(test_ei), size=min(n, len(test_ei)), replace=False)
        s_rows, g_rows = test_ei[idx], test_ej[idx]
        z_g_all = torch.from_numpy(setup["z"][g_rows]).float().to(DEV)
        return [dict(start_state=setup["proprio"][s], goal_state=setup["proprio"][g], z_g=z_g_all[k:k + 1],
                     start_row=int(s), goal_row=int(g))
                for k, (s, g) in enumerate(zip(s_rows, g_rows))]

    if cfg.eval_protocol != "same_episode":
        raise ValueError(f"unknown eval_protocol '{cfg.eval_protocol}'")
    goal_offset = int(cfg.env.paper_goal_offset)
    if cfg.eval_pool == "dataset":
        p = sample_same_episode_pairs_from_h5(mech.h5_path(ROOT), mech.state_h5_key, goal_offset, n, seed,
                                              exclude_eps=setup["train_eps"], mech=mech)
        z_g_all = torch.cat([encode_frame(p["goal_pixels"][b:b + 32]) for b in range(0, len(p["goal_pixels"]), 32)])
    elif cfg.eval_pool == "test_episodes":
        p = sample_same_episode_pairs_from_setup(setup, goal_offset, n, seed)
        z_g_all = torch.from_numpy(p["z_g"]).float().to(DEV)
    else:
        raise ValueError(f"unknown eval_pool '{cfg.eval_pool}'")
    return [dict(start_state=p["start_state"][k], goal_state=p["goal_state"][k], z_g=z_g_all[k:k + 1],
                 start_row=int(p["start_row"][k]), goal_row=int(p["goal_row"][k]))
            for k in range(len(p["start_row"]))]


# ------------------------------------------------------------------
# rollout loop
# ------------------------------------------------------------------

def actor_act_fn(actor_net):
    def act(z_s, z_g):
        with torch.no_grad():
            return torch.tanh(actor_net(torch.cat([z_s, z_g], dim=-1))).cpu().numpy()[0]
    return act


def random_act_fn(env, seed):
    """The paper's Random baseline: uniform actions from the env's action space."""
    env.action_space.seed(int(seed))
    return lambda z_s, z_g: env.action_space.sample()


def run_episodes(env_name, act_fn, env, encode_frame, pairs, seed, max_steps, log_progress=False):
    """pairs: output of build_eval_pairs. act_fn(z_s (1,d), z_g (1,d)) -> action ndarray.

    The goal embedding for each episode is taken from the LIVE env's own reset() info when
    it provides one: PushT's reset() always internally does _set_state(goal_state) +
    render() and returns that image as info['goal'] (see env.py reset()/_get_info()) --
    using it directly is the exact per-episode ground truth and avoids re-deriving the same
    render from a separately-fetched dataset frame or a cached landmark embedding. Falls
    back to the pair's precomputed z_g for environments that don't expose this (Two-Room)."""
    mech = ENV_MECHANICS[env_name]
    episodes = []
    t0 = time.time()
    for k, p in enumerate(pairs):
        ep_seed = int((seed * 100_000 + k) % (2**31 - 1))
        _, info = env.reset(seed=ep_seed, options=mech.reset_options(p["start_state"], p["goal_state"]))
        z_g = encode_frame(np.asarray(info["goal"])[None]) if "goal" in info else p["z_g"]
        success, final_dist, t = False, None, 0
        for t in range(max_steps):
            pixels = env.render()[None]  # (1,H,W,3)
            z_s = encode_frame(pixels)  # (1,d)
            a = act_fn(z_s, z_g)
            obs, reward, terminated, truncated, info = env.step(a)
            success, final_dist = mech.step_result(obs, reward, terminated, truncated, info)
            if success:
                break

        episodes.append(dict(start_row=p["start_row"], goal_row=p["goal_row"], success=bool(success),
                              steps=t + 1, final_dist=final_dist))
        if log_progress and k % 10 == 0:
            log(f"  episode {k}/{len(pairs)}: success={success} steps={t+1} "
                f"final_dist={final_dist:.1f} elapsed={time.time()-t0:.1f}s")
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


def rollout_eval(cfg, actor_net, setup, n_episodes, seed, ckpt_dir, env=None, encode_frame=None):
    """Standalone-use entrypoint (builds env/encoder if not given). actor_net=None runs the
    random-action baseline instead. For repeated calls in one process (e.g. periodic checks
    during training), build env/encoder/pairs once and call run_episodes directly."""
    if env is None:
        env = make_env(cfg.env.name)
    if encode_frame is None:
        encode_frame = make_encoder(ckpt_dir)
    pairs = build_eval_pairs(cfg, setup, n_episodes, seed, encode_frame)
    max_steps = eval_max_steps(cfg)
    log(f"[rollout] protocol={cfg.eval_protocol}"
        f"{' pool=' + cfg.eval_pool + ' goal_offset=' + str(cfg.env.paper_goal_offset) if cfg.eval_protocol == 'same_episode' else ''}"
        f" pairs={len(pairs)} budget={max_steps} steps policy={'random' if actor_net is None else 'actor'}")
    act_fn = random_act_fn(env, seed) if actor_net is None else actor_act_fn(actor_net)
    episodes = run_episodes(cfg.env.name, act_fn, env, encode_frame, pairs, seed, max_steps, log_progress=True)
    return summarize(episodes)
