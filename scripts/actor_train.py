"""Unified live-rollout actor training -- config-driven consolidation of
tworoom_b3_actor_train.py/pusht_b3_actor_train.py. Adds an AWR-trained actor on top of B3's
V/Q value learning and evaluates it with real live rollouts, answering "does the value-
ranking proxy convert into real task success." Two independently-tracked peaks (V's
Spearman peak vs. the actor's own live-rollout success-rate peak) -- see module docstrings
in the original pair for the full rationale, unchanged here.

Run as:
    python scripts/actor_train.py env=pusht tier=mixed_large variant=auxphi seed=1
    python scripts/actor_train.py env=pusht tier=mixed_large variant=stitch phi_mode=none seed=1
"""

import os
import pickle
import sys
import time
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, open_dict
from scipy import stats as sps

sys.path.insert(0, str(Path(__file__).resolve().parent))
from actor_rollout_utils import (actor_act_fn, build_eval_pairs, eval_max_steps, make_encoder, make_env,
                                 run_episodes, summarize)
from common.checkpoint_io import load_checkpoint, save_checkpoint
from common.graph_lib import DEV
from common.log_util import log
from common.training import (EMA_TAU, GAMMA, GOAL_GAMMA, GOALS_PER_TRANSITION, HIDDEN, MLP,
                              TAU_EXPECTILE, build_her_tuples, build_stitch_her_tuples, ema_update)
from datatiers import build_tier_setups

ROOT = Path(__file__).resolve().parent.parent

CROSS_PHI_LIMIT = 20.0  # bounded-Dijkstra radius for cross-episode aux pairs (phi_mode=sparse
                         # only -- dense mode just indexes the cached phi_dist matrix directly).
                         # Generous relative to a single identification edge's weight (1.0):
                         # cross_s_idx/cross_g_idx are themselves identification-edge endpoints
                         # (see common.graph_lib.build_cross_episode_id_pairs), so a path of
                         # length <= id_weight always exists and reachability is never the
                         # limiting factor -- this constant only caps how far Dijkstra searches
                         # for a possibly-shorter alternate route.


def _out_dirs(cfg):
    out_dir = ROOT / "outputs" / f"b3_{cfg.env.output_prefix}" / "actor"
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    return out_dir, ckpt_dir


def run_tag(cfg):
    """Optional suffix distinguishing hyperparameter variants of the same (tier, variant)
    -- e.g. run_tag=_hg96 for her_goal_gamma=0.96 -- so they don't collide with the
    default run's checkpoint/result files. Empty by default."""
    return getattr(cfg, "run_tag", "") or ""


def ckpt_path(cfg, tier, variant, seed):
    _, ckpt_dir = _out_dirs(cfg)
    return ckpt_dir / f"{tier}__{variant}{run_tag(cfg)}_actor__s{seed}.pt"


def get_setup(cfg, tier):
    return build_tier_setups(cfg, [tier])[tier]


# Matches ada_sweep.py's own SEEDS constant -- the fixed seed sweep every tier/variant runs.
ALL_SEEDS = (0, 1, 2)


def her_phi_pairs_cached(cache_path, setup, action, cfg, s_idx, goal_idx):
    """phi at exactly this seed's HER (s_idx, goal_idx) pairs, via a small on-disk cache of
    *exact pairs* shared across all seeds training on this tier -- NOT a cache of each
    source's full reachable set (see common.graph_lib.phi_for_union_pairs's docstring for
    why that blew a 20GB job's memory budget on Reacher's densely-connected expert_1000
    graph: a bounded search from a typical source reaches a large fraction of all nodes, so
    "cache the reachable set" is nearly as big as the dense matrix phi_mode=sparse exists to
    avoid).

    HER's source rows are fully seed-independent (build_her_tuples iterates every real
    transition in every training episode regardless of seed; only the goal offset is
    seed-randomized) but goal rows differ per seed, so this seed's own (s_idx, goal_idx)
    pairs won't already be in a cache built by a different seed alone. Instead, on a cache
    miss this rebuilds the cache from the UNION of all ALL_SEEDS' HER pairs (cheap -- pure
    numpy, no Dijkstra) and resolves that whole union in one bounded-Dijkstra pass, so the
    expensive part (one bounded search per unique source -- the union's unique-source count
    is the same as any single seed's, since sources are already seed-independent) still only
    happens once per tier, not once per seed, while the cache itself stays small (bounded by
    how many pairs are actually queried across 3 seeds, not by how many nodes are reachable).

    Concurrent writers (two seeds launched in parallel) aren't coordinated: each may rebuild
    the same union if both start before either finishes, but the result is still correct and
    at most 2x the work, never memory-unsafe -- and any seed launched after either finishes
    gets a fully warm cache with zero Dijkstra calls."""
    cache = {}
    if cache_path.exists():
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)

    keys = list(zip((int(x) for x in s_idx), (int(x) for x in goal_idx)))
    if all(k in cache for k in keys):
        return np.array([cache[k] for k in keys], dtype=np.float32)

    all_s, all_g = [s_idx], [goal_idx]
    for sd in ALL_SEEDS:
        if sd == cfg.seed:
            continue
        ss, _, gg, _, _ = build_her_tuples(
            setup["ep_idx"], setup["step_idx"], action, seed=sd,
            allowed_episode_ids=setup["train_eps"], goal_gamma=cfg.her_goal_gamma,
        )
        all_s.append(ss); all_g.append(gg)
    all_s = np.concatenate(all_s); all_g = np.concatenate(all_g)
    union_keys = set(zip((int(x) for x in all_s), (int(x) for x in all_g))) | set(cache.keys())
    uk_s = np.array([k[0] for k in union_keys], dtype=np.int64)
    uk_g = np.array([k[1] for k in union_keys], dtype=np.int64)

    gap = np.abs(setup["step_idx"][uk_g] - setup["step_idx"][uk_s])
    limit = int(gap.max()) + 1
    log(f"[phi-pairs-cache] {cache_path.name}: building shared cache over {len(union_keys)} "
        f"unique pairs (union of seeds {ALL_SEEDS}), limit={limit}")
    from common.graph_lib import phi_for_union_pairs
    vals = phi_for_union_pairs(setup["graph"], uk_s, uk_g, limit=limit, n_jobs=2)
    cache = {(int(s), int(g)): float(v) for s, g, v in zip(uk_s, uk_g, vals)}

    tmp_path = cache_path.with_suffix(".tmp.pkl")
    with open(tmp_path, "wb") as f:
        pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp_path, cache_path)
    log(f"[phi-pairs-cache] wrote {cache_path.name}: {len(cache)} pairs cached")

    return np.array([cache[k] for k in keys], dtype=np.float32)


def run_actor_condition(cfg, tier, variant, setup, seed, n_steps, eval_every,
                         rollout_eval_every, rollout_eval_episodes):
    path = ckpt_path(cfg, tier, variant, seed)
    z, action = setup["z"], setup["action"]
    phi_dist = setup["phi_dist"]
    d = setup["d"]
    act_dim = action.shape[1]
    aux_lambda = 0.0 if variant in ("baseline", "stitch") else cfg.aux_lambda
    tag = f"{cfg.env.name}/{tier}/{variant}{run_tag(cfg)}"
    if variant == "auxphi" and phi_dist is None and setup.get("graph") is None:
        raise SystemExit(f"[{tag}] variant=auxphi needs graph distances but the setup was built "
                          f"with need_graph=false (use phi_mode=sparse for large tiers)")
    if variant == "stitch" and setup.get("id_i") is None:
        raise SystemExit(f"[{tag}] variant=stitch needs identification edges but the setup was "
                          f"built without them (use phi_mode=none, sparse, or need_graph=true)")

    ckpt = load_checkpoint(path)
    if ckpt is not None and ckpt.get("done"):
        log(f"[{tag}/s{seed}] already complete ({ckpt['step']} steps, "
            f"peak_rho={ckpt.get('peak_rho'):.4f}@{ckpt.get('peak_step')}, "
            f"peak_success_rate={ckpt.get('peak_success_rate')}@{ckpt.get('peak_success_step')}) "
            f"-- skipping")
        return ckpt

    torch.manual_seed(seed)
    z_t = torch.from_numpy(z).to(DEV)
    act_t = torch.from_numpy(action).float().to(DEV)

    v_net = MLP(2 * d, HIDDEN).to(DEV)
    v_target = MLP(2 * d, HIDDEN).to(DEV)
    q_net = MLP(2 * d + act_dim, HIDDEN).to(DEV)
    q_target = MLP(2 * d + act_dim, HIDDEN).to(DEV)
    actor_net = MLP(2 * d, HIDDEN, out_dim=act_dim).to(DEV)
    v_target.load_state_dict(v_net.state_dict())
    q_target.load_state_dict(q_net.state_dict())

    from stable_worldmodel.wm.gcrl.module import ExpectileLoss
    expectile_loss = ExpectileLoss(tau=TAU_EXPECTILE)
    opt = torch.optim.Adam(
        list(v_net.parameters()) + list(q_net.parameters()) + list(actor_net.parameters()), lr=3e-4)

    s_idx, next_idx, goal_idx, act_idx, done_arr = build_her_tuples(
        setup["ep_idx"], setup["step_idx"], action, seed=seed, allowed_episode_ids=setup["train_eps"],
        goal_gamma=cfg.her_goal_gamma,
    )
    n_tuples = len(s_idx)

    if variant == "stitch":
        # No aux loss, no phi, no Dijkstra -- the identification edges instead expand the
        # HER tuple pool itself with cross-episode goals they license (see common.training.
        # build_stitch_her_tuples). Training below is otherwise identical to variant=baseline.
        st_s, st_next, st_goal, st_act, st_done = build_stitch_her_tuples(
            setup["ep_idx"], setup["step_idx"], setup["id_i"], setup["id_j"],
            allowed_episode_ids=setup["train_eps"],
            goals_per_edge=cfg.get("stitch_goals_per_edge", GOALS_PER_TRANSITION),
            hop_gamma=cfg.get("stitch_hop_gamma", GOAL_GAMMA),
            seed=seed,
        )
        log(f"[{tag}/s{seed}] stitch: {n_tuples} same-episode HER tuples + {len(st_s)} "
            f"cross-episode stitched tuples")
        s_idx = np.concatenate([s_idx, st_s])
        next_idx = np.concatenate([next_idx, st_next])
        goal_idx = np.concatenate([goal_idx, st_goal])
        act_idx = np.concatenate([act_idx, st_act])
        done_arr = np.concatenate([done_arr, st_done])
        n_tuples = len(s_idx)

    # phi at exactly the HER pairs, precomputed once. The dense matrix path just gathers from
    # it; the sparse path (large tiers, where the NxN matrix doesn't fit) computes these pairs
    # directly with bounded Dijkstra. Both give the training loop the same phi_her[batch].
    phi_her = None
    if variant == "auxphi":
        if phi_dist is not None:
            phi_her = phi_dist[s_idx, goal_idx].astype(np.float32)
        else:
            from datatiers import _out_dirs as _tier_out_dirs
            _, _, tier_cache_dir, _ = _tier_out_dirs(cfg)
            cache_path = tier_cache_dir / f"{tier}_her_phi_pairs.pkl"
            log(f"[{tag}/s{seed}] sparse phi: {n_tuples} HER pairs (shared pair cache: {cache_path.name})")
            phi_her = her_phi_pairs_cached(cache_path, setup, action, cfg, s_idx, goal_idx)

    # Cross-episode aux batch: the HER-based aux term above only ever reads phi at
    # same-episode pairs (a HER goal is always a future state in the SAME trajectory), so V
    # never sees the graph's opinion about a cross-episode pair during training and has to
    # generalize to cross-episode goals on its own -- even though cross-episode stitching is
    # the graph's whole reason for existing. This adds a second, independent aux term on
    # pairs drawn from the graph's own identification edges that cross an episode boundary
    # (real stitching links the graph itself found, not arbitrary far-apart pairs), so phi
    # stays small/bounded by construction. Off by default (aux_cross_lambda=0) to preserve
    # every existing run's behavior; see [[pusht-within-episode-id-edges]] for why the graph's
    # cross-episode connectivity needed to exist in the first place.
    aux_cross_lambda = float(cfg.get("aux_cross_lambda", 0.0)) if variant == "auxphi" else 0.0
    cross_s_idx = cross_g_idx = phi_cross = None
    if aux_cross_lambda > 0.0:
        from common.graph_lib import build_cross_episode_id_pairs
        cs, cg = build_cross_episode_id_pairs(setup["id_i"], setup["id_j"], setup["ep_idx"])
        train_mask = (np.isin(setup["ep_idx"][cs], setup["train_eps"])
                      & np.isin(setup["ep_idx"][cg], setup["train_eps"]))
        cross_s_idx, cross_g_idx = cs[train_mask], cg[train_mask]
        if len(cross_s_idx) == 0:
            log(f"[{tag}/s{seed}] WARNING: no cross-episode identification edges among train "
                f"episodes -- cross-episode aux term disabled for this run")
            cross_s_idx = cross_g_idx = None
        else:
            if phi_dist is not None:
                phi_cross = phi_dist[cross_s_idx, cross_g_idx].astype(np.float32)
            else:
                from common.graph_lib import phi_for_pairs as _phi_for_pairs
                phi_cross = _phi_for_pairs(setup["graph"], cross_s_idx, cross_g_idx, limit=CROSS_PHI_LIMIT)
            log(f"[{tag}/s{seed}] cross-episode aux pairs: {len(cross_s_idx)}, "
                f"mean phi={phi_cross.mean():.2f} (aux_cross_lambda={aux_cross_lambda})")

    train_ei, train_ej, train_true_d = setup["train_eval"]
    test_ei, test_ej, test_true_d = setup["test_eval"]
    train_zsg = torch.cat([z_t[train_ei], z_t[train_ej]], dim=-1)
    test_zsg = torch.cat([z_t[test_ei], z_t[test_ej]], dim=-1)

    rng = np.random.default_rng(seed + 1)
    history = []
    start_step = 1
    best_rho, best_step, best_actor_state = -2.0, None, None
    best_success_rate, best_success_step, best_success_actor_state = -1.0, None, None

    if ckpt is not None:
        v_net.load_state_dict(ckpt["v_net"]); v_target.load_state_dict(ckpt["v_target"])
        q_net.load_state_dict(ckpt["q_net"]); q_target.load_state_dict(ckpt["q_target"])
        actor_net.load_state_dict(ckpt["actor_net"])
        opt.load_state_dict(ckpt["opt"])
        history = ckpt["history"]
        start_step = ckpt["step"] + 1
        rng.bit_generator.state = ckpt["numpy_rng"]
        torch.set_rng_state(ckpt["torch_rng"])
        if torch.cuda.is_available() and ckpt.get("torch_cuda_rng") is not None:
            torch.cuda.set_rng_state(ckpt["torch_cuda_rng"])
        best_rho = ckpt.get("peak_rho", -2.0)
        best_step = ckpt.get("peak_step")
        best_actor_state = ckpt.get("peak_actor_state")
        best_success_rate = ckpt.get("peak_success_rate", -1.0)
        best_success_step = ckpt.get("peak_success_step")
        best_success_actor_state = ckpt.get("peak_success_actor_state")
        log(f"[{tag}/s{seed}] resuming from step {start_step}/{n_steps} "
            f"({len(history)} eval points so far, best_rho={best_rho:.4f}@{best_step}, "
            f"best_success_rate={best_success_rate}@{best_success_step})")
    else:
        log(f"[{tag}/s{seed}] starting fresh, n_tuples={n_tuples}, variant={variant}, "
            f"beta={cfg.beta}, her_goal_gamma={cfg.her_goal_gamma}, n_steps={n_steps}")

    if start_step > n_steps:
        return ckpt

    rollout_env = make_env(cfg.env.name)
    encode_frame = make_encoder(cfg.env.ckpt_dir_resolved)
    # (start, goal) pairs for the periodic rollout checks, fixed once per run -- protocol per
    # cfg.eval_protocol (see actor_rollout_utils.py). These checks drive the success-selected
    # actor, so the protocol is recorded in the checkpoint for actor_rollout_eval.py to check.
    rollout_pairs = build_eval_pairs(cfg, setup, rollout_eval_episodes, cfg.rollout_eval_pair_seed, encode_frame)
    rollout_max_steps = eval_max_steps(cfg)
    log(f"[{tag}/s{seed}] rollout checks: protocol={cfg.eval_protocol}, {len(rollout_pairs)} pairs, "
        f"budget={rollout_max_steps} steps")

    t0 = time.time()
    for step in range(start_step, n_steps + 1):
        batch = rng.integers(0, n_tuples, 256)
        s, nx, g, a_i, dn = s_idx[batch], next_idx[batch], goal_idx[batch], act_idx[batch], done_arr[batch]

        zs, zn, zg = z_t[s], z_t[nx], z_t[g]
        a = act_t[a_i]
        mask = (~torch.from_numpy(dn).to(DEV)).float().unsqueeze(-1)
        reward = -mask

        sg = torch.cat([zs, zg], dim=-1)
        ng = torch.cat([zn, zg], dim=-1)
        sag = torch.cat([zs, a, zg], dim=-1)

        with torch.no_grad():
            q = q_target(sag)
        v = v_net(sg)
        value_loss = expectile_loss(v, q.detach())

        aux_loss = torch.tensor(0.0, device=DEV)
        if variant == "auxphi":
            phi_sg = torch.from_numpy(phi_her[batch]).float().to(DEV).unsqueeze(-1)
            pseudo_v = -(1 - GAMMA ** phi_sg) / (1 - GAMMA)
            if cfg.aux_standardize:
                # Match only the ORDERING the graph gets right, not its absolute scale.
                # Measured on Push-T mixed_large: graph distance ranks true expert step-gaps
                # at Spearman +0.9952, but pseudo_v's absolute scale (median -5.85) is ~2x
                # the value the Bellman/expectile objective converges to (baseline V median
                # -2.76) -- because graph distance measures how many steps the DATA took
                # while V learns the OPTIMAL cost-to-go, and 78% of this tier is WeakPolicy
                # data that wanders. Regressing raw pseudo_v drags V off the TD fixed point
                # and inverts the actor's data preference (it ends up weighting noisy data
                # 3x MORE than expert data). See investigations/pusht_lowscore/.
                v_n = (v - v.mean()) / (v.std() + 1e-6)
                p_n = (pseudo_v - pseudo_v.mean()) / (pseudo_v.std() + 1e-6)
                aux_loss = ((v_n - p_n) ** 2).mean()
            else:
                aux_loss = ((v - pseudo_v) ** 2).mean()

        aux_loss_cross = torch.tensor(0.0, device=DEV)
        if cross_s_idx is not None:
            cbatch = rng.integers(0, len(cross_s_idx), 256)
            v_cross = v_net(torch.cat([z_t[cross_s_idx[cbatch]], z_t[cross_g_idx[cbatch]]], dim=-1))
            phi_c = torch.from_numpy(phi_cross[cbatch]).float().to(DEV).unsqueeze(-1)
            pseudo_v_cross = -(1 - GAMMA ** phi_c) / (1 - GAMMA)
            if cfg.aux_standardize:
                v_n = (v_cross - v_cross.mean()) / (v_cross.std() + 1e-6)
                p_n = (pseudo_v_cross - pseudo_v_cross.mean()) / (pseudo_v_cross.std() + 1e-6)
                aux_loss_cross = ((v_n - p_n) ** 2).mean()
            else:
                aux_loss_cross = ((v_cross - pseudo_v_cross) ** 2).mean()

        with torch.no_grad():
            next_v = v_target(ng)
            q_tgt = reward + GAMMA * mask * next_v
        q_pred = q_net(sag)
        critic_loss = ((q_pred - q_tgt) ** 2).mean()

        with torch.no_grad():
            q_sa_for_actor = q_net(sag)
            v_s_for_actor = v_net(sg)
            advantage = q_sa_for_actor - v_s_for_actor
            if cfg.awr_normalize_adv:
                # Standard AWR practice, and specifically protective here: any constant
                # drift in V (e.g. from the aux term) shifts every advantage by the same
                # amount and silently rescales every weight. Measured drift on Push-T
                # mixed_large: adv mean -0.689 (baseline) vs -0.236 (auxphi).
                advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-6)
            weight = torch.exp(cfg.beta * advantage).clamp(max=cfg.awr_weight_clip)
        a_pred = torch.tanh(actor_net(sg))
        actor_loss = (weight * ((a_pred - a) ** 2).sum(-1, keepdim=True)).mean()

        loss = (value_loss + critic_loss + actor_loss
                + (aux_lambda * aux_loss if variant == "auxphi" else 0.0)
                + aux_cross_lambda * aux_loss_cross)
        opt.zero_grad()
        loss.backward()
        opt.step()
        ema_update(v_target, v_net, EMA_TAU)
        ema_update(q_target, q_net, EMA_TAU)

        if step % eval_every == 0 or step == 1:
            with torch.no_grad():
                v_train = v_net(train_zsg).squeeze(-1).cpu().numpy()
                v_test = v_net(test_zsg).squeeze(-1).cpu().numpy()
            rho_train, _ = sps.spearmanr(train_true_d, -v_train)
            rho_test, _ = sps.spearmanr(test_true_d, -v_test)
            if rho_test > best_rho:
                best_rho, best_step = rho_test, step
                best_actor_state = {k: t.detach().cpu().clone() for k, t in actor_net.state_dict().items()}
            history.append(dict(step=step, value_loss=float(value_loss.item()),
                                 critic_loss=float(critic_loss.item()), actor_loss=float(actor_loss.item()),
                                 aux_loss=float(aux_loss.item()), aux_loss_cross=float(aux_loss_cross.item()),
                                 spearman_train=float(rho_train), spearman_test=float(rho_test)))
            is_done = step == n_steps

            if step % rollout_eval_every == 0 or step == 1:
                actor_net.eval()
                episodes = run_episodes(cfg.env.name, actor_act_fn(actor_net), rollout_env, encode_frame,
                                         rollout_pairs, seed=cfg.rollout_eval_pair_seed, max_steps=rollout_max_steps)
                actor_net.train()
                roll_summary = summarize(episodes)
                sr = roll_summary["success_rate"]
                if sr >= best_success_rate:  # >= not > -- on a tie, prefer the LATER (more-
                    # trained) checkpoint rather than freezing on whichever step hit that rate
                    # first; with only rollout_eval_episodes samples per check, ties are common
                    # and a later-trained actor is the more useful of two equally-scoring ones.
                    best_success_rate, best_success_step = sr, step
                    best_success_actor_state = {k: t.detach().cpu().clone() for k, t in actor_net.state_dict().items()}
                log(f"[{tag}/s{seed}] step={step} ROLLOUT CHECK: success_rate={sr:.3f} "
                    f"({sum(e['success'] for e in episodes)}/{len(episodes)}) "
                    f"best_success_rate={best_success_rate:.3f}@{best_success_step}")

            save_checkpoint(path, step, v_net, v_target, q_net, q_target, opt, history, rng, is_done,
                             extra=dict(actor_net=actor_net.state_dict(), peak_step=best_step,
                                        peak_rho=best_rho, peak_actor_state=best_actor_state,
                                        peak_success_step=best_success_step, peak_success_rate=best_success_rate,
                                        peak_success_actor_state=best_success_actor_state,
                                        eval_protocol=cfg.eval_protocol))
            if (step // eval_every) % 5 == 0 or step == 1:
                log(f"[{tag}/s{seed}] step={step} vloss={value_loss.item():.3f} "
                    f"closs={critic_loss.item():.3f} aloss={actor_loss.item():.3f} "
                    f"train_rho={rho_train:.4f} test_rho={rho_test:.4f} "
                    f"best_rho={best_rho:.4f}@{best_step} elapsed={time.time()-t0:.1f}s")

    return load_checkpoint(path)


@hydra.main(version_base=None, config_path="../config/graph", config_name="actor")
def main(cfg: DictConfig):
    from common.envs import ENV_MECHANICS

    log(f"device={DEV} env={cfg.env.name}")
    with open_dict(cfg.env):
        cfg.env.ckpt_dir_resolved = str(ENV_MECHANICS[cfg.env.name].ckpt_dir(ROOT))
    setup = get_setup(cfg, cfg.tier)

    ckpt = run_actor_condition(cfg, cfg.tier, cfg.variant, setup, cfg.seed, cfg.n_steps, cfg.eval_every,
                                cfg.rollout_eval_every, cfg.rollout_eval_episodes)
    log(f"[{cfg.env.name}/{cfg.tier}/{cfg.variant}/s{cfg.seed}] done: peak_rho={ckpt.get('peak_rho'):.4f}"
        f"@{ckpt.get('peak_step')}, peak_success_rate={ckpt.get('peak_success_rate')}"
        f"@{ckpt.get('peak_success_step')}, checkpoint={ckpt_path(cfg, cfg.tier, cfg.variant, cfg.seed)}")


if __name__ == "__main__":
    main()
