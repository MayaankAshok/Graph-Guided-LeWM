"""Experiment 6 (viability-value.tex sec:experiments) -- closed-loop success rate: does the
trained viability critic, wired in as LeWM's planning cost, beat plain latent-L2 MPC on the
env's own success predicate, not just on ranking correlation?

Every prior result in this repo's viability_* pipeline is either self-evaluation (held-out
hindsight labels) or open-loop candidate scoring against a privileged oracle
(viability_eval_audit.py). Neither one actually plans: a real MPC controller replans on a
receding horizon, so a candidate that scores well under one cost can lead to a different
state than the one another cost would have reached. This script is the first one that steps
the real environment closed-loop and reports the env's native `terminated` success rate.

Uses the established `model.criterion` monkeypatch seam called by CEM through `get_cost`.
Only the criterion and critic-loading code differ from the standard planner; the CEM solver,
WorldModelPolicy, World.evaluate machinery, and paired same-seed episode sampling are shared.

Costs (`viability.conditions`, comma-separated):
    l2            LeWM's own MSE(z_pred, z_goal)                          -- the baseline
    viability     -log(max(eps, V(z_pred, z_goal, h)))                    -- eq. (lv-cost)
    hybrid        L2 + beta * (-log V)                                     -- eq. (hybrid), raw scale
    eht           E[T] in predictor steps straight from the hitting-time head's pmf
                  (viability_train_ht.py critics only; '> B_max' counted as B_max + 1)
    et            sum_{h in grid} (1 - V(z, g, h))  ~ expected hitting time (restricted mean
                  survival time over the critic's h grid). Does not saturate when every
                  candidate is "viable within h" -- the failure mode of -log V at a loose h.
    hybrid_std    std(L2) + std(-log V(h))    proposal sec. planner-integration: each cost
    hybrid_et_std std(L2) + std(et)           robustly standardised by median/IQR fitted on
                  an OFFLINE candidate pool of model-generated endpoints (the Exp-2 audit's
                  z_pred bank for the same goal offset; no oracle field is read).
    continuation  amortized inverse-CEM residual from the final three predicted latents
    continuation_hybrid  standardized L2 + weight * continuation

Horizon (`viability.h_mode`):
    fixed      h = viability.h on every plan call (the historical behaviour; at the LAST plan
               call of a 50-step episode this asks "reachable within 25 MORE steps" when the
               true remaining budget is 0, so the cost is flat exactly when it must decide).
    remaining  h = min(h_max, max(0, B - t - H_p*5)), eq. (remaining-budget), clipped to the
               critic's training grid: the plan-call index is counted by wrapping
               solver.solve (all live envs replan synchronously every
               receding_horizon*action_block steps), so t = 25 * n_calls here.

Run (matches E4's protocol structure; num_eval must be <= the eval_budget-vs-horizon-valid
episode count):
    python scripts/viability_live_rollout.py eval.num_eval=50
    python scripts/viability_live_rollout.py eval.num_eval=50 eval.goal_offset_steps=50 \
        eval.eval_budget=100 viability.h=35   # the offset where the ranking win is largest
    python scripts/viability_live_rollout.py eval.num_eval=50 viability.h_mode=remaining \
        viability.conditions=viability,et,hybrid,hybrid_std,hybrid_et_std
"""

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import json
import sys
import time
from pathlib import Path

import hydra
import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from common.envs import ENV_MECHANICS
from common.continuation import continuation_criterion, load_continuation_head
from common.lewm_loader import load_lewm
from common.log_util import log
from viability_eval_audit import load_critic


def img_transform(cfg):
    return transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(**spt.data.dataset_stats.ImageNet),
        transforms.Resize(size=cfg.eval.img_size),
    ])


def get_episodes_length(dataset, episodes):
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    episode_idx = dataset.get_col_data(col_name)
    step_idx = dataset.get_col_data("step_idx")
    return np.array([np.max(step_idx[episode_idx == e]) + 1 for e in episodes])


class PlanClock:
    """Tracks the plan-call index so the critic can be asked about the TRUE remaining budget,
    eq. (remaining-budget): h_rem = max(0, B - t - H_p*5). Wraps `solver.solve`; every call
    advances t by receding_horizon*action_block env steps (WorldModelPolicy replans all live
    envs together whenever their action buffers empty, so one solve == one time step for
    every env). `fixed` mode reproduces the historical behaviour (h constant)."""

    def __init__(self, mode, h_fixed, budget, plan_cfg, h_cap):
        self.mode, self.h_fixed, self.budget, self.h_cap = mode, h_fixed, budget, h_cap
        self.step_per_call = plan_cfg.receding_horizon * plan_cfg.action_block
        self.horizon_steps = plan_cfg.horizon * plan_cfg.action_block
        self.n_calls = 0
        self.h_log = []

    @property
    def h(self):
        if self.mode == "fixed":
            return self.h_fixed
        t = self.n_calls * self.step_per_call
        # clipped to the largest horizon the critic was trained on (proposal: "horizon inputs
        # are clipped only to the largest horizon used in training")
        return min(max(0, self.budget - t - self.horizon_steps), self.h_cap)

    def attach(self, solver):
        orig = solver.solve

        def solve(info_dict, init_action=None):
            self.h_log.append(self.h)
            out = orig(info_dict, init_action=init_action)
            self.n_calls += 1
            return out

        solver.solve = solve
        self.n_calls = 0
        self.h_log = []


def _split(info_dict):
    pred_emb = info_dict["predicted_emb"]        # (B, S, T-1, dim)
    goal_emb = info_dict["goal_emb"][..., -1:, :].expand_as(pred_emb)
    return pred_emb[..., -1, :], goal_emb[..., -1, :].detach()   # (B, S, dim) each


@torch.no_grad()
def cost_neglogv(critic, p, g, h, eps=1e-3):
    h_t = torch.full(p.shape[:-1], float(h), device=p.device)
    mean_v, _ = critic.prob(p, g, h_t)
    return -torch.log(mean_v.clamp_min(eps))


@torch.no_grad()
def cost_et(critic, p, g, h_grid):
    """sum_h (1 - V(z, g, h)) over the critic's training grid -- a restricted-mean-survival-time
    estimate of the hitting time in grid units (0 = already there, len(grid) = never within
    h_max). Independent of the remaining budget."""
    out = torch.zeros(p.shape[:-1], device=p.device)
    for h in h_grid:
        mean_v, _ = critic.prob(p, g, torch.full(p.shape[:-1], float(h), device=p.device))
        out += 1.0 - mean_v
    return out


def cost_l2(p, g):
    return ((p - g) ** 2).sum(-1)


def fit_standardizer(critic, audit_npz, h_grid, eps=1e-3):
    """median / IQR of each cost on an offline pool of model-generated endpoints (the Exp-2
    audit's z_pred bank; only z_pred / z_goal are read -- no oracle field). Returns
    {'l2': (med, iqr), 'et': (med, iqr), 'v': {h: (med, iqr)}}."""
    d = np.load(audit_npz)
    z = torch.as_tensor(d["z_pred"], device="cuda")                      # (n, K, D)
    g = torch.as_tensor(d["z_goal"], device="cuda")[:, None].expand_as(z)
    stats = {}

    def mi(x):
        x = x.flatten().cpu().numpy()
        q1, q2, q3 = np.percentile(x, [25, 50, 75])
        return float(q2), float(max(q3 - q1, 1e-8))

    stats["l2"] = mi(cost_l2(z, g))
    stats["et"] = mi(cost_et(critic, z, g, h_grid))
    stats["v"] = {int(h): mi(cost_neglogv(critic, z, g, h, eps)) for h in h_grid}
    return stats


def make_criterion(kind, critic, clock, beta, h_grid, std=None, eps=1e-3):
    """kind in {viability, hybrid, et, hybrid_std, hybrid_et_std}; `clock.h` is read on every
    call so the horizon follows the plan-call index in `remaining` mode."""

    def criterion(info_dict):
        p, g = _split(info_dict)
        h = clock.h
        if kind == "viability":
            return cost_neglogv(critic, p, g, h, eps)
        if kind == "hybrid":
            return cost_l2(p, g) + beta * cost_neglogv(critic, p, g, h, eps)
        if kind == "et":
            return cost_et(critic, p, g, h_grid)
        if kind == "eht":                       # hitting-time head only: E[T] from the pmf
            return critic.expected_bins(p, g)
        if kind == "hybrid_std":
            hk = min(std["v"], key=lambda k: abs(k - h))
            return ((cost_l2(p, g) - std["l2"][0]) / std["l2"][1]
                    + (cost_neglogv(critic, p, g, h, eps) - std["v"][hk][0]) / std["v"][hk][1])
        if kind == "hybrid_et_std":
            return ((cost_l2(p, g) - std["l2"][0]) / std["l2"][1]
                    + (cost_et(critic, p, g, h_grid) - std["et"][0]) / std["et"][1])
        raise ValueError(kind)

    return criterion


@hydra.main(version_base=None, config_path=str(ROOT / "config" / "eval"), config_name="pusht")
def main(cfg: DictConfig):
    t0 = time.time()
    mech = ENV_MECHANICS["pusht"]
    res_dir = ROOT / "outputs" / "pusht" / "critic_training"
    res_dir.mkdir(parents=True, exist_ok=True)
    vcfg = cfg.get("viability", {})
    h = int(vcfg.get("h", 25))
    h_mode = str(vcfg.get("h_mode", "fixed"))
    beta = float(vcfg.get("beta", 20.0))
    raw_conditions = vcfg.get("conditions", "l2,viability,hybrid")
    conditions = (list(raw_conditions) if not isinstance(raw_conditions, str)
                  else raw_conditions.split(","))
    critic_path = vcfg.get("critic", str(ROOT / "outputs" / "pusht" / "critic_training" / "critic_full_s0" / "critic.pt"))
    audit_npz = vcfg.get("audit_npz", str(ROOT / "outputs" / "pusht" / "experiments" / "viability_exp2" /
                                         f"raw_offset{cfg.eval.goal_offset_steps}.npz"))
    run_tag = str(vcfg.get("tag", ""))
    continuation_path = vcfg.get(
        "continuation_head", str(ROOT / "outputs/pusht/experiments/continuation_head/continuation_head.pt"))
    continuation_weight = float(vcfg.get("continuation_weight", 1.0))
    continuation_kappa = float(vcfg.get("continuation_kappa", 1.0))
    assert h_mode in ("fixed", "remaining"), h_mode

    assert cfg.plan_config.horizon * cfg.plan_config.action_block <= cfg.eval.eval_budget

    log(f"=== viability live rollout: h_mode={h_mode} (h={h} if fixed) conditions={conditions} "
        f"num_eval={cfg.eval.num_eval} offset={cfg.eval.goal_offset_steps} "
        f"budget={cfg.eval.eval_budget} ===")

    h5_path = str(mech.h5_path(ROOT))
    dataset = swm.data.HDF5Dataset(path=h5_path, keys_to_cache=list(cfg.dataset.keys_to_cache))
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    log(f"dataset: {h5_path}")

    process = {}
    for col in cfg.dataset.keys_to_cache:
        if col == "pixels":
            continue
        processor = preprocessing.StandardScaler()
        col_data = dataset.get_col_data(col)
        col_data = col_data[~np.isnan(col_data).any(axis=1)]
        processor.fit(col_data)
        process[col] = processor
        if col != "action":
            process[f"goal_{col}"] = process[col]

    transform = {"pixels": img_transform(cfg), "goal": img_transform(cfg)}

    # -- sample eval episodes ONCE, shared by both cost conditions -- a paired comparison,
    # identical to eval.py's/e4's sampling formula/seed so results are comparable across scripts
    ep_indices, _ = np.unique(dataset.get_col_data(col_name), return_index=True)
    episode_len = get_episodes_length(dataset, ep_indices)
    max_start_idx = episode_len - cfg.eval.goal_offset_steps - 1
    max_start_idx_dict = {ep_id: max_start_idx[i] for i, ep_id in enumerate(ep_indices)}
    max_start_per_row = np.array([max_start_idx_dict[e] for e in dataset.get_col_data(col_name)])
    valid_mask = dataset.get_col_data("step_idx") <= max_start_per_row
    valid_indices = np.nonzero(valid_mask)[0]
    log(f"{valid_mask.sum()} valid starting points found for evaluation.")

    g = np.random.default_rng(cfg.seed)
    random_episode_indices = g.choice(len(valid_indices) - 1, size=cfg.eval.num_eval, replace=False)
    random_episode_indices = np.sort(valid_indices[random_episode_indices])
    eval_episodes = dataset.get_row_data(random_episode_indices)[col_name]
    eval_start_idx = dataset.get_row_data(random_episode_indices)["step_idx"]
    if len(eval_episodes) < cfg.eval.num_eval:
        raise ValueError("Not enough episodes with sufficient length for evaluation.")

    ckpt_dir = mech.ckpt_dir(ROOT)
    continuation_kinds = {"continuation", "continuation_hybrid"}
    viability_kinds = {"viability", "hybrid", "et", "eht", "hybrid_std", "hybrid_et_std"}
    critic = None
    cargs = {"h_max": 50, "members": 0}
    if any(cost in viability_kinds for cost in conditions):
        critic, cargs = load_critic(critic_path)
        log(f"critic: {critic_path} (kind={type(critic).__name__}, h_max={cargs['h_max']}, "
            f"members={cargs.get('members', 1)})")
    continuation_head = None
    if any(cost in continuation_kinds for cost in conditions):
        continuation_head, continuation_meta = load_continuation_head(continuation_path)
        log(f"continuation head: {continuation_path} "
            f"(best_step={continuation_meta.get('best_step')}, "
            f"validation_rho={continuation_meta.get('best_validation_spearman', float('nan')):.3f}, "
            f"kappa={continuation_kappa})")
    if h_mode == "fixed" and h > cargs["h_max"]:
        log(f"[warn] h={h} exceeds critic h_max={cargs['h_max']}; extrapolating past training grid")
    h_grid = list(range(0, cargs["h_max"] + 1, 5))
    std = None
    if any(c.endswith("_std") for c in conditions):
        std = fit_standardizer(critic, audit_npz, h_grid)
        log(f"standardizer from {audit_npz}: l2 med/iqr={std['l2'][0]:.1f}/{std['l2'][1]:.1f} "
            f"et={std['et'][0]:.2f}/{std['et'][1]:.2f} "
            f"-logV: " + " ".join(f"h{k}={v[0]:.2f}/{v[1]:.2f}" for k, v in std["v"].items()))

    results = {}
    for cost in conditions:
        log(f"\n--- condition: cost={cost} ---")
        model = load_lewm(ckpt_dir=ckpt_dir, device="cuda")
        model.interpolate_pos_encoding = True
        clock = PlanClock(h_mode, h, cfg.eval.eval_budget, cfg.plan_config, h_cap=cargs["h_max"])
        if cost in continuation_kinds:
            model.criterion = continuation_criterion(
                continuation_head, cost, continuation_weight, continuation_kappa)
        elif cost != "l2":
            model.criterion = make_criterion(cost, critic, clock, beta, h_grid, std)

        config = swm.PlanConfig(**cfg.plan_config)
        solver = hydra.utils.instantiate(cfg.solver, model=model)
        clock.attach(solver)
        policy = swm.policy.WorldModelPolicy(
            solver=solver, config=config, process=process, transform=transform)

        world = swm.World(env_name=cfg.world.env_name, num_envs=cfg.eval.num_eval,
                          max_episode_steps=2 * cfg.eval.eval_budget, image_shape=(224, 224))
        world.set_policy(policy)

        t_run = time.time()
        metrics = world.evaluate(
            dataset=dataset,
            start_steps=eval_start_idx.tolist(),
            goal_offset=cfg.eval.goal_offset_steps,
            eval_budget=cfg.eval.eval_budget,
            episodes_idx=eval_episodes.tolist(),
            callables=OmegaConf.to_container(cfg.eval.get("callables"), resolve=True),
            video=None,
        )
        dt = time.time() - t_run
        log(f"[{cost}] success_rate={metrics.get('success_rate')}  ({dt:.1f}s)  "
            f"plan calls={clock.n_calls} h per call={clock.h_log}")
        results[cost] = dict(
            metrics={k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in metrics.items()},
            secs=dt, plan_calls=clock.n_calls, h_per_call=clock.h_log)
        world.close()

    log("\n=== SUMMARY ===")
    for cost in conditions:
        log(f"{cost} success_rate: {results[cost]['metrics'].get('success_rate')}")

    out = dict(num_eval=cfg.eval.num_eval, goal_offset_steps=cfg.eval.goal_offset_steps,
              eval_budget=cfg.eval.eval_budget, seed=cfg.seed, h=h, h_mode=h_mode, beta=beta,
              conditions=conditions, critic_path=critic_path, audit_npz=audit_npz,
              continuation_head=continuation_path, continuation_weight=continuation_weight,
              continuation_kappa=continuation_kappa,
              standardizer=std, results=results, secs=time.time() - t0)
    tag = "_".join(conditions)
    hname = f"h{h}" if h_mode == "fixed" else "hrem"
    p = res_dir / f"live_rollout_pusht_n{cfg.eval.num_eval}_off{cfg.eval.goal_offset_steps}_{hname}_{tag}{run_tag}.json"
    p.write_text(json.dumps(out, indent=2))
    log(f"wrote {p} ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
