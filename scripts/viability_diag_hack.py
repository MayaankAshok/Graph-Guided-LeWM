"""Diagnostic: why does pure -log V planning lose to L2 in closed-loop control despite
tying/beating it on open-loop candidate ranking (viability_eval_audit.py)?

Hypothesis: CEM optimizes the criterion directly, so (unlike the fixed candidate pool the
open-loop audit scored) it actively searches for whatever latent the critic rates highest --
including states the critic is simply wrong about. This wraps the criterion to record, at
every solver call, the chosen (lowest-cost) candidate's V AND its L2-distance-to-goal side by
side, plus saves per-episode videos, so we can see concretely whether the viability planner is
converging on states that are far by L2 but rated highly viable (the reward-hacking
signature) versus something more mundane (e.g. a degenerate near-zero-action optimum).

Run:
    python scripts/viability_diag_hack.py eval.num_eval=6 eval.goal_offset_steps=25 \
        eval.eval_budget=50 +viability.h=25
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
from common.lewm_loader import load_lewm
from common.log_util import log
from viability_eval_audit import load_critic
from viability_live_rollout import get_episodes_length, img_transform

TRACE = []


def make_traced_viability_criterion(critic, h, eps=1e-3):
    h_t = None

    def criterion(info_dict):
        nonlocal h_t
        pred_emb = info_dict["predicted_emb"]
        goal_emb = info_dict["goal_emb"][..., -1:, :].expand_as(pred_emb)
        p = pred_emb[..., -1, :]
        g = goal_emb[..., -1, :].detach()
        dev = p.device
        if h_t is None or h_t.device != dev:
            h_t = torch.full(p.shape[:-1], float(h), device=dev)
        mean_v, std_v = critic.prob(p, g, h_t)
        l2 = ((p - g) ** 2).sum(-1)
        cost = -torch.log(mean_v.clamp_min(eps))
        # per env (dim 0), record the chosen (lowest-cost) candidate's stats -- this is what
        # the solver will actually act on once it converges
        best = cost.argmin(dim=-1)
        TRACE.append(dict(
            v=mean_v.gather(-1, best[:, None]).squeeze(-1).detach().cpu().numpy().tolist(),
            v_std=std_v.gather(-1, best[:, None]).squeeze(-1).detach().cpu().numpy().tolist(),
            l2=l2.gather(-1, best[:, None]).squeeze(-1).sqrt().detach().cpu().numpy().tolist(),
            l2_pop_min=l2.min(-1).values.detach().cpu().numpy().tolist(),
            l2_pop_median=l2.median(-1).values.detach().cpu().numpy().tolist(),
        ))
        return cost

    return criterion


@hydra.main(version_base=None, config_path=str(ROOT / "config" / "eval"), config_name="pusht")
def main(cfg: DictConfig):
    mech = ENV_MECHANICS["pusht"]
    res_dir = ROOT / "outputs" / "pusht" / "critic_training" / "diag_hack"
    res_dir.mkdir(parents=True, exist_ok=True)
    h = int(cfg.get("viability", {}).get("h", 25))
    critic_path = cfg.get("viability", {}).get(
        "critic", str(ROOT / "outputs" / "pusht" / "critic_training" / "critic_full_s0" / "critic.pt"))

    log(f"=== diagnostic: viability-only closed loop, h={h}, num_eval={cfg.eval.num_eval} "
        f"offset={cfg.eval.goal_offset_steps} budget={cfg.eval.eval_budget} ===")

    h5_path = str(mech.h5_path(ROOT))
    dataset = swm.data.HDF5Dataset(path=h5_path, keys_to_cache=list(cfg.dataset.keys_to_cache))
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"

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
    eval_episodes = dataset.get_row_data(random_episode_indices)[col_name]
    eval_start_idx = dataset.get_row_data(random_episode_indices)["step_idx"]

    ckpt_dir = mech.ckpt_dir(ROOT)
    critic, cargs = load_critic(critic_path)
    log(f"critic h_max={cargs['h_max']}")

    model = load_lewm(ckpt_dir=ckpt_dir, device="cuda")
    model.interpolate_pos_encoding = True
    model.criterion = make_traced_viability_criterion(critic, h)

    config = swm.PlanConfig(**cfg.plan_config)
    solver = hydra.utils.instantiate(cfg.solver, model=model)
    policy = swm.policy.WorldModelPolicy(solver=solver, config=config, process=process, transform=transform)

    world = swm.World(env_name=cfg.world.env_name, num_envs=cfg.eval.num_eval,
                      max_episode_steps=2 * cfg.eval.eval_budget, image_shape=(224, 224))
    world.set_policy(policy)

    t0 = time.time()
    metrics = world.evaluate(
        dataset=dataset, start_steps=eval_start_idx.tolist(), goal_offset=cfg.eval.goal_offset_steps,
        eval_budget=cfg.eval.eval_budget, episodes_idx=eval_episodes.tolist(),
        callables=OmegaConf.to_container(cfg.eval.get("callables"), resolve=True),
        video=str(res_dir),
    )
    log(f"success_rate={metrics.get('success_rate')} episode_successes={metrics.get('episode_successes')} "
        f"({time.time()-t0:.1f}s)")

    # CEM solves envs SEQUENTIALLY (each call's batch dim is 1 env), so TRACE is one
    # env's whole episode of calls, then the next env's, etc -- not interleaved.
    v_all = np.concatenate([t["v"] for t in TRACE])
    l2_all = np.concatenate([t["l2"] for t in TRACE])
    l2_min_all = np.concatenate([t["l2_pop_min"] for t in TRACE])
    l2_med_all = np.concatenate([t["l2_pop_median"] for t in TRACE])
    n_total = len(v_all)
    per_env = n_total // cfg.eval.num_eval
    log(f"n_calls={n_total}  ({per_env} per env, assuming equal split across {cfg.eval.num_eval} envs)")
    log(f"OVERALL: chosen-V mean={v_all.mean():.3f} min={v_all.min():.3f} | "
        f"population median-L2 mean={l2_med_all.mean():.1f} (cross-episode ceiling ~19.7) | "
        f"corr(chosen-V, chosen-L2)={np.corrcoef(v_all, l2_all)[0,1]:+.3f} | "
        f"frac(chosen-V>0.9 AND pop-median-L2>50)={float(((v_all>0.9)&(l2_med_all>50)).mean()):.2f}")
    for env_i in range(cfg.eval.num_eval):
        sl = slice(env_i * per_env, (env_i + 1) * per_env)
        succ = bool(np.asarray(metrics["episode_successes"])[env_i])
        v, l2, l2_min, l2_med = v_all[sl], l2_all[sl], l2_min_all[sl], l2_med_all[sl]
        log(f"  env {env_i} success={succ}  chosen-V: {np.round(v, 3).tolist()}")
        log(f"           chosen-L2: {np.round(l2, 2).tolist()}")
        log(f"           pop median-L2: {np.round(l2_med, 1).tolist()}  pop min-L2: {np.round(l2_min, 1).tolist()}")

    out = dict(v=v_all.tolist(), l2=l2_all.tolist(), l2_min=l2_min_all.tolist(),
              l2_med=l2_med_all.tolist(), per_env=per_env,
              episode_successes=[bool(x) for x in metrics["episode_successes"]],
              success_rate=float(metrics.get("success_rate")))
    (res_dir / "trace.json").write_text(json.dumps(out, indent=2))
    log(f"wrote {res_dir / 'trace.json'} and {res_dir}/episode_*.mp4")


if __name__ == "__main__":
    main()
