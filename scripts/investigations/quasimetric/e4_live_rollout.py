"""E4 -- wire the quasimetric head into CEM planning, measure live rollout success rate.

Swaps jepa.py's L2 planning cost (Section 2.2 of docs/quasimetric-jepa-proposal.md) for the
E3-trained quasimetric head, via a minimal monkeypatch of the loaded model's `criterion`
method -- the exact seam stable_worldmodel's CEMSolver calls (get_cost -> criterion, see
stable_worldmodel/solver/cem.py and .../wm/lewm/lewm.py, which is a package-vendored twin of
this repo's own jepa.py -- same class, same method signatures). Nothing about rollout /
goal-encoding / CEM optimization itself changes; only what number scores a candidate.

This is a self-contained script (does not import or modify eval.py/jepa.py) that duplicates
eval.py's plumbing rather than editing it in place, matching how E1-E3 were built --
eval.py belongs to the base train.py pipeline (see CLAUDE.md's "two mostly-separate
codepaths"), and it already carries local uncommitted changes unrelated to this project.

Runs BOTH conditions (raw L2 baseline, quasimetric) in one process against the SAME sampled
eval episodes/start states (identical seed) for a paired comparison, under the paper's live
eval protocol (start = a random dataset state, goal = the state 25 timesteps later in the same
trajectory, 50 env steps to reach it, success = the env's own `terminated` flag) -- the exact
protocol CLAUDE.md's `eval.py policy=lewm-pusht eval.num_eval=50` uses to validate checkpoint
health (96+-2.83% paper, 94% reproduced here).

This is the FIRST stage in the ladder that touches actual control -- E0-E3 all measured
correlation/asymmetry on frozen embeddings, never planning outcomes. No formal gate is set
(docs/quasimetric-jepa-proposal.md's gates stop at E3); this reports whether E3's correlation
gain (and its grounded asymmetry) translates into more successful live rollouts than the L2
baseline it replaces.

Run:
    python scripts/investigations/quasimetric/e4_live_rollout.py
    python scripts/investigations/quasimetric/e4_live_rollout.py eval.num_eval=8   # smoke test
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

ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.envs import ENV_MECHANICS
from common.lewm_loader import load_lewm
from common.log_util import log
from common.quasimetric import LatentQuasimetric
from probe_gate import Arm


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


def recompute_quasimetric_norm_stats(cache_dir, n_episodes, seed, train_frac, grid=16):
    """Reproduces exactly the (mu, sd) Arm.fit_norm computed during E3 training --
    deterministic given the same seed/split, so nothing needed saving from that run."""
    tag = f"ep{n_episodes}_seed{seed}_g{grid}"
    z = np.load(cache_dir / f"z_{tag}.npy", mmap_mode="r")
    ep_idx = np.load(cache_dir / f"ep_idx_{tag}.npy")
    uniq = np.unique(ep_idx)
    shuffled = np.random.default_rng(seed).permutation(uniq)
    n_tr = int(len(shuffled) * train_frac)
    rows_tr = np.nonzero(np.isin(ep_idx, shuffled[:n_tr]))[0]
    arm = Arm("lewm", z, z.shape[1])
    arm.fit_norm(rows_tr, seed)
    return arm.mu[0].copy(), arm.sd[0].copy()


def make_quasimetric_criterion(quasi_model, mu, sd, clip=10.0):
    """Drop-in replacement for jepa.py's criterion(info_dict) -- same inputs/outputs, only the
    scoring function changes (learned asymmetric d_Q instead of symmetric MSE)."""
    mu_t = torch.from_numpy(mu).float()
    sd_t = torch.from_numpy(sd).float()

    def criterion(info_dict):
        pred_emb = info_dict["predicted_emb"]   # (B, S, T-1, dim)
        goal_emb = info_dict["goal_emb"]         # (B, S, T, dim)
        goal_emb = goal_emb[..., -1:, :].expand_as(pred_emb)
        p = pred_emb[..., -1, :]                  # (B, S, dim)
        g = goal_emb[..., -1, :].detach()
        dev = p.device
        pn = torch.clamp((p - mu_t.to(dev)) / sd_t.to(dev), -clip, clip)
        gn = torch.clamp((g - mu_t.to(dev)) / sd_t.to(dev), -clip, clip)
        return quasi_model(pn, gn)   # (B, S)

    return criterion


@hydra.main(version_base=None, config_path=str(ROOT / "config" / "eval"), config_name="pusht")
def main(cfg: DictConfig):
    t0 = time.time()
    mech = ENV_MECHANICS["pusht"]
    res_dir = ROOT / "outputs" / "quasimetric"
    res_dir.mkdir(parents=True, exist_ok=True)

    assert cfg.plan_config.horizon * cfg.plan_config.action_block <= cfg.eval.eval_budget

    log(f"=== E4 live rollout: L2 vs. quasimetric planning cost (num_eval={cfg.eval.num_eval}) ===")

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

    # -- sample eval episodes ONCE, shared by both cost conditions -- a paired comparison
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

    quasi_mu, quasi_sd = recompute_quasimetric_norm_stats(
        Path(os.environ.get("PROBE_CACHE_DIR", str(ROOT / "outputs" / "probe_gate_pusht" / "cache"))),
        n_episodes=18685, seed=0, train_frac=0.8,
    )
    # QUASI_HEAD_NAME lets this same script retest a later head revision (e.g. E3-v2's
    # predictor-rollout-aware head) against the identical L2 baseline/protocol without
    # duplicating the file -- defaults to the original E3 head, so old invocations/results
    # are unaffected.
    head_name = os.environ.get("QUASI_HEAD_NAME", "e3_quasimetric_head_pusht.pt")
    quasi_model = LatentQuasimetric(latent_dim=192, hidden_dim=256, proj_dim=64)
    quasi_model.load_state_dict(
        torch.load(res_dir / head_name, map_location="cpu"))
    quasi_model = quasi_model.to("cuda").eval()
    quasi_model.requires_grad_(False)
    log(f"quasimetric head: {head_name}")

    results = {}
    for cost in ["l2", "quasimetric"]:
        log(f"\n--- condition: cost={cost} ---")
        model = load_lewm(ckpt_dir=ckpt_dir, device="cuda")
        model.interpolate_pos_encoding = True
        if cost == "quasimetric":
            model.criterion = make_quasimetric_criterion(quasi_model, quasi_mu, quasi_sd)

        config = swm.PlanConfig(**cfg.plan_config)
        solver = hydra.utils.instantiate(cfg.solver, model=model)
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
        log(f"[{cost}] success_rate={metrics.get('success_rate')}  ({dt:.1f}s)")
        results[cost] = dict(
            metrics={k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in metrics.items()},
            secs=dt)
        world.close()

    log("\n=== SUMMARY ===")
    log(f"L2 baseline success_rate:      {results['l2']['metrics'].get('success_rate')}")
    log(f"Quasimetric-JEPA success_rate: {results['quasimetric']['metrics'].get('success_rate')}")

    out = dict(num_eval=cfg.eval.num_eval, goal_offset_steps=cfg.eval.goal_offset_steps,
              eval_budget=cfg.eval.eval_budget, seed=cfg.seed, quasi_head=head_name,
              results=results, secs=time.time() - t0)
    suffix = "" if head_name == "e3_quasimetric_head_pusht.pt" else f"_{Path(head_name).stem}"
    p = res_dir / f"e4_live_rollout_pusht_n{cfg.eval.num_eval}{suffix}.json"
    p.write_text(json.dumps(out, indent=2))
    log(f"wrote {p} ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
