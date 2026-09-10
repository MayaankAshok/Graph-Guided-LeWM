"""E11 -- does the LEARNED CEM's predicted cost track TRUE reachability (E10's Oracle-CEM),
not just the cheap Euclidean proxy E5 checked it against?

E5 (e5_cost_hacking_diagnostic.py) correlates each condition's (l2 baseline, quasimetric
v1/v2/v3) one-shot-CEM predicted cost against `build_true_distance_oracle`'s Euclidean-proxy
distance from the ACTUAL resulting state to the goal. E10 (e10_oracle_cem.py) showed that
proxy itself only reaches Spearman rho=0.607 against genuine ground-truth min-actions-to-goal
(J^oracle, a 38k-real-physics-rollout CEM search per pair) -- so a good E5 correlation number
doesn't confirm the learned cost actually tracks reachability; it could just track the proxy's
own blind spots. This script reuses E5's exact one-shot-CEM machinery (same 200 pairs, same
seed, same 4 conditions) but correlates each condition's pred_cost DIRECTLY against E10's
oracle_steps (loaded from its per-shard result files, joined by pair index -- both scripts use
the identical eval-pair sampling formula/seed, so pair i in one is pair i in the other).

Needs outputs/quasimetric/e10_oracle_cem_pusht_n{num_eval}_shard*of*.jsonl already present
(produced by running e10_oracle_cem.py, sharded, first) with the SAME num_eval/seed.

Run (matches E10's num_eval=200 default from the sweep already run):
    python scripts/investigations/quasimetric/e11_oracle_vs_learned_cem.py eval.num_eval=200
"""

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import glob
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
from scipy import stats as sps
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms

ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.envs import ENV_MECHANICS
from common.lewm_loader import load_lewm
from common.log_util import log
from common.quasimetric import LatentQuasimetric
from e4_live_rollout import (get_episodes_length, img_transform,
                             make_quasimetric_criterion, recompute_quasimetric_norm_stats)
from e5_cost_hacking_diagnostic import HEAD_NAMES


def load_oracle_results(res_dir, num_eval):
    """Join E10's per-shard .jsonl outputs into a (num_eval,) array of oracle_steps (NaN
    where the oracle never succeeded) plus the euclid-proxy distance E10 already computed,
    indexed by pair_id -- both scripts share the identical eval-pair sampling formula/seed."""
    pattern = str(res_dir / f"e10_oracle_cem_pusht_n{num_eval}_shard*of*.jsonl")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(
            f"no E10 oracle results found matching {pattern} -- run e10_oracle_cem.py "
            f"(sharded) with eval.num_eval={num_eval} first")
    oracle_steps = np.full(num_eval, np.nan)
    euclid_proxy = np.full(num_eval, np.nan)
    n_loaded = 0
    for f in files:
        for line in open(f):
            if not line.strip():
                continue
            row = json.loads(line)
            pid = row["pair_id"]
            oracle_steps[pid] = row["oracle_steps"] if row["oracle_steps"] is not None else np.nan
            euclid_proxy[pid] = row["euclid_proxy_dist"]
            n_loaded += 1
    if n_loaded < num_eval:
        log(f"WARNING: only {n_loaded}/{num_eval} oracle pairs found across {len(files)} shard "
            f"files -- some pairs will be dropped from the correlation (missing shard/still running?)")
    return oracle_steps, euclid_proxy


@hydra.main(version_base=None, config_path=str(ROOT / "config" / "eval"), config_name="pusht")
def main(cfg: DictConfig):
    t0 = time.time()
    mech = ENV_MECHANICS["pusht"]
    res_dir = ROOT / "outputs" / "quasimetric"
    res_dir.mkdir(parents=True, exist_ok=True)

    assert cfg.plan_config.horizon * cfg.plan_config.action_block == cfg.eval.goal_offset_steps, (
        "E11 relies on exactly one CEM solve covering the full goal_offset -- config drifted")
    eval_budget_one_shot = cfg.plan_config.horizon * cfg.plan_config.action_block

    log(f"=== E11 oracle vs. learned-CEM cost (num_eval={cfg.eval.num_eval}, "
        f"one-shot eval_budget={eval_budget_one_shot}) ===")

    oracle_steps, euclid_proxy = load_oracle_results(res_dir, cfg.eval.num_eval)
    oracle_solved = ~np.isnan(oracle_steps)
    log(f"loaded E10 oracle results: {oracle_solved.sum()}/{cfg.eval.num_eval} pairs solved by "
        f"the oracle, mean oracle_steps={np.nanmean(oracle_steps):.2f}")

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

    # -- SAME eval episode sampling as e4/e5/e10 (identical seed/logic) -- pair i here IS
    # pair i in E10's output, by construction.
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
    quasi_mu, quasi_sd = recompute_quasimetric_norm_stats(
        Path(os.environ.get("PROBE_CACHE_DIR", str(ROOT / "outputs" / "probe_gate_pusht" / "cache"))),
        n_episodes=18685, seed=0, train_frac=0.8,
    )

    results = {}
    for cond, head_name in HEAD_NAMES.items():
        log(f"\n--- condition: {cond} ---")
        model = load_lewm(ckpt_dir=ckpt_dir, device="cuda")
        model.interpolate_pos_encoding = True

        if head_name is not None:
            quasi_model = LatentQuasimetric(latent_dim=192, hidden_dim=256, proj_dim=64)
            quasi_model.load_state_dict(torch.load(res_dir / head_name, map_location="cpu"))
            quasi_model = quasi_model.to("cuda").eval()
            quasi_model.requires_grad_(False)
            model.criterion = make_quasimetric_criterion(quasi_model, quasi_mu, quasi_sd)

        config = swm.PlanConfig(**cfg.plan_config)
        solver = hydra.utils.instantiate(cfg.solver, model=model)

        captured_costs = {}
        n_calls = [0]
        real_solve = solver.solve
        def wrapped_solve(info_dict, init_action=None, _real=real_solve, _bucket=captured_costs, _n=n_calls):
            out = _real(info_dict, init_action=init_action)
            if _n[0] == 0:
                _bucket["costs"] = np.asarray(out["costs"])
            _n[0] += 1
            return out
        solver.solve = wrapped_solve

        policy = swm.policy.WorldModelPolicy(
            solver=solver, config=config, process=process, transform=transform)

        world = swm.World(env_name=cfg.world.env_name, num_envs=cfg.eval.num_eval,
                          max_episode_steps=2 * eval_budget_one_shot, image_shape=(224, 224))
        world.set_policy(policy)

        t_run = time.time()
        metrics = world.evaluate(
            dataset=dataset,
            start_steps=eval_start_idx.tolist(),
            goal_offset=cfg.eval.goal_offset_steps,
            eval_budget=eval_budget_one_shot,
            episodes_idx=eval_episodes.tolist(),
            callables=OmegaConf.to_container(cfg.eval.get("callables"), resolve=True),
            video=None,
        )
        dt = time.time() - t_run
        world.close()

        pred_cost = captured_costs.get("costs")
        pred_cost = np.asarray(pred_cost).reshape(-1)[:cfg.eval.num_eval] if pred_cost is not None else None

        rho_oracle = rho_euclid = None
        mask = np.zeros(0, dtype=bool)
        if pred_cost is not None:
            mask = oracle_solved
            if mask.sum() > 2:
                rho_oracle = float(sps.spearmanr(pred_cost[mask], oracle_steps[mask]).statistic)
            euclid_mask = ~np.isnan(euclid_proxy)
            if euclid_mask.sum() > 2:
                rho_euclid = float(sps.spearmanr(pred_cost[euclid_mask], euclid_proxy[euclid_mask]).statistic)

        log(f"[{cond}] one-shot success_rate={metrics.get('success_rate')}  "
            f"Spearman(pred_cost, ORACLE_steps)={rho_oracle}  "
            f"Spearman(pred_cost, euclid_proxy)={rho_euclid}  ({dt:.1f}s)")

        results[cond] = dict(
            success_rate=metrics.get("success_rate"),
            spearman_pred_cost_vs_oracle_steps=rho_oracle,
            spearman_pred_cost_vs_euclid_proxy=rho_euclid,
            n_pairs_used=int(mask.sum()),
            pred_cost_mean=float(pred_cost.mean()) if pred_cost is not None else None,
            pred_cost_std=float(pred_cost.std()) if pred_cost is not None else None,
            pred_cost=pred_cost.tolist() if pred_cost is not None else None,
            secs=dt,
        )

    log("\n=== SUMMARY ===")
    log(f"{'condition':>16s}  {'success%':>9s}  {'rho(pred,ORACLE)':>18s}  {'rho(pred,euclid)':>17s}")
    for cond in HEAD_NAMES:
        r = results[cond]
        log(f"{cond:>16s}  {str(r['success_rate']):>9s}  "
            f"{str(r['spearman_pred_cost_vs_oracle_steps']):>18s}  "
            f"{str(r['spearman_pred_cost_vs_euclid_proxy']):>17s}")

    out = dict(num_eval=cfg.eval.num_eval, goal_offset_steps=cfg.eval.goal_offset_steps,
              eval_budget=eval_budget_one_shot, seed=cfg.seed, results=results,
              secs=time.time() - t0)
    p = res_dir / f"e11_oracle_vs_learned_cem_pusht_n{cfg.eval.num_eval}.json"
    p.write_text(json.dumps(out, indent=2))
    log(f"wrote {p} ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
