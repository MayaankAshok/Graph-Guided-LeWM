"""E5 -- does the CEM-chosen plan's PREDICTED cost track its TRUE physical outcome?

The direct test the E3-v3 writeup flagged as unverified: three increasingly sophisticated
attempts to fix the quasimetric head each made E4's live rollout worse, and the leading
(unconfirmed) hypothesis was "optimizer exploitation" -- CEM's 30-iteration, 300-candidate
search finds action sequences whose PREDICTED cost is low without the resulting state
actually being close to the goal, a failure mode a separately-trained scalar cost is much
more exposed to than plain L2 against a real target embedding.

Method: `config/eval/pusht.yaml`'s plan_config has horizon=5, action_block=5,
receding_horizon=5 (== horizon, so NO partial truncation -- the full planned action sequence
always executes before any replanning) and goal_offset_steps=25 == horizon*action_block. So
calling `world.evaluate(..., eval_budget=25, ...)` triggers EXACTLY ONE CEM solve per env and
executes its ENTIRE chosen plan, with no second replan muddying the picture. Monkeypatching
`policy.solver.solve` captures that one solve's predicted cost (`outputs['costs']`, the
elite-mean cost of the final CEM iteration) as a side effect; `world.infos['state']` after
`world.evaluate()` returns holds each env's TRUE resulting state (this reuses
`world.evaluate`/`_evaluate_from_dataset` entirely unmodified -- the same machinery already
validated by E4 -- rather than hand-rolling env reset/pixel-transform/CEM-batching logic,
which would be easy to get subtly wrong).

For each of 4 conditions (l2, and the v1/v2/v3 quasimetric heads), on the SAME 50 sampled
(start, goal) pairs used throughout E4: compute the real oracle distance from the true
resulting state to the real goal state, and correlate it against the model's own predicted
cost for the plan that produced that state. A well-calibrated cost should show POSITIVE
correlation (higher predicted cost -> genuinely farther from goal). A near-zero or negative
correlation, especially one that gets WORSE from v1 to v3 (mirroring E4's live-rollout
decline), would be direct, non-hypothetical evidence for the optimizer-exploitation account.

Run:
    python scripts/investigations/quasimetric/e5_cost_hacking_diagnostic.py
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

HEAD_NAMES = {
    "l2": None,
    "quasimetric_v1": "e3_quasimetric_head_pusht.pt",
    "quasimetric_v2": "e3v2_quasimetric_head_pusht.pt",
    "quasimetric_v3": "e3v3_quasimetric_head_pusht.pt",
}


@hydra.main(version_base=None, config_path=str(ROOT / "config" / "eval"), config_name="pusht")
def main(cfg: DictConfig):
    t0 = time.time()
    mech = ENV_MECHANICS["pusht"]
    res_dir = ROOT / "outputs" / "quasimetric"
    res_dir.mkdir(parents=True, exist_ok=True)

    assert cfg.plan_config.horizon * cfg.plan_config.action_block == cfg.eval.goal_offset_steps, (
        "E5 relies on exactly one CEM solve covering the full goal_offset -- config drifted")
    eval_budget_one_shot = cfg.plan_config.horizon * cfg.plan_config.action_block

    log(f"=== E5 cost-hacking diagnostic (num_eval={cfg.eval.num_eval}, "
        f"one-shot eval_budget={eval_budget_one_shot}) ===")

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

    # -- SAME eval episode sampling as e4_live_rollout.py (identical seed/logic), so results
    # are directly comparable to E4's success-rate numbers
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

    # real goal states, read directly from the h5 (row = start_row + goal_offset, same
    # episode -- matches _extract_init_goal's own construction) for the oracle distance
    import h5py
    with h5py.File(h5_path, "r", swmr=True) as f5:
        ep_offset = f5["ep_offset"][:]
        state_col = f5["state"]
        ep_ids = np.asarray(eval_episodes).astype(np.int64)
        goal_rows = ep_offset[ep_ids] + np.asarray(eval_start_idx).astype(np.int64) + cfg.eval.goal_offset_steps
        order = np.argsort(goal_rows)
        goal_states = np.empty((len(goal_rows), state_col.shape[1]), dtype=state_col.dtype)
        goal_states[order] = state_col[goal_rows[order]]
        # oracle fit on a broad real-state sample (same convention as graph_gate.py/E3-v3)
        true_dist_oracle = mech.build_true_distance_oracle(np.asarray(state_col[::37]))

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

        captured_costs = {}   # filled by the wrapped solve()
        n_calls = [0]
        real_solve = solver.solve
        def wrapped_solve(info_dict, init_action=None, _real=real_solve, _bucket=captured_costs, _n=n_calls):
            out = _real(info_dict, init_action=init_action)
            # first call per env-batch = the one-shot solve this diagnostic relies on
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
            eval_budget=eval_budget_one_shot,   # exactly one CEM solve, full-horizon execution
            episodes_idx=eval_episodes.tolist(),
            callables=OmegaConf.to_container(cfg.eval.get("callables"), resolve=True),
            video=None,
        )
        dt = time.time() - t_run

        final_state = np.asarray(world.infos["state"])
        if final_state.ndim == 3:   # (n_envs, T, state_dim) if a history dim is kept
            final_state = final_state[:, -1]
        true_dist = true_dist_oracle(final_state, goal_states)
        true_dist = np.diagonal(true_dist) if true_dist.ndim == 2 else true_dist

        pred_cost = captured_costs.get("costs")
        if pred_cost is not None:
            pred_cost = np.asarray(pred_cost).reshape(-1)[:cfg.eval.num_eval]

        rho = float(sps.spearmanr(pred_cost, true_dist).statistic) if pred_cost is not None else None
        log(f"[{cond}] one-shot success_rate={metrics.get('success_rate')}  "
            f"true_dist mean={true_dist.mean():.3f}+-{true_dist.std():.3f}  "
            f"Spearman(pred_cost, true_dist)={rho}  ({dt:.1f}s)")

        results[cond] = dict(
            success_rate=metrics.get("success_rate"),
            true_dist_mean=float(true_dist.mean()), true_dist_std=float(true_dist.std()),
            pred_cost_mean=float(pred_cost.mean()) if pred_cost is not None else None,
            pred_cost_std=float(pred_cost.std()) if pred_cost is not None else None,
            spearman_pred_cost_vs_true_dist=rho,
            true_dist=true_dist.tolist(),
            pred_cost=pred_cost.tolist() if pred_cost is not None else None,
            secs=dt,
        )
        world.close()

    log("\n=== SUMMARY ===")
    for cond in HEAD_NAMES:
        r = results[cond]
        log(f"{cond:>16s}: one-shot success={r['success_rate']:>5}%  "
            f"true_dist={r['true_dist_mean']:.3f}  "
            f"Spearman(cost,true_dist)={r['spearman_pred_cost_vs_true_dist']}")

    out = dict(num_eval=cfg.eval.num_eval, goal_offset_steps=cfg.eval.goal_offset_steps,
              eval_budget=eval_budget_one_shot, seed=cfg.seed, results=results,
              secs=time.time() - t0)
    p = res_dir / f"e5_cost_hacking_diagnostic_pusht_n{cfg.eval.num_eval}.json"
    p.write_text(json.dumps(out, indent=2))
    log(f"wrote {p} ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
