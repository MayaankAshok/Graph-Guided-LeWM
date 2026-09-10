"""E7 -- does CEM's search itself behave differently (not just end up differently) under the
quasimetric cost vs. L2?

E6 ruled out the two architectural hypotheses raised in the E3-v3/E5 writeup: the psi
projection is NOT a lossy bottleneck (it beats raw 192-dim L2 at tracking true distance on
the same broad candidate population, 0.91 vs 0.58 Spearman) and the phi asymmetric term is
NOT creating a rough landscape by being frequently active -- it is ZERO for 93.7% of
candidates, near-inert. The per-candidate cost is, if anything, MORE accurate under d_Q than
under raw L2 on a broad population. Yet E4/E5 show CEM under d_Q still converges to
genuinely worse live outcomes. That combination -- accurate pointwise cost, worse
optimization result -- is the signature of an optimizer-level failure, not a cost-accuracy
failure: CEM's elite-averaging update implicitly assumes a roughly UNIMODAL "good" region: it
fits a new Gaussian mean/variance to the top-k elites every iteration, which only works if
those elites cluster into one basin. If d_Q's low-cost SET is multi-modal (several
disconnected good regions, e.g. corresponding to different valid pushing strategies) while
raw L2's happens to be more unimodal, CEM's mean-averaging would blend elites from different
modes into an action sequence that is not close to any of them -- worse than either mode
alone, even though each individual candidate's cost was scored accurately.

This is directly observable without any new ground-truth data: `stable_worldmodel`'s
CEMSolver already ships `VarNormRecorder` (does the search variance actually shrink/converge
over the 30 iterations?) and `EliteSpreadRecorder` (how spread out -- in ACTION space -- are
the top-k elites at each iteration?) callbacks. Attach both, run one real CEM solve per
episode (identical setup to E5) for L2 and each quasimetric head, and compare the
iteration-by-iteration trajectories. A flat or non-shrinking elite spread under d_Q where L2's
shrinks cleanly is direct evidence for the multi-modal-averaging account; if both shrink
similarly, that hypothesis is not supported either and the explanation is still open.

Run:
    python scripts/investigations/quasimetric/e7_cem_convergence_trace.py
"""

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import json
import sys
import time
from pathlib import Path

import hydra
import numpy as np
import stable_worldmodel as swm
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn import preprocessing

ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.envs import ENV_MECHANICS
from common.lewm_loader import load_lewm
from common.log_util import log
from common.quasimetric import LatentQuasimetric
from e4_live_rollout import (get_episodes_length, img_transform,
                             make_quasimetric_criterion, recompute_quasimetric_norm_stats)
from stable_worldmodel.solver.callbacks import BestCostRecorder, EliteSpreadRecorder, VarNormRecorder

HEAD_NAMES = {
    "l2": None,
    "quasimetric_v1": "e3_quasimetric_head_pusht.pt",
    "quasimetric_v3": "e3v3_quasimetric_head_pusht.pt",
}


@hydra.main(version_base=None, config_path=str(ROOT / "config" / "eval"), config_name="pusht")
def main(cfg: DictConfig):
    t0 = time.time()
    mech = ENV_MECHANICS["pusht"]
    res_dir = ROOT / "outputs" / "quasimetric"
    res_dir.mkdir(parents=True, exist_ok=True)

    eval_budget_one_shot = cfg.plan_config.horizon * cfg.plan_config.action_block
    log(f"=== E7 CEM convergence trace (num_eval={cfg.eval.num_eval}) ===")

    h5_path = str(mech.h5_path(ROOT))
    dataset = swm.data.HDF5Dataset(path=h5_path, keys_to_cache=list(cfg.dataset.keys_to_cache))

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

    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
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

        cb_var = VarNormRecorder(reduction="mean")
        cb_spread = EliteSpreadRecorder(reduction="mean")
        cb_best = BestCostRecorder(reduction="mean")
        solver = swm.solver.CEMSolver(
            model=model, batch_size=cfg.solver.batch_size, num_samples=cfg.solver.num_samples,
            var_scale=cfg.solver.var_scale, n_steps=cfg.solver.n_steps, topk=cfg.solver.topk,
            device=cfg.solver.device, seed=cfg.seed, callbacks=[cb_var, cb_spread, cb_best],
        )

        config = swm.PlanConfig(**cfg.plan_config)
        policy = swm.policy.WorldModelPolicy(
            solver=solver, config=config, process=process, transform=transform)
        world = swm.World(env_name=cfg.world.env_name, num_envs=cfg.eval.num_eval,
                          max_episode_steps=2 * eval_budget_one_shot, image_shape=(224, 224))
        world.set_policy(policy)

        t_run = time.time()
        metrics = world.evaluate(
            dataset=dataset, start_steps=eval_start_idx.tolist(),
            goal_offset=cfg.eval.goal_offset_steps, eval_budget=eval_budget_one_shot,
            episodes_idx=eval_episodes.tolist(),
            callables=OmegaConf.to_container(cfg.eval.get("callables"), resolve=True),
            video=None,
        )
        dt = time.time() - t_run

        # history: list of per-env batches (batch_size=1 -> one list per episode), each a
        # list of per-iteration scalars (30 CEM steps). Average across episodes per step.
        def stack_history(cb):
            arr = np.array(cb.history, dtype=np.float64)   # (n_envs, n_steps)
            return arr.mean(axis=0).tolist(), arr.std(axis=0).tolist()

        var_mean, var_std = stack_history(cb_var)
        spread_mean, spread_std = stack_history(cb_spread)
        best_mean, best_std = stack_history(cb_best)

        log(f"[{cond}] success_rate={metrics.get('success_rate')}  ({dt:.1f}s)  "
            f"var_norm: iter0={var_mean[0]:.3f} iter29={var_mean[-1]:.3f}  "
            f"elite_spread: iter0={spread_mean[0]:.3f} iter29={spread_mean[-1]:.3f}")

        results[cond] = dict(
            success_rate=metrics.get("success_rate"), secs=dt,
            var_norm_mean=var_mean, var_norm_std=var_std,
            elite_spread_mean=spread_mean, elite_spread_std=spread_std,
            best_cost_mean=best_mean, best_cost_std=best_std,
        )
        world.close()

    log("\n=== SUMMARY (iteration 0 -> iteration 29, mean over episodes) ===")
    for cond in HEAD_NAMES:
        r = results[cond]
        log(f"{cond:>16s}: success={r['success_rate']:>5}%  "
            f"var_norm {r['var_norm_mean'][0]:.3f}->{r['var_norm_mean'][-1]:.3f}  "
            f"elite_spread {r['elite_spread_mean'][0]:.3f}->{r['elite_spread_mean'][-1]:.3f}")

    out = dict(num_eval=cfg.eval.num_eval, seed=cfg.seed, results=results, secs=time.time() - t0)
    p = res_dir / f"e7_cem_convergence_trace_pusht_n{cfg.eval.num_eval}.json"
    p.write_text(json.dumps(out, indent=2))
    log(f"wrote {p} ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
