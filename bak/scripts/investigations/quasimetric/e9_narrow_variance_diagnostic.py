"""E9 -- does the quasimetric head's discriminability specifically degrade in the NARROW,
converged neighborhood CEM's later iterations actually explore, vs. the BROAD, wide-variance
population E6 tested on?

E8 (full n=50) showed d_Q trails L2 at every CEM iteration count (1..30), with the gap
NARROWING but not closed by iteration 30 -- ambiguous between "d_Q converges more slowly to
the same ceiling" (fixable by just running CEM longer) and "d_Q converges to a genuinely
worse ceiling" (a training/architecture issue). This diagnostic tests the mechanism directly:
E3-v3's candidate pool -- and E6's accuracy numbers built from it -- used ONLY broad,
wide-variance (var_scale=1.0) candidates, resembling CEM's INITIAL exploration (iteration 0).
The head never saw anything resembling CEM's LATE-iteration candidate distribution: narrow,
centered on a specific partially-converged mean, not zero.

Method: run each cost's OWN real CEM solve to a partial iteration count (n_steps=15, the
midpoint of E8's sweep where the gap was still large), capture the ACTUAL converged
mean/std it reached (via CEMSolver's own `outputs['mean']`/`outputs['var']`, which is
literally what it would sample its NEXT iteration from), then draw a FRESH batch of K=20
candidates from that exact narrow distribution -- the same real distribution CEM itself
would explore next. Roll them through the real predictor (for the cost) and the real
environment (for ground truth, unavoidable for non-expert actions), and compute within-group
Spearman for raw_l2, psi_only, and full_dQ against true outcome distance -- directly
comparable to E6's broad-population numbers (raw_l2=0.5847, psi_only=0.9126, full_dQ=0.9205).

If psi_only/full_dQ's correlation drops much more sharply than raw_l2's when moving from the
broad to the narrow population, that confirms a fine-grained discriminability collapse
specific to the learned head -- the mechanism behind E8's persistent gap, and the direct
target for a retraining fix (augment training with narrow, CEM-converged-like candidates,
not just broad exploration-like ones).

Run:
    python scripts/investigations/quasimetric/e9_narrow_variance_diagnostic.py
"""

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import json
import sys
import time
from pathlib import Path

import h5py
import hydra
import numpy as np
import stable_worldmodel as swm
import torch
from omegaconf import DictConfig, OmegaConf
from scipy import stats as sps
from sklearn import preprocessing

ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.envs import ENV_MECHANICS
from common.lewm_loader import load_lewm
from common.log_util import log
from common.quasimetric import LatentQuasimetric
from e3_train_quasimetric import episode_split
from e3v2_train_predictor_aware import predictor_rollout_from_z0
from e4_live_rollout import get_episodes_length, img_transform, make_quasimetric_criterion
from probe_gate import Arm

N_STEPS_PARTIAL = int(os.environ.get("E9_N_STEPS_PARTIAL", 15))
K_CANDIDATES = int(os.environ.get("E9_K", 20))
CONDITIONS = {"l2": None, "quasimetric_v3": "e3v3_quasimetric_head_pusht.pt"}


def within_group_spearman(values, true_dist, group_id):
    rhos = []
    for g in np.unique(group_id):
        idx = np.nonzero(group_id == g)[0]
        if len(idx) > 2 and np.ptp(true_dist[idx]) > 1e-6 and np.ptp(values[idx]) > 1e-9:
            r = sps.spearmanr(true_dist[idx], values[idx]).statistic
            if np.isfinite(r):
                rhos.append(r)
    return float(np.mean(rhos)), len(rhos)


@hydra.main(version_base=None, config_path=str(ROOT / "config" / "eval"), config_name="pusht")
def main(cfg: DictConfig):
    t0 = time.time()
    mech = ENV_MECHANICS["pusht"]
    res_dir = ROOT / "outputs" / "quasimetric"
    res_dir.mkdir(parents=True, exist_ok=True)
    eval_budget_one_shot = cfg.plan_config.horizon * cfg.plan_config.action_block

    log(f"=== E9 narrow-variance diagnostic (num_eval={cfg.eval.num_eval}, "
        f"n_steps_partial={N_STEPS_PARTIAL}, K={K_CANDIDATES}) ===")

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

    with h5py.File(h5_path, "r", swmr=True) as f5:
        ep_offset = f5["ep_offset"][:]
        state_col = f5["state"]
        action_col_full = f5["action"]
        ep_ids = np.asarray(eval_episodes).astype(np.int64)
        start_idx_arr = np.asarray(eval_start_idx).astype(np.int64)
        start_rows = ep_offset[ep_ids] + start_idx_arr
        goal_rows = start_rows + cfg.eval.goal_offset_steps

        def fetch_rows(rows):
            order = np.argsort(rows)
            out = np.empty((len(rows), state_col.shape[1]), dtype=state_col.dtype)
            out[order] = state_col[rows[order]]
            return out

        start_states = fetch_rows(start_rows)
        goal_states = fetch_rows(goal_rows)
        true_dist_oracle = mech.build_true_distance_oracle(np.asarray(state_col[::37]))
        action_all = action_col_full[:]   # full column, cheap (~19MB) -- for action mean/std

    act_mean, act_std = action_all.mean(0), action_all.std(0)
    act_std[act_std < 1e-6] = 1.0

    # z0 (real cached embeddings) and normalization stats, matching training exactly
    cache_dir = Path(os.environ.get("PROBE_CACHE_DIR", str(ROOT / "outputs" / "probe_gate_pusht" / "cache")))
    tag = "ep18685_seed0_g16"
    z_all = np.load(cache_dir / f"z_{tag}.npy", mmap_mode="r")
    ep_idx_cache = np.load(cache_dir / f"ep_idx_{tag}.npy")
    train_eps, test_eps, rows_tr, rows_te = episode_split(ep_idx_cache, 0.8, 0)
    arm = Arm("lewm", z_all, z_all.shape[1])
    arm.fit_norm(rows_tr, 0)
    mu, sd = arm.mu[0], arm.sd[0]
    z0_all = np.asarray(z_all[start_rows]).astype(np.float32)   # (n_envs, 192) raw
    z_goal_all = np.asarray(z_all[goal_rows]).astype(np.float32)

    ckpt_dir = mech.ckpt_dir(ROOT)
    device = "cuda"

    results = {}
    for cond, head_name in CONDITIONS.items():
        log(f"\n--- condition: {cond} ---")
        model = load_lewm(ckpt_dir=ckpt_dir, device=device)
        model.interpolate_pos_encoding = True
        quasi_model = None
        if head_name is not None:
            quasi_model = LatentQuasimetric(latent_dim=192, hidden_dim=256, proj_dim=64)
            quasi_model.load_state_dict(torch.load(res_dir / head_name, map_location="cpu"))
            quasi_model = quasi_model.to(device).eval()
            quasi_model.requires_grad_(False)
            from e4_live_rollout import recompute_quasimetric_norm_stats
            qmu, qsd = recompute_quasimetric_norm_stats(cache_dir, 18685, 0, 0.8)
            model.criterion = make_quasimetric_criterion(quasi_model, qmu, qsd)

        solver = swm.solver.CEMSolver(
            model=model, batch_size=cfg.solver.batch_size, num_samples=cfg.solver.num_samples,
            var_scale=cfg.solver.var_scale, n_steps=N_STEPS_PARTIAL, topk=cfg.solver.topk,
            device=cfg.solver.device, seed=cfg.seed,
        )
        captured = {}
        real_solve = solver.solve
        def wrapped_solve(info_dict, init_action=None, _real=real_solve, _bucket=captured):
            out = _real(info_dict, init_action=init_action)
            if "mean" not in _bucket:
                _bucket["mean"] = out["mean"][-1].numpy() if isinstance(out["mean"], list) else np.asarray(out["mean"])
                _bucket["var"] = out["var"][-1].numpy() if isinstance(out["var"], list) else np.asarray(out["var"])
            return out
        solver.solve = wrapped_solve

        config = swm.PlanConfig(**cfg.plan_config)
        policy = swm.policy.WorldModelPolicy(
            solver=solver, config=config, process=process, transform=transform)
        world = swm.World(env_name=cfg.world.env_name, num_envs=cfg.eval.num_eval,
                          max_episode_steps=2 * eval_budget_one_shot, image_shape=(224, 224))
        world.set_policy(policy)
        world.evaluate(
            dataset=dataset, start_steps=eval_start_idx.tolist(),
            goal_offset=cfg.eval.goal_offset_steps, eval_budget=eval_budget_one_shot,
            episodes_idx=eval_episodes.tolist(),
            callables=OmegaConf.to_container(cfg.eval.get("callables"), resolve=True),
            video=None,
        )
        world.close()

        cem_mean = captured["mean"]   # (n_envs, horizon, action_dim*action_block) z-scored units
        cem_std = captured["var"]     # same shape -- CEMSolver's own "var" is actually the std
        n_envs = cem_mean.shape[0]
        log(f"captured CEM state at n_steps={N_STEPS_PARTIAL}: mean_norm="
            f"{np.linalg.norm(cem_mean, axis=(1,2)).mean():.3f}  "
            f"std_norm={np.linalg.norm(cem_std, axis=(1,2)).mean():.3f}")

        # -- sample K fresh candidates per env from its OWN converged N(mean, std), exactly
        # matching CEM's own next-iteration sampling formula (candidates = randn*std + mean)
        env_local = mech.make_env()
        raw_l2_all, psi_all, full_all, true_dist_all, group_id_all = [], [], [], [], []
        t_env = time.time()
        rng = np.random.default_rng(cfg.seed + 1)
        for i in range(n_envs):
            cand_z = (rng.standard_normal((K_CANDIDATES,) + cem_mean.shape[1:]).astype(np.float32)
                     * cem_std[i][None] + cem_mean[i][None])   # (K, horizon=5, 10)
            # de-normalize per REAL action (2-dim), not per flattened 10-dim block: split the
            # flat 10 = 5 real actions x 2 dims BEFORE broadcasting act_mean/act_std (shape (2,))
            cand_raw = (cand_z.reshape(K_CANDIDATES, 5, 5, 2) * act_std.reshape(1, 1, 1, -1)
                       + act_mean.reshape(1, 1, 1, -1)).reshape(K_CANDIDATES, 25, 2)   # (K, 25, 2)

            true_d = np.empty(K_CANDIDATES, dtype=np.float32)
            for k in range(K_CANDIDATES):
                env_local.reset(options={"state": start_states[i].astype(np.float64).copy(),
                                         "goal_state": goal_states[i].astype(np.float64).copy()})
                obs = None
                for t in range(cand_raw.shape[1]):
                    obs, reward, terminated, truncated, info = env_local.step(cand_raw[k, t])
                    if terminated or truncated:
                        break
                final_state = np.asarray(obs["state"], dtype=np.float64)
                true_d[k] = true_dist_oracle(final_state[None, :], goal_states[i][None, :].astype(np.float64))[0, 0]

            with torch.no_grad():
                z0_t = torch.from_numpy(np.tile(z0_all[i], (K_CANDIDATES, 1))).float().to(device)
                blocks_t = torch.from_numpy(cand_z.reshape(K_CANDIDATES, 5, 10)).float().to(device)
                z_hat = predictor_rollout_from_z0(model, z0_t, blocks_t).cpu().numpy()

            z_goal_i = np.tile(z_goal_all[i], (K_CANDIDATES, 1))
            raw_l2 = np.linalg.norm(z_hat - z_goal_i, axis=1)
            raw_l2_all.append(raw_l2)
            true_dist_all.append(true_d)
            group_id_all.append(np.full(K_CANDIDATES, i, dtype=np.int64))

            if quasi_model is not None:
                z_hat_n = np.clip((z_hat - mu) / sd, -10.0, 10.0)
                z_goal_n = np.clip((z_goal_i - mu) / sd, -10.0, 10.0)
                with torch.no_grad():
                    za = torch.from_numpy(z_hat_n).float().to(device)
                    zg = torch.from_numpy(z_goal_n).float().to(device)
                    psi_all.append(torch.norm(quasi_model.psi(za) - quasi_model.psi(zg), dim=-1).cpu().numpy())
                    full_all.append(quasi_model(za, zg).cpu().numpy())

            if i % 10 == 0:
                log(f"  [{cond}] env {i}/{n_envs} elapsed={time.time()-t_env:.1f}s")
        env_local.close()

        raw_l2_all = np.concatenate(raw_l2_all)
        true_dist_all = np.concatenate(true_dist_all)
        group_id_all = np.concatenate(group_id_all)
        rho_raw, ng_raw = within_group_spearman(raw_l2_all, true_dist_all, group_id_all)
        log(f"[{cond}] narrow-region within-group Spearman: raw_l2={rho_raw:.4f} (n_groups={ng_raw})")
        cond_res = dict(raw_l2=rho_raw, n_groups_raw=ng_raw,
                        true_dist_mean=float(true_dist_all.mean()), true_dist_std=float(true_dist_all.std()))
        if quasi_model is not None:
            psi_all = np.concatenate(psi_all)
            full_all = np.concatenate(full_all)
            rho_psi, ng_psi = within_group_spearman(psi_all, true_dist_all, group_id_all)
            rho_full, ng_full = within_group_spearman(full_all, true_dist_all, group_id_all)
            log(f"[{cond}] narrow-region within-group Spearman: psi_only={rho_psi:.4f} full_dQ={rho_full:.4f}")
            cond_res.update(psi_only=rho_psi, full_dQ=rho_full, n_groups_psi=ng_psi)
        results[cond] = cond_res

    log("\n=== SUMMARY: narrow-region (converged-neighborhood) vs. broad-region (E6) Spearman ===")
    log(f"broad  (E6, var_scale=1.0 exploration): raw_l2=0.5847  psi_only=0.9126  full_dQ=0.9205")
    log(f"narrow (this run, n_steps={N_STEPS_PARTIAL} converged neighborhood):")
    log(f"  l2 condition's own neighborhood:            raw_l2={results['l2']['raw_l2']:.4f}")
    log(f"  quasimetric_v3's own neighborhood: raw_l2={results['quasimetric_v3']['raw_l2']:.4f}  "
        f"psi_only={results['quasimetric_v3'].get('psi_only')}  full_dQ={results['quasimetric_v3'].get('full_dQ')}")

    out = dict(num_eval=cfg.eval.num_eval, n_steps_partial=N_STEPS_PARTIAL, k=K_CANDIDATES,
              seed=cfg.seed, results=results, secs=time.time() - t0)
    p = res_dir / f"e9_narrow_variance_diagnostic_pusht_n{cfg.eval.num_eval}.json"
    p.write_text(json.dumps(out, indent=2))
    log(f"wrote {p} ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
