"""E1 -- IDM ensemble sanity check (docs/experiment-plan-predictor-stitching.md).

Trains a small inverse-dynamics-model ensemble on real adjacent (z_t, z_{t+1}, a_t)
triples and checks whether it recovers the real action on HELD-OUT same-episode
transitions meaningfully better than predicting the marginal (mean) action. This is the
kill criterion for the whole predictor-based cross-episode-stitching line: if the ensemble
can't even recover a KNOWN action from a pair of latents it was never trained on, there is
nothing to build E2 (edge discovery/verification) on.

Run as:
    python scripts/idm_gate.py env=tworoom
"""

import json
import sys
from pathlib import Path

import h5py
import hydra
import numpy as np
from omegaconf import DictConfig

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.envs import ENV_MECHANICS
from common.graph_lib import find_transition_edges
from common.idm_ensemble import ensemble_predict, train_idm_ensemble
from common.log_util import log
from graph_gate import load_landmarks

ROOT = Path(__file__).resolve().parent.parent


def load_landmarks_and_actions(mech, cfg):
    # Reuses the B0 gate's landmark cache (outputs/b0_<prefix>) rather than re-encoding
    # under a fresh cache namespace -- it's already been computed at n_episodes=300.
    out_dir = ROOT / "outputs" / f"b0_{cfg.env.output_prefix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    z, ep_idx, step_idx, state = load_landmarks(
        mech, mech.h5_path(ROOT), out_dir, cfg.n_episodes, mech.ckpt_dir(ROOT),
    )
    f = h5py.File(mech.h5_path(ROOT), "r", swmr=True, rdcc_nbytes=256 * 1024 * 1024)
    ep_offset = f["ep_offset"][:]
    global_idx = ep_offset[ep_idx] + step_idx
    action_col = f["action"][:]
    action = action_col[global_idx].astype(np.float32)
    # LeWM consumes Z-SCORED actions: train.py runs utils.get_column_normalizer over every
    # non-pixel column (action included) and eval.py fits a sklearn StandardScaler on the
    # same column. Push-T's action std is ~0.206, so raw actions are ~5x too small and the
    # predictor reads them as a near-stationary command -- which is exactly what made it look
    # like a no-motion baseline. Stats are taken over the FULL dataset column, matching what
    # both train.py and eval.py fit on, not just the sampled landmarks.
    valid = action_col[~np.isnan(action_col).any(axis=1)].astype(np.float64)
    act_mean = valid.mean(0).astype(np.float32)
    act_std = valid.std(0).astype(np.float32)
    f.close()
    action_norm = (action - act_mean) / act_std
    log(f"[action-norm] mean={act_mean} std={act_std} "
        f"(raw actions are {float(np.mean(1.0 / act_std)):.2f}x too small for the model)")
    return z, ep_idx, step_idx, state, action_norm, (act_mean, act_std)


@hydra.main(version_base=None, config_path="../config/graph", config_name="idm_gate")
def main(cfg: DictConfig):
    mech = ENV_MECHANICS[cfg.env.name]

    log(f"=== E1: IDM ensemble sanity check (env={cfg.env.name}) ===")
    z, ep_idx, step_idx, state, action, act_stats = load_landmarks_and_actions(mech, cfg)
    log(f"landmarks: {z.shape[0]} frames from {len(np.unique(ep_idx))} episodes")

    trans_i, trans_j = find_transition_edges(ep_idx, step_idx)
    log(f"real adjacent (transition) pairs: {len(trans_i)}")

    uniq_eps = np.unique(ep_idx)
    rng = np.random.default_rng(0)
    shuffled = rng.permutation(uniq_eps)
    n_train = int(len(shuffled) * cfg.train_frac)
    train_eps, test_eps = shuffled[:n_train], shuffled[n_train:]

    trans_ep = ep_idx[trans_i]  # trans_i/trans_j always share an episode by construction
    train_mask = np.isin(trans_ep, train_eps)
    test_mask = np.isin(trans_ep, test_eps)
    tr_i, tr_j = trans_i[train_mask], trans_j[train_mask]
    te_i, te_j = trans_i[test_mask], trans_j[test_mask]
    log(f"train pairs: {len(tr_i)} ({len(train_eps)} episodes), "
        f"test pairs: {len(te_i)} ({len(test_eps)} episodes)")

    log(f"training {cfg.n_members}-member IDM ensemble ({cfg.n_steps} steps/member)...")
    models = train_idm_ensemble(
        z, action, tr_i, tr_j, action_dim=mech.action_dim,
        n_members=cfg.n_members, hidden=cfg.hidden, n_steps=cfg.n_steps,
        batch_size=cfg.batch_size, lr=cfg.lr, seed=cfg.seed,
    )

    preds = ensemble_predict(models, z, te_i, te_j)  # (M, n_test, action_dim)
    ensemble_mean = preds.mean(axis=0)
    true_a = action[te_i]

    member_mse = float(np.mean((preds - true_a[None]) ** 2))
    ensemble_mse = float(np.mean((ensemble_mean - true_a) ** 2))

    train_preds = ensemble_predict(models, z, tr_i, tr_j)
    train_true_a = action[tr_i]
    train_ensemble_mse = float(np.mean((train_preds.mean(axis=0) - train_true_a) ** 2))

    train_a = action[tr_i]
    marginal_mean = train_a.mean(axis=0, keepdims=True)  # the "predict the marginal" baseline
    baseline_mse = float(np.mean((marginal_mean - true_a) ** 2))

    var_per_pair = preds.var(axis=0).mean(axis=-1)  # (n_test,) mean ensemble variance per pair
    improvement = 1.0 - ensemble_mse / baseline_mse
    passed = improvement > 0.3  # not a principled cutoff, just "clearly better" -- inspect the
                                # actual numbers below rather than trusting this alone

    log(f"[E1] member-avg MSE={member_mse:.5f}  ensemble-mean MSE={ensemble_mse:.5f}  "
        f"marginal-baseline MSE={baseline_mse:.5f}")
    log(f"[E1] train-set ensemble MSE={train_ensemble_mse:.5f}  "
        f"(test/train ratio={ensemble_mse / train_ensemble_mse:.2f}x)")
    log(f"[E1] ensemble Var(a) on held-out pairs: mean={var_per_pair.mean():.5f} "
        f"std={var_per_pair.std():.5f}")
    log(f"[E1] improvement over marginal baseline: {100 * improvement:.1f}%")
    log(f"[E1] kill criterion (>30% MSE reduction vs. marginal baseline): "
        f"{'PASSED' if passed else 'FAILED'}")

    out_dir = ROOT / "outputs" / f"idm_gate_{cfg.env.output_prefix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = dict(
        env=cfg.env.name, n_episodes=cfg.n_episodes, n_members=cfg.n_members,
        n_train_pairs=int(len(tr_i)), n_test_pairs=int(len(te_i)),
        member_avg_mse=member_mse, ensemble_mean_mse=ensemble_mse,
        marginal_baseline_mse=baseline_mse, improvement_frac=improvement,
        ensemble_var_mean=float(var_per_pair.mean()), ensemble_var_std=float(var_per_pair.std()),
        passed_kill_criterion=passed,
    )
    out_path = out_dir / "e1_idm_gate_results.json"
    out_path.write_text(json.dumps(summary, indent=2))
    log(f"\nwrote {out_path}")
    log(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
