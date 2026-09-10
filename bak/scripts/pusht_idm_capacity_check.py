"""Standalone check: does the IDM ensemble actually recover ground-truth actions on held-out
REAL transitions, at the full 300-episode Push-T landmark scale? Follow-up to
pusht_expert30_idm_probe.py's surprise finding -- at expert_30 (3,241 train / 760 test pairs),
EVERY hidden size tried (256/512/1024) gave NEGATIVE MSE reduction vs. the marginal (mean
training action) baseline. Before concluding anything about model capacity in general, check
whether that is an expert_30 data-scarcity artifact or holds at 10x the data too.

Run as:
    python scripts/pusht_idm_capacity_check.py
"""

import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.envs import ENV_MECHANICS
from common.graph_lib import find_transition_edges
from common.idm_ensemble import ensemble_predict, train_idm_ensemble
from common.log_util import log
from graph_gate import load_landmarks

ROOT = Path(__file__).resolve().parent.parent
N_EPISODES = 300


def main():
    mech = ENV_MECHANICS["pusht"]
    out_dir = ROOT / "outputs" / "b0_pusht"
    z, ep_idx, step_idx, state = load_landmarks(
        mech, mech.h5_path(ROOT), out_dir, N_EPISODES, mech.ckpt_dir(ROOT), seed=0)
    log(f"landmarks: {z.shape[0]} frames from {len(np.unique(ep_idx))} episodes")

    f = h5py.File(mech.h5_path(ROOT), "r", swmr=True, rdcc_nbytes=256 * 1024 * 1024)
    ep_offset = f["ep_offset"][:]
    global_idx = ep_offset[ep_idx] + step_idx
    action = f["action"][:][global_idx].astype(np.float32)
    f.close()

    trans_i, trans_j = find_transition_edges(ep_idx, step_idx)
    uniq_eps = np.unique(ep_idx)
    rng = np.random.default_rng(0)
    shuffled = rng.permutation(uniq_eps)
    n_train = int(len(shuffled) * 0.8)
    train_eps, test_eps = shuffled[:n_train], shuffled[n_train:]
    trans_ep = ep_idx[trans_i]
    tr_i, tr_j = trans_i[np.isin(trans_ep, train_eps)], trans_j[np.isin(trans_ep, train_eps)]
    te_i, te_j = trans_i[np.isin(trans_ep, test_eps)], trans_j[np.isin(trans_ep, test_eps)]
    log(f"IDM train pairs: {len(tr_i)} ({len(train_eps)} episodes), "
        f"test pairs: {len(te_i)} ({len(test_eps)} episodes)")

    marginal_action = action[tr_i].mean(axis=0, keepdims=True)
    mse_marginal = float(((marginal_action - action[te_i]) ** 2).mean())
    log(f"marginal-baseline (predict mean training action) held-out MSE={mse_marginal:.5f}")

    for hidden in (256, 512, 1024):
        m = train_idm_ensemble(z, action, tr_i, tr_j, action_dim=mech.action_dim,
                                n_members=1, hidden=hidden, n_steps=3000, batch_size=256,
                                lr=1e-3, seed=0)
        pred = ensemble_predict(m, z, te_i, te_j).mean(axis=0)
        mse = float(((pred - action[te_i]) ** 2).mean())
        log(f"[hidden={hidden}] held-out REAL action-recovery MSE={mse:.5f} "
            f"(reduction={100*(1-mse/mse_marginal):.1f}%)")


if __name__ == "__main__":
    main()
