"""Re-score the E2 distance-bucket diagnosis using REAL environment stepping with the
specific inferred action, instead of oracle_dist(i,j) (which only asks whether i and j are
mutually reachable via SOME path -- never validates the action actually tried). Per-user
direction: treat the oracle-based pairwise reachability check as insufficient going forward
for scoring a candidate edge's realizing action -- use actual environment stepping instead.

Run as:
    python scripts/env_verified_buckets.py env=tworoom
"""

import sys
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.envs import ENV_MECHANICS
from common.graph_lib import (
    DEV, SKIP, action_continuation_lookup, env_action_verified_distance, find_transition_edges,
    oracle_pairwise_sample, raw_topk_cross_episode_pairs,
)
from common.idm_ensemble import ensemble_predict, train_idm_ensemble
from common.lewm_loader import load_lewm
from common.log_util import log
from idm_gate import load_landmarks_and_actions

ROOT = Path(__file__).resolve().parent.parent
SPEED = 5.0  # Two-Room's empirically fixed dataset speed, see scripts/env_verify_cycle_edges.py


@hydra.main(version_base=None, config_path="../config/graph", config_name="idm_gate")
def main(cfg: DictConfig):
    mech = ENV_MECHANICS[cfg.env.name]
    log(f"=== E2 action-verified distance buckets (env={cfg.env.name}) ===")
    z, ep_idx, step_idx, state, action, act_stats = load_landmarks_and_actions(mech, cfg)
    true_dist_oracle = mech.build_true_distance_oracle(state)
    trans_i, trans_j = find_transition_edges(ep_idx, step_idx)
    cont_valid, cont_actions, cont_target_row = action_continuation_lookup(
        ep_idx, step_idx, action, n_cont=SKIP - 1,
    )

    cand_i, cand_j = raw_topk_cross_episode_pairs(z, ep_idx, k=20)
    log(f"candidate pool: {len(cand_i)} unique cross-episode pairs")

    n_sample = 2000
    keep0, oracle_d = oracle_pairwise_sample(true_dist_oracle, state, cand_i, cand_j,
                                              n_sample, seed=0)
    ci, cj = cand_i[keep0], cand_j[keep0]
    raw_dist = np.linalg.norm(z[ci] - z[cj], axis=1)
    good_oracle = oracle_d <= 2

    uniq_eps = np.unique(ep_idx)
    shuffled = np.random.default_rng(0).permutation(uniq_eps)
    n_train = int(len(shuffled) * cfg.train_frac)
    train_eps = shuffled[:n_train]
    trans_ep = ep_idx[trans_i]
    tr_i, tr_j = trans_i[np.isin(trans_ep, train_eps)], trans_j[np.isin(trans_ep, train_eps)]
    models = train_idm_ensemble(
        z, action, tr_i, tr_j, action_dim=mech.action_dim,
        n_members=cfg.n_members, hidden=cfg.hidden, n_steps=cfg.n_steps,
        batch_size=cfg.batch_size, lr=cfg.lr, seed=cfg.seed,
    )
    mean_action_norm = ensemble_predict(models, z, ci, cj).mean(axis=0).astype(np.float32)
    # de-normalize for the simulator; keep the normalized copy for anything model-facing
    _am, _as = act_stats
    mean_action = (mean_action_norm * _as + _am).astype(np.float32)

    log("=== scoring via REAL environment stepping + ORACLE distance on the outcome ===")
    keep1, after_dist = env_action_verified_distance(mech, state, ci, cj, mean_action,
                                                       true_dist_oracle, cap=n_sample,
                                                       seed=1, speed=SPEED)
    good_env = after_dist <= 2
    raw_dist_k = raw_dist[keep1]
    good_oracle_k = good_oracle[keep1]

    log(f"[compare] oracle-based good: {good_oracle_k.mean():.3f}  "
        f"action-verified good: {good_env.mean():.3f}  "
        f"agreement (both agree): {(good_oracle_k == good_env).mean():.3f}")

    # also compute the predictor's own forward-model direction/distance for the same sample,
    # for a like-for-like comparison against latent-only proxies. Uses the real skip-action
    # block convention: the IDM's inferred connecting action spliced into slot 1, j's own
    # real next (skip-1) actions filling the rest (see cycle_consistency_filter's docstring
    # / [[lewm-predictor-action-block-convention]]) -- NOT the superseded single-action
    # zero-padding this used to do. j's without a real continuation (too close to their
    # episode's end) can't be spliced and are excluded from this arm only.
    model = load_lewm(ckpt_dir=mech.ckpt_dir(ROOT), device=DEV)
    cj_k = cj[keep1]
    usable_latent = cont_valid[cj_k]
    n_dropped = int((~usable_latent).sum())
    if n_dropped:
        log(f"[latent-arm] dropping {n_dropped}/{len(cj_k)} candidates whose j has no real "
            f"continuation (too close to its episode's end)")
    ur = np.nonzero(usable_latent)[0]
    hybrid = np.concatenate([mean_action_norm[keep1][ur], cont_actions[cj_k[ur]]], axis=1)
    zi_t = torch.from_numpy(z[ci[keep1][ur]]).float().to(DEV).unsqueeze(1)
    act_t = torch.from_numpy(hybrid).float().to(DEV).unsqueeze(1)
    with torch.no_grad():
        act_emb = model.action_encoder(act_t)
        z_hat = model.predict(zi_t, act_emb)[:, -1].cpu().numpy()
    target_row = cont_target_row[cj_k[ur]]
    dist_before_latent = np.linalg.norm(z[ci[keep1][ur]] - z[target_row], axis=1)
    dist_after_latent = np.linalg.norm(z_hat - z[target_row], axis=1)
    improved_latent_ur = dist_after_latent <= dist_before_latent
    # scatter back to keep1's index space; unusable rows are excluded via usable_latent below,
    # not silently treated as "not improved"
    improved_latent = np.zeros(len(cj_k), dtype=bool)
    improved_latent[ur] = improved_latent_ur

    log("\nbuckets, ORACLE-based good vs. ACTION-VERIFIED good, and whether the latent "
        "relative-improvement proxy predicts the action-verified outcome "
        "(restricted to candidates with a real action-block continuation):")
    buckets = [0, 1, 2, 3, 5, 10, 1000]
    for b0, b1 in zip(buckets[:-1], buckets[1:]):
        m = (raw_dist_k >= b0) & (raw_dist_k < b1) & usable_latent
        if m.sum() < 5:
            continue
        prec_among_proxy_kept = (
            good_env[m & improved_latent].mean() if (m & improved_latent).sum() else float("nan")
        )
        log(f"raw_dist[{b0},{b1}): n={m.sum():4d}  oracle_good={good_oracle_k[m].mean():.3f}  "
            f"action_verified_good={good_env[m].mean():.3f}  "
            f"mean_after_dist={after_dist[m].mean():.2f}  "
            f"latent_proxy_kept_frac={improved_latent[m].mean():.3f}  "
            f"precision_among_proxy_kept={prec_among_proxy_kept:.3f}")


if __name__ == "__main__":
    main()
