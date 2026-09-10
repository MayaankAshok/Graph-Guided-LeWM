"""E2 (revised) -- IDM-predictor cycle consistency (docs/graph-proposal/predictor-stitching.tex).

E2's original design (discover_predicted_edges) borrowed a metric neighbor's REAL action and
checked only the predictor's own self-consistency -- it failed decisively (3.7% precision vs.
92.7% baseline, [[tworoom-e2-edge-discovery-failed]]) because a predictor's reconstruction
error isn't evidence of correctness once the action is off-manifold for that specific state.

This tests a different, better-targeted design: for each candidate cross-episode pair (i, j)
from a WIDE, unfiltered top-k nearest-neighbor pool (deliberately looser than the calibrated
identification-edge threshold, so there's real signal to discriminate), the IDM ensemble
infers what action would connect z_i DIRECTLY to z_j (not a borrowed, unrelated action). If
the ensemble agrees (variance below a data-driven separator between real transitions and
genuinely random pairs -- calibrate_variance_threshold) AND running its mean action through
the INDEPENDENTLY-trained frozen predictor points in the right DIRECTION (cosine similarity
to the true target direction, not an absolute distance floor -- the original absolute-floor
design mis-scaled badly, see [[tworoom-e2b-cycle-consistency-diagnosed]] and
docs/graph-proposal/latent-diagnostics.tex Figures 3-4), the edge is kept. Two
separately-trained models must agree, unlike E2's single-model self-consistency check.

Run as:
    python scripts/cycle_consistency_gate.py env=tworoom
"""

import json
import sys
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.envs import ENV_MECHANICS
from common.graph_lib import (
    DEV, SKIP, action_continuation_lookup, build_identification_edges,
    calibrate_variance_threshold, cycle_consistency_filter, find_transition_edges,
    oracle_pairwise_sample, raw_topk_cross_episode_pairs,
)
from common.idm_ensemble import ensemble_predict, train_idm_ensemble
from common.lewm_loader import load_lewm
from common.log_util import log
from idm_gate import load_landmarks_and_actions

ROOT = Path(__file__).resolve().parent.parent


@hydra.main(version_base=None, config_path="../config/graph", config_name="cycle_consistency_gate")
def main(cfg: DictConfig):
    mech = ENV_MECHANICS[cfg.env.name]

    log(f"=== E2 (revised): IDM-predictor cycle consistency (env={cfg.env.name}) ===")
    z, ep_idx, step_idx, state, action, act_stats = load_landmarks_and_actions(mech, cfg)
    log(f"landmarks: {z.shape[0]} frames from {len(np.unique(ep_idx))} episodes")

    true_dist_oracle = mech.build_true_distance_oracle(state)
    trans_i, trans_j = find_transition_edges(ep_idx, step_idx)
    cont_valid, cont_actions, cont_target_row = action_continuation_lookup(
        ep_idx, step_idx, action, n_cont=SKIP - 1,
    )

    log(f"=== building wide, unfiltered top-{cfg.k_candidates} candidate pool ===")
    cand_i, cand_j = raw_topk_cross_episode_pairs(z, ep_idx, k=cfg.k_candidates)
    log(f"candidate pool: {len(cand_i)} unique cross-episode pairs")

    log("=== scoring the UNFILTERED pool against the oracle (before) ===")
    keep0, oracle_d0 = oracle_pairwise_sample(true_dist_oracle, state, cand_i, cand_j,
                                               cfg.n_oracle_sample, seed=0)
    before_precision = float(np.mean(oracle_d0 <= 2))
    log(f"[E2b] unfiltered pool: {len(keep0)} oracle-scored | precision (oracle_dist<=2): "
        f"{before_precision:.3f}")

    log("=== training IDM ensemble (same convention as E1) ===")
    uniq_eps = np.unique(ep_idx)
    rng = np.random.default_rng(0)
    shuffled = rng.permutation(uniq_eps)
    n_train = int(len(shuffled) * cfg.train_frac)
    train_eps, test_eps = shuffled[:n_train], shuffled[n_train:]
    trans_ep = ep_idx[trans_i]
    train_mask = np.isin(trans_ep, train_eps)
    test_mask = np.isin(trans_ep, test_eps)
    tr_i, tr_j = trans_i[train_mask], trans_j[train_mask]
    te_i, te_j = trans_i[test_mask], trans_j[test_mask]
    models = train_idm_ensemble(
        z, action, tr_i, tr_j, action_dim=mech.action_dim,
        n_members=cfg.n_members, hidden=cfg.hidden, n_steps=cfg.n_steps,
        batch_size=cfg.batch_size, lr=cfg.lr, seed=cfg.seed,
    )

    log("=== calibrating thresholds (docs/graph-proposal/latent-diagnostics.tex Figs 3/4/6) ===")
    held_out_preds = ensemble_predict(models, z, te_i, te_j)
    held_out_var = held_out_preds.var(axis=0).mean(axis=-1)

    rand_i = np.random.default_rng(11).integers(0, len(z), 20_000)
    rand_j = np.random.default_rng(12).integers(0, len(z), 20_000)
    rand_keep = rand_i != rand_j
    random_var = ensemble_predict(models, z, rand_i[rand_keep], rand_j[rand_keep]).var(axis=0).mean(axis=-1)

    var_threshold = calibrate_variance_threshold(held_out_var, random_var)
    log(f"[E2] variance threshold {var_threshold:.5f} -- geometric-mean separator between "
        f"real transitions (p95={np.quantile(held_out_var, 0.95):.5f}) and random pairs "
        f"(p05={np.quantile(random_var, 0.05):.5f}), per Figure 6")

    direction_threshold = cfg.direction_threshold
    log(f"[E2] direction threshold cos>{direction_threshold:.2f} -- Figures 3/4 show the "
        f"predictor's magnitude is unreliable at this scale but its direction is (median "
        f"cosine 0.924 on real transitions), so gate on direction instead of an absolute floor")

    model = load_lewm(ckpt_dir=mech.ckpt_dir(ROOT), device=DEV)

    log("=== running the two-model cycle-consistency filter on the candidate pool ===")
    keep_mask, var_a, cos_dir, usable_mask = cycle_consistency_filter(
        model, models, z, cand_i, cand_j, cont_valid, cont_actions, cont_target_row,
        var_threshold, direction_threshold,
    )
    kept_i, kept_j = cand_i[keep_mask], cand_j[keep_mask]
    n_usable = int(usable_mask.sum())
    log(f"[E2b] {len(kept_i)} / {n_usable} usable candidates pass both checks "
        f"({100 * len(kept_i) / max(1, n_usable):.1f}% of usable, "
        f"{100 * len(kept_i) / max(1, len(cand_i)):.1f}% of all {len(cand_i)})")

    if len(kept_i) == 0:
        log("[E2b] nothing passed the filter -- cannot score precision/recall, stopping here")
        after_precision, recall = None, None
    else:
        log("=== scoring the FILTERED (kept) pool against the oracle (after) ===")
        keep1, oracle_d1 = oracle_pairwise_sample(true_dist_oracle, state, kept_i, kept_j,
                                                   cfg.n_oracle_sample, seed=1)
        after_precision = float(np.mean(oracle_d1 <= 2))
        log(f"[E2b] filtered pool: {len(keep1)} oracle-scored | precision (oracle_dist<=2): "
            f"{after_precision:.3f}")

        # recall relative to the unfiltered pool's own true positives: of the pairs already
        # oracle-scored as genuinely good in the "before" sample, what fraction also survive
        # the filter's variance+forward checks (evaluated on that same scored subsample)?
        good0_mask = oracle_d0 <= 2
        if good0_mask.sum() >= 10:
            i0, j0 = cand_i[keep0][good0_mask], cand_j[keep0][good0_mask]
            keep_on_good, _, _, usable_on_good = cycle_consistency_filter(
                model, models, z, i0, j0, cont_valid, cont_actions, cont_target_row,
                var_threshold, direction_threshold,
            )
            if usable_on_good.sum() >= 10:
                recall = float(keep_on_good[usable_on_good].mean())
                log(f"[E2b] recall on the {int(usable_on_good.sum())}/{good0_mask.sum()} "
                    f"usable known-good pairs from the 'before' sample: {recall:.3f}")
            else:
                recall = None
        else:
            recall = None

    log("=== reference: existing plain-metric k-capped identification edges ===")
    id_i, id_j, rho_hat, eps2 = build_identification_edges(z, ep_idx, step_idx, k=4)
    keep_b, oracle_d_baseline = oracle_pairwise_sample(true_dist_oracle, state, id_i, id_j,
                                                        cfg.n_oracle_sample, seed=2)
    baseline_precision = float(np.mean(oracle_d_baseline <= 2))
    log(f"[E2b] reference baseline (k=4 calibrated kNN): {len(id_i)} edges, "
        f"precision={baseline_precision:.3f}, for context only (different candidate pool "
        f"size, not a like-for-like recall comparison)")

    out_dir = ROOT / "outputs" / f"idm_gate_{cfg.env.output_prefix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = dict(
        env=cfg.env.name,
        k_candidates=cfg.k_candidates, n_candidates=int(len(cand_i)),
        before_precision=before_precision,
        var_threshold=var_threshold, direction_threshold=direction_threshold,
        n_usable=n_usable, n_kept=int(len(kept_i)),
        kept_frac_of_usable=len(kept_i) / max(1, n_usable),
        kept_frac_of_all=len(kept_i) / max(1, len(cand_i)),
        after_precision=after_precision, recall_on_known_good=recall,
        reference_baseline_precision=baseline_precision,
    )
    out_path = out_dir / "e2b_cycle_consistency_gate_results.json"
    out_path.write_text(json.dumps(summary, indent=2))
    log(f"\nwrote {out_path}")
    log(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
