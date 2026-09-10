"""E2 -- edge discovery + verification precision vs. the Two-Room oracle
(docs/graph-proposal/predictor-stitching.tex).

Runs LeWM-predictor-based edge discovery (common.graph_lib.discover_predicted_edges) on
Two-Room's B0 landmark set, scores every discovered edge against the true oracle distance,
and checks whether the IDM ensemble's prediction variance (same mechanism as stage E1)
actually tracks edge correctness -- both (a) precision as a function of a variance
threshold and (b) the raw correlation. Compares against the existing plain-metric k-capped
identification edges' own false-edge rate as the number to beat (the E2 kill criterion).

Run as:
    python scripts/edge_discovery_gate.py env=tworoom
"""

import json
import sys
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig
from scipy import stats as sps

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.envs import ENV_MECHANICS
from common.graph_lib import (
    DEV, SKIP, build_identification_edges, discover_predicted_edges,
    find_block_transition_edges, find_transition_edges, oracle_pairwise_sample,
    predictor_error_floor,
)
from common.idm_ensemble import ensemble_predict, train_idm_ensemble
from common.lewm_loader import load_lewm
from common.log_util import log
from idm_gate import load_landmarks_and_actions

ROOT = Path(__file__).resolve().parent.parent


@hydra.main(version_base=None, config_path="../config/graph", config_name="edge_discovery_gate")
def main(cfg: DictConfig):
    mech = ENV_MECHANICS[cfg.env.name]

    log(f"=== E2: edge discovery + verification precision (env={cfg.env.name}) ===")
    z, ep_idx, step_idx, state, action, act_stats = load_landmarks_and_actions(mech, cfg)
    log(f"landmarks: {z.shape[0]} frames from {len(np.unique(ep_idx))} episodes")

    true_dist_oracle = mech.build_true_distance_oracle(state)
    trans_i, trans_j = find_transition_edges(ep_idx, step_idx)
    block_i, action_block, block_j = find_block_transition_edges(ep_idx, step_idx, action, skip=SKIP)

    log("=== loading predictor and calibrating its error floor ===")
    model = load_lewm(ckpt_dir=mech.ckpt_dir(ROOT), device=DEV)
    floor = predictor_error_floor(model, z, action_block, block_i, block_j,
                                   quantile=cfg.error_floor_quantile)

    log("=== discovering predicted-successor edges ===")
    disc_i, disc_j, disc_sqerr = discover_predicted_edges(
        model, z, ep_idx, block_i, action_block, k_actions=cfg.k_actions, error_floor=floor,
    )
    if len(disc_i) == 0:
        raise SystemExit("[E2] discovered zero edges -- nothing to evaluate, stop here")

    log("=== training IDM ensemble (same convention as E1) ===")
    uniq_eps = np.unique(ep_idx)
    rng = np.random.default_rng(0)
    shuffled = rng.permutation(uniq_eps)
    n_train = int(len(shuffled) * cfg.train_frac)
    train_eps = shuffled[:n_train]
    trans_ep = ep_idx[trans_i]
    train_mask = np.isin(trans_ep, train_eps)
    tr_i, tr_j = trans_i[train_mask], trans_j[train_mask]
    models = train_idm_ensemble(
        z, action, tr_i, tr_j, action_dim=mech.action_dim,
        n_members=cfg.n_members, hidden=cfg.hidden, n_steps=cfg.n_steps,
        batch_size=cfg.batch_size, lr=cfg.lr, seed=cfg.seed,
    )

    log("=== scoring discovered edges against the oracle ===")
    keep, oracle_d = oracle_pairwise_sample(true_dist_oracle, state, disc_i, disc_j, cfg.n_oracle_sample)
    disc_i, disc_j, disc_sqerr = disc_i[keep], disc_j[keep], disc_sqerr[keep]
    preds = ensemble_predict(models, z, disc_i, disc_j)
    var_a = preds.var(axis=0).mean(axis=-1)

    thr = cfg.oracle_false_edge_threshold
    overall_precision = float(np.mean(oracle_d <= thr))
    log(f"[E2] {len(disc_i)} discovered edges scored | overall precision "
        f"(oracle_dist<={thr}): {overall_precision:.3f}")

    rho, p = sps.spearmanr(var_a, oracle_d)
    log(f"[E2] Spearman(Var(a), oracle_dist) = {rho:.4f} (p={p:.1e}) "
        f"-- positive means higher ensemble variance goes with a worse (longer) real "
        f"distance, i.e. variance does track correctness")

    precision_by_quantile = {}
    for q in (0.1, 0.25, 0.5, 0.75, 1.0):
        tau = float(np.quantile(var_a, q))
        mask = var_a <= tau
        if mask.sum() >= 10:
            prec = float(np.mean(oracle_d[mask] <= thr))
            precision_by_quantile[str(q)] = dict(n=int(mask.sum()), var_threshold=tau, precision=prec)
            log(f"[E2] keep lowest-variance {int(q * 100)}% ({mask.sum()} edges, "
                f"tau={tau:.5f}): precision={prec:.3f}")

    log("=== baseline: existing plain-metric k-capped identification edges ===")
    id_i, id_j, rho_hat, eps2 = build_identification_edges(z, ep_idx, step_idx, k=cfg.k_actions)
    keep_b, oracle_d_baseline = oracle_pairwise_sample(
        true_dist_oracle, state, id_i, id_j, cfg.n_oracle_sample, seed=1,
    )
    baseline_precision = float(np.mean(oracle_d_baseline <= thr))
    log(f"[E2] baseline: {len(id_i)} kNN identification edges, {len(keep_b)} oracle-scored | "
        f"precision (oracle_dist<={thr}): {baseline_precision:.3f}")

    best_q, best_precision = max(
        ((q, v["precision"]) for q, v in precision_by_quantile.items()), key=lambda kv: kv[1],
    )
    passed = best_precision > baseline_precision + 0.05  # not a principled margin -- a coarse
    # pass/fail flag only, inspect the actual precision numbers logged above
    log(f"[E2] best discovered-edge precision={best_precision:.3f} (quantile {best_q}) vs. "
        f"baseline kNN precision={baseline_precision:.3f}")
    log(f"[E2] kill criterion (discovered precision clearly beats baseline kNN precision): "
        f"{'PASSED' if passed else 'FAILED'}")

    out_dir = ROOT / "outputs" / f"idm_gate_{cfg.env.output_prefix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = dict(
        env=cfg.env.name,
        n_discovered_raw=int(len(disc_i)),
        error_floor=floor,
        overall_precision=overall_precision,
        spearman_var_vs_oracle=dict(rho=float(rho), p=float(p)),
        precision_by_variance_quantile=precision_by_quantile,
        baseline_n_edges=int(len(id_i)), baseline_n_oracle_scored=int(len(keep_b)),
        baseline_precision=baseline_precision,
        best_quantile=best_q, best_precision=best_precision,
        passed_kill_criterion=passed,
    )
    out_path = out_dir / "e2_edge_discovery_gate_results.json"
    out_path.write_text(json.dumps(summary, indent=2))
    log(f"\nwrote {out_path}")
    log(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
