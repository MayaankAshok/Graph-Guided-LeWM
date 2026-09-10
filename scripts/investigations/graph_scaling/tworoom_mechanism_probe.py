"""Mechanism probe: is the graph's advantage over doing nothing coming from cross-episode
STITCHING (identification edges providing shortcuts / connectivity between otherwise-
separate trajectories), or does it ALSO improve ranking quality within a single episode's
own reachable set (a denoising/consistency effect, independent of stitching)?

Method: build two graphs per tier from the identical calibration (rho_hat, eps2) and
transition edges -- (a) the real graph (transitions + identification edges, i.e. what
B0-B4 actually use) and (b) a transitions-only graph (identification edges dropped
entirely, so cross-episode landmark pairs are only connected if Dijkstra happens to find
no path at all -- most cross-episode pairs become disconnected). Evaluate Spearman(true,
graph) on a FIXED set of held-out-style pairs, split into same-episode and cross-episode
subsets.

Expected/uninteresting result if the graph is PURE stitching: same-episode Spearman is
about equal between (a) and (b) (transitions-only already gives the exact literal
trajectory distance, hard to beat, for the straight-line portion) and cross-episode
Spearman collapses for (b) (nothing connects those pairs without identification edges).

Informative result if the graph does MORE than stitching: same-episode Spearman is
measurably higher for (a) than (b) too -- meaning identification edges are finding
genuine shortcuts THROUGH other episodes that beat the real trajectory's own (possibly
suboptimal/wandering) path back to a same-episode goal, not just plugging disconnected
components together.

Local, CPU/GPU-light, no RL training -- pure graph construction + Dijkstra + Spearman.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from tworoom_b0_graph_gate import build_true_distance_oracle, log
from tworoom_graph_variants import (
    OUT_DIR, TIERS, compute_calibration, evaluate_phi_quality, find_graph_edges,
    load_tier_raw_arrays, make_eval_pairs, phi_dist_at_pairs,
)
from scipy.sparse import coo_matrix


def build_graph(n, trans_i, trans_j, id_i, id_j):
    rows = np.concatenate([trans_i, trans_j, id_i, id_j])
    cols = np.concatenate([trans_j, trans_i, id_j, id_i])
    weights = np.ones(len(rows), dtype=np.float64)
    return coo_matrix((weights, (rows, cols)), shape=(n, n)).tocsr()


def main():
    true_dist_oracle = build_true_distance_oracle()
    results = {}

    for tier in TIERS:
        log(f"\n{'='*70}\n{tier}\n{'='*70}")
        z, ep_idx, step_idx, proprio, action = load_tier_raw_arrays(tier)
        n = z.shape[0]
        rho_hat, eps2 = compute_calibration(z, ep_idx, step_idx)
        trans_i, trans_j, id_i, id_j = find_graph_edges(z, ep_idx, step_idx, eps2)
        log(f"[{tier}] n={n} transition_edges={len(trans_i)} identification_edges={len(id_i)} "
            f"(rho_hat={rho_hat:.4f} eps2={eps2:.2f})")

        empty = np.array([], dtype=np.int64)
        graph_full = build_graph(n, trans_i, trans_j, id_i, id_j)
        graph_transonly = build_graph(n, trans_i, trans_j, empty, empty)

        ei, ej, true_d, same_ep = make_eval_pairs(ep_idx, proprio, true_dist_oracle)
        log(f"[{tier}] eval pairs: {len(ei)} total ({int(same_ep.sum())} same-episode, "
            f"{int((~same_ep).sum())} cross-episode)")

        phi_full = phi_dist_at_pairs(graph_full, ei, ej)
        phi_transonly = phi_dist_at_pairs(graph_transonly, ei, ej)

        q_full = evaluate_phi_quality(phi_full, same_ep, true_d, "full_graph")
        q_transonly = evaluate_phi_quality(phi_transonly, same_ep, true_d, "transitions_only")

        for q in (q_full, q_transonly):
            row = q
            log(f"[{tier}] {row['label']:<18} overall={row['overall']['spearman']:.4f} "
                f"same_ep={row['same_episode']['spearman'] if row['same_episode'] else float('nan'):.4f} "
                f"cross_ep={row['cross_episode']['spearman'] if row['cross_episode'] else float('nan'):.4f}")

        results[tier] = dict(
            n=n, n_transition_edges=int(len(trans_i)), n_identification_edges=int(len(id_i)),
            rho_hat=rho_hat, eps2=eps2, full_graph=q_full, transitions_only=q_transonly,
        )

    out_path = OUT_DIR / "mechanism_probe_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    log(f"\nwrote {out_path}")

    log(f"\n{'='*100}\nSUMMARY: Spearman(true, graph), full graph vs transitions-only (no identification edges)\n{'='*100}")
    log(f"{'tier':<13}{'same_ep (full)':>16}{'same_ep (trans-only)':>22}{'delta':>9}"
        f"{'cross_ep (full)':>18}{'cross_ep (trans-only)':>23}")
    for tier in TIERS:
        r = results[tier]
        sf = r["full_graph"]["same_episode"]["spearman"] if r["full_graph"]["same_episode"] else float("nan")
        st = r["transitions_only"]["same_episode"]["spearman"] if r["transitions_only"]["same_episode"] else float("nan")
        cf = r["full_graph"]["cross_episode"]["spearman"] if r["full_graph"]["cross_episode"] else float("nan")
        ct = r["transitions_only"]["cross_episode"]["spearman"] if r["transitions_only"]["cross_episode"] else float("nan")
        log(f"{tier:<13}{sf:>16.4f}{st:>22.4f}{sf-st:>9.4f}{cf:>18.4f}{ct:>23.4f}")


if __name__ == "__main__":
    main()
