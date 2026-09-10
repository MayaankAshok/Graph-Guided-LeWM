"""Benchmark for non-O(n^2) alternatives to the brute-force identification-edge scan.

B1 flagged this as a hard blocker for Push-T scale: identification edges scale ~n^1.96 in
COUNT (not just the O(n^2) pairwise scan needed to find them), extrapolating to ~1.5e10
edges -- infeasible either way. This benchmark checks two escapes from the pairwise-scan
side specifically (a real vector-search index, not just a faster way to do the same
all-pairs comparison):

  1. sklearn radius_neighbors (KD-tree/ball-tree, exact) -- same edges as brute force in
     principle, different data structure. At this latent dimensionality (192) tree
     methods are known to degrade toward brute-force query cost -- this measures whether
     that actually shows up here, or whether it's still a useful win at this data scale.

  2. FAISS HNSW (approximate k-NN, genuine ANN) -- the standard sub-quadratic vector-
     search approach, unaffected by dimensionality the way trees are. Retrieves top-k
     neighbors per point and filters to the calibrated threshold; recall against the
     brute-force ground truth tells us directly whether k is large enough and whether the
     resulting graph is downstream-usable.

For all 6 tiers (up to mixed_large's 10,396 landmarks, the largest available locally):
wall-clock construction time, edge-set recall/precision against brute force, and
downstream Spearman quality (same pipeline as tworoom_mechanism_probe.py) using each
method's resulting graph. Local, CPU-only-friendly, no RL training.
"""

import gc
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from tworoom_b0_graph_gate import build_true_distance_oracle, build_weighted_graph, log
from tworoom_graph_variants import (
    OUT_DIR, TIERS, build_id_edges_bruteforce, build_id_edges_faiss_hnsw, build_id_edges_kdtree,
    compute_calibration, edge_set_recall_precision, evaluate_phi_quality, find_graph_edges,
    load_tier_raw_arrays, make_eval_pairs, phi_dist_at_pairs,
)

METHODS = ["bruteforce", "kdtree", "faiss_hnsw"]


def main():
    true_dist_oracle = build_true_distance_oracle()
    results = {}

    for tier in TIERS:
        log(f"\n{'='*70}\n{tier}\n{'='*70}")
        z, ep_idx, step_idx, proprio, action = load_tier_raw_arrays(tier)
        n = z.shape[0]
        rho_hat, eps2 = compute_calibration(z, ep_idx, step_idx)

        # transition edges are identical across methods -- only the identification-edge
        # construction method varies -- so get them once via the existing exact scan.
        trans_i, trans_j, _, _ = find_graph_edges(z, ep_idx, step_idx, eps2)

        ei, ej, true_d, same_ep = make_eval_pairs(ep_idx, proprio, true_dist_oracle)

        method_edges = {}
        tier_results = dict(n=n, n_transition_edges=int(len(trans_i)), rho_hat=rho_hat, eps2=eps2, methods={})

        for method in METHODS:
            if method == "bruteforce":
                id_i, id_j, elapsed = build_id_edges_bruteforce(z, ep_idx, step_idx, eps2)
            elif method == "kdtree":
                id_i, id_j, elapsed = build_id_edges_kdtree(z, ep_idx, eps2)
            elif method == "faiss_hnsw":
                id_i, id_j, elapsed = build_id_edges_faiss_hnsw(z, ep_idx, eps2, k=256)
            method_edges[method] = (id_i, id_j)

            graph = build_weighted_graph(n, trans_i, trans_j, id_i, id_j, id_weight=1.0)
            phi = phi_dist_at_pairs(graph, ei, ej)
            quality = evaluate_phi_quality(phi, same_ep, true_d, method)

            if method == "bruteforce":
                recall, precision = 1.0, 1.0
            else:
                ref_i, ref_j = method_edges["bruteforce"]
                recall, precision, _ = edge_set_recall_precision(ref_i, ref_j, id_i, id_j, n)

            ov = quality["overall"]["spearman"] if quality["overall"] else float("nan")
            log(f"[{tier}] {method:<12} n_edges={len(id_i):>8} time={elapsed:>7.2f}s "
                f"recall={recall:.4f} precision={precision:.4f} overall_spearman={ov:.4f}")

            tier_results["methods"][method] = dict(
                n_edges=int(len(id_i)), time_sec=elapsed, recall=recall, precision=precision, quality=quality,
            )
            del graph, phi
            gc.collect()

        results[tier] = tier_results
        method_edges.clear()
        gc.collect()

    out_path = OUT_DIR / "graph_construction_bench_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    log(f"\nwrote {out_path}")

    log(f"\n{'='*100}\nSUMMARY\n{'='*100}")
    log(f"{'tier':<13}{'n':>7}  " + "".join(f"{m+' time(s)':>18}{m+' recall':>16}{m+' spearman':>18}  " for m in METHODS))
    for tier in TIERS:
        r = results[tier]
        row = f"{tier:<13}{r['n']:>7}  "
        for m in METHODS:
            mm = r["methods"][m]
            ov = mm["quality"]["overall"]["spearman"] if mm["quality"]["overall"] else float("nan")
            row += f"{mm['time_sec']:>18.2f}{mm['recall']:>16.4f}{ov:>18.4f}  "
        log(row)


if __name__ == "__main__":
    main()
