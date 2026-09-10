"""Follow-up to tworoom_graph_construction_bench.py: that benchmark found brute-force,
KD-tree, and (uncapped) FAISS-HNSW all reach identical edge sets and identical downstream
Spearman at Two-Room's scale (n<=10,396) -- and brute-force is actually the FASTEST of the
three there, since chunked cdist on this few points is cheap regardless of the O(n^2)
comparison count. So construction wall-clock is not actually the problem at this scale.

The real problem B1 already identified is EDGE COUNT: identification edges scale ~n^1.96,
extrapolating to ~1.5e10 at Push-T scale -- infeasible for Dijkstra memory/compute and for
the dense phi_dist lookups the training loop needs, regardless of how fast they're found.
A capped top-k ANN retrieval (keep only the k nearest neighbors per point that also pass
the calibration threshold, instead of ALL of them) is the actual fix for that, at the cost
of missing some legitimate identification edges for points with more than k qualifying
neighbors. This sweep measures that cost directly: as k shrinks, how much of the true
(uncapped) edge set is lost, and how much downstream Spearman quality degrades -- the
answer needed before deciding whether a top-k cap is viable ahead of B5's Push-T scale-up.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from tworoom_b0_graph_gate import build_true_distance_oracle, build_weighted_graph, log
from tworoom_graph_variants import (
    OUT_DIR, TIERS, build_id_edges_bruteforce, compute_calibration, edge_set_recall_precision,
    evaluate_phi_quality, find_graph_edges, load_tier_raw_arrays, make_eval_pairs, phi_dist_at_pairs,
)

import faiss

K_VALUES = [4, 8, 16, 32, 64]


def build_id_edges_faiss_capped(z, ep_idx, eps2, k, M=32):
    """Like build_id_edges_faiss_hnsw but genuinely caps out-degree at k -- does NOT
    widen the search to chase full recall the way the uncapped benchmark did. This is
    the realistic Push-T-scale operating point: bounded edges per node, accept missing
    some qualifying neighbors beyond the cap."""
    z32 = np.ascontiguousarray(z.astype(np.float32))
    n, d = z32.shape
    index = faiss.IndexHNSWFlat(d, M)
    index.hnsw.efConstruction = 200
    index.add(z32)
    index.hnsw.efSearch = max(64, 2 * k)
    D, I = index.search(z32, k + 1)
    rows, cols = [], []
    for i in range(n):
        for rank in range(1, k + 1):
            j = int(I[i, rank])
            if j < 0 or D[i, rank] >= eps2:
                continue  # NOTE: not `break` here -- capped search order isn't guaranteed
                # monotonic once efSearch is this tight relative to k, unlike the uncapped case
            if ep_idx[j] != ep_idx[i]:
                a, b = (i, j) if i < j else (j, i)
                rows.append(a)
                cols.append(b)
    if rows:
        pairs = np.unique(np.stack([np.array(rows), np.array(cols)], axis=1), axis=0)
        return pairs[:, 0], pairs[:, 1]
    return np.array([], dtype=np.int64), np.array([], dtype=np.int64)


def main():
    true_dist_oracle = build_true_distance_oracle()
    results = {}

    for tier in TIERS:
        log(f"\n{'='*70}\n{tier}\n{'='*70}")
        z, ep_idx, step_idx, proprio, action = load_tier_raw_arrays(tier)
        n = z.shape[0]
        rho_hat, eps2 = compute_calibration(z, ep_idx, step_idx)
        trans_i, trans_j, _, _ = find_graph_edges(z, ep_idx, step_idx, eps2)
        ref_i, ref_j, _ = build_id_edges_bruteforce(z, ep_idx, step_idx, eps2)
        ei, ej, true_d, same_ep = make_eval_pairs(ep_idx, proprio, true_dist_oracle)

        graph_ref = build_weighted_graph(n, trans_i, trans_j, ref_i, ref_j, id_weight=1.0)
        phi_ref = phi_dist_at_pairs(graph_ref, ei, ej)
        q_ref = evaluate_phi_quality(phi_ref, same_ep, true_d, "uncapped")
        ov_ref = q_ref["overall"]["spearman"]
        log(f"[{tier}] uncapped     n_edges={len(ref_i):>8} (100.0% of true) overall_spearman={ov_ref:.4f}")

        tier_row = dict(n=n, n_uncapped_edges=int(len(ref_i)), uncapped_spearman=ov_ref, k_sweep={})
        for k in K_VALUES:
            id_i, id_j = build_id_edges_faiss_capped(z, ep_idx, eps2, k)
            recall, precision, _ = edge_set_recall_precision(ref_i, ref_j, id_i, id_j, n)
            graph = build_weighted_graph(n, trans_i, trans_j, id_i, id_j, id_weight=1.0)
            phi = phi_dist_at_pairs(graph, ei, ej)
            q = evaluate_phi_quality(phi, same_ep, true_d, f"k{k}")
            ov = q["overall"]["spearman"] if q["overall"] else float("nan")
            pct_edges = 100.0 * len(id_i) / max(1, len(ref_i))
            log(f"[{tier}] k={k:<4}       n_edges={len(id_i):>8} ({pct_edges:5.1f}% of true) "
                f"recall={recall:.4f} overall_spearman={ov:.4f} (delta={ov-ov_ref:+.4f})")
            tier_row["k_sweep"][k] = dict(n_edges=int(len(id_i)), pct_of_uncapped=pct_edges,
                                           recall=recall, spearman=ov, delta_vs_uncapped=ov - ov_ref)
        results[tier] = tier_row

    out_path = OUT_DIR / "edge_cap_sweep_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    log(f"\nwrote {out_path}")

    log(f"\n{'='*100}\nSUMMARY: overall Spearman vs top-k cap on identification-edge out-degree\n{'='*100}")
    log(f"{'tier':<13}{'uncapped':>10}" + "".join(f"{'k='+str(k):>10}" for k in K_VALUES))
    for tier in TIERS:
        r = results[tier]
        row = f"{tier:<13}{r['uncapped_spearman']:>10.4f}"
        for k in K_VALUES:
            row += f"{r['k_sweep'][k]['spearman']:>10.4f}"
        log(row)


if __name__ == "__main__":
    main()
