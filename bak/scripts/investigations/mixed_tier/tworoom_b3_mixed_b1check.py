"""Direct answer to: is graph distance actually related to true distance-to-goal on the
mixed tier, for pairs that touch noisy-origin states -- not just the expert-only pairs
tworoom_b3_mixed_diagnosis.py's diagnostic 4 already checked (which found graph quality on
expert-only pairs is unaffected by adding noisy nodes, Spearman 0.9688 vs 0.9666). This
runs the same B0/B1 evaluate() methodology (Spearman(true, euclidean) vs Spearman(true,
graph) over sampled source/target pairs, split same-room/cross-room), but on the mixed
tier's own graph, stratified by node origin (expert vs noisy) -- the direct test of
whether Phi is simply a worse proxy for true distance wherever noisy states are involved,
which would explain why shaping (applied uniformly, weighted by how often HER tuples touch
each population) can hurt overall value learning even though it's excellent on the
majority (83.6%) expert-episode portion.
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats as sps
from scipy.sparse.csgraph import dijkstra

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from tworoom_b0_graph_gate import DEV, WALL_CENTER_X, build_true_distance_oracle, build_weighted_graph, find_graph_edges, log
from tworoom_b3_datatiers import NOISY_EP_ID_OFFSET, Q_CALIB, _build_mixed_arrays

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "outputs" / "b3_tworoom"
rng = np.random.default_rng(0)

N_QUERY_SOURCES = 300     # more than B0's 100, since we need enough per-stratum after splitting by origin
N_TARGETS_PER_SOURCE = 40


def rho_hat_of(z, ep_idx, step_idx):
    d = z.shape[1]
    order = np.lexsort((step_idx, ep_idx))
    ep_o, step_o = ep_idx[order], step_idx[order]
    adj = (ep_o[1:] == ep_o[:-1]) & (step_o[1:] == step_o[:-1] + 1)
    z_o = z[order]
    return float(np.mean(np.sum(z_o[:-1][adj] * z_o[1:][adj], axis=1)) / d)


def main():
    log(f"device={DEV}")
    z, ep_idx, step_idx, proprio, action = _build_mixed_arrays()
    n, d = z.shape
    is_noisy = ep_idx >= NOISY_EP_ID_OFFSET
    log(f"[mixed] {n} rows, {int((~is_noisy).sum())} expert-origin, {int(is_noisy.sum())} noisy-origin")

    eps2 = 2 * (1 - rho_hat_of(z, ep_idx, step_idx)) * sps.chi2.ppf(Q_CALIB, d)
    trans_i, trans_j, id_i, id_j = find_graph_edges(z, ep_idx, step_idx, eps2)
    graph = build_weighted_graph(n, trans_i, trans_j, id_i, id_j, id_weight=1.0)

    true_dist_oracle = build_true_distance_oracle()

    # sample sources so we get a good number from BOTH origins, not just proportional (noisy
    # is only 18.6% of rows, proportional sampling would starve the noisy-source stratum)
    expert_rows = np.nonzero(~is_noisy)[0]
    noisy_rows = np.nonzero(is_noisy)[0]
    src_idx = np.concatenate([
        rng.choice(expert_rows, size=N_QUERY_SOURCES // 2, replace=False),
        rng.choice(noisy_rows, size=min(N_QUERY_SOURCES // 2, len(noisy_rows)), replace=False),
    ])
    log(f"[eval] {len(src_idx)} query sources ({N_QUERY_SOURCES // 2} expert-origin, "
        f"{min(N_QUERY_SOURCES // 2, len(noisy_rows))} noisy-origin)")

    D_graph_full = dijkstra(graph, indices=src_idx, directed=False)  # (S, n)

    rows = []
    for si, s in enumerate(src_idx):
        tgt = rng.choice(n, size=N_TARGETS_PER_SOURCE, replace=False)
        tgt = tgt[tgt != s]
        true_d = true_dist_oracle(proprio[s:s+1], proprio[tgt])[0]
        eucl_d = np.linalg.norm(z[s:s+1] - z[tgt], axis=1)
        graph_d = D_graph_full[si, tgt]
        src_room = proprio[s, 0] < WALL_CENTER_X
        tgt_room = proprio[tgt, 0] < WALL_CENTER_X
        cross = src_room != tgt_room
        src_noisy = is_noisy[s]
        tgt_noisy = is_noisy[tgt]
        for k in range(len(tgt)):
            rows.append((true_d[k], eucl_d[k], graph_d[k], bool(cross[k]), bool(src_noisy), bool(tgt_noisy[k])))

    true_d = np.array([r[0] for r in rows]); eucl_d = np.array([r[1] for r in rows])
    graph_d = np.array([r[2] for r in rows]); cross = np.array([r[3] for r in rows])
    src_noisy_a = np.array([r[4] for r in rows]); tgt_noisy_a = np.array([r[5] for r in rows])
    finite = np.isfinite(true_d) & np.isfinite(graph_d) & np.isfinite(eucl_d)
    log(f"[eval] {(~finite).sum()} non-finite pairs dropped, {finite.sum()} remain")
    true_d, eucl_d, graph_d, cross = true_d[finite], eucl_d[finite], graph_d[finite], cross[finite]
    src_noisy_a, tgt_noisy_a = src_noisy_a[finite], tgt_noisy_a[finite]
    touches_noisy = src_noisy_a | tgt_noisy_a
    both_noisy = src_noisy_a & tgt_noisy_a
    both_expert = ~src_noisy_a & ~tgt_noisy_a

    def report(mask, name):
        if mask.sum() < 10:
            log(f"[eval] {name}: too few pairs ({mask.sum()}), skipping")
            return None
        rho_eucl, p_eucl = sps.spearmanr(true_d[mask], eucl_d[mask])
        rho_graph, p_graph = sps.spearmanr(true_d[mask], graph_d[mask])
        log(f"[eval] {name} (n={mask.sum()}): "
            f"Spearman(true, euclidean)={rho_eucl:.4f} (p={p_eucl:.1e})  "
            f"Spearman(true, graph)={rho_graph:.4f} (p={p_graph:.1e})")
        return dict(n=int(mask.sum()), spearman_euclidean=float(rho_eucl), spearman_graph=float(rho_graph))

    log("\n=== overall / room split (standard B0 report) ===")
    results = {}
    results["overall"] = report(np.ones_like(cross, dtype=bool), "overall")
    results["same_room"] = report(~cross, "same-room")
    results["cross_room"] = report(cross, "cross-room")

    log("\n=== origin-stratified (THE question: is graph dist related to true dist for noisy-touching pairs?) ===")
    results["both_expert"] = report(both_expert, "both-expert-origin")
    results["touches_noisy"] = report(touches_noisy, "touches-noisy (>=1 endpoint noisy-origin)")
    results["both_noisy"] = report(both_noisy, "both-noisy-origin")

    log("\n=== origin x room cross-tab ===")
    results["both_expert_same_room"] = report(both_expert & ~cross, "both-expert, same-room")
    results["both_expert_cross_room"] = report(both_expert & cross, "both-expert, cross-room")
    results["touches_noisy_same_room"] = report(touches_noisy & ~cross, "touches-noisy, same-room")
    results["touches_noisy_cross_room"] = report(touches_noisy & cross, "touches-noisy, cross-room")

    out_path = OUT_DIR / "b3_mixed_b1check.json"
    out_path.write_text(json.dumps(results, indent=2))
    log(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
