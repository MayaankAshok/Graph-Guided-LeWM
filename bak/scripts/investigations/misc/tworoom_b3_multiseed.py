"""Multi-seed robustness check for B3: is baseline > shaped a real pattern, or noise
from a single 5000-step run with tiny MLPs, both far from converged?

Reuses the expensive one-time setup (graph, full pairwise Dijkstra distance matrix,
HER tuples) from tworoom_b3_gciql_shaping.py and re-runs run_condition() across
multiple seeds per condition.
"""
import json
import sys
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401
import numpy as np
from scipy import stats as sps
from scipy.sparse.csgraph import dijkstra

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import tworoom_b3_gciql_shaping as m

SEEDS = [0, 1, 2]


def main():
    m.log(f"device={m.DEV}")
    d_land = np.load(m.LANDMARKS)
    z, ep_idx, step_idx, proprio = d_land["z"], d_land["ep_idx"], d_land["step_idx"], d_land["proprio"]
    n, d = z.shape

    f = h5py.File(m.H5_PATH, "r", swmr=True)
    ep_offset = f["ep_offset"][:]
    global_idx = ep_offset[ep_idx] + step_idx
    action = f["action"][:][global_idx].astype(np.float32)
    f.close()

    order = np.lexsort((step_idx, ep_idx))
    ep_o, step_o = ep_idx[order], step_idx[order]
    adj = (ep_o[1:] == ep_o[:-1]) & (step_o[1:] == step_o[:-1] + 1)
    z_o = z[order]
    rho_hat = float(np.mean(np.sum(z_o[:-1][adj] * z_o[1:][adj], axis=1)) / d)
    eps2 = 2 * (1 - rho_hat) * sps.chi2.ppf(m.Q_CALIB, d)
    trans_i, trans_j, id_i, id_j = m.find_graph_edges(z, ep_idx, step_idx, eps2)
    graph = m.build_weighted_graph(n, trans_i, trans_j, id_i, id_j, id_weight=1.0)
    m.log(f"graph: {len(trans_i)} transition edges, {len(id_i)} identification edges")

    phi_dist = dijkstra(graph, indices=np.arange(n), directed=False)
    finite_max = phi_dist[np.isfinite(phi_dist)].max()
    phi_dist = np.where(np.isfinite(phi_dist), phi_dist, finite_max * 2)
    m.log("phi_dist precomputed")

    true_dist_oracle = m.build_true_distance_oracle()

    results = {}
    for use_shaping, name in [(False, "baseline"), (True, "shaped")]:
        results[name] = []
        for seed in SEEDS:
            m.log(f"\n{'='*50}\n{name} seed={seed}\n{'='*50}")
            s_idx, next_idx, goal_idx, act_idx, done = m.build_her_tuples(
                ep_idx, step_idx, action, seed=seed
            )
            history = m.run_condition(
                f"{name}_s{seed}", use_shaping, z, proprio, s_idx, next_idx, goal_idx,
                act_idx, done, action, phi_dist, true_dist_oracle, d, seed=seed,
            )
            results[name].append(dict(seed=seed, history=history))

    out_path = m.OUT_DIR / "b3_multiseed_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    m.log(f"\nwrote {out_path}")

    m.log("\n" + "=" * 50 + "\nSUMMARY (final-step spearman per seed)\n" + "=" * 50)
    for name in ["baseline", "shaped"]:
        finals = [r["history"][-1]["spearman_v_vs_true"] for r in results[name]]
        m.log(f"{name}: {[f'{v:.4f}' for v in finals]}  mean={np.mean(finals):.4f} std={np.std(finals):.4f}")


if __name__ == "__main__":
    main()
