"""Does the graph rank CANDIDATE endpoints for a fixed (start, goal) better than LeWM's own
terminal L2, and does the answer flip with horizon? Reuses the existing Experiment-2
privileged-oracle audit artifacts (viability_inversion_audit.py's raw_offset{N}.npz: for
every (task, candidate) it stores z_pred, z_goal, and the oracle's true minimum recovery
time T), and scores the SAME candidates with a graph-distance cost, so the comparison is
exactly apples-to-apples with the already-published L2 numbers in docs/gas-mpc/main.tex
(Spearman(L2, oracle) vs. oracle: 0.72 at offset 25, 0.36 at offset 50).

This is a different question from scripts/gas_mpc_graph_shortrange_corr.py, which asked
whether distance-between-two-real-states tracks elapsed trajectory time. Here start and
goal are fixed per task; only the CANDIDATE endpoint varies, which is exactly what CEM's
own cost function has to rank every iteration.

    python scripts/graph_candidate_ranking_diag.py \
        --audit 25:outputs/pusht/bak/superseded_2026-09-19/experiments/viability_exp2/raw_offset25.npz \
        --audit 50:outputs/pusht/bak/superseded_2026-09-19/experiments/viability_exp2/raw_offset50.npz \
        --audit 100:outputs/pusht/bak/superseded_2026-09-19/experiments/viability_minsteps_large/raw_offset100.npz
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import stats as sps
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.gas import TDR
from common.log_util import log


def load_tdr_standalone(path):
    """Loads TDR weights directly, bypassing gas_mpc_prepare.load_tdr's cache_train.npz
    provenance check (not available on a machine without the full training cache) -- the
    checkpoint itself is the same seed-0 asset every headline result in main.tex uses."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    c = ck["cfg"]
    tdr = TDR(192, c["tdr_dim"], c["tdr_hidden"], c["tdr_layers"])
    tdr.load_state_dict(ck["tdr"])
    tdr.eval()
    return tdr


def dist_node_to_goal(centers, graph_coo, h_td, hg):
    d = np.linalg.norm(centers - hg, axis=1)
    thresh = max(h_td, 1.2 * float(d.min()))
    attach = np.nonzero(d <= thresh)[0]
    n = len(centers)
    rows = np.concatenate([graph_coo.row, np.full(len(attach), n), attach])
    cols = np.concatenate([graph_coo.col, attach, np.full(len(attach), n)])
    vals = np.concatenate([graph_coo.data, d[attach], d[attach]])
    aug = csr_matrix((vals, (rows, cols)), shape=(n + 1, n + 1))
    return dijkstra(aug, directed=False, indices=n)[:n]


def graph_costs(centers, graph_coo, h_td, hs, hg):
    """hs: (K, dim) candidate psi; hg: (dim,) goal psi. Returns (ctg, direct, geo), each (K,)."""
    dn = dist_node_to_goal(centers, graph_coo, h_td, hg)
    direct = np.linalg.norm(hs - hg, axis=1)
    K = len(hs)
    geo = np.empty(K)
    for i in range(K):
        d_start = np.linalg.norm(centers - hs[i], axis=1)
        near = np.nonzero(d_start <= h_td)[0]
        if len(near) == 0:
            near = np.arange(len(centers))
        geo[i] = float(np.min(dn[near] + d_start[near]))
    return np.minimum(direct, geo), direct, geo


def per_task_spearman(cost, T):
    if len(cost) < 3 or np.ptp(T) == 0:
        return float("nan")
    return float(sps.spearmanr(cost, T).statistic)


def boot_ci(values, n=5000, seed=0):
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if len(v) < 2:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), size=(n, len(v)))].mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h-td", type=float, default=8.0)
    ap.add_argument("--tdr-path", default="outputs/pusht/tdr_full_s0.pt")
    ap.add_argument("--graph-path", default="outputs/pusht/graph_full_s0_htd8_te0.9.pkl")
    ap.add_argument("--audit", action="append", required=True,
                    help="offset:path/to/raw_offsetN.npz, repeatable")
    args = ap.parse_args()

    tdr = load_tdr_standalone(args.tdr_path)
    with open(args.graph_path, "rb") as fh:
        g = pickle.load(fh)
    centers = np.asarray(g["centers"], dtype=np.float32)
    graph_coo = g["graph"].tocoo()
    log(f"[diag] graph: {len(centers)} nodes, h_td={args.h_td}")

    summary_all = {}
    for spec in args.audit:
        offset_s, path = spec.split(":", 1)
        offset = int(offset_s)
        d = np.load(path)
        z_pred, z_goal = d["z_pred"], d["z_goal"]
        T = d["oracle_min_steps"].astype(np.float64)
        c_pred = d["c_pred"]  # LeWM's own ||z_pred - z_goal||^2, already stored by the audit
        n_tasks, K, D = z_pred.shape
        has_true = "z_true" in d.files and "c_true" in d.files
        with torch.no_grad():
            hs_all = tdr.phi(torch.from_numpy(z_pred.reshape(-1, D))).numpy().reshape(n_tasks, K, -1)
            hg_all = tdr.phi(torch.from_numpy(z_goal)).numpy()
            if has_true:
                z_true, c_true = d["z_true"], d["c_true"]
                hs_true_all = tdr.phi(torch.from_numpy(z_true.reshape(-1, D))).numpy().reshape(n_tasks, K, -1)
        keys = ["l2", "tdr", "geo", "ctg"] + (["l2_true", "tdr_true", "geo_true", "ctg_true"] if has_true else [])
        rows = {k: [] for k in keys}
        for t in range(n_tasks):
            ctg, direct, geo = graph_costs(centers, graph_coo, args.h_td, hs_all[t], hg_all[t])
            Tt = T[t]
            rows["l2"].append(per_task_spearman(c_pred[t], Tt))
            rows["tdr"].append(per_task_spearman(direct, Tt))
            rows["geo"].append(per_task_spearman(geo, Tt))
            rows["ctg"].append(per_task_spearman(ctg, Tt))
            if has_true:
                ctg_t, direct_t, geo_t = graph_costs(centers, graph_coo, args.h_td, hs_true_all[t], hg_all[t])
                rows["l2_true"].append(per_task_spearman(c_true[t], Tt))
                rows["tdr_true"].append(per_task_spearman(direct_t, Tt))
                rows["geo_true"].append(per_task_spearman(geo_t, Tt))
                rows["ctg_true"].append(per_task_spearman(ctg_t, Tt))
        summary = {}
        for k, v in rows.items():
            v = np.array(v)
            ok = np.isfinite(v)
            summary[k] = dict(mean=float(v[ok].mean()), ci95=boot_ci(v[ok]), n_tasks=int(ok.sum()))
        summary_all[offset] = summary
        log(f"[offset {offset}] n_tasks={n_tasks} K={K} censored_frac={float((T > d['oracle_cap']).mean()):.2f} "
            + " ".join(f"{k}={summary[k]['mean']:.3f} [{summary[k]['ci95'][0]:.2f},{summary[k]['ci95'][1]:.2f}]"
                       for k in keys))

    log("summary (mean per-task Spearman(cost, oracle T), 95% task-bootstrap CI):")
    log(f"{'offset':>8} {'l2':>22} {'tdr':>22} {'geo':>22} {'ctg':>22}")
    for offset in sorted(summary_all):
        s = summary_all[offset]
        log(f"{offset:>8} " + " ".join(
            f"{s[k]['mean']:>7.3f}[{s[k]['ci95'][0]:.2f},{s[k]['ci95'][1]:.2f}]" for k in ("l2", "tdr", "geo", "ctg")))


if __name__ == "__main__":
    main()
