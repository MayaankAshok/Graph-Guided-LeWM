"""Does OUR's ACTUAL deployed cost (subgoal_tdr + phase-8 final switch) rank Experiment-2's
audit candidates better than L2? Unlike scripts/graph_candidate_ranking_diag.py (which scores
every candidate by its own min(tdr-direct-to-goal, graph-route-to-goal), i.e. the `ctg`
objective), this reproduces gas_mpc_eval.py's GraphOracle.select + the phase-8 final-phase
switch exactly: ONE subgoal node v* (or the final-goal decision) is picked ONCE per task from
the task's START latent, and every candidate in that task is then scored against that same
fixed target -- ||psi(candidate) - psi(subgoal medoid)|| if not final, or LeWM's own
||candidate - goal||^2 (z-space L2, phase 8's final_metric=l2) if the start is already within
one lookahead of the goal.

Needs z_start per task (not saved by viability_inversion_audit.py) -- see
scripts/encode_rows_standalone.py, run once to backfill it from the audit's stored start_row.

    python scripts/graph_our_cost_ranking_diag.py --z-start outputs/pusht/z_start_25_50.npz \
        --audit 25:outputs/pusht/bak/superseded_2026-09-19/experiments/viability_exp2/raw_offset25.npz \
        --audit 50:outputs/pusht/bak/superseded_2026-09-19/experiments/viability_exp2/raw_offset50.npz
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

# phase 8's OUR config (docs/gas-mpc/main.tex sec:phase8 and the Objectives paragraph)
H_TD = 8.0
LOOKAHEAD = 13.7          # one CEM horizon, calibrated TDR units
FINAL_THRESH = 13.7       # final_thresh=13.7 (one lookahead) in OUR


def load_tdr_standalone(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    c = ck["cfg"]
    tdr = TDR(192, c["tdr_dim"], c["tdr_hidden"], c["tdr_layers"])
    tdr.load_state_dict(ck["tdr"])
    tdr.eval()
    return tdr


def goal_info(centers, graph_coo, h_td, hg):
    d = np.linalg.norm(centers - hg, axis=1)
    thresh = max(h_td, 1.2 * float(d.min()))
    attach = np.nonzero(d <= thresh)[0]
    n = len(centers)
    rows = np.concatenate([graph_coo.row, np.full(len(attach), n), attach])
    cols = np.concatenate([graph_coo.col, attach, np.full(len(attach), n)])
    vals = np.concatenate([graph_coo.data, d[attach], d[attach]])
    aug = csr_matrix((vals, (rows, cols)), shape=(n + 1, n + 1))
    dist, pred = dijkstra(aug, directed=False, indices=n, return_predecessors=True)
    return dist[:n], pred[:n]


def select_subgoal(centers, dist, pred, h_td, lookahead, final_thresh, hc, hg):
    """GraphOracle.select, Alg. 1: nearest node within h_td of hc minimising
    dist_to_goal + ||hc-node||, then walk the shortest path toward the goal until the
    cumulative TDR distance from hc reaches `lookahead`. Returns (node_index or -1, final)."""
    d_goal = float(np.linalg.norm(hg - hc))
    if d_goal <= final_thresh:
        return -1, True
    d = np.linalg.norm(centers - hc, axis=1)
    near = np.nonzero(d <= h_td)[0]
    if len(near) == 0:
        near = np.arange(len(centers))
    score = dist[near] + d[near]
    if not np.isfinite(score).any():
        return int(near[np.argmin(d[near])]), False
    v = int(near[np.argmin(score)])
    total = float(d[v])
    while total < lookahead:
        nxt = int(pred[v])
        if nxt < 0 or nxt >= len(centers):
            return -1, True
        total += float(np.linalg.norm(centers[nxt] - centers[v]))
        v = nxt
    return v, False


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
    ap.add_argument("--h-td", type=float, default=H_TD)
    ap.add_argument("--lookahead", type=float, default=LOOKAHEAD)
    ap.add_argument("--final-thresh", type=float, default=FINAL_THRESH)
    ap.add_argument("--tdr-path", default="outputs/pusht/tdr_full_s0.pt")
    ap.add_argument("--graph-path", default="outputs/pusht/graph_full_s0_htd8_te0.9.pkl")
    ap.add_argument("--z-start", required=True, help="npz with z_start_<offset> arrays, from encode_rows_standalone.py")
    ap.add_argument("--audit", action="append", required=True, help="offset:path/to/raw_offsetN.npz")
    ap.add_argument("--no-l2-switch", action="store_true",
                    help="phase 8's final-phase override is final_thresh=13.7 + final_metric=l2 "
                         "(z-space L2 against the true goal once close). With this flag, the final "
                         "phase instead costs TDR distance to the true goal -- OUR's graph/TDR "
                         "mechanism end to end, never falling back to raw L2.")
    args = ap.parse_args()

    tdr = load_tdr_standalone(args.tdr_path)
    with open(args.graph_path, "rb") as fh:
        g = pickle.load(fh)
    centers = np.asarray(g["centers"], dtype=np.float32)
    node_z = np.asarray(g["node_z"], dtype=np.float32)  # medoid frame latent, per node
    graph_coo = g["graph"].tocoo()
    z_start_all = np.load(args.z_start)
    log(f"[diag] graph: {len(centers)} nodes, h_td={args.h_td} lookahead={args.lookahead} "
        f"final_thresh={args.final_thresh} final_metric={'tdr (no L2 switch)' if args.no_l2_switch else 'l2 (phase 8)'}")

    summary_all = {}
    for spec in args.audit:
        offset_s, path = spec.split(":", 1)
        offset = int(offset_s)
        d = np.load(path)
        z_pred, z_goal = d["z_pred"], d["z_goal"]
        z_start = z_start_all[f"z_start_{offset}"]
        T = d["oracle_min_steps"].astype(np.float64)
        c_pred = d["c_pred"]
        n_tasks, K, D = z_pred.shape
        with torch.no_grad():
            hs_all = tdr.phi(torch.from_numpy(z_pred.reshape(-1, D))).numpy().reshape(n_tasks, K, -1)
            hg_all = tdr.phi(torch.from_numpy(z_goal)).numpy()
            hc_all = tdr.phi(torch.from_numpy(z_start)).numpy()
            hnode_all = tdr.phi(torch.from_numpy(node_z)).numpy()

        our_rows, l2_rows = [], []
        n_final = 0
        for t in range(n_tasks):
            dist, pred = goal_info(centers, graph_coo, args.h_td, hg_all[t])
            v, final = select_subgoal(centers, dist, pred, args.h_td, args.lookahead,
                                      args.final_thresh, hc_all[t], hg_all[t])
            Tt = T[t]
            if final:
                n_final += 1
                if args.no_l2_switch:
                    cost = np.linalg.norm(hs_all[t] - hg_all[t], axis=1)        # TDR distance to true goal
                else:
                    cost = np.linalg.norm(z_pred[t] - z_goal[t], axis=1) ** 2   # phase-8 final_metric=l2
            else:
                cost = np.linalg.norm(hs_all[t] - hnode_all[v], axis=1)     # subgoal_tdr
            our_rows.append(per_task_spearman(cost, Tt))
            l2_rows.append(per_task_spearman(c_pred[t], Tt))
        our_rows, l2_rows = np.array(our_rows), np.array(l2_rows)
        ok_our, ok_l2 = np.isfinite(our_rows), np.isfinite(l2_rows)
        summary_all[offset] = dict(
            our=dict(mean=float(our_rows[ok_our].mean()), ci95=boot_ci(our_rows[ok_our]), n_tasks=int(ok_our.sum())),
            l2=dict(mean=float(l2_rows[ok_l2].mean()), ci95=boot_ci(l2_rows[ok_l2]), n_tasks=int(ok_l2.sum())),
            n_final=n_final, n_tasks=n_tasks)
        log(f"[offset {offset}] n_tasks={n_tasks} n_final={n_final}/{n_tasks} (start already within one lookahead) "
            f"our={summary_all[offset]['our']['mean']:.3f} {summary_all[offset]['our']['ci95']} "
            f"l2={summary_all[offset]['l2']['mean']:.3f} {summary_all[offset]['l2']['ci95']}")

    log("summary (mean per-task Spearman(cost, oracle T), 95% task-bootstrap CI):")
    for offset in sorted(summary_all):
        s = summary_all[offset]
        log(f"  offset {offset}: OUR(subgoal_tdr+final-switch)={s['our']['mean']:.3f} "
            f"[{s['our']['ci95'][0]:.2f},{s['our']['ci95'][1]:.2f}]  l2={s['l2']['mean']:.3f} "
            f"[{s['l2']['ci95'][0]:.2f},{s['l2']['ci95'][1]:.2f}]  n_final={s['n_final']}/{s['n_tasks']}")


if __name__ == "__main__":
    main()
