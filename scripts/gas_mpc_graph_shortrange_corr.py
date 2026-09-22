"""Spearman correlation of graph distance vs. true trajectory length, held-out episodes,
stratified into short/mid/long offset buckets. Answers docs/gas-mpc/main.tex's open TODO:
"[the graph] is only a guide of long term structure (not really of short term or even mid
range??) -- [TODO] Show this with correlations."

Buckets (env steps; same-episode offset = ground truth trajectory length, since these are
expert continuations): short 10-30, mid 40-60, long 80-120.

For each bucket, samples several offsets spanning it from held-out episodes (the same
disjoint, reject-presolved convention as gas_mpc_make_tasks.py's task200u pool), encodes
start/goal frames with the frozen LeWM encoder, and computes four candidate metrics:
    l2   raw latent L2 (no graph, no TDR)
    tdr  direct TDR distance (no graph)
    geo  pure graph route distance: min over nodes within h_td of start of
         (node distance from start) + (Dijkstra distance from node to the goal, goal
         attached as a temporary node -- same construction as GraphOracle.goal_info)
    ctg  min(tdr, geo) -- the objective gas_mpc_eval.py actually scores candidates with
Then reports Spearman(metric, offset) pooled within each bucket.

    python scripts/gas_mpc_graph_shortrange_corr.py --graph-seed 0 --h-td 8 --te 0.9 \
        --n-gaps 5 --n-per-gap 60
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401 -- registers the Blosc filter the h5's `pixels` column uses
import numpy as np
import torch
from scipy import stats as sps
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.gas import DEV
from common.heldout_tasks import sample_disjoint_rows
from common.lewm_loader import load_lewm
from common.log_util import log
from gas_mpc_prepare import MECH, OUT, graph_path, load_tdr, task_heldout_episodes
from planning_cost_gate import make_encode_frame

ROOT = Path(__file__).resolve().parent.parent
DOC = ROOT / "docs" / "gas-mpc"
BUCKETS = {"short": (10, 30), "mid": (40, 60), "long": (80, 120)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph-seed", type=int, default=0)
    ap.add_argument("--h-td", type=float, default=8.0)
    ap.add_argument("--te", type=float, default=0.9)
    ap.add_argument("--n-gaps", type=int, default=5, help="distinct offsets sampled per bucket")
    ap.add_argument("--n-per-gap", type=int, default=60, help="held-out pairs per offset")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    mech = MECH
    h5_path, ckpt_dir = mech.h5_path(ROOT), mech.ckpt_dir(ROOT)
    with h5py.File(h5_path, "r", swmr=True, rdcc_nbytes=512 * 1024 * 1024) as f:
        ep_offset = f["ep_offset"][:].astype(np.int64)
        ep_len = f["ep_len"][:].astype(np.int64)
        n_rows = int(ep_offset[-1] + ep_len[-1])
        ep_col = np.repeat(np.arange(len(ep_len)), ep_len)
        step_col = np.concatenate([np.arange(l) for l in ep_len])
        state_col = mech.read_state_slice(f, 0, n_rows)
        held = task_heldout_episodes(ep_col, args.graph_seed)
        log(f"[corr] {n_rows} rows, {len(ep_len)} episodes, {len(held)} held out")

        rng = np.random.default_rng(args.seed)
        plan, gap_of = [], {}
        for name, (lo, hi) in BUCKETS.items():
            gaps = sorted(set(np.linspace(lo, hi, args.n_gaps).round().astype(int).tolist()))
            for gap in gaps:
                s, gl = sample_disjoint_rows(ep_col, step_col, state_col, args.n_per_gap,
                                             int(rng.integers(1 << 30)), "same_episode", int(gap),
                                             held, reject=mech.goal_reached)
                plan.append((name, gap, s, gl))
                gap_of[gap] = name
        log(f"[corr] buckets -> gaps: " +
            ", ".join(f"{name}:{sorted({g for n2, g, *_ in plan if n2 == name})}" for name in BUCKETS))

        all_rows = np.unique(np.concatenate([np.r_[s, gl] for _, _, s, gl in plan]))
        pixels = f["pixels"][all_rows]  # sorted unique fancy index, one read

    model = load_lewm(Path(ckpt_dir), device=DEV)
    encode = make_encode_frame(model, batch=256)
    z_uniq = encode(pixels)
    z_of = dict(zip(all_rows.tolist(), z_uniq))

    tdr, _ = load_tdr(args.graph_seed, 192)
    with open(graph_path(args.graph_seed, args.h_td, args.te), "rb") as fh:
        g = pickle.load(fh)
    centers = np.asarray(g["centers"], dtype=np.float32)
    graph_coo = g["graph"].tocoo()
    n_nodes = len(centers)

    def dist_to_goal(hg):
        """Dijkstra distance from every node to the goal, goal attached as a temporary node
        within max(h_td, 1.2*nearest) -- identical construction to GraphOracle.goal_info."""
        d = np.linalg.norm(centers - hg, axis=1)
        thresh = max(args.h_td, 1.2 * float(d.min()))
        attach = np.nonzero(d <= thresh)[0]
        n = n_nodes
        rows = np.concatenate([graph_coo.row, np.full(len(attach), n), attach])
        cols = np.concatenate([graph_coo.col, attach, np.full(len(attach), n)])
        vals = np.concatenate([graph_coo.data, d[attach], d[attach]])
        aug = csr_matrix((vals, (rows, cols)), shape=(n + 1, n + 1))
        return dijkstra(aug, directed=False, indices=n)[:n]

    results = {name: dict(offset=[], l2=[], tdr=[], geo=[], ctg=[]) for name in BUCKETS}
    with torch.no_grad():
        for name, gap, s, gl in plan:
            zs = np.stack([z_of[r] for r in s])
            zg = np.stack([z_of[r] for r in gl])
            hs = tdr.phi(torch.from_numpy(zs).to(DEV)).cpu().numpy()
            hg = tdr.phi(torch.from_numpy(zg).to(DEV)).cpu().numpy()
            l2 = np.linalg.norm(zs - zg, axis=1)
            tdr_d = np.linalg.norm(hs - hg, axis=1)
            geo = np.empty(len(s))
            for i in range(len(s)):
                dist_nodes = dist_to_goal(hg[i])
                d_start = np.linalg.norm(centers - hs[i], axis=1)
                near = np.nonzero(d_start <= args.h_td)[0]
                if len(near) == 0:
                    near = np.arange(n_nodes)
                geo[i] = float(np.min(dist_nodes[near] + d_start[near]))
            ctg = np.minimum(tdr_d, geo)
            results[name]["offset"].extend([int(gap)] * len(s))
            results[name]["l2"].extend(l2.tolist())
            results[name]["tdr"].extend(tdr_d.tolist())
            results[name]["geo"].extend(geo.tolist())
            results[name]["ctg"].extend(ctg.tolist())
            log(f"  [{name} gap={gap}] {len(s)} pairs encoded")

    summary = {}
    for name, d in results.items():
        offset = np.array(d["offset"])
        row = {"n": len(offset), "gaps": sorted(set(offset.tolist()))}
        for metric in ("l2", "tdr", "geo", "ctg"):
            row[f"rho_{metric}"] = float(sps.spearmanr(d[metric], offset)[0])
        summary[name] = row
        log(f"[{name}] n={row['n']} gaps={row['gaps']} " +
            " ".join(f"{m}={row[f'rho_{m}']:.3f}" for m in ("l2", "tdr", "geo", "ctg")))

    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"graph_shortrange_corr_s{args.graph_seed}_htd{args.h_td:g}_te{args.te:g}.json"
    out.write_text(json.dumps(dict(args=vars(args), buckets=BUCKETS, summary=summary,
                                   raw=results), indent=1))
    log(f"wrote {out}")


if __name__ == "__main__":
    main()
