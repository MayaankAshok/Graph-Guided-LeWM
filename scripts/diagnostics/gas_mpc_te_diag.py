"""TE-threshold diagnostic for the GAS TD-aware graph (config CLAUDE.md's "TE transition"
question) -- cheap enough to run before committing to a full rebuild-and-eval sweep at a
new TE. Two checks per candidate graph, no CEM/env/critic involved:

  reachability   On the ACTUAL fixed eval pairs (outputs/pusht/pairs/
                 pairs_*_task200.json, first --n-tasks of them, same set phase6/7 use): run
                 the real Alg.-1 subgoal-selection logic (gas_mpc_eval.GraphOracle.goal_info
                 + .select) and report frac_final / frac_fallback / frac_unreachable per
                 protocol. These are exactly the failure modes that silently degrade a live
                 rollout (fallback = no node near the current state; unreachable = goal not
                 connected to any nearby node) but never crash anything, so they're invisible
                 unless checked directly.
  ranking        B0-gate style (the same methodology used since Two-Room B0): sample random
                 same-episode (start, start+gap) pairs at varying gaps (true distance =
                 gap, since these are logged, mostly-expert trajectories), and Spearman-
                 correlate the graph's best augmented distance
                 min_v dist_to_goal[v] + ||psi(start) - centers[v]||
                 against the true gap -- alongside the no-graph baseline ||psi(start)-psi(goal)||
                 for reference. Also reports the disconnected fraction (no finite path at all).

Needs no new graph builds: graph_full_s{seed}_htd{h_td}_te{te}.pkl for each --te-list value
must already exist (build with `gas_mpc_prepare.py graph --seed S --h-td H --te T` first if not).

Run (on Ada, needs GPU + PUSHT_H5_PATH):
    python scripts/diagnostics/gas_mpc_te_diag.py --te-list 0.9,0.99,0.999
    python scripts/diagnostics/gas_mpc_te_diag.py --te-list 0.9,0.99,0.999 --seed 0 --n-tasks 50 \
        --protocols cross,same50,same100 --n-rank 500 --rank-max-gap 200
"""

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import hdf5plugin  # noqa: F401 -- registers the Blosc filter the h5's `pixels` column uses
import h5py
import numpy as np
import torch
from scipy import stats as sps

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from common.envs import ENV_MECHANICS
from common.lewm_loader import load_lewm
from common.log_util import log
from gas_mpc.gas_mpc_eval import PAIRS_DIR, PROTOCOLS, GraphOracle
from gas_mpc.gas_mpc_prepare import graph_path, load_tdr
from diagnostics.planning_cost_gate import DEV, make_encode_frame

DIAG_SEED = 20260916  # independent of TASK_SEED (eval pairs) and any training seed


def load_fixed_pairs(protocol, n_tasks, task_total=200):
    proto = PROTOCOLS[protocol]
    tag = "cross_episode" if proto["pairing"] == "cross_episode" else f"same_episode_off{proto['offset']}"
    path = PAIRS_DIR / f"pairs_{tag}_task{task_total}.json"
    pairs = json.loads(path.read_text())
    assert pairs["n"] >= n_tasks, f"{path} only has {pairs['n']} tasks, need {n_tasks}"
    return dict(start_row=pairs["start_row"][:n_tasks], goal_row=pairs["goal_row"][:n_tasks])


def encode_rows(f, encode, rows):
    rows = np.asarray(rows)
    order = np.argsort(rows)
    pix = f["pixels"][rows[order]]  # h5 fancy-index needs increasing order
    pix = pix[np.argsort(order)]    # restore original order
    return encode(pix)


def sample_gap_pairs(ep_offset, ep_len, n, max_gap, rng):
    n_ep = len(ep_len)
    starts, goals, gaps = [], [], []
    while len(starts) < n:
        gap = int(rng.integers(5, max_gap + 1))
        e = int(rng.integers(n_ep))
        if ep_len[e] <= gap + 1:
            continue
        s0 = int(rng.integers(0, ep_len[e] - gap))
        starts.append(ep_offset[e] + s0)
        goals.append(ep_offset[e] + s0 + gap)
        gaps.append(gap)
    return np.array(starts), np.array(goals), np.array(gaps, dtype=np.float64)


def reachability_check(go, hc_all, hg_all):
    go.stats = dict(n_goals=0, n_attached=[], n_reachable=[], n_select=0, n_fallback=0,
                    n_unreachable=0, n_final=0, n_walk=[])
    go.cache = {}
    for hc, hg in zip(hc_all, hg_all):
        gi = go.goal_info(hg)
        go.select(hc, gi)
    return go.summary()


def ranking_check(go, hc_all, hg_all, true_gap):
    go.cache = {}
    graph_dist = np.empty(len(hc_all))
    raw_dist = np.empty(len(hc_all))
    n_disc = 0
    for i, (hc, hg) in enumerate(zip(hc_all, hg_all)):
        gi = go.goal_info(hg)
        aug = gi["dist"] + np.linalg.norm(go.centers - hc, axis=1)
        best = float(np.min(aug)) if np.isfinite(aug).any() else np.inf
        if not np.isfinite(best):
            n_disc += 1
            best = 10_000.0
        graph_dist[i] = best
        raw_dist[i] = float(np.linalg.norm(hg - hc))
    sp_graph = sps.spearmanr(graph_dist, true_gap).correlation
    sp_raw = sps.spearmanr(raw_dist, true_gap).correlation
    return dict(spearman_graph=float(sp_graph), spearman_raw=float(sp_raw),
                disconnected_frac=n_disc / len(hc_all))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--te-list", default="0.9,0.99,0.999")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--h-td", type=float, default=8.0)
    ap.add_argument("--n-tasks", type=int, default=50)
    ap.add_argument("--protocols", default="cross,same50,same100")
    ap.add_argument("--n-rank", type=int, default=500)
    ap.add_argument("--rank-max-gap", type=int, default=200)
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    te_list = [float(x) for x in args.te_list.split(",")]
    protocols = args.protocols.split(",")
    mech = ENV_MECHANICS["pusht"]
    h5_path, ckpt_dir = mech.h5_path(ROOT), mech.ckpt_dir(ROOT)

    log(f"[setup] loading LeWM + TDR seed={args.seed}")
    model = load_lewm(Path(ckpt_dir), device=DEV)
    encode = make_encode_frame(model, batch=args.batch)
    tdr, _ = load_tdr(args.seed, 192)

    results = {}
    with h5py.File(h5_path, "r", swmr=True, rdcc_nbytes=512 * 1024 * 1024) as f:
        ep_offset = f["ep_offset"][:].astype(np.int64)
        ep_len = f["ep_len"][:].astype(np.int64)

        # -- reachability pairs, one encode pass per protocol, reused across every TE --
        reach_psi = {}
        for proto in protocols:
            pairs = load_fixed_pairs(proto, args.n_tasks)
            t0 = time.time()
            z_start = encode_rows(f, encode, pairs["start_row"])
            z_goal = encode_rows(f, encode, pairs["goal_row"])
            with torch.no_grad():
                hc = tdr.phi(torch.from_numpy(z_start).to(DEV)).cpu().numpy()
                hg = tdr.phi(torch.from_numpy(z_goal).to(DEV)).cpu().numpy()
            reach_psi[proto] = (hc, hg)
            log(f"[setup] {proto}: encoded {len(pairs['start_row'])} pairs in {time.time() - t0:.1f}s")

        # -- ranking pairs (varying same-episode gaps), one encode pass, reused across every TE --
        rng = np.random.default_rng(DIAG_SEED)
        s_rows, g_rows, gaps = sample_gap_pairs(ep_offset, ep_len, args.n_rank, args.rank_max_gap, rng)
        t0 = time.time()
        z_s = encode_rows(f, encode, s_rows)
        z_g = encode_rows(f, encode, g_rows)
        with torch.no_grad():
            rank_hc = tdr.phi(torch.from_numpy(z_s).to(DEV)).cpu().numpy()
            rank_hg = tdr.phi(torch.from_numpy(z_g).to(DEV)).cpu().numpy()
        log(f"[setup] ranking: encoded {args.n_rank} gap-pairs (gap 5-{args.rank_max_gap}) in {time.time() - t0:.1f}s")

    for te in te_list:
        gpath = graph_path(args.seed, args.h_td, te)
        assert gpath.exists(), f"missing {gpath} -- build it first with gas_mpc_prepare.py graph"
        g = pickle.load(open(gpath, "rb"))
        go = GraphOracle(g, tdr, h_td=args.h_td, subgoal_threshold=args.h_td, lookahead=0.0,
                         step_units=1.0, n_waypoints=0)
        log(f"\n=== TE={te} ({gpath.name}) === stats={g['stats']}")

        for proto in protocols:
            hc, hg = reach_psi[proto]
            r = reachability_check(go, hc, hg)
            log(f"[reach] TE={te} proto={proto} n_goals={r['n_goals']} "
                f"mean_attached={r['mean_attached']:.1f} mean_reachable={r['mean_reachable']:.1f} "
                f"frac_final={r['frac_final']:.3f} frac_fallback={r['frac_fallback']:.3f} "
                f"frac_unreachable={r['frac_unreachable']:.3f}")
            results.setdefault(str(te), {}).setdefault("reach", {})[proto] = r

        rk = ranking_check(go, rank_hc, rank_hg, gaps)
        log(f"[rank]  TE={te} spearman_graph={rk['spearman_graph']:.4f} "
            f"spearman_raw(no-graph)={rk['spearman_raw']:.4f} disconnected_frac={rk['disconnected_frac']:.3f}")
        results[str(te)]["rank"] = rk
        results[str(te)]["graph_stats"] = g["stats"]

    out = ROOT / "outputs" / "pusht" / f"te_diag_s{args.seed}_htd{args.h_td:g}.json"
    out.write_text(json.dumps(results, indent=1))
    log(f"\n[done] wrote {out}")


if __name__ == "__main__":
    main()
