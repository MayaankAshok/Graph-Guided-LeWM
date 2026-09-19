"""Per-pair diagnostic for the hierarchical-MPC screen: how far is each (start, goal) pair
by the graph's own estimate, and which pairs does each objective actually win?

For every pair of a protocol/seed: the graph geodesic D_g(start -> goal) (goal attached as
in planning, Alg.-1 start selection), converted to env steps with the TDR's step-gap
calibration; the direct TDR distance; the raw latent L2; the initial block position/angle
error. Then, for each cached method result on the same pairs, success rate inside bins of
D_g (steps) and of initial block error. Writes docs/gas-mpc/pair_diag_{protocol}_s{seed}.tex.

    python scripts/gas_mpc_pair_diag.py --protocol cross --seed 0
"""

import argparse
import json
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.log_util import log
from gas_mpc_eval import EVAL_DIR, PAIRS_DIR, PROTOCOLS, GraphOracle
from gas_mpc_prepare import OUT, ensure_psi, graph_path, load_cache, load_tdr

ROOT = Path(__file__).resolve().parent.parent
DOC = ROOT / "docs" / "gas-mpc"


def units_to_steps(u, med):
    """Monotone interpolation of the TDR calibration (median TDR distance per step gap),
    linear extrapolation beyond the largest calibrated gap."""
    gaps = np.array(sorted(int(k) for k in med))
    units = np.array([med[g] if g in med else med[str(g)] for g in gaps])
    u = np.asarray(u, dtype=np.float64)
    out = np.interp(u, units, gaps)
    slope = (gaps[-1] - gaps[-2]) / (units[-1] - units[-2])
    beyond = u > units[-1]
    out[beyond] = gaps[-1] + (u[beyond] - units[-1]) * slope
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", default="cross")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--h-td", type=float, default=8.0)
    ap.add_argument("--te", type=float, default=0.9)
    ap.add_argument("--graph-seed", type=int, default=0)
    args = ap.parse_args()
    proto = PROTOCOLS[args.protocol]
    pair_tag = "cross_episode" if proto["pairing"] == "cross_episode" else f"same_episode_off{proto['offset']}"
    pairs = json.loads((PAIRS_DIR / f"pairs_{pair_tag}_n{args.n}_s{args.seed}.json").read_text())
    cache = load_cache()
    sr, gr = np.array(pairs["start_row"]), np.array(pairs["goal_row"])
    assert np.allclose(cache["state"][sr], np.array(pairs["start_state"], dtype=np.float32), atol=1e-3), \
        "cache rows do not line up with h5 rows"
    psi = ensure_psi(args.graph_seed)
    tdr, ck = load_tdr(args.graph_seed, cache["z"].shape[1])
    med = ck["history"][-1]["median_by_gap"]
    with open(graph_path(args.graph_seed, args.h_td, args.te), "rb") as fh:
        g = pickle.load(fh)
    oracle = GraphOracle(g, tdr, args.h_td, args.h_td, 0.0, 1.0, 5)

    rows = []
    for i in range(len(sr)):
        hs, hg = psi[sr[i]], psi[gr[i]]
        gi = oracle.goal_info(hg)
        d = np.linalg.norm(oracle.centers - hs, axis=1)
        near = np.nonzero(d <= args.h_td)[0]
        if len(near) == 0:
            near = np.arange(oracle.n)
        geo = float(np.min(gi["dist"][near] + d[near]))
        direct = float(np.linalg.norm(hs - hg))
        D = min(geo, direct)
        st, gt = np.array(pairs["start_state"][i]), np.array(pairs["goal_state"][i])
        block = float(np.linalg.norm(st[2:4] - gt[2:4]))
        ang = abs(float(st[4] - gt[4])); ang = min(ang, 2 * np.pi - ang)
        agent = float(np.linalg.norm(st[:2] - gt[:2]))
        rows.append(dict(pair=i, D_units=D, D_steps=float(units_to_steps([D], med)[0]), geo_units=geo,
                         direct_units=direct, l2_z=float(np.linalg.norm(cache["z"][sr[i]] - cache["z"][gr[i]])),
                         block_px=block, block_deg=float(np.degrees(ang)), agent_px=agent))
    D_steps = np.array([r["D_steps"] for r in rows])
    block_px = np.array([r["block_px"] for r in rows]); block_deg = np.array([r["block_deg"] for r in rows])
    log(f"[{args.protocol} s{args.seed}] graph distance start->goal (steps): median {np.median(D_steps):.0f}, "
        f"q25/q75 {np.percentile(D_steps, 25):.0f}/{np.percentile(D_steps, 75):.0f}, max {D_steps.max():.0f}; "
        f"pairs with D_g > budget {proto['budget']}: {int((D_steps > proto['budget']).sum())}/{len(rows)}")

    # method results on these pairs
    methods = {}
    for p in sorted(EVAL_DIR.glob(f"*__{args.protocol}__s{args.seed}__n{args.n}.json")):
        if re.search(r"__c\d+$", p.stem):
            continue
        r = json.loads(p.read_text())
        methods[r["method"]] = np.array(r["first_hit_step"]) >= 0
    bins_D = [(0, 50), (50, 100), (100, 200), (200, 1e9)]
    near_block = (block_px < 60) & (block_deg < 30)
    lines = ["\\begin{table}[h]\n\\centering\\small",
             f"\\caption{{Per-pair diagnostic, {args.protocol} protocol, seed {args.seed}, {args.n} pairs. "
             "$D_g$: graph geodesic from start to goal (Alg.-1 start selection, goal attached as in "
             "planning), in env steps via the TDR step-gap calibration. `block near': initial block "
             "position error $<60$\\,px and angle error $<30^\\circ$. Cells: successes / pairs in the bin.}",
             f"\\label{{tab:pair-diag-{args.protocol}-s{args.seed}}}",
             "\\begin{tabular}{l" + "r" * (len(bins_D) + 2) + "}\n\\toprule",
             "objective & " + " & ".join("$D_g\\in[%d,%s)$" % (a, ("%d" % b) if b < 1e8 else "\\infty") for a, b in bins_D)
             + " & block near & block far \\\\",
             "\\# pairs & " + " & ".join(str(int(((D_steps >= a) & (D_steps < b)).sum())) for a, b in bins_D)
             + f" & {int(near_block.sum())} & {int((~near_block).sum())} \\\\\n\\midrule"]
    for m, ok in methods.items():
        cells = [f"{int(ok[(D_steps >= a) & (D_steps < b)].sum())}" for a, b in bins_D]
        cells += [str(int(ok[near_block].sum())), str(int(ok[~near_block].sum()))]
        lines.append(m.replace("_", "\\_") + " & " + " & ".join(cells) + " \\\\")
        log(f"  {m:55s} " + " ".join(f"[{a},{b:.0f}):{int(ok[(D_steps >= a) & (D_steps < b)].sum())}"
                                     for a, b in bins_D) + f"  near:{int(ok[near_block].sum())}/{int(near_block.sum())}"
            f" far:{int(ok[~near_block].sum())}/{int((~near_block).sum())}")
    lines.append("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    DOC.mkdir(parents=True, exist_ok=True)
    out = DOC / f"pair_diag_{args.protocol}_s{args.seed}.tex"
    out.write_text("\n".join(lines))
    (OUT / f"pair_diag_{args.protocol}_s{args.seed}.json").write_text(json.dumps(dict(
        rows=rows, methods={m: ok.tolist() for m, ok in methods.items()}), indent=1))
    log(f"wrote {out}")


if __name__ == "__main__":
    main()
