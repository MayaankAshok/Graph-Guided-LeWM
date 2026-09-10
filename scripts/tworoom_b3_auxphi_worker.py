"""Single (tier, seed) auxiliary-Phi-regression training run, CLI-invokable so the sweep
orchestrator (tworoom_b3_auxphi_sweep.py) can launch multiple seeds as separate OS
processes running truly in parallel (not Python threads -- the training loop is numpy/CUDA
work that wouldn't actually parallelize under the GIL, so separate processes is the correct
way to get real concurrency here). Reuses the exact training loop from
tworoom_b3_mixed_auxphi.py (baseline's plain -1/0 TD objective + an auxiliary regression of
V(s,g) toward a Phi-derived pseudo-value, bypassing the noisy reward-shaping channel) and
the same resumable checkpoint infrastructure, just generalized to any tier by name.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tworoom_b0_graph_gate import log
from tworoom_b3_datatiers import _build_setup, build_tier_setups, get_true_dist_matrix
from tworoom_b3_mixed_auxphi import run_condition_resumable
from tworoom_b3_mixed_large import TIER as MIXED_LARGE_TIER, _build_mixed_large_arrays

ROOT = Path(__file__).resolve().parent.parent
B4_DIR = ROOT / "outputs" / "b4_tworoom"
B4_CKPT_DIR = B4_DIR / "checkpoints"
B4_CACHE_DIR = B4_DIR / "tier_cache"
for d_ in (B4_CKPT_DIR, B4_CACHE_DIR):
    d_.mkdir(parents=True, exist_ok=True)


def get_setup(tier):
    if tier == MIXED_LARGE_TIER:
        arrays = _build_mixed_large_arrays()
        return _build_setup(tier, *arrays)
    return build_tier_setups([tier])[tier]


def _is_capped_source(source):
    return source == "transonly" or (source.startswith("k") and source[1:].isdigit())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--aux-lambda", type=float, default=0.3)
    ap.add_argument("--distance-source", default="graph")
    args = ap.parse_args()
    if args.distance_source not in ("graph", "euclidean", "oracle") and not _is_capped_source(args.distance_source):
        raise ValueError(f"unknown --distance-source '{args.distance_source}' "
                          "(expected graph, euclidean, oracle, transonly, or k<N>)")

    # keep the already-completed "graph" condition's name unchanged (auxphi_lam0.3, no
    # source suffix) so this B4 ablation doesn't orphan or duplicate that work; only new
    # arms get a name suffix. "graph" checkpoints stay in B3's own directory (it's the
    # auxphi fix itself, not a B4-ablation-specific artifact); euclidean/oracle/transonly/
    # k<N> -- all new arms -- live under outputs/b4_tworoom/ instead.
    suffix = "" if args.distance_source == "graph" else f"_{args.distance_source}"
    name = f"auxphi{suffix}_lam{args.aux_lambda}"
    ckpt_dir = None if args.distance_source == "graph" else B4_CKPT_DIR
    log(f"[worker] tier={args.tier} seed={args.seed} name={name} "
        f"distance_source={args.distance_source} setup starting...")
    setup = get_setup(args.tier)

    dist_matrix = None
    if args.distance_source == "oracle":
        dist_matrix = get_true_dist_matrix(args.tier, setup["proprio"], cache_dir=B4_CACHE_DIR)
    elif _is_capped_source(args.distance_source):
        from tworoom_graph_variants import get_ablation_dist_matrix
        dist_matrix = get_ablation_dist_matrix(args.tier, args.distance_source, setup, B4_CACHE_DIR)

    run_condition_resumable(args.tier, name, args.aux_lambda, setup, args.seed,
                             distance_source=args.distance_source, dist_matrix=dist_matrix,
                             ckpt_dir=ckpt_dir)
    log(f"[{args.tier}/{name}/s{args.seed}] worker finished")


if __name__ == "__main__":
    main()
