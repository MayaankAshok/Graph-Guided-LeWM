"""Re-export shim -- everything except load_tier_raw_arrays moved to common/graph_lib.py
(environment-agnostic; that module's docstring has the full history). load_tier_raw_arrays
stays here since it's genuinely Two-Room-tier-dispatch-specific (references
tworoom_b3_datatiers/tworoom_b3_mixed_large's tier-building functions directly), used only
by the out-of-scope historical diagnostic scripts (tworoom_mechanism_probe.py,
tworoom_graph_construction_bench.py, tworoom_edge_cap_sweep.py), which keep working
unchanged via this shim.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.graph_lib import (  # noqa: F401
    build_id_edges_bruteforce, build_id_edges_faiss_capped, build_id_edges_faiss_hnsw,
    build_id_edges_kdtree, build_identification_edges, compute_calibration,
    edge_set_recall_precision, evaluate_phi_quality, full_phi_dist_matrix,
    get_ablation_dist_matrix, make_eval_pairs, phi_dist_at_pairs,
)
from tworoom_b0_graph_gate import build_true_distance_oracle, find_graph_edges, log  # noqa: F401
from tworoom_b3_datatiers import _build_mixed_arrays, _load_real_tier_arrays
from tworoom_b3_mixed_large import TIER as MIXED_LARGE_TIER
from tworoom_b3_mixed_large import _build_mixed_large_arrays

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "outputs" / "graph_variants"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TIERS = ["expert_10", "expert_25", "expert_50", "expert_100", "mixed", "mixed_large"]
N_EVAL_PAIRS = 2000  # fixed eval-pair sample size for the quality checks


def load_tier_raw_arrays(tier):
    """(z, ep_idx, step_idx, proprio, action) for any of the 6 tiers -- same dispatch as
    tworoom_b3_datatiers.build_tier_setups, factored out so both diagnostic scripts can
    reuse it without pulling in the training-loop machinery."""
    if tier.startswith("expert_"):
        n = int(tier.split("_")[1])
        return _load_real_tier_arrays(n)
    elif tier == "mixed":
        return _build_mixed_arrays()
    elif tier == MIXED_LARGE_TIER:
        return _build_mixed_large_arrays()
    raise ValueError(f"unknown tier '{tier}'")
