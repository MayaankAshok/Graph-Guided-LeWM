"""Why does variant=stitch underperform baseline/auxphi on Push-T's expert tiers?

Two candidate explanations, both testable without training anything:
(1) identification edges themselves are unreliable (the endpoints aren't actually the same
    state), so a stitched tuple built on a bad edge is a wholesale garbage relabeling.
(2) even on a GOOD edge, build_stitch_her_tuples samples the goal uniformly across the
    entire post-edge suffix of the other episode -- so the goal can be arbitrarily far past
    the identified point, at which distance the start row's real action has no relationship
    to reaching it (unlike ordinary same-episode HER, where the real trajectory actually
    does pass through the goal row).

Measures both directly against Push-T's true-state oracle (the same one used everywhere
else in this pipeline), and contrasts against ordinary same-episode HER as the reference
scale for "what a legitimately-labeled tuple's true distance looks like."

Run (Ada, one small tier -- landmarks already cached):
    python scripts/investigations/pusht_stitch_diagnosis/diagnose.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.envs import ENV_MECHANICS
from common.graph_lib import build_identification_edges, find_transition_edges
from common.log_util import log
from common.training import build_her_tuples, build_stitch_her_tuples
from graph_gate import load_landmarks

ROOT = Path(__file__).resolve().parents[3]
TIER_N_EPISODES = 100  # expert_100 -- landmarks already cached from prior runs


def paired_true_dist(oracle, state, idx_a, idx_b, chunk=500):
    """oracle(...) returns a full (S,T) cross matrix -- only the diagonal (a_k vs b_k) is
    wanted, so this chunks to avoid an O(n^2) matrix for large samples."""
    out = np.empty(len(idx_a), dtype=np.float64)
    for s in range(0, len(idx_a), chunk):
        e = min(s + chunk, len(idx_a))
        a, b = idx_a[s:e], idx_b[s:e]
        m = oracle(state[a], state[b])
        out[s:e] = np.diagonal(m)
    return out


def summarize(name, vals):
    p = np.percentile(vals, [10, 50, 90])
    log(f"{name}: n={len(vals)} mean={vals.mean():.2f} p10={p[0]:.2f} p50={p[1]:.2f} p90={p[2]:.2f}")


def main():
    mech = ENV_MECHANICS["pusht"]
    out_dir = ROOT / "outputs" / "b3_pusht"
    z, ep_idx, step_idx, state = load_landmarks(
        mech, mech.h5_path(ROOT), out_dir, TIER_N_EPISODES, mech.ckpt_dir(ROOT),
    )
    n = len(ep_idx)
    log(f"loaded {n} landmarks, {len(np.unique(ep_idx))} episodes")

    oracle = mech.build_true_distance_oracle(state)

    # Reference scale: true distance covered by ONE real transition (definitely-legitimate,
    # definitely-adjacent pairs) -- anchors what "close" and "far" mean in this metric.
    trans_i, trans_j = find_transition_edges(ep_idx, step_idx)
    sub = np.random.default_rng(0).choice(len(trans_i), size=min(3000, len(trans_i)), replace=False)
    one_step_dist = paired_true_dist(oracle, state, trans_i[sub], trans_j[sub])
    summarize("ONE REAL STEP (reference scale)", one_step_dist)

    # Identification edges: are the two endpoints actually the same state?
    id_i, id_j, rho_hat, eps2 = build_identification_edges(z, ep_idx, step_idx, k=4)
    log(f"identification edges: {len(id_i)} (rho_hat={rho_hat:.4f} eps2={eps2:.2f})")
    sub = np.random.default_rng(1).choice(len(id_i), size=min(3000, len(id_i)), replace=False)
    edge_dist = paired_true_dist(oracle, state, id_i[sub], id_j[sub])
    summarize("IDENTIFICATION EDGE ENDPOINTS (i.e. z_i,t1 vs z_j,t2)", edge_dist)
    bad_frac = (edge_dist > np.percentile(one_step_dist, 90)).mean()
    log(f"fraction of identification edges farther apart (true state) than a typical real "
        f"step: {bad_frac:.3f}")

    # Ordinary same-episode HER: the reference for "a legitimately-labeled tuple."
    rng = np.random.default_rng(0)
    action = np.zeros((n, 2), dtype=np.float32)  # true_dist doesn't need real actions
    her_s, her_next, her_g, her_a, her_done = build_her_tuples(ep_idx, step_idx, action, seed=0)
    sub = rng.choice(len(her_s), size=min(3000, len(her_s)), replace=False)
    her_dist = paired_true_dist(oracle, state, her_s[sub], her_g[sub])
    her_offset = step_idx[her_g[sub]] - step_idx[her_s[sub]]
    summarize("SAME-EPISODE HER (s,g) true distance", her_dist)
    log(f"same-episode HER goal offset (steps): mean={her_offset.mean():.1f} "
        f"p50={np.median(her_offset):.1f} p90={np.percentile(her_offset,90):.1f}")

    # Stitched cross-episode HER: the thing actually used by variant=stitch.
    st_s, st_next, st_g, st_a, st_done = build_stitch_her_tuples(ep_idx, step_idx, id_i, id_j, seed=0)
    log(f"stitched tuples: {len(st_s)}")
    sub = rng.choice(len(st_s), size=min(3000, len(st_s)), replace=False)
    st_dist = paired_true_dist(oracle, state, st_s[sub], st_g[sub])
    summarize("STITCHED (s,g) true distance", st_dist)

    # How much of the stitched tuple's true distance is "the edge itself" vs "how far past
    # the edge point the goal was sampled" -- isolates hypothesis (2) from hypothesis (1).
    ep_of_s = ep_idx[st_s[sub]]
    ep_of_g = ep_idx[st_g[sub]]
    # for each sampled stitched tuple, find its originating edge endpoint in g's episode
    # (approximately -- just report goal offset past whichever id edge landed in that episode
    # is not tracked per-tuple, so instead report the raw temporal gap in g's own episode
    # from episode start as a proxy for "how deep into an unrelated episode did we reach")
    log("stitched tuple goal episode differs from start episode: "
        f"{(ep_of_s != ep_of_g).mean():.3f} (should be 1.0 by construction)")

    frac_worse_than_her_p90 = (st_dist > np.percentile(her_dist, 90)).mean()
    log(f"fraction of stitched (s,g) pairs farther apart (true state) than the 90th "
        f"percentile of ordinary same-episode HER pairs: {frac_worse_than_her_p90:.3f}")


if __name__ == "__main__":
    main()
