"""E1 -- is there a REAL, data-grounded asymmetry to learn, and how big is it?

The original proposal's "Test 1" computed ||z_A - z_B|| vs ||z_B - z_A|| on a plain L2
norm -- a tautology (norms are symmetric by definition), not a measurement of the
environment. This replaces it with a grounded measurement: the repo's own DIRECTED
transition graph (common.graph_lib.build_weighted_graph -- edges follow real env
transitions forward-in-time only, plus symmetric identification edges linking
near-duplicate states across episodes). A path backward through this graph only exists
if some OTHER episode's trajectory happens to traverse the reverse direction. That is a
genuine, data-grounded notion of reachability, not a property of the distance function.

For the same "active pushing" pairs the original Test 1 used (block displaced > threshold
over a fixed window), this reports:
  - forward graph-geodesic distance A->B (should track the true step gap closely --
    it IS mostly the same transition edges)
  - backward graph-geodesic distance B->A (finite only if the offline data contains some
    route back; otherwise unreachable)
  - fraction of pairs with NO finite backward path at all -- the real irreversibility rate
  - for pairs that DO have a backward path, the real asymmetry gap (d_back - d_fwd) -- this
    is the actual target magnitude a quasimetric head needs to reproduce, not a made-up
    constant margin
  - the raw latent L2 gap for the same pairs, for comparison (trivially ~0, included only
    to make the contrast with the graph-grounded numbers explicit)

Run:
    python scripts/investigations/quasimetric/e1_directed_asymmetry.py env=pusht
"""

import json
import sys
import time
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig
from scipy.sparse.csgraph import dijkstra

ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from common.envs import ENV_MECHANICS
from common.graph_lib import build_graph_edges_for_env, build_weighted_graph
from common.log_util import log
from graph_gate import load_landmarks

WINDOW = 15          # matches the original Test 1's Delta t
DISP_THRESH = 20.0   # matches the original Test 1's "active pushing" filter (pixels)
ID_WEIGHT = 1.0       # established default, see build_weighted_graph's docstring


def group_by_episode(ep_idx, step_idx):
    order = np.argsort(ep_idx, kind="stable")
    ep_sorted = ep_idx[order]
    bounds = np.searchsorted(ep_sorted, np.unique(ep_sorted), side="left").tolist()
    bounds.append(len(ep_sorted))
    segs = [order[bounds[a]:bounds[a + 1]] for a in range(len(bounds) - 1)]
    return [s[np.argsort(step_idx[s], kind="stable")] for s in segs if len(s) > WINDOW]


def find_active_pushing_pairs(state, ep_idx, step_idx, block_cols=(2, 3)):
    """Same construction as the original Test 1: within-episode pairs (t, t+WINDOW) where
    the block moved more than DISP_THRESH pixels -- isolates active pushes from idle frames."""
    A, B = [], []
    for seg in group_by_episode(ep_idx, step_idx):
        L = len(seg)
        blk = state[seg][:, block_cols]
        disp = np.linalg.norm(blk[WINDOW:] - blk[:L - WINDOW], axis=1)
        active = np.where(disp > DISP_THRESH)[0]
        A.append(seg[active])
        B.append(seg[active + WINDOW])
    return np.concatenate(A), np.concatenate(B)


@hydra.main(version_base=None, config_path=str(ROOT / "config" / "graph"),
            config_name="graph_gate")
def main(cfg: DictConfig):
    t0 = time.time()
    mech = ENV_MECHANICS[cfg.env.name]
    out_dir = ROOT / "outputs" / f"b0_{cfg.env.output_prefix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    res_dir = ROOT / "outputs" / "quasimetric"
    res_dir.mkdir(parents=True, exist_ok=True)

    log(f"=== E1 directed graph asymmetry: {cfg.env.name} ===")
    z, ep_idx, step_idx, state = load_landmarks(
        mech, mech.h5_path(ROOT), out_dir, cfg.env.n_episodes, mech.ckpt_dir(ROOT),
    )
    log(f"landmarks: {z.shape[0]} frames, {len(np.unique(ep_idx))} episodes")

    trans_i, trans_j, id_i, id_j, rho_hat, eps2 = build_graph_edges_for_env(cfg.env, z, ep_idx, step_idx)
    log(f"graph edges: {len(trans_i)} transition, {len(id_i)} identification "
        f"(rho_hat={rho_hat:.4f}, eps2={eps2:.4g})")
    graph = build_weighted_graph(z.shape[0], trans_i, trans_j, id_i, id_j, ID_WEIGHT)

    A, B = find_active_pushing_pairs(state, ep_idx, step_idx)
    log(f"active-pushing pairs (block moved > {DISP_THRESH}px over {WINDOW} steps): {len(A)}")
    if len(A) == 0:
        raise RuntimeError("no active-pushing pairs found -- check block_cols / DISP_THRESH")

    # subsample sources to keep dijkstra cheap (each unique source costs one row of the
    # sparse solve regardless of how many targets are queried against it)
    rng = np.random.default_rng(0)
    if len(A) > 4000:
        keep = rng.choice(len(A), 4000, replace=False)
        A, B = A[keep], B[keep]

    uniqA = np.unique(A)
    uniqB = np.unique(B)
    log(f"unique sources: {len(uniqA)} forward, {len(uniqB)} backward -- running dijkstra...")
    D_fwd_rows = dijkstra(graph, indices=uniqA, directed=True)   # (|uniqA|, n)
    D_bwd_rows = dijkstra(graph, indices=uniqB, directed=True)   # (|uniqB|, n)
    fwd_pos = {a: i for i, a in enumerate(uniqA)}
    bwd_pos = {b: i for i, b in enumerate(uniqB)}

    d_fwd = np.array([D_fwd_rows[fwd_pos[a], b] for a, b in zip(A, B)])
    d_bwd = np.array([D_bwd_rows[bwd_pos[b], a] for a, b in zip(A, B)])
    d_l2_fwd = np.linalg.norm(z[A] - z[B], axis=1)
    d_l2_bwd = np.linalg.norm(z[B] - z[A], axis=1)  # identical to d_l2_fwd -- included to
    # make the contrast with the graph numbers explicit in the saved json, not because it's
    # informative on its own (that IS the point being made)

    finite_fwd = np.isfinite(d_fwd)
    finite_bwd = np.isfinite(d_bwd)
    irreversible_frac = float((~finite_bwd).mean())
    both_finite = finite_fwd & finite_bwd
    gap = d_bwd[both_finite] - d_fwd[both_finite]

    log(f"forward graph distance:  {d_fwd[finite_fwd].mean():.2f} +- {d_fwd[finite_fwd].std():.2f}"
        f"  ({finite_fwd.mean():.1%} finite; window={WINDOW} so this should track ~{WINDOW})")
    log(f"backward graph distance: {d_bwd[finite_bwd].mean():.2f} +- {d_bwd[finite_bwd].std():.2f}"
        f"  ({finite_bwd.mean():.1%} finite -- rest have NO route back in the offline data)")
    log(f"irreversible fraction (no backward path at all): {irreversible_frac:.1%}")
    log(f"for the {both_finite.sum()} pairs with a backward route: "
        f"real asymmetry gap d_bwd - d_fwd = {gap.mean():.2f} +- {gap.std():.2f}")
    log(f"raw latent L2 gap (baseline, by definition): "
        f"{np.abs(d_l2_fwd - d_l2_bwd).max():.8f}")

    res = dict(
        env=cfg.env.name, n_pairs=int(len(A)),
        n_edges_transition=int(len(trans_i)), n_edges_identification=int(len(id_i)),
        rho_hat=float(rho_hat), eps2=float(eps2),
        forward=dict(mean=float(d_fwd[finite_fwd].mean()), std=float(d_fwd[finite_fwd].std()),
                     frac_finite=float(finite_fwd.mean())),
        backward=dict(mean=float(d_bwd[finite_bwd].mean()) if finite_bwd.any() else None,
                      std=float(d_bwd[finite_bwd].std()) if finite_bwd.any() else None,
                      frac_finite=float(finite_bwd.mean())),
        irreversible_fraction=irreversible_frac,
        asymmetry_gap_where_reversible=dict(
            n=int(both_finite.sum()),
            mean=float(gap.mean()) if both_finite.any() else None,
            std=float(gap.std()) if both_finite.any() else None,
        ),
        raw_l2_gap_max=float(np.abs(d_l2_fwd - d_l2_bwd).max()),
        secs=time.time() - t0,
    )
    p = res_dir / f"e1_directed_asymmetry_{cfg.env.name}.json"
    p.write_text(json.dumps(res, indent=2))
    log(f"wrote {p} ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
