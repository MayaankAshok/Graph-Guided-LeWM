"""Shared latent-graph construction and calibration library -- environment-agnostic.
Consolidated from tworoom_b0_graph_gate.py (calibrate/build_weighted_graph/find_graph_edges)
and tworoom_graph_variants.py (everything else here); both old files now import-and-re-export
from this module so their existing dependents (the out-of-scope historical diagnostic scripts:
tworoom_mechanism_probe.py, tworoom_graph_construction_bench.py, tworoom_edge_cap_sweep.py,
tworoom_b1_graph_diagnostics.py, etc.) keep working unchanged.

New in this consolidation: build_identification_edges() generalizes the "try the calibrated
threshold, fall back to the heuristic (uncalibrated) top-k construction if it fails"
logic that was previously hand-written ad hoc in pusht_b0_graph_gate.py's main(). That logic
was never actually Push-T-specific -- Two-Room's calibration just always happens to pass it
(see [[pusht-b5-b0-gate]] for why Push-T's fails: rho_hat statistically >= 1 at every
landmark scale tested). Any future environment gets this fallback for free.
"""

import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
from scipy import stats as sps
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

from common.log_util import log

DEV = "cuda" if torch.cuda.is_available() else "cpu"

# One predictor "step" spans this many real env steps, and the action fed for that step is
# the concatenation of the real actions taken in between (frame-skip, see docs/lewm.txt App.
# D and [[lewm-predictor-action-block-convention]]) -- both envs have action_dim=2, so a
# block is 5*2=10 dims, exactly action_encoder's input_dim, no padding needed. The E2 gates
# below used to feed a single action zero-padded to 10 dims instead (common.lewm_loader.
# pad_action, now superseded) -- that convention is no longer used anywhere in this file.
SKIP = 5


# ============================================================
# Calibration (rho_hat, adjacent vs unrelated displacement nulls)
# ============================================================

def calibrate(z, ep_idx, step_idx, quantiles=(0.5, 0.01, 1e-3, 1e-6), rng=None):
    """Full diagnostic version -- logs adjacent/unrelated displacement stats and a small
    quantile sweep. Used by graph_gate.py's B0-style gate check.

    rng: caller's shared np.random.Generator, threaded through so a caller's subsequent
    random draws (e.g. evaluate()'s source/target sampling) advance from where this left
    off -- matching the original tworoom_b0_graph_gate.py's single-module-rng behavior
    exactly, needed for bit-exact reproduction of already-published results. Defaults to a
    fresh seed-0 generator for standalone callers that don't care about that threading."""
    if rng is None:
        rng = np.random.default_rng(0)
    d = z.shape[1]
    order = np.lexsort((step_idx, ep_idx))
    z_o, ep_o, step_o = z[order], ep_idx[order], step_idx[order]

    same_ep = ep_o[1:] == ep_o[:-1]
    consec = step_o[1:] == step_o[:-1] + 1
    adj_mask = same_ep & consec
    z_t, z_t1 = z_o[:-1][adj_mask], z_o[1:][adj_mask]
    n_adj = z_t.shape[0]
    log(f"[calib] {n_adj} adjacent (same-episode consecutive) pairs")

    rho_hat = float(np.mean(np.sum(z_t * z_t1, axis=1)) / d)
    adj_sqdisp = np.sum((z_t - z_t1) ** 2, axis=1)
    log(f"[calib] rho_hat = {rho_hat:.4f}")
    log(f"[calib] adjacent ||dz||^2: mean={adj_sqdisp.mean():.2f} std={adj_sqdisp.std():.2f} "
        f"(null-implied mean 2(1-rho)d = {2*(1-rho_hat)*d:.2f})")

    n_unrel = min(200_000, n_adj * 4)
    i = rng.integers(0, z.shape[0], n_unrel)
    j = rng.integers(0, z.shape[0], n_unrel)
    diff_ep = ep_idx[i] != ep_idx[j]
    i, j = i[diff_ep], j[diff_ep]
    unrel_sqdisp = np.sum((z[i] - z[j]) ** 2, axis=1)
    log(f"[calib] unrelated (cross-episode random) ||dz||^2: mean={unrel_sqdisp.mean():.2f} "
        f"std={unrel_sqdisp.std():.2f} (n={len(i)}) (marginal-null-implied mean 2d = {2*d})")

    thresholds = {}
    for q in quantiles:
        # BUG (found empirically): connecting i,j means asserting "this pair looks like it
        # could be adjacent" -- that's a claim we should only make when ||dz||^2 sits in the
        # SMALL (left) tail of the adjacent-pair null, i.e. eps^2 = 2(1-rho)*chi2^{-1}(q) for
        # SMALL q. An earlier version used chi2^{-1}(1-alpha) with small alpha, which pushes
        # eps^2 UP as alpha shrinks -- exactly backwards.
        eps2 = 2 * (1 - rho_hat) * sps.chi2.ppf(q, d)
        frac_adj_below = float(np.mean(adj_sqdisp < eps2))
        frac_unrel_below = float(np.mean(unrel_sqdisp < eps2))
        thresholds[q] = eps2
        log(f"[calib] q={q:g}: eps^2={eps2:.2f}  "
            f"P(adjacent < eps^2)={frac_adj_below:.4f}  P(unrelated < eps^2)={frac_unrel_below:.6f}")

    return rho_hat, thresholds, adj_sqdisp, unrel_sqdisp


def compute_calibration(z, ep_idx, step_idx, q_calib=1e-3):
    """Quiet single-value version (rho_hat, eps2) -- used inside the RL-training tier-setup
    path (build_identification_edges, the "capped" identification-edge construction) where
    the full calibrate() logging would be noise on every worker invocation.

    eps2 is derived directly from the measured adjacent-pair squared displacement and the
    embedding's effective degrees of freedom (participation ratio of its covariance), NOT
    through calibrate()'s rho_hat/d formula (2*(1-rho_hat)*chi2.ppf(q,d)), which implicitly
    assumes E[||z||^2]==d and dof==d. That assumption is close enough on Two-Room but wrong
    enough on Push-T to flip eps2 negative: measured E[||z||^2]=193.2 (not d=192) and
    effective dof ~84 (not d=192, per the covariance's participation ratio), pushing
    rho_hat statistically >= 1. The formula below is a mean of squared displacements
    (always >= 0) times a chi2 quantile (always >= 0 for d_eff>0, q in (0,1)), so it is
    well-defined and positive by construction for any environment -- no "calibration
    failed" branch is needed downstream (see build_identification_edges).

    NOTE: calibrate()'s own formula (used by Two-Room's "bruteforce" identification-edge
    path) is deliberately left as-is, to preserve its validated bit-exact historical
    numbers -- see CLAUDE.md. This fix is scoped to the "capped" path, which is what
    actually broke (Push-T); Two-Room's bruteforce calibration never failed."""
    n, d = z.shape
    order = np.lexsort((step_idx, ep_idx))
    ep_o, step_o = ep_idx[order], step_idx[order]
    adj = (ep_o[1:] == ep_o[:-1]) & (step_o[1:] == step_o[:-1] + 1)
    if not np.any(adj):
        raise ValueError("compute_calibration: no adjacent (same-episode consecutive) "
                          "landmark pairs found -- does every episode have <= 1 row?")
    z_o = z[order]
    zt, zt1 = z_o[:-1][adj], z_o[1:][adj]
    rho_hat = float(np.mean(np.sum(zt * zt1, axis=1)) / d)  # kept for logging only --
                                                              # eps2 no longer derived from it

    adj_sqdisp = np.sum((zt - zt1) ** 2, axis=1)
    zc = z - z.mean(axis=0, keepdims=True)
    eig = np.clip(np.linalg.eigvalsh(np.cov(zc, rowvar=False)), 0, None)
    eig_sum = float(eig.sum())
    d_eff = (eig_sum ** 2) / float(np.sum(eig ** 2)) if eig_sum > 0 else float(d)

    eps2 = float(adj_sqdisp.mean() / d_eff * sps.chi2.ppf(q_calib, d_eff))
    return rho_hat, eps2


# ============================================================
# Transition edges (consecutive same-episode frames -- always exact, always cheap)
# ============================================================

def find_transition_edges(ep_idx, step_idx):
    """Return (transition_i, transition_j), i<j, undirected. Standalone version of the
    transition-edge half of find_graph_edges(), for callers using a non-brute-force
    identification-edge method (e.g. build_identification_edges) that don't want find_graph_edges'
    O(n^2) identification scan running too."""
    order = np.lexsort((step_idx, ep_idx))
    ep_o, step_o = ep_idx[order], step_idx[order]
    same_ep = ep_o[1:] == ep_o[:-1]
    consec = step_o[1:] == step_o[:-1] + 1
    trans_mask = same_ep & consec
    a = order[:-1][trans_mask]
    b = order[1:][trans_mask]
    log(f"[graph] transition edges: {len(a)}")
    return a, b


# ============================================================
# Identification-edge construction: brute force (exact, O(n^2)) + faster alternatives
# ============================================================

def find_graph_edges(z, ep_idx, step_idx, eps2, batch=1000):
    """Return (transition_i, transition_j, identification_i, identification_j), each i<j,
    undirected. Exact O(n^2) chunked pairwise scan for identification edges -- the
    original B0-B4 method; do not reintroduce as the default for a new environment without
    a specific reason (see B1's landmark-count^1.96 edge-count blowup)."""
    n = z.shape[0]
    order = np.lexsort((step_idx, ep_idx))

    ep_o, step_o = ep_idx[order], step_idx[order]
    same_ep = ep_o[1:] == ep_o[:-1]
    consec = step_o[1:] == step_o[:-1] + 1
    trans_mask = same_ep & consec
    a = order[:-1][trans_mask]
    b = order[1:][trans_mask]
    log(f"[graph] transition edges: {len(a)}")

    z_t = torch.from_numpy(z).to(DEV)
    ep_t = torch.from_numpy(ep_idx).to(DEV)
    id_rows, id_cols = [], []
    for i0 in range(0, n, batch):
        i1 = min(n, i0 + batch)
        chunk = z_t[i0:i1]  # (b,d)
        sqd = torch.cdist(chunk, z_t, p=2) ** 2  # (b,n)
        diff_ep = ep_t[i0:i1].unsqueeze(1) != ep_t.unsqueeze(0)
        below = (sqd < eps2) & diff_ep
        idx_i, idx_j = torch.nonzero(below, as_tuple=True)
        idx_i = idx_i + i0
        keep = idx_i < idx_j
        id_rows.append(idx_i[keep].cpu().numpy())
        id_cols.append(idx_j[keep].cpu().numpy())
        if (i0 // batch) % 10 == 0:
            log(f"[graph] identification-edge scan {i1}/{n}")

    id_rows = np.concatenate(id_rows) if id_rows else np.array([], dtype=np.int64)
    id_cols = np.concatenate(id_cols) if id_cols else np.array([], dtype=np.int64)
    log(f"[graph] identification edges (cross-episode, ||dz||^2<eps^2): {len(id_rows)}")

    return a, b, id_rows, id_cols


def build_id_edges_bruteforce(z, ep_idx, step_idx, eps2):
    """Exact, O(n^2) chunked pairwise scan -- used as ground truth for recall/precision and
    as the timing baseline in construction-method comparisons."""
    t0 = time.time()
    _, _, id_i, id_j = find_graph_edges(z, ep_idx, step_idx, eps2)
    return id_i, id_j, time.time() - t0


def build_id_edges_kdtree(z, ep_idx, eps2):
    """Exact radius-neighbors query via sklearn's tree-based index. Same edge SET as brute
    force (up to floating-point tie-breaking at the threshold boundary)."""
    from sklearn.neighbors import NearestNeighbors

    t0 = time.time()
    eps = float(np.sqrt(eps2))
    nn = NearestNeighbors(radius=eps, algorithm="auto")
    nn.fit(z)
    neighbors = nn.radius_neighbors(z, return_distance=False)
    rows, cols = [], []
    for i, nbrs in enumerate(neighbors):
        for j in nbrs:
            if j > i and ep_idx[j] != ep_idx[i]:
                rows.append(i)
                cols.append(j)
    return np.array(rows, dtype=np.int64), np.array(cols, dtype=np.int64), time.time() - t0


def build_id_edges_faiss_hnsw(z, ep_idx, eps2, k=64, M=32):
    """Approximate k-NN via FAISS's HNSW graph index, widened (k=64 default) to chase full
    recall of the true within-threshold neighborhood rather than capping out-degree."""
    import faiss

    t0 = time.time()
    z32 = np.ascontiguousarray(z.astype(np.float32))
    n, d = z32.shape
    index = faiss.IndexHNSWFlat(d, M)
    index.hnsw.efConstruction = 200
    index.add(z32)
    index.hnsw.efSearch = max(128, 2 * k)  # efSearch below k starves the top-k results
    D, I = index.search(z32, k + 1)  # +1 since each point is its own nearest neighbor
    rows, cols = [], []
    for i in range(n):
        for rank in range(1, k + 1):  # skip rank 0 (self)
            j = int(I[i, rank])
            if j < 0:
                continue
            if D[i, rank] >= eps2:
                break  # faiss returns neighbors in increasing distance order
            if j > i and ep_idx[j] != ep_idx[i]:
                rows.append(i)
                cols.append(j)
            elif j < i and ep_idx[j] != ep_idx[i]:
                rows.append(j)
                cols.append(i)
    if rows:
        pairs = np.unique(np.stack([np.array(rows), np.array(cols)], axis=1), axis=0)
        id_i, id_j = pairs[:, 0], pairs[:, 1]
    else:
        id_i, id_j = np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    return id_i, id_j, time.time() - t0


def build_id_edges_faiss_capped(z, ep_idx, step_idx, eps2, k, min_step_gap=5, M=32):
    """FAISS-HNSW k-NN with out-degree genuinely capped at k (does NOT widen the search to
    chase full recall). This is the default, recommended construction method: confirmed on
    Two-Room to cost nothing in downstream Spearman quality even at k=4 (often slightly
    better), and to actually beat the uncapped graph on the real RL-training metric at every
    tier tested (see [[tworoom-b4-extension-edge-cap-rl]]). eps2 must be finite: this
    function is meant purely as a fast approximation of the exact O(n^2) brute-force
    threshold scan (find_graph_edges), not a different edge criterion -- every edge it
    returns must also satisfy the same threshold the brute-force scan would apply, so its
    output is always a SUBSET of what the exact scan would find (fewer edges, from HNSW's
    approximate recall and the k-cap, never a DIFFERENT edge). An unbounded eps2 would
    accept a node's k nearest neighbors regardless of true distance -- not a subset of any
    thresholded search's output, so it's not supported here; see build_identification_edges,
    which always calls this with a finite, calibrated eps2.

    Same-episode kNN neighbours are allowed as identification edges too, as long as their
    step gap is >= min_step_gap -- excluding only a small window avoids duplicating
    transition edges (gap=1) or near-duplicate consecutive frames, while letting a wandering
    trajectory that loops back near an earlier state get the shortcut its revisit actually
    earns. Without this, a noisy/wandering rollout (e.g. Push-T's WeakPolicy) keeps the full
    loop length in phi even when it revisits a near-identical state, since every same-episode
    neighbour used to be discarded outright regardless of step gap."""
    import faiss

    z32 = np.ascontiguousarray(z.astype(np.float32))
    n, d = z32.shape
    index = faiss.IndexHNSWFlat(d, M)
    index.hnsw.efConstruction = 200
    index.add(z32)
    index.hnsw.efSearch = max(64, 2 * k)
    D, I = index.search(z32, k + 1)
    rows, cols = [], []
    for i in range(n):
        for rank in range(1, k + 1):
            j = int(I[i, rank])
            if j < 0 or D[i, rank] >= eps2:
                continue  # not `break` -- capped search order isn't guaranteed monotonic
                # once efSearch is this tight relative to k, unlike the uncapped case
            if ep_idx[j] != ep_idx[i] or abs(int(step_idx[j]) - int(step_idx[i])) >= min_step_gap:
                a, b = (i, j) if i < j else (j, i)
                rows.append(a)
                cols.append(b)
    if rows:
        pairs = np.unique(np.stack([np.array(rows), np.array(cols)], axis=1), axis=0)
        return pairs[:, 0], pairs[:, 1]
    return np.array([], dtype=np.int64), np.array([], dtype=np.int64)


def build_identification_edges(z, ep_idx, step_idx, k, q=1e-3, eps2_override=None):
    """Calibrated identification edges via build_id_edges_faiss_capped's k-capped,
    threshold-checked kNN search -- a fast approximation of the exact O(n^2) brute-force
    threshold scan (find_graph_edges), not a different edge criterion: every edge returned
    also satisfies the same eps2 threshold the brute-force scan would apply, so the output
    is always a SUBSET of what the exact scan would find.

    compute_calibration()'s eps2 is derived directly from the measured adjacent-pair
    squared displacement and the embedding's effective degrees of freedom, which is
    well-defined and positive by construction (see its docstring for why the older
    rho_hat/d formula could go negative on Push-T and this one structurally can't). There is
    deliberately no "calibration failed, fall back to an unbounded top-k" branch any more --
    an unbounded eps2 was never really an approximation of a thresholded search to begin
    with, it silently redefines "identification edge" as "one of the k nearest neighbors,
    however far away", which is a different (and much noisier) criterion, not a degraded
    version of the same one.

    eps2_override: if given, skip compute_calibration entirely and use this value directly
    -- e.g. an EMPIRICAL quantile of real adjacent-pair squared displacement (measured
    directly off the histogram), which is not the same number as compute_calibration's
    chi2-formula at the same quantile (the real distribution has heavier tails than the
    chi2 approximation -- measured on Push-T's 300-episode set: empirical 99th pct = 14.16
    vs the formula's q=0.99 value of 3.12, a 4.5x gap). rho_hat is still computed (cheap)
    purely for logging; it plays no role in eps2 when overridden.

    Returns (id_i, id_j, rho_hat, eps2)."""
    rho_hat, calibrated_eps2 = compute_calibration(z, ep_idx, step_idx, q_calib=q)
    eps2 = calibrated_eps2 if eps2_override is None else float(eps2_override)
    if not (eps2 > 0):
        raise ValueError(
            f"non-positive eps2={eps2!r} (rho_hat={rho_hat:.4f}) -- this should be "
            "structurally impossible with the adj-sqdisp/d_eff formula (both factors are "
            ">= 0 by construction) or with a sane eps2_override; investigate rather than "
            "silently falling back to an unbounded threshold."
        )
    id_i, id_j = build_id_edges_faiss_capped(z, ep_idx, step_idx, eps2, k)
    log(f"[graph] identification edges (k={k}-capped, "
        f"eps^2={eps2:.2f}{' [override]' if eps2_override is not None else ' [calibrated]'}, "
        f"rho_hat={rho_hat:.4f}): {len(id_i)}")
    return id_i, id_j, rho_hat, eps2


def build_graph_edges_for_env(env_cfg, z, ep_idx, step_idx, q=1e-3, eps2_override=None):
    """Dispatches on env_cfg.id_edge_method ("bruteforce" or "capped") -- the one identification-
    edge construction choice that's a per-environment config value, not hardcoded (see
    config/graph/env/*.yaml for why the two known environments differ). Shared by
    graph_gate.py and datatiers.py so this dispatch is written once.

    eps2_override: passed through to the "capped" path only (see build_identification_edges).
    Two-Room's "bruteforce" path doesn't support it -- not needed there, its calibration
    never fails and this override exists specifically for Push-T's identification-threshold
    experiments.

    Returns (trans_i, trans_j, id_i, id_j, rho_hat, eps2)."""
    if env_cfg.id_edge_method == "bruteforce":
        rho_hat, thresholds, _, _ = calibrate(z, ep_idx, step_idx, quantiles=(q,))
        eps2 = thresholds[q]
        trans_i, trans_j, id_i, id_j = find_graph_edges(z, ep_idx, step_idx, eps2)
        return trans_i, trans_j, id_i, id_j, rho_hat, eps2
    elif env_cfg.id_edge_method == "capped":
        trans_i, trans_j = find_transition_edges(ep_idx, step_idx)
        id_i, id_j, rho_hat, eps2 = build_identification_edges(
            z, ep_idx, step_idx, k=env_cfg.id_edge_k, q=q, eps2_override=eps2_override,
        )
        return trans_i, trans_j, id_i, id_j, rho_hat, eps2
    raise ValueError(f"unknown env.id_edge_method '{env_cfg.id_edge_method}'")


def build_cross_episode_id_pairs(id_i, id_j, ep_idx):
    """Identification-edge endpoints that cross an episode boundary -- the graph's own
    opinion of which states in DIFFERENT episodes are "the same", i.e. exactly the stitching
    links a value function needs to see in order to generalize to cross-episode goals (see
    actor_train.py's cross-episode aux term). Both directions of each edge are returned
    (id edges are symmetric, but the directed transition graph can make phi(s,g) != phi(g,s)
    once a shortest path leaves the single edge, so both are worth training on)."""
    cross = ep_idx[id_i] != ep_idx[id_j]
    ci, cj = id_i[cross], id_j[cross]
    return np.concatenate([ci, cj]), np.concatenate([cj, ci])


# ============================================================
# Weighted graph + dense distance matrix
# ============================================================

def build_weighted_graph(n, trans_i, trans_j, id_i, id_j, id_weight):
    """weight=1 per real env step for transition edges; id_weight per identification edge.

    Transition edges are DIRECTED, trans_i -> trans_j only (trans_i is always the
    earlier-step frame -- guaranteed by find_transition_edges/find_graph_edges' construction).
    An env step is not generally reversible (e.g. pushing a block in Push-T can't be undone
    by reversing the action), so Dijkstra should not be allowed to walk a transition
    backward -- that asymmetry is exactly what makes phi(s, g) a value-function-like quantity
    rather than a plain undirected geodesic. This does change Two-Room's historical bit-exact
    numbers too (its transitions are only "roughly" reversible), a deliberate tradeoff.
    Identification edges stay symmetric (added both directions) -- "same state" genuinely is
    a symmetric relation, unlike a transition.

    Callers MUST pass directed=True to scipy's dijkstra/csgraph functions on this graph now --
    directed=False re-symmetrizes every edge regardless of the matrix's actual structure and
    would silently undo this.

    id_weight=0 ("same state, free to teleport") badly UNDERperforms raw Euclidean distance:
    real identification-edge slop is the same scale as one real transition step, so treating
    it as free lets Dijkstra chain "approximately-same-state" hops into large, illegitimate
    shortcuts. id_weight=1 (one step-equivalent per hop) fixes this and is the default."""
    rows = np.concatenate([trans_i, id_i, id_j])
    cols = np.concatenate([trans_j, id_j, id_i])
    weights = np.concatenate([
        np.ones(len(trans_i)),
        np.full(len(id_i), id_weight), np.full(len(id_i), id_weight),
    ])
    return coo_matrix((weights, (rows, cols)), shape=(n, n)).tocsr()


def full_phi_dist_matrix(graph, n, chunk=2000):
    """Dense NxN graph-geodesic distance matrix, float32 (halves memory vs. dijkstra's
    native float64). The RL training loop samples arbitrary (s,g) pairs every batch and
    needs the full matrix cached, unlike the few-thousand-fixed-pairs diagnostics below.

    Built in row-chunks: a single dijkstra(indices=arange(n)) call materialises the whole
    thing at scipy's native float64 first, which is 124 GB at expert_1000's 124,479
    landmarks (vs 62 GB for the float32 we keep) and OOMs before returning anything.
    Chunking caps the float64 intermediate at (chunk, n) and casts each block down as it
    goes. The infinite-entry fill is also done per block, to avoid a full-size boolean mask."""
    D = np.empty((n, n), dtype=np.float32)
    t0 = time.time()
    finite_max = 0.0
    for i0 in range(0, n, chunk):
        i1 = min(n, i0 + chunk)
        block = dijkstra(graph, indices=np.arange(i0, i1), directed=True)
        finite = np.isfinite(block)
        if finite.any():
            finite_max = max(finite_max, float(block[finite].max()))
        D[i0:i1] = block.astype(np.float32)
        if (i0 // chunk) % 5 == 0:
            log(f"[phi-dense] rows {i1}/{n}, elapsed={time.time()-t0:.1f}s")
    np.nan_to_num(D, copy=False, posinf=finite_max * 2 if finite_max else 1.0)
    log(f"[phi-dense] {n}x{n} float32 ({D.nbytes/2**30:.1f} GiB) in {time.time()-t0:.1f}s")
    return D


def phi_for_pairs(graph, s_rows, g_rows, limit, chunk=400):
    """Graph-geodesic distance for exactly the given (s,g) pairs, via chunked *bounded*
    multi-source Dijkstra -- the scalable alternative to full_phi_dist_matrix.

    The dense NxN matrix is what caps tier size: at expert_1000's 124,479 landmarks it is
    62 GB as float32 (and scipy's own float64 intermediate is 124 GB), plus ~7h of unbounded
    all-pairs Dijkstra. But the auxiliary-regression loss only ever reads phi at the HER
    tuples' own (s, goal) pairs, which are same-episode and therefore always joined by a
    transition-edge path no longer than their step gap. So we need distances only up to
    `limit` = the largest such gap, and only at those pairs -- both of which this exploits.
    Bounded Dijkstra explores just the ball of radius `limit` around each source, and the
    (chunk, n) output is materialised chunk-by-chunk instead of all at once."""
    s_rows = np.asarray(s_rows)
    g_rows = np.asarray(g_rows)
    out = np.empty(len(s_rows), dtype=np.float32)
    uniq, inv = np.unique(s_rows, return_inverse=True)
    t0 = time.time()
    for i0 in range(0, len(uniq), chunk):
        i1 = min(len(uniq), i0 + chunk)
        D = dijkstra(graph, indices=uniq[i0:i1], directed=True, limit=limit)
        mask = (inv >= i0) & (inv < i1)
        out[mask] = D[inv[mask] - i0, g_rows[mask]]
        if (i0 // chunk) % 25 == 0:
            log(f"[phi-pairs] {i1}/{len(uniq)} unique sources, elapsed={time.time()-t0:.1f}s")
    n_inf = int(np.sum(~np.isfinite(out)))
    if n_inf:
        # Shouldn't happen for same-episode pairs (the transition path is <= limit by
        # construction), so surface it rather than silently clamping a real disconnection.
        log(f"[phi-pairs] WARNING: {n_inf}/{len(out)} pairs beyond limit={limit} -- clamping")
        out[~np.isfinite(out)] = limit
    log(f"[phi-pairs] done: {len(out)} pairs from {len(uniq)} sources in {time.time()-t0:.1f}s")
    return out


_pool_graph, _pool_limit = None, None


def _phi_pool_init(graph, limit):
    global _pool_graph, _pool_limit
    _pool_graph, _pool_limit = graph, limit


def _phi_pool_chunk_dense(sources):
    return dijkstra(_pool_graph, indices=sources, directed=True, limit=_pool_limit)


def phi_for_union_pairs(graph, s_rows, g_rows, limit, n_jobs=1, chunk=None, mem_budget_bytes=256 * 2**20):
    """Same contract and result as phi_for_pairs (bounded multi-source Dijkstra, chunked),
    with the same memory discipline: each dense (chunk, n) block is reduced to just the
    specific (s, g) entries this call actually needs and dropped immediately -- NOT cached
    as a full per-source reachable set.

    That distinction matters: an earlier version of the cross-seed sharing this function
    supports tried caching each source's full reachable-within-`limit` row so any later
    seed's queries could reuse it. On Reacher's expert_1000 graph (dense identification-edge
    connectivity -- 477k id edges over 160k nodes), a `limit`-radius search from a typical
    source reaches a large fraction of ALL nodes, not a small neighborhood -- so "cache the
    reachable set" is nearly the same size as the dense NxN matrix phi_mode=sparse exists to
    avoid, and OOMed a 20GB job budget. This function stays exact-pairs-only (bounded by
    how many pairs are actually queried, not by how many nodes are reachable) and relies on
    the CALLER unioning multiple seeds' pair sets into one call to still get cross-seed
    sharing -- see actor_train.py's her_phi_pairs helper, which unions all seeds' HER pairs
    before calling this once per tier.

    n_jobs: scipy's dijkstra call is single-threaded; different source chunks are
    independent, so this parallelizes across OS processes via multiprocessing -- an exact
    speedup (more cores doing the same work), not an approximation. chunk: if None, sized
    from mem_budget_bytes so at most ~n_jobs dense (chunk, n) float64 blocks are ever in
    flight at once."""
    s_rows = np.asarray(s_rows)
    g_rows = np.asarray(g_rows)
    n = graph.shape[0]
    if chunk is None:
        chunk = max(20, min(400, int(mem_budget_bytes / max(1, n_jobs) / (n * 8))))

    out = np.empty(len(s_rows), dtype=np.float32)
    uniq, inv = np.unique(s_rows, return_inverse=True)
    bounds = [(i0, min(len(uniq), i0 + chunk)) for i0 in range(0, len(uniq), chunk)]
    t0 = time.time()

    def extract(i0, i1, D):
        mask = (inv >= i0) & (inv < i1)
        out[mask] = D[inv[mask] - i0, g_rows[mask]]

    if n_jobs > 1 and len(bounds) > 1:
        chunk_arrays = [uniq[i0:i1] for i0, i1 in bounds]
        with Pool(processes=min(n_jobs, len(bounds)), initializer=_phi_pool_init,
                  initargs=(graph, limit)) as pool:
            for k, D in enumerate(pool.imap(_phi_pool_chunk_dense, chunk_arrays)):
                extract(*bounds[k], D)
                if k % 5 == 0:
                    log(f"[phi-union] {bounds[k][1]}/{len(uniq)} unique sources, elapsed={time.time()-t0:.1f}s")
    else:
        for k, (i0, i1) in enumerate(bounds):
            D = dijkstra(graph, indices=uniq[i0:i1], directed=True, limit=limit)
            extract(i0, i1, D)
            if k % 5 == 0:
                log(f"[phi-union] {i1}/{len(uniq)} unique sources, elapsed={time.time()-t0:.1f}s")

    n_inf = int(np.sum(~np.isfinite(out)))
    if n_inf:
        log(f"[phi-union] WARNING: {n_inf}/{len(out)} pairs beyond limit={limit} -- clamping")
        out[~np.isfinite(out)] = limit
    log(f"[phi-union] done: {len(out)} pairs from {len(uniq)} sources in {time.time()-t0:.1f}s")
    return out


def get_ablation_dist_matrix(tier, source, setup, cache_dir):
    """Dense NxN distance matrix for a B4-style distance-source ablation arm beyond the
    original three (graph/euclidean/oracle): "transonly" (identification edges dropped) and
    "k{K}" for any integer K (identification edges capped to each node's top-K neighbors).
    Cached to f"{tier}_phi_dist_{source}.npy" under cache_dir, same convention as the oracle
    matrix, so re-running a worker after an interruption doesn't rebuild it."""
    cache_path = Path(cache_dir) / f"{tier}_phi_dist_{source}.npy"
    if cache_path.exists():
        log(f"[{tier}/{source}] loading cached dist matrix")
        return np.load(cache_path)

    z, ep_idx, step_idx = setup["z"], setup["ep_idx"], setup["step_idx"]
    n = z.shape[0]
    _, eps2 = compute_calibration(z, ep_idx, step_idx)
    trans_i, trans_j, _, _ = find_graph_edges(z, ep_idx, step_idx, eps2)

    if source == "transonly":
        id_i, id_j = np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    elif source.startswith("k") and source[1:].isdigit():
        k = int(source[1:])
        log(f"[{tier}/{source}] building k={k}-capped identification edges...")
        id_i, id_j = build_id_edges_faiss_capped(z, ep_idx, step_idx, eps2, k)
    else:
        raise ValueError(f"unknown ablation distance source '{source}'")

    log(f"[{tier}/{source}] {len(trans_i)} transition edges, {len(id_i)} identification edges "
        f"-- computing full {n}x{n} dijkstra...")
    t0 = time.time()
    graph = build_weighted_graph(n, trans_i, trans_j, id_i, id_j, id_weight=1.0)
    D = full_phi_dist_matrix(graph, n)
    log(f"[{tier}/{source}] dijkstra done in {time.time() - t0:.1f}s")
    np.save(cache_path, D)
    return D


# ============================================================
# Quality diagnostics: recall/precision, same-episode-vs-cross-episode Spearman breakdown
# ============================================================

def edge_set_recall_precision(id_i_ref, id_j_ref, id_i_cand, id_j_cand, n):
    """Sparse-matrix set comparison (not Python-level set ops), so this stays cheap even at
    large edge counts. recall = how much of the reference edge set the candidate method
    reproduces; precision = how much of the candidate's edge set is in the reference."""
    ref = coo_matrix((np.ones(len(id_i_ref), dtype=bool), (id_i_ref, id_j_ref)), shape=(n, n)).tocsr()
    cand = coo_matrix((np.ones(len(id_i_cand), dtype=bool), (id_i_cand, id_j_cand)), shape=(n, n)).tocsr()
    overlap = ref.multiply(cand)
    n_overlap = int(overlap.nnz)
    n_ref = len(id_i_ref)
    n_cand = len(id_i_cand)
    recall = n_overlap / n_ref if n_ref > 0 else float("nan")
    precision = n_overlap / n_cand if n_cand > 0 else float("nan")
    return recall, precision, n_overlap


def make_eval_pairs(ep_idx, proprio, true_dist_oracle, n_pairs=2000, seed=202):
    """Fixed random (s,g) landmark-row pairs with true distances, split by same-episode vs
    cross-episode so denoising and stitching effects can be told apart."""
    rng = np.random.default_rng(seed)
    n = len(ep_idx)
    ei = rng.integers(0, n, n_pairs)
    ej = rng.integers(0, n, n_pairs)
    keep = ei != ej
    ei, ej = ei[keep], ej[keep]
    uniq_src, src_row = np.unique(ei, return_inverse=True)
    D = true_dist_oracle(proprio[uniq_src], proprio[ej])
    true_d = D[src_row, np.arange(len(ei))]
    finite = np.isfinite(true_d)
    ei, ej, true_d = ei[finite], ej[finite], true_d[finite]
    same_ep = ep_idx[ei] == ep_idx[ej]
    return ei, ej, true_d, same_ep


def phi_dist_at_pairs(graph, ei, ej):
    """Graph-geodesic distance for exactly the (ei, ej) pairs needed, via multi-source
    Dijkstra from only the unique sources queried -- not a dense NxN matrix. Cheaper than
    full_phi_dist_matrix when only a few thousand fixed eval pairs are ever looked up."""
    uniq_src, src_row = np.unique(ei, return_inverse=True)
    D = dijkstra(graph, indices=uniq_src, directed=True)  # (n_uniq_src, n)
    finite = np.isfinite(D)
    finite_max = D[finite].max() if finite.any() else 1.0
    phi_d = D[src_row, ej]
    phi_d = np.where(np.isfinite(phi_d), phi_d, finite_max * 2)
    return phi_d.astype(np.float32)


def evaluate_phi_quality(phi_d, same_ep, true_d, label):
    def _report(mask, name):
        if mask.sum() < 10:
            return None
        rho, p = sps.spearmanr(true_d[mask], phi_d[mask])
        return dict(n=int(mask.sum()), spearman=float(rho), p=float(p))

    return dict(
        label=label,
        overall=_report(np.ones_like(same_ep, dtype=bool), "overall"),
        same_episode=_report(same_ep, "same_episode"),
        cross_episode=_report(~same_ep, "cross_episode"),
    )


# ============================================================
# Predictor-based edge discovery (stage E2, docs/graph-proposal/predictor-stitching.tex)
# ============================================================

def oracle_pairwise_sample(true_dist_oracle, state, i_idx, j_idx, cap, seed=0):
    """Diagonal of the oracle distance for (i_idx[k], j_idx[k]) pairs, subsampled to `cap`
    pairs to bound the number of Dijkstra calls a grid-based oracle (e.g. Two-Room's) needs.
    Returns the kept index (into the original i_idx/j_idx) alongside the distances, so
    callers can subset any per-edge arrays (e.g. ensemble variance) the same way."""
    n = len(i_idx)
    if n > cap:
        rng = np.random.default_rng(seed)
        keep = rng.choice(n, size=cap, replace=False)
    else:
        keep = np.arange(n)
    D = true_dist_oracle(state[i_idx[keep]], state[j_idx[keep]])
    return keep, np.diagonal(D)


def env_action_verified_distance(mech, state, i_idx, j_idx, actions, true_dist_oracle, cap,
                                  seed=0, speed=None):
    """Ground truth for a SPECIFIC candidate action: step the REAL environment (real physics,
    not the predictor's forecast), then score the resulting state with the ORACLE's true
    maze-respecting distance to the target -- NOT raw Euclidean distance, and not the
    learned latent graph distance being validated. Raw Euclidean after the real step is
    NOT a safe substitute: two states can sit close in raw position while separated by a
    wall, so a real step that never crosses the wall can still land "close" in Euclidean
    terms without truly reaching the target -- exactly the confound the oracle exists to
    resolve (see main.tex's B0: raw Euclidean only correlates 0.44 with true distance,
    the graph geodesic 0.95). Unlike oracle_pairwise_sample (which only asks whether i and
    j are mutually reachable via SOME path and never validates any specific action), this
    combines real physics with the wall-aware oracle metric for the actual outcome.

    Subsampled to `cap` pairs since real env stepping (reset + step per pair) is much slower
    than the grid oracle's own batched Dijkstra. `speed`: if the environment's dynamics have
    a fixed speed that reset() would otherwise randomize per-episode (Two-Room: 5.0, see
    scripts/env_verify_cycle_edges.py), force it explicitly so the replay isn't confounded --
    leave None for environments without this quirk.

    Returns (kept_index, after_oracle_dist): kept_index into the original
    i_idx/j_idx/actions (so callers can subset other per-candidate arrays, e.g. latent
    distance, the same way), and the oracle's true distance from the resulting real state to
    state_j."""
    n = len(i_idx)
    if n > cap:
        rng = np.random.default_rng(seed)
        keep = rng.choice(n, size=cap, replace=False)
    else:
        keep = np.arange(n)

    env = mech.make_env()
    after_state = np.empty((len(keep), state.shape[1]), dtype=np.float32)
    for k, idx in enumerate(keep):
        si, sj, a = state[i_idx[idx]], state[j_idx[idx]], actions[idx]
        env.reset(options=mech.reset_options(np.asarray(si, dtype=np.float32),
                                              np.asarray(sj, dtype=np.float32)))
        if speed is not None:
            env.variation_space["agent"]["speed"].set_value(np.array([speed], dtype=np.float32))
        _, _, _, _, info = env.step(np.asarray(a, dtype=np.float32))
        after_state[k] = info["state"]

    D = true_dist_oracle(after_state, state[j_idx[keep]])
    return keep, np.diagonal(D)


def find_block_transition_edges(ep_idx, step_idx, action, skip=SKIP):
    """Real `skip`-step block transitions matching the checkpoint's ACTUAL training
    convention (frame-skip, [[lewm-predictor-action-block-convention]]): source landmark row
    `block_i`, the real concatenated `skip` actions taken in between (`action_block`, shape
    (N, skip*action_dim)), and the real landmark row `skip` steps later (`block_j`). This
    replaces the superseded single-action zero-padding convention (common.lewm_loader.
    pad_action) everywhere in this file -- a predictor call spans `skip` real env steps, not
    one. Assumes landmark rows are every real step of an episode with no gaps (same
    assumption latent_diagnostics.build_block_history_chains makes; true for load_landmarks)."""
    order = np.lexsort((step_idx, ep_idx))
    ep_o = ep_idx[order]
    boundaries = np.nonzero(np.diff(ep_o))[0] + 1
    starts_ep = np.concatenate(([0], boundaries))
    ends_ep = np.concatenate((boundaries, [len(order)]))

    block_i, action_rows, block_j = [], [], []
    for s, e in zip(starts_ep, ends_ep):
        ep_rows = order[s:e]
        L = len(ep_rows)
        for t in range(L - skip):
            block_i.append(ep_rows[t])
            action_rows.append(ep_rows[t:t + skip])
            block_j.append(ep_rows[t + skip])
    block_i = np.array(block_i, dtype=np.int64)
    action_rows = np.array(action_rows, dtype=np.int64)
    block_j = np.array(block_j, dtype=np.int64)
    action_block = action[action_rows].reshape(len(block_i), -1)
    log(f"[graph] block transition edges (skip={skip}): {len(block_i)}")
    return block_i, action_block, block_j


def action_continuation_lookup(ep_idx, step_idx, action, n_cont):
    """For every landmark row, the real next `n_cont` actions taken in the SAME episode
    (`cont_actions`, shape (N, n_cont*action_dim)) and the landmark row `n_cont` steps later
    (`cont_target_row`), where both exist -- `cont_valid` is False for rows within `n_cont`
    steps of their episode's end, which have no real continuation to borrow.

    Used to build the hybrid action block for the cross-episode splice-consistency check
    ([[lewm-predictor-action-block-convention]]): an inferred single connecting action in
    slot 1, plus a target row's own real next `n_cont` actions in the remaining slots, so a
    `1+n_cont`-action block can be fed to the predictor under its real training convention
    instead of a single action zero-padded to the wrong width."""
    n = len(ep_idx)
    cont_valid = np.zeros(n, dtype=bool)
    cont_actions = np.zeros((n, n_cont * action.shape[1]), dtype=np.float32)
    cont_target_row = np.full(n, -1, dtype=np.int64)

    order = np.lexsort((step_idx, ep_idx))
    ep_o = ep_idx[order]
    boundaries = np.nonzero(np.diff(ep_o))[0] + 1
    starts_ep = np.concatenate(([0], boundaries))
    ends_ep = np.concatenate((boundaries, [len(order)]))
    for s, e in zip(starts_ep, ends_ep):
        ep_rows = order[s:e]
        L = len(ep_rows)
        for t in range(L - n_cont):
            r = ep_rows[t]
            cont_valid[r] = True
            cont_actions[r] = action[ep_rows[t:t + n_cont]].reshape(-1)
            cont_target_row[r] = ep_rows[t + n_cont]
    log(f"[graph] {cont_valid.mean() * 100:.1f}% of landmarks have a real {n_cont}-action "
        f"continuation available")
    return cont_valid, cont_actions, cont_target_row


def predictor_error_floor(model, z, action_block, block_i, block_j, quantile=0.9,
                           sample=5000, seed=0):
    """Empirical `skip`-step block forward-prediction error floor for the frozen LeWM
    predictor, measured on real transitions -- same "calibrate from an empirical null"
    pattern as compute_calibration(), but for the predictor's own accuracy rather than latent
    distance. Used as discover_predicted_edges()'s acceptance threshold: a discovered edge is
    trusted only if its predicted-vs-actual error is no worse than the predictor typically
    achieves on data it has real ground truth for.

    block_i/action_block/block_j: from find_block_transition_edges -- the real concatenated
    action block, NOT a single action zero-padded (see [[lewm-predictor-action-block-convention]]
    for why that convention was wrong)."""
    rng = np.random.default_rng(seed)
    n = len(block_i)
    idx = rng.choice(n, size=min(sample, n), replace=False)
    i, j = block_i[idx], block_j[idx]

    zi = torch.from_numpy(z[i]).float().to(DEV).unsqueeze(1)
    act = torch.from_numpy(action_block[idx]).float().to(DEV).unsqueeze(1)
    with torch.no_grad():
        act_emb = model.action_encoder(act)
        pred = model.predict(zi, act_emb)[:, -1].cpu().numpy()
    sq_err = np.sum((pred - z[j]) ** 2, axis=1)
    floor = float(np.quantile(sq_err, quantile))
    log(f"[predictor] block error on {len(idx)} real transitions: "
        f"mean={sq_err.mean():.4f} q{quantile:.2f}={floor:.4f}")
    return floor


def discover_predicted_edges(model, z, ep_idx, block_i, action_block, k_actions=4,
                              error_floor=None, M=32, chunk=20000):
    """Predictor-based cross-episode edge DISCOVERY (as opposed to the metric-kNN
    identification edges elsewhere in this file): for each landmark, roll its current metric
    neighbors' own recorded action BLOCKS through the frozen predictor, and accept a directed
    edge to whichever real landmark (in a DIFFERENT episode) the predicted state lands near.
    This can propose edges the metric scan can never find -- two trajectories one block apart
    without ever being close right now. See docs/graph-proposal/predictor-stitching.tex,
    Algorithm 2.

    Candidate action blocks are drawn from each landmark's own current metric neighbors (via
    the same FAISS-HNSW index used for the nearest-successor lookup), not sampled arbitrarily
    -- this keeps the predictor query in-distribution, unlike a uniform/random action. Only
    neighbors with a real `skip`-action block available (block_i, from
    find_block_transition_edges) can be borrowed -- a neighbor too close to its episode's end
    has none, and is skipped rather than falling back to a zero-padded single action (the
    superseded convention, see [[lewm-predictor-action-block-convention]]).

    error_floor: from predictor_error_floor() -- required, no default, so callers must
    calibrate it themselves rather than trusting an arbitrary constant.

    Returns (disc_i, disc_j, disc_sqerr): disc_i -> disc_j directed edges and the squared
    prediction error that got each one accepted (not the realizing action -- callers needing
    it can recompute it from disc_i's kNN neighbor action blocks)."""
    import faiss
    assert error_floor is not None, "discover_predicted_edges requires a calibrated error_floor"

    z32 = np.ascontiguousarray(z.astype(np.float32))
    n, d = z32.shape
    index = faiss.IndexHNSWFlat(d, M)
    index.hnsw.efConstruction = 200
    index.add(z32)
    index.hnsw.efSearch = max(64, 2 * k_actions)

    _, nbr_idx = index.search(z32, k_actions + 1)
    nbr_idx = nbr_idx[:, 1:]  # drop self (always its own nearest neighbor, distance 0)

    has_block = np.zeros(n, dtype=bool)
    has_block[block_i] = True
    block_lookup = np.zeros((n, action_block.shape[1]), dtype=np.float32)
    block_lookup[block_i] = action_block

    src_all = np.repeat(np.arange(n), k_actions)
    nbr_all = nbr_idx.reshape(-1)
    usable = has_block[nbr_all]
    log(f"[discover] {100 * usable.mean():.1f}% of (landmark, neighbor) pairs have a real "
        f"action block to borrow (rest are too close to their episode's end)")
    src = src_all[usable]
    cand_actions = block_lookup[nbr_all[usable]]

    disc_i, disc_j, disc_sqerr = [], [], []
    for lo in range(0, len(src), chunk):
        hi = min(lo + chunk, len(src))
        zi = torch.from_numpy(z[src[lo:hi]]).float().to(DEV).unsqueeze(1)
        act = torch.from_numpy(cand_actions[lo:hi]).float().to(DEV).unsqueeze(1)
        with torch.no_grad():
            act_emb = model.action_encoder(act)
            z_hat = model.predict(zi, act_emb)[:, -1].cpu().numpy()

        D_hat, I_hat = index.search(np.ascontiguousarray(z_hat), 5)
        for row in range(hi - lo):
            i = int(src[lo + row])
            for rank in range(5):
                j = int(I_hat[row, rank])
                if j < 0 or ep_idx[j] == ep_idx[i]:
                    continue
                if D_hat[row, rank] < error_floor:
                    disc_i.append(i)
                    disc_j.append(j)
                    disc_sqerr.append(float(D_hat[row, rank]))
                break  # only the nearest different-episode neighbor is ever considered

    disc_i, disc_j, disc_sqerr = np.array(disc_i), np.array(disc_j), np.array(disc_sqerr)
    if len(disc_i):
        pairs, uniq_idx = np.unique(np.stack([disc_i, disc_j], axis=1), axis=0, return_index=True)
        disc_i, disc_j, disc_sqerr = pairs[:, 0], pairs[:, 1], disc_sqerr[uniq_idx]
    log(f"[discover] {len(src)} (landmark, candidate-action) pairs tried -> "
        f"{len(disc_i)} unique discovered edges pass error_floor={error_floor:.4f}")
    return disc_i, disc_j, disc_sqerr


def raw_topk_cross_episode_pairs(z, ep_idx, k=20, M=32):
    """Unfiltered top-k nearest cross-episode neighbor pairs, NO distance threshold --
    diagnostic-only helper for testing whether a downstream verification step (e.g.
    cycle_consistency_filter) can recover high precision from a deliberately looser,
    higher-recall candidate pool than build_identification_edges' calibrated threshold
    produces. Not a recommended graph-construction method on its own -- see
    build_id_edges_faiss_capped's docstring for why an unbounded threshold was deliberately
    removed from the main pipeline; this exists purely to give a verification step a
    population with real signal to discriminate (a mix of true and false candidates),
    unlike testing it on an already-92%-precision set."""
    import faiss
    z32 = np.ascontiguousarray(z.astype(np.float32))
    n, d = z32.shape
    index = faiss.IndexHNSWFlat(d, M)
    index.hnsw.efConstruction = 200
    index.add(z32)
    index.hnsw.efSearch = max(64, 2 * k)
    _, I = index.search(z32, k + 1)
    rows, cols = [], []
    for i in range(n):
        for rank in range(1, k + 1):
            j = int(I[i, rank])
            if j >= 0 and ep_idx[j] != ep_idx[i]:
                a, b = (i, j) if i < j else (j, i)
                rows.append(a)
                cols.append(b)
    if rows:
        pairs = np.unique(np.stack([np.array(rows), np.array(cols)], axis=1), axis=0)
        return pairs[:, 0], pairs[:, 1]
    return np.array([], dtype=np.int64), np.array([], dtype=np.int64)


def radius_cross_episode_pairs(z, ep_idx, radius, M=32):
    """Cross-episode neighbor pairs within `radius` Euclidean distance -- unlike
    raw_topk_cross_episode_pairs's fixed rank cutoff (k=20), which can return candidates far
    beyond what a single real environment step actually covers (measured: Push-T's top-20
    pool averages ~4.3 units apart, vs. a real single step's ~1.15-unit median move -- see
    [[pusht-1step-predictor-results]]), this returns every cross-episode pair close enough to
    plausibly BE one real step apart. Calibrate `radius` from the real adjacent-step distance
    distribution (e.g. its 95th percentile), not an arbitrary constant."""
    import faiss
    z32 = np.ascontiguousarray(z.astype(np.float32))
    n, d = z32.shape
    index = faiss.IndexHNSWFlat(d, M)
    index.hnsw.efConstruction = 200
    index.add(z32)
    index.hnsw.efSearch = 128
    lims, D, I = index.range_search(z32, radius ** 2)
    rows, cols = [], []
    for i in range(n):
        for k in range(lims[i], lims[i + 1]):
            j = int(I[k])
            if j != i and ep_idx[j] != ep_idx[i]:
                a, b = (i, j) if i < j else (j, i)
                rows.append(a)
                cols.append(b)
    if rows:
        pairs = np.unique(np.stack([np.array(rows), np.array(cols)], axis=1), axis=0)
        return pairs[:, 0], pairs[:, 1]
    return np.array([], dtype=np.int64), np.array([], dtype=np.int64)


def calibrate_variance_threshold(real_var, random_var, real_quantile=0.5, random_quantile=0.5):
    """Separate the "connectable" cluster (real transitions, and the wide candidate pool,
    which docs/graph-proposal/latent-diagnostics.tex's Figure 6 shows sits at the same scale)
    from the "genuinely unrelated" cluster (random pairs, an order of magnitude higher) by
    the geometric mean of the two populations' medians -- a data-driven separator sitting in
    the gap between the two clusters' bulk, rather than an arbitrary quantile of only one of
    them. Defaults to medians rather than tail quantiles deliberately: both distributions have
    enough spread that their tails cross even though the bulk is separated by roughly 10x
    (real p95 and random p05 overlap in practice) -- a threshold only needs to separate the
    bulk of each population, not achieve zero overlap at the extremes."""
    hi = np.quantile(real_var, real_quantile)
    lo = np.quantile(random_var, random_quantile)
    if not hi < lo:
        log(f"[calibrate_variance_threshold] WARNING: real-pair quantile ({hi:.5f}) is not "
            f"below the random-pair quantile ({lo:.5f}) -- proceeding with their geometric "
            f"mean anyway, but inspect the two distributions before trusting this threshold")
    return float(np.sqrt(hi * lo))

