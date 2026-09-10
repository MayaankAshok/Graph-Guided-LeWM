"""One-off investigation: does the SOURCE of cross-trajectory identification edges matter,
independent of the standard k-capped/calibrated construction actor_train.py normally uses?

Six conditions on Push-T's expert_1000 tier, goal=25/50 protocol (the LeWM paper's own
offset/budget):
  1) baseline (no graph) -- already run, see
     outputs/b3_pusht/actor/rollout_eval/expert_1000__baseline_defaulteps__s*__sel-success__same_episode_goal25.json
  2) auxphi, transition edges only (no identification edges at all) -- phi(s,g) = real step-gap
  3) auxphi, cross-episode edges within q10 (10th pct of real one-step latent displacement),
     weight 0 ("same state, free teleport")
  4) same edge set as (3), weight 1 (one step-equivalent per hop)
  5) auxphi, cross-episode edges within q90, weight 1 -- current auxphi's own capped/
     calibrated construction is hypothesized to sit close to this
  6) auxphi, edges <q10 at weight 0 AND [q10,q90) at weight 1 (two edge groups, one graph)

q10/q90 are measured directly off this tier's own real same-episode consecutive-frame
latent Euclidean distances (NOT common.graph_lib.compute_calibration's chi2-fitted eps^2,
a different parametric estimate of roughly the same thing) -- a purely empirical,
environment-specific scale.

No shared library file is touched: this script builds its own graph inline (reusing
common.graph_lib's primitives -- find_transition_edges, radius_cross_episode_pairs) and
hands it to actor_train.run_actor_condition exactly like datatiers.py's own setup dict does,
under a distinct run_tag (_pct{condition}) so it can't collide with any other run's
checkpoint. Evaluation reuses actor_rollout_eval.py UNMODIFIED (see run_pct_sweep.sh) --
identification edges never enter the live-rollout eval path, only training.

Run one condition/seed:
    python scripts/investigations/pusht_percentile_edges/run.py env=pusht tier=expert_1000 \
        seed=0 +pct_condition=3
"""

import sys
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig, open_dict
from scipy.sparse import coo_matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from actor_train import ckpt_path as _ckpt_path
from actor_train import run_actor_condition
from common.envs import ENV_MECHANICS
from common.graph_lib import DEV, find_graph_edges, find_transition_edges
from common.log_util import log
from common.training import precompute_eval_set
from datatiers import _load_real_tier_arrays

ROOT = Path(__file__).resolve().parents[3]

CONDITIONS = {
    2: "transonly",
    3: "q10_w0",
    4: "q10_w1",
    5: "q90_w1",
    6: "q10_w0_q90_w1",
}


def adjacent_distances(z, ep_idx, step_idx):
    """Euclidean (not squared) distance between every real same-episode consecutive-frame
    pair in this tier -- the empirical population q10/q90 are measured off."""
    order = np.lexsort((step_idx, ep_idx))
    ep_o, step_o = ep_idx[order], step_idx[order]
    same_ep = ep_o[1:] == ep_o[:-1]
    consec = step_o[1:] == step_o[:-1] + 1
    mask = same_ep & consec
    z_o = z[order]
    zt, zt1 = z_o[:-1][mask], z_o[1:][mask]
    return np.linalg.norm(zt - zt1, axis=1)


def _pair_keys(i, j, n):
    return i.astype(np.int64) * np.int64(n) + j.astype(np.int64)


def cross_episode_pairs_within(z, ep_idx, step_idx, radius):
    """Exact cross-episode pairs with Euclidean distance < radius, via common.graph_lib's
    brute-force chunked GPU scan (find_graph_edges) -- NOT common.graph_lib.
    radius_cross_episode_pairs's FAISS-HNSW range_search, which crashes with a
    faiss::FaissException ('labels == nullptr && distances == nullptr' in
    RangeSearchResult::do_allocation) on this Ada build's faiss (1.15.0), reproduced
    independently of this script even on a small subsample of real landmark embeddings (a
    genuine FAISS bug, not a caller error). Brute-force is exact anyway, which matches a
    percentile-threshold spec better than an approximate kNN search would -- the O(n^2) scan
    is a GPU chunked cdist (see find_graph_edges), not the same scaling wall as an O(n^2)
    CPU-bound method."""
    _, _, id_i, id_j = find_graph_edges(z, ep_idx, step_idx, radius ** 2)
    return id_i, id_j


def build_weighted_graph_groups(n, trans_i, trans_j, edge_groups):
    """Same construction as common.graph_lib.build_weighted_graph, generalized to several
    identification-edge groups at DIFFERENT weights in one graph (needed by condition 6:
    <q10 edges at weight 0, [q10,q90) edges at weight 1, simultaneously) -- not added to the
    shared library since every other caller only ever needs one uniform weight."""
    rows = [trans_i]
    cols = [trans_j]
    weights = [np.ones(len(trans_i))]
    for gi, gj, w in edge_groups:
        rows += [gi, gj]
        cols += [gj, gi]
        weights += [np.full(len(gi), w), np.full(len(gi), w)]
    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    weights = np.concatenate(weights)
    return coo_matrix((weights, (rows, cols)), shape=(n, n)).tocsr()


def build_condition_graph(cond, n, trans_i, trans_j, z, ep_idx, step_idx):
    if cond == 2:
        id_i = id_j = np.array([], dtype=np.int64)
        graph = build_weighted_graph_groups(n, trans_i, trans_j, [])
        log(f"[pct-edges] cond2 (transonly): 0 identification edges -- phi(s,g) = real step-gap")
        return graph, id_i, id_j

    adj_dist = adjacent_distances(z, ep_idx, step_idx)
    q10 = float(np.quantile(adj_dist, 0.10))
    q90 = float(np.quantile(adj_dist, 0.90))
    log(f"[pct-edges] adjacent-frame latent distance over {len(adj_dist)} real consecutive "
        f"pairs: q10={q10:.4f} q90={q90:.4f}")

    if cond in (3, 4):
        id_i, id_j = cross_episode_pairs_within(z, ep_idx, step_idx, q10)
        w = 0.0 if cond == 3 else 1.0
        log(f"[pct-edges] cond{cond}: {len(id_i)} cross-episode edges within q10={q10:.4f}, weight={w}")
        graph = build_weighted_graph_groups(n, trans_i, trans_j, [(id_i, id_j, w)])
        return graph, id_i, id_j

    if cond == 5:
        id_i, id_j = cross_episode_pairs_within(z, ep_idx, step_idx, q90)
        log(f"[pct-edges] cond5: {len(id_i)} cross-episode edges within q90={q90:.4f}, weight=1.0")
        graph = build_weighted_graph_groups(n, trans_i, trans_j, [(id_i, id_j, 1.0)])
        return graph, id_i, id_j

    if cond == 6:
        id10_i, id10_j = cross_episode_pairs_within(z, ep_idx, step_idx, q10)
        id90_i, id90_j = cross_episode_pairs_within(z, ep_idx, step_idx, q90)
        keys10 = _pair_keys(id10_i, id10_j, n)
        keys90 = _pair_keys(id90_i, id90_j, n)
        between_mask = ~np.isin(keys90, keys10)  # q10 pairs are always a subset (smaller radius)
        bet_i, bet_j = id90_i[between_mask], id90_j[between_mask]
        log(f"[pct-edges] cond6: {len(id10_i)} edges <q10 (w=0), {len(bet_i)} edges in "
            f"[q10,q90) (w=1) -- q10={q10:.4f} q90={q90:.4f}")
        graph = build_weighted_graph_groups(n, trans_i, trans_j, [(id10_i, id10_j, 0.0), (bet_i, bet_j, 1.0)])
        all_i = np.concatenate([id10_i, bet_i])
        all_j = np.concatenate([id10_j, bet_j])
        return graph, all_i, all_j

    raise ValueError(f"unknown pct_condition {cond} (must be 2-6)")


@hydra.main(version_base=None, config_path="../../../config/graph", config_name="actor")
def main(cfg: DictConfig):
    cond = int(cfg.pct_condition)
    if cond not in CONDITIONS:
        raise ValueError(f"pct_condition must be one of {sorted(CONDITIONS)} (2-6); got {cond}")
    tag = CONDITIONS[cond]

    log(f"device={DEV} env={cfg.env.name} tier={cfg.tier} pct_condition={cond} ({tag}) seed={cfg.seed}")
    cfg.run_tag = f"_pct{cond}"
    with open_dict(cfg.env):
        cfg.env.ckpt_dir_resolved = str(ENV_MECHANICS[cfg.env.name].ckpt_dir(ROOT))

    mech = ENV_MECHANICS[cfg.env.name]
    n_episodes = int(cfg.tier.split("_")[1])
    z, ep_idx, step_idx, state, action = _load_real_tier_arrays(cfg, mech, n_episodes)
    n, d = z.shape

    uniq_eps = np.unique(ep_idx)
    rng = np.random.default_rng(0)
    shuffled = rng.permutation(uniq_eps)
    n_train = int(len(shuffled) * cfg.train_frac)
    train_eps, test_eps = shuffled[:n_train], shuffled[n_train:]
    train_rows = np.nonzero(np.isin(ep_idx, train_eps))[0]
    test_rows = np.nonzero(np.isin(ep_idx, test_eps))[0]
    log(f"[{cfg.tier}] {n} landmarks, {len(uniq_eps)} episodes "
        f"({len(train_eps)} train / {len(test_eps)} test episodes)")

    trans_i, trans_j = find_transition_edges(ep_idx, step_idx)
    graph, id_i, id_j = build_condition_graph(cond, n, trans_i, trans_j, z, ep_idx, step_idx)

    true_dist_oracle = mech.build_true_distance_oracle(state)
    train_eval = precompute_eval_set(train_rows, state, true_dist_oracle, cfg.n_eval_pairs, seed=100)
    test_eval = precompute_eval_set(test_rows, state, true_dist_oracle, cfg.n_eval_pairs, seed=101)
    log(f"[{cfg.tier}] eval pairs: {len(train_eval[0])} train, {len(test_eval[0])} test")

    setup = dict(z=z, ep_idx=ep_idx, step_idx=step_idx, proprio=state, action=action,
                 train_eps=train_eps, test_eps=test_eps, phi_dist=None, graph=graph,
                 id_i=id_i, id_j=id_j, train_eval=train_eval, test_eval=test_eval, d=d)

    ckpt = run_actor_condition(cfg, cfg.tier, "auxphi", setup, cfg.seed, cfg.n_steps, cfg.eval_every,
                                cfg.rollout_eval_every, cfg.rollout_eval_episodes)
    log(f"[{cfg.env.name}/{cfg.tier}/pct{cond}({tag})/s{cfg.seed}] done: "
        f"peak_rho={ckpt.get('peak_rho'):.4f}@{ckpt.get('peak_step')}, "
        f"peak_success_rate={ckpt.get('peak_success_rate')}@{ckpt.get('peak_success_step')}, "
        f"checkpoint={_ckpt_path(cfg, cfg.tier, 'auxphi', cfg.seed)}")


if __name__ == "__main__":
    main()
