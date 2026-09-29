"""Held-out task sampling; occupied intervals are inclusive and local to one pool."""
import numpy as np


def fixed_episode_split(episode_ids, heldout_frac):
    """The final evaluation split is independent of every learned-asset seed."""
    ids = np.asarray(episode_ids)
    if not 0 < heldout_frac < 1 or len(ids) < 2:
        raise ValueError("Need at least two episodes and a holdout fraction between zero and one")
    perm = np.random.default_rng(0).permutation(len(ids))
    n = max(1, int(len(ids) * heldout_frac))
    return np.sort(ids[perm[n:]]), np.sort(ids[perm[:n]])


def validate_training_cache(cache):
    """Fail before accessing any frame arrays in a cache without a verified split."""
    if not bool(cache.get("training_only", False)):
        raise ValueError("A physically training-only cache is required; run gas_mpc_prepare.py encode")
    train, held = fixed_episode_split(np.arange(int(cache["n_source_episodes"])),
                                     float(cache["heldout_frac"]))
    if not np.array_equal(cache["episode_id"], train) or not np.array_equal(cache["heldout_episode_ids"], held):
        raise ValueError("Training cache does not match the fixed final holdout")
    return held


def validate_test_cache(cache):
    """Require embeddings from exactly the fixed final holdout."""
    if not bool(cache.get("heldout_only", False)):
        raise ValueError("A physically held-out test cache is required; run gas_mpc_prepare.py encode")
    train, held = fixed_episode_split(np.arange(int(cache["n_source_episodes"])),
                                     float(cache["heldout_frac"]))
    if not np.array_equal(cache["episode_id"], held) or not np.array_equal(cache["training_episode_ids"], train):
        raise ValueError("Test cache does not match the fixed final holdout")
    return held


def source_rows(cache, rows):
    """Translate compact training-cache rows back to evaluation-dataset rows."""
    rows = np.asarray(rows)
    ep = np.searchsorted(cache["ep_offset"], rows, side="right") - 1
    return cache["source_ep_offset"][ep] + rows - cache["ep_offset"][ep]


def cache_rows(cache, source_rows):
    """Translate dataset rows into one split cache, rejecting rows outside its episodes."""
    rows = np.asarray(source_rows, dtype=np.int64)
    source, length = cache["source_ep_offset"], cache["ep_len"]
    ep = np.searchsorted(source, rows, side="right") - 1
    if np.any(ep < 0) or np.any(rows >= source[ep] + length[ep]):
        raise ValueError("Dataset row is outside the selected embedding split")
    return cache["ep_offset"][ep] + rows - source[ep]


def sample_disjoint_rows(ep_col, step_col, state_col, n, seed, pairing, offset,
                         heldout_eps, reject):
    ep_col, step_col = np.asarray(ep_col), np.asarray(step_col)
    if n < 1 or offset < 1:
        raise ValueError("Task count and trajectory offset must be positive")
    rng = np.random.default_rng(seed)
    rows = np.flatnonzero(np.isin(ep_col, heldout_eps))
    if not len(rows):
        raise ValueError("No dataset rows belong to the held-out episodes")
    occupied, starts, goals = {}, [], []

    def free(ep, lo, hi):
        return all(hi < a or lo > b for a, b in occupied.get(ep, []))

    if pairing == "same_episode":
        goals_all = rows + offset
        valid = goals_all < len(ep_col)
        rows, goals_all = rows[valid], goals_all[valid]
        valid = ((ep_col[rows] == ep_col[goals_all]) &
                 (step_col[goals_all] == step_col[rows] + offset))
        rows = rows[valid]
        candidates = ((int(s), int(s + offset)) for s in rng.permutation(rows))
    elif pairing == "cross_episode":
        if len(np.unique(ep_col[rows])) < 2:
            raise ValueError("Cross-episode tasks require at least two held-out episodes")
        # Bounded rejection sampling: fail explicitly instead of relaxing constraints.
        candidates = ((int(rng.choice(rows)), int(rng.choice(rows)))
                      for _ in range(max(10000, 1000 * n)))
    else:
        raise ValueError(pairing)
    for s, g in candidates:
        se, ge = ep_col[s].item(), ep_col[g].item()
        if pairing == "cross_episode" and se == ge:
            continue
        intervals = ([(se, int(step_col[s]), int(step_col[g]))]
                     if se == ge else [(se, int(step_col[s]), int(step_col[s])),
                                       (ge, int(step_col[g]), int(step_col[g]))])
        if not all(free(*interval) for interval in intervals):
            continue
        if reject is not None and np.asarray(reject(
                np.asarray(state_col[s], dtype=np.float64),
                np.asarray(state_col[g], dtype=np.float64))).any():
            continue
        for ep, lo, hi in intervals:
            occupied.setdefault(ep, []).append((lo, hi))
        starts.append(s)
        goals.append(g)
        if len(starts) == n:
            order = np.argsort(starts) if pairing == "same_episode" else np.arange(n)
            return np.asarray(starts)[order], np.asarray(goals)[order]
    raise ValueError(f"Could only sample {len(starts)}/{n} nontrivial, non-overlapping "
                     "held-out tasks; reduce the task count")
