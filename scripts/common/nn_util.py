"""Evaluation helper shared by active diagnostics."""

import numpy as np


def precompute_eval_set(row_pool, proprio, true_dist_oracle, n_pairs, seed):
    """Sample fixed state-goal pairs and retain those with finite oracle distance."""
    rng = np.random.default_rng(seed)
    ei = rng.choice(row_pool, size=n_pairs, replace=True)
    ej = rng.choice(row_pool, size=n_pairs, replace=True)
    ei, ej = ei[ei != ej], ej[ei != ej]
    uniq_src, src_row = np.unique(ei, return_inverse=True)
    true_d = true_dist_oracle(proprio[uniq_src], proprio[ej])[src_row, np.arange(len(ei))]
    finite = np.isfinite(true_d)
    return ei[finite], ej[finite], true_d[finite]
