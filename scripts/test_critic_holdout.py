"""Run with python scripts/test_critic_holdout.py."""
import numpy as np
from common.heldout_tasks import critic_episode_split

for seed in range(5):
    train, val, held = critic_episode_split(18685, .02, .1, seed)
    np.testing.assert_array_equal(held, np.sort(np.random.default_rng(0).permutation(18685)[:373]))
    assert not set(train) & set(held)
    assert not set(val) & set(held)
    assert not set(train) & set(val)
    assert len(train) + len(val) + len(held) == 18685
print('Five-seed critic holdout checks passed')
