"""Run with: python scripts/tests/test_heldout_tasks.py (numpy only)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from common.heldout_tasks import sample_disjoint_rows


def check():
    ep = np.repeat(np.arange(5), 30)
    step = np.tile(np.arange(30), 5)
    state = np.arange(len(ep))[:, None]
    held = np.array([1, 3, 4])
    reject = lambda s, g: (s[..., 0] % 3 == 0) | (g[..., 0] % 3 == 0)
    for pairing, count in [("same_episode", 9), ("cross_episode", 20)]:
        args = (ep, step, state, count, 123, pairing, 4, held, reject)
        s, g = sample_disjoint_rows(*args)
        s2, g2 = sample_disjoint_rows(*args)
        assert np.array_equal(s, s2) and np.array_equal(g, g2)
        assert np.isin(ep[s], held).all() and np.isin(ep[g], held).all()
        assert not reject(state[s], state[g]).any()
        occupied = {}
        for a, b in zip(s, g):
            if pairing == "same_episode":
                assert ep[a] == ep[b] and step[b] - step[a] == 4
                intervals = [(ep[a], step[a], step[b])]
            else:
                assert ep[a] != ep[b]
                intervals = [(ep[a], step[a], step[a]), (ep[b], step[b], step[b])]
            for e, lo, hi in intervals:
                assert all(hi < x or lo > y for x, y in occupied.get(e, []))
                occupied.setdefault(e, []).append((lo, hi))
    for pairing in ["same_episode", "cross_episode"]:
        try:
            sample_disjoint_rows(ep, step, state, 100, 0, pairing, 4, held,
                                 lambda s, g: np.array(True))
        except ValueError as exc:
            assert "Could only sample 0/100" in str(exc)
        else:
            raise AssertionError("Impossible task pools must fail")
    print("Held-out task sampling checks passed")


if __name__ == "__main__":
    check()
