"""True (not predictor-imagined) same-episode start->goal latent distance at a given horizon,
sampled the same way predictor_error_by_horizon.py samples its rollout starts -- for horizons
the task200u pool has no protocol at (e.g. H=5; the pool's shortest offset is 25).

Reuses predictor_error_by_horizon.load_cache/sample_starts so the (episode, t) draw is directly
comparable to that script's own H=5/25/50/100 rows, then reports ||z[t] - z[t+H]|| instead of
rolling the predictor forward.

    python scripts/direct_horizon_distance.py --horizons 5,25,50,100 --n-samples 3000
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.log_util import log
from predictor_error_by_horizon import LOOKBACK, load_cache, sample_starts


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--horizons", default="5,25,50,100")
    ap.add_argument("--n-samples", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    horizons = [int(h) for h in args.horizons.split(",")]
    rng = np.random.default_rng(args.seed)
    cache = load_cache()
    z = cache["z"]

    for h in horizons:
        starts = sample_starts(cache, h, args.n_samples, rng)
        dist = np.linalg.norm(z[starts] - z[starts + h], axis=-1)
        log(f"[h={h:>3}] n={len(dist)} mean={dist.mean():.3f} median={np.median(dist):.3f} std={dist.std():.3f}")


if __name__ == "__main__":
    main()
