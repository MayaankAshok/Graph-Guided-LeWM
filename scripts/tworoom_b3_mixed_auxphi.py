"""Candidate improvement, aimed at actually BEATING baseline (not just recovering it).

Diagnosis so far (tworoom_b3_mixed_diagnosis.py, tworoom_b3_mixed_b1check.py,
tworoom_b3_mixed_peak_probe.py): Phi (graph distance) is an excellent, uniformly-accurate
proxy for true distance (~0.96-0.97 Spearman everywhere, including noisy-touching pairs).
The problem is HOW that information reaches the network: potential-based reward shaping
folds Phi into the per-step reward, which then gets bootstrapped through TD -- and TD
compounds whatever noise is in that per-step reward. We measured real per-tuple noise: the
shaped-reward increment has much higher variance on real-expert-episode HER tuples
(std 0.68) than noisy-episode ones (std 0.47), because real trajectories are far less
monotonic toward HER-sampled goals (91 avg steps vs a ~25-step direct path). The peak-probe
showed this manifests as a literal non-monotonic hump in shaped's learned V(s,g) at
mid-range distances, even though shaped's peak V has MORE overall structure than baseline's
(near-flat) peak V.

Fix tested here: don't touch the reward at all -- train on baseline's plain -1/0 TD
objective (the "flat but honest" signal), and ADD a direct auxiliary regression loss
pulling V(s,g) toward a Phi-derived pseudo-value:

    pseudo_v(s,g) = -(1 - gamma^Phi(s,g)) / (1 - gamma)

i.e. "the value Phi(s,g) steps would have under pure geometric discounting, if Phi were
the literal step count." This uses the exact same accurate distance information as
shaping, but as a single, non-bootstrapped regression target per tuple -- it can't get
compounded across a trajectory of noisy per-step increments the way the reward-shaping
term does. If the mechanism diagnosis is right, this should let V absorb Phi's genuine
ranking accuracy without inheriting the reward-shaping noise, and should beat both
baseline and the lambda-scaled shaped variant.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy import stats as sps

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.training import peak_stats, run_condition_resumable  # noqa: F401
from tworoom_b0_graph_gate import DEV
from tworoom_b3_datatiers import CKPT_DIR, SEEDS, build_tier_setups
from tworoom_b3_gciql_shaping import log

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "outputs" / "b3_tworoom"

AUX_LAMBDAS = [0.3]
TIER = "mixed"


def main():
    log(f"device={DEV}")
    setup = build_tier_setups([TIER])[TIER]

    results = {}
    for lam in AUX_LAMBDAS:
        name = f"auxphi_lam{lam}"
        results[name] = []
        for seed in SEEDS:
            # ckpt_dir passed explicitly -- run_condition_resumable's old implicit fallback
            # to a hardcoded ckpt_path() import was removed when this moved to
            # common/training.py (see that module's docstring).
            history = run_condition_resumable(TIER, name, lam, setup, seed, ckpt_dir=CKPT_DIR)
            results[name].append(dict(seed=seed, history=history))

    existing = json.loads((OUT_DIR / f"b3_datatiers_{TIER}_results.json").read_text())
    fix_existing = {}
    fix_path = OUT_DIR / "b3_mixed_fix_results.json"
    if fix_path.exists():
        fix_existing = json.loads(fix_path.read_text())

    out_path = OUT_DIR / "b3_mixed_auxphi_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    log(f"wrote {out_path}")

    log("\n" + "=" * 70 + f"\nSUMMARY: peak held-out-test Spearman on '{TIER}' tier\n" + "=" * 70)
    for name in ["baseline", "shaped"]:
        peaks = [peak_stats(r["history"])["spearman_test"] for r in existing[name]]
        log(f"  {name} (main sweep): mean peak = {np.mean(peaks):.4f} +- {np.std(peaks):.4f}  "
            f"({[f'{p:.4f}' for p in peaks]})")
    for name, runs in fix_existing.items():
        peaks = [peak_stats(r["history"])["spearman_test"] for r in runs]
        log(f"  {name} (lambda-fix): mean peak = {np.mean(peaks):.4f} +- {np.std(peaks):.4f}  "
            f"({[f'{p:.4f}' for p in peaks]})")
    for lam in AUX_LAMBDAS:
        name = f"auxphi_lam{lam}"
        peaks = [peak_stats(r["history"])["spearman_test"] for r in results[name]]
        log(f"  {name}: mean peak = {np.mean(peaks):.4f} +- {np.std(peaks):.4f}  "
            f"({[f'{p:.4f}' for p in peaks]})")


if __name__ == "__main__":
    main()
