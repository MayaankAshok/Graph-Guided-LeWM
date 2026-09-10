"""Complementary test to the lambda-scaling fix (tworoom_b3_mixed_fix.py): does shaping
just need enough TOTAL data to win on the mixed tier, the same way it needed enough data
in the pure size sweep (10/25/50/100 real episodes, where shaped only beat baseline at
100)? Keeps the same 50 real expert episodes as the original mixed tier (for continuity --
same underlying expert data, same seed) but scales up the noisy-rollout side until total
frame count matches expert_100's ~9101, instead of the original mixed tier's ~5645.

Runs standard baseline vs full-strength (lambda=1) shaped -- the same mechanism as the
main sweep, NOT the lambda-scaled fix -- because the question here is "does data volume
alone fix it," a different question from "does weakening shaping fix it" (already
answered in tworoom_b3_mixed_fix.py: yes, mostly, but doesn't cross baseline). Same
eval/peak methodology throughout, unchanged.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tworoom_b3_datatiers import (
    MIXED_NOISE, MIXED_REPEAT_PROB, MIXED_SEED, NOISY_EP_ID_OFFSET, SEEDS,
    _build_setup, _load_real_tier_arrays, run_condition_resumable,
)
from tworoom_b3_gciql_shaping import log
from tworoom_rollout_collector import collect_rollouts, encode_pixels

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "outputs" / "b3_tworoom"

TIER = "mixed_large"
MIXED_LARGE_N_EXPERT = 50   # same 50 real episodes as the original mixed tier (same seed -> same data)
MIXED_LARGE_N_NOISY = 220   # ~21 steps/ep observed at this noise level -> ~4600 frames -> total ~9100


def _build_mixed_large_arrays():
    ez, e_ep, e_step, e_proprio, e_action = _load_real_tier_arrays(MIXED_LARGE_N_EXPERT)

    rollout = collect_rollouts(
        n_episodes=MIXED_LARGE_N_NOISY, action_noise=MIXED_NOISE, action_repeat_prob=MIXED_REPEAT_PROB,
        seed=MIXED_SEED, cache_name=f"mixed_noisy_n{MIXED_LARGE_N_NOISY}_noise{MIXED_NOISE}",
    )
    log(f"[{TIER}] encoding {len(rollout['pixels'])} freshly-collected frames with the frozen LeWM encoder...")
    nz = encode_pixels(rollout["pixels"])
    n_ep_idx = rollout["ep_idx"] + NOISY_EP_ID_OFFSET

    z = np.concatenate([ez, nz], axis=0)
    ep_idx = np.concatenate([e_ep, n_ep_idx], axis=0)
    step_idx = np.concatenate([e_step, rollout["step_idx"]], axis=0)
    proprio = np.concatenate([e_proprio, rollout["proprio"]], axis=0)
    action = np.concatenate([e_action, rollout["action"]], axis=0)
    log(f"[{TIER}] combined: {len(ez)} real-expert frames ({MIXED_LARGE_N_EXPERT} eps) + "
        f"{len(nz)} noisy-rollout frames ({MIXED_LARGE_N_NOISY} eps) = {len(z)} total")
    return z, ep_idx, step_idx, proprio, action


def peak_stats(history):
    i = max(range(len(history)), key=lambda k: history[k]["spearman_test"])
    return history[i]


def main():
    log(f"device set by tworoom_b0_graph_gate")
    arrays = _build_mixed_large_arrays()
    setup = _build_setup(TIER, *arrays)

    results = {}
    for use_shaping, name in [(False, "baseline"), (True, "shaped")]:
        results[name] = []
        for seed in SEEDS:
            history = run_condition_resumable(TIER, name, use_shaping, setup, seed)
            results[name].append(dict(seed=seed, history=history))

    out_path = OUT_DIR / f"b3_datatiers_{TIER}_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    log(f"wrote {out_path}")

    log("\n" + "=" * 70 + f"\nSUMMARY: peak held-out-test Spearman on '{TIER}' tier "
        f"({len(setup['z'])} landmarks)\n" + "=" * 70)
    for name in ["baseline", "shaped"]:
        peaks = [peak_stats(r["history"])["spearman_test"] for r in results[name]]
        log(f"  {name}: mean peak = {np.mean(peaks):.4f} +- {np.std(peaks):.4f}  "
            f"({[f'{p:.4f}' for p in peaks]})")


if __name__ == "__main__":
    main()
