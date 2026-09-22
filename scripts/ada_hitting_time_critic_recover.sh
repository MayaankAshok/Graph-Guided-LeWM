#!/usr/bin/env bash
# Complete phase 1 after the Weibull candidate failed numerically; preserve it as a failed model.
set -euo pipefail
cd /home2/mayaank.ashok/lewm_research
PY=/home2/mayaank.ashok/.venv/bin/python
OUT=outputs/pusht/hitting_time
CACHE=/ssd_scratch/mayaank.ashok/planning_trainonly/pusht/cache_train.npz
CUDA_VISIBLE_DEVICES=2 MUJOCO_EGL_DEVICE_ID=2 "$PY" scripts/viability_long_horizon_audit.py \
  --cache "$CACHE" --labels "$OUT/labels_s0.npz" --eval-n 20000 --pred-rollouts 4000 \
  --critics \
    legacy=outputs/pusht/critic_s0_tdr_holdout/critic.pt \
    categorical="$OUT/softmax_s0/critic.pt" \
    hazard="$OUT/hazard_s0/critic.pt" \
  --out "$OUT/long_horizon_audit_s0.json" \
  > "$OUT/logs/long_horizon_audit_s0.log" 2>&1
printf '%s\n' 'weibull: failed (NaN training loss; no selected checkpoint)' > "$OUT/weibull_s0/FAILED"
touch "$OUT/phase1.done"
