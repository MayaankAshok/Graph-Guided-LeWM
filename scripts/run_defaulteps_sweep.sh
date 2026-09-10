#!/usr/bin/env bash
# Companion to run_eps99pct_sweep.sh: same tiers/variants/seeds/eval horizons, but at the
# DEFAULT calibrated identification threshold (common.graph_lib.compute_calibration, no
# id_eps2_override -- eps2 ~1.2-1.3 depending on tier, vs. the other sweep's fixed 14.16
# empirical-99th-percentile override). This is the direct comparison the eps99pct sweep
# needs to tell whether its data-scale trend (auxphi loses at expert_100, ties at
# expert_300, wins at expert_1000) is specific to the looser threshold or shows up at the
# default one too.
#
# Distinct run_tag (_defaulteps) so this can't collide with or resume from either the
# eps99pct sweep's checkpoints or the earlier plain-run_tag expert_100 checkpoints trained
# before config/graph/env/pusht.yaml's periodic-eval default changed from 100/200 to
# 50/100 steps -- that config change affects which checkpoint gets picked as
# success-selected, so those earlier checkpoints are not directly comparable to this run
# even though they used the same eps2.
#
# need_graph=false is passed for baseline runs (baseline never reads phi_dist -- aux_lambda=0
# -- so building the dense NxN matrix for it is pure waste; this is what stalled
# expert_300/baseline/seed-2 in the eps99pct sweep).
#
# Run from ~/lewm_research: bash scripts/run_defaulteps_sweep.sh

set -uo pipefail
cd ~/lewm_research

export PUSHT_H5_PATH=/ssd_scratch/mayaank.ashok/lewm_data/datasets/pusht_expert_train.h5
export PUSHT_TIER_CACHE_DIR=/ssd_scratch/mayaank.ashok/lewm_pusht_cache/tier_cache_defaulteps
mkdir -p "$PUSHT_TIER_CACHE_DIR"
source .venv/bin/activate
PY=/home2/mayaank.ashok/lewm_research/.venv/bin/python
EVAL_DIR=outputs/b3_pusht/actor/rollout_eval
mkdir -p outputs/b3_pusht/actor/sweep_logs

run_tier() {
  local TIER=$1 N_PARALLEL=$2
  local BASE_OVR="run_tag=_defaulteps"
  if [ "$TIER" = "expert_1000" ]; then
    BASE_OVR="$BASE_OVR phi_mode=sparse"
  fi

  echo "=== $TIER: training (n_parallel=$N_PARALLEL) ==="
  local GPU=0 RUNNING=0
  for VARIANT in baseline auxphi; do
    local OVR="$BASE_OVR"
    if [ "$VARIANT" = "baseline" ]; then
      OVR="$OVR need_graph=false"
    fi
    for SEED in 0 1 2; do
      LOG=outputs/b3_pusht/actor/sweep_logs/${TIER}_${VARIANT}_s${SEED}_defaulteps.log
      CUDA_VISIBLE_DEVICES=$GPU nohup $PY scripts/actor_train.py env=pusht tier=$TIER variant=$VARIANT seed=$SEED $OVR > "$LOG" 2>&1 &
      echo "launched train $TIER/$VARIANT/s$SEED on gpu$GPU -> $LOG"
      GPU=$((1 - GPU))
      RUNNING=$((RUNNING + 1))
      if [ "$RUNNING" -ge "$N_PARALLEL" ]; then
        wait -n
        RUNNING=$((RUNNING - 1))
      fi
    done
  done
  wait
  echo "=== $TIER: training done ==="

  for GOAL in "25 50" "50 100" "100 200"; do
    set -- $GOAL
    local OFFSET=$1 BUDGET=$2
    echo "=== $TIER: eval goal=$OFFSET budget=$BUDGET ==="
    GPU=0 RUNNING=0
    for VARIANT in baseline auxphi; do
      for SEED in 0 1 2; do
        LOG=outputs/b3_pusht/actor/sweep_logs/${TIER}_${VARIANT}_s${SEED}_defaulteps_goal${OFFSET}.log
        CUDA_VISIBLE_DEVICES=$GPU nohup $PY scripts/actor_rollout_eval.py env=pusht tier=$TIER variant=$VARIANT seed=$SEED select=success n_episodes=100 $BASE_OVR env.paper_goal_offset=$OFFSET env.paper_eval_budget=$BUDGET > "$LOG" 2>&1 &
        GPU=$((1 - GPU))
        RUNNING=$((RUNNING + 1))
        if [ "$RUNNING" -ge "$N_PARALLEL" ]; then
          wait -n
          RUNNING=$((RUNNING - 1))
        fi
      done
    done
    wait
    for VARIANT in baseline auxphi; do
      for SEED in 0 1 2; do
        SRC="$EVAL_DIR/${TIER}__${VARIANT}_defaulteps__s${SEED}__sel-success__same_episode.json"
        DST="$EVAL_DIR/${TIER}__${VARIANT}_defaulteps__s${SEED}__sel-success__same_episode_goal${OFFSET}.json"
        if [ -f "$SRC" ]; then mv "$SRC" "$DST"; else echo "WARNING: missing $SRC"; fi
      done
    done
    echo "=== $TIER: goal=$OFFSET eval done ==="
  done
}

run_tier expert_100  3
run_tier expert_300  3
run_tier expert_1000 2   # smaller parallelism -- same caution as the eps99pct sweep

echo "=== ALL DONE ==="
