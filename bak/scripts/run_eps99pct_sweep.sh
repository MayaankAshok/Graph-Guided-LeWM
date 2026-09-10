#!/usr/bin/env bash
# One-shot sweep: identification-edge threshold = the EMPIRICAL 99th percentile of real
# 1-step transition distances (eps2=14.16, measured off the actual histogram, not
# compute_calibration()'s chi2-formula approximation at the same quantile -- see
# config/graph/actor.yaml's id_eps2_override comment for why those two disagree).
#
# Trains expert_100 / expert_300 / expert_1000, {baseline, auxphi}, 3 seeds each (18 runs),
# then evaluates every trained actor at three live-rollout horizons (goal=25/50/100,
# budget=50/100/200 -- same 2x ratio used throughout), success-selected checkpoint only.
#
# Everything here writes to a dedicated cache dir and run_tag (_eps99pct) so none of it
# touches or overwrites the existing her_goal_gamma=0.99-default-threshold results.
#
# Run from ~/lewm_research: bash scripts/run_eps99pct_sweep.sh

set -uo pipefail
cd ~/lewm_research

export PUSHT_H5_PATH=/ssd_scratch/mayaank.ashok/lewm_data/datasets/pusht_expert_train.h5
export PUSHT_TIER_CACHE_DIR=/ssd_scratch/mayaank.ashok/lewm_pusht_cache/tier_cache_eps99pct
mkdir -p "$PUSHT_TIER_CACHE_DIR"
source /home2/mayaank.ashok/.venv/bin/activate
PY=/home2/mayaank.ashok/.venv/bin/python
EVAL_DIR=outputs/b3_pusht/actor/rollout_eval
mkdir -p outputs/b3_pusht/actor/sweep_logs

run_tier() {
  local TIER=$1 N_PARALLEL=$2
  local OVR="id_eps2_override=14.16 run_tag=_eps99pct"
  if [ "$TIER" = "expert_1000" ]; then
    OVR="$OVR phi_mode=sparse"
  fi

  echo "=== $TIER: training (n_parallel=$N_PARALLEL) ==="
  local GPU=0 RUNNING=0
  for VARIANT in baseline auxphi; do
    for SEED in 0 1 2; do
      LOG=outputs/b3_pusht/actor/sweep_logs/${TIER}_${VARIANT}_s${SEED}_eps99pct.log
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
        LOG=outputs/b3_pusht/actor/sweep_logs/${TIER}_${VARIANT}_s${SEED}_eps99pct_goal${OFFSET}.log
        CUDA_VISIBLE_DEVICES=$GPU nohup $PY scripts/actor_rollout_eval.py env=pusht tier=$TIER variant=$VARIANT seed=$SEED select=success n_episodes=100 $OVR env.paper_goal_offset=$OFFSET env.paper_eval_budget=$BUDGET > "$LOG" 2>&1 &
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
        SRC="$EVAL_DIR/${TIER}__${VARIANT}_eps99pct__s${SEED}__sel-success__same_episode.json"
        DST="$EVAL_DIR/${TIER}__${VARIANT}_eps99pct__s${SEED}__sel-success__same_episode_goal${OFFSET}.json"
        if [ -f "$SRC" ]; then mv "$SRC" "$DST"; else echo "WARNING: missing $SRC"; fi
      done
    done
    echo "=== $TIER: goal=$OFFSET eval done ==="
  done
}

run_tier expert_100  3
run_tier expert_300  3
run_tier expert_1000 2   # smaller parallelism -- ~10x bigger than anything run through the
                         # actor stage so far, no memory-footprint measurement to trust 3-way

echo "=== ALL DONE ==="
