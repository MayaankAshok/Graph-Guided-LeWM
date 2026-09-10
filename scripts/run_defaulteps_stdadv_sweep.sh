#!/usr/bin/env bash
# Companion to run_defaulteps_sweep.sh: identical tiers/variants/seeds/eval horizons and
# identification-edge threshold (default calibrated eps2, same tier_cache reused below --
# aux_standardize/awr_normalize_adv only change the actor-training LOSS, not the graph/
# phi_dist, so the already-computed dense matrices are still valid), but with two fixes
# enabled that were OFF in that sweep:
#   aux_standardize=true    -- regress V onto the graph's ORDERING (z-scored pseudo_v)
#                               instead of its raw behavioral-distance scale
#   awr_normalize_adv=true  -- z-score the AWR advantage before exp(beta*advantage), so a
#                               constant drift in V from the aux term can't silently shift
#                               which actions get clamped at awr_weight_clip
# Motivation: actor.yaml's own comments document a measured advantage-drift on Push-T
# (mixed_large: adv mean -0.689 baseline vs -0.236 auxphi) that these two flags exist
# specifically to fix, and neither was enabled in the defaulteps sweep. This is the direct
# test of whether that drift explains auxphi losing to baseline there.
#
# Distinct run_tag (_defaulteps_stdadv) so this can't collide with or resume from the
# plain _defaulteps checkpoints -- same reasoning as that script's own run_tag comment.
#
# Run from ~/lewm_research: bash scripts/run_defaulteps_stdadv_sweep.sh

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
  local BASE_OVR="run_tag=_defaulteps_stdadv aux_standardize=true awr_normalize_adv=true"
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
      LOG=outputs/b3_pusht/actor/sweep_logs/${TIER}_${VARIANT}_s${SEED}_defaulteps_stdadv.log
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
        LOG=outputs/b3_pusht/actor/sweep_logs/${TIER}_${VARIANT}_s${SEED}_defaulteps_stdadv_goal${OFFSET}.log
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
        SRC="$EVAL_DIR/${TIER}__${VARIANT}_defaulteps_stdadv__s${SEED}__sel-success__same_episode.json"
        DST="$EVAL_DIR/${TIER}__${VARIANT}_defaulteps_stdadv__s${SEED}__sel-success__same_episode_goal${OFFSET}.json"
        if [ -f "$SRC" ]; then mv "$SRC" "$DST"; else echo "WARNING: missing $SRC"; fi
      done
    done
    echo "=== $TIER: goal=$OFFSET eval done ==="
  done
}

run_tier expert_100  3
run_tier expert_300  3
run_tier expert_1000 2   # smaller parallelism -- same caution as the reference sweeps

echo "=== ALL DONE ==="
