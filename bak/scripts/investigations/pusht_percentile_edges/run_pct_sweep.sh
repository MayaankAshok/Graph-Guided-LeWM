#!/usr/bin/env bash
# Trains + evaluates conditions 2-6 (see run.py's module docstring) for Push-T's expert_1000
# tier at the paper's goal=25/50 protocol, 3 seeds each. Condition 1 (baseline) is already
# done -- see outputs/b3_pusht/actor/rollout_eval/expert_1000__baseline_defaulteps__s*__sel-success__same_episode_goal25.json
#
# Run from ~/lewm_research (gnode with 3 free GPUs expected): bash scripts/investigations/pusht_percentile_edges/run_pct_sweep.sh

set -uo pipefail
cd ~/lewm_research

export PUSHT_H5_PATH=/ssd_scratch/mayaank.ashok/lewm_data/datasets/pusht_expert_train.h5
source /home2/mayaank.ashok/.venv/bin/activate
PY=/home2/mayaank.ashok/.venv/bin/python
EVAL_DIR=outputs/b3_pusht/actor/rollout_eval
LOG_DIR=outputs/b3_pusht/actor/sweep_logs
mkdir -p "$LOG_DIR"

CONDITIONS="2 3 4 5 6"
SEEDS="0 1 2"
N_PARALLEL=3

echo "=== training: pct conditions {$CONDITIONS} x seeds {$SEEDS} ==="
GPU=0 RUNNING=0
for COND in $CONDITIONS; do
  for SEED in $SEEDS; do
    LOG="$LOG_DIR/expert_1000_pct${COND}_s${SEED}_train.log"
    # Distinct cache dir per condition -- her_phi_pairs_cached's cache file is keyed only by
    # tier (not by which graph produced the phi values), so conditions 2-6 MUST NOT share a
    # tier_cache_dir or a later condition would silently read an earlier condition's phi
    # values for the same (s_idx, goal_idx) keys. Confirmed by a smoke test that would have
    # done exactly this.
    CUDA_VISIBLE_DEVICES=$GPU PUSHT_TIER_CACHE_DIR="/home2/mayaank.ashok/lewm_research/outputs/b3_pusht/tier_cache_pct${COND}" \
      nohup $PY scripts/investigations/pusht_percentile_edges/run.py \
      env=pusht tier=expert_1000 seed=$SEED +pct_condition=$COND \
      env.paper_goal_offset=25 env.paper_eval_budget=50 > "$LOG" 2>&1 &
    echo "launched train pct$COND/s$SEED on gpu$GPU -> $LOG"
    GPU=$(( (GPU + 1) % 3 ))
    RUNNING=$((RUNNING + 1))
    if [ "$RUNNING" -ge "$N_PARALLEL" ]; then
      wait -n
      RUNNING=$((RUNNING - 1))
    fi
  done
done
wait
echo "=== training done ==="

echo "=== eval: goal=25 budget=50 ==="
GPU=0 RUNNING=0
for COND in $CONDITIONS; do
  for SEED in $SEEDS; do
    LOG="$LOG_DIR/expert_1000_pct${COND}_s${SEED}_eval.log"
    CUDA_VISIBLE_DEVICES=$GPU nohup $PY scripts/actor_rollout_eval.py \
      env=pusht tier=expert_1000 variant=auxphi seed=$SEED run_tag=_pct${COND} \
      select=success n_episodes=100 phi_mode=sparse \
      env.paper_goal_offset=25 env.paper_eval_budget=50 > "$LOG" 2>&1 &
    GPU=$(( (GPU + 1) % 3 ))
    RUNNING=$((RUNNING + 1))
    if [ "$RUNNING" -ge "$N_PARALLEL" ]; then
      wait -n
      RUNNING=$((RUNNING - 1))
    fi
  done
done
wait

for COND in $CONDITIONS; do
  for SEED in $SEEDS; do
    SRC="$EVAL_DIR/expert_1000__auxphi_pct${COND}__s${SEED}__sel-success__same_episode.json"
    DST="$EVAL_DIR/expert_1000__auxphi_pct${COND}__s${SEED}__sel-success__same_episode_goal25.json"
    if [ -f "$SRC" ]; then mv "$SRC" "$DST"; else echo "WARNING: missing $SRC"; fi
  done
done
echo "=== ALL DONE ==="
