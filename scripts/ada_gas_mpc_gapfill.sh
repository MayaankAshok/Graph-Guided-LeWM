#!/usr/bin/env bash
# Fills remaining gaps in the fixed-task-set (tasks200, first 50) tables that
# ada_gas_mpc_reseed.sh's subgoal_family() does not cover:
#   - subgoal_tdr same100 (plain, no critic) -- missing for every seed
#   - subgoal_tdr + critic beta in {0.5, 1, 4} on same25 and same50 -- reseed only
#     does beta=2 on those protocols; beta=0.5/1/4 previously only ran for seed 0
# Runs alongside the still-active reseed sweep, round-robined across all 4 GPUs
# (all 4 are already busy with reseed, but there is memory headroom).
#   nohup srun --jobid=2697130 --overlap bash scripts/ada_gas_mpc_gapfill.sh > outputs/pusht/logs/ada_gapfill_driver.log 2>&1 &
cd /home2/mayaank.ashok/lewm_research || exit 1
export PUSHT_H5_PATH=/ssd_scratch/mayaank.ashok/lewm_data/datasets/pusht_expert_train.h5
export MUJOCO_GL=egl OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PY=/home2/mayaank.ashok/.venv/bin/python
export N=50
mkdir -p outputs/pusht/logs
echo "[gapfill] armed $(date)"
LA="+mpc.lookahead=13.7"
T=""

gaps_for_seed() {
  local s=$1
  echo "SEED=$s subgoal_tdr same100 $LA $T"
  echo "SEED=$s subgoal_tdr same25 $LA $T +mpc.critic_beta=0.5"
  echo "SEED=$s subgoal_tdr same25 $LA $T +mpc.critic_beta=1"
  echo "SEED=$s subgoal_tdr same25 $LA $T +mpc.critic_beta=4"
  echo "SEED=$s subgoal_tdr same50 $LA $T +mpc.critic_beta=0.5"
  echo "SEED=$s subgoal_tdr same50 $LA $T +mpc.critic_beta=1"
  echo "SEED=$s subgoal_tdr same50 $LA $T +mpc.critic_beta=4"
}

# Round-robin all 5 seeds' gap lines across 4 GPUs so no single queue is much longer.
mapfile -t ALL < <(for s in 0 1 2 3 4; do gaps_for_seed "$s"; done)
Q0=(); Q1=(); Q2=(); Q3=()
i=0
for line in "${ALL[@]}"; do
  case $((i % 4)) in
    0) Q0+=("$line") ;;
    1) Q1+=("$line") ;;
    2) Q2+=("$line") ;;
    3) Q3+=("$line") ;;
  esac
  i=$((i+1))
done

run_line() {
  local pre=()
  while [[ $# -gt 0 && $1 == *=* && $1 != +* ]]; do pre+=("$1"); shift; done
  env "${pre[@]}" bash scripts/gas_mpc_run.sh "$@"
}
run_queue() {
  local gpu=$1; shift
  echo "[gapfill q$gpu] starting $(date) (${#@} lines)"
  for line in "$@"; do
    # shellcheck disable=SC2086
    CUDA_VISIBLE_DEVICES=$gpu run_line $line
  done
}
L=outputs/pusht/logs
run_queue 0 "${Q0[@]}" > $L/gapfill_q0.log 2>&1 &
run_queue 1 "${Q1[@]}" > $L/gapfill_q1.log 2>&1 &
run_queue 2 "${Q2[@]}" > $L/gapfill_q2.log 2>&1 &
run_queue 3 "${Q3[@]}" > $L/gapfill_q3.log 2>&1 &
wait
echo "[gapfill] all queues finished $(date)"
