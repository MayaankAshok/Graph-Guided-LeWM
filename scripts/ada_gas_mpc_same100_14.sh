#!/usr/bin/env bash
# Fills same100 for critic beta=1 and beta=4 (currently missing), all 5 seeds -- completes
# the critic-weight table across all 4 protocols. 10 runs, spread across all 4 GPUs.
#   nohup srun --jobid=2697130 --overlap bash scripts/ada_gas_mpc_same100_14.sh > outputs/pusht/logs/ada_same100_14_driver.log 2>&1 &
cd /home2/mayaank.ashok/lewm_research || exit 1
export PUSHT_H5_PATH=/ssd_scratch/mayaank.ashok/lewm_data/datasets/pusht_expert_train.h5
export MUJOCO_GL=egl OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PY=/home2/mayaank.ashok/.venv/bin/python
export N=50
mkdir -p outputs/pusht/logs
echo "[same100-14] armed $(date)"
LA="+mpc.lookahead=13.7"
T=""

run_line() {
  local pre=()
  while [[ $# -gt 0 && $1 == *=* && $1 != +* ]]; do pre+=("$1"); shift; done
  env "${pre[@]}" bash scripts/gas_mpc_run.sh "$@"
}
run_queue() {
  local gpu=$1; shift
  echo "[q$gpu] starting $(date) (${#@} lines)"
  for line in "$@"; do
    # shellcheck disable=SC2086
    CUDA_VISIBLE_DEVICES=$gpu run_line $line
  done
}
L=outputs/pusht/logs
run_queue 0 "SEED=0 subgoal_tdr same100 $LA $T +mpc.critic_beta=1" "SEED=0 subgoal_tdr same100 $LA $T +mpc.critic_beta=4" > $L/same100_14_q0.log 2>&1 &
run_queue 1 "SEED=1 subgoal_tdr same100 $LA $T +mpc.critic_beta=1" "SEED=1 subgoal_tdr same100 $LA $T +mpc.critic_beta=4" > $L/same100_14_q1.log 2>&1 &
run_queue 2 "SEED=2 subgoal_tdr same100 $LA $T +mpc.critic_beta=1" "SEED=2 subgoal_tdr same100 $LA $T +mpc.critic_beta=4" > $L/same100_14_q2.log 2>&1 &
run_queue 3 "SEED=3 subgoal_tdr same100 $LA $T +mpc.critic_beta=1" "SEED=3 subgoal_tdr same100 $LA $T +mpc.critic_beta=4" "SEED=4 subgoal_tdr same100 $LA $T +mpc.critic_beta=1" "SEED=4 subgoal_tdr same100 $LA $T +mpc.critic_beta=4" > $L/same100_14_q3.log 2>&1 &
wait
echo "[same100-14] all done $(date)"
