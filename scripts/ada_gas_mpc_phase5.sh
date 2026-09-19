#!/usr/bin/env bash
# Phase 5 (job 2697130): the runs phases 3-4 pointed at.
#   beta in {4, 8} on all 5 seeds; retrieval + beta in {2, 4} on all 5 seeds; offsets 50/100
#   at beta=4; then 100 pairs per seed (n=100 -> the first 50 pairs of a seed are the same
#   rows as the n=50 file since both come from default_rng(seed)... they are NOT: n changes
#   the draw, so n=100 is a fresh, larger pair set) for L2 and the two finalists on seed 0.
#   nohup srun --jobid=2697130 --overlap bash scripts/ada_gas_mpc_phase5.sh > outputs/pusht/logs/ada_driver5.log 2>&1 &
cd /home2/mayaank.ashok/lewm_research || exit 1
export PUSHT_H5_PATH=/ssd_scratch/mayaank.ashok/lewm_data/datasets/pusht_expert_train.h5
export MUJOCO_GL=egl OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PY=/home2/mayaank.ashok/.venv/bin/python
mkdir -p outputs/pusht/logs
echo "[phase5] starting $(date)"
LA="+mpc.lookahead=13.7"
Q0=(
  "SEED=1 subgoal_tdr cross $LA +mpc.critic_beta=4"
  "SEED=2 subgoal_tdr cross $LA +mpc.critic_beta=4"
  "SEED=3 subgoal_tdr cross $LA +mpc.critic_beta=4"
  "SEED=4 subgoal_tdr cross $LA +mpc.critic_beta=4"
  "subgoal_tdr cross $LA +mpc.critic_beta=8"
  "SEED=1 subgoal_tdr cross $LA +mpc.critic_beta=8"
  "SEED=2 subgoal_tdr cross $LA +mpc.critic_beta=8"
  "N=100 l2 cross"
)
Q1=(
  "SEED=1 subgoal_tdr cross $LA +mpc.critic_beta=1 +mpc.retrieval=true"
  "SEED=2 subgoal_tdr cross $LA +mpc.critic_beta=1 +mpc.retrieval=true"
  "SEED=3 subgoal_tdr cross $LA +mpc.critic_beta=1 +mpc.retrieval=true"
  "SEED=4 subgoal_tdr cross $LA +mpc.critic_beta=1 +mpc.retrieval=true"
  "subgoal_tdr cross $LA +mpc.critic_beta=4 +mpc.retrieval=true"
  "SEED=1 subgoal_tdr cross $LA +mpc.critic_beta=4 +mpc.retrieval=true"
  "SEED=2 subgoal_tdr cross $LA +mpc.critic_beta=4 +mpc.retrieval=true"
  "N=100 subgoal_tdr cross $LA +mpc.critic_beta=4"
)
Q2=(
  "subgoal_tdr same50 $LA +mpc.critic_beta=4"
  "subgoal_tdr same100 $LA +mpc.critic_beta=4"
  "SEED=1 subgoal_tdr same50 $LA +mpc.critic_beta=2"
  "SEED=2 subgoal_tdr same50 $LA +mpc.critic_beta=2"
  "SEED=1 subgoal_tdr same100 $LA +mpc.critic_beta=2"
  "SEED=1 l2 same100"
  "SEED=3 subgoal_tdr cross $LA +mpc.critic_beta=8"
  "SEED=4 subgoal_tdr cross $LA +mpc.critic_beta=8"
  "N=100 subgoal_tdr cross $LA +mpc.critic_beta=1 +mpc.retrieval=true"
)
run_line() {
  local pre=()
  while [[ $# -gt 0 && $1 == *=* && $1 != +* ]]; do pre+=("$1"); shift; done
  env "${pre[@]}" bash scripts/gas_mpc_run.sh "$@"
}
run_queue() {
  local gpu=$1; shift
  for line in "$@"; do
    # shellcheck disable=SC2086
    CUDA_VISIBLE_DEVICES=$gpu run_line $line
  done
}
L=outputs/pusht/logs
run_queue 0 "${Q0[@]}" > $L/ada5_q0.log 2>&1 &
run_queue 1 "${Q1[@]}" > $L/ada5_q1.log 2>&1 &
run_queue 2 "${Q2[@]}" > $L/ada5_q2.log 2>&1 &
wait
echo "[phase5] all queues finished $(date)"
