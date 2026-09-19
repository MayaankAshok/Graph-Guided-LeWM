#!/usr/bin/env bash
# Phase 2 of the hierarchical-MPC screen (both branches of the plan):
#   Branch B (low-level is the bottleneck): replan rate, retrieval warm-start, critic hybrid.
#   Branch A (confirm + sweep): extra seeds at offset 50 / cross, H_TD 12/16, lookahead, same100.
# Waits for the phase-1 driver to finish, then runs three queues, one per GPU. Lines may start
# with VAR=value assignments (e.g. SEED=1). Launch from the login node:
#   nohup srun --jobid=<id> --overlap bash scripts/ada_gas_mpc_phase2.sh > outputs/pusht/logs/ada_driver2.log 2>&1 &
cd /home2/mayaank.ashok/lewm_research || exit 1
export PUSHT_H5_PATH=/ssd_scratch/mayaank.ashok/lewm_data/datasets/pusht_expert_train.h5
export MUJOCO_GL=egl OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PY=/home2/mayaank.ashok/.venv/bin/python
export N=${N:-50}
mkdir -p outputs/pusht/logs

# (phase-1 leftovers -- subgoal cross, dir cross, ctg same100 -- are folded into the queues below)
echo "[phase2] starting $(date)"

LA="+mpc.lookahead=13.7"
Q0=(
  "subgoal cross $LA"
  "subgoal_tdr cross $LA +mpc.receding=2"
  "subgoal_tdr cross $LA +mpc.retrieval=true"
  "l2 cross +mpc.retrieval=true"
  "subgoal_tdr cross $LA +mpc.retrieval=true +mpc.receding=2"
  "subgoal_tdr cross $LA +mpc.receding=1"
  "ctg cross +mpc.retrieval=true"
)
Q1=(
  "dir cross $LA"
  "subgoal_tdr same50 $LA +mpc.receding=2"
  "subgoal_tdr same50 $LA +mpc.retrieval=true"
  "subgoal_tdr same50 $LA +mpc.critic_beta=1"
  "subgoal_tdr cross $LA +mpc.critic_beta=1"
  "subgoal_tdr same50 $LA +mpc.h_td=12"
  "subgoal_tdr cross $LA +mpc.h_td=12"
  "subgoal_tdr same50 $LA +mpc.h_td=16"
  "l2 same50 +mpc.retrieval=true"
)
Q2=(
  "ctg same100"
  "SEED=1 subgoal_tdr same50 $LA"
  "SEED=1 l2 same50"
  "SEED=2 subgoal_tdr same50 $LA"
  "SEED=2 l2 same50"
  "SEED=1 subgoal_tdr cross $LA"
  "subgoal_tdr same50 +mpc.lookahead=8"
  "subgoal_tdr same50 +mpc.lookahead=20"
  "subgoal_tdr same100 $LA"
  "SEED=2 subgoal_tdr cross $LA"
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
run_queue 0 "${Q0[@]}" > outputs/pusht/logs/ada2_q0.log 2>&1 &
run_queue 1 "${Q1[@]}" > outputs/pusht/logs/ada2_q1.log 2>&1 &
run_queue 2 "${Q2[@]}" > outputs/pusht/logs/ada2_q2.log 2>&1 &
wait
echo "[phase2] all queues finished $(date)"
