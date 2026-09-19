#!/usr/bin/env bash
# Phase 3: the critic composition won phase 2 (subgoal_tdr + critic: cross 26% vs 10%,
# offset 50 72% vs 58%). Controls, combinations, beta sweep, and extra seeds for the winner.
# Each queue waits for its GPU's phase-2 queue to finish (per-queue sentinel), then runs.
#   nohup srun --jobid=<id> --overlap bash scripts/ada_gas_mpc_phase3.sh > outputs/pusht/logs/ada_driver3.log 2>&1 &
cd /home2/mayaank.ashok/lewm_research || exit 1
export PUSHT_H5_PATH=/ssd_scratch/mayaank.ashok/lewm_data/datasets/pusht_expert_train.h5
export MUJOCO_GL=egl OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PY=/home2/mayaank.ashok/.venv/bin/python
export N=${N:-50}
mkdir -p outputs/pusht/logs
echo "[phase3] armed $(date)"

LA="+mpc.lookahead=13.7"
C1="+mpc.critic_beta=1"
Q0=(   # waits for phase-2 Q0 (last line: ctg cross retrieval)
  "l2 cross $C1"
  "subgoal_tdr cross $LA $C1 +mpc.retrieval=true"
  "subgoal cross $LA $C1"
  "subgoal_tdr cross $LA $C1 +mpc.h_td=12"
  "SEED=3 subgoal_tdr cross $LA $C1"
  "SEED=3 l2 cross"
)
Q1=(   # waits for phase-2 Q1 (last line: l2 same50 retrieval)
  "l2 same50 $C1"
  "subgoal_tdr cross $LA +mpc.critic_beta=0.5"
  "subgoal_tdr cross $LA +mpc.critic_beta=2"
  "subgoal_tdr same50 $LA $C1 +mpc.retrieval=true"
  "subgoal_tdr same50 $LA $C1 +mpc.h_td=12"
  "SEED=4 subgoal_tdr cross $LA $C1"
  "SEED=4 l2 cross"
)
Q2=(   # waits for phase-2 Q2 (last line: SEED=2 subgoal_tdr cross)
  "SEED=1 subgoal_tdr cross $LA $C1"
  "SEED=1 l2 cross"
  "SEED=2 subgoal_tdr cross $LA $C1"
  "SEED=2 l2 cross"
  "subgoal_tdr same100 $LA $C1"
  "subgoal cross $LA +mpc.retrieval=true"
)
wait_for() {  # wait until the given result log has its final '=== ... success=' line
  until grep -q "^=== .*success=" "$1" 2>/dev/null; do sleep 60; done
}
run_line() {
  local pre=()
  while [[ $# -gt 0 && $1 == *=* && $1 != +* ]]; do pre+=("$1"); shift; done
  env "${pre[@]}" bash scripts/gas_mpc_run.sh "$@"
}
run_queue() {
  local gpu=$1 sentinel=$2; shift 2
  wait_for "$sentinel"
  echo "[phase3 q$gpu] starting $(date)"
  for line in "$@"; do
    # shellcheck disable=SC2086
    CUDA_VISIBLE_DEVICES=$gpu run_line $line
  done
}
L=outputs/pusht/logs
run_queue 0 "$L/ctg__mpc.retrieval=true__cross__s0__n50.log" "${Q0[@]}" > $L/ada3_q0.log 2>&1 &
run_queue 1 "$L/l2__mpc.retrieval=true__same50__s0__n50.log" "${Q1[@]}" > $L/ada3_q1.log 2>&1 &
run_queue 2 "$L/subgoal_tdr__mpc.lookahead=13.7__cross__s2__n50.log" "${Q2[@]}" > $L/ada3_q2.log 2>&1 &
wait
echo "[phase3] all queues finished $(date)"
