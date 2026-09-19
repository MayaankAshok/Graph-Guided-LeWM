#!/usr/bin/env bash
# Phase 4: beta sweep on the critic term was monotone on seed 0 (0.5/1/2 -> 14/26/38% cross);
# run beta=2 on seeds 1-4 (matched L2 rows come from phase 3), beta=4 on seed 0, and beta=2
# at offset 50. Each queue starts when its GPU's phase-3 queue has finished.
#   nohup srun --jobid=<id> --overlap bash scripts/ada_gas_mpc_phase4.sh > outputs/pusht/logs/ada_driver4.log 2>&1 &
cd /home2/mayaank.ashok/lewm_research || exit 1
export PUSHT_H5_PATH=/ssd_scratch/mayaank.ashok/lewm_data/datasets/pusht_expert_train.h5
export MUJOCO_GL=egl OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PY=/home2/mayaank.ashok/.venv/bin/python
export N=${N:-50}
mkdir -p outputs/pusht/logs
echo "[phase4] armed $(date)"
LA="+mpc.lookahead=13.7"
C2="+mpc.critic_beta=2"
Q0=(
  "subgoal_tdr cross $LA +mpc.critic_beta=4"
  "SEED=1 subgoal_tdr cross $LA $C2"
  "SEED=3 subgoal_tdr cross $LA $C2"
)
Q1=(
  "subgoal_tdr same50 $LA $C2"
  "SEED=2 subgoal_tdr cross $LA $C2"
  "subgoal_tdr cross $LA $C2 +mpc.retrieval=true"
)
Q2=(
  "SEED=4 subgoal_tdr cross $LA $C2"
  "subgoal_tdr same100 $LA $C2"
  "l2 cross +mpc.critic_beta=2"
)
wait_for() { until grep -q "^=== .*success=" "$1" 2>/dev/null; do sleep 60; done; }
run_line() {
  local pre=()
  while [[ $# -gt 0 && $1 == *=* && $1 != +* ]]; do pre+=("$1"); shift; done
  env "${pre[@]}" bash scripts/gas_mpc_run.sh "$@"
}
run_queue() {
  local gpu=$1 sentinel=$2; shift 2
  wait_for "$sentinel"
  echo "[phase4 q$gpu] starting $(date)"
  for line in "$@"; do
    # shellcheck disable=SC2086
    CUDA_VISIBLE_DEVICES=$gpu run_line $line
  done
}
L=outputs/pusht/logs
run_queue 0 "$L/l2__cross__s3__n50.log" "${Q0[@]}" > $L/ada4_q0.log 2>&1 &
run_queue 1 "$L/l2__cross__s4__n50.log" "${Q1[@]}" > $L/ada4_q1.log 2>&1 &
run_queue 2 "$L/subgoal__mpc.lookahead=13.7__mpc.retrieval=true__cross__s0__n50.log" "${Q2[@]}" > $L/ada4_q2.log 2>&1 &
wait
echo "[phase4] all queues finished $(date)"
