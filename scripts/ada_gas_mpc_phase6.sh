#!/usr/bin/env bash
# Phase 6: the fixed 200-task sets (first 50 tasks), every seed on the SAME tasks -- seeds are
# CEM repeats. Finalists + controls: cross x {l2, subgoal_tdr, crit1, crit2, crit4, ret+crit1},
# offset 50 x {l2, subgoal_tdr, crit2}, offset 100 x {l2, crit2}, seeds 0-4.
# Queues start when their GPU's phase-5 queue is done (per-queue sentinel).
#   nohup srun --jobid=2697130 --overlap bash scripts/ada_gas_mpc_phase6.sh > outputs/pusht/logs/ada_driver6.log 2>&1 &
cd /home2/mayaank.ashok/lewm_research || exit 1
export PUSHT_H5_PATH=/ssd_scratch/mayaank.ashok/lewm_data/datasets/pusht_expert_train.h5
export MUJOCO_GL=egl OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PY=/home2/mayaank.ashok/.venv/bin/python
export N=50
mkdir -p outputs/pusht/logs
echo "[phase6] armed $(date)"
LA="+mpc.lookahead=13.7"
T=""
cross_set() {  # all cross-episode finalists for one seed
  local s=$1
  echo "SEED=$s l2 cross $T"
  echo "SEED=$s subgoal_tdr cross $LA $T"
  echo "SEED=$s subgoal_tdr cross $LA $T +mpc.critic_beta=2"
  echo "SEED=$s subgoal_tdr cross $LA $T +mpc.critic_beta=1"
  echo "SEED=$s subgoal_tdr cross $LA $T +mpc.critic_beta=4"
  echo "SEED=$s subgoal_tdr cross $LA $T +mpc.critic_beta=1 +mpc.retrieval=true"
}
mapfile -t Q0 < <(cross_set 0; cross_set 3; for s in 0 1 2 3 4; do echo "SEED=$s l2 same100 $T"; done)
mapfile -t Q1 < <(cross_set 1; cross_set 4; for s in 0 1 2 3 4; do echo "SEED=$s subgoal_tdr same100 $LA $T +mpc.critic_beta=2"; done)
mapfile -t Q2 < <(cross_set 2; for s in 0 1 2 3 4; do echo "SEED=$s l2 same50 $T"; echo "SEED=$s subgoal_tdr same50 $LA $T"; echo "SEED=$s subgoal_tdr same50 $LA $T +mpc.critic_beta=2"; done)
wait_for() { until grep -q "^=== .*success=" "$1" 2>/dev/null; do sleep 60; done; }
run_line() {
  local pre=()
  while [[ $# -gt 0 && $1 == *=* && $1 != +* ]]; do pre+=("$1"); shift; done
  env "${pre[@]}" bash scripts/gas_mpc_run.sh "$@"
}
run_queue() {
  local gpu=$1 sentinel=$2; shift 2
  [ -n "$sentinel" ] && wait_for "$sentinel"
  echo "[phase6 q$gpu] starting $(date) (${#@} lines)"
  for line in "$@"; do
    # shellcheck disable=SC2086
    CUDA_VISIBLE_DEVICES=$gpu run_line $line
  done
}
L=outputs/pusht/logs
run_queue 0 "$L/l2__cross__s0__n100.log" "${Q0[@]}" > $L/ada6_q0.log 2>&1 &
run_queue 1 "$L/subgoal_tdr__mpc.lookahead=13.7__mpc.critic_beta=4__cross__s0__n100.log" "${Q1[@]}" > $L/ada6_q1.log 2>&1 &
run_queue 2 "" "${Q2[@]}" > $L/ada6_q2.log 2>&1 &
wait
echo "[phase6] all queues finished $(date)"
