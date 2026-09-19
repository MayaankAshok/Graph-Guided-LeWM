#!/usr/bin/env bash
# Phase 7: the paper's STATE-BASED regime on the LeWM latent -- TDR expectile 0.999, TE 0.99
# (12.1% of frames kept, 2,534 nodes, 1 component), lookahead re-calibrated to that TDR's
# 25-step distance (9.65 units). Winner (subgoal_tdr + critic beta=2) on 5 cross seeds and
# offsets 50/100, plus the no-critic subgoal for the graph's own effect.
#   nohup srun --jobid=2697130 --overlap bash scripts/ada_gas_mpc_phase7.sh > outputs/pusht/logs/ada_driver7.log 2>&1 &
cd /home2/mayaank.ashok/lewm_research || exit 1
export PUSHT_H5_PATH=/ssd_scratch/mayaank.ashok/lewm_data/datasets/pusht_expert_train.h5
export MUJOCO_GL=egl OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PY=/home2/mayaank.ashok/.venv/bin/python
export N=${N:-50}
mkdir -p outputs/pusht/logs
echo "[phase7] armed $(date)"
T="GAS_MPC_TDR_TAG=_e0.999"
S="+mpc.lookahead=9.65 +mpc.te=0.99"
Q0=(
  "$T subgoal_tdr cross $S +mpc.critic_beta=2"
  "$T SEED=1 subgoal_tdr cross $S +mpc.critic_beta=2"
  "$T subgoal_tdr same50 $S"
)
Q1=(
  "$T SEED=2 subgoal_tdr cross $S +mpc.critic_beta=2"
  "$T subgoal_tdr same50 $S +mpc.critic_beta=2"
  "$T subgoal_tdr cross $S"
)
Q2=(
  "$T SEED=3 subgoal_tdr cross $S +mpc.critic_beta=2"
  "$T SEED=4 subgoal_tdr cross $S +mpc.critic_beta=2"
  "$T subgoal_tdr same100 $S +mpc.critic_beta=2"
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
  echo "[phase7 q$gpu] starting $(date)"
  for line in "$@"; do
    # shellcheck disable=SC2086
    CUDA_VISIBLE_DEVICES=$gpu run_line $line
  done
}
L=outputs/pusht/logs
run_queue 0 "$L/subgoal_tdr__mpc.lookahead=13.7__mpc.critic_beta=2__mpc.te=0.99__same50__s0__n50.log" "${Q0[@]}" > $L/ada7_q0.log 2>&1 &
run_queue 1 "$L/subgoal_tdr__mpc.lookahead=13.7__mpc.critic_beta=2__mpc.te=0.999__same50__s0__n50.log" "${Q1[@]}" > $L/ada7_q1.log 2>&1 &
run_queue 2 "$L/subgoal_tdr__mpc.lookahead=13.7__mpc.te=0.999__cross__s0__n50.log" "${Q2[@]}" > $L/ada7_q2.log 2>&1 &
wait
echo "[phase7] all queues finished $(date)"
