#!/usr/bin/env bash
# Ada driver for the hierarchical-MPC screen: three sequential queues, one per GPU, each
# line "method protocol [overrides]" handled by scripts/gas_mpc_run.sh (resumable).
# Launch FROM THE LOGIN NODE so it survives the ssh session:
#   nohup srun --jobid=<id> --overlap bash scripts/ada_gas_mpc.sh > outputs/pusht/logs/ada_driver.log 2>&1 &
cd /home2/mayaank.ashok/lewm_research || exit 1
export PUSHT_H5_PATH=/ssd_scratch/mayaank.ashok/lewm_data/datasets/pusht_expert_train.h5
export MUJOCO_GL=egl OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PY=/home2/mayaank.ashok/.venv/bin/python
export N=${N:-50} SEED=${SEED:-0}
mkdir -p outputs/pusht/logs

Q0=(
  "tdr cross"
  "ctg same25"
  "ctg same50"
  "ctg cross"
  "subgoal same25 +mpc.lookahead=13.7"
  "subgoal same50 +mpc.lookahead=13.7"
  "subgoal cross +mpc.lookahead=13.7"
)
Q1=(
  "path same25"
  "path same50"
  "path cross"
  "dir same25 +mpc.lookahead=13.7"
  "dir same50 +mpc.lookahead=13.7"
  "dir cross +mpc.lookahead=13.7"
)
Q2=(
  "subgoal_tdr same25 +mpc.lookahead=13.7"
  "subgoal_tdr same50 +mpc.lookahead=13.7"
  "subgoal_tdr cross +mpc.lookahead=13.7"
  "l2 same100"
  "tdr same100"
  "ctg same100"
)
run_queue() {
  local gpu=$1; shift
  for line in "$@"; do
    # shellcheck disable=SC2086
    CUDA_VISIBLE_DEVICES=$gpu bash scripts/gas_mpc_run.sh $line
  done
}
run_queue 0 "${Q0[@]}" > outputs/pusht/logs/ada_q0.log 2>&1 &
run_queue 1 "${Q1[@]}" > outputs/pusht/logs/ada_q1.log 2>&1 &
run_queue 2 "${Q2[@]}" > outputs/pusht/logs/ada_q2.log 2>&1 &
wait
echo "[ada_gas_mpc] all queues finished $(date)"
