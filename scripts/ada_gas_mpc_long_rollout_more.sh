#!/usr/bin/env bash
# L2-baseline long-chunk variants, same idea as ada_gas_mpc_same100_longrollout.sh (which did
# same100 as 2x100 instead of the default 8x25): here same50 as 2x50 (receding=10 blocks,
# budget 100) and cross as 3x100 (receding=20 blocks, budget 250 -> last chunk truncated to
# 50 steps by the budget). 5 seeds each, 10 runs total spread across 4 GPUs.
#   nohup srun --jobid=2701284 --overlap bash scripts/ada_gas_mpc_long_rollout_more.sh > outputs/pusht/logs/ada_long_rollout_more_driver.log 2>&1 &
cd /home2/mayaank.ashok/lewm_research || exit 1
export PUSHT_H5_PATH=/ssd_scratch/mayaank.ashok/lewm_data/datasets/pusht_expert_train.h5
export MUJOCO_GL=egl OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PY=/home2/mayaank.ashok/.venv/bin/python
export N=50
mkdir -p outputs/pusht/logs
echo "[long-rollout-more] armed $(date)"
OV50="plan_config.horizon=10 +mpc.receding=10 +mpc.tag=_h10"
OV100="plan_config.horizon=20 +mpc.receding=20 +mpc.tag=_h20"

run_queue() {
  local gpu=$1; shift
  echo "[q$gpu] starting $(date) (${#@} lines)"
  for line in "$@"; do
    CUDA_VISIBLE_DEVICES=$gpu bash -c "$line"
  done
}
L=outputs/pusht/logs
run_queue 0 "SEED=0 bash scripts/gas_mpc_run.sh l2 same50 $OV50" "SEED=0 bash scripts/gas_mpc_run.sh l2 cross $OV100" "SEED=1 bash scripts/gas_mpc_run.sh l2 same50 $OV50" > $L/long_rollout_more_q0.log 2>&1 &
run_queue 1 "SEED=1 bash scripts/gas_mpc_run.sh l2 cross $OV100" "SEED=2 bash scripts/gas_mpc_run.sh l2 same50 $OV50" "SEED=2 bash scripts/gas_mpc_run.sh l2 cross $OV100" > $L/long_rollout_more_q1.log 2>&1 &
run_queue 2 "SEED=3 bash scripts/gas_mpc_run.sh l2 same50 $OV50" "SEED=3 bash scripts/gas_mpc_run.sh l2 cross $OV100" "SEED=4 bash scripts/gas_mpc_run.sh l2 same50 $OV50" > $L/long_rollout_more_q2.log 2>&1 &
run_queue 3 "SEED=4 bash scripts/gas_mpc_run.sh l2 cross $OV100" > $L/long_rollout_more_q3.log 2>&1 &
wait
echo "[long-rollout-more] all done $(date)"
