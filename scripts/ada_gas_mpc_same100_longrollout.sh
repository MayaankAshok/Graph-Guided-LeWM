#!/usr/bin/env bash
# L2-baseline variant: instead of same100's default 8 replans of 25 steps each
# (receding=5 blocks -> 25 env steps/call, budget 200), this plans the FULL 100-step
# chunk at once (horizon=receding=20 blocks -> 100 env steps/call), so budget 200 gives
# exactly 2 open-loop rollouts. 5 seeds, one per GPU (GPU 3 takes the extra seed).
#   nohup srun --jobid=2701284 --overlap bash scripts/ada_gas_mpc_same100_longrollout.sh > outputs/pusht/logs/ada_same100_longrollout_driver.log 2>&1 &
cd /home2/mayaank.ashok/lewm_research || exit 1
export PUSHT_H5_PATH=/ssd_scratch/mayaank.ashok/lewm_data/datasets/pusht_expert_train.h5
export MUJOCO_GL=egl OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PY=/home2/mayaank.ashok/.venv/bin/python
export N=50
mkdir -p outputs/pusht/logs
echo "[same100-longrollout] armed $(date)"
OV="plan_config.horizon=20 +mpc.receding=20 +mpc.tag=_h20"

run_queue() {
  local gpu=$1; shift
  echo "[q$gpu] starting $(date) (${#@} seeds)"
  for seed in "$@"; do
    CUDA_VISIBLE_DEVICES=$gpu SEED=$seed bash scripts/gas_mpc_run.sh l2 same100 $OV
  done
}
L=outputs/pusht/logs
run_queue 0 0 > $L/same100_longrollout_q0.log 2>&1 &
run_queue 1 1 > $L/same100_longrollout_q1.log 2>&1 &
run_queue 2 2 > $L/same100_longrollout_q2.log 2>&1 &
run_queue 3 3 4 > $L/same100_longrollout_q3.log 2>&1 &
wait
echo "[same100-longrollout] all done $(date)"
