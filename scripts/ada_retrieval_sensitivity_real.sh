#!/usr/bin/env bash
# Real-simulator version of the retrieval-sensitivity diagnostic
# (scripts/retrieval_sensitivity_real_diag.py): runs a training-bank node's real action block
# through the LIVE simulator from a held-out query's real state, no predictor involved.
# Three envs in parallel on gnode003's 4 GPUs.
set -u
cd /home2/mayaank.ashok/lewm_research || exit 1
source /home2/mayaank.ashok/.venv/bin/activate
export MUJOCO_GL=egl

L=outputs/logs/retrieval_sensitivity_real_20260921
mkdir -p "$L"

run_env() {
  local env=$1 gpu=$2 h5var=$3 h5path=$4
  export GAS_MPC_ENV=$env
  export "${h5var}=${h5path}"
  export CUDA_VISIBLE_DEVICES=$gpu
  export MUJOCO_EGL_DEVICE_ID=$gpu
  export DIAG_N_QUERIES=50
  export DIAG_BATCH=50
  echo "[$(date)] launching $env on gpu $gpu"
  python scripts/retrieval_sensitivity_real_diag.py > "$L/${env}.log" 2>&1
  echo "[$(date)] finished $env (exit $?)"
}

run_env pusht   0 PUSHT_H5_PATH   /ssd_scratch/mayaank.ashok/lewm_data/datasets/pusht_expert_train.h5 &
run_env cube    1 CUBE_H5_PATH    /ssd_scratch/mayaank.ashok/lewm_data/datasets/cube_single_expert.h5 &
run_env reacher 2 REACHER_H5_PATH /ssd_scratch/mayaank.ashok/lewm_data/datasets/reacher.h5 &
wait
echo "[$(date)] all envs finished"
