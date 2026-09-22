#!/usr/bin/env bash
# Retrieval-sensitivity diagnostic (scripts/retrieval_sensitivity_diag.py): does "same action
# sequence from a nearby latent -> similar result" hold, per env, and is the error convergent
# or divergent in the initial latent offset? Three envs in parallel on gnode003's 4 GPUs (only
# needs pixels/actions from the h5 + the frozen encoder/predictor -- no live env, no MUJOCO_GL).
set -u
cd /home2/mayaank.ashok/lewm_research || exit 1
source /home2/mayaank.ashok/.venv/bin/activate

L=outputs/logs/retrieval_sensitivity_20260921
mkdir -p "$L"

run_env() {
  local env=$1 gpu=$2 h5var=$3 h5path=$4
  export GAS_MPC_ENV=$env
  export "${h5var}=${h5path}"
  export CUDA_VISIBLE_DEVICES=$gpu
  export DIAG_N_QUERIES=150
  echo "[$(date)] launching $env on gpu $gpu"
  python scripts/retrieval_sensitivity_diag.py > "$L/${env}.log" 2>&1
  echo "[$(date)] finished $env (exit $?)"
}

run_env pusht   0 PUSHT_H5_PATH   /ssd_scratch/mayaank.ashok/lewm_data/datasets/pusht_expert_train.h5 &
run_env cube    1 CUBE_H5_PATH    /ssd_scratch/mayaank.ashok/lewm_data/datasets/cube_single_expert.h5 &
run_env reacher 2 REACHER_H5_PATH /ssd_scratch/mayaank.ashok/lewm_data/datasets/reacher.h5 &
wait
echo "[$(date)] all envs finished"
