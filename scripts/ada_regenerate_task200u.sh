#!/usr/bin/env bash
set -euo pipefail
cd /home2/mayaank.ashok/lewm_research
export PUSHT_H5_PATH=/ssd_scratch/mayaank.ashok/lewm_data/datasets/pusht_expert_train.h5
export REACHER_H5_PATH=/ssd_scratch/mayaank.ashok/lewm_data/datasets/reacher.h5
export CUBE_H5_PATH=/ssd_scratch/mayaank.ashok/lewm_data/datasets/ogbench/cube_single_expert.h5
export GAS_MPC_POOL=task200u OMP_NUM_THREADS=4
failed=0
for env in pusht reacher cube; do
  echo "Regenerating $env"
  GAS_MPC_ENV=$env /home2/mayaank.ashok/.venv/bin/python scripts/gas_mpc_make_tasks.py --tasks 200 || failed=1
done
exit "$failed"
