#!/usr/bin/env bash
# L2 single-pass diagnostic: LeWM's CEM planner, one solve (budget=25, no replan), on the
# fixed 200-task same-episode-offset-25 set, single seed. See scripts/l2_single_pass_diagnostic.py.
#   nohup srun --jobid=2698367 --overlap bash scripts/ada_l2_single_pass_diagnostic.sh > outputs/pusht/pairs/logs/ada_l2_single_pass.log 2>&1 &
cd /home2/mayaank.ashok/lewm_research || exit 1
export PUSHT_H5_PATH=/ssd_scratch/mayaank.ashok/lewm_data/datasets/pusht_expert_train.h5
export MUJOCO_GL=egl OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PY=/home2/mayaank.ashok/.venv/bin/python
export CUDA_VISIBLE_DEVICES=0
mkdir -p outputs/pusht/pairs/logs
echo "[l2_single_pass] armed $(date)"
"$PY" scripts/l2_single_pass_diagnostic.py eval.num_eval=200 +diag.seed=0 +diag.chunk=25
echo "[l2_single_pass] done $(date)"
