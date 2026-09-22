#!/usr/bin/env bash
# Fast verification pass for the determinism fix (env-reset seed threaded through, global
# random/np.random seeded per chunk, single-threaded BLAS): 2 reps, small N.
#   SEED=<nonzero> N=10 REPS=2 bash scripts/repro_check_l2_same100_verify.sh
cd /home2/mayaank.ashok/lewm_research || exit 1
export PUSHT_H5_PATH=${PUSHT_H5_PATH:-/ssd_scratch/mayaank.ashok/lewm_data/datasets/pusht_expert_train.h5}
export MUJOCO_GL=egl
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PY=/home2/mayaank.ashok/.venv/bin/python
export N=${N:-10}
export SEED=${SEED:?SEED must be set to a nonzero value}
REPS=${REPS:-2}
mkdir -p outputs/pusht/logs

for ((r = 0; r < REPS; r++)); do
  CUDA_VISIBLE_DEVICES=$r bash scripts/gas_mpc_run.sh "[$((r+1))/${REPS}]" l2 same100 \
    "+mpc.tag=_reprov${r}" "+mpc.graph_seed=${GRAPH_SEED:-1}" \
    > "outputs/pusht/logs/reprov_l2_same100_s${SEED}_rep${r}.log" 2>&1 &
done
wait
echo "[repro_check_verify] all ${REPS} reps finished $(date)"
