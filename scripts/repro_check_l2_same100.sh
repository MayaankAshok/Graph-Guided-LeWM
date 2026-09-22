#!/usr/bin/env bash
# Reproducibility check: N independent runs of the L2 baseline, same protocol/seed,
# distinguished only by +mpc.tag so their per-chunk caches (outputs/pusht/eval/) don't
# collide -- each run genuinely re-executes CEM rather than reading a shared cache.
# Compare the resulting outputs/pusht/eval/*_repro*.json success rates/details to judge
# determinism.
#
#   SEED=<nonzero> N=50 REPS=3 bash scripts/repro_check_l2_same100.sh
cd /home2/mayaank.ashok/lewm_research || exit 1
export PUSHT_H5_PATH=${PUSHT_H5_PATH:-/ssd_scratch/mayaank.ashok/lewm_data/datasets/pusht_expert_train.h5}
export MUJOCO_GL=egl OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PY=/home2/mayaank.ashok/.venv/bin/python
export N=${N:-50}
export SEED=${SEED:?SEED must be set to a nonzero value}
REPS=${REPS:-3}
mkdir -p outputs/pusht/logs

for ((r = 0; r < REPS; r++)); do
  CUDA_VISIBLE_DEVICES=$r bash scripts/gas_mpc_run.sh "[$((r+1))/${REPS}]" l2 same100 \
    "+mpc.tag=_repro${r}" \
    > "outputs/pusht/logs/repro_l2_same100_s${SEED}_rep${r}.log" 2>&1 &
done
wait
echo "[repro_check] all ${REPS} reps finished $(date)"
