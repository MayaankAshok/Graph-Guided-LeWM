#!/usr/bin/env bash
# Seed-variance training for the main regime (TDR expectile 0.99, TE 0.9) only -- the
# state-based regime (e0.999, phase 7) is NOT included. Produces tdr_full_s{1..4}.pt,
# graph_full_s{1..4}_htd8_te0.9.pkl, psi_full_s{1..4}.npy, and critic_full_s{1..4}/critic.pt,
# so the phase6 sweep can eventually pick up per-seed models instead of reusing seed 0's for
# every CEM seed (today `mpc.seed`/graph_seed=0 always loads the SAME TDR+critic; see
# scripts/gas_mpc_eval.py -- this script does not change that wiring, only builds the models).
#
# Both feeder caches (outputs/pusht/cache_full.npz, outputs/pusht/critic_training/cache_full_s0.npz --
# the same file, symlinked) already hold the FULL dataset, so no re-encoding is needed per
# seed -- only the TDR/critic training itself differs by seed.
#
# 3 GPUs busy at once (of 4 idle on this allocation), one queue per GPU: the two (slow,
# ~100min on a 2080 Ti) critics split across 2 queues, the four (fast, ~25min) TDR+graph
# builds run sequentially on the third -- this balances total wall time (~200min) instead of
# round-robin queuing 8 jobs 3-at-a-time, which would leave a long tail on the last critic.
# GPU 3 stays free for the ongoing phase6/7 eval sweep.
#
#   nohup srun --jobid=2697130 --overlap bash scripts/ada_seed_variance_training.sh \
#       > outputs/pusht/logs/ada_seedvar_driver.log 2>&1 &
cd /home2/mayaank.ashok/lewm_research || exit 1
export PUSHT_H5_PATH=/ssd_scratch/mayaank.ashok/lewm_data/datasets/pusht_expert_train.h5
export MUJOCO_GL=egl OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PY=/home2/mayaank.ashok/.venv/bin/python
mkdir -p outputs/pusht/logs outputs/pusht/critic_training/logs
echo "[seedvar] armed $(date)"

run_tdr() {  # resumable (gas_mpc_prepare.py skips a stage whose output already exists)
  local seed=$1 gpu=$2
  local log=outputs/pusht/logs/tdr_s${seed}.log
  echo "[seedvar] tdr seed=$seed gpu=$gpu -> $log $(date)"
  CUDA_VISIBLE_DEVICES=$gpu $PY scripts/gas_mpc_prepare.py all --seed "$seed" --h-td 8 --te 0.9 >> "$log" 2>&1
  echo "[seedvar] tdr seed=$seed done $(date)"
}

run_critic() {  # NOT resumable -- only called for seeds with no existing critic_full_s{seed}/
  local seed=$1 gpu=$2
  local out=outputs/pusht/critic_training/critic_full_s${seed}
  local log=outputs/pusht/critic_training/logs/critic_s${seed}.log
  echo "[seedvar] critic seed=$seed gpu=$gpu -> $log $(date)"
  CUDA_VISIBLE_DEVICES=$gpu $PY scripts/viability_train.py --cache outputs/pusht/critic_training/cache_full_s0.npz \
      --out "$out" --seed "$seed" --steps 60000 --log-every 200 --eval-every 2000 >> "$log" 2>&1
  echo "[seedvar] critic seed=$seed done $(date)"
}

( run_critic 1 0; run_critic 2 0 ) &
( run_critic 3 1; run_critic 4 1 ) &
( run_tdr 1 2; run_tdr 2 2; run_tdr 3 2; run_tdr 4 2 ) &
wait
echo "[seedvar] all done $(date)"
