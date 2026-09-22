#!/usr/bin/env bash
# log P(true same25 action | CEM's sampling distribution) vs. CEM iteration, zero-mean (OUR)
# vs. retrieval-mean (OUR+retr) init, for all 3 environments. Extends the single-iteration-1
# log-P diagnostic in docs/gas-mpc/main.tex to iterations {1, 5, 10, 20} via
# +mpc.logp_steps='0,4,9,19' (0-indexed CEM steps; gas_mpc_eval.py's MeanVarRecorder callback
# snapshots the PRE-update (mean, var) -- the distribution actually used to sample that
# iteration's candidates -- into outputs/<env>/logp_diag/eval/*_logp.npz).
# +mpc.budget=25 forces exactly one CEM solve per task (same25's default budget is 50, i.e.
# up to 2 replans; this diagnostic wants one representative solve, matching this project's
# existing single-pass convention, e.g. the "Single-pass L2 CEM at offset 25" diagnostic).
# GAS_MPC_OUT keeps this out of the real outputs/<env>/eval/ used for reporting.
# n=50, seed=0 only -- this is about the shape of one representative solve's trajectory, not
# a headline success-rate comparison.
set -u
cd /home2/mayaank.ashok/lewm_research || exit 1
export PY=/home2/mayaank.ashok/.venv/bin/python
export MUJOCO_GL=egl OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2

run_env() {
  local env=$1 h5_var=$2 h5_path=$3 htd=$4 la=$5 extra_env=$6
  local out="outputs/$env/logp_diag"
  mkdir -p "$out"
  for f in cache_train.npz cache_full.npz "tdr_full_s0.pt" "psi_train_s0.npy" \
           "graph_full_s0_htd${htd}_te0.9.pkl"; do
    ln -sf "../$f" "$out/$f" 2>/dev/null
  done
  ln -sf "../critic_s0_tdr_holdout" "$out/critic_s0_tdr_holdout" 2>/dev/null
  ln -sf "../pairs" "$out/pairs" 2>/dev/null

  local OVER="+mpc.h_td=$htd +mpc.te=0.9 +mpc.lookahead=$la +mpc.final_thresh=$la +mpc.final_metric=l2 +mpc.critic_beta=1 +mpc.critic_cost=et +mpc.graph_seed=0 +mpc.critic=$out/critic_s0_tdr_holdout/critic.pt +mpc.logp_steps='0,4,9,19'"

  echo "[logp-diag] $env: OUR (zero-mean) $(date)"
  # shellcheck disable=SC2086
  env GAS_MPC_ENV=$env $h5_var=$h5_path $extra_env GAS_MPC_OUT="$out" \
    $PY scripts/gas_mpc_eval.py +mpc.method=subgoal_tdr +mpc.protocol=same25 +mpc.seed=0 \
    +mpc.budget=25 $OVER eval.num_eval=50 >> "$out/our.log" 2>&1

  echo "[logp-diag] $env: OUR+retr $(date)"
  # shellcheck disable=SC2086
  env GAS_MPC_ENV=$env $h5_var=$h5_path $extra_env GAS_MPC_OUT="$out" \
    $PY scripts/gas_mpc_eval.py +mpc.method=subgoal_tdr +mpc.protocol=same25 +mpc.seed=0 \
    +mpc.budget=25 $OVER +mpc.retrieval=true eval.num_eval=50 >> "$out/retr.log" 2>&1
}

CUDA_VISIBLE_DEVICES=0 run_env pusht PUSHT_H5_PATH /ssd_scratch/mayaank.ashok/lewm_data/datasets/pusht_expert_train.h5 8 13.7 "" &
CUDA_VISIBLE_DEVICES=1 MUJOCO_EGL_DEVICE_ID=1 run_env cube CUBE_H5_PATH /ssd_scratch/mayaank.ashok/lewm_data/datasets/ogbench/cube_single_expert.h5 7.25 13.63 "CUBE_CKPT_DIR=/home2/mayaank.ashok/lewm_research/data/checkpoints/models--quentinll--lewm-cube" &
CUDA_VISIBLE_DEVICES=2 MUJOCO_EGL_DEVICE_ID=2 run_env reacher REACHER_H5_PATH /ssd_scratch/mayaank.ashok/lewm_data/datasets/reacher.h5 3.72 5.83 "" &
wait
echo "[logp-diag] all envs finished $(date)"
