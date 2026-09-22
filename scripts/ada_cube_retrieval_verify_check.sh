#!/usr/bin/env bash
# Regression check: does +mpc.retrieval_verify=true (predictor-verified retrieval, added for
# Reacher's failure mode) cost Cube anything? Same OUR+retr config as
# ada_cube_retrieval_full.sh plus +mpc.retrieval_verify=true, 3 seeds (0-2), same100 and cross
# only -- matches ada_cube_variants.sh's original screening scope, cheap. If Cube's numbers
# hold within noise of the unverified OUR+retr (Table 19, docs/gas-mpc/main.tex), the verify
# gate is safe to leave on as a single cross-environment default; if it costs real points, it
# should stay Reacher-only (a per-env or per-config flag, not universal).
set -u
cd /home2/mayaank.ashok/lewm_research || exit 1

export GAS_MPC_ENV=cube
export CUBE_H5_PATH=/ssd_scratch/mayaank.ashok/lewm_data/datasets/ogbench/cube_single_expert.h5
export CUBE_CKPT_DIR=/home2/mayaank.ashok/lewm_research/data/checkpoints/models--quentinll--lewm-cube
export PY=/home2/mayaank.ashok/.venv/bin/python
export MUJOCO_GL=egl OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export N=50

L=outputs/cube/logs/retrieval_verify_check_20260920
mkdir -p "$L"

OVERRIDES="+mpc.h_td=7.25 +mpc.te=0.9 +mpc.lookahead=13.63 +mpc.final_thresh=13.63 +mpc.final_metric=l2 +mpc.critic_beta=1 +mpc.critic_cost=et +mpc.retrieval=true +mpc.retrieval_verify=true"

JOBS=()
for seed in 0 1 2; do
  for proto in same100 cross; do
    JOBS+=("$seed|$proto")
  done
done

run_slot() {
  local slot=$1
  local gpu=$2
  local i seed proto
  for ((i = slot; i < ${#JOBS[@]}; i += 4)); do
    IFS='|' read -r seed proto <<< "${JOBS[$i]}"
    echo "[$(date)] slot=$slot gpu=$gpu seed=$seed proto=$proto"
    # shellcheck disable=SC2086
    CUDA_VISIBLE_DEVICES=$gpu MUJOCO_EGL_DEVICE_ID=$gpu SEED=$seed \
      bash scripts/gas_mpc_run.sh "[$((i + 1))/${#JOBS[@]}]" subgoal_tdr "$proto" $OVERRIDES
  done
}

echo "[cube-retrieval-verify-check] start $(date), ${#JOBS[@]} runs"
for slot in 0 1 2 3; do
  gpu=$slot
  run_slot "$slot" "$gpu" > "$L/slot${slot}.log" 2>&1 &
done
wait
echo "[cube-retrieval-verify-check] finished $(date)"
