#!/usr/bin/env bash
# Cube diagnosis/tuning sweep: three CEM/asset seeds, same100 and cross only.
# Run on the current 80 GB gnode003 allocation; two eval workers on each physical GPU 0--3.
set -u
cd /home2/mayaank.ashok/lewm_research || exit 1

export GAS_MPC_ENV=cube
export CUBE_H5_PATH=/ssd_scratch/mayaank.ashok/lewm_data/datasets/ogbench/cube_single_expert.h5
export CUBE_CKPT_DIR=/home2/mayaank.ashok/lewm_research/data/checkpoints/models--quentinll--lewm-cube
export PY=/home2/mayaank.ashok/.venv/bin/python
export MUJOCO_GL=egl OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export N=50

L=outputs/cube/logs/variants_20260919
mkdir -p "$L"

JOBS=()
for seed in 0 1 2; do
  for proto in same100 cross; do
    # Tighter final switch: keep graph/TDR guidance until roughly 12 steps from the goal.
    JOBS+=("$seed|$proto|late|+mpc.h_td=7.25 +mpc.te=0.9 +mpc.lookahead=13.63 +mpc.final_thresh=7.25 +mpc.final_metric=l2 +mpc.critic_beta=1 +mpc.critic_cost=et")
    # Critic ablation: isolates whether its TDR-derived labels are misaligned with Cube success.
    JOBS+=("$seed|$proto|nocrit|+mpc.h_td=7.25 +mpc.te=0.9 +mpc.lookahead=13.63 +mpc.final_thresh=13.63 +mpc.final_metric=l2")
    # Demonstration retrieval: warm-start CEM with a graph-compatible logged grasp/carry block.
    JOBS+=("$seed|$proto|retrieval|+mpc.h_td=7.25 +mpc.te=0.9 +mpc.lookahead=13.63 +mpc.final_thresh=13.63 +mpc.final_metric=l2 +mpc.critic_beta=1 +mpc.critic_cost=et +mpc.retrieval=true")
    # Stronger feasibility preference, previously useful in the original Cube screen.
    JOBS+=("$seed|$proto|beta2|+mpc.h_td=7.25 +mpc.te=0.9 +mpc.lookahead=13.63 +mpc.final_thresh=13.63 +mpc.final_metric=l2 +mpc.critic_beta=2 +mpc.critic_cost=et")
  done
done

run_slot() {
  local slot=$1
  local gpu=$2
  local i seed proto name overrides
  for ((i = slot; i < ${#JOBS[@]}; i += 8)); do
    IFS='|' read -r seed proto name overrides <<< "${JOBS[$i]}"
    echo "[$(date)] slot=$slot gpu=$gpu seed=$seed proto=$proto variant=$name"
    # shellcheck disable=SC2086
    CUDA_VISIBLE_DEVICES=$gpu MUJOCO_EGL_DEVICE_ID=$gpu SEED=$seed \
      bash scripts/gas_mpc_run.sh "[$((i + 1))/${#JOBS[@]}]" subgoal_tdr "$proto" $overrides
  done
}

echo "[cube-variants] start $(date), ${#JOBS[@]} runs"
for slot in 0 1 2 3 4 5 6 7; do
  gpu=$((slot / 2))
  run_slot "$slot" "$gpu" > "$L/slot${slot}.log" 2>&1 &
done
wait
echo "[cube-variants] finished $(date)"
