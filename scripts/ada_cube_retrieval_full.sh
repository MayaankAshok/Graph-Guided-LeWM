#!/usr/bin/env bash
# Cube retrieval-variant full sweep: the single "retrieval" config from
# scripts/ada_cube_variants.sh (2026-09-19), extended to all 5 CEM seeds and all 4
# protocols (same25/same50/same100/cross), matching the OUR config used for
# docs/gas-mpc/main.tex Table 3 / tab:cube-task-critic-refresh but with +mpc.retrieval=true.
# Resumable: gas_mpc_eval.py skips any (seed, protocol) whose output json already
# exists, so re-running this script only fills in missing combos -- safe to re-launch
# after a partial run, a node change, or to add seeds/protocols later.
# Run on a 4-GPU allocation; two eval workers per physical GPU 0--3 (8 slots).
set -u
cd /home2/mayaank.ashok/lewm_research || exit 1

export GAS_MPC_ENV=cube
export CUBE_H5_PATH=/ssd_scratch/mayaank.ashok/lewm_data/datasets/ogbench/cube_single_expert.h5
export CUBE_CKPT_DIR=/home2/mayaank.ashok/lewm_research/data/checkpoints/models--quentinll--lewm-cube
export PY=/home2/mayaank.ashok/.venv/bin/python
export MUJOCO_GL=egl OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export N=50

L=outputs/cube/logs/retrieval_full_20260919
mkdir -p "$L"

# Same overrides as the "retrieval" arm of ada_cube_variants.sh: OUR's exact config
# (h_td=7.25, te=0.9, lookahead=13.63, final_thresh=13.63, final_metric=l2,
# critic_beta=1, critic_cost=et) plus +mpc.retrieval=true. graph_seed / critic path are
# filled in per-seed by gas_mpc_run.sh itself.
OVERRIDES="+mpc.h_td=7.25 +mpc.te=0.9 +mpc.lookahead=13.63 +mpc.final_thresh=13.63 +mpc.final_metric=l2 +mpc.critic_beta=1 +mpc.critic_cost=et +mpc.retrieval=true"

JOBS=()
for seed in 0 1 2 3 4; do
  for proto in same25 same50 same100 cross; do
    JOBS+=("$seed|$proto")
  done
done

run_slot() {
  local slot=$1
  local gpu=$2
  local i seed proto
  for ((i = slot; i < ${#JOBS[@]}; i += 8)); do
    IFS='|' read -r seed proto <<< "${JOBS[$i]}"
    echo "[$(date)] slot=$slot gpu=$gpu seed=$seed proto=$proto"
    # shellcheck disable=SC2086
    CUDA_VISIBLE_DEVICES=$gpu MUJOCO_EGL_DEVICE_ID=$gpu SEED=$seed \
      bash scripts/gas_mpc_run.sh "[$((i + 1))/${#JOBS[@]}]" subgoal_tdr "$proto" $OVERRIDES
  done
}

echo "[cube-retrieval-full] start $(date), ${#JOBS[@]} runs (existing outputs are skipped automatically)"
for slot in 0 1 2 3 4 5 6 7; do
  gpu=$((slot / 2))
  run_slot "$slot" "$gpu" > "$L/slot${slot}.log" 2>&1 &
done
wait
echo "[cube-retrieval-full] finished $(date)"
