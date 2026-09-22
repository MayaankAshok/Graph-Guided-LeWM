#!/usr/bin/env bash
# Push-T OUR+retr sweep: OUR's exact phase-8 config (scripts/ada_gas_mpc_phase8.sh, config B:
# subgoal_tdr, lookahead/final_thresh=13.7, final_metric=l2, critic_beta=1, critic_cost=et)
# plus +mpc.retrieval=true, all 5 CEM seeds, all 4 protocols. Mirrors
# scripts/ada_cube_retrieval_full.sh. Protocols run in REVERSE order (cross first, same25
# last): JOBS is built protocol-major with that order, so cross gets the lowest indices and
# is picked up earliest by every slot's round-robin.
# Resumable: gas_mpc_eval.py skips any (seed, protocol) whose output already exists.
# 2 eval workers per physical GPU, all 4 GPUs (0-3) even if squeue reports fewer -- nvidia-smi
# on the node is the ground truth for what's actually free.
set -u
cd /home2/mayaank.ashok/lewm_research || exit 1

export GAS_MPC_ENV=pusht
export PUSHT_H5_PATH=/ssd_scratch/mayaank.ashok/lewm_data/datasets/pusht_expert_train.h5
export PY=/home2/mayaank.ashok/.venv/bin/python
export MUJOCO_GL=egl OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export N=50

L=outputs/pusht/logs/retrieval_full_20260919
mkdir -p "$L"

# OUR's exact config (h_td=8.0, te=0.9 are gas_mpc_eval.py's defaults, so no override needed
# here -- see scripts/ada_gas_mpc_phase8.sh config B) plus retrieval=true.
OVERRIDES="+mpc.lookahead=13.7 +mpc.final_thresh=13.7 +mpc.final_metric=l2 +mpc.critic_beta=1 +mpc.critic_cost=et +mpc.retrieval=true"

JOBS=()
for proto in cross same100 same50 same25; do
  for seed in 0 1 2 3 4; do
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
    CUDA_VISIBLE_DEVICES=$gpu SEED=$seed \
      bash scripts/gas_mpc_run.sh "[$((i + 1))/${#JOBS[@]}]" subgoal_tdr "$proto" $OVERRIDES
  done
}

echo "[pusht-retrieval-full] start $(date), ${#JOBS[@]} runs, protocol order: cross,same100,same50,same25"
for slot in 0 1 2 3 4 5 6 7; do
  gpu=$((slot / 2))
  run_slot "$slot" "$gpu" > "$L/slot${slot}.log" 2>&1 &
done
wait
echo "[pusht-retrieval-full] finished $(date)"
