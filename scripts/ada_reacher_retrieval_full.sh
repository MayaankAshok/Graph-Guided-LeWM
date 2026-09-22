#!/usr/bin/env bash
# Reacher OUR+retr sweep: OUR's exact config (method B in scripts/ada_gas_mpc_reacher.sh:
# subgoal_tdr, lookahead/final_thresh/h_td from outputs/reacher/calib.json, te=0.9,
# critic_beta=1, critic_cost=et) plus +mpc.retrieval=true, all 5 CEM seeds, all 4 protocols.
# Protocols run in REVERSE order (cross first, same25 last): JOBS is protocol-major with that
# order, so cross gets the lowest indices and is picked up earliest by every slot.
# Resumable: gas_mpc_eval.py skips any (seed, protocol) whose output already exists.
#
# CAUTION (explicit user request overrides the usual default): scripts/ada_gas_mpc_reacher.sh
# defaults to 1 eval process per GPU because 2/GPU killed 7/60 Reacher runs on 2026-09-17 from
# EGL render contexts piling onto GPU 0. MUJOCO_EGL_DEVICE_ID=$gpu below is the fix for that
# specific cause and was never stress-tested at 2/GPU afterward. Running 2/GPU x 4 GPUs anyway,
# per instruction; watch slot logs for CUDA/EGL OOM if this regresses.
# All 4 GPUs (0-3) even if squeue reports fewer -- nvidia-smi on the node is ground truth.
set -u
cd /home2/mayaank.ashok/lewm_research || exit 1

export GAS_MPC_ENV=reacher
export REACHER_H5_PATH=/ssd_scratch/mayaank.ashok/lewm_data/datasets/reacher.h5
export PY=/home2/mayaank.ashok/.venv/bin/python
export MUJOCO_GL=egl OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export N=50

OUT=outputs/reacher
L="$OUT/logs/retrieval_full_20260919"
mkdir -p "$L"

HTD=$($PY -c "import json; c=json.load(open('$OUT/calib.json')); assert c.get('scope')=='training_only'; print(c['h_td'])")
LA=$($PY -c "import json; c=json.load(open('$OUT/calib.json')); assert c.get('scope')=='training_only'; print(c['lookahead'])")
echo "[reacher-retrieval-full] calib h_td=$HTD lookahead=$LA"

OVERRIDES="+mpc.lookahead=$LA +mpc.h_td=$HTD +mpc.te=0.9 +mpc.final_thresh=$LA +mpc.final_metric=l2 +mpc.critic_beta=1 +mpc.critic_cost=et +mpc.retrieval=true"

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
    CUDA_VISIBLE_DEVICES=$gpu MUJOCO_EGL_DEVICE_ID=$gpu SEED=$seed \
      bash scripts/gas_mpc_run.sh "[$((i + 1))/${#JOBS[@]}]" subgoal_tdr "$proto" $OVERRIDES
  done
}

echo "[reacher-retrieval-full] start $(date), ${#JOBS[@]} runs, protocol order: cross,same100,same50,same25"
for slot in 0 1 2 3 4 5 6 7; do
  gpu=$((slot / 2))
  run_slot "$slot" "$gpu" > "$L/slot${slot}.log" 2>&1 &
done
wait
echo "[reacher-retrieval-full] finished $(date)"
