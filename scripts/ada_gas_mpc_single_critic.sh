#!/usr/bin/env bash
# Push-T ablation: replace OUR's five-member critic mean with one ensemble member.
# Keep every other headline setting paired with the held-out task200u evaluation.
set -euo pipefail
cd /home2/mayaank.ashok/lewm_research
export PUSHT_H5_PATH=/ssd_scratch/mayaank.ashok/lewm_data/datasets/pusht_expert_train.h5
export MUJOCO_GL=egl OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PY=/home2/mayaank.ashok/.venv/bin/python
export GAS_MPC_POOL=task200u
export N=${N:-50}
SEEDS=(${SEEDS:-0 1 2 3 4})
PROTOS=(${PROTOS:-same25 same50 same100 cross})
N_GPU=${N_GPU:-4}
SLOTS_PER_GPU=${SLOTS_PER_GPU:-2}
mkdir -p outputs/pusht/logs

JOBS=()
for proto in "${PROTOS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    # Pair each learned-asset/CEM seed with one different zero-based member. Across five
    # seeds this exercises every ensemble position once without multiplying the matrix 5x.
    JOBS+=("$seed|$proto|$seed")
  done
done

NSLOT=$((N_GPU * SLOTS_PER_GPU))
echo "[single-critic] start $(date), jobs=${#JOBS[@]} slots=$NSLOT"
run_slot() {
  local slot=$1
  local gpu=$((slot / SLOTS_PER_GPU))
  local i seed proto member
  for ((i = slot; i < ${#JOBS[@]}; i += NSLOT)); do
    IFS='|' read -r seed proto member <<< "${JOBS[$i]}"
    CUDA_VISIBLE_DEVICES=$gpu MUJOCO_EGL_DEVICE_ID=$gpu SEED=$seed \
      bash scripts/gas_mpc_run.sh "[$((i + 1))/${#JOBS[@]}]" subgoal_tdr "$proto" \
      +mpc.lookahead=13.53 +mpc.final_thresh=13.53 +mpc.final_metric=l2 \
      +mpc.critic_beta=1 +mpc.critic_cost=et +mpc.critic_member="$member" \
      +mpc.tag=_singlecritic
  done
  echo "[single-critic slot$slot gpu$gpu] finished $(date)"
}
for ((slot = 0; slot < NSLOT; slot++)); do
  run_slot "$slot" > "outputs/pusht/logs/ada_single_critic_slot$slot.log" 2>&1 &
done
wait
echo "[single-critic] all slots finished $(date)"
