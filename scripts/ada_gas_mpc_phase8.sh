#!/usr/bin/env bash
# Phase 8: closing the short-range gap and the beta dependence (docs/gas-mpc/main.tex, phase 8).
# Configurations (every protocol, fixed task set, 50 tasks):
#   A  subgoal_tdr, final-goal switch at one lookahead (13.7 units), z-space L2 in the final phase
#   B  A + critic, cost = budget-capped expected hitting time (et), std composition, beta=1
#   E  B with beta=2
#   C  A + critic et, compose=steps (both terms in env steps, no standardisation, beta=1 nominal)
#   D  A + hard feasibility filter tau=0.5 (no weighted critic term)
# Jobs = SEEDS x CFG x PROTOS, short protocols first, round-robin over SLOTS_PER_GPU * N_GPU
# sequential queues (two processes per GPU by default -- the per-process slowdown is small).
# Every eval is resumable / skip-if-done, so re-running only does the missing work.
#   nohup srun --jobid=<JOB> --overlap bash scripts/ada_gas_mpc_phase8.sh > outputs/pusht/logs/ada_driver8.log 2>&1 &
#   SEEDS="3 4" CFG="A B C" nohup srun --jobid=<JOB> --overlap bash scripts/ada_gas_mpc_phase8.sh > outputs/pusht/logs/ada_driver8b.log 2>&1 &
cd /home2/mayaank.ashok/lewm_research || exit 1
export PUSHT_H5_PATH=/ssd_scratch/mayaank.ashok/lewm_data/datasets/pusht_expert_train.h5
export MUJOCO_GL=egl OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PY=/home2/mayaank.ashok/.venv/bin/python
export N=${N:-50}
SEEDS=(${SEEDS:-0 1 2})
CFG=(${CFG:-A B E C D})
PROTOS=(${PROTOS:-same25 same50 same100 cross})
N_GPU=${N_GPU:-3}
SLOTS_PER_GPU=${SLOTS_PER_GPU:-2}
mkdir -p outputs/pusht/logs
L=outputs/pusht/logs
echo "[phase8] armed $(date) seeds=${SEEDS[*]} configs=${CFG[*]} protos=${PROTOS[*]} slots=$((N_GPU * SLOTS_PER_GPU))"

# gap calibration tables once per seed, sequentially (each loads the 1.8 GB latent cache)
for s in "${SEEDS[@]}"; do
  [ -f outputs/pusht/gap_calib_s$s.json ] && continue
  $PY -c "import sys; sys.path.insert(0, 'scripts'); from gas_mpc_eval import gap_calibration; print(gap_calibration($s))" \
    >> $L/gap_calib.log 2>&1
done
echo "[phase8] calibration done $(date)"

A="+mpc.lookahead=13.7 +mpc.final_thresh=13.7 +mpc.final_metric=l2"
declare -A CONFIG=(
  [A]="$A"
  [B]="$A +mpc.critic_beta=1 +mpc.critic_cost=et"
  [D]="$A +mpc.critic_filter=0.5"
  [C]="$A +mpc.critic_beta=1 +mpc.critic_cost=et +mpc.compose=steps"
  [E]="$A +mpc.critic_beta=2 +mpc.critic_cost=et"
)

# job list: protocol-major so the short ones finish first; one job = "seed|proto|overrides"
JOBS=()
for proto in "${PROTOS[@]}"; do
  for s in "${SEEDS[@]}"; do
    for c in "${CFG[@]}"; do
      JOBS+=("$s|$proto|${CONFIG[$c]}")
    done
  done
done
NSLOT=$((N_GPU * SLOTS_PER_GPU))

run_slot() {
  local slot=$1
  local gpu=$((slot / SLOTS_PER_GPU))
  echo "[phase8 slot$slot gpu$gpu] starting $(date)"
  local i
  for ((i = slot; i < ${#JOBS[@]}; i += NSLOT)); do
    IFS='|' read -r s proto over <<< "${JOBS[$i]}"
    # shellcheck disable=SC2086
    CUDA_VISIBLE_DEVICES=$gpu SEED=$s bash scripts/gas_mpc_run.sh subgoal_tdr $proto $over
  done
  echo "[phase8 slot$slot gpu$gpu] finished $(date)"
}

for ((k = 0; k < NSLOT; k++)); do
  run_slot $k > "$L/ada8_slot$k.log" 2>&1 &
done
wait
echo "[phase8] all slots finished $(date)"
