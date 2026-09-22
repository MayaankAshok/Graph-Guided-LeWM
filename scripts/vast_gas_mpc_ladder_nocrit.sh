#!/usr/bin/env bash
# Ablation ladder (docs/gas-mpc/main.tex / iclr2027 tab:ladder), non-critic rows only --
# Ada (which holds the per-seed critic checkpoints) is unreachable, so critic-based rows
# (old -logV beta sweep, ET critic, ET-steps, hard filter) are deferred to a follow-up run.
# Configurations, every protocol, task200u pool, 50 tasks, 5 seeds (L2 just 1 seed to verify
# against the existing archived task200u results):
#   L2v  l2, seed=2 only -- sanity check against already-complete task200u results
#   TDR  tdr terminal cost (no graph)
#   CTG  graph cost-to-go (no subgoal)
#   SG   graph subgoal, lookahead 13.7 (no final switch)
#   A    SG + final-goal switch at 13.7, z-space L2 in the final phase
# Single-GPU, 4 parallel slots (measured: 4x4-thread procs beat 8x4-thread and 4x16-thread
# on this RTX 3090 -- see benchmark in session notes).
#   nohup bash scripts/vast_gas_mpc_ladder_nocrit.sh > outputs/pusht/logs/vast_ladder_nocrit.log 2>&1 &
cd "$(dirname "$0")/.." || exit 1
export PUSHT_H5_PATH=${PUSHT_H5_PATH:-/workspace/lewm_data/pusht_expert_train.h5}
export MUJOCO_GL=egl OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PY=.venv/bin/python
export N=${N:-50}
SEEDS=(${SEEDS:-0 1 2 3 4})
PROTOS=(${PROTOS:-same25 same50 same100 cross})
NSLOT=${NSLOT:-4}
mkdir -p outputs/pusht/logs
L=outputs/pusht/logs
echo "[ladder_nocrit] armed $(date) seeds=${SEEDS[*]} protos=${PROTOS[*]} slots=$NSLOT"

SG="+mpc.lookahead=13.7"
A="$SG +mpc.final_thresh=13.7 +mpc.final_metric=l2"

JOBS=()
# L2 verification: one seed only, all protocols
for proto in "${PROTOS[@]}"; do
  JOBS+=("2|l2|$proto|")
done
# Full 5-seed ladder for the non-critic graph methods
for proto in "${PROTOS[@]}"; do
  for s in "${SEEDS[@]}"; do
    JOBS+=("$s|tdr|$proto|")
    JOBS+=("$s|ctg|$proto|")
    JOBS+=("$s|subgoal_tdr|$proto|$SG")
    JOBS+=("$s|subgoal_tdr|$proto|$A")
  done
done
echo "[ladder_nocrit] ${#JOBS[@]} jobs queued"

run_slot() {
  local slot=$1
  echo "[ladder_nocrit slot$slot] starting $(date)"
  local i
  for ((i = slot; i < ${#JOBS[@]}; i += NSLOT)); do
    IFS='|' read -r s method proto over <<< "${JOBS[$i]}"
    # shellcheck disable=SC2086
    CUDA_VISIBLE_DEVICES=0 SEED=$s bash scripts/gas_mpc_run.sh $method $proto $over
  done
  echo "[ladder_nocrit slot$slot] finished $(date)"
}

for ((k = 0; k < NSLOT; k++)); do
  run_slot $k > "$L/vast_ladder_nocrit_slot$k.log" 2>&1 &
done
wait
echo "[ladder_nocrit] all slots finished $(date)"
