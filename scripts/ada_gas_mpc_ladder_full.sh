#!/usr/bin/env bash
# Ablation ladder, Push-T, task200u, all 5 seeds x 4 protocols. Revised scope (session
# 2026-09-21): kept L2 / TDR / ctg / subgoal-bare / A / A+old-critic-beta1 / A+ET-critic
# beta-sweep {0,0.5,1,2,4} (beta=1 = "OUR", already archived); dropped old-critic beta
# {0.5,2,4} and A+ET-steps/hard-filter (not requested this round). Added: L2 + ET critic
# beta=1 (no graph); subgoal (non-TDR variant) + A's final-switch structure + ET critic
# beta=1. The H_TD ablation (config 10 at H_TD in {4,12,16}) is a SEPARATE follow-up driver
# (ada_gas_mpc_ladder_htd.sh) once ada_gas_mpc_graphs_htd.sh finishes building those graphs.
# Resumable: skips any (config, protocol, seed) already archived.
#   nohup srun --jobid=<JOB> --overlap bash scripts/ada_gas_mpc_ladder_full.sh \
#     > outputs/pusht/logs/ada_ladder_full.log 2>&1 &
cd /home2/mayaank.ashok/lewm_research || exit 1
export PUSHT_H5_PATH=${PUSHT_H5_PATH:-/ssd_scratch/mayaank.ashok/lewm_data/datasets/pusht_expert_train.h5}
export MUJOCO_GL=egl OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PY=/home2/mayaank.ashok/.venv/bin/python
export N=${N:-50}
SEEDS=(${SEEDS:-0 1 2 3 4})
PROTOS=(${PROTOS:-same25 same50 same100 cross})
N_GPU=${N_GPU:-4}
SLOTS_PER_GPU=${SLOTS_PER_GPU:-1}
mkdir -p outputs/pusht/logs
L=outputs/pusht/logs
echo "[ladder_full] armed $(date) seeds=${SEEDS[*]} protos=${PROTOS[*]} slots=$((N_GPU * SLOTS_PER_GPU))"

SG="+mpc.lookahead=13.7"
A="$SG +mpc.final_thresh=13.7 +mpc.final_metric=l2"
declare -A CONFIG=(
  [1_l2]=""
  [2_tdr]=""
  [3_ctg]=""
  [4_subgoal]="$SG"
  [5_A]="$A"
  [7_critold1]="$A +mpc.critic_beta=1"
  [10_et_b0]="$A +mpc.critic_beta=0 +mpc.critic_cost=et"
  [10_et_b0.5]="$A +mpc.critic_beta=0.5 +mpc.critic_cost=et"
  [10_et_b1]="$A +mpc.critic_beta=1 +mpc.critic_cost=et"
  [10_et_b2]="$A +mpc.critic_beta=2 +mpc.critic_cost=et"
  [10_et_b4]="$A +mpc.critic_beta=4 +mpc.critic_cost=et"
  [l2_crit]="+mpc.critic_beta=1 +mpc.critic_cost=et"
  [subgoal_crit]="$A +mpc.critic_beta=1 +mpc.critic_cost=et"
)
declare -A METHOD=(
  [1_l2]="l2" [2_tdr]="tdr" [3_ctg]="ctg"
  [4_subgoal]="subgoal_tdr" [5_A]="subgoal_tdr" [7_critold1]="subgoal_tdr"
  [10_et_b0]="subgoal_tdr" [10_et_b0.5]="subgoal_tdr" [10_et_b1]="subgoal_tdr"
  [10_et_b2]="subgoal_tdr" [10_et_b4]="subgoal_tdr"
  [l2_crit]="l2" [subgoal_crit]="subgoal"
)
STAGES=(1_l2 2_tdr 3_ctg 4_subgoal 5_A 7_critold1 10_et_b0 10_et_b0.5 10_et_b1 10_et_b2 10_et_b4 l2_crit subgoal_crit)

JOBS=()
for proto in "${PROTOS[@]}"; do
  for s in "${SEEDS[@]}"; do
    for st in "${STAGES[@]}"; do
      JOBS+=("$s|${METHOD[$st]}|$proto|${CONFIG[$st]}")
    done
  done
done
echo "[ladder_full] ${#JOBS[@]} jobs queued (already-archived combos skip near-instantly)"

NSLOT=$((N_GPU * SLOTS_PER_GPU))
run_slot() {
  local slot=$1
  local gpu=$((slot / SLOTS_PER_GPU))
  echo "[ladder_full slot$slot gpu$gpu] starting $(date)"
  local i
  for ((i = slot; i < ${#JOBS[@]}; i += NSLOT)); do
    IFS='|' read -r s method proto over <<< "${JOBS[$i]}"
    # shellcheck disable=SC2086
    CUDA_VISIBLE_DEVICES=$gpu SEED=$s bash scripts/gas_mpc_run.sh "[$((i+1))/${#JOBS[@]}]" $method $proto $over
  done
  echo "[ladder_full slot$slot gpu$gpu] finished $(date)"
}

for ((k = 0; k < NSLOT; k++)); do
  run_slot $k > "$L/ada_ladder_full_slot$k.log" 2>&1 &
done
wait
echo "[ladder_full] all slots finished $(date)"
