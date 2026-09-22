#!/usr/bin/env bash
# H_TD ablation: config 10 (A + ET critic, beta=1) at H_TD in {4,12,16} (H_TD=8 already
# covered by ada_gas_mpc_ladder_full.sh's 10_et_b1 stage). Requires graphs built by
# ada_gas_mpc_graphs_htd.sh first (graph_full_s{seed}_htd{H}_te0.9.pkl for H in 4,12,16).
#   nohup srun --jobid=<JOB> --overlap bash scripts/ada_gas_mpc_ladder_htd.sh \
#     > outputs/pusht/logs/ada_ladder_htd.log 2>&1 &
cd /home2/mayaank.ashok/lewm_research || exit 1
export PUSHT_H5_PATH=${PUSHT_H5_PATH:-/ssd_scratch/mayaank.ashok/lewm_data/datasets/pusht_expert_train.h5}
export MUJOCO_GL=egl OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PY=/home2/mayaank.ashok/.venv/bin/python
export N=${N:-50}
SEEDS=(${SEEDS:-0 1 2 3 4})
PROTOS=(${PROTOS:-same25 same50 same100 cross})
HTDS=(${HTDS:-4 12 16})
N_GPU=${N_GPU:-4}
mkdir -p outputs/pusht/logs
L=outputs/pusht/logs

A_BASE="+mpc.lookahead=13.7 +mpc.final_thresh=13.7 +mpc.final_metric=l2 +mpc.critic_beta=1 +mpc.critic_cost=et"

JOBS=()
for proto in "${PROTOS[@]}"; do
  for s in "${SEEDS[@]}"; do
    for h in "${HTDS[@]}"; do
      JOBS+=("$s|$proto|$A_BASE +mpc.h_td=$h")
    done
  done
done
echo "[ladder_htd] ${#JOBS[@]} jobs queued: htd=${HTDS[*]} seeds=${SEEDS[*]} protos=${PROTOS[*]}"

run_slot() {
  local slot=$1
  echo "[ladder_htd slot$slot] starting $(date)"
  local i
  for ((i = slot; i < ${#JOBS[@]}; i += N_GPU)); do
    IFS='|' read -r s proto over <<< "${JOBS[$i]}"
    # shellcheck disable=SC2086
    CUDA_VISIBLE_DEVICES=$slot SEED=$s bash scripts/gas_mpc_run.sh "[$((i+1))/${#JOBS[@]}]" subgoal_tdr $proto $over
  done
  echo "[ladder_htd slot$slot] finished $(date)"
}

for ((k = 0; k < N_GPU; k++)); do
  run_slot $k > "$L/ada_ladder_htd_slot$k.log" 2>&1 &
done
wait
echo "[ladder_htd] all slots finished $(date)"
