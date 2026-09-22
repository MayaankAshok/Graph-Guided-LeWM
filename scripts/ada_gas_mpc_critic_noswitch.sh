#!/usr/bin/env bash
# Missing ladder cell: graph subgoal + ET critic beta=1, WITHOUT the final L2 switch
# (isolates the critic's contribution independent of the switch -- completes the
# switch x critic 2x2 factorial: config4=neither, config5(A)=switch only,
# config10(OUR)=both, this=critic only). Push-T, task200u, 5 seeds x 4 protocols.
#   nohup srun --jobid=<JOB> --overlap bash scripts/ada_gas_mpc_critic_noswitch.sh \
#     > outputs/pusht/logs/ada_critic_noswitch.log 2>&1 &
cd /home2/mayaank.ashok/lewm_research || exit 1
export PUSHT_H5_PATH=${PUSHT_H5_PATH:-/ssd_scratch/mayaank.ashok/lewm_data/datasets/pusht_expert_train.h5}
export MUJOCO_GL=egl OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PY=/home2/mayaank.ashok/.venv/bin/python
export N=${N:-50}
SEEDS=(${SEEDS:-0 1 2 3 4})
PROTOS=(${PROTOS:-same25 same50 same100 cross})
N_GPU=${N_GPU:-4}
mkdir -p outputs/pusht/logs
L=outputs/pusht/logs

OVER="+mpc.lookahead=13.7 +mpc.critic_beta=1 +mpc.critic_cost=et"

JOBS=()
for proto in "${PROTOS[@]}"; do
  for s in "${SEEDS[@]}"; do
    JOBS+=("$s|$proto")
  done
done
echo "[critic_noswitch] ${#JOBS[@]} jobs queued"

run_slot() {
  local slot=$1
  local i
  for ((i = slot; i < ${#JOBS[@]}; i += N_GPU)); do
    IFS='|' read -r s proto <<< "${JOBS[$i]}"
    # shellcheck disable=SC2086
    CUDA_VISIBLE_DEVICES=$slot SEED=$s bash scripts/gas_mpc_run.sh "[$((i+1))/${#JOBS[@]}]" subgoal_tdr $proto $OVER
  done
}

for ((k = 0; k < N_GPU; k++)); do
  run_slot $k > "$L/ada_critic_noswitch_slot$k.log" 2>&1 &
done
wait
echo "[critic_noswitch] all slots finished $(date)"
