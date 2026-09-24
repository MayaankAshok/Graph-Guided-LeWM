#!/usr/bin/env bash
# Extends tab:budget-sweep: Push-T cross-episode, OUR vs L2, 3 seeds, budget 750
# (+mpc.budget=750, tag _bud750). 6 jobs, 1 per GPU over 3 GPUs.
#   nohup srun --jobid=<JOB> --overlap bash scripts/ada_gas_mpc_cross_bud750.sh \
#     > outputs/pusht/logs/ada_cross_bud750.log 2>&1 &
cd /home2/mayaank.ashok/lewm_research || exit 1
export PUSHT_H5_PATH=${PUSHT_H5_PATH:-/ssd_scratch/mayaank.ashok/lewm_data/datasets/pusht_expert_train.h5}
export MUJOCO_GL=egl OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1
export PY=/home2/mayaank.ashok/.venv/bin/python
export N=${N:-50}
N_GPU=${N_GPU:-3}
BUD="+mpc.budget=750"
OUR="+mpc.lookahead=13.7 +mpc.final_thresh=13.7 +mpc.final_metric=l2 +mpc.critic_beta=1 +mpc.critic_cost=et"
JOBS=(
  "0|l2|$BUD" "1|l2|$BUD" "2|l2|$BUD"
  "0|subgoal_tdr|$OUR $BUD" "1|subgoal_tdr|$OUR $BUD" "2|subgoal_tdr|$OUR $BUD"
)
L=outputs/pusht/logs
mkdir -p $L
echo "[bud750] ${#JOBS[@]} jobs over $N_GPU GPUs $(date)"
for ((k = 0; k < N_GPU; k++)); do
  (
    local_gpu=$k
    for ((i = k; i < ${#JOBS[@]}; i += N_GPU)); do
      IFS='|' read -r s m ov <<< "${JOBS[$i]}"
      echo "[$(date +%H:%M:%S)] gpu$local_gpu $m s$s"
      # shellcheck disable=SC2086
      CUDA_VISIBLE_DEVICES=$local_gpu SEED=$s bash scripts/gas_mpc_run.sh "[$((i+1))/${#JOBS[@]}]" $m cross $ov
    done
  ) > $L/cross_bud750_gpu$k.log 2>&1 &
done
wait
echo "[bud750] done $(date)"
