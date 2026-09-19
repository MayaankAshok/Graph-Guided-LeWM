#!/usr/bin/env bash
# Push-T old/new task-pool x critic comparison.  The sole critic configuration is phase-8 B:
# subgoal_tdr + final L2 switch + budget-capped ET critic (beta=1, std composition).
#
# Cells:
#   L2           on new tasks (task200u)
#   old critic   on old tasks (task200)
#   old critic   on new tasks (task200u)
#   new critic   on new tasks (task200u)
#
# NEW_CRITIC_TPL must name the critic being trained now and include a literal %s for its
# learned-asset seed, e.g. 'outputs/pusht/critic_training/critic_full_new_s%s/critic.pt'.  It is required
# so a new training run cannot accidentally be evaluated with the historical critic.
#
# Run from Ada's login node after staging Push-T, for example:
#   NEW_CRITIC_TPL='outputs/pusht/critic_training/critic_full_new_s%s/critic.pt' \
#     nohup srun --jobid=<JOB> --overlap bash scripts/ada_gas_mpc_task_critic_matrix.sh \
#     > outputs/pusht/logs/ada_task_critic_matrix.log 2>&1 &
cd /home2/mayaank.ashok/lewm_research || exit 1
export PUSHT_H5_PATH=/ssd_scratch/mayaank.ashok/lewm_data/datasets/pusht_expert_train.h5
export MUJOCO_GL=egl OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PY=/home2/mayaank.ashok/.venv/bin/python
export N=${N:-50}
SEEDS=(${SEEDS:-0 1 2 3 4})
PROTOS=(${PROTOS:-same25 same50 same100 cross})
N_GPU=${N_GPU:-4}
SLOTS_PER_GPU=${SLOTS_PER_GPU:-2}
OLD_CRITIC_TPL=${OLD_CRITIC_TPL:-outputs/pusht/critic_training/critic_full_s%s/critic.pt}
: "${NEW_CRITIC_TPL:?Set NEW_CRITIC_TPL, e.g. outputs/pusht/critic_training/critic_full_new_s%s/critic.pt}"
mkdir -p outputs/pusht/logs
L=outputs/pusht/logs

B='+mpc.lookahead=13.7 +mpc.final_thresh=13.7 +mpc.final_metric=l2 +mpc.critic_beta=1 +mpc.critic_cost=et'
JOBS=()
for proto in "${PROTOS[@]}"; do
  for s in "${SEEDS[@]}"; do
    # L2 has no learned critic and is intentionally evaluated once per CEM seed.
    JOBS+=("$s|$proto|task200u|l2|+mpc.tag=_matrix_newtasks")
    old_critic=$(printf "$OLD_CRITIC_TPL" "$s")
    new_critic=$(printf "$NEW_CRITIC_TPL" "$s")
    [ -f "$old_critic" ] || { echo "[matrix] missing old critic: $old_critic"; exit 1; }
    [ -f "$new_critic" ] || { echo "[matrix] missing new critic: $new_critic"; exit 1; }
    JOBS+=("$s|$proto|task200|subgoal_tdr|$B +mpc.critic=$old_critic +mpc.tag=_matrix_oldcritic_oldtasks")
    JOBS+=("$s|$proto|task200u|subgoal_tdr|$B +mpc.critic=$old_critic +mpc.tag=_matrix_oldcritic_newtasks")
    JOBS+=("$s|$proto|task200u|subgoal_tdr|$B +mpc.critic=$new_critic +mpc.tag=_matrix_newcritic_newtasks")
  done
done

NSLOT=$((N_GPU * SLOTS_PER_GPU))
echo "[matrix] armed $(date) seeds=${SEEDS[*]} protos=${PROTOS[*]} n=$N slots=$NSLOT jobs=${#JOBS[@]}"
run_slot() {
  local slot=$1
  local gpu=$((slot / SLOTS_PER_GPU))
  local i s proto pool method over
  echo "[matrix slot$slot gpu$gpu] starting $(date)"
  for ((i = slot; i < ${#JOBS[@]}; i += NSLOT)); do
    IFS='|' read -r s proto pool method over <<< "${JOBS[$i]}"
    # shellcheck disable=SC2086
    CUDA_VISIBLE_DEVICES=$gpu GAS_MPC_POOL=$pool SEED=$s bash scripts/gas_mpc_run.sh \
      "[$((i + 1))/${#JOBS[@]}]" "$method" "$proto" $over
  done
  echo "[matrix slot$slot gpu$gpu] finished $(date)"
}
for ((k = 0; k < NSLOT; k++)); do
  run_slot "$k" > "$L/ada_task_critic_matrix_slot$k.log" 2>&1 &
done
wait
echo "[matrix] all slots finished $(date)"
