#!/usr/bin/env bash
# Push-T critic without the state column: retrain the viability critic with
# `--label-source tdr` (tau from a TDR ball, cross-negatives gated by the calibrated TDR
# gap distance -- viability_train.py docstring) for seeds 0-4, then run the phase-8
# headline config B (subgoal_tdr + final switch + et critic, beta=1) on every protocol with
# the new critics. Results carry `+mpc.tag=_tdrlab`, so they sit next to (not on top of)
# the state-labelled B rows in gas_mpc_report.py; L2 and state-B for the same seeds/tasks
# already exist from phase 8.
#
# Stage 1 trains critics s0..s3 one per GPU and s4 alongside s0 (z cache 1.8 GB + psi on
# each GPU; two fit on a 1080 Ti). Stage 2 = SEEDS x PROTOS eval jobs, protocol-major,
# round-robin over N_GPU * SLOTS_PER_GPU sequential queues. Both stages are resumable.
#   nohup srun --jobid=<JOB> --overlap bash scripts/ada_gas_mpc_tdrlabel.sh > outputs/pusht/logs/ada_driver_tdrlab.log 2>&1 &
#   STAGE=eval nohup srun ... bash scripts/ada_gas_mpc_tdrlabel.sh ...      # skip training
cd /home2/mayaank.ashok/lewm_research || exit 1
export PUSHT_H5_PATH=/ssd_scratch/mayaank.ashok/lewm_data/datasets/pusht_expert_train.h5
export MUJOCO_GL=egl OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PY=/home2/mayaank.ashok/.venv/bin/python
export N=${N:-50}
SEEDS=(${SEEDS:-0 1 2 3 4})
PROTOS=(${PROTOS:-same25 same50 same100 cross})
N_GPU=${N_GPU:-4}
SLOTS_PER_GPU=${SLOTS_PER_GPU:-2}
STAGE=${STAGE:-all}
CRITIC_STEPS=${CRITIC_STEPS:-60000}
XNEG_TDR_FACTOR=${XNEG_TDR_FACTOR:-1.25}
mkdir -p outputs/pusht/logs outputs/pusht/critic_training/logs
L=outputs/pusht/logs
echo "[tdrlab] armed $(date) stage=$STAGE seeds=${SEEDS[*]} protos=${PROTOS[*]} slots=$((N_GPU * SLOTS_PER_GPU))"

critic_dir() { echo "outputs/pusht/critic_training/critic_full_s${1}_tdr"; }

critic_done() {   # critic.pt exists and was saved at the final step
  [ -f "$(critic_dir "$1")/critic.pt" ] && $PY -c "
import torch, sys; ck = torch.load('$(critic_dir "$1")/critic.pt', map_location='cpu', weights_only=False)
sys.exit(0 if ck['step'] >= $CRITIC_STEPS else 1)" 2>/dev/null
}

run_critic() {
  local s=$1 gpu=$2
  if critic_done "$s"; then echo "[train s$s] critic present"; return 0; fi
  [ -f outputs/pusht/tdr_full_s$s.pt ] || { echo "[train s$s] missing tdr_full_s$s.pt"; return 1; }
  echo "[train s$s gpu$gpu] critic (tdr labels) $(date)"
  CUDA_VISIBLE_DEVICES=$gpu $PY scripts/viability_train.py --cache outputs/pusht/critic_training/cache_full_s0.npz \
      --out "$(critic_dir "$s")" --seed "$s" --steps "$CRITIC_STEPS" --log-every 200 --eval-every 2000 \
      --label-source tdr --tdr outputs/pusht/tdr_full_s$s.pt --xneg-tdr-factor "$XNEG_TDR_FACTOR" \
      >> outputs/pusht/critic_training/logs/critic_s${s}_tdr.log 2>&1 || { echo "[train s$s] critic FAILED"; return 1; }
  echo "[train s$s] critic done $(date)"
}

if [ "$STAGE" = all ] || [ "$STAGE" = train ]; then
  ( run_critic 0 0; ) &
  ( run_critic 4 0; ) &
  ( run_critic 1 1; ) &
  ( run_critic 2 2; ) &
  ( run_critic 3 3; ) &
  wait
  echo "[tdrlab] training stage finished $(date)"
  for s in "${SEEDS[@]}"; do critic_done "$s" || echo "[tdrlab] WARNING critic s$s not done"; done
fi
[ "$STAGE" = train ] && exit 0

B="+mpc.lookahead=13.7 +mpc.final_thresh=13.7 +mpc.final_metric=l2 +mpc.critic_beta=1 +mpc.critic_cost=et +mpc.tag=_tdrlab"
JOBS=()
for proto in "${PROTOS[@]}"; do
  for s in "${SEEDS[@]}"; do
    critic_done "$s" || continue
    JOBS+=("$s|$proto|$B +mpc.critic=$(critic_dir "$s")/critic.pt")
  done
done
NSLOT=$((N_GPU * SLOTS_PER_GPU))
echo "[tdrlab] ${#JOBS[@]} eval jobs over $NSLOT slots $(date)"

run_slot() {
  local slot=$1
  local gpu=$((slot / SLOTS_PER_GPU))
  echo "[tdrlab slot$slot gpu$gpu] starting $(date)"
  local i
  for ((i = slot; i < ${#JOBS[@]}; i += NSLOT)); do
    IFS='|' read -r s proto over <<< "${JOBS[$i]}"
    # shellcheck disable=SC2086
    CUDA_VISIBLE_DEVICES=$gpu SEED=$s bash scripts/gas_mpc_run.sh subgoal_tdr $proto $over
  done
  echo "[tdrlab slot$slot gpu$gpu] finished $(date)"
}

for ((k = 0; k < NSLOT; k++)); do
  run_slot $k > "$L/ada_tdrlab_slot$k.log" 2>&1 &
done
wait
echo "[tdrlab] all slots finished $(date)"
