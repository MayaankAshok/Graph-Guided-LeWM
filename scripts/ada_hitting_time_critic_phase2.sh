#!/usr/bin/env bash
# Select the best long-range head, train seeds 1--2, then run all Push-T task200u goals.
set -euo pipefail
cd /home2/mayaank.ashok/lewm_research
export PY=/home2/mayaank.ashok/.venv/bin/python
export PUSHT_H5_PATH=/ssd_scratch/mayaank.ashok/lewm_data/datasets/pusht_expert_train.h5
export MUJOCO_GL=egl OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export GAS_MPC_POOL=task200u N=50
CACHE=/ssd_scratch/mayaank.ashok/planning_trainonly/pusht/cache_train.npz
OUT=outputs/pusht/hitting_time
LABELS=$OUT/labels_s0.npz
mkdir -p "$OUT/logs"
while [[ ! -f "$OUT/phase1.done" ]]; do sleep 30; done

WINNER=$("$PY" -c "import json; d=json.load(open('$OUT/long_horizon_audit_s0.json'))['sets']['predicted']; names=[n for n in ['categorical','hazard','weibull'] if n in d]; print(max(names,key=lambda n:d[n]['gt50_exact']['spearman_cost_vs_TG']))")
case "$WINNER" in
  categorical) HEAD=softmax ;;
  hazard) HEAD=hazard ;;
  weibull) HEAD=weibull ;;
  *) echo "unknown winner $WINNER"; exit 1 ;;
esac
printf '%s\n' "$WINNER" > "$OUT/winner.txt"

pids=()
for seed in 1 2; do
  if [[ ! -f "$OUT/${HEAD}_s${seed}/critic.pt" ]]; then
    gpu=$seed
    CUDA_VISIBLE_DEVICES=$gpu MUJOCO_EGL_DEVICE_ID=$gpu "$PY" scripts/viability_train_ht.py \
      --cache "$CACHE" --labels "$LABELS" --out "$OUT/${HEAD}_s${seed}" \
      --seed "$seed" --b-max 45 --head "$HEAD" --steps 60000 --eval-every 2000 \
      > "$OUT/logs/${HEAD}_s${seed}.log" 2>&1 &
    pids+=("$!")
  fi
done
for pid in "${pids[@]}"; do wait "$pid"; done

JOBS=()
for proto in same25 same50 same100 cross; do
  for seed in 0 1 2; do JOBS+=("$seed|$proto"); done
done
NSLOT=8
run_slot() {
  local slot=$1
  local gpu=$((slot / 2))
  local i seed proto critic
  for ((i=slot; i<${#JOBS[@]}; i+=NSLOT)); do
    IFS='|' read -r seed proto <<< "${JOBS[$i]}"
    critic="$OUT/${HEAD}_s${seed}/critic.pt"
    CUDA_VISIBLE_DEVICES=$gpu MUJOCO_EGL_DEVICE_ID=$gpu SEED=$seed \
      bash scripts/gas_mpc_run.sh "[$((i + 1))/${#JOBS[@]}]" subgoal_tdr "$proto" \
      +mpc.h_td=8 +mpc.te=0.9 +mpc.lookahead=13.53 \
      +mpc.final_thresh=13.53 +mpc.final_metric=l2 \
      +mpc.critic_beta=1 +mpc.critic_cost=et +mpc.critic="$critic" \
      +mpc.tag="_ht_${WINNER}" \
      > "$OUT/logs/eval_${WINNER}_${proto}_s${seed}.log" 2>&1
  done
}
for slot in $(seq 0 $((NSLOT - 1))); do run_slot "$slot" & done
wait
touch "$OUT/phase2.done"
