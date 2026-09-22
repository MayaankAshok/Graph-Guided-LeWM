#!/usr/bin/env bash
# Build training-only long-horizon labels, train the three one-pass candidates, and audit them.
set -euo pipefail
cd /home2/mayaank.ashok/lewm_research
export PY=/home2/mayaank.ashok/.venv/bin/python
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
CACHE=/ssd_scratch/mayaank.ashok/planning_trainonly/pusht/cache_train.npz
OUT=outputs/pusht/hitting_time
LABELS=$OUT/labels_s0.npz
mkdir -p "$OUT/logs"

# Do not overlap the memory-heavy 2.3M-node FAISS/Dijkstra build with the currently running
# evaluation chain. WAIT_PID is the chain's parent, supplied by the launcher.
if [[ -n "${WAIT_PID:-}" ]]; then
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 30; done
fi

if [[ ! -f "$LABELS" ]]; then
  CUDA_VISIBLE_DEVICES="" "$PY" scripts/viability_graph_labels.py \
    --cache "$CACHE" --out "$LABELS" --seed 0 --b-max 45 \
    --goals-train 20000 --goals-val 2000 --dijkstra-chunk 20 --workers 16 \
    > "$OUT/logs/labels_s0.log" 2>&1
fi

pids=()
heads=(softmax hazard weibull)
for gpu in 0 1 2; do
  head=${heads[$gpu]}
  CUDA_VISIBLE_DEVICES=$gpu MUJOCO_EGL_DEVICE_ID=$gpu "$PY" scripts/viability_train_ht.py \
    --cache "$CACHE" --labels "$LABELS" --out "$OUT/${head}_s0" \
    --seed 0 --b-max 45 --head "$head" --steps 60000 --eval-every 2000 \
    > "$OUT/logs/${head}_s0.log" 2>&1 &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid"; done

CUDA_VISIBLE_DEVICES=2 MUJOCO_EGL_DEVICE_ID=2 "$PY" scripts/viability_long_horizon_audit.py \
  --cache "$CACHE" --labels "$LABELS" --eval-n 20000 --pred-rollouts 4000 \
  --critics \
    legacy=outputs/pusht/critic_s0_tdr_holdout/critic.pt \
    categorical="$OUT/softmax_s0/critic.pt" \
    hazard="$OUT/hazard_s0/critic.pt" \
    weibull="$OUT/weibull_s0/critic.pt" \
  --out "$OUT/long_horizon_audit_s0.json" \
  > "$OUT/logs/long_horizon_audit_s0.log" 2>&1
touch "$OUT/phase1.done"
