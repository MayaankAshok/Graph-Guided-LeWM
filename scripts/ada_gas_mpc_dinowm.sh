#!/usr/bin/env bash
# DINO-WM baseline on Push-T: the original dino_wm checkpoint (dinowm/outputs/pusht/checkpoints/
# model_latest.pth) planned with the same CEM as LeWM (+mpc.wm=dinowm +mpc.method=l2, i.e. DINO-WM's
# own terminal visual+proprio MSE), every protocol x 5 CEM seeds on the task200u pool. Loader and
# input normalisation: scripts/common/dinowm_loader.py. Resumable through gas_mpc_run.sh's per-chunk
# caches. Jobs are pulled from a shared queue (longest protocols first), one slot per GPU.
#
# Launch from the LOGIN node inside an existing allocation:
#   nohup srun --jobid=<JOB> --overlap bash scripts/ada_gas_mpc_dinowm.sh \
#     > outputs/pusht/dinowm_driver.log 2>&1 &
#
# Knobs: SEEDS="0 1 2 3 4" PROTOS="cross same100 same50 same25" N=50 GPUS="0 1 2 3"
set -u
cd /home2/mayaank.ashok/lewm_research || exit 1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1
export PY=/home2/mayaank.ashok/.venv/bin/python
export PUSHT_H5_PATH=/ssd_scratch/mayaank.ashok/lewm_data/datasets/pusht_expert_train.h5
SEEDS=(${SEEDS:-0 1 2 3 4})
PROTOS=(${PROTOS:-cross same100 same50 same25})
GPUS=(${GPUS:-0 1 2 3})
export N=${N:-50}

JOBS=()   # protocol|seed
for proto in "${PROTOS[@]}"; do
  for s in "${SEEDS[@]}"; do
    JOBS+=("$proto|$s")
  done
done

LOGD=outputs/pusht/dinowm_logs
mkdir -p "$LOGD"
QFILE=$LOGD/queue.next
echo 0 > "$QFILE"
echo "[dinowm] ${#JOBS[@]} jobs (protos=${PROTOS[*]} seeds=${SEEDS[*]} n=$N) over GPUs ${GPUS[*]} $(date)"

next_job() {   # atomically pop the next job index from the shared counter
  (
    flock 9
    local i; i=$(cat "$QFILE")
    echo $((i + 1)) > "$QFILE"
    echo "$i"
  ) 9> "$QFILE.lock"
}

for slot in "${!GPUS[@]}"; do
  (
    local_gpu=${GPUS[$slot]}
    while true; do
      i=$(next_job)
      [ "$i" -ge "${#JOBS[@]}" ] && break
      IFS='|' read -r proto s <<< "${JOBS[$i]}"
      logf="$LOGD/${proto}_s${s}.log"
      echo "[$(date +%H:%M:%S)] slot$slot gpu$local_gpu $proto s$s -> $logf"
      env CUDA_VISIBLE_DEVICES=$local_gpu SEED=$s \
          bash scripts/gas_mpc_run.sh "[$((i + 1))/${#JOBS[@]}]" l2 "$proto" +mpc.wm=dinowm > "$logf" 2>&1
      grep "^=== .*success=" "$logf" | tail -1
    done
  ) > "$LOGD/slot$slot.log" 2>&1 &
done
wait
echo "[dinowm] all slots finished $(date)"
