#!/usr/bin/env bash
# GCIQL baseline (LeWM paper protocol, scripts/gciql_train.py) on one env: train once on every
# training episode (task-pool episodes excluded), then evaluate the same policy checkpoint on all
# four goal protocols of the task200u pool through gas_mpc_eval.py +mpc.method=gciql.
# Trains from the cached training-only LeWM latents (outputs/<env>/cache_train.npz).
# Resumable: each phase resumes from outputs/<env>/gciql/checkpoints/gciql_*/resume.pt,
# and evals skip finished chunks.
#
# Launch from the LOGIN node inside an existing allocation:
#   nohup srun --jobid=<JOB> --overlap bash scripts/ada_gciql.sh > outputs/pusht/gciql_driver.log 2>&1 &
#
# Knobs: GAS_MPC_ENV=pusht TRAIN_GPU=0 GPUS="0 1 2" PROTOS="cross same100 same50 same25" N=50
#        PRECISION=32 (1080 Ti has no bf16; paper used bf16-mixed) SKIP_TRAIN=1 EXTRA="max_epochs=..."
set -u
cd /home2/mayaank.ashok/lewm_research || exit 1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1
export PY=/home2/mayaank.ashok/.venv/bin/python
export GAS_MPC_ENV=${GAS_MPC_ENV:-pusht}
case $GAS_MPC_ENV in
  pusht)   export PUSHT_H5_PATH=${PUSHT_H5_PATH:-/ssd_scratch/mayaank.ashok/lewm_data/datasets/pusht_expert_train.h5} ;;
  reacher) export REACHER_H5_PATH=${REACHER_H5_PATH:-/ssd_scratch/mayaank.ashok/lewm_data/datasets/reacher.h5} ;;
  cube)    export CUBE_H5_PATH=${CUBE_H5_PATH:-/ssd_scratch/mayaank.ashok/lewm_data/datasets/ogbench/cube_single_expert.h5} ;;
esac
OUTD=${GAS_MPC_OUT:-outputs/$GAS_MPC_ENV}
LOGD=$OUTD/gciql/logs
mkdir -p "$LOGD"
TRAIN_GPU=${TRAIN_GPU:-0}
GPUS=(${GPUS:-0 1 2})
PROTOS=(${PROTOS:-cross same100 same50 same25})
export N=${N:-50}

if [ "${SKIP_TRAIN:-0}" != 1 ]; then
  echo "[gciql] training on GPU $TRAIN_GPU $(date) -> $LOGD/train.log"
  # shellcheck disable=SC2086
  CUDA_VISIBLE_DEVICES=$TRAIN_GPU $PY scripts/gciql_train.py precision=${PRECISION:-32} ${EXTRA:-} \
      >> "$LOGD/train.log" 2>&1 || { echo "[gciql] training FAILED, see $LOGD/train.log"; exit 1; }
  echo "[gciql] training done $(date)"
fi

# One protocol per GPU slot (the policy is deterministic, so one CEM/env seed per protocol).
for i in "${!PROTOS[@]}"; do
  (
    slot=$((i % ${#GPUS[@]}))
    local_gpu=${GPUS[$slot]}
    proto=${PROTOS[$i]}
    echo "[$(date +%H:%M:%S)] gpu$local_gpu gciql $proto"
    env CUDA_VISIBLE_DEVICES=$local_gpu MUJOCO_EGL_DEVICE_ID=$local_gpu SEED=0 \
        bash scripts/gas_mpc_run.sh "[$((i + 1))/${#PROTOS[@]}]" gciql "$proto" > "$LOGD/eval_${proto}.log" 2>&1
    grep "^=== .*success=" "$LOGD/eval_${proto}.log" | tail -1
  ) &
  # never more concurrent evals than GPUs
  if (( (i + 1) % ${#GPUS[@]} == 0 )); then wait; fi
done
wait
echo "[gciql] all evals finished $(date)"
