#!/usr/bin/env bash
# Random-action baseline (+mpc.method=random) across every env, every protocol, 5 seeds --
# the floor every other GAS-MPC method must clear. No model, no CEM, no graph: each env step
# is env.action_space.sample() (stable_worldmodel's RandomPolicy), so this needs no TDR/graph/
# critic assets and no meaningful GPU compute -- only the env's own step/render. Still uses
# gas_mpc_run.sh (resumable per-chunk caches) so a re-run only fills in missing (env, protocol,
# seed) combinations. Envs must already have their pairs/pairs_*_task200u.json pools and
# staged datasets (data stage of ada_gas_mpc_reacher.sh / ada_gas_mpc_cube.sh) -- this script
# does not create either.
#
# Launch from the LOGIN node inside an existing allocation:
#   nohup srun --jobid=<JOB> --overlap bash scripts/ada_gas_mpc_random_baseline.sh \
#     > outputs/random_baseline_driver.log 2>&1 &
#
# Knobs (env vars): ENVS="pusht reacher cube" SEEDS="0 1 2 3 4" PROTOS="same25 same50 same100 cross"
# N=200 N_GPU=4 SLOTS_PER_GPU=1 (Reacher/Cube EGL: 2 slots/GPU OOMed before MUJOCO_EGL_DEVICE_ID
# was pinned per-slot; this script pins it, but keep 1 unless you've verified 2 fits).
set -u
cd /home2/mayaank.ashok/lewm_research || exit 1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1
export PY=/home2/mayaank.ashok/.venv/bin/python
ENVS=(${ENVS:-pusht reacher cube})
SEEDS=(${SEEDS:-0 1 2 3 4})
PROTOS=(${PROTOS:-same25 same50 same100 cross})
N_GPU=${N_GPU:-4}
SLOTS_PER_GPU=${SLOTS_PER_GPU:-1}
export N=${N:-200}
SCRATCH=/ssd_scratch/mayaank.ashok/lewm_data

declare -A H5PATH=(
  [pusht]="PUSHT_H5_PATH=$SCRATCH/datasets/pusht_expert_train.h5"
  [reacher]="REACHER_H5_PATH=$SCRATCH/datasets/reacher.h5"
  [cube]="CUBE_H5_PATH=$SCRATCH/datasets/ogbench/cube_single_expert.h5"
)

JOBS=()   # env|protocol|seed
for e in "${ENVS[@]}"; do
  for proto in "${PROTOS[@]}"; do
    for s in "${SEEDS[@]}"; do
      JOBS+=("$e|$proto|$s")
    done
  done
done

NSLOT=$((N_GPU * SLOTS_PER_GPU))
echo "[random] ${#JOBS[@]} jobs (envs=${ENVS[*]} protos=${PROTOS[*]} seeds=${SEEDS[*]} n=$N) over $NSLOT slots $(date)"
mkdir -p outputs/random_baseline_logs

for ((k = 0; k < NSLOT; k++)); do
  (
    gpu=$((k / SLOTS_PER_GPU))
    for ((i = k; i < ${#JOBS[@]}; i += NSLOT)); do
      IFS='|' read -r e proto s <<< "${JOBS[$i]}"
      mkdir -p "outputs/$e/logs"
      logf="outputs/random_baseline_logs/${e}_${proto}_s${s}.log"
      echo "[$(date +%H:%M:%S)] slot$k gpu$gpu $e $proto s$s -> $logf"
      # shellcheck disable=SC2086
      env GAS_MPC_ENV="$e" MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=$gpu MUJOCO_EGL_DEVICE_ID=$gpu \
          ${H5PATH[$e]} SEED=$s bash scripts/gas_mpc_run.sh random "$proto" > "$logf" 2>&1
      grep "^=== .*success=" "$logf" | tail -1
    done
  ) > "outputs/random_baseline_logs/slot$k.log" 2>&1 &
done
wait
echo "[random] all slots finished $(date)"
