#!/usr/bin/env bash
# Build graphs at H_TD in {4,12,16} x 5 seeds (TE=0.9), needed before the H_TD ablation
# sweep on config 10 (A + ET critic beta=1). H_TD=8 graphs already exist. TDR checkpoints
# are H_TD-independent (H_TD is graph-construction only), so no TDR retraining needed.
#   nohup srun --jobid=<JOB> --overlap bash scripts/ada_gas_mpc_graphs_htd.sh > outputs/pusht/logs/ada_graphs_htd.log 2>&1 &
cd /home2/mayaank.ashok/lewm_research || exit 1
export PUSHT_H5_PATH=${PUSHT_H5_PATH:-/ssd_scratch/mayaank.ashok/lewm_data/datasets/pusht_expert_train.h5}
export MUJOCO_GL=egl OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
PY=/home2/mayaank.ashok/.venv/bin/python
SEEDS=(0 1 2 3 4)
HTDS=(4 12 16)
N_GPU=4
mkdir -p outputs/pusht/logs
L=outputs/pusht/logs

JOBS=()
for h in "${HTDS[@]}"; do
  for s in "${SEEDS[@]}"; do
    JOBS+=("$s|$h")
  done
done
echo "[graphs_htd] ${#JOBS[@]} graph builds queued: htd=${HTDS[*]} seeds=${SEEDS[*]}"

run_slot() {
  local slot=$1
  local i
  for ((i = slot; i < ${#JOBS[@]}; i += N_GPU)); do
    IFS='|' read -r s h <<< "${JOBS[$i]}"
    echo "[graphs_htd slot$slot] seed=$s htd=$h $(date)"
    CUDA_VISIBLE_DEVICES=$slot $PY scripts/gas_mpc_prepare.py graph --seed "$s" --h-td "$h" --te 0.9 \
      >> "$L/graphs_htd_s${s}_htd${h}.log" 2>&1
  done
}

for ((k = 0; k < N_GPU; k++)); do
  run_slot $k > "$L/ada_graphs_htd_slot$k.log" 2>&1 &
done
wait
echo "[graphs_htd] all done $(date)"
