#!/usr/bin/env bash
# Cube L2 baseline audit (2026-09-18). gas_mpc_eval's L2 same25 on Cube is 56.8% (5 seeds, first
# 50 of the fixed 200-task pool) while the LeWM paper's Fig. 6 reports 74% for the same frozen
# checkpoint at the same protocol (goal offset 25, budget 50). Four arms, same checkpoint, same
# dataset, one process per GPU, every run ~3 min (200-task runs ~11 min):
#   ref    upstream eval.py, paper protocol verbatim: 50 tasks drawn by cfg.seed (42/43/44, each
#          seed = its own task draw + CEM seed), CEM n_steps=30 (config/eval/solver/cem.yaml)
#   ref10  the same with solver.n_steps=10 -- the paper's stated CEM setting for Cube (App. D:
#          "30 iterations in PushT and 10 iterations in the other environments")
#   pool   gas_mpc_eval l2 same25 over the WHOLE 200-task pool (the first-50 prefix has 22%
#          pre-solved tasks vs 38.4% of all dataset windows -- a hard draw), seeds 0/1/2
#   cem10  gas_mpc_eval l2 same25, first 50, solver.n_steps=10, seeds 0/1/2 (tag _cem10)
# Launch from the LOGIN node: nohup srun --jobid=<JOB> --overlap bash scripts/ada_cube_baseline_check.sh \
#     > outputs/cube/logs/baseline_check.log 2>&1 &
set -u
cd /home2/mayaank.ashok/lewm_research || exit 1
export GAS_MPC_ENV=cube
export SCRATCH=/ssd_scratch/mayaank.ashok/lewm_data
export CUBE_H5_PATH=$SCRATCH/datasets/ogbench/cube_single_expert.h5
export CUBE_CKPT_DIR=/home2/mayaank.ashok/lewm_research/data/checkpoints/models--quentinll--lewm-cube
export MUJOCO_GL=egl OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1
export PY=/home2/mayaank.ashok/.venv/bin/python
# eval.py resolves <STABLEWM_HOME>/datasets/<dataset_name>.h5 and <STABLEWM_HOME>/checkpoints/<policy>
export STABLEWM_HOME=$SCRATCH
mkdir -p "$SCRATCH/checkpoints"
[ -e "$SCRATCH/checkpoints/lewm-cube" ] || ln -s "$CUBE_CKPT_DIR" "$SCRATCH/checkpoints/lewm-cube"
L=outputs/cube/logs/baseline_check
mkdir -p "$L"
echo "[check] armed $(date) host=$(hostname)"

ref() {   # ref <seed> <n_steps>
  local s=$1 it=$2
  local logf="$L/ref_cem${it}_s${s}.log"   # separate `local`: one-line form expands $it before assigning it
  if grep -q "success_rate" "$logf" 2>/dev/null; then echo "[ref cem$it s$s] cached: $(grep -o "'success_rate': [0-9.]*" "$logf")"; return; fi
  echo "[$(date +%H:%M:%S)] ref cem$it seed $s -> $logf"
  $PY eval.py --config-name=cube policy=lewm-cube +cache_dir=null seed="$s" eval.num_eval=50 \
      solver.n_steps="$it" > "$logf" 2>&1
  grep -o "'success_rate': [0-9.]*" "$logf" | tail -1
}
ours() {  # ours <seed> <n> [extra hydra overrides]
  local s=$1 n=$2; shift 2
  SEED=$s N=$n bash scripts/gas_mpc_run.sh l2 same25 "$@"
}

ARMS=${ARMS:-ref pool cem10}   # which slots to run (e.g. ARMS=ref to redo only the eval.py arm)
REF_SEEDS=${REF_SEEDS:-42 43 44}  # eval.py seeds (each = its own 50-task draw + CEM seed)
REF_ITERS=${REF_ITERS:-30 10}
REF_GPU=${REF_GPU:-0}
[[ " $ARMS " == *" ref "* ]] && ( export CUDA_VISIBLE_DEVICES=$REF_GPU MUJOCO_EGL_DEVICE_ID=$REF_GPU
  for it in $REF_ITERS; do for s in $REF_SEEDS; do ref $s $it; done; done
) >> "$L/slot_ref_gpu$REF_GPU.log" 2>&1 &
[[ " $ARMS " == *" pool "* ]] && ( export CUDA_VISIBLE_DEVICES=1 MUJOCO_EGL_DEVICE_ID=1
  ours 0 200; ours 1 200
) > "$L/slot1.log" 2>&1 &
[[ " $ARMS " == *" cem10 "* ]] && ( export CUDA_VISIBLE_DEVICES=2 MUJOCO_EGL_DEVICE_ID=2
  ours 2 200
  for s in 0 1 2; do ours $s 50 solver.n_steps=10 +mpc.tag=_cem10; done
) > "$L/slot2.log" 2>&1 &
wait
echo "[check] finished $(date)"
