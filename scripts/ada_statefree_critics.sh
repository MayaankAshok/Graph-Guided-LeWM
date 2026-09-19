#!/usr/bin/env bash
# Sequential jobs keep full caches within the allocation's 20 GB memory limit.
set -euo pipefail
cd /home2/mayaank.ashok/lewm_research
PY=/home2/mayaank.ashok/.venv/bin/python
export MUJOCO_GL=egl OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PUSHT_H5_PATH=/ssd_scratch/mayaank.ashok/lewm_data/datasets/pusht_expert_train.h5
export REACHER_H5_PATH=/ssd_scratch/mayaank.ashok/lewm_data/datasets/reacher.h5
export CUBE_H5_PATH=/ssd_scratch/mayaank.ashok/lewm_data/datasets/ogbench/cube_single_expert.h5
mkdir -p outputs/pusht/logs/statefree_critics
for env in pusht reacher cube; do
  export GAS_MPC_ENV=$env CUDA_VISIBLE_DEVICES=0
  out=outputs/$env
  mkdir -p "$out/logs"
  if [ ! -f "$out/cache_train.npz" ]; then
    echo "[$env] preparing training-only cache $(date)"
    "$PY" scripts/gas_mpc_prepare.py encode > "$out/logs/statefree_encode.log" 2>&1
  fi
  for s in 0 1 2 3 4; do
    tdr=$out/tdr_full_s$s.pt
    if [ ! -f "$tdr" ]; then
      "$PY" scripts/gas_mpc_prepare.py tdr --seed "$s" > "$out/logs/statefree_tdr_s$s.log" 2>&1
    fi
    dest=$out/critic_s${s}_tdr_holdout
    if [ -f "$dest/critic.pt" ] && "$PY" - "$dest/critic.pt" <<'PY'
import sys, torch
ck = torch.load(sys.argv[1], map_location='cpu', weights_only=False)
assert ck['step'] >= 60000
assert ck['args']['label_source'] == 'tdr' and ck['args']['exclude_tdr_holdout']
assert ck['args'].get('training_only')
assert not set(ck['evaluation_episodes']) & (set(ck['train_episodes']) | set(ck['val_episodes']))
PY
    then continue; fi
    echo "[$env s$s] state-free critic, fixed holdout excluded $(date)"
    "$PY" scripts/viability_train.py --env "$env" --cache "$out/cache_train.npz" \
      --out "$dest" --seed "$s" --steps 60000 --label-source tdr --tdr "$tdr" \
      --exclude-tdr-holdout > "$out/logs/critic_s${s}_tdr_holdout.log" 2>&1
    echo "[$env s$s] finished $(date)"
  done
done
echo "[statefree] Push-T, Reacher, and Cube finished $(date)"
