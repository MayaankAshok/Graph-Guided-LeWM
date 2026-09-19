#!/usr/bin/env bash
# Re-encode only training episodes; retain verified TDR weights and replace diagnostics.
# Run sequentially: the allocation has 20 GB RAM and graph preparation must run alone.
set -euo pipefail
cd /home2/mayaank.ashok/lewm_research
export PY=/home2/mayaank.ashok/.venv/bin/python
export CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PUSHT_H5_PATH=/ssd_scratch/mayaank.ashok/lewm_data/datasets/pusht_expert_train.h5
export REACHER_H5_PATH=/ssd_scratch/mayaank.ashok/lewm_data/datasets/reacher.h5
export CUBE_H5_PATH=/ssd_scratch/mayaank.ashok/lewm_data/datasets/ogbench/cube_single_expert.h5
for env in pusht reacher cube; do
  export GAS_MPC_ENV=$env
  export GAS_MPC_TRAIN_CACHE_DIR=/ssd_scratch/mayaank.ashok/planning_trainonly/$env
  out=outputs/$env
  mkdir -p "$out/logs"
  echo "[$env] encoding training episodes only $(date)"
  "$PY" scripts/gas_mpc_prepare.py encode > "$out/logs/encode_trainonly.log" 2>&1
  for s in 0 1 2 3 4; do
    "$PY" scripts/gas_mpc_prepare.py refresh --seed "$s" > "$out/logs/tdr_refresh_s$s.log" 2>&1
  done
  "$PY" - "$out" "$env" <<'PY'
import json, pickle, sys, torch
from pathlib import Path
out, env = Path(sys.argv[1]), sys.argv[2]
def archive(path, group):
    dest = out / 'legacy' / group / path.name
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        raise FileExistsError(f'Refusing to overwrite {dest}')
    path.rename(dest)
for path in out.glob('gap_calib_*.json'):
    if json.loads(path.read_text()).get('scope') != 'training_only':
        archive(path, 'full_dataset_calibration')
path = out / 'calib.json'
if path.exists() and json.loads(path.read_text()).get('scope') != 'training_only':
    archive(path, 'full_dataset_calibration')
for path in out.glob('psi_full_*.npy'):
    archive(path, 'full_dataset_features')
for path in out.glob('graph_full_*.pkl'):
    with path.open('rb') as file:
        graph = pickle.load(file)
    if not graph.get('training_only'):
        archive(path, 'graphs_before_training_only')
ck = torch.load(out/'tdr_full_s0.pt', map_location='cpu', weights_only=False)
assert ck['cfg']['diagnostic_scope'] == 'training_only'
med = {int(k): float(v) for k, v in ck['history'][-1]['median_by_gap'].items()}
import numpy as np
gaps = sorted(med)
h_td = 8.0 if env == 'pusht' else float(np.interp(12, gaps, [med[g] for g in gaps]))
calib = dict(scope='training_only', h_td=round(h_td,2), lookahead=round(med[25],2),
             htd_steps=12, median_by_gap=med, xneg_min_dist=0.0)
(out/'calib.json').write_text(json.dumps(calib, indent=2))
print(env,calib)
PY
  htd=$($PY -c "import json; print(json.load(open('$out/calib.json'))['h_td'])")
  for s in 0 1 2 3 4; do
    echo "[$env s$s] training-only features, calibration and graph $(date)"
    "$PY" -c "import sys; sys.path.insert(0,'scripts'); from gas_mpc_eval import gap_calibration; gap_calibration($s)" \
      > "$out/logs/gap_calib_trainonly_s$s.log" 2>&1
    "$PY" scripts/gas_mpc_prepare.py graph --seed "$s" --h-td "$htd" --te 0.9 \
      > "$out/logs/graph_trainonly_s$s.log" 2>&1
  done
  echo "[$env] train-only preparation finished $(date)"
done
echo "[trainonly] all three environments finished $(date)"
