#!/usr/bin/env bash
# Push-T TDR-labelled critics: seeds 1..4 on four allocated GPUs.
# From the login node:
# nohup srun --jobid=<FOUR_GPU_JOB> --overlap bash scripts/ada_critic_holdout_seeds14.sh \
#   > outputs/pusht/logs/critic_holdout_seeds14_driver.log 2>&1 &
set -euo pipefail
cd /home2/mayaank.ashok/lewm_research
PY=/home2/mayaank.ashok/.venv/bin/python
export GAS_MPC_ENV=pusht MUJOCO_GL=egl
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PUSHT_H5_PATH=/ssd_scratch/mayaank.ashok/lewm_data/datasets/pusht_expert_train.h5
out=outputs/pusht
mkdir -p "$out/logs"
# Direct-node mode requires explicit confirmation that all four GPUs are usable.
if [[ ${DIRECT_FOUR_GPU_ACCESS:-0} != 1 ]]; then
  if [[ -z ${SLURM_JOB_ID:-} ]]; then
    echo 'Run inside a four-GPU allocation or set DIRECT_FOUR_GPU_ACCESS=1.' >&2; exit 1
  fi
  job=$(scontrol show job "$SLURM_JOB_ID" -o)
  if [[ ! $job =~ gres/gpu=([0-9]+) ]] || (( ${BASH_REMATCH[1]:-0} < 4 )); then
    echo 'This job must allocate at least four GPUs.' >&2; exit 1
  fi
fi
if [[ -n ${WAIT_PREPARATION_PID:-} ]]; then
  echo "Waiting for preparation PID $WAIT_PREPARATION_PID to finish $(date)"
  while kill -0 "$WAIT_PREPARATION_PID" 2>/dev/null; do sleep 30; done
  echo "Preparation exited; checking critic prerequisites $(date)"
fi
check() {
  "$PY" - "$1" "$2" <<'PY'
import json, sys, torch, numpy as np
from pathlib import Path
sys.path.insert(0, 'scripts')
from common.heldout_tasks import validate_training_cache
mode, seed = sys.argv[1], int(sys.argv[2])
p = Path('outputs/pusht')
with np.load(p/'cache_train.npz') as d:
    split = {k: d[k] for k in ('training_only', 'n_source_episodes', 'heldout_frac',
                              'episode_id', 'heldout_episode_ids')}
held = validate_training_cache(split)
train = split['episode_id']
pools = list((p/'pairs').glob('*task200u.json'))
assert len(pools) == 4, 'Expected four task200u protocol pools'
for f in pools:
    pool = json.loads(f.read_text())
    assert np.array_equal(np.sort(pool['heldout_eps']), held), f'Holdout mismatch: {f}'
    assert np.isin(pool['start_ep'] + pool['goal_ep'], held).all(), f'Invalid endpoints: {f}'
if mode == 'pre':
    assert torch.cuda.device_count() >= 4, 'Four visible allocated GPUs required'
    ck = torch.load(p/f'tdr_full_s{seed}.pt', map_location='cpu', weights_only=False)
    assert ck['done'] and ck['cfg']['diagnostic_scope'] == 'training_only'
    assert np.array_equal(ck['excluded_episode_ids'], held)
    assert np.isin(np.r_[ck['train_episode_ids'], ck['diagnostic_episode_ids']], train).all()
else:
    ck = torch.load(p/f'critic_s{seed}_tdr_holdout/critic.pt', map_location='cpu', weights_only=False)
    assert ck['step'] >= 60000 and ck['args']['seed'] == seed
    assert ck['args']['label_source'] == 'tdr' and ck['args']['training_only']
    assert ck['args']['exclude_tdr_holdout']
    assert np.array_equal(ck['evaluation_episodes'], held)
    used = np.r_[ck['train_episodes'], ck['val_episodes']]
    assert np.isin(used, train).all() and not np.isin(used, held).any()
    assert not np.intersect1d(ck['train_episodes'], ck['val_episodes']).size
print(f'[verified] seed {seed}: {len(held)} evaluation episodes excluded ({mode})', flush=True)
PY
}
# Check every seed before starting any worker.
for seed in 1 2 3 4; do check pre "$seed"; done
run_seed() {
  local seed=$1
  local gpu=$((seed - 1))
  local dest=$out/critic_s${seed}_tdr_holdout
  if [[ -f $dest/critic.pt ]]; then
    check post "$seed"
    echo "[seed $seed] verified completed checkpoint; skipping"
    return
  fi
  # Refuse to overwrite partial or otherwise unverified runs.
  if [[ -d $dest ]] && [[ -n $(ls -A "$dest") ]]; then
    echo "Refusing to overwrite nonempty $dest" >&2; return 1
  fi
  echo "[seed $seed GPU $gpu] starting $(date)"
  CUDA_VISIBLE_DEVICES=$gpu "$PY" scripts/viability_train.py \
    --env pusht --cache "$out/cache_train.npz" --out "$dest" \
    --seed "$seed" --steps 60000 --log-every 200 --eval-every 2000 \
    --label-source tdr --tdr "$out/tdr_full_s${seed}.pt" \
    --xneg-tdr-factor 1.25 --exclude-tdr-holdout
  check post "$seed"
  echo "[seed $seed] finished $(date)"
}
pids=()
for seed in 1 2 3 4; do
  run_seed "$seed" > "$out/logs/critic_s${seed}_tdr_holdout.log" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
if ((status)); then echo 'One or more critic jobs failed; inspect per-seed logs.' >&2; exit 1; fi
echo '[done] all four critics completed and passed holdout checks'
