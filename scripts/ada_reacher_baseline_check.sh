#!/usr/bin/env bash
# Reproduce the published LeWM Reacher baseline with the paper's evaluation protocol:
# 50 SAC-dataset starts, goals 25 environment steps ahead, budget 50, CEM 300/10/top-30,
# horizon 5 action blocks and open-loop execution of all 5 blocks (paper App. D and F.1).
#
# The current task200u headline pool is intentionally held-out, nontrivial and disjoint; it is
# not the paper's random dataset draw. This script uses upstream eval.py so each seed controls
# both the 50-task draw and CEM, exactly as in the published evaluation.
#
# Launch on the login node inside the current allocation:
#   nohup srun --jobid=<JOB> --overlap bash scripts/ada_reacher_baseline_check.sh \
#     > outputs/reacher/logs/baseline_check/driver.log 2>&1 &
set -u
cd /home2/mayaank.ashok/lewm_research || exit 1

export SCRATCH=${SCRATCH:-/ssd_scratch/mayaank.ashok/lewm_data}
export REACHER_H5_PATH=${REACHER_H5_PATH:-$SCRATCH/datasets/reacher.h5}
export REACHER_CKPT_DIR=${REACHER_CKPT_DIR:-/home2/mayaank.ashok/lewm_research/data/checkpoints/models--quentinll--lewm-reacher}
export STABLEWM_HOME=$SCRATCH
export MUJOCO_GL=egl OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1
export PY=/home2/mayaank.ashok/.venv/bin/python
SEEDS=(${SEEDS:-42 43 44 45 46})
N_GPU=${N_GPU:-4}
L=outputs/reacher/logs/baseline_check
mkdir -p "$L" "$SCRATCH/datasets/dmc" "$SCRATCH/checkpoints"
[ -e "$SCRATCH/datasets/dmc/reacher_random.h5" ] || ln -s "$REACHER_H5_PATH" "$SCRATCH/datasets/dmc/reacher_random.h5"
[ -e "$SCRATCH/checkpoints/lewm-reacher" ] || ln -s "$REACHER_CKPT_DIR" "$SCRATCH/checkpoints/lewm-reacher"

ref() {
  local seed=$1
  local logf="$L/paper_cem10_s${seed}.log"
  if grep -q "success_rate" "$logf" 2>/dev/null; then
    echo "[seed $seed] cached: $(grep -o "'success_rate': [0-9.]*" "$logf" | tail -1)"
    return
  fi
  echo "[$(date +%H:%M:%S)] paper protocol seed $seed -> $logf"
  "$PY" eval.py --config-name=reacher policy=lewm-reacher +cache_dir=null seed="$seed" \
    eval.num_eval=50 solver.n_steps=10 > "$logf" 2>&1
  grep -o "'success_rate': [0-9.]*" "$logf" | tail -1
}

echo "[reacher baseline] armed $(date) host=$(hostname) seeds=${SEEDS[*]}"
for ((gpu = 0; gpu < N_GPU; gpu++)); do
  (
    # Existing jobs have priority. Start this queue when its physical GPU becomes idle.
    while nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader | grep -q '[0-9]'; do sleep 30; done
    export CUDA_VISIBLE_DEVICES=$gpu MUJOCO_EGL_DEVICE_ID=$gpu
    for ((i = gpu; i < ${#SEEDS[@]}; i += N_GPU)); do ref "${SEEDS[$i]}"; done
  ) > "$L/slot_gpu${gpu}.log" 2>&1 &
done
wait

"$PY" - "$L" <<'PY'
import glob, re, statistics, sys
rows = []
for path in sorted(glob.glob(f"{sys.argv[1]}/paper_cem10_s*.log")):
    text = open(path).read()
    seed = int(re.search(r"_s(\d+)\.log$", path).group(1))
    matches = re.findall(r"'success_rate': ([0-9.]+)", text)
    if matches:
        rows.append((seed, float(matches[-1])))
print("paper-protocol results:", rows)
if rows:
    vals = [x[1] for x in rows]
    print(f"mean={statistics.mean(vals):.1f} sd={statistics.stdev(vals):.1f}" if len(vals) > 1 else f"mean={vals[0]:.1f}")
PY
echo "[reacher baseline] finished $(date)"
