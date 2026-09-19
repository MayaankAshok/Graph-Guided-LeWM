#!/usr/bin/env bash
# Reacher GAS-MPC pipeline on Ada, end to end and resumable: every stage skips work whose
# output already exists, so re-running the script after a crash / new allocation only does
# what is missing. Stages (STAGES="data encode train eval" by default):
#   data    stage /share1's reacher.tar.zst onto the node's /ssd_scratch and extract reacher.h5
#   encode  frozen LeWM encoder over every frame -> outputs/reacher/cache_train.npz
#           (one GPU; the cache is seed-independent and shared by every later stage)
#   train   per seed, ONE process per GPU (SEEDS round-robin over N_GPU queues):
#             TDR (gas_mpc_prepare.py tdr) -> calibration (seed 0 only: H_TD = TDR distance
#             of HTD_STEPS env steps, LOOKAHEAD = of 25 env steps, written to calib.json and
#             shared by every seed so all seeds carry one method tag) -> graph
#             (gas_mpc_prepare.py graph) -> viability critic (viability_train.py --env reacher)
#   eval    SEEDS x METHODS x PROTOS, SLOTS_PER_GPU processes per GPU (default 1 -- 2 OOMs on Reacher):
#             l2   LeWM's own terminal L2 (baseline)
#             old  subgoal_tdr + critic, -log V, std composition, beta=1  (the Push-T phase-3 config;
#                  not in the default METHODS since 2026-09-17)
#             B    subgoal_tdr + final-goal switch at one lookahead + z-space L2 final phase
#                  + budget-capped expected-hitting-time critic term, beta=1  (Push-T phase-8 winner)
#           gas_mpc_run.sh / gas_mpc_eval.py skip any (method, protocol, seed) whose result exists.
#           The stage first checks that the live env renders the dataset's frames (render_check):
#           the venv must have mujoco==3.5.0 dm_control==1.0.37 -- see the function's comment.
# Launch from the LOGIN node inside an existing allocation (mem=20G, 3 GPUs):
#   nohup srun --jobid=<JOB> --overlap bash scripts/ada_gas_mpc_reacher.sh > outputs/reacher/logs/driver.log 2>&1 &
# Knobs (env vars): SEEDS="0 1 2 3 4" N_GPU=3 N=50 STAGES=... SMOKE=1 (tiny everything under
# outputs/reacher/smoke/, for a plumbing check before the real run).
# Every eval result is one file under outputs/reacher/eval/; to redo the whole sweep,
# delete that directory's contents (the fixed task sets live in pairs/, keep those) and rerun.
set -u
cd /home2/mayaank.ashok/lewm_research || exit 1
export GAS_MPC_ENV=reacher
export REACHER_H5_PATH=${REACHER_H5_PATH:-/ssd_scratch/mayaank.ashok/lewm_data/datasets/reacher.h5}
export MUJOCO_GL=egl OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1
export PY=/home2/mayaank.ashok/.venv/bin/python
SEEDS=(${SEEDS:-0 1 2 3 4})
N_GPU=${N_GPU:-3}
SLOTS_PER_GPU=${SLOTS_PER_GPU:-1}   # eval processes per GPU. 2 killed 7/60 runs on 2026-09-17 -- that was every
                                    # slot's EGL context piling onto GPU 0 (see MUJOCO_EGL_DEVICE_ID below), not
                                    # the model; with the fix 2 may fit (~2 GB EGL + <2.5 GB CUDA each), untested
STAGES=${STAGES:-data encode train eval}
PROTOS=(${PROTOS:-same25 same50 same100 cross})
METHODS=(${METHODS:-l2 B})               # `old` (the Push-T phase-3 config) is opt-in: METHODS="l2 old B"
SMOKE=${SMOKE:-0}

# ---- hyperparameters (see the "Reacher" paragraph in docs/gas-mpc/main.tex) ----
TDR_STEPS=${TDR_STEPS:-50000}      # Push-T value; expectile 0.99, 32-d, batch 1024
TE=${TE:-0.9}                      # TE filter, Push-T value
HTD_STEPS=${HTD_STEPS:-12}         # H_TD expressed in env steps (Push-T's 8 TDR units ~ 12 steps)
CRITIC_STEPS=${CRITIC_STEPS:-60000}
XNEG_MIN_DIST=${XNEG_MIN_DIST:-2.5}   # rad, joint-space L2: the data never moves > 2.35 rad (p99) in
                                      # 50 steps, so cross-episode pairs farther apart than this
                                      # are safe "unreachable within h_max" negatives
export N=${N:-50}
TDR_EXTRA=""
if [ "$SMOKE" = 1 ]; then
  export GAS_MPC_OUT=outputs/reacher/smoke
  export GAS_MPC_MAX_EPISODES=40        # encode only the first 40 episodes
  TDR_STEPS=300; CRITIC_STEPS=300; export N=2; TDR_EXTRA="--eval-every 100 --heldout-frac 0.1"
  SEEDS=(${SEEDS[@]:0:1}); PROTOS=(same25 cross)
fi
OUT=${GAS_MPC_OUT:-outputs/reacher}
L=$OUT/logs
mkdir -p "$L"
echo "[reacher] armed $(date) stages=[$STAGES] seeds=${SEEDS[*]} methods=${METHODS[*]} protos=${PROTOS[*]} out=$OUT smoke=$SMOKE"

has_stage() { [[ " $STAGES " == *" $1 "* ]]; }

# ---------------------------------------------------------------- data
stage_data() {
  if [ -f "$REACHER_H5_PATH" ]; then echo "[data] $REACHER_H5_PATH present"; return; fi
  local D; D=$(dirname "$(dirname "$REACHER_H5_PATH")")
  mkdir -p "$D/datasets"
  echo "[data] staging reacher.tar.zst -> $D $(date)"
  [ -f "$D/reacher.tar.zst" ] || rsync -a ada:/share1/mayaank.ashok/lewm_data/reacher.tar.zst "$D/"
  tar --zstd -xf "$D/reacher.tar.zst" -C "$D/datasets"
  ls -la "$D/datasets"
  echo "[data] done $(date)"
}

# ---------------------------------------------------------------- encode
stage_encode() {
  if [ -f "$OUT/cache_train.npz" ]; then echo "[encode] $OUT/cache_train.npz present"; return; fi
  echo "[encode] start $(date)"
  CUDA_VISIBLE_DEVICES=0 $PY scripts/gas_mpc_prepare.py encode >> "$L/encode.log" 2>&1
  [ -f "$OUT/cache_train.npz" ] || { echo "[encode] FAILED (see $L/encode.log)"; exit 1; }
  echo "[encode] done $(date)"
}

# ---------------------------------------------------------------- train
calibrate() {
  # H_TD and LOOKAHEAD in TDR units from seed 0's TDR: piecewise-linear interpolation of the
  # median TDR distance by step gap (the TDR checkpoint's own diagnostic table).
  $PY - "$OUT" "$HTD_STEPS" <<'PYEOF'
import json, sys, torch
out, htd_steps = sys.argv[1], float(sys.argv[2])
ck = torch.load(f"{out}/tdr_full_s0.pt", map_location="cpu", weights_only=False)
assert ck["cfg"].get("diagnostic_scope") == "training_only"
med = {int(k): float(v) for k, v in ck["history"][-1]["median_by_gap"].items()}
gaps = sorted(med)
import numpy as np
h_td = float(np.interp(htd_steps, gaps, [med[g] for g in gaps]))
la = float(np.interp(25, gaps, [med[g] for g in gaps]))
calib = dict(h_td=round(h_td, 2), lookahead=round(la, 2), htd_steps=htd_steps, median_by_gap=med, scope="training_only")
json.dump(calib, open(f"{out}/calib.json", "w"), indent=1)
print(calib)
PYEOF
}

critic_done() {   # critic.pt exists and was saved at the final step
  [ -f "$OUT/critic_s${1}_tdr_holdout/critic.pt" ] && $PY -c "
import torch, sys; ck = torch.load('$OUT/critic_s${1}_tdr_holdout/critic.pt', map_location='cpu', weights_only=False)
sys.exit(0 if ck.get('step', 0) >= $CRITIC_STEPS and ck['args'].get('training_only') else 1)" 2>/dev/null
}

train_seed() {
  local s=$1 gpu=$2
  echo "[train s$s gpu$gpu] tdr $(date)"
  # shellcheck disable=SC2086
  CUDA_VISIBLE_DEVICES=$gpu $PY scripts/gas_mpc_prepare.py tdr --seed "$s" --tdr-steps "$TDR_STEPS" $TDR_EXTRA \
    >> "$L/tdr_s$s.log" 2>&1 || { echo "[train s$s] TDR FAILED"; return 1; }
  if [ "$s" = "${SEEDS[0]}" ] && [ ! -f "$OUT/calib.json" ]; then
    calibrate >> "$L/calib.log" 2>&1 || { echo "[train s$s] calibration FAILED"; return 1; }
  fi
  until [ -f "$OUT/calib.json" ]; do sleep 30; done      # other seeds wait for seed 0's table
  local HTD; HTD=$($PY -c "import json; c=json.load(open('$OUT/calib.json')); assert c.get('scope')=='training_only'; print(c['h_td'])")
  echo "[train s$s gpu$gpu] graph h_td=$HTD $(date)"
  CUDA_VISIBLE_DEVICES=$gpu $PY scripts/gas_mpc_prepare.py graph --seed "$s" --h-td "$HTD" --te "$TE" \
    >> "$L/graph_s$s.log" 2>&1 || { echo "[train s$s] graph FAILED"; return 1; }
  if critic_done "$s"; then echo "[train s$s] critic present"; else
    echo "[train s$s gpu$gpu] critic $(date)"
    CUDA_VISIBLE_DEVICES=$gpu $PY scripts/viability_train.py --env reacher --cache "$OUT/cache_train.npz" \
      --out "$OUT/critic_s${s}_tdr_holdout" --seed "$s" --steps "$CRITIC_STEPS" --label-source tdr --tdr "$OUT/tdr_full_s$s.pt" --exclude-tdr-holdout --xneg-min-dist "$XNEG_MIN_DIST" \
      >> "$L/critic_s$s.log" 2>&1 || { echo "[train s$s] critic FAILED"; return 1; }
  fi
  echo "[train s$s gpu$gpu] done $(date)"
}

stage_train() {
  # one process per GPU: seed i goes to queue i % N_GPU, queues run their seeds sequentially
  local q
  for ((q = 0; q < N_GPU; q++)); do
    (
      for ((i = q; i < ${#SEEDS[@]}; i += N_GPU)); do train_seed "${SEEDS[$i]}" "$q"; done
    ) > "$L/train_q$q.log" 2>&1 &
  done
  wait
  for s in "${SEEDS[@]}"; do
    critic_done "$s" && $PY -c "
import sys, json; sys.path.insert(0, 'scripts')
from gas_mpc_prepare import graph_path
c = json.load(open('$OUT/calib.json')); sys.exit(0 if graph_path($s, c['h_td'], $TE).exists() else 1)" \
      || { echo "[train] seed $s incomplete -- see $L/train_q*.log"; exit 1; }
  done
  echo "[train] all seeds done $(date)"
}

# ---------------------------------------------------------------- eval
render_check() {
  # The live env must draw the frames the encoder was trained on. reacher.h5 was rendered with
  # mujoco 3.5/3.6 (Mar 2026); mujoco 3.12.0 (what the venv had until 2026-09-17) draws the floor
  # WITHOUT its checker texture. Same state, same arm pixels, but the encoder's latent moves by
  # about a 25-step gap (15 vs 17 units), which cost a full sweep: L2 same25 came out 65% instead
  # of the paper's 86%. Pin: pip install mujoco==3.5.0 dm_control==1.0.37 (only dm_control
  # constrains mujoco). This compares three dataset frames with a live render of their states.
  $PY - "$REACHER_H5_PATH" <<'PYEOF' || { echo "[render] live render does not match the dataset -- fix the venv (mujoco==3.5.0 dm_control==1.0.37)"; return 1; }
import sys, h5py, hdf5plugin, numpy as np, mujoco
import stable_worldmodel as swm
f = h5py.File(sys.argv[1], "r")
rows = [0, 100000, 1000000]
world = swm.World(env_name="swm/ReacherDMControl-v0", num_envs=1, max_episode_steps=10, image_shape=(224, 224), task="qpos_match")
world.reset(seed=None)
w = world.envs.envs[0]
while not hasattr(w, "_get_pixels"):          # the AddPixelsWrapper: exactly what the policy sees
    w = w.env
diffs = []
for r in rows:
    w.unwrapped.set_state(np.array(f["qpos"][r]), np.array(f["qvel"][r]))
    live = w._get_pixels()[0]["pixels"].astype(float)
    diffs.append(float(np.abs(live - f["pixels"][r].astype(float)).mean()))
world.close()
print(f"[render] mujoco {mujoco.__version__}: dataset-vs-live mean abs pixel diff {np.round(diffs, 2).tolist()} (must be < 4; 3.12.0 gave ~18)")
sys.exit(0 if max(diffs) < 4 else 1)
PYEOF
}

method_spec() {   # method name -> "<gas_mpc_eval method> <hydra overrides>" (graph_seed / critic path are added by gas_mpc_run.sh)
  local HTD LA
  HTD=$($PY -c "import json; c=json.load(open('$OUT/calib.json')); assert c.get('scope')=='training_only'; print(c['h_td'])")
  LA=$($PY -c "import json; c=json.load(open('$OUT/calib.json')); assert c.get('scope')=='training_only'; print(c['lookahead'])")
  case $1 in
    l2)  echo "l2" ;;
    old) echo "subgoal_tdr +mpc.lookahead=$LA +mpc.h_td=$HTD +mpc.te=$TE +mpc.critic_beta=1" ;;
    B)   echo "subgoal_tdr +mpc.lookahead=$LA +mpc.h_td=$HTD +mpc.te=$TE +mpc.final_thresh=$LA +mpc.final_metric=l2 +mpc.critic_beta=1 +mpc.critic_cost=et" ;;
    *)   echo "unknown method $1" >&2; return 1 ;;
  esac
}

stage_eval() {
  render_check > "$L/render_check.log" 2>&1; local rc=$?
  grep '^\[render\]' "$L/render_check.log"
  [ $rc = 0 ] || { echo "[eval] render check FAILED (see $L/render_check.log)"; exit 1; }
  # fixed task sets once, before six processes could race to create them
  $PY scripts/gas_mpc_make_tasks.py --tasks 200 >> "$L/make_tasks.log" 2>&1 || { echo "[eval] make_tasks FAILED"; exit 1; }
  local JOBS=() proto s m spec
  for proto in "${PROTOS[@]}"; do for s in "${SEEDS[@]}"; do for m in "${METHODS[@]}"; do
    spec=$(method_spec "$m") || exit 1
    JOBS+=("$s|$proto|${spec%% *}|${spec#* }")       # seed | protocol | method | overrides ('' for l2)
  done; done; done
  local NSLOT=$((N_GPU * SLOTS_PER_GPU)) k
  echo "[eval] ${#JOBS[@]} jobs over $NSLOT slots $(date)"
  for ((k = 0; k < NSLOT; k++)); do
    (
      gpu=$((k / SLOTS_PER_GPU))
      for ((i = k; i < ${#JOBS[@]}; i += NSLOT)); do
        IFS='|' read -r s proto meth over <<< "${JOBS[$i]}"
        [ "$over" = "$meth" ] && over=""                   # l2 has no overrides
        # MUJOCO_EGL_DEVICE_ID: dm_control's EGL renderer enumerates the PHYSICAL GPUs and defaults
        # to device 0 regardless of CUDA_VISIBLE_DEVICES, so without it every slot's ~2 GB render
        # context lands on GPU 0 (seen 2026-09-17: 3 x 2 GB "G" contexts on GPU 0, "C" work on the
        # right GPUs; this is what OOMed 2 slots per GPU earlier, not the model). Same index as
        # CUDA_VISIBLE_DEVICES: identical GPUs, both PCI order on gnode003.
        # shellcheck disable=SC2086
        CUDA_VISIBLE_DEVICES=$gpu MUJOCO_EGL_DEVICE_ID=$gpu SEED=$s bash scripts/gas_mpc_run.sh "$meth" "$proto" $over
      done
    ) > "$L/eval_slot$k.log" 2>&1 &
  done
  wait
  echo "[eval] all slots finished $(date)"
  $PY scripts/gas_mpc_report.py >> "$L/report.log" 2>&1 && echo "[eval] report written (docs/gas-mpc/results_table_reacher.tex)"
}

has_stage data && stage_data
has_stage encode && stage_encode
has_stage train && stage_train
has_stage eval && stage_eval
echo "[reacher] finished $(date)"
