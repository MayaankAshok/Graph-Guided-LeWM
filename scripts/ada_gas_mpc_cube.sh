#!/usr/bin/env bash
# OGBench Cube (quentinll/lewm-cube, single-cube pick-and-place) GAS-MPC pipeline on Ada, end to
# end and resumable: every stage skips work whose output already exists, so re-running after a
# crash / new allocation only does what is missing. Stages (STAGES="data ckpt encode train eval" by default):
#   data    download cube_single_expert.tar.zst (46 GB) straight onto the compute node's /ssd_scratch
#           -- NOT via /share1 (no room there) -- and extract cube_single_expert.h5 next to it.
#           Direct curl from the node when it can reach huggingface.co; otherwise the bytes are
#           streamed through the login node (`ssh ada curl ... >> file`), still never written to
#           /share1. Both paths resume a partial file. The archive is deleted once the h5 is out
#           (KEEP_ARCHIVE=1 keeps it). Ends with an `inspect` of the h5's columns: the pipeline
#           needs pixels/action/ep_offset/ep_len plus the two state columns CubeMechanics reads
#           (privileged_block_0_pos, proprio_effector_pos) -- if the names differ, the log says
#           what IS there and the stage fails before any GPU time is spent.
#   ckpt    quentinll/lewm-cube's config.json + weights.pt (~72 MB) into /home2 (fits the quota)
#   encode  frozen LeWM encoder over every frame -> outputs/cube/cache_train.npz
#           (one GPU; seed-independent, shared by every later stage)
#   train   per seed, ONE process per GPU (SEEDS round-robin over N_GPU queues):
#             TDR (gas_mpc_prepare.py tdr) -> calibration (seed 0 only: H_TD = TDR distance of
#             HTD_STEPS env steps, LOOKAHEAD = of 25 env steps, XNEG_MIN_DIST = p99 of the block's
#             50-step displacement unless given; written to calib.json and shared by every seed)
#             -> graph (gas_mpc_prepare.py graph) -> viability critic (viability_train.py --env cube)
#   eval    SEEDS x METHODS x PROTOS live rollouts through
#           config/eval/cube.yaml's World + callables. Needs ogbench in the venv (pip on the
#           COMPUTE node, see stage_eval) and opens with gas_mpc_cube_render_check.py (live render
#           == dataset frames, passed 2026-09-18 with mujoco 3.5.0 + ogbench 1.2.1).
# Launch from the LOGIN node inside an existing allocation (mem=20G, N_GPU GPUs):
#   nohup srun --jobid=<JOB> --overlap bash scripts/ada_gas_mpc_cube.sh > outputs/cube/logs/driver.log 2>&1 &
# Knobs (env vars): SEEDS="0 1 2 3 4" N_GPU=3 N=50 STAGES=... SMOKE=1 (tiny everything under
# outputs/cube/smoke/, for a plumbing check before the real run; still needs the data).
set -u
cd /home2/mayaank.ashok/lewm_research || exit 1
export GAS_MPC_ENV=cube
export SCRATCH=${SCRATCH:-/ssd_scratch/mayaank.ashok/lewm_data}
export CUBE_H5_PATH=${CUBE_H5_PATH:-$SCRATCH/datasets/ogbench/cube_single_expert.h5}
export CUBE_CKPT_DIR=${CUBE_CKPT_DIR:-/home2/mayaank.ashok/lewm_research/data/checkpoints/models--quentinll--lewm-cube}
export MUJOCO_GL=egl OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1
export PY=/home2/mayaank.ashok/.venv/bin/python
HF_DATA=https://huggingface.co/datasets/quentinll/lewm-cube/resolve/main/cube_single_expert.tar.zst
HF_CKPT=https://huggingface.co/quentinll/lewm-cube/resolve/main
SEEDS=(${SEEDS:-0 1 2 3 4})
N_GPU=${N_GPU:-3}
SLOTS_PER_GPU=${SLOTS_PER_GPU:-1}
STAGES=${STAGES:-data ckpt encode train eval}
PROTOS=(${PROTOS:-same25 same50 same100 cross})
METHODS=(${METHODS:-l2 B})
SMOKE=${SMOKE:-0}
KEEP_ARCHIVE=${KEEP_ARCHIVE:-0}

# ---- hyperparameters: the Reacher/Push-T values; nothing here has been tuned on cube ----
TDR_STEPS=${TDR_STEPS:-50000}      # expectile 0.99, 32-d, batch 1024
TE=${TE:-0.9}                      # TE filter
HTD_STEPS=${HTD_STEPS:-12}         # H_TD in env steps (Push-T's 8 TDR units ~ 12 steps)
CRITIC_STEPS=${CRITIC_STEPS:-60000}
XNEG_MIN_DIST=${XNEG_MIN_DIST:-auto}  # metres, block-position L2 between cross-episode critic negatives;
                                      # auto = p99 of the block's 50-step displacement in the data
                                      # (Reacher's 2.5 rad was chosen the same way: just above its p99)
export N=${N:-50}
TDR_EXTRA=""
if [ "$SMOKE" = 1 ]; then
  export GAS_MPC_OUT=outputs/cube/smoke
  export GAS_MPC_MAX_EPISODES=40
  TDR_STEPS=300; CRITIC_STEPS=300; export N=2; TDR_EXTRA="--eval-every 100 --heldout-frac 0.1"
  SEEDS=(${SEEDS[@]:0:1}); PROTOS=(same25 cross)
fi
OUT=${GAS_MPC_OUT:-outputs/cube}
L=$OUT/logs
mkdir -p "$L"
echo "[cube] armed $(date) host=$(hostname) stages=[$STAGES] seeds=${SEEDS[*]} methods=${METHODS[*]} protos=${PROTOS[*]} out=$OUT smoke=$SMOKE"

has_stage() { [[ " $STAGES " == *" $1 "* ]]; }

# ---------------------------------------------------------------- download helpers
node_has_internet() { curl -sSIL --max-time 20 -o /dev/null "$1" 2>/dev/null; }

remote_size() {   # content-length of a URL after redirects, queried from wherever can reach it
  local url=$1 out
  out=$(curl -sSIL --max-time 30 "$url" 2>/dev/null) || out=$(ssh ada "curl -sSIL --max-time 30 '$url'")
  printf '%s' "$out" | tr -d '\r' | grep -i '^content-length:' | tail -1 | awk '{print $2}'
}

fetch() {         # fetch URL -> local file, resumable; direct from the node, else streamed through ada
  local url=$1 dst=$2 want have
  want=$(remote_size "$url")
  have=$([ -f "$dst" ] && stat -c %s "$dst" || echo 0)
  if [ -n "$want" ] && [ "$have" = "$want" ]; then echo "[fetch] $(basename "$dst") complete ($have bytes)"; return 0; fi
  mkdir -p "$(dirname "$dst")"
  if node_has_internet "$url"; then
    echo "[fetch] direct curl -> $dst (have $have / ${want:-?} bytes) $(date)"
    curl -L -C - --retry 10 --retry-delay 30 -o "$dst" "$url" 2>> "$L/fetch.log"
  else
    echo "[fetch] node cannot reach huggingface.co; streaming through the login node -> $dst (have $have / ${want:-?}) $(date)"
    # `ssh ada curl` writes to OUR stdout, so the bytes land on this node's /ssd_scratch and never
    # touch /share1. Range request from the current size makes it resumable.
    ssh ada "curl -sSL --retry 10 --retry-delay 30 -r ${have}- '$url'" >> "$dst"
  fi
  have=$(stat -c %s "$dst")
  if [ -n "$want" ] && [ "$have" != "$want" ]; then
    echo "[fetch] $(basename "$dst"): $have of $want bytes -- rerun to resume"; return 1
  fi
  echo "[fetch] $(basename "$dst") done ($have bytes) $(date)"
}

# ---------------------------------------------------------------- data
inspect_h5() {
  $PY - "$CUBE_H5_PATH" <<'PYEOF'
import sys, h5py, hdf5plugin, numpy as np
sys.path.insert(0, "scripts")
from common.envs import ENV_MECHANICS
m = ENV_MECHANICS["cube"]
with h5py.File(sys.argv[1], "r") as f:
    cols = {k: (f[k].shape, str(f[k].dtype)) for k in f.keys()}
    for k, v in sorted(cols.items()):
        print(f"[inspect] {k:40s} {v[0]} {v[1]}")
    need = ["pixels", "action", "ep_offset", "ep_len"]
    missing = [k for k in need if k not in f]
    ok_block = any(k in f for k in m.BLOCK_KEYS); ok_eff = any(k in f for k in m.EFFECTOR_KEYS)
    if missing or not ok_block or not ok_eff:
        print(f"[inspect] FAIL: missing {missing}; block col {m.BLOCK_KEYS} present={ok_block}; "
              f"effector col {m.EFFECTOR_KEYS} present={ok_eff}. Edit CubeMechanics.BLOCK_KEYS/EFFECTOR_KEYS "
              f"in scripts/common/envs.py to the names listed above.")
        sys.exit(1)
    ep_len = f["ep_len"][:]
    a = f["action"]
    print(f"[inspect] {len(ep_len)} episodes, {int(ep_len.sum())} frames, ep_len min/median/max "
          f"{ep_len.min()}/{int(np.median(ep_len))}/{ep_len.max()}, pixels {f['pixels'].shape[1:]}, "
          f"action dim {a.shape[1]} (CubeMechanics.action_dim={m.action_dim})")
    st = m.read_state_slice(f, 0, int(ep_len[0]))
    print(f"[inspect] state (block xyz | effector xyz) episode 0: first {st[0].round(3).tolist()} "
          f"last {st[-1].round(3).tolist()}")
    assert a.shape[1] == m.action_dim, "action dim mismatch"
PYEOF
}

stage_data() {
  if [ -f "$CUBE_H5_PATH" ]; then echo "[data] $CUBE_H5_PATH present"; inspect_h5 || exit 1; return; fi
  [ -d /ssd_scratch ] || { echo "[data] /ssd_scratch does not exist here -- this stage must run ON THE COMPUTE NODE (srun/adag), not the login node"; exit 1; }
  mkdir -p "$SCRATCH/datasets"
  echo "[data] free space on /ssd_scratch: $(df -h /ssd_scratch | tail -1 | awk '{print $4}') (archive 46 GB + the extracted h5; Push-T's h5 was 3.3x its .zst)"
  local ARC=$SCRATCH/cube_single_expert.tar.zst
  fetch "$HF_DATA" "$ARC" || exit 1
  echo "[data] archive layout:"; tar --zstd -tf "$ARC" | head -5
  echo "[data] extracting -> $SCRATCH/datasets $(date)"
  tar --zstd -xf "$ARC" -C "$SCRATCH/datasets" || { echo "[data] extraction FAILED"; exit 1; }
  if [ ! -f "$CUBE_H5_PATH" ]; then
    # the archive's top-level name has not always matched what the config expects (CLAUDE.md);
    # find the h5 and put the expected path in front of it instead of guessing
    local found; found=$(find "$SCRATCH/datasets" -name '*.h5' -o -name '*.hdf5' | head -1)
    [ -n "$found" ] || { echo "[data] no .h5 found under $SCRATCH/datasets after extraction"; ls -laR "$SCRATCH/datasets" | head -40; exit 1; }
    echo "[data] extracted h5 is $found, expected $CUBE_H5_PATH -- symlinking"
    mkdir -p "$(dirname "$CUBE_H5_PATH")" && ln -s "$found" "$CUBE_H5_PATH"
  fi
  ls -laL "$CUBE_H5_PATH"
  [ "$KEEP_ARCHIVE" = 1 ] || { rm -f "$ARC"; echo "[data] removed $ARC (KEEP_ARCHIVE=1 to keep)"; }
  echo "[data] done $(date)"
  inspect_h5 || exit 1
}

# ---------------------------------------------------------------- ckpt
stage_ckpt() {
  if [ -f "$CUBE_CKPT_DIR/weights.pt" ] && [ -f "$CUBE_CKPT_DIR/config.json" ]; then echo "[ckpt] $CUBE_CKPT_DIR present"; return; fi
  fetch "$HF_CKPT/config.json" "$CUBE_CKPT_DIR/config.json" || exit 1
  fetch "$HF_CKPT/weights.pt" "$CUBE_CKPT_DIR/weights.pt" || exit 1
  $PY -c "import json; c=json.load(open('$CUBE_CKPT_DIR/config.json')); print('[ckpt] action_encoder input_dim', c['action_encoder']['input_dim'], '(expect 25 = 5 actions x 5 dims)')"
}

# ---------------------------------------------------------------- encode
stage_encode() {
  if [ -f "$OUT/cache_train.npz" ]; then echo "[encode] $OUT/cache_train.npz present"; return; fi
  [ -f "$CUBE_H5_PATH" ] || { echo "[encode] $CUBE_H5_PATH missing -- run the data stage on the compute node first"; exit 1; }
  echo "[encode] start $(date)"
  CUDA_VISIBLE_DEVICES=0 $PY scripts/gas_mpc_prepare.py encode >> "$L/encode.log" 2>&1
  [ -f "$OUT/cache_train.npz" ] || { echo "[encode] FAILED (see $L/encode.log)"; exit 1; }
  echo "[encode] done $(date)"
}

# ---------------------------------------------------------------- train
calibrate() {
  # H_TD and LOOKAHEAD in TDR units from seed 0's TDR (piecewise-linear interpolation of the
  # median TDR distance by step gap), plus the cross-episode negative radius from the data.
  $PY - "$OUT" "$HTD_STEPS" "$XNEG_MIN_DIST" <<'PYEOF'
import json, sys, torch, numpy as np
out, htd_steps, xneg = sys.argv[1], float(sys.argv[2]), sys.argv[3]
ck = torch.load(f"{out}/tdr_full_s0.pt", map_location="cpu", weights_only=False)
assert ck["cfg"].get("diagnostic_scope") == "training_only"
med = {int(k): float(v) for k, v in ck["history"][-1]["median_by_gap"].items()}
gaps = sorted(med)
h_td = float(np.interp(htd_steps, gaps, [med[g] for g in gaps]))
la = float(np.interp(25, gaps, [med[g] for g in gaps]))
if xneg == "auto":
    d = np.load(f"{out}/cache_train.npz")
    st, off, ln = d["state"], d["ep_offset"], d["ep_len"]
    disp = []
    for o, n in zip(off, ln):
        if n > 50:
            b = st[o:o + n, :3]
            disp.append(np.linalg.norm(b[50:] - b[:-50], axis=1))
    disp = np.concatenate(disp)
    xneg_v = float(np.percentile(disp, 99))
    print(f"block 50-step displacement (m): p50 {np.percentile(disp, 50):.3f} p90 {np.percentile(disp, 90):.3f} "
          f"p99 {xneg_v:.3f} max {disp.max():.3f} -> xneg_min_dist={xneg_v:.3f}")
else:
    xneg_v = float(xneg)
calib = dict(h_td=round(h_td, 2), lookahead=round(la, 2), htd_steps=htd_steps, xneg_min_dist=round(xneg_v, 4), median_by_gap=med, scope="training_only")
json.dump(calib, open(f"{out}/calib.json", "w"), indent=1)
print(calib)
PYEOF
}

critic_done() {
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
  until [ -f "$OUT/calib.json" ]; do sleep 30; done
  local HTD XNEG
  HTD=$($PY -c "import json; c=json.load(open('$OUT/calib.json')); assert c.get('scope')=='training_only'; print(c['h_td'])")
  XNEG=$($PY -c "import json; c=json.load(open('$OUT/calib.json')); assert c.get('scope')=='training_only'; print(c['xneg_min_dist'])")
  echo "[train s$s gpu$gpu] graph h_td=$HTD $(date)"
  CUDA_VISIBLE_DEVICES=$gpu $PY scripts/gas_mpc_prepare.py graph --seed "$s" --h-td "$HTD" --te "$TE" \
    >> "$L/graph_s$s.log" 2>&1 || { echo "[train s$s] graph FAILED"; return 1; }
  if critic_done "$s"; then echo "[train s$s] critic present"; else
    echo "[train s$s gpu$gpu] critic xneg_min_dist=$XNEG $(date)"
    CUDA_VISIBLE_DEVICES=$gpu $PY scripts/viability_train.py --env cube --cache "$OUT/cache_train.npz" \
      --out "$OUT/critic_s${s}_tdr_holdout" --seed "$s" --steps "$CRITIC_STEPS" --label-source tdr --tdr "$OUT/tdr_full_s$s.pt" --exclude-tdr-holdout --xneg-min-dist "$XNEG" \
      >> "$L/critic_s$s.log" 2>&1 || { echo "[train s$s] critic FAILED"; return 1; }
  fi
  echo "[train s$s gpu$gpu] done $(date)"
}

stage_train() {
  [ -f "$OUT/cache_train.npz" ] || { echo "[train] $OUT/cache_train.npz missing -- run encode first"; exit 1; }
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

# ---------------------------------------------------------------- eval (opt-in, untested on cube)
method_spec() {
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
  # ogbench must be pip-installed ON THE COMPUTE NODE (the venv's /usr/bin/python3 is 3.10 there
  # but 3.6 on the login node, so pip does not run on ada): srun --overlap ... pip install --no-deps ogbench
  $PY -c "import ogbench" 2>/dev/null || { echo "[eval] the venv has no ogbench -- srun --jobid=<JOB> --overlap $PY -m pip install --no-deps ogbench"; exit 1; }
  # live render must reproduce the dataset frames (Reacher's mujoco lesson); 3 rows, < 4 mean abs diff
  $PY scripts/gas_mpc_cube_render_check.py > "$L/render_check.log" 2>&1; local rc=$?
  grep '^\[render\]' "$L/render_check.log"
  [ $rc = 0 ] || { echo "[eval] render check FAILED (see $L/render_check.log)"; exit 1; }
  $PY scripts/gas_mpc_make_tasks.py --tasks 200 >> "$L/make_tasks.log" 2>&1 || { echo "[eval] make_tasks FAILED (see $L/make_tasks.log)"; exit 1; }
  local JOBS=() proto s m spec
  for proto in "${PROTOS[@]}"; do for s in "${SEEDS[@]}"; do for m in "${METHODS[@]}"; do
    spec=$(method_spec "$m") || exit 1
    JOBS+=("$s|$proto|${spec%% *}|${spec#* }")
  done; done; done
  local NSLOT=$((N_GPU * SLOTS_PER_GPU)) k
  echo "[eval] ${#JOBS[@]} jobs over $NSLOT slots $(date)"
  for ((k = 0; k < NSLOT; k++)); do
    (
      gpu=$((k / SLOTS_PER_GPU))
      for ((i = k; i < ${#JOBS[@]}; i += NSLOT)); do
        IFS='|' read -r s proto meth over <<< "${JOBS[$i]}"
        [ "$over" = "$meth" ] && over=""
        # shellcheck disable=SC2086
        CUDA_VISIBLE_DEVICES=$gpu MUJOCO_EGL_DEVICE_ID=$gpu SEED=$s bash scripts/gas_mpc_run.sh "$meth" "$proto" $over
      done
    ) > "$L/eval_slot$k.log" 2>&1 &
  done
  wait
  echo "[eval] all slots finished $(date)"
  $PY scripts/gas_mpc_report.py >> "$L/report.log" 2>&1 && echo "[eval] report written (docs/gas-mpc/results_table_cube.tex)"
}

has_stage data && stage_data
has_stage ckpt && stage_ckpt
has_stage encode && stage_encode
has_stage train && stage_train
has_stage eval && stage_eval
echo "[cube] finished $(date)"
