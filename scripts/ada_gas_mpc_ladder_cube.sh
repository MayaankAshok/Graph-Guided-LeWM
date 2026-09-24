#!/usr/bin/env bash
# Cube version of the paper's Table 2 ablation (tab:ablation, Push-T rows L2 and (a)-(h)),
# task200u, 5 seeds x 4 protocols. Same mechanisms as ada_gas_mpc_ladder_full.sh /
# ada_gas_mpc_critic_noswitch.sh, but with Cube's calibrated lookahead / H_TD from
# outputs/cube/calib.json instead of Push-T's 13.7 / 8. Rows L2 and (g) use exactly the
# override strings of ada_gas_mpc_cube.sh's method_spec (l2 / B), so already-archived runs
# are reused. Requires the cube train stage (TDR, graph, critic per seed) to be done.
# Resumable: skips any (config, protocol, seed) already archived.
#   nohup srun --jobid=<JOB> --overlap bash scripts/ada_gas_mpc_ladder_cube.sh \
#     > outputs/cube/logs/ada_ladder_cube.log 2>&1 &
cd /home2/mayaank.ashok/lewm_research || exit 1
export GAS_MPC_ENV=cube
export SCRATCH=${SCRATCH:-/ssd_scratch/mayaank.ashok/lewm_data}
export CUBE_H5_PATH=${CUBE_H5_PATH:-$SCRATCH/datasets/ogbench/cube_single_expert.h5}
export CUBE_CKPT_DIR=${CUBE_CKPT_DIR:-/home2/mayaank.ashok/lewm_research/data/checkpoints/models--quentinll--lewm-cube}
export MUJOCO_GL=egl OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1
export PY=/home2/mayaank.ashok/.venv/bin/python
export N=${N:-50}
SEEDS=(${SEEDS:-0 1 2 3 4})
PROTOS=(${PROTOS:-same25 same50 same100 cross})
N_GPU=${N_GPU:-3}
SLOTS_PER_GPU=${SLOTS_PER_GPU:-1}   # MuJoCo EGL: 1 per GPU, as in the Cube/Reacher drivers
TE=${TE:-0.9}
OUT=outputs/cube
L=$OUT/logs
mkdir -p "$L"

[ -f "$OUT/calib.json" ] || { echo "[ladder_cube] $OUT/calib.json missing -- run ada_gas_mpc_cube.sh train first"; exit 1; }
HTD=$($PY -c "import json; c=json.load(open('$OUT/calib.json')); assert c.get('scope')=='training_only'; print(c['h_td'])")
LA=$($PY -c "import json; c=json.load(open('$OUT/calib.json')); assert c.get('scope')=='training_only'; print(c['lookahead'])")
$PY -c "import ogbench" 2>/dev/null || { echo "[ladder_cube] venv has no ogbench (see ada_gas_mpc_cube.sh stage_eval)"; exit 1; }
echo "[ladder_cube] armed $(date) host=$(hostname) h_td=$HTD lookahead=$LA seeds=${SEEDS[*]} protos=${PROTOS[*]} slots=$((N_GPU * SLOTS_PER_GPU))"

G="+mpc.lookahead=$LA +mpc.h_td=$HTD +mpc.te=$TE"
SW="+mpc.final_thresh=$LA +mpc.final_metric=l2"
ET="+mpc.critic_beta=1 +mpc.critic_cost=et"
# row | method | overrides   (row labels follow tab:ablation)
ROWS=(
  "l2|l2|"
  "a_tdr|tdr|"
  "b_subgoal|subgoal_tdr|$G"
  "c_switch|subgoal_tdr|$G $SW"
  "d_l2_et|l2|$ET"
  "e_noswitch_et|subgoal_tdr|$G $ET"
  "f_logv|subgoal_tdr|$G $SW +mpc.critic_beta=1"
  "g_ours|subgoal_tdr|$G $SW $ET"
  "h_l2scored|subgoal|$G $SW $ET"
)

JOBS=()
for proto in "${PROTOS[@]}"; do
  for s in "${SEEDS[@]}"; do
    for r in "${ROWS[@]}"; do
      JOBS+=("$s|$proto|$r")
    done
  done
done
echo "[ladder_cube] ${#JOBS[@]} jobs queued (already-archived combos skip near-instantly)"

NSLOT=$((N_GPU * SLOTS_PER_GPU))
run_slot() {
  local slot=$1
  local gpu=$((slot / SLOTS_PER_GPU))
  echo "[ladder_cube slot$slot gpu$gpu] starting $(date)"
  local i
  for ((i = slot; i < ${#JOBS[@]}; i += NSLOT)); do
    IFS='|' read -r s proto row method over <<< "${JOBS[$i]}"
    echo "[ladder_cube slot$slot] row=$row"
    # shellcheck disable=SC2086
    CUDA_VISIBLE_DEVICES=$gpu MUJOCO_EGL_DEVICE_ID=$gpu SEED=$s bash scripts/gas_mpc_run.sh "[$((i+1))/${#JOBS[@]}]" $method $proto $over
  done
  echo "[ladder_cube slot$slot gpu$gpu] finished $(date)"
}

for ((k = 0; k < NSLOT; k++)); do
  run_slot $k > "$L/ada_ladder_cube_slot$k.log" 2>&1 &
done
wait
echo "[ladder_cube] all slots finished $(date)"
