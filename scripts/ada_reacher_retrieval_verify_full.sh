#!/usr/bin/env bash
# Predictor-verified retrieval on Reacher: same OUR+retr config as
# ada_reacher_retrieval_full.sh, plus +mpc.retrieval_verify=true (score the retrieved block
# against a zero-action candidate with this run's own model.get_cost -- predictor rollout +
# configured criterion -- and fall back to zero-mean when retrieval loses). Motivated by
# docs/gas-mpc/main.tex's log-P diagnostic: on Reacher, retrieval's iteration-1 placement was
# WORSE than zero-mean on all 50/50 held-out same25 tasks, so this should reject most/all of
# them and (if the diagnosis is right) recover most of OUR+retr's lost ground vs.\ plain OUR.
# All 5 seeds, all 4 protocols, reverse order (cross first, same25 last), 2 workers/GPU x 4 GPUs.
# Resumable: gas_mpc_eval.py skips any (seed, protocol) whose output already exists.
set -u
cd /home2/mayaank.ashok/lewm_research || exit 1

export GAS_MPC_ENV=reacher
export REACHER_H5_PATH=/ssd_scratch/mayaank.ashok/lewm_data/datasets/reacher.h5
export PY=/home2/mayaank.ashok/.venv/bin/python
export MUJOCO_GL=egl OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export N=50

OUT=outputs/reacher
L="$OUT/logs/retrieval_verify_full_20260920"
mkdir -p "$L"

HTD=$($PY -c "import json; c=json.load(open('$OUT/calib.json')); assert c.get('scope')=='training_only'; print(c['h_td'])")
LA=$($PY -c "import json; c=json.load(open('$OUT/calib.json')); assert c.get('scope')=='training_only'; print(c['lookahead'])")
echo "[reacher-retrieval-verify-full] calib h_td=$HTD lookahead=$LA"

OVERRIDES="+mpc.lookahead=$LA +mpc.h_td=$HTD +mpc.te=0.9 +mpc.final_thresh=$LA +mpc.final_metric=l2 +mpc.critic_beta=1 +mpc.critic_cost=et +mpc.retrieval=true +mpc.retrieval_verify=true"

JOBS=()
for proto in cross same100 same50 same25; do
  for seed in 0 1 2 3 4; do
    JOBS+=("$seed|$proto")
  done
done

run_slot() {
  local slot=$1
  local gpu=$2
  local i seed proto
  for ((i = slot; i < ${#JOBS[@]}; i += 8)); do
    IFS='|' read -r seed proto <<< "${JOBS[$i]}"
    echo "[$(date)] slot=$slot gpu=$gpu seed=$seed proto=$proto"
    # shellcheck disable=SC2086
    CUDA_VISIBLE_DEVICES=$gpu MUJOCO_EGL_DEVICE_ID=$gpu SEED=$seed \
      bash scripts/gas_mpc_run.sh "[$((i + 1))/${#JOBS[@]}]" subgoal_tdr "$proto" $OVERRIDES
  done
}

echo "[reacher-retrieval-verify-full] start $(date), ${#JOBS[@]} runs, protocol order: cross,same100,same50,same25"
for slot in 0 1 2 3 4 5 6 7; do
  gpu=$((slot / 2))
  run_slot "$slot" "$gpu" > "$L/slot${slot}.log" 2>&1 &
done
wait
echo "[reacher-retrieval-verify-full] finished $(date)"
