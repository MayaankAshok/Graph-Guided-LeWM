#!/usr/bin/env bash
# Predictor-verified retrieval on Push-T: same OUR+retr config as
# ada_pusht_retrieval_full.sh (phase-8 config B: subgoal_tdr, lookahead/final_thresh=13.7,
# final_metric=l2, critic_beta=1, critic_cost=et), plus +mpc.retrieval_verify=true (score the
# retrieved block against a zero-action candidate with this run's own model.get_cost --
# predictor rollout + configured criterion -- and fall back to zero-mean when retrieval
# loses). Mirrors ada_reacher_retrieval_verify_full.sh / ada_cube_retrieval_verify_check.sh.
# Motivated by docs/gas-mpc/main.tex sec:retrieval-action-diag: unlike Reacher, Push-T's
# retrieved action is a GOOD match to the truth (r=0.90, beats random on 49/50 tasks) -- the
# log-P diagnostic says the failure is not a bad retrieval, it's an unnecessary one that adds
# variance without benefit. If that diagnosis is right, verification (which only rejects
# retrievals that score worse than zero-mean under the run's own predicted cost, not by
# ground-truth distance) should keep almost everything and change little; a real recovery
# would suggest the earlier diagnosis was incomplete.
# All 5 seeds, all 4 protocols, reverse order (cross first, same25 last).
# This node (gnode006) has 3x1080Ti, 2 workers/GPU = 6 slots (not the 4-GPU/8-slot layout used
# by the Reacher/Cube variants) -- check nvidia-smi -L before reusing this on a different node.
# Resumable: gas_mpc_eval.py skips any (seed, protocol) whose output already exists.
set -u
cd /home2/mayaank.ashok/lewm_research || exit 1

export GAS_MPC_ENV=pusht
export PUSHT_H5_PATH=/ssd_scratch/mayaank.ashok/lewm_data/datasets/pusht_expert_train.h5
export PY=/home2/mayaank.ashok/.venv/bin/python
export MUJOCO_GL=egl OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export N=50

L=outputs/pusht/logs/retrieval_verify_full_20260921
mkdir -p "$L"

OVERRIDES="+mpc.lookahead=13.7 +mpc.final_thresh=13.7 +mpc.final_metric=l2 +mpc.critic_beta=1 +mpc.critic_cost=et +mpc.retrieval=true +mpc.retrieval_verify=true"

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
  for ((i = slot; i < ${#JOBS[@]}; i += 6)); do
    IFS='|' read -r seed proto <<< "${JOBS[$i]}"
    echo "[$(date)] slot=$slot gpu=$gpu seed=$seed proto=$proto"
    # shellcheck disable=SC2086
    CUDA_VISIBLE_DEVICES=$gpu SEED=$seed \
      bash scripts/gas_mpc_run.sh "[$((i + 1))/${#JOBS[@]}]" subgoal_tdr "$proto" $OVERRIDES
  done
}

echo "[pusht-retrieval-verify-full] start $(date), ${#JOBS[@]} runs, protocol order: cross,same100,same50,same25"
for slot in 0 1 2 3 4 5; do
  gpu=$((slot / 2))
  run_slot "$slot" "$gpu" > "$L/slot${slot}.log" 2>&1 &
done
wait
echo "[pusht-retrieval-verify-full] finished $(date)"
