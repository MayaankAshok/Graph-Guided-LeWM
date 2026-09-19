#!/usr/bin/env bash
# Re-run everything that used the wrong (seed-0-for-every-CEM-seed) TDR/graph/critic on the
# fixed 200-task set, now that gas_mpc_run.sh auto-matches +mpc.graph_seed/+mpc.critic to the
# CEM seed. The old subgoal_tdr*/critic task200 results were deleted first (l2 needs no
# trained model, so its existing results stayed and only its gaps are filled here).
#
# GPU 2 and 3: seeds 0/1/3, whose TDR+critic are already trained -- start immediately.
# GPU 0 and 1: seeds 2/4, whose critics are still training (ada_seed_variance_training.sh) --
#   wait for that script's own "critic seed=X done" line before starting.
#   nohup srun --jobid=2697130 --overlap bash scripts/ada_gas_mpc_reseed.sh > outputs/pusht/logs/ada_reseed_driver.log 2>&1 &
cd /home2/mayaank.ashok/lewm_research || exit 1
export PUSHT_H5_PATH=/ssd_scratch/mayaank.ashok/lewm_data/datasets/pusht_expert_train.h5
export MUJOCO_GL=egl OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PY=/home2/mayaank.ashok/.venv/bin/python
export N=50
mkdir -p outputs/pusht/logs
echo "[reseed] armed $(date)"
LA="+mpc.lookahead=13.7"
T=""

subgoal_family() {  # all subgoal_tdr(+critic) work for one seed
  local s=$1
  echo "SEED=$s subgoal_tdr cross $LA $T"
  echo "SEED=$s subgoal_tdr same25 $LA $T"
  echo "SEED=$s subgoal_tdr same50 $LA $T"
  echo "SEED=$s subgoal_tdr cross $LA $T +mpc.critic_beta=1"
  echo "SEED=$s subgoal_tdr cross $LA $T +mpc.critic_beta=2"
  echo "SEED=$s subgoal_tdr same25 $LA $T +mpc.critic_beta=2"
  echo "SEED=$s subgoal_tdr same50 $LA $T +mpc.critic_beta=2"
  echo "SEED=$s subgoal_tdr same100 $LA $T +mpc.critic_beta=2"
  echo "SEED=$s subgoal_tdr cross $LA $T +mpc.critic_beta=4"
  echo "SEED=$s subgoal_tdr cross $LA $T +mpc.critic_beta=1 +mpc.retrieval=true"
}

# ---- l2 gaps (seed-independent, safe to run anytime) ----
mapfile -t L2GAPS < <(
  echo "SEED=3 l2 cross $T"
  echo "SEED=4 l2 cross $T"
  for s in 0 1 2 3 4; do echo "SEED=$s l2 same50 $T"; done
  for s in 0 1 2 3 4; do echo "SEED=$s l2 same100 $T"; done
)

mapfile -t Q2 < <(subgoal_family 0; subgoal_family 1)
mapfile -t Q3 < <(subgoal_family 3; printf '%s\n' "${L2GAPS[@]}")
mapfile -t Q0 < <(subgoal_family 2)
mapfile -t Q1 < <(subgoal_family 4)

run_line() {
  local pre=()
  while [[ $# -gt 0 && $1 == *=* && $1 != +* ]]; do pre+=("$1"); shift; done
  env "${pre[@]}" bash scripts/gas_mpc_run.sh "$@"
}
wait_for() { until grep -q "^\[seedvar\] critic seed=$1 done" outputs/pusht/logs/ada_seedvar_driver.log 2>/dev/null; do sleep 60; done; }
run_queue() {
  local gpu=$1 sentinel_seed=$2; shift 2
  [ -n "$sentinel_seed" ] && wait_for "$sentinel_seed"
  echo "[reseed q$gpu] starting $(date) (${#@} lines)"
  for line in "$@"; do
    # shellcheck disable=SC2086
    CUDA_VISIBLE_DEVICES=$gpu run_line $line
  done
}
L=outputs/pusht/logs
run_queue 2 "" "${Q2[@]}" > $L/reseed_q2.log 2>&1 &
run_queue 3 "" "${Q3[@]}" > $L/reseed_q3.log 2>&1 &
run_queue 0 2 "${Q0[@]}" > $L/reseed_q0.log 2>&1 &
run_queue 1 4 "${Q1[@]}" > $L/reseed_q1.log 2>&1 &
wait
echo "[reseed] all queues finished $(date)"
