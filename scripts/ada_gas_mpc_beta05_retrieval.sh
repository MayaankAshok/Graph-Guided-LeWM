#!/usr/bin/env bash
# Two-phase fill on the fixed 200-task set (tasks200, first 50), all 5 seeds:
#   Phase A: subgoal_tdr + critic beta=0.5 on same100 and cross (currently missing;
#            same25/same50 already exist).
#   Phase B (starts once Phase A's queues all finish): subgoal_tdr + retrieval + critic
#            beta=1 on same25/same50/same100 -- cross already exists (ret_crit1, 34.4+-5.6).
# Round-robined across all 4 GPUs (all idle right now).
#   nohup srun --jobid=2697130 --overlap bash scripts/ada_gas_mpc_beta05_retrieval.sh > outputs/pusht/logs/ada_beta05ret_driver.log 2>&1 &
cd /home2/mayaank.ashok/lewm_research || exit 1
export PUSHT_H5_PATH=/ssd_scratch/mayaank.ashok/lewm_data/datasets/pusht_expert_train.h5
export MUJOCO_GL=egl OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PY=/home2/mayaank.ashok/.venv/bin/python
export N=50
mkdir -p outputs/pusht/logs
echo "[beta05ret] armed $(date)"
LA="+mpc.lookahead=13.7"
T=""

run_line() {
  local pre=()
  while [[ $# -gt 0 && $1 == *=* && $1 != +* ]]; do pre+=("$1"); shift; done
  env "${pre[@]}" bash scripts/gas_mpc_run.sh "$@"
}
round_robin() {  # splits stdin lines into Q0..Q3 (nameref arrays)
  local -n q0=$1 q1=$2 q2=$3 q3=$4; shift 4
  local i=0 line
  while IFS= read -r line; do
    case $((i % 4)) in
      0) q0+=("$line") ;;
      1) q1+=("$line") ;;
      2) q2+=("$line") ;;
      3) q3+=("$line") ;;
    esac
    i=$((i+1))
  done
}
run_queue() {
  local gpu=$1; shift
  echo "[q$gpu] starting $(date) (${#@} lines)"
  for line in "$@"; do
    # shellcheck disable=SC2086
    CUDA_VISIBLE_DEVICES=$gpu run_line $line
  done
}
L=outputs/pusht/logs

# ---- Phase A: beta=0.5 on same100 and cross, all seeds ----
mapfile -t PHASE_A < <(
  for s in 0 1 2 3 4; do echo "SEED=$s subgoal_tdr same100 $LA $T +mpc.critic_beta=0.5"; done
  for s in 0 1 2 3 4; do echo "SEED=$s subgoal_tdr cross $LA $T +mpc.critic_beta=0.5"; done
)
AQ0=(); AQ1=(); AQ2=(); AQ3=()
round_robin AQ0 AQ1 AQ2 AQ3 < <(printf '%s\n' "${PHASE_A[@]}")
run_queue 0 "${AQ0[@]}" > $L/beta05_a_q0.log 2>&1 &
run_queue 1 "${AQ1[@]}" > $L/beta05_a_q1.log 2>&1 &
run_queue 2 "${AQ2[@]}" > $L/beta05_a_q2.log 2>&1 &
run_queue 3 "${AQ3[@]}" > $L/beta05_a_q3.log 2>&1 &
wait
echo "[beta05ret] phase A (beta=0.5) finished $(date)"

# ---- Phase B: retrieval + critic beta=1 on same25/same50/same100, all seeds ----
mapfile -t PHASE_B < <(
  for s in 0 1 2 3 4; do echo "SEED=$s subgoal_tdr same25 $LA $T +mpc.critic_beta=1 +mpc.retrieval=true"; done
  for s in 0 1 2 3 4; do echo "SEED=$s subgoal_tdr same50 $LA $T +mpc.critic_beta=1 +mpc.retrieval=true"; done
  for s in 0 1 2 3 4; do echo "SEED=$s subgoal_tdr same100 $LA $T +mpc.critic_beta=1 +mpc.retrieval=true"; done
)
BQ0=(); BQ1=(); BQ2=(); BQ3=()
round_robin BQ0 BQ1 BQ2 BQ3 < <(printf '%s\n' "${PHASE_B[@]}")
run_queue 0 "${BQ0[@]}" > $L/beta05_b_q0.log 2>&1 &
run_queue 1 "${BQ1[@]}" > $L/beta05_b_q1.log 2>&1 &
run_queue 2 "${BQ2[@]}" > $L/beta05_b_q2.log 2>&1 &
run_queue 3 "${BQ3[@]}" > $L/beta05_b_q3.log 2>&1 &
wait
echo "[beta05ret] phase B (retrieval) finished $(date)"
echo "[beta05ret] all done $(date)"
