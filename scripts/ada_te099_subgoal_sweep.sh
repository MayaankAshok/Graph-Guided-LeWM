#!/usr/bin/env bash
# TE=0.99 transition, decided from scripts/gas_mpc_te_diag.py's diagnostic (seed 0: fully
# reachable at TE=0.99, ranking Spearman statistically tied with TE=0.9, ~27% fewer nodes /
# ~37% faster graph build). Step 1 rebuilds seeds 1-4's graphs at TE=0.99 (h_td=8, main TDR
# e0.99 -- seed 0's graph_full_s0_htd8_te0.99.pkl already exists, built during the diagnostic).
# Step 2 runs subgoal_tdr with NO critic (mpc.critic_beta left at its 0.0 default) across all
# 4 protocols x 5 seeds = 20 evals, N=50 (first 50 of the fixed 200-task set per protocol).
# TDR/psi/critic are untouched by this -- TE only changes graph construction (see CLAUDE.md).
#
#   nohup srun --jobid=2697130 --overlap bash scripts/ada_te099_subgoal_sweep.sh \
#       > outputs/pusht/logs/ada_te099_driver.log 2>&1 &
cd /home2/mayaank.ashok/lewm_research || exit 1
export PUSHT_H5_PATH=/ssd_scratch/mayaank.ashok/lewm_data/datasets/pusht_expert_train.h5
export MUJOCO_GL=egl OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PY=/home2/mayaank.ashok/.venv/bin/python
export N=50
mkdir -p outputs/pusht/logs
echo "[te099] armed $(date)"

echo "[te099] rebuilding graphs seeds 1-4 at TE=0.99 (seed 0 already exists)"
for s in 1 2 3 4; do
  CUDA_VISIBLE_DEVICES=0 $PY scripts/gas_mpc_prepare.py graph --seed "$s" --h-td 8 --te 0.99 \
    >> outputs/pusht/logs/graph_s${s}_te0.99.log 2>&1
  echo "[te099] graph seed=$s done $(date)"
done

LA="+mpc.lookahead=13.7"
TE="+mpc.te=0.99"
T=""
run_line() {
  local pre=()
  while [[ $# -gt 0 && $1 == *=* && $1 != +* ]]; do pre+=("$1"); shift; done
  env "${pre[@]}" bash scripts/gas_mpc_run.sh "$@"
}
# One protocol per GPU queue.
mapfile -t Q0 < <(for s in 0 1 2 3 4; do echo "SEED=$s subgoal_tdr cross $LA $TE $T"; done)
mapfile -t Q1 < <(for s in 0 1 2 3 4; do echo "SEED=$s subgoal_tdr same50 $LA $TE $T"; done)
mapfile -t Q2 < <(for s in 0 1 2 3 4; do echo "SEED=$s subgoal_tdr same100 $LA $TE $T"; done)
mapfile -t Q3 < <(for s in 0 1 2 3 4; do echo "SEED=$s subgoal_tdr same25 $LA $TE $T"; done)
run_queue() {
  local gpu=$1; shift
  echo "[te099 q$gpu] starting $(date) (${#@} lines)"
  for line in "$@"; do
    # shellcheck disable=SC2086
    CUDA_VISIBLE_DEVICES=$gpu run_line $line
  done
}
L=outputs/pusht/logs
run_queue 0 "${Q0[@]}" > $L/te099_q0.log 2>&1 &
run_queue 1 "${Q1[@]}" > $L/te099_q1.log 2>&1 &
run_queue 2 "${Q2[@]}" > $L/te099_q2.log 2>&1 &
run_queue 3 "${Q3[@]}" > $L/te099_q3.log 2>&1 &
wait
echo "[te099] all done $(date)"
