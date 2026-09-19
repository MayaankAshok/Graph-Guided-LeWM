#!/usr/bin/env bash
# Phase 6c: same-episode offset-25 protocol on the fixed 200-task set (first 50 tasks),
# same three finalist configs as same50: l2, subgoal_tdr, subgoal_tdr + critic beta=2.
# Runs entirely on GPU 3, which ada_seed_variance_training.sh deliberately leaves free
# for eval work while it trains TDR/critic seeds 1-4 on GPUs 0-2. Single queue, no waiting
# -- GPUs 0-2 are already double-booked (phase6 eval + seed-variance training), so this
# avoids adding more contention there.
#   nohup srun --jobid=2697130 --overlap bash scripts/ada_gas_mpc_phase6_same25.sh > outputs/pusht/logs/ada_driver6c.log 2>&1 &
cd /home2/mayaank.ashok/lewm_research || exit 1
export PUSHT_H5_PATH=/ssd_scratch/mayaank.ashok/lewm_data/datasets/pusht_expert_train.h5
export MUJOCO_GL=egl OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PY=/home2/mayaank.ashok/.venv/bin/python
export N=50
export CUDA_VISIBLE_DEVICES=3
mkdir -p outputs/pusht/logs
echo "[phase6c] armed $(date)"
LA="+mpc.lookahead=13.7"
T=""
mapfile -t Q3 < <(
  for s in 0 1 2 3 4; do echo "SEED=$s l2 same25 $T"; done
  for s in 0 1 2 3 4; do echo "SEED=$s subgoal_tdr same25 $LA $T"; done
  for s in 0 1 2 3 4; do echo "SEED=$s subgoal_tdr same25 $LA $T +mpc.critic_beta=2"; done
)
run_line() {
  local pre=()
  while [[ $# -gt 0 && $1 == *=* && $1 != +* ]]; do pre+=("$1"); shift; done
  env "${pre[@]}" bash scripts/gas_mpc_run.sh "$@"
}
L=outputs/pusht/logs
echo "[phase6c q3] starting $(date) (${#Q3[@]} lines)"
for line in "${Q3[@]}"; do
  # shellcheck disable=SC2086
  run_line $line
done > $L/ada6c_q3.log 2>&1
echo "[phase6c] all runs finished $(date)"
