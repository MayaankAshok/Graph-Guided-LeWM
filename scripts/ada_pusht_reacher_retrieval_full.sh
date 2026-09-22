#!/usr/bin/env bash
# Runs the Push-T and Reacher OUR+retr sweeps sequentially (Push-T first, then Reacher),
# not concurrently: each sweep already uses 8 slots (2 workers/GPU x 4 GPUs), and Reacher's
# EGL renderer is a known OOM risk even at 2/GPU alone (see ada_reacher_retrieval_full.sh) --
# running both sweeps at once would double that to 4 processes/GPU and compound it.
# Both sub-scripts are independently resumable (gas_mpc_eval.py skips any (seed, protocol)
# already done), so re-running this script after an interruption only fills in what's missing.
#   nohup srun --jobid=<JOB> --overlap bash scripts/ada_pusht_reacher_retrieval_full.sh \
#     > outputs/pusht/logs/retrieval_full_pusht_then_reacher.out 2>&1 &
set -u
cd /home2/mayaank.ashok/lewm_research || exit 1

echo "[pusht+reacher retrieval] start $(date)"
bash scripts/ada_pusht_retrieval_full.sh
bash scripts/ada_reacher_retrieval_full.sh
echo "[pusht+reacher retrieval] finished $(date)"
