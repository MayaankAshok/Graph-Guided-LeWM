#!/bin/bash
# One seed of the cross-episode LeWM-CEM baseline on Ada. Usage: ada_cross_episode_baseline.sh SEED GPU
# Launch from the LOGIN node (anything backgrounded inside an srun step dies with the step):
#   nohup srun --jobid=<id> --overlap bash scripts/ada_cross_episode_baseline.sh 0 0 > /dev/null 2>&1 &
SEED=$1; GPU=${2:-0}
cd /home2/mayaank.ashok/lewm_research
source /home2/mayaank.ashok/.venv/bin/activate
export PUSHT_H5_PATH=/ssd_scratch/mayaank.ashok/lewm_data/datasets/pusht_expert_train.h5
export CUDA_VISIBLE_DEVICES=$GPU
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
O=outputs/pusht/pairs
mkdir -p $O
echo "=== seed $SEED gpu $GPU start $(date) on $(hostname)" > $O/cross_episode_s$SEED.log
python scripts/viability_cross_episode_baseline.py eval.num_eval=50 "+cross.seeds=[$SEED]" +cross.max_budget=250 \
  >> $O/cross_episode_s$SEED.log 2>&1
echo "=== seed $SEED done $(date) exit=$?" >> $O/cross_episode_s$SEED.log
