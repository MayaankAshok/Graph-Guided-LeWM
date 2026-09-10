#!/bin/bash
cd /home2/mayaank.ashok/lewm_research
source /home2/mayaank.ashok/.venv/bin/activate
export PUSHT_H5_PATH=/ssd_scratch/mayaank.ashok/lewm_data/datasets/pusht_expert_train.h5
export PUSHT_TIER_CACHE_DIR=/ssd_scratch/mayaank.ashok/lewm_pusht_cache/tier_cache
export PUSHT_ROLLOUT_CACHE_DIR=/ssd_scratch/mayaank.ashok/lewm_pusht_cache/rollout_cache
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8

run_one() {
  tier=$1; seed=$2; gpu=$3
  export CUDA_VISIBLE_DEVICES=$gpu
  python scripts/actor_train.py env=pusht tier=$tier variant=stitch phi_mode=none seed=$seed \
  && python scripts/actor_rollout_eval.py env=pusht tier=$tier variant=stitch phi_mode=none seed=$seed select=success n_episodes=100 \
  && python scripts/actor_rollout_eval.py env=pusht tier=$tier variant=stitch phi_mode=none seed=$seed select=rho n_episodes=100
  echo "DONE tier=$tier seed=$seed exit=$?"
}

TIERS=(expert_100 expert_300 expert_1000)
SEEDS=(0 1 2)
slot_pid=(0 0 0)
slot_job=("" "" "")
queue=()
for tier in "${TIERS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    queue+=("$tier $seed")
  done
done

i=0
while [ $i -lt ${#queue[@]} ] || [ -n "${slot_pid[0]}${slot_pid[1]}${slot_pid[2]}" ]; do
  for gpu in 0 1 2; do
    pid=${slot_pid[$gpu]}
    if [ -z "$pid" ] || [ "$pid" = "0" ] || ! kill -0 "$pid" 2>/dev/null; then
      slot_pid[$gpu]=""
      if [ $i -lt ${#queue[@]} ]; then
        job="${queue[$i]}"
        tier="${job% *}"
        seed="${job#* }"
        run_one "$tier" "$seed" "$gpu" > "stitch_v2_${tier}_s${seed}.log" 2>&1 &
        slot_pid[$gpu]=$!
        i=$((i+1))
      fi
    fi
  done
  sleep 10
done
wait
echo ALL-DONE-EXPERT-SWEEP-V2
