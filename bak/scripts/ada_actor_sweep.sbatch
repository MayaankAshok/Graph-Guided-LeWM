#!/bin/bash
# SLURM array job for the actor + live-rollout sweep (tworoom_b3_actor_train.py +
# tworoom_b3_actor_rollout_eval.py): all 6 tiers x 2 variants x 5 seeds = 60 independent
# tasks, expanded from the single-seed local pilot now that cluster compute is available.
#
# Verified Ada layout (checked directly, not assumed):
#   - Repo lives on /home2/mayaank.ashok/lewm_research -- NFS-mounted, reachable from both
#     the login node and every compute node (unlike /share1, which is login-node-only).
#   - The LeWM checkpoint (data/checkpoints/models--quentinll--lewm-tworooms/) is small
#     (72MB) and already sits there permanently -- no per-job staging needed, the default
#     (un-overridden) CKPT_DIR just resolves correctly.
#   - The dataset master copy (compressed, tworoom.tar.zst, ~3.3GB) lives once on
#     /share1/mayaank.ashok/lewm_data/ -- reachable from a compute node via `scp
#     ada:/share1/...` (confirmed working: SSH from a compute node back to the login node,
#     by hostname or internal IP, succeeds -- it's the raw /share1 filesystem mount that's
#     login-node-only, not SSH reachability). Every task re-stages its own copy to this
#     node's local /ssd_scratch and unpacks there, since scratch is per-node, not shared,
#     and a task can land on any node.
#   - partition=u22, no --account needed (both confirmed by test allocations that
#     succeeded without specifying one).
#
#SBATCH --job-name=lewm_actor_sweep
#SBATCH --partition=u22
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --array=0-59%4
#SBATCH --output=logs/actor_sweep_%A_%a.out
#SBATCH --error=logs/actor_sweep_%A_%a.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

SCRATCH="/ssd_scratch/$USER"
DATA_DIR="$SCRATCH/lewm_data"

# --- stage the dataset onto this node's local scratch, every task, every time ---
mkdir -p "$SCRATCH"
scp ada:/share1/mayaank.ashok/lewm_data/tworoom.tar.zst "$SCRATCH/tworoom.tar.zst"
rm -rf "$DATA_DIR"          # in case a previous job left a stale/partial copy on this node
mkdir -p "$DATA_DIR"
tar --zstd -xf "$SCRATCH/tworoom.tar.zst" -C "$DATA_DIR"
rm -f "$SCRATCH/tworoom.tar.zst"

# Locate the extracted h5 rather than assuming its exact path within the archive --
# find() is robust regardless of whether it lands at the top level or nested.
H5_FILE=$(find "$DATA_DIR" -name "*.h5" | head -1)
if [ -z "$H5_FILE" ]; then
    echo "ERROR: no .h5 file found after extracting tworoom.tar.zst -- check archive contents" >&2
    exit 1
fi
export LEWM_H5_PATH="$H5_FILE"
echo "[task $SLURM_ARRAY_TASK_ID] LEWM_H5_PATH=$LEWM_H5_PATH"

# --- index -> (tier, variant, seed), same scheme as the local PowerShell pairing ---
TIERS=(expert_10 expert_25 expert_50 expert_100 mixed mixed_large)
VARIANTS=(baseline auxphi)
SEEDS=(0 1 2 3 4)

N_SEEDS=${#SEEDS[@]}
N_VARIANTS=${#VARIANTS[@]}

idx=$SLURM_ARRAY_TASK_ID
seed_idx=$(( idx % N_SEEDS )); idx=$(( idx / N_SEEDS ))
variant_idx=$(( idx % N_VARIANTS )); idx=$(( idx / N_VARIANTS ))
tier_idx=$idx

TIER=${TIERS[$tier_idx]}
VARIANT=${VARIANTS[$variant_idx]}
SEED=${SEEDS[$seed_idx]}

echo "[task $SLURM_ARRAY_TASK_ID] tier=$TIER variant=$VARIANT seed=$SEED host=$(hostname) gpu=$(nvidia-smi -L 2>/dev/null || echo none)"

source .venv/bin/activate

# Train (resumable -- if this task's walltime runs out before done=True, resubmitting the
# SAME array index reruns this script and picks up from the last checkpoint automatically,
# same as the local runs). Cache-building (landmarks/graph/phi_dist) is NOT pre-warmed by
# this script -- run ada_warm_caches.py once, sequentially, before submitting this array,
# so concurrent same-tier tasks never race to build the same cache file.
python scripts/tworoom_b3_actor_train.py --tier "$TIER" --variant "$VARIANT" --seed "$SEED"

# Rollout eval at BOTH peaks -- this is the direct comparison the single-seed local pilot
# was built to enable (does the best success rate land at the same step as the best
# Spearman peak, or not).
python scripts/tworoom_b3_actor_rollout_eval.py --tier "$TIER" --variant "$VARIANT" --seed "$SEED" \
    --select rho --n-episodes 100
python scripts/tworoom_b3_actor_rollout_eval.py --tier "$TIER" --variant "$VARIANT" --seed "$SEED" \
    --select success --n-episodes 100

echo "[task $SLURM_ARRAY_TASK_ID] done"
