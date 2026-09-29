---
name: ada-cluster
description: >-
  Use this skill when preparing, dispatching, monitoring, or debugging research experiments,
  evaluations, and data pipelines on the IIIT-H Ada HPC cluster.
---

# Ada HPC Cluster Operations Skill (`ada-cluster`)

This skill defines operational procedures, storage constraints, and execution recipes for the IIIT-H Ada cluster (`ada.iiit.ac.in`).

---

## 1. Cluster Storage Hierarchy (Strict Separation)

| Storage Location | Accessibility | Quota / Lifecycle | Purpose & Rules |
|---|---|---|---|
| `/share1/mayaank.ashok/lewm_data/` | **Login node (`ada`) ONLY** | 100GB | Master compressed datasets (`.zst`). **Does not exist on compute nodes.** |
| `/ssd_scratch/mayaank.ashok/` | **Active Slurm allocation only** | Local SSD (Purged ~7 days) | Fast decompressed dataset cache (`pusht_expert_train.h5`). Never assume persistence across allocation changes. |
| `/home2/mayaank.ashok/` | Shared filesystem | **30GB strict quota** | Code repositories and Python venv (`/home2/mayaank.ashok/.venv`). **Never put decompressed datasets here!** Push-T is 43GB decompressed and will immediately exceed quota. |
| `/home2/.../checkpoints/` | Shared filesystem | Permanent | Pretrained LeWM model weights (~72MB per env). |

> [!WARNING]
> **Never access `/share1` from a compute node**; it will fail with not found / permission error.
> **Never decompress full datasets into `/home2`**; verify quota with `quota -s`.

**Never create, submit, or allocate a Slurm job.** Do not run `sinteractive`, `salloc`, `sbatch`,
or `srun` without `--jobid`. Reuse the user's active allocation; if none is active, stop and ask
the user instead of starting one. Do not SSH directly to a compute node.

---

## 2. Cluster Hardware & Environment Setup

- **Compute Partition**: `u22`.
- **Python Module**: `module load u22/python/3.12.4`
- **Venv Activation**: `source /home2/mayaank.ashok/.venv/bin/activate`
- **Pip Cache**: `export PIP_CACHE_DIR=/share1/mayaank.ashok/pip_cache` (run pip installs only on login node `ada`).

---

## 3. Critical Cluster Execution Invariants

### Invariant 1: Headless MuJoCo EGL & GPU Pairing
When running multi-GPU evaluations or workers, both environment variables must be exported before importing PyTorch or MuJoCo:
```bash
export CUDA_VISIBLE_DEVICES=$SLOT
export MUJOCO_EGL_DEVICE_ID=$SLOT
```
*Why:* PyTorch remaps the selected physical device to logical `cuda:0`, but MuJoCo's headless EGL renderer requires the physical device ID. Mismatch causes rendering crashes or contention.

### Invariant 2: Action Z-Scoring Invariant
Any script feeding actions to `model.action_encoder` or `model.predict` must z-score actions using the FULL dataset mean/std (Push-T: $\sigma_a = 0.206$; un-normalized actions make motions $4.8\times$ too small). De-normalize back to raw $[-1, 1]$ before `env.step()`.

### Invariant 3: Training-Only Data Provenance
Evaluations and test episodes must NEVER enter training or asset preparation.
- Use `cache_train.npz` and `psi_train_s*.npy`.
- Staged on Ada at: `/ssd_scratch/mayaank.ashok/planning_trainonly/<env>/`.

### Invariant 4: Slurm Memory & FAISS Cgroup Limit
Interactive jobs have `mem=20G`. A FAISS graph index over 2.3M frames run alongside model training will trigger a silent kernel cgroup-OOM kill without Python traceback. **Always run FAISS builds in isolation.**

---

## 4. Operational Runbook & Command Recipes

### Checking Allocation & Job Status
```bash
# Find active job ID
squeue -u mayaank.ashok -o "%.18i %.9P %.24j %.8T %.10M %.20R"
```

### Staging Data to Compute Node Scratch
```bash
# Run from ada inside the existing allocation with srun --jobid=<jobid> --overlap:
srun --jobid=<jobid> --overlap bash -lc '
mkdir -p /ssd_scratch/mayaank.ashok/lewm_data/datasets
rsync -avP ada:/share1/mayaank.ashok/lewm_data/pusht_expert_train.h5.zst /ssd_scratch/mayaank.ashok/lewm_data/
zstd -d /ssd_scratch/mayaank.ashok/lewm_data/pusht_expert_train.h5.zst \
  -o /ssd_scratch/mayaank.ashok/lewm_data/datasets/pusht_expert_train.h5
export PUSHT_H5_PATH=/ssd_scratch/mayaank.ashok/lewm_data/datasets/pusht_expert_train.h5
'
```

### Dispatching Work from Login Node (`ada`)
```bash
srun --jobid=<jobid> --overlap bash -lc 'source /home2/mayaank.ashok/.venv/bin/activate && cd /home2/mayaank.ashok/lewm_research && python evaluator.py --config config/evaluations/pusht_headline.yaml --run'
```

Keep long-running work attached to the existing `srun` step. Never detach a new `srun` or
submit a background Slurm job. If the allocation changes, `/ssd_scratch` is local to the old
node and must be staged again through the new active allocation.
