---
name: lewm-experimenter
description: >-
  Use this skill when running long-horizon planning evaluations (gas_mpc_eval.py), benchmark sweeps,
  or ablation studies across Push-T, Reacher, and Cube, enforcing protocol integrity and baseline comparability.
---

# LEWM Experimenter & Evaluation Skill (`lewm-experimenter`)

This skill defines the standardized benchmark protocols, pre-flight checklists, and execution commands for evaluating long-horizon planning algorithms against the LeWM world model.

---

## 1. Standard Benchmark Protocol (The ICLR Protocol)

To ensure comparability with published results and `docs/knowledge/`:
- **Task Pool**: The sole active pool is `task200u` (`gas_mpc_eval.POOL`). Pre-solved pairs where the environment's success predicate is satisfied at $t=0$ are rejected at generation time (`EXP-015`).
- **Sample Prefix**: First 50 tasks (`eval.num_eval=50`).
- **Seeds**: 5 seeds (`seed=0, 1, 2, 3, 4`). Headline metrics are mean $\pm$ standard error across 5 seeds.
- **Protocols**:
  - `same25`: Intra-episode start and goal at offset $\Delta t = 25$. Execution budget: 50 steps.
  - `same50`: Intra-episode offset $\Delta t = 50$. Budget: 100 steps.
  - `same100`: Intra-episode offset $\Delta t = 100$. Budget: 200 steps.
  - `cross`: Inter-episode start and goal sampled uniformly at random. Budget: 250 steps (Push-T), 200 steps (Reacher).
- **Control Chunking**: Receding horizon of 5 blocks ($5 \times 5 = 25$ real environment steps executed per plan iteration, `EXP-014`).

---

## 2. Pre-Flight Verification Checklist

Before starting any benchmark run, verify these 5 invariant checkpoints:

| Check | Requirement | Verification Method |
|---|---|---|
| **1. Action Normalization** | Model receives z-scored actions ($\sigma_a=0.206$ on Push-T). Env receives raw $[-1, 1]$. | Inspect `StandardScaler` / `process` pipeline. Raw action feeds cause silent $4.8\times$ shrinkage (`EXP-011`). |
| **2. Checkpoint Health** | Pretrained weights loaded correctly without dimensional collapse. | Check off-diagonal cosine similarity ($\sim 0.02-0.06$) and participation ratio (`EXP-002`, `EXP-011`). |
| **3. Data Isolation** | Test/eval episodes strictly excluded from training and calibration. | Confirm learned assets read `cache_train.npz` and evaluations read `cache_test.npz`. |
| **4. Headless EGL** | Headless rendering bound to the matching GPU worker. | Confirm `CUDA_VISIBLE_DEVICES` == `MUJOCO_EGL_DEVICE_ID` before imports (`EXP-018`). |
| **5. Task Pool** | Validated `task200u` tasks loaded. | Confirm log outputs `Loaded pool task200u` (`EXP-015`). |

---

## 3. Evaluation Command Recipes

### Baseline Evaluation (Vanilla LeWM CEM/L2)
```bash
python scripts/gas_mpc/gas_mpc_eval.py \
  +mpc.method=l2 \
  +mpc.protocol=cross \
  eval.num_eval=50 \
  seed=0
```

### Reference Benchmark: OUR Method (ICLR 2027 Headline)
```bash
python scripts/gas_mpc/gas_mpc_eval.py \
  +mpc.method=subgoal_tdr \
  +mpc.protocol=cross \
  +mpc.critic_kind=ht \
  +mpc.critic_weight=1.0 \
  +mpc.final_switch=true \
  +mpc.final_metric=l2 \
  +mpc.final_horizon=1 \
  eval.num_eval=50 \
  seed=0
```

### Headline Performance Targets (`EXP-007`)
Any new unified planner must be benchmarked against these results (5 seeds, `task200u`, prefix 50):

| Method | Push-T (Cross) | Reacher (Cross) | Cube (Cross) |
|---|---|---|---|
| **Vanilla LeWM L2** | $10.4\% \pm 1.6\%$ | $29.2\% \pm 2.8\%$ | $16.4\% \pm 2.1\%$ |
| **OUR Method** | $\mathbf{42.4\% \pm 3.1\%}$ | $\mathbf{65.6\% \pm 3.2\%}$ | $\mathbf{27.6\% \pm 2.4\%}$ |

### Reporting & Results Aggregation
```bash
# Generate markdown table from experiment logs
python scripts/gas_mpc/gas_mpc_report.py outputs/pusht/
```
Once results are aggregated, synthesize findings into an atomic knowledge card (`EXP-XXX`) following the `research-kb` skill.
