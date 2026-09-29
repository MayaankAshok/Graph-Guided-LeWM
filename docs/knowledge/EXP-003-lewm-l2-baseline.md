---
id: EXP-003
title: Vanilla LeWM CEM/L2 Baseline Long-Horizon Failure
status: established
created: 2026-09-27
tags: [baselines, lewm, cem, failure-modes, benchmarks]
derived_from: [EXP-001, EXP-002]
leads_to: [EXP-004, EXP-007, EXP-011, EXP-015, EXP-016, EXP-017, EXP-018]
opens_questions:
  - "Why does L2 perform exceptionally well at same25 (>91%) despite the predictor error and shell curvature?"
  - "Can a planner exploit L2's short-range precision while delegating long-range guidance to a different mechanism?"
---

# EXP-003: Vanilla LeWM CEM/L2 Baseline Long-Horizon Failure

## 1. Motivation & Provenance
- **Source Documents**: `docs/iclr2027/main.pdf` Section 1, Section 5.1, Section 5.2 (Table 1: `tab:main-results`); `docs/gas-mpc/main.pdf` Section 1, Section 3, Section 4 (`results_table.tex`), Section 7, and Section 8.
- **Problem Statement**:
  LeWorldModel (LeWM, Le Lidec et al., 2025) achieved state-of-the-art results on short-horizon imitation benchmarks using pure Cross-Entropy Method (CEM) planning over a frozen latent JEPA predictor. In the original paper, planning was evaluated almost exclusively on short horizons (e.g. 25-step lookahead towards same-episode goals).

  Following the insights from [[EXP-001]](EXP-001-predictor-drift.md) (predictor error accumulation) and [[EXP-002]](EXP-002-gaussian-shell-ood.md) (Gaussian shell OOD collapse), this experiment systematically evaluates the vanilla LeWM baseline across varying horizon lengths (`same25`, `same50`, `same100`) and arbitrary cross-episode pairings (`cross`) on a standardized, unfiltered evaluation suite (`task200u`).

## 2. Experimental Protocol & Method
- **Script**: `scripts/gas_mpc/gas_mpc_eval.py +mpc.method=l2 +mpc.protocol=<protocol> eval.num_eval=50`
- **Output Artifacts**: `docs/gas-mpc/results.json`, `docs/gas-mpc/results_reacher.json`, `docs/gas-mpc/results_cube.json`.
- **Planner**: Unchanged LeWM CEM (300 action candidates, 30 iterations, top-30 elite selection, planning horizon 5 blocks = 25 steps, receding horizon 5 blocks).
- **Objective**: Terminal latent L2 distance $\|\hat{z}_5 - z_{\text{goal}}\|^2$.
- **Protocols & Budgets**:
  - `same25`: Goal is 25 steps ahead in the same episode (budget 50 steps).
  - `same50`: Goal is 50 steps ahead in the same episode (budget 100 steps).
  - `same100`: Goal is 100 steps ahead in the same episode (budget 200 steps).
  - `cross`: Goal is randomly sampled from a completely different held-out episode (budget 250 steps for Push-T, 200 for Reacher/Cube).
- **Evaluation Suite**: Sanitized `task200u` pool (first 50 fixed tasks), evaluated across 5 random seeds (0 to 4), total 250 task instances per protocol. Pre-solved pairs at $t=0$ are strictly excluded ([[EXP-015]](EXP-015-task-pool-integrity.md)).

## 3. Empirical Results
Submitted vanilla LeWM baseline from `docs/iclr2027/main.tex` Table `tab:main` (mean % $\pm$ sample standard deviation over five CEM seeds, 50 fixed tasks per seed):

The earlier `docs/gas-mpc/results.json` records Push-T L2 cross at 10.4%; the submitted table reports 14.4%. They are separate recorded runs and are not pooled here.

| Benchmark | same25 | same50 | same100 | cross |
| :--- | :---: | :---: | :---: | :---: |
| **Push-T** | **91.2% $\pm$ 1.0** | 40.8% $\pm$ 3.7 | 13.2% $\pm$ 3.2 | **14.4% $\pm$ 3.9** |
| **Reacher** | **77.2% $\pm$ 4.8** | 90.8% $\pm$ 1.0 | 82.8% $\pm$ 1.0 | **29.2% $\pm$ 2.4** |
| **Cube** | **56.0% $\pm$ 3.8** | 34.4% $\pm$ 3.2 | 35.6% $\pm$ 1.5 | **16.4% $\pm$ 1.5** |

### Failure Anatomy in Cross-Episode Tasks:
- In Push-T cross-episode tasks, vanilla LeWM achieves 14.4% success.
- Success occurs almost exclusively on degenerate pairs where the block happens to start near the goal block pose by chance ($< 60\text{px}$ distance, $< 30^\circ$ rotation).
- On all failing episodes:
  - The robotic end-effector accurately reaches the goal end-effector position (best agent distance: $14\text{px}$).
  - However, the block is left far from its goal pose (final block distance: $106\text{px}$, rotation error $91^\circ$).
  - **The Saliency Trap**: Terminal L2 gradient is dominated by the robot arm / end-effector representation. The planner finds it easier to minimize $\|\hat{z} - z_g\|$ by parking the arm at the goal location rather than executing complex contact sequences to push the object.

## 4. Synthesis & Next Questions
- **Established Fact**: Vanilla LeWM reaches 91.2% on Push-T same25 but only 14.4% on Push-T cross.
- **Core Dilemma**:
  1. For short horizons ($H \le 25$), L2 planning is fast, precise, and highly reliable.
  2. For long horizons ($H > 25$), L2 provides no topological guidance, gets stuck in local minima, and triggers Gaussian shell OOD dynamics.
- **Next Steps**:
  - We need a global progress signal that understands multi-step temporal topology without destroying the short-range execution precision of the local controller. (Explored via TDR in [[EXP-004]](EXP-004-tdr-calibration.md) and Graph search in [[EXP-005]](EXP-005-gas-graph-stitching.md)).
