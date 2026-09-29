---
id: EXP-015
title: Evaluation Protocol and Task Pool Integrity (task200u vs legacy task200)
status: established
created: 2026-09-27
tags: [evaluation-protocol, benchmarks, task-pools, data-hygiene, artifacts]
derived_from: [EXP-003, EXP-007, EXP-010]
leads_to: []
opens_questions:
  - "How can automated benchmark assertions prevent pre-solved or degenerate task pairs from entering evaluation pools?"
---

# EXP-015: Evaluation Protocol and Task Pool Integrity (task200u vs legacy task200)

## 1. Motivation & Provenance
- **Source Documents**: `docs/iclr2027/main.pdf` Section 5.1 ("Setup", lines 473–478); `docs/gas-mpc/main.pdf` Section 9 ("Task pools: pre-solved tasks and the task200u convention", recorded 2026-09-18).
- **Problem Statement**:
  In empirical benchmark evaluation, subtle data filtering flaws can silently distort reported success numbers and produce misleading scientific conclusions. 

  During the development of `gas-mpc` (documented in `docs/gas-mpc/main.tex` Section 9 on 2026-09-18), an audit of evaluation task files revealed a critical data hygiene issue in the legacy `task200` task pool: **many randomly sampled task pairs were already satisfied at timestep $t=0$ before any planning action was taken!**

  This document establishes the provenance of the sanitized `task200u` benchmark suite and details why legacy results must never be mixed with `task200u` numbers.

## 2. Experimental Protocol & Method
- **Active scripts**: `scripts/gas_mpc/gas_mpc_make_tasks.py`, `scripts/tests/test_heldout_tasks.py`.
- **Historical driver** (retired from the former flat scripts folder): `ada_regenerate_task200u.sh`.
- **Task Artifacts**: `outputs/pusht/pairs/pairs_*_task200u.json`, `outputs/cube/pairs/pairs_*_task200u.json`.
- **The Audit Protocol**:
  - Load all 200 task pairs across all protocols (`same25`, `same50`, `same100`, `cross`) in the legacy `task200` files.
  - Evaluate the environment's ground-truth success predicate on the initial state $(s_0, s_{\text{goal}})$ at $t=0$.
  - Record the fraction of pre-solved pairs per environment and protocol.
- **The Sanitized `task200u` Convention**:
  - `gas_mpc_make_tasks.py` was updated to explicitly reject any candidate pair if `env.is_success(s_0, s_{\text{goal}}) == True`.
  - Rejection sampling was repeated until exactly 200 strictly non-trivial pairs were generated per protocol.

## 3. Empirical Results
- **Pre-Solved Tasks in the Legacy Pool (`task200`)**:
  - **Push-T**: 0% pre-solved across all protocols (the initial arm and block are never coincident with goal pose).
  - **Reacher**: 0% pre-solved across all protocols.
  - **OGBench Cube**: Severe distortion!
    - `same25`: **22.0%** (11 of the first 50 tasks were already solved at $t=0$!).
    - `same50`: **16.0%** (8 of 50 tasks pre-solved).
    - `same100`: **44.0%** (22 of 50 tasks pre-solved!).
    - `cross`: 0% pre-solved.

- **Consequences of the Legacy Pool on Cube**:
  - In the legacy `task200` pool, a completely dead controller that does nothing at all would report a 44% "success rate" on same100!
  - Prior reported baseline numbers (e.g. L2 same25 reading 74% in older configs) were heavily inflated by these pre-solved artifacts.
  - In the sanitized `task200u` pool, the true baseline performance of vanilla L2 is established as:
    - Cube same25: **56.0%**
    - Cube same50: **34.4%**
    - Cube same100: **35.6%**
    - Cube cross: **16.4%**

## 4. Synthesis & Next Questions
- **Established Invariant**:
  - All valid evaluations in this project must strictly specify `POOL = 'task200u'` (`gas_mpc_eval.py`).
  - Results tagged with `task200` and `task200u` are completely non-comparable and must never be combined into a single table.
- **Data Provenance Rule**: Any newly introduced evaluation environment must run an automated initial-state predicate audit to guarantee zero pre-solved tasks.
