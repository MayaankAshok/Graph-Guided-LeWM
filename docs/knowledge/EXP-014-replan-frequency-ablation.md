---
id: EXP-014
title: Plan-Length Ablation (Horizon Equals Receding Horizon)
status: established
created: 2026-09-27
tags: [receding-horizon, replanning-frequency, open-loop-execution, cem-control]
derived_from: [EXP-001, EXP-007]
leads_to: []
opens_questions:
  - "At a fixed planning horizon, does more frequent replanning improve or degrade performance?"
  - "Can a closed-loop policy execute actions without the discrete replanning latency of MPC?"
---

# EXP-014: Plan-Length Ablation (Horizon Equals Receding Horizon)

## 1. Motivation & Provenance
- **Source Documents**: `docs/iclr2027/main.pdf` Appendix B.1, Table 3 (`tab:replan-frequency-appendix`: "Push-T L2 plan-length ablation"); `docs/gas-mpc/main.pdf` Section 11 ("Replan frequency ablation: fewer, longer L2 chunks", recorded 2026-09-19).
- **Problem Statement**:
  In Model Predictive Control (MPC), a critical design parameter is the **receding horizon execution frequency**: how many predicted action steps are executed open-loop in the environment before triggering the next replanning call?

  In the LeWM baseline and OUR Method, the controller optimizes a 25-step trajectory (5 blocks of 5 env steps) and executes all 25 steps open-loop before replanning (`receding_horizon = 5` blocks = 25 steps). 

  This study varies **plan length and execution chunk together** (`horizon = receding`) to test whether shorter or longer plans improve control. It does not isolate replanning frequency at a fixed planning horizon.

## 2. Experimental Protocol & Method
- **Historical drivers** (retired from the former flat scripts folder): `ada_gas_mpc_l2_rh1.sh`, `ada_gas_mpc_l2_rh10.sh`.
- **Variants Evaluated**:
  - `rh = horizon = 1`: Plan and execute 1 block (5 environment steps), then replan.
  - `rh = horizon = 5` (default): Plan and execute 5 blocks (25 steps).
  - `rh = horizon = 10`: Plan and execute 10 blocks (50 steps), tested at same100 and cross.
- **Protocols**: Push-T `task200u` first 50 tasks, CEM seeds 0–4. The 5- and 25-step variants cover all four protocols; the 50-step variant covers same100 and cross.

## 3. Empirical Results
Success from `docs/iclr2027/main.tex` Table `tab:replan-frequency-appendix` (mean % $\pm$ sample standard deviation over five CEM seeds; 50 fixed tasks per seed):

| Protocol | 5-step plan | 25-step plan (default) | 50-step plan |
| :--- | :---: | :---: | :---: |
| **same25** | 64.0% $\pm$ 1.8 | **91.2% $\pm$ 1.0** | not tested |
| **same50** | 19.2% $\pm$ 2.7 | **40.8% $\pm$ 3.7** | 34.4% $\pm$ 2.7 |
| **same100** | 1.2% $\pm$ 1.6 | **13.2% $\pm$ 3.2** | 12.8% $\pm$ 2.4 |
| **cross** | 4.4% $\pm$ 1.5 | 14.4% $\pm$ 3.9 | **15.2% $\pm$ 4.8** |

- The 5-step variant both looks ahead only 5 steps and replans more often. Its poor result cannot identify which change is responsible. The 50-step variant is close to the default at same100 and cross; the 100-step variant in the cited table performs worse.

## 4. Synthesis & Next Questions
- **Established Fact**: With `horizon = receding`, 25-step plans outperform 5-step plans on every tested protocol; 50-step plans are competitive at same100 and cross.
- **Open mechanism**: A fixed-horizon ablation is needed to separate replanning frequency from lookahead length.
