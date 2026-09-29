---
id: EXP-006
title: Expected Hitting-Time Critic (ET vs -log P) and Survival Formulations
status: established
created: 2026-09-27
tags: [critic, hitting-time, viability, bellman-backup, cost-composition]
derived_from: [EXP-005]
leads_to: [EXP-007, EXP-008, EXP-016, EXP-017]
opens_questions:
  - "Can the hitting-time critic be trained directly as a continuous scalar regressor without survival bins?"
  - "Why does the critic cost need to be turned off in the final lookahead horizon to prevent short-horizon regression?"
---

# EXP-006: Expected Hitting-Time Critic (ET vs -log P) and Survival Formulations

## 1. Motivation & Provenance
- **Source Documents**: `docs/iclr2027/main.pdf` Section 3.3 ("Reachability score and the remaining budget"), Appendix J.6 (Proposition 11: "Survival identity, truncation, and ranking"); `docs/gas-mpc/main.pdf` Section 2 ("Final-phase switch and critic terms"), Section 6 (Phases 4 & 8), and `docs/gas-mpc/hitting-time-critic-study.md`.
- **Problem Statement**:
  In [[EXP-005]](EXP-005-gas-graph-stitching.md), graph subgoals plus the local switch lifted Push-T cross-episode success from 14.4% to 17.2%. However, planning frequently failed because CEM selected candidate rollout endpoints $\hat{z}_5$ that appeared close to graph nodes in TDR space but were dynamically infeasible from the current physical contact state.

  To filter candidate endpoints, we need a **reachability critic** that evaluates whether a predicted latent $\hat{z}$ can reach the goal within the remaining episode budget $h_{\text{rem}}$.

  This experiment investigates: **how should the reachability critic be formulated, and why did standard probability formulations ($-\log V$) fail compared to budget-capped expected hitting time ($\mathbb{E}[T]$)?**

## 2. Experimental Protocol & Method
- **Active scripts**: `scripts/critics/viability_train.py`, `scripts/common/viability.py`.
- **Historical drivers** (retired from the former flat scripts folder): `viability_train_ht.py`, `ada_hitting_time_critic_phase1.sh`.
- **Output Checkpoints**: `outputs/pusht/critic_training/critic_full_s0/`, `outputs/pusht/critic_training/ht_full_s0/`.
- **Architectures**:
  1. *Viability Ensemble (`ViabilityCritic`)*: The submitted method uses five MLP members predicting binary reachability $V(z, z_g, h) \in [0, 1]$ over discrete horizon budgets. Trained with hindsight labels, cross-episode negatives, and Bellman backups through the frozen LeWM predictor.
  2. *Categorical Hitting Time Head (`HittingTimeHead`)*: Single-pass network predicting probability mass function $p(T = b \mid z, z_g)$ over 47 classes (multiples of 5 env steps up to 225, plus a censored $>225$ tail bin).
- **Cost Formulations in CEM**:
  - *Negative Log-Likelihood*: $C_{\text{nlv}} = -\log V(\hat{z}_5, z_g, h_{\text{rem}})$.
  - *Budget-Capped Expected Hitting Time*:
    $$\mathbb{E}[T]_{h_{\text{rem}}} = 5 \sum_{j \in \{0, 5, \dots, h_{\text{rem}}\}} \big(1 - V(\hat{z}_5, z_g, j)\big)$$
    representing the restricted survival sum in environment steps. Feasible endpoints are ordered by how early they hit the goal; endpoints exceeding the budget cap plateau at $h_{\text{rem}}$.

## 3. Empirical Results
- **The Collapse of $-\log V$**:
  - Across testing, $-\log V$ proved catastrophic for multi-horizon planning.
  - *Saturation*: When $V \approx 0$ or $V \approx 1$, small classifier calibration errors produce astronomical gradients under $-\log$.
  - *Short-Horizon Degradation*: On Push-T same25, graph plus switch plus $-\log V$ scored 77.6%, versus 91.2% for L2 (submitted ablation).
  - *Weight Sensitivity*: $-\log V$ required a completely different balancing weight $\beta$ for each protocol (e.g. $\beta=0.1$ for same25, $\beta=5.0$ for cross). No single hyperparameter configuration worked across tasks.
- **The Expected Hitting Time ($\mathbb{E}[T]$) Breakthrough**:
  - Replacing $-\log V$ with budget-capped expected hitting time $\mathbb{E}[T]$ completely eliminated the saturation and weighting instability:
  - *Linear Metric*: $\mathbb{E}[T]$ is expressed directly in environment steps (0 to 50 steps), matching the scale of physical time.
  - *Single Universal Weight*: A single weight $\beta = 1.0$ (with standard deviation normalization) works across all protocols: same25, same50, same100, and cross.
  - *Cross-Episode Performance*: Adding the ET critic to graph subgoals with the local switch raised Push-T cross-episode success from **17.2% to 42.4%**.

## 4. Synthesis & Next Questions
- **Established Fact**: A feasibility critic is indispensable for cross-episode planning with a frozen world model. The critic must be formulated as a bounded expected hitting time ($\mathbb{E}[T]$) rather than negative log-probability ($-\log V$) to prevent saturation and hyperparameter sensitivity.
- **The Remaining Short-Range Flaw**:
  - Even with $\mathbb{E}[T]$, evaluating the critic on short-range tasks (same25) still produced a slight degradation compared to vanilla L2.
  - *Root Cause*: Near the goal, the critic's spatial resolution (trained in 5-step chunks) is far too coarse compared to LeWM's smooth latent L2 gradient.
- **Resolution**: The planner must implement a **final-phase switch**: when within one lookahead horizon of the goal ($h \le 13.7$ TDR units), the graph and critic must yield control completely to pure goal L2. (Detailed in [[EXP-007]](EXP-007-gas-mpc-phases-to-our-method.md)).
