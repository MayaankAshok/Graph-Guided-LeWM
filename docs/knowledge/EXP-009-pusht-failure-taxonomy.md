---
id: EXP-009
title: Push-T Failure Taxonomy and Under-Actuated Contact Overshoot
status: established
created: 2026-09-27
tags: [failure-analysis, pusht, contact-dynamics, under-actuated, overshoot]
derived_from: [EXP-007, EXP-008]
leads_to: [EXP-013]
opens_questions:
  - "Why does CEM fail to arrest momentum near the goal, causing 57% of close approaches to regress?"
  - "Can a learned reactive terminal controller lock the object into the goal zone once within tolerance?"
---

# EXP-009: Push-T Failure Taxonomy and Under-Actuated Contact Overshoot

## 1. Motivation & Provenance
- **Source Documents**: `docs/iclr2027/main.pdf` Section 6 ("Limitations"); `docs/gas-mpc/pusht-failure-mode-study.md` and `docs/gas-mpc/pusht-failure-analysis.json`.
- **Problem Statement**:
  Even though OUR Method elevated cross-episode Push-T success from 14.4% to 42.4% ([[EXP-007]](EXP-007-gas-mpc-phases-to-our-method.md)), **57.6% of cross-episode tasks and 53.6% of same100 tasks still fail**.

  To understand what prevents further progress, this study (conducted in `docs/gas-mpc/pusht-failure-mode-study.md`) decomposes failing trajectories across all 4 protocols into sub-components: agent position, block position, and block orientation.

  This study asks: **what is the physical mechanism of failure when OUR Method fails? Does the planner get lost, orient incorrectly, abandon the block, or run out of precision?**

## 2. Experimental Protocol & Method
- **Source Artifacts**: `docs/gas-mpc/pusht-failure-mode-study.md`, `docs/gas-mpc/pusht-failure-analysis.json`, and per-task `_traj.npz` files (`outputs/pusht/eval/`).
- **Data**: Full state trajectories from all 250 evaluation episodes per protocol across 5 seeds on the fixed `task200u` pool.
- **Push-T Success Predicate**:
  $$\text{Success} \iff \|\text{agent} - \text{goal}_{\text{agent}}\| < 20\text{px} \land \|\text{block} - \text{goal}_{\text{block}}\| < 20\text{px} \land |\theta_{\text{block}} - \theta_{\text{goal}}| < 20^\circ$$
  Both position and angle tolerances must be satisfied simultaneously.
- **Decomposition**: Trajectories were inspected at their single closest approach (minimum combined error) and at episode termination.

## 3. Empirical Results
- **Headline Failure Taxonomy (Cross-Episode, $n=250$)**:
  - **Success**: 42.4% (106 episodes)
  - **Close but not enough**: 47.6% (119 episodes) — the agent and block are both near the target, but never satisfy both position and angle thresholds simultaneously.
  - **Block never gets close**: 5.6% (14 episodes) — minimum block error $> 60\text{px}$.
  - **Orientation-only miss**: 2.4% (6 episodes) — position satisfied, angle exceeds $20^\circ$.
  - **Agent abandons block**: 2.0% (5 episodes) — agent reaches target, block left behind.

> **Primary Takeaway**: Failure is overwhelmingly **"ran out of precision near the goal" (47.6%)**, NOT getting lost or giving up on the block.

- **The Primary Bottleneck: "Block Off, Agent OK"**:
  At the single closest approach of each failing episode:
  - In cross-episode: **54.2%** of failures have the agent within tolerance, but the block slightly outside.
  - In same50: **54.7%** of failures have the agent within tolerance, block outside.
  - Push-T is fundamentally under-actuated: the agent controls the block only through unilateral contact. Reaching the agent goal pose is easy, but precisely finishing the push requires subtle micro-adjustments that open-loop 5-action blocks cannot deliver.

- **The Non-Monotonic Regression Paradox (Overshoot)**:
  Comparing each failing episode's best-ever error to its final-step error:
  - In same100 tasks: **57.5% (77 of 134 failing episodes) got $\ge 15\text{px}$ closer earlier, then ended worse!**
  - In same25 tasks: 34.5% regressed after getting close.
  - In cross tasks: median best-ever error was $48\text{px}$, but final error drifted to $82\text{px}$.
  - **The Bumping Failure**: The robot successfully navigates the block into the near-goal zone, but subsequent receding-horizon plan blocks maintain forward momentum or attempt to adjust pose, accidentally knocking the block away from the target zone!

## 4. Synthesis & Next Questions
- **Established Fact**: Long-horizon graph-guided MPC successfully solves global navigation across the arena, but suffers from an **end-game precision and momentum bottleneck**.
- **Implication**:
  - CEM sampling with 5-step receding horizon blocks is too coarse for micro-manipulation. When executing 5 actions open-loop, inertia causes the end-effector to plow through the block.
  - The controller lacks an active "stopping or settling" mechanism once near the target.
- **Future Vectors**:
  1. Could a closed-loop reactive terminal policy (e.g. trained specifically for local insertion/settling) take over once within $15\text{px}$?
  2. Does a flow-matching or continuous trajectory optimization approach generate smoother, non-overshooting terminal decelerations?
