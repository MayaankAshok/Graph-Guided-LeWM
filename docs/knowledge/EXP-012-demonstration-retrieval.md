---
id: EXP-012
title: Demonstration Retrieval and Nearest-Neighbor Action Seeding
status: established
created: 2026-09-27
tags: [retrieval, action-seeding, warm-start, ogbench-cube, cem-proposals]
derived_from: [EXP-005, EXP-010]
leads_to: []
opens_questions:
  - "Why does demonstration retrieval provide substantial lift on Cube but zero benefit on Push-T?"
  - "Can retrieval-augmented CEM be replaced by an amortized actor policy?"
---

# EXP-012: Demonstration Retrieval and Nearest-Neighbor Action Seeding

## 1. Motivation & Provenance
- **Source Documents**: `docs/iclr2027/main.pdf` Appendix C ("Demonstration retrieval"); `docs/gas-mpc/main.pdf` Section 5.1–5.4 ("Does retrieval's action quality explain why it only helps Cube?"), Section 12.
- **Problem Statement**:
  Standard CEM samples action candidate blocks from a zero-mean Gaussian distribution or the previous iteration's shifted elite distribution. In complex manipulation tasks with narrow bottlenecks (such as grasping a cube in 3D space, [[EXP-010]](EXP-010-cube-3d-transfer-bottleneck.md)), randomly sampling an effective grasp action sequence from scratch has near-zero probability.

  To address this, the `gas-mpc` pipeline investigated **Demonstration Retrieval**: finding the nearest offline graph node $v_{\text{NN}}$ in TDR space and injecting its logged expert demonstration actions into CEM's initial candidate population as a "warm-start" proposal.

## 2. Experimental Protocol & Method
- **Historical drivers** (retired from the former flat scripts folder): `retrieval_sensitivity_diag.py`, `retrieval_sensitivity_real_diag.py`, `ada_pusht_retrieval_full.sh`, `ada_reacher_retrieval_full.sh`.
- **Mechanism**:
  - At each plan call, query the FAISS index to find the $k$ nearest training frame embeddings to the current latent $z_0$.
  - Extract the 5-action block executed by the expert following that retrieved frame.
  - Seed 10% to 20% of CEM's initial 300 candidate trajectories with the retrieved action sequence perturbed by small Gaussian noise.
- **Evaluation**: Tested across Push-T, Reacher, and Cube on the `task200u` pool.

## 3. Empirical Results
- **The Divergent Effect Across Environments**:
  - **OGBench Cube**:
    - Adding retrieval to OUR Method lifts cross-episode success from **27.6% to 48.0%** (+20.4 points).
    - In same25, retrieval lifts success from 62.0% to 86.4%.
    - Retrieval provides the critical initial velocity sequence that guides the gripper into contact with the cube.
  - **Push-T**:
    - Adding retrieval leaves cross-episode success at **42.4%** in the submitted table.
    - Replaying retrieved actions produced higher selection regret than pure random CEM.
  - **Reacher**:
    - Slightly negative: cross-episode 65.6% vs 63.6% with retrieval.

- **Mechanistic Cause (The Local Action Invariance Test)**:
  - Audits in `docs/gas-mpc/main.tex` tested retrieval's core assumption: *does executing the retrieved action from a nearby latent $z_0 \approx z_{\text{NN}}$ produce the expected next latent?*
  - In Push-T, two states that are close in latent space often differ slightly in relative agent-block contact angle. Replaying the demonstrator's action when contact is slightly shifted causes the agent to slip off the T-block, resulting in severe dynamic divergence.
  - In Cube, the arm is in free space during approach; small latent differences do not cause catastrophic collisions before the grasp occurs.

## 4. Synthesis & Next Questions
- **Established Fact**: Retrieval helps Cube strongly in the submitted benchmark, while it does not improve Push-T cross and slightly reduces Reacher cross success.
- **Simplification Takeaway**: Retrieval adds FAISS search and action-noise tuning; a simpler planner should be compared against the Cube gain before discarding it.
