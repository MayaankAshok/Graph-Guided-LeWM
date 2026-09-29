---
id: EXP-008
title: Short-Horizon Regression and Boundary Switching Discontinuities
status: established
created: 2026-09-27
tags: [failure-modes, short-horizon, switching-thresholds, cem-chattering, continuity]
derived_from: [EXP-006, EXP-007]
leads_to: [EXP-009]
opens_questions:
  - "Can a smooth homotopy / potential blending function replace the hard threshold switch at theta_final?"
  - "Does a single continuous flow or diffusion policy eliminate the boundary chattering entirely?"
---

# EXP-008: Short-Horizon Regression and Boundary Switching Discontinuities

## 1. Motivation & Provenance
- **Source Documents**: `docs/iclr2027/main.pdf` Section 5.2 (Table 1: same25 performance rows), Section 6 ("Limitations"); `docs/gas-mpc/main.pdf` Section 5.6–5.7 ("Live CEM trace"), Section 6 (Phase 6: "short-range gap", Phase 7: "the final-phase switch", Phase 8).
- **Problem Statement**:
  Throughout all phases of `gas-mpc` development, a persistent and vexing pathology was observed: **incorporating global graph subgoals and reachability critics consistently degraded short-horizon performance on same25 tasks compared to vanilla LeWM L2**.

  In the submitted Push-T ablation, graph plus switch plus $-\log V$ scores 77.6% same25 versus 91.2% for L2. OUR Method with the ET critic and local switch still trails L2 on Push-T and Reacher same25:
  - Push-T same25: **88.4% $\pm$ 2.3** vs. **91.2% $\pm$ 1.0** (L2)
  - Reacher same25: **76.0% $\pm$ 4.6** vs. **77.2% $\pm$ 4.8** (L2)
  - Cube same25: **62.0% $\pm$ 4.7** vs. **56.0% $\pm$ 3.8** (L2; OUR is higher)

  This experiment investigates: **why does global guidance harm local planning, and what are the failure mechanisms introduced by hard threshold switching?**

## 2. Experimental Protocol & Method
- **Scripts**: `scripts/paper/diagnostics/gas_mpc_cem_trace_audit.py`.
- **Protocol**:
  - Audit CEM optimization traces across 30 iterations for same25 tasks.
  - Track the mean and variance of candidate action sequences, the distribution of cost components (subgoal TDR, critic ET, and terminal L2), and the spatial trajectories of the predicted rollouts.
  - Compare the cost surface geometry of vanilla L2 vs. OUR composite objective near the goal boundary.

## 3. Empirical Results
- **Mechanism 1: Hard Threshold Chattering (The Discontinuity Trap)**:
  - When the agent starts near the switching boundary ($\|h(z_0) - h(z_g)\| \approx 13.7$), small candidate movements cause some rollouts to trigger the final L2 mode while other rollouts trigger the graph subgoal mode.
  - This splits the CEM elite sample set into two distinct clusters: one cluster pursuing the true goal directly, and another cluster aiming at an intermediate graph cluster medoid.
  - Averaging two distinct elite clusters in CEM results in a mean action sequence that executes neither strategy cleanly, causing the robot to hesitate or stall.
- **Mechanism 2: Discrete Node Lateral Wander**:
  - Graph cluster medoids are discrete data frames from offline demonstrations.
  - Even when a cluster medoid is along the general route to the goal, it is almost never exactly collinear with the continuous line-of-sight to the goal.
  - Minimizing distance to the discrete medoid causes the end-effector to swerve laterally away from the direct contact trajectory, wasting precious budget steps.
- **Mechanism 3: Coarse Critic Resolution Near Goal**:
  - The hitting-time critic is trained with temporal discretization (5-step bins up to $h_{\text{max}}=50$).
  - Push-T success requires extreme spatial precision: agent+block position error $< 20\text{px}$ and orientation error $< 20^\circ$.
  - At a remaining distance of 5 to 10 environment steps, the critic output is nearly flat or dominated by classification noise. Injecting this noisy signal into CEM corrupts the pristine gradient of latent L2 distance.

## 4. Synthesis & Next Questions
- **Established Fact**: Piecewise, multi-objective planning functions create non-convex discontinuities and chattering in sampling-based optimizers like CEM. Local precision is degraded whenever a discrete graph node or temporally quantized critic competes with direct goal alignment.
- **Design Implication**:
  - Handcrafted heuristic switching (e.g. `final_thresh=13.7`) is an unstable patch that treats the symptom rather than the disease.
  - A truly elegant long-horizon architecture must have **continuous scale consistency**: smoothly and automatically transitioning from topological guidance at long horizons to fine metric alignment at short horizons without discrete mode switches.
