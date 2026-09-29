---
id: EXP-013
title: Extended Budget (750-step) Plateau and Limit Cycles
status: established
created: 2026-09-27
tags: [extended-budget, limit-cycles, pusht, asymptotic-performance, failure-modes]
derived_from: [EXP-007, EXP-009]
leads_to: []
opens_questions:
  - "Why does tripling the execution budget fail to resolve failing episodes?"
  - "How can long-horizon planners break out of cyclical attractor orbits in latent space?"
---

# EXP-013: Extended Budget (750-step) Plateau and Limit Cycles

## 1. Motivation & Provenance
- **Source Documents**: `docs/gas-mpc/main.pdf` Section 5.8 ("Cross-episode with a 750-step budget", Ada job 2712749, recorded 2026-09-23); `docs/iclr2027/main.pdf` Section 6 ("Limitations").
- **Problem Statement**:
  A common conjecture in long-horizon planning is that failing episodes merely "ran out of time" under the standard evaluation budget ($B = 250$ steps on Push-T). If true, simply extending the execution horizon should allow the receding-horizon controller to eventually maneuver the block to the goal.

  To test this hypothesis, an experiment was conducted on Ada (job `2712749`, recorded in `docs/gas-mpc/main.tex` Section 5.8), tripling the budget from 250 to 750 steps on Push-T cross-episode tasks.

## 2. Experimental Protocol & Method
- **Script / Run**: Ada job `2712749`, `docs/gas-mpc/main.tex` Section 5.8.
- **Configuration**: OUR Method (`subgoal_tdr` + `final_metric=l2` + `final_thresh=13.7` + budget-capped `ET` critic + $\beta=1.0$).
- **Protocol**: Push-T cross-episode tasks on the fixed `task200u` pool, running for a maximum of **750 environment steps** (30 replanning calls $\times$ 25 steps) instead of the standard 250 steps.

## 3. Empirical Results
- **Success Rate Comparison (Push-T Cross)**:
  - Standard Budget (250 steps): **42.4%**
  - Extended Budget (750 steps): **44.0%**
  - **Marginal Gain**: Tripling the budget yielded only **+1.6%** (4 additional successful episodes out of 250).

- **The Limit Cycle Phenomenon**:
  - Trajectory analysis of failing episodes revealed that the controller does not steadily make progress towards the goal.
  - Instead, after initial progress stalls (usually around steps 150–200), the agent enters **stable limit cycles**:
    1. *The Orbiting Trap*: The robot end-effector circles around the block without making contact, because any contact perturbation increases the predicted cost under the current subgoal.
    2. *Subgoal Bouncing*: The Dijkstra shortest path fluctuates between two alternative graph branches at successive replanning intervals, causing the agent to oscillate back and forth between two waypoints.
    3. *The Pushed-Away Wall Trap*: The block gets jammed against the arena boundary or corner, where offline demonstrations rarely venture. In these out-of-distribution physical configurations, the world model predictor predicts no valid escape actions.

## 4. Synthesis & Next Questions
- **Established Fact**: Long-horizon failure in Graph-MPC is **asymptotic, not temporal**. The remaining 56% failure rate on Push-T cross cannot be resolved by allocating more execution time.
- **Key Takeaway**: Failures are caused by structural attractors (limit cycles, OOD contact states, and discrete graph oscillations). Progress requires breaking these topological deadlocks through continuous stochastic exploration or reactive recovery policies.
