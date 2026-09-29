---
id: EXP-017
title: Theoretical Guarantees: Replanning Bounds, Local Convergence, and the Exact-Model L2 Trap
status: established
created: 2026-09-27
tags: [theory, proofs, replanning-bounds, bellman-contraction, exact-model-trap, iclr-theorems]
derived_from: [EXP-002, EXP-003, EXP-006, EXP-007]
leads_to: []
opens_questions:
  - "Can theoretical progress bounds be established for continuous generative trajectory planners without discrete graph assumptions?"
---

# EXP-017: Theoretical Guarantees: Replanning Bounds, Local Convergence, and the Exact-Model L2 Trap

## 1. Motivation & Provenance
- **Source Documents**: `docs/iclr2027/main.pdf` Section 4 ("Theory: from the planning cost to goal progress"), Appendix J ("Detailed theory and proofs": Lemmas 5–14, Propositions 9–17, Theorems 1–3).
- **Problem Statement**:
  To establish rigorous mathematical foundations for graph-guided MPC with frozen world models, the ICLR 2027 manuscript derived formal theoretical bounds on replanning complexity, convergence, and representation limits.

  This document synthesizes the formal mathematical theorems, error propagation bounds, and the fundamental **Exact-Model L2 Trap** proven in the submission.

## 2. Experimental Protocol & Method
- **Theoretical Setting (`docs/iclr2027/main.pdf` Section 4 & Appendix J.1)**: Continuous state space $\mathcal{S}$, compact action space $\mathcal{A}$, deterministic transition dynamics $s_{t+1} = f(s_t, a_t)$, and observation mapping $o_t = \phi(s_t)$.
- **Proof Machinery**:
  - Lyapunov stability and contraction analysis in latent space.
  - Finite-horizon dynamic programming and Bellman non-expansiveness over discrete horizon lattices.
  - Worst-case geometric curvature bounds for non-convex obstacle manifolds.

## 3. Empirical Results
The formal theoretical guarantees established in `docs/iclr2027/main.pdf` Section 4 & Appendix J:

### Theorem 1: Normalized-Cost Graph Progress (Replanning Complexity Bound)
Let $D_g(v_0)$ be the initial Dijkstra graph distance to the goal, $r_{\text{term}}$ be the terminal switching radius ($H_{\text{TD}}$), $\Delta$ be the nominal step progress per lookahead horizon, and let errors be bounded by:
- Predictor model error: $\sup \|\hat{z}_H - z_{t+H}\| \le \epsilon_m$
- CEM sampling search error: $\epsilon_s$
- Environment stochastic execution error: $\epsilon_{\text{exec}}$
- Critic ranking error: $\epsilon_c$

If the combined error condition satisfies $\epsilon_{\text{total}} = \epsilon_m + \epsilon_s + \epsilon_{\text{exec}} + \beta \epsilon_c < \Delta$, then the system enters the terminal goal region within a bounded number of replanning calls:
$$K \le \left\lceil \frac{D_g(v_0) - r_{\text{term}}}{\Delta - \epsilon_{\text{total}}} \right\rceil$$

### Theorem 2: Local Convergence After the Switch
Once the state enters the terminal radius ($\|h(z_0) - h(z_g)\| \le r_{\text{term}}$) and the controller switches to pure goal L2 (`final_metric=l2`), if the local dynamics Jacobian satisfies a local contractivity condition in latent space with margin $\mu > 0$, the latent distance to the goal contracts exponentially:
$$\|z_{t+k\cdot H} - z_g\|_2 \le (1 - \mu)^k \|z_{\text{term}} - z_g\|_2 + \frac{\epsilon_m + \epsilon_s}{\mu}$$
guaranteeing precise local alignment within the environment's tolerance predicate.

### Theorem 3: Finite-Horizon Critic Error Propagation
The multi-horizon reachability critic operator $\mathcal{T}$ applied across discrete horizons $h \in \{0, 5, \dots, h_{\text{max}}\}$ is a $\gamma$-contraction in supremum norm:
$$\|\mathcal{T} V_1 - \mathcal{T} V_2\|_\infty \le \gamma \|V_1 - V_2\|_\infty$$
Under Bellman updates with frozen predictor rollouts, the propagation of valuation error across a trajectory of length $H$ is bounded by $\frac{\epsilon_{\text{Bellman}}}{1 - \gamma} (1 - \gamma^H)$.

### Proposition: The Exact-Model L2 Trap
A foundational theoretical question is: *is the failure of terminal L2 planning solely due to model prediction error?*

The paper formally proves **NO**:
> **Proposition (Exact-Model L2 Trap)**: There exist smooth, deterministic dynamical systems where the world model is mathematically exact ($\epsilon_m = 0$) and the action optimizer is globally optimal ($\epsilon_s = 0$), yet terminal latent L2 planning at any fixed horizon $H$ gets trapped in non-goal stationary points or limit cycles.

*Proof Intuition*: On any non-convex obstacle manifold (e.g. pushing a block around a barrier), the gradient of the Euclidean distance to the goal points directly into the obstacle. A local controller with horizon $H < H_{\text{obstacle}}$ cannot see past the obstacle, creating an un-crossable local minimum regardless of model accuracy.

### Proposition: Survival Score as Truncated Expected Hitting Time
The paper establishes the exact conditions under which the discrete survival sum matches physical time:
If $V(z, z_g, h) = \mathbb{P}(T \le h \mid z, z_g)$ is well-calibrated, then:
$$\sum_{j=0}^{h_{\text{rem}}} \big(1 - V(z, z_g, j)\big) \cdot \Delta t = \mathbb{E}\left[ \min(T, h_{\text{rem}}) \mid z, z_g \right]$$
proving that the budget-capped ET critic measures the true conditional truncated expectation of remaining time.

## 4. Synthesis & Next Questions
- **Established Fact**: The failure of standard world-model planning is fundamentally structural, not merely a symptom of inaccurate world models. Topological guidance is mathematically required for non-convex tasks.
- **Guideline for Simplification**: Any new unified architecture must preserve the global non-convex routing capability proven in Theorem 1 while avoiding the fragile multi-component error accumulation ($\epsilon_m + \epsilon_s + \epsilon_{\text{exec}} + \beta \epsilon_c$).
