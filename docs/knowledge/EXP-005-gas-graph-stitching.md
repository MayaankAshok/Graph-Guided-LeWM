---
id: EXP-005
title: Offline Experience Graph (GAS Graph) Topology, TE Filtering, and Shortcuts
status: established
created: 2026-09-27
tags: [graph, dijkstra, subgoals, gas, clustering, topology]
derived_from: [EXP-004]
leads_to: [EXP-006, EXP-007, EXP-010, EXP-012]
opens_questions:
  - "Can continuous neural path planning (e.g. flow-based geodesics) replace discrete Dijkstra graph search?"
  - "How severely does graph sparsity in 3D contact domains degrade shortest path quality?"
---

# EXP-005: Offline Experience Graph (GAS Graph) Topology, TE Filtering, and Shortcuts

## 1. Motivation & Provenance
- **Source Documents**: `docs/iclr2027/main.pdf` Section 3.1 ("Temporal representation and graph"), Section 3.2, Appendix A (Algorithm 1: Graph-guided MPC, Algorithm 2: Offline sequential clustering and graph construction); `docs/gas-mpc/main.pdf` Section 2 ("Graph"), Section 6 (Phases 1 & 3).
- **Problem Statement**:
  While [[EXP-004]](EXP-004-tdr-calibration.md) established that TDR reliably measures temporal distance along known trajectories, it cannot navigate across disconnected episodes. 

  To bridge disparate offline experiences, Graph-Assisted Stitching (GAS) constructs a topological graph $G=(V, E)$ over offline dataset frames in TDR space. By finding the shortest path via Dijkstra's algorithm, the planner can extract an intermediate reachable **subgoal** $v^\star$ located exactly one planning horizon ahead ($H_{\text{TD}}$).

  This experiment evaluates: **how should the graph be constructed to avoid spurious shortcuts, and how effective is pure graph-subgoal guidance for LeWM's CEM controller?**

## 2. Experimental Protocol & Method
- **Script**: `scripts/gas_mpc/gas_mpc_prepare.py graph`
- **Output Artifacts**: `outputs/pusht/graph/graph_te0.9_htd13.7.pt`.
- **Graph Construction (GAS Algorithm 2)**:
  1. *Sequential TD Clustering*: Form node clusters sequentially at radius $r = H_{\text{TD}} / 2$. A new cluster is formed only when a frame's distance to all existing cluster medoids exceeds $r$.
  2. *Transition-Edge (TE) Filtering*: Filter candidate edges using a transition-edge classifier score $\theta_{\text{TE}} \ge 0.90$ to reject dynamically impossible transitions between nearby poses.
  3. *Edge Construction*: Place directed edges between cluster medoids within TDR distance $H_{\text{TD}}$, weighted by their TDR distance.
  4. *Query Attachment*: At inference, attach the current latent $z_0$ and goal latent $z_g$ as temporary nodes linked to all clusters within $\max(H_{\text{TD}}, 1.2 \cdot d_{\min})$.
- **Subgoal Objectives in CEM**:
  - `subgoal`: Picks node $v^\star$ along the Dijkstra path at cumulative TDR distance $\approx H_{\text{TD}}$, scores endpoints by $\|\hat{z}_5 - z_{\text{med}(v^\star)}\|^2$ in latent space.
  - `subgoal_tdr`: Same node $v^\star$, but scores endpoints by $\|\psi(\hat{z}_5) - \psi(z_{\text{med}(v^\star)})\|$ in TDR space.
  - `ctg`: Graph Cost-To-Go without committing to a single node: $\min_v (\|\psi(\hat{z}_5) - c_v\| + D_g[v])$.

## 3. Empirical Results
- **Graph Topology & Size (Push-T)**:
  - The submitted Push-T graph has about 20.5k nodes (`docs/iclr2027/main.tex`, `tab:hyper-env`).
  - On the filtered graph, 96.2% of cross-episode test pairs are connected within a Dijkstra horizon of 225 steps.
- **The Shortcut Catastrophe**:
  - If TE filtering is disabled or the connection radius is broadened beyond $H_{\text{TD}}$, false identity edges connect states where the robot is on opposite sides of the T-block.
  - This creates "ghost shortcuts": Dijkstra finds a 20-step path through an un-crossable collision, leading CEM into dead ends.
- **Planning Performance with Pure Graph Guidance**:
  - Submitted Push-T ablation (`docs/iclr2027/main.tex`, `tab:ablation`): graph subgoal without switch or critic reaches **51.6%** same50 and **15.2%** cross (L2: 40.8% / 14.4%). Adding the local switch reaches **62.4%** same50 and **17.2%** cross.

## 4. Synthesis & Next Questions
- **Established Fact**: Graph subgoals improve Push-T same50 from 40.8% to 51.6% without the local switch, or 62.4% with it. Cross improves only modestly before adding the critic.
- **Why Graph Subgoaling Alone Remains Weak on Cross-Episode**:
  - The graph identifies *what* state to aim for, but CEM's stochastic search frequently samples rollouts that reach the vicinity of the node without establishing the proper contact dynamics.
  - Without a feasibility check, the planner cannot distinguish between a rollout that moves the block towards the subgoal and a rollout that merely slips past the block.
- **Resolution**: CEM requires a **feasibility / reachability critic** to filter out un-achievable endpoints and prioritize dynamically viable trajectories. (Addressed in [[EXP-006]](EXP-006-hitting-time-critic.md)).
