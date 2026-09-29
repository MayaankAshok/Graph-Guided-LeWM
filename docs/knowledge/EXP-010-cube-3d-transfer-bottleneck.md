---
id: EXP-010
title: 3D Contact Manipulation (Cube) Transfer Bottleneck and Topological Disconnects
status: established
created: 2026-09-27
tags: [cube, 3d-manipulation, ogbench, transfer-limits, graph-sparsity, contact-physics]
derived_from: [EXP-005, EXP-007]
leads_to: [EXP-012, EXP-015]
opens_questions:
  - "Why does explicit graph search fail in high-dimensional 3D contact domains?"
  - "Can continuous generative planners (e.g. flow matching on latent trajectories) overcome the exponential sparsity of discrete graphs in 3D manipulation?"
---

# EXP-010: 3D Contact Manipulation (Cube) Transfer Bottleneck and Topological Disconnects

## 1. Motivation & Provenance
- **Source Documents**: `docs/iclr2027/main.pdf` Section 5.2 ("Cube Benchmark"), Appendix B.2 (Table 4: "Cube ablation ladder"); `docs/gas-mpc/main.pdf` Section 8 ("Cube: OGBench single cube"), Section 12 ("Cube diagnosis/tuning sweep"), and `docs/gas-mpc/eval-inventory-cube.md`.
- **Problem Statement**:
  In 2D benchmarks, OUR Method improved cross-episode success over vanilla LeWM by 28.0 percentage points on Push-T and 36.4 on Reacher ([[EXP-007]](EXP-007-gas-mpc-phases-to-our-method.md)).

  However, when transferred to 3D robotic manipulation (the OGBench **Cube** task, requiring a 7-DoF Franka arm to grasp, lift, and reposition a cube to an arbitrary 3D target), the improvement was much smaller: from **16.4% to 27.6%**. 

  More alarmingly, ablation studies revealed that **the graph search and reachability critic provided virtually zero marginal benefit over pure TDR** on Cube. 

  This experiment investigates: **why does the graph-guided MPC paradigm fail to transfer effectively to 3D contact-rich manipulation?**

## 2. Experimental Protocol & Method
- **Benchmark**: OGBench `cube-single-expert-v0` (46.2 GB dataset, 7-DoF arm, 6-DoF cube pose, vision-based RGB inputs).
- **Records**: `docs/gas-mpc/eval-inventory-cube.md`, `results_cube.json`.
- **Historical driver** (retired from the former flat scripts folder): `ada_gas_mpc_ladder_cube.sh`.
- **Ablation Protocol**: Evaluate Vanilla L2, Pure TDR (`tdr`), Pure Graph Subgoals (`subgoal_tdr`), and OUR Method across 5 seeds on the fixed `task200u` pool (250 episodes per protocol).

## 3. Empirical Results
- **The Cube Performance Ladder (Cross-Episode)**:
  - Vanilla LeWM (L2): **16.4%**
  - Pure TDR (`tdr`, no graph, no critic): **36.8%** (+20.4 points over L2)
  - Graph subgoal (no switch or critic): **28.4%**
  - Graph subgoal + local switch + ET critic (OUR Method): **27.6%**

> **Critical Empirical Finding**: In the submitted Cube ablation (`docs/iclr2027/main.tex`, `tab:ablation-cube`), TDR alone outperforms OUR on cross-episode tasks (36.8% vs 27.6%). The graph and critic do not improve that protocol.

- **Root Causes of the 3D Transfer Bottleneck**:
  1. *Exponential Curse of Dimensionality*: 
     - Push-T state space is effectively 5-dimensional ($x, y, \theta$ of block + $x, y$ of agent).
     - Cube state space is 13-dimensional ($7$ robot joint positions + $6$-DoF cube pose). 
     - The submitted graphs have 20.5k Push-T nodes and 16k Cube nodes; graph connectivity and usefulness differ across these state spaces.
  2. *Thin Contact Manifolds & Non-Euclidean Barriers*:
     - To move a cube, the gripper fingers must close within millimeters of the cube surface. The manifold of grasping states occupies near-zero measure in the state space.
     - Two frames where the gripper is 2 cm apart can look nearly identical in latent space, yet one is a secure grasp while the other is an empty grasp. 
     - TDR and identity edges blur across this micro-contact barrier, creating graph shortcuts that assume the cube can be lifted without a pre-grasp approach.
  3. *Unconnected Query Attachment*:
     - In 3D manipulation, test goal states are frequently located $> 2 \times H_{\text{TD}}$ away from any training node in TDR space. 
     - As a result, the goal fails to attach to the graph, leaving Dijkstra search disconnected and forcing the planner to fall back to unguided search.

## 4. Synthesis & Next Questions
- **Established Fact**: Explicit discrete graph construction (GAS-style) **does not scale** from planar navigation to 3D multi-body contact manipulation. Discrete offline nodes cannot capture the continuous, thin contact manifolds required for grasping.
- **Strategic Implications for Radical Rethink**:
  - The entire complexity of maintaining an offline graph (clustering, Dijkstra search, node attachment, TE classifiers) yields almost no return in high-dimensional 3D spaces.
  - The only component that transferred robustly was the continuous metric space ($\psi$).
  - **Conclusion**: We must abandon explicit discrete graph stitching in favor of continuous generative trajectory models or implicit quasimetric energy models that naturally interpolate through high-dimensional continuous contact spaces without discrete nodes.
