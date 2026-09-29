---
id: EXP-004
title: Temporal Distance Representation (TDR) Mapping and Metric Calibration
status: established
created: 2026-09-27
tags: [tdr, metric-learning, calibration, gas, representations]
derived_from: [EXP-002, EXP-003]
leads_to: [EXP-005, EXP-007]
opens_questions:
  - "Can TDR be trained end-to-end with the world model encoder instead of as a post-hoc MLP on frozen latents?"
  - "Why does pure TDR objective without a graph fail on cross-episode tasks?"
---

# EXP-004: Temporal Distance Representation (TDR) Mapping and Metric Calibration

## 1. Motivation & Provenance
- **Source Documents**: `docs/iclr2027/main.pdf` Section 3.1 ("Temporal representation and graph"), Appendix H, Appendix I (Table 7); `docs/gas-mpc/main.pdf` Section 2 ("TDR"), Section 6 (Phases 1–2).
- **Problem Statement**:
  Because [[EXP-002]](EXP-002-gaussian-shell-ood.md) showed that Euclidean latent distance $\|\hat{z} - z_g\|$ drives the planner off the data shell, and [[EXP-003]](EXP-003-lewm-l2-baseline.md) demonstrated that vanilla L2 fails beyond 25 steps, we require a distance metric where Euclidean distance directly reflects the **minimum dynamical time-to-reach**.

  Following Graph-Assisted Stitching (GAS, Baek et al., ICML 2025), we learn a Temporal Distance Representation (TDR) mapping $\psi(z)$, projecting frozen LeWM latents into a metric space where $\|\psi(s) - \psi(g)\|_2 \approx \text{dynamical step distance}$.

## 2. Experimental Protocol & Method
- **Script**: `scripts/gas_mpc/gas_mpc_prepare.py tdr`
- **Output Artifacts**: `outputs/pusht/tdr/tdr_s0.pt`, `outputs/pusht/psi_train_s0.npy`.
- **Architecture**: $\psi(z): \mathbb{R}^{192} \to \mathbb{R}^{32}$, a 3-layer MLP (hidden dim 512, LayerNorm, GELU activations).
- **Training Loss**: HILP / GAS expectile temporal difference loss:
  $$\mathcal{L}_{\text{TD}}(\psi) = \mathbb{E}\left[ |\tau - \mathbf{1}_{\delta < 0}| \cdot \delta^2 \right], \quad \delta = 1 + \gamma \min_{i \in \{1, 2\}} \|\psi_i'(s') - \psi_i'(g)\| - \|\psi(s) - \psi(g)\|$$
  with expectile $\tau = 0.99$, discount $\gamma = 0.99$, and twin target networks.
- **Relabeling**: 62.5% same-trajectory future goals, 37.5% uniform random dataset goals. Trained for 50,000 batches of size 1024 on `cache_train.npz` (held-out evaluation episodes strictly excluded).
- **Calibration Probe**:
  - Compute pairwise $\|\psi(z_t) - \psi(z_{t+\Delta})\|$ across logged trajectories for step offsets $\Delta \in [1, 200]$.
  - Measure Spearman rank correlation against true environment step differences and identify the median TDR distance corresponding to 1 planning horizon.

## 3. Empirical Results
- **Rank Correlation**:
  - Push-T: Spearman correlation between $\|\psi(s) - \psi(g)\|$ and ground-truth step distance is **0.94** on held-out episodes (vs. 0.61 for raw latent L2).
  - Reacher: Spearman correlation is **0.98**.
- **Submitted environment-specific scales** (`docs/iclr2027/main.tex`, `tab:hyper-env`): graph threshold $H_{\text{TD}}=u_{12}$ is 7.59 / 3.72 / 7.25 TDR units for Push-T / Reacher / Cube. The 25-step lookahead and local switch radius $\lambda=u_{25}$ is 13.53 / 5.83 / 13.63, respectively. These quantities serve different roles and must be calibrated per environment.
- **Pure TDR Planning (`tdr` objective)**:
  - In `gas_mpc_eval.py +mpc.method=tdr`, scoring candidate rollout endpoints by $\|\psi(\hat{z}_5) - \psi(z_{\text{goal}})\|$:
  - On Push-T same50: improves from 40.8% (L2) to 48.4%.
  - On Push-T cross: reaches **14.0%** (vs 14.4% for L2).

## 4. Synthesis & Next Questions
- **Established Fact**: TDR successfully straightens the curved latent manifold into a 32-dimensional quasi-metric space that linearly scales with true transition time up to ~50 steps.
- **Why Pure TDR Fails on Cross-Episode Goals**:
  - TDR is trained on finite continuous trajectories. For two states from completely different episodes with no shared trajectory, the triangle inequality in TDR space does not guarantee an unobstructed path—there may be obstacles, irreversible state changes, or contact boundaries between them.
  - Merely knowing the distance to the goal does not tell CEM *which direction to push the block*.
- **Resolution**: We must stitch offline trajectories into an explicit **topological graph** to find valid intermediate subgoals across episodes. (Evaluated in [[EXP-005]](EXP-005-gas-graph-stitching.md)).
