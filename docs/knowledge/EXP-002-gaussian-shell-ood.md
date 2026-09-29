---
id: EXP-002
title: Latent Space Geometry and Gaussian Shell OOD Drift
status: established
created: 2026-09-27
tags: [geometry, gaussian-shells, ood, latent-space, cursed-dimensions]
derived_from: [EXP-001]
leads_to: [EXP-003, EXP-004, EXP-017]
opens_questions:
  - "Can a non-Euclidean latent distance metric enforce geodesics along the shell without an explicit offline graph?"
  - "Does normalizing latent representations to the unit hypersphere (e.g. cosine distance) prevent the interior collapse?"
---

# EXP-002: Latent Space Geometry and Gaussian Shell OOD Drift

## 1. Motivation & Provenance
- **Source Documents**: `docs/iclr2027/main.pdf` Section 1, Appendix F.1 (Lemma 1, Proposition 4), Appendix F.3, and Appendix J.9 ("Why a Gaussian shell is not a failure theorem"); `docs/gas-mpc/main.pdf` Section 2.
- **Problem Statement**:
  Because [[EXP-001]](EXP-001-predictor-drift.md) proved that multi-step rollouts beyond 25 steps diverge in relative accuracy, long-horizon tasks must be approached by chaining multiple 25-step receding-horizon segments ($N \times 25$ steps). 

  However, when CEM optimizes terminal L2 distance to a distant goal $\|\hat{z}_{25} - z_{\text{goal}}\|^2$, each 25-step plan greedily chooses actions that aim the predicted endpoint directly along the straight line connecting the current latent $z_0$ to $z_{\text{goal}}$. 

  This experiment investigates: **what is the geometric structure of the LeWM latent space, and what happens to the dynamics model when forced to plan along straight Euclidean chords between distant states?**

## 2. Experimental Protocol & Method
- **Scripts**: `scripts/paper/diagnostics/gas_mpc_gaussian_shell_diag.py`, `scripts/paper/diagnostics/gas_mpc_gaussian_shell_interp_diag.py`
- **Figures / Artifacts**: `docs/iclr2027/figures/fig_gaussian_shell_r_hist.png`
- **Data**: 2,336,736 encoded Push-T frames (192-dimensional ViT CLS tokens, `outputs/pusht/cache_train.npz`).
- **Mathematical Framework (`docs/iclr2027/main.pdf` Appendix F.1)**:
  - *Lemma 1 (Distance Between Random Pairs)*: For independent $z_1, z_2 \sim \mathcal{N}(\mu, \sigma^2 I_d)$, $\mathbb{E}\|z_1 - z_2\| = \sqrt{2} \mathbb{E}\|z - \mu\|$. For $d=192$, the ratio of standard deviation to mean is $< 7.3\%$.
  - *Proposition 4 (Gaussian Shell)*: Mass concentrates in a thin spherical shell of radius $R \approx \sigma \sqrt{d}$, with interior probability $\Pr[r \le (1-\varepsilon)\sigma\sqrt{d}] \le \exp(-d\varepsilon^2/4)$.
  - *Midpoint Chordal Drop*: The midpoint of two independent shell samples satisfies $\frac{1}{2}(z_1 + z_2) - \mu \sim \mathcal{N}(0, \frac{\sigma^2}{2}I_d)$, concentrating at radius $\sigma \sqrt{d/2} = R / \sqrt{2} \approx 0.707 R$.

## 3. Empirical Results
The verified empirical measurements from `docs/iclr2027/main.pdf` (Appendix F.1–F.3):
- **Measured Shell Radius**:
  - The empirical norm distribution of Push-T latents has mean radius $R = \mathbf{13.88}$ (line 1092).
  - The measured random-pair distance is $\mathbf{19.60 \pm 2.43}$, matching the theoretical prediction $\sqrt{2} \times 13.88 \approx \mathbf{19.63}$.
- **Midpoint Collapse to the Empty Interior**:
  - Linear chords connecting distant states ($z_0 \to z_g$) plunge through the center of the sphere.
  - At the midpoint $\alpha = 0.5$, the latent norm drops by a factor of $1/\sqrt{2}$ to $\|z(0.5)\|_2 \approx \mathbf{9.81}$ (and for orthogonal cross-episode vectors, drops to $7.2 \pm 1.1$).
  - In 192 dimensions, the probability of sampling a training state with norm $\le 9.81$ is bounded by $\exp(-192 \times (1 - 1/\sqrt{2})^2 / 4) < 10^{-11}$. The interior is an empty void.
- **Predictor OOD Collapse**:
  - When CEM optimizes $\|\hat{z}_5 - z_g\|^2$, it forces the predictor to propose rollouts that move along the chord toward the interior.
  - In this interior void where latent norms are 30–50% smaller than any training data, the predictor dynamics break down, producing stagnant or non-physical predictions.

## 4. Synthesis & Next Questions
- **Established Fact**: Minimizing terminal Euclidean distance in high-dimensional latent space inherently induces Out-Of-Distribution (OOD) drift because Euclidean chords cut through the empty interior of the data shell.
- **Distinction from Failure Theorems (`docs/iclr2027/main.pdf` Appendix J.9)**:
  - As noted in Appendix J.9, a shell geometry alone is not a failure theorem (arc length and chord distance can preserve ranking on a sphere). Rather, the failure occurs because the world model predictor is not trained on interior points, causing dynamics extrapolation failure during optimization.
- **Implication**: Planning must be guided along the **surface of the manifold** via geodesic graph subgoals ([[EXP-005]](EXP-005-gas-graph-stitching.md)) or metric representations ([[EXP-004]](EXP-004-tdr-calibration.md)).
- **Open Questions Arising**:
  1. How much does this geometric failure explain vanilla LeWM's catastrophic collapse on cross-episode goals? (Addressed in [[EXP-003]](EXP-003-lewm-l2-baseline.md)).
  2. Can a Temporal Distance Representation (TDR) project this geometry into a space where Euclidean distances reflect true dynamical step distances? (Addressed in [[EXP-004]](EXP-004-tdr-calibration.md)).
