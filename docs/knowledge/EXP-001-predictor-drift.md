---
id: EXP-001
title: Predictor Horizon Error Accumulation and Gaussian Noise Limit
status: established
created: 2026-09-27
tags: [dynamics, lewm-predictor, error-accumulation, bounds]
derived_from: []
leads_to: [EXP-002, EXP-003, EXP-011, EXP-014, EXP-016]
opens_questions:
  - "Can multi-step rollout drift be bounded by regularizing latent geometry instead of autoregressive rollout?"
  - "Does a single-step jump or flow-matching predictor suffer less variance accumulation than autoregressive 5-step block prediction?"
---

# EXP-001: Predictor Horizon Error Accumulation and Gaussian Noise Limit

## 1. Motivation & Provenance
In world-model-based MPC, the controller optimizes action sequences by rolling out the dynamics model in latent space over a planning horizon $H$. LeWorldModel (LeWM) uses a transformer-based latent predictor that consumes a 3-step latent history ($z_{t-2}, z_{t-1}, z_t$) and 3 action blocks to predict the next latent state $z_{t+1}$. 

When attempting long-horizon planning, a fundamental question is: **how quickly does the latent prediction diverge from ground truth as the horizon $H$ increases, and does it degrade gracefully or collapse to random noise?**

## 2. Experimental Protocol & Method
- **Script**: `scripts/paper/diagnostics/predictor_error_by_horizon.py`
- **Output Artifact**: `outputs/pusht/diagnostics/predictor_error_by_horizon_s0.json`, `docs/iclr2027/figures/fig_predictor_error_by_horizon.png`
- **Model**: Pretrained frozen LeWM checkpoint (`quentinll/lewm-pusht`), 192-dimensional CLS latent representation.
- **Data**: Training-only cache (`cache_train.npz`, 2,336,736 frames, 18,685 Push-T episodes; held-out evaluation episodes strictly excluded).
- **Protocol**:
  - Sample 3,000 start frames and roll the frozen predictor forward autoregressively under the real logged expert actions (z-scored).
  - Compare the predicted latent $\hat{z}_{t+H}$ against the true encoded latent $z_{t+H}$ at horizons $H \in \{5, 25, 50, 100\}$ environment steps (1, 5, 10, 20 predictor blocks).
  - Concurrently measure the true start-to-goal task distance $\|z_t - z_{t+H}\|_2$ between frames $H$ steps apart.
  - Draw a reference distribution of pairwise distances between uniformly random unrelated frames.

## 3. Empirical Results
The definitive numbers from `docs/iclr2027/main.pdf` (Table 5 / `tab:predictor-error-and-task-distance`) and `predictor_error_by_horizon_s0.json`:

| Horizon ($H$) | Predictor Error (Median $\pm$ Std) | Start $\to$ Goal Distance (Median $\pm$ Std) | Relative Error Ratio |
| :--- | :---: | :---: | :---: |
| **$H=5$ (1 block)** | $1.00 \pm 0.56$ | $5.29 \pm 2.76$ | 18.9% |
| **$H=25$ (5 blocks)** | $2.41 \pm 2.09$ | $17.06 \pm 4.19$ | 14.1% |
| **$H=50$ (10 blocks)** | $4.28 \pm 3.82$ | $19.45 \pm 3.29$ | 22.0% |
| **$H=100$ (20 blocks)** | $7.80 \pm 5.35$ | $19.60 \pm 2.43$ | 39.8% |
| **Random-Pair Reference** | - | $19.63 \pm 1.39$ | - |

### Key Takeaways from the Data:
1. **Graceful Degradation, Not Noise Collapse**:
   - The predictor rollout error stays **well below the scale of two unrelated frames (19.63)** even at $H=100$ ($7.80 \pm 5.35$ vs $19.63$). 
   - The predictor does *not* collapse to an uninformative Gaussian random guess; it retains directional mutual information with the true future state.
2. **Task Distance Saturation (The Gaussian Shell Scale)**:
   - In contrast, the true start-to-goal distance between frames in the same episode rapidly saturates: at $H=50$ it reaches $19.45$, and at $H=100$ it reaches $19.60$—becoming indistinguishable from completely random pairs ($19.63$).
   - This matches theoretical predictions for high-dimensional spherical shells ($\sqrt{2} \times 13.88 \approx 19.63$, [[EXP-002]](EXP-002-gaussian-shell-ood.md)).
3. **Growing Relative Error Ratio**:
   - Relative to the task scale, predictor error is only **14.1%** of the gap at $H=25$, but climbs to **39.8%** by $H=100$.
   - At $H=100$, predictor error consumes a massive fraction (~40%) of the distance that CEM is trying to optimize, severely disrupting terminal L2 gradient descent.

## 4. Synthesis & Next Questions
- **Established Fact**: Long-horizon rollout failure is driven by two coupled phenomena: (1) predictor error compounds to ~40% of task scale by step 100, and (2) task distance saturates to the random-pair sphere diameter (~19.6) past step 50, destroying Euclidean discrimination.
- **Open Questions Arising**:
  1. *Geometric Question*: Why does task distance saturate at ~19.6, and what is the underlying manifold shape? (Detailed in [[EXP-002]](EXP-002-gaussian-shell-ood.md)).
  2. *Control Question*: How does this 14.1% vs 39.8% relative error scale impact replanning frequency? (Detailed in [[EXP-014]](EXP-014-replan-frequency-ablation.md)).
