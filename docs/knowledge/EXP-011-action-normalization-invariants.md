---
id: EXP-011
title: Action Space Z-Scoring Invariants, Checkpoint Health, and Silent Failure Modes
status: established
created: 2026-09-27
tags: [invariants, action-normalization, silent-failures, sanity-checks, cluster-ops]
derived_from: [EXP-001, EXP-003]
leads_to: [EXP-018]
opens_questions:
  - "How can we implement automated compile-time or runtime assertion wrappers to guarantee no un-normalized actions ever reach a world model?"
  - "Can action normalization be made self-contained inside the model checkpoint rather than requiring external preprocessing?"
---

# EXP-011: Action Space Z-Scoring Invariants, Checkpoint Health, and Silent Failure Modes

## 1. Motivation & Provenance
- **Source Documents**: `docs/iclr2027/main.pdf` Appendix E ("Action normalization invariant"); `docs/gas-mpc/main.pdf` Section 10 ("Held-out task-pool and critic refresh"); `AGENTS.md` ("Action normalization", "Checkpoint health").
- **Problem Statement**:
  In world-model research, the most dangerous failures are **silent failures**—scenarios where code runs without raising an exception, but generates mathematically incorrect predictions that mimic scientific phenomena (e.g. leading researchers to write papers concluding that "predictors are unresponsive at long horizons").

  This document formalizes the hard-won operational invariants, data hygiene requirements, and diagnostic checks that must govern all ongoing research and future automated pipelines in this codebase (originally recorded in `AGENTS.md` and `notes.md`).

## 2. Experimental Protocol & Method
- **Audits**: `scripts/tests/test_training_only_cache.py`; checkpoint health checks are described in `AGENTS.md`.
- **Verification Harness**: Validating action normalization statistics, latent covariance spectra, and episode data split isolation.

## 3. Empirical Results

### 1. The Action Z-Scoring Invariant (The 4.8x Silent Trap)
- **The Invariant**: LeWM's `model.action_encoder(...)` and `model.predict(...)` strictly expect **Z-SCORED actions**:
  $$a_{\text{norm}} = \frac{a_{\text{raw}} - \mu_a}{\sigma_a}$$
  computed across the **FULL dataset** action column, NOT raw actions in $[-1, 1]$.
- **The Failure Mode**:
  - If raw actions are fed, PyTorch executes without error.
  - However, because Push-T action standard deviation is $\sigma_a \approx 0.206$, feeding raw actions makes the input $\sim 4.8\times$ too small in magnitude.
  - The predictor outputs near-zero latent movement regardless of the action taken.
  - In Two-Room ($\sigma_a \approx 0.868$), the error is only $1.15\times$, which easily slips into noise and goes unnoticed.
- **The Rule**:
  - Always normalize actions before sending to the model.
  - Always de-normalize back to raw units before calling `env.step()`.
  - Validate every new pipeline against the baseline `eval.py policy=lewm-pusht eval.num_eval=50`, which must achieve $\sim 94\%$ success on Push-T.

### 2. Checkpoint Health & Latent Collapse Invariant
- A checkpoint that loads without error is not necessarily a trained, functioning model.
- A collapsed encoder maps diverse images to nearly identical latents.
- **Health Checks**:
  1. *Off-Diagonal Cosine Similarity*: Compute pairwise cosine similarity between random, temporally decorrelated frames. Healthy: $\cos \theta \in [0.02, 0.06]$. Collapsed: $\cos \theta \approx 1.0$.
  2. *Covariance Participation Ratio (PR)*: Measures the effective dimensionality of the latent covariance matrix:
     $$\text{PR} = \frac{(\text{Tr}(\Sigma))^2}{\text{Tr}(\Sigma^2)}$$
     Must be a healthy fraction of the embedding dim ($D=192$), not single digits.
  3. *Checkpoint Provenance*: Never point planning scripts to fresh local checkpoints (e.g. `data/checkpoints/lewm/weights_epoch_1.pt`). Always use official HF pretrained weights (`quentinll/lewm-<env>`) loaded via `scripts/common/lewm_loader.py` (which remaps ViT block attribute names).

### 3. Training-Only Data Isolation Invariant
- Evaluation episodes must be completely excluded from all learned assets (TDR training, graph construction, critic training, and threshold calibrations).
- `cache_train.npz` and `psi_train_s*.npy` represent the physically filtered training split.
- `cache_full.npz` is strictly an evaluation artifact; scripts may only read its normalization metadata, never its frame arrays.

## 4. Synthesis & Next Questions
- **Established Fact**: Any future agentic improvement pipeline or automated Ada cluster runner must execute these sanity checks as automated pre-flight assertions before spending GPU hours.
- **Next Steps**: Integrate these checks into automated smoke tests for any newly proposed architecture.
