---
id: EXP-016
title: Candidate Ranking Within CEM Elites and Horizon Correlation Audits
status: established
created: 2026-09-27
tags: [spearman-correlation, candidate-ranking, cem-elites, oracle-audit, diagnostics]
derived_from: [EXP-001, EXP-003, EXP-006, EXP-007]
leads_to: []
opens_questions:
  - "Can a planner maintain high candidate ranking correlation without requiring an ensemble or multi-step rollout?"
---

# EXP-016: Candidate Ranking Within CEM Elites and Horizon Correlation Audits

## 1. Motivation & Provenance
- **Source Documents**: `docs/iclr2027/main.tex` Section 5.3 and Appendix G (`tab:cem-trace-rho`); `docs/gas-mpc/main.tex` graph-versus-L2 ranking audit (`tab:graph-vs-l2-zpred`).
- **Problem Statement**:
  Candidate ranking against a privileged recovery-time oracle tests whether a planning cost favors actions that leave the environment easier to recover from.

  The two cited audits use different candidate banks. Their correlations must remain separate.

## 2. Experimental Protocol & Method
- **Scripts**: `scripts/paper/diagnostics/gas_mpc_cem_trace_audit.py`.
- **Wide fixed bank**: 50 Push-T tasks, 32 candidates per task, scored on predictor-imagined endpoints; compare L2, TDR, and graph costs to privileged simulator recovery time. This is the graph-versus-L2 audit in `docs/gas-mpc/main.tex`.
- **CEM elite bank**: 25 tasks per protocol, CEM seed 0, asset seed 0, and 30 elites selected by OUR at iteration 1. Execute each elite in the simulator, estimate recovery time with a privileged CEM oracle, then correlate that time with OUR and L2 costs on the same imagined endpoint. Fully censored tasks are excluded. This is the manuscript Appendix G audit.

## 3. Empirical Results
- **Wide fixed candidate bank (Push-T)**, mean per-task Spearman with recovery time; 95% task-bootstrap intervals are in `docs/gas-mpc/main.tex`:

| Cost | Offset 10 | Offset 25 | Offset 50 |
| :--- | :---: | :---: | :---: |
| Terminal latent L2 | 0.183 | 0.719 | 0.356 |
| TDR (no graph) | 0.343 | 0.731 | 0.664 |
| Pure graph | 0.363 | 0.710 | 0.672 |

- **OUR-selected CEM elites (Push-T)**, mean per-task Spearman with privileged recovery time from `docs/iclr2027/main.tex` Table `tab:cem-trace-rho`:

| Protocol | Tasks with non-degenerate labels | OUR cost | L2 cost |
| :--- | ---: | ---: | ---: |
| same25 | 25 | 0.327 | 0.396 |
| same50 | 24 | 0.203 | 0.086 |
| same100 | 11 | 0.191 | -0.064 |

The wide-bank audit shows that L2 ranking falls from 0.719 at offset 25 to 0.356 at offset 50. In OUR-selected elites, L2 ranks better at same25, while OUR ranks better at same50 and same100. These are different candidate distributions and should not be pooled.

## 4. Synthesis & Next Questions
- **Established observation**: The L2 cost loses ranking quality at longer offsets in both audits. The elite audit is affected by censoring and by selecting candidates with OUR's cost, so its absolute correlations cannot be compared directly to the wide-bank audit.
- **Next test**: Evaluate a new simplified cost on the same fixed candidate bank and the same CEM elites before comparing closed-loop success.
