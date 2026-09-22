# Reacher L2 baseline reproduction (2026-09-19)

## Result

The published 86% baseline is **not reproducible as a stable mean** with the released Reacher
checkpoint and SAC dataset. Three evaluations separate task sampling from controller health:

| Evaluation | CEM | Tasks | Seeds | Success (%) |
|---|---:|---:|---|---:|
| Updated headline prefix (`task200u`, first 50) | 30 | fixed 50 | 0--4 | 74 / 80 / 70 / 84 / 78 = **77.2 +/- 4.8** |
| Paper protocol through upstream `eval.py` | 10 | 50 fresh SAC windows per seed | 42--46 | 72 / 74 / 82 / 82 / 80 = **78.0 +/- 4.0** |
| Updated held-out pool, full diagnostic | 30 | fixed 200 | 0--4 | 85.0 / 81.0 / 75.5 / 79.0 / 84.0 = **80.9 +/- 3.4** |

Standard deviations are population SDs, matching `gas_mpc_report.py`. The older `task200`
first-50 prefix produced 86.8 +/- 4.3 with CEM-30 and 84.0 +/- 2.4 with CEM-10. Thus 86 can
be recovered on that older task draw, but not on the new evaluation population or on five
independent draws of the paper protocol.

## Protocol verification

The local paper (`docs/references/lewm-2603.19312.pdf`, Fig. 6 and Apps. D/F.1) reports 86%
and specifies: SAC data, goal 25 environment steps ahead, budget 50, 300 CEM candidates,
10 iterations outside Push-T, top 30, horizon 5 blocks, and open-loop execution of all five
5-action blocks. `scripts/ada_reacher_baseline_check.sh` runs these settings through upstream
`eval.py`, including its per-seed random task draw and full-dataset action `StandardScaler`.
The live environment remains on the previously verified MuJoCo 3.5.0 render path.

## Why the updated first 50 are low

`sample_disjoint_rows` accepts 200 randomized tasks and then sorts same-episode pairs by
`start_row`. Evaluation takes the first 50 of that sorted file, so the prefix is the earliest
quarter of the selected dataset rows rather than a random 50-task prefix. For Reacher:

- start-row ranges by quarter: 10,793--439,719; 459,553--906,117;
  906,154--1,441,726; 1,475,820--1,967,961;
- mean initial joint displacement by quarter: 0.720, 0.684, 0.679, 0.598 radians;
- expanding from the first 50 to all 200 raises L2 from 77.2% to 80.9%.

This is a real prefix-order bias, but it explains only part of the gap to 86. The independent
paper-protocol reproduction is also 78.0%, so ordinary 50-task sampling variation and/or an
unreleased paper-time pipeline difference remains.

## Recommendation

Do not tune L2 until it reaches 86 and do not compare OUR-on-50 against L2-on-200. Either keep
the current fixed first-50 pool and report 77.2% transparently, or prospectively remove the
row sort, regenerate every environment's task pools, and rerun **both** L2 and OUR on the same
new prefixes. The latter is the methodologically cleaner fix, but it invalidates all current
`task200u` headline evaluations and should be an explicit experiment-version change.

Raw paper-protocol logs and the five `_full200audit` JSON manifests are stored beside this file.
