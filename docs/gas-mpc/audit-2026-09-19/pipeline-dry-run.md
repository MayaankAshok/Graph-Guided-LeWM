# Planning pipeline dry run — 2026-09-19

Inspected on Ada's gnode003. The main dry run launched no training or evaluation.
The separately isolated execution rehearsal is documented in `pipeline-smoke-test.md`.
Driver: `scripts/ada_planning_pipeline.py` (installed on Ada).

Default evaluation: first 50 tasks of each fixed 200-task `task200u` pool.
Protocols: same25, same50, same100, cross. CEM and learned-asset seeds: 0–4,
matched per run and recorded separately by the evaluator's `seed` and `mpc.graph_seed`.
OUR denotes the former configuration B.

| Environment | Preparation | Critics to train (60,000 steps each) | L2 evaluations | OUR evaluations |
|---|---|---|---:|---:|
| Push-T | Skip all current preparation stages | 0 | 20 | 20 |
| Reacher | Skip all current preparation stages | 0, 1, 2, 3, 4 | 20 | 20 |
| Cube | Skip all current preparation stages | 0, 1, 2, 3, 4 | 20 | 20 |

Preparation checks cover encoding, five TDRs, calibration, five psi caches,
five gap calibrations, five graphs, and all four task pools.
Existing Push-T critics 1–4 pass the completion and holdout checks and are skipped.
The archived provisional seed-0 critic is not used.

| Environment | Graph h_td | OUR lookahead and final threshold |
|---|---:|---:|
| Push-T | 8.0 | 13.53 |
| Reacher | 3.72 | 5.83 |
| Cube | 7.25 | 13.63 |

OUR uses subgoal_tdr, final goal L2, budget-capped ET critic cost, beta 1,
std composition, TE threshold 0.9, and the corresponding corrected critic.
Thresholds come from current training-only calibration.

If executed now: 11 critic trainings and 120 evaluation runs (6,000 task rollouts).
Environments run sequentially; preparation runs alone; training and evaluation
use four GPU queues with at most one worker per GPU. Seed 4 runs after an earlier
seed in the same queue when all five critics need training.

Stages are skipped only after their output passes validation. Incomplete critics
restart rather than resume optimizer state. Evaluation filenames include a
fingerprint of current code/configuration, task pool, normalization, world-model
checkpoint, and (for OUR) planning assets, preventing reuse of old results after
those inputs change. The evaluator can resume chunks for an unchanged run.

Staged HDF5 datasets, pretrained LeWM checkpoints, and pre-existing evaluation/
action-normalization cache metadata are prerequisites; the driver does not
download datasets or retrain LeWM. Two-Room's historical diagnostic pipeline is
outside this driver.

Dry run on a compute node:

```bash
cd /home2/mayaank.ashok/lewm_research
/home2/mayaank.ashok/.venv/bin/python scripts/ada_planning_pipeline.py --dry-run
```

Execution requires explicitly adding `--run` instead of `--dry-run`.
Full stage commands are saved in `pusht.json`, `reacher.json`, and `cube.json`
alongside this report. Evaluation fingerprint tags for missing critics are
provisional in the dry-run plans; execution computes them after training.
