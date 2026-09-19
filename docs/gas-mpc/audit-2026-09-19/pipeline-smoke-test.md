# End-to-end rehearsal — 2026-09-19

Completed on Ada gnode003. Runtime rehearsal passed for Push-T, Reacher, Cube.
No main training assets or main evaluation results were written.

| Environment | Encode | TDR seeds | Psi/calibration/graph seeds | Task generation | Critic seeds | L2 runs | OUR runs |
|---|---|---|---|---|---|---:|---:|
| Push-T | Passed | 0–4 passed | 0–4 passed | Passed | 0–4 passed | 20 passed | 20 passed |
| Reacher | Passed | 0–4 passed | 0–4 passed | Passed | 0–4 passed | 20 passed | 20 passed |
| Cube | Passed | 0–4 passed | 0–4 passed | Passed | 0–4 passed | 20 passed | 20 passed |

All evaluations exercised actual rendering, the frozen pretrained encoder and
predictor, CEM, real environment action stepping and result aggregation. All four
protocols and five seeds were covered for both methods. The normal shared Bash
runner was exercised for both methods; other smoke evaluations called the same
Hydra-configured evaluator within one process per environment to avoid repeated
interpreter/library startup. The main pipeline still uses the runner for all jobs.

Smoke configuration:

- Twenty source episodes per environment, selected strictly from main-training
  episodes; zero overlap with the real final evaluation split.
- Real pixels/actions/state in a separate HDF5 fixture; up to 128 frames per episode.
- Two TDR and critic updates per seed, all five seeds freshly trained.
- A separate two-task held-out fixture pool for each protocol, evaluated on its
  first task with a budget of five real environment steps.
- CEM has eight candidates, top-k two, one iteration; normal five-action blocks
  and predictor horizon/history are retained.

Isolation verified:

- All smoke data/assets/results are under
  `/ssd_scratch/mayaank.ashok/pipeline_smoke/20260919_e2e/`.
- `SMOKE_ONLY.json` identifies every fixture as ineligible for the main run.
- Evaluation records contain `smoke_test: true`, use task2u and SMOKE tags, and
  are rejected by normal evaluation-result validation.
- Every two-step critic fails the normal 60,000-step completion requirement.
- Smoke evaluator mode refuses the canonical main output directory.
- A post-rehearsal main dry-run still schedules Push-T critic seed 0, Reacher/Cube
  critics 0–4, and all 120 first-50-task evaluations. It skips only verified main assets.

Independent matrix verification checked exactly 40 unique
(method, protocol, seed) combinations per environment, valid result counts and
budgets, finite critic weights via stage validation, and correct episode splits.
The verified report is `smoke-summary.json` alongside this document.

Run a new isolated rehearsal on the compute node:

```bash
cd /home2/mayaank.ashok/lewm_research
/home2/mayaank.ashok/.venv/bin/python scripts/ada_planning_pipeline.py --smoke --gpus 4
```

Without an explicit smoke-root override this creates a fresh timestamped scratch
directory. Default invocation remains planning-only; `--run` is required for
main execution. The rehearsal verifies execution paths, not scientific performance
or full-scale memory/duration. Its scores must not enter headline tables.
