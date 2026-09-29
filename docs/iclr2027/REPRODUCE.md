# Reproducing the submitted ICLR 2027 results

Run from the repository root with the Python environment in `requirements-ada.txt` (or the
equivalent local environment). The benchmark commands below use the fixed `task200u` pool:
the first 50 of 200 pre-solved-pair-filtered tasks per protocol, CEM seeds 0–4, and
separate training and held-out test embeddings. The command without `--run` previews work;
`--run` builds missing assets, checks cached assets, and resumes completed evaluations.

## Inputs and device setup

Place each downloaded HDF5 dataset and the corresponding pretrained LeWM
`config.json`/`weights.pt` at the paths in `config/evaluations/env/`, or set
`PUSHT_H5_PATH`, `REACHER_H5_PATH`, `CUBE_H5_PATH` and the matching
`PUSHT_CKPT_DIR`, `REACHER_CKPT_DIR`, `CUBE_CKPT_DIR`. Use the Hugging Face
`quentinll/lewm-{pusht,reacher,cube}` dataset and model repositories; the dataset
archives must be downloaded and extracted before evaluation. On Ada, stage dataset
archives from `/share1/mayaank.ashok/lewm_data/` via the **login** node, extract
HDF5 files under compute-node `/ssd_scratch/mayaank.ashok/`, and point the H5 variables
there. Set `GAS_MPC_TRAIN_CACHE_DIR` to a persistent-for-the-allocation scratch path
such as `/ssd_scratch/mayaank.ashok/planning_trainonly/pusht` when running Push-T.
Recreate these caches after scratch is purged. Check checkpoint health and action
normalization as described in `AGENTS.md` before interpreting a new run.

Set `evaluation.gpu_slots` in the composed config to the physical GPU IDs assigned
on the current node. The evaluator sets both `CUDA_VISIBLE_DEVICES` and
`MUJOCO_EGL_DEVICE_ID` to each worker's physical slot before importing the model or
MuJoCo. On a four-GPU allocation, set `[0, 1, 2, 3]`; on one GPU, keep `[0]`.

## Main table, ablations, retrieval, and ladder

```bash
python evaluator.py --config config/evaluations/paper_pusht.yaml --run
python evaluator.py --config config/evaluations/paper_reacher.yaml --run
python evaluator.py --config config/evaluations/paper_cube.yaml --run
```

These commands build the train/test encoded caches, TDR, graphs, and ET critics as
needed. They evaluate the random and L2 baselines, OUR method, the Push-T and Cube
ablation rows, Push-T critic-weight variants, and OUR with demonstration retrieval.
They cover panel A of Table 1 except GCIQL, the Push-T and Cube ablations, the
retrieval table, and the shared rows of the full Push-T ladder. Each completed
result is in `outputs/<env>/eval/`; `scripts/gas_mpc/gas_mpc_report.py` aggregates the five
seeds by method and protocol into `docs/gas-mpc/results*.json` and `.tex`:

```bash
python scripts/gas_mpc/gas_mpc_report.py --n 50
GAS_MPC_ENV=reacher python scripts/gas_mpc/gas_mpc_report.py --n 50
GAS_MPC_ENV=cube python scripts/gas_mpc/gas_mpc_report.py --n 50
```

On PowerShell, use `$env:GAS_MPC_ENV='reacher'` (or `cube`) before that command.
Use the same PowerShell form for the GCIQL training commands below.
The paper uses the mean and standard deviation of the five per-seed success
percentages. The report includes every result present in the output directory,
so use the method tags and config fingerprints in the JSON when comparing an
isolated rerun with an archived table.

GCIQL is trained once per environment on the training-only cache, then evaluated
on the same held-out task pool. `precision=32` is required on Ada's GTX 1080 Ti:

```bash
python scripts/baselines/gciql_train.py precision=32
GAS_MPC_ENV=reacher python scripts/baselines/gciql_train.py precision=32
GAS_MPC_ENV=cube python scripts/baselines/gciql_train.py precision=32
python evaluator.py --config config/evaluations/paper_gciql_pusht.yaml --run
python evaluator.py --config config/evaluations/paper_gciql_reacher.yaml --run
python evaluator.py --config config/evaluations/paper_gciql_cube.yaml --run
```

For the graph-radius rows of the Push-T ladder, build separate graphs while reusing
the same train/test embeddings, TDR, critic, and tasks:

```bash
python evaluator.py --config config/evaluations/paper_pusht_htd4.yaml --run
python evaluator.py --config config/evaluations/paper_pusht_htd12.yaml --run
python evaluator.py --config config/evaluations/paper_pusht_htd16.yaml --run
```

## Diagnostics and appendix figures

The L2 plan-length appendix uses the default 25-step plan from the main run and
these 5-, 50-, and 100-step variants:

```bash
python evaluator.py --config config/evaluations/paper_pusht_plan5.yaml --run
python evaluator.py --config config/evaluations/paper_pusht_plan50.yaml --run
python evaluator.py --config config/evaluations/paper_pusht_plan100.yaml --run
```

The predictor-error figure/table and task-distance rows use disjoint data:
predictor error samples `cache_train.npz`; fixed-task distances use `cache_test.npz`.
The two shell figures use training frames. To regenerate their image files:

```bash
python scripts/paper/diagnostics/predictor_error_by_horizon.py --horizons 5,25,50,100 --n-samples 500 --seed 0 --out-dir outputs/pusht/diagnostics
cp outputs/pusht/diagnostics/predictor_error_by_horizon_s0.png docs/iclr2027/figures/fig_predictor_error_by_horizon.png
python scripts/paper/diagnostics/task_start_goal_distance.py --out-dir outputs/pusht/diagnostics
python scripts/paper/diagnostics/gas_mpc_gaussian_shell_diag.py --env pusht --n-samples 200000 --fig-out docs/iclr2027/figures/fig_gaussian_shell_r_hist.png
python scripts/paper/diagnostics/gas_mpc_gaussian_shell_interp_diag.py --env pusht --offset 100 --n-pairs 200000 --fig-out docs/iclr2027/figures/fig_gaussian_shell_interp_off100_hist.png
```

The CEM ranking table uses a privileged simulator oracle on 25 tasks, seed 0,
for each same-episode offset. This is substantially more expensive than the
latent diagnostics:

```bash
python scripts/paper/diagnostics/gas_mpc_cem_trace_audit.py +mpc.protocol=same25
python scripts/paper/diagnostics/gas_mpc_cem_trace_audit.py +mpc.protocol=same50
python scripts/paper/diagnostics/gas_mpc_cem_trace_audit.py +mpc.protocol=same100
```

The Push-T budget-window entries can be read from `success_by_budget` in each
L2/OUR result JSON and averaged over seeds. For a window with prior cumulative
success `p0` and current cumulative success `p1`, its hazard is
`(p1 - p0) / (100 - p0)`. The success-tolerance entries (20/25/30 px and degrees)
require the per-chunk `__c*_traj.npz` files saved by `gas_mpc_eval.py`:
compute position and wrapped-angle error for each trajectory against its saved
`goal_state`, mark a task successful if any step meets both thresholds, then
average percentages over seeds. The aggregate JSON's `min_pos_err` and
`final_ang_err` cannot reconstruct these joint per-step events.

```bash
python scripts/paper/diagnostics/paper_diagnostics.py --eval-dir outputs/pusht/eval
```

The command reports all config-tagged Push-T runs found there; match the L2 and
OUR method tags with the paper table and keep their five seed rows together.

The compute-cost table is tied to one GTX 1080 Ti. Preparation stage times are
in `outputs/pusht/logs/evaluator/`; evaluation JSON records `secs` and `wall`.
Report the GPU model, asset sizes, stage times, and per-decision timing when
rerunning on different hardware. The parameter-count and hyperparameter tables
are model/config metadata, and the approach diagram is a hand-authored TeX figure.

## Scope and paper-record caveats

Table 1 panel B quotes results from prior papers on their own task sets; this
repository cannot regenerate those external numbers. The submitted Push-T
hyperparameter table lists graph radius 7.59 and lookahead 13.53, while the
evaluation drivers that produced the recorded Push-T headline and ablation runs
used graph radius 8 and lookahead 13.7. The configs above retain the recorded
run settings. This discrepancy should be resolved in the manuscript record
before claiming an exact numerical reproduction of every printed cell. Random
sampling, GPU hardware, dependency versions, and finite-seed variation can also
change newly generated values.

## GAS-MPC and ICLR result coverage

This section maps the results recorded in `docs/gas-mpc/main.tex` and
`docs/iclr2027/main.tex` to the current commands. The ICLR numbers are the refreshed
training-only-asset, held-out `task200u` results (first 50 tasks, CEM seeds 0--4).
The living GAS-MPC log also contains screening runs, older pools, single-seed
mechanism studies, and diagnostics; those are historical findings rather than
additional headline estimates. Do not combine those rows with the refreshed paper
tables. Results are written below `outputs/<env>/eval/`; preserve those JSONs and
their config fingerprints when preparing reports.

### ICLR headline results and ablations

`paper_pusht.yaml`, `paper_reacher.yaml`, and `paper_cube.yaml` contain the exact
method variants and paper protocol. The three commands at the start of this file
regenerate Table 1 (random, L2, OUR and comparison methods), the Push-T and Cube
component tables, Push-T critic-weight ladder, and retrieval comparisons. The
recorded held-out headline (success %, mean ± sample SD over seeds 0--4) is:

| Environment | Method | Same-ep 25 | Same-ep 50 | Same-ep 100 | Cross-ep |
|---|---|---:|---:|---:|---:|
| Push-T | L2 | 91.2 ± 1.0 | 40.8 ± 3.7 | 13.2 ± 3.2 | 14.4 ± 3.9 |
| Push-T | OUR | 88.4 ± 2.3 | 70.0 ± 2.2 | 46.4 ± 3.2 | 42.4 ± 3.2 |
| Reacher | L2 | 77.2 ± 4.8 | 90.8 ± 1.0 | 82.8 ± 1.0 | 29.2 ± 2.4 |
| Reacher | OUR | 76.0 ± 4.6 | 98.0 ± 1.3 | 97.6 ± 0.8 | 65.6 ± 2.0 |
| Cube | L2 | 56.0 ± 3.8 | 34.4 ± 3.2 | 35.6 ± 1.5 | 16.4 ± 1.5 |
| Cube | OUR | 62.0 ± 4.7 | 49.6 ± 4.6 | 50.4 ± 3.2 | 27.6 ± 6.1 |

Cube retrieval is the strongest later Cube result in the GAS-MPC log. Its exact
configuration is OUR plus `retrieval=true`, all five seeds and four protocols:
86.4 ± 4.1 / 59.2 ± 2.4 / 78.8 ± 3.0 / 48.0 ± 2.8. Reacher and Push-T retrieval
results are included in the paper table but retrieval does not improve those
environments. The paper's external literature comparison (Table 1B) is not
recomputed here: those methods were evaluated on different task sets.

To render the refreshed per-seed result tables from evaluation JSONs:

```bash
python scripts/gas_mpc/gas_mpc_report.py --n 50
$env:GAS_MPC_ENV='reacher'; python scripts/gas_mpc/gas_mpc_report.py --n 50
$env:GAS_MPC_ENV='cube'; python scripts/gas_mpc/gas_mpc_report.py --n 50
```

On bash, prefix the latter commands with `GAS_MPC_ENV=reacher` or `GAS_MPC_ENV=cube`.
The report includes every matching JSON in the output directory; compare method tags,
task-pool tags, learned-asset seed, and config fingerprint before using a row.

### GCIQL baseline

GCIQL is trained independently for each environment using the training-only cache,
then evaluated on the held-out task pool. Re-run its training and evaluation after
the paper evaluator has prepared the training embeddings:

```bash
python scripts/baselines/gciql_train.py precision=32
$env:GAS_MPC_ENV='reacher'; python scripts/baselines/gciql_train.py precision=32
$env:GAS_MPC_ENV='cube'; python scripts/baselines/gciql_train.py precision=32
python evaluator.py --config config/evaluations/paper_gciql_pusht.yaml --run
python evaluator.py --config config/evaluations/paper_gciql_reacher.yaml --run
python evaluator.py --config config/evaluations/paper_gciql_cube.yaml --run
```

`precision=32` is intentional for the Ada GTX 1080 Ti. The deterministic single
training-seed GCIQL results in Table 1 have no across-seed standard deviation.

### Additional ICLR tables and figures

The Push-T radius ladder and L2 plan-length ablation each rebuild only their changed
graph or evaluation variant while reusing cached embeddings, TDR, critic, and tasks:

```bash
python evaluator.py --config config/evaluations/paper_pusht_htd4.yaml --run
python evaluator.py --config config/evaluations/paper_pusht_htd12.yaml --run
python evaluator.py --config config/evaluations/paper_pusht_htd16.yaml --run
python evaluator.py --config config/evaluations/paper_pusht_plan5.yaml --run
python evaluator.py --config config/evaluations/paper_pusht_plan50.yaml --run
python evaluator.py --config config/evaluations/paper_pusht_plan100.yaml --run
```

Regenerate predictor/task-distance diagnostics, latent shell figures, and single-call
CEM ranking audits with the following commands (the predictor diagnostic uses only
`cache_train.npz`; fixed-task diagnostics use `cache_test.npz`):

```bash
python scripts/paper/diagnostics/predictor_error_by_horizon.py --horizons 5,25,50,100 --n-samples 500 --seed 0 --out-dir outputs/pusht/diagnostics
Copy-Item outputs/pusht/diagnostics/predictor_error_by_horizon_s0.png docs/iclr2027/figures/fig_predictor_error_by_horizon.png
python scripts/paper/diagnostics/task_start_goal_distance.py --out-dir outputs/pusht/diagnostics
python scripts/paper/diagnostics/gas_mpc_gaussian_shell_diag.py --env pusht --n-samples 200000 --fig-out docs/iclr2027/figures/fig_gaussian_shell_r_hist.png
python scripts/paper/diagnostics/gas_mpc_gaussian_shell_interp_diag.py --env pusht --offset 100 --n-pairs 200000 --fig-out docs/iclr2027/figures/fig_gaussian_shell_interp_off100_hist.png
python scripts/paper/diagnostics/gas_mpc_cem_trace_audit.py +mpc.protocol=same25
python scripts/paper/diagnostics/gas_mpc_cem_trace_audit.py +mpc.protocol=same50
python scripts/paper/diagnostics/gas_mpc_cem_trace_audit.py +mpc.protocol=same100
python scripts/paper/diagnostics/paper_diagnostics.py --eval-dir outputs/pusht/eval
```

The budget-window values are derived from `success_by_budget` in each L2 and OUR
JSON. The tolerance table (20/25/30 px and degrees) requires saved
`__c*_traj.npz` rollout chunks; aggregate JSON alone does not contain per-step joint
position/angle events. Compute-cost values require the recorded evaluator stage logs
and per-evaluation `secs`/`wall`; they are tied to the reported GTX 1080 Ti. Parameter
counts and hyperparameter tables are read from model/config metadata, and the approach
diagram is authored in TeX rather than generated by a result script.

### GAS-MPC exploratory and historical results

The GAS-MPC log's headline mechanism comparison and per-seed final results are the
same refreshed runs described above. The historical Push-T screening narrative also
records a different, earlier protocol/result set (for example, its five-seed screen
reports L2/OUR 91.2/90.8 at same25, 48.4/64.4 at same50, 11.2/50.4 at same100,
and 10.4/44.4 cross). These are not the final paper numbers: task pools and critic
assets were refreshed, so the older values cannot be recreated by simply running the
current `paper_*.yaml` configs.

For an isolated legacy baseline or method screen, use the historical evaluator and
explicitly supply the environment, protocol, method, CEM seed, and the matching saved
assets. For example, the following is a single current-protocol L2 cross-episode run;
change the environment variables and flags only when reproducing a specific logged
experiment:

```bash
python scripts/gas_mpc/gas_mpc_eval.py +mpc.method=l2 +mpc.protocol=cross eval.num_eval=50
python scripts/gas_mpc/gas_mpc_eval.py +mpc.method=subgoal_tdr +mpc.protocol=cross +mpc.critic_beta=1 +mpc.critic_cost=et +mpc.final_thresh=13.7 +mpc.final_metric=l2 +mpc.compose=std eval.num_eval=50
```

For the frozen ICLR headline pool and strict training/test asset isolation, use the
composable `evaluator.py --config ... --run` commands above. `gas_mpc_eval.py`
commands reproduce historical runs only when their matching assets, task pool, and
configuration are present. Its CLI changed over the life of the log; use the command
and config recorded in each historical section of `docs/gas-mpc/main.tex`.

The GAS-MPC living log also preserves exploratory retrieval, viability, graph-ranking,
and simulator studies outside the submitted ICLR results. Their one-off drivers were
removed from the active script tree; the archived results remain readable in
`docs/gas-mpc/main.tex`, but those exploratory runs are not reproducible from the
current checkout. Screening-only runs (one to three seeds, smaller task sets, budget
overrides, or old task pools) must remain labeled screening and must not replace the
five-seed paper tables.
