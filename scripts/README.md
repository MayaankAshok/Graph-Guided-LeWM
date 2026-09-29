# Research scripts

Run commands from the repository root. The active benchmark entry point is
`python evaluator.py --config config/evaluations/paper_pusht.yaml --run`.
See [the reproduction guide](../docs/iclr2027/REPRODUCE.md) for all environments,
paper tables, and appendix figures.

```text
scripts/
  gas_mpc/             preparation, fixed tasks, evaluation, result tables
  baselines/           GCIQL and the historical cross-episode baseline
  critics/             viability training, loading, and oracle audits
  diagnostics/         optional GAS-MPC and simulator investigations
  paper/diagnostics/   submitted-paper figures and appendix analyses
  common/              shared model, environment, graph, and critic code
  tools/               knowledge-base management
  tests/               runnable integrity checks
```

The scripts do not all generate submitted ICLR results. The table distinguishes
result producers from imported helpers and optional investigations.

| Script | Role in GAS-MPC / submitted ICLR results |
| --- | --- |
| [gas_mpc/gas_mpc_prepare.py](gas_mpc/gas_mpc_prepare.py) | Required: separate train/test encoding, TDR, and graph assets. |
| [gas_mpc/gas_mpc_make_tasks.py](gas_mpc/gas_mpc_make_tasks.py) | Required: fixed held-out `task200u` pools, rejecting pre-solved pairs. |
| [gas_mpc/gas_mpc_eval.py](gas_mpc/gas_mpc_eval.py) | Required: headline methods, ablations, retrieval, and plan-length evaluations. |
| [gas_mpc/gas_mpc_report.py](gas_mpc/gas_mpc_report.py) | Aggregates evaluation JSON into GAS-MPC result tables. |
| [baselines/gciql_train.py](baselines/gciql_train.py) | Trains the GCIQL comparison policy; its loader is imported by the evaluator. |
| [baselines/viability_cross_episode_baseline.py](baselines/viability_cross_episode_baseline.py) | Pair sampling, transforms, and rollout helpers are imported by current evaluations. Its standalone CLI reproduces an earlier baseline, not the current headline protocol. |
| [critics/viability_train.py](critics/viability_train.py) | Required for OUR: trains the viability ensemble used for budget-capped expected hitting time. |
| [critics/viability_eval_audit.py](critics/viability_eval_audit.py) | Its critic loader is required by OUR. Its standalone CLI scores critics against saved privileged oracle outcomes. |
| [critics/viability_inversion_audit.py](critics/viability_inversion_audit.py) | Oracle worker helpers support the paper's CEM ranking audit; the standalone CLI is a supporting viability experiment. |
| [diagnostics/planning_cost_gate.py](diagnostics/planning_cost_gate.py) | Its frame encoder is used by preparation and oracle diagnostics. Its standalone cost-ranking experiment is optional. |
| [diagnostics/gas_mpc_cube_render_check.py](diagnostics/gas_mpc_cube_render_check.py) | Cube rendering/state parity check; supports validation, does not generate headline percentages. |
| [diagnostics/gas_mpc_graph_shortrange_corr.py](diagnostics/gas_mpc_graph_shortrange_corr.py) | Optional graph-distance versus trajectory-offset investigation. |
| [diagnostics/gas_mpc_te_diag.py](diagnostics/gas_mpc_te_diag.py) | Optional temporal-efficiency/subgoal investigation. |
| [diagnostics/gas_mpc_support_audit.py](diagnostics/gas_mpc_support_audit.py) | Historical training-support/route investigation; expects legacy `cache_full.npz` and `psi_full_s*.npy` banks. |
| [diagnostics/gas_mpc_oracle_audit.py](diagnostics/gas_mpc_oracle_audit.py) | Historical physics reachability/encoder-inversion investigation; also expects a legacy full-data bank. |
| [diagnostics/l2_single_pass_diagnostic.py](diagnostics/l2_single_pass_diagnostic.py) | Historical single-call L2 prediction versus execution diagnostic; separate from headline closed-loop evaluation. |
| [diagnostics/plot_l2_single_pass_diagnostic.py](diagnostics/plot_l2_single_pass_diagnostic.py) | Plots that historical diagnostic's saved `task200` result. |
| [tools/kb_manager.py](tools/kb_manager.py) | Maintains knowledge cards and their index; does not generate benchmark results. |
| [tests/test_evaluator.py](tests/test_evaluator.py) | Checks config composition, script locations, and local import routes. |
| [tests/test_heldout_tasks.py](tests/test_heldout_tasks.py) | Checks deterministic, disjoint, nontrivial task sampling. |
| [tests/test_training_only_cache.py](tests/test_training_only_cache.py) | Checks training/test embedding isolation and training-only calibration. |
| [tests/test_kb_manager.py](tests/test_kb_manager.py) | Checks knowledge-card integrity and index generation. |
| [paper/diagnostics/](paper/diagnostics/) | Six reproduction tools: predictor error, task distances, two latent-shell figures, CEM ranking audit, and budget/tolerance analysis. |
| [common/](common/) | Shared dependencies; includes compatibility code for recorded baselines and critic checkpoints. |

Direct script commands now use their subfolder paths, for example:

```bash
python scripts/gas_mpc/gas_mpc_prepare.py encode
python scripts/gas_mpc/gas_mpc_eval.py +mpc.method=l2 +mpc.protocol=same25 eval.num_eval=50
python scripts/baselines/gciql_train.py precision=32
python scripts/tools/kb_manager.py validate
python scripts/tests/test_evaluator.py
```

The original flat script paths have been replaced. Current launchers, imports,
reproduction commands, knowledge-card references, and workspace skills use the new
paths. Historical audit records can still contain the old paths; their recorded
results and archived assets have not been migrated. Optional historical diagnostics
are not a substitute for the paper's training-only, held-out protocol.
