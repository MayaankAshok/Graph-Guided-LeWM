# Paper reproduction scripts

Run commands from the repository root; full benchmark commands are in
[`docs/iclr2027/REPRODUCE.md`](../../docs/iclr2027/REPRODUCE.md).

- Root-level `gas_mpc_prepare.py`, `gas_mpc_eval.py`, `gas_mpc_make_tasks.py`,
  `gas_mpc_report.py`, `gciql_train.py`, `viability_train.py`, and
  `viability_eval_audit.py` build assets, evaluate methods, and train/load the
  GCIQL and ET-critic baselines. They stay together because the evaluator and
  scripts import them by their current module names.
- `diagnostics/` contains the paper's predictor, task-distance, latent-shell, CEM
  ranking, and rollout-summary diagnostics.
- `../common/` contains shared environment, model, graph, and viability code.

Historical exploratory scripts outside the submitted paper workflow were removed.
