# Active GAS-MPC evaluation flow

Run one entrypoint with a composed evaluation config:

```bash
python evaluator.py --config config/evaluations/pusht_headline.yaml        # preview
python evaluator.py --config config/evaluations/pusht_headline.yaml --run  # execute missing work
```

`reacher_headline.yaml` and `cube_headline.yaml` select the other environments. Each config extends a base, an environment, and method fragments. The base defines seeds 0–4, all four `task200u` protocols, the first 50 tasks, and physical GPU slots. A new evaluation set can override any of these values without creating a shell driver. Set `evaluation.gpu_slots` to the physical GPU IDs assigned on the current node before running. Each worker sets matching CUDA and EGL device IDs before launching PyTorch or MuJoCo.

The runner resolves these requirements in order:

```text
downloaded HDF5 dataset + pretrained LeWM weights
    -> cache_train.npz + cache_test.npz from disjoint episodes
    -> TDR per asset seed
    -> graph and ET critic per asset seed
    -> fixed held-out task200u pool
    -> CEM evaluation per method, protocol, and seed
```

Existing artifacts are checked and reused. Missing derived artifacts run their current Python producers (`gas_mpc_prepare.py`, `viability_train.py`, `gas_mpc_make_tasks.py`); `gas_mpc_eval.py` retains its resumable per-chunk result cache. Invalid derived artifacts are archived under their configured output or training-cache root before rebuilding. Dataset and pretrained weights are external inputs. Point `dataset.path` and `model.directory` to caches, or set `dataset.prepare` and `model.prepare` to lists of argv commands in an environment config. Those commands can use `{dataset}`, `{checkpoint_dir}`, `{output_root}`, and `{python}` path placeholders. On Ada, stage compressed archives from `/share1` through the login node before extracting to `/ssd_scratch`; compute nodes cannot mount `/share1`.

The encoder computes action mean and standard deviation from the full HDF5 action column and writes separate `cache_train.npz` and `cache_test.npz` embedding files. TDR, graph, critic, and calibration consume training embeddings. Task sampling and evaluation use the held-out test cache. Both files record and verify the fixed episode split.

The paper's evaluation sets and diagnostic commands are listed in `docs/iclr2027/REPRODUCE.md`. Historical shell sweeps have been retired; archived result files remain in place.

Workflow scripts are grouped under `scripts/gas_mpc/`, `scripts/critics/`, and
`scripts/baselines/`; the evaluator launches their current paths. See the
[script inventory](../scripts/README.md) for each script's role in GAS-MPC and
the submitted ICLR results, plus optional diagnostics, tools, and checks.
