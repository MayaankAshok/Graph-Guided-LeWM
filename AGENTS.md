# LEWM — graph-assisted CEM planning

## Current research scope (2026-09-17)

The active project improves long-horizon and cross-episode goal reaching through graph-assisted
planning inside LeWM's CEM planner. The LeWM encoder and predictor stay frozen; CEM is the
low-level controller. Learn a Temporal Distance Representation (TDR), build a GAS-style graph,
and use shortest paths to supply subgoals or cost-to-go. Supporting experiments evaluate
viability/expected-hitting-time costs and switching to direct goal L2 in the final horizon.

IQL/GCIQL actor training, graph-derived auxiliary-value (`auxphi`) regression, potential-based
reward shaping, and GAS with a learned low-level policy are historical and outside active scope.
Do not launch or extend those pipelines unless explicitly requested. Shared helpers remain usable
by planning code; their presence does not make actor training active. Preserve historical results.

Push-T is the primary study; Reacher is the current transfer study. Cube evaluation scripts exist,
but availability alone does not establish validated transfer results. Two-Room supplies historical
diagnostics. Base `train.py`/`jepa.py`/`eval.py` is the upstream LeWM pipeline with separate configs
and dataset/checkpoint conventions.

## Active workflow

- `docs/gas-mpc/main.tex`: current methodology, findings, and results log. Compile from
  `docs/gas-mpc/` with `latexmk -pdf -interaction=nonstopmode main.tex`.
- `scripts/gas_mpc_prepare.py`: full-dataset encoding, TDR training, and graph preparation;
  resumable stages `encode`, `tdr`, `graph`, and `all`.
- `scripts/gas_mpc_eval.py`: CEM evaluation, using base eval configs with `+mpc.*` overrides.
  `l2` is the LeWM baseline; `tdr`, `ctg`, `subgoal`, `subgoal_tdr`, `dir`, and `path`
  separate planning mechanisms. Critic, retrieval, and final-phase options are in its header.
- `scripts/gas_mpc_make_tasks.py`: shared tasks; `gas_mpc_report.py`: result tables;
  `gas_mpc_pair_diag.py` and `l2_single_pass_diagnostic.py`: mechanism/prediction audits.
- `scripts/gas_mpc_run.sh` and `scripts/ada_gas_mpc*.sh`: resumable drivers and Ada sweeps.
  Inspect each driver's queue before running it.
  Parallel drivers should invoke the shared runner as
  `bash scripts/gas_mpc_run.sh "[<current>/<total>]" <method> <protocol> ...`.
  The bracketed first argument is deliberately ignored by `gas_mpc_run.sh`, but remains in
  the Bash command line for `htop` progress tracking. Assign GPU slots explicitly and keep
  `slot` and its derived `gpu` in separate Bash `local` declarations.
- `scripts/common/gas.py`: shared TDR/graph implementation, also containing historical policy
  code. `scripts/common/viability.py` and `scripts/viability_*.py` support critic training and
  audits. Continuation-cost scripts are an experimental branch.

`GAS_MPC_ENV` selects the environment. Default output roots are `outputs/pusht/` for Push-T
and `outputs/<env>/` otherwise; `GAS_MPC_OUT` overrides the root for isolated diagnostics.
Preparation and several critic scripts use argparse.

Final evaluation episodes must be absent from learned-asset preparation, including diagnostics,
feature inference, retrieval, and threshold calibration. `cache_train.npz` is the physically
filtered training cache; `psi_train_s*.npy` contains only its frames. `cache_full.npz` remains
an evaluation/legacy artifact; preparation may read only its pre-existing action-normalization
metadata, never its frame arrays. On Ada, `ada_prepare_trainonly.sh` builds training caches
under `/ssd_scratch/mayaank.ashok/planning_trainonly/<env>/` and links them into each output
folder. Rebuild these caches when scratch is purged or the allocation changes nodes.
TDR and calibration assets require training-only provenance; old diagnostics/calibrations are
archived. Internal validation episodes are selected only from non-evaluation episodes.

Example baseline (requires the dataset and pretrained checkpoint):
```bash
python scripts/gas_mpc_eval.py +mpc.method=l2 +mpc.protocol=same25 eval.num_eval=50
```
Graph objectives require prepared TDR/graph assets; critic terms require a trained critic.
Use the current writeup and sweep scripts for complete configurations and environment-specific
scales. Do not transfer Push-T thresholds blindly.

## Evaluation requirements

Call the former configuration B **OUR method**: `subgoal_tdr`, final switch to goal L2
at one environment-calibrated lookahead, budget-capped ET critic, beta 1, std composition.
Default evaluation uses the first 50 tasks of each fixed 200-task `task200u` pool;
evaluate seeds 0--4 on same25, same50, same100, cross for L2 and OUR.
Use a different task prefix only when explicitly requested.

Compare methods on the fixed 200-task pool per protocol; `eval.num_eval` selects a prefix.
Record task identity/count, CEM seed, and learned-asset seed separately. `mpc.tasks` is not
supported.

Protocols: `same25`, `same50`, `same100`, and `cross`. Same-episode goals are at that
trajectory offset, with default budget twice the offset capped at 250. Cross-episode budgets
are environment-specific (Push-T 250, Reacher 200). Success is the environment's own predicate.
Separate budget overrides and single-pass diagnostics from headline runs, screening from
multi-seed validation, and predictor-imagined costs from realized environment outcomes.

Verify action normalization, real 5-action blocks, predictor history, checkpoint health,
rendering, and remaining-budget conventions before interpreting results. Graph-distance
correlation or critic accuracy alone does not establish improved live control.

Preserve shared dependencies and archived results.
## Ada cluster (IIIT-H)

### GPU assignment (2026-09-19)

The current compute node, `gnode003`, has four GPUs (physical IDs 0–3), and the user
has authorized access to all four. This is specific to the current node; verify availability
again when the allocation changes. `ada_planning_pipeline.py` defaults to four GPU queues.
For every GPU worker, set both `CUDA_VISIBLE_DEVICES` and `MUJOCO_EGL_DEVICE_ID`
to its assigned physical GPU ID before importing MuJoCo or creating render contexts.
CUDA exposes that single device as logical GPU 0; EGL still uses the physical index.
This keeps Reacher and Cube rendering on the same GPU as their model computation
with the installed MuJoCo 3.5.0. Smoke workers require the same mapping before imports.

### Storage layout — the one thing that's easy to get backwards

| What | Where | Node access |
|---|---|---|
| Code (research scripts) | `/home2/mayaank.ashok/lewm_research/` | Both login (`ada`) and compute (`adag`) |
| venv (research scripts) | `/home2/mayaank.ashok/.venv/` | Both — activate with `source /home2/mayaank.ashok/.venv/bin/activate` |
| Code + venv (base `train.py` repo) | `/home/mayaank.ashok/LEWM` | Both — but 25GB quota, NFS, never put data/checkpoints here |
| Dataset master copies (compressed `.zst`) | `/share1/mayaank.ashok/lewm_data/` | **Login node (`ada`) ONLY** — not mounted on compute nodes at all |
| Per-job dataset staging (decompressed) | `/ssd_scratch/mayaank.ashok/` | **Compute nodes ONLY** — fast, purged after ~7 days; not present on login node |
| Research-script dataset, **small only** (decompressed, permanent) | `/home2/mayaank.ashok/lewm_research/data/datasets/` | Both — only for datasets small enough to fit `/home2`'s 30GB quota alongside the repo/venv. Two-Room's `tworoom.h5` fits (3.4GB compressed); **Push-T's `pusht_expert_train.h5` does not** — it decompresses to ~43GB, over quota by itself. For Push-T (and cube/reacher), stage into `/ssd_scratch` per-job instead, same as the base `train.py` pipeline below — set `PUSHT_H5_PATH` to point there. |
| Research-script checkpoints | `/home2/mayaank.ashok/lewm_research/data/checkpoints/models--quentinll--lewm-<env>/` | Both — small (~72MB), lives permanently, no staging |

**Rule of thumb: anything touching `/share1` must run via `ssh ada` (login node), never `ssh
adag` (compute node) — `/share1` simply does not exist from a compute node and mkdir/curl there
will fail with a permission or not-found error.** Conversely, `/ssd_scratch` only exists on
compute nodes. SSH connectivity from a compute node back to the login node (`ssh ada` or `ssh
172.16.0.2`, run *from inside* an `adag` shell) does work, so a compute-node job can `scp`/`rsync`
a file down from `/share1` if it needs it staged locally — just don't expect to read `/share1` as
a local path from a compute node, and don't try to write into it from one either.

`/home2` has a **30GB per-user quota** (confirmed via `quota -s` — not the aggregate filesystem
free space `df -h` reports, which is misleadingly huge and does not reflect the real limit).

**`~/.ssh/config` for direct access from a machine driving this (e.g. Codex's Bash tool):**
```
Host ada
    HostName ada.iiit.ac.in
    User mayaank.ashok

Host adag
    HostName gnode076          # update every time the running compute-node allocation changes
    User mayaank.ashok
    ProxyJump ada
    StrictHostKeyChecking accept-new
```
**When this changes** (happens whenever the interactive allocation gets a new node): update
`HostName` above, then re-stage `/ssd_scratch` from scratch (it's compute-node-local disk, none
of it survives a node change) — full checklist in `notes.md`'s "When the Ada compute-node
allocation changes" section.

### Getting a compute node
```bash
tmux
sinfo -O "NodeList:35,Partition:12,StateCompact:8,CPUsState:15,Gres:18,GresUsed:18" -t idle,mix
sinteractive -c16 -g2 -w <NODE_NAME>
```
Partition `u22` (26+ idle nodes at last check); no `--account` flag needed. Python module:
`u22/python/3.12.4`.

### Interactive-allocation operations (Ada job facts, 2026-09-12)

Find the active allocation's job ID before running a command through Slurm:
```bash
squeue -u mayaank.ashok -o "%.18i %.9P %.24j %.8T %.10M %.20R"
```
Use the numeric value in the first column as `<jobid>`. The interactive allocation has
`mem=20G` per job. A FAISS build over the 2.3M-frame cache run alongside training is
cgroup-OOM-killed without a Python traceback, so run graph builds alone.

If direct compute-node SSH is unavailable because FAISS has starved `sshd` (load can exceed
200) or `pam_slurm_adopt` rejects the session, run work through the login node:
```bash
ssh ada "srun --jobid=<jobid> --overlap bash -c '... '"
```
Start background work from the login node, not inside an existing `srun` step:
```bash
nohup srun --overlap bash script.sh &
```
Backgrounding from inside an `srun` step dies when that step ends. NFS can make live logs
read through `ada` stale; read them through `srun` or `adag` for current output.

### Python environment setup, in order (order matters for torch)
1. For research scripts: `module load u22/python/3.12.4 && python3 -m venv /home2/mayaank.ashok/.venv && source /home2/mayaank.ashok/.venv/bin/activate`
2. Run `pip install` from the **login node** (`ssh ada`, not a compute-node job) — package
   installation doesn't need a GPU, and `/share1` (a sane pip-cache location, 100GB quota vs
   `/home2`'s 30GB) isn't reachable from compute nodes anyway:
   ```bash
   export PIP_CACHE_DIR=/share1/mayaank.ashok/pip_cache
   pip install -r requirements-ada.txt   # or requirements.txt for the base train.py repo
   ```
3. Install torch **last**, from the CUDA index matching the node's actual driver (check with
   `nvidia-smi`) — `requirements-ada.txt`'s unpinned torch dependency otherwise gets silently
   overwritten by whatever's newest on PyPI, which can be newer than the node's driver supports:
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 --force-reinstall
   pip install 'fsspec<=2026.6.0'   # torch's reinstall usually regresses this above datasets' ceiling
   ```
4. Skip the `env` extra's Box2D dependency (broken `swig` wrapper in this environment, and none
   of LEWM's datasets need it anyway — pusht=pymunk, reacher=dm_control, cube=ogbench,
   tworoom=custom): `pip install "stable-worldmodel[train]"` then
   `pip install opencv-python-headless pygame pymunk shapely` directly.

### Dataset download (all 4 known datasets)

Each HF dataset repo ships a plain compressed archive, not something `load_dataset` can
auto-fetch (its auto-download path only matches bare `.h5`/`.hdf5`/`.lance`, not `.zst`).
Download manually, **from the login node**:

| `data=` group | HF repo → file | Compressed size | Extract with |
|---|---|---|---|
| `pusht` | `quentinll/lewm-pusht` → `pusht_expert_train.h5.zst` | 13.1 GB | `zstd -d pusht_expert_train.h5.zst` (plain file, not tar) |
| `ogb` (cube) | `quentinll/lewm-cube` → `cube_single_expert.tar.zst` | 46.2 GB | `tar --zstd -xf cube_single_expert.tar.zst` |
| `tworoom` | `quentinll/lewm-tworooms` → `tworoom.tar.zst` | 3.4 GB | `tar --zstd -xf tworoom.tar.zst` |
| `dmc` (reacher) | `quentinll/lewm-reacher` → `reacher.tar.zst` | 23.8 GB | `tar --zstd -xf reacher.tar.zst` |

```bash
ssh ada "mkdir -p /share1/mayaank.ashok/lewm_data && cd \$_ && nohup curl -L -C - -O https://huggingface.co/datasets/quentinll/lewm-<name>/resolve/main/<file> > download.log 2>&1 & disown"
```
(`-C -` resumes a partial download; `nohup ... & disown` so it survives the SSH session ending —
these are large enough to take minutes.)

Each of `quentinll/lewm-<env>` is **both** a dataset repo (ships the `.zst` above) and a separate
model repo (ships `config.json` + `weights.pt`) under the same name — confirmed via
`HfApi().model_info()`/`.dataset_info()`, not assumed. Checkpoint download (small, ~72MB, safe to
run directly, no `nohup` needed):
```bash
ssh ada "mkdir -p /home2/mayaank.ashok/lewm_research/data/checkpoints/models--quentinll--lewm-<env> && cd \$_ && curl -sL -o config.json https://huggingface.co/quentinll/lewm-<env>/resolve/main/config.json && curl -sL -o weights.pt https://huggingface.co/quentinll/lewm-<env>/resolve/main/weights.pt"
```

For the `train.py` pipeline: `load_dataset` resolves `dataset.name` under
`$LOCAL_DATASET_DIR/datasets/` (differs from README.md, which is stale for this installed
`stable_worldmodel` version). Decompress into `/ssd_scratch/mayaank.ashok/lewm_data/datasets/`
per job. For the research scripts (`tworoom_*.py`/`pusht_*.py`), only Two-Room's small dataset
gets a permanent decompressed copy under `/home2/mayaank.ashok/lewm_research/data/datasets/`
(`LEWM_H5_PATH` — no per-job staging needed). Push-T's is too big for `/home2`'s quota: keep the
compressed `.zst` master on `/share1`, decompress into `/ssd_scratch` per allocation, and point
`PUSHT_H5_PATH` at that instead:
```bash
# on adag, once per allocation (or once per 7-day /ssd_scratch purge cycle):
mkdir -p /ssd_scratch/mayaank.ashok/lewm_data/datasets
rsync -avP ada:/share1/mayaank.ashok/lewm_data/pusht_expert_train.h5.zst /ssd_scratch/mayaank.ashok/lewm_data/
zstd -d /ssd_scratch/mayaank.ashok/lewm_data/pusht_expert_train.h5.zst \
  -o /ssd_scratch/mayaank.ashok/lewm_data/datasets/pusht_expert_train.h5
export PUSHT_H5_PATH=/ssd_scratch/mayaank.ashok/lewm_data/datasets/pusht_expert_train.h5
```

**Always verify the extracted path** with `tar --zstd -tf <file>.tar.zst | head` (for the tar
archives) before assuming it matches a config's expected `dataset.name` — it hasn't always.

### Action normalization — verify before trusting any predictor result

**LeWM's predictor and `action_encoder` require Z-SCORED actions, never raw ones.** This is
not documented in `jepa.py` itself — it's imposed by the data pipeline on both sides:
`train.py` runs `utils.get_column_normalizer` over every non-pixel column loaded (`action`
included), a `ZScoreNormalizer` fit on that dataset's own mean/std; `eval.py` fits a
`sklearn.preprocessing.StandardScaler` per cached column and hands it to
`swm.policy.WorldModelPolicy` as `process`. Any script that calls `model.action_encoder(...)`
or `model.predict(...)` directly (rather than going through `eval.py`) must normalize the
same way by hand, using the FULL dataset action column's mean/std (not just a landmark
subsample).

**The failure mode is silent, not a crash** — feed raw actions and the predictor doesn't
error, it just predicts near-zero motion, which reads exactly like "the predictor doesn't
matter much at longer horizons." This produced a fully-written, plausible-sounding WRONG
conclusion in this project before a paper-reproduction check caught it. Magnitude is
environment-dependent and explains why some past results were more affected than others:
Push-T's action std (0.206) makes raw actions ~4.8x too small (catastrophic); Two-Room's
(0.868) makes them only ~1.15x too small (easy to miss, mostly absorbed into noise).

**Before trusting any predictor-facing code:**
1. Compute `act_mean`/`act_std` from the full dataset action column.
2. Normalize actions before they reach `action_encoder`/`predict`.
3. If the same code also steps the REAL environment, de-normalize back to raw units first —
   `env.step()` wants raw actions; only the model wants z-scored ones.
4. Validate against `eval.py policy=lewm-pusht eval.num_eval=50`, which should land near the
   paper's reported Push-T success rate (96.0 ± 2.83; this repo reproduced 94.0%). Don't trust
   a new predictor pipeline until it clears this bar.

### Checkpoint health — verify before trusting any result

A checkpoint that loads without error is not necessarily a *trained* one. Before running any
latent-geometry or RL experiment against a `quentinll/lewm-<env>` checkpoint, check it isn't
collapsed: off-diagonal cosine similarity between different frames' embeddings should be small
(~0.02–0.06 measured so far, not ~1.0), and the participation ratio of the embedding covariance
(measured on a large, genuinely-random — not temporally-correlated — sample; small/correlated
batches understate it badly) should be a healthy fraction of the embedding dim, not single
digits. `scripts/investigations/misc/diag_collapse.py` / `scripts/investigations/misc/pusht_diag_collapse.py` do this check. A
from-scratch, few-epoch local checkpoint (e.g. `data/checkpoints/lewm/weights_epoch_1.pt`) is
**not** the same thing as the real pretrained HF checkpoint — download the latter explicitly
(see above) rather than pointing scripts at whatever's already sitting in `data/checkpoints/lewm/`.

Loading an externally-produced HF checkpoint (not one trained locally with the currently-installed
`stable_pretraining`) needs `scripts/common/lewm_loader.py`'s `load_lewm` remapping path
— it remaps ViT block attribute names that changed between the checkpoint's original library
version and what's installed now. `swm.wm.utils.load_pretrained()` lacks this remap and will only
work for a checkpoint trained fresh with the current library version.
