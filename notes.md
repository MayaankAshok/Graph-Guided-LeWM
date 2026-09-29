# Running LEWM on Ada (IIIT-H cluster)

## Active graph-assisted CEM workflow (2026-09-17)

Current research is graph-assisted planning in LeWM CEM. IQL/GCIQL actor and graph-shaping
sweeps below are historical. Use `AGENTS.md` and `docs/gas-mpc/main.tex` for current scope.
Code: `/home2/mayaank.ashok/lewm_research/`; activate
`source /home2/mayaank.ashok/.venv/bin/activate`. Sync edited files before Ada runs (no git pull).
Stage large datasets on compute-node `/ssd_scratch`, pulling compressed masters from login-node
`/share1`; set `PUSHT_H5_PATH` to the staged Push-T dataset.

Prepare and evaluate with `python evaluator.py --config config/evaluations/paper_pusht.yaml --run`.
See `docs/iclr2027/REPRODUCE.md` for all submitted-paper commands.
`GAS_MPC_ENV` selects the environment; `GAS_MPC_OUT` overrides the output root. Default roots:
`outputs/pusht/` (Push-T), `outputs/<env>/` (other environments).
Keep large assets within quota and back up scratch assets before the allocation ends.
Node changes require restaging data and restoring/rebuilding scratch-local planning assets.
Old tier/rollout-cache commands below apply to historical actor experiments.
Regenerate current tables with `scripts/gas_mpc/gas_mpc_report.py`.

Evaluation preferences (2026-09-19): call the former configuration B **OUR method**.
OUR is `subgoal_tdr` with a final switch to goal L2 at one calibrated lookahead,
budget-capped expected-hitting-time critic cost, beta 1, and std composition.
Default evaluation is the **first 50 tasks of each fixed 200-task `task200u` pool**,
for CEM/learned-asset seeds 0--4 and protocols same25, same50, same100, cross.
Do not silently increase this default to 200. `evaluator.py`
implements the resumable Push-T/Reacher/Cube workflow and defaults to preview;
`--run` executes missing work, and `evaluation.num_eval` changes the evaluated prefix.
Superseded assets are in `outputs/<env>/bak/superseded_2026-09-19/` on Ada and locally.
Corrected Push-T critics seeds 1--4 completed 60k steps; clean seed 0 remains to train.
Historical pipeline rehearsal used the retired Ada orchestrator to execute preparation,
five short TDR/critic trainings and L2/OUR evaluations for all four protocols in
isolated `/ssd_scratch/mayaank.ashok/pipeline_smoke/<run>/<env>/` directories.
Fixtures use only main-training episodes. Smoke evaluations have an explicit
`smoke_test` marker, reduced task count/budget and separate paths; they must
never count as preparation/training/evaluation success in the main run.
2026-09-19 rehearsal passed on gnode003: five two-step TDRs and critics per
environment, plus all 40 L2/OUR seed/protocol combinations per environment.
The same evaluator is batched in smoke mode to avoid repeated imports; the
actual Bash runner is exercised for both methods. Main mode uses the Bash
runner for every job. Verified normal validation rejects every smoke result
and every short critic; smoke mode rejects the canonical main output path.
Main dry-run still schedules 11 critics and 120 first-50-task evaluations.

Strict final-holdout isolation (2026-09-18): the retired Ada preparation driver rebuilt
training-only caches, TDR diagnostics, features, gap calibration and graphs for Push-T,
Reacher and Cube. It reuses the verified TDR weights and archives earlier diagnostic tables
and preparation assets. It does not restart critic training. Active preparation loads
`cache_train.npz`; `psi_train_s*.npy` has no final-evaluation frames. Training caches are
symlinked from `/ssd_scratch/mayaank.ashok/planning_trainonly/<env>/` and require rebuilding
after a scratch purge/node change. Frozen LeWM normalization is inherited from existing
aggregate statistics; no evaluation-episode actions are read to recompute it. Earlier
evaluation tables retain their original protocol and calibration and do not establish
results for this stricter pipeline.

## Earlier base-pipeline and actor setup notes

The remaining sections retain earlier records; their actor sweeps, old code paths, and
deployment statements do not define the current CEM workflow.

## Storage layout

| What | Where |
|---|---|
| Code | `/home/mayaank.ashok/LEWM` (synced via `scripts/sync_to_remote.sh`) |
| Master dataset copy (compressed) | `/share1/mayaank.ashok/lewm_data/` (check `/share1/dataset` first — may already be there) |
| Per-job dataset staging (decompressed) + checkpoints | `/ssd_scratch/mayaank.ashok/` (fast, purged after ~7 days — copy checkpoints back to `/share1` before then) |
| Never | `/home` for data/checkpoints — 25GB quota, NFS |

`/share1` is visible on the master/login node only (not compute nodes), so it's a staging point, not somewhere a job can read from directly — decompress into `/ssd_scratch` at the start of each job instead.

## When the Ada compute-node allocation changes (new gnodeXXX)

Happened for real on 2026-09-02: the running interactive allocation moved from `gnode058` to
`gnode059` mid-session. `/ssd_scratch` is **compute-node-local disk** — none of it survives a
node change, even though `/home2` (the repo/venv) does. Checklist, in order:

1. **Update `~/.ssh/config`'s `adag` entry** to the new node name (on the machine driving this,
   e.g. via Claude's Bash tool):
   ```
   Host adag
       HostName gnode059          # <- update this
       User mayaank.ashok
       ProxyJump ada
       StrictHostKeyChecking accept-new
   ```
2. **Verify both hops work** before assuming anything else does:
   ```bash
   ssh ada "echo ada-ok"
   ssh adag "echo adag-ok; hostname; nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv"
   ```
3. **The venv itself is fine** (`/home2` is node-independent) — but only from `adag`, never
   `ada` (the login node's plain `python3` is a different, older interpreter with no torch;
   this is a standing gotcha, not new). Sanity check: `ssh adag "source /home2/mayaank.ashok/
   .venv/bin/activate && python -c 'import torch; print(torch.__version__,
   torch.cuda.is_available())'"`.
4. **Re-stage `/ssd_scratch` from scratch** — the dataset and any tier caches pointed there are
   gone on the new node, even though they're still safely on `/share1` (dataset master) or
   nowhere at all (tier caches were never anywhere but the old node's now-unreachable
   `/ssd_scratch` — they must be rebuilt, not just re-copied):
   ```bash
   mkdir -p /ssd_scratch/mayaank.ashok/lewm_data/datasets
   rsync -avP ada:/share1/mayaank.ashok/lewm_data/pusht_expert_train.h5.zst /ssd_scratch/mayaank.ashok/lewm_data/
   zstd -d /ssd_scratch/mayaank.ashok/lewm_data/pusht_expert_train.h5.zst \
     -o /ssd_scratch/mayaank.ashok/lewm_data/datasets/pusht_expert_train.h5
   export PUSHT_H5_PATH=/ssd_scratch/mayaank.ashok/lewm_data/datasets/pusht_expert_train.h5
   export PUSHT_TIER_CACHE_DIR=/ssd_scratch/mayaank.ashok/lewm_pusht_cache/tier_cache
   export PUSHT_ROLLOUT_CACHE_DIR=/ssd_scratch/mayaank.ashok/lewm_pusht_cache/rollout_cache
   ```
   Rebuilding a tier's cache (identification graph + phi_dist matrix) costs real time again --
   `mixed_large`'s took ~4.4 minutes on Ada last time (see the graph-shaping-project section
   below) -- budget for that after any node change, it isn't instant.

## Python environment (one-time setup)

`uv` isn't available on Ada (no root to install it, `snap install` is denied). Use the system module + stdlib `venv` + `pip` instead. Module names vary by node image — `u18/python/3.10.2` didn't exist on `gnode027`; what worked there was `u22/python/3.12.4`. Check what's actually available on your node first:

```bash
module avail 2>&1 | grep -i python
```

```bash
module load u22/python/3.12.4   # or whatever module avail shows
cd ~/LEWM
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
export PIP_CACHE_DIR=/scratch/mayaank.ashok/pip_cache   # keep the download cache off the 25GB /home quota
```

`stable-worldmodel` only requires Python `>=3.10`, so 3.12 is fine (no need to match README's `3.10` exactly).

None of the `module avail` software (namd/gromacs/amber/etc.) is relevant to this project. CUDA/cuDNN modules also aren't needed — the pip `torch` wheel bundles its own CUDA runtime; only the GPU driver on the node matters (check with `nvidia-smi`). If `pip install torch` resolves to an unexpectedly huge/new CUDA stack (e.g. `cu13`), pin an older, smaller, more mature one instead:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### Install stable-worldmodel — skip the `env` extra's Box2D dependency

`pip install "stable-worldmodel[train,env]"` will try to build `box2d-py`, which needs `swig`. The `swig` PyPI wrapper package (github.com/nightlark/swig-pypi) is broken in this environment — its console-script entry point (`from swig import swig`) fails with `ModuleNotFoundError` on both 4.5.0 and 4.4.1, so don't chase that further.

Box2D (LunarLander/CarRacing/BipedalWalker envs) isn't actually needed by LEWM — none of its datasets (pusht=pymunk, reacher=dm_control, cube=ogbench, tworoom) use it. Confirmed against this repo's local `.venv`, which only has `opencv-python-headless`, `pygame`, `pymunk`, `shapely` from the `env` extra — no `box2d-py`, no `ale-py`, no `ogbench`, no `craftax`, no `minigrid`, no `stable-baselines3`. Install that same subset instead of the full extra:

```bash
pip install "stable-worldmodel[train]"
pip install opencv-python-headless pygame pymunk shapely
```

## Get Node
```bash
tmux

sinfo -O "NodeList:85,Partition:12,StateCompact:8,CPUsState:15,Gres:12,GresUsed:12" -t idle,mix
sinteractive -w NODE_NAME

srun --time=12:00:00 -C16 -G2 --pty bash -w gnode058

# Detailed real-time specs & active GPU allocations for a specific node:
scontrol show node <NODE_NAME> | grep -E "NodeName|CPUAlloc|CPUTot|Gres|GresUsed"
```

## Env vars to set before training

```bash
export LOCAL_DATASET_DIR=/ssd_scratch/mayaank.ashok/lewm_data
export STABLEWM_HOME=/ssd_scratch/mayaank.ashok/stable_worldmodel
```

- `LOCAL_DATASET_DIR` → passed as `cache_dir` to `swm.data.load_dataset(...)` at train.py:55-57.
- `STABLEWM_HOME` → read by `get_cache_dir()` in `stable_worldmodel`, used for the checkpoint dir at train.py:109. Unset, it defaults to `~/.stable_worldmodel` (i.e. `/home`).

After the run, copy `/ssd_scratch/mayaank.ashok/stable_worldmodel/checkpoints` back to `/share1` before the scratch purge.

## Dataset: download once to /share1, stage to /ssd_scratch per job

Checked each HF dataset repo's actual file listing directly — they're plain compressed archives, not something `load_dataset` can auto-fetch on its own (its HF auto-download path only matches bare `.h5`/`.hdf5`/`.lance` entries, not `.zst`-compressed ones, so it does **not** work for these repos — download manually instead).

| data= group | HF file | Compressed size | Extract with |
|---|---|---|---|
| `pusht` | `quentinll/lewm-pusht` → `pusht_expert_train.h5.zst` | 13.1 GB | `zstd -d pusht_expert_train.h5.zst` (plain file, not tar) |
| `ogb` (cube) | `quentinll/lewm-cube` → `cube_single_expert.tar.zst` | 46.2 GB | `tar --zstd -xf cube_single_expert.tar.zst` |
| `tworoom` | `quentinll/lewm-tworooms` → `tworoom.tar.zst` | 3.4 GB | `tar --zstd -xf tworoom.tar.zst` |
| `dmc` (reacher) | `quentinll/lewm-reacher` → `reacher.tar.zst` | 23.8 GB | `tar --zstd -xf reacher.tar.zst` |

All four together is ~86.5GB compressed — tight against `/share1`'s 100GB quota, so only download the dataset(s) you're actually training on. Check `/share1/dataset` first in case a CVIT admin already mirrored one.

**1. Download once, to `/share1` (login node, needs internet):**
```bash
mkdir -p /share1/mayaank.ashok/lewm_data
cd /share1/mayaank.ashok/lewm_data
curl -L -C - -O https://huggingface.co/datasets/quentinll/lewm-pusht/resolve/main/pusht_expert_train.h5.zst
# -C - resumes a partial download if the connection drops
```

**2. At the start of each job, stage + decompress into `/ssd_scratch`:**
```bash
mkdir -p /ssd_scratch/mayaank.ashok/lewm_data/datasets
cd /ssd_scratch/mayaank.ashok/lewm_data/
rsync -avP mayaank.ashok@ada:/share1/mayaank.ashok/lewm_data/pusht_expert_train.h5.zst ./
cd /ssd_scratch/mayaank.ashok/lewm_data/datasets
zstd -d ../pusht_expert_train.h5.zst -o pusht_expert_train.h5
```

`load_dataset` resolves `dataset.name` under `$LOCAL_DATASET_DIR/datasets/`, so files need to land there. Note this **differs from README.md**, which says to place files directly under `$STABLEWM_HOME` — that's stale/for a different package version; the installed `stable_worldmodel` (0.1.1) code appends a `datasets/` subfolder (checked directly in `stable_worldmodel/data/utils.py`). If a config's default name doesn't resolve, check both locations.

Also, `pusht.yaml`'s default `dataset.name` is `pusht_expert_train.lance` (a Lance dataset), but the HF repo only ships `pusht_expert_train.h5.zst` — a format mismatch. Override the name on the CLI to point at what you actually downloaded (see run command below); format is auto-detected from the file, so this just works once the name matches an existing path.

For the `.tar.zst` archives (cube/tworoom/reacher), the extracted top-level name should already match each config's `dataset.name` (`ogbench/cube_single_expert.h5`, `tworoom.h5`, `reacher.h5`) — verify with `tar --zstd -tf <file>.tar.zst | head` before assuming, and adjust the extraction path if it doesn't match.

## Run command

```bash
cd ~/LEWM
source .venv/bin/activate
LOCAL_DATASET_DIR=/ssd_scratch/mayaank.ashok/lewm_data \
STABLEWM_HOME=/ssd_scratch/mayaank.ashok/stable_worldmodel \
python train.py data=pusht data.dataset.name=pusht_expert_train.h5 trainer.precision=16-mixed \
  loader.num_workers=1 loader.prefetch_factor=3 loader.batch_size=64 \
  +trainer.accumulate_grad_batches=2
```

Other datasets (once downloaded + extracted per above):
```bash
python train.py data=ogb     # ogb.yaml's default name already matches the extracted archive
python train.py data=tworoom
python train.py data=dmc
```


```powershell
wsl rsync -avz --progress /mnt/c/Mayaank/IIITH/CSTAR/LEWM/scripts/ mayaank.ashok@ada.iiit.ac.in:/share1/mayaank.ashok/lewm/scripts/
wsl rsync -avz --progress /mnt/c/Mayaank/IIITH/CSTAR/LEWM/docs/ mayaank.ashok@ada.iiit.ac.in:/share1/mayaank.ashok/lewm/docs/
wsl rsync -avz --progress /mnt/c/Mayaank/IIITH/CSTAR/LEWM/notes.md /mnt/c/Mayaank/IIITH/CSTAR/LEWM/requirements-ada.txt mayaank.ashok@ada.iiit.ac.in:/share1/mayaank.ashok/lewm/
```

## Historical Two-Room research scripts (B0-B4 + actor sweep) -- separate from the train.py/pusht setup above

The `tworoom_*.py` scripts (latent-graph analysis, GCIQL/actor training, live-rollout eval)
are a different codepath from `train.py` above -- own data loading (`H5_PATH`/`CKPT_DIR` in
`tworoom_b0_graph_gate.py` / `tworoom_lewm_loader.py`, not `swm.data.load_dataset`), own
checkpoint/resume system. Everything below is verified directly against the actual account
(not assumed from the pusht setup's conventions, which differ in a few places).

**Verified storage layout:**

| What | Where | Verified how |
|---|---|---|
| Code | `/home2/mayaank.ashok/lewm_research/` | NFS home is `/home2`, not `/home` (`whoami`+`$HOME` check). Reachable from both login and compute nodes. Deliberately a different directory from the pre-existing `~/LEWM` (that one's the base repo checkout -- `jepa.py`/`eval.py`/`module.py` -- not these research scripts). |
| venv | `/home2/mayaank.ashok/.venv/` | **Update (2026-09-10):** moved out of `lewm_research/` to sit directly under home (same /home2 filesystem/quota, just not nested under the repo). Activate with `source /home2/mayaank.ashok/.venv/bin/activate` from anywhere -- no need to `cd` into `lewm_research` first. All the venv's own internal absolute paths (pyvenv.cfg, bin/activate, console-script shebangs) were fixed with `sed` after the move; every script/doc that used to `source .venv/bin/activate` after `cd lewm_research` was updated to this new absolute path instead. |
| LeWM checkpoint | `data/checkpoints/models--quentinll--lewm-tworooms/{config.json,weights.pt}` inside the repo above | Small (72MB) -- lives permanently in the repo, no per-job staging. Downloaded directly: `quentinll/lewm-tworooms` is BOTH a dataset repo (ships `tworoom.tar.zst`) and a separate model repo (ships `config.json`+`weights.pt`) under the same name -- confirmed via `HfApi().model_info(...)` / `.dataset_info(...)`, not assumed. |
| Dataset master copy (compressed) | `/share1/mayaank.ashok/lewm_data/tworoom.tar.zst` (~3.3GB) | `/share1` is mounted on the login node only (confirmed: `ls /share1/...` works via plain `ssh ada`, fails with "No such file or directory" from inside an `srun` compute-node shell). |
| Per-job dataset staging (decompressed) | `/ssd_scratch/mayaank.ashok/lewm_data/` | Confirmed present and large (880GB) on compute nodes; NOT present on the login node at all. |

**The one thing that's easy to get backwards:** the raw `/share1` filesystem mount is
login-node-only, but SSH connectivity FROM a compute node BACK to the login node (by
hostname `ada` or internal IP `172.16.0.2`) works fine -- confirmed directly with `ssh
172.16.0.2 hostname` and `ssh ada hostname` run from inside an `srun` shell, both
succeeding. So `scp ada:/share1/...` (or `rsync -e ssh`) from a compute-node job script is
a valid way to pull the master copy onto that node's local scratch -- there's no need to
avoid share1 as a staging point, just don't expect to read it as a local path from a
compute node.

**Env var overrides** (added to `tworoom_b0_graph_gate.py` / `tworoom_lewm_loader.py`,
unset by default so local Windows runs are unchanged):
```bash
export LEWM_H5_PATH=/ssd_scratch/mayaank.ashok/lewm_data/tworoom_extracted/tworoom.h5   # wherever the tar actually extracts to -- verify with `find`, don't assume
export LEWM_CKPT_DIR=/some/other/checkpoint/dir   # only needed if NOT using the default in-repo location above
```

**SSH config for direct access** (`~/.ssh/config` on the machine driving this, e.g. via
Claude's Bash tool):
```
Host ada
    HostName ada.iiit.ac.in
    User mayaank.ashok

Host adag
    HostName gnode058          # update this every time the running compute-node allocation changes
    User mayaank.ashok
    ProxyJump ada
    StrictHostKeyChecking accept-new
```
`ssh adag` then lands directly on the currently-running compute node in one hop.

**Cluster specifics confirmed by actually submitting test jobs** (don't re-derive these --
just use them): partition `u22` (26+ idle nodes at last check, gpu:3 or gpu:4 per node);
no `--account` flag needed (`srun --partition=u22 --gres=gpu:1 ...` succeeds without one);
python module is `u22/python/3.12.4` (matches the pusht setup's own finding above).

**`/home2` has a real per-user quota -- 30GB (confirmed via `quota -s`, not the aggregate
filesystem free space `df -h` shows, which is misleadingly huge).** Hit this directly: a
`pip install --force-reinstall` died mid-swap with "Disk quota exceeded", which also
corrupted torch's install (mixed old/new `.so` files, `undefined symbol` on import). Two
old, unrelated checkouts (`~/lewm2`, `~/LEWM` -- pre-existing base-repo checkouts for
`train.py`/`jepa.py`, not these research scripts) were eating 15.6GB of it; removed both
(confirmed unused first) to get back to ~15GB free of 30GB.

**Installing torch: order matters.** `stable_worldmodel`/`stable_pretraining`/`lightning`
declare an unpinned torch dependency, so installing `requirements-ada.txt` pulls in
whatever's newest on plain PyPI (`torch==2.13.0+cu130` as of writing) regardless of what
you installed before it -- silently overwriting a correctly-chosen CUDA build. CUDA 13.0
needs driver >=~580; the actual node driver here is 570.211.01, so that torch fails with
`CUDA initialization: driver too old` (or `cuda.is_available()==False`). Fix: install
`requirements-ada.txt` FIRST, then `torch`+`torchvision` from the matching CUDA index
LAST, with `--force-reinstall` -- nothing after that touches torch again, so it sticks.
That last step also usually regresses `fsspec` back above `datasets`' `<=2026.6.0` ceiling
(same conflict as before) -- `pip install 'fsspec<=2026.6.0'` once more fixes it without
touching torch.

**Pip cache: keep it on `/share1`, not `/ssd_scratch`, and always run pip from the LOGIN
node.** A scratch-node cache is useless from a different node next time; share1 persists
and has a much bigger quota (100GB vs home2's 30GB). Since `/share1` isn't mounted on
compute nodes at all, this means running `pip install` from `ssh ada` directly (fine --
package installation doesn't need a GPU), not from a compute-node job:
```bash
ssh ada
source /home2/mayaank.ashok/.venv/bin/activate
export PIP_CACHE_DIR=/share1/mayaank.ashok/pip_cache
pip install ...
```

**One-time setup, in order:**
1. Download the checkpoint straight into the repo (login node has internet, confirmed):
   ```bash
   ssh ada "mkdir -p /home2/mayaank.ashok/lewm_research/data/checkpoints/models--quentinll--lewm-tworooms && cd \$_ && curl -sL -o config.json https://huggingface.co/quentinll/lewm-tworooms/resolve/main/config.json && curl -sL -o weights.pt https://huggingface.co/quentinll/lewm-tworooms/resolve/main/weights.pt"
   ```
2. Download the dataset once to share1 (nohup'd -- 3.3GB, takes a few minutes; login node internet confirmed working):
   ```bash
   ssh ada "mkdir -p /share1/mayaank.ashok/lewm_data && cd \$_ && nohup curl -L -C - -o tworoom.tar.zst https://huggingface.co/datasets/quentinll/lewm-tworooms/resolve/main/tworoom.tar.zst > download.log 2>&1 & disown"
   ```
3. Set up the Python env in `/home2/mayaank.ashok/lewm_research` per the "Python environment" section above (module load, venv, `pip install -r requirements-ada.txt`, torch from the matching CUDA wheel index).
4. Run `scripts/ada_warm_caches.py` ONCE, sequentially (not as an array job) -- builds landmarks/graph/phi_dist for all 6 tiers up front. Needed because `scripts/ada_actor_sweep.sbatch` runs 60 array tasks (6 tiers x 2 variants x 5 seeds) that could land on different nodes at nearly the same time; letting each lazily build its own tier's cache on first touch would race multiple concurrent writers against the same `tier_cache/*.npy` file (`np.save` isn't atomic the way the checkpoint writer is). Building each tier's cache once, first, means every array task after that just does a safe read.
5. Submit the sweep: `sbatch scripts/ada_actor_sweep.sbatch`, then `scripts/aggregate_actor_sweep.py` to pull results together (safe to run mid-sweep as a progress check too).
