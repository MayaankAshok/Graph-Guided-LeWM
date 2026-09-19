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

Final evaluation episodes are excluded from learned-asset preparation, including diagnostics,
feature inference, retrieval, and calibration. Use the physically filtered `cache_train.npz`
and `psi_train_s*.npy`. Preparation may read only pre-existing normalization metadata from
the legacy/evaluation `cache_full.npz`. Ada training caches live on node-local scratch and
must be rebuilt after a node change or purge. Older TDR diagnostics and calibrations are
archived; active loaders require training-only provenance.

Example baseline (requires the dataset and pretrained checkpoint):
```bash
python scripts/gas_mpc_eval.py +mpc.method=l2 +mpc.protocol=same25 eval.num_eval=50
```
Graph objectives require prepared TDR/graph assets; critic terms require a trained critic.
Use the current writeup and sweep scripts for complete configurations and environment-specific
scales. Do not transfer Push-T thresholds blindly.

## Evaluation requirements

Compare methods on the fixed 200-task pool per protocol; `eval.num_eval` selects a prefix.
Record task identity/count, CEM seed, and learned-asset seed separately. `mpc.tasks` is not
supported. Since 2026-09-18 the pool is `task200u` (`gas_mpc_eval.POOL`): pairs the env's own
success predicate already accepts at t=0 are rejected at sampling (`gas_mpc_make_tasks.py`).
Every result tagged `task200` used the older unfiltered pool (on Cube 22/16/44% of the first 50
same25/50/100 tasks were pre-solved; Push-T/Reacher none) -- do not mix the two in one table.

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

### Storage layout — the one thing that's easy to get backwards

| What | Where | Node access |
|---|---|---|
| Code (research scripts) | `/home2/mayaank.ashok/lewm_research/` | Both login (`ada`) and compute (`adag`) |
| venv (research scripts) | `/home2/mayaank.ashok/.venv/` | Both — moved out of `lewm_research/` on 2026-09-10 to sit directly under home (still same /home2 filesystem/quota, just not nested under the repo). Activate with `source /home2/mayaank.ashok/.venv/bin/activate` from anywhere, no `cd` into `lewm_research` needed first. |
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

**`~/.ssh/config` for direct access from a machine driving this (e.g. Claude's Bash tool):**
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
1. `module load u22/python/3.12.4 && python3 -m venv .venv && source .venv/bin/activate`
   (for the research scripts, the resulting `.venv` was later relocated to
   `/home2/mayaank.ashok/.venv` — see the storage-layout table above — so a fresh setup there
   should just create it at `/home2/mayaank.ashok/.venv` directly instead of nesting it under
   `lewm_research/`.)
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

## Latent viability critic (`scripts/viability_*.py`, `scripts/common/viability.py`) -- supporting critic workflow (recorded 2026-09-11)

Records the earlier viability proposal implementation on the frozen
Push-T LeWM. Use `docs/gas-mpc/main.tex` for current CEM integration and task conventions. Three supporting stages, all argparse (not Hydra), all runnable on the local GPU:

1. `viability_cache.py --episodes N` encodes N random episodes once (CLS latent, raw
   action, 7-dim state, episode layout) into `outputs/pusht/critic_training/cache_N_sS.npz`. Needs
   `import hdf5plugin` (the h5's `pixels` are Blosc-compressed; the script does it).
   ~700 frames/s locally, so 1000 episodes is ~3 min. Action mean/std come from the FULL
   dataset column, not the subset.
2. `viability_train.py --cache ... --out ...` trains the K-member critic
   `V(z, z_g, h)` (h in ENV STEPS, grid = multiples of 5 up to `--h-max`) with: hindsight
   labels `1[h >= tau]` where tau is the FIRST step the logged trajectory satisfies the
   env's success predicate towards the goal frame (`PushTMechanics.goal_reached`, from the
   dataset `state` column, no simulator); cross-episode negatives (block-pose-filtered by
   `--xneg-min-block-dist`); predictor-imagined rollouts (logged-action rows get hindsight
   labels, perturbed/random rows are label-free); a Bellman backup through the frozen
   predictor with an EMA target ensemble aggregated pessimistically (`--bellman-kappa`);
   and a horizon-monotonicity hinge. The proposal's `L_cal` is deliberately NOT
   implemented (unspecified; BCE is proper; calibration is measured, not trained).
   ~195 ms/step locally; 20k steps ~65 min. Self-evaluation is on held-out episodes'
   hindsight labels only -- it cannot see oracle ground truth.
3. `viability_eval_audit.py --critic ... --audit <dir>` is the external test: scores the
   critic against the privileged recovery-time oracle `T` stored by
   `viability_inversion_audit.py` (`z_pred`, `z_true`, `z_goal`, `oracle_min_steps`),
   reporting AUROC/ECE per budget, Spearman(V, -T), selection regret vs LeWM's L2, and
   the oracle-best candidate's percentile under V.

`common/viability.py` holds the shared pieces: `ViabilityCritic`, `predictor_step` /
`imagine` / `history_at` (the predictor needs 3 latents + 3 action blocks; the critic reads a
single z), `sample_action_blocks`, `bellman_target`, `first_hit_time`, `auroc`, `ece`.
Actions reaching the predictor are z-scored per the rule above; raw `[-1, 1]` actions are
only used to sample perturbations.

### Closed loop: the horizon must track the remaining budget (fixed 2026-09-12)

`viability_live_rollout.py` (Experiment 6) originally queried the critic at ONE fixed `h`
for the whole episode because stable_worldmodel's criterion seam exposes no step count.
With `receding_horizon=5` blocks that means the LAST plan call of a 50-step episode asks
"reachable within 25 MORE steps" when the true remaining budget is 0 -- flat cost exactly
when it must decide. This, not the critic, was most of the pure-`-log V` collapse: 14% ->
66% at offset 25 (L2: 94%) and 22% -> 66% at offset 50 (L2: 60%) with the SAME checkpoint,
once `+viability.h_mode=remaining` (a `PlanClock` wrapping `solver.solve`; every solve = one
25-step advance for all live envs; `h` clipped to the critic's `h_max`) was used. Always
run closed-loop with `h_mode=remaining`; results without it predate the fix (`_h25`/`_h35`
in the filename vs `_hrem`). Other costs there: `et` (sum_h (1-V), an expected-hitting-time
cost that does not saturate), `hybrid_std`/`hybrid_et_std` (the proposal's median/IQR
standardisation, fitted on the Exp-2 audit's `z_pred` bank for the same offset). Pure V is
still 28 pts under L2 at offset 25; the standardised hybrid ties it (92 vs 94).

### Graph hitting-time labels are NOT a drop-in replacement for the critic (2026-09-12)

`viability_graph_label_test.py` scores offline-supported graph hitting time `T_G` (directed
per-frame transition edges + the validated k-capped FAISS id edges, audit episodes added to
the node set) against the Exp-2 oracle with no model trained, same metrics as
`viability_eval_audit.py`. Temporal-only edges leave 70-80% of candidate endpoints
disconnected from the goal and rank no better than L2; stitching only becomes useful at an
id-edge radius 10-20x the calibrated eps^2 (14-29 vs 1.3), and even then the raw label sits
below the trained critic (offset 50, z_pred: 0.56-0.64 vs 0.70) while looser radii add
shortcuts that raise selection regret. Any behavioural label is ~5x pessimistic on Push-T's
loose success predicate (expert continuation at offset 50: T_G 16-25 steps vs oracle 4 --
the block is already inside the 20 px tolerance, the expert spends the time repositioning
the agent). See [[viability-bellman-vs-graph-labels]].

### Revised critic: hitting-time distribution head (`viability_train_ht.py`, 2026-09-12)

Implemented at the user's decision so both critics can be compared
under identical protocols (proposal Sec. "Revised Critic"). `common.viability.HittingTimeHead`
predicts `p(T = b | z, z_g)` over 47 classes (0..45 predictor blocks, '> 45'); `V(z,g,h)` is
the cumulative sum up to `h//5`, `expected_bins` gives E[T]. Same `.prob(z, zg, h)` interface
as the ensemble, so `viability_eval_audit.py` / `viability_live_rollout.py` dispatch on the
checkpoint's `kind` field and run unchanged (audit also reports E[T] ranking; live rollout has
an `eht` cost). Pipeline: `viability_graph_labels.py` (directed per-frame transition edges +
k=4 FAISS id edges at eps^2 = empirical q0.99 of adjacent displacement ~14 -- NOT the
calibrated 1.3, which stitches nothing -- reversed bounded Dijkstra per goal, ~90 s for the
1000-episode cache; pairs: same-episode fwd / identity / bwd / cross; stores T_G at
start+5k too) -> `viability_train_ht.py` (50% logged pairs, 50% predictor rollouts under
LOGGED actions labelled with the represented frame's T_G; exact CE + censored tail loss,
beyond-bound weight 0.25; select by val loss; ~75 ms/step locally). Known label properties:
goal-FRAME not goal-set (+1 bin on ~21% of labels), behavioural (~5x oracle on Push-T), and
`V(.,.,0)` is only trained through identity pairs -- rank by E[T] (or best-h) at offset 25,
not the CDF at h=0.
Matched full-dataset result (2026-09-12, `outputs/pusht/critic_training/ht_full_s0/`, proposal
tab:revised-run2): ties the original critic on offset-25 oracle ranking (0.727), below it at
offset 50 (0.674 vs 0.701, 2x regret); closed loop (h_rem) offset 25 pure 58 / E[T] 76 /
std-hybrid 90 vs original 66 / 76 / 92; offset 50 pure 22 / 48 / 50 vs 66 / 58 / 58 (L2 60).
Kept as a baseline, not the mainline. On the full graph 96% of bank pairs are reachable
within 225 steps, but the label test on the same graph (`graph_label_test_full.json`) shows
planner endpoints are still 35-41% disconnected at the calibrated radius (Spearman 0.39-0.50,
at/below L2), 13-15% at q90 (eps^2 5), 7-9% at q99 (eps^2 14, Spearman 0.63-0.66); temporal-only
gets WORSE with more data (85-92%). The loose radius no longer raises regret at full density
(k=4 cap keeps edges short) -- the 1000-episode shortcut effect was a sparsity artifact.

### Historical cross-episode baseline (2026-09-12; predates current fixed task pool)

`viability_cross_episode_baseline.py` scores LeWM's own CEM/L2 planner (paper config) when
start and goal are two uniform-random rows from DIFFERENT episodes -- no distance or
reachability filter. Pairs are fixed per seed (`outputs/pusht/pairs/
pairs_cross_episode_n50_s{seed}.json`, from `default_rng(seed)` alone, byte-identical on
local and Ada); score any new planner on exactly these. Result, 5 seeds x 50 pairs:
**10.8% pooled at 250 steps** (per seed 4/16/6/14/14), and the budget barely matters (4.4% at
25, 7.6% at 50, 8.8% at 100, 10.8% from 175 on). One rollout at `max_budget` gives every
smaller budget exactly (L2 is budget-agnostic, envs freeze on success) -- `run_mode=separate`
exists for budget-aware costs. Successes are the pairs whose block is already near its goal
(39% when block<60px & angle<30deg, 7% otherwise); on failures the agent reaches its own
target (best 14px) while the block gets to ~54px/26deg at best and then drifts off again
(final 106px/91deg). `+cross.pairing=same_episode` reruns the paper protocol through the
same evaluator as a plumbing check (90% at offset 25, paper 96). On Ada: one seed per GPU via
`scripts/ada_cross_episode_baseline.sh SEED GPU` (launched with `nohup srun --overlap` from
the login node), then `+cross.aggregate=true` combines the per-seed JSONs. A seed takes
~15-30 min (10 plan calls x 50 envs) on either the local 3050 Ti or a 1080 Ti.
