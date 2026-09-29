# LEWM — Long-Horizon World-Model Planning & Research Engine

## Current research scope (Post-ICLR, 2026-09-27)

The ICLR 2027 manuscript ("Graph-Guided Long-Horizon MPC with LeWorldModel") has been finalized and submitted.

The current research phase focuses on **Radical Simplification of Long-Horizon Planning**:
The submitted configuration (OUR Method: frozen LeWM + TDR mapping + offline GAS graph with Dijkstra + budget-capped expected hitting time critic + CEM + final L2 threshold switch) established cross-episode gains of +28 points on Push-T, +36.4 on Reacher, and +11.2 on Cube, but represents an overcomplicated, piecewise patchwork.

Active research aims to **simplify and unify the architecture**: investigating whether unified end-to-end long-horizon planners (e.g., continuous flow/diffusion trajectory generators, quasimetric energy models, or direct goal-conditioned value functions) can match or surpass the composite stack without heuristic threshold switching, or multi-objective balancing.

## Research Knowledge Base & Emergent Idea Web

All empirical findings, mechanistic phenomena, failure modes, and hypotheses are maintained as **atomic cards** in `docs/knowledge/`:
- **Card Schema**: (1) Motivation & Provenance, (2) Method & Protocol, (3) Empirical Results, (4) Synthesis & Next Questions.
- **Master Index**: `docs/knowledge/INDEX.md` maintains a live Mermaid DAG and the active research frontier of open questions.
- **Tracked Cards**: `EXP-001` through `EXP-018` capture all empirical findings, geometry bounds, and failure taxonomy from `docs/gas-mpc/` and `docs/iclr2027/`.
- **CLI Tooling**: `python scripts/tools/kb_manager.py {validate,sync-index,new,questions,search,report}`. Always validate and synchronize after updating or adding cards.
- **Agent Ideation Protocol**: Research proposals must formulate concrete RFC hypotheses addressing open questions in `INDEX.md` with explicit radical simplification rationales and minimal falsification tests.

## Dedicated Workspace Skills

Agents should invoke the specialized skills located under `.agents/skills/`:
- **`research-kb`** (`.agents/skills/research-kb/SKILL.md`): Knowledge base ideation, RFC proposal drafting, card scaffolding, schema validation, and index/graph synchronization.
- **`ada-cluster`** (`.agents/skills/ada-cluster/SKILL.md`): IIIT-H Ada cluster operations, strict storage hierarchy (`/share1` vs `/ssd_scratch` vs `/home2`), Slurm commands, and headless EGL GPU pairing.
- **`lewm-experimenter`** (`.agents/skills/lewm-experimenter/SKILL.md`): Long-horizon planning evaluation (`gas_mpc_eval.py`), pre-flight checks, 5-seed `task200u` benchmark sweeps, and result aggregation.

## Active workflow

For new GAS-MPC benchmark sets, use root `evaluator.py` with a composed file under `config/evaluations/` (`docs/evaluator.md`). It resolves dataset, checkpoint, encoded-cache, TDR, graph, critic, task-pool, and evaluation requirements in order. The shell drivers below are historical reproduction paths.

- `docs/knowledge/`: Core research knowledge base, empirical cards (`EXP-001` through `EXP-018`), and active open questions.
- `docs/iclr2027/main.tex`: Final submitted ICLR manuscript.
- `docs/gas-mpc/main.tex`: Historical methodology, findings, and phase-by-phase results log.
- `scripts/gas_mpc/gas_mpc_prepare.py`: full-dataset encoding, TDR training, and graph preparation;
  resumable stages `encode`, `tdr`, `graph`, and `all`.
- `scripts/gas_mpc/gas_mpc_eval.py`: CEM evaluation, using base eval configs with `+mpc.*` overrides.
  `l2` is the LeWM baseline; `tdr`, `ctg`, `subgoal`, `subgoal_tdr`, `dir`, and `path`
  separate planning mechanisms. Critic, retrieval, and final-phase options are in its header.
- `scripts/gas_mpc/gas_mpc_make_tasks.py`: shared tasks;
  `scripts/gas_mpc/gas_mpc_report.py`: result tables.
- `scripts/paper/diagnostics/`: submitted-paper figures and appendix audits;
  `scripts/diagnostics/`: optional graph, simulator, and single-pass investigations.
- `evaluator.py` and `config/evaluations/`: composable, resumable preparation and evaluation.
  See `docs/iclr2027/REPRODUCE.md` for the submitted paper's commands. Assign physical GPU
  slots in each config before running on Ada.
- `scripts/common/gas.py`: shared TDR/graph implementation. `scripts/common/viability.py`
  and `scripts/critics/` support critic training, loading, and oracle audits.
- `scripts/baselines/`: GCIQL training and the historical cross-episode baseline;
  the latter also supplies shared task and rollout helpers.
- `scripts/tools/`: knowledge-base tooling; `scripts/tests/`: runnable integrity checks.
  See `scripts/README.md` for each script's GAS-MPC and ICLR result coverage.

`GAS_MPC_ENV` selects the environment. Default output roots are `outputs/pusht/` for Push-T
and `outputs/<env>/` otherwise; `GAS_MPC_OUT` overrides the root for isolated diagnostics.
Preparation and several critic scripts use argparse.

Final evaluation episodes must be absent from learned-asset preparation, including diagnostics,
feature inference, retrieval, and threshold calibration. `cache_train.npz` and `cache_test.npz`
hold separate training and final evaluation embeddings; `psi_train_s*.npy` contains only
training frames. Action-normalization statistics come from the full HDF5 action column.
On Ada, set `GAS_MPC_TRAIN_CACHE_DIR` to `/ssd_scratch/mayaank.ashok/planning_trainonly/<env>/`
and run `evaluator.py` to build training caches. Rebuild when scratch is purged or the
allocation changes nodes.
TDR and calibration assets require training-only provenance; old diagnostics/calibrations are
archived. Internal validation episodes are selected only from non-evaluation episodes.

Example baseline (requires the dataset and pretrained checkpoint):
```bash
python scripts/gas_mpc/gas_mpc_eval.py +mpc.method=l2 +mpc.protocol=same25 eval.num_eval=50
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
supported. The sole pool is `task200u` (`gas_mpc_eval.POOL`): pairs the env's own
success predicate already accepts at t=0 are rejected at sampling (`gas_mpc_make_tasks.py`).

Protocols: `same25`, `same50`, `same100`, and `cross`. Same-episode goals are at that
trajectory offset, with default budget twice the offset capped at 250. Cross-episode budgets
are environment-specific (Push-T 250, Reacher 200). Success is the environment's own predicate.
Separate budget overrides and single-pass diagnostics from headline runs, screening from
multi-seed validation, and predictor-imagined costs from realized environment outcomes.

### Mandatory Pre-Flight Invariants Checklist

Before interpreting or publishing any benchmark or diagnostic results, agents MUST verify:
1. **Action Z-Scoring**: Actions reaching `action_encoder` or `predict` must be z-scored via full-dataset column statistics (Push-T $\sigma_a = 0.206$; raw actions cause silent $4.8\times$ shrinkage). De-normalize back to $[-1, 1]$ before `env.step()`.
2. **Checkpoint Health**: Model weights must not be collapsed (off-diagonal cosine similarity $\sim 0.02-0.06$, participation ratio healthy).
3. **Data Provenance Isolation**: Learned assets use `cache_train.npz`; task sampling and evaluation use `cache_test.npz`. The episode sets must be disjoint.
4. **Task Pool Integrity**: Strictly evaluate on `task200u` (first 50 tasks prefix, 5 seeds). Pre-solved pairs rejected at $t=0$.
5. **Headless EGL GPU Matching**: `CUDA_VISIBLE_DEVICES` == `MUJOCO_EGL_DEVICE_ID` == physical GPU slot, set before any PyTorch or MuJoCo import.

### Baseline Benchmark Reference Targets (`EXP-007`)
Any proposed radical simplification architecture must compete against the submitted `docs/iclr2027/main.tex` Table `tab:main` numbers (mean $\pm$ sample standard deviation over CEM seeds 0-4 on `task200u` prefix 50; historical `docs/gas-mpc/results.json` records a different earlier run):

| Method | Push-T (Cross) | Reacher (Cross) | Cube (Cross) |
|---|---|---|---|
| **Vanilla LeWM L2** | $14.4\% \pm 3.9\%$ | $29.2\% \pm 2.4\%$ | $16.4\% \pm 1.5\%$ |
| **OUR Method (ICLR 2027)** | $\mathbf{42.4\% \pm 3.2\%}$ | $\mathbf{65.6\% \pm 2.0\%}$ | $\mathbf{27.6\% \pm 6.1\%}$ |

Preserve shared dependencies and archived results.

## Ada cluster (IIIT-H)

### GPU assignment (2026-09-19)

The current compute node, `gnode003`, has four GPUs (physical IDs 0–3), and the user
has authorized access to all four. This is specific to the current node; verify availability
again when the allocation changes. Configure `evaluation.gpu_slots` for the active node.
For every GPU worker, set both `CUDA_VISIBLE_DEVICES` and `MUJOCO_EGL_DEVICE_ID`
to its assigned physical GPU ID before importing MuJoCo or creating render contexts.
CUDA exposes that single device as logical GPU 0; EGL still uses the physical index.
This keeps Reacher and Cube rendering on the same GPU as their model computation
with the installed MuJoCo 3.5.0. Smoke workers require the same mapping before imports.

### Storage layout — the one thing that's easy to get backwards

| What | Where | Node access |
|---|---|---|
| Code (research scripts) | `/home2/mayaank.ashok/lewm_research/` | Login node (`ada`) and shared home filesystem |
| venv (research scripts) | `/home2/mayaank.ashok/.venv/` | Shared home filesystem — activate with `source /home2/mayaank.ashok/.venv/bin/activate` |
| Code + venv (base `train.py` repo) | `/home/mayaank.ashok/LEWM` | Shared filesystem; 25GB quota, NFS, never put data/checkpoints here |
| Dataset master copies (compressed `.zst`) | `/share1/mayaank.ashok/lewm_data/` | Login node (`ada`) only; not mounted on compute nodes |
| Per-job dataset staging (decompressed) | `/ssd_scratch/mayaank.ashok/` | Existing compute allocation only; fast, purged after ~7 days |
| Research-script dataset, **small only** (decompressed, permanent) | `/home2/mayaank.ashok/lewm_research/data/datasets/` | Shared filesystem, only for datasets that fit `/home2`'s 30GB quota. Two-Room fits; Push-T does not and must be staged to `/ssd_scratch`. |
| Research-script checkpoints | `/home2/mayaank.ashok/lewm_research/data/checkpoints/models--quentinll--lewm-<env>/` | Shared filesystem; small (~72MB), no staging |

`/share1` is available from the login node (`ada`) only. `/ssd_scratch` is available only inside
the active compute allocation. Run compute-side commands through `srun` as described below; do
not SSH directly to a compute node.

`/home2` has a **30GB per-user quota** (confirmed via `quota -s` — not the aggregate filesystem
free space `df -h` reports, which is misleadingly huge and does not reflect the real limit).

**`~/.ssh/config` for login-node access:**
```
Host ada
    HostName ada.iiit.ac.in
    User mayaank.ashok
```

### Existing Slurm allocation only

**Do not create, submit, or allocate a Slurm job.** Never run `sinteractive`, `salloc`, `sbatch`,
or `srun` without `--jobid`. Reuse the user's active allocation; if there is no active allocation,
stop and ask the user instead of starting one.

Find the active allocation's job ID on the login node:
```bash
squeue -u mayaank.ashok -o "%.18i %.9P %.24j %.8T %.10M %.20R"
```
Use the numeric job ID for the existing allocation, and run compute work from `ada` through:
```bash
srun --jobid=<jobid> --overlap bash -lc '...'
```
The interactive allocation has `mem=20G` per job. A FAISS build over the 2.3M-frame cache can
be cgroup-OOM-killed when run alongside training, so run graph builds alone. Read live logs
through an `srun --jobid=<jobid> --overlap ...` command; NFS reads from `ada` can be stale.

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
# inside the existing allocation, launched from ada through srun --jobid=<jobid> --overlap:
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

## Latent viability critic (`scripts/critics/viability_*.py`, `scripts/common/viability.py`) -- supporting critic workflow (recorded 2026-09-11)

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
`scripts/baselines/viability_cross_episode_baseline.py` can be invoked per seed, then
`+cross.aggregate=true` combines the per-seed JSONs. A seed takes
~15-30 min (10 plan calls x 50 envs) on either the local 3050 Ti or a 1080 Ti.
