# LEWM

Research repo for LeWM (Latent World Models) and a downstream "latent graph shaping" offline-RL
project built on top of it. Two mostly-separate codepaths live here:

- **`train.py`/`jepa.py`/`eval.py`** — the base LeWM training/eval pipeline (Hydra configs under
  `config/`), operating on `stable_worldmodel`'s own dataset/checkpoint conventions.
- **`scripts/tworoom_*.py`, `scripts/pusht_*.py`** — the latent-graph research project (see
  `docs/graph-proposal/main.tex` for the full writeup). Own data loading (`H5_PATH`/`CKPT_DIR`
  module constants, not `swm.data.load_dataset`), own checkpoint/resume system, own output dirs
  (`outputs/b0_tworoom/`, `outputs/b3_pusht/`, etc.). Do not confuse this with the `train.py`
  pipeline's conventions below — they differ in several places (see the dedicated section below).

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

A second, separate defect used to compound with this one in the `scripts/*.py` E2 gates: they
called the predictor via `common.lewm_loader.pad_action`, the superseded single-action
zero-padding convention, instead of the real 5-action block (frame-skip=5) the checkpoint was
actually trained on. Normalizing scale does not fix that mismatch on its own — **as of
2026-09-09 this is fixed**: `common.graph_lib.predictor_error_floor`, `discover_predicted_edges`,
and `cycle_consistency_filter` (used by `edge_discovery_gate.py`, `cycle_consistency_gate.py`,
`env_verified_buckets.py`) now use `find_block_transition_edges`/`action_continuation_lookup`
to build real 5-action blocks instead of `pad_action`. `pad_action` itself is still in
`common/lewm_loader.py` for reference but nothing in `scripts/common/graph_lib.py` calls it
anymore — grep for `pad_action` before trusting any NEW predictor-facing code that still uses
it. See [[lewm-predictor-action-block-convention]] and [[e2-idm-rerun-normalized-actions]] for
what changed numerically (Push-T's E2 discovery precision dropped from a saturated-looking
0.993 to a real 0.429/0.787 once the block convention was corrected).

### Checkpoint health — verify before trusting any result

A checkpoint that loads without error is not necessarily a *trained* one. Before running any
latent-geometry or RL experiment against a `quentinll/lewm-<env>` checkpoint, check it isn't
collapsed: off-diagonal cosine similarity between different frames' embeddings should be small
(~0.02–0.06 measured so far, not ~1.0), and the participation ratio of the embedding covariance
(measured on a large, genuinely-random — not temporally-correlated — sample; small/correlated
batches understate it badly) should be a healthy fraction of the embedding dim, not single
digits. `scripts/diag_collapse.py` / `scripts/pusht_diag_collapse.py` do this check. A
from-scratch, few-epoch local checkpoint (e.g. `data/checkpoints/lewm/weights_epoch_1.pt`) is
**not** the same thing as the real pretrained HF checkpoint — download the latter explicitly
(see above) rather than pointing scripts at whatever's already sitting in `data/checkpoints/lewm/`.

Loading an externally-produced HF checkpoint (not one trained locally with the currently-installed
`stable_pretraining`) needs `scripts/tworoom_lewm_loader.py`'s `load_tworoom_lewm(ckpt_dir=...)`
— it remaps ViT block attribute names that changed between the checkpoint's original library
version and what's installed now. `swm.wm.utils.load_pretrained()` lacks this remap and will only
work for a checkpoint trained fresh with the current library version.

## Unified config-driven pipeline (`scripts/common/`, `config/graph/`) -- use this for new work

The `tworoom_*.py`/`pusht_*.py` pairs below were consolidated into one config-driven layer:
`scripts/common/` (`lewm_loader.py`, `graph_lib.py`, `training.py`, `checkpoint_io.py`,
`envs.py`) holds everything environment-agnostic, plus one `EnvMechanics` class per
environment in `envs.py` for the genuinely irreducible per-environment control flow (oracle,
live-env reset/step, noisy-rollout policy). Unified entry points live directly in `scripts/`
with no environment prefix: `graph_gate.py` (B0 gate), `datatiers.py`/`worker.py` (B3 tier
sweep), `rollout_collector.py`, `actor_train.py`/`actor_rollout_eval.py`/
`actor_rollout_utils.py`, `ada_sweep.py`, `aggregate_sweep.py`. Config lives in
`config/graph/` (its own subtree, deliberately separate from `config/train/`/`config/eval/`
which belong to the unrelated `train.py` pipeline) -- `env=tworoom`/`env=pusht` selects the
environment via Hydra override, e.g. `python scripts/graph_gate.py env=pusht`.

**Adding a third environment** means: one `config/graph/env/<name>.yaml`, one
`EnvMechanics` subclass in `common/envs.py` implementing `build_true_distance_oracle`,
`make_env`, `reset_options`, `step_result`, `collect_noisy_rollout` -- no new script files.

**Validated bit-exact** against the original `tworoom_*.py`/`pusht_*.py` scripts before this
layer was trusted for anything: `graph_gate.py env=tworoom` reproduces the historical B0
result to full float precision (0.9515333078449236/0.43939877995042154), `env=pusht`
reproduces it to 5-6 significant figures (the tiny residual is FAISS's own run-to-run
non-determinism, not the extraction). One real subtlety hit during validation: Two-Room's
original B0 script shared ONE module-level `rng` between `calibrate()` and `evaluate()`
(both lived in the same file), while Push-T's never did (different file, different `rng`) --
an accident of file structure, not a design choice, but reproducing the exact historical
numbers required preserving it via `env.share_calibration_rng` in each env's config rather
than picking one behavior for both.

**The old Two-Room library files are NOT deleted** -- 6 of them (`tworoom_b0_graph_gate.py`,
`tworoom_graph_variants.py`, `tworoom_b3_gciql_shaping.py`, `tworoom_b3_convergence.py`,
`tworoom_b3_datatiers.py`, `tworoom_b3_mixed_auxphi.py`) are now re-export shims (e.g.
`tworoom_b0_graph_gate.calibrate` just re-exports `common.graph_lib.calibrate`) since ~24
completed historical diagnostic scripts (`tworoom_b1_graph_diagnostics.py`,
`tworoom_mechanism_probe.py`, `tworoom_edge_cap_sweep.py`, the whole mixed-tier investigation
thread, etc.) still import from them and were deliberately left untouched -- they answered a
specific question and are done, not part of the ongoing repeatable pipeline.
`tworoom_rollout_collector.py` also stays for the same reason (`tworoom_b3_datatiers.py`'s
still-live `_build_mixed_arrays` needs it directly, not through a shim).

**13 fully-redundant scripts WERE deleted** (2026-09-01, verified zero remaining
dependents first): the entire old Push-T pipeline (`pusht_b0_graph_gate.py`,
`pusht_rollout_collector.py`, `pusht_actor_rollout_utils.py`, `pusht_b3_datatiers.py`,
`pusht_b3_worker.py`, `pusht_b3_actor_train.py`, `pusht_b3_actor_rollout_eval.py`,
`ada_pusht_b3_sweep.py`, `ada_pusht_actor_rollout_sweep.py` -- Push-T never had a
distance-source ablation, so nothing there needed keeping) plus three Two-Room-side leaf
scripts superseded with no capability loss (`ada_direct_sweep.py`, `ada_warm_caches.py`,
`aggregate_actor_sweep.py`, `tworoom_b3_auxphi_sweep.py`).

**NOT deleted, despite looking redundant at first glance:** `tworoom_b3_actor_train.py`,
`tworoom_b3_actor_rollout_eval.py`, `tworoom_actor_rollout_utils.py`,
`tworoom_b3_auxphi_worker.py`, `tworoom_b4_ablation_sweep.py`, `ada_b4_edge_ablation_sweep.py`,
`ada_b4ext_actor_rollout_sweep.py` -- these support a `--distance-source` ablation
(`graph`/`euclidean`/`oracle`/`transonly`/`k<N>`) that reproduces a published result
(`sec:b4ext-actor-rollout` in `main.tex`) and has **no equivalent in the new unified layer**
(`worker.py`/`actor_train.py` only ever train the default "graph" condition). Don't delete
these, and don't assume the new layer can reproduce a B4-style ablation until that gap is
actually closed.

The new layer IS deployed to Ada (synced 2026-09-02 via `scp` of `scripts/common/`, the
unprefixed entry points, and `config/graph/`; hydra/omegaconf were already in the venv).
Re-sync any edited file with `scp` before running it there -- there is no git pull on Ada.

### Live-rollout evaluation protocol (`eval_protocol` in `config/graph/actor.yaml`)

The LeWM paper's Push-T policy numbers (Fig. 6: GCBC 75%, GCIVL 33%, GCIQL 20%, Random 2%)
are measured under a specific protocol (paper App. F.1; stable-worldmodel
`scripts/plan/eval_ff.py` + `World._evaluate_from_dataset`): start = a random dataset
state, goal = the state exactly **25 timesteps later in the same trajectory**, **50 env
steps** to reach it, success = the env's own `terminated` (`eval_state`: agent+block
position error < 20 px and block angle error < pi/9). The policy is given the *dataset
frame* at the goal row as its goal image. Their GCIQL (`scripts/train/gciql.py` in
github.com/galilai-group/stable-worldmodel) is frozen DINOv2-small patch embeddings +
6-layer transformer V/Q heads, expectile 0.9, gamma 0.99, AWR on V(s')-V(s) with
alpha 10, trained on the full ~18.7k-episode dataset.

Our original convention (`eval_protocol=cross_episode`) was much harder -- arbitrary
held-out (s, g) pairs from *different* trajectories (the Spearman eval pairs) with a
250-step budget -- and every Push-T actor scored exactly 0.000 under it (see
`[[pusht-b5-b3-rl-training]]`). `eval_protocol=same_episode` (now the default) implements
the paper's protocol; `eval_pool=dataset` draws pairs from the whole h5 minus the tier's
train episodes (needs `PUSHT_H5_PATH` at eval time), `eval_pool=test_episodes` from the
tier's held-out episodes only. `policy=random` in `actor_rollout_eval.py` runs the
paper's Random baseline (measured at 4-9% on 100 episodes here vs the paper's 2% on 50).
Result files now carry the protocol in their name
(`{tier}__{variant}__s{seed}__sel-{rho|success}__{protocol}.json`); files without it
predate this and are all `cross_episode`. `aggregate_sweep.py --protocol` filters
accordingly. The success-selected actor is picked by the *training-time* rollout checks,
which use the same `eval_protocol` -- a checkpoint records which one in its
`eval_protocol` field, and `actor_rollout_eval.py select=success` warns on a mismatch.
`ada_sweep.py --mode actor --eval-only --selects rho --protocol same_episode` re-scores an
existing sweep's trained actors under a protocol without retraining.

Related knobs added alongside: `her_goal_gamma` (HER future-goal offset ~ Geom(1-gamma);
0.9 = the established default, mean 10-step goals; the paper's critic uses 0.99),
`need_graph=false` (skip graph + the dense NxN `phi_dist` entirely -- baseline-only, for
large `expert_N` tiers like `expert_1000` that would otherwise need an O(n^2) matrix;
`variant=auxphi` refuses to run without it), and `run_tag` (suffix for hyperparameter
variants so they get their own checkpoint/result files, e.g. `run_tag=_hg96`).

### Scaling `auxphi` past the dense-matrix wall (`phi_mode=sparse`)

The dense NxN `phi_dist` is what caps tier size: at `expert_1000`'s 124,479 landmarks it is
62 GB as float32, and scipy's `dijkstra(indices=arange(n))` materialises its own float64
copy (124 GB) before returning anything, so it OOMs rather than merely being slow.
`phi_mode=sparse` (config/graph/actor.yaml) keeps the graph but skips the matrix:
`actor_train.py` computes phi at *only* the HER pairs the aux loss actually reads, via
`common.graph_lib.phi_for_pairs` (chunked bounded multi-source Dijkstra). This is exact,
not an approximation -- HER pairs are same-episode, so a transition-edge path no longer than
their step gap always exists, which both bounds the search radius and guarantees no pair is
unreachable. `full_phi_dist_matrix` is also chunked now, so the dense path itself no longer
OOMs at scale. `common.training.run_condition_resumable` (the Spearman-only B3 sweep path)
still requires the dense matrix.

### Two knobs from the Push-T underperformance investigation

`aux_standardize` and `awr_normalize_adv` (both default **false** = exact historical
behaviour, so no published Two-Room number moves). See
`scripts/investigations/pusht_lowscore/` for the measurements that motivated them --
in short, `pseudo_v` carries the graph's *behavioural* distance scale while IQL's expectile
V learns the *optimal* cost-to-go, and those diverge exactly when the data is far from
optimal.

**The 15 remaining completed one-time investigations live under `scripts/investigations/`**,
grouped by theme (moved 2026-09-02, verified zero remaining dependents outside the group
first): `mixed_tier/` (the reward-shaping-on-noisy-data failure investigation --
`tworoom_b3_mixed_diagnosis.py`, `_b1check.py`, `_distance_bins.py`, `_peak_probe.py`,
`_fix.py`), `graph_scaling/` (`tworoom_graph_construction_bench.py`,
`tworoom_edge_cap_sweep.py`, `tworoom_mechanism_probe.py`), `early_diagnostics/`
(`tworoom_b1_graph_diagnostics.py`, `tworoom_b2_encoder_swap.py`), `misc/`
(`tworoom_b3_multiseed.py`, `tworoom_distance_histograms.py`, `pilot_signals.py` -- the
actual origin of this whole research line -- `diag_collapse.py`, `pusht_diag_collapse.py`).
Each subfolder has an `__init__.py`; moved files' `sys.path.insert` was adjusted to still
find `scripts/`'s main library files (now 3 `.parent`s up instead of 1, since they're 2
levels deeper). One cross-dependency needed fixing at the time: `tworoom_b1_graph_
diagnostics.py`'s `load_landmarks` is still imported by the active shims
`tworoom_b3_convergence.py`/`tworoom_b3_datatiers.py`, now via
`investigations.early_diagnostics.tworoom_b1_graph_diagnostics` -- if you ever move
something new in or out of `investigations/`, grep for `from <filename> import` across all
of `scripts/*.py` (not just other investigation scripts) before moving, the same way this
move was checked.

## Research scripts (`tworoom_*.py`, `pusht_*.py`) conventions

- Full writeup, methodology, and results: `docs/graph-proposal/main.tex` (compile with `latexmk
  -pdf -interaction=nonstopmode main.tex` from `docs/graph-proposal/`).
- Each environment gets its own `<env>_b0_graph_gate.py` (checkpoint load, landmark encode,
  calibration, graph construction, Spearman-vs-oracle gate check) and later
  `<env>_b3_datatiers.py`/`<env>_b3_worker.py` (offline RL training tiers) — Two-Room's version is
  the reference implementation; Push-T's is a direct port with documented deviations where
  Two-Room's approach didn't transfer (see the B5 section of the doc).
- Identification-edge construction should default to the k-capped FAISS-HNSW method
  (`tworoom_graph_variants.build_id_edges_faiss_capped`, k=4), not the brute-force O(n²) scan —
  this was a hard-won fix for a real scaling wall, confirmed to also *improve* quality on
  Two-Room, not just enable scale. Do not reintroduce the brute-force scan for a new environment
  without a specific reason.
- The calibrated identification-edge threshold (rho_hat/eps² from one-step-displacement
  correlation) is **not guaranteed to transfer** across environments — it failed outright on
  Push-T (rho_hat statistically ≥ 1). Always sanity-check `eps2 > 0` after calibration before
  trusting it; if it fails, fall back to the heuristic (uncalibrated) top-k construction by
  passing `eps2=float("inf")` into the same capped-FAISS builder — no other code changes needed.
- Offline RL training uses mode 2 (auxiliary regression toward a graph-derived pseudo-value),
  not mode 3 (potential-based reward shaping) — mode 3 was tried first and found to mostly not
  work (TD compounds reward-shaping noise); mode 2 is the established, reusable mechanism for any
  new environment. `tworoom_b3_mixed_auxphi.run_condition_resumable` is written generically over
  a `setup` dict and is reused unchanged across environments; only the `setup`-building code
  (tier definitions, graph construction, oracle) needs to be written per environment.
- Direct-SSH sweep orchestration (`ada_*.py` scripts) is used instead of SLURM `sbatch`/`srun`
  arrays — a `subprocess.Popen` pool with an `N_PARALLEL` concurrency cap, `CUDA_VISIBLE_DEVICES`
  cycled across `N_GPUS`, thread-limiting env vars, and skip-if-done via each combo's own
  checkpoint `done` flag. Run these directly inside an existing interactive Ada allocation
  (`adag`), not submitted as a separate job.
- **Push-T's own generated caches (not just the dataset) can blow `/home2`'s 30GB quota too.**
  `pusht_rollout_collector.py`'s per-tier rollout cache holds raw 224x224x3 pixel frames (up to
  `n_episodes*250` of them for `mixed`/`mixed_large` — `mixed_large`'s alone is ~3.8GB), and
  `pusht_b3_datatiers.py`'s per-tier `phi_dist` cache is a dense NxN float32 matrix
  (`mixed_large`'s is ~4GB). These hit exactly this quota mid-sweep once already (`OSError:
  [Errno 122] Disk quota exceeded`, home2 at 36GB/30GB). Both now take an env var override
  (`PUSHT_ROLLOUT_CACHE_DIR`, `PUSHT_TIER_CACHE_DIR`) — set both to a compute-node-local
  `/ssd_scratch` dir before running any Push-T sweep with `mixed`/`mixed_large` tiers involved,
  same as `PUSHT_H5_PATH`:
  ```bash
  export PUSHT_TIER_CACHE_DIR=/ssd_scratch/mayaank.ashok/lewm_pusht_cache/tier_cache
  export PUSHT_ROLLOUT_CACHE_DIR=/ssd_scratch/mayaank.ashok/lewm_pusht_cache/rollout_cache
  ```
  If quota is already blown, `outputs/b3_pusht/rollout_cache/*.npz` is safe to delete outright
  (pure input cache, regenerates in ~1-2 min); `outputs/b3_pusht/tier_cache/*.npy` is expensive
  to regenerate (`mixed_large`'s took ~4.4 min) so `mv` it to `/ssd_scratch` instead of deleting.
- **The local dev machine has a recurring large-contiguous-allocation problem, not (necessarily)
  Ada.** `np.stack`ing tens of thousands of frames, `IncrementalPCA` with a large `batch_size`,
  and `scipy.sparse.csgraph.dijkstra`'s internal float64 all-pairs matrix (allocated at float64
  regardless of what dtype the input graph or output cast uses — a ~30K-landmark tier needs
  ~7.4GB just for that intermediate) have all hit `numpy._core._exceptions._ArrayMemoryError` on
  this machine at sizes that should comfortably fit in reported free RAM. Don't take a local
  failure at this specific error as evidence a large tier/landmark-count is infeasible in
  general — validate correctness locally at a small, cheap size (e.g. override a noisy-episode
  or landmark count for the check only) and confirm the real size on Ada, which has not shown
  this failure mode.
