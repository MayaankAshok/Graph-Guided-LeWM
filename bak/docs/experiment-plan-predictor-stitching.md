# Experiment set: validating predictor-based cross-episode stitching

## Context

This document develops a chain of extensions to the latent-graph pipeline
(`scripts/common/graph_lib.py`, `scripts/actor_train.py`): (1) using the LeWM
predictor to *verify* candidate identification edges via dynamical consistency,
(2) using it to *discover* new cross-episode edges the metric kNN scan can't
see, (3) using backprop to invert the frozen predictor and find a connecting
action directly, (4) training an ensemble of small inverse-dynamics models
(IDM) and using their prediction variance as a confidence/OOD signal, and (5)
using IDM-identified actions to generate synthetic cross-episode HER tuples
fed into IQL training, potentially replacing the existing `auxphi` aux-loss
term entirely.

The open concern, reasoned through but not yet measured: standard same-episode
HER is "free" because its relabeled goal is trivially true by construction
(whatever the trajectory actually reached). Synthetic cross-episode HER breaks
that — the action is inferred, not observed, so a wrong inversion is real label
noise fed directly into `critic_loss`/`actor_loss`, not a softly-weighted aux
term. This repo already has one precedent for "theoretically safe, empirically
still lost" (mode-3 potential shaping, provably policy-invariant, lost on 4/5
tiers) — synthetic HER has a *weaker* theoretical guarantee than that, so
there's no basis to assume the gap vs. baseline stays positive, especially as
the graph grows and the number of candidate cross-episode pairs (and thus the
absolute count of any false edges) grows with it.

This experiment set is designed to answer that empirically, in stages, so each
stage's kill criterion is checked *before* investing in the next, the same
gated-ladder style the existing B0-B5 experiments already use
(`docs/graph-proposal/main.tex`). Two-Room first (has a trustworthy oracle
distance to diagnose against), then Push-T once the mechanism is trusted —
matching how B0-B4 and B5 were historically sequenced.

## Principles carried over from the existing pipeline (reuse, don't reinvent)

- Every new mechanism is a config-gated knob defaulting to **off**, the same
  pattern as `aux_standardize`, `awr_normalize_adv`, `aux_cross_lambda` in
  `config/graph/actor.yaml` — no existing published number should move.
- Calibrate every threshold from an empirical null (as `calibrate()` /
  `compute_calibration()` in `scripts/common/graph_lib.py` already do), never
  a hardcoded constant.
- `eval_protocol=same_episode` for all live-rollout numbers (the paper
  protocol) — `cross_episode` is floored at 0% for every known Push-T
  condition and cannot show an effect either way.
- `run_tag` discipline (`config/graph/actor.yaml`) so a changed graph-
  construction mechanism always forces a fresh checkpoint/result file instead
  of silently reusing a stale one.
- Reuse the existing B4 distance-source ablation infrastructure —
  `tworoom_b4_ablation_sweep.py` / `ada_b4_edge_ablation_sweep.py`, which
  ablate the *distance source* fed to the aux-regression term
  (none/euclidean/graph/oracle) — as the harness for E4/E5's new conditions,
  by adding a new distance-source value, rather than building a parallel
  sweep mechanism. (Note: `tworoom_b3_actor_train.py`'s own
  `--distance-source` flag is a *different*, already-existing ablation — it
  only takes `graph`/`transonly`/`k<N>`, the edge-capping axis from the B4
  extension, confirmed via its argparse validation. Don't conflate the two;
  E4/E5 need the former.)

## New building blocks needed

- **IDM ensemble**: N=5-8 small MLP heads `(z1, z2) -> action`, trained on
  real adjacent `(z_t, z_{t+1}, a_t)` triples already available from the
  landmark/action arrays loaded in `scripts/common/training.py`. New file,
  e.g. `scripts/common/idm_ensemble.py`.
- **Predictor-based edge discovery**: reuse `model.action_encoder` /
  `model.predict` from the checkpoint loaded via
  `scripts/common/lewm_loader.py`, and the FAISS-HNSW index already built
  inside `build_id_edges_faiss_capped` (`scripts/common/graph_lib.py:227`) for
  nearest-neighbor lookup of predicted next-states. Add as a new function in
  `graph_lib.py`, e.g. `discover_predicted_edges(...)`.
- **Edge verification**: dynamical-consistency check (apply `j`'s real action
  to `z_i` and vice versa, compare to real successors) as a filter function in
  `graph_lib.py`, thresholded against the predictor's own empirical one-step
  error floor (measured the same way `compute_calibration` measures its null).
- **Synthetic HER tuple generator**: new function alongside `build_her_tuples`
  in `scripts/common/training.py`, gated by a new flag (e.g.
  `her_stitch_frac`, default `0.0`) controlling what fraction of each training
  batch's HER tuples are drawn from verified synthetic cross-episode edges
  instead of real same-episode relabeling.

## Experiment ladder

**E1 — IDM ensemble sanity check** (Two-Room, no RL)
Train the ensemble on real adjacent pairs; measure held-out inverse-dynamics
accuracy on held-out same-episode transitions.
*Kill criterion:* if held-out action-prediction error isn't meaningfully
better than predicting the marginal action distribution, stop — the whole
downstream mechanism has nothing to build on.

**E2 — Edge discovery + verification precision vs. the Two-Room oracle** (no RL)
Run predictor-rollout discovery on Two-Room's existing B0 landmark set
(`scripts/graph_gate.py`). For every proposed edge, look up the true oracle
distance already computed there. Measure: (a) precision of accepted edges
(fraction with oracle distance ≤ 2) as a function of the IDM-ensemble
variance threshold, (b) correlation between `Var(âᵢ)` and actual oracle-
distance error — this directly tests whether ensemble variance is a valid
confidence signal, as claimed, or mostly reflects action multimodality.
*Kill criterion:* if precision at any usable variance threshold isn't clearly
better than the existing plain-metric k-capped identification edges' own
false-edge rate (measurable the same way), the discovery+verification
machinery adds nothing over what's already built — stop before any RL run.

**E3 — Graph-quality proxy at scale** (Two-Room, no RL, mirrors B0/B1)
Add E2's verified edges on top of the existing k-capped graph; re-run the B0
Spearman-vs-oracle gate (`scripts/graph_gate.py`) at landmark-count scales
50/300/1000 episodes (mirroring B1). This isolates the "especially as graph
size grows" question at the graph-quality level, before any expensive RL
sweep.
*Kill criterion:* if Spearman doesn't improve over the existing k-capped graph
at any scale, or degrades at the larger scales (false-edge count outgrowing
useful new connections), that's the answer to the scaling question directly.

**E4 — Design A: verified edges feed the graph only, no synthetic actor
targets** (Two-Room, then Push-T)
Wire E3's edges into `phi_dist` exactly like normal identification edges — no
new loss term, no synthetic action ever reaches `actor_loss`. This isolates
whether edge discovery alone helps, using the mechanism already proven safe
(mode-2 aux regression). Sweep tiers × 3 seeds via the existing B4 ablation
harness; live-rollout eval at `eval_protocol=same_episode`.
*Kill criterion:* if this doesn't help, Design B (below) — strictly riskier —
is even less likely to.

**E5 — Design B: synthetic cross-episode HER fed into `critic_loss` /
`actor_loss`** (Two-Room first — this is the specific mechanism this
conversation raised the concern about)
Implement as an additive, confidence-gated fraction of the HER pool
(`her_stitch_frac`, default 0), restricted to edges that passed E2's
precision bar. Sweep `her_stitch_frac ∈ {0, 0.05, 0.15, 0.30, 1.0}` (1.0 =
full replacement, as originally proposed) at a fixed tier, 3 seeds — this
produces the actual baseline-vs-synthetic-HER gap curve, including whether it
ever goes negative. Cross with tier size — `config/graph/datatiers.yaml`'s
default `dataset_sizes: [10, 25, 50, 100]` gives `expert_10`/`expert_25`/
`expert_50`/`expert_100` plus `mixed`/`mixed_large` out of the box; a larger
scale point needs a `dataset_sizes` override (e.g. 300+) together with
`phi_mode=sparse` (`config/graph/actor.yaml`'s existing knob for tiers too
large for the dense `phi_dist` matrix) — to test whether the safe fraction
shrinks, or the gap's sign flips, as the graph grows.
*Kill criterion / concern being tested directly:* if the gap turns negative
at any tested fraction or scale, that confirms the "no guaranteed positive
gap" concern and defines the safe operating range, if one exists.

**E6 — Repeat E4/E5's winning configuration on Push-T**
Only after Two-Room shows a clear non-negative signal. Use
`eval_protocol=same_episode`. Explicitly watch for the already-diagnosed
Push-T failure mode (expert-vs-noisy AWR weight ratio inverting on noisy
tiers, see `scripts/investigations/pusht_lowscore/`) recurring under
synthetic HER.

## Metrics logged at every applicable stage

- `Spearman(phi, oracle)` — Two-Room only (E2, E3)
- `Var(âᵢ)` vs. true oracle-distance error, correlation — Two-Room only (E2)
- Edge precision at threshold — Two-Room only (E2)
- Peak value-Spearman proxy (existing metric, all RL stages)
- Live-rollout success rate under `eval_protocol=same_episode` (primary
  metric, E4-E6)
- Expert-vs-noisy AWR weight ratio (the diagnostic that caught the Push-T
  failure mode in `scripts/investigations/pusht_lowscore/`) — log this for
  every `her_stitch_frac > 0` run; it's the earliest warning sign if the
  actor's data preference is being corrupted again.

## Explicit non-goals / guardrails

- Never chain multiple predicted or synthetic hops within one edge or one HER
  tuple — one verified step only.
- Do not run E5 before E2's precision bar is established on real oracle data
  — an unverified synthetic action reaching `actor_loss` directly is the
  single riskiest configuration in this whole chain.
- Do not skip Two-Room and go straight to Push-T for any new mechanism —
  Push-T has no oracle to diagnose a regression against if something goes
  wrong.

## Deliverable

A short internal report (need not be folded into `main.tex`) answering the
two concrete open questions from this conversation: (1) does IDM ensemble
variance actually track ground-truth confidence (E2), and (2) is the
baseline-vs-synthetic-HER gap ever negative, and at what fraction/scale (E5).

## Verification

Each stage is verified against its own stated kill criterion using existing
tooling plus the new additions above:
- E1-E3: `scripts/graph_gate.py` (existing B0/B1 gate-check pattern), no
  training run needed.
- E4-E6: the existing B4 distance-source ablation harness
  (`tworoom_b4_ablation_sweep.py` / `ada_b4_edge_ablation_sweep.py`) extended
  with a new distance-source value (E4) and the new `her_stitch_frac` knob
  (E5/E6). Confirmed `scripts/ada_sweep.py`'s `VARIANTS = ["baseline",
  "auxphi"]` (line 39) is a hardcoded module constant, not an arbitrary CLI
  override — adding a `her_stitch` condition means editing that constant (and
  its use in the Hydra-override builder, `env_cli()`) directly, not just
  passing a new flag. Once added, run the full tier × seed grid through the
  same direct-SSH orchestration pattern on Ada, following the existing
  skip-if-done / `run_tag` conventions documented in `CLAUDE.md`.
