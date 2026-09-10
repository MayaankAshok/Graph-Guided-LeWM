# Quasimetric-JEPA: Reconciling Isotropic Regularization with Dynamical Reachability for Latent World Models

**Target Venues:** NeurIPS / ICLR
**Track:** Self-Supervised Learning / Robot Learning / World Models
**Status:** Research proposal + experiment ladder, E0-E5, closed out for now. E0-E3 passed
their offline gates; E4 (live rollout) shows the trained head **regresses** live planning
success (76.0% vs. 94.0% for raw L2), and two further retries (E3-v2: 64.0%, E3-v3, the most
rigorous attempt — ranking loss over real-env-grounded CEM-like candidates: 60.0%) each
improved their own offline diagnostic while making live rollout **monotonically worse**. E5
(direct diagnostic: does the CEM-chosen plan's predicted cost track its true physical
outcome?) **refutes the "cost-hacking" explanation** for this — quasimetric conditions
correlate predicted cost with true outcome distance far MORE strongly than L2 does
(0.54-0.65 vs. 0.06) — and instead points to a harder-to-navigate learned cost landscape as
the likely mechanism. Net result: the representation-level findings (E0-E2) stand; no version
of the planning-cost swap (Section 2's core proposal) has been shown to help, and E5 gives a
specific, falsifiable reason why not. Not a paper draft; the planning result is not supported
by the evidence gathered.

---

## 1. Motivation

Joint-Embedding Predictive Architectures (JEPAs) learn world models by predicting environment
dynamics in a compact latent space rather than generating pixels. **LeWorldModel (LeWM)** (Maes
et al., 2026) trains such a JEPA end-to-end from raw pixels with a two-term loss:
$$\mathcal{L}_{\text{LeWM}} = \underbrace{\|\hat{z}_{t+1} - z_{t+1}\|_2^2}_{\mathcal{L}_{\text{pred}}} + \lambda \underbrace{\text{SIGReg}(Z)}_{\mathcal{L}_{\text{anti-collapse}}}$$
where SIGReg regularizes the latent distribution toward $\mathcal{N}(0, I_d)$ via the Epps-Pulley
test statistic along random 1D projections (Cramer-Wold).

Its planning cost is plain squared latent L2 ([jepa.py:138](../jepa.py#L138)):
$$\mathcal{C}(\hat{z}_H, z_g) = \|\hat{z}_H - z_g\|_2^2$$

L2 is symmetric. Push-T's contact dynamics are not: an end-effector can push a block one way
but has no way to pull it back without disengaging and re-approaching from the other side.
A symmetric cost cannot express that asymmetry by construction — this is the one part of the
original document's argument that survives scrutiny as-is, and it motivates the fix below.

Two other claims in an earlier version of this document did not survive re-checking against
their own scripts, and this version drops them rather than repeat them:

- A "symmetry residual" test that reported $\big|\|z_A-z_B\|-\|z_B-z_A\|\big| = 0$ as a finding.
  That equality holds for *any* norm on *any* pair of vectors — it is the definition of a norm,
  not a measurement of LeWM. It said nothing about whether real asymmetry exists to learn.
  Replaced by [E1](#e1----is-there-a-real-asymmetry-to-learn-and-how-big-is-it) below, which
  measures it against the repo's own directed transition graph instead.
- A "topological barrier" test on Two-Room reported latent distance across a solid wall
  (22.16) vs. to the doorway (17.76) as a *failure*. Its own script prints a line the prior
  writeup omitted: **0.00% of wall pairs rank closer than the door pairs** — every single
  pair is correctly ordered. That is the metric working, not failing, and the pairing itself
  is confounded (same-room vs. cross-room, not barrier vs. no-barrier). Dropped outright
  rather than reframed, since it does not obviously generalize to the pairs planning actually
  needs to rank close.

The Gaussian-annulus argument (all pairwise latent distances concentrate near $\sqrt{2d}$) is
mathematically correct but was previously used to claim "no gradient beyond 5 steps" — that
conclusion does not follow, since it is a fact about *independent* pairs, and planning compares
temporally *dependent* candidates. [E2](#e2----where-does-latent-l2-actually-stop-discriminating)
below measures the thing that actually matters: adjacent-bin discriminability along a real
step-gap curve, not the unconditional pairwise mean.

The cross-episode collapse claim is real and independently corroborated by this repo's own
prior work: Push-T's cross-episode Spearman is 0.25 by this document's own script, and the
repo's `eval_protocol=cross_episode` scored every Push-T actor condition at exactly 0% success
(see [[pusht-b5-b3-rl-training]]). This is the actual problem worth solving.

---

## 2. Proposed Fix: Quasimetric-JEPA

Replace the L2 planning cost with a learned, frozen-latent, **post-hoc** asymmetric head
$d_Q(z_i, z_j)$ that need not be a true quasimetric to be useful — it only needs to (a) beat
raw L2 on ranking real dynamical distance, and (b) recover asymmetry where it genuinely exists.
A full quasimetric (directed triangle inequality) is a stretch goal, not the gate.

### 2.1 Head architecture (`scripts/common/quasimetric.py`)

Asymmetric potential decomposition, in the style of IQE / QRL (Wang & Isola 2022) but with a
single scalar potential rather than a multi-interval embedding — noted here as a known
limitation, not a novelty claim:
$$d_Q(z_i, z_j) = \underbrace{\mathrm{ReLU}\big(\Phi(z_j) - \Phi(z_i)\big)}_{\text{directional potential}} + \underbrace{\|\Psi(z_i) - \Psi(z_j)\|_2}_{\text{symmetric local metric}}$$

```python
class LatentQuasimetric(nn.Module):
    def __init__(self, latent_dim=192, hidden_dim=256, proj_dim=64):
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim), nn.Mish(),
            nn.Linear(hidden_dim, 1),
        )
        self.psi = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.Mish(),
            nn.Linear(hidden_dim, proj_dim),
        )

    def forward(self, z_src, z_dst):
        potential_diff = F.relu(self.phi(z_dst) - self.phi(z_src)).squeeze(-1)
        spatial_dist = torch.norm(self.psi(z_src) - self.psi(z_dst), p=2, dim=-1)
        return potential_diff + spatial_dist
```

**A first training attempt at this head, before the ladder below, scored *worse* than raw L2**
(held-out same-episode Spearman 0.629 vs. L2's 0.767 on Push-T) and its asymmetry gap came
entirely from a constant `+10.0` margin in the loss rather than anything learned per-pair —
i.e. it did not actually pass its own bar. That result is why this document no longer treats
the head as validated and instead proposes the ladder in Section 3 before any more paper-shaped
claims. The training objective needs to move toward the QRL-style constrained formulation
(maximize $d_Q(s,g)$ subject to $d_Q(s,s') \le 1$ on real transitions) rather than regressing
$d_Q \to k$ directly — regressing to step-count learns *behavioral* distance, not cost-to-go,
which is the same divergence already diagnosed in this repo's IQL work
([[pusht-underperformance-diagnosed]]). The random-pair contrastive term (`L_rand`, forcing
every cross-episode pair to $d \ge 50$) is very likely actively harmful to the one thing this
project is supposed to fix — cross-episode stitching — since many cross-episode pairs in expert
data are near-identical states, not maximally distant ones. Both are ladder items, not
committed design decisions.

### 2.2 Integration into planning

Swap the terminal cost at [jepa.py:138](../jepa.py#L138):
```python
# cost = F.mse_loss(pred_terminal_z, goal_z, ...)          # current
cost = quasimetric_head(pred_terminal_z, goal_z)             # proposed
```
Not attempted until the ladder below clears its gates — there is no point wiring a head into
planning before it beats the baseline it replaces.

---

## 3. Experiment Ladder

Each stage gates the next. Push-T is the primary environment (Two-Room's transitions are
"roughly reversible" per [graph_lib.py](../scripts/common/graph_lib.py)'s own docstring, so it
is a weak test of an asymmetry fix); Two-Room is used only where it's already cheap to check
alongside Push-T. All stages run on Ada (`gnode087`, 3x RTX 2080 Ti), reusing this repo's
existing landmark caches and `common.graph_lib`/`common.envs` infrastructure rather than
building parallel plumbing.

**Overall gate for this whole line of work:** a trained quasimetric head must beat raw latent
L2 on ranking real dynamical distance (same-episode Spearman, held-out episodes). Beating this
repo's own graph-geodesic metric (Spearman ~0.95 on Two-Room, ~0.99 on Push-T same-episode) is
explicitly **not** required to keep going — the head only needs to be a better *drop-in
replacement for L2* in the planning cost, not a replacement for the graph pipeline.

### E0 — Does the frozen latent even retain enough to learn a metric from?

Gate question: if the encoder already discarded the state, no post-hoc head can recover it and
this becomes an encoder-retraining project instead. Reused `scripts/probe_gate.py` rather than
writing a new probe — it already runs exactly this comparison (frozen `z` vs. a raw-pixel floor
vs. a dimension-matched random-projection floor, episode-level train/test split, full 2.34M-frame
Push-T dataset) as the kill criterion for a related proposal ([[experiment-plan-predictor-stitching]]).

```
python scripts/probe_gate.py env=pusht n_episodes=18685
```

**Result** (2,336,736 frames / 18,685 episodes, episode-level held-out test; all four arms use
an identical probe architecture/budget, so the comparison is purely about the representation):

| Arm (dim) | Probe A: state R^2 | Probe B: gap Spearman (free-form) | Probe C: gap Spearman (metric form) | Edge-precision @5/10/25 steps (C) |
|---|---|---|---|---|
| **lewm (192)** | **0.9939** | **0.9577** | **0.9404** | 0.877 / 0.901 / 0.913 |
| pixels (768, raw-pixel floor) | 0.9866 | 0.9239 | 0.9234 | 0.878 / 0.881 / 0.895 |
| randproj (192, dim-matched floor) | 0.9692 | 0.9188 | 0.9172 | 0.878 / 0.872 / 0.885 |
| state (6, privileged-state ceiling) | — | 0.9431 | 0.9328 | 0.864 / 0.899 / 0.911 |

**Verdict: PASS, decisively, on the repo's own kill criterion** (`[E1] kill criterion (LeWM
beats the raw-pixel floor on probe A AND probe C): PASSED`). The frozen latent isn't just
above the floor, it's the best of the four arms on every probe — including probe C, the
metric-form recovery a quasimetric head actually needs, where it edges out even the raw
privileged-state ceiling (0.9404 vs. 0.9328; plausibly an artifact of the ceiling arm's much
lower input dimensionality interacting with the shared probe architecture, not evidence LeWM's
z carries more information than ground truth — not worth over-reading). This is a stronger and
more direct signal than the dropped architecture-only argument in the prior version of this
document: a metric head on frozen z is not just plausible, it already recovers the *undirected*
magnitude well above every floor. The open question the rest of the ladder addresses is
direction (asymmetry) and cross-episode stitching specifically, not whether the information
exists at all.

### E1 — Is there a real asymmetry to learn, and how big is it?

The dropped Test 1 measured a tautology. This measures asymmetry against something real: the
repo's own directed transition graph (`common.graph_lib.build_weighted_graph` — edges follow
real env transitions forward-in-time only, plus symmetric identification edges linking
near-duplicate states across episodes). A route *backward* through this graph exists only if
some other episode's trajectory happens to traverse it — a genuine, data-grounded notion of
reachability, not a property of the distance function computing it.

```
python scripts/investigations/quasimetric/e1_directed_asymmetry.py env=pusht
```

Method: sample the same "active pushing" pairs as the original Test 1 (block displaced > 20px
over a 15-step window, 300-episode landmark set, 12,056 candidate pairs, 4,000 subsampled),
build the directed weighted graph (`id_weight=1`, the repo's established default), and compute
forward (A->B) and backward (B->A) graph-geodesic distance for each pair via Dijkstra.

**Result:**

| Quantity | Value |
|---|---|
| Forward graph distance (A->B) | 14.82 +- 0.75 (100% finite; tracks the 15-step window almost exactly, as expected) |
| Backward graph distance (B->A) | 21.36 +- 3.24, but only **0.5% finite** |
| **Irreversible fraction** (no route back exists anywhere in the offline data) | **99.5%** |
| Asymmetry gap where a backward route does exist (n=22) | +7.55 +- 2.68 steps |
| Raw latent L2 asymmetry (baseline, by construction) | 0.00000000 |

**Verdict: real, large, and mostly not "harder" but "impossible."** For 99.5% of active-pushing
transitions, the offline dataset contains no trajectory that ever returns to the pre-push
configuration — this is a much stronger and better-grounded claim than the prior document's
constant-margin framing. It also reframes what the head should target: the dominant signal is
not "penalize the reverse direction by some margin," it's "most reverse queries should map to
a large, saturating distance (unreachable-in-data), not a graded one." The head's `L_bwd` term
(Section 2.1's constant `+10.0` margin) does not currently distinguish these — worth revisiting
before the next training attempt.

### E2 — Where does latent L2 actually stop discriminating?

Replaces the dropped $\sqrt{2d}$ argument with the thing planning actually consumes: given two
candidate action sequences whose predicted terminal states are $k$ and $k+1$ true steps from
the start, can L2 tell them apart? Measured as adjacent-bin discriminability
$d' = (\mu_{k+1}-\mu_k) / \sqrt{(\sigma_k^2+\sigma_{k+1}^2)/2}$ over step-gap bins from 1 to 250,
same-episode pairs, full 2.34M-frame Push-T dataset.

```
python scripts/investigations/quasimetric/e2_saturation.py env=pusht n_episodes=18685
```

**Result** (2,336,736 frames, 15,000 same-episode pairs per step-gap bin, bins from 1 to 250
true steps):

| Step-gap bin | Latent L2 mean +- std | True (normalized state) mean +- std | Adjacent-bin d' (latent) |
|---|---|---|---|
| 1 | 1.28 +- 0.80 | 0.11 +- 0.07 | — |
| 4-5 | 5.03 +- 2.63 | 0.49 +- 0.28 | +0.73 |
| 19-25 | 15.08 +- 4.07 | 1.63 +- 0.77 | +0.59 |
| 36-50 | 18.33 +- 3.33 | 2.32 +- 0.98 | +0.35 |
| 51-70 | 18.83 +- 2.79 | 2.74 +- 1.02 | **+0.16** |
| 71-100 | 19.02 +- 2.32 | 3.18 +- 1.02 | +0.08 |
| 151-250 | 19.59 +- 1.84 | 3.80 +- 0.92 | +0.20 |
| random cross-episode | 19.61 +- 1.43 (matches sqrt(2*192)=19.60) | — | — |

Headline same-episode Spearman(true gap, latent L2), all bins pooled: **0.8649**.

**Verdict: real saturation exists, but not where the original document put it.** Adjacent-bin
discriminability ($d'$) first drops under the 0.2-sigma threshold at the **36-50 -> 51-70** step
boundary, not at 5 steps — the original "H=5" claim (derived from the unconditional
$\sqrt{2d}$ argument, which does not condition on temporal dependence) overstated the severity
by roughly 8-10x. L2 stays clearly informative through at least ~35 steps and only becomes
genuinely noise-dominated past ~50-70. Note also the true-state curve's own $d'$ stays around
0.3-0.4 even at 100-250 steps while latent $d'$ falls toward ~0.07-0.2 in the same range — a
real, if modest, gap between the latent metric and ground truth in the far tail. That gap, not
a 5-step wall, is the actual target for a quasimetric head: recovering discriminability past
~50 steps, where L2 genuinely has little left to give.

### E3 — Train the head the QRL-informed way, gate against L2

Changes from the first attempt in Section 2.1, each traced to an E0-E2 finding:
- `L_rand`'s fixed $D_{max}=50$ contrastive margin is **dropped entirely** — it fights
  cross-episode stitching, the thing this whole line of work exists to fix.
- The reverse/asymmetry penalty is restricted to active-pushing pairs (E1's exact
  construction) with a margin grounded in E1's own measurement, instead of a flat +10
  applied to every pair.
- A local upper-bound constraint is added: real one-step transitions must have
  $d_Q(s,s') \le 1$, enforced as a hard hinge penalty rather than full Lagrangian dual
  ascent — a stated simplification of the QRL (Wang & Isola 2022) formulation, not a full
  reproduction.
- The correlation-producing term is regression to $\log(1+k)$ on same-episode pairs across a
  spread of step gaps — kept from probe_gate.py's probe C rather than invented, since that
  exact recipe already scored held-out Spearman 0.9404 on this cache (E0).

```
python scripts/investigations/quasimetric/e3_train_quasimetric.py env=pusht n_episodes=18685
```

**Result** (2,336,736 frames, 20,000 training steps, ~520s on gnode087; 1,494,800 train /
373,700 test rank pairs, 619,955 train / 154,274 test active-pushing pairs, episode-level
held-out split):

| Quantity | Value |
|---|---|
| Held-out Spearman(true gap, $d_Q$) | **0.8605** (train 0.8838) |
| Held-out Spearman(true gap, raw L2), same pairs | 0.8164 |
| **Gate** | **PASSED** (0.8605 > 0.8164) |
| Learned asymmetry, held-out active-pushing pairs: $d_Q$(fwd) | 3.36 +- 1.51 |
| Learned asymmetry: $d_Q$(bwd) | 15.45 +- 6.02 |
| Learned asymmetry gap (bwd - fwd) | 12.08 +- 6.61 |
| Fraction of held-out pairs with bwd > fwd | **97.3%** |
| Raw L2 asymmetry on the same pairs (baseline, by construction) | 0.0 |

**Verdict: gate passed, with one honest caveat.** The trained head beats raw L2 on held-out
rank correlation (0.8605 vs. 0.8164, on identical pairs) — the E3 bar this document set. It
does **not** reach probe C's standalone 0.9404: adding the local-constraint and asymmetry
terms costs roughly 0.08 of correlation relative to training on the rank objective alone, a
real three-way tradeoff between correlation, scale-anchoring, and asymmetry rather than a free
win from adding more structure. The asymmetry result is the more interesting number: on
held-out active-pushing pairs, **97.3%** are correctly ordered backward-harder-than-forward —
close to E1's independently-measured, ground-truth-grounded irreversibility rate of 99.5% on
the same pair construction. That correspondence, not the Spearman margin, is the strongest
evidence this head learned something real about Push-T's dynamics rather than fitting noise:
two different measurements (a directed graph over real transitions in E1, and a trained head
in E3) landed on close to the same irreversibility rate independently.

Weights: `outputs/quasimetric/e3_quasimetric_head_pusht.pt`. Full results:
`outputs/quasimetric/e3_train_quasimetric_pusht.json`.

### E4 — Wire into planning, measure live rollout

Swapped the CEM planning cost per Section 2.2, via a minimal monkeypatch of the loaded
model's `criterion` method (the exact seam `stable_worldmodel`'s CEM solver calls) rather
than editing `jepa.py`/`eval.py` in place — a self-contained script, matching how E1-E3 were
built. Ran both conditions (raw L2 baseline, E3's quasimetric head) against the SAME 50
sampled eval episodes for a paired comparison, under the paper's live protocol (start = a
random dataset state, goal = the state 25 steps later in the same trajectory, 50-step
budget, success = the env's own `terminated` flag) — the exact protocol CLAUDE.md's
established checkpoint-health check (`eval.py policy=lewm-pusht eval.num_eval=50`) uses.

**Note on scope:** this swaps the CEM/MPC planning cost (Section 2.2's direct target), not
the separate offline-RL actor pipeline (`actor_train.py`/`actor_rollout_eval.py`, which
trains a policy network via IQL against graph-derived pseudo-values and is a different
mechanism entirely). An earlier draft of this section conflated the two; corrected here.
Only the paper's `same_episode` protocol was run — a true `cross_episode` CEM-planning eval
would need deeper changes to `stable_worldmodel`'s eval harness and was out of scope for
this pass.

```
python scripts/investigations/quasimetric/e4_live_rollout.py
```

**Result** (50 episodes per condition, ~63s / ~74s wall time, gnode087):

| Condition | Success rate |
|---|---|
| L2 baseline (unmodified `jepa.py` cost) | **94.0%** (47/50) |
| Quasimetric-JEPA (E3 head) | **76.0%** (38/50) |

The L2 number is not new — it reproduces CLAUDE.md's already-documented reproduction of the
paper's protocol (94.0% here vs. paper 96.0+-2.83%) to the exact percentage point, which is a
useful sanity check on this harness: the comparison methodology is not the source of what
follows.

**Verdict: real regression, not noise.** The quasimetric head, despite passing its E3 gate
(better held-out rank correlation than raw L2, 0.86 vs. 0.82, plus grounded asymmetry),
makes live planning **worse** — an 18-point drop, and the failures are not evenly spread
noise: 12 of the 50 episodes flip from success under L2 to failure under the quasimetric
cost. This is the ladder doing its job: E0-E3 all looked encouraging, and the one stage that
touches actual control is where the encouraging signal breaks.

The leading hypothesis, **not verified here**: CEM scores thousands of *predicted* rollout
terminal states per episode (`jepa.py`'s own predictor, evaluated on CEM-sampled action
candidates), not the *real, encoded* dataset frame pairs E3's rank/asymmetry losses were
trained and evaluated on. A held-out correlation gate on real frames says nothing about the
head's behavior on the out-of-distribution predicted embeddings CEM actually queries it
with — plausibly noisier and further off-manifold than anything E3 saw, especially early in
CEM's sampling before the action distribution has narrowed. The asymmetric potential term
($\Phi$, a ReLU of a scalar difference) is also a more jagged cost surface than smooth
squared L2, which is exactly the kind of shape CEM's variance-based elite selection is more
likely to be misled by. Neither of these is confirmed; both are testable follow-ups (e.g.
retraining E3 with pairs drawn from actual predictor rollouts rather than only real frames)
rather than claims this document is making.

**What this does not do:** it does not undermine E0-E3's own findings (the frozen latent is
informative, real irreversibility exists, L2 does saturate past ~50 steps, and a head can be
trained to beat L2 on held-out rank correlation with grounded asymmetry) — those are measured
facts about the representation and about offline training, and stand regardless. What it
shows is that offline correlation is not sufficient evidence for a planning-cost swap.

### E3-v2 / E4 retest — predictor-rollout-aware training

Tested the leading hypothesis directly: retrained the head with an added loss term built
from the model's own predictor, rolled forward autoregressively using REAL recorded
5-action blocks (`action_encoder`'s validated 10-dim block convention, chained 5 blocks deep
to match `plan_config.horizon=5` / `action_block=5` exactly — reproducing `jepa.py`'s
`rollout` inner loop verified line-by-line, starting from a cached real z0 rather than
re-encoding pixels). Each pair regresses `d_Q(predicted_terminal_embedding, real_25-step-
later_embedding)` toward a fixed `log(1+25)` target — a scale/domain-anchoring term, added
alongside E3's unchanged rank/local/asymmetry terms on real frames, not replacing them.
Full design and code: `scripts/investigations/quasimetric/e3v2_train_predictor_aware.py`.

```
python scripts/investigations/quasimetric/e3v2_train_predictor_aware.py env=pusht n_episodes=18685
QUASI_HEAD_NAME=e3v2_quasimetric_head_pusht.pt python scripts/investigations/quasimetric/e4_live_rollout.py
```

**Result:**

| Quantity | Value |
|---|---|
| E3-v2 gate: held-out Spearman(true gap, d_Q), real pairs | **0.8666** (train 0.8861) — slightly *better* than E3-v1's 0.8605 |
| vs. raw L2 (same pairs) | 0.8164 — gate still PASSED |
| Predictor-rollout diagnostic: d_Q(predicted, true 25-step target) | 3.30 +- 0.62 (target log(1+25) = 3.26 — on scale) |
| Predictor-rollout diagnostic: d_Q(predicted, random/shuffled target) | 8.53 +- 9.58 (correctly farther) |
| **E4 retest: live rollout success rate** | **64.0%** (32/50) |
| E4 v1 (original E3 head), for comparison | 76.0% (38/50) |
| L2 baseline, for comparison | 94.0% (47/50) |

**Verdict: the fix made it worse, not better.** Every offline diagnostic looked *healthier*
than E3-v1 — the gate improved, and the new predictor-rollout diagnostic shows the head's
predicted-vs-true-target distance sits right on its training target with good separation from
a random target. None of that transferred to live control: success rate dropped a further 12
points, to 64.0%, an even larger gap from the L2 baseline than before.

**Revised hypothesis for why:** the predictor-rollout diagnostic measures the wrong thing for
what CEM needs. It checks that d_Q's *mean* over predicted-vs-true-target pairs sits near a
fixed scalar — but CEM does not need "the average predicted rollout is roughly 25-ish units
from a real continuation"; it needs to *rank hundreds of different candidate rollouts against
each other* to find the best one. A loss that pulls every predictor-rollout embedding's score
toward the same constant target, using only REAL action sequences (never the diverse,
often near-random action candidates CEM actually samples, especially early in each solve),
plausibly *flattens* exactly the discriminative signal CEM's elite-selection step depends on
— punishing variance the offline mean-based diagnostic can't see. This reverses the earlier
guess that predictor-vs-real distribution shift was simply "unseen and therefore noisy": the
attempted fix suggests the shift matters in a way that a fixed-target anchoring term actively
fights against, not one it corrects.

**What would follow from here, not yet attempted:** train the predictor-rollout term as a
*ranking* signal instead of a fixed-target regression — sample several candidate action
sequences (including non-expert, higher-variance ones resembling CEM's actual sampling) from
the same start state, and require d_Q to rank them by their real eventual distance to a fixed
goal, rather than regressing every rollout toward one number. That is a materially different
and larger experiment (needs actual CEM-like negative action sampling, not just real
trajectory continuations) — attempted next, below.

### E3-v3 / E4 retest — ranking loss over real-env-grounded CEM-like candidates

Built the experiment E3-v2's postmortem called for. For 2,000 sampled start states (train
episodes), generated K=20 candidate action sequences each — 25 real actions, i.i.d.
z-scored Gaussian noise per real action, `var_scale=1.0` — matching
`config/eval/solver/cem.yaml`'s CEM sampling distribution exactly, not just real trajectory
continuations. Each candidate's GROUND TRUTH outcome distance came from actually stepping
the real Push-T physics simulator (there is no dataset row that tells you what a random
action sequence achieves, so this is unavoidable): reset to the start state, de-normalize
the z-scored actions back to raw units via the exact inverse used by
`stable_worldmodel.policy.WorldModelPolicy` (`process['action'].inverse_transform`, verified
by reading its source directly), step 25 times, measure true distance from the resulting
state to the fixed goal (the real 25-step-later continuation, matching E4's own protocol)
via the same projected/normalized oracle used throughout this project. The SAME candidates
were also rolled through the model's actual predictor to get their predicted embeddings.
A pairwise margin-ranking loss then requires `d_Q` to rank candidates in the same order as
their real physics outcome — supervising relative ordering directly, not a fixed scalar.
Full design and code: `scripts/investigations/quasimetric/e3v3_train_ranking_cem_like.py`.

Real env stepping turned out to be cheap for Push-T (~30ms per 25-step rollout), so the pool
reached 40,000 real rollouts (2,000 groups x 20 candidates) rather than a token-sized one.

```
python scripts/investigations/quasimetric/e3v3_train_ranking_cem_like.py env=pusht n_episodes=18685
QUASI_HEAD_NAME=e3v3_quasimetric_head_pusht.pt python scripts/investigations/quasimetric/e4_live_rollout.py
```

**Result:**

| Quantity | Value |
|---|---|
| E3-v3 gate: held-out Spearman(true gap, d_Q), real pairs | 0.8390 — PASSED vs. raw L2's 0.8164 |
| Within-group Spearman(true CEM-candidate outcome distance, d_Q) | **0.9205**, mean over 2,000 real-env-grounded groups (40,000 rollouts) |
| **E4 retest: live rollout success rate** | **60.0%** (30/50) |

| Condition | Live rollout success rate |
|---|---|
| L2 baseline | 94.0% |
| E3-v1 (real frames only) | 76.0% |
| E3-v2 (fixed-target predictor anchor) | 64.0% |
| **E3-v3 (ranking loss, real-env-grounded CEM candidates)** | **60.0%** |

**Verdict: worse again, and the trend itself is now the finding.** This was the most
carefully targeted fix of the three — real physics ground truth, CEM's actual sampling
distribution, a ranking objective that directly matches what elite selection consumes — and
by its own within-group metric it works well (0.92 Spearman on 2,000 genuinely
CEM-representative comparisons, the strongest and most relevant diagnostic run in this whole
document). It still made live control worse than the version before it. Three attempts, each
more sophisticated and each passing progressively more targeted offline checks, produced a
**monotonic decline** in live success: 76.0% -> 64.0% -> 60.0%, moving further from L2's
94.0% each time, not closer.

That monotonic trend across three structurally different fixes is more informative than any
one of them individually: it weighs against "the training distribution didn't match what CEM
queries" as the operative explanation (E3-v3 fixed that about as directly as a training
distribution can be fixed, and it still made things worse), and toward something more
architectural. The leading open hypothesis, **not tested here**: CEM is a 30-iteration
zeroth-order optimizer that aggressively narrows toward whatever minimizes the cost it is
given — a well-known failure mode in model-based RL is that this kind of optimizer exploits
any exploitable irregularity in a *learned* cost surface (the "optimizer's curse"), finding
action sequences whose predicted embedding scores well under `d_Q` without actually being
physically close to the goal. Squared L2 against a real target embedding is comparatively
hard to exploit this way — anything that minimizes it is, by construction, close to that
embedding in whatever geometry the encoder itself organizes. A separately-trained scalar
network, even one that is accurate on realistic distributions of pairs, has far more degrees
of freedom to be *locally* wrong in a way a 300-candidates x 30-iterations search is well
suited to find and climb toward. If that is the real mechanism, better training data alone —
which is what all three attempts here changed — cannot fix it; what would be needed instead
is something that constrains the cost surface's smoothness or robustness under active
optimization (e.g. adversarial training against CEM-discovered failures, an ensemble/
uncertainty penalty, or a Lipschitz/smoothness constraint on the head), not a better dataset.
This is offered as the most consistent explanation of the pattern actually observed, not as
something separately verified — the direct check (e.g. comparing the *real* physical outcome
of the CEM-chosen action sequence against what `d_Q` predicted for it, on the actual failed
E4 episodes) would confirm or rule it out.

### E5 — does predicted cost track true outcome? (the direct check, run)

Ran the check flagged above. `plan_config`'s horizon(5) x action_block(5) = 25 =
`goal_offset_steps` exactly, and `receding_horizon` = `horizon` (no partial-plan truncation)
— so capping `eval_budget=25` forces `world.evaluate()` to do exactly ONE CEM solve per
episode, execute its entire chosen plan, and stop, with no second replan to muddy the
picture. `stable_worldmodel`'s `CEMSolver.solve` was wrapped (not modified) to capture the
elite-mean cost of its own chosen plan, while `world.infos['state']` after the (otherwise
completely unmodified) `world.evaluate()` call gives the TRUE resulting state — reusing the
same reset/callables/CEM-batching machinery E4 already validated rather than hand-rolling it.
For each of the 4 conditions, on the same 50 sampled (start, goal) pairs: the true oracle
distance from that real resulting state to the real goal, correlated against the model's own
predicted cost for the plan that produced it. Full code:
`scripts/investigations/quasimetric/e5_cost_hacking_diagnostic.py`.

```
python scripts/investigations/quasimetric/e5_cost_hacking_diagnostic.py
```

**Result** (n=50, one-shot eval_budget=25 — a stricter, single-replan variant of E4's
protocol, so these success rates are not directly the E4 numbers, though similarly ordered):

| Condition | One-shot success | True dist-to-goal (mean +- std) | Spearman(predicted cost, true dist) |
|---|---|---|---|
| L2 baseline | 76.0% | **0.297 +- 0.267** (best) | 0.056 (not significant; SE ~ 0.14 at n=50) |
| quasimetric v1 | 62.0% | 0.540 +- 0.633 | **0.649** |
| quasimetric v2 | 54.0% | 0.634 +- 0.631 | 0.537 |
| quasimetric v3 | 54.0% | 0.543 +- 0.565 | **0.653** |

**Verdict: this refutes the cost-hacking hypothesis as stated, not confirms it — and the
actual pattern is more interesting.** Under cost-hacking, the quasimetric conditions should
show weak or negative correlation between their own predicted cost and true outcome (CEM
finding candidates that score well without actually being close). The opposite happened:
**all three quasimetric heads correlate with true outcome distance far MORE strongly than L2
does** (0.54-0.65 vs. 0.056, essentially zero for L2). The model is not being fooled by its
own cost among the candidates CEM actually converges to — locally, its ranking is honest.

What's actually happening is a systematic **absolute offset**: every quasimetric condition
converges to a genuinely worse neighborhood of action space (true distance roughly 2x L2's)
while still ranking that neighborhood's candidates correctly relative to each other. L2's
own near-zero correlation among its chosen elites is itself informative, not a contradiction
— restriction of range: CEM has already spent 30 iterations squeezing out the bulk of L2's
discriminative signal by the time it returns elites, so what's left is close to noise, yet
the neighborhood it lands in is still the better one. The quasimetric heads retain more
locally usable signal within their elite set, but that set itself sits in a worse region.

This points away from "the cost function can be fooled" and toward something like: **the
learned cost surface has a different, harder-to-navigate shape than smooth squared L2** —
plausibly the asymmetric potential term ($\Phi$, a ReLU of a scalar difference) creating flat
or oddly-curved regions that CEM's Gaussian-narrowing search converges into a genuinely
different (and worse) attractor than L2's simpler bowl, not because it's deceived there but
because that's just where a rougher landscape leads a 30-iteration local search. This is
still not fully isolated (would need e.g. directly visualizing/profiling the cost-vs-action
landscape, or an ablation removing $\Phi$ entirely to test whether the asymmetric term
specifically is the source of the roughness) but it is a materially more precise, and better
evidenced, account than the unconfirmed exploitation guess it replaces.

**Where this leaves the whole line of work:** four separate attempts (E3-v1 through v3, each
passing progressively better offline gates, plus this diagnostic) have not found a way to
make a learned quasimetric cost help CEM planning on Push-T — every version tested makes live
control worse than the L2 baseline it was meant to replace, and E5 now gives a specific,
falsifiable reason why (landscape shape under active optimization, not miscalibration or
distribution mismatch, both of which were tested and addressed without closing the gap). The
representation-level findings (E0-E2: informative latent, real irreversibility, real
saturation past ~50 steps) remain solid measurements and are not in question. The planning
result is the part of this proposal that is not supported by the evidence gathered so far.

---

## 4. Reproducibility

```powershell
# E0 -- frozen-latent sufficiency (reused, not new)
python scripts/probe_gate.py env=pusht n_episodes=18685

# E1 -- real directed-graph asymmetry
python scripts/investigations/quasimetric/e1_directed_asymmetry.py env=pusht

# E2 -- saturation curve
python scripts/investigations/quasimetric/e2_saturation.py env=pusht n_episodes=18685

# E3 -- train the head, gate against L2 (~520s on an RTX 2080 Ti)
python scripts/investigations/quasimetric/e3_train_quasimetric.py env=pusht n_episodes=18685

# E4 -- live rollout, both cost conditions (~180s total)
python scripts/investigations/quasimetric/e4_live_rollout.py
python scripts/investigations/quasimetric/e4_live_rollout.py eval.num_eval=8   # smoke test

# E3-v2 -- predictor-rollout-aware retrain (~1040s), then retest E4 with QUASI_HEAD_NAME
python scripts/investigations/quasimetric/e3v2_train_predictor_aware.py env=pusht n_episodes=18685
QUASI_HEAD_NAME=e3v2_quasimetric_head_pusht.pt python scripts/investigations/quasimetric/e4_live_rollout.py

# E3-v3 -- ranking loss over real-env-grounded CEM-like candidates (~1210s), then retest E4
python scripts/investigations/quasimetric/e3v3_train_ranking_cem_like.py env=pusht n_episodes=18685
QUASI_HEAD_NAME=e3v3_quasimetric_head_pusht.pt python scripts/investigations/quasimetric/e4_live_rollout.py

# E5 -- does the CEM-chosen plan's predicted cost track its true outcome? (~255s, all 4 conditions)
python scripts/investigations/quasimetric/e5_cost_hacking_diagnostic.py
```

On Ada: run from `/home2/mayaank.ashok/lewm_research`, `PROBE_CACHE_DIR` and `PUSHT_H5_PATH`
pointed at `/ssd_scratch/mayaank.ashok/` per this repo's usual staging convention (see
`CLAUDE.md`). E0/E2/E3/E3-v2/E3-v3 reuse the same memmapped 2.34M-frame cache; E1 reuses the
existing 300-episode landmark cache. Neither needs re-encoding once the cache exists. E4/E5
additionally need `PUSHT_CKPT_DIR` (or its default) and a trained head, e.g.
`outputs/quasimetric/e3_quasimetric_head_pusht.pt`
from a completed E3 run.

The old `scripts/investigations/test_quasimetric_failures.py` and `verify_quasimetric_fix.py`
are left in place for reference (they are what E1's numbers replace and what motivated the
"first attempt" note in Section 2.1) but should not be cited as evidence going forward.
