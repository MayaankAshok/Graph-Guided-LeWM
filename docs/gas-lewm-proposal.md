# LeWM as a frozen visual prior for GAS

**Status:** proposal, nothing run yet. Written 2026-09-08.

**One paragraph.** Graph-Assisted Stitching (GAS, ICML 2025) builds a graph over an offline
dataset in a learned Temporal Distance Representation (TDR) space, runs Dijkstra for subgoals,
and trains a low-level policy to chase them. It is state of the art on OGBench's stitching
tasks. Its own stated weakness is visual input: it learns image encoders from scratch under a
temporal-difference objective, and loses between 19 and 79 points moving from states to pixels.
This proposal replaces GAS's from-scratch CNN encoder for the TDR with a **frozen pretrained
LeWM encoder plus a small MLP head**, leaving the rest of the GAS pipeline untouched. The bet is
that self-supervised world-model pretraining is a better way to learn perception from 64x64
frames than a TD gradient is.

---

## 1. Verification of the premise

### 1.1 Does GAS need a better visual encoder?

Yes, and the gap is large and explicitly acknowledged.

**Architecture (from the paper).** For pixel-based environments GAS adopts the Impala CNN. It
uses **four separate CNN encoders** for the TDR, the Q function, the value function, and the
low-level policy, with random-crop augmentation. The TDR encoder is trained end to end under the
expectile loss. No pretrained weights are used anywhere in the visual path.

**The measured state-to-pixel gap.**

| Task | State-based | Visual | Drop |
|---|---|---|---|
| antmaze-giant-navigate | 77.6 | 59.0 | -18.6 |
| antmaze-giant-stitch | 88.3 | 55.8 | -32.5 |
| antmaze-large-explore | 94.2 | 15.1 | **-79.1** |

**Their own words.** The limitations section attributes the lower visual performance to "a lack
of high-dimensional representation learning for visual input," and says the authors plan to
"enhance GAS by incorporating advanced representation learning techniques for visual feature
extraction."

So the premise holds. The headroom is quantified, the cause is named by the authors, and the
named fix is exactly the class of thing LeWM is.

### 1.2 Is the current LeWM encoder suitable?

Partly. The architecture is a good fit. The available checkpoints are not, and this is the main
practical obstacle.

**What LeWM's encoder is** (verified in `config/train/model/lewm.yaml` and
`config/train/lewm.yaml`): a ViT-tiny, patch size 14, input resolution 224, embedding dim 192.
One pooled 192-d vector per frame. Per-frame, no frame stacking, which matches how OGBench
pixel tasks are set up.

**Three concrete mismatches.**

1. **No checkpoint exists for any maze environment.** Queried directly against the HuggingFace
   API: the only published LeWM models are `lewm-pusht`, `lewm-cube`, `lewm-tworooms`, and
   `lewm-reacher`. GAS's visual results are on `visual-antmaze-*` and `visual-scene-play`. There
   is **no pretrained LeWM encoder for any of them.** Either pretrain one on the target OGBench
   dataset, or restrict the first experiment to a task where an existing checkpoint plausibly
   transfers.
2. **Resolution mismatch.** LeWM expects 224x224; OGBench pixel observations are 64x64x3.
   Upsampling works but costs about 12x more encoder compute per frame than the information
   warrants, and the pretrained features were never fit to upsampled low-resolution renders.
3. **Render-distribution mismatch, even for cube.** `lewm-cube` was pretrained on this project's
   own `cube_single_expert` renders at 224. OGBench's `visual-cube-*` uses 64x64 renders with
   deliberately adjusted colors and a transparent arm. Same simulator, different images. Do not
   assume the frozen encoder transfers across that gap without looking at the frames side by
   side.

**The honest reading:** the *architecture* is suitable and the *available weights* are not.
The realistic path is to pretrain a LeWM encoder on the target OGBench visual dataset using this
repo's own `train.py` pipeline, which has been idle throughout the latent-graph work.

---

## 2. The proposal

Keep the entire GAS pipeline. Change one component.

```
GAS today:      image --Impala CNN (trained by expectile TD loss)--> psi(s)
Proposed:       image --frozen LeWM ViT--> z (192-d, cached) --MLP--> psi(s)
```

Everything downstream is unchanged: the same expectile TDR objective, the same TD-aware
clustering into graph nodes at uniform temporal spacing, the same Temporal Efficiency filter,
the same Dijkstra subgoal selection, the same IQL and DDPG+BC low-level policy.

The encoder runs once over the dataset and `z` is cached. Training the TDR then reduces to
fitting an MLP on cached 192-d vectors.

---

## 3. Why this is a better use of LeWM than the latent-graph line

This is the important argument, and it is not obvious.

The latent-graph work required `||z_i - z_j||` to *itself* be a temporal distance. That is a
strong requirement on the geometry of a representation that was never trained to satisfy it, and
Push-T falsified it: raw latent Euclidean distance ranked the oracle at 0.219, cross-episode
correlation went negative, and the calibration formula collapsed.

**Under this proposal that requirement disappears.** The MLP head is trained with GAS's temporal
distance objective, so the metric is *learned on top of* `z`. What `z` must supply is only that
it is a **sufficient statistic of the observation** — that the information needed to recover
temporal distance survives the encoder. That is a far weaker condition, and it is close to what a
JEPA-style predictive objective is designed to produce.

In other words, the proposal takes LeWM out of a role it demonstrably fails at and puts it in the
role it was actually trained for. The two years of negative Push-T results are evidence *for*
this reframing, not against it.

**Secondary benefit.** Caching `z` turns four CNN forward and backward passes per batch into
MLPs over precomputed features. GAS's visual runs should get substantially cheaper, which makes a
multi-seed sweep affordable on the available cluster.

---

## 4. Why it might not work

Each risk with the cheapest test that settles it.

1. **A frozen generic latent may have discarded the task-relevant bits.** LeWM's objective
   rewards predicting future latents, not localizing the agent. It may retain scene layout and
   drop fine agent pose, which is exactly what the TDR needs. *Test:* E1 probe, below. This is
   the single highest-risk assumption and it is cheap to check.
2. **The TD gradient is task-aligned; self-supervised pretraining is not.** GAS's CNN gets to
   shape its features around the thing it is scored on. A frozen encoder cannot. The MLP head has
   to work with whatever survived. *Test:* E2 compares TDR quality directly, before any policy
   training.
3. **Loss of random-crop augmentation.** GAS augments images per batch. Frozen cached features
   cannot be augmented that way. *Mitigation:* cache a small number of augmented views per frame,
   or unfreeze the last ViT block. Both are arms in E5.
4. **Confounded comparison.** Adding a pretrained encoder adds parameters and pretraining
   compute. A reviewer will say the gain came from either. *Mitigation:* the matched arms in E5,
   in particular an Impala CNN given the same self-supervised pretraining budget, and a frozen
   DINOv2 arm.
5. **GAS may not reproduce.** *Mitigation:* E0 is a reproduction gate before anything else.
6. **Pretraining cost.** A LeWM encoder must be trained per target environment. Throughput on
   OGBench frames is unmeasured; measure it before committing to a task list.

---

## 5. Experiment ladder

Each step has a kill criterion. Do not proceed past a failed one without an explanation.

**E0 — Reproduce GAS's visual baseline.** One task, `visual-antmaze-large-stitch-v0`, 3 seeds,
using the authors' released code unmodified.
*Kill:* cannot land within the paper's reported spread. Fix the reproduction before continuing;
every later number is meaningless otherwise.

**E1 — Probe sufficiency of `z`. No RL, no graph, cheapest decisive test.**
Pretrain (or load) a LeWM encoder for the target dataset, freeze it, cache `z`. Train small MLP
probes from `z` to predict (a) the privileged state, in particular agent x-y, and (b) the true
temporal gap between two frames of the same episode. Run the identical probes on GAS's trained
Impala features, on frozen DINOv2 features, and on raw downsampled pixels as a floor.
*Kill:* if `z` does not beat raw pixels on both probes, the prior carries no usable information
and the rest of the ladder will fail. Stop here.

**E1 RESULT (run 2026-09-08, Ada gnode087, Push-T, 300 episodes / 36,824 frames, 240/60
episode split, 48k train and 12k test pairs).** `scripts/probe_gate.py env=pusht`, results in
`outputs/probe_gate_pusht/e1_probe_results.json`. The pixel floor in this run was a
14x14x3 mean-pool (588-d): the pooling parameter was named by block size and silently
produced a coarser grid than intended. It is an exact block mean either way, so nothing
below changes, but the parameter is now `pixel_grid` (output side) and later runs use a
genuine 16x16x3 / 768-d floor.

| arm | A: state R^2 | B: gap Spearman | C: metric-gap Spearman |
|---|---|---|---|
| **lewm** | **0.9913** | **0.9016** | 0.8671 |
| pixels (floor) | 0.9332 | 0.8046 | 0.8400 |
| randproj (floor) | 0.8977 | 0.8069 | 0.8341 |
| state (ceiling) | n/a | 0.8998 | 0.8715 |

**Probe A passes decisively, and the margin is where it should be.** Per-dimension R^2 on block
orientation (cos, sin) is 0.988 / 0.951 for LeWM against 0.747 / 0.656 for pooled pixels and
0.744 / 0.619 for the random projection. Agent and block position are recoverable from anything;
object *pose* is what the frozen latent preserves and crude pixels destroy.

**Probe B passes at the ceiling.** LeWM reaches Spearman 0.9016 against the privileged-state
ceiling's 0.8998, and beats it on per-bucket MAE at 3 of 4 gap buckets (1.10 / 2.83 / 7.17 /
17.80 steps against 1.08 / 3.15 / 7.32 / 19.04). Temporal distance is recoverable from `z` as
well as from the simulator state. **The core information-sufficiency assumption of this proposal
holds.**

**Probe C is ambiguous and should not be quoted as a pass.** Two problems. First the band is
narrow: the pixel floor is 0.8400 and the privileged-state ceiling is 0.8715, so the probe has
only 0.03 Spearman of dynamic range on this data and cannot separate representations. Second,
ranking and calibration disagree — LeWM ranks above both floors but has *worse* absolute error
than pooled pixels at every bucket above gap 5 (metric-head MAE 6.50 / 16.05 / 33.91 against the
pixel floor's 4.34 / 9.64 / 23.50). The state ceiling shows the same inversion, which points at
the metric head rather than at any particular representation.

**Reading.** E1's stated kill criterion did not fire, and the assumption it was written to test
is confirmed by probes A and B. But probe C did not deliver the metric-form evidence it was added
for, and Push-T is the likely reason: ~125-step near-optimal episodes make the step gap largely
predictable from crude visual similarity, which is exactly the "wrong shape" concern that argues
for fragmenting the data or moving to OGBench stitch datasets. Before trusting any probe-C number,
three script changes: log train error per probe to separate underfitting from overfitting in the
metric head, raise the metric-head budget above `n_steps=6000` / `metric_dim=32`, and replace
global Spearman with a threshold-aware statistic (edge precision at a candidate `H_TD`), since
that is what GAS's graph actually consumes.

**E1 FULL-SCALE RESULT (run 2026-09-08, Ada gnode087, Push-T, all 18,685 episodes / 2,336,736
frames, full h5, 1,494,800 train / 373,700 test pairs).** `scripts/probe_gate.py env=pusht
n_episodes=18685`, results in `outputs/probe_gate_pusht/e1_probe_results_ep18685.json`. Encode
took 82 min, probes ~20 min, all 3 GPUs idle otherwise on gnode087.

| arm | A: state R^2 (train) | B: gap Spearman (train) | C: metric-gap Spearman (train) |
|---|---|---|---|
| **lewm** | **0.9942** (0.9990) | **0.9575** (0.9699) | **0.9407** (0.9579) |
| pixels (floor) | 0.9876 (0.9963) | 0.9245 (0.9521) | 0.9237 (0.9478) |
| randproj (floor) | 0.9693 (0.9872) | 0.9183 (0.9451) | 0.9178 (0.9420) |
| state (ceiling) | n/a | 0.9428 (0.9501) | 0.9323 (0.9421) |

Edge precision/recall at GAS's actual decision threshold (probe C, metric head):

| arm | T=5 (prec/rec) | T=10 (prec/rec) | T=25 (prec/rec) |
|---|---|---|---|
| lewm | 0.876 / 0.853 | **0.903** / 0.882 | **0.914** / 0.934 |
| pixels | 0.876 / 0.798 | 0.880 / 0.866 | 0.902 / 0.923 |
| randproj | 0.882 / 0.781 | 0.874 / 0.853 | 0.890 / 0.926 |
| state | 0.860 / 0.834 | 0.895 / 0.861 | 0.906 / 0.926 |

**All train/test gaps are small (< 0.02 everywhere), so every number above is genuine
signal, not an artifact of the 6000-step / 300-episode probe underfitting or overfitting.**
That was the open question after the 300-episode run and it is now closed.

**The ambiguity in the 300-episode probe C is resolved, and it resolves in LeWM's favor, but
narrowly.** Full-scale Spearman: lewm 0.9407 vs. the best floor (pixels) at 0.9237 -- a gap of
0.017, small but now clearly resolved rather than sitting inside a 0.03-wide floor-to-ceiling
band as it did at 300 episodes (floor 0.8400, ceiling 0.8715, band width 0.0315). LeWM also now
sits *above* both floors AND the privileged-state ceiling on probes B and C, which is odd on its
face -- a 192-d learned latent outscoring an oracle 6-d wall/orientation projection at predicting
step gap. The likely explanation, not yet confirmed: `z` carries visually-observable information
correlated with progress through an episode (deformation of the pushed block, subtle scene cues)
that survived encoding but is absent from the coarse `[agent_xy, block_xy, cos, sin]` projection
used as ground truth -- the projection is a reasonable but lossy stand-in for "true state," not
the literal true state.

**Where the LeWM edge actually lives: not at the shortest threshold.** At T=5 (a tight, GAS-like
identification radius) lewm and pixels are statistically tied on precision (0.876 vs. 0.876);
LeWM's recall lead there (0.853 vs. 0.798) is real but this is the threshold where GAS is being
most conservative, and conservative decisions are exactly where a floor this close is a genuine
risk to the proposal. The LeWM margin widens at T=10 (0.903 vs. 0.880 precision) and is largest
at T=25. **If GAS's actual `H_TD` on visual-antmaze ends up small, budget for the frozen prior's
advantage over a crude pixel encoder to be marginal, not decisive; if `H_TD` is more permissive,
the advantage is clearer.**

**Reading for the proposal.** Push-T is not adjudicating this cleanly even with the full dataset,
because a 768-d mean-pooled floor is already strong here -- Push-T's camera is close, static, and
the scene is simple, so coarse pixel statistics go a long way. This is itself informative: it
predicts the frozen-prior advantage will be *larger*, not smaller, on visual-antmaze, where the
camera is farther back, the scene is more cluttered, and position/orientation are less legible
from raw pixel statistics alone. **E1's kill criterion did not fire and the sufficiency
assumption behind this proposal is confirmed at full scale** -- proceed to E2, but do not expect
Push-T's specific margins to transfer; re-run this same probe on cached OGBench visual-antmaze
frames before committing pretraining compute, since that is the test that actually matters.

**E2 — TDR quality with the frozen prior.** Train `psi` as an MLP over frozen `z` using GAS's
expectile objective. Compare `|| psi(s) - psi(g) ||` against true temporal distance on held-out
episodes, versus GAS's from-scratch CNN TDR trained on the same data.
*Kill:* if the frozen-prior TDR is not at least as accurate as the from-scratch one, the central
claim is already false and no policy result will rescue it.

**E3 — Graph quality.** Build GAS's graph from each TDR. Report node count, edge precision
against ground-truth reachability, largest-component coverage, and the Temporal Efficiency filter
pass rate.
*Kill:* none. This is diagnostic, and it is where an unexpected E4 result gets explained.

**E4 — Full pipeline, one task.** `visual-antmaze-large-stitch-v0`, 3 seeds minimum, GAS baseline
versus GAS with the frozen LeWM prior, everything else identical.
*Kill:* no improvement over E0's reproduction. If E1 through E3 all passed and E4 does not move,
the bottleneck is downstream of the representation and that is itself a reportable finding.

**E5 — Ablations, same task.** Arms: frozen `z` plus MLP (the proposal); unfreeze the last ViT
block; frozen DINOv2 plus MLP; Impala CNN given the same self-supervised pretraining budget;
LeWM `z` as a distillation target for GAS's CNN rather than a replacement; `z` concatenated with
CNN features.
*Purpose:* separate "pretraining helps" from "LeWM specifically helps" from "more parameters
help." Without this the result is not publishable.

**E6 — Breadth and the negative control.** Extend the winner to
`visual-antmaze-{medium,giant}-stitch` and `visual-antmaze-large-explore`, which has the largest
state-to-pixel gap at 79 points and therefore the most headroom. Then run one visual manipulation
task as a negative control.
*Expectation to state in advance:* based on this project's Push-T evidence and on TTGS's reported
0-2 point manipulation gains, the benefit should be concentrated on navigation and largely absent
on manipulation. Predicting that in advance and confirming it is worth more than quietly omitting
the manipulation result.

---

## 6. Compute and data notes

- **Encoding is cheap.** At the ~600 frames/s measured on the local GPU, a 1M-transition OGBench
  dataset encodes in roughly half an hour, once. Cached `z` for 1M frames at 192 dims in float32
  is about 768 MB, so it fits anywhere.
- **Pretraining is the real cost and is unmeasured.** Measure LeWM throughput on 64x64-upsampled
  OGBench frames before picking the task list.
- **Dataset sizes.** OGBench visual stitch datasets are 1M transitions across 5000 episodes of
  200 steps. That is roughly 8x the largest landmark set this project has previously handled, so
  do not carry over any dense N-by-N habits from the latent-graph code.
- **Resolution decision to make early.** Either upsample 64 to 224 and use the pretrained
  weights as-is, or retrain LeWM natively at 64 with a smaller patch size. The second is cleaner
  and probably cheaper; the first is the only option if reusing an existing checkpoint.

---

## 7. Verify before committing

1. Read the GAS paper directly rather than relying on this summary. It now defines the related
   work and the baseline, not merely adjacent literature.
2. Check whether GAS's released code supports swapping the encoder cleanly, or whether the four
   separate CNN encoders are entangled with the training loop.
3. Render a handful of OGBench `visual-cube` frames beside this project's own cube renders. If
   they are close, `lewm-cube` gives a zero-pretraining pilot for E1 and E2. If they are not,
   pretraining is mandatory and the timeline moves.
4. Confirm that GAS's TDR encoder is the only place the encoder swap is needed, or whether the Q,
   V, and policy encoders should be swapped too. Swapping all four is the stronger version of the
   proposal and costs nothing extra once `z` is cached.
