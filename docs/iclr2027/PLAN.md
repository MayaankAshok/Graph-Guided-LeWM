# ICLR 2027 plan for the graph-assisted latent MPC paper

Written 2026-09-18. Companion to `main.tex` (paper draft) in this directory.
Numbers below are copied from `docs/gas-mpc/main.tex`, `results_table*.tex`, and the
Cube eval JSONs on Ada (job 2699936) as of 2026-09-18 13:51 IST.

---

## 0. The deadline changes everything

| ICLR 2027 | Date (AoE) |
|---|---|
| Abstract registration | **Fri 18 Sep 2026** (today) |
| Full paper | **Fri 25 Sep 2026** (7 days) |
| Reviews out / discussion | 5 Nov / 5-18 Nov 2026 |
| Decisions | 16 Dec 2026 |

Sources: [CFP](https://iclr.cc/Conferences/2027/CallForPapers),
[Author guidelines](https://iclr.cc/Conferences/2027/AuthorGuidelines).

Recommendation:
1. Register the abstract today. It costs nothing, the title/abstract can be edited until
   the 25th, and a registered submission can be withdrawn. Without it there is no ICLR
   2027 option at all.
2. Decide go/no-go on **Tue 22 Sep** using the P0 checklist in section 4. If the Cube
   baseline discrepancy (section 1.3) and the held-out-task check are not resolved by then,
   submit to ICML 2027 (deadline expected late Jan 2027) with the full comparison set
   instead. A rejected ICLR submission with a thin baseline set is not a free shot: the
   paper is de-anonymised after review under the 2027 policy.
3. The 7-day version of the paper is Push-T (full ablation ladder) + Reacher + Cube
   (headline only). Two-Room is a stretch goal (section 4.2).

---

## 1. Suitability assessment

### 1.1 What exists today

Headline configuration **B** = TDR graph subgoal one lookahead ahead + final-goal switch at
one lookahead with LeWM's own z-space L2 + budget-capped expected-hitting-time critic term
(standardised, beta = 1). Baseline **L2** = LeWM's CEM with terminal latent L2. Same fixed
tasks, same CEM, same predictor. 50 tasks x 5 CEM seeds unless noted.

| Env | Method | same-ep 25 | same-ep 50 | same-ep 100 | cross-ep |
|---|---|---|---|---|---|
| Push-T | L2 | 91.2 +- 3.0 | 48.4 +- 4.6 | 11.2 +- 2.0 | 10.4 +- 3.4 |
| Push-T | B  | 90.8 +- 2.7 | **64.4** +- 4.6 | **50.4** +- 5.0 | **44.4** +- 3.9 |
| Reacher | L2 | 86.8 +- 4.8 | 96.0 +- 1.4 | 76.0 +- 1.4 | 24.4 +- 1.7 |
| Reacher | B  | 80.8 +- 5.2 | 97.6 +- 2.6 | **98.0** +- 0.0 | **64.0** +- 2.8 |
| Cube (partial, 18 Sep) | L2 | 56.8 (5 s) | 50.8 (5 s) | 62.0 (5 s) | 12.0 (2 s) |
| Cube (partial, 18 Sep) | B  | 66.8 (5 s) | 56.4 (5 s) | 70.4 (5 s) | 34.0 (1 s) |
| Two-Room | -- | no GAS-MPC run exists | | | |

Push-T also has the full ablation ladder (subgoal only -> final switch -> critic variants
-> beta sweep -> composition variants -> retrieval -> TE / H_TD / lookahead sweeps), the
per-pair geodesic diagnostic, the single-pass predicted-vs-realized audit, and the oracle
Spearman audit of L2 vs the critic. Reacher and Cube have only L2 vs B.

### 1.2 Is "four environments" sufficient?

The count is not the problem. LeWM itself uses exactly these four; SAGE and Hi-LeWM use
two (Push-T, Cube); TRM uses three. Four is at the upper end for this literature. Three
other things matter more to a reviewer:

1. **Two-Room has no result.** "Four environments" is currently three, one of them
   half-finished. Two-Room is also the environment where TRM reports LeWM going from 7% to
   96.7% with a learned temporal cost, so a reviewer who knows that paper will ask.
2. **Breadth of baselines, not of environments.** Every table is "L2 vs ours". The 2026
   LeWM-planning literature has at least six methods that attack the same long-horizon
   failure on the same frozen model (section 2). At least two of them must appear as rows.
3. **Cube's baseline is below the published number** (next item). An unexplained weak
   baseline invalidates the Cube column.

### 1.3 Known problems a reviewer will find (fix or disclose)

- **Cube L2 baseline -- RESOLVED 2026-09-18** (`docs/gas-mpc/main.tex` sec:cube-baseline-audit).
  No computation mismatch: `gas_mpc_eval` matches upstream `World._evaluate_from_dataset`
  step for step, and tasks 0-49 of the 200-task runs reproduce the old 54/52/58 exactly.
  The 56.8 is a hard task draw: the first-50 prefix has 22% pre-solved tasks vs 38.4% of
  all dataset windows (success is set by block displacement), and the paper's 74 is itself
  a single 50-task draw (upstream `eval.py`, same checkpoint: 68/82/58/66/74 at CEM-30,
  72/82/64 at the paper's CEM-10). Corrected L2 = whole 200-task pool, 3 seeds:
  61.5/63.5/68.0 = **64.3**, matching the population prediction 65.4 and SAGE's 66.7.
  B on the same pool: see the tex section. Quote the 200-task numbers for Cube same25.
- **Goal-in-graph leakage.** Eval start/goal frames are dataset frames, and the graph is
  built on 98% of episodes, so the goal frame is (almost always) a graph node's member.
  This is LeWM's standard protocol, but for a graph method it is a fair objection. Run the
  headline on tasks drawn from the 2% held-out episodes (or re-split with a larger holdout)
  and show the gap is small.
- **Critic labels use privileged state.** Hindsight labels come from the env's success
  predicate evaluated on the dataset's state column; cross-episode negatives are gated on
  block pose / joint distance. The encoder, predictor, TDR and graph are pixel-only, the
  critic is not. Say so in the method section, not in a footnote, and give the pixel-only
  fallback (graph hitting-time labels: `viability_train_ht.py`; weaker but label-free).
- **Reacher short-range loss.** B is 6 points under L2 at same25 (80.8 vs 86.8). The
  10-seed rerun says it is CEM noise (82.6 vs 85.2, p = 0.37) but the sign is consistent.
  Report the 10-seed number and the `critic_final=false` variant.
- **Many moving parts.** TDR + graph + subgoal rule + final switch + critic + ET cost +
  standardisation. The defence is the ladder (each row adds one part and the reader sees
  what it buys) and the two-mechanism story (direction vs feasibility). Do not add parts.

### 1.4 Verdict

As it stands: a solid workshop paper or a borderline main-track submission. The story is
clean, the mechanistic analysis is better than most of the competing papers, and the
cross-episode result (10 -> 44% Push-T, 24 -> 64% Reacher, every seed) is a genuinely new
number in this line. What is missing is external comparison. With SAGE-style and
TRM-style rows, the held-out-task check, and a fixed Cube column, it is a
competitive main-track paper. Two-Room on top makes the "four environments" claim true.

---

## 2. Positioning and novelty

### 2.1 The neighbourhood (read these before writing Related Work)

Same frozen LeWM, same long-horizon failure, 2026:

| Paper | What it adds on top of frozen LeWM | Learns | Stitching / cross-episode? |
|---|---|---|---|
| SAGE (2607.17973) | Learned subgoal generator + subgoal-conditioned action prior; CEM refines | generator + prior on 400k expert windows | No (aligned same-trajectory windows). Push-T H=150: 12.7 -> 64.7; Cube 26.7 -> 67.3 |
| Hi-LeWM / "Mind the Gap" (2607.12547) | Macro-action encoder + high-level predictor; CEM over macro-actions | macro-action encoder, HL predictor | No. Naive version loses to flat; empirical-macro CEM: d=50 64.0, d=75 32.7 |
| HWM (2604.03208) | Multi-scale world models in shared latent; long-horizon predictions are subgoals | extra world models + action encoder | No |
| FF-JEPA (2606.09311) | Action-free latent planner predicting next subgoal | planner network (diffusion variant) | No. Push-T t=75: 91.8 |
| TRM (2605.22164) | Learned pairwise temporal cost replaces terminal L2 | small pairwise cost head | Two-Room 7 -> 96.7 (LeWM); Push-T "exposes limits" |
| TD-JEPA (2607.25337) | Directed temporal cost trained into the encoder | encoder retrained | No |
| RC-aux (2605.07278) | Budget-conditioned reachability supervision into the encoder; reachability-aware planner | encoder continued/retrained | No |
| VLWM (2606.21775) | Variable-length predictor | predictor retrained | No |
| DA-LeWM (2608.18746), SCALE (2608.16287) | Representation-side geometry fixes | encoder | No |
| GC-IDM (2605.08732) | Amortise CEM into a goal-conditioned IDM | IDM | No |
| RP1 (2608.18669) | Learned critic + learned plan optimiser | both | No |

Graph-search lineage (offline data as a graph, Dijkstra for subgoals):
SPTM (ICLR 2018), SoRB (NeurIPS 2019), SGM (NeurIPS 2020), LEAP, GAS (ICML 2025),
TTGS (2510.07257, test-time graph on OGBench with frozen GCRL policies), ALPS (ICML 2026,
Laplacian latent + k-means + Dijkstra + CEM with behaviour prior on OGBench).
All of these use a learned goal-conditioned *policy* (or a state-space model with a BC
prior) as the low-level controller; none put the graph inside a pixel world model's MPC.

Critic-in-MPC lineage: POLO (terminal value in MPC), TD-MPC / TD-MPC2, Dreamer/Director,
goal-conditioned terminal value estimation for MPC. All train the value with the model or
with rewards; none use a budget-clocked hitting-time term.

### 2.2 What is actually new here

1. **Dataset-as-graph inside a frozen pixel world model's MPC.** The graph is over the
   offline dataset in a TDR space learned on frozen latents; nothing in the world model or
   planner is retrained; there is no learned subgoal generator, policy or macro-action space.
   GAS/TTGS/ALPS use RL policies at the low level; SAGE/Hi-LeWM/HWM/FF-JEPA learn
   generators or extra predictors. This is the "no new generative model, data-supported by
   construction" corner of the design space, and Hi-LeWM's own diagnosis ("subgoals must
   remain data-supported, executable, and matched to the temporal scale") is exactly what a
   dataset graph gives for free.
2. **Cross-episode goal reaching as the evaluation target.** Every LeWM-planning paper
   evaluates goals drawn from the same trajectory at some offset. We evaluate arbitrary
   (start, goal) pairs from different episodes, which is the stitching problem GAS was built
   for, and show latent L2 sits at ~10% there regardless of budget.
3. **A budget-clocked expected-hitting-time cost.** V(z, g, h) with h = the budget left
   after the plan being scored; cost = capped sum over h of (1 - V). It removes the beta
   dependence that a -log V term has and keeps a progress gradient among feasible endpoints.
   RC-aux has budget-conditioned reachability, but trained into the encoder; TRM/TD-JEPA
   have temporal costs but no budget clock.
4. **The mechanistic decomposition** (this is what a reviewer will remember): the graph
   subgoal fixes *direction* (planner keeps converging; final error 76 -> 19 px at offset
   50), the critic fixes *feasibility* (stops the block drifting; final error 163 -> 26 px
   on cross-episode), the final-phase switch fixes *resolution* (medoid radius ~ success
   tolerance). Neither the subgoal nor the critic alone moves cross-episode; together they
   quadruple it. Plus the executor finding: the graph knows a 45-120-step route for every
   cross-episode pair, CEM cannot execute it without the feasibility term.

### 2.3 Claims to make and claims to avoid

Make: "no retraining of the world model", "no learned subgoal generator", "one configuration
across protocols and environments" (beta = 1 everywhere; only H_TD / lookahead are
recalibrated from the TDR's own step-gap table), "first cross-episode numbers for latent MPC
on Push-T / Reacher / Cube" (hedge: "to our knowledge").

Avoid: "state of the art on Push-T long horizon" (SAGE's same-episode numbers are higher at
H=150 and it uses the same frozen model); "pixels only" (the critic labels are not);
"the graph is what matters" (the RL version of GAS on this latent found the opposite; here
the graph is one of three necessary pieces).

### 2.4 Suggested framing (one sentence)

*Long-horizon and cross-episode goal reaching with a frozen pixel world model can be
recovered without retraining it or learning a generator: use the offline dataset itself as a
graph to supply data-supported subgoals, and a budget-aware hitting-time critic to keep the
planner on feasible routes.*

Title candidates:
- "Planning on the Dataset Graph: Subgoals and Hitting-Time Critics for Long-Horizon Latent MPC"
- "Graph-Assisted Latent MPC: Stitching Offline Data Inside a Frozen World Model's Planner"
- "Direction and Feasibility: What a Frozen World Model's Planner Needs for Long Horizons"

---

## 3. Comparisons and analyses

### 3.1 External baselines (in priority order, with cost)

| Priority | Baseline | Why | Cost |
|---|---|---|---|
| P0 | L2 with more compute (horizon 10 blocks, 600 samples, 60 iters, receding 1) | "Did you just give L2 too little search?" | eval only, ~1 h/protocol on a 1080 Ti |
| P0 | Random policy / zero-action floor | success by luck on near pairs (block-near pairs succeed at 39%) | trivial |
| P0 | TDR-only terminal cost, graph cost-to-go (`tdr`, `ctg`) on the fixed tasks, 5 seeds | separates metric from graph from critic; currently seed-0 only | eval only |
| P1 | TRM-style learned pairwise cost as terminal ranking (no graph, no critic) | closest "replace the cost" paper; cheap to reimplement (small MLP on frozen latents, same-trajectory pairs) | 1 day |
| P1 | SAGE (official code if released; else generator-only ablation) | strongest same-episode competitor; its generator cannot stitch by construction, which is our argument | 1-2 days if code exists |
| P1 | Oracle subgoal (true future latent from the same episode at +25 steps) | upper bound on subgoal quality (Hi-LeWM does this: 73.3% at d=50) | eval only, same-episode only |
| P2 | GCRL low-level policy with the same graph (GAS proper, HIQL) | shows CEM is the stronger executor; historical numbers exist (GAS-direct 50.7 at offset 25 on 1000 eps) but on a smaller tier | rerun on full data: 1 day |
| P2 | Hi-LeWM empirical-macro CEM | second hierarchical competitor | only if code exists |
| P2 | DINO-WM or PLDM as the frozen model with our graph + critic | "is it LeWM-specific?" | 2-3 days |

### 3.2 Internal ablation ladder (Push-T exists; replicate the bold rows on Reacher and Cube)

1. L2
2. + TDR terminal cost (no graph)
3. + graph cost-to-go (no subgoal commitment)
4. **+ graph subgoal one lookahead ahead** (subgoal_tdr)
5. **+ final-phase switch at one lookahead, z-space L2 in the final phase** (A)
6. + critic as -log V, standardised, beta in {0.5, 1, 2, 4} (old rows; shows beta dependence)
7. **+ critic as expected hitting time, beta = 1** (B) -- headline
8. ET at beta = 2 (E); ET in calibrated step units without standardisation (C); hard
   feasibility filter tau = 0.5 (D); `critic_final=false`
9. Critic without graph (L2 + critic): shows the composition is needed
10. Retrieval warm-start; replan every 5/10 steps (rh1/rh2): what did not work
11. Graph knobs: TE in {0.9, 0.99}, H_TD in {8, 12, 16}, lookahead in {8, 13.7, 20}
12. Critic training ablations: no Bellman, no cross-episode negatives, graph hitting-time
    labels instead of privileged labels (pixel-only critic)
13. Learned-asset seed vs CEM seed: retrain TDR/graph/critic on 3 seeds, report both variances

### 3.3 Mechanistic analyses (each is one figure or one small table)

| Analysis | Question it answers | Status |
|---|---|---|
| Success vs graph geodesic D_g bins (per-pair diag) | Does the objective know a route, and who executes it? | exists for cross s0; extend to all seeds and protocols |
| Success vs initial block/joint error | Are wins just the near pairs? | exists |
| Median final error and closest approach, per method | direction vs drift | exists (32 -> 76 px L2; 19 -> 19 px subgoal) |
| First-hit-step distributions | progress gradient; beta slows the planner | exists for beta sweep |
| Predicted vs realized terminal cost (single pass) | model-based planning gap; 88.5% optimistic | exists for L2; add for B |
| Oracle Spearman of L2 / TDR / critic vs privileged recovery time per offset | why L2 ranking goes flat past 50 steps; where the critic is worse (offset 25) | exists (0.72 / 0.36 L2; 0.60 / 0.70 critic) |
| Fraction of CEM population the critic marks feasible, per protocol | why -log V saturates at short range (67% pass at 25; 39-41% at 100/cross) | exists for D |
| Frac_final / frac_fallback / frac_unreachable of Alg.-1 per protocol | silent failure modes of the graph | script exists (`gas_mpc_te_diag.py`) |
| TDR calibration table (median distance per step gap) and graph stats per env | one TDR unit in env steps; node counts (Push-T 20.5k, Reacher ~535, Cube ?) | exists |
| Route visualisation: 2-3 cross-episode pairs, the graph path medoid frames, the executed frames | the qualitative figure | make |
| Held-out-task check | leakage | run |
| Compute: prepare time (encode / TDR / graph / critic) and per-decision planning time vs L2 | "is it practical?" | measure; graph Dijkstra is once per goal |

### 3.4 Statistics

- Report mean +- sd over CEM seeds and the pooled count (hits/250). Use paired comparisons
  (same tasks, same seeds): a sign test or Wilcoxon over per-task success across seeds, and
  the "solved in >= 1 seed / in every seed" columns already in the report.
- For the headline rows, move from 50 to the full 200 tasks (5 seeds x 200 = 1000 pairs per
  cell); one 50-task protocol run is 14-16 min on a 1080 Ti, so 200 tasks is ~1 h per cell.
- Separate learned-asset seed from CEM seed in the appendix table.

---

## 4. Seven-day schedule (if going for 25 Sep)

### 4.1 P0 (must be done by the go/no-go on 22 Sep)
- [ ] Register abstract on OpenReview today; confirm one author qualifies as a reciprocal
      reviewer and is registered for 3+ reviews (else the submission is desk-rejected).
- [ ] Cube: finish the 7 remaining cross-episode runs; resolve the L2 baseline gap
      (sample tasks from episode starts as LeWM/SAGE do, check CEM iters, predicate).
- [ ] Held-out-task check on Push-T (headline B and L2, 3 seeds).
- [ ] `tdr` and `ctg` on the fixed tasks, 5 seeds (fills ladder rows 2-3).
- [ ] L2 with more compute (ladder row 1').
- [ ] Reacher 10-seed same25 number and `critic_final=false` row into the appendix.

### 4.2 P1 (25 Sep if time; otherwise rebuttal / next venue)
- [ ] TRM-style learned cost baseline.
- [ ] Oracle-subgoal upper bound.
- [ ] Two-Room end to end (`GAS_MPC_ENV=tworoom`; dataset is small and already on /home2).
- [ ] Route figure; compute table; 200-task headline rows.

### 4.3 Writing
- Day 1-2: method + experiments sections from `main.tex` here; tables auto-generated by
  `gas_mpc_report.py` (add a `--paper` mode that emits the 4-column mean +- sd rows).
- Day 3-4: related work (section 2.1 above), intro, figures.
- Day 5: limitations, reproducibility statement, AI-use statement, appendix tables.
- Day 6: full read-through against the checklist in section 5; anonymisation pass.
- Day 7: buffer.

---

## 5. Conference checklist (ICLR 2027 specifics)

Policies (verify on the guidelines page before submitting):
- 9 pages main text at submission (10 at rebuttal / camera-ready); references, appendix,
  reproducibility statement, ethics statement and the AI-use statement do not count.
- **Mandatory AI-use statement** in the paper (does not count toward the limit). It must say
  which of the "required disclosure" tasks LLMs were used for (data generation, theory /
  conceptual framework, mathematical claims) and is recommended for code, figures,
  literature analysis, editing. This project used an LLM assistant for code, analysis
  scripts, literature search and drafting; disclose that plainly.
- Reciprocal reviewing: at least one author must be registered to review 3+ papers and have
  a prior publication at a listed venue; authors on 3+ submissions review 6. A new-author
  team may submit at most one paper with no qualified reviewer.
- No author changes after the abstract deadline; author order may change until the 25th.
- Double blind; arXiv posting allowed; papers are de-anonymised after the review period
  regardless of outcome (new in 2027). Cite our own prior write-ups in third person.
- Reproducibility statement (recommended, no page cost): point to the code, the fixed task
  files, the seeds, and the asset checkpoints.
- Style: `iclr2027_conference.sty` from the official zip; `main.tex` here loads it if
  present and falls back to `article` so it compiles today.

Writing:
- Lead with the cross-episode result and the two-mechanism story; the same-episode gains
  are supporting evidence, not the headline (SAGE beats us there).
- One figure on page 1 or 2: the ladder as a bar chart (L2 -> subgoal -> switch -> critic)
  across the four protocols, or the per-pair geodesic-bin plot. Reviewers read figure 1 and
  table 1 first.
- Every number in the text must be in a table or figure; every table must say tasks x seeds.
- State the privileged-label dependency of the critic in the method section.
- Define "budget" and "offset" once, with the 2x-offset rule and the cross-episode budgets
  (Push-T 250, Reacher 200, Cube ?).
- Keep the executor finding: it is the most quotable sentence ("the graph knows a route,
  CEM cannot execute it without a feasibility term").
- Put the render-version bug (MuJoCo 3.12 vs 3.5 floor texture, 65% -> 87%) in the
  appendix as a reproducibility note; it will save other people weeks.
- Limitations: privileged critic labels; single frozen model family (LeWM); graph built on
  the eval dataset's episodes; expert-only data (no suboptimal-data tier study); Reacher
  short-range loss; Cube partial.
