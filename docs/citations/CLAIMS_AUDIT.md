# Cited-claims audit — ICLR 2027 draft (`docs/iclr2027/main.tex`)

Audited 2026-09-24. Every sentence in `main.tex` that cites a reference, or describes a cited
paper by name, was split into single factual claims. Each claim was checked against the full
text of the cited PDF in [`pdfs/`](pdfs/). Line numbers (L…) refer to `main.tex`.

**Status key**

| Status | Meaning |
|---|---|
| **SUPPORTED** | The source says this. Evidence is quoted or paraphrased. |
| **PARTIAL** | Broadly right, but imprecise, over-general, or true for only some of the cited works. Fix the wording. |
| **INCORRECT** | The source contradicts the claim. Must fix. |
| **NOT IN SOURCE** | The fact is true for our setup, but it comes from our code or data, not from the cited paper. Change the attribution. |

**Totals: 74 claims. 50 SUPPORTED, 13 PARTIAL, 6 INCORRECT, 5 NOT IN SOURCE.**

---

## A. Must fix (INCORRECT)

| # | Where | Claim | What the source says | Suggested fix |
|---|---|---|---|---|
| A1 | L86, L159 | PLDM plans by "select[ing] the sequence whose predicted **terminal** latent lies closest to the goal"; PLDM plans "with zero-shot **CEM** … scoring candidates by **terminal** latent distance" | PLDM uses **MPPI**: "We use MPPI [72] in all our experiments with planning". Its cost (Eq. 5) is a discounted **sum over every step t = 0..H** of the goal distance, **plus an uncertainty penalty** (Eq. 6). It is neither CEM nor terminal-only. | Drop PLDM from both lists, or say "plan by sampling-based MPC (CEM or MPPI) toward a goal latent". |
| A2 | L103 | TRM is an example of "a **retrained encoder** whose geometry directly encodes temporal cost" | TRM: "TRM **freezes the trained encoder and predictor**, then learns a continuous pairwise ranking signal". "We study this failure in a **fixed encoder**". | Move TRM out of the retrained-encoder group, e.g. "…a learned pairwise temporal cost on the frozen model \citep{trm}, or a retrained encoder… \citep{tdjepa,rcaux}". L165 already describes it correctly. |
| A3 | L191 | "TTGS and **ALPS** do the same **at test time for frozen OGBench policies**" | ALPS is not a test-time wrapper around frozen policies. In **pre-training** it learns a Laplacian representation, a forward model and a behaviour prior, then clusters the dataset. At decision time it runs Dijkstra over the cluster graph and reaches each subgoal with **CEM on its learned forward model**. (TTGS does match: "guides a frozen policy at test time".) | "TTGS wraps frozen OGBench policies with test-time graph search \citep{ttgs}; ALPS runs Dijkstra over a Laplacian cluster graph and reaches subgoals with CEM on a learned model \citep{alps}." |
| A4 | L108 (novelty) | "To our knowledge, **no prior method places the dataset graph inside a pixel world model's planner itself**." | **ALPS nearly does this.** It builds a cluster graph over the offline dataset, runs Dijkstra to pick subgoals, and executes them with CEM on a learned forward model. It also reports **pixel-based** OGBench variants (visual-antmaze, 64×64×3 images). The differences: ALPS trains its own forward model, representation and behaviour prior, while we use a **frozen, pretrained** JEPA world model and its unchanged planner and cost interface. | Narrow the claim: "…inside the planner of a frozen, pretrained pixel world model". Discuss ALPS as the closest prior work in Related Work. A reviewer who knows ALPS will raise this first. |
| A5 | L161 | `stableworldmodel` "stud[ies] which design choices matter" | stable-worldmodel is a **software platform and benchmark suite**: "an open-source platform for standardized and reproducible world modeling research and evaluation". It does not study design choices. | Cite it as the evaluation platform: "\citet{jepawm-study} study which design choices matter; stable-worldmodel \citep{stableworldmodel} standardises their evaluation". |
| A6 | L364 (with L226) | "The encoder, predictor, **CEM hyperparameters** … are **LeWM's, unchanged**" (L226: "300 candidates, **30 iterations**, 30 elites") | LeWM App. D: "CEM samples 300 candidate action sequences and optimizes them for a maximum of **30 iterations in PushT and 10 iterations in the other environments**". Our `config/eval/solver/cem.yaml` uses `n_steps: 30` for every environment, and `cube.yaml`/`reacher.yaml` don't override it. So Cube and Reacher run 30 iterations, not LeWM's published 10. | Either say "LeWM's released evaluation config (30 iterations in every environment; the LeWM paper reports 10 outside Push-T)", or rerun Cube and Reacher at 10. Using 30 favours the L2 baseline, so this is a disclosure issue, not a fairness one. |

## B. Should fix (PARTIAL)

| # | Where | Claim | Issue | Suggested fix |
|---|---|---|---|---|
| B1 | L86 | DINO-WM is a "joint-embedding world model **trained from pixels**" | DINO-WM trains only the predictor, on **frozen, pretrained DINOv2** patch features. LeWM itself contrasts DINO-WM's "Frozen Pre-trained Encoder" with its own end-to-end training. | "…world models that plan in a learned or pretrained latent from pixels". |
| B2 | L100 | This work "targets the same failure **in the same frozen-model setting**" | True for SAGE, Hi-LeWM and FF-JEPA (all frozen LeWM), and for TRM (frozen encoder). **Not** for TD-JEPA and RC-aux, which retrain or continue training the encoder, as the same sentence says. HWM builds on DINO-WM, V-JEPA 2-AC and PLDM, not LeWM. | "…targets the same failure, mostly on a frozen LeWM-style model…". |
| B3 | L167 | DA-LeWM and SCALE "push **similar signals** [i.e. temporal cost] into the encoder at training time" | Both are training-time encoder changes, but the signal is not temporal. **DA-LeWM** adds "inverse-dynamics and demonstration-conditioned goal-action heads". **SCALE** correlates latent distances "with distances in a standardized task-relevant **state** space". | "…push planning-relevant structure (temporal cost, reachability, action, or state distance) into the encoder at training time". |
| B4 | L161 | "the planning horizon is **a known weak point of all of them**", supported by `jepawm-study` + `stableworldmodel` | `jepawm-study` gives only indirect support: "The already existing pick-and-place tasks require too long horizons to be solved by our current planning procedure". `stableworldmodel` only says planning "depend[s] on" accurate long-horizon prediction. Neither shows it for *all* of them. | Soften to "and long horizons remain a documented difficulty \citep{jepawm-study}". Better, back it with Hi-LeWM or SAGE, whose LeWM baselines fall from 94% at H=25 to 12.7% at H=150. |
| B5 | L106, L188 | SPTM builds "a graph over the offline dataset" / "over **replay buffers**" in the "offline goal-conditioned RL" line | SPTM is navigation. Its graph is built from an exploration walk-through ("the agent builds the graph by appending observations … adding shortcut connections based on detected visual similarities"). It has no replay buffer and isn't offline GCRL. | Acceptable as lineage; say "over stored experience" instead of "replay buffers". |
| B6 | L106 | The line "hand[s] the resulting subgoal to a **learned goal-conditioned policy**" (includes ALPS) | ALPS reaches subgoals with CEM on a learned forward model plus a behaviour prior, not with a GC policy. L192's parenthetical already says this correctly. | Add "(or a model-based controller)". |
| B7 | L197 | Director "put[s] a learned value at the end of a short model rollout" (in "Value functions inside **MPC**") | Director trains manager and worker **policies** by actor-critic in imagination, and values bootstrap imagined rollouts during training. It does no MPC at decision time. | Move Director to the hierarchical paragraph, or say "…or bootstrap imagined rollouts with a learned value \citep{director}". |
| B8 | L190 | GAS clusters "a temporal distance representation \citep{hilp,**qrl**}" | GAS's TDR comes from HILP ("proposed a method to learn a Temporal Distance Representation … (Park et al., 2024c)"). QRL appears in GAS only as a baseline. | Fine as a pointer to the concept. For precision, "\citep{hilp}; cf. quasimetric RL \citep{qrl}". |
| B9 | L181 | HWM, SAGE, FF-JEPA and Hi-LeWM each "train a **generative** component whose training pairs are same-trajectory windows" | Same-trajectory training is **supported**: SAGE uses "aligned windows from expert trajectories"; Hi-LeWM's macro-actions are "encoded from training trajectories"; HWM builds macro-actions "from primitive action chunks"; FF-JEPA predicts next subgoals along trajectories. But HWM's high-level model and Hi-LeWM's predictor are deterministic predictors, not generative models. The follow-on "cannot lie on a route that no single trajectory took" is our inference, not something they state. | "…trains a learned subgoal model on same-trajectory windows, so its subgoals are not built to lie on…". |
| B10 | L224 | LeWM maps a frame "to a 192-d **CLS** latent" | LeWM: "The observation embedding z_t is constructed from the [CLS] token embedding of the last layer, **followed by a projection step** … a 1-layer MLP with Batch Normalization". | "…the projected 192-d CLS embedding". |
| B11 | L224 | The predictor "maps **three past latents and three action blocks**" | Supported for Push-T and Cube ("history length is set to 3 for the PushT and OGBench-Cube environments, and to 1 for TwoRoom"). Reacher's history length isn't stated in the LeWM paper. | Say it holds for the released checkpoints, or check Reacher's `config.json`. |
| B12 | L258 | "HILP/GAS's target-stabilisation trick: computing q over the **minimum of two independent networks**" | Neither paper's text describes a two-network minimum. Both describe a single target network (HILP Eq. 5 "V̄ denotes the target network"; GAS Eq. 5). The twin-network minimum is in their released code (and OGBench's HILP implementation). | "…following HILP's and GAS's released implementations". |
| B13 | L364 | "graph-construction rules are … **GAS's, unchanged**" | The clustering, TE filter, edges and Dijkstra match GAS (see C-rows below). The **goal-attach rule**, linking to every node within max(H_TD, 1.2·d_min), is not in the GAS paper. GAS's Algorithm 1 selects "reachable nodes within H_TD". | Say the goal-attach radius is ours. |

## C. Attribution fixes (NOT IN SOURCE)

| # | Where | Claim | What the cited source actually says | Suggested fix |
|---|---|---|---|---|
| C1 | L387 | "Push-T \citep{lewm}: **18,685** expert episodes" | LeWM: "We follow the same setup and dataset as Zhou et al. [18], which contains **20,000** expert episodes with an average length of 196 steps". DINO-WM says **18,500**. 18,685 is the episode count in our released training file. The three numbers disagree. | "18,685 episodes in the released Push-T training set \citep{dinowm,lewm}". Don't imply the count comes from LeWM. |
| C2 | L387 | Push-T success "position errors below **20 px** and block angle error below **π/9**" | Not stated in LeWM or DINO-WM. It comes from the environment code. | Name the code, as L391 already does for Cube. |
| C3 | L391 | "OGBench Cube \citep{ogbench}: **10,000 episodes of 201 frames**" | OGBench's cube datasets are the OGBench `play`/`noisy` datasets. The 10,000 × 200-step dataset is **LeWM's**: "We collect 10,000 episodes, each consisting of 200 steps … using the data-collection heuristic provided in the benchmark library". | "OGBench Cube \citep{ogbench}, with LeWM's 10,000-episode dataset \citep{lewm}". |
| C4 | L391 | Cube success "within **0.04 m**" | Not in the OGBench paper. The paper already names `CubeEnv._compute_successes`. | OK as written. |
| C5 | L248–254 | γ = 0.99, expectile τ = 0.99, 32-d TDR "(HILP/GAS)" | The objective's **form** is HILP/GAS (supported, D-rows). The **values** are ours or task-specific: HILP tunes its Hilbert-representation discount and expectile per environment, and GAS tunes its TDR expectile per task (its Fig. 7 ablation). The 32-d output **does** match GAS ("we reduce the output dimension to 32 for the TDR"). | Fine as written, since the citation attaches to the method. Optionally add "(our values)". |

## D. Verified claims (SUPPORTED)

| # | Where | Key | Claim | Evidence in source |
|---|---|---|---|---|
| D1 | L86 | lewm | Joint-embedding WM trained from pixels | End-to-end JEPA from pixels, "Pixels Based" (Fig. 2) |
| D2 | L86, L227 | lewm | Planning cost is terminal squared L2 to goal latent | "C(ẑ_H) = ‖ẑ_H − z_g‖²₂ … terminal latent goal-matching objective" (Eq. 4) |
| D3 | L93, L160 | lewm | Published planner is CEM | "which we solve using the Cross-Entropy Method (CEM)" |
| D4 | L86 | dinowm | CEM with terminal latent distance | "We utilize the cross-entropy method (CEM)"; "MSE between the predicted latent state at the final timestep T and the goal latent state" |
| D5 | L159 | dinowm | Zero-shot planning | Title; "achieves zero-shot behavioral solutions at test time" |
| D6 | L159 | vjepa2 | V-JEPA 2-AC plans with CEM on a terminal latent distance, zero-shot | Energy "‖P(â₁:T; s_k, z_k) − z_g‖₁" optimized "using the cross-entropy method"; "deploy V-JEPA 2-AC zero-shot" (note: L1 distance, not L2; our L159 says only "distance") |
| D7 | L86 | pldm | Trained from pixels, joint-embedding | "Observations are 64 × 64 pixels images"; "JEPA architecture" (the planning claims are A1) |
| D8 | L158 | planet | Plans in a reconstruction-based latent | RSSM with decoder; plans with CEM, "we do not use a policy or value network" |
| D9 | L158 | dreamerv3 | Learns policies in a reconstruction-based latent | "reconstruct the inputs to ensure informative representations"; actor-critic learning (Fig. 3) |
| D10 | L161 | jepawm-study | Studies which design choices matter | "Summary of all candidate design choices studied in this work" (Table 1) |
| D11 | L101, L177 | sage | Learns a subgoal generator and a subgoal-conditioned action prior that CEM refines | "multi-horizon latent subgoal generator"; "subgoal-conditioned action generator … uses a frozen latent world model to evaluate and refine these proposals"; "CEM refinement procedure" |
| D12 | L100, L184 | sage | Same frozen model | "Using the same frozen LeWM, candidate budget, and CEM refinement procedure as the baselines" |
| D13 | L184 | sage | 64.7% at H=150 on Push-T | "When target offset H=150, success rises from 12.7% to 64.7% on PushT" |
| D14 | L183, L688 | sage | Stronger same-episode numbers than ours at long offsets | Same 2H budget ("PushT uses an environment budget of 2H"). SAGE H=50: 81.3% vs our same50. SAGE H=150: 64.7% (we don't evaluate H=150). |
| D15 | L689 | sage | Trained on 400k expert windows | "Both components in SAGE use aligned windows from expert trajectories … we sample 400k valid training windows" |
| D16 | L101, L179 | hilewm | Learns macro-actions and a high-level predictor, frozen LeWM | "freezes the pretrained low-level LeWM and adds a high-level predictor"; "latent macro-action encoder … and a high-level predictor" |
| D17 | L108 | hilewm | Quote "data-supported, executable, and matched to the temporal scale" | Verbatim: "subgoals must remain data-supported, executable, and matched to the temporal scale at which they are used" |
| D18 | L108 | hilewm | Identifies conditions for a subgoal interface to transfer to a compact frozen WM | "they identify the conditions under which this hierarchical interface transfers to compact frozen LeWM" |
| D19 | L179 | hilewm | Naive version loses to flat planning | "The straightforward Hi-LeWM planner often underperforms flat LeWM" |
| D20 | L179 | hilewm | …because CEM exploits the learned subgoal space | "CEM can exploit the terminal latent objective by selecting macro-actions that look good at the final step but do not yield useful intermediate subgoals" |
| D21 | L102, L176 | hwm | World models at several time scales | "HWM learns world models at multiple temporal scales within a shared latent space" |
| D22 | L176 | hwm | Long-horizon predictions used as subgoals | "predictions from the long-horizon model serve as subgoals for the short-horizon model via latent matching" |
| D23 | L102, L178 | ffjepa | Action-free latent planner | "we introduce an action-free latent planner that predicts the next subgoal given the current state" |
| D24 | L100 | ffjepa | Frozen LeWM setting | "We build our method on top of the LeWM JEPA world model"; "trained on the latent space defined by the world model's frozen encoder" |
| D25 | L103, L166 | tdjepa | Retrained encoder encoding temporal cost | "retains the LeWM encoder-predictor backbone and mines a directed temporal cost from reward-free trajectories" (trained, not frozen) |
| D26 | L103, L167 | rcaux | Training-time encoder fix | "RC-aux keeps the world-model backbone unchanged and adds planning-aligned supervision"; "continuation-training and matched-from-scratch settings" |
| D27 | L171 | rcaux | Budget-conditioned reachability | "budget-conditioned reachability supervision, together with temporal hard negatives" |
| D28 | L165 | trm | Learns a pairwise temporal cost from logged trajectories | "a small temporal pairwise cost trained from logged trajectories" |
| D29 | L165 | trm | Used as the terminal ranking | "uses it as a terminal cost on predicted terminal latent states against goal latent states" |
| D30 | L168 | valueguided | Shapes geometry toward a value | "shaping their representation space so that the negative goal-conditioned value function … is approximated by a distance" |
| D31 | L168 | straightening | Shapes geometry toward a straight path | "a curvature regularizer that encourages locally straightened latent trajectories" |
| D32 | L188 | sorb | Graph over replay buffer, learned distance, waypoints for GC policy | "a goal-conditioned value function provides edge weights, and nodes are taken to be previously seen observations in a replay buffer"; "generate this sequence of subgoals" |
| D33 | L188 | sgm | Graph over stored states, learned distance, planning for GC policy | "stores states and feasible transitions in a sparse memory"; "two-way consistency" aggregation for goal-conditioned RL |
| D34 | L188 | sptm | Graph plus a learned network for retrieval and navigation | "a (non-parametric) graph … and a (parametric) deep network capable of retrieving nodes" (see B5 for the replay-buffer caveat) |
| D35 | L189 | gas | Clusters a TDR into nodes | "clustered in the TDR space at intervals of the target temporal distance H_TD … centroids of each cluster are treated as graph nodes" |
| D36 | L189, L266–272 | gas | Temporal-efficiency filter, as in our Eq. (TE) | Eq. 7–9: s_opt = F(s_cur, H_TD), s_reached = s_{cur+H_TD}, TE = cos(ψ(s_opt)−ψ(s_cur), ψ(s_reached)−ψ(s_cur)). "e.g., TE ≥ 0.9" matches our θ_TE = 0.9. |
| D37 | L273–275 | gas | Sequential clustering: assign to nearest centre within H_TD/2, else open a new one, then reset centres to means | Algorithm 2 (App. A): "if ‖h_i − h_c‖ > H_TD/2 then Create a new cluster"; "cluster centers are updated to the mean values". (GAS's **main text** says "within H_TD"; its Algorithm 2 and our code use H_TD/2.) |
| D38 | L276 | gas | Edges between nodes within H_TD | "edges are added between nodes within H_TD" |
| D39 | L278 | gas | One Dijkstra pass gives cost-to-go from every node | "Dijkstra's algorithm is used to precompute the shortest distances from all graph nodes to the final goal" |
| D40 | L189 | gas | Searches the graph for subgoals | "utilizes a shortest-path algorithm to sequentially select subgoals" |
| D41 | L116, L248 | hilp, gas | TDR via expectile TD with V = −‖ψ(s)−ψ(g)‖ and a target network | HILP Eq. 5 and final objective; GAS Eq. 4–6 ("V(s,g) = −‖ψ(s)−ψ(g)‖₂ … expectile loss") |
| D42 | L250–252 | hilp, gas | Goal from own trajectory at a geometric offset w.p. 0.625, else uniform | HILP: "geometric distribution over the future states within the same trajectory (with probability 0.625), or uniformly from the dataset"; GAS App.: same, "inherited from the original hyperparameter settings" |
| D43 | L191 | ttgs | Test-time graph search for frozen OGBench policies | "A shortest-path search with Dijkstra's algorithm yields a sequence of subgoals that guides a frozen policy at test time"; "no additional supervision or parameter updates"; evaluated on OGBench |
| D44 | L192–193 | alps | Executes subgoals with a state-space model and a behaviour prior | "a one-step model in the original state space"; "includes a behavior prior to bias CEM optimization" |
| D45 | L192 | ogbench | OGBench is the benchmark TTGS/ALPS evaluate on | Both papers evaluate on OGBench tasks |
| D46 | L197 | polo | Learned value at the end of a short model rollout | "approximate value functions can help reduce the planning horizon"; terminal value in MPC |
| D47 | L197 | tdmpc | Same | "local trajectory optimization over a short horizon, and use a learned terminal value function to estimate long-term return" |
| D48 | L197 | tdmpc2 | Same | "bootstrapping return estimates beyond horizon H with a learned terminal value function" |
| D49 | L199 | gcterminal | Goal-conditioned terminal values for multi-task MPC | "an MPC framework with goal-conditioned terminal value learning to achieve multitask policy optimization" |
| D50 | L199–200 | polo…gcterminal | "Those values are trained with rewards or with the model" | TD-MPC(2) learn values by TD on rewards; POLO and gcterminal use reward-based value learning |

---

## E. Other `refs.bib` problems found during this pass

- **`scale` — author list was incomplete. Fixed in `refs.bib`.** arXiv's metadata lists 3 authors, but the v2 PDF lists 11: Jiaming Hu, Yan Zheng, Tian Wang, Florian Dubost, Alejandro Mottini, Junze Liu, Arvind Srinivasan, Kai Zhong, Kun Qian, Sharon Gao, Qingjun Cui.
- **`dalewm` — author order differs.** The PDF gives "Jiawei Wang, **Yushen Zuo, Ke Rui**, Yichun Feng, Minglei Li". arXiv metadata and our bib have "Wang, **Rui, Zuo**, …". The PDF is normally authoritative for author order. Not changed; confirm on the arXiv abstract page.

## G. Resolution (2026-09-24)

The paper was edited to address every item below. Line numbers above refer to the pre-edit text.

| Item | Resolution |
|---|---|
| A3, A4, B6 | Fixed by the author. ALPS is now described accurately and named as the closest prior work, and the novelty claim is narrowed to "a frozen, pretrained pixel world model … through its planning cost alone". |
| A1 | PLDM is now described as MPPI with a summed, uncertainty-penalised cost. The Intro keeps only LeWM and DINO-WM as terminal-L2 examples. |
| A2, B2 | TRM is now "a learned pairwise temporal cost"; only TD-JEPA and RC-aux are "retrained encoder". "Same frozen-model setting" became "mostly on a frozen LeWM-style model". |
| A5, B4 | stable-worldmodel is now cited as the evaluation platform. The horizon weakness is now backed by the jepawm-study pick-and-place remark and SAGE's LeWM baseline (12.7% at H=150). |
| A6 | The Method now says 30 iterations is LeWM's released evaluation config (`config/eval/solver/cem.yaml`, from LeWM's initial commit), used in every environment, and that the LeWM paper reports 10 outside Push-T. "What is frozen" now says "LeWM's released" hyperparameters. |
| B1 | The Intro now says "joint-embedding latent of pixel observations" instead of "trained from pixels". |
| B3 | "similar signals" is now "planning-relevant structure (temporal cost and reachability, inverse dynamics, or state distance)". |
| B5 | "replay buffers" is now "stored experience". |
| B7 | Director is now described as bootstrapping imagined rollouts to train hierarchical policies. |
| B8 | GAS clusters "HILP's temporal distance representation, a relative of quasimetric value learning". |
| B9 | "generative component" is now "subgoal model"; "cannot lie" is now "are not built to lie". |
| B10, B11 | The latent is now "(the projected CLS embedding)", and history 3 is "LeWM's default history" (`history_size: 3` in LeWM's train config). Reacher's checkpoint config was not checked locally. |
| B12 | Now attributed to GAS's released implementation. Verified in `M_utils/agents/gas.py`: `next_v = jnp.minimum(next_v1, next_v2)`. |
| B13 | The goal-attachment radius is now marked "(our rule …)", and "What is frozen" notes it is the one change to GAS's rules. |
| C1, C2 | Push-T now cites DINO-WM and LeWM for "the 18,685 expert episodes of LeWM's released training file". The success thresholds are attributed to "the environment's own success check". |
| C3 | Cube now cites LeWM for the 10,000-episode dataset. |
| C4, C5 | No change needed. |
| E (dalewm author order) | Fixed 2026-09-25. `refs.bib` now follows the PDF: Wang, Zuo, Rui, Feng, Li. |
| New (L408, GCIQL recipe) | Added after the audit. Its hyperparameters were cited to the LeWM paper, which does not list them. Fixed 2026-09-25: now "the recipe of their released code (stable-worldmodel's `gciql` configuration: …)". Table (B) of `tab:main` was checked against LeWM Fig. 6, and all 16 numbers match. |

## F. Suggested order of work

1. **A4** (ALPS and the novelty claim) and **A3** (ALPS description). Reviewers will check these first.
2. **A1** (PLDM is MPPI with a summed cost) and **A2** (TRM uses a frozen encoder). Both are simple factual errors.
3. **A6** (CEM iterations on Cube and Reacher). Decide whether to disclose or rerun.
4. **C1–C3** (dataset counts and sources in Setup).
5. The B-rows, as wording edits.
