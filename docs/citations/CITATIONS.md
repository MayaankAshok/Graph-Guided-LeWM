# Citation tracker — ICLR 2027 draft (`docs/iclr2027/main.tex`)

Built 2026-09-24 from `main.tex`, `refs.bib`, and `main.aux`. Line numbers (L…) refer to `main.tex`.
The paper cites **30** references. `stableworldmodel`, `qrl` and `director` were dropped from Related Work on 2026-09-24. Their bib entries and PDFs are kept, but their entries below are historical. All 33 PDFs are in [`pdfs/`](pdfs/), named `<bibkey>.pdf`.
`refs.bib` also has **6 uncited** entries (`hiql`, `leap`, `vlwm`, `gcidm`, `rp1`, `fastlewm`).
They were verified too but not downloaded.

---

## 1. `refs.bib` verification (all 39 entries)

**How each entry was checked**
- **Title and authors:** compared with the arXiv API record, using the latest arXiv version.
- **Venue:** taken from four sources:
  - the venue line printed in each PDF (e.g. "Proceedings of the 43rd ICML", "Published as a conference paper at ICLR 2025");
  - arXiv's comment and journal-ref fields;
  - Semantic Scholar's venue field;
  - Crossref, for DreamerV3's Nature version.
- DBLP blocked automated queries, so it was not used.

**What changed**
- Every entry now has a `url` field. ICLR's `.bst` prints it as "URL …" in the reference list.
- Every entry marked **Fixed** was corrected in `refs.bib` on 2026-09-24.
- The paper compiles cleanly: no BibTeX warnings and no undefined citations.

"Venue source" says where the venue came from: **PDF** = the paper's own header or footer, **arXiv** = arXiv's comment or journal-ref field, **S2** = Semantic Scholar.

| # | Key | Title / authors | Venue (now in bib) | Venue source | Status | Link |
|---|---|---|---|---|---|---|
| 1 | `lewm` | OK | arXiv 2026 (preprint) | PDF says "Preprint" | OK | [arXiv:2603.19312](https://arxiv.org/abs/2603.19312) |
| 2 | `gas` | OK | ICML 2025 | PDF: Proc. 42nd ICML; arXiv comment | OK (removed redundant `note`) | [arXiv:2506.07744](https://arxiv.org/abs/2506.07744) |
| 3 | `hilp` | OK | ICML 2024 | PDF: Proc. 41st ICML | OK | [arXiv:2402.15567](https://arxiv.org/abs/2402.15567) |
| 4 | `qrl` | OK | ICML 2023 | PDF: Proc. 40th ICML; arXiv journal-ref | OK | [arXiv:2304.01203](https://arxiv.org/abs/2304.01203) |
| 5 | `hiql` *(uncited)* | OK | NeurIPS 2023 | arXiv comment; S2 | OK | [arXiv:2307.11949](https://arxiv.org/abs/2307.11949) |
| 6 | `sptm` | OK | ICLR 2018 | PDF header | OK | [arXiv:1803.00653](https://arxiv.org/abs/1803.00653) |
| 7 | `sorb` | OK | NeurIPS 2019 | S2 (pp. 15220–15231) | OK | [arXiv:1906.05253](https://arxiv.org/abs/1906.05253) |
| 8 | `sgm` | OK | NeurIPS 2020 | PDF; arXiv comment | OK | [arXiv:2003.06417](https://arxiv.org/abs/2003.06417) |
| 9 | `ttgs` | OK | **ICML 2026** (bib said arXiv 2025) | PDF: Proc. 43rd ICML (v2, May 2026) | **Fixed: venue** | [arXiv:2510.07257](https://arxiv.org/abs/2510.07257) |
| 10 | `alps` | **Authors were a placeholder** → Shehmar, Schlegel, Taylor, Machado | ICML 2026 | PDF: Proc. 43rd ICML; arXiv journal-ref | **Fixed: authors, link** | [arXiv:2602.05031](https://arxiv.org/abs/2602.05031) |
| 11 | `sage` | OK | arXiv 2026 | PDF says "Preprint" | OK | [arXiv:2607.17973](https://arxiv.org/abs/2607.17973) |
| 12 | `hilewm` | **Surnames only** → full names (Niccolò Caselli, …, Sathya Kamesh Bhethanabhotla) | arXiv 2026, note: WM@Booth 2026 workshop | PDF: "Accepted to WM@Booth 2026" | **Fixed: authors, note** | [arXiv:2607.12547](https://arxiv.org/abs/2607.12547) |
| 13 | `hwm` | OK | arXiv 2026 | S2 | OK | [arXiv:2604.03208](https://arxiv.org/abs/2604.03208) |
| 14 | `ffjepa` | OK | arXiv 2026 | S2 | OK | [arXiv:2606.09311](https://arxiv.org/abs/2606.09311) |
| 15 | `trm` | **Title was v1**. Now v2: *World Model Control by Trajectory Reachability Metrics* | arXiv 2026 | arXiv | **Fixed: title** | [arXiv:2605.22164](https://arxiv.org/abs/2605.22164) |
| 16 | `tdjepa` | OK | arXiv 2026 | S2 | OK | [arXiv:2607.25337](https://arxiv.org/abs/2607.25337) |
| 17 | `rcaux` | OK | arXiv 2026 | PDF says "Preprint" | OK | [arXiv:2605.07278](https://arxiv.org/abs/2605.07278) |
| 18 | `vlwm` *(uncited)* | OK | arXiv 2026 | S2 | OK | [arXiv:2606.21775](https://arxiv.org/abs/2606.21775) |
| 19 | `gcidm` *(uncited)* | OK | arXiv 2026 | S2 | OK | [arXiv:2605.08732](https://arxiv.org/abs/2605.08732) |
| 20 | `rp1` *(uncited)* | OK | arXiv 2026 | arXiv comment: "Preprint" | OK | [arXiv:2608.18669](https://arxiv.org/abs/2608.18669) |
| 21 | `dalewm` | **Author order** now follows the PDF (Wang, Zuo, Rui, Feng, Li); arXiv metadata lists Rui before Zuo | arXiv 2026 | arXiv | **Fixed: author order** | [arXiv:2608.18746](https://arxiv.org/abs/2608.18746) |
| 22 | `scale` | **Author list incomplete**: arXiv metadata gives 3 authors, the v2 PDF lists 11 (Hu, Zheng, Wang, Dubost, Mottini, Liu, Srinivasan, Zhong, Qian, Gao, Cui) | arXiv 2026 | PDF says "Preprint" | **Fixed: authors** | [arXiv:2608.16287](https://arxiv.org/abs/2608.16287) |
| 23 | `straightening` | **Authors were a placeholder** → Wang, Bounou, Zhou, Balestriero, Rudner, LeCun, Ren | **ICML 2026** (bib said arXiv) | PDF: Proc. 43rd ICML; arXiv: "ICML2026 Camera Ready" | **Fixed: authors, venue** | [arXiv:2603.12231](https://arxiv.org/abs/2603.12231) |
| 24 | `valueguided` | OK | arXiv 2026, note: World Modeling Workshop 2026 | arXiv comment (poster, Mila) | OK | [arXiv:2601.00844](https://arxiv.org/abs/2601.00844) |
| 25 | `jepawm-study` | OK | **TMLR 2026** (bib said arXiv 2025) | PDF: "Published in TMLR (05/2026)" | **Fixed: venue, year** | [arXiv:2512.24497](https://arxiv.org/abs/2512.24497) |
| 26 | `stableworldmodel` | **Authors were a placeholder** → Maes, Le Lidec, Facury, … (12 authors) | arXiv 2026 | PDF says "Preprint" | **Fixed: authors** | [arXiv:2605.21800](https://arxiv.org/abs/2605.21800) |
| 27 | `dinowm` | OK | **ICML 2025** (bib said arXiv 2024) | S2 (PDF has no venue line) | **Fixed: venue, year** | [arXiv:2411.04983](https://arxiv.org/abs/2411.04983) |
| 28 | `pldm` | **Two names wrong**: "Kynghyun" → Kyunghyun Cho; "Rodriguez" → Rudner | **NeurIPS 2025** (bib said arXiv) | PDF footer "NeurIPS 2025"; S2 | **Fixed: authors, venue** | [arXiv:2502.14819](https://arxiv.org/abs/2502.14819) |
| 29 | `vjepa2` | "Assran, Mahmoud and others" → full 30-author list; first author is Mido Assran | arXiv 2025 | S2 | **Fixed: authors** | [arXiv:2506.09985](https://arxiv.org/abs/2506.09985) |
| 30 | `planet` | OK | ICML 2019 | PDF: Proc. 36th ICML | OK | [arXiv:1811.04551](https://arxiv.org/abs/1811.04551) |
| 31 | `dreamerv3` | **Published title differs**: *Mastering Diverse Control Tasks through World Models* | **Nature 640:647–653, 2025** (bib said arXiv 2023) | Crossref, DOI 10.1038/s41586-025-08744-2 | **Fixed: title, venue, DOI** | [doi:10.1038/s41586-025-08744-2](https://doi.org/10.1038/s41586-025-08744-2) · [arXiv:2301.04104](https://arxiv.org/abs/2301.04104) |
| 32 | `director` | OK | NeurIPS 2022 | S2 (arXiv PDF is the pre-review version) | OK | [arXiv:2206.04114](https://arxiv.org/abs/2206.04114) |
| 33 | `polo` | OK | ICLR 2019 | arXiv comment "Accepted at ICLR 2019" | OK | [arXiv:1811.01848](https://arxiv.org/abs/1811.01848) |
| 34 | `tdmpc` | OK | ICML 2022 | PDF: Proc. 39th ICML | OK | [arXiv:2203.04955](https://arxiv.org/abs/2203.04955) |
| 35 | `tdmpc2` | OK | ICLR 2024 | PDF header | OK | [arXiv:2310.16828](https://arxiv.org/abs/2310.16828) |
| 36 | `gcterminal` | **Authors were a placeholder, no ID** → Morita, Yamamori, Yagi, Sugimoto, Morimoto | arXiv 2024 | PDF: "Preprint submitted to Neural Networks"; no published version found | **Fixed: authors, ID** | [arXiv:2410.04929](https://arxiv.org/abs/2410.04929) |
| 37 | `ogbench` | OK | ICLR 2025 | PDF header | OK | [arXiv:2410.20092](https://arxiv.org/abs/2410.20092) |
| 38 | `leap` *(uncited)* | OK | NeurIPS 2019 | arXiv comment; S2 | OK | [arXiv:1911.08453](https://arxiv.org/abs/1911.08453) |
| 39 | `fastlewm` *(uncited)* | **Authors were a placeholder** → Yuntian Gao, Xiangyu Xu | arXiv 2026 | S2 | **Fixed: authors** | [arXiv:2606.26217](https://arxiv.org/abs/2606.26217) |

**Left to check by hand**
- **`dinowm`:** the ICML 2025 venue comes only from Semantic Scholar, because the arXiv PDF has no venue line. It is very likely right, but it has not been checked against PMLR.
- **Published PDFs:** for ICML/NeurIPS/ICLR papers the link goes to arXiv, not the proceedings page. Swap in proceedings URLs if you prefer them.
- **`dreamerv3` wording:** the paper's text (L158) says "PlaNet and Dreamer", which still fits. The reference now uses the Nature title.

## 2. Wording in the paper that may not match the source

> Superseded by the full claim-by-claim audit in [CLAIMS_AUDIT.md](CLAIMS_AUDIT.md) (74 claims). The table below is the first pass.

| Where | Claim in paper | What the source says |
|---|---|---|
| L191, Related Work | "TTGS \citep{ttgs} and ALPS \citep{alps} do the same at test time for frozen OGBench policies" | **Correct for TTGS**: a training-free graph-search wrapper around existing GCRL policies. **Not correct for ALPS**: it learns a Laplacian representation and a model, then does hierarchical *decision-time planning*. It is a model-based planner, not a wrapper around frozen policies. Suggest: "TTGS wraps frozen OGBench policies with test-time graph search, and ALPS plans hierarchically in a learned Laplacian representation." |
| L103 vs L165 | TRM is grouped with "a retrained encoder" (L103) but described as a learned pairwise cost used as the terminal ranking (L165) | These two descriptions conflict. L165 matches the paper's title ("World Model Control by Trajectory Reachability Metrics"). Check L103. |
| L108, Intro | Quote: ``data-supported, executable, and matched to the temporal scale'' | **Verbatim**, found in Hi-LeWM. The full sentence ends "…at which they are used." The truncation is fine. |
| L688, Limitations | SAGE trains "on 400k expert windows" | **Verified**: SAGE samples 400k training windows from expert trajectories for each component. |
| L179, Related Work | Hi-LeWM "shows the naive version loses to flat planning because CEM exploits the learned subgoal space" | **Verified**: "The straightforward Hi-LeWM planner often underperforms flat LeWM", and "CEM can exploit the terminal latent objective by selecting macro-actions…". |
| L387, Setup | Push-T 18,685 episodes, 20 px / π/9 success, cited to `lewm` | These numbers were **not found in the LeWM PDF** text. They come from our dataset and environment code. Citing LeWM for the environment/dataset is fine, but don't imply the numbers come from that paper. |
| L391, Setup | Cube success threshold 0.04 m, cited to `ogbench` | Not found in the OGBench PDF. It comes from the code (`CubeEnv._compute_successes`), and the paper says so. OK as written. |

---

## 3. Every citation and what we cite it for (order of first appearance in the paper)

Numbered by where each reference is **first** cited. Keys cited together in one `\citep{…}` keep that group's order. Every use of the key is listed under its entry.

### 1. `lewm` — LeWorldModel (Maes, Le Lidec, Scieur, LeCun, Balestriero; arXiv 2026) · [arXiv](https://arxiv.org/abs/2603.19312) · [pdf](pdfs/lewm.pdf)
- **L86 (Intro):** One of the joint-embedding world models that plan by CEM search with a terminal L2 latent cost. This is the default objective we improve.
- **L160 (Related Work, planning with latent WMs):** Listed with DINO-WM/PLDM/V-JEPA 2-AC as models that plan zero-shot with CEM using terminal latent distance.
- **L224 (Method):** The frozen model we use: ViT encoder → 192-d CLS latent; predictor takes 3 past latents + 3 action blocks; an action block is 5 env actions (frame-skip 5).
- **L387 (Setup):** Source of the Push-T environment and dataset.

### 2. `dinowm` — DINO-WM (Zhou, Pan, LeCun, Pinto; ICML 2025) · [arXiv](https://arxiv.org/abs/2411.04983) · [pdf](pdfs/dinowm.pdf)
- **L86 (Intro):** Example of a pixel-trained joint-embedding world model that plans by CEM + L2.
- **L159 (Related Work):** Plans zero-shot with CEM in a joint-embedding latent using terminal distance.

### 3. `pldm` — PLDM: Learning from Reward-Free Offline Data (Sobal et al.; NeurIPS 2025) · [arXiv](https://arxiv.org/abs/2502.14819) · [pdf](pdfs/pldm.pdf)
- **L86 (Intro)** and **L159 (Related Work):** Same role as DINO-WM.

### 4. `sage` — SAGE (Cheng, Zhang, Wang; arXiv 2026) · [arXiv](https://arxiv.org/abs/2607.17973) · [pdf](pdfs/sage.pdf)
- **L101 (Intro):** Learns a subgoal generator and an action prior. It is an example of "learning something new on top of the frozen model".
- **L177 (Related Work, hierarchical):** Learns a subgoal generator and a subgoal-conditioned action prior that CEM refines.
- **L688 (Limitations):** Reports higher *same-episode* success than ours at long offsets with the same frozen model, by training on 400k expert windows. We do not claim same-episode SOTA, and we argue a single-trajectory generator cannot stitch across episodes.

### 5. `hilewm` — Mind the Gap / Hi-LeWM (Caselli et al.; arXiv 2026, WM@Booth workshop) · [arXiv](https://arxiv.org/abs/2607.12547) · [pdf](pdfs/hilewm.pdf)
- **L101 (Intro):** Learns a macro-action space and a high-level predictor.
- **L108 (Intro):** Our main motivation. We quote its conditions for subgoals to transfer to a frozen compact world model: "data-supported, executable, and matched to the temporal scale".
- **L179 (Related Work):** The naive hierarchical version loses to flat planning because CEM exploits the learned subgoal space.

### 6. `hwm` — Hierarchical Planning with Latent World Models (Zhang et al.; arXiv 2026) · [arXiv](https://arxiv.org/abs/2604.03208) · [pdf](pdfs/hwm.pdf)
- **L102 (Intro):** World models at several time scales.
- **L176 (Related Work):** Uses long-horizon predictions as subgoals.

### 7. `ffjepa` — FF-JEPA (Masip et al.; arXiv 2026) · [arXiv](https://arxiv.org/abs/2606.09311) · [pdf](pdfs/ffjepa.pdf)
- **L102 (Intro)** and **L178 (Related Work):** Learns an action-free latent planner.

### 8. `tdjepa` — Temporal-Distance JEPA (Bai, Xiong; arXiv 2026) · [arXiv](https://arxiv.org/abs/2607.25337) · [pdf](pdfs/tdjepa.pdf)
- **L103 (Intro):** Retrains the encoder so its geometry encodes temporal cost.
- **L166 (Related Work, repairing the cost):** Pushes a temporal-distance signal into the encoder at training time.

### 9. `rcaux` — RC-aux (Li et al.; arXiv 2026) · [arXiv](https://arxiv.org/abs/2605.07278) · [pdf](pdfs/rcaux.pdf)
- **L103 (Intro)** and **L167 (Related Work):** Same role as `tdjepa`: an encoder-side training fix.

### 10. `trm` — World Model Control by Trajectory Reachability Metrics (Li et al.; arXiv 2026) · [arXiv](https://arxiv.org/abs/2605.22164) · [pdf](pdfs/trm.pdf)
- **L103 (Intro):** Grouped with encoder-geometry fixes. **See §2: this may conflict with L165.**
- **L165 (Related Work):** Learns a pairwise temporal cost from logged trajectories and uses it as the terminal ranking. This is the closest "cost repair on a frozen model" prior work.

### 11. `sptm` — Semi-parametric Topological Memory (Savinov, Dosovitskiy, Koltun; ICLR 2018) · [arXiv](https://arxiv.org/abs/1803.00653) · [pdf](pdfs/sptm.pdf)
- **L106 (Intro):** Older offline-GCRL line that builds a dataset graph and uses shortest paths to pick subgoals.
- **L188 (Related Work):** Builds a graph over the replay buffer with a learned distance and plans waypoints for a goal-conditioned policy.

### 12. `sorb` — Search on the Replay Buffer (Eysenbach, Salakhutdinov, Levine; NeurIPS 2019) · [arXiv](https://arxiv.org/abs/1906.05253) · [pdf](pdfs/sorb.pdf)
- **L106 (Intro)**, **L188 (Related Work):** Same role as SPTM.

### 13. `sgm` — Sparse Graphical Memory (Emmons et al.; NeurIPS 2020) · [arXiv](https://arxiv.org/abs/2003.06417) · [pdf](pdfs/sgm.pdf)
- **L106 (Intro)**, **L188 (Related Work):** Same role as SPTM.

### 14. `gas` — Graph-Assisted Stitching (Baek et al.; ICML 2025) · [arXiv](https://arxiv.org/abs/2506.07744) · [pdf](pdfs/gas.pdf)
- **L106 (Intro):** Part of the dataset-graph line.
- **L116 (Intro, our method):** Source of the temporal distance representation (TDR) we learn on frozen LeWM latents.
- **L189 (Related Work):** Clusters a TDR into nodes, applies a temporal-efficiency filter, and searches the graph for subgoals.
- **L248 (Method, Eq. TDR):** Our TDR training follows HILP/GAS: expectile TD, EMA target, γ = 0.99, the goal-sampling mix, and the min-of-two-networks target trick.

### 15. `ttgs` — Test-Time Graph Search (Opryshko et al.; ICML 2026) · [arXiv](https://arxiv.org/abs/2510.07257) · [pdf](pdfs/ttgs.pdf)
- **L106 (Intro)**, **L191 (Related Work):** A training-free graph search over the dataset at test time, wrapped around frozen OGBench GCRL policies.

### 16. `alps` — Laplacian Representations for Decision-Time Planning (Shehmar, Schlegel, Taylor, Machado; ICML 2026) · [arXiv](https://arxiv.org/abs/2602.05031) · [pdf](pdfs/alps.pdf)
- **L106 (Intro)**, **L191 (Related Work):** Cited as test-time search for frozen OGBench policies. **See §2: this description is inaccurate.**

### 17. `hilp` — Foundation Policies with Hilbert Representations (Park, Kreiman, Levine; ICML 2024) · [arXiv](https://arxiv.org/abs/2402.15567) · [pdf](pdfs/hilp.pdf)
- **L116 (Intro):** Co-source of the TDR, with GAS.
- **L190 (Related Work):** One of the temporal distance representations GAS builds on.
- **L248 (Method):** Source of the TDR objective and the stabilisation trick.

### 18. `planet` — PlaNet: Learning Latent Dynamics for Planning from Pixels (Hafner et al.; ICML 2019) · [arXiv](https://arxiv.org/abs/1811.04551) · [pdf](pdfs/planet.pdf)
- **L158 (Related Work):** Background. It plans in a *reconstruction-based* latent, in contrast to joint-embedding models.

### 19. `dreamerv3` — Mastering Diverse Control Tasks through World Models (Hafner, Pasukonis, Ba, Lillicrap; Nature 2025) · [doi](https://doi.org/10.1038/s41586-025-08744-2) · [arXiv](https://arxiv.org/abs/2301.04104) · [pdf](pdfs/dreamerv3.pdf) (arXiv version)
- **L158 (Related Work):** Background, paired with PlaNet. It learns policies in a reconstruction-based latent.

### 20. `vjepa2` — V-JEPA 2 (Assran et al.; arXiv 2025) · [arXiv](https://arxiv.org/abs/2506.09985) · [pdf](pdfs/vjepa2.pdf)
- **L159 (Related Work):** V-JEPA 2-AC is another model that plans with CEM and terminal latent distance.

### 21. `jepawm-study` — What Drives Success in Physical Planning with JEPA World Models? (Terver et al.; TMLR 2026) · [arXiv](https://arxiv.org/abs/2512.24497) · [pdf](pdfs/jepawm-study.pdf)
- **L161 (Related Work):** A design-choice study. We use it to support the point that the planning horizon is a known weak spot.

### 22. `stableworldmodel` — stable-worldmodel platform (Maes et al.; arXiv 2026) · [arXiv](https://arxiv.org/abs/2605.21800) · [pdf](pdfs/stableworldmodel.pdf)
- **L161 (Related Work):** Same sentence as `jepawm-study`. (We also use this library in code, but the paper does not cite it for that.)

### 23. `dalewm` — Decision-Metric Alignment in Latent World Models (Wang et al.; arXiv 2026) · [arXiv](https://arxiv.org/abs/2608.18746) · [pdf](pdfs/dalewm.pdf)
- **L167 (Related Work):** Pushes decision-relevant signals into the encoder at training time.

### 24. `scale` — SCALE (Hu, Zheng, Wang; arXiv 2026) · [arXiv](https://arxiv.org/abs/2608.16287) · [pdf](pdfs/scale.pdf)
- **L167 (Related Work):** Same group as DA-LeWM: state-calibrated encoder geometry.

### 25. `valueguided` — Value-Guided Action Planning with JEPA World Models (Destrade et al.; arXiv 2026, workshop) · [arXiv](https://arxiv.org/abs/2601.00844) · [pdf](pdfs/valueguided.pdf)
- **L168 (Related Work):** Shapes the latent geometry toward a value function.

### 26. `straightening` — Temporal Straightening for Latent Planning (Wang et al.; ICML 2026) · [arXiv](https://arxiv.org/abs/2603.12231) · [pdf](pdfs/straightening.pdf)
- **L168 (Related Work):** Shapes the latent geometry toward straight trajectories.

### 27. `qrl` — Quasimetric RL (Wang, Torralba, Isola, Zhang; ICML 2023) · [arXiv](https://arxiv.org/abs/2304.01203) · [pdf](pdfs/qrl.pdf)
- **L190 (Related Work):** Alternative temporal-distance representation, cited next to HILP.

### 28. `ogbench` — OGBench (Park, Frans, Eysenbach, Levine; ICLR 2025) · [arXiv](https://arxiv.org/abs/2410.20092) · [pdf](pdfs/ogbench.pdf)
- **L192 (Related Work):** The benchmark TTGS/ALPS evaluate on.
- **L391 (Setup):** Source of the Cube environment and dataset: 10,000 episodes of 201 frames. The success threshold is taken from code.

### 29. `polo` — POLO: Plan Online, Learn Offline (Lowrey et al.; ICLR 2019) · [arXiv](https://arxiv.org/abs/1811.01848) · [pdf](pdfs/polo.pdf)
- **L197 (Related Work):** Puts a learned value at the end of a short model rollout. This is precedent for our terminal-cost replacement.

### 30. `tdmpc` — TD-MPC (Hansen, Wang, Su; ICML 2022) · [arXiv](https://arxiv.org/abs/2203.04955) · [pdf](pdfs/tdmpc.pdf)
- **L197 (Related Work):** Same role as POLO.

### 31. `tdmpc2` — TD-MPC2 (Hansen, Su, Wang; ICLR 2024) · [arXiv](https://arxiv.org/abs/2310.16828) · [pdf](pdfs/tdmpc2.pdf)
- **L197 (Related Work):** Same role as POLO.

### 32. `director` — Director: Deep Hierarchical Planning from Pixels (Hafner et al.; NeurIPS 2022) · [arXiv](https://arxiv.org/abs/2206.04114) · [pdf](pdfs/director.pdf)
- **L197 (Related Work):** Grouped with the value-at-the-end-of-rollout methods. It is also hierarchical, with a learned manager/worker in latent space.

### 33. `gcterminal` — Goal-Conditioned Terminal Value Estimation for MPC (Morita et al.; arXiv 2024) · [arXiv](https://arxiv.org/abs/2410.04929) · [pdf](pdfs/gcterminal.pdf)
- **L199 (Related Work):** Goal-conditioned terminal values for multi-task MPC. This is the closest precedent for a goal-conditioned terminal cost inside MPC.
