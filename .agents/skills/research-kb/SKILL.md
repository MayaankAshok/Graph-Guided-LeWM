---
name: research-kb
description: >-
  Use this skill when proposing new research hypotheses, conducting radical simplification research ideation,
  querying open research questions, scaffolding or updating atomic knowledge cards in docs/knowledge/,
  and maintaining index/graph integrity using scripts/tools/kb_manager.py.
---

# Research Knowledge Base & Ideation Skill (`research-kb`)

This skill governs research ideation, atomic knowledge management, and hypothesis testing in the post-ICLR phase of the LEWM project.

---

## 1. Post-ICLR Research Mandate: Radical Simplification

The composite planning stack established in the ICLR 2027 submission (**OUR Method**: frozen LeWM + TDR mapping + offline GAS graph with Dijkstra + budget-capped expected hitting time critic + CEM + final L2 threshold switch) demonstrated strong empirical gains (+32% Push-T, +36% Reacher, +11% Cube). However, it represents an overcomplicated, piecewise patchwork with discrete graph lookups, heuristic threshold switching, and multi-objective balancing.

**Active Research Goal:**
Investigate whether unified end-to-end long-horizon planners (e.g., continuous flow/diffusion trajectory generators, quasimetric energy models, or direct goal-conditioned value functions) can match or surpass the composite stack without:
- Discrete graph lookups and expensive offline topological indexing.
- Heuristic threshold switching ($\theta_{\text{final}}$) and short-horizon regression chattering.
- Multi-objective parameter balancing ($\beta$ weights between distance, critic, and L2).

---

## 2. Navigating the Active Research Frontier

Before proposing any new experiment or code change, an agent must inspect the active frontier of open questions.

### CLI Commands
```bash
# List all unresolved research questions grouped by parent card & tags
python scripts/tools/kb_manager.py questions

# Search existing findings and cards by keyword
python scripts/tools/kb_manager.py search "diffusion"
python scripts/tools/kb_manager.py search "overshoot"

# Check overall KB status
python scripts/tools/kb_manager.py report
```

### Knowledge Base Structure
All cards live in `docs/knowledge/`:
- `docs/knowledge/INDEX.md`: Master index with live Mermaid DAG and categorized open questions.
- `EXP-001` to `EXP-018`: Established empirical findings and mechanistic phenomena.

---

## 3. Formulating an RFC (Request for Comments) Proposal

Every new research initiative must begin with a concrete RFC hypothesis card targeting an open question from `docs/knowledge/INDEX.md`.

An RFC proposal must clearly specify:
1. **Target Bottleneck**: Which open question or failure mode is addressed (e.g., `EXP-008` switching chattering, `EXP-009` contact overshoot, `EXP-010` 3D graph sparsity).
2. **Radical Simplification Rationale**: Exactly which composite components are eliminated or unified.
3. **Falsifiable Claim & Minimal Test**: The simplest possible experiment that can falsify the hypothesis before launching massive cluster sweeps.
4. **Benchmark Comparison**: Clear target metric against the headline benchmark (`EXP-007` OUR method: Push-T 42.4%, Reacher 65.6%, Cube 27.6% on cross-episode).

---

## 4. Card Creation & Lifecycle Protocol

### Step 1: Scaffold the Card
```bash
python scripts/tools/kb_manager.py new \
  --title "Latent Flow Matching Trajectory Generator" \
  --derived-from EXP-001 EXP-005 \
  --tags "flow-matching" "diffusion" "planning" "radical-simplification" \
  --status "open_hypothesis"
```

### Step 2: Fill in the 4-Section Schema
Every card must strictly follow this 4-section schema:
```markdown
# EXP-XXX: <Title>

## 1. Motivation & Provenance
<!-- What is the problem being addressed? Where is the need for this experiment derived from? -->

## 2. Experimental Protocol & Method
<!-- Exact scripts, configurations, data splits, environment setup, and evaluation metrics used. -->

## 3. Empirical Results
<!-- Quantitative findings, benchmark numbers, comparison tables, error bounds, and statistical significance. -->

## 4. Synthesis & Next Questions
<!-- What does this resolve from the initial problem statement? What new hypotheses or questions arise? -->
```

### Step 3: Card Status Progression
- `open_hypothesis`: Initial proposal / RFC awaiting experimental execution.
- `in_progress`: Experiments actively running on cluster or local GPU.
- `established`: Peer-reviewed/validated empirical findings with reproducible numbers.
- `superseded`: Prior hypotheses or obsolete configurations replaced by newer insights.

### Step 4: Validate and Synchronize
```bash
# Validate frontmatter, required sections, and link integrity
python scripts/tools/kb_manager.py validate

# Rebuild INDEX.md and the live Mermaid DAG
python scripts/tools/kb_manager.py sync-index
```
*(Always run both commands before completing any research turn!)*
