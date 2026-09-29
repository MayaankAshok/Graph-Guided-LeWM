# Research Knowledge Base: Architecture & Workflow Guide

## Purpose
This directory (`docs/knowledge/`) houses the persistent, evolving **Web of Ideas** for research on long-horizon world-model planning. 

For submitted benchmark figures, use `docs/iclr2027/main.tex` and name its table and statistic in each card. `docs/gas-mpc/results*.json` and `results_table*.tex` include earlier runs; keep their numbers labeled as historical rather than combining them with submitted rows.

Rather than maintaining monolithic unstructured research notes, each markdown file in this directory represents **one atomic result, experiment, or empirical finding**, interconnected into a Directed Acyclic Graph (DAG) that leads from established facts to the active research frontier.

---

## The 4-Section Card Anatomy
Every experiment card follows a standardized schema:

```markdown
---
id: EXP-XXX
title: <Clear descriptive title>
status: established | in_progress | open_hypothesis | superseded
created: YYYY-MM-DD
tags: [tag1, tag2]
derived_from: [EXP-YYY]
leads_to: [EXP-ZZZ]
opens_questions:
  - "<Falsifiable question or hypothesis arising from this result>"
---

# EXP-XXX: <Title>

## 1. Motivation & Provenance
What is the exact problem being investigated? Where did the motivation come from (referencing parent cards)?

## 2. Experimental Protocol & Method
The exact, reproducible implementation details: scripts, configurations, environment setups, data splits, and evaluation protocols.

## 3. Empirical Results
The definitive numbers: benchmark tables, statistical bounds, error metrics, failure breakdowns, and concrete takeaways.

## 4. Synthesis & Next Questions
What did this experiment resolve? What new contradictions, bottlenecks, or hypotheses did it reveal?
```

---

## Tooling & CLI Commands
The knowledge base is managed by [`scripts/tools/kb_manager.py`](../../scripts/tools/kb_manager.py):

| Command | Action |
| :--- | :--- |
| `python scripts/tools/kb_manager.py validate` | Verifies all YAML frontmatter, validates that section headers exist, and checks for broken ID links. |
| `python scripts/tools/kb_manager.py sync-index` | Automatically updates [`INDEX.md`](INDEX.md) with a live Mermaid DAG, active open questions list, and master registry table. |
| `python scripts/tools/kb_manager.py sync-index --check` | Exits with an error if the generated index is stale; suitable for CI. |
| `python scripts/tools/kb_manager.py new --title "..." --derived-from EXP-001` | Scaffolds a new card with the next available ID, template sections, and frontmatter. |
| `python scripts/tools/kb_manager.py report` | Summarizes card counts by status (`established`, `in_progress`, `open_hypothesis`). |

---

## Agentic Ideation Protocol (RFC System)
When engaging in research ideation, the AI agent:
1. Reviews unresolved open questions from [`INDEX.md`](INDEX.md).
2. Proposes 2–3 structured **Research Proposals / Hypotheses** designed for **Radical Simplification**:
   - Stating the core architectural idea.
   - Grounding it in specific open questions from the web.
   - Explaining why it reduces complexity (eliminating components).
   - Formulating a minimal falsification test / smoke experiment.
3. Upon human review and alignment, the agent scaffolds accepted proposals as `open_hypothesis` cards and links them into the DAG.
