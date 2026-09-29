#!/usr/bin/env python3
"""Knowledge Base Manager (scripts/tools/kb_manager.py)

Manages the 'Web of Ideas' research knowledge base in docs/knowledge/.
Features:
  - Scaffolds new atomic experiment/result cards (4-section schema).
  - Validates card frontmatter, schema, and ID cross-references.
  - Automatically compiles and synchronizes the master INDEX.md with a live Mermaid DAG.
  - Lists open research questions and identifies unaddressed bottlenecks.

Usage:
  python scripts/tools/kb_manager.py validate
  python scripts/tools/kb_manager.py sync-index
  python scripts/tools/kb_manager.py new --title "My New Finding" --derived-from EXP-001
  python scripts/tools/kb_manager.py questions
  python scripts/tools/kb_manager.py report
"""

import argparse
import csv
import datetime
import io
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


KB_DIR = Path("docs/knowledge")
INDEX_PATH = KB_DIR / "INDEX.md"

CARD_TEMPLATE = """---
id: {card_id}
title: {title}
status: {status}
created: {date}
tags: [{tags}]
derived_from: [{derived_from}]
leads_to: [{leads_to}]
opens_questions:
{open_questions_yaml}---

# {card_id}: {title}

## 1. Motivation & Provenance
<!-- What is the problem being addressed? Where is the need for this experiment derived from? -->


## 2. Experimental Protocol & Method
<!-- Exact scripts, configurations, data splits, environment setup, and evaluation metrics used. -->


## 3. Empirical Results
<!-- Quantitative findings, benchmark numbers, comparison tables, error bounds, and statistical significance. -->


## 4. Synthesis & Next Questions
<!-- What does this resolve from the initial problem statement? What new hypotheses or questions arise? -->

"""


def parse_frontmatter(content: str) -> Tuple[Optional[Dict], str]:
    """Parse the fixed card schema without an optional YAML dependency."""
    match = re.match(r"^---\r?\n(.*?)\r?\n---\r?\n(.*)$", content, re.DOTALL)
    if not match:
        return None, content
    fm_str, body = match.group(1), match.group(2)
    data = {}
    for line in fm_str.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if line.startswith("  - "):
            if "opens_questions" not in data:
                raise ValueError("list item outside opens_questions")
            data["opens_questions"].append(line[4:].strip().strip('"\''))
            continue
        if line[0].isspace() or ":" not in line:
            raise ValueError(f"unsupported frontmatter line: {line}")
        k, v = (part.strip() for part in line.split(":", 1))
        if k in data:
            raise ValueError(f"duplicate frontmatter field: {k}")
        if k == "opens_questions":
            if v:
                raise ValueError("opens_questions must be a block list")
            data[k] = []
        elif k in {"tags", "derived_from", "leads_to"}:
            if not (v.startswith("[") and v.endswith("]")):
                raise ValueError(f"{k} must be an inline list")
            data[k] = [item.strip() for item in next(csv.reader(io.StringIO(v[1:-1]), skipinitialspace=True), []) if item.strip()]
        else:
            data[k] = v.strip('"\'')
    return data, body


def load_all_cards(kb_dir: Path = KB_DIR) -> Dict[str, Dict]:
    """Load cards; reject malformed files and duplicate IDs instead of hiding them."""
    cards = {}
    if not kb_dir.exists():
        return cards

    for p in sorted(kb_dir.glob("*.md")):
        if p.name.upper() in ("INDEX.MD", "README.MD", "TEMPLATE.MD"):
            continue
        content = p.read_text(encoding="utf-8")
        try:
            fm, body = parse_frontmatter(content)
        except ValueError as exc:
            raise ValueError(f"{p}: {exc}") from exc
        if not fm or "id" not in fm:
            raise ValueError(f"{p}: missing card frontmatter or id")
        
        card_id = str(fm["id"]).strip()
        if card_id in cards:
            raise ValueError(f"{p}: duplicate id {card_id} (also {cards[card_id]['path']})")
        cards[card_id] = {
            "id": card_id,
            "title": fm.get("title", p.stem),
            "status": fm.get("status", "established"),
            "created": str(fm.get("created", "")),
            "tags": fm.get("tags", []) if isinstance(fm.get("tags"), list) else [fm.get("tags")],
            "derived_from": fm.get("derived_from", []) if isinstance(fm.get("derived_from"), list) else [fm.get("derived_from")] if fm.get("derived_from") else [],
            "leads_to": fm.get("leads_to", []) if isinstance(fm.get("leads_to"), list) else [fm.get("leads_to")] if fm.get("leads_to") else [],
            "opens_questions": fm.get("opens_questions", []) if isinstance(fm.get("opens_questions"), list) else [fm.get("opens_questions")] if fm.get("opens_questions") else [],
            "path": p,
            "filename": p.name,
            "body": body,
        }
    return cards


def validate_kb(kb_dir: Path = KB_DIR) -> bool:
    """Validate cross-references, required fields, and section headers."""
    try:
        cards = load_all_cards(kb_dir)
    except ValueError as exc:
        print(f"[FAIL] {exc}")
        return False
    print(f"[kb_manager] Loaded {len(cards)} cards from {kb_dir}")
    errors = []
    if not cards:
        errors.append("No cards found")

    valid_statuses = {"established", "in_progress", "open_hypothesis", "superseded"}

    for cid, c in cards.items():
        fm, _ = parse_frontmatter(c["path"].read_text(encoding="utf-8"))
        for field in ("title", "status", "created", "tags", "derived_from", "leads_to", "opens_questions"):
            if field not in fm:
                errors.append(f"[{cid}] Missing frontmatter field: {field}")
        if not re.fullmatch(r"EXP-\d{3,}", cid) or not c["filename"].startswith(cid + "-"):
            errors.append(f"[{cid}] ID and filename must use EXP-NNN-title.md")
        try:
            datetime.date.fromisoformat(c["created"])
        except ValueError:
            errors.append(f"[{cid}] Invalid created date: {c['created']}")
        # Validate status
        status = c.get("status")
        if status not in valid_statuses:
            errors.append(f"[{cid}] Invalid status '{status}'. Expected one of: {sorted(valid_statuses)}")

        # Validate section headers in body
        required_sections = [
            "## 1. Motivation & Provenance",
            "## 2. Experimental Protocol & Method",
            "## 3. Empirical Results",
            "## 4. Synthesis & Next Questions"
        ]
        for sec in required_sections:
            if not re.search(r"^" + re.escape(sec) + r"\s*$", c["body"], re.MULTILINE):
                errors.append(f"[{cid}] Missing required section: '{sec}'")

        # Validate derived_from links
        for parent_id in c["derived_from"]:
            parent_id = str(parent_id).strip()
            if parent_id and parent_id not in cards:
                errors.append(f"[{cid}] 'derived_from' refers to non-existent card ID: '{parent_id}'")

        # Validate leads_to links
        for child_id in c["leads_to"]:
            child_id = str(child_id).strip()
            if child_id and child_id not in cards:
                errors.append(f"[{cid}] 'leads_to' refers to non-existent card ID: '{child_id}'")
            elif child_id and cid not in cards[child_id]["derived_from"]:
                errors.append(f"[{cid}] 'leads_to' {child_id} is missing reciprocal 'derived_from'")
        for parent_id in c["derived_from"]:
            if parent_id in cards and cid not in cards[parent_id]["leads_to"]:
                errors.append(f"[{cid}] 'derived_from' {parent_id} is missing reciprocal 'leads_to'")

        for target in re.findall(r"\[[^]]+\]\(([^)]+)\)", c["body"]):
            if not re.match(r"^[a-z]+://", target) and not (c["path"].parent / target.split("#", 1)[0]).exists():
                errors.append(f"[{cid}] Broken markdown link: {target}")
        for target in re.findall(r"`((?:docs|scripts)/[^`]+)`", c["body"]):
            target = target.split()[0]
            if not (kb_dir.parent.parent / target).exists():
                errors.append(f"[{cid}] Missing cited file: {target}")

    edges = {(parent, cid) for cid, c in cards.items() for parent in c["derived_from"] if parent in cards}
    edges.update((cid, child) for cid, c in cards.items() for child in c["leads_to"] if child in cards)
    def reaches(start, node, seen):
        if node == start:
            return True
        if node in seen:
            return False
        return any(reaches(start, dst, seen | {node}) for src, dst in edges if src == node)
    for src, dst in sorted(edges):
        if reaches(src, dst, set()):
            errors.append(f"Graph cycle through {src} -> {dst}")
            break

    print("\n--- Validation Results ---")
    if errors:
        print(f"[FAIL] Found {len(errors)} ERROR(S):")
        for e in errors:
            print(f"  - {e}")
    else:
        print("[OK] All card schemas and referenced IDs are valid!")

    return len(errors) == 0


def generate_mermaid_dag(cards: Dict[str, Dict]) -> str:
    """Generate Mermaid graph from cards and their connections."""
    lines = ["```mermaid", "graph TD"]

    # Class definitions for styling
    lines.append("  %% Styles")
    lines.append("  classDef established fill:#e1f5fe,stroke:#0288d1,stroke-width:2px,color:#01579b;")
    lines.append("  classDef in_progress fill:#fff9c4,stroke:#fbc02d,stroke-width:2px,color:#f57f17;")
    lines.append("  classDef open_hypothesis fill:#f3e5f5,stroke:#8e24aa,stroke-width:2px,color:#4a148c;")
    lines.append("  classDef superseded fill:#eeeeee,stroke:#9e9e9e,stroke-width:1px,color:#616161,stroke-dasharray: 5 5;")

    # Nodes
    for cid, c in sorted(cards.items()):
        clean_title = c['title'].replace('"', "'").replace("[", "(").replace("]", ")")
        if len(clean_title) > 36:
            clean_title = clean_title[:33] + "..."
        node_label = f"\"{cid}<br/><b>{clean_title}</b>\""
        lines.append(f"  {cid}[{node_label}]")

    # Links
    edges = set()
    for cid, c in sorted(cards.items()):
        for p in c["derived_from"]:
            p = str(p).strip()
            if p in cards:
                edges.add((p, cid))
        for child in c["leads_to"]:
            child = str(child).strip()
            if child in cards:
                edges.add((cid, child))

    for src, dst in sorted(edges):
        lines.append(f"  {src} --> {dst}")

    # Apply classes
    for cid, c in sorted(cards.items()):
        st = c.get("status", "established")
        lines.append(f"  class {cid} {st};")

    lines.append("```")
    return "\n".join(lines)


def sync_index(kb_dir: Path = KB_DIR, index_path: Path = INDEX_PATH, check: bool = False):
    """Rebuild INDEX.md from all loaded cards."""
    if not validate_kb(kb_dir):
        return False
    cards = load_all_cards(kb_dir)
    print(f"[kb_manager] Synchronizing index with {len(cards)} cards...")

    mermaid_code = generate_mermaid_dag(cards)

    # Collect all open questions
    all_questions = []
    for c in cards.values():
        for q in c["opens_questions"]:
            if q and str(q).strip():
                all_questions.append((c["id"], c["title"], str(q).strip()))

    lines = [
        "# Research Knowledge Base: Web of Ideas",
        "",
        "> **Current Paradigm:** Radical Architectural Rethink & Simplification of Long-Horizon World-Model Planning.",
        f"> **Active Cards:** {len(cards)} | **Open Questions:** {len(all_questions)}",
        "",
        "Welcome to the research knowledge base. Each file in this directory represents **one atomic result or experiment**, structured across four sections: (1) Motivation & Provenance, (2) Method & Protocol, (3) Empirical Results, and (4) Synthesis & Next Questions.",
        "",
        "---",
        "",
        "## Dependency & Emergent Idea Graph",
        "",
        mermaid_code,
        "",
        "---",
        "",
        "## Active Research Frontier: Unresolved Open Questions",
        "",
        "The following open questions and bottlenecks have been identified from empirical results and represent active vectors for new experiments:",
        ""
    ]

    if not all_questions:
        lines.append("*No open questions registered yet.*")
    else:
        for cid, title, q in all_questions:
            c_file = cards[cid]["filename"] if cid in cards else f"{cid}.md"
            lines.append(f"- **[[{cid}]({c_file})]** *({title})*: {q}")

    lines.extend([
        "",
        "---",
        "",
        "## Master Card Registry",
        "",
        "| ID | Title | Status | Tags | Derived From | Leads To |",
        "| :--- | :--- | :---: | :--- | :--- | :--- |"
    ])

    for cid, c in sorted(cards.items()):
        status_badge = {
            "established": "`established`",
            "in_progress": "`in_progress`",
            "open_hypothesis": "`hypothesis`",
            "superseded": "`superseded`"
        }.get(c['status'], c['status'])
        
        tags_str = ", ".join(c.get("tags", []))
        
        def make_link(target_id):
            target_id = str(target_id).strip()
            if target_id in cards:
                return f"[{target_id}]({cards[target_id]['filename']})"
            return target_id

        derived_str = ", ".join([make_link(x) for x in c.get("derived_from", []) if x]) or "-"
        leads_str = ", ".join([make_link(x) for x in c.get("leads_to", []) if x]) or "-"
        
        lines.append(f"| **[{cid}]({c['filename']})** | {c['title']} | {status_badge} | {tags_str} | {derived_str} | {leads_str} |")

    lines.extend([
        "",
        "---",
        "",
        "## How to Use This Knowledge Base",
        "",
        "1. **Scaffold a new card**: Run `python scripts/tools/kb_manager.py new --title \"...\" --derived-from EXP-XXX`.",
        "2. **Validate integrity**: Run `python scripts/tools/kb_manager.py validate` to check links and schemas.",
        "3. **Synchronize index & DAG**: Run `python scripts/tools/kb_manager.py sync-index` to update this document.",
        "4. **Agentic Ideation**: Agents inspect the open questions above, propose hypotheses targeting open bottlenecks, and scaffold new cards.",
        ""
    ])

    content = "\n".join(lines)
    if check:
        if not index_path.exists() or index_path.read_text(encoding="utf-8") != content:
            print(f"[FAIL] {index_path} is stale; run sync-index")
            return False
        print(f"[OK] {index_path} is current")
        return True
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(content, encoding="utf-8")
    print(f"[kb_manager] Successfully wrote index to {index_path}")
    return True


def create_new_card(title: str, derived_from: List[str], tags: List[str], status: str = "open_hypothesis", card_id: Optional[str] = None):
    """Create a new experiment card."""
    if KB_DIR.exists() and any(KB_DIR.glob("EXP-*.md")) and not validate_kb(KB_DIR):
        raise ValueError("fix the knowledge base before creating a card")
    cards = load_all_cards(KB_DIR)
    missing = set(derived_from) - cards.keys()
    if missing:
        raise ValueError(f"unknown parent cards: {', '.join(sorted(missing))}")
    
    if not card_id:
        # Determine next ID
        existing_nums = []
        for cid in cards.keys():
            m = re.match(r"EXP-(\d+)", cid)
            if m:
                existing_nums.append(int(m.group(1)))
        next_num = max(existing_nums, default=0) + 1
        card_id = f"EXP-{next_num:03d}"

    if not re.fullmatch(r"EXP-\d{3,}", card_id) or card_id in cards:
        raise ValueError(f"invalid or duplicate card ID: {card_id}")

    # Build filename slug
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", title.lower()).strip("-")
    filename = f"{card_id}-{slug}.md"
    file_path = KB_DIR / filename

    if file_path.exists():
        print(f"[kb_manager] Error: File {file_path} already exists!")
        return

    today_str = datetime.date.today().isoformat()
    tags_formatted = ", ".join(f'"{t}"' for t in tags)
    derived_formatted = ", ".join(f'"{d}"' for d in derived_from)
    open_q_yaml = '  - "What is the primary falsifiable claim of this hypothesis?"\n'

    content = CARD_TEMPLATE.format(
        card_id=card_id,
        title=title,
        status=status,
        date=today_str,
        tags=tags_formatted,
        derived_from=derived_formatted,
        leads_to="",
        open_questions_yaml=open_q_yaml
    )

    KB_DIR.mkdir(parents=True, exist_ok=True)
    file_path.write_text(content, encoding="utf-8")
    print(f"[kb_manager] Created new card: {file_path}")
    
    # Sync index after creating
    sync_index()


def list_questions(kb_dir: Path = KB_DIR):
    """List open questions across all cards, categorized by card ID and tags."""
    cards = load_all_cards(kb_dir)
    print(f"\n=== Active Research Frontier: Open Questions ({len(cards)} cards) ===\n")
    total_q = 0
    for cid, c in sorted(cards.items()):
        questions = [str(q).strip() for q in c.get("opens_questions", []) if q and str(q).strip()]
        if not questions:
            continue
        tags_str = ", ".join(c.get("tags", []))
        print(f"[{cid}] {c['title']} (tags: {tags_str})")
        for q in questions:
            total_q += 1
            print(f"  ? {q}")
        print()
    print(f"Total open questions tracked: {total_q}\n")


def search_cards(query: str, kb_dir: Path = KB_DIR):
    """Search cards for query keywords in title, tags, or body content."""
    cards = load_all_cards(kb_dir)
    query_lower = query.lower()
    matches = []
    for cid, c in sorted(cards.items()):
        in_title = query_lower in c['title'].lower()
        in_tags = any(query_lower in t.lower() for t in c.get('tags', []))
        in_body = query_lower in c['body'].lower()
        if in_title or in_tags or in_body:
            matches.append((cid, c, in_title, in_tags, in_body))
    print(f"\nSearch results for '{query}': {len(matches)} match(es)\n")
    for cid, c, in_title, in_tags, in_body in matches:
        reasons = []
        if in_title: reasons.append("title")
        if in_tags: reasons.append("tags")
        if in_body: reasons.append("content")
        print(f"- [{cid}] {c['title']} ({', '.join(reasons)})")
        print(f"  File: {c['filename']}")
        print(f"  Status: {c['status']} | Tags: {', '.join(c.get('tags', []))}")


def main():
    parser = argparse.ArgumentParser(description="Knowledge Base Manager")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("validate", help="Validate cards, links, and schemas")
    sync_p = subparsers.add_parser("sync-index", help="Rebuild INDEX.md and Mermaid DAG")
    sync_p.add_argument("--check", action="store_true", help="Fail if INDEX.md is stale")
    
    new_p = subparsers.add_parser("new", help="Scaffold a new card")
    new_p.add_argument("--title", required=True, help="Title of the experiment or finding")
    new_p.add_argument("--derived-from", nargs="*", default=[], help="Parent card IDs (e.g. EXP-001)")
    new_p.add_argument("--tags", nargs="*", default=["research"], help="Topic tags")
    new_p.add_argument("--status", default="open_hypothesis", choices=["established", "in_progress", "open_hypothesis", "superseded"])
    new_p.add_argument("--id", help="Explicit ID (e.g. EXP-010)")

    subparsers.add_parser("questions", help="List active open questions from all cards")

    search_p = subparsers.add_parser("search", help="Search cards by keyword")
    search_p.add_argument("query", help="Search keyword or phrase")

    subparsers.add_parser("report", help="Summary report")

    args = parser.parse_args()

    if args.command == "validate":
        ok = validate_kb()
        sys.exit(0 if ok else 1)
    elif args.command == "sync-index":
        sys.exit(0 if sync_index(check=args.check) else 1)
    elif args.command == "new":
        create_new_card(args.title, args.derived_from, args.tags, args.status, args.id)
    elif args.command == "questions":
        list_questions()
    elif args.command == "search":
        search_cards(args.query)
    elif args.command == "report":
        cards = load_all_cards(KB_DIR)
        print(f"Total cards: {len(cards)}")
        for st in ["established", "in_progress", "open_hypothesis"]:
            count = sum(1 for c in cards.values() if c["status"] == st)
            print(f"  - {st}: {count}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
