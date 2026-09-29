"""Small integrity check for the fixed knowledge-card format."""

import sys
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.kb_manager import load_all_cards, parse_frontmatter, sync_index, validate_kb


def demo():
    with TemporaryDirectory() as directory:
        kb = Path(directory) / "docs" / "knowledge"
        kb.mkdir(parents=True)
        card = kb / "EXP-001-example.md"
        card.write_text("""---
id: EXP-001
title: Example
status: open_hypothesis
created: 2026-09-27
tags: [research]
derived_from: []
leads_to: []
opens_questions:
  - "Can this be tested?"
---
# EXP-001: Example
## 1. Motivation & Provenance
## 2. Experimental Protocol & Method
## 3. Empirical Results
## 4. Synthesis & Next Questions
""", encoding="utf-8")
        assert parse_frontmatter(card.read_text(encoding="utf-8"))[0]["opens_questions"] == ["Can this be tested?"]
        assert validate_kb(kb)
        index = kb / "INDEX.md"
        assert sync_index(kb, index)
        assert sync_index(kb, index, check=True)
        assert "Open Questions:** 1" in index.read_text(encoding="utf-8")
        index.write_text("stale", encoding="utf-8")
        assert not sync_index(kb, index, check=True)
        card.write_text(card.read_text(encoding="utf-8") + "\n[missing](gone.md)\n", encoding="utf-8")
        assert not validate_kb(kb)
        card.write_text(card.read_text(encoding="utf-8").replace("\n[missing](gone.md)\n", ""), encoding="utf-8")
        (kb / "EXP-002-duplicate.md").write_text(card.read_text(encoding="utf-8"), encoding="utf-8")
        try:
            load_all_cards(kb)
        except ValueError as exc:
            assert "duplicate id" in str(exc)
        else:
            raise AssertionError("duplicate card ID was accepted")


if __name__ == "__main__":
    demo()
