"""Config composition and dependency sanity check; no GPU or dataset needed."""

import ast
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from evaluator import ROOT, check_config, compose


def demo():
    scripts = ROOT / "scripts"
    assert not list(scripts.glob("*.py")), "Workflow scripts belong in subfolders"
    packages = {path.name for path in scripts.iterdir() if path.is_dir()}
    moved_names = {path.stem for path in scripts.rglob("*.py") if path.name != "__init__.py"}
    for path in [ROOT / "evaluator.py", *scripts.rglob("*.py")]:
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"), filename=str(path))):
            if isinstance(node, ast.ImportFrom) and node.module:
                assert node.module not in moved_names, f"{path}: stale flat import {node.module}"
                if node.module.split(".")[0] in packages:
                    target = scripts.joinpath(*node.module.split("."))
                    assert target.with_suffix(".py").is_file() or (target / "__init__.py").is_file(), \
                        f"{path}: missing local module {node.module}"
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if node.value.startswith("scripts/") and node.value.endswith(".py"):
                    assert (ROOT / node.value).is_file(), f"{path}: missing script {node.value}"
    for environment in ("pusht", "reacher", "cube"):
        cfg = compose(f"config/evaluations/{environment}_headline.yaml")
        check_config(cfg)
        assert cfg.environment == environment
        assert list(cfg.evaluation.methods) == ["l2", "our"]
        assert "critic" not in cfg.methods.l2.requires
        assert {"embeddings_train", "embeddings_test", "tdr", "graph", "critic", "tasks"} <= set(cfg.methods.our.requires)
        assert list(cfg.evaluation.seeds) == [0, 1, 2, 3, 4]
        assert list(cfg.evaluation.protocols) == ["same25", "same50", "same100", "cross"]
        cfg.evaluation.pool = "unfiltered"
        try:
            check_config(cfg)
        except ValueError as exc:
            assert "task200u" in str(exc)
        else:
            raise AssertionError("unsupported task pool was accepted")
    for path in (Path(__file__).resolve().parents[2] / "config/evaluations").glob("paper_*.yaml"):
        paper = compose(path)
        check_config(paper)
        assert int(paper.evaluation.num_eval) == 50
        assert list(paper.evaluation.protocols)
        assert list(paper.evaluation.methods)


if __name__ == "__main__":
    demo()
