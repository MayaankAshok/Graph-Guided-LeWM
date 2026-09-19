"""Read-only task provenance audit. Run with Python (standard library only)."""
import hashlib
import ast
import json
import math
import re
import statistics
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
EVAL = ROOT / "outputs/pusht/eval"
BASE = "subgoal_tdr_la13.7_th8_htd8_te0.9_ft13.7_fl2_crit1_ccet"
PROTOS = {"same25": "same_episode_off25", "same50": "same_episode_off50",
          "same100": "same_episode_off100", "cross": "cross_episode"}


def errors(p):
    pos, ang = [], []
    for s, g in zip(p["start_state"], p["goal_state"]):
        pos.append(math.sqrt(sum((s[i] - g[i]) ** 2 for i in range(4))))
        a = abs(s[4] - g[4])
        ang.append(min(a, 2 * math.pi - a))
    return pos, ang


def signature(p):
    keys = ["start_row", "goal_row", "start_state", "goal_state"]
    raw = json.dumps({k: p[k] for k in keys}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()


def validate_heldout(p):
    assert p["n"] == 200 and p["nonoverlap"]
    held = set(p["heldout_eps"])
    assert set(p["start_ep"]) <= held and set(p["goal_ep"]) <= held
    pos, ang = errors(p)
    assert not any(x < 20 and a < math.pi / 9 for x, a in zip(pos, ang))
    occupied = {}
    for se, ge, ss, gs in zip(p["start_ep"], p["goal_ep"], p["start_step"], p["goal_step"]):
        if p["pairing"] == "same_episode":
            assert se == ge and gs - ss == p["offset"]
            intervals = [(se, ss, gs)]
        else:
            assert se != ge
            intervals = [(se, ss, ss), (ge, gs, gs)]
        for ep, lo, hi in intervals:
            assert all(hi < a or lo > b for a, b in occupied.get(ep, []))
            occupied.setdefault(ep, []).append((lo, hi))


def main():
    output = {}
    for proto, tag in PROTOS.items():
        pools = {}
        for f in sorted((HERE / "pairs").glob(f"pairs_{tag}_task200*.json")):
            p = json.loads(f.read_text())
            if "heldout_eps" in p:
                validate_heldout(p)
            pools[f.name] = {"signature": signature(p), "errors": errors(p),
                             "pairs": p, "heldout": "heldout_eps" in p}
        records = []
        groups = {}
        files = {f.name: f for f in EVAL.glob(f"*__{proto}__*__s*__n50*.json")}
        files.update({f.name: f for f in (HERE / "eval").glob(f"*__{proto}__*__s*__n50*.json")})
        for f in sorted(files.values()):
            if re.search(r"__c\d+$", f.stem):
                continue
            r = json.loads(f.read_text())
            if r["method"] not in (BASE, "l2") and "matrix_" not in r["method"]:
                continue
            pos, ang, hit = [], [], []
            for ci in (0, 25):
                stem = f.stem
                if stem.endswith("__heldout_disjoint"):
                    stem = stem[:-len("__heldout_disjoint")] + f"__heldout_disjoint__c{ci}"
                else:
                    stem += f"__c{ci}"
                chunk = f.parent / (stem + ".json")
                if chunk.exists():
                    c = json.loads(chunk.read_text())
                    pos.extend(c["initial_pos_err"])
                    ang.extend(c["initial_ang_err"])
                    hit.extend(c["first_hit_step"])
            matches = []
            deltas = {}
            for name, pool in pools.items():
                pp, aa = pool["errors"]
                if len(pos) == 50:
                    delta = max(abs(x - y) for x, y in zip(pos, pp[:50]))
                    adelta = max(abs(x - y) for x, y in zip(ang, aa[:50]))
                    deltas[name] = [delta, adelta]
                    if delta < 1e-8 and adelta < 1e-8:
                        matches.append(name)
            consistent = len(hit) == 50 and hit == r["first_hit_step"]
            assert consistent, f"Missing or inconsistent chunks: {f}"
            assert matches, f"Task provenance did not match any snapshot: {f}"
            calculated = 100 * sum(x >= 0 for x in r["first_hit_step"]) / r["n"]
            assert abs(calculated - r["success_rate"]) < 1e-8, f
            row = {"file": f.name, "seed": r["seed"], "rate": r["success_rate"],
                   "pool_label": r.get("pool", "task200"), "matches": matches,
                   "run_suffix": "heldout_disjoint" if "__heldout_disjoint" in f.stem else "legacy",
                   "max_error_deltas": deltas, "chunks_consistent": consistent,
                   "mpc": r["mpc"]}
            records.append(row)
            groups.setdefault((r["method"], row["pool_label"], row["run_suffix"]), []).append(row)
        summary = []
        for (method, label, suffix), rows in groups.items():
            rows.sort(key=lambda r: r["seed"])
            rates = [r["rate"] for r in rows]
            summary.append({"method": method, "label": label, "run_suffix": suffix,
                            "seeds": [r["seed"] for r in rows], "rates": rates,
                            "mean": statistics.mean(rates),
                   "sd": statistics.stdev(rates) if len(rates) > 1 else None,
                   "population_sd": statistics.pstdev(rates)})
        identity = {n: {"sha256": p["signature"], "heldout": p["heldout"],
                         "first_start_row": p["pairs"]["start_row"][0],
                         "first_errors": [p["errors"][0][0], p["errors"][1][0]]}
                    for n, p in pools.items()}
        pool_comparisons = []
        names = list(pools)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                ap, bp = pools[a]["pairs"], pools[b]["pairs"]
                aset = set(zip(ap["start_row"][:50], ap["goal_row"][:50]))
                bset = set(zip(bp["start_row"][:50], bp["goal_row"][:50]))
                pool_comparisons.append({"a": a, "b": b,
                                         "identical_all_200": pools[a]["signature"] == pools[b]["signature"],
                                         "shared_first50_pairs": len(aset & bset)})
        config_comparisons = []
        for old in records:
            if "matrix_oldcritic_oldtasks" not in old["file"]:
                continue
            new = next((r for r in records if "matrix_oldcritic_newtasks" in r["file"]
                        and r["run_suffix"] == "heldout_disjoint"
                        and r["seed"] == old["seed"]), None)
            if new:
                differences = {k: [old["mpc"].get(k), new["mpc"].get(k)]
                               for k in set(old["mpc"]) | set(new["mpc"])
                               if k not in ("tag", "force") and old["mpc"].get(k) != new["mpc"].get(k)}
                config_comparisons.append({"seed": old["seed"], "differences": differences})
        output[proto] = {"pools": identity, "pool_comparisons": pool_comparisons,
                         "config_comparisons": config_comparisons,
                         "summary": summary, "records": records}
        print(proto)
        for s in summary:
            print(s["label"], s["run_suffix"], s["method"].replace(BASE, "B"), s["rates"],
                  f'{s["mean"]:.1f}', f'{s["sd"]:.2f}' if s["sd"] is not None else "")
        for row in records:
            print(" ", row["seed"], row["pool_label"], row["matches"], row["chunks_consistent"])
    (HERE / "findings.json").write_text(json.dumps(output, indent=2) + "\n")


def reproduce_loader_bug():
    """Exercise the actual loader without importing GPU/research dependencies."""
    source = ROOT / "scripts/viability_cross_episode_baseline.py"
    tree = ast.parse(source.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
              and n.name == "load_or_make_pairs")
    old = {"start_row": [1], "goal_row": [26]}
    new = {"start_row": [101], "goal_row": [126], "nonoverlap": True}
    ns = {"json": json, "log": lambda x: None, "sample_pairs": lambda **kw: new}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(source), "exec"), ns)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "pairs_same_episode_off25_task200.json"
        path.write_text(json.dumps(old))
        result = ns["load_or_make_pairs"](path, heldout_eps=[9])
        assert result == new and json.loads(path.read_text()) == new
        assert json.loads(path.with_name(path.stem + ".full_dataset.json").read_text()) == old
    print("Confirmed: current shared loader replaces task200 with held-out tasks")


if __name__ == "__main__":
    reproduce_loader_bug()
    main()
