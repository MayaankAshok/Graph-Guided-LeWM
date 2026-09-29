"""Regenerate docs/gas-mpc/results_table.tex (and results.json) from task200u evaluations.

Layout, per result count:
  0. one short paragraph stating the conventions shared by every table in the block
  1. algo tables -- one per (base objective, critic on/off), e.g. l2 / subgoal_tdr /
                    subgoal_tdr + critic. Columns = protocols (same-ep 25/50/100, cross-ep).
                    Rows = variants within that algo (graph knobs, retrieval, beta, ...).
                    Cells = mean +/- sd over seeds (n seeds tested), pooled per protocol.
  2. appendix    -- at the end of the document, one per-seed table per (block, protocol),
                    every config with >= 1 seed of data.
Captions carry only what is specific to that table; tables use the [H] float placement
(docs/gas-mpc/main.tex loads the float package).

    python scripts/gas_mpc/gas_mpc_report.py
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gas_mpc.gas_mpc_prepare import ENV, OUT

SUFFIX = "" if ENV == "pusht" else f"_{ENV}"     # GAS_MPC_ENV=reacher -> results_table_reacher.tex etc.

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / "docs" / "gas-mpc"
PROTO_ORDER = ["same25", "same50", "same100", "cross"]
PROTO_LABEL = {"same25": "same-ep 25", "same50": "same-ep 50", "same100": "same-ep 100",
               "cross": "cross-ep"}
BASE_ORDER = ["random", "gciql", "l2", "tdr", "ctg", "subgoal", "subgoal_tdr", "dir", "path"]
BASE_TITLE = {"random": "Random", "gciql": "GCIQL", "l2": "L2", "tdr": "TDR (no graph)", "ctg": "CTG (terminal)",
              "subgoal": "subgoal", "subgoal_tdr": "subgoal\\_tdr",
              "dir": "Direction", "path": "Waypoint path"}


def tex_escape(s):
    return s.replace("_", "\\_")


def parse_tag(tag):
    """'subgoal_tdr_la13.7_th8_htd8_te0.9_ret_crit2_rh2' -> dict of the knobs."""
    d = dict(tag=tag, la=None, th=None, htd=None, te=None, crit=0.0, ret=False, rh=5, sup=0.0,
             ft=None, flt=0.0, fl2=False, cc="nlv", steps=False)
    m = re.match(r"^(random|gciql|l2|tdr|ctg|subgoal_tdr|subgoal|dir|path)(.*)$", tag)
    d["base"], rest = m.group(1), m.group(2)
    for k, v in re.findall(r"_(la|th|htd|te|crit|rh|sup|ft|flt)([0-9.]+)", rest):
        d[k] = float(v) if k != "rh" else int(v)
    d["ret"] = "_ret" in rest
    d["fl2"] = "_fl2" in rest
    d["steps"] = "_steps" in rest
    mm = re.search(r"_cc(etfull|et|nlv)", rest)
    if mm:
        d["cc"] = mm.group(1)
    d["crit"] = float(d["crit"] or 0.0)
    d["flt"] = float(d["flt"] or 0.0)
    return d


def algo_key(t):
    return (t["base"], t["crit"] > 0 or t["flt"] > 0)


def algo_title(key):
    base, has_crit = key
    title = BASE_TITLE.get(base, tex_escape(base))
    return title + (" + critic" if has_crit else "")


def variant_label(t):
    """Row label within an algo table: only the knobs that differ from the algo's defaults."""
    parts = []
    if t["htd"] not in (None, 8.0):
        parts.append(f"$H_{{\\mathrm{{TD}}}}$={t['htd']:g}")
    if t["te"] not in (None, 0.9):
        parts.append(f"TE={t['te']:g}")
    if t["la"] not in (None, 13.7):
        parts.append(f"lookahead={t['la']:g}")
    if t["rh"] != 5:
        parts.append(f"replan={t['rh']}")
    if t["ret"]:
        parts.append("retrieval")
    if t["ft"] is not None:
        parts.append(f"final@{t['ft']:g}")
    if t["fl2"]:
        parts.append("final L2")
    if t["crit"] > 0:
        parts.append(f"$\\beta$={t['crit']:g}")
    if t["cc"] != "nlv":
        parts.append({"et": "ET($h_{\\mathrm{rem}}$)", "etfull": "ET(full)"}[t["cc"]])
    if t["steps"]:
        parts.append("steps")
    if t["flt"] > 0:
        parts.append(f"filter $\\tau$={t['flt']:g}")
    if t["sup"] > 0:
        parts.append(f"support={t['sup']:g}")
    return ", ".join(parts) if parts else "default"


def load_results(n=None):
    """{(pool_tag, n): {method: {protocol: {seed: result}}}}"""
    rows = {}
    for p in sorted((OUT / "eval").glob("*.json")):
        if re.search(r"__c\d+$", p.stem):        # per-chunk cache, not a method result
            continue
        r = json.loads(p.read_text())
        if n is not None and r["n"] != n:
            continue
        pool = r.get("pool", "")
        if pool != "task200u":
            continue
        method = re.sub(r"_cfg[0-9a-f]{12}$", "", r["method"])
        rows.setdefault((pool, r["n"]), {}).setdefault(method, {}) \
            .setdefault(r["protocol"], {})[r["seed"]] = r
    return rows


def pooled(seed_dict):
    hits = sum(r["n_success"] for r in seed_dict.values())
    tot = sum(r["n"] for r in seed_dict.values())
    return 100.0 * hits / tot, hits, tot


def method_sort_key(m):
    t = parse_tag(m)
    return (BASE_ORDER.index(t["base"]), t["crit"], t["flt"], t["htd"] or 0, t["la"] or 0, t["rh"], t["ret"],
            t["ft"] or 0, t["fl2"], t["cc"], t["steps"])


def table(caption, label, cols, body_lines, row_header="configuration"):
    return (["\\begin{table}[H]\n\\centering\\small", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
             "\\begin{tabular}{l" + "r" * len(cols) + "}\n\\toprule",
             f"{row_header} & " + " & ".join(cols) + " \\\\\n\\midrule"]
            + body_lines + ["\\bottomrule\n\\end{tabular}\n\\end{table}\n"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=None)
    args = ap.parse_args()
    DOC.mkdir(parents=True, exist_ok=True)
    rows = load_results(args.n)
    out_json = {}
    main_lines, main_appendix = [], []      # fixed 200-task set -> results_table.tex (in main.tex)
    legacy_lines, legacy_appendix = [], []  # older per-seed pair sets -> legacy_results_table.tex
    # Only the fixed held-out task pool enters active tables.
    for (T, n), methods in sorted(rows.items(), key=lambda kv: (kv[0][0] != "", kv[0][0], -kv[0][1]), reverse=True):
        conv = T if T else "perseed"
        lines, appendix = (main_lines, main_appendix) if T else (legacy_lines, legacy_appendix)
        protos = [p for p in PROTO_ORDER if any(p in v for v in methods.values())]
        labels = [PROTO_LABEL[p] for p in protos]
        by_algo = {}
        for m in methods:
            by_algo.setdefault(algo_key(parse_tag(m)), []).append(m)
        jd = out_json.setdefault(conv + f"_n{n}", {})

        # ---- 0. the shared conventions, once ----
        if T:
            pool_note = "200 tasks, pre-solved pairs rejected"
            lines.append(f"\\subsection*{{Fixed task set \\texttt{{{T}}} ({pool_note}, first {n} used)}}")
            lines.append(f"All tables in this block: the same {n} tasks for every configuration and every seed; a seed "
                         "changes only CEM's random draws. Cells are success rates in \\%, mean $\\pm$ sd over the "
                         "seeds tested (per-seed counts and values are in the appendix). Default settings unless a row "
                         "says otherwise: lookahead 13.7 TDR units, $H_{\\mathrm{TD}}=8$, TE 0.9, open-loop 25 steps, "
                         "no retrieval.\n")
        else:
            lines.append(f"\\subsection*{{Per-seed pair sets ({n} pairs per seed)}}")
            lines.append(f"All tables in this block: each seed draws its own {n} (start, goal) pairs; within a seed every "
                         "configuration sees the same pairs. Cells are success rates in \\%, mean $\\pm$ sd over the "
                         "seeds tested (per-seed counts and values are in the appendix). Default settings unless a row "
                         "says otherwise: lookahead 13.7 TDR units, $H_{\\mathrm{TD}}=8$, TE 0.9, open-loop 25 steps, "
                         "no retrieval.\n")

        # ---- 1. one table per algo (base objective x critic on/off) ----
        jd["algo"] = {}
        for key in sorted(by_algo, key=lambda k: (BASE_ORDER.index(k[0]), k[1])):
            body = []
            for m in sorted(by_algo[key], key=method_sort_key):
                cells = []
                for p in protos:
                    seed_dict = methods[m].get(p, {})
                    if not seed_dict:
                        cells.append("--")
                        continue
                    rates = [r["success_rate"] for r in seed_dict.values()]
                    mean, sd, ns = float(np.mean(rates)), float(np.std(rates)), len(rates)
                    cells.append(f"{mean:.1f} $\\pm$ {sd:.1f}")
                    jd["algo"].setdefault(m, {})[p] = dict(mean=mean, sd=sd, n_seeds=ns)
                body.append(variant_label(parse_tag(m)) + " & " + " & ".join(cells) + " \\\\")
            lines += table(f"{algo_title(key)}.", f"tab:gas-mpc-{key[0]}{'-crit' if key[1] else ''}-{conv}-n{n}",
                           labels, body, row_header="variant")

        # ---- 2. per-seed tables (appendix, collected for the end of the document) ----
        for p in protos:
            multi = {m: v[p] for m, v in methods.items() if p in v}
            if not multi:
                continue
            seeds = sorted({sd for v in multi.values() for sd in v})
            body = []
            for m in sorted(multi, key=method_sort_key):
                cells, rates, oks = [], [], []
                for sd in seeds:
                    r = multi[m].get(sd)
                    cells.append(f"{r['success_rate']:.0f}" if r else "--")
                    if r:
                        rates.append(r["success_rate"]); oks.append(np.array(r["first_hit_step"]) >= 0)
                pr = pooled(multi[m])
                if T:
                    ok = np.stack(oks)
                    tail = f" & {np.mean(rates):.1f} $\\pm$ {np.std(rates):.1f} & {int(ok.any(0).sum())} / {int(ok.all(0).sum())}"
                else:
                    tail = f" & {pr[0]:.1f}"
                body.append(tex_escape(m) + " & " + " & ".join(cells) + tail + " \\\\")
                jd.setdefault("pooled", {}).setdefault(p, {})[m] = dict(
                    per_seed={sd: multi[m][sd]["success_rate"] for sd in multi[m]}, hits=pr[1], total=pr[2],
                    pooled=pr[0], mean=float(np.mean(rates)), sd=float(np.std(rates)))
            if T:
                cap = f"Fixed task set ({T}, first {n}), {PROTO_LABEL[p]}, per seed (CEM repeats); `solved' = tasks solved in $\\ge$1 seed / in every seed run."
                cols = [f"s{sd}" for sd in seeds] + ["mean $\\pm$ sd", "solved"]
            else:
                cap = f"Per-seed pair sets ({n} pairs), {PROTO_LABEL[p]}, per seed, and pooled over the seeds each row has."
                cols = [f"s{sd}" for sd in seeds] + ["pooled"]
            appendix += table(cap, f"tab:gas-mpc-pooled-{p}-{conv}-n{n}", cols, body)

    if main_appendix:
        main_lines.append("\\clearpage")
        main_lines.append("\\subsection*{Per-seed results}")
        main_lines.append("Every test in the blocks above, broken down by seed.\n")
        main_lines += main_appendix
    if legacy_appendix:
        legacy_lines.append("\\clearpage")
        legacy_lines.append("\\subsection*{Per-seed results}")
        legacy_lines.append("Every test in the blocks above, broken down by seed.\n")
        legacy_lines += legacy_appendix

    (DOC / f"results_table{SUFFIX}.tex").write_text("\n".join(main_lines) if main_lines else "% no results yet\n")
    (DOC / f"results{SUFFIX}.json").write_text(json.dumps(out_json, indent=1))
    print("\n".join(main_lines) if main_lines else "no results yet")
    print(f"wrote {DOC / f'results_table{SUFFIX}.tex'}")


if __name__ == "__main__":
    main()
