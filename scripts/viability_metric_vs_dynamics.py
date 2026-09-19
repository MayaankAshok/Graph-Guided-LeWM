"""Experiment 2 Run B: separate metric failure from dynamics failure.

Reads the artifact-complete Run A output of scripts/viability_inversion_audit.py and, for
every offset, compares two terminal costs against the shared recovery-time oracle T:

    c_pred = ||z_pred - z_g||^2   LeWM's deployed cost (predicted endpoint)
    c_true = ||z_true - z_g||^2   the identical metric at the endpoint actually reached

Both costs rank the SAME candidate pool against the SAME oracle labels, so their paired
difference is the loss attributable to endpoint prediction ("dynamics penalty"), and
whatever disagreement c_true still shows is the terminal metric's own failure. A third,
privileged column `state_dist` (raw 7-dim state distance to the goal) is reported as a
control: it is not a latent cost, it just shows what an endpoint-only Euclidean metric can
do at best with the correct endpoint.

Reports (all per offset, aggregated over tasks with task-bootstrap 95% CIs):
    ranking quality    per-task Spearman(cost, T)
    inversions         pairwise closer-but-slower rate and mean recovery-step gap
    selection regret   T[argmin cost] - min T, over tasks with a recoverable candidate
    oracle-best rank   percentile the cost assigns to the fastest-recovering candidate
    dynamics penalty   paired c_pred - c_true difference for each of the above
    coverage           censored fraction, informative tasks, tied-label fraction, slot-0
                       oracle validation, and the stronger-oracle (Run C) subset results
    strata             all of the above inside model-error quartiles (global thresholds
                       fitted on the run) plus the lowest-error decile

Figures: fig_strata.png (inversion rate and rank agreement vs model-error stratum, paired
curves) and fig_gallery.png (low-error candidates where both latent costs prefer the
slower-to-recover endpoint, rendered from the simulator).

    python scripts/viability_metric_vs_dynamics.py --out outputs/pusht/experiments/viability_exp2 --goal-offsets 25 50
    python scripts/viability_metric_vs_dynamics.py --out outputs/pusht/experiments/viability_exp2 --select-recheck  # -> Run C
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats as sps

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.log_util import log

COSTS = ("c_pred", "c_true", "state_dist")
LATENT_COSTS = ("c_pred", "c_true")
STRATA = ("Q1", "Q2", "Q3", "Q4", "D1")
RECHECK_GROUPS = ("low_error_inversion", "high_error_inversion", "non_inversion")


# ------------------------------------------------------------------
# loading
# ------------------------------------------------------------------

def load_run(out_dir, offset):
    raw = np.load(out_dir / f"raw_offset{offset}.npz")
    if "c_true" not in raw.files:
        raise SystemExit(f"{out_dir}/raw_offset{offset}.npz predates the Experiment-2 save format "
                         "(no c_true); rerun scripts/viability_inversion_audit.py")
    T = raw["oracle_min_steps"].astype(np.int64)
    cap = int(raw["oracle_cap"])
    return dict(
        c_pred=raw["c_pred"].astype(np.float64), c_true=raw["c_true"].astype(np.float64),
        state_dist=raw["endpoint_dist"].astype(np.float64), e_model=raw["e_model"].astype(np.float64),
        T=T, cap=cap, censored=T > cap, final_states=raw["final_states"], goal_state=raw["goal_state"],
        natural_budget=max(0, offset - raw["candidate_actions"].shape[2]),
    )


# ------------------------------------------------------------------
# per-task statistics
# ------------------------------------------------------------------

def inverted_pairs(cost, T):
    """(K,K) mask: [a,b] true iff cost ranks a strictly closer but the oracle recovers a slower."""
    return (cost[:, None] < cost[None, :]) & (T[:, None] > T[None, :])


def task_stats(cost, T, member=None):
    """Statistics for one task's candidate pool, optionally restricted to a stratum `member`.

    Pairwise quantities use pairs with both members inside the stratum; the restricted
    Spearman needs >= 3 members with a non-tied oracle label. NaN marks 'no data'.
    """
    K = len(cost)
    member = np.ones(K, bool) if member is None else member
    both = member[:, None] & member[None, :]
    left = (cost[:, None] < cost[None, :]) & both
    inv = inverted_pairs(cost, T) & both
    n_left = int(left.sum())
    out = dict(inv_rate=inv.sum() / n_left if n_left else np.nan, n_pairs=n_left,
               gaps=(T[:, None] - T[None, :])[inv].tolist())
    sub_c, sub_T = cost[member], T[member]
    out["spearman"] = (float(sps.spearmanr(sub_c, sub_T).statistic)
                       if member.sum() >= 3 and np.ptp(sub_T) > 0 else np.nan)
    return out


def selection_stats(cost, T, cap):
    """Regret of the cost's pick and the percentile it assigns to the oracle-best candidate."""
    K = len(cost)
    pick = int(np.argmin(cost))
    recoverable = T.min() <= cap
    pct = (sps.rankdata(cost, method="average") - 1) / max(K - 1, 1)
    return dict(
        regret=float(T[pick] - T.min()) if recoverable else np.nan,  # censored pick -> lower bound
        pick_censored=bool(T[pick] > cap) if recoverable else np.nan,
        # best-ranked among candidates tied at the fastest recovery (0 = ranked first)
        best_pct=float(pct[T == T.min()].min()) if np.ptp(T) > 0 else np.nan,
    )


# ------------------------------------------------------------------
# aggregation
# ------------------------------------------------------------------

def boot_ci(values, n=5000, seed=0):
    """Task-bootstrap 95% CI of the nan-mean; tasks (not candidate pairs) are the unit."""
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if len(v) < 2:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), size=(n, len(v)))].mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def agg(values, seed=0):
    v = np.asarray(values, float)
    ok = np.isfinite(v)
    return dict(mean=float(v[ok].mean()) if ok.any() else float("nan"), ci95=boot_ci(v, seed=seed),
                n_tasks=int(ok.sum()))


def paired(a, b, seed=0):
    """Paired (a - b) over tasks where both are defined."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    return agg(np.where(ok, a - b, np.nan), seed=seed)


def stratum_masks(e_model):
    """Global model-error thresholds fitted on the whole run (all tasks x candidates)."""
    edges = np.quantile(e_model, [0.25, 0.5, 0.75])
    d1 = np.quantile(e_model, 0.10)
    masks = {
        "Q1": e_model <= edges[0],
        "Q2": (e_model > edges[0]) & (e_model <= edges[1]),
        "Q3": (e_model > edges[1]) & (e_model <= edges[2]),
        "Q4": e_model > edges[2],
        "D1": e_model <= d1,
    }
    return masks, dict(q25=float(edges[0]), q50=float(edges[1]), q75=float(edges[2]), d10=float(d1))


def pooled_rank_spearman(cost, T, member):
    """Rank within each task first, then pool stratum members across tasks (the brief's
    'pool candidates only after computing per-task rankings')."""
    rc = np.stack([sps.rankdata(c) for c in cost]) / cost.shape[1]
    rT = np.stack([sps.rankdata(t) for t in T]) / T.shape[1]
    x, y = rc[member], rT[member]
    if len(x) < 3 or np.ptp(y) == 0:
        return float("nan"), int(len(x))
    return float(sps.spearmanr(x, y).statistic), int(len(x))


def analyse_cost(cost, run, member=None):
    """Aggregate one cost over tasks (optionally inside a stratum)."""
    T, cap = run["T"], run["cap"]
    n = len(cost)
    rows = [task_stats(cost[i], T[i], None if member is None else member[i]) for i in range(n)]
    out = dict(
        spearman=agg([r["spearman"] for r in rows]),
        inv_rate=agg([r["inv_rate"] for r in rows]),
        gap_mean=float(np.mean(sum((r["gaps"] for r in rows), []))) if any(r["gaps"] for r in rows) else float("nan"),
        n_pairs=int(sum(r["n_pairs"] for r in rows)),
        _per_task=dict(spearman=[r["spearman"] for r in rows], inv_rate=[r["inv_rate"] for r in rows]),
    )
    if member is None:
        sel = [selection_stats(cost[i], T[i], cap) for i in range(n)]
        out.update(
            regret=agg([s["regret"] for s in sel]),
            pick_censored_frac=float(np.nanmean([s["pick_censored"] for s in sel if s["pick_censored"] is not np.nan])),
            best_pct=agg([s["best_pct"] for s in sel]),
        )
        out["_per_task"].update(regret=[s["regret"] for s in sel], best_pct=[s["best_pct"] for s in sel])
    else:
        rho, m = pooled_rank_spearman(cost, T, member)
        out.update(pooled_rank_spearman=rho, n_members=m)
    return out


def coverage(run):
    T, cap, censored = run["T"], run["cap"], run["censored"]
    n, K = T.shape
    iu = np.triu_indices(K, 1)
    tied = np.array([(T[i][:, None] == T[i][None, :])[iu].mean() for i in range(n)])
    unc = ~censored
    tied_unc = []
    for i in range(n):
        m = unc[i][:, None] & unc[i][None, :]
        if m[iu].any():
            tied_unc.append(float((T[i][:, None] == T[i][None, :])[iu][m[iu]].mean()))
    # censored endpoints whose block already sits within ~2x the success tolerance of the
    # goal are almost certainly oracle false negatives (fine placement defeats a CEM that
    # starts from full-range random actions), so their share is reported explicitly
    fs, gs = run["final_states"], run["goal_state"]
    pos = np.linalg.norm(fs[:, :, :4] - gs[:, None, :4], axis=-1)
    ang = np.abs(fs[:, :, 4] - gs[:, None, 4])
    near_goal = (pos < 40) & (np.minimum(ang, 2 * np.pi - ang) < np.pi / 6)
    return dict(
        n_tasks=int(n), n_candidates=int(K), oracle_cap=cap,
        unreached_fraction=float(censored.mean()),
        near_goal_endpoint_fraction=float(near_goal.mean()),
        near_goal_censored_fraction=float(censored[near_goal].mean()) if near_goal.any() else float("nan"),
        fully_censored_tasks=int(censored.all(axis=1).sum()),
        n_informative_tasks=int(sum(np.ptp(T[i]) > 0 for i in range(n))),
        tied_label_pair_fraction=float(tied.mean()),
        tied_label_pair_fraction_uncensored=float(np.mean(tied_unc)) if tied_unc else float("nan"),
        natural_recovery_budget=int(run["natural_budget"]),
        viable_candidate_fraction=float((T <= run["natural_budget"]).mean()),
        slot0_recovered_within_natural_budget=float((T[:, 0] <= run["natural_budget"]).mean()),
        slot0_censored_fraction=float(censored[:, 0].mean()),
        median_T=float(np.median(T)), median_T_uncensored=float(np.median(T[unc])) if unc.any() else float("nan"),
    )


# ------------------------------------------------------------------
# Run C: subset selection and integration
# ------------------------------------------------------------------

def slow_members(cost, T):
    """(n,K) mask of candidates that are the closer-but-slower member of >= 1 inverted pair."""
    return np.stack([inverted_pairs(cost[i], T[i]).any(axis=1) for i in range(len(cost))])


def any_member(cost, T):
    return np.stack([(lambda p: p.any(axis=1) | p.any(axis=0))(inverted_pairs(cost[i], T[i]))
                     for i in range(len(cost))])


def select_recheck_subset(run, masks, n_per_group, seed):
    slow = slow_members(run["c_pred"], run["T"]) | slow_members(run["c_true"], run["T"])
    involved = any_member(run["c_pred"], run["T"]) | any_member(run["c_true"], run["T"])
    groups = {
        "low_error_inversion": slow & masks["Q1"],
        "high_error_inversion": slow & masks["Q4"],
        "non_inversion": ~involved,
    }
    rng = np.random.default_rng(seed)
    items = []
    for g, m in groups.items():
        idx = np.argwhere(m)
        take = idx[rng.permutation(len(idx))[:n_per_group]] if len(idx) else idx
        items += [dict(task=int(t), cand=int(c), group=g) for t, c in take]
    return dict(seed=int(seed), groups=list(groups), n_available={g: int(m.sum()) for g, m in groups.items()},
                items=items)


def recheck_report(run, masks, out_dir, offset):
    path = out_dir / f"recheck_offset{offset}.npz"
    if not path.exists():
        return None
    rc = np.load(path)
    task, cand, group = rc["task"], rc["cand"], rc["group"].astype(str)
    old, new, cap = rc["old_steps"].astype(np.int64), rc["new_steps"].astype(np.int64), int(rc["oracle_cap"])
    T_upd = run["T"].copy()
    T_upd[task, cand] = np.minimum(T_upd[task, cand], new[:])
    report = dict(oracle=f"{int(rc['oracle_trials'])}x{int(rc['oracle_npop'])}x{int(rc['oracle_niter'])}", groups={})
    for g in RECHECK_GROUPS:
        m = group == g
        if not m.any():
            continue
        drop = (old - new)[m]
        report["groups"][g] = dict(
            n=int(m.sum()), improved_fraction=float((new[m] < old[m]).mean()),
            mean_drop_when_improved=float(drop[drop > 0].mean()) if (drop > 0).any() else 0.0,
            uncensored_fraction=float(((old[m] > cap) & (new[m] <= cap)).mean()),
            censored_before=float((old[m] > cap).mean()), censored_after=float((new[m] > cap).mean()),
        )
        # inversion survival: pairs where this rechecked SLOW member was ranked closer but
        # recovered slower; still inverted iff its stronger-oracle time still exceeds the
        # fast member's Run-A time
        for cost_name in LATENT_COSTS:
            cost = run[cost_name]
            before = after = 0
            for t, c, tn in zip(task[m], cand[m], new[m]):
                pairs = inverted_pairs(cost[t], run["T"][t])[c]
                before += int(pairs.sum())
                after += int((pairs & (tn > run["T"][t])).sum())
            report["groups"][g][f"{cost_name}_inversion_survival"] = (after / before if before else float("nan"))
            report["groups"][g][f"{cost_name}_n_inverted_pairs"] = before
    # whole-run low-error inversion rate with the rechecked times substituted (partial
    # coverage, so this bounds how much a stronger oracle could move the headline number)
    run_upd = dict(run, T=T_upd, censored=T_upd > cap)
    report["Q1_inversion_rate_with_recheck"] = {
        c: analyse_cost(run[c], run_upd, masks["Q1"])["inv_rate"]["mean"] for c in LATENT_COSTS}
    report["overall_inversion_rate_with_recheck"] = {
        c: analyse_cost(run[c], run_upd)["inv_rate"]["mean"] for c in LATENT_COSTS}
    return report


# ------------------------------------------------------------------
# figures
# ------------------------------------------------------------------

def fig_strata(analysis, offsets, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    quart = ("Q1", "Q2", "Q3", "Q4")
    fig, axes = plt.subplots(len(offsets), 3, figsize=(13, 3.6 * len(offsets)), squeeze=False)
    style = {"c_pred": dict(color="tab:red", marker="o", label="$C^{pred}$ (predicted endpoint)"),
             "c_true": dict(color="tab:blue", marker="s", label="$C^{true}$ (true endpoint)"),
             "state_dist": dict(color="gray", marker="^", ls="--", label="state distance (control)")}
    x = np.arange(len(quart))
    for r, off in enumerate(offsets):
        strata = analysis[str(off)]["strata"]
        for c, ax_key, ttl in ((0, "inv_rate", "pairwise inversion rate"),
                               (1, "spearman", "per-task Spearman(cost, T) within stratum"),
                               (2, "pooled_rank_spearman", "pooled within-task-rank Spearman")):
            ax = axes[r, c]
            for cost in COSTS:
                if ax_key == "pooled_rank_spearman":
                    y = np.array([strata[q][cost][ax_key] for q in quart])
                    ax.plot(x, y, **style[cost])
                else:
                    y = np.array([strata[q][cost][ax_key]["mean"] for q in quart])
                    ci = np.array([strata[q][cost][ax_key]["ci95"] for q in quart])
                    ax.errorbar(x, y, yerr=np.abs(ci.T - y), capsize=3, **style[cost])
            ax.set_xticks(x)
            ax.set_xticklabels([f"{q}\n(e<={strata[q]['edge']:.2f})" if q != "Q4" else f"{q}\n(e>{strata['Q3']['edge']:.2f})"
                                for q in quart], fontsize=8)
            ax.set_title(f"offset {off}: {ttl}", fontsize=10)
            ax.grid(alpha=0.3)
            if c == 0:
                ax.set_ylabel("closer-but-slower pair fraction")
                ax.legend(fontsize=8)
            if r == len(offsets) - 1:
                ax.set_xlabel("model-error quartile (global thresholds)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def find_gallery_pairs(run, masks, n_examples):
    """Low-error pairs where BOTH latent costs prefer the slower-recovering endpoint and BOTH
    endpoints are recoverable (uncensored) -- a censored slow member would dominate the gap
    and mostly shows oracle censoring, not the metric; one pair per task, largest gap first."""
    T, cap = run["T"], run["cap"]
    found = []
    for i in range(len(T)):
        both = inverted_pairs(run["c_pred"][i], T[i]) & inverted_pairs(run["c_true"][i], T[i])
        both &= masks["Q1"][i][:, None] & masks["Q1"][i][None, :] & (T[i] <= cap)[:, None] & (T[i] <= cap)[None, :]
        if not both.any():
            continue
        gap = np.where(both, T[i][:, None] - T[i][None, :], -1)
        a, b = np.unravel_index(int(np.argmax(gap)), gap.shape)
        found.append((int(gap[a, b]), i, int(a), int(b)))
    found.sort(reverse=True)
    return found[:n_examples]


def fig_gallery(runs, masks_by_off, offsets, path, n_examples, cap_note):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from common.envs import ENV_MECHANICS

    mech = ENV_MECHANICS["pusht"]
    env = mech.make_env()

    def render(state, goal):
        env.reset(seed=0, options=mech.reset_options(state, goal))
        return env.render()

    rows = []
    for off in offsets:
        for gap, i, a, b in find_gallery_pairs(runs[off], masks_by_off[off], n_examples):
            rows.append((off, i, a, b))
    if not rows:
        log("[gallery] no low-error pairs where both latent costs prefer the slower endpoint")
        env.close()
        return
    fig, axes = plt.subplots(len(rows), 3, figsize=(9.5, 3.2 * len(rows)), squeeze=False)
    for r, (off, i, a, b) in enumerate(rows):
        run = runs[off]
        goal = run["goal_state"][i]
        panels = [(run["final_states"][i][a], f"A: preferred by both costs\n"
                   f"Cpred={run['c_pred'][i][a]:.1f} Ctrue={run['c_true'][i][a]:.1f}\n"
                   f"T={_fmt_T(run['T'][i][a], run['cap'])}  e_model={run['e_model'][i][a]:.2f}"),
                  (run["final_states"][i][b], f"B: ranked farther by both\n"
                   f"Cpred={run['c_pred'][i][b]:.1f} Ctrue={run['c_true'][i][b]:.1f}\n"
                   f"T={_fmt_T(run['T'][i][b], run['cap'])}  e_model={run['e_model'][i][b]:.2f}"),
                  (goal, f"goal (offset {off}, task {i})")]
        for c, (state, title) in enumerate(panels):
            ax = axes[r, c]
            ax.imshow(render(state, goal))
            ax.set_title(title, fontsize=8)
            ax.axis("off")
    fig.suptitle("Accurate prediction, wrong preference: both latent costs rank A closer, oracle recovers B faster"
                 + cap_note, fontsize=9, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(path, dpi=130)
    plt.close(fig)
    env.close()


def _fmt_T(t, cap):
    return f">{cap} (censored)" if t > cap else str(int(t))


# ------------------------------------------------------------------
# main
# ------------------------------------------------------------------

def analyse_offset(run, out_dir, offset):
    masks, edges = stratum_masks(run["e_model"])
    overall = {c: analyse_cost(run[c], run) for c in COSTS}
    per = {c: overall[c].pop("_per_task") for c in COSTS}
    penalty = {k: paired(per["c_pred"][k], per["c_true"][k]) for k in ("spearman", "inv_rate", "regret", "best_pct")}
    strata = {}
    for q in STRATA:
        strata[q] = {c: analyse_cost(run[c], run, masks[q]) for c in COSTS}
        for c in COSTS:
            strata[q][c].pop("_per_task")
        strata[q]["edge"] = {"Q1": edges["q25"], "Q2": edges["q50"], "Q3": edges["q75"], "Q4": float("inf"),
                             "D1": edges["d10"]}[q]
        strata[q]["penalty_inv_rate"] = paired(
            [task_stats(run["c_pred"][i], run["T"][i], masks[q][i])["inv_rate"] for i in range(len(run["T"]))],
            [task_stats(run["c_true"][i], run["T"][i], masks[q][i])["inv_rate"] for i in range(len(run["T"]))])
        strata[q]["tied_label_fraction"] = _tied_within(run["T"], masks[q])
    return dict(overall=overall, dynamics_penalty=penalty, coverage=coverage(run), model_error_edges=edges,
                strata=strata, recheck=recheck_report(run, masks, out_dir, offset)), masks


def _tied_within(T, member):
    ties, tot = 0, 0
    for i in range(len(T)):
        m = member[i]
        if m.sum() < 2:
            continue
        sub = T[i][m]
        eq = (sub[:, None] == sub[None, :])[np.triu_indices(len(sub), 1)]
        ties += int(eq.sum())
        tot += len(eq)
    return ties / tot if tot else float("nan")


def print_report(analysis, offsets):
    def f(x):
        return "nan" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:+.3f}"

    def ci(a):
        return f"{a['mean']:+.3f} [{a['ci95'][0]:+.3f},{a['ci95'][1]:+.3f}] (n={a['n_tasks']})"

    for off in offsets:
        A = analysis[str(off)]
        cov = A["coverage"]
        log(f"\n===== offset {off}: {cov['n_tasks']} tasks x {cov['n_candidates']} candidates, oracle cap {cov['oracle_cap']} =====")
        log(f"coverage: censored={cov['unreached_fraction']:.3f} (near-goal endpoints {cov['near_goal_endpoint_fraction']:.3f}, "
            f"of which censored {cov['near_goal_censored_fraction']:.3f}) fully-censored tasks={cov['fully_censored_tasks']} "
            f"informative tasks={cov['n_informative_tasks']} tied-label pairs={cov['tied_label_pair_fraction']:.3f} "
            f"(uncensored-only {cov['tied_label_pair_fraction_uncensored']:.3f})")
        log(f"oracle check: slot-0 (dataset continuation) recovered within natural budget "
            f"{cov['natural_recovery_budget']}: {cov['slot0_recovered_within_natural_budget']:.2f}; "
            f"slot-0 censored {cov['slot0_censored_fraction']:.2f}; median T={cov['median_T']:.0f}")
        log(f"{'cost':>11} {'spearman(cost,T)':>36} {'inversion rate':>36} {'gap':>6} {'regret (steps)':>36} {'oracle-best pct':>36}")
        for c in COSTS:
            o = A["overall"][c]
            log(f"{c:>11} {ci(o['spearman']):>36} {ci(o['inv_rate']):>36} {o['gap_mean']:>6.1f} "
                f"{ci(o['regret']):>36} {ci(o['best_pct']):>36}")
        p = A["dynamics_penalty"]
        log(f"dynamics penalty (c_pred - c_true, paired over tasks): spearman {ci(p['spearman'])} | "
            f"inv_rate {ci(p['inv_rate'])} | regret {ci(p['regret'])} | best_pct {ci(p['best_pct'])}")
        log(f"model-error strata (global thresholds q25={A['model_error_edges']['q25']:.2f} "
            f"q50={A['model_error_edges']['q50']:.2f} q75={A['model_error_edges']['q75']:.2f} d10={A['model_error_edges']['d10']:.2f})")
        log(f"{'stratum':>8} {'cost':>11} {'inv_rate':>30} {'gap':>6} {'spearman|stratum':>30} {'pooled-rank rho':>16} {'members':>8} {'tied':>6}")
        for q in STRATA:
            S = A["strata"][q]
            for c in COSTS:
                s = S[c]
                log(f"{q:>8} {c:>11} {ci(s['inv_rate']):>30} {s['gap_mean']:>6.1f} {ci(s['spearman']):>30} "
                    f"{f(s['pooled_rank_spearman']):>16} {s['n_members']:>8} {S['tied_label_fraction']:>6.2f}")
            log(f"{q:>8} {'penalty':>11} {ci(S['penalty_inv_rate']):>30}")
        R = A.get("recheck")
        if R:
            log(f"stronger-oracle recheck ({R['oracle']}):")
            for g, r in R["groups"].items():
                log(f"  {g:>21}: n={r['n']} improved={r['improved_fraction']:.2f} "
                    f"mean_drop={r['mean_drop_when_improved']:.1f} censored {r['censored_before']:.2f}->{r['censored_after']:.2f} | "
                    f"inversion survival c_pred={f(r.get('c_pred_inversion_survival'))} "
                    f"c_true={f(r.get('c_true_inversion_survival'))}")
            log(f"  Q1 inversion rate with rechecked T substituted (mixed oracle strengths, indicative only): "
                + ", ".join(f"{c}={v:.3f}" for c, v in R["Q1_inversion_rate_with_recheck"].items()))
        else:
            log("stronger-oracle recheck: not run yet (use --select-recheck, then scripts/viability_oracle_recheck.py)")


def _jsonable(o):
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True, help="Run A output dir")
    p.add_argument("--goal-offsets", type=int, nargs="+", default=[25, 50])
    p.add_argument("--select-recheck", action="store_true",
                   help="write recheck_subset_offset{N}.json for scripts/viability_oracle_recheck.py")
    p.add_argument("--recheck-per-group", type=int, default=40)
    p.add_argument("--recheck-seed", type=int, default=23)
    p.add_argument("--no-figures", action="store_true")
    p.add_argument("--gallery-examples", type=int, default=3, help="per offset")
    args = p.parse_args()
    out_dir = Path(args.out).resolve()

    runs, masks_by_off, analysis = {}, {}, {}
    for off in args.goal_offsets:
        runs[off] = load_run(out_dir, off)
        analysis[str(off)], masks_by_off[off] = analyse_offset(runs[off], out_dir, off)
        if args.select_recheck:
            subset = select_recheck_subset(runs[off], masks_by_off[off], args.recheck_per_group, args.recheck_seed)
            (out_dir / f"recheck_subset_offset{off}.json").write_text(json.dumps(subset, indent=1))
            log(f"[offset={off}] recheck subset: {len(subset['items'])} endpoints "
                f"(available {subset['n_available']}) -> recheck_subset_offset{off}.json")

    print_report(analysis, args.goal_offsets)
    (out_dir / "analysis.json").write_text(json.dumps(_jsonable(analysis), indent=2))
    log(f"\n[done] wrote {out_dir / 'analysis.json'}")
    if not args.no_figures:
        fig_strata(analysis, args.goal_offsets, out_dir / "fig_strata.png")
        log(f"[done] wrote {out_dir / 'fig_strata.png'}")
        try:
            fig_gallery(runs, masks_by_off, args.goal_offsets, out_dir / "fig_gallery.png",
                        args.gallery_examples, "")
            log(f"[done] wrote {out_dir / 'fig_gallery.png'}")
        except Exception as e:  # rendering needs the simulator; the numbers above do not
            log(f"[gallery] skipped: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
