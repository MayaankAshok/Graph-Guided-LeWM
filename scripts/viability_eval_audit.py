"""Score a trained viability critic against the privileged recovery-time oracle of an
Experiment-1/2 audit run (viability_inversion_audit.py, raw_offset{N}.npz).

The training script can only self-evaluate on trajectory-derived hindsight labels. This is
the external test: for every (task, candidate) the audit stores z_pred (what the planner
would score), z_true (the encoded executed endpoint), z_goal, and T = the oracle's minimum
recovery time, right-censored at oracle_cap. For each budget h on the critic's grid with
h <= cap the ground-truth label is 1[T <= h], and we report, for z_pred and z_true inputs:

    auroc / ece / brier   of V(z, z_goal, h) against 1[T <= h], pooled over tasks
    spearman(V, -T)       per task (ties in T are common at short offsets), mean over tasks
    selection             T of the candidate V picks (argmax at h = natural budget) vs the
                          one LeWM's c_pred picks (argmin) vs the best available
    oracle-best rank      percentile rank V assigns to the fastest-recovering candidate

Usage:
    python scripts/viability_eval_audit.py --critic outputs/pusht/critic_training/critic_v0/critic.pt \
        --audit outputs/pusht/experiments/viability_exp2 --goal-offsets 25 50
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import stats as sps

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.log_util import log
from common.viability import SKIP, HittingTimeHead, ViabilityCritic, auroc, ece

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def load_critic(path):
    """Either critic kind, by the checkpoint's `kind` field: the horizon-conditioned ensemble
    (viability_train.py) or the hitting-time head (viability_train_ht.py). Both expose
    prob(z, zg, h) -> (V, std) and a["h_max"] for the horizon grid."""
    ck = torch.load(path, map_location=DEV, weights_only=False)
    a = ck["args"]
    if ck.get("kind") == "hitting_time":
        critic = HittingTimeHead(**ck["head_kwargs"]).to(DEV)
        critic.load_state_dict(ck["head"])
    else:
        critic = ViabilityCritic(h_max=a["h_max"], n_members=a["members"], hidden=a["hidden"], depth=a["depth"]).to(DEV)
        critic.load_state_dict(ck["critic"])
    critic.eval()
    return critic, a


@torch.no_grad()
def score(critic, z, zg, h):
    """z (n, K, D), zg (n, D), scalar h -> mean/std over the ensemble, each (n, K)."""
    n, K, D = z.shape
    zt = torch.as_tensor(z.reshape(-1, D), device=DEV)
    gt = torch.as_tensor(np.repeat(zg, K, axis=0), device=DEV)
    ht = torch.full((n * K,), float(h), device=DEV)
    m, s = critic.prob(zt, gt, ht)
    return m.view(n, K).cpu().numpy(), s.view(n, K).cpu().numpy()


def evaluate_offset(critic, d, offset, h_grid):
    T, cap = d["oracle_min_steps"].astype(np.float64), float(d["oracle_cap"])
    c_pred = d["c_pred"]
    n, K = T.shape
    # the audit's 'natural budget': offset minus the 25-step candidate horizon (HORIZON*SKIP)
    natural = int(d["natural_budget"]) if "natural_budget" in d.files else max(0, offset - 25)
    out = {"n_tasks": int(n), "n_candidates": int(K), "oracle_cap": cap, "natural_budget": natural,
           "censored_frac": float((T > cap).mean()), "by_input": {}}
    for name in ("z_pred", "z_true"):
        res = {}
        for h in h_grid:
            if h > cap:
                continue                      # T > cap is 'not within cap', unknown beyond it
            V, S = score(critic, d[name], d["z_goal"], h)
            y = (T <= h).astype(np.float64)
            v, yy = V.ravel(), y.ravel()
            res[int(h)] = dict(
                auroc=auroc(v, yy), ece=ece(v, yy), brier=float(((v - yy) ** 2).mean()),
                pos_frac=float(yy.mean()), mean_v=float(v.mean()), mean_std=float(S.mean()),
            )
        # ranking against recovery time, per task. h=`natural` is the fair comparison point
        # (the recovery budget actually remaining after the candidate horizon; 0 when the
        # goal frame IS the candidate horizon, e.g. offset == candidate horizon); we ALSO
        # sweep the whole grid and report the best h, since a fixed h can be a poor match
        # to what the oracle's budget actually spans (see CLAUDE.md doubts on this script).
        h_nat = min(h_grid, key=lambda h: abs(h - min(natural, cap)))

        def _rank_stats(h):
            V, _ = score(critic, d[name], d["z_goal"], h)
            rhos, regret_v, best_rank = [], [], []
            for i in range(n):
                if np.ptp(T[i]) > 0:
                    r = sps.spearmanr(V[i], -T[i]).statistic
                    if np.isfinite(r):
                        rhos.append(r)
                regret_v.append(T[i][int(np.argmax(V[i]))] - T[i].min())
                best = int(np.argmin(T[i]))
                best_rank.append(float((V[i] > V[i][best]).mean()))   # 0 = V ranks it first
            return dict(spearman_v_vs_negT_mean=float(np.mean(rhos)) if rhos else float("nan"),
                       regret_steps_V=float(np.mean(regret_v)),
                       oracle_best_percentile_under_V=float(np.mean(best_rank)))

        by_h = {int(h): _rank_stats(h) for h in h_grid if h <= cap}
        # hitting-time head: also rank by its expected hitting time (no h needed; the CDF at
        # a single h -- especially h=0, an 'is this the goal node' query -- can be flat)
        if hasattr(critic, "expected_bins"):
            with torch.no_grad():
                nK, D = d[name].shape[0] * d[name].shape[1], d[name].shape[2]
                zt = torch.as_tensor(d[name].reshape(-1, D), device=DEV)
                gt = torch.as_tensor(np.repeat(d["z_goal"], K, axis=0), device=DEV)
                E = critic.expected_bins(zt, gt).view(n, K).cpu().numpy()
            rhos_e = [sps.spearmanr(-E[i], -T[i]).statistic for i in range(n) if np.ptp(T[i]) > 0]
            res["expected_hitting_time"] = dict(
                spearman_negE_vs_negT_mean=float(np.nanmean(rhos_e)),
                regret_steps=float(np.mean([T[i][int(np.argmin(E[i]))] - T[i].min() for i in range(n)])),
                oracle_best_percentile=float(np.mean([(E[i] < E[i][int(np.argmin(T[i]))]).mean() for i in range(n)])))
        best_h = max(by_h, key=lambda h: by_h[h]["spearman_v_vs_negT_mean"])
        l2_rhos = [sps.spearmanr(-c_pred[i], -T[i]).statistic for i in range(n) if np.ptp(T[i]) > 0]
        res["spearman_lewm_l2"] = float(np.nanmean(l2_rhos))
        regret_l2 = float(np.mean([T[i][int(np.argmin(c_pred[i]))] - T[i].min() for i in range(n)]))
        res["natural_h"] = int(h_nat)
        res["at_natural_h"] = by_h[h_nat]
        res["best_h"] = int(best_h)
        res["at_best_h"] = by_h[best_h]
        res["by_h"] = by_h
        res["regret_steps_lewm_l2"] = regret_l2
        out["by_input"][name] = res
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--critic", required=True)
    p.add_argument("--audit", required=True, help="directory holding raw_offset{N}.npz")
    p.add_argument("--goal-offsets", type=int, nargs="+", default=[25, 50])
    p.add_argument("--out", default=None)
    args = p.parse_args()
    critic, cargs = load_critic(args.critic)
    h_grid = list(range(0, cargs["h_max"] + 1, SKIP))
    results = {"critic": args.critic, "critic_args": cargs, "offsets": {}}
    for off in args.goal_offsets:
        path = Path(args.audit) / f"raw_offset{off}.npz"
        if not path.exists():
            log(f"[skip] {path} not found")
            continue
        d = np.load(path)
        r = evaluate_offset(critic, d, off, h_grid)
        results["offsets"][str(off)] = r
        for name, res in r["by_input"].items():
            hs = [h for h in h_grid if h in res]
            line = " ".join(f"h{h}:auroc={res[h]['auroc']:.2f}/ece={res[h]['ece']:.2f}" for h in hs)
            log(f"[offset={off}] {name}: {line}")
            nat, best = res["at_natural_h"], res["at_best_h"]
            if "expected_hitting_time" in res:
                e = res["expected_hitting_time"]
                log(f"[offset={off}] {name}: expected hitting time: spearman(-E,-T)={e['spearman_negE_vs_negT_mean']:.3f} "
                    f"regret={e['regret_steps']:.1f} rank={e['oracle_best_percentile']:.2f}")
            log(f"[offset={off}] {name}: at natural h={res['natural_h']}: spearman(V,-T)={nat['spearman_v_vs_negT_mean']:.3f} "
                f"regret={nat['regret_steps_V']:.1f} rank={nat['oracle_best_percentile_under_V']:.2f} | "
                f"at best h={res['best_h']}: spearman={best['spearman_v_vs_negT_mean']:.3f} regret={best['regret_steps_V']:.1f} | "
                f"LeWM L2: spearman={res['spearman_lewm_l2']:.3f} regret={res['regret_steps_lewm_l2']:.1f}")
    out = Path(args.out or Path(args.critic).parent / "eval_audit.json")
    out.write_text(json.dumps(results, indent=2))
    log(f"[done] wrote {out}")


if __name__ == "__main__":
    main()
