"""Can a latent-space comparison replace the dataset's state column in the two places the
viability critic's TRAINING uses it (viability_train.py)?

  role 1  hindsight time-to-goal tau = first k <= delta at which the env's success predicate
          (state-based) holds along the logged trajectory t .. t+delta   (first_hit_time)
  role 2  cross-episode negative filter: keep (t, g) as a label-0 pair only when the two
          block poses are far apart (xneg_distance >= 100 px on Push-T)

For each role the candidate substitutes are a ball in raw latent L2 ||z_k - z_g|| and a ball
in TDR distance ||psi_k - psi_g||, with the radius chosen on a calibration half of the pairs
and every number reported on the other half.  The no-state baselines are tau = delta (role 1)
and no filter (role 2).  Ground truth is the state column, used here ONLY to score.

Writes outputs/<env>/latent_label_diag_s<seed>.json.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.envs import ENV_MECHANICS  # noqa: E402
from common.log_util import log  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
ENV = os.environ.get("GAS_MPC_ENV", "pusht")
MECH = ENV_MECHANICS[ENV]
OUT = Path(os.environ.get("GAS_MPC_OUT", ROOT / "outputs" / ENV))
SKIP = 5


def auroc(score_pos_is_high, label):
    """Rank-based AUROC; label bool (N,), score float (N,)."""
    from scipy import stats as sps
    label = np.asarray(label, bool)
    r = sps.rankdata(score_pos_is_high)
    n1, n0 = label.sum(), (~label).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((r[label].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


# ----------------------------------------------------------------------------------------
# role 1: hindsight tau
# ----------------------------------------------------------------------------------------

def sample_hindsight_pairs(ep_offset, ep_len, episodes, n, delta_max, rng):
    """(t, delta) with t + delta inside the episode; delta ~ U{0..min(delta_max, avail)}."""
    lens = ep_len[episodes]
    # episodes weighted by length, like a uniform frame draw
    e = rng.choice(episodes, size=n, p=lens / lens.sum())
    step = (rng.random(n) * ep_len[e]).astype(np.int64)
    avail = ep_len[e] - 1 - step
    dmax = np.minimum(avail, delta_max)
    delta = (rng.random(n) * (dmax + 1)).astype(np.int64)
    delta = np.minimum(delta, dmax)
    t = ep_offset[e] + step
    return t, delta


def segment_distances(z, t, delta, delta_max, metric):
    """(N, delta_max+1) distances d_k = ||f(z_{t+k}) - f(z_{t+delta})|| for k <= delta, +inf beyond."""
    n, L = len(t), delta_max + 1
    d = np.full((n, L), np.inf, np.float32)
    g = t + delta
    zg = z[g]
    for k in range(L):
        ok = delta >= k
        if not ok.any():
            break
        d[ok, k] = np.linalg.norm(z[t[ok] + k] - zg[ok], axis=1)
    return d


def state_tau(state, t, delta, delta_max):
    n, L = len(t), delta_max + 1
    g = t + delta
    hit = np.zeros((n, L), bool)
    for k in range(L):
        ok = delta >= k
        hit[ok, k] = MECH.goal_reached(state[t[ok] + k], state[g[ok]])
    return hit.argmax(1), hit  # argmax of first True; k = delta always hits


def latent_tau(d, r):
    """first k with d_k < r (d_delta = 0 so always <= delta)."""
    return (d < r).argmax(1)


def label_agreement(tau_a, tau_b, h_grid):
    """mean over pairs and budgets h of 1[(h >= tau_a) == (h >= tau_b)]."""
    la = h_grid[None, :] >= tau_a[:, None]
    lb = h_grid[None, :] >= tau_b[:, None]
    return float((la == lb).mean())


def role1(cache, psi, args, rng):
    ep_offset, ep_len = cache["ep_offset"], cache["ep_len"]
    n_ep = len(ep_len)
    eps = rng.choice(n_ep, size=min(args.episodes, n_ep), replace=False)
    t, delta = sample_hindsight_pairs(ep_offset, ep_len, eps, args.pairs, args.delta_max, rng)
    h_grid = np.arange(0, args.h_max + 1, SKIP)
    log(f"[role1] {len(t)} (t, delta) pairs from {len(eps)} episodes, delta_max={args.delta_max}")

    tau_s, hit = state_tau(cache["state"], t, delta, args.delta_max)
    matters = tau_s < delta
    log(f"[role1] predicate fires early (tau < delta) on {100 * matters.mean():.1f}% of pairs; "
        f"mean delta - tau = {(delta - tau_s).mean():.2f} steps; "
        f"among delta >= 25: {100 * matters[delta >= 25].mean():.1f}%, delta - tau = {(delta - tau_s)[delta >= 25].mean():.2f}")

    metrics = {"l2": segment_distances(cache["z"], t, delta, args.delta_max, "l2"),
               "tdr": segment_distances(psi, t, delta, args.delta_max, "tdr")}

    # per-frame separability: does distance to the goal frame predict the env predicate,
    # on non-trivial frames (k < delta)?
    kk = np.arange(args.delta_max + 1)[None, :]
    valid = kk < delta[:, None]
    out = {"n_pairs": int(len(t)), "n_episodes": int(len(eps)), "frac_tau_lt_delta": float(matters.mean()),
           "mean_delta_minus_tau": float((delta - tau_s).mean()), "h_grid": h_grid.tolist(),
           "baseline_tau_eq_delta": dict(label_agreement=label_agreement(delta, tau_s, h_grid),
                                         mean_abs_err=float(np.abs(delta - tau_s).mean()),
                                         exact=float((delta == tau_s).mean())),
           "metrics": {}}
    log(f"[role1] baseline tau=delta (no predicate): label agreement {out['baseline_tau_eq_delta']['label_agreement']:.4f}, "
        f"|err| {out['baseline_tau_eq_delta']['mean_abs_err']:.2f}")

    half = rng.random(len(t)) < 0.5   # calibration half
    for name, d in metrics.items():
        auc = auroc(-d[valid], hit[valid])
        # radius sweep on the calibration half, chosen by label agreement
        cand = np.quantile(d[valid], np.linspace(0.001, 0.5, 200))
        best_r, best_a = None, -1
        for r in cand:
            a = label_agreement(latent_tau(d[half], r), tau_s[half], h_grid)
            if a > best_a:
                best_a, best_r = a, float(r)
        te = ~half
        tau_l = latent_tau(d[te], best_r)
        err = tau_l - tau_s[te]
        rec = dict(auroc_frame_predicate=auc, radius=best_r, calib_label_agreement=best_a,
                   test_label_agreement=label_agreement(tau_l, tau_s[te], h_grid),
                   test_mean_abs_err=float(np.abs(err).mean()), test_exact=float((err == 0).mean()),
                   test_early_frac=float((err < 0).mean()), test_late_frac=float((err > 0).mean()),
                   test_label_agreement_when_matters=label_agreement(tau_l[matters[te]], tau_s[te][matters[te]], h_grid),
                   baseline_label_agreement_when_matters=label_agreement(delta[te][matters[te]], tau_s[te][matters[te]], h_grid))
        # by delta bucket
        by = {}
        for lo, hi in ((0, 10), (10, 25), (25, 40), (40, args.delta_max + 1)):
            m = (delta[te] >= lo) & (delta[te] < hi)
            if m.any():
                by[f"{lo}-{hi - 1}"] = dict(n=int(m.sum()), latent=label_agreement(tau_l[m], tau_s[te][m], h_grid),
                                            baseline=label_agreement(delta[te][m], tau_s[te][m], h_grid))
        rec["by_delta"] = by
        out["metrics"][name] = rec
        log(f"[role1] {name}: frame AUROC {auc:.3f} | r={best_r:.3f} | label agreement test {rec['test_label_agreement']:.4f} "
            f"(baseline {out['baseline_tau_eq_delta']['label_agreement']:.4f}) | when it matters: latent "
            f"{rec['test_label_agreement_when_matters']:.4f} vs baseline {rec['baseline_label_agreement_when_matters']:.4f} | "
            f"|err| {rec['test_mean_abs_err']:.2f} exact {rec['test_exact']:.3f} early {rec['test_early_frac']:.3f} "
            f"late {rec['test_late_frac']:.3f}")
        for k, v in by.items():
            log(f"[role1]   delta {k}: latent {v['latent']:.4f} baseline {v['baseline']:.4f} (n={v['n']})")
    return out


# ----------------------------------------------------------------------------------------
# role 2: cross-episode negative filter
# ----------------------------------------------------------------------------------------

def role2(cache, psi, args, rng):
    n = args.cross
    N = len(cache["z"])
    frame_ep = np.repeat(np.arange(len(cache["ep_len"])), cache["ep_len"])
    t = rng.integers(0, N, size=n * 2)
    g = rng.integers(0, N, size=n * 2)
    ok = frame_ep[t] != frame_ep[g]
    t, g = t[ok][:n], g[ok][:n]
    st, sg = cache["state"][t], cache["state"][g]
    import torch
    block_d = MECH.xneg_distance(torch.from_numpy(st), torch.from_numpy(sg)).numpy()
    keep_s = block_d >= args.xneg
    near20 = block_d < 20
    reached = MECH.goal_reached(st, sg)
    log(f"[role2] {n} cross-episode pairs: state filter keeps {100 * keep_s.mean():.1f}%; "
        f"block < 20 px on {100 * near20.mean():.2f}%; full predicate true on {100 * reached.mean():.3f}%")
    out = {"n_pairs": int(n), "state_keep_rate": float(keep_s.mean()), "frac_block_lt20": float(near20.mean()),
           "frac_predicate_true": float(reached.mean()),
           "no_filter": dict(kept_block_lt20=float(near20.mean()), kept_predicate_true=float(reached.mean()),
                             kept_block_lt_xneg=float((~keep_s).mean())),
           "metrics": {}}
    half = rng.random(n) < 0.5
    def pair_dist(X, chunk=20000):
        return np.concatenate([np.linalg.norm(X[t[i:i + chunk]] - X[g[i:i + chunk]], axis=1)
                               for i in range(0, n, chunk)])
    dists = {"l2": pair_dist(cache["z"]), "tdr": pair_dist(psi)}
    from scipy import stats as sps
    for name, d in dists.items():
        auc = auroc(d, keep_s)
        rho = float(sps.spearmanr(d, block_d)[0])
        # threshold matched to the state filter's keep rate on the calibration half
        thr = float(np.quantile(d[half], 1 - keep_s[half].mean()))
        te = ~half
        keep_l = d[te] >= thr
        rec = dict(auroc_keep=auc, spearman_vs_block_dist=rho, threshold=thr, test_keep_rate=float(keep_l.mean()),
                   precision=float(keep_s[te][keep_l].mean()),          # kept & truly far
                   recall=float(keep_l[keep_s[te]].mean()),
                   kept_block_lt_xneg=float((~keep_s[te])[keep_l].mean()),
                   kept_block_lt20=float(near20[te][keep_l].mean()),
                   kept_predicate_true=float(reached[te][keep_l].mean()),
                   kept_block_dist_p5=float(np.percentile(block_d[te][keep_l], 5)),
                   kept_block_dist_median=float(np.median(block_d[te][keep_l])))
        # stricter latent threshold: what keep rate is needed to push block<20 among kept to ~0?
        strict = {}
        for q in (0.5, 0.6, 0.7, 0.8):
            thr_q = float(np.quantile(d[half], 1 - q))
            kq = d[te] >= thr_q
            strict[f"keep{q:g}"] = dict(threshold=thr_q, kept_block_lt_xneg=float((~keep_s[te])[kq].mean()),
                                        kept_block_lt20=float(near20[te][kq].mean()),
                                        kept_predicate_true=float(reached[te][kq].mean()))
        rec["by_keep_rate"] = strict
        out["metrics"][name] = rec
        log(f"[role2] {name}: AUROC(keep) {auc:.3f} spearman(d, block_d) {rho:.3f} | at matched keep rate "
            f"{rec['test_keep_rate']:.3f}: precision {rec['precision']:.3f} recall {rec['recall']:.3f}, kept pairs with "
            f"block<{args.xneg:g}px {100 * rec['kept_block_lt_xneg']:.1f}% (no filter {100 * out['no_filter']['kept_block_lt_xneg']:.1f}%), "
            f"block<20px {100 * rec['kept_block_lt20']:.2f}% (no filter {100 * near20.mean():.2f}%), "
            f"predicate true {100 * rec['kept_predicate_true']:.3f}%")
        for k, v in strict.items():
            log(f"[role2]   {k}: block<{args.xneg:g} {100 * v['kept_block_lt_xneg']:.1f}%  block<20 {100 * v['kept_block_lt20']:.2f}%  "
                f"predicate {100 * v['kept_predicate_true']:.3f}%")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tdr-seed", type=int, default=0)
    p.add_argument("--episodes", type=int, default=4000, help="episodes sampled for role 1")
    p.add_argument("--pairs", type=int, default=60000, help="(t, delta) pairs for role 1")
    p.add_argument("--cross", type=int, default=300000, help="cross-episode pairs for role 2")
    p.add_argument("--delta-max", type=int, default=60)
    p.add_argument("--h-max", type=int, default=50)
    p.add_argument("--xneg", type=float, default=100.0)
    args = p.parse_args()
    rng = np.random.default_rng(args.seed)
    t0 = time.time()
    from gas_mpc_prepare import load_cache, ensure_psi
    cache = load_cache()
    psi = ensure_psi(args.tdr_seed)
    log(f"[load] z {cache['z'].shape} psi {psi.shape} in {time.time() - t0:.0f}s")
    res = dict(env=ENV, args=vars(args), role1_hindsight=role1(cache, psi, args, rng), role2_cross=role2(cache, psi, args, rng))
    path = OUT / f"latent_label_diag_s{args.seed}.json"
    path.write_text(json.dumps(res, indent=1))
    log(f"[done] {path} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
