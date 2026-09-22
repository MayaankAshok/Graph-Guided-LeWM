"""Compare the legacy budget-conditioned critic with one-pass hitting-time heads.

The audit uses only the label bank's held-out episodes. It reports CDF quality through 225
steps and, most importantly for CEM, rank correlation between each critic's expected-time
cost and graph hitting time on both logged and predictor-imagined endpoints.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from scipy import stats as sps

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.lewm_loader import load_lewm
from common.viability import SKIP, auroc, ece
from viability_eval_audit import load_critic
from viability_train import LatentCache
from viability_train_ht import Bank, Sampler

DEV = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def infer(critic, z, zg, horizons, batch=4096):
    rows = {"cdf": {h: [] for h in horizons}, "cost": [], "seconds": 0.0,
            "forward_calls_per_batch": 1 if hasattr(critic, "restricted_expected_steps") else len(range(0, critic.h_max + 1, SKIP))}
    if DEV == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for lo in range(0, len(z), batch):
        a = torch.as_tensor(z[lo:lo + batch], device=DEV)
        g = torch.as_tensor(zg[lo:lo + batch], device=DEV)
        for h in horizons:
            p, _ = critic.prob(a, g, torch.full((len(a),), float(h), device=DEV))
            rows["cdf"][h].append(p.cpu().numpy())
        if hasattr(critic, "restricted_expected_steps"):
            cost = critic.restricted_expected_steps(a, g, critic.h_max)
        else:
            cost = torch.zeros(len(a), device=DEV)
            for h in range(0, critic.h_max + 1, SKIP):
                p, _ = critic.prob(a, g, torch.full((len(a),), float(h), device=DEV))
                cost += SKIP * (1 - p)
        rows["cost"].append(cost.cpu().numpy())
    if DEV == "cuda":
        torch.cuda.synchronize()
    rows["seconds"] = time.time() - t0
    rows["cost"] = np.concatenate(rows["cost"])
    rows["cdf"] = {h: np.concatenate(v) for h, v in rows["cdf"].items()}
    return rows


def metrics(pred, tg, goal, horizons):
    exact = tg >= 0
    long = exact & (tg > 50)
    out = {"n": int(len(tg)), "n_exact": int(exact.sum()), "n_exact_gt50": int(long.sum()),
           "seconds": pred["seconds"], "forward_calls_per_batch": pred["forward_calls_per_batch"], "cdf": {}}
    for h in horizons:
        y = ((tg >= 0) & (tg <= h)).astype(float)
        p = pred["cdf"][h]
        out["cdf"][str(h)] = {"auroc": auroc(p, y), "brier": float(np.mean((p - y) ** 2)), "ece": ece(p, y)}
    for name, mask in (("all_exact", exact), ("gt50_exact", long)):
        out[name] = {"spearman_cost_vs_TG": float(sps.spearmanr(pred["cost"][mask], tg[mask]).statistic)}
    per_goal = []
    for g in np.unique(goal[exact]):
        m = exact & (goal == g)
        if m.sum() >= 3 and np.ptp(tg[m]) > 0:
            r = sps.spearmanr(pred["cost"][m], tg[m]).statistic
            if np.isfinite(r):
                per_goal.append(r)
    out["per_goal_spearman_mean"] = float(np.mean(per_goal)) if per_goal else float("nan")
    out["n_informative_goals"] = len(per_goal)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--critics", nargs="+", required=True, help="name=checkpoint")
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-n", type=int, default=20000)
    p.add_argument("--pred-rollouts", type=int, default=4000)
    args = p.parse_args()

    cache = LatentCache(args.cache, DEV)
    bank = Bank(args.labels, 1, DEV, 45)
    rng = torch.Generator(device=DEV); rng.manual_seed(args.seed)
    idx = torch.randint(0, bank.n, (min(args.eval_n, bank.n),), device=DEV, generator=rng)
    logged = (cache.z[bank.start[idx]].cpu().numpy(), cache.z[bank.goal[idx]].cpu().numpy(),
              bank.tg[idx].cpu().numpy(), bank.goal[idx].cpu().numpy())

    model = load_lewm(Path(cache.meta["ckpt_dir"]), device=DEV); model.eval()
    sampler = Sampler(cache, bank, SimpleNamespace(imagine_blocks=5), rng)
    zi, zg, tg = sampler.predicted(model, min(args.pred_rollouts, len(bank.hist_ok)))
    valid = (tg > -2).reshape(-1)
    predicted = (zi.reshape(-1, zi.shape[-1])[valid].cpu().numpy(),
                 zg[:, None].expand(-1, zi.shape[1], -1).reshape(-1, zi.shape[-1])[valid].cpu().numpy(),
                 tg.reshape(-1)[valid].cpu().numpy(),
                 np.repeat(np.arange(len(zg)), zi.shape[1])[valid.cpu().numpy()])
    del model

    horizons = [25, 50, 100, 150, 225]
    out = {"args": vars(args), "device": DEV, "sets": {}}
    for set_name, (z, goal_z, tg, goal) in (("logged", logged), ("predicted", predicted)):
        out["sets"][set_name] = {}
        for spec in args.critics:
            name, path = spec.split("=", 1)
            critic, cargs = load_critic(path)
            pred = infer(critic, z, goal_z, horizons)
            out["sets"][set_name][name] = dict(metrics(pred, tg, goal, horizons), h_max=int(critic.h_max), args=cargs)
            print(set_name, name, out["sets"][set_name][name], flush=True)
            del critic
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
