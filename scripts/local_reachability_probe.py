"""Monte-Carlo probe of Push-T's local model-reachable latent set."""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from common.lewm_loader import load_lewm
from common.viability import HISTORY, SKIP, imagine


def summarize(x):
    return {"mean": float(np.mean(x)), "median": float(np.median(x)),
            "q10": float(np.quantile(x, .1)), "q90": float(np.quantile(x, .9))}


def min_dist(points, targets, batch=128):
    p2 = np.sum(points * points, axis=1)
    out = []
    for lo in range(0, len(targets), batch):
        t = targets[lo:lo + batch]
        d2 = np.sum(t * t, axis=1)[:, None] + p2[None] - 2 * t @ points.T
        out.append(np.sqrt(np.maximum(d2.min(axis=1), 0)))
    return np.concatenate(out)


def analyze(delta, radius, queries, rng):
    norms = np.linalg.norm(delta, axis=1)
    unit = delta[norms > 1e-8] / norms[norms > 1e-8, None]
    centered = delta - delta.mean(0)
    _, singular, vt = np.linalg.svd(centered, full_matrices=False)
    variance = singular ** 2
    frac = np.cumsum(variance) / variance.sum()
    k95 = int(np.searchsorted(frac, .95) + 1)
    pr = float(variance.sum() ** 2 / np.sum(variance ** 2))

    coeff = rng.standard_normal((queries, k95))
    directions = coeff @ vt[:k95]
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    targets = radius * directions
    distance = min_dist(delta, targets)
    opposite_distance = min_dist(delta, -targets)
    coverage = {}
    for tol in (.25, .5, 1.0):
        hit, opposite_hit = distance <= tol, opposite_distance <= tol
        coverage[str(tol)] = {
            "fraction": float(hit.mean()),
            "antipodal_fraction": float(opposite_hit.mean()),
            "antipodal_disagreement": float(np.mean(hit != opposite_hit)),
        }

    probe = unit[:min(512, len(unit))]
    best_reverse_cos = np.max((-probe) @ unit.T, axis=1)
    return {
        "displacement_norm": summarize(norms),
        "effective_dimension_participation_ratio": pr,
        "dimensions_for_95pct_variance": k95,
        "mean_direction_resultant": float(np.linalg.norm(unit.mean(0))),
        "best_reverse_direction_cosine": summarize(best_reverse_cos),
        "fraction_with_reverse_cosine_above_0.9": float(np.mean(best_reverse_cos > .9)),
        "radius": radius,
        "nearest_reachable_distance_on_sphere": summarize(distance),
        "coverage_by_tolerance": coverage,
    }, vt[:2]


def choose_rows(offsets, lengths, count, rng):
    rows = []
    valid_ep = np.flatnonzero(lengths > 2 * (HISTORY - 1) * SKIP + 5 * SKIP)
    for ep in rng.choice(valid_ep, size=count, replace=False):
        step = rng.integers((HISTORY - 1) * SKIP,
                            lengths[ep] - 5 * SKIP)
        rows.append(int(offsets[ep] + step))
    return np.asarray(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cache", type=Path)
    p.add_argument("--ckpt", type=Path,
                   default=ROOT / "data/checkpoints/models--quentinll--lewm-pusht")
    p.add_argument("--out", type=Path,
                   default=ROOT / "outputs/pusht/diagnostics/pusht_local_reachability")
    p.add_argument("--starts", type=int, default=12)
    p.add_argument("--samples", type=int, default=4096)
    p.add_argument("--queries", type=int, default=512)
    p.add_argument("--radius", type=float, default=2.0)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--seed", type=int, default=73)
    p.add_argument("--action-sampler", choices=("cem", "legal-uniform"), default="cem")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    with np.load(args.cache) as data:
        z = data["z"]
        action = data["action"]
        offsets, lengths = data["ep_offset"], data["ep_len"]
        act_mean, act_std = data["act_mean"], data["act_std"]
        rows = choose_rows(offsets, lengths, args.starts, rng)
        z_hist = np.stack([z[rows - 2 * SKIP], z[rows - SKIP], z[rows]], axis=1)
        past = np.stack([action[r - 2 * SKIP:r] for r in rows]).reshape(args.starts, 2, -1)

    model = load_lewm(args.ckpt, device="cuda")
    shape = (args.starts, args.samples, 5, SKIP * 2)
    if args.action_sampler == "cem":
        norm = rng.standard_normal(shape).astype(np.float32)
    else:
        raw = rng.uniform(-1, 1, size=(args.starts, args.samples, 5 * SKIP, 2)).astype(np.float32)
        norm = ((raw - act_mean) / act_std).reshape(shape)
    past = ((past.reshape(args.starts, 2, SKIP, 2) - act_mean) / act_std).reshape(
        args.starts, 2, SKIP * 2)

    endpoints = {1: [], 5: []}
    for start in range(args.starts):
        parts = {1: [], 5: []}
        for lo in range(0, args.samples, args.batch):
            hi = min(lo + args.batch, args.samples)
            n = hi - lo
            zh = torch.from_numpy(np.repeat(z_hist[start:start + 1], n, axis=0)).cuda()
            ap = torch.from_numpy(np.repeat(past[start:start + 1], n, axis=0)).cuda()
            af = torch.from_numpy(norm[start, lo:hi]).cuda()
            pred, _, _ = imagine(model, zh, ap, af)
            pred = pred.float().cpu().numpy()
            parts[1].append(pred[:, 0])
            parts[5].append(pred[:, -1])
        for horizon in endpoints:
            endpoints[horizon].append(np.concatenate(parts[horizon]) - z_hist[start, -1])

    per_start, bases = {1: [], 5: []}, {1: [], 5: []}
    for horizon in endpoints:
        for delta in endpoints[horizon]:
            stats, basis = analyze(delta, args.radius, args.queries, rng)
            per_start[horizon].append(stats)
            bases[horizon].append(basis)

    result = {"config": vars(args) | {"cache": str(args.cache), "ckpt": str(args.ckpt),
                                      "out": str(args.out)}, "rows": rows.tolist(), "horizons": {}}
    for horizon in endpoints:
        keys = per_start[horizon][0]
        aggregate = {}
        for key in keys:
            if isinstance(keys[key], (int, float)):
                aggregate[key] = summarize([x[key] for x in per_start[horizon]])
        for tol in ("0.25", "0.5", "1.0"):
            aggregate[f"coverage_tol_{tol}"] = summarize([
                x["coverage_by_tolerance"][tol]["fraction"] for x in per_start[horizon]])
            aggregate[f"antipodal_disagreement_tol_{tol}"] = summarize([
                x["coverage_by_tolerance"][tol]["antipodal_disagreement"]
                for x in per_start[horizon]])
        result["horizons"][str(horizon)] = {"aggregate_across_starts": aggregate,
                                               "per_start": per_start[horizon]}

    (args.out / "summary.json").write_text(json.dumps(result, indent=2))
    fig, axes = plt.subplots(args.starts, 2, figsize=(10, 3 * args.starts), squeeze=False)
    theta = np.linspace(0, 2 * np.pi, 300)
    for i in range(args.starts):
        for col, horizon in enumerate((1, 5)):
            xy = endpoints[horizon][i] @ bases[horizon][i].T
            axes[i, col].scatter(xy[:, 0], xy[:, 1], s=2, alpha=.2)
            axes[i, col].plot(args.radius * np.cos(theta), args.radius * np.sin(theta), "k--")
            axes[i, col].set_aspect("equal")
            axes[i, col].set_title(f"start {i}, horizon {horizon} block(s)")
    fig.tight_layout()
    fig.savefig(args.out / "reachable_sets_pca.png", dpi=150)
    print(json.dumps(result["horizons"], indent=2))


if __name__ == "__main__":
    main()
