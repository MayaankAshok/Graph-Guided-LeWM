"""Inverse-CEM test of local sphere coverage and continuation cost on Push-T."""

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from common.lewm_loader import load_lewm
from common.log_util import log
from common.viability import SKIP, imagine
from local_reachability_probe import choose_rows


@torch.inference_mode()
def inverse_cem(model, z_hist, a_past, target, horizon, population, iterations,
                elite, rng, chunk=1024, action_clip=3.0):
    """Minimum model-terminal L2 for each independent (history, target) query."""
    queries, action_dim = len(target), SKIP * 2
    mean = np.zeros((queries, horizon, action_dim), np.float32)
    std = np.ones_like(mean)
    best = np.full(queries, np.inf, np.float32)
    zh = torch.as_tensor(z_hist, dtype=torch.float32, device="cuda")
    ap = torch.as_tensor(a_past, dtype=torch.float32, device="cuda")
    tg = torch.as_tensor(target, dtype=torch.float32, device="cuda")
    zh = zh.repeat_interleave(population, 0)
    ap = ap.repeat_interleave(population, 0)
    tg = tg.repeat_interleave(population, 0)

    for _ in range(iterations):
        actions = mean[:, None] + std[:, None] * rng.standard_normal(
            (queries, population, horizon, action_dim)).astype(np.float32)
        actions = np.clip(actions, -action_clip, action_clip)
        flat = actions.reshape(queries * population, horizon, action_dim)
        costs = []
        for lo in range(0, len(flat), chunk):
            hi = min(lo + chunk, len(flat))
            future = torch.from_numpy(flat[lo:hi]).cuda()
            endpoint = imagine(model, zh[lo:hi], ap[lo:hi], future)[0][:, -1]
            costs.append(torch.linalg.vector_norm(endpoint - tg[lo:hi], dim=1).cpu().numpy())
        costs = np.concatenate(costs).reshape(queries, population)
        best = np.minimum(best, costs.min(1))
        elite_idx = np.argpartition(costs, elite - 1, axis=1)[:, :elite]
        selected = actions[np.arange(queries)[:, None], elite_idx]
        mean = selected.mean(1)
        std = np.maximum(selected.std(1), .05)
    return best


def normalize_actions(raw, mean, std):
    return ((raw.reshape(*raw.shape[:-2], -1, SKIP, 2) - mean) / std).reshape(
        *raw.shape[:-2], -1, SKIP * 2)


@torch.inference_mode()
def model_context_after_candidates(model, z_hist, a_past, future, chunk=512):
    end_hist, end_past, endpoint = [], [], []
    for lo in range(0, len(future), chunk):
        hi = min(lo + chunk, len(future))
        pred, z_seq, a_seq = imagine(
            model,
            torch.from_numpy(z_hist[lo:hi]).float().cuda(),
            torch.from_numpy(a_past[lo:hi]).float().cuda(),
            torch.from_numpy(future[lo:hi]).float().cuda())
        end_hist.append(z_seq[:, -3:].float().cpu().numpy())
        end_past.append(a_seq[:, -2:].float().cpu().numpy())
        endpoint.append(pred[:, -1].float().cpu().numpy())
    return np.concatenate(end_hist), np.concatenate(end_past), np.concatenate(endpoint)


def rank_metrics(score, recovery, viable_steps=25):
    rhos, inversions = [], []
    for s, t in zip(score, recovery):
        if np.ptp(s) > 0 and np.ptp(t) > 0:
            rhos.append(stats.spearmanr(s, t).statistic)
        ordered = s[:, None] < s[None]
        inversions.append(np.sum(ordered & (t[:, None] > t[None])) / max(ordered.sum(), 1))
    chosen = recovery[np.arange(len(recovery)), np.argmin(score, axis=1)]
    return {
        "mean_within_task_spearman": float(np.nanmean(rhos)),
        "mean_pairwise_inversion_rate": float(np.mean(inversions)),
        "selected_mean_recovery_steps": float(np.mean(chosen)),
        "selected_median_recovery_steps": float(np.median(chosen)),
        "selected_viable_fraction": float(np.mean(chosen <= viable_steps)),
        "selected_censored_fraction": float(np.mean(chosen > 100)),
    }


def standardize_rows(x):
    return (x - x.mean(1, keepdims=True)) / np.maximum(x.std(1, keepdims=True), 1e-8)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", type=Path, default=ROOT / "outputs/pusht/critic_training/cache_full_s0.npz")
    p.add_argument("--audit", type=Path, default=ROOT / "outputs/pusht/experiments/viability_exp2/raw_offset50.npz")
    p.add_argument("--ckpt", type=Path,
                   default=ROOT / "data/checkpoints/models--quentinll--lewm-pusht")
    p.add_argument("--out", type=Path,
                   default=ROOT / "outputs/pusht/diagnostics/pusht_inverse_reachability")
    p.add_argument("--sphere-starts", type=int, default=8)
    p.add_argument("--sphere-pairs", type=int, default=16)
    p.add_argument("--basis-samples", type=int, default=2048)
    p.add_argument("--rank-tasks", type=int, default=12)
    p.add_argument("--population", type=int, default=96)
    p.add_argument("--sphere-iters", type=int, default=10)
    p.add_argument("--rank-iters", type=int, default=8)
    p.add_argument("--elite", type=int, default=12)
    p.add_argument("--radius", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=91)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    t0 = time.time()

    with np.load(args.cache) as data:
        z, action = data["z"], data["action"]
        offsets, lengths = data["ep_offset"], data["ep_len"]
        act_mean, act_std = data["act_mean"], data["act_std"]
        sphere_rows = choose_rows(offsets, lengths, args.sphere_starts, rng)
        sphere_hist = np.stack([z[sphere_rows - 2 * SKIP], z[sphere_rows - SKIP],
                                z[sphere_rows]], axis=1)
        sphere_past_raw = np.stack([action[r - 2 * SKIP:r] for r in sphere_rows])

        with np.load(args.audit) as audit:
            n_tasks = min(args.rank_tasks, len(audit["start_row"]))
            start_rows = audit["start_row"][:n_tasks].astype(np.int64)
            candidates_raw = audit["candidate_actions"][:n_tasks].astype(np.float32)
            goal = audit["z_goal"][:n_tasks].astype(np.float32)
            terminal_l2 = audit["c_pred"][:n_tasks].astype(np.float32)
            audit_endpoint = audit["z_pred"][:n_tasks].astype(np.float32)
            recovery = audit["oracle_min_steps"][:n_tasks].astype(np.int64)
            rank_hist_task = np.stack([z[start_rows - 2 * SKIP], z[start_rows - SKIP],
                                       z[start_rows]], axis=1)
            rank_past_raw_task = np.stack([action[r - 2 * SKIP:r] for r in start_rows])

    model = load_lewm(args.ckpt, device="cuda")

    # A local PCA basis from the model's own initial-CEM action distribution, then paired
    # radius-r targets q and -q. Inverse CEM removes the previous Monte-Carlo density limit.
    sphere_past = normalize_actions(sphere_past_raw[:, None], act_mean, act_std)[:, 0]
    basis_actions = rng.standard_normal(
        (args.sphere_starts * args.basis_samples, 1, SKIP * 2)).astype(np.float32)
    basis_hist = np.repeat(sphere_hist, args.basis_samples, axis=0)
    basis_past = np.repeat(sphere_past, args.basis_samples, axis=0)
    _, _, basis_endpoint = model_context_after_candidates(
        model, basis_hist, basis_past, basis_actions)
    basis_delta = basis_endpoint.reshape(args.sphere_starts, args.basis_samples, -1) \
        - sphere_hist[:, None, -1]

    targets, query_hist, query_past = [], [], []
    for i, delta in enumerate(basis_delta):
        centered = delta - delta.mean(0)
        _, singular, vt = np.linalg.svd(centered, full_matrices=False)
        frac = np.cumsum(singular ** 2) / np.sum(singular ** 2)
        k95 = np.searchsorted(frac, .95) + 1
        coeff = rng.standard_normal((args.sphere_pairs, k95))
        direction = coeff @ vt[:k95]
        direction /= np.linalg.norm(direction, axis=1, keepdims=True)
        direction = np.concatenate([direction, -direction])
        targets.append(sphere_hist[i, -1] + args.radius * direction)
        query_hist.append(np.repeat(sphere_hist[i:i + 1], len(direction), axis=0))
        query_past.append(np.repeat(sphere_past[i:i + 1], len(direction), axis=0))
    targets, query_hist, query_past = map(np.concatenate, (targets, query_hist, query_past))
    log(f"[sphere] {len(targets)} targets; inverse CEM {args.population}x{args.sphere_iters}")
    sphere_residual = inverse_cem(
        model, query_hist, query_past, targets, 1, args.population, args.sphere_iters,
        args.elite, rng)
    paired = sphere_residual.reshape(args.sphere_starts, 2, args.sphere_pairs)
    sphere_result = {
        "queries": int(len(sphere_residual)),
        "residual_mean": float(sphere_residual.mean()),
        "residual_median": float(np.median(sphere_residual)),
        "residual_q90": float(np.quantile(sphere_residual, .9)),
        "coverage_at_0.25": float(np.mean(sphere_residual <= .25)),
        "coverage_at_0.5": float(np.mean(sphere_residual <= .5)),
        "coverage_at_1.0": float(np.mean(sphere_residual <= 1.0)),
        "antipodal_disagreement_at_0.5": float(np.mean(
            (paired[:, 0] <= .5) != (paired[:, 1] <= .5))),
        "antipodal_disagreement_at_1.0": float(np.mean(
            (paired[:, 0] <= 1.0) != (paired[:, 1] <= 1.0))),
    }

    # Reconstruct the predictor history at each saved 25-action candidate endpoint, then
    # solve for the best additional 25 actions. Compare that residual with real recovery T.
    n_candidates = candidates_raw.shape[1]
    future = normalize_actions(candidates_raw, act_mean, act_std).reshape(
        n_tasks * n_candidates, 5, SKIP * 2)
    rank_hist = np.repeat(rank_hist_task, n_candidates, axis=0)
    rank_past = normalize_actions(rank_past_raw_task[:, None], act_mean, act_std)[:, 0]
    rank_past = np.repeat(rank_past, n_candidates, axis=0)
    end_hist, end_past, reconstructed_endpoint = model_context_after_candidates(
        model, rank_hist, rank_past, future)
    reconstruction_error = np.linalg.norm(
        reconstructed_endpoint.reshape(n_tasks, n_candidates, -1)
        - audit_endpoint, axis=2)
    rank_goal = np.repeat(goal, n_candidates, axis=0)
    log(f"[ranking] {len(rank_goal)} endpoints; continuation CEM "
        f"{args.population}x{args.rank_iters}x5 blocks")
    continuation = inverse_cem(
        model, end_hist, end_past, rank_goal, 5, args.population, args.rank_iters,
        args.elite, rng).reshape(n_tasks, n_candidates)

    scores = {"terminal_l2": terminal_l2, "continuation_residual": continuation}
    l2z, contz = standardize_rows(terminal_l2), standardize_rows(continuation)
    for weight in (.25, .5, 1.0, 2.0):
        scores[f"hybrid_{weight:g}"] = l2z + weight * contz
    rank_result = {name: rank_metrics(score, recovery) for name, score in scores.items()}
    rank_result["oracle_pool_upper_bound_viable_fraction"] = float(
        np.mean(np.min(recovery, axis=1) <= 25))
    rank_result["endpoint_reconstruction_error"] = {
        "mean": float(reconstruction_error.mean()),
        "max": float(reconstruction_error.max()),
    }

    result = {
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "sphere": sphere_result,
        "ranking": rank_result,
        "seconds": time.time() - t0,
    }
    (args.out / "summary.json").write_text(json.dumps(result, indent=2))
    np.savez_compressed(args.out / "raw.npz", sphere_residual=sphere_residual,
                        sphere_paired=paired, terminal_l2=terminal_l2,
                        continuation_residual=continuation, recovery_steps=recovery)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].hist(sphere_residual, bins=50, alpha=.8)
    axes[0].axvline(.5, color="black", linestyle="--", label="tolerance 0.5")
    axes[0].axvline(1.0, color="tab:red", linestyle="--", label="tolerance 1.0")
    axes[0].set(xlabel="best inverse-CEM latent residual", ylabel="targets",
                title="Radius-2 sphere targets")
    axes[0].legend()
    names = list(scores)
    values = [rank_result[name]["selected_viable_fraction"] for name in names]
    axes[1].bar(np.arange(len(names)), values)
    axes[1].set_xticks(np.arange(len(names)), names, rotation=35, ha="right")
    axes[1].set_ylim(0, 1)
    axes[1].set(ylabel="fraction recoverable within 25 steps",
                title="Cost-selected candidate viability")
    fig.tight_layout()
    fig.savefig(args.out / "summary.png", dpi=180)
    log(f"[done] {args.out / 'summary.json'} ({time.time() - t0:.1f}s)")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
