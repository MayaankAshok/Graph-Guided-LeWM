"""Train a history-conditioned head to amortize 25-action inverse-CEM residuals."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from common.continuation import ContinuationHead
from common.lewm_loader import load_lewm
from common.log_util import log
from reachability_inverse_cem import model_context_after_candidates, normalize_actions, rank_metrics


def mean_task_spearman(score, target):
    values = [stats.spearmanr(a, b).statistic for a, b in zip(score, target)
              if np.ptp(a) > 0 and np.ptp(b) > 0]
    return float(np.nanmean(values))


def pairwise_loss(prediction, target, gap=.05):
    pred_gap = prediction[:, :, None] - prediction[:, None, :]
    target_gap = target[:, :, None] - target[:, None, :]
    upper = torch.triu(torch.ones_like(target_gap, dtype=torch.bool), diagonal=1)
    keep = upper & (target_gap.abs() > gap)
    return F.softplus(-torch.sign(target_gap[keep]) * pred_gap[keep]).mean()


@torch.no_grad()
def predict(head, history, goal, task_ids):
    h = torch.from_numpy(history[task_ids]).float().cuda()
    g = torch.from_numpy(goal[task_ids]).float().cuda()
    return head(h, g).cpu().numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", type=Path, default=ROOT / "outputs/pusht/critic_training/cache_full_s0.npz")
    p.add_argument("--audit", type=Path, default=ROOT / "outputs/pusht/experiments/viability_exp2/raw_offset50.npz")
    p.add_argument("--labels", type=Path,
                   default=ROOT / "outputs/pusht/diagnostics/pusht_inverse_reachability_full/raw.npz")
    p.add_argument("--ckpt", type=Path,
                   default=ROOT / "data/checkpoints/models--quentinll--lewm-pusht")
    p.add_argument("--out", type=Path,
                   default=ROOT / "outputs/pusht/experiments/continuation_head/continuation_head.pt")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--rank-weight", type=float, default=.25)
    p.add_argument("--seed", type=int, default=137)
    args = p.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    t0 = time.time()

    with np.load(args.audit) as audit, np.load(args.labels) as labels:
        start_rows = audit["start_row"].astype(np.int64)
        candidates_raw = audit["candidate_actions"].astype(np.float32)
        goal = audit["z_goal"].astype(np.float32)
        terminal_l2 = audit["c_pred"].astype(np.float32)
        recovery = audit["oracle_min_steps"].astype(np.int64)
        continuation = labels["continuation_residual"].astype(np.float32)
    n_tasks, n_candidates = continuation.shape
    assert candidates_raw.shape[:2] == continuation.shape

    with np.load(args.cache) as cache:
        z, action = cache["z"], cache["action"]
        act_mean, act_std = cache["act_mean"], cache["act_std"]
        initial_history = np.stack([z[start_rows - 10], z[start_rows - 5], z[start_rows]], 1)
        past_raw = np.stack([action[row - 10:row] for row in start_rows])

    model = load_lewm(args.ckpt, device="cuda")
    future = normalize_actions(candidates_raw, act_mean, act_std).reshape(
        n_tasks * n_candidates, 5, 10)
    history = np.repeat(initial_history, n_candidates, axis=0)
    past = normalize_actions(past_raw[:, None], act_mean, act_std)[:, 0]
    past = np.repeat(past, n_candidates, axis=0)
    end_history, _, _ = model_context_after_candidates(model, history, past, future)
    end_history = end_history.reshape(n_tasks, n_candidates, 3, -1)
    goal_grouped = np.repeat(goal[:, None], n_candidates, axis=1)
    del model
    torch.cuda.empty_cache()

    order = rng.permutation(n_tasks)
    train_ids, val_ids, test_ids = order[:35], order[35:40], order[40:]
    train_vectors = np.concatenate([
        end_history[train_ids].reshape(-1, end_history.shape[-1]),
        goal_grouped[train_ids].reshape(-1, goal_grouped.shape[-1])])
    z_mean, z_std = train_vectors.mean(0), train_vectors.std(0)
    target = np.log1p(continuation)
    target_mean, target_std = target[train_ids].mean(), target[train_ids].std()
    target_normalized = (target - target_mean) / max(target_std, 1e-6)
    q1, l2_median, q3 = np.quantile(terminal_l2[train_ids], [.25, .5, .75])

    head = ContinuationHead(hidden=(256, 128)).cuda()
    head.set_stats(z_mean, z_std, target_mean, target_std, l2_median, q3 - q1)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)
    train_h = torch.from_numpy(end_history[train_ids]).float().cuda()
    train_g = torch.from_numpy(goal_grouped[train_ids]).float().cuda()
    train_y = torch.from_numpy(target_normalized[train_ids]).float().cuda()
    best_rho, best_state, best_step = -np.inf, None, 0

    for step in range(args.steps + 1):
        head.train()
        prediction = head(train_h, train_g)
        regression = F.smooth_l1_loss(prediction, train_y)
        ranking = pairwise_loss(prediction, train_y)
        loss = regression + args.rank_weight * ranking
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            head.eval()
            val_prediction = predict(head, end_history, goal_grouped, val_ids)
            val_rho = mean_task_spearman(val_prediction, continuation[val_ids])
            if val_rho > best_rho:
                best_rho, best_step = val_rho, step
                best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
            if step % 250 == 0:
                log(f"step={step} loss={loss.item():.4f} regression={regression.item():.4f} "
                    f"ranking={ranking.item():.4f} val_rho={val_rho:.3f}")

    head.load_state_dict(best_state)
    head.eval()
    split_metrics = {}
    for name, ids in (("train", train_ids), ("validation", val_ids), ("test", test_ids)):
        prediction = predict(head, end_history, goal_grouped, ids)
        split_metrics[name] = {
            "n_tasks": int(len(ids)),
            "spearman_vs_inverse_cem": mean_task_spearman(prediction, continuation[ids]),
            "ranking_vs_real_recovery": rank_metrics(prediction, recovery[ids]),
            "terminal_l2_vs_real_recovery": rank_metrics(terminal_l2[ids], recovery[ids]),
        }

    payload = {
        "state_dict": best_state,
        "z_dim": 192,
        "hidden": (256, 128),
        "best_step": best_step,
        "best_validation_spearman": best_rho,
        "train_task_ids": train_ids.tolist(),
        "validation_task_ids": val_ids.tolist(),
        "test_task_ids": test_ids.tolist(),
        "source_labels": str(args.labels),
    }
    torch.save(payload, args.out)
    result = {
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "best_step": best_step,
        "best_validation_spearman": best_rho,
        "splits": split_metrics,
        "seconds": time.time() - t0,
    }
    args.out.with_suffix(".json").write_text(json.dumps(result, indent=2))
    log(f"wrote {args.out} and {args.out.with_suffix('.json')} ({time.time() - t0:.1f}s)")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
