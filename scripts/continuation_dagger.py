"""CEM-in-the-loop data aggregation for the Push-T continuation-cost ensemble."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from common.continuation import ContinuationEnsemble, ContinuationHead
from common.lewm_loader import load_lewm
from common.log_util import log
from continuation_train import mean_task_spearman, pairwise_loss
from reachability_inverse_cem import inverse_cem, model_context_after_candidates, normalize_actions


def choose_problems(offsets, lengths, count, goal_offset, rng, excluded_episodes=()):
    valid = np.flatnonzero(lengths > 10 + goal_offset)
    valid = np.setdiff1d(valid, np.asarray(excluded_episodes, dtype=np.int64))
    episodes = rng.choice(valid, size=count, replace=False)
    steps = np.array([rng.integers(10, lengths[e] - goal_offset) for e in episodes])
    return episodes, offsets[episodes] + steps


@torch.inference_mode()
def ensemble_predictions(ensemble, history, goal, batch=4096):
    means, stds = [], []
    for lo in range(0, len(history), batch):
        hi = min(lo + batch, len(history))
        h = torch.from_numpy(history[lo:hi]).float().cuda()
        g = torch.from_numpy(goal[lo:hi]).float().cuda()
        p = ensemble.predictions(h, g)
        means.append(p.mean(0).cpu().numpy())
        stds.append(p.std(0, unbiased=False).cpu().numpy())
    return np.concatenate(means), np.concatenate(stds)


def train_ensemble(data, validation_groups, args, seed):
    group_ids = data["group_id"]
    validation_episodes = np.unique(data["episode_id"][
        np.isin(group_ids, validation_groups)])
    train_groups = np.unique(group_ids[~np.isin(data["episode_id"], validation_episodes)])
    train_rows = np.isin(group_ids, train_groups)
    vectors = np.concatenate([data["history"][train_rows].reshape(-1, 192),
                              data["goal"][train_rows]])
    z_mean, z_std = vectors.mean(0), vectors.std(0)
    target = np.log1p(data["label"])
    target_mean, target_std = target[train_rows].mean(), target[train_rows].std()
    target_n = (target - target_mean) / max(target_std, 1e-6)
    l2 = ((data["history"][:, -1] - data["goal"]) ** 2).sum(1)
    q1, l2_median, q3 = np.quantile(l2[train_rows], [.25, .5, .75])

    members = []
    for member_id in range(args.members):
        torch.manual_seed(seed + member_id)
        member = ContinuationHead(hidden=(256, 128)).cuda()
        member.set_stats(z_mean, z_std, target_mean, target_std, l2_median, q3 - q1)
        optimizer = torch.optim.AdamW(member.parameters(), lr=args.lr, weight_decay=1e-4)
        member_rng = np.random.default_rng(seed + 1009 * member_id)
        for _ in range(args.train_steps):
            chosen = member_rng.choice(train_groups, size=min(args.groups_per_batch,
                                                               len(train_groups)), replace=True)
            batches = [np.flatnonzero(group_ids == group) for group in chosen]
            rows = np.concatenate(batches)
            h = torch.from_numpy(data["history"][rows]).float().cuda()
            g = torch.from_numpy(data["goal"][rows]).float().cuda()
            y = torch.from_numpy(target_n[rows]).float().cuda()
            prediction = member(h, g)
            regression = F.smooth_l1_loss(prediction, y)
            rank_losses, pos = [], 0
            for batch_rows in batches:
                width = len(batch_rows)
                rank_loss = pairwise_loss(
                    prediction[pos:pos + width][None], y[pos:pos + width][None])
                rank_losses.append(torch.nan_to_num(rank_loss))
                pos += width
            loss = regression + args.rank_weight * torch.stack(rank_losses).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        member.eval()
        members.append(member)

    ensemble = ContinuationEnsemble(members).cuda().eval()
    val_rhos = []
    for group in validation_groups:
        rows = np.flatnonzero(group_ids == group)
        mean, _ = ensemble_predictions(ensemble, data["history"][rows], data["goal"][rows])
        if len(rows) > 2:
            val_rhos.append(mean_task_spearman(mean[None], data["label"][rows][None]))
    return ensemble, float(np.mean(val_rhos))


def save_ensemble(path, ensemble, validation_rho, round_no, data_size, args):
    payload = {
        "members": [{k: v.detach().cpu() for k, v in member.state_dict().items()}
                    for member in ensemble.members],
        "z_dim": 192,
        "hidden": (256, 128),
        "best_step": round_no,
        "best_validation_spearman": validation_rho,
        "dagger_round": round_no,
        "training_examples": data_size,
        "kappa": args.kappa,
    }
    torch.save(payload, path)


def collect(model, ensemble, initial_history, past, goal, args, rng):
    starts, population, horizon = len(goal), args.outer_population, 5
    mean = np.zeros((starts, horizon, 10), np.float32)
    std = np.ones_like(mean)
    last = None
    for _ in range(args.outer_iters):
        actions = mean[:, None] + std[:, None] * rng.standard_normal(
            (starts, population, horizon, 10)).astype(np.float32)
        actions = np.clip(actions, -3, 3)
        flat = actions.reshape(starts * population, horizon, 10)
        histories = np.repeat(initial_history, population, axis=0)
        pasts = np.repeat(past, population, axis=0)
        end_history, end_past, endpoint = model_context_after_candidates(
            model, histories, pasts, flat)
        goals = np.repeat(goal, population, axis=0)
        head_mean, head_std = ensemble_predictions(ensemble, end_history, goals)
        l2 = ((endpoint - goals) ** 2).sum(1).reshape(starts, population)
        head_mean = head_mean.reshape(starts, population)
        head_std = head_std.reshape(starts, population)
        l2z = (l2 - ensemble.l2_median.item()) / ensemble.l2_iqr.item()
        cost = l2z + args.outer_weight * (head_mean + args.kappa * head_std)
        elite_idx = np.argpartition(cost, args.outer_topk - 1, axis=1)[:, :args.outer_topk]
        elite = actions[np.arange(starts)[:, None], elite_idx]
        mean, std = elite.mean(1), np.maximum(elite.std(1), .05)
        last = actions, end_history.reshape(starts, population, 3, 192), \
            end_past.reshape(starts, population, 2, 10), head_std, cost

    actions, histories, pasts, uncertainty, cost = last
    selected_history, selected_past, selected_goal = [], [], []
    for i in range(starts):
        elite = np.argsort(cost[i])[:args.save_elites]
        uncertain = np.argsort(uncertainty[i])[::-1]
        chosen = list(elite)
        for index in uncertain:
            if index not in chosen:
                chosen.append(int(index))
            if len(chosen) == args.save_elites + args.save_uncertain:
                break
        selected_history.append(histories[i, chosen])
        selected_past.append(pasts[i, chosen])
        selected_goal.append(np.repeat(goal[i:i + 1], len(chosen), axis=0))
    return map(np.concatenate, (selected_history, selected_past, selected_goal))


def initial_data(args, model, cache):
    with np.load(args.audit) as audit, np.load(args.labels) as labels:
        rows = audit["start_row"].astype(np.int64)
        candidates = audit["candidate_actions"].astype(np.float32)
        goal = audit["z_goal"].astype(np.float32)
        label = labels["continuation_residual"].astype(np.float32)
    n_tasks, n_candidates = label.shape
    z, action = cache["z"], cache["action"]
    episode_id = np.searchsorted(cache["ep_offset"], rows, side="right") - 1
    history = np.stack([z[rows - 10], z[rows - 5], z[rows]], 1)
    past_raw = np.stack([action[row - 10:row] for row in rows])
    future = normalize_actions(candidates, cache["act_mean"], cache["act_std"]).reshape(
        n_tasks * n_candidates, 5, 10)
    history = np.repeat(history, n_candidates, axis=0)
    past = normalize_actions(past_raw[:, None], cache["act_mean"], cache["act_std"])[:, 0]
    past = np.repeat(past, n_candidates, axis=0)
    end_history, _, _ = model_context_after_candidates(model, history, past, future)
    return {
        "history": end_history.astype(np.float32),
        "goal": np.repeat(goal, n_candidates, axis=0).astype(np.float32),
        "label": label.ravel(),
        "group_id": np.repeat(np.arange(n_tasks), n_candidates),
        "episode_id": np.repeat(episode_id, n_candidates),
        "round": np.zeros(n_tasks * n_candidates, np.int16),
    }


def save_data(path, data):
    np.savez_compressed(path, **data)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", type=Path, default=ROOT / "outputs/pusht/critic_training/cache_full_s0.npz")
    p.add_argument("--audit", type=Path, default=ROOT / "outputs/pusht/experiments/viability_exp2/raw_offset50.npz")
    p.add_argument("--labels", type=Path,
                   default=ROOT / "outputs/pusht/diagnostics/pusht_inverse_reachability_full/raw.npz")
    p.add_argument("--ckpt", type=Path,
                   default=ROOT / "data/checkpoints/models--quentinll--lewm-pusht")
    p.add_argument("--out", type=Path, default=ROOT / "outputs/pusht/experiments/continuation_dagger")
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument("--starts-per-round", type=int, default=64)
    p.add_argument("--members", type=int, default=5)
    p.add_argument("--train-steps", type=int, default=600)
    p.add_argument("--groups-per-batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--rank-weight", type=float, default=.25)
    p.add_argument("--outer-population", type=int, default=96)
    p.add_argument("--outer-iters", type=int, default=8)
    p.add_argument("--outer-topk", type=int, default=12)
    p.add_argument("--outer-weight", type=float, default=.25)
    p.add_argument("--kappa", type=float, default=1.0)
    p.add_argument("--save-elites", type=int, default=8)
    p.add_argument("--save-uncertain", type=int, default=8)
    p.add_argument("--teacher-population", type=int, default=96)
    p.add_argument("--teacher-iters", type=int, default=8)
    p.add_argument("--teacher-elite", type=int, default=12)
    p.add_argument("--seed", type=int, default=211)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    data_path = args.out / "aggregate.npz"
    rng = np.random.default_rng(args.seed)
    t0 = time.time()

    with np.load(args.cache) as source:
        cache = {key: source[key] for key in source.files}
    model = load_lewm(args.ckpt, device="cuda")
    if data_path.exists():
        with np.load(data_path) as saved:
            data = {key: saved[key] for key in saved.files}
        completed = int(data["round"].max())
        log(f"resuming {len(data)} fields / {len(data['label'])} examples after round {completed}")
    else:
        data = initial_data(args, model, cache)
        save_data(data_path, data)
        completed = 0
        log(f"bootstrapped {len(data['label'])} examples")

    validation_groups = np.arange(40, 50)
    validation_episodes = np.unique(data["episode_id"][
        np.isin(data["group_id"], validation_groups)])
    history = []
    ensemble = None
    for round_no in range(completed, args.rounds + 1):
        log(f"[round {round_no}] training {args.members} heads on {len(data['label'])} examples")
        ensemble, val_rho = train_ensemble(
            data, validation_groups, args, args.seed + 10_000 * round_no)
        ckpt = args.out / f"round{round_no}.pt"
        save_ensemble(ckpt, ensemble, val_rho, round_no, len(data["label"]), args)
        record = {"round": round_no, "examples": len(data["label"]),
                  "validation_spearman": val_rho}
        history.append(record)
        log(f"[round {round_no}] validation rho={val_rho:.3f}; wrote {ckpt}")
        if round_no == args.rounds:
            break

        round_rng = np.random.default_rng(args.seed + round_no + 1)
        episodes, rows = choose_problems(
            cache["ep_offset"], cache["ep_len"], args.starts_per_round, 50,
            round_rng, validation_episodes)
        initial_history = np.stack([cache["z"][rows - 10], cache["z"][rows - 5],
                                    cache["z"][rows]], 1)
        past_raw = np.stack([cache["action"][row - 10:row] for row in rows])
        past = normalize_actions(past_raw[:, None], cache["act_mean"], cache["act_std"])[:, 0]
        goal = cache["z"][rows + 50]
        log(f"[round {round_no + 1}] collecting from {len(rows)} fresh outer-CEM problems")
        selected_history, selected_past, selected_goal = collect(
            model, ensemble, initial_history, past, goal, args, round_rng)
        log(f"[round {round_no + 1}] teacher-labelling {len(selected_goal)} hard candidates")
        label = inverse_cem(
            model, selected_history, selected_past, selected_goal, 5,
            args.teacher_population, args.teacher_iters, args.teacher_elite, round_rng)
        next_group = int(data["group_id"].max()) + 1
        per_group = args.save_elites + args.save_uncertain
        added = {
            "history": selected_history.astype(np.float32),
            "goal": selected_goal.astype(np.float32),
            "label": label.astype(np.float32),
            "group_id": np.repeat(np.arange(next_group, next_group + len(rows)), per_group),
            "episode_id": np.repeat(episodes, per_group),
            "round": np.full(len(label), round_no + 1, np.int16),
        }
        data = {key: np.concatenate([data[key], added[key]]) for key in data}
        save_data(data_path, data)
        record.update({"added": len(label), "teacher_residual_mean": float(label.mean()),
                       "teacher_residual_q90": float(np.quantile(label, .9))})
        (args.out / "progress.json").write_text(json.dumps(history, indent=2))
        log(f"[round {round_no + 1}] checkpointed aggregate ({len(data['label'])} examples)")

    final_path = args.out / "continuation_dagger.pt"
    save_ensemble(final_path, ensemble, history[-1]["validation_spearman"],
                  args.rounds, len(data["label"]), args)
    result = {"config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "rounds": history, "examples": len(data["label"]),
              "final_checkpoint": str(final_path), "seconds": time.time() - t0}
    (args.out / "summary.json").write_text(json.dumps(result, indent=2))
    log(f"done: {final_path} ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
