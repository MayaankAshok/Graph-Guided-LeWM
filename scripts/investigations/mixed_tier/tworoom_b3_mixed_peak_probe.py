"""Mechanism probe, take 2. tworoom_b3_mixed_distance_bins.py used the *final* (step
50000, already decayed post-peak) checkpoint weights, since the resumable pipeline only
ever keeps the latest checkpoint, not a peak snapshot. That's the wrong regime to inspect
-- the headline baseline-vs-shaped comparison is peak-vs-peak, and post-decay both
conditions are already somewhat degenerate. This script re-trains baseline and shaped on
the mixed tier (same hyperparameters, same HER tuples, same eval), but additionally keeps
an in-memory snapshot of v_net's weights at the step where test_rho is highest, then runs
the same true-distance-binned Spearman analysis + the shaped-vs-baseline value-difference
identity check on THAT snapshot instead of the final one. One seed (0) for a first read.
"""

import copy
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy import stats as sps

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from tworoom_b0_graph_gate import DEV, build_true_distance_oracle, log
from tworoom_b3_convergence import precompute_eval_set
from tworoom_b3_datatiers import (
    BATCH_SIZE, EMA_TAU, EVAL_EVERY, GAMMA, HIDDEN, N_STEPS, TAU_EXPECTILE, build_tier_setups,
)
from tworoom_b3_gciql_shaping import MLP, build_her_tuples, ema_update

SEED = 0
N_BINS = 10


def train_with_peak_snapshot(tier, use_shaping, setup, seed):
    z, action, phi_dist, d = setup["z"], setup["action"], setup["phi_dist"], setup["d"]
    torch.manual_seed(seed)
    z_t = torch.from_numpy(z).to(DEV)
    act_t = torch.from_numpy(action).float().to(DEV)

    v_net = MLP(2 * d, HIDDEN).to(DEV)
    v_target = MLP(2 * d, HIDDEN).to(DEV)
    q_net = MLP(2 * d + act_t.shape[1], HIDDEN).to(DEV)
    q_target = MLP(2 * d + act_t.shape[1], HIDDEN).to(DEV)
    v_target.load_state_dict(v_net.state_dict())
    q_target.load_state_dict(q_net.state_dict())

    from stable_worldmodel.wm.gcrl.module import ExpectileLoss
    expectile_loss = ExpectileLoss(tau=TAU_EXPECTILE)
    opt = torch.optim.Adam(list(v_net.parameters()) + list(q_net.parameters()), lr=3e-4)

    s_idx, next_idx, goal_idx, act_idx, done_arr = build_her_tuples(
        setup["ep_idx"], setup["step_idx"], action, seed=seed, allowed_episode_ids=setup["train_eps"]
    )
    n_tuples = len(s_idx)

    train_ei, train_ej, train_true_d = setup["train_eval"]
    test_ei, test_ej, test_true_d = setup["test_eval"]
    train_zsg = torch.cat([z_t[train_ei], z_t[train_ej]], dim=-1)
    test_zsg = torch.cat([z_t[test_ei], z_t[test_ej]], dim=-1)

    rng = np.random.default_rng(seed + 1)
    best_rho, best_step, best_state = -2.0, None, None
    t0 = time.time()
    for step in range(1, N_STEPS + 1):
        batch = rng.integers(0, n_tuples, BATCH_SIZE)
        s, nx, g, a_i, dn = s_idx[batch], next_idx[batch], goal_idx[batch], act_idx[batch], done_arr[batch]

        zs, zn, zg = z_t[s], z_t[nx], z_t[g]
        a = act_t[a_i]
        mask = (~torch.from_numpy(dn).to(DEV)).float().unsqueeze(-1)
        reward = -mask
        if use_shaping:
            phi_s = torch.from_numpy(-phi_dist[s, g]).float().to(DEV).unsqueeze(-1)
            phi_n = torch.from_numpy(-phi_dist[nx, g]).float().to(DEV).unsqueeze(-1)
            reward = reward + GAMMA * phi_n - phi_s

        sg = torch.cat([zs, zg], dim=-1)
        ng = torch.cat([zn, zg], dim=-1)
        sag = torch.cat([zs, a, zg], dim=-1)

        with torch.no_grad():
            q = q_target(sag)
        v = v_net(sg)
        value_loss = expectile_loss(v, q.detach())

        with torch.no_grad():
            next_v = v_target(ng)
            q_tgt = reward + GAMMA * mask * next_v
        q_pred = q_net(sag)
        critic_loss = ((q_pred - q_tgt) ** 2).mean()

        loss = value_loss + critic_loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        ema_update(v_target, v_net, EMA_TAU)
        ema_update(q_target, q_net, EMA_TAU)

        if step % EVAL_EVERY == 0 or step == 1:
            with torch.no_grad():
                v_test = v_net(test_zsg).squeeze(-1).cpu().numpy()
            rho_test, _ = sps.spearmanr(test_true_d, -v_test)
            if rho_test > best_rho:
                best_rho, best_step = rho_test, step
                best_state = copy.deepcopy(v_net.state_dict())
            if step % (EVAL_EVERY * 40) == 0:
                log(f"  step={step} test_rho={rho_test:.4f} (best={best_rho:.4f}@{best_step}) "
                    f"elapsed={time.time()-t0:.1f}s")

    log(f"  done: best test_rho={best_rho:.4f} at step {best_step}/{N_STEPS} elapsed={time.time()-t0:.1f}s")
    return best_state, best_step, best_rho


def main():
    log(f"device={DEV}  effective horizon 1/(1-gamma) = {1/(1-GAMMA):.1f} steps (gamma={GAMMA})")
    setup = build_tier_setups(["mixed"])["mixed"]
    z, proprio, phi_dist, d = setup["z"], setup["proprio"], setup["phi_dist"], setup["d"]
    z_t = torch.from_numpy(z).to(DEV)

    true_dist_oracle = build_true_distance_oracle()
    all_rows = np.arange(len(z))
    ei, ej, true_d = precompute_eval_set(all_rows, proprio, true_dist_oracle, n_pairs=8000, seed=999)
    graph_d = phi_dist[ei, ej]
    zsg = torch.cat([z_t[ei], z_t[ej]], dim=-1)
    order = np.argsort(true_d)
    n = len(true_d)
    bin_edges = np.linspace(0, n, N_BINS + 1).astype(int)
    log(f"[eval set] n={len(ei)}, true_dist range [{true_d.min():.1f}, {true_d.max():.1f}] "
        f"(median={np.median(true_d):.1f}), graph_dist range [{graph_d.min():.1f}, {graph_d.max():.1f}] "
        f"(median={np.median(graph_d):.1f})")

    v_by_name = {}
    for use_shaping, name in [(False, "baseline"), (True, "shaped")]:
        log(f"\n=== training {name} (seed={SEED}) with peak snapshotting ===")
        best_state, best_step, best_rho = train_with_peak_snapshot("mixed", use_shaping, setup, SEED)

        v_net = MLP(2 * d, HIDDEN).to(DEV)
        v_net.load_state_dict(best_state)
        v_net.eval()
        with torch.no_grad():
            v = v_net(zsg).squeeze(-1).cpu().numpy()
        v_by_name[name] = v
        rho, p = sps.spearmanr(true_d, -v)
        log(f"\n--- {name} PEAK snapshot (step={best_step}, recorded best_rho={best_rho:.4f}) "
            f"--- recomputed-on-this-8000-pair-set Spearman = {rho:.4f} (p={p:.1e})")
        log(f"{'bin':>4} {'true_dist range':>20} {'n':>6} {'spearman':>10} {'mean_V':>10} {'mean_graph_d':>13}")
        for b in range(N_BINS):
            idx = order[bin_edges[b]:bin_edges[b + 1]]
            if len(idx) < 20:
                continue
            rho_b, _ = sps.spearmanr(true_d[idx], -v[idx])
            log(f"{b:>4} {f'{true_d[idx].min():.0f}-{true_d[idx].max():.0f}':>20} {len(idx):>6} "
                f"{rho_b:>10.3f} {v[idx].mean():>10.3f} {graph_d[idx].mean():>13.1f}")

    log("\n=== V(shaped,peak) - V(baseline,peak) vs graph_dist ===")
    delta = v_by_name["shaped"] - v_by_name["baseline"]
    rho_delta, p_delta = sps.spearmanr(graph_d, delta)
    slope, intercept = np.polyfit(graph_d, delta, 1)
    log(f"Spearman(graph_dist, delta) = {rho_delta:.4f} (p={p_delta:.1e})")
    log(f"linear fit: delta ~= {slope:.4f} * graph_dist + {intercept:.4f} (identity predicts slope near +1)")
    for b in range(N_BINS):
        idx = order[bin_edges[b]:bin_edges[b + 1]]
        if len(idx) < 20:
            continue
        log(f"  bin {b} (true_dist {true_d[idx].min():.0f}-{true_d[idx].max():.0f}): "
            f"mean_delta={delta[idx].mean():.3f} mean_graph_d={graph_d[idx].mean():.1f}")


if __name__ == "__main__":
    main()
