"""B3 (scoped), take 2: train/test split, frequent cheap evaluation, run to convergence.

Addresses two real gaps in the first pass (tworoom_b3_gciql_shaping.py):
  1. No held-out data at all -- eval pairs were drawn from the same landmark pool used
     for HER training tuples. Here: 100 episodes, 80 train / 20 held-out test episodes.
     HER training tuples come ONLY from train episodes; both a train-pairs eval set and
     a test-pairs eval set are tracked throughout training.
  2. Only 5000 steps, and the curves were still visibly climbing -- not converged.
     Here: eval sets are precomputed ONCE (true distances computed a single time via
     one batched multi-source Dijkstra call, not re-run every eval checkpoint), which
     makes each eval essentially free (two MLP forward passes + a Spearman calc). That
     lets us evaluate every 50 steps for tens of thousands of steps without eval
     overhead dominating wall-clock, and actually see a plateau.

The graph (transition + identification edges, calibrated threshold) is built over ALL
100 episodes' landmarks -- Phi is a fixed deterministic function of the frozen encoder
and calibration statistics, not something fit to minimize eval error, so including
test-episode states as graph nodes is not leakage in the ML sense; it's required for
Phi(s,g) to even be defined for held-out states.
"""

import json
import sys
import time
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401
import numpy as np
import torch
from scipy import stats as sps
from scipy.sparse.csgraph import dijkstra

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.training import precompute_eval_set  # noqa: F401
from tworoom_b0_graph_gate import DEV, H5_PATH, build_true_distance_oracle, build_weighted_graph, find_graph_edges
from investigations.early_diagnostics.tworoom_b1_graph_diagnostics import load_landmarks
from tworoom_b3_gciql_shaping import MLP, ema_update, build_her_tuples, log

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "outputs" / "b3_tworoom"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_EPISODES = 100
TRAIN_FRAC = 0.8
Q_CALIB = 1e-3
GAMMA = 0.99
TAU_EXPECTILE = 0.9
EMA_TAU = 0.005
HIDDEN = 256
BATCH_SIZE = 256
N_STEPS = 50000
EVAL_EVERY = 50
N_EVAL_PAIRS = 1000
GOALS_PER_TRANSITION = 4
GOAL_GAMMA = 0.9
SEEDS = [0, 1, 2]


def run_condition(name, use_shaping, z, s_idx, next_idx, goal_idx, act_idx, done, action,
                   phi_dist, train_eval, test_eval, d, seed):
    torch.manual_seed(seed)
    n_tuples = len(s_idx)
    z_t = torch.from_numpy(z).to(DEV)
    act_t = torch.from_numpy(action).float().to(DEV)

    v_net = MLP(2 * d, HIDDEN).to(DEV)
    v_target = MLP(2 * d, HIDDEN).to(DEV)
    v_target.load_state_dict(v_net.state_dict())
    q_net = MLP(2 * d + act_t.shape[1], HIDDEN).to(DEV)
    q_target = MLP(2 * d + act_t.shape[1], HIDDEN).to(DEV)
    q_target.load_state_dict(q_net.state_dict())

    from stable_worldmodel.wm.gcrl.module import ExpectileLoss
    expectile_loss = ExpectileLoss(tau=TAU_EXPECTILE)
    opt = torch.optim.Adam(list(v_net.parameters()) + list(q_net.parameters()), lr=3e-4)

    train_ei, train_ej, train_true_d = train_eval
    test_ei, test_ej, test_true_d = test_eval
    train_zsg = torch.cat([z_t[train_ei], z_t[train_ej]], dim=-1)
    test_zsg = torch.cat([z_t[test_ei], z_t[test_ej]], dim=-1)

    rng = np.random.default_rng(seed + 1)
    history = []
    t0 = time.time()
    for step in range(1, N_STEPS + 1):
        batch = rng.integers(0, n_tuples, BATCH_SIZE)
        s, nx, g, a_i, dn = s_idx[batch], next_idx[batch], goal_idx[batch], act_idx[batch], done[batch]

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
                v_train = v_net(train_zsg).squeeze(-1).cpu().numpy()
                v_test = v_net(test_zsg).squeeze(-1).cpu().numpy()
            rho_train, _ = sps.spearmanr(train_true_d, -v_train)
            rho_test, _ = sps.spearmanr(test_true_d, -v_test)
            history.append(dict(step=step, value_loss=float(value_loss.item()),
                                 critic_loss=float(critic_loss.item()),
                                 spearman_train=float(rho_train), spearman_test=float(rho_test)))
            if step % (EVAL_EVERY * 20) == 0 or step == 1:
                log(f"[{name}] step={step} vloss={value_loss.item():.3f} closs={critic_loss.item():.3f} "
                    f"train_rho={rho_train:.4f} test_rho={rho_test:.4f} elapsed={time.time()-t0:.1f}s")

    return history


def check_convergence(history, key, frac=0.1):
    """Compare the mean of the last `frac` of eval points to the mean of the `frac`
    before that; report the delta as a simple plateau diagnostic."""
    vals = [h[key] for h in history]
    n = len(vals)
    k = max(1, int(n * frac))
    last = np.mean(vals[-k:])
    prev = np.mean(vals[-2 * k:-k]) if n >= 2 * k else np.mean(vals[:k])
    return float(last), float(last - prev)


def main():
    log(f"device={DEV}")
    z, ep_idx, step_idx, proprio = load_landmarks(N_EPISODES)
    n, d = z.shape
    log(f"landmarks: {n} frames, d={d}, {len(np.unique(ep_idx))} episodes")

    f = h5py.File(H5_PATH, "r", swmr=True, rdcc_nbytes=256 * 1024 * 1024)
    ep_offset = f["ep_offset"][:]
    global_idx = ep_offset[ep_idx] + step_idx
    action = f["action"][:][global_idx].astype(np.float32)
    f.close()

    uniq_eps = np.unique(ep_idx)
    rng = np.random.default_rng(0)
    shuffled = rng.permutation(uniq_eps)
    n_train = int(len(shuffled) * TRAIN_FRAC)
    train_eps, test_eps = shuffled[:n_train], shuffled[n_train:]
    log(f"episode split: {len(train_eps)} train, {len(test_eps)} test")

    train_rows = np.nonzero(np.isin(ep_idx, train_eps))[0]
    test_rows = np.nonzero(np.isin(ep_idx, test_eps))[0]
    log(f"landmark rows: {len(train_rows)} train, {len(test_rows)} test")

    log("building graph over ALL landmarks (Phi must be defined for held-out states too)...")
    order = np.lexsort((step_idx, ep_idx))
    ep_o, step_o = ep_idx[order], step_idx[order]
    adj = (ep_o[1:] == ep_o[:-1]) & (step_o[1:] == step_o[:-1] + 1)
    z_o = z[order]
    rho_hat = float(np.mean(np.sum(z_o[:-1][adj] * z_o[1:][adj], axis=1)) / d)
    eps2 = 2 * (1 - rho_hat) * sps.chi2.ppf(Q_CALIB, d)
    log(f"rho_hat={rho_hat:.4f} eps2={eps2:.2f}")

    trans_i, trans_j, id_i, id_j = find_graph_edges(z, ep_idx, step_idx, eps2)
    graph = build_weighted_graph(n, trans_i, trans_j, id_i, id_j, id_weight=1.0)
    log(f"graph: {len(trans_i)} transition edges, {len(id_i)} identification edges")

    log("precomputing full pairwise graph-distance matrix...")
    t0 = time.time()
    phi_dist = dijkstra(graph, indices=np.arange(n), directed=False)
    finite_max = phi_dist[np.isfinite(phi_dist)].max()
    phi_dist = np.where(np.isfinite(phi_dist), phi_dist, finite_max * 2)
    log(f"done in {time.time()-t0:.1f}s")

    true_dist_oracle = build_true_distance_oracle()

    log("precomputing FIXED train/test eval sets (true distance computed once each)...")
    train_eval = precompute_eval_set(train_rows, proprio, true_dist_oracle, N_EVAL_PAIRS, seed=100)
    test_eval = precompute_eval_set(test_rows, proprio, true_dist_oracle, N_EVAL_PAIRS, seed=101)
    log(f"train eval pairs: {len(train_eval[0])}, test eval pairs: {len(test_eval[0])}")

    results = {}
    for use_shaping, name in [(False, "baseline"), (True, "shaped")]:
        results[name] = []
        for seed in SEEDS:
            log(f"\n{'='*50}\n{name} seed={seed}\n{'='*50}")
            s_idx, next_idx, goal_idx, act_idx, done = build_her_tuples(
                ep_idx, step_idx, action, seed=seed, allowed_episode_ids=train_eps
            )
            log(f"n_tuples={len(s_idx)} (train episodes only)")
            history = run_condition(f"{name}_s{seed}", use_shaping, z, s_idx, next_idx, goal_idx,
                                     act_idx, done, action, phi_dist, train_eval, test_eval, d, seed)
            results[name].append(dict(seed=seed, history=history))

    out_path = OUT_DIR / "b3_convergence_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    log(f"\nwrote {out_path}")

    log("\n" + "=" * 60 + "\nCONVERGENCE SUMMARY\n" + "=" * 60)
    for name in ["baseline", "shaped"]:
        for split in ["spearman_train", "spearman_test"]:
            finals, deltas = [], []
            for r in results[name]:
                last, delta = check_convergence(r["history"], split)
                finals.append(last); deltas.append(delta)
            log(f"{name} {split}: final={np.mean(finals):.4f}+-{np.std(finals):.4f}  "
                f"plateau_delta(last10% vs prev10%)={np.mean(deltas):+.4f} "
                f"({'converged' if abs(np.mean(deltas)) < 0.01 else 'STILL MOVING'})")


if __name__ == "__main__":
    main()
