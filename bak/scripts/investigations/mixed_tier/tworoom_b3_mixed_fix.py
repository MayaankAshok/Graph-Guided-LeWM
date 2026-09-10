"""Candidate fix for the mixed-tier shaping failure, tested on the mixed tier ONLY first
(per instruction -- it's the worst comparison and the most realistic tier, so verify a fix
works there before touching anything else).

What tworoom_b3_mixed_diagnosis.py ruled out and found (see outputs/b3_tworoom/b3_mixed_diagnosis.json
and memory tworoom-b3-datatier-sweep):
  - NOT graph pollution: Spearman(true, graph-dist) on identical expert-only pairs is the
    SAME whether the graph also contains noisy nodes or not (0.9688 vs 0.9666). Adding
    noisy nodes doesn't create bad Dijkstra shortcuts for expert-only queries.
  - NOT disconnection: 1 connected component, 100% of both expert- and noisy-origin nodes
    reachable, noisy-origin nodes have comparable identification-edge degree to expert ones.
  - NOT a shaped-reward blowup: no outliers (|increment|>5 fraction = 0 in both groups).
  - IS a population-conditional scale/consistency mismatch: the shaping increment
    (gamma*Phi(next)-Phi(s)) has mean=0.458 std=0.681 on tuples from real expert episodes
    vs mean=0.751 std=0.466 on tuples from the freshly-collected noisy episodes -- i.e.
    real "expert" trajectories track monotonic graph-distance progress toward HER-sampled
    goals LESS consistently than the noisy-collector's direct waypoint-following behavior,
    even though Phi itself is accurate (per the point above). Also: mixed's baseline peak
    (0.340) beats expert_50's baseline (0.272) -- extra heterogeneous data helps sparse-
    reward TD -- while mixed's shaped peak (0.205) is flat-to-worse than expert_50's shaped
    (0.212) -- shaped does NOT get the same benefit from the extra data.

Candidate fix tested here: shaping-strength coefficient lambda, reward = reward +
lambda*(gamma*Phi(next)-Phi(s)), lambda in SHAPING_LAMBDAS. This is a standard, well-
grounded variance/influence control (not a population-conditional hack -- it doesn't need
to know which episodes are "noisy", so it's usable in a real deployment where that label
doesn't exist) and preserves the exact potential-based telescoping-sum structure for any
fixed lambda (Phi' := lambda*Phi is itself a valid potential). If shrinking lambda lets
shaped stop losing to baseline (or start winning) on the mixed tier, that's evidence the
mechanism is "shaping's per-step variance/inconsistency overwhelms its signal when the
data is small/heterogeneous," fixable by turning it down; if it does NOT help even at
small lambda, that points away from a scale/variance story and toward something more
fundamental about applying potential-based shaping across behaviorally-distinct sub-
populations.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy import stats as sps

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from tworoom_b0_graph_gate import DEV
from tworoom_b3_datatiers import (
    BATCH_SIZE, EMA_TAU, EVAL_EVERY, GAMMA, HIDDEN, N_STEPS, SEEDS, TAU_EXPECTILE,
    build_tier_setups, ckpt_path, load_checkpoint, save_checkpoint,
)
from tworoom_b3_gciql_shaping import MLP, build_her_tuples, ema_update, log

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "outputs" / "b3_tworoom"

SHAPING_LAMBDAS = [0.1, 0.15, 0.3, 0.5]  # 0.3's result: mean peak 0.311 vs baseline's 0.340 and
# original (lambda=1) shaped's 0.205 -- closes ~79% of the gap. Bracketing to find the best point.
TIER = "mixed"


def run_condition_resumable(tier, name, shaping_lambda, setup, seed):
    path = ckpt_path(tier, name, seed)
    z, action, phi_dist = setup["z"], setup["action"], setup["phi_dist"]
    d = setup["d"]

    ckpt = load_checkpoint(path)
    if ckpt is not None and ckpt.get("done"):
        log(f"[{tier}/{name}/s{seed}] already complete ({ckpt['step']} steps) -- skipping")
        return ckpt["history"]

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
    history = []
    start_step = 1

    if ckpt is not None:
        v_net.load_state_dict(ckpt["v_net"]); v_target.load_state_dict(ckpt["v_target"])
        q_net.load_state_dict(ckpt["q_net"]); q_target.load_state_dict(ckpt["q_target"])
        opt.load_state_dict(ckpt["opt"])
        history = ckpt["history"]
        start_step = ckpt["step"] + 1
        rng.bit_generator.state = ckpt["numpy_rng"]
        torch.set_rng_state(ckpt["torch_rng"])
        if torch.cuda.is_available() and ckpt.get("torch_cuda_rng") is not None:
            torch.cuda.set_rng_state(ckpt["torch_cuda_rng"])
        log(f"[{tier}/{name}/s{seed}] resuming from step {start_step}/{N_STEPS} "
            f"({len(history)} eval points so far)")
    else:
        log(f"[{tier}/{name}/s{seed}] starting fresh, n_tuples={n_tuples}, lambda={shaping_lambda}")

    if start_step > N_STEPS:
        return history

    t0 = time.time()
    for step in range(start_step, N_STEPS + 1):
        batch = rng.integers(0, n_tuples, BATCH_SIZE)
        s, nx, g, a_i, dn = s_idx[batch], next_idx[batch], goal_idx[batch], act_idx[batch], done_arr[batch]

        zs, zn, zg = z_t[s], z_t[nx], z_t[g]
        a = act_t[a_i]
        mask = (~torch.from_numpy(dn).to(DEV)).float().unsqueeze(-1)
        reward = -mask

        phi_s = torch.from_numpy(-phi_dist[s, g]).float().to(DEV).unsqueeze(-1)
        phi_n = torch.from_numpy(-phi_dist[nx, g]).float().to(DEV).unsqueeze(-1)
        reward = reward + shaping_lambda * (GAMMA * phi_n - phi_s)

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
            is_done = step == N_STEPS
            save_checkpoint(path, step, v_net, v_target, q_net, q_target, opt, history, rng, is_done)
            if step % (EVAL_EVERY * 20) == 0 or step == 1:
                log(f"[{tier}/{name}/s{seed}] step={step} vloss={value_loss.item():.3f} "
                    f"closs={critic_loss.item():.3f} train_rho={rho_train:.4f} test_rho={rho_test:.4f} "
                    f"elapsed={time.time()-t0:.1f}s")

    return history


def peak_stats(history):
    i = max(range(len(history)), key=lambda k: history[k]["spearman_test"])
    return history[i]


def main():
    log(f"device={DEV}")
    setup = build_tier_setups([TIER])[TIER]

    results = {}
    for lam in SHAPING_LAMBDAS:
        name = f"shaped_lam{lam}"
        results[name] = []
        for seed in SEEDS:
            history = run_condition_resumable(TIER, name, lam, setup, seed)
            results[name].append(dict(seed=seed, history=history))

    # pull in the already-computed baseline / original shaped from the main sweep for comparison
    existing = json.loads((OUT_DIR / f"b3_datatiers_{TIER}_results.json").read_text())

    out_path = OUT_DIR / "b3_mixed_fix_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    log(f"wrote {out_path}")

    log("\n" + "=" * 70 + f"\nSUMMARY: peak held-out-test Spearman on '{TIER}' tier\n" + "=" * 70)
    for name in ["baseline", "shaped"]:
        peaks = [peak_stats(r["history"])["spearman_test"] for r in existing[name]]
        log(f"  {name} (from main sweep): mean peak = {np.mean(peaks):.4f} +- {np.std(peaks):.4f}  "
            f"({[f'{p:.4f}' for p in peaks]})")
    for lam in SHAPING_LAMBDAS:
        name = f"shaped_lam{lam}"
        peaks = [peak_stats(r["history"])["spearman_test"] for r in results[name]]
        log(f"  {name}: mean peak = {np.mean(peaks):.4f} +- {np.std(peaks):.4f}  "
            f"({[f'{p:.4f}' for p in peaks]})")


if __name__ == "__main__":
    main()
