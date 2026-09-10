"""Fills the "GCIVL and live-rollout success rates remain open" gap flagged since B3's
first pass: everything in B3/B4 measured Spearman(V(s,g), true_distance), a value-ranking
proxy. It says nothing about whether a POLICY extracted from that value function actually
reaches the goal, because no policy was ever trained -- only V(s,g) and Q(s,a,g). This adds
an actor network trained via advantage-weighted regression (AWR), the standard GCIQL/IQL
policy-extraction recipe:

    weight = exp(beta * (Q(s,a,g) - V(s,g))), clipped
    actor_loss = weight * ||actor(s,g) - a||^2   (weighted behavior cloning toward the
                                                    real action taken in the HER tuple)

V/Q training is otherwise UNCHANGED from tworoom_b3_mixed_auxphi.py -- baseline's plain
-1/0 TD objective, optionally plus the mode-2 auxiliary-Phi-regression loss
(variant="auxphi"). The actor is a pure addition; its gradients don't touch V/Q (the
advantage is computed under torch.no_grad()).

Per-instruction scope: expert_100 and mixed_large tiers, baseline vs auxphi (mode 2, graph
distance source), single seed to start (more seeds via repeated invocation with --seed).

Resumability: reuses tworoom_b3_datatiers.save_checkpoint/load_checkpoint verbatim (same
write-verify + one-generation-backup + fsync hardening as every other script in this
campaign), via the `extra` payload field for the actor's weights and TWO peak-step actor
SNAPSHOTS (kept alongside the resumable "latest" state, not instead of it -- resume still
always uses the latest state; the snapshots are additional data in the same file, exactly
like tworoom_b3_mixed_peak_probe.py did once as a one-off, just made resumable here).

TWO independently-tracked peaks, not one -- there's no reason they have to coincide:
  - peak_step/peak_rho/peak_actor_state: the original criterion, V's peak held-out-test
    Spearman (cheap, a forward pass every eval_every steps).
  - peak_success_step/peak_success_rate/peak_success_actor_state: a SEPARATE peak from a
    small (ROLLOUT_EVAL_EPISODES), FIXED-pair-subset live rollout check every
    ROLLOUT_EVAL_EVERY steps, using the actor's OWN task-success rate directly. V's ranking
    accuracy is only a proxy for "is the AWR-derived actor good right now"; the first
    single-seed live-rollout run (before this was added) showed a much smaller success-rate
    gain than the Spearman gap suggested, raising exactly this question. Both peaks matter:
    tworoom_b3_actor_rollout_eval.py's --select flag picks which one to do the final,
    larger-sample evaluation on.

n_steps default is 20,000, truncated from the campaign-wide 50,000 -- every peak observed
in the first single-seed pass (V's Spearman peak, all 4 tier/variant combos) landed at or
below 12,650 steps, so 20k leaves ~58% margin without paying for 30k steps of pure
post-peak decay (and the periodic rollout checks that would run through it for nothing).

Logging: every eval step (50) is recorded to history; console print every 250 steps (more
frequent than the 1000-step cadence used elsewhere in this campaign, per instruction).
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy import stats as sps

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tworoom_b0_graph_gate import DEV
from tworoom_b3_datatiers import (
    BATCH_SIZE, EMA_TAU, EVAL_EVERY, GAMMA, HIDDEN, TAU_EXPECTILE,
    build_tier_setups, load_checkpoint, save_checkpoint,
)
from tworoom_b3_gciql_shaping import MLP, build_her_tuples, ema_update, log
from tworoom_b3_datatiers import _build_setup
from tworoom_b3_mixed_large import TIER as MIXED_LARGE_TIER, _build_mixed_large_arrays
from tworoom_actor_rollout_utils import make_encoder, make_env, run_episodes, summarize

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "outputs" / "b3_tworoom" / "actor"
CKPT_DIR = OUT_DIR / "checkpoints"
CKPT_DIR.mkdir(parents=True, exist_ok=True)

BETA = 3.0              # AWR temperature (standard IQL/GCIQL default)
AWR_WEIGHT_CLIP = 100.0
AUX_LAMBDA = 0.3        # mode-2 weight, unchanged from tworoom_b3_mixed_auxphi.py
CONSOLE_EVERY_EVALS = 5  # print every 5*EVAL_EVERY = 250 steps

# Periodic in-training rollout check, tracked as an INDEPENDENT peak from V's Spearman peak.
# There's no a priori reason the actor's own best rollout success rate lands at the same
# step as V's best ranking accuracy -- V's Spearman peak is a proxy for "is V a good ranker
# right now", not "is the AWR-derived actor good right now". A small, fixed (same pairs every
# check, for a fair step-to-step trend) subset keeps the added cost bounded: ~150 steps/
# episode worst case, ~22ms/step observed -> ~12 episodes/check is a few tens of seconds,
# tractable at a several-thousand-step interval over a 50k-step budget.
ROLLOUT_EVAL_EVERY = 2500
ROLLOUT_EVAL_EPISODES = 12
ROLLOUT_EVAL_PAIR_SEED = 424242


def ckpt_path(tier, variant, seed, distance_source="graph"):
    # distance_source only matters for variant="auxphi"; "graph" (the original, already-run
    # condition) and "baseline" both keep the original filename unchanged so existing
    # checkpoints from the first actor-rollout campaign are still found and reused.
    suffix = "" if (variant == "baseline" or distance_source == "graph") else f"_{distance_source}"
    return CKPT_DIR / f"{tier}__{variant}{suffix}_actor__s{seed}.pt"


def run_actor_condition(tier, variant, setup, seed, n_steps, eval_every,
                         rollout_eval_every=ROLLOUT_EVAL_EVERY, rollout_eval_episodes=ROLLOUT_EVAL_EPISODES,
                         distance_source="graph", dist_matrix=None):
    path = ckpt_path(tier, variant, seed, distance_source)
    z, action = setup["z"], setup["action"]
    phi_dist = dist_matrix if dist_matrix is not None else setup["phi_dist"]
    d = setup["d"]
    act_dim = action.shape[1]
    tag = f"{tier}/{variant}" + ("" if (variant == "baseline" or distance_source == "graph") else f"_{distance_source}")

    ckpt = load_checkpoint(path)
    if ckpt is not None and ckpt.get("done"):
        log(f"[{tag}/s{seed}] already complete ({ckpt['step']} steps, "
            f"peak_rho={ckpt.get('peak_rho'):.4f}@{ckpt.get('peak_step')}, "
            f"peak_success_rate={ckpt.get('peak_success_rate')}@{ckpt.get('peak_success_step')}) "
            f"-- skipping")
        return ckpt

    torch.manual_seed(seed)
    z_t = torch.from_numpy(z).to(DEV)
    act_t = torch.from_numpy(action).float().to(DEV)

    v_net = MLP(2 * d, HIDDEN).to(DEV)
    v_target = MLP(2 * d, HIDDEN).to(DEV)
    q_net = MLP(2 * d + act_dim, HIDDEN).to(DEV)
    q_target = MLP(2 * d + act_dim, HIDDEN).to(DEV)
    actor_net = MLP(2 * d, HIDDEN, out_dim=act_dim).to(DEV)
    v_target.load_state_dict(v_net.state_dict())
    q_target.load_state_dict(q_net.state_dict())

    from stable_worldmodel.wm.gcrl.module import ExpectileLoss
    expectile_loss = ExpectileLoss(tau=TAU_EXPECTILE)
    opt = torch.optim.Adam(
        list(v_net.parameters()) + list(q_net.parameters()) + list(actor_net.parameters()), lr=3e-4)

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
    best_rho, best_step, best_actor_state = -2.0, None, None
    best_success_rate, best_success_step, best_success_actor_state = -1.0, None, None

    if ckpt is not None:
        v_net.load_state_dict(ckpt["v_net"]); v_target.load_state_dict(ckpt["v_target"])
        q_net.load_state_dict(ckpt["q_net"]); q_target.load_state_dict(ckpt["q_target"])
        actor_net.load_state_dict(ckpt["actor_net"])
        opt.load_state_dict(ckpt["opt"])
        history = ckpt["history"]
        start_step = ckpt["step"] + 1
        rng.bit_generator.state = ckpt["numpy_rng"]
        torch.set_rng_state(ckpt["torch_rng"])
        if torch.cuda.is_available() and ckpt.get("torch_cuda_rng") is not None:
            torch.cuda.set_rng_state(ckpt["torch_cuda_rng"])
        best_rho = ckpt.get("peak_rho", -2.0)
        best_step = ckpt.get("peak_step")
        best_actor_state = ckpt.get("peak_actor_state")
        best_success_rate = ckpt.get("peak_success_rate", -1.0)
        best_success_step = ckpt.get("peak_success_step")
        best_success_actor_state = ckpt.get("peak_success_actor_state")
        log(f"[{tag}/s{seed}] resuming from step {start_step}/{n_steps} "
            f"({len(history)} eval points so far, best_rho={best_rho:.4f}@{best_step}, "
            f"best_success_rate={best_success_rate}@{best_success_step})")
    else:
        log(f"[{tag}/s{seed}] starting fresh, n_tuples={n_tuples}, variant={variant}, "
            f"beta={BETA}, n_steps={n_steps}")

    if start_step > n_steps:
        return ckpt

    # Built once, reused for every periodic in-training rollout check (and the fixed pair
    # subset is the SAME across all checks within this run, so success-rate is comparable
    # step-to-step rather than confounded by which pairs got sampled).
    rollout_env = make_env()
    encode_frame = make_encoder()
    pair_rng = np.random.default_rng(ROLLOUT_EVAL_PAIR_SEED)
    n_test_pairs = len(test_ei)
    rollout_pair_idx = pair_rng.choice(n_test_pairs, size=min(rollout_eval_episodes, n_test_pairs), replace=False)

    t0 = time.time()
    for step in range(start_step, n_steps + 1):
        batch = rng.integers(0, n_tuples, BATCH_SIZE)
        s, nx, g, a_i, dn = s_idx[batch], next_idx[batch], goal_idx[batch], act_idx[batch], done_arr[batch]

        zs, zn, zg = z_t[s], z_t[nx], z_t[g]
        a = act_t[a_i]
        mask = (~torch.from_numpy(dn).to(DEV)).float().unsqueeze(-1)
        reward = -mask

        sg = torch.cat([zs, zg], dim=-1)
        ng = torch.cat([zn, zg], dim=-1)
        sag = torch.cat([zs, a, zg], dim=-1)

        with torch.no_grad():
            q = q_target(sag)
        v = v_net(sg)
        value_loss = expectile_loss(v, q.detach())

        aux_loss = torch.tensor(0.0, device=DEV)
        if variant == "auxphi":
            phi_sg = torch.from_numpy(phi_dist[s, g]).float().to(DEV).unsqueeze(-1)
            pseudo_v = -(1 - GAMMA ** phi_sg) / (1 - GAMMA)
            aux_loss = ((v - pseudo_v) ** 2).mean()

        with torch.no_grad():
            next_v = v_target(ng)
            q_tgt = reward + GAMMA * mask * next_v
        q_pred = q_net(sag)
        critic_loss = ((q_pred - q_tgt) ** 2).mean()

        # AWR actor loss. Advantage computed under no_grad from the CURRENT (not target)
        # V/Q -- standard IQL-AWR convention -- and detached so this loss's backward pass
        # only updates actor_net, never V/Q's own parameters.
        with torch.no_grad():
            q_sa_for_actor = q_net(sag)
            v_s_for_actor = v_net(sg)
            advantage = q_sa_for_actor - v_s_for_actor
            weight = torch.exp(BETA * advantage).clamp(max=AWR_WEIGHT_CLIP)
        a_pred = torch.tanh(actor_net(sg))
        actor_loss = (weight * ((a_pred - a) ** 2).sum(-1, keepdim=True)).mean()

        loss = value_loss + critic_loss + (AUX_LAMBDA * aux_loss if variant == "auxphi" else 0.0) + actor_loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        ema_update(v_target, v_net, EMA_TAU)
        ema_update(q_target, q_net, EMA_TAU)

        if step % eval_every == 0 or step == 1:
            with torch.no_grad():
                v_train = v_net(train_zsg).squeeze(-1).cpu().numpy()
                v_test = v_net(test_zsg).squeeze(-1).cpu().numpy()
            rho_train, _ = sps.spearmanr(train_true_d, -v_train)
            rho_test, _ = sps.spearmanr(test_true_d, -v_test)
            if rho_test > best_rho:
                best_rho, best_step = rho_test, step
                best_actor_state = {k: t.detach().cpu().clone() for k, t in actor_net.state_dict().items()}
            history.append(dict(step=step, value_loss=float(value_loss.item()),
                                 critic_loss=float(critic_loss.item()), actor_loss=float(actor_loss.item()),
                                 aux_loss=float(aux_loss.item()),
                                 spearman_train=float(rho_train), spearman_test=float(rho_test)))
            is_done = step == n_steps

            if step % rollout_eval_every == 0 or step == 1:
                actor_net.eval()
                episodes = run_episodes(actor_net, rollout_env, encode_frame, setup, rollout_pair_idx,
                                         seed=ROLLOUT_EVAL_PAIR_SEED)
                actor_net.train()
                roll_summary = summarize(episodes)
                sr = roll_summary["success_rate"]
                if sr > best_success_rate:
                    best_success_rate, best_success_step = sr, step
                    best_success_actor_state = {k: t.detach().cpu().clone() for k, t in actor_net.state_dict().items()}
                log(f"[{tag}/s{seed}] step={step} ROLLOUT CHECK: success_rate={sr:.3f} "
                    f"({sum(e['success'] for e in episodes)}/{len(episodes)}) "
                    f"best_success_rate={best_success_rate:.3f}@{best_success_step}")

            save_checkpoint(path, step, v_net, v_target, q_net, q_target, opt, history, rng, is_done,
                             extra=dict(actor_net=actor_net.state_dict(), peak_step=best_step,
                                        peak_rho=best_rho, peak_actor_state=best_actor_state,
                                        peak_success_step=best_success_step, peak_success_rate=best_success_rate,
                                        peak_success_actor_state=best_success_actor_state))
            if (step // eval_every) % CONSOLE_EVERY_EVALS == 0 or step == 1:
                log(f"[{tag}/s{seed}] step={step} vloss={value_loss.item():.3f} "
                    f"closs={critic_loss.item():.3f} aloss={actor_loss.item():.3f} "
                    f"train_rho={rho_train:.4f} test_rho={rho_test:.4f} "
                    f"best_rho={best_rho:.4f}@{best_step} elapsed={time.time()-t0:.1f}s")

    return load_checkpoint(path)


def get_setup(tier):
    if tier == MIXED_LARGE_TIER:
        arrays = _build_mixed_large_arrays()
        return _build_setup(tier, *arrays)
    return build_tier_setups([tier])[tier]


# Truncated from the campaign-wide default of 50,000: every peak observed so far (V's
# Spearman peak, across baseline/auxphi on both expert_100 and mixed_large) landed at or
# below 12,650 steps -- 20,000 leaves comfortable margin (>7,350 steps / ~58%) without
# paying for 30,000 further steps of pure post-peak decay and periodic rollout checks on
# an already-decaying actor.
ACTOR_N_STEPS_DEFAULT = 20_000


def _is_capped_source(source):
    return source == "transonly" or (source.startswith("k") and source[1:].isdigit())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", required=True, choices=["expert_10", "expert_25", "expert_50",
                                                        "expert_100", "mixed", "mixed_large"])
    ap.add_argument("--variant", required=True, choices=["baseline", "auxphi"])
    ap.add_argument("--distance-source", default="graph",
                     help="only matters for --variant auxphi: 'graph' (default, the original "
                          "full-graph condition), 'transonly', or 'k<N>' (e.g. k4, k8) -- see "
                          "tworoom_b3_auxphi_worker.py / the B4-extension ablation")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-steps", type=int, default=ACTOR_N_STEPS_DEFAULT)
    ap.add_argument("--eval-every", type=int, default=EVAL_EVERY)
    ap.add_argument("--rollout-eval-every", type=int, default=ROLLOUT_EVAL_EVERY)
    ap.add_argument("--rollout-eval-episodes", type=int, default=ROLLOUT_EVAL_EPISODES)
    args = ap.parse_args()
    if args.distance_source != "graph" and not _is_capped_source(args.distance_source):
        raise ValueError(f"unknown --distance-source '{args.distance_source}' (expected graph, "
                          "transonly, or k<N>)")

    log(f"device={DEV}")
    setup = get_setup(args.tier)

    dist_matrix = None
    if args.variant == "auxphi" and args.distance_source != "graph":
        from tworoom_b3_auxphi_worker import B4_CACHE_DIR
        from tworoom_graph_variants import get_ablation_dist_matrix
        dist_matrix = get_ablation_dist_matrix(args.tier, args.distance_source, setup, B4_CACHE_DIR)

    ckpt = run_actor_condition(args.tier, args.variant, setup, args.seed, args.n_steps, args.eval_every,
                                args.rollout_eval_every, args.rollout_eval_episodes,
                                distance_source=args.distance_source, dist_matrix=dist_matrix)
    log(f"[{args.tier}/{args.variant}/s{args.seed}] done: peak_rho={ckpt.get('peak_rho'):.4f}"
        f"@{ckpt.get('peak_step')}, peak_success_rate={ckpt.get('peak_success_rate')}"
        f"@{ckpt.get('peak_success_step')}, "
        f"checkpoint={ckpt_path(args.tier, args.variant, args.seed, args.distance_source)}")


if __name__ == "__main__":
    main()
