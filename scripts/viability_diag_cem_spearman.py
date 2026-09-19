"""Does V's ranking against the REAL oracle collapse specifically on the population a real
CEM search visits, as opposed to the tamer expert+noise+random pool viability_eval_audit.py
scored? viability_diag_hack.py's trace showed near-zero correlation between the critic's own
score and its own (predicted-latent) notion of distance during a live solve -- but that used
predicted-latent-L2 as a distance proxy, not the ground-truth oracle. This is the same
question asked properly: run a CEM search matching stable_worldmodel's actual CEMSolver
(same update rule, same num_samples/n_steps/topk -- see config/eval/solver/cem.yaml) under
each of two objectives (L2, viability), snapshot its population early/mid/late, execute a
subsample of each snapshot in the true simulator, query the same privileged recovery-time
oracle viability_inversion_audit.py uses, and report Spearman(score, -T) PER SNAPSHOT PER
OBJECTIVE. This directly tests whether V ranks well on a population that was itself searched
to please L2 but poorly on a population searched to please V (self-referential collapse) --
the mechanism viability_diag_hack.py's trace pointed at without a ground-truth check.

stable_worldmodel's CEMSolver samples candidates ~ N(mean, var) with NO clipping to the
action space and operates directly in the model's input space -- since actions there are
z-scored (mean 0/std 1 by construction) and var_scale=1.0, its N(0,1) init is already the
z-scored action distribution; nothing else needs de/normalizing except immediately before
stepping the real simulator, where raw = z*std+mean and IS clipped to the env's [-1,1] range.

Run:
    python scripts/viability_diag_cem_spearman.py --n-tasks 12 --n-per-snapshot 16
"""

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import torch
from scipy import stats as sps

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.envs import ENV_MECHANICS
from common.lewm_loader import load_lewm
from common.log_util import log
from planning_cost_gate import DEV, HISTORY, HORIZON, SKIP, latent_rollout, make_encode_frame, sample_problems
from viability_eval_audit import load_critic
from viability_inversion_audit import _init_oracle_worker, oracle_endpoint_job

ACTION_DIM_Z = SKIP * 2  # z-scored action block width for Push-T (SKIP real actions x 2 dims)
N_SAMPLES, N_STEPS, TOPK = 300, 30, 30  # matches config/eval/solver/cem.yaml exactly


@torch.no_grad()
def cem_search(model, z_hist, act_hist_pre, z_goal, cost_fn, snapshot_iters, seed):
    """One CEM run, matching stable_worldmodel.solver.cem.CEMSolver.solve() exactly (see
    CLAUDE.md/this file's docstring for the fidelity argument). Returns {iter: (N_SAMPLES,
    HORIZON, ACTION_DIM_Z) z-scored action populations} for the requested snapshot iters.
    `cost_fn(pred_terminal_emb) -> (N_SAMPLES,) cost, lower=better` scores one iteration's
    rolled-out terminal embeddings."""
    gen = torch.Generator(device=DEV).manual_seed(seed)
    mean = torch.zeros(HORIZON, ACTION_DIM_Z, device=DEV)
    var = torch.ones(HORIZON, ACTION_DIM_Z, device=DEV)
    snapshots = {}
    z_hist_b = z_hist.unsqueeze(0).expand(N_SAMPLES, -1, -1)
    for step in range(N_STEPS):
        cand = torch.randn(N_SAMPLES, HORIZON, ACTION_DIM_Z, generator=gen, device=DEV) * var + mean
        cand[0] = mean
        if step in snapshot_iters:
            snapshots[step] = cand.detach().clone()
        act_hist = torch.cat([act_hist_pre.unsqueeze(0).expand(N_SAMPLES, -1, -1), cand[:, :1]], dim=1)
        pred = latent_rollout(model, z_hist_b, act_hist, cand[:, 1:])[:, -1]  # (N_SAMPLES, D) numpy
        cost = cost_fn(torch.as_tensor(pred, device=DEV))
        topk_idx = torch.topk(cost, k=TOPK, largest=False).indices
        elite = cand[topk_idx]
        mean, var = elite.mean(0), elite.std(0).clamp_min(1e-3)
    return snapshots


def run(args):
    mech = ENV_MECHANICS["pusht"]
    root = Path(args.root).resolve()
    h5_path, ckpt_dir = mech.h5_path(root), mech.ckpt_dir(root)
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    log(f"[setup] h5={h5_path} device={DEV} n_tasks={args.n_tasks} offset={args.goal_offset} "
        f"h_eval={args.h} n_per_snapshot={args.n_per_snapshot} snapshots={args.snapshot_iters}")

    model = load_lewm(Path(ckpt_dir), device=DEV)
    encode = make_encode_frame(model)
    critic, cargs = load_critic(args.critic)
    env = mech.make_env()
    probs = sample_problems(h5_path, mech, args.goal_offset, args.n_tasks, args.seed)
    n = len(probs["start_row"])

    def l2_cost(pred, z_goal):
        return ((pred - z_goal) ** 2).sum(-1)

    def v_cost(pred, z_goal, h):
        h_t = torch.full((pred.shape[0],), float(h), device=DEV)
        mean_v, _ = critic.prob(pred, z_goal.expand_as(pred), h_t)
        return -torch.log(mean_v.clamp_min(1e-3))

    def rmst_cost(pred, z_goal, H):
        """D_RMST(z,g;H) = sum_{h=0}^{H-1} (1-V(z,g,h)) -- restricted-mean-survival-time-style
        aggregate over the WHOLE horizon grid instead of one fixed h. Tests whether the
        single-h collapse in cem_spearman's summary is an inference-time h-choice artifact
        (this should then rank well) or a genuine local-resolution gap shared across h
        (this should then collapse too, since it reuses the critic's coarse z-pathway)."""
        n_here = pred.shape[0]
        hs = torch.arange(H, device=DEV, dtype=torch.float32)
        p = pred.unsqueeze(1).expand(-1, H, -1).reshape(n_here * H, -1)
        g = z_goal.expand_as(pred).unsqueeze(1).expand(-1, H, -1).reshape(n_here * H, -1)
        h_t = hs[None, :].expand(n_here, -1).reshape(-1)
        mean_v, _ = critic.prob(p, g, h_t)
        return (1.0 - mean_v).view(n_here, H).sum(-1)

    pool = ProcessPoolExecutor(max_workers=args.oracle_workers, initializer=_init_oracle_worker,
                               mp_context=get_context("spawn")) if args.oracle_workers > 1 else None
    global _ORACLE_ENV
    import viability_inversion_audit as via
    if pool is None:
        via._ORACLE_ENV = env

    rows = []  # one row per (task, condition, snapshot_iter, candidate)
    rng = np.random.default_rng(args.seed + 555)
    for i in range(n):
        z_hist = torch.as_tensor(encode(probs["hist_pixels"][i]), device=DEV)
        z_goal_np = encode(probs["goal_pixels"][i:i + 1])[0]
        z_goal = torch.as_tensor(z_goal_np, device=DEV)
        mean_a, std_a = probs["act_mean"], probs["act_std"]
        past = torch.as_tensor(((probs["past_action"][i] - mean_a) / std_a).reshape(HISTORY - 1, ACTION_DIM_Z),
                               dtype=torch.float32, device=DEV)

        for cond, cost_fn in (("l2", lambda p: l2_cost(p, z_goal)),
                              ("viability", lambda p: v_cost(p, z_goal, args.h)),
                              ("rmst", lambda p: rmst_cost(p, z_goal, args.rmst_h))):
            snaps = cem_search(model, z_hist, past, z_goal,
                              cost_fn, args.snapshot_iters, seed=args.seed * 7919 + i * 13 + hash(cond) % 1000)
            for it, cand_z in snaps.items():
                idx = rng.choice(N_SAMPLES, size=min(args.n_per_snapshot, N_SAMPLES), replace=False)
                cand_z = cand_z[idx].cpu().numpy()                       # (n_per_snapshot, HORIZON, 10) z-scored blocks
                n_here = len(idx)
                # un-block (n, HORIZON, SKIP*2) -> (n, HORIZON*SKIP, 2) raw-per-step shape, matching
                # make_candidates' layout, so the inverse of sample_problems' normalize-then-reshape applies
                cand_raw = np.clip(
                    cand_z.reshape(n_here, HORIZON * SKIP, 2) * std_a + mean_a, -1.0, 1.0)

                # predicted terminal embedding for THIS exact subsample (for scoring)
                act_hist_np = np.concatenate(
                    [np.repeat(past.cpu().numpy()[None], len(idx), axis=0), cand_z[:, :1]], axis=1)
                pred = latent_rollout(model, np.repeat(z_hist.cpu().numpy()[None], len(idx), axis=0),
                                      act_hist_np, cand_z[:, 1:])[:, -1]
                l2 = ((pred - z_goal_np) ** 2).sum(-1)
                with torch.no_grad():
                    v_mean, v_std = critic.prob(torch.as_tensor(pred, device=DEV),
                                                z_goal.expand(len(idx), -1),
                                                torch.full((len(idx),), float(args.h), device=DEV))
                v_mean, v_std = v_mean.cpu().numpy(), v_std.cpu().numpy()
                with torch.no_grad():
                    rmst = rmst_cost(torch.as_tensor(pred, device=DEV), z_goal, args.rmst_h).cpu().numpy()

                # execute in the real simulator, query the oracle
                final_state, final_pixels = [], []
                for k, actions in enumerate(cand_raw):
                    seed_k = int(args.seed * 1_000_003 + i * 10_007 + hash((cond, it)) % 100_000 + k)
                    env.reset(seed=seed_k, options=mech.reset_options(probs["start_state"][i], probs["goal_state"][i]))
                    obs = None
                    for a in actions:
                        obs, reward, terminated, truncated, info = env.step(a)
                        if terminated:
                            break
                    final_state.append(np.asarray(obs["state"], dtype=np.float32))
                    final_pixels.append(env.render())
                final_state = np.stack(final_state)

                jobs = [(state, probs["goal_state"][i], via.oracle_seed(args.seed, args.goal_offset, i, k) + it * 97,
                        args.oracle_max_steps, args.oracle_trials, args.oracle_npop, args.oracle_niter,
                        args.oracle_elite_frac) for k, state in enumerate(final_state)]
                values = pool.map(oracle_endpoint_job, jobs) if pool else map(oracle_endpoint_job, jobs)
                for k, (steps, best) in enumerate(values):
                    rows.append(dict(task=i, cond=cond, iter=int(it), l2=float(l2[k]), v=float(v_mean[k]),
                                    v_std=float(v_std[k]), rmst=float(rmst[k]), oracle_min_steps=int(steps)))
        log(f"[task {i+1}/{n}] rows so far={len(rows)}")

    if pool:
        pool.shutdown()
    env.close()

    summary = {}
    for cond in ("l2", "viability", "rmst"):
        summary[cond] = {}
        for it in args.snapshot_iters:
            sub = [r for r in rows if r["cond"] == cond and r["iter"] == it]
            by_task = {}
            for r in sub:
                by_task.setdefault(r["task"], []).append(r)
            rho_v, rho_l2, rho_rmst, censored = [], [], [], []
            for t, rs in by_task.items():
                T = np.array([r["oracle_min_steps"] for r in rs], dtype=np.float64)
                if np.ptp(T) < 1e-9:
                    continue
                v = np.array([r["v"] for r in rs])
                l2 = np.array([r["l2"] for r in rs])
                rmst = np.array([r["rmst"] for r in rs])
                censored.append(float((T > args.oracle_max_steps).mean()))
                rv = sps.spearmanr(v, -T).statistic
                rl = sps.spearmanr(-l2, -T).statistic
                rr = sps.spearmanr(-rmst, -T).statistic  # lower RMST = more viable, like L2
                if np.isfinite(rv):
                    rho_v.append(rv)
                if np.isfinite(rl):
                    rho_l2.append(rl)
                if np.isfinite(rr):
                    rho_rmst.append(rr)
            summary[cond][str(it)] = dict(
                spearman_V_vs_negT=float(np.mean(rho_v)) if rho_v else float("nan"),
                spearman_negL2_vs_negT=float(np.mean(rho_l2)) if rho_l2 else float("nan"),
                spearman_negRMST_vs_negT=float(np.mean(rho_rmst)) if rho_rmst else float("nan"),
                n_informative_tasks=len(rho_v), censored_frac=float(np.mean(censored)) if censored else float("nan"),
            )
            log(f"[{cond} iter={it}] Spearman(V,-T)={summary[cond][str(it)]['spearman_V_vs_negT']:+.3f}  "
                f"Spearman(-L2,-T)={summary[cond][str(it)]['spearman_negL2_vs_negT']:+.3f}  "
                f"Spearman(-RMST,-T)={summary[cond][str(it)]['spearman_negRMST_vs_negT']:+.3f}  "
                f"n_tasks={summary[cond][str(it)]['n_informative_tasks']}")

    (out_dir / "rows.json").write_text(json.dumps(rows))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    log(f"[done] wrote {out_dir}/summary.json ({len(rows)} rows)")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=str(Path(__file__).resolve().parent.parent))
    p.add_argument("--out", default=None)
    p.add_argument("--critic", default=None)
    p.add_argument("--n-tasks", type=int, default=12)
    p.add_argument("--n-per-snapshot", type=int, default=16)
    p.add_argument("--snapshot-iters", type=int, nargs="+", default=[0, 14, 29])
    p.add_argument("--goal-offset", type=int, default=25)
    p.add_argument("--h", type=int, default=25, help="viability horizon, matches the live-rollout default")
    p.add_argument("--rmst-h", type=int, default=50, help="RMST horizon H, sums (1-V) over h=0..H-1; "
                   "defaults to the critic's h_max")
    p.add_argument("--oracle-max-steps", type=int, default=100)
    p.add_argument("--oracle-trials", type=int, default=2)
    p.add_argument("--oracle-npop", type=int, default=64)
    p.add_argument("--oracle-niter", type=int, default=5)
    p.add_argument("--oracle-elite-frac", type=float, default=0.15)
    p.add_argument("--oracle-workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=31)
    args = p.parse_args()
    args.out = args.out or str(Path(args.root) / "outputs" / "pusht" / "critic_training" / "diag_cem_spearman")
    args.critic = args.critic or str(Path(args.root) / "outputs" / "pusht" / "critic_training" / "critic_full_s0" / "critic.pt")
    run(args)


if __name__ == "__main__":
    main()
