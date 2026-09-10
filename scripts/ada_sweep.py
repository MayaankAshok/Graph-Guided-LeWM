"""Unified direct-SSH sweep orchestrator -- config-driven consolidation of
ada_pusht_b3_sweep.py/ada_pusht_actor_rollout_sweep.py (and, in spirit,
ada_direct_sweep.py/ada_b4ext_actor_rollout_sweep.py's Two-Room equivalents). Same pattern
throughout this project: a subprocess.Popen pool with an N_PARALLEL concurrency cap,
CUDA_VISIBLE_DEVICES cycled across N_GPUS, thread-limiting env vars, skip-if-done via each
combo's own checkpoint `done` flag. No SLURM -- run directly inside an existing interactive
Ada allocation.

Run as (on the target node, from the repo root, venv active):
    python scripts/ada_sweep.py --env pusht --mode datatiers
    python scripts/ada_sweep.py --env pusht --mode actor
    python scripts/ada_sweep.py --env tworoom --mode actor --tiers expert_10 expert_25

--mode datatiers launches one worker.py call per (tier, variant, seed).
--mode actor chains actor_train.py -> actor_rollout_eval.py(select=rho) ->
actor_rollout_eval.py(select=success) per combo, same as the original actor-rollout
orchestrators. --protocol picks the live-rollout protocol (see actor_rollout_utils.py;
default same_episode = the LeWM paper's), --selects which actor selections to evaluate,
and --eval-only skips combos whose actor checkpoint isn't already trained (so an
existing sweep can be re-scored under a new protocol without launching any training --
use --selects rho for that, since a checkpoint's success-selected actor was picked under
whatever protocol its own training used, recorded in its `eval_protocol` field).
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from actor_train import ckpt_path as actor_ckpt_path
from common.checkpoint_io import load_checkpoint
from datatiers import ckpt_path as datatiers_ckpt_path

ROOT = Path(__file__).resolve().parent.parent
PYTHON = ROOT / ".venv" / "bin" / "python"
VARIANTS = ["baseline", "auxphi"]
SEEDS = [0, 1, 2]

DEFAULT_TIERS = {
    "datatiers": lambda cfg_sizes: [f"expert_{n}" for n in cfg_sizes] + ["mixed", "mixed_large"],
}


def log(*a):
    print(*a, flush=True)


def env_cli(env, tier, variant, seed, extra=()):
    return [f"env={env}", f"tier={tier}", f"variant={variant}", f"seed={seed}", *extra]


class FakeCfg:
    """Minimal stand-in so ckpt_path()'s cfg.env.output_prefix/cfg.tier/etc. lookups work
    without spinning up a real Hydra config just to compute a path."""
    def __init__(self, env, tier, variant, seed):
        self.env = type("E", (), {"output_prefix": env, "name": env})()
        self.tier, self.variant, self.seed = tier, variant, seed
        self.run_tag = ""


def is_done_datatiers(env, tier, variant, seed):
    ckpt = load_checkpoint(datatiers_ckpt_path(FakeCfg(env, tier, variant, seed), tier, variant, seed))
    return ckpt is not None and ckpt.get("done", False)


def is_actor_trained(env, tier, variant, seed):
    ckpt = load_checkpoint(actor_ckpt_path(FakeCfg(env, tier, variant, seed), tier, variant, seed))
    return ckpt is not None and ckpt.get("done", False)


def is_done_actor(env, tier, variant, seed, eval_dir, protocol, selects):
    for sel in selects:
        if not (eval_dir / f"{tier}__{variant}__s{seed}__sel-{sel}__{protocol}.json").exists():
            return False
    return is_actor_trained(env, tier, variant, seed)


def launch(cmd_args, log_path, gpu_id, n_parallel, env_overrides=None):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    n_threads = str(max(1, os.cpu_count() // n_parallel // 2))
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[var] = n_threads
    if env_overrides:
        env.update(env_overrides)
    f = open(log_path, "w")
    p = subprocess.Popen(cmd_args, stdout=f, stderr=subprocess.STDOUT, cwd=str(ROOT), env=env)
    return p, f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", required=True, choices=["tworoom", "pusht", "reacher"])
    ap.add_argument("--mode", required=True, choices=["datatiers", "actor"])
    ap.add_argument("--tiers", nargs="+", default=None)
    ap.add_argument("--dataset-sizes", nargs="+", type=int, default=[10, 25, 50, 100])
    ap.add_argument("--n-parallel", type=int, default=3)
    ap.add_argument("--n-gpus", type=int, default=2)
    ap.add_argument("--poll-seconds", type=int, default=10)
    ap.add_argument("--protocol", default="same_episode", choices=["same_episode", "cross_episode"],
                    help="actor mode: live-rollout protocol for both training's rollout checks and the evals")
    ap.add_argument("--selects", nargs="+", default=["rho", "success"], choices=["rho", "success"],
                    help="actor mode: which selected actors to run actor_rollout_eval.py on")
    ap.add_argument("--n-episodes", type=int, default=100, help="actor mode: eval episodes per combo")
    ap.add_argument("--eval-only", action="store_true",
                    help="actor mode: only (re-)evaluate combos whose actor is already trained; never train")
    args = ap.parse_args()

    tiers = args.tiers or ([f"expert_{n}" for n in args.dataset_sizes] + ["mixed", "mixed_large"])
    combos = [(t, v, s) for t in tiers for v in VARIANTS for s in SEEDS]

    out_dir = ROOT / "outputs" / f"b3_{args.env}"
    if args.mode == "actor":
        eval_dir = out_dir / "actor" / "rollout_eval"
        log_dir = out_dir / "actor" / "sweep_logs"
        is_done = lambda t, v, s: is_done_actor(args.env, t, v, s, eval_dir, args.protocol, args.selects)
    else:
        log_dir = out_dir / "sweep_logs"
        is_done = lambda t, v, s: is_done_datatiers(args.env, t, v, s)
    log_dir.mkdir(parents=True, exist_ok=True)

    todo = [c for c in combos if not is_done(*c)]
    if args.mode == "actor" and args.eval_only:
        untrained = [c for c in todo if not is_actor_trained(args.env, *c)]
        for c in untrained:
            log(f"--eval-only: skipping {c[0]}/{c[1]}/s{c[2]} (no finished actor checkpoint)")
        todo = [c for c in todo if c not in untrained]
    log(f"env={args.env} mode={args.mode}"
        f"{' protocol=' + args.protocol + ' selects=' + ','.join(args.selects) if args.mode == 'actor' else ''}: "
        f"{len(combos)} total combos, {len(combos) - len(todo)} already done, {len(todo)} to run, "
        f"{args.n_parallel} at a time across {args.n_gpus} GPUs")

    def build_cmd(tier, variant, seed):
        if args.mode == "datatiers":
            return [str(PYTHON), "scripts/worker.py", *env_cli(args.env, tier, variant, seed)]
        # actor mode: chain train -> eval(rho) -> eval(success) in one shell command, same
        # pattern as the original ada_b4ext_actor_rollout_sweep.py/ada_pusht_actor_rollout_sweep.py
        proto = f"eval_protocol={args.protocol}"
        extra_flags = [proto]
        if tier in ("expert_300", "expert_1000"):
            if variant == "baseline":
                extra_flags.append("need_graph=false")
            elif variant == "auxphi":
                extra_flags.append("phi_mode=sparse")
        steps = []
        if not args.eval_only:
            steps.append([str(PYTHON), "scripts/actor_train.py", *env_cli(args.env, tier, variant, seed, extra=extra_flags)])
        for sel in args.selects:
            steps.append([str(PYTHON), "scripts/actor_rollout_eval.py",
                          *env_cli(args.env, tier, variant, seed,
                                   extra=[f"select={sel}", f"n_episodes={args.n_episodes}", *extra_flags])])
        shell_cmd = " && ".join(" ".join(c) for c in steps)
        return ["bash", "-c", shell_cmd]

    running = []
    next_gpu = 0
    idx = 0
    t0 = time.time()
    n_ok, n_failed = 0, 0

    while idx < len(todo) or running:
        while len(running) < args.n_parallel and idx < len(todo):
            tier, variant, seed = todo[idx]
            gpu = next_gpu % args.n_gpus
            next_gpu += 1
            log_path = log_dir / f"{tier}_{variant}_s{seed}.log"
            cmd = build_cmd(tier, variant, seed)
            p, fh = launch(cmd, log_path, gpu, args.n_parallel)
            log(f"[{time.time()-t0:.0f}s] launched {tier}/{variant}/s{seed} on gpu{gpu} "
                f"pid={p.pid} -> {log_path}")
            running.append(dict(proc=p, fh=fh, tier=tier, variant=variant, seed=seed, log_path=log_path))
            idx += 1

        time.sleep(args.poll_seconds)
        still_running = []
        for entry in running:
            ret = entry["proc"].poll()
            if ret is None:
                still_running.append(entry)
                continue
            entry["fh"].close()
            tier, variant, seed = entry["tier"], entry["variant"], entry["seed"]
            if ret == 0:
                n_ok += 1
                log(f"[{time.time()-t0:.0f}s] {tier}/{variant}/s{seed}: OK ({n_ok} done, {n_failed} failed, "
                    f"{len(todo)-idx} queued, {len(still_running)} running)")
            else:
                n_failed += 1
                log(f"[{time.time()-t0:.0f}s] {tier}/{variant}/s{seed}: FAILED (exit={ret}) -- "
                    f"see {entry['log_path']}")
        running = still_running

    log(f"\nALL DONE in {time.time()-t0:.0f}s -- {n_ok} ok, {n_failed} failed out of {len(todo)} run "
        f"({len(combos)-len(todo)} were already done)")


if __name__ == "__main__":
    main()
