"""Extends the live-rollout actor campaign (tworoom_b3_actor_train.py /
tworoom_b3_actor_rollout_eval.py) to the B4-extension distance sources (transonly, k4, k8 --
see tworoom-b4-extension-edge-cap-rl memory / outputs/b4_tworoom/edge_ablation_results.json),
which so far were only checked on the Spearman-vs-oracle proxy metric, not real task success.
Mirrors exactly what tworoom_b3_actor_rollout_eval.py's campaign already did for the original
baseline/graph conditions -- same actor (AWR), same dual peak-tracking (rho vs success), same
final 100-episode evaluation at both peaks.

Scope: only the "auxphi" variant needs a per-source rerun (baseline doesn't depend on any
distance source at all -- its results from the original 60-combo actor sweep already cover
every tier/seed and don't need repeating here). 4 tiers (expert_50, expert_100, mixed,
mixed_large) x 3 sources (transonly, k4, k8) x 5 seeds = 60 runs, chaining train -> eval(rho)
-> eval(success) per combo, same direct-SSH orchestration pattern as ada_direct_sweep.py and
ada_b4_edge_ablation_sweep.py (no SLURM, N_PARALLEL processes, thread-capped, skip-if-done).

Run this ON the target node (e.g. via `ssh adag` or directly inside an existing tmux session
there), from the repo root, with the venv active:

    python scripts/ada_b4ext_actor_rollout_sweep.py
"""

import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tworoom_b3_actor_train import ckpt_path
from tworoom_b3_datatiers import load_checkpoint

ROOT = Path(__file__).resolve().parent.parent
PYTHON = ROOT / ".venv" / "bin" / "python"
WORKER_TRAIN = Path(__file__).resolve().parent / "tworoom_b3_actor_train.py"
WORKER_EVAL = Path(__file__).resolve().parent / "tworoom_b3_actor_rollout_eval.py"
EVAL_DIR = ROOT / "outputs" / "b3_tworoom" / "actor" / "rollout_eval"
LOG_DIR = ROOT / "outputs" / "b3_tworoom" / "actor" / "sweep_logs_b4ext"
LOG_DIR.mkdir(parents=True, exist_ok=True)

TIERS = ["expert_50", "expert_100", "mixed", "mixed_large"]
SOURCES = ["transonly", "k4", "k8"]
SEEDS = [0, 1, 2, 3, 4]
VARIANT = "auxphi"
N_PARALLEL = 3
N_GPUS = 2
POLL_SECONDS = 10


def log(*a):
    print(*a, flush=True)


def is_done(tier, source, seed):
    rho_path = EVAL_DIR / f"{tier}__{VARIANT}_{source}__s{seed}__sel-rho.json"
    success_path = EVAL_DIR / f"{tier}__{VARIANT}_{source}__s{seed}__sel-success.json"
    if not (rho_path.exists() and success_path.exists()):
        return False
    ckpt = load_checkpoint(ckpt_path(tier, VARIANT, seed, source))
    return ckpt is not None and ckpt.get("done", False)


def launch(tier, source, seed, gpu_id):
    log_path = LOG_DIR / f"{tier}_{source}_s{seed}.log"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    n_threads = str(max(1, os.cpu_count() // N_PARALLEL // 2))
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[var] = n_threads
    cmd = (
        f"{PYTHON} {WORKER_TRAIN} --tier {tier} --variant {VARIANT} --distance-source {source} "
        f"--seed {seed} && "
        f"{PYTHON} {WORKER_EVAL} --tier {tier} --variant {VARIANT} --distance-source {source} "
        f"--seed {seed} --select rho --n-episodes 100 && "
        f"{PYTHON} {WORKER_EVAL} --tier {tier} --variant {VARIANT} --distance-source {source} "
        f"--seed {seed} --select success --n-episodes 100"
    )
    f = open(log_path, "w")
    p = subprocess.Popen(["bash", "-c", cmd], stdout=f, stderr=subprocess.STDOUT, cwd=str(ROOT), env=env)
    return p, f, log_path


def main():
    combos = [(t, s, seed) for t in TIERS for s in SOURCES for seed in SEEDS]
    todo = [c for c in combos if not is_done(*c)]
    log(f"{len(combos)} total combos, {len(combos) - len(todo)} already done, {len(todo)} to run, "
        f"{N_PARALLEL} at a time across {N_GPUS} GPUs")

    running = []
    next_gpu = 0
    idx = 0
    t0 = time.time()
    n_ok, n_failed = 0, 0

    while idx < len(todo) or running:
        while len(running) < N_PARALLEL and idx < len(todo):
            tier, source, seed = todo[idx]
            gpu = next_gpu % N_GPUS
            next_gpu += 1
            p, fh, log_path = launch(tier, source, seed, gpu)
            log(f"[{time.time()-t0:.0f}s] launched {tier}/{source}/s{seed} on gpu{gpu} "
                f"pid={p.pid} -> {log_path}")
            running.append(dict(proc=p, fh=fh, tier=tier, source=source, seed=seed, log_path=log_path))
            idx += 1

        time.sleep(POLL_SECONDS)
        still_running = []
        for entry in running:
            ret = entry["proc"].poll()
            if ret is None:
                still_running.append(entry)
                continue
            entry["fh"].close()
            tier, source, seed = entry["tier"], entry["source"], entry["seed"]
            if ret == 0:
                n_ok += 1
                log(f"[{time.time()-t0:.0f}s] {tier}/{source}/s{seed}: OK ({n_ok} done, {n_failed} failed, "
                    f"{len(todo)-idx} queued, {len(still_running)} running)")
            else:
                n_failed += 1
                log(f"[{time.time()-t0:.0f}s] {tier}/{source}/s{seed}: FAILED (exit={ret}) -- "
                    f"see {entry['log_path']}")
        running = still_running

    log(f"\nALL DONE in {time.time()-t0:.0f}s -- {n_ok} ok, {n_failed} failed out of {len(todo)} run "
        f"({len(combos)-len(todo)} were already done)")


if __name__ == "__main__":
    main()
