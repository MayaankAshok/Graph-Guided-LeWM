"""Extends the B4 distance-source ablation (tworoom_b3_auxphi_worker.py /
tworoom_b3_mixed_auxphi.py's mode-2 auxiliary-regression loop) with two new arms motivated
by the local mechanism-probe + edge-cap-sweep campaign (see
outputs/graph_variants/mechanism_probe_results.json and edge_cap_sweep_results.json):

  transonly -- identification edges dropped entirely (pure transition-chain graph). The
    static Spearman probe found same-episode ranking is measurably WORSE without them
    (+0.05 to +0.14 Spearman gap across tiers) -- this checks whether that gap survives at
    the real RL-training level, the way the graph-vs-euclidean gap did in the original B4.
  k4, k8 -- identification edges capped to each node's top-k FAISS-HNSW neighbors instead
    of every threshold-qualifying one. The static probe found this costs ZERO Spearman
    quality (often slightly better) while cutting edge count 3-12x -- this is the fix for
    B1's O(n^1.96) edge-count blocker, checked here at the RL-training level before trusting
    it ahead of B5's Push-T scale-up.

Scope (per instruction, narrower than the original 6-tier x 3-seed B4 run): 4 tiers
(expert_50, expert_100, mixed, mixed_large -- skipping the two smallest expert tiers, where
the graph-vs-euclidean signal was already weakest), 5 seeds, 3 processes in parallel across
however many GPUs this node has. Direct-SSH orchestration (no SLURM sbatch/srun), same
pattern as ada_direct_sweep.py: subprocess.Popen pool, CUDA_VISIBLE_DEVICES cycling,
thread-limiting env vars (unconstrained BLAS/OpenMP defaulted to using every visible core
per process last time, causing severe oversubscription at N_PARALLEL>1), skip-if-done based
on checkpoint state so this is safe to re-run after an interruption.

Run this ON the target node (e.g. via `ssh adag` or directly inside an existing tmux
session there), from the repo root, with the venv active:

    python scripts/ada_b4_edge_ablation_sweep.py
"""

import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tworoom_b3_datatiers import ckpt_path, load_checkpoint
from tworoom_b3_auxphi_worker import B4_CKPT_DIR

ROOT = Path(__file__).resolve().parent.parent
PYTHON = ROOT / ".venv" / "bin" / "python"
WORKER = Path(__file__).resolve().parent / "tworoom_b3_auxphi_worker.py"
LOG_DIR = ROOT / "outputs" / "b4_tworoom" / "edge_ablation_logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

TIERS = ["expert_50", "expert_100", "mixed", "mixed_large"]
SOURCES = ["transonly", "k4", "k8"]
SEEDS = [0, 1, 2, 3, 4]
AUX_LAMBDA = 0.3
N_PARALLEL = 3
N_GPUS = 2  # matches gnode058 per ada_direct_sweep.py; adjust if run on a different node
POLL_SECONDS = 10


def log(*a):
    print(*a, flush=True)


def name_for(source):
    return f"auxphi_{source}_lam{AUX_LAMBDA}"


def is_done(tier, source, seed):
    path = B4_CKPT_DIR / f"{tier}__{name_for(source)}__s{seed}.pt"
    ckpt = load_checkpoint(path)
    return ckpt is not None and ckpt.get("done", False)


def launch(tier, source, seed, gpu_id):
    log_path = LOG_DIR / f"{tier}_{source}_s{seed}.log"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    # See ada_direct_sweep.py's note: unconstrained BLAS/OpenMP defaults to every visible
    # core per process -- with N_PARALLEL processes that oversubscribes the node badly even
    # though this is a tiny-MLP workload with no real use for that much parallelism.
    n_threads = str(max(1, os.cpu_count() // N_PARALLEL // 2))
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[var] = n_threads
    cmd = [str(PYTHON), str(WORKER), "--tier", tier, "--seed", str(seed),
           "--aux-lambda", str(AUX_LAMBDA), "--distance-source", source]
    f = open(log_path, "w")
    p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=str(ROOT), env=env)
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
