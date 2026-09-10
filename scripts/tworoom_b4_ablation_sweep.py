"""B4, updated: the distance-source ablation for auxiliary-Phi regression (the method that
actually beat baseline everywhere, per tworoom_b3_auxphi_sweep.py), across ALL 6 tiers.

The original B4 plan (raw Euclidean vs graph vs noise-free oracle, all fed through reward
SHAPING) was designed to answer "is shaping's failure about graph/Phi noise, or something
more fundamental." That question is already answered directly: Phi is excellent everywhere
(~0.96-0.97 Spearman vs true distance, no degradation on noisy-touching pairs -- see
tworoom_b3_mixed_diagnosis.py), and the real problem was reward-shaping's TD-bootstrap
channel compounding non-monotonic-behavior noise, not Phi's own accuracy.

This is the updated, more informative version: apply the SAME 3-way distance-source
ablation to auxphi instead, which sidesteps that TD-compounding problem entirely (see
tworoom_b3_mixed_auxphi.py). This directly tests the proposal's original core thesis --
does the graph's superior distance estimate matter -- on the mechanism that actually works:

  graph      -- already complete (auxphi_lam0.3, no suffix) from tworoom_b3_auxphi_sweep.py
  euclidean  -- pseudo_v built from raw ||z_s - z_g||, no graph at all (known-bad proxy,
                ~0.44 Spearman vs true distance per B0)
  oracle     -- pseudo_v built from the ground-truth wall-respecting pixel distance
                (~1.0 Spearman by construction) -- not deployable (needs privileged
                position info), but tells us how much headroom the graph estimate leaves
                on the table

If graph clearly beats euclidean and sits close to oracle, that validates the whole
proposal's core claim under the mechanism that actually works. If euclidean does nearly as
well, the graph isn't what's doing the work here -- an important, much less flattering
result worth knowing either way.

Parallelization: for each (tier, distance_source) pair, 3 seeds run as separate OS
processes in parallel (same reasoning as tworoom_b3_auxphi_sweep.py -- real concurrency,
not GIL-limited threads), one (tier, source) batch at a time. For "oracle", the (tier,
source)-shared true-distance matrix is pre-built/cached ONCE by this orchestrator itself
before launching the 3 seed workers, so they don't all redundantly recompute it in
parallel (same-size cost class as phi_dist, e.g. ~864MB for mixed_large) -- each worker's
own lookup then just hits the cache.
"""

import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tworoom_b0_graph_gate import log
from tworoom_b3_datatiers import ckpt_path, get_true_dist_matrix, load_checkpoint
from tworoom_b3_auxphi_worker import B4_CACHE_DIR, B4_CKPT_DIR, get_setup

ROOT = Path(__file__).resolve().parent.parent
WORKER = Path(__file__).resolve().parent / "tworoom_b3_auxphi_worker.py"
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
LOG_DIR = ROOT / "outputs" / "b4_tworoom" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

TIERS = ["expert_10", "expert_25", "expert_50", "expert_100", "mixed", "mixed_large"]
SEEDS = [0, 1, 2]
AUX_LAMBDA = 0.3
DISTANCE_SOURCES = ["euclidean", "oracle"]  # "graph" already complete, not re-run here


def name_for(source):
    suffix = "" if source == "graph" else f"_{source}"
    return f"auxphi{suffix}_lam{AUX_LAMBDA}"


def is_done(tier, name, seed, source):
    path = ckpt_path(tier, name, seed) if source == "graph" else B4_CKPT_DIR / f"{tier}__{name}__s{seed}.pt"
    ckpt = load_checkpoint(path)
    return ckpt is not None and ckpt.get("done", False)


def main():
    t0 = time.time()
    status = {}
    for tier in TIERS:
        for source in DISTANCE_SOURCES:
            name = name_for(source)
            key = f"{tier}/{source}"
            log(f"\n{'=' * 70}\n{key}: {len(SEEDS)} seeds in parallel\n{'=' * 70}")

            seeds_to_run = [s for s in SEEDS if not is_done(tier, name, s, source)]
            skipped = [s for s in SEEDS if s not in seeds_to_run]
            if skipped:
                log(f"  skipping already-complete seeds {skipped}")
            if not seeds_to_run:
                log(f"[{key}] all seeds already complete, nothing to launch")
                status[key] = True
                continue

            if source == "oracle":
                log(f"  pre-building/caching the true-distance matrix for {tier} "
                    f"(shared across all seeds, avoids 3x redundant computation)...")
                setup = get_setup(tier)
                get_true_dist_matrix(tier, setup["proprio"], cache_dir=B4_CACHE_DIR)

            procs = []
            for seed in seeds_to_run:
                log_path = LOG_DIR / f"{tier}_{source}_s{seed}.log"
                f = open(log_path, "w")
                p = subprocess.Popen(
                    [str(PYTHON), str(WORKER), "--tier", tier, "--seed", str(seed),
                     "--aux-lambda", str(AUX_LAMBDA), "--distance-source", source],
                    stdout=f, stderr=subprocess.STDOUT, cwd=str(ROOT),
                )
                procs.append((p, f, seed, log_path))
                log(f"  launched seed={seed} pid={p.pid} -> {log_path}")

            key_ok = True
            for p, f, seed, log_path in procs:
                ret = p.wait()
                f.close()
                ok = ret == 0
                key_ok = key_ok and ok
                log(f"  {key} seed={seed}: {'OK' if ok else f'FAILED (exit {ret})'}")
                if not ok:
                    log(f"    see {log_path} for details")

            status[key] = key_ok
            log(f"[{key}] done ({'all OK' if key_ok else 'SOME FAILED'}), "
                f"elapsed so far {time.time() - t0:.1f}s")

    log(f"\nALL DONE, total elapsed {time.time() - t0:.1f}s")
    log("status: " + ", ".join(f"{k}={'OK' if ok else 'FAILED'}" for k, ok in status.items()))


if __name__ == "__main__":
    main()
