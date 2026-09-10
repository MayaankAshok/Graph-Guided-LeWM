"""E2 -- where does latent Euclidean distance stop being informative?

The quasimetric proposal claimed "no gradient beyond 5 steps" from the sqrt(2d)
concentration result. That argument is about INDEPENDENT pairs and does not
transfer: pairs 5-25 steps apart are strongly dependent, and same-episode
Spearman is ~0.72-0.77. This measures the real thing instead.

For same-episode pairs binned by true step gap k, report the latent L2
distance distribution per bin, and the discriminability between ADJACENT bins

    d_prime = (mu_{k+1} - mu_k) / sqrt((var_k + var_{k+1}) / 2)

d_prime is what a planner actually consumes: it is the signal-to-noise ratio of
"is this candidate closer than that one". Saturation is where d_prime falls
under ~0.2, not where the mean approaches sqrt(2d).

Also reports the same curve for the projected true-state distance, so a flat
latent curve can be distinguished from a genuinely flat ground truth.

Run:
    python scripts/investigations/quasimetric/e2_saturation.py env=pusht
"""

import json
import os
import sys
import time
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from common.envs import ENV_MECHANICS
from common.log_util import log

BINS = [(1, 1), (2, 3), (4, 5), (6, 8), (9, 12), (13, 18), (19, 25),
        (26, 35), (36, 50), (51, 70), (71, 100), (101, 150), (151, 250)]


def load_cache(cache_dir, n_episodes, seed=0, grid=16):
    """Reuse probe_gate.py's memmapped landmark cache -- same frames, no re-encode."""
    tag = f"ep{n_episodes}_seed{seed}_g{grid}"
    need = ["z", "ep_idx", "step_idx", "state"]
    paths = {k: cache_dir / f"{k}_{tag}.npy" for k in need}
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"probe_gate cache missing: {missing}\nRun scripts/probe_gate.py env=pusht "
            f"n_episodes={n_episodes} first (it writes this cache).")
    return {k: np.load(p, mmap_mode="r") for k, p in paths.items()}


def sample_same_episode_pairs(ep_idx, step_idx, bins, per_bin, seed=0):
    """Draw `per_bin` same-episode (i, j) row pairs whose step gap falls in each bin."""
    rng = np.random.RandomState(seed)
    order = np.argsort(ep_idx, kind="stable")
    ep_sorted = np.asarray(ep_idx)[order]
    bounds = np.searchsorted(ep_sorted, np.unique(ep_sorted), side="left").tolist()
    bounds.append(len(ep_sorted))
    segments = [order[bounds[a]:bounds[a + 1]] for a in range(len(bounds) - 1)]
    segments = [s for s in segments if len(s) >= 2]
    # within each episode, order rows by step so gap == index difference
    segments = [s[np.argsort(np.asarray(step_idx)[s], kind="stable")] for s in segments]

    out = {}
    for lo, hi in bins:
        I, J, G = [], [], []
        tries = 0
        while len(I) < per_bin and tries < per_bin * 60:
            tries += 1
            seg = segments[rng.randint(len(segments))]
            L = len(seg)
            gap = rng.randint(lo, hi + 1)
            if gap >= L:
                continue
            t = rng.randint(0, L - gap)
            I.append(seg[t]); J.append(seg[t + gap]); G.append(gap)
        if I:
            out[(lo, hi)] = (np.array(I), np.array(J), np.array(G))
    return out


def dprime(mu_a, var_a, mu_b, var_b):
    return float((mu_b - mu_a) / np.sqrt(max((var_a + var_b) / 2.0, 1e-12)))


@hydra.main(version_base=None, config_path=str(ROOT / "config" / "graph"),
            config_name="probe_gate")
def main(cfg: DictConfig):
    t0 = time.time()
    mech = ENV_MECHANICS[cfg.env.name]
    out_dir = ROOT / "outputs" / "quasimetric"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(os.environ.get("PROBE_CACHE_DIR",
                                    str(ROOT / "outputs" / f"probe_gate_{cfg.env.output_prefix}" / "cache")))
    per_bin = int(os.environ.get("E2_PER_BIN", 20000))

    log(f"=== E2 saturation curve: {cfg.env.name} (cache={cache_dir}) ===")
    d = load_cache(cache_dir, cfg.n_episodes, cfg.seed)
    z, ep_idx, step_idx, state = d["z"], d["ep_idx"], d["step_idx"], d["state"]
    log(f"cache: {z.shape[0]} frames, {len(np.unique(ep_idx))} episodes, dim={z.shape[1]}")

    pairs = sample_same_episode_pairs(ep_idx, step_idx, BINS, per_bin, seed=cfg.seed)
    proj = mech._project_state
    # standardize the true-state projection the same way the oracle does, so the two
    # curves are on comparable footing
    ref = proj(np.asarray(state[::97]))
    smean, sstd = ref.mean(0, keepdims=True), ref.std(0, keepdims=True)
    sstd[sstd < 1e-6] = 1.0

    rows = []
    for (lo, hi), (I, J, G) in pairs.items():
        zi, zj = np.asarray(z[I]), np.asarray(z[J])
        dl = np.linalg.norm(zi - zj, axis=1)
        si = (proj(np.asarray(state[I])) - smean) / sstd
        sj = (proj(np.asarray(state[J])) - smean) / sstd
        ds = np.linalg.norm(si - sj, axis=1)
        rows.append(dict(bin=f"{lo}-{hi}", lo=lo, hi=hi, n=int(len(I)),
                         gap_mean=float(G.mean()),
                         lat_mean=float(dl.mean()), lat_std=float(dl.std()),
                         true_mean=float(ds.mean()), true_std=float(ds.std())))
        log(f"  gap {lo:>3d}-{hi:<3d} n={len(I):6d}  latent {dl.mean():7.3f} +- {dl.std():6.3f}"
            f"   true {ds.mean():7.3f} +- {ds.std():6.3f}")

    # random cross-episode reference: the sqrt(2d) ceiling
    rng = np.random.RandomState(cfg.seed)
    ra, rb = rng.randint(0, z.shape[0], 50000), rng.randint(0, z.shape[0], 50000)
    keep = np.asarray(ep_idx)[ra] != np.asarray(ep_idx)[rb]
    d_rand = np.linalg.norm(np.asarray(z[ra[keep]]) - np.asarray(z[rb[keep]]), axis=1)
    log(f"random cross-episode reference: {d_rand.mean():.3f} +- {d_rand.std():.3f} "
        f"(sqrt(2d)={np.sqrt(2*z.shape[1]):.3f})")

    # adjacent-bin discriminability
    dps = []
    for a, b in zip(rows[:-1], rows[1:]):
        dps.append(dict(
            from_bin=a["bin"], to_bin=b["bin"],
            dprime_latent=dprime(a["lat_mean"], a["lat_std"] ** 2, b["lat_mean"], b["lat_std"] ** 2),
            dprime_true=dprime(a["true_mean"], a["true_std"] ** 2, b["true_mean"], b["true_std"] ** 2),
        ))
        log(f"  d' {a['bin']:>8s} -> {b['bin']:<8s}  latent {dps[-1]['dprime_latent']:+7.4f}"
            f"   true {dps[-1]['dprime_true']:+7.4f}")

    # first adjacent-bin boundary where the latent signal drops under 0.2 sigma
    sat = next((p["from_bin"] for p in dps if p["dprime_latent"] < 0.2), None)
    log(f"\nlatent d' first drops below 0.2 at bin boundary: {sat or 'never (no saturation)'}")

    # global Spearman over all sampled same-episode pairs, as a single headline number
    allI = np.concatenate([p[0] for p in pairs.values()])
    allJ = np.concatenate([p[1] for p in pairs.values()])
    allG = np.concatenate([p[2] for p in pairs.values()])
    sub = rng.choice(len(allG), min(200000, len(allG)), replace=False)
    dl_all = np.linalg.norm(np.asarray(z[allI[sub]]) - np.asarray(z[allJ[sub]]), axis=1)
    rho = float(spearmanr(allG[sub], dl_all).statistic)
    log(f"same-episode Spearman(gap, latent L2) over all bins: {rho:.4f}")

    res = dict(env=cfg.env.name, n_frames=int(z.shape[0]), n_episodes=int(len(np.unique(ep_idx))),
               per_bin=per_bin, bins=rows, dprime=dps,
               random_ref=dict(mean=float(d_rand.mean()), std=float(d_rand.std()),
                               sqrt_2d=float(np.sqrt(2 * z.shape[1]))),
               saturation_bin=sat, spearman_same_episode=rho, secs=time.time() - t0)
    p = out_dir / f"e2_saturation_{cfg.env.name}_ep{cfg.n_episodes}.json"
    p.write_text(json.dumps(res, indent=2))
    log(f"wrote {p} ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
