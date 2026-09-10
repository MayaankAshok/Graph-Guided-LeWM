"""Stitching showcase, built with methodological safeguards against p-hacking (see the
conversation this came from -- the naive "arbitrary cross-episode pair" eval already gave a
null result: baseline 8%, auxphi 9%, both near the random floor, because most arbitrary
pairs are genuinely far apart and the result is dominated by raw long-horizon-control
difficulty, not by whether the graph can stitch).

The claim under test: auxphi's advantage over baseline should be concentrated on
cross-episode pairs the GRAPH says are close (via identification edges -- true stitching
opportunities), and should vanish on pairs it says are far (negative control).

Safeguards (fixed BEFORE looking at any actor's success/fail outcome):
  1. Bucket boundaries fixed in advance: near <20, mid 20-50, far >=50 (true graph-distance
     steps). Never tuned post-hoc.
  2. Every bucket is reported, including far as the negative control.
  3. Success is the real physical eval_state() criterion (via mech.step_result), never
     phi_dist itself -- phi_dist only decides which pairs to ask about.
  4. Near-bucket pairs are cross-validated against the TRUE oracle distance
     (PushTMechanics.build_true_distance_oracle, privileged z-scored state distance,
     computed independently of the graph) -- rules out "the graph invented a shortcut that
     isn't real."
  5. Near-bucket pairs are also reported by hop count (phi_dist itself, since every edge in
     this graph has weight 1, so phi_dist IS the shortest-path hop count) and raw latent
     Euclidean distance -- separates genuine multi-hop stitching from a trivial
     near-duplicate pair findable by plain nearest-neighbor lookup.

Run as:
    python scripts/investigations/pusht_lowscore/stitching_eval.py env=pusht tier=expert_1000
    python scripts/investigations/pusht_lowscore/stitching_eval.py env=pusht tier=mixed_large
"""

import sys
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig, open_dict
from scipy import stats as sps

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from actor_rollout_utils import actor_act_fn, make_encoder, make_env, run_episodes, summarize
from actor_train import ckpt_path, get_setup
from common.checkpoint_io import load_checkpoint
from common.envs import ENV_MECHANICS
from common.graph_lib import DEV, phi_for_pairs
from common.log_util import log
from common.training import HIDDEN, MLP
from render_rollout_videos import select_actor_state

ROOT = Path(__file__).resolve().parent.parent.parent.parent

# Safeguard 1: fixed before running anything, never tuned to the outcome.
BUCKETS = [("near", 0, 20), ("mid", 20, 50), ("far", 50, float("inf"))]
N_CANDIDATES = 6000       # cross-episode candidate pool to bucket from
N_PER_BUCKET = 40         # rollouts per bucket per variant (kept equal across buckets)
DIJKSTRA_LIMIT = 100      # bounded search radius; anything beyond comes back clamped -> "far"
RNG_SEED = 12345


def sample_cross_episode_pairs(ep_idx, n, seed):
    rng = np.random.default_rng(seed)
    n_rows = len(ep_idx)
    ei = rng.integers(0, n_rows, n * 2)
    ej = rng.integers(0, n_rows, n * 2)
    keep = ep_idx[ei] != ep_idx[ej]
    ei, ej = ei[keep][:n], ej[keep][:n]
    return ei, ej


def load_actor(cfg, variant, d, act_dim):
    ck = load_checkpoint(ckpt_path(cfg, cfg.tier, variant, cfg.seed))
    if ck is None:
        raise SystemExit(f"no checkpoint for {cfg.tier}/{variant}/s{cfg.seed}")
    state, step = select_actor_state(cfg, ck, variant)
    net = MLP(2 * d, HIDDEN, out_dim=act_dim).to(DEV)
    net.load_state_dict(state)
    net.eval()
    log(f"[{cfg.tier}] loaded {variant} (select={cfg.select} -> step {step})")
    return net


def build_pairs(setup, rows_s, rows_g):
    """One pair dict per (row_s, row_g), in actor_rollout_utils.run_episodes' expected
    format. z_g is a placeholder -- run_episodes overrides it from the live env's own
    info['goal'] for PushT (see actor_rollout_utils module docstring); this only matters
    for environments that don't provide one."""
    state = setup["proprio"]
    return [dict(start_state=state[s], goal_state=state[g], z_g=None,
                 start_row=int(s), goal_row=int(g)) for s, g in zip(rows_s, rows_g)]


@hydra.main(version_base=None, config_path="../../../config/graph", config_name="actor")
def main(cfg: DictConfig):
    with open_dict(cfg):
        cfg.phi_mode = "sparse"  # need setup["graph"]; skip the dense matrix (not used here)
    mech = ENV_MECHANICS[cfg.env.name]
    with open_dict(cfg.env):
        cfg.env.ckpt_dir_resolved = str(mech.ckpt_dir(ROOT))

    setup = get_setup(cfg, cfg.tier)
    ep_idx, z, proprio, d = setup["ep_idx"], setup["z"], setup["proprio"], setup["d"]
    act_dim = setup["action"].shape[1]

    log(f"\n[{cfg.tier}] sampling {N_CANDIDATES} cross-episode candidate pairs "
        f"from {len(ep_idx)} landmarks...")
    ei, ej = sample_cross_episode_pairs(ep_idx, N_CANDIDATES, RNG_SEED)
    phi_d = phi_for_pairs(setup["graph"], ei, ej, limit=DIJKSTRA_LIMIT)

    # Bucket assignment (safeguard 1/2): fixed boundaries, computed before any actor runs.
    bucket_rows = {}
    for name, lo, hi in BUCKETS:
        mask = (phi_d >= lo) & (phi_d < hi)
        idx = np.nonzero(mask)[0]
        log(f"[{cfg.tier}] bucket '{name}' [{lo},{hi}): {len(idx)}/{N_CANDIDATES} candidates")
        bucket_rows[name] = idx

    # Safeguard 4: independently validate the near bucket against the TRUE oracle distance
    # (privileged z-scored state distance, nothing to do with the graph).
    near_idx = bucket_rows["near"]
    if len(near_idx) > 0:
        oracle = mech.build_true_distance_oracle(proprio)
        sub = near_idx[:min(len(near_idx), 500)]
        s_rows, g_rows = ei[sub], ej[sub]
        oracle_d = np.array([float(oracle(proprio[s:s+1], proprio[g:g+1])[0, 0])
                              for s, g in zip(s_rows, g_rows)])
        euclid_d = np.linalg.norm(z[s_rows] - z[g_rows], axis=1)
        rho, _ = sps.spearmanr(phi_d[sub], oracle_d)
        log(f"\n[{cfg.tier}] SAFEGUARD 4 (near bucket vs TRUE oracle distance, n={len(sub)}):")
        log(f"  oracle_dist: median={np.median(oracle_d):.3f} p90={np.percentile(oracle_d,90):.3f} "
            f"(graph-close pairs are also oracle-close if this is small)")
        log(f"  Spearman(phi_dist, oracle_dist) on near bucket = {rho:+.4f}")
        log(f"[{cfg.tier}] SAFEGUARD 5 (near bucket hop-count / Euclidean distance, n={len(sub)}):")
        log(f"  phi_dist (= hop count, edge weight 1): median={np.median(phi_d[sub]):.1f} "
            f"(>2 means genuine multi-hop, not a single direct identification edge)")
        log(f"  raw latent Euclidean ||z_s - z_g||: median={np.median(euclid_d):.2f} "
            f"p10={np.percentile(euclid_d,10):.2f}")
        frac_trivial = float(np.mean(phi_d[sub] <= 2))
        log(f"  fraction of 'near' pairs that are trivial (phi_dist<=2, i.e. essentially a "
            f"single identification-edge hop): {100*frac_trivial:.1f}%")

    # Actors: same checkpoints for every bucket, identical (start,goal) pairs given to both.
    baseline_net = load_actor(cfg, "baseline", d, act_dim)
    auxphi_net = load_actor(cfg, "auxphi", d, act_dim)
    env = make_env(cfg.env.name)
    encode_frame = make_encoder(cfg.env.ckpt_dir_resolved)
    max_steps = mech.max_episode_steps

    log(f"\n[{cfg.tier}] === live-rollout success rate by bucket (identical pairs, "
        f"{N_PER_BUCKET}/bucket, max_steps={max_steps}) ===")
    log(f"{'bucket':<8}{'phi_dist range':<18}{'n':>5}{'baseline':>12}{'auxphi':>12}{'delta':>10}")
    for name, lo, hi in BUCKETS:
        idx = bucket_rows[name]
        if len(idx) == 0:
            log(f"{name:<8}{'[' + str(lo) + ',' + str(hi) + ')':<18} -- no candidates, skipped")
            continue
        rng = np.random.default_rng(RNG_SEED + 1)
        pick = rng.choice(idx, size=min(N_PER_BUCKET, len(idx)), replace=False)
        pairs = build_pairs(setup, ei[pick], ej[pick])

        base_eps = run_episodes(cfg.env.name, actor_act_fn(baseline_net), env, encode_frame,
                                 pairs, seed=RNG_SEED, max_steps=max_steps)
        aux_eps = run_episodes(cfg.env.name, actor_act_fn(auxphi_net), env, encode_frame,
                                pairs, seed=RNG_SEED, max_steps=max_steps)
        base_sr = summarize(base_eps)["success_rate"]
        aux_sr = summarize(aux_eps)["success_rate"]
        log(f"{name:<8}{'[' + str(lo) + ',' + str(hi) + ')':<18}{len(pairs):>5}"
            f"{100*base_sr:>11.1f}%{100*aux_sr:>11.1f}%{100*(aux_sr-base_sr):>+9.1f}%")

    log(f"\n[{cfg.tier}] done.")


if __name__ == "__main__":
    main()
