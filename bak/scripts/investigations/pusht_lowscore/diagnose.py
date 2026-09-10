"""Why does the latent-graph auxiliary term (auxphi) UNDERperform the plain baseline on
Push-T, when it beats baseline at 6/6 tiers on Two-Room?

Everything downstream of the graph on Push-T is scored against
PushTMechanics.build_true_distance_oracle, which is NOT a dynamical distance at all -- it's
a z-scored Euclidean distance in a 6-D projected state space (agent xy, block xy, cos/sin of
block angle). For a contact-manipulation task, straight-line state distance and actual
time-to-reach are very different things (you must walk around the block, contact it, push it
in the right direction). Two-Room's oracle, by contrast, is a genuine grid-geodesic through
the real wall/door geometry.

This script grades the oracle, the graph, and raw latent Euclidean against a ground truth
that IS dynamical for expert data: the number of env steps between two frames of the SAME
expert trajectory. For near-optimal expert demos, that step gap is (close to) the real
time-to-reach, which is exactly the quantity the value function is supposed to rank.

Measures, on one tier:
  H1  Spearman(oracle, true step-gap)         -- is the metric everything is scored against real?
  H2  phi_dist vs true step-gap at fixed gaps  -- do identification edges create false shortcuts?
  H3  pseudo_v (the aux regression target) vs the TD-learned V it is added to -- scale clash?

Run as:
    python scripts/investigations/pusht_lowscore/diagnose.py env=pusht tier=mixed_large
"""

import sys
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, open_dict
from scipy import stats as sps

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from actor_train import ckpt_path
from common.checkpoint_io import load_checkpoint
from common.envs import ENV_MECHANICS
from common.graph_lib import DEV
from common.log_util import log
from common.training import GAMMA, HIDDEN, MLP, build_her_tuples
from datatiers import NOISY_EP_ID_OFFSET, build_tier_setups

ROOT = Path(__file__).resolve().parent.parent.parent.parent
OFFSETS = [1, 2, 5, 10, 25, 50, 100]
N_PER_OFFSET = 3000


def episode_rows(setup, only=None):
    """Row indices grouped per episode, in step order. only: 'expert' | 'noisy' | None."""
    ep_idx, step_idx = setup["ep_idx"], setup["step_idx"]
    order = np.lexsort((step_idx, ep_idx))
    ep_o = ep_idx[order]
    bnd = np.nonzero(np.diff(ep_o))[0] + 1
    groups = [order[s:e] for s, e in zip(np.concatenate(([0], bnd)), np.concatenate((bnd, [len(order)])))]
    if only == "expert":
        groups = [g for g in groups if ep_idx[g[0]] < NOISY_EP_ID_OFFSET]
    elif only == "noisy":
        groups = [g for g in groups if ep_idx[g[0]] >= NOISY_EP_ID_OFFSET]
    return groups


def fixed_gap_pairs(groups, gap, n, rng):
    """(s_rows, g_rows) exactly `gap` env steps apart within one episode."""
    S, G = [], []
    for rows in groups:
        if len(rows) <= gap:
            continue
        t = np.arange(len(rows) - gap)
        S.append(rows[t]); G.append(rows[t + gap])
    if not S:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    S, G = np.concatenate(S), np.concatenate(G)
    pick = rng.choice(len(S), size=min(n, len(S)), replace=False)
    return S[pick], G[pick]


def pairwise_oracle(oracle, state, s_rows, g_rows, chunk=512):
    """Per-pair oracle distance (the diagonal of its (S,T) output), chunked."""
    out = np.empty(len(s_rows), dtype=np.float64)
    for i in range(0, len(s_rows), chunk):
        sl = slice(i, min(i + chunk, len(s_rows)))
        out[sl] = np.diag(oracle(state[s_rows[sl]], state[g_rows[sl]]))
    return out


@hydra.main(version_base=None, config_path="../../../config/graph", config_name="actor")
def main(cfg: DictConfig):
    mech = ENV_MECHANICS[cfg.env.name]
    with open_dict(cfg.env):
        cfg.env.ckpt_dir_resolved = str(mech.ckpt_dir(ROOT))
    setup = build_tier_setups(cfg, [cfg.tier])[cfg.tier]
    z, state, phi = setup["z"], setup["proprio"], setup["phi_dist"]
    oracle = mech.build_true_distance_oracle(state)
    rng = np.random.default_rng(0)

    log("\n" + "=" * 78)
    log(f"H1/H2: metrics vs TRUE step-gap, same-episode pairs -- tier={cfg.tier}")
    log("=" * 78)

    for pool in ("expert", "noisy"):
        groups = episode_rows(setup, only=pool)
        if not groups:
            continue
        all_gap, all_orc, all_phi, all_euc = [], [], [], []
        log(f"\n--- {pool} episodes ({len(groups)} episodes) ---")
        log(f"{'gap':>5}{'n':>7}{'phi_dist med':>14}{'phi<gap %':>11}{'oracle med':>12}{'lat-eucl med':>14}")
        for gap in OFFSETS:
            s, g = fixed_gap_pairs(groups, gap, N_PER_OFFSET, rng)
            if len(s) < 50:
                continue
            phi_d = phi[s, g].astype(np.float64)
            orc_d = pairwise_oracle(oracle, state, s, g)
            euc_d = np.linalg.norm(z[s] - z[g], axis=1).astype(np.float64)
            log(f"{gap:>5}{len(s):>7}{np.median(phi_d):>14.1f}"
                f"{100 * np.mean(phi_d < gap):>11.1f}{np.median(orc_d):>12.3f}{np.median(euc_d):>14.2f}")
            all_gap.append(np.full(len(s), gap)); all_orc.append(orc_d)
            all_phi.append(phi_d); all_euc.append(euc_d)

        gap_a = np.concatenate(all_gap); orc_a = np.concatenate(all_orc)
        phi_a = np.concatenate(all_phi); euc_a = np.concatenate(all_euc)
        log(f"\n  Spearman vs true step-gap ({pool}, n={len(gap_a)}):")
        log(f"    graph phi_dist          : {sps.spearmanr(gap_a, phi_a).statistic:+.4f}")
        log(f"    CURRENT oracle          : {sps.spearmanr(gap_a, orc_a).statistic:+.4f}   <-- what "
            f"the gate + rho-selection are scored against")
        log(f"    raw latent euclidean    : {sps.spearmanr(gap_a, euc_a).statistic:+.4f}")

    log("\n" + "=" * 78)
    log("H3: aux regression target (pseudo_v) vs the TD-learned V it is added to")
    log("=" * 78)
    s_idx, next_idx, goal_idx, act_idx, done = build_her_tuples(
        setup["ep_idx"], setup["step_idx"], setup["action"], seed=0,
        allowed_episode_ids=setup["train_eps"], goal_gamma=cfg.her_goal_gamma,
    )
    batch = rng.choice(len(s_idx), size=20000, replace=False)
    s, g = s_idx[batch], goal_idx[batch]
    phi_sg = phi[s, g].astype(np.float64)
    pseudo_v = -(1 - GAMMA ** phi_sg) / (1 - GAMMA)
    log(f"  HER pairs (n={len(s)}), her_goal_gamma={cfg.her_goal_gamma}")
    log(f"    phi_dist  : med={np.median(phi_sg):.1f} p90={np.percentile(phi_sg, 90):.1f} "
        f"max={phi_sg.max():.1f}")
    log(f"    pseudo_v  : med={np.median(pseudo_v):.2f} p10={np.percentile(pseudo_v, 10):.2f} "
        f"min={pseudo_v.min():.2f}  (floor at -{1/(1-GAMMA):.0f})")

    d = setup["d"]
    zt = torch.from_numpy(z).to(DEV)
    for variant in ("baseline", "auxphi"):
        with open_dict(cfg):
            cfg.variant, cfg.run_tag = variant, ""
        ck = load_checkpoint(ckpt_path(cfg, cfg.tier, variant, cfg.seed))
        if ck is None:
            log(f"    [{variant}] no checkpoint -- skipped")
            continue
        v_net = MLP(2 * d, HIDDEN).to(DEV)
        v_net.load_state_dict(ck["v_net"])
        v_net.eval()
        with torch.no_grad():
            v = v_net(torch.cat([zt[s], zt[g]], dim=-1)).squeeze(-1).cpu().numpy()
        log(f"    V({variant:8s}): med={np.median(v):.2f} p10={np.percentile(v, 10):.2f} "
            f"min={v.min():.2f}   corr(V, -pseudo_v) spearman={sps.spearmanr(v, pseudo_v).statistic:+.4f}")


if __name__ == "__main__":
    main()
