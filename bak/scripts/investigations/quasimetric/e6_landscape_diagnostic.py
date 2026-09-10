"""E6 -- isolate WHICH piece of the quasimetric head is responsible for E5's finding
(locally-honest ranking, but a worse absolute convergence neighborhood than L2).

Reuses E3-v3's already-cached CEM-candidate pool (2,000 groups x 20 real-env-grounded
candidates, predicted embeddings + true outcome distance -- no new env stepping) to run four
metrics against the SAME broad candidate population, side by side:

  1. raw_l2      ||z_hat - z_goal||_2 in the FULL 192-dim, UN-normalized space -- exactly the
                 metric jepa.py's `criterion` uses (F.mse_loss, no z-scoring at all). The
                 baseline this whole project is trying to beat.
  2. psi_only    ||psi(z_hat) - psi(z_goal)||_2 -- the quasimetric head's SYMMETRIC component
                 alone, projected down to proj_dim=64. Isolates whether the dimensionality
                 bottleneck (192 -> 64) is discarding information raw L2 still has access to.
  3. phi_only    ReLU(phi(z_goal) - phi(z_hat)) alone -- the ASYMMETRIC potential term alone.
                 Isolates whether this term's contribution is even informative about true
                 distance on this population, or just noise/bias riding on top of psi.
  4. full_dQ     psi_only + phi_only -- the complete trained head (already measured at 0.9205
                 within-group Spearman for v3; recomputed here for a direct side-by-side).

If raw_l2 correlates with true_dist as well as or better than full_dQ on this SAME broad
population (not just among CEM's own narrowed elites, where restriction of range attenuates
raw_l2's correlation per E5's own finding), that says the learned head isn't adding
discriminative value here even before CEM's optimization dynamics get involved -- pointing at
the architecture/training, not just "CEM finds a rough landscape." If psi_only alone already
matches full_dQ, phi_only is dead weight (or actively harmful) and the fix is architectural
(drop or shrink it, or increase proj_dim). If phi_only carries most of the signal and psi_only
is weak, the dimensionality bottleneck is the more likely target for a fix.

Run:
    python scripts/investigations/quasimetric/e6_landscape_diagnostic.py pool_head=v3
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import stats as sps

ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.log_util import log
from common.quasimetric import LatentQuasimetric
from e3_train_quasimetric import episode_split
from probe_gate import Arm

POOL_HEAD = sys.argv[1].split("=")[1] if len(sys.argv) > 1 and sys.argv[1].startswith("pool_head=") else "v3"
HEAD_FILES = {"v1": "e3_quasimetric_head_pusht.pt", "v2": "e3v2_quasimetric_head_pusht.pt",
             "v3": "e3v3_quasimetric_head_pusht.pt"}


def within_group_spearman(values, true_dist, group_id):
    rhos = []
    for g in np.unique(group_id):
        idx = np.nonzero(group_id == g)[0]
        if len(idx) > 2 and np.ptp(true_dist[idx]) > 1e-6 and np.ptp(values[idx]) > 1e-9:
            r = sps.spearmanr(true_dist[idx], values[idx]).statistic
            if np.isfinite(r):
                rhos.append(r)
    return float(np.mean(rhos)), len(rhos)


def best_pick_true_dist(values, true_dist, group_id):
    """For each group, the true_dist of the candidate this metric would pick as best
    (argmin). Directly tests whether high rank-correlation (within_group_spearman) actually
    translates into picking the genuinely closest candidate, not just ranking well overall."""
    picks = []
    for g in np.unique(group_id):
        idx = np.nonzero(group_id == g)[0]
        if len(idx) < 2:
            continue
        best = idx[np.argmin(values[idx])]
        picks.append(true_dist[best])
    picks = np.array(picks)
    return float(picks.mean()), float(picks.std()), float(np.median(picks))


def main():
    res_dir = ROOT / "outputs" / "quasimetric"
    cache_dir = Path(os.environ.get("PROBE_CACHE_DIR", str(ROOT / "outputs" / "probe_gate_pusht" / "cache")))

    pool_path = res_dir / "e3v3_cem_pool_g2000_k20.npz"
    log(f"loading candidate pool: {pool_path}")
    pool = np.load(pool_path)
    z_hat = pool["z_hat"].astype(np.float32)          # (N, 192) raw, un-normalized predicted embeddings
    true_dist = pool["true_dist"].astype(np.float32)  # (N,) real physics ground truth
    group_id = pool["group_id"]
    goal_row = pool["goal_row"]
    n = len(true_dist)
    log(f"pool: {n} rows, {len(np.unique(group_id))} groups")

    tag = "ep18685_seed0_g16"
    z_all = np.load(cache_dir / f"z_{tag}.npy", mmap_mode="r")
    z_goal = np.asarray(z_all[goal_row]).astype(np.float32)   # (N, 192) raw, un-normalized

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # -- metric 1: raw L2 in full 192-dim, un-normalized (exactly jepa.py's baseline metric)
    raw_l2 = np.linalg.norm(z_hat - z_goal, axis=1)
    rho_raw_l2, ng1 = within_group_spearman(raw_l2, true_dist, group_id)
    log(f"[raw_l2 , full 192-dim, un-normalized]  within-group Spearman = {rho_raw_l2:.4f}  (n_groups={ng1})")

    # -- normalization stats, matching how the head was actually trained/queried (Arm.fit_norm
    # on the SAME train split, seed=0) -- needed to feed z_hat/z_goal through the head correctly
    ep_idx = np.load(cache_dir / f"ep_idx_{tag}.npy")
    train_eps, test_eps, rows_tr, rows_te = episode_split(ep_idx, 0.8, 0)
    arm = Arm("lewm", z_all, z_all.shape[1])
    arm.fit_norm(rows_tr, 0)
    mu, sd = arm.mu[0], arm.sd[0]

    z_hat_n = np.clip((z_hat - mu) / sd, -10.0, 10.0)
    z_goal_n = np.clip((z_goal - mu) / sd, -10.0, 10.0)

    head_path = res_dir / HEAD_FILES[POOL_HEAD]
    log(f"loading head: {head_path}")
    model = LatentQuasimetric(latent_dim=192, hidden_dim=256, proj_dim=64).to(device)
    model.load_state_dict(torch.load(head_path, map_location="cpu"))
    model.eval()
    model.requires_grad_(False)

    def chunked(fn, n, chunk=8192):
        return np.concatenate([fn(a, min(a + chunk, n)) for a in range(0, n, chunk)])

    with torch.no_grad():
        def psi_dist(a, b):
            za = torch.from_numpy(z_hat_n[a:b]).float().to(device)
            zg = torch.from_numpy(z_goal_n[a:b]).float().to(device)
            return torch.norm(model.psi(za) - model.psi(zg), p=2, dim=-1).cpu().numpy()

        def phi_term(a, b):
            za = torch.from_numpy(z_hat_n[a:b]).float().to(device)
            zg = torch.from_numpy(z_goal_n[a:b]).float().to(device)
            return torch.relu(model.phi(zg) - model.phi(za)).squeeze(-1).cpu().numpy()

        def full_dq(a, b):
            za = torch.from_numpy(z_hat_n[a:b]).float().to(device)
            zg = torch.from_numpy(z_goal_n[a:b]).float().to(device)
            return model(za, zg).cpu().numpy()

        psi_only = chunked(psi_dist, n)
        phi_only = chunked(phi_term, n)
        full_d_q = chunked(full_dq, n)

    rho_psi, ng2 = within_group_spearman(psi_only, true_dist, group_id)
    rho_phi, ng3 = within_group_spearman(phi_only, true_dist, group_id)
    rho_full, ng4 = within_group_spearman(full_d_q, true_dist, group_id)
    log(f"[psi_only, proj_dim=64, normalized ]  within-group Spearman = {rho_psi:.4f}  (n_groups={ng2})")
    log(f"[phi_only, asymmetric ReLU term    ]  within-group Spearman = {rho_phi:.4f}  (n_groups={ng3})")
    log(f"[full_dQ , psi + phi (the trained head)]  within-group Spearman = {rho_full:.4f}  (n_groups={ng4})")

    # -- flat-region diagnostic: what fraction of candidates have phi_only == 0 exactly
    # (the ReLU saturated, so the asymmetric term contributes nothing and the cost degenerates
    # to psi_only alone for that candidate)?
    frac_relu_zero = float((phi_only < 1e-6).mean())
    log(f"fraction of candidates with phi_only == 0 (ReLU saturated): {frac_relu_zero:.1%}")

    # -- the test that actually connects to E4/E5: does high rank-correlation translate into
    # picking the genuinely closest candidate? For each group, take the argmin under each
    # metric and report ITS true_dist -- this is what CEM effectively does (pick the lowest-
    # cost candidate), just without the 30-iteration narrowing.
    bp_raw = best_pick_true_dist(raw_l2, true_dist, group_id)
    bp_psi = best_pick_true_dist(psi_only, true_dist, group_id)
    bp_full = best_pick_true_dist(full_d_q, true_dist, group_id)
    oracle = best_pick_true_dist(true_dist, true_dist, group_id)  # picking by true_dist itself -- the ceiling
    log(f"best-pick true_dist (mean/std/median), n_groups={len(np.unique(group_id))}:")
    log(f"  raw_l2  : {bp_raw[0]:.4f} / {bp_raw[1]:.4f} / {bp_raw[2]:.4f}")
    log(f"  psi_only: {bp_psi[0]:.4f} / {bp_psi[1]:.4f} / {bp_psi[2]:.4f}")
    log(f"  full_dQ : {bp_full[0]:.4f} / {bp_full[1]:.4f} / {bp_full[2]:.4f}")
    log(f"  oracle  : {oracle[0]:.4f} / {oracle[1]:.4f} / {oracle[2]:.4f}  (picks by true_dist itself -- ceiling)")

    res = dict(
        pool_head=POOL_HEAD, n_rows=n, n_groups=len(np.unique(group_id)),
        within_group_spearman=dict(raw_l2=rho_raw_l2, psi_only=rho_psi, phi_only=rho_phi, full_dQ=rho_full),
        frac_relu_saturated=frac_relu_zero,
        best_pick_true_dist=dict(
            raw_l2=dict(mean=bp_raw[0], std=bp_raw[1], median=bp_raw[2]),
            psi_only=dict(mean=bp_psi[0], std=bp_psi[1], median=bp_psi[2]),
            full_dQ=dict(mean=bp_full[0], std=bp_full[1], median=bp_full[2]),
            oracle=dict(mean=oracle[0], std=oracle[1], median=oracle[2]),
        ),
    )
    p = res_dir / f"e6_landscape_diagnostic_{POOL_HEAD}.json"
    p.write_text(json.dumps(res, indent=2))
    log(f"wrote {p}")


if __name__ == "__main__":
    main()
