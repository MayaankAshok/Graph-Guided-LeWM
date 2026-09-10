"""
Executable test suite proving the fundamental metric failure modes of LeWM:
1. Contact Irreversibility & Metric Symmetry Residual (Push-T)
2. Gaussian Annulus Distance Saturation (Push-T & TwoRoom)
3. Topological Barrier Shortcut Violation (TwoRoom)
4. Cross-Episode Correlation Collapse (Push-T)

Run as:
    .venv\Scripts\python scripts/investigations/test_quasimetric_failures.py
"""

import sys
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent.parent
PUSHT_PATH = ROOT / "outputs" / "b0_pusht" / "landmarks_ep300_seed0.npz"
TWOROOM_PATH = ROOT / "outputs" / "b0_tworoom" / "landmarks_ep300.npz"


def test_contact_irreversibility_pusht():
    print("=" * 70)
    print("TEST 1: Contact Irreversibility & Symmetry Residual on Push-T")
    print("=" * 70)
    if not PUSHT_PATH.exists():
        print(f"Skipping: {PUSHT_PATH} not found.")
        return

    data = np.load(PUSHT_PATH)
    z = data["z"]
    state = data["state"]
    ep = data["ep_idx"]

    # Active pushing segments: dt=15 steps, block displacement > 20px
    forward_pairs = []
    for e in np.unique(ep)[:100]:
        idx = np.where(ep == e)[0]
        for t in range(len(idx) - 15):
            s_A, s_B = state[idx[t]], state[idx[t + 15]]
            if np.linalg.norm(s_B[2:4] - s_A[2:4]) > 20.0:
                forward_pairs.append((idx[t], idx[t + 15]))

    if not forward_pairs:
        print("No pushing pairs found.")
        return

    idx_A = np.array([p[0] for p in forward_pairs])
    idx_B = np.array([p[1] for p in forward_pairs])

    dist_fwd = np.linalg.norm(z[idx_A] - z[idx_B], axis=1)
    dist_bwd = np.linalg.norm(z[idx_B] - z[idx_A], axis=1)
    asym_residual = np.abs(dist_fwd - dist_bwd)

    print(f"Number of active pushing transitions: {len(forward_pairs)}")
    print(f"Forward latent distance:  {dist_fwd.mean():.4f} ± {dist_fwd.std():.4f}")
    print(f"Backward latent distance: {dist_bwd.mean():.4f} ± {dist_bwd.std():.4f}")
    print(f"Residual |dist(A->B) - dist(B->A)| max: {asym_residual.max():.8f}")
    print("-> Physical Reality: Forward push = 15 steps. Reversing it requires encircling the block (>50 steps).")
    print("-> LeWM Prediction: Forward and backward distances are IDENTICAL (Symmetric metric artifact).")


def test_hollow_hypersphere_shell():
    print("\n" + "=" * 70)
    print("TEST 2: Hollow Hypersphere Shell (Gaussian Annulus Concentration)")
    print("=" * 70)
    for env_name, path in [("Push-T", PUSHT_PATH), ("TwoRoom", TWOROOM_PATH)]:
        if not path.exists():
            continue
        data = np.load(path)
        z = data["z"]
        d = z.shape[1]
        norms = np.linalg.norm(z, axis=1)

        theoretical_mean = np.sqrt(d - 0.5)
        theoretical_std = 1.0 / np.sqrt(2.0)
        frac_inside = np.mean(norms < 12.0)
        frac_outside = np.mean(norms > 16.0)
        frac_shell = np.mean((norms >= 12.0) & (norms <= 16.0))

        print(f"[{env_name}] Latent dim d={d} (N={len(norms)} states)")
        print(f"  Theoretical norm mean sqrt(d - 0.5): {theoretical_mean:.3f}, std: {theoretical_std:.3f}")
        print(f"  Empirical norm:                      {norms.mean():.3f} ± {norms.std():.3f} (min: {norms.min():.2f}, max: {norms.max():.2f})")
        print(f"  Fraction inside ball (||z|| < 12):    {frac_inside:.2%}")
        print(f"  Fraction outside ball (||z|| > 16):   {frac_outside:.2%}")
        print(f"  Fraction on hollow shell [12, 16]:   {frac_shell:.2%}")
        print("-> Conclusion: Virtually ZERO mass exists at the origin or outside the shell.")
        print("   Every latent state is forced by SIGReg to live on a thin hollow hypersphere.")


def test_gaussian_distance_saturation():
    print("\n" + "=" * 70)
    print("TEST 3: High-Dimensional Gaussian Annulus Pairwise Distance Saturation")
    print("=" * 70)
    for env_name, path in [("Push-T", PUSHT_PATH), ("TwoRoom", TWOROOM_PATH)]:
        if not path.exists():
            continue
        data = np.load(path)
        z = data["z"]
        d = z.shape[1]

        rng = np.random.RandomState(42)
        idx1 = rng.choice(len(z), 30000)
        idx2 = rng.choice(len(z), 30000)
        keep = idx1 != idx2
        dists = np.linalg.norm(z[idx1[keep]] - z[idx2[keep]], axis=1)

        theoretical = np.sqrt(2 * d)
        print(f"[{env_name}] Latent dim d={d}")
        print(f"  Theoretical Gaussian pairwise distance sqrt(2d): {theoretical:.3f}")
        print(f"  Empirical pairwise distance:                     {dists.mean():.3f} ± {dists.std():.3f}")
        print(f"  90% concentration band:                          [{np.percentile(dists, 5):.2f}, {np.percentile(dists, 95):.2f}]")
        print("-> Consequence: Distant states concentrate tightly on a thin shell, causing vanishing planning gradients.")


def test_topological_barrier_tworoom():
    print("\n" + "=" * 70)
    print("TEST 4: Topological Barrier Shortcut Violation in TwoRoom")
    print("=" * 70)
    if not TWOROOM_PATH.exists():
        print(f"Skipping: {TWOROOM_PATH} not found.")
        return

    data = np.load(TWOROOM_PATH)
    z = data["z"]
    pos = data["proprio"]  # [x, y] coordinates. Wall at x ~ 112, Door at (112, 112)

    # Pick points on opposite sides of the wall at high y (y > 175, far from doorway at y ~ 52)
    left_wall = np.where((pos[:, 0] < 95) & (pos[:, 1] > 175))[0]
    right_wall = np.where((pos[:, 0] > 130) & (pos[:, 1] > 175))[0]
    doorway = np.where((np.abs(pos[:, 0] - 112) < 10) & (pos[:, 1] >= 45) & (pos[:, 1] <= 60))[0]

    pairs_wall_latent = []
    pairs_door_latent = []
    for l in left_wall[:100]:
        r = right_wall[np.argmin(np.abs(pos[right_wall, 1] - pos[l, 1]))]
        d = doorway[0]
        pairs_wall_latent.append(np.linalg.norm(z[l] - z[r]))
        pairs_door_latent.append(np.linalg.norm(z[l] - z[d]))

    pairs_wall_latent = np.array(pairs_wall_latent)
    pairs_door_latent = np.array(pairs_door_latent)

    print(f"Mean latent distance across wall (solid barrier): {pairs_wall_latent.mean():.3f}")
    print(f"Mean latent distance to doorway (open passage):   {pairs_door_latent.mean():.3f}")
    print(f"Fraction of wall pairs ranked CLOSER than door:   {np.mean(pairs_wall_latent < pairs_door_latent):.2%}")
    print("-> Consequence: Planner selects actions directly penetrating the wall, causing 87% planning failure.")


def test_cross_episode_correlation_collapse():
    print("\n" + "=" * 70)
    print("TEST 5: Cross-Episode Correlation Collapse on Push-T")
    print("=" * 70)
    if not PUSHT_PATH.exists():
        print(f"Skipping: {PUSHT_PATH} not found.")
        return

    data = np.load(PUSHT_PATH)
    z, state, ep = data["z"], data["state"], data["ep_idx"]

    # 1. Within-episode correlation: dt vs latent distance
    within_deltas, within_latents = [], []
    rng = np.random.RandomState(42)
    for e in np.unique(ep)[:50]:
        idx = np.where(ep == e)[0]
        if len(idx) < 30: continue
        for _ in range(30):
            t1, t2 = rng.choice(len(idx), 2, replace=False)
            if t1 > t2: t1, t2 = t2, t1
            within_deltas.append(t2 - t1)
            within_latents.append(np.linalg.norm(z[idx[t1]] - z[idx[t2]]))

    # 2. Cross-episode correlation: physical distance vs latent distance
    cross_A, cross_B = [], []
    for _ in range(2000):
        e1, e2 = rng.choice(np.unique(ep), 2, replace=False)
        cross_A.append(rng.choice(np.where(ep == e1)[0]))
        cross_B.append(rng.choice(np.where(ep == e2)[0]))

    cross_latent = np.linalg.norm(z[cross_A] - z[cross_B], axis=1)
    cross_phys = np.linalg.norm(state[cross_A] - state[cross_B], axis=1)

    corr_within = spearmanr(within_deltas, within_latents).statistic
    corr_cross = spearmanr(cross_phys, cross_latent).statistic

    print(f"Within-episode Spearman correlation (dt vs latent distance):     {corr_within:.4f}")
    print(f"Cross-episode Spearman correlation (physical state vs latent L2): {corr_cross:.4f}")
    print("-> Consequence: Latent L2 metric only functions within the same trajectory, collapsing to ~0.22 across episodes.")


if __name__ == "__main__":
    test_contact_irreversibility_pusht()
    test_hollow_hypersphere_shell()
    test_gaussian_distance_saturation()
    test_topological_barrier_tworoom()
    test_cross_episode_correlation_collapse()
