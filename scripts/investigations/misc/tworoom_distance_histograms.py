"""Squared latent-distance histograms for three pair types, plus a same-trajectory
distance-vs-timestep-delta scatter. Uses the cached 300-episode B0 landmarks
(no re-encoding) so this is purely a re-analysis of already-verified encodings.

Three comparison sets:
  A. Consecutive (adjacent) pairs within the same trajectory -- the null used to
     estimate rho_hat and calibrate the identification threshold.
  B. Random (non-adjacent) pairs within the same trajectory, at varying |delta step|.
  C. Random pairs from different trajectories -- the "unrelated" reference.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as sps

ROOT = Path(__file__).resolve().parent.parent
LANDMARKS = ROOT / "outputs" / "b0_tworoom" / "landmarks_ep300.npz"
OUT_DIR = ROOT / "outputs" / "b0_tworoom"
OUT_DIR.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(0)
D = 192


def main():
    d = np.load(LANDMARKS)
    z, ep_idx, step_idx = d["z"], d["ep_idx"], d["step_idx"]
    n = z.shape[0]
    print(f"loaded {n} landmarks")

    # ---- group frames by episode for fast within-episode sampling ----
    order = np.argsort(ep_idx, kind="stable")
    ep_sorted = ep_idx[order]
    boundaries = np.nonzero(np.diff(ep_sorted))[0] + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [len(order)]))
    episodes = [order[s:e] for s, e in zip(starts, ends)]  # list of arrays of landmark-row indices
    episodes = [e for e in episodes if len(e) >= 2]
    print(f"{len(episodes)} episodes with >=2 frames")

    # ---- Set A: adjacent (consecutive-step) pairs, same trajectory ----
    ord2 = np.lexsort((step_idx, ep_idx))
    ep_o, step_o = ep_idx[ord2], step_idx[ord2]
    adj_mask = (ep_o[1:] == ep_o[:-1]) & (step_o[1:] == step_o[:-1] + 1)
    ai, aj = ord2[:-1][adj_mask], ord2[1:][adj_mask]
    sqd_adjacent = np.sum((z[ai] - z[aj]) ** 2, axis=1)
    rho_hat = float(np.mean(np.sum(z[ai] * z[aj], axis=1)) / D)
    print(f"Set A (adjacent): n={len(sqd_adjacent)}  mean={sqd_adjacent.mean():.1f}  rho_hat={rho_hat:.4f}")

    # ---- Set B: random (non-adjacent) pairs within the same trajectory ----
    N_B = 60000
    bi, bj, delta_step = [], [], []
    tries = 0
    while len(bi) < N_B and tries < N_B * 5:
        tries += 1
        ep_rows = episodes[rng.integers(0, len(episodes))]
        if len(ep_rows) < 2:
            continue
        pick = rng.choice(len(ep_rows), size=2, replace=False)
        r1, r2 = ep_rows[pick[0]], ep_rows[pick[1]]
        bi.append(r1); bj.append(r2)
        delta_step.append(abs(int(step_idx[r1]) - int(step_idx[r2])))
    bi, bj, delta_step = np.array(bi), np.array(bj), np.array(delta_step)
    sqd_random_same = np.sum((z[bi] - z[bj]) ** 2, axis=1)
    print(f"Set B (random, same trajectory): n={len(sqd_random_same)}  mean={sqd_random_same.mean():.1f}  "
          f"delta_step range=[{delta_step.min()},{delta_step.max()}]")

    # ---- Set C: random pairs from different trajectories ----
    N_C = 150000
    ci = rng.integers(0, n, N_C)
    cj = rng.integers(0, n, N_C)
    keep = ep_idx[ci] != ep_idx[cj]
    ci, cj = ci[keep], cj[keep]
    sqd_diff = np.sum((z[ci] - z[cj]) ** 2, axis=1)
    print(f"Set C (different trajectories): n={len(sqd_diff)}  mean={sqd_diff.mean():.1f}")

    eps2 = 2 * (1 - rho_hat) * sps.chi2.ppf(1e-3, D)
    print(f"identification threshold eps^2 (q=1e-3) = {eps2:.1f}")

    # ============================================================
    # Figure 1: overlaid density histograms
    # ============================================================
    fig, ax = plt.subplots(figsize=(9, 5.5))
    bins = np.linspace(0, 650, 90)

    ax.hist(sqd_adjacent, bins=bins, density=True, alpha=0.55, color="#3A4CC4",
            label=f"A: consecutive, same trajectory (n={len(sqd_adjacent):,})")
    ax.hist(sqd_random_same, bins=bins, density=True, alpha=0.5, color="#96590A",
            label=f"B: random pair, same trajectory (n={len(sqd_random_same):,})")
    ax.hist(sqd_diff, bins=bins, density=True, alpha=0.45, color="#186F63",
            label=f"C: random pair, different trajectories (n={len(sqd_diff):,})")

    # theoretical chi-squared curve for set A (rho=rho_hat: models real one-step
    # transitions -- what the identification threshold is actually calibrated against)
    # and for set C (rho=0: models two fully independent draws from the marginal) --
    # a check of BOTH parametric assumptions against the data, not just one
    xs = np.linspace(1, 650, 500)
    chi2_pdf_A = sps.chi2.pdf(xs / (2 * (1 - rho_hat)), D) / (2 * (1 - rho_hat))
    chi2_pdf_C = sps.chi2.pdf(xs / 2.0, D) / 2.0
    ax.plot(xs, chi2_pdf_A, "--", color="#12161F", linewidth=1.5,
            label=r"theoretical $\chi^2_d$ for A ($\rho=\hat\rho$, real one-step transitions)")
    ax.plot(xs, chi2_pdf_C, "-.", color="#12161F", linewidth=1.5,
            label=r"theoretical $\chi^2_d$ for C ($\rho=0$, independent draws)")

    ax.axvline(eps2, color="crimson", linewidth=1.5, linestyle=":")
    ax.text(eps2 + 8, ax.get_ylim()[1] * 0.92 if ax.get_ylim()[1] else 0.01,
            r"identification cutoff $\varepsilon^2$" + f"\n(q=1e-3) = {eps2:.0f}",
            color="crimson", fontsize=9, va="top")

    ax.set_xlabel(r"squared latent distance $\|\Delta z\|^2$")
    ax.set_ylabel("probability density")
    ax.set_title("Two-Room: squared latent-distance distributions by pair type\n"
                  "(300 episodes, 27,675 landmarks, real pretrained checkpoint)")
    ax.legend(fontsize=9, loc="upper right")
    ax.set_xlim(0, 650)
    fig.tight_layout()
    out1 = OUT_DIR / "distance_histograms.png"
    fig.savefig(out1, dpi=150)
    print(f"wrote {out1}")

    # ============================================================
    # Figure 2: same-trajectory scatter, latent distance vs |delta step|
    # ============================================================
    fig2, ax2 = plt.subplots(figsize=(9, 5.5))
    ax2.scatter(delta_step, sqd_random_same, s=3, alpha=0.12, color="#96590A", edgecolors="none")

    # median trend line per delta_step bin -- finer resolution near delta=1 where the
    # latent distance is climbing fastest, coarser further out where it's flat
    max_ds = delta_step.max()
    bin_edges = np.concatenate([np.arange(0, 21, 1), np.arange(25, max_ds + 5, 5)])
    bin_idx = np.digitize(delta_step, bin_edges)
    med_x, med_y = [], []
    for b in range(1, len(bin_edges)):
        m = bin_idx == b
        if m.sum() >= 20:
            med_x.append((bin_edges[b - 1] + bin_edges[b]) / 2)
            med_y.append(np.median(sqd_random_same[m]))
    ax2.plot(med_x, med_y, color="#3A4CC4", linewidth=2.2,
             label="median (1-step bins up to 20, 5-step bins beyond)")

    ax2.axhline(2 * D, color="#186F63", linestyle="--", linewidth=1.3,
                label=r"$2d$ = fully-uncorrelated (marginal-null) level")
    ax2.axhline(eps2, color="crimson", linestyle=":", linewidth=1.3,
                label=r"identification cutoff $\varepsilon^2$")

    ax2.set_xlabel(r"$|\Delta\,\mathrm{step}|$ within the same trajectory")
    ax2.set_ylabel(r"squared latent distance $\|\Delta z\|^2$")
    ax2.set_title("Two-Room: latent distance vs. temporal separation, same trajectory")
    ax2.legend(fontsize=9, loc="lower right")
    ax2.set_xlim(0, max_ds)
    ax2.set_ylim(0, 650)
    fig2.tight_layout()
    out2 = OUT_DIR / "distance_vs_timestep_scatter.png"
    fig2.savefig(out2, dpi=150)
    print(f"wrote {out2}")


if __name__ == "__main__":
    main()
