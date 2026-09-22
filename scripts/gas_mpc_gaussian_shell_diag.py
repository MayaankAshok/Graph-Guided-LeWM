"""Gaussian-shell diagnostic: does the LeWM latent look like a high-dimensional isotropic
Gaussian, or does it concentrate more/less tightly than one would?

Motivation (docs/gas-mpc/main.tex "graph gives long-horizon structure" narrative): the claim
is that a straight-line latent path between two real points passes through the *interior* of
the data cloud, which is anomalously low-density in high dimensions because an isotropic
Gaussian's mass concentrates on a thin shell around radius sigma*sqrt(d) (see
https://aseemrb.me/blog/high-dimensional-gaussians-on-a-sphere/). This script only checks the
premise: does ||z - mean(z)|| for real encoded frames look like the textbook chi(d) shell
distribution predicted for an isotropic Gaussian fit to the same mean/variance? It does not
yet touch the interpolation-goes-OOD claim itself (that needs a second script).

Diagnostic only -- reads outputs/<env>/cache_train.npz (or cache_full.npz as a fallback,
loudly noted) purely to look at the marginal geometry of the latent cloud; it prepares no
learned asset and is exempt from the training-only provenance rule in CLAUDE.md for that
reason.

Run (Push-T; needs outputs/pusht/cache_train.npz or cache_full.npz -- run
gas_mpc_prepare.py encode first if neither exists):
    python scripts/gas_mpc_gaussian_shell_diag.py
    python scripts/gas_mpc_gaussian_shell_diag.py --env reacher --n-samples 200000
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
from scipy import stats as sps

ROOT = Path(__file__).resolve().parent.parent


def load_latents(out_dir: Path, n_samples: int, seed: int):
    train_path = out_dir / "cache_train.npz"
    full_path = out_dir / "cache_full.npz"
    if train_path.exists():
        path = train_path
    elif full_path.exists():
        path = full_path
        print(f"[gaussian_shell_diag] {train_path.name} not found; falling back to "
              f"{full_path.name} (all episodes, incl. eval -- fine for this diagnostic, "
              f"which only looks at marginal latent geometry, not a learned asset).")
    else:
        raise FileNotFoundError(
            f"neither {train_path} nor {full_path} exists -- run "
            f"'python scripts/gas_mpc_prepare.py encode' first")
    with np.load(path) as d:
        z = d["z"]
    n_total = z.shape[0]
    rng = np.random.default_rng(seed)
    if n_total > n_samples:
        idx = rng.choice(n_total, size=n_samples, replace=False)
        z_sample = z[idx]
    else:
        z_sample = z
    return z, z_sample, path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default=os.environ.get("GAS_MPC_ENV", "pusht"))
    ap.add_argument("--out-dir", default=None,
                     help="override outputs/<env> root (default outputs/<env>/)")
    ap.add_argument("--n-samples", type=int, default=200_000,
                     help="subsample size for the histogram (mean/var are still estimated "
                          "from the full population)")
    ap.add_argument("--n-bins", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fig-out", default=None,
                     help="default outputs/<env>/diagnostics/gaussian_shell_r_hist.png")
    ap.add_argument("--json-out", default=None,
                     help="default outputs/<env>/diagnostics/gaussian_shell_r_stats.json")
    args = ap.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else ROOT / "outputs" / args.env
    fig_out = Path(args.fig_out) if args.fig_out else out_dir / "diagnostics" / "gaussian_shell_r_hist.png"
    json_out = Path(args.json_out) if args.json_out else out_dir / "diagnostics" / "gaussian_shell_r_stats.json"
    fig_out.parent.mkdir(parents=True, exist_ok=True)

    z_full, z_sample, src_path = load_latents(out_dir, args.n_samples, args.seed)
    n_total, d = z_full.shape
    print(f"[gaussian_shell_diag] loaded {n_total} frames, d={d}, from {src_path}; "
          f"histogram uses {z_sample.shape[0]} of them")

    # Full-population moments (cheap: two passes over an (N, d) float32 array).
    mu = z_full.mean(axis=0)
    per_dim_var = z_full.var(axis=0, ddof=1)
    trace_cov = per_dim_var.sum()
    sigma2_iso = trace_cov / d          # matched isotropic variance: trace(Sigma)/d
    sigma_iso = float(np.sqrt(sigma2_iso))
    # Participation ratio: (sum var_i)^2 / sum(var_i^2), in [1, d]. d means perfectly
    # isotropic; small values mean most of the variance sits in a few directions, in which
    # case the isotropic chi(d) curve below is a poor model of the true shape.
    participation_ratio = float(trace_cov ** 2 / np.sum(per_dim_var ** 2))

    # Actual: distance of each sampled point from the data centroid.
    r = np.linalg.norm(z_sample - mu, axis=1)

    # Theoretical: if z ~ N(mu, sigma_iso^2 * I_d), then ||z - mu|| / sigma_iso ~ chi(df=d).
    chi_dist = sps.chi(df=d, scale=sigma_iso)
    r_grid = np.linspace(0, r.max() * 1.05, 2000)
    theo_pdf = chi_dist.pdf(r_grid)
    theo_mean = float(chi_dist.mean())
    theo_std = float(chi_dist.std())

    emp_mean = float(r.mean())
    emp_std = float(r.std())

    stats_out = {
        "env": args.env,
        "source_cache": str(src_path),
        "n_total_frames": int(n_total),
        "n_histogram_samples": int(z_sample.shape[0]),
        "latent_dim": int(d),
        "sigma_isotropic": sigma_iso,
        "participation_ratio": participation_ratio,
        "participation_ratio_frac_of_d": participation_ratio / d,
        "mean_offset_norm": float(np.linalg.norm(mu)),
        "empirical_r_mean": emp_mean,
        "empirical_r_std": emp_std,
        "theoretical_chi_mean": theo_mean,
        "theoretical_chi_std": theo_std,
        "mean_ratio_empirical_over_theoretical": emp_mean / theo_mean,
        "std_ratio_empirical_over_theoretical": emp_std / theo_std,
    }
    json_out.write_text(json.dumps(stats_out, indent=2))
    print(json.dumps(stats_out, indent=2))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(r, bins=args.n_bins, density=True, alpha=0.55, color="tab:blue",
            label=f"actual $\\|z-\\mu\\|$ ({args.env}, n={z_sample.shape[0]})")
    ax.plot(r_grid, theo_pdf, color="tab:red", lw=2,
            label=f"theoretical $\\chi_{{{d}}}(\\sigma={sigma_iso:.3g})$, isotropic-Gaussian fit")
    ax.axvline(emp_mean, color="tab:blue", ls="--", lw=1,
               label=f"actual mean = {emp_mean:.2f}")
    ax.axvline(theo_mean, color="tab:red", ls="--", lw=1,
               label=f"theoretical mean = {theo_mean:.2f}")
    ax.set_xlabel(r"$\|z-\mu\|$")
    ax.set_ylabel("density")
    ax.set_title(f"{args.env}: latent norm-from-centroid, actual vs.\ isotropic-Gaussian shell\n"
                 f"participation ratio {participation_ratio:.1f}/{d} "
                 f"({100*participation_ratio/d:.0f}% of dims effectively used)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_out, dpi=150)
    print(f"[gaussian_shell_diag] wrote {fig_out}")
    print(f"[gaussian_shell_diag] wrote {json_out}")


if __name__ == "__main__":
    main()
