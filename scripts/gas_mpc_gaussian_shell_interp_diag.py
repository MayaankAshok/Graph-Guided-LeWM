"""Gaussian-shell diagnostic, part 2: does the straight-line latent midpoint between two real
points fall inside the shell (docs/gas-mpc/main.tex's OOD-interpolation claim), for pairs of
frames drawn `--offset` steps apart within the same episode (default 100, i.e. the same-ep-100
protocol's own horizon)?

For each sampled (start, goal) pair this reports/plots three radii (all centred on the full
origin, i.e. r(x) = ||x||):
    r_start, r_goal   -- real encoded frames, expected to sit on the chi(d) shell (part 1).
    r_mid_lerp        -- ||0.5*(z_start+z_goal)||, the naive straight-line midpoint a
                          z-space-L2 CEM plan is implicitly aiming through.
    r_mid_real        -- the actual logged frame at the trajectory's true midpoint (start +
                          offset//2), i.e. what a real, non-interpolated state at that point in
                          time looks like -- the control for "maybe the task dynamics also dip
                          off-shell at the midpoint, so it isn't specifically about interpolating".

Diagnostic only -- reads outputs/<env>/cache_train.npz (or cache_full.npz as a fallback,
loudly noted); prepares no learned asset.

Run (Push-T; needs outputs/pusht/cache_train.npz or cache_full.npz):
    python scripts/gas_mpc_gaussian_shell_interp_diag.py
    python scripts/gas_mpc_gaussian_shell_interp_diag.py --offset 50 --n-pairs 50000
    # exact match to the paper's fixed same-ep-100 held-out eval pairs (n=200):
    python scripts/gas_mpc_gaussian_shell_interp_diag.py --eval-pairs outputs/pusht/pairs/pairs_same_episode_off100_task200u.json
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
from scipy import stats as sps

ROOT = Path(__file__).resolve().parent.parent


def load_cache(out_dir: Path):
    train_path = out_dir / "cache_train.npz"
    full_path = out_dir / "cache_full.npz"
    if train_path.exists():
        path = train_path
    elif full_path.exists():
        path = full_path
        print(f"[gaussian_shell_interp_diag] {train_path.name} not found; falling back to "
              f"{full_path.name} (all episodes, incl. eval -- fine for this diagnostic).")
    else:
        raise FileNotFoundError(
            f"neither {train_path} nor {full_path} exists -- run "
            f"'python scripts/gas_mpc_prepare.py encode' first")
    with np.load(path) as d:
        return d["z"], d["ep_offset"], d["ep_len"], path


def sample_offset_pairs(ep_offset, ep_len, offset, n_pairs, seed):
    """Uniformly sample (start_row, goal_row) pairs `offset` apart within a single episode,
    across all episodes long enough to support the offset -- no held-out or overlap
    restriction (diagnostic only, not an eval task pool)."""
    rng = np.random.default_rng(seed)
    eligible = np.where(ep_len > offset)[0]
    if eligible.size == 0:
        raise ValueError(f"no episode has more than {offset} steps")
    eps = rng.choice(eligible, size=n_pairs, replace=True)
    max_start = ep_len[eps] - offset - 1
    starts = (rng.random(n_pairs) * (max_start + 1)).astype(np.int64)
    start_rows = ep_offset[eps] + starts
    goal_rows = start_rows + offset
    return start_rows, goal_rows


def load_eval_pairs(path, ep_offset, offset_expected):
    d = json.loads(Path(path).read_text())
    assert d["pairing"] == "same_episode"
    assert d["offset"] == offset_expected, f"pairs file offset {d['offset']} != --offset {offset_expected}"
    return np.array(d["start_row"], dtype=np.int64), np.array(d["goal_row"], dtype=np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default=os.environ.get("GAS_MPC_ENV", "pusht"))
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--offset", type=int, default=100, help="steps apart within episode (same-ep-100 default)")
    ap.add_argument("--n-pairs", type=int, default=20_000,
                     help="pairs to freshly sample (ignored if --eval-pairs given)")
    ap.add_argument("--eval-pairs", default=None,
                     help="use the exact fixed task200u json instead of fresh sampling, "
                          "e.g. outputs/pusht/pairs/pairs_same_episode_off100_task200u.json")
    ap.add_argument("--n-bins", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fig-out", default=None)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else ROOT / "outputs" / args.env
    tag = f"off{args.offset}"
    fig_out = Path(args.fig_out) if args.fig_out else out_dir / "diagnostics" / f"gaussian_shell_interp_{tag}_hist.png"
    json_out = Path(args.json_out) if args.json_out else out_dir / "diagnostics" / f"gaussian_shell_interp_{tag}_stats.json"
    fig_out.parent.mkdir(parents=True, exist_ok=True)

    z, ep_offset, ep_len, src_path = load_cache(out_dir)
    n_total, d = z.shape
    mu = z.mean(axis=0)
    sigma_iso = float(np.sqrt(z.var(axis=0, ddof=1).sum() / d))
    chi_dist = sps.chi(df=d, scale=sigma_iso)

    if args.eval_pairs:
        start_rows, goal_rows = load_eval_pairs(args.eval_pairs, ep_offset, args.offset)
        source = f"eval pairs file {args.eval_pairs}"
    else:
        start_rows, goal_rows = sample_offset_pairs(ep_offset, ep_len, args.offset, args.n_pairs, args.seed)
        source = f"{args.n_pairs} freshly sampled offset-{args.offset} pairs (seed {args.seed})"
    n_pairs = start_rows.shape[0]
    print(f"[gaussian_shell_interp_diag] {n_pairs} pairs from {source}, cache {src_path}")

    z_start = z[start_rows]
    z_goal = z[goal_rows]
    z_mid_lerp = 0.5 * (z_start + z_goal)
    z_mid_real = z[start_rows + args.offset // 2]

    r_start = np.linalg.norm(z_start, axis=1)
    r_goal = np.linalg.norm(z_goal, axis=1)
    r_mid_lerp = np.linalg.norm(z_mid_lerp, axis=1)
    r_mid_real = np.linalg.norm(z_mid_real, axis=1)

    theo_mean, theo_std = float(chi_dist.mean()), float(chi_dist.std())
    # how far below the shell does the interpolated midpoint sit, in shell-sigma units?
    z_score = (theo_mean - r_mid_lerp.mean()) / theo_std
    # fraction of interpolated midpoints below the 1st percentile of the real shell distribution
    p01 = float(chi_dist.ppf(0.01))
    frac_below_p01 = float((r_mid_lerp < p01).mean())

    stats_out = {
        "env": args.env, "offset": args.offset, "n_pairs": int(n_pairs), "source": source,
        "latent_dim": int(d), "sigma_isotropic": sigma_iso,
        "shell_theoretical_mean": theo_mean, "shell_theoretical_std": theo_std,
        "r_start_mean": float(r_start.mean()), "r_start_std": float(r_start.std()),
        "r_goal_mean": float(r_goal.mean()), "r_goal_std": float(r_goal.std()),
        "r_mid_real_mean": float(r_mid_real.mean()), "r_mid_real_std": float(r_mid_real.std()),
        "r_mid_lerp_mean": float(r_mid_lerp.mean()), "r_mid_lerp_std": float(r_mid_lerp.std()),
        "shell_sigma_units_below_shell_mean": float(z_score),
        "shell_p01_radius": p01,
        "frac_lerp_midpoints_below_shell_p01": frac_below_p01,
    }
    json_out.write_text(json.dumps(stats_out, indent=2))
    print(json.dumps(stats_out, indent=2))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    r_all_max = max(r_start.max(), r_goal.max(), r_mid_real.max(), r_mid_lerp.max())
    r_grid = np.linspace(0, r_all_max * 1.05, 2000)
    bins = np.linspace(0, r_all_max * 1.05, args.n_bins)

    # sized to the paper's rendered width (0.56 * 5.5in ICLR text width) so fonts print at
    # body-text size without scaling. Rendered by LaTeX with the paper's own `times` package
    # (Times text, Computer Modern math) so the fonts match the document exactly.
    plt.rcParams.update({"text.usetex": True, "font.family": "serif",
                         "text.latex.preamble": r"\usepackage{times}",
                         "font.size": 9, "axes.labelsize": 10, "legend.fontsize": 9})
    fig, ax = plt.subplots(figsize=(3.1, 2.3))
    ax.hist(r_mid_lerp, bins=bins, density=True, alpha=0.55, color="tab:blue",
            label="straight-line\nmidpoint")
    ax.plot(r_grid, chi_dist.pdf(r_grid), color="tab:red", lw=2,
            label=f"$\\chi_{{{d}}}$ shell")
    ax.axvline(r_mid_lerp.mean(), color="tab:blue", ls="--", lw=1.5)
    ax.axvline(theo_mean, color="tab:red", ls="--", lw=1.5)
    ax.set_xlabel(r"$\|z\|$")
    ax.set_ylabel("density")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(fig_out, dpi=300)
    print(f"[gaussian_shell_interp_diag] wrote {fig_out}")
    print(f"[gaussian_shell_interp_diag] wrote {json_out}")


if __name__ == "__main__":
    main()
