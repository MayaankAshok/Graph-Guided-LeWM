"""Scatter: CEM's predicted latent cost (what it optimized against its OWN predictor's
imagined rollout) vs. the realized latent distance after actually simulating the chosen
open-loop action sequence in the real env and re-encoding the true final frame -- same
quantity (sum of squared differences over the 192-d CLS latent, LeWM's own criterion),
computed twice: once inside the predictor's head, once against physical reality. For the
l2_single_pass_diagnostic run. Color-coded by success/failure (PushT's own eval_state: pos
err < 20px and angle err < pi/9). See docs/gas-mpc/main.tex, Diagnostics section.

Run: python scripts/diagnostics/plot_l2_single_pass_diagnostic.py
"""

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "outputs" / "pusht" / "pairs" / "l2_single_pass_off25_B25__task200__s0__n200__v2.json"
OUT = ROOT / "docs" / "gas-mpc" / "fig_l2_single_pass_predicted_vs_realized.png"


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import spearmanr

    d = json.loads(RES.read_text())
    pc = np.array(d["predicted_cost"])
    rl = np.array(d["realized_latent_dist"])
    fh = np.array(d["first_hit_step"])
    success = fh >= 0

    rho, p = spearmanr(pc, rl)
    lo, hi = min(pc.min(), rl.min()), max(pc.max(), rl.max())

    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    ax.plot([lo, hi], [lo, hi], color="gray", ls=":", lw=1, alpha=0.7, label="predicted = realized")
    ax.scatter(pc[success], rl[success], c="tab:blue", marker="o", s=28, alpha=0.75,
               label=f"success (n={int(success.sum())})", zorder=3)
    ax.scatter(pc[~success], rl[~success], c="tab:red", marker="x", s=34, alpha=0.85,
               label=f"failure (n={int((~success).sum())})", zorder=3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo * 0.7, hi * 1.4)
    ax.set_ylim(lo * 0.7, hi * 1.4)
    ax.set_xlabel(r"predicted CEM cost $\|\hat z_{25}-z_g\|^2$ (predictor's own rollout, log scale)")
    ax.set_ylabel(r"realized latent distance $\|z_{25}-z_g\|^2$ (true rollout, log scale)")
    ax.set_title(f"L2 single-pass (budget=25, offset=25, n=200)\n"
                 f"Spearman($C^{{pred}}$, realized latent dist.) = {rho:.3f} (p={p:.1e})")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT, dpi=150)
    print(f"wrote {OUT}")
    print(f"spearman(predicted_cost, realized_latent_dist) = {rho:.4f} (p={p:.2e})")
    print(f"success {int(success.sum())}/{len(success)} ({100*success.mean():.1f}%)")
    print(f"realized > predicted in {100*(rl > pc).mean():.1f}% of tasks")
    print(f"median realized latent dist: success {np.median(rl[success]):.2f}, "
          f"failure {np.median(rl[~success]):.2f}")
    print(f"median predicted cost: success {np.median(pc[success]):.3f}, "
          f"failure {np.median(pc[~success]):.3f}")


if __name__ == "__main__":
    main()
