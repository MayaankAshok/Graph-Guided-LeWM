"""Direct test of a specific mechanism: does potential-based shaping's value function
V'(s,g) = V(s,g) - Phi(s,g) = V(s,g) + d_graph(s,g) invert the true-distance ranking for
FAR-apart pairs, because the base value V(s,g) saturates near -1/(1-gamma) (~-100 for
gamma=0.99) while the added +d_graph term is unbounded and grows linearly with distance?
If so, shaped's per-distance-bin Spearman should degrade or invert specifically at large
true_dist, while baseline's should stay flat/mildly-degrading (it saturates too, but
doesn't invert -- it has no unbounded term added on top).

Uses the ALREADY-TRAINED mixed/baseline/s0 and mixed/shaped/s0 checkpoints from the main
sweep (both complete, no new training needed) -- this is a mechanism probe on an existing
model, not a new training run.
"""

import sys
from pathlib import Path

import numpy as np
import torch
from scipy import stats as sps

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from tworoom_b0_graph_gate import DEV, build_true_distance_oracle, log
from tworoom_b3_convergence import precompute_eval_set
from tworoom_b3_datatiers import CKPT_DIR, GAMMA, build_tier_setups
from tworoom_b3_gciql_shaping import MLP

HIDDEN = 256
N_PAIRS = 8000
N_BINS = 10


def load_v_net(tier, name, seed, d):
    path = CKPT_DIR / f"{tier}__{name}__s{seed}.pt"
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    v_net = MLP(2 * d, HIDDEN)
    v_net.load_state_dict(ckpt["v_net"])
    v_net.to(DEV).eval()
    return v_net, ckpt["step"], ckpt["done"]


def main():
    log(f"device={DEV}  effective horizon 1/(1-gamma) = {1/(1-GAMMA):.1f} steps (gamma={GAMMA})")
    setup = build_tier_setups(["mixed"])["mixed"]
    z, proprio, phi_dist, d = setup["z"], setup["proprio"], setup["phi_dist"], setup["d"]
    z_t = torch.from_numpy(z).to(DEV)

    true_dist_oracle = build_true_distance_oracle()
    all_rows = np.arange(len(z))
    ei, ej, true_d = precompute_eval_set(all_rows, proprio, true_dist_oracle, n_pairs=N_PAIRS, seed=999)
    graph_d = phi_dist[ei, ej]
    log(f"[eval set] n={len(ei)} pairs, true_dist range [{true_d.min():.1f}, {true_d.max():.1f}] "
        f"(median={np.median(true_d):.1f}), graph_dist range [{graph_d.min():.1f}, {graph_d.max():.1f}] "
        f"(median={np.median(graph_d):.1f})")

    zsg = torch.cat([z_t[ei], z_t[ej]], dim=-1)

    order = np.argsort(true_d)
    n = len(true_d)
    bin_edges = np.linspace(0, n, N_BINS + 1).astype(int)

    v_by_name = {}
    for name in ["baseline", "shaped"]:
        v_net, step, done = load_v_net("mixed", name, 0, d)
        with torch.no_grad():
            v = v_net(zsg).squeeze(-1).cpu().numpy()
        v_by_name[name] = v
        rho, p = sps.spearmanr(true_d, -v)
        log(f"\n=== {name} (step={step}, done={done}) === overall Spearman(true_dist, -V) = {rho:.4f} (p={p:.1e})")
        log(f"{'bin':>4} {'true_dist range':>20} {'n':>6} {'spearman':>10} {'mean_V':>10} {'mean_graph_d':>13}")
        for b in range(N_BINS):
            idx = order[bin_edges[b]:bin_edges[b + 1]]
            if len(idx) < 20:
                continue
            rho_b, _ = sps.spearmanr(true_d[idx], -v[idx])
            log(f"{b:>4} {f'{true_d[idx].min():.0f}-{true_d[idx].max():.0f}':>20} {len(idx):>6} "
                f"{rho_b:>10.3f} {v[idx].mean():>10.3f} {graph_d[idx].mean():>13.1f}")

    # direct check of the theoretical identity: V'(s,g) - V(s,g) ~= +graph_dist(s,g) (up to the
    # boundary/gamma^T term the telescoping sum drops). If the mechanism is real, this
    # difference should track graph_dist roughly linearly, especially at large distance.
    log("\n=== V'(shaped) - V(baseline) vs graph_dist (identity predicts slope ~+1 at large d) ===")
    delta = v_by_name["shaped"] - v_by_name["baseline"]
    rho_delta, p_delta = sps.spearmanr(graph_d, delta)
    slope, intercept = np.polyfit(graph_d, delta, 1)
    log(f"Spearman(graph_dist, V_shaped - V_baseline) = {rho_delta:.4f} (p={p_delta:.1e})")
    log(f"linear fit: delta ~= {slope:.4f} * graph_dist + {intercept:.4f}  (theory predicts slope near +1)")
    for b in range(N_BINS):
        idx = order[bin_edges[b]:bin_edges[b + 1]]
        if len(idx) < 20:
            continue
        log(f"  bin {b} (true_dist {true_d[idx].min():.0f}-{true_d[idx].max():.0f}): "
            f"mean_delta={delta[idx].mean():.3f} mean_graph_d={graph_d[idx].mean():.1f}")


if __name__ == "__main__":
    main()
