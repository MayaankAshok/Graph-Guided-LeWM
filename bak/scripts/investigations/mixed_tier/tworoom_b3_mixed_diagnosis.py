"""Why does graph-potential shaping fail worst on the mixed tier (50 real expert + 50
live noisy-rollout episodes) -- worse than on any pure-expert tier, including expert_10
which has 1/10th the data? See docs/graph-proposal/main.tex sec:b3-results and memory
tworoom-b3-datatier-sweep for the result this is debugging.

Central trick: the mixed tier's 50 real-expert episodes are the EXACT same 50 episodes
(same seed=0 selection inside load_landmarks) used to build the standalone expert_50
tier. tworoom_b3_datatiers._build_mixed_arrays concatenates real-expert rows FIRST, so
rows [0:n_expert_rows) of the mixed tier's z/ep_idx/step_idx/proprio are row-for-row
IDENTICAL to expert_50's arrays -- same states, same true distances. That lets us hold
the data fixed and isolate one variable: does adding 50 noisy episodes' worth of nodes
to the GRAPH degrade Phi even for pairs that never directly touch a noisy state, by
giving Dijkstra bad shortcuts through the noisy component?

Diagnostics, in order:
  1. Identification-edge composition and degree, split by node origin (expert/noisy).
  2. False-edge rate (measured true-position gap > 12px), split by edge composition.
  3. Connectivity: does the noisy component hang off the expert component, or is it
     stitched in throughout (i.e. are there many expert<->noisy identification edges)?
  4. THE key test: for the SAME expert-origin (s,g) row pairs, compare Phi computed on
     the mixed-tier graph (which also contains noisy nodes) vs Phi computed on the
     standalone expert_50 graph (pure). If mixed-graph Phi is worse/biased even here,
     noisy nodes are polluting the graph as bad shortcuts, not just being individually
     mis-estimated where they're the query state.
  5. HER contamination: of the actual training tuples (same construction as training),
     what fraction come from noisy episodes, and how does the shaped-reward increment
     (gamma*Phi(next)-Phi(s)) differ in scale between noisy-episode and expert-episode
     tuples?
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats as sps
from scipy.sparse.csgraph import connected_components

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from tworoom_b0_graph_gate import DEV, build_true_distance_oracle, build_weighted_graph, find_graph_edges, log
from tworoom_b3_datatiers import (
    GAMMA, MIXED_N_EXPERT, NOISY_EP_ID_OFFSET, Q_CALIB, TIER_CACHE_DIR,
    _build_mixed_arrays, _build_setup, _load_real_tier_arrays, build_tier_setups,
)
from tworoom_b3_gciql_shaping import build_her_tuples

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "outputs" / "b3_tworoom"
rng = np.random.default_rng(0)


def rho_hat_of(z, ep_idx, step_idx):
    d = z.shape[1]
    order = np.lexsort((step_idx, ep_idx))
    ep_o, step_o = ep_idx[order], step_idx[order]
    adj = (ep_o[1:] == ep_o[:-1]) & (step_o[1:] == step_o[:-1] + 1)
    z_o = z[order]
    if adj.sum() == 0:
        return float("nan")
    return float(np.mean(np.sum(z_o[:-1][adj] * z_o[1:][adj], axis=1)) / d)


def main():
    log(f"device={DEV}")
    results = {}

    # ---- load mixed tier (from cache) and its constituent arrays ----
    log("\n=== loading mixed tier ===")
    mz, mep, mstep, mproprio, maction = _build_mixed_arrays()
    n = mz.shape[0]
    is_noisy = mep >= NOISY_EP_ID_OFFSET
    n_expert_rows = int((~is_noisy).sum())
    log(f"[mixed] {n} rows total, {n_expert_rows} expert-origin, {n - n_expert_rows} noisy-origin")

    d = mz.shape[1]
    eps2 = 2 * (1 - rho_hat_of(mz, mep, mstep)) * sps.chi2.ppf(Q_CALIB, d)
    trans_i, trans_j, id_i, id_j = find_graph_edges(mz, mep, mstep, eps2)
    graph = build_weighted_graph(n, trans_i, trans_j, id_i, id_j, id_weight=1.0)

    # ---- diagnostic 0: calibration -- does pooling noisy transitions shift rho_hat/eps2? ----
    log("\n=== diagnostic 0: calibration, pooled vs origin-stratified ===")
    rho_pooled = rho_hat_of(mz, mep, mstep)
    rho_expert_only = rho_hat_of(mz[~is_noisy], mep[~is_noisy], mstep[~is_noisy])
    rho_noisy_only = rho_hat_of(mz[is_noisy], mep[is_noisy], mstep[is_noisy])
    eps2_expert_only = 2 * (1 - rho_expert_only) * sps.chi2.ppf(Q_CALIB, d)
    eps2_noisy_only = 2 * (1 - rho_noisy_only) * sps.chi2.ppf(Q_CALIB, d)
    log(f"rho_hat: pooled={rho_pooled:.4f} (eps2={eps2:.2f})  "
        f"expert-only={rho_expert_only:.4f} (eps2={eps2_expert_only:.2f})  "
        f"noisy-only={rho_noisy_only:.4f} (eps2={eps2_noisy_only:.2f})")
    results["calibration"] = dict(rho_pooled=rho_pooled, eps2_pooled=float(eps2),
                                   rho_expert_only=rho_expert_only, eps2_expert_only=float(eps2_expert_only),
                                   rho_noisy_only=rho_noisy_only, eps2_noisy_only=float(eps2_noisy_only))

    # ---- diagnostic 1: identification-edge composition ----
    log("\n=== diagnostic 1: identification-edge composition ===")
    i_noisy, j_noisy = is_noisy[id_i], is_noisy[id_j]
    ee = int(np.sum(~i_noisy & ~j_noisy))
    en = int(np.sum(i_noisy != j_noisy))
    nn = int(np.sum(i_noisy & j_noisy))
    log(f"identification edges: expert-expert={ee} expert-noisy={en} noisy-noisy={nn} (total={len(id_i)})")

    deg = np.zeros(n, dtype=np.int64)
    np.add.at(deg, id_i, 1)
    np.add.at(deg, id_j, 1)
    log(f"avg id-degree: expert-origin nodes={deg[~is_noisy].mean():.2f}  noisy-origin nodes={deg[is_noisy].mean():.2f}")
    n_noisy_isolated = int(np.sum(is_noisy & (deg == 0)))
    log(f"noisy-origin nodes with ZERO identification edges (rely on transition edges only): "
        f"{n_noisy_isolated}/{n - n_expert_rows} ({100*n_noisy_isolated/max(1,n-n_expert_rows):.1f}%)")
    results["edge_composition"] = dict(
        expert_expert=ee, expert_noisy=en, noisy_noisy=nn, total=int(len(id_i)),
        avg_degree_expert=float(deg[~is_noisy].mean()), avg_degree_noisy=float(deg[is_noisy].mean()),
        n_noisy_isolated=n_noisy_isolated,
    )

    # ---- diagnostic 2: false-edge rate stratified by composition ----
    log("\n=== diagnostic 2: false-edge rate by composition (true gap > 12px) ===")
    true_dist_oracle = build_true_distance_oracle()

    def false_rate(mask, n_sample=2000):
        idx = np.nonzero(mask)[0]
        if len(idx) == 0:
            return float("nan"), 0
        sample = rng.choice(idx, size=min(n_sample, len(idx)), replace=False)
        td = np.array([
            true_dist_oracle(mproprio[id_i[k]:id_i[k]+1], mproprio[id_j[k]:id_j[k]+1])[0, 0]
            for k in sample
        ])
        return float(np.mean(td > 12.0)), len(sample)

    fr_ee, n_ee_s = false_rate(~i_noisy & ~j_noisy)
    fr_en, n_en_s = false_rate(i_noisy != j_noisy)
    fr_nn, n_nn_s = false_rate(i_noisy & j_noisy)
    log(f"false-edge rate: expert-expert={fr_ee*100:.1f}% (n={n_ee_s})  "
        f"expert-noisy={fr_en*100:.1f}% (n={n_en_s})  noisy-noisy={fr_nn*100:.1f}% (n={n_nn_s})")
    results["false_edge_rate"] = dict(expert_expert=fr_ee, expert_noisy=fr_en, noisy_noisy=fr_nn)

    # ---- diagnostic 3: connectivity ----
    log("\n=== diagnostic 3: connectivity ===")
    n_comp, labels = connected_components(graph, directed=False)
    sizes = np.bincount(labels)
    giant = np.argmax(sizes)
    log(f"{n_comp} connected components, giant component size={sizes.max()}/{n} ({100*sizes.max()/n:.1f}%)")
    frac_expert_in_giant = float(np.mean(labels[~is_noisy] == giant))
    frac_noisy_in_giant = float(np.mean(labels[is_noisy] == giant))
    log(f"fraction IN giant component: expert-origin={frac_expert_in_giant*100:.1f}%  noisy-origin={frac_noisy_in_giant*100:.1f}%")
    results["connectivity"] = dict(n_components=int(n_comp), giant_frac=float(sizes.max()/n),
                                    frac_expert_in_giant=frac_expert_in_giant, frac_noisy_in_giant=frac_noisy_in_giant)

    # ---- diagnostic 4: THE key test -- matched expert-only quality, mixed graph vs pure expert_50 graph ----
    log("\n=== diagnostic 4: expert-only Phi quality, mixed-graph vs pure-expert_50-graph (matched rows) ===")
    ez, e_ep, e_step, e_proprio, e_action = _load_real_tier_arrays(MIXED_N_EXPERT)
    same_rows = (mz[:n_expert_rows].shape == ez.shape) and np.allclose(mz[:n_expert_rows], ez)
    log(f"row-alignment check (mixed[:n_expert_rows] == standalone expert_{MIXED_N_EXPERT} arrays): {same_rows}")
    if not same_rows:
        log("WARNING: row alignment assumption failed -- diagnostic 4 skipped, see script docstring")
        results["matched_expert_quality"] = None
    else:
        expert50_setup = build_tier_setups(["expert_50"])["expert_50"]
        phi_expert50 = expert50_setup["phi_dist"]  # (n_expert_rows, n_expert_rows)

        log("[mixed] loading/using cached mixed phi_dist for the same rows...")
        mixed_setup = _build_setup("mixed", mz, mep, mstep, mproprio, maction)
        phi_mixed_full = mixed_setup["phi_dist"]

        n_pairs = 3000
        si = rng.integers(0, n_expert_rows, n_pairs)
        gi = rng.integers(0, n_expert_rows, n_pairs)
        keep = si != gi
        si, gi = si[keep], gi[keep]

        true_d = np.array([
            true_dist_oracle(mproprio[si[k]:si[k]+1], mproprio[gi[k]:gi[k]+1])[0, 0] for k in range(len(si))
        ])
        phi_d_mixed = phi_mixed_full[si, gi]
        phi_d_pure = phi_expert50[si, gi]

        finite = np.isfinite(true_d) & np.isfinite(phi_d_mixed) & np.isfinite(phi_d_pure)
        true_d, phi_d_mixed, phi_d_pure = true_d[finite], phi_d_mixed[finite], phi_d_pure[finite]

        rho_mixed, _ = sps.spearmanr(true_d, phi_d_mixed)
        rho_pure, _ = sps.spearmanr(true_d, phi_d_pure)
        delta = phi_d_mixed - phi_d_pure
        log(f"Spearman(true, graph-dist) on IDENTICAL expert-only pairs: "
            f"mixed-tier graph={rho_mixed:.4f}   pure expert_50 graph={rho_pure:.4f}")
        log(f"phi_dist(mixed) - phi_dist(expert_50) over these pairs: "
            f"mean={delta.mean():.3f} std={delta.std():.3f} "
            f"frac_shorter_in_mixed={float(np.mean(delta < -1e-6)):.3f} "
            f"frac_identical={float(np.mean(np.abs(delta) < 1e-6)):.3f} "
            f"frac_longer_in_mixed={float(np.mean(delta > 1e-6)):.3f}")
        results["matched_expert_quality"] = dict(
            n_pairs=int(len(si)), spearman_mixed_graph=float(rho_mixed), spearman_pure_expert50_graph=float(rho_pure),
            delta_mean=float(delta.mean()), delta_std=float(delta.std()),
            frac_shorter_in_mixed=float(np.mean(delta < -1e-6)), frac_longer_in_mixed=float(np.mean(delta > 1e-6)),
        )

    # ---- diagnostic 5: HER contamination in actual training tuples ----
    log("\n=== diagnostic 5: HER training-tuple contamination ===")
    mixed_setup = results.get("_mixed_setup") or _build_setup("mixed", mz, mep, mstep, mproprio, maction)
    train_eps = mixed_setup["train_eps"]
    s_idx, next_idx, goal_idx, act_idx, done_arr = build_her_tuples(
        mep, mstep, maction, seed=0, allowed_episode_ids=train_eps
    )
    tuple_is_noisy = is_noisy[s_idx]  # goal/next share the episode with s, so this is the whole tuple's origin
    frac_noisy_tuples = float(tuple_is_noisy.mean())
    log(f"train HER tuples: {len(s_idx)} total, {frac_noisy_tuples*100:.1f}% from noisy episodes")

    phi_dist = mixed_setup["phi_dist"]
    phi_s = -phi_dist[s_idx, goal_idx]
    phi_n = -phi_dist[next_idx, goal_idx]
    shaped_increment = GAMMA * phi_n - phi_s

    for label, mask in [("expert-episode tuples", ~tuple_is_noisy), ("noisy-episode tuples", tuple_is_noisy)]:
        if mask.sum() == 0:
            continue
        inc = shaped_increment[mask]
        log(f"  {label} (n={mask.sum()}): shaped increment mean={inc.mean():.4f} std={inc.std():.4f} "
            f"min={inc.min():.4f} max={inc.max():.4f} |inc|>5 frac={float(np.mean(np.abs(inc) > 5)):.4f}")
    results["her_contamination"] = dict(
        n_tuples=int(len(s_idx)), frac_noisy_tuples=frac_noisy_tuples,
        expert_increment=dict(mean=float(shaped_increment[~tuple_is_noisy].mean()),
                               std=float(shaped_increment[~tuple_is_noisy].std())),
        noisy_increment=dict(mean=float(shaped_increment[tuple_is_noisy].mean()),
                              std=float(shaped_increment[tuple_is_noisy].std())) if tuple_is_noisy.sum() else None,
    )

    out_path = OUT_DIR / "b3_mixed_diagnosis.json"
    out_path.write_text(json.dumps(results, indent=2))
    log(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
