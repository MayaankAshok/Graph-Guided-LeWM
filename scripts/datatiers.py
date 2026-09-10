"""Unified B3 data-tier sweep -- config-driven consolidation of tworoom_b3_datatiers.py/
pusht_b3_datatiers.py. Builds expert_N/mixed/mixed_large tier setups and runs the
baseline-vs-auxphi comparison via common.training.run_condition_resumable (unchanged --
already environment-agnostic before this consolidation).

Run as:
    python scripts/datatiers.py env=tworoom              # full sweep, all tiers/variants/seeds
    python scripts/datatiers.py env=pusht                 # full sweep for Push-T

For single-combo invocation (used by worker.py/the Ada orchestrator), see worker.py.
"""

import json
import os
import sys
import time
from pathlib import Path

import h5py
import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.envs import ENV_MECHANICS
from common.graph_lib import DEV, build_graph_edges_for_env, build_weighted_graph, full_phi_dist_matrix
from common.log_util import log
from common.training import peak_stats, precompute_eval_set, run_condition_resumable
from graph_gate import load_landmarks
from rollout_collector import collect_rollouts, encode_pixels

ROOT = Path(__file__).resolve().parent.parent
NOISY_EP_ID_OFFSET = 1_000_000  # keeps freshly-collected episode ids disjoint from real ones


def _out_dirs(cfg):
    out_dir = ROOT / "outputs" / f"b3_{cfg.env.output_prefix}"
    ckpt_dir = out_dir / "checkpoints"
    tier_cache_dir = Path(
        os.environ.get(f"{cfg.env.name.upper()}_TIER_CACHE_DIR", str(out_dir / "tier_cache"))
    )
    rollout_cache_dir = Path(
        os.environ.get(f"{cfg.env.name.upper()}_ROLLOUT_CACHE_DIR", str(out_dir / "rollout_cache"))
    )
    for d in (out_dir, ckpt_dir, tier_cache_dir, rollout_cache_dir):
        d.mkdir(parents=True, exist_ok=True)
    return out_dir, ckpt_dir, tier_cache_dir, rollout_cache_dir


def _load_real_tier_arrays(cfg, mech, n_episodes):
    out_dir, _, _, _ = _out_dirs(cfg)
    z, ep_idx, step_idx, state = load_landmarks(
        mech, mech.h5_path(ROOT), out_dir, n_episodes, mech.ckpt_dir(ROOT),
    )
    f = h5py.File(mech.h5_path(ROOT), "r", swmr=True, rdcc_nbytes=256 * 1024 * 1024)
    ep_offset = f["ep_offset"][:]
    global_idx = ep_offset[ep_idx] + step_idx
    action = f["action"][:][global_idx].astype(np.float32)
    f.close()
    return z, ep_idx, step_idx, state, action


def _build_mixed_arrays(cfg, mech, large=False):
    _, _, _, rollout_cache_dir = _out_dirs(cfg)
    n_expert = cfg.env.mixed_large_n_expert if large else cfg.env.mixed_n_expert
    n_noisy = cfg.env.mixed_large_n_noisy if large else cfg.env.mixed_n_noisy
    tag = "mixed_large" if large else "mixed"

    ez, e_ep, e_step, e_state, e_action = _load_real_tier_arrays(cfg, mech, n_expert)

    policy_kwargs = OmegaConf.to_container(cfg.env.rollout_policy_kwargs)
    cache_name = f"{tag}_noisy_n{n_noisy}_" + "_".join(f"{k}{v}" for k, v in sorted(policy_kwargs.items()))
    rollout = collect_rollouts(
        cfg.env.name, n_episodes=n_noisy, seed=7, cache_dir=rollout_cache_dir,
        cache_name=cache_name, **policy_kwargs,
    )
    log(f"[{tag}] encoding {len(rollout['pixels'])} freshly-collected frames with the frozen LeWM encoder...")
    nz = encode_pixels(rollout["pixels"], mech.ckpt_dir(ROOT))
    n_ep_idx = rollout["ep_idx"] + NOISY_EP_ID_OFFSET

    z = np.concatenate([ez, nz], axis=0)
    ep_idx = np.concatenate([e_ep, n_ep_idx], axis=0)
    step_idx = np.concatenate([e_step, rollout["step_idx"]], axis=0)
    state = np.concatenate([e_state, rollout["state"]], axis=0)
    action = np.concatenate([e_action, rollout["action"]], axis=0)
    log(f"[{tag}] combined: {len(ez)} real-expert frames ({n_expert} eps) + "
        f"{len(nz)} noisy-rollout frames ({n_noisy} eps) = {len(z)} total")
    return z, ep_idx, step_idx, state, action


def _build_setup(cfg, mech, tier_key, z, ep_idx, step_idx, state, action):
    _, _, tier_cache_dir, _ = _out_dirs(cfg)
    cache_path = tier_cache_dir / f"{tier_key}_phi_dist.npy"
    n, d = z.shape

    uniq_eps = np.unique(ep_idx)
    rng = np.random.default_rng(0)
    shuffled = rng.permutation(uniq_eps)
    n_train = int(len(shuffled) * cfg.train_frac)
    train_eps, test_eps = shuffled[:n_train], shuffled[n_train:]
    train_rows = np.nonzero(np.isin(ep_idx, train_eps))[0]
    test_rows = np.nonzero(np.isin(ep_idx, test_eps))[0]
    log(f"[{tier_key}] {n} landmarks, {len(uniq_eps)} episodes "
        f"({len(train_eps)} train / {len(test_eps)} test episodes, "
        f"{len(train_rows)} / {len(test_rows)} rows)")

    phi_mode = cfg.get("phi_mode", "dense")
    if phi_mode == "none":
        # variant=stitch only: no weighted graph, no Dijkstra, no phi_dist at all -- only
        # the raw identification edges are needed (common.training.build_stitch_her_tuples
        # turns them directly into cross-episode HER tuples, with no distance value
        # involved). Takes precedence over need_graph, same as phi_mode=sparse below.
        trans_i, trans_j, id_i, id_j, rho_hat, eps2 = build_graph_edges_for_env(
            cfg.env, z, ep_idx, step_idx, eps2_override=cfg.get("id_eps2_override", None),
        )
        log(f"[{tier_key}] graph: {len(trans_i)} transition edges, {len(id_i)} identification edges "
            f"(rho_hat={rho_hat:.4f} eps2={eps2:.2f}); phi_mode=none -- no graph object, no phi_dist")
        true_dist_oracle = mech.build_true_distance_oracle(state)
        train_eval = precompute_eval_set(train_rows, state, true_dist_oracle, cfg.n_eval_pairs, seed=100)
        test_eval = precompute_eval_set(test_rows, state, true_dist_oracle, cfg.n_eval_pairs, seed=101)
        log(f"[{tier_key}] eval pairs: {len(train_eval[0])} train, {len(test_eval[0])} test")
        return dict(z=z, ep_idx=ep_idx, step_idx=step_idx, proprio=state, action=action,
                    train_eps=train_eps, test_eps=test_eps, phi_dist=None, graph=None,
                    id_i=id_i, id_j=id_j,
                    train_eval=train_eval, test_eval=test_eval, d=d)

    if phi_mode == "sparse":
        # Keep the graph, skip the dense NxN matrix: actor_train.py computes phi at just the
        # HER pairs it needs via common.graph_lib.phi_for_pairs. Required above ~40k
        # landmarks, where the dense matrix stops fitting (expert_1000: 62 GB).
        trans_i, trans_j, id_i, id_j, rho_hat, eps2 = build_graph_edges_for_env(
            cfg.env, z, ep_idx, step_idx, eps2_override=cfg.get("id_eps2_override", None),
        )
        log(f"[{tier_key}] graph: {len(trans_i)} transition edges, {len(id_i)} identification edges "
            f"(rho_hat={rho_hat:.4f} eps2={eps2:.2f}); "
            f"phi_mode=sparse -- no dense phi_dist")
        graph = build_weighted_graph(n, trans_i, trans_j, id_i, id_j, id_weight=1.0)
        true_dist_oracle = mech.build_true_distance_oracle(state)
        train_eval = precompute_eval_set(train_rows, state, true_dist_oracle, cfg.n_eval_pairs, seed=100)
        test_eval = precompute_eval_set(test_rows, state, true_dist_oracle, cfg.n_eval_pairs, seed=101)
        log(f"[{tier_key}] eval pairs: {len(train_eval[0])} train, {len(test_eval[0])} test")
        return dict(z=z, ep_idx=ep_idx, step_idx=step_idx, proprio=state, action=action,
                    train_eps=train_eps, test_eps=test_eps, phi_dist=None, graph=graph,
                    id_i=id_i, id_j=id_j,
                    train_eval=train_eval, test_eval=test_eval, d=d)

    if not cfg.get("need_graph", True):
        # Baseline-only use (e.g. a large expert_N tier for reproducing the LeWM paper's
        # GCIQL number): the dense NxN phi_dist is the one O(n^2)-memory piece of the
        # pipeline, and only the auxphi variant reads it. actor_train.py refuses to run
        # variant=auxphi on a setup built this way.
        log(f"[{tier_key}] need_graph=false: skipping graph construction and phi_dist")
        phi_dist = None
        id_i = id_j = None
    else:
        trans_i, trans_j, id_i, id_j, rho_hat, eps2 = build_graph_edges_for_env(
            cfg.env, z, ep_idx, step_idx, eps2_override=cfg.get("id_eps2_override", None),
        )
        log(f"[{tier_key}] graph: {len(trans_i)} transition edges, {len(id_i)} identification edges "
            f"(rho_hat={rho_hat:.4f} eps2={eps2:.2f})")
        graph = build_weighted_graph(n, trans_i, trans_j, id_i, id_j, id_weight=1.0)

        if cache_path.exists():
            log(f"[{tier_key}] loading cached phi_dist")
            phi_dist = np.load(cache_path)
        else:
            log(f"[{tier_key}] precomputing full pairwise graph-distance matrix...")
            t0 = time.time()
            phi_dist = full_phi_dist_matrix(graph, n)
            log(f"[{tier_key}] done in {time.time()-t0:.1f}s")
            np.save(cache_path, phi_dist)

    true_dist_oracle = mech.build_true_distance_oracle(state)
    train_eval = precompute_eval_set(train_rows, state, true_dist_oracle, cfg.n_eval_pairs, seed=100)
    test_eval = precompute_eval_set(test_rows, state, true_dist_oracle, cfg.n_eval_pairs, seed=101)
    log(f"[{tier_key}] eval pairs: {len(train_eval[0])} train, {len(test_eval[0])} test")

    return dict(z=z, ep_idx=ep_idx, step_idx=step_idx, proprio=state, action=action,
                train_eps=train_eps, test_eps=test_eps, phi_dist=phi_dist,
                id_i=id_i, id_j=id_j,
                train_eval=train_eval, test_eval=test_eval, d=d)


def tiers_for(cfg):
    return [f"expert_{n}" for n in cfg.dataset_sizes] + ["mixed", "mixed_large"]


def build_tier_setups(cfg, tiers=None):
    mech = ENV_MECHANICS[cfg.env.name]
    tiers = tiers if tiers is not None else tiers_for(cfg)
    setups = {}
    for tier in tiers:
        if tier.startswith("expert_"):
            n = int(tier.split("_")[1])
            arrays = _load_real_tier_arrays(cfg, mech, n)
            setups[tier] = _build_setup(cfg, mech, tier, *arrays)
        elif tier == "mixed":
            arrays = _build_mixed_arrays(cfg, mech, large=False)
            setups[tier] = _build_setup(cfg, mech, "mixed", *arrays)
        elif tier == "mixed_large":
            arrays = _build_mixed_arrays(cfg, mech, large=True)
            setups[tier] = _build_setup(cfg, mech, "mixed_large", *arrays)
        else:
            raise ValueError(f"unknown tier '{tier}'")
    return setups


def ckpt_path(cfg, tier, name, seed):
    _, ckpt_dir, _, _ = _out_dirs(cfg)
    return ckpt_dir / f"{tier}__{name}__s{seed}.pt"


@hydra.main(version_base=None, config_path="../config/graph", config_name="datatiers")
def main(cfg: DictConfig):
    log(f"device={DEV} env={cfg.env.name}")
    out_dir, ckpt_dir, _, _ = _out_dirs(cfg)
    setups = build_tier_setups(cfg)

    all_results = {}
    for tier in tiers_for(cfg):
        setup = setups[tier]
        all_results[tier] = {}
        for aux_lambda, name in [(0.0, "baseline"), (cfg.aux_lambda, "auxphi")]:
            all_results[tier][name] = []
            for seed in cfg.seeds:
                history = run_condition_resumable(
                    tier, name, aux_lambda, setup, seed, ckpt_dir=ckpt_dir, distance_source="graph",
                )
                all_results[tier][name].append(dict(seed=seed, history=history))

        out_path = out_dir / f"b3_datatiers_{tier}_results.json"
        out_path.write_text(json.dumps(all_results[tier], indent=2))
        log(f"wrote {out_path}")

    log("\n" + "=" * 70 + "\nSUMMARY: peak held-out-test Spearman by tier\n" + "=" * 70)
    for tier in tiers_for(cfg):
        log(f"\n--- {tier} ---")
        for name in ["baseline", "auxphi"]:
            peaks = []
            for r in all_results[tier][name]:
                p = peak_stats(r["history"])
                peaks.append(p["spearman_test"])
                log(f"  {name} seed={r['seed']}: peak={p['spearman_test']:.4f} @ step {p['step']}")
            log(f"  {name} mean peak = {np.mean(peaks):.4f} +- {np.std(peaks):.4f}")

    full_out = out_dir / "b3_datatiers_all_results.json"
    full_out.write_text(json.dumps(all_results, indent=2))
    log(f"\nwrote {full_out}")


if __name__ == "__main__":
    main()
