"""E3 -- train the quasimetric head, gated on beating raw latent L2.

Three changes from the first attempt (scripts/common/quasimetric.py's
compute_quasimetric_loss, Section 2.1 of docs/quasimetric-jepa-proposal.md), each directly
motivated by an E0-E2 finding:

  - L_rand (the first attempt's fixed D_max=50 margin forcing every cross-episode pair apart)
    is dropped entirely. Flagged in the proposal as actively fighting cross-episode stitching,
    since many cross-episode pairs in expert data are near-duplicate states, not maximally
    distant ones -- this whole line of work exists to fix cross-episode stitching, not hurt it.

  - The reverse/asymmetry penalty is no longer a flat +10 constant applied to every pair.
    It is restricted to active-pushing pairs (identical construction to E1: block displaced
    >20px over a 15-step window -- see e1_directed_asymmetry.find_active_pushing_pairs) and
    its margin is grounded in E1's own measurement on exactly this pair population (forward
    graph distance ~14.8, backward ~21.4 among the rare pairs where a route back exists at
    all, and NO route back for 99.5% of them) rather than invented.

  - A local upper-bound constraint is added: real one-step transitions (literal adjacent env
    steps, gap=1) must have d_Q(s, s') <= 1. This anchors the head's scale to "1 unit ~= 1 real
    env step" the way a pure correlation loss cannot, and is the one piece of the QRL-style
    (Wang & Isola 2022) "maximize distance subject to local step cost <= 1" formulation kept
    here as a hard constraint rather than the full dual-ascent treatment -- a simplification
    stated plainly, not a full QRL reproduction.

The correlation-producing term itself is UNCHANGED from probe_gate.py's probe C (metric-form
regression to log1p(true step gap)), because that recipe is not hypothetical here: run on this
exact cache, it already scores held-out Spearman 0.9404 on Push-T (see
outputs/probe_gate_pusht/e1_probe_results_ep18685.json), clearing the E3 gate (raw L2's own
Spearman on the same held-out pairs, measured fresh below) by a wide margin on its own. E3's
job is to show that ADDING the asymmetric potential (phi) and the E1-grounded asymmetry term
on top does not break that, while producing a real, grounded, non-flat asymmetry signal.

Gate (docs/quasimetric-jepa-proposal.md Section 3, E3): held-out same-episode
Spearman(true step gap, d_Q forward) must beat raw latent L2's Spearman on the SAME held-out
pairs. Beating the graph-geodesic metric is explicitly not required.

Run:
    python scripts/investigations/quasimetric/e3_train_quasimetric.py env=pusht
"""

import json
import os
import sys
import time
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from scipy import stats as sps

ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.graph_lib import DEV
from common.log_util import log
from common.quasimetric import LatentQuasimetric
from probe_gate import Arm, _chunked, make_pairs
from e1_directed_asymmetry import find_active_pushing_pairs

N_STEPS = int(os.environ.get("E3_N_STEPS", 20000))
BATCH_SIZE = int(os.environ.get("E3_BATCH_SIZE", 512))
LOCAL_BATCH = 128       # guaranteed real one-step (gap=1) pairs per training step
ASYM_BATCH = 128        # active-pushing pairs per training step, for the asymmetry term
LR = 1e-3
W_RANK = 1.0            # weight on the (proven) log1p(gap) correlation term
W_LOCAL = 2.0           # weight on the d_Q(s, s') <= 1 constraint for real one-step transitions
W_ASYM = 0.5            # weight on the E1-grounded reverse-asymmetry margin
ASYM_MARGIN = 6.0       # ~ E1's measured forward/backward gap (21.36 - 14.82 = 6.5) among
                        # active-pushing pairs, in the same step-count units as d_Q's other terms


def load_cache(cache_dir, n_episodes, seed=0, grid=16):
    tag = f"ep{n_episodes}_seed{seed}_g{grid}"
    need = ["z", "ep_idx", "step_idx", "state", "pooled"]
    paths = {k: cache_dir / f"{k}_{tag}.npy" for k in need}
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"probe_gate cache missing: {missing}\nRun scripts/probe_gate.py env=pusht "
            f"n_episodes={n_episodes} first (it writes this cache).")
    return {k: np.load(p, mmap_mode="r") for k, p in paths.items()}


def episode_split(ep_idx, frac_train=0.8, seed=0):
    uniq = np.unique(ep_idx)
    shuffled = np.random.default_rng(seed).permutation(uniq)
    n_tr = int(len(shuffled) * frac_train)
    train_eps, test_eps = set(map(int, shuffled[:n_tr])), set(map(int, shuffled[n_tr:]))
    is_train = np.isin(ep_idx, shuffled[:n_tr])
    return train_eps, test_eps, np.nonzero(is_train)[0], np.nonzero(~is_train)[0]


def gap1_pairs(ep_idx, step_idx, episodes, n_per_episode, seed):
    """Guaranteed literal one-env-step transitions (gap == 1), for the local constraint --
    a dedicated draw rather than relying on make_pairs' random gap to hit 1 often enough."""
    rng = np.random.default_rng(seed)
    order = np.lexsort((step_idx, ep_idx))
    ep_o = ep_idx[order]
    bounds = np.nonzero(np.diff(ep_o))[0] + 1
    starts = np.concatenate(([0], bounds))
    ends = np.concatenate((bounds, [len(order)]))
    s_rows, g_rows = [], []
    for a, b in zip(starts, ends):
        rows = order[a:b]
        if int(ep_idx[rows[0]]) not in episodes or len(rows) < 2:
            continue
        t = rng.integers(0, len(rows) - 1, n_per_episode)
        s_rows.append(rows[t])
        g_rows.append(rows[t + 1])
    s = np.concatenate(s_rows)
    return s, np.concatenate(g_rows)


@hydra.main(version_base=None, config_path=str(ROOT / "config" / "graph"),
            config_name="probe_gate")
def main(cfg: DictConfig):
    t0 = time.time()
    res_dir = ROOT / "outputs" / "quasimetric"
    res_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(os.environ.get("PROBE_CACHE_DIR",
                                    str(ROOT / "outputs" / f"probe_gate_{cfg.env.output_prefix}" / "cache")))

    log(f"=== E3 train quasimetric head: {cfg.env.name} (device={DEV}) ===")
    cache = load_cache(cache_dir, cfg.n_episodes, cfg.seed)
    ep_idx, step_idx = np.asarray(cache["ep_idx"]), np.asarray(cache["step_idx"])
    state = np.asarray(cache["state"])
    n = len(ep_idx)
    log(f"cache: {n} frames, {len(np.unique(ep_idx))} episodes, z dim {cache['z'].shape[1]}")

    train_eps, test_eps, rows_tr, rows_te = episode_split(ep_idx, cfg.train_frac, cfg.seed)
    log(f"split: {len(train_eps)} train / {len(test_eps)} test episodes "
        f"({len(rows_tr)} / {len(rows_te)} frames)")

    arm = Arm("lewm", cache["z"], cache["z"].shape[1])
    arm.fit_norm(rows_tr, cfg.seed)

    # correlation pairs (probe-C-style, log1p(gap) regression target)
    tr = make_pairs(ep_idx, step_idx, train_eps, cfg.pairs_per_episode, cfg.seed, cfg.max_gap)
    te = make_pairs(ep_idx, step_idx, test_eps, cfg.pairs_per_episode, cfg.seed + 1, cfg.max_gap)
    tr_s, tr_g, tr_gap = tr
    te_s, te_g, te_gap = te
    log(f"rank pairs: {len(tr_gap)} train / {len(te_gap)} test "
        f"(gap median={np.median(te_gap):.0f} max={te_gap.max():.0f})")
    gtr = torch.from_numpy(np.log1p(tr_gap)).float().to(DEV)

    # local one-step-transition pairs, train split only
    loc_s, loc_g = gap1_pairs(ep_idx, step_idx, train_eps, n_per_episode=20, seed=cfg.seed + 2)
    log(f"local (gap=1) pairs: {len(loc_s)}")

    # E1-style active-pushing pairs, split by episode membership (train for the loss, held-out
    # test for reporting the learned asymmetry against E1's ground-truth-grounded numbers).
    # find_active_pushing_pairs' default block_cols=(2,3) indexes raw state columns
    # directly (block x,y), matching PushTMechanics._project_state's own column layout.
    A_all, B_all = find_active_pushing_pairs(state, ep_idx, step_idx)
    is_train_pair = np.isin(ep_idx[A_all], list(train_eps))
    asym_tr_A, asym_tr_B = A_all[is_train_pair], B_all[is_train_pair]
    asym_te_A, asym_te_B = A_all[~is_train_pair], B_all[~is_train_pair]
    log(f"active-pushing pairs: {len(asym_tr_A)} train / {len(asym_te_A)} test")

    model = LatentQuasimetric(latent_dim=arm.dim, hidden_dim=256, proj_dim=64).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    rng = np.random.default_rng(cfg.seed)

    log(f"training {N_STEPS} steps, batch={BATCH_SIZE} rank / {LOCAL_BATCH} local / {ASYM_BATCH} asym")
    for step in range(N_STEPS):
        rb = rng.integers(0, len(tr_gap), BATCH_SIZE)
        z_s, z_g = arm.get(tr_s[rb]), arm.get(tr_g[rb])
        d_fwd = model(z_s, z_g)
        loss_rank = F.smooth_l1_loss(d_fwd, gtr[rb])

        lb = rng.integers(0, len(loc_s), LOCAL_BATCH)
        zl_s, zl_g = arm.get(loc_s[lb]), arm.get(loc_g[lb])
        d_local = model(zl_s, zl_g)
        loss_local = F.relu(d_local - 1.0).pow(2).mean()

        ab = rng.integers(0, len(asym_tr_A), ASYM_BATCH)
        za, zb = arm.get(asym_tr_A[ab]), arm.get(asym_tr_B[ab])
        d_af = model(za, zb)   # forward: A -> B (the real push direction)
        d_ab = model(zb, za)   # reverse: B -> A
        loss_asym = F.relu(d_af.detach() + ASYM_MARGIN - d_ab).mean()

        loss = W_RANK * loss_rank + W_LOCAL * loss_local + W_ASYM * loss_asym
        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % max(1, N_STEPS // 10) == 0:
            log(f"  step {step}/{N_STEPS} loss={loss.item():.4f} "
                f"(rank={loss_rank.item():.4f} local={loss_local.item():.4f} "
                f"asym={loss_asym.item():.4f}) ({time.time()-t0:.0f}s)")

    model.eval()

    # --- evaluation: rank gate (d_Q forward vs. raw L2, same held-out pairs) ---
    with torch.no_grad():
        def d_q_fwd(s_rows, g_rows):
            return _chunked(lambda ix: model(arm.get(s_rows[ix]), arm.get(g_rows[ix]))
                            .cpu().numpy(), len(s_rows))
        dq_te = d_q_fwd(te_s, te_g)
        k = min(len(tr_gap), 100000)
        dq_tr = d_q_fwd(tr_s[:k], tr_g[:k])

    z_mm = cache["z"]  # stays memmapped -- index directly, no full-array materialization
    l2_te = np.linalg.norm(np.asarray(z_mm[te_s]) - np.asarray(z_mm[te_g]), axis=1)

    rho_q = float(sps.spearmanr(dq_te, te_gap).statistic)
    rho_q_tr = float(sps.spearmanr(dq_tr, tr_gap[:k]).statistic)
    rho_l2 = float(sps.spearmanr(l2_te, te_gap).statistic)
    gate_passed = rho_q > rho_l2
    log(f"\n=== GATE: held-out Spearman(true gap, d_Q) = {rho_q:.4f}  "
        f"(train {rho_q_tr:.4f})  vs. raw L2 = {rho_l2:.4f}  "
        f"-> {'PASSED' if gate_passed else 'FAILED'}")

    # --- evaluation: learned asymmetry on held-out active-pushing pairs ---
    with torch.no_grad():
        d_fwd_te = _chunked(lambda ix: model(arm.get(asym_te_A[ix]), arm.get(asym_te_B[ix]))
                            .cpu().numpy(), len(asym_te_A))
        d_bwd_te = _chunked(lambda ix: model(arm.get(asym_te_B[ix]), arm.get(asym_te_A[ix]))
                            .cpu().numpy(), len(asym_te_A))
    l2_fwd_te = np.linalg.norm(np.asarray(z_mm[asym_te_A]) - np.asarray(z_mm[asym_te_B]), axis=1)
    l2_bwd_te = np.linalg.norm(np.asarray(z_mm[asym_te_B]) - np.asarray(z_mm[asym_te_A]), axis=1)
    gap_learned = d_bwd_te - d_fwd_te
    log(f"held-out active-pushing asymmetry: d_Q(fwd)={d_fwd_te.mean():.2f}+-{d_fwd_te.std():.2f}  "
        f"d_Q(bwd)={d_bwd_te.mean():.2f}+-{d_bwd_te.std():.2f}  "
        f"gap={gap_learned.mean():.2f}+-{gap_learned.std():.2f}  "
        f"(fraction with bwd>fwd: {(gap_learned > 0).mean():.1%})")
    log(f"raw L2 asymmetry on the same pairs (baseline, by construction): "
        f"{np.abs(l2_fwd_te - l2_bwd_te).max():.8f}")

    res = dict(
        env=cfg.env.name, n_frames=int(n), n_steps=N_STEPS,
        n_train_episodes=len(train_eps), n_test_episodes=len(test_eps),
        n_rank_pairs_train=int(len(tr_gap)), n_rank_pairs_test=int(len(te_gap)),
        n_local_pairs=int(len(loc_s)),
        n_asym_pairs_train=int(len(asym_tr_A)), n_asym_pairs_test=int(len(asym_te_A)),
        gate=dict(spearman_quasimetric=rho_q, spearman_quasimetric_train=rho_q_tr,
                  spearman_raw_l2=rho_l2, passed=gate_passed),
        learned_asymmetry=dict(
            d_fwd_mean=float(d_fwd_te.mean()), d_fwd_std=float(d_fwd_te.std()),
            d_bwd_mean=float(d_bwd_te.mean()), d_bwd_std=float(d_bwd_te.std()),
            gap_mean=float(gap_learned.mean()), gap_std=float(gap_learned.std()),
            frac_bwd_gt_fwd=float((gap_learned > 0).mean()),
        ),
        raw_l2_asymmetry_max=float(np.abs(l2_fwd_te - l2_bwd_te).max()),
        secs=time.time() - t0,
    )
    p = res_dir / f"e3_train_quasimetric_{cfg.env.name}.json"
    p.write_text(json.dumps(res, indent=2))
    torch.save(model.state_dict(), res_dir / f"e3_quasimetric_head_{cfg.env.name}.pt")
    log(f"wrote {p} ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
