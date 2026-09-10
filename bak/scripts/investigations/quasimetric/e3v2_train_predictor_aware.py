"""E3-v2 -- retrain the quasimetric head with pairs from ACTUAL predictor rollouts, to test
E4's leading hypothesis: E3's head was trained only on real encoded frame pairs, but E4's CEM
solver scores it on the model's own multi-step PREDICTED terminal embeddings, which are a
different, out-of-distribution population the head never saw during training.

What "actual predictor rollouts" means here, concretely: `jepa.py`'s LeWM.rollout applies the
predictor autoregressively, one call per CEM horizon-step, each consuming ONE 10-dim action
block (5 real actions concatenated -- action_encoder's input_dim=10 IS this block, per
CLAUDE.md's action-block correction, not a padded single action). With
config/eval/pusht.yaml's plan_config (horizon=5, action_block=5, history_len=1 default), a
full CEM cost evaluation applies exactly 5 chained predictor calls (25 real steps) to a single
starting embedding. `predictor_rollout_from_z0` below reproduces that exact loop (verified
against jepa.py / stable_worldmodel's wm/lewm/lewm.py, a package-vendored twin) but starts
from an already-encoded z0 (the probe_gate cache's own `z`) instead of re-encoding pixels --
cheaper, and the two are the same embedding by construction.

Training pairs are built from REAL recorded action sequences (via
common.graph_lib.find_block_transition_edges, skip=5 -- the validated block convention,
NOT the superseded pad_action padding), chained 5 blocks deep, so each pair has a
well-defined ground-truth target: "the embedding actually recorded 25 real steps later along
the SAME real trajectory". This is the natural, unambiguous reading of "actual" rollouts
(genuine recorded actions, not synthetic CEM-noise actions) and keeps the label well-posed.

Only ONE loss term changes from E3: a new `L_pred_anchor` regresses
d_Q(z_hat_5block_predicted, z_target_real) toward log(1+25) -- a fixed target, since every
pair spans exactly the horizon=5 block CEM actually plans over, matching what E4 evaluates.
This is a scale/domain-anchoring term (teach the head to score PREDICTED embeddings on the
same scale as real ones), not a new rank-discrimination signal on its own -- it works
alongside E3's unchanged rank/local/asymmetry terms on real frames, not instead of them.
Local and asymmetry terms are UNCHANGED (still real-frame-only) -- a scoped limitation, not
claimed to be fixed here.

Run:
    python scripts/investigations/quasimetric/e3v2_train_predictor_aware.py env=pusht n_episodes=18685
"""

import json
import os
import sys
import time
from pathlib import Path

import h5py
import hydra
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from scipy import stats as sps

ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.envs import ENV_MECHANICS
from common.graph_lib import DEV, find_block_transition_edges
from common.lewm_loader import load_lewm
from common.log_util import log
from common.quasimetric import LatentQuasimetric
from probe_gate import Arm, _chunked, make_pairs
from e1_directed_asymmetry import find_active_pushing_pairs
from e3_train_quasimetric import episode_split, gap1_pairs

N_STEPS = int(os.environ.get("E3_N_STEPS", 20000))
BATCH_SIZE = int(os.environ.get("E3_BATCH_SIZE", 512))
LOCAL_BATCH = 128
ASYM_BATCH = 128
PRED_BATCH = int(os.environ.get("E3_PRED_BATCH", 96))   # predictor-rollout pairs per step --
# smaller than the others since each one costs 5 chained transformer forward passes
LR = 1e-3
W_RANK = 1.0
W_LOCAL = 2.0
W_ASYM = 0.5
W_PRED = 1.0             # weight on the new predictor-rollout anchor term
ASYM_MARGIN = 6.0
PRED_HORIZON = 5          # matches plan_config.horizon in config/eval/pusht.yaml
PRED_HISTORY_SIZE = 3     # matches jepa.py LeWM.rollout's default history_size (unmodified
                          # by config/eval/pusht.yaml or policy.py -- E4 uses this exact value)
PRED_GAP_STEPS = 25       # = PRED_HORIZON * action_block(5) -- the real-step gap every
                          # predictor-rollout pair spans, matching goal_offset_steps


def predictor_rollout_from_z0(model, z0, action_blocks, history_size=PRED_HISTORY_SIZE):
    """Reproduces jepa.py/stable_worldmodel LeWM.rollout's autoregressive inner loop exactly
    for the H=1 (history_len=1) case, starting from an already-encoded z0 instead of pixels.
    z0: (B, D). action_blocks: (B, n_blocks, 10) -- REAL 5-action blocks, sequence order.
    Returns the predicted embedding after applying ALL n_blocks blocks: (B, D)."""
    B, n_blocks, _ = action_blocks.shape
    emb = z0.unsqueeze(1)              # (B, 1, D) -- H=1 history convention
    act = action_blocks[:, :1]         # (B, 1, 10) -- act_0
    for t in range(n_blocks - 1):
        act_emb = model.action_encoder(act)
        pred = model.predict(emb[:, -history_size:], act_emb[:, -history_size:])[:, -1:]
        emb = torch.cat([emb, pred], dim=1)
        act = torch.cat([act, action_blocks[:, t + 1:t + 2]], dim=1)
    act_emb = model.action_encoder(act)
    pred = model.predict(emb[:, -history_size:], act_emb[:, -history_size:])[:, -1:]
    return pred[:, 0]   # (B, D)


def build_block_chain_lookup(ep_idx, step_idx, action, skip=5):
    """Dense per-row lookup arrays from find_block_transition_edges' edge list: next_row[r]
    (the row `skip` real steps later in the same episode, or -1) and next_action_block[r]
    (the real skip*action_dim block taken to get there). chain_ok[r] additionally marks rows
    with a full PRED_HORIZON-deep valid chain (not near an episode boundary)."""
    n = len(ep_idx)
    block_i, action_block, block_j = find_block_transition_edges(ep_idx, step_idx, action, skip=skip)
    next_row = np.full(n, -1, dtype=np.int64)
    next_action_block = np.zeros((n, action_block.shape[1]), dtype=np.float32)
    next_row[block_i] = block_j
    next_action_block[block_i] = action_block

    chain_ok = np.ones(n, dtype=bool)
    cur = np.arange(n)
    for _ in range(PRED_HORIZON):
        valid = cur >= 0
        chain_ok &= valid
        cur = np.where(valid, next_row[np.where(valid, cur, 0)], -1)
    return next_row, next_action_block, chain_ok


def sample_predictor_chains(rows_pool, next_row, next_action_block, n, rng):
    """Sample n starting rows (from rows_pool, all chain_ok) and walk PRED_HORIZON blocks
    deep. Returns start_rows (n,), action_blocks (n, PRED_HORIZON, 10), target_rows (n,)."""
    start = rng.choice(rows_pool, size=n, replace=True)
    action_blocks = np.zeros((n, PRED_HORIZON, next_action_block.shape[1]), dtype=np.float32)
    cur = start.copy()
    for t in range(PRED_HORIZON):
        action_blocks[:, t] = next_action_block[cur]
        cur = next_row[cur]
    return start, action_blocks, cur   # cur is now the target row after PRED_HORIZON blocks


@hydra.main(version_base=None, config_path=str(ROOT / "config" / "graph"),
            config_name="probe_gate")
def main(cfg: DictConfig):
    t0 = time.time()
    mech = ENV_MECHANICS[cfg.env.name]
    res_dir = ROOT / "outputs" / "quasimetric"
    res_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(os.environ.get("PROBE_CACHE_DIR",
                                    str(ROOT / "outputs" / f"probe_gate_{cfg.env.output_prefix}" / "cache")))

    log(f"=== E3-v2 train predictor-aware quasimetric head: {cfg.env.name} (device={DEV}) ===")

    tag = f"ep{cfg.n_episodes}_seed{cfg.seed}_g{cfg.pixel_grid}"
    need = ["z", "ep_idx", "step_idx", "state"]
    cache = {k: np.load(cache_dir / f"{k}_{tag}.npy", mmap_mode="r") for k in need}
    ep_idx, step_idx = np.asarray(cache["ep_idx"]), np.asarray(cache["step_idx"])
    state = np.asarray(cache["state"])
    n = len(ep_idx)
    log(f"cache: {n} frames, {len(np.unique(ep_idx))} episodes, z dim {cache['z'].shape[1]}")

    train_eps, test_eps, rows_tr, rows_te = episode_split(ep_idx, cfg.train_frac, cfg.seed)
    log(f"split: {len(train_eps)} train / {len(test_eps)} test episodes "
        f"({len(rows_tr)} / {len(rows_te)} frames)")

    arm = Arm("lewm", cache["z"], cache["z"].shape[1])
    arm.fit_norm(rows_tr, cfg.seed)

    # -- real-frame pairs, UNCHANGED from E3
    tr = make_pairs(ep_idx, step_idx, train_eps, cfg.pairs_per_episode, cfg.seed, cfg.max_gap)
    te = make_pairs(ep_idx, step_idx, test_eps, cfg.pairs_per_episode, cfg.seed + 1, cfg.max_gap)
    tr_s, tr_g, tr_gap = tr
    te_s, te_g, te_gap = te
    log(f"rank pairs (real frames): {len(tr_gap)} train / {len(te_gap)} test")
    gtr = torch.from_numpy(np.log1p(tr_gap)).float().to(DEV)

    loc_s, loc_g = gap1_pairs(ep_idx, step_idx, train_eps, n_per_episode=20, seed=cfg.seed + 2)
    log(f"local (gap=1) pairs: {len(loc_s)}")

    A_all, B_all = find_active_pushing_pairs(state, ep_idx, step_idx)
    is_train_pair = np.isin(ep_idx[A_all], list(train_eps))
    asym_tr_A, asym_tr_B = A_all[is_train_pair], B_all[is_train_pair]
    asym_te_A, asym_te_B = A_all[~is_train_pair], B_all[~is_train_pair]
    log(f"active-pushing pairs: {len(asym_tr_A)} train / {len(asym_te_A)} test")

    # -- NEW: real 5-action-block chains for predictor rollouts (skip=5, matches action_block)
    log("loading action column and building block-chain lookup...")
    with h5py.File(str(mech.h5_path(ROOT)), "r", swmr=True) as f5:
        action = f5["action"][:len(ep_idx)]   # cache rows == h5 rows (identity, verified:
        # n_episodes covers the full dataset so encode_dataset's per-episode concatenation
        # in sorted-episode order reproduces the raw h5 row order exactly)
    next_row, next_action_block, chain_ok = build_block_chain_lookup(ep_idx, step_idx, action, skip=5)
    tr_pool = np.nonzero(chain_ok & np.isin(ep_idx, list(train_eps)))[0]
    te_pool = np.nonzero(chain_ok & np.isin(ep_idx, list(test_eps)))[0]
    log(f"predictor-chain-eligible rows: {len(tr_pool)} train / {len(te_pool)} test "
        f"(of {n}, {(chain_ok.mean()):.1%} overall)")

    model = load_lewm(ckpt_dir=mech.ckpt_dir(ROOT), device=DEV)
    model.interpolate_pos_encoding = True

    quasi_model = LatentQuasimetric(latent_dim=arm.dim, hidden_dim=256, proj_dim=64).to(DEV)
    opt = torch.optim.AdamW(quasi_model.parameters(), lr=LR, weight_decay=1e-4)
    rng = np.random.default_rng(cfg.seed)
    log_target_pred = torch.tensor(np.log1p(PRED_GAP_STEPS), dtype=torch.float32, device=DEV)

    log(f"training {N_STEPS} steps, batch={BATCH_SIZE} rank / {LOCAL_BATCH} local / "
        f"{ASYM_BATCH} asym / {PRED_BATCH} predictor-rollout (horizon={PRED_HORIZON})")
    for step in range(N_STEPS):
        rb = rng.integers(0, len(tr_gap), BATCH_SIZE)
        d_fwd = quasi_model(arm.get(tr_s[rb]), arm.get(tr_g[rb]))
        loss_rank = F.smooth_l1_loss(d_fwd, gtr[rb])

        lb = rng.integers(0, len(loc_s), LOCAL_BATCH)
        d_local = quasi_model(arm.get(loc_s[lb]), arm.get(loc_g[lb]))
        loss_local = F.relu(d_local - 1.0).pow(2).mean()

        ab = rng.integers(0, len(asym_tr_A), ASYM_BATCH)
        d_af = quasi_model(arm.get(asym_tr_A[ab]), arm.get(asym_tr_B[ab]))
        d_ab = quasi_model(arm.get(asym_tr_B[ab]), arm.get(asym_tr_A[ab]))
        loss_asym = F.relu(d_af.detach() + ASYM_MARGIN - d_ab).mean()

        # -- predictor-rollout anchor term
        p_start, p_blocks, p_target = sample_predictor_chains(tr_pool, next_row, next_action_block,
                                                               PRED_BATCH, rng)
        with torch.no_grad():
            z0 = torch.from_numpy(np.asarray(cache["z"][p_start])).float().to(DEV)
            blocks_t = torch.from_numpy(p_blocks).float().to(DEV)
            z_hat = predictor_rollout_from_z0(model, z0, blocks_t)   # (B, D) raw scale
            z_hat_n = torch.clamp((z_hat - torch.from_numpy(arm.mu).to(DEV)) /
                                  torch.from_numpy(arm.sd).to(DEV), -10.0, 10.0)
        z_tgt_n = arm.get(p_target)
        d_pred = quasi_model(z_hat_n, z_tgt_n)
        loss_pred = F.smooth_l1_loss(d_pred, log_target_pred.expand_as(d_pred))

        loss = (W_RANK * loss_rank + W_LOCAL * loss_local + W_ASYM * loss_asym
               + W_PRED * loss_pred)
        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % max(1, N_STEPS // 10) == 0:
            log(f"  step {step}/{N_STEPS} loss={loss.item():.4f} "
                f"(rank={loss_rank.item():.4f} local={loss_local.item():.4f} "
                f"asym={loss_asym.item():.4f} pred={loss_pred.item():.4f}) "
                f"({time.time()-t0:.0f}s)")

    quasi_model.eval()
    model.eval()

    # --- evaluation: same E3 gate, on real pairs (apples-to-apples with the original E3) ---
    with torch.no_grad():
        def d_q_fwd(s_rows, g_rows):
            return _chunked(lambda ix: quasi_model(arm.get(s_rows[ix]), arm.get(g_rows[ix]))
                            .cpu().numpy(), len(s_rows))
        dq_te = d_q_fwd(te_s, te_g)
        k = min(len(tr_gap), 100000)
        dq_tr = d_q_fwd(tr_s[:k], tr_g[:k])

    z_mm = cache["z"]
    l2_te = np.linalg.norm(np.asarray(z_mm[te_s]) - np.asarray(z_mm[te_g]), axis=1)
    rho_q = float(sps.spearmanr(dq_te, te_gap).statistic)
    rho_q_tr = float(sps.spearmanr(dq_tr, tr_gap[:k]).statistic)
    rho_l2 = float(sps.spearmanr(l2_te, te_gap).statistic)
    log(f"\n=== E3 gate (real pairs, unchanged eval): Spearman(d_Q)={rho_q:.4f} "
        f"(train {rho_q_tr:.4f})  vs raw L2={rho_l2:.4f}  "
        f"-> {'PASSED' if rho_q > rho_l2 else 'FAILED'}")

    # --- NEW: held-out predictor-rollout quality -- does d_Q correctly place PREDICTED
    # terminal embeddings near their real 25-step-later target, vs. random/near/far targets?
    with torch.no_grad():
        n_eval_chains = min(20000, len(te_pool))
        ps, pb, pt = sample_predictor_chains(te_pool, next_row, next_action_block,
                                             n_eval_chains, rng)
        z0 = torch.from_numpy(np.asarray(cache["z"][ps])).float().to(DEV)
        blocks_t = torch.from_numpy(pb).float().to(DEV)

        def rollout_chunked(z0_np, blocks_np, chunk=2048):
            outs = []
            for a in range(0, len(z0_np), chunk):
                zc = torch.from_numpy(z0_np[a:a+chunk]).float().to(DEV)
                bc = torch.from_numpy(blocks_np[a:a+chunk]).float().to(DEV)
                outs.append(predictor_rollout_from_z0(model, zc, bc).cpu().numpy())
            return np.concatenate(outs)

        z_hat_all = rollout_chunked(np.asarray(cache["z"][ps]), pb)
        z_hat_n = np.clip((z_hat_all - arm.mu) / arm.sd, -10.0, 10.0)
        z_tgt_n = np.clip((np.asarray(z_mm[pt]) - arm.mu) / arm.sd, -10.0, 10.0)

        d_pred_true = _chunked(lambda ix: quasi_model(
            torch.from_numpy(z_hat_n[ix]).float().to(DEV),
            torch.from_numpy(z_tgt_n[ix]).float().to(DEV)).cpu().numpy(), len(ps))
        # mismatched (shuffled) targets, as a negative control -- d_Q should score these
        # farther than the true 25-step-later target on average
        shuf = rng.permutation(len(pt))
        z_tgt_shuf_n = z_tgt_n[shuf]
        d_pred_shuf = _chunked(lambda ix: quasi_model(
            torch.from_numpy(z_hat_n[ix]).float().to(DEV),
            torch.from_numpy(z_tgt_shuf_n[ix]).float().to(DEV)).cpu().numpy(), len(ps))

    log(f"held-out predictor-rollout quality (n={n_eval_chains}, gap=25 fixed): "
        f"d_Q(pred, true target)={d_pred_true.mean():.2f}+-{d_pred_true.std():.2f}  "
        f"d_Q(pred, random target)={d_pred_shuf.mean():.2f}+-{d_pred_shuf.std():.2f}  "
        f"(target log(1+25)={np.log1p(25):.2f})")

    res = dict(
        env=cfg.env.name, n_frames=int(n), n_steps=N_STEPS,
        gate=dict(spearman_quasimetric=rho_q, spearman_quasimetric_train=rho_q_tr,
                  spearman_raw_l2=rho_l2, passed=bool(rho_q > rho_l2)),
        predictor_rollout_quality=dict(
            n=int(n_eval_chains), gap_steps=PRED_GAP_STEPS,
            d_pred_true_mean=float(d_pred_true.mean()), d_pred_true_std=float(d_pred_true.std()),
            d_pred_shuf_mean=float(d_pred_shuf.mean()), d_pred_shuf_std=float(d_pred_shuf.std()),
            target_log1p_gap=float(np.log1p(PRED_GAP_STEPS)),
        ),
        secs=time.time() - t0,
    )
    p = res_dir / f"e3v2_train_predictor_aware_{cfg.env.name}.json"
    p.write_text(json.dumps(res, indent=2))
    torch.save(quasi_model.state_dict(), res_dir / f"e3v2_quasimetric_head_{cfg.env.name}.pt")
    log(f"wrote {p} ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
