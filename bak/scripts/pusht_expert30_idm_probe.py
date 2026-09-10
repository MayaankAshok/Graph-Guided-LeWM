"""Exploratory probe: on a small, clean expert_30 Push-T landmark set, build cross-episode
kNN candidates filtered at distance <=5 (the user's stated upper bound on real consecutive-
state distance -- see adjacent_dist quantiles logged below), run the IDM ensemble on those
candidates, and report how its inferred actions score under pred1's single-step prediction
error vs. a random-action control and the real in-distribution reference.

This is a follow-up to docs/graph-proposal/latent-diagnostics-pusht.tex Sec. 5: the 300-episode
landmark set's cross-episode kNN candidates turned out to be mostly latent-space coincidences,
not real transitions, even at a matched distance scale. expert_30 is a much smaller, purely-
expert (no noisy rollouts), less symmetric pool -- worth checking whether the same failure mode
holds here before concluding the mechanism is dead for Push-T entirely.

Run as:
    python scripts/pusht_expert30_idm_probe.py
"""

import sys
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.envs import ENV_MECHANICS
from common.graph_lib import DEV, find_transition_edges
from common.idm_ensemble import (
    classifier_predict_proba, ensemble_predict, train_idm_ensemble, train_transition_classifier,
)
from common.lewm_loader import load_local_jepa_checkpoint
from common.log_util import log
from graph_gate import load_landmarks

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "docs" / "graph-proposal" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
N_EPISODES = 30
DIST_THRESHOLD = 5.0  # user-specified upper bound on real consecutive-state distance


def main():
    mech = ENV_MECHANICS["pusht"]
    out_dir = ROOT / "outputs" / "b0_pusht"
    out_dir.mkdir(parents=True, exist_ok=True)

    log(f"=== expert_{N_EPISODES} IDM probe (env=pusht, dist_threshold={DIST_THRESHOLD}) ===")
    z, ep_idx, step_idx, state = load_landmarks(
        mech, mech.h5_path(ROOT), out_dir, N_EPISODES, mech.ckpt_dir(ROOT), seed=0)
    log(f"landmarks: {z.shape[0]} frames from {len(np.unique(ep_idx))} episodes")

    f = h5py.File(mech.h5_path(ROOT), "r", swmr=True, rdcc_nbytes=256 * 1024 * 1024)
    ep_offset = f["ep_offset"][:]
    global_idx = ep_offset[ep_idx] + step_idx
    action = f["action"][:][global_idx].astype(np.float32)
    f.close()

    trans_i, trans_j = find_transition_edges(ep_idx, step_idx)
    adjacent_dist = np.linalg.norm(z[trans_i] - z[trans_j], axis=1)
    log(f"real adjacent-step distance: n={len(adjacent_dist)} mean={adjacent_dist.mean():.3f} "
        f"median={np.median(adjacent_dist):.3f} q90={np.quantile(adjacent_dist, 0.9):.3f} "
        f"q99={np.quantile(adjacent_dist, 0.99):.3f} max={adjacent_dist.max():.3f}")

    # --- kNN candidates filtered at distance <= DIST_THRESHOLD, brute force -----------------
    # n is small here (a few thousand landmarks for 30 episodes), so a full pairwise distance
    # matrix is cheap and exact -- no need for FAISS/HNSW approximation or the range_search
    # blowup risk seen on the full 300-episode pool.
    n = z.shape[0]
    log(f"computing brute-force pairwise distances over {n} landmarks...")
    d2 = np.sum(z**2, axis=1)[:, None] + np.sum(z**2, axis=1)[None, :] - 2 * z @ z.T
    np.clip(d2, 0, None, out=d2)
    dist = np.sqrt(d2)
    same_ep = ep_idx[:, None] == ep_idx[None, :]
    iu, ju = np.triu_indices(n, k=1)
    keep = (~same_ep[iu, ju]) & (dist[iu, ju] <= DIST_THRESHOLD)
    cand_i, cand_j = iu[keep], ju[keep]
    cand_dist = dist[cand_i, cand_j]
    log(f"[candidates] {len(cand_i)} cross-episode pairs within distance<={DIST_THRESHOLD} "
        f"(out of {len(iu)} possible pairs): mean={cand_dist.mean():.3f} "
        f"median={np.median(cand_dist):.3f}")

    # --- train IDM ensemble on this tier's real transitions ---------------------------------
    uniq_eps = np.unique(ep_idx)
    rng = np.random.default_rng(0)
    shuffled = rng.permutation(uniq_eps)
    n_train = int(len(shuffled) * 0.8)
    train_eps, test_eps = shuffled[:n_train], shuffled[n_train:]
    trans_ep = ep_idx[trans_i]
    tr_i, tr_j = trans_i[np.isin(trans_ep, train_eps)], trans_j[np.isin(trans_ep, train_eps)]
    te_i, te_j = trans_i[np.isin(trans_ep, test_eps)], trans_j[np.isin(trans_ep, test_eps)]
    log(f"IDM train pairs: {len(tr_i)} ({len(train_eps)} episodes), "
        f"test pairs: {len(te_i)} ({len(test_eps)} episodes)")

    idm_models = train_idm_ensemble(z, action, tr_i, tr_j, action_dim=mech.action_dim,
                                     n_members=6, hidden=256, n_steps=3000, batch_size=256,
                                     lr=1e-3, seed=0)

    # --- is the ensemble actually capacity-limited on the task it WAS trained for? -----------
    # Before blaming the splice test's failure on model size, check whether the IDM can even
    # recover the GROUND-TRUTH action on held-out REAL (in-distribution) transitions -- the
    # one setting where we know the right answer and it has seen comparable training data.
    # If this is already near-ceiling, a bigger MLP can't help the splice test: that failure
    # is happening on OUT-OF-DISTRIBUTION candidate pairs (distance ~3.4, vs. training pairs'
    # ~1.1-3.6), which is an extrapolation problem, not a within-distribution fitting problem
    # -- more capacity fits the training distribution more precisely, it does not manufacture
    # supervision for a region with zero training examples.
    pred_te = ensemble_predict(idm_models, z, te_i, te_j).mean(axis=0)
    mse_model = float(((pred_te - action[te_i]) ** 2).mean())
    marginal_action = action[tr_i].mean(axis=0, keepdims=True)
    mse_marginal = float(((marginal_action - action[te_i]) ** 2).mean())
    log(f"[capacity check, hidden=256] held-out REAL action-recovery MSE={mse_model:.5f} "
        f"vs. marginal-baseline MSE={mse_marginal:.5f}  "
        f"(reduction={100*(1-mse_model/mse_marginal):.1f}%)")

    # quick capacity sweep: does a bigger single model do any better in-distribution?
    for hidden in (512, 1024):
        big = train_idm_ensemble(z, action, tr_i, tr_j, action_dim=mech.action_dim,
                                  n_members=1, hidden=hidden, n_steps=3000, batch_size=256,
                                  lr=1e-3, seed=0)
        pred_big = ensemble_predict(big, z, te_i, te_j).mean(axis=0)
        mse_big = float(((pred_big - action[te_i]) ** 2).mean())
        log(f"[capacity check, hidden={hidden}] held-out REAL action-recovery MSE={mse_big:.5f} "
            f"(reduction={100*(1-mse_big/mse_marginal):.1f}%)")

    a_idm = ensemble_predict(idm_models, z, cand_i, cand_j)  # (n_members, n, action_dim)
    idm_var = a_idm.var(axis=0).mean(axis=-1)
    a_idm_mean = a_idm.mean(axis=0).astype(np.float32)
    log(f"[idm] inferred-action ensemble variance over candidates: mean={idm_var.mean():.5f} "
        f"median={np.median(idm_var):.5f}")

    real_var = ensemble_predict(idm_models, z, te_i, te_j).var(axis=0).mean(axis=-1)
    rand_i2 = rng.integers(0, n, 20000)
    rand_j2 = rng.integers(0, n, 20000)
    keep_r = rand_i2 != rand_j2
    rand_var = ensemble_predict(idm_models, z, rand_i2[keep_r], rand_j2[keep_r]).var(axis=0).mean(axis=-1)
    log(f"[idm-var] held-out real={real_var.mean():.5f}  candidates={idm_var.mean():.5f}  "
        f"genuinely-random={rand_var.mean():.5f}")

    # --- score the inferred action with pred1 (genuine 1-step predictor) --------------------
    log("loading pred1 (genuine 1-step predictor)...")
    model = load_local_jepa_checkpoint(
        ckpt_dir=ROOT / "data" / "checkpoints" / "lewm_pusht_1step_predictor",
        weights_file="weights_epoch_20.pt", device=DEV)

    def single_action_error(zi_idx, zj_idx, a1):
        zi_t = torch.from_numpy(z[zi_idx]).float().to(DEV).unsqueeze(1)
        act_t = torch.from_numpy(a1).float().to(DEV).unsqueeze(1)
        with torch.no_grad():
            act_emb = model.action_encoder(act_t)
            pred = model.predict(zi_t, act_emb)[:, -1].cpu().numpy()
        return np.linalg.norm(pred - z[zj_idx], axis=1)

    rand_a1 = action[trans_i[rng.integers(0, len(trans_i), size=len(cand_i))]]
    hybrid_err = single_action_error(cand_i, cand_j, a_idm_mean)
    random_err = single_action_error(cand_i, cand_j, rand_a1)
    hybrid_ratio = hybrid_err / np.maximum(cand_dist, 1e-8)
    random_ratio = random_err / np.maximum(cand_dist, 1e-8)

    # real single-step reference at the SAME (SKIP=1) convention, for comparison
    real_err = single_action_error(trans_i, trans_j, action[trans_i])
    real_ratio = real_err / np.maximum(adjacent_dist, 1e-8)

    log(f"[splice] candidate gap: mean={cand_dist.mean():.3f} median={np.median(cand_dist):.3f}")
    log(f"[splice] IDM error: mean={hybrid_err.mean():.3f} median={np.median(hybrid_err):.3f} "
        f"ratio median={np.median(hybrid_ratio):.3f} frac<1={np.mean(hybrid_ratio < 1):.3f}")
    log(f"[splice] random error: mean={random_err.mean():.3f} median={np.median(random_err):.3f} "
        f"ratio median={np.median(random_ratio):.3f} frac<1={np.mean(random_ratio < 1):.3f}")
    log(f"[splice] real reference: mean={real_err.mean():.3f} median={np.median(real_err):.3f} "
        f"ratio median={np.median(real_ratio):.3f} frac<1={np.mean(real_ratio < 1):.3f}")

    # --- does selecting by IDM ensemble confidence (low variance) find a more-likely-real ---
    # --- subset, rather than treating the whole candidate pool as one population? -----------
    # If low ensemble variance ("all 6 members agree on the connecting action") tracks actual
    # connectivity, the splice test should look progressively more like the real reference
    # (higher frac<1, IDM beating random by a wider margin) as we restrict to more confident
    # candidates. If variance carries no such signal, these numbers should stay flat.
    order = np.argsort(idm_var)  # ascending: most confident (lowest variance) first
    log("=== selecting candidates by IDM ensemble confidence (ascending variance) ===")
    for frac in (1.0, 0.5, 0.25, 0.1, 0.05):
        m = max(10, int(len(order) * frac))
        sel = order[:m]
        h_err, r_err = hybrid_err[sel], random_err[sel]
        h_rat, r_rat = hybrid_ratio[sel], random_ratio[sel]
        log(f"[confidence top {int(frac*100):>3}%] n={m} var<= {idm_var[sel].max():.5f}  |  "
            f"IDM: mean={h_err.mean():.3f} median={np.median(h_err):.3f} "
            f"frac<1={np.mean(h_rat < 1):.3f}  |  random: mean={r_err.mean():.3f} "
            f"median={np.median(r_err):.3f} frac<1={np.mean(r_rat < 1):.3f}  |  "
            f"candidate gap mean={cand_dist[sel].mean():.3f}")

    # --- scatter: IDM ensemble variance vs. splice prediction error -------------------------
    # Does low variance (ensemble agreement) predict low prediction error (a correct action)?
    # Plotted for both the IDM's inferred action and the random-action control on the SAME
    # candidates/x-axis, so any real variance->error relationship should show up as the IDM
    # series sloping down-left while random stays flat -- not just both scattering identically.
    corr_idm = np.corrcoef(idm_var, hybrid_err)[0, 1]
    corr_rand = np.corrcoef(idm_var, random_err)[0, 1]
    log(f"[scatter] pearson corr(idm_var, IDM error)={corr_idm:.3f}  "
        f"corr(idm_var, random error)={corr_rand:.3f}")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(idm_var, hybrid_err, s=10, alpha=0.4, label=f"IDM action (r={corr_idm:.2f})")
    ax.scatter(idm_var, random_err, s=10, alpha=0.4, label=f"random $a_1$ (r={corr_rand:.2f})")
    ax.axhline(np.median(real_err), color="black", linestyle="--", linewidth=1,
               label=f"real single-step median error ({np.median(real_err):.2f})")
    ax.set_xscale("log")
    ax.set_xlabel("IDM ensemble variance (log scale)")
    ax.set_ylabel("prediction error, ||pred(z_i, a) - z_j||")
    ax.set_title(f"IDM ensemble variance vs. splice prediction error (expert_{N_EPISODES}, pusht)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"pusht_expert{N_EPISODES}_idmvar_vs_error.png", dpi=150)
    plt.close(fig)

    # === improving the ensemble: a discriminative "is this real?" classifier ================
    # The action-regression ensemble always outputs SOME action and its variance turned out
    # not to track correctness at all (scatter above, r~=0). A binary classifier asks the
    # more direct question this task actually needs answered. Train it with DISTANCE-MATCHED
    # negatives -- for each real positive transition, find a cross-episode pair at nearly the
    # same latent distance -- so it can't just learn "small distance = real"; it has to find
    # an actual distinguishing feature (if one exists).
    log("=== improving the ensemble: distance-matched discriminative classifier ===")
    train_pt_mask = np.isin(ep_idx, train_eps)
    cross_mask_all = (~same_ep[iu, ju]) & train_pt_mask[iu] & train_pt_mask[ju]
    neg_pool_i, neg_pool_j = iu[cross_mask_all], ju[cross_mask_all]
    neg_pool_dist = dist[neg_pool_i, neg_pool_j]
    sort_order = np.argsort(neg_pool_dist)
    sorted_d = neg_pool_dist[sort_order]

    pos_dist = adjacent_dist[np.isin(trans_ep, train_eps)]  # matches tr_i/tr_j order
    idx_r = np.clip(np.searchsorted(sorted_d, pos_dist), 0, len(sorted_d) - 1)
    idx_l = np.clip(idx_r - 1, 0, len(sorted_d) - 1)
    use_l = np.abs(sorted_d[idx_l] - pos_dist) < np.abs(sorted_d[idx_r] - pos_dist)
    matched = np.where(use_l, idx_l, idx_r)
    neg_sel = sort_order[matched]
    neg_i_train, neg_j_train = neg_pool_i[neg_sel], neg_pool_j[neg_sel]
    match_err = np.abs(dist[neg_i_train, neg_j_train] - pos_dist)
    log(f"[classifier] {len(tr_i)} positives, {len(neg_i_train)} distance-matched negatives "
        f"(mean distance-match error={match_err.mean():.4f}, max={match_err.max():.4f})")

    clf_models = train_transition_classifier(
        z, tr_i, tr_j, neg_i_train, neg_j_train, n_members=5, hidden=256,
        n_steps=3000, batch_size=256, lr=1e-3, seed=0)

    # sanity check: held-out real transitions vs. genuinely random pairs (NOT distance-matched
    # -- an easy population) should separate cleanly if the classifier learned anything at all
    p_real_heldout = classifier_predict_proba(clf_models, z, te_i, te_j)
    p_random = classifier_predict_proba(clf_models, z, rand_i2[keep_r], rand_j2[keep_r])
    log(f"[classifier] sanity: P(real) on held-out real transitions: mean={p_real_heldout.mean():.3f} "
        f"median={np.median(p_real_heldout):.3f}  |  on genuinely random pairs: "
        f"mean={p_random.mean():.3f} median={np.median(p_random):.3f}")

    # the actual question: does classifier probability on the candidate pool correlate with
    # whether the IDM's (or a random) action actually gets closer to the target?
    p_cand = classifier_predict_proba(clf_models, z, cand_i, cand_j)
    corr_clf_idm = np.corrcoef(p_cand, hybrid_err)[0, 1]
    corr_clf_rand = np.corrcoef(p_cand, random_err)[0, 1]
    log(f"[classifier] P(real) on candidate pool: mean={p_cand.mean():.3f} "
        f"median={np.median(p_cand):.3f}")
    log(f"[classifier] pearson corr(P(real), IDM error)={corr_clf_idm:.3f}  "
        f"corr(P(real), random error)={corr_clf_rand:.3f}")

    clf_order = np.argsort(-p_cand)  # descending: most-likely-real first
    log("=== selecting candidates by classifier P(real) (descending) ===")
    for frac in (1.0, 0.5, 0.25, 0.1, 0.05):
        m = max(10, int(len(clf_order) * frac))
        sel = clf_order[:m]
        h_rat, r_rat = hybrid_ratio[sel], random_ratio[sel]
        log(f"[P(real) top {int(frac*100):>3}%] n={m} P(real)>= {p_cand[sel].min():.3f}  |  "
            f"IDM: mean={hybrid_err[sel].mean():.3f} frac<1={np.mean(h_rat < 1):.3f}  |  "
            f"random: mean={random_err[sel].mean():.3f} frac<1={np.mean(r_rat < 1):.3f}  |  "
            f"candidate gap mean={cand_dist[sel].mean():.3f}")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(p_cand, hybrid_err, s=10, alpha=0.4, label=f"IDM action (r={corr_clf_idm:.2f})")
    ax.scatter(p_cand, random_err, s=10, alpha=0.4, label=f"random $a_1$ (r={corr_clf_rand:.2f})")
    ax.axhline(np.median(real_err), color="black", linestyle="--", linewidth=1,
               label=f"real single-step median error ({np.median(real_err):.2f})")
    ax.set_xlabel("classifier P(real transition)")
    ax.set_ylabel("prediction error, ||pred(z_i, a) - z_j||")
    ax.set_title(f"Distance-matched classifier P(real) vs. splice error (expert_{N_EPISODES}, pusht)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"pusht_expert{N_EPISODES}_classifier_vs_error.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
