"""Ground-truth environment verification of IDM-inferred connecting actions
(docs/graph-proposal/predictor-stitching.tex, follow-up to E2b).

E2b found the two-model (IDM + predictor) cycle-consistency check passes almost every
candidate regardless of correctness, and diagnosed a near-zero inferred action as the
likely cause -- but a near-zero action isn't necessarily wrong if it's genuinely what
connects two already-close states. This settles that empirically, in the REAL environment
(not a latent-space proxy): reset Two-Room at state_i, execute the IDM's inferred mean
action, and check the ACTUAL resulting position against state_j and against two controls
(doing nothing; a random action of matching magnitude), using the env's own success
threshold (terminated = distance < 16.0) as the criterion.

Run as:
    python scripts/env_verify_cycle_edges.py env=tworoom
"""

import json
import sys
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.envs import ENV_MECHANICS
from common.graph_lib import find_transition_edges, raw_topk_cross_episode_pairs
from common.idm_ensemble import ensemble_predict, train_idm_ensemble
from common.log_util import log
from idm_gate import load_landmarks_and_actions

ROOT = Path(__file__).resolve().parent.parent
SUCCESS_DIST = 16.0  # TwoRoomEnv.step's own termination threshold
SPEED = 5.0          # empirically the dataset's fixed speed (mean 4.989, std 0.147, max 5.0
                     # over 200 episodes' unclamped real transitions) -- forced explicitly so
                     # the replay isn't confounded by reset()'s random per-episode speed draw


def _step_from(env, mech, state_i, state_j, action, rng):
    # reset_options is per-environment: Two-Room wants {"state","target_state"}, Push-T
    # {"state","goal_state"}. Going through mech keeps this gate portable instead of
    # hardcoding Two-Room's key (which is why it had never run on Push-T).
    env.reset(options=mech.reset_options(np.asarray(state_i, dtype=np.float32),
                                          np.asarray(state_j, dtype=np.float32)))
    # Two-Room's reset() draws a random per-episode agent speed, which would confound the
    # replay, so it is pinned. Other environments have no such variation -- Push-T's
    # variation_space has no "agent"/"speed" key at all and raised KeyError here, which is
    # why this gate had never run outside Two-Room. Pin it only where it exists.
    try:
        env.variation_space["agent"]["speed"].set_value(np.array([SPEED], dtype=np.float32))
    except (KeyError, AttributeError, TypeError):
        pass
    obs, _, _, _, info = env.step(np.asarray(action, dtype=np.float32))
    # Two-Room exposes the post-step state on info, Push-T on obs. Accept either.
    for src in (info, obs):
        if isinstance(src, dict) and "state" in src:
            return np.asarray(src["state"], dtype=np.float32).reshape(-1)
    raise KeyError("no 'state' key on the env's step() obs or info")


@hydra.main(version_base=None, config_path="../config/graph", config_name="cycle_consistency_gate")
def main(cfg: DictConfig):
    mech = ENV_MECHANICS[cfg.env.name]

    log(f"=== ground-truth env verification of IDM-inferred actions (env={cfg.env.name}) ===")
    z, ep_idx, step_idx, state, action, act_stats = load_landmarks_and_actions(mech, cfg)
    trans_i, trans_j = find_transition_edges(ep_idx, step_idx)

    log(f"=== building wide, unfiltered top-{cfg.k_candidates} candidate pool ===")
    cand_i, cand_j = raw_topk_cross_episode_pairs(z, ep_idx, k=cfg.k_candidates)
    min_real_dist = float(cfg.get("min_real_dist", 0.0))
    if min_real_dist > 0:
        real_d = np.linalg.norm(state[cand_i] - state[cand_j], axis=1)
        far_mask = real_d >= min_real_dist
        log(f"[env-verify] restricting to real_dist>={min_real_dist}: "
            f"{far_mask.sum()} / {len(cand_i)} candidates ({100 * far_mask.mean():.2f}%)")
        cand_i, cand_j = cand_i[far_mask], cand_j[far_mask]
    rng = np.random.default_rng(42)
    n_test = min(500, len(cand_i))
    sample = rng.choice(len(cand_i), size=n_test, replace=False)
    ci, cj = cand_i[sample], cand_j[sample]
    log(f"testing {n_test} candidate pairs in the real environment")

    log("=== training IDM ensemble (same convention as E1/E2/E2b) ===")
    uniq_eps = np.unique(ep_idx)
    shuffled = np.random.default_rng(0).permutation(uniq_eps)
    n_train = int(len(shuffled) * cfg.train_frac)
    train_eps = shuffled[:n_train]
    trans_ep = ep_idx[trans_i]
    train_mask = np.isin(trans_ep, train_eps)
    tr_i, tr_j = trans_i[train_mask], trans_j[train_mask]
    models = train_idm_ensemble(
        z, action, tr_i, tr_j, action_dim=mech.action_dim,
        n_members=cfg.n_members, hidden=cfg.hidden, n_steps=cfg.n_steps,
        batch_size=cfg.batch_size, lr=cfg.lr, seed=cfg.seed,
    )
    preds = ensemble_predict(models, z, ci, cj)
    mean_action_norm = preds.mean(axis=0)
    var_a = preds.var(axis=0).mean(axis=-1)
    # The IDM is now trained on z-scored actions (idm_gate.load_landmarks_and_actions), but
    # env.step() takes RAW actions -- de-normalize before touching the simulator. The
    # magnitude-matched random control below is built from the raw action too, so it stays a
    # like-for-like control.
    _am, _as = act_stats
    mean_action = mean_action_norm * _as + _am

    log("=== stepping the REAL environment for each candidate ===")
    env = mech.make_env()
    dist_before, dist_after_action, dist_after_random, dist_after_zero = [], [], [], []
    for k in range(n_test):
        si, sj = state[ci[k]], state[cj[k]]
        dist_before.append(float(np.linalg.norm(si - sj)))

        after = _step_from(env, mech, si, sj, mean_action[k], rng)
        dist_after_action.append(float(np.linalg.norm(after - sj)))

        # control 1: random direction, same magnitude as the inferred action
        mag = float(np.linalg.norm(mean_action[k]))
        theta = rng.uniform(0, 2 * np.pi)
        rand_action = mag * np.array([np.cos(theta), np.sin(theta)], dtype=np.float32)
        after_r = _step_from(env, mech, si, sj, rand_action, rng)
        dist_after_random.append(float(np.linalg.norm(after_r - sj)))

        # control 2: do nothing
        after_z = _step_from(env, mech, si, sj, np.zeros(mech.action_dim, dtype=np.float32), rng)
        dist_after_zero.append(float(np.linalg.norm(after_z - sj)))

    dist_before = np.array(dist_before)
    dist_after_action = np.array(dist_after_action)
    dist_after_random = np.array(dist_after_random)
    dist_after_zero = np.array(dist_after_zero)

    def summarize(name, arr):
        succ = float(np.mean(arr < SUCCESS_DIST))
        log(f"[env-verify] {name:28s} mean={arr.mean():7.2f}  median={np.median(arr):7.2f}  "
            f"success(<{SUCCESS_DIST})={succ:.3f}")
        return dict(mean=float(arr.mean()), median=float(np.median(arr)), success_rate=succ)

    log(f"--- action-norm stats: mean={np.linalg.norm(mean_action,axis=1).mean():.4f} "
        f"(real training actions average ~1.20) ---")
    r_before = summarize("before (no step)", dist_before)
    r_action = summarize("after inferred action", dist_after_action)
    r_random = summarize("after random-dir control", dist_after_random)
    r_zero = summarize("after zero-action control", dist_after_zero)

    improved_over_before = float(np.mean(dist_after_action < dist_before))
    improved_over_random = float(np.mean(dist_after_action < dist_after_random))
    log(f"[env-verify] inferred action improves on doing nothing: {improved_over_before:.3f} "
        f"of pairs | improves on a random same-magnitude action: {improved_over_random:.3f}")

    out_dir = ROOT / "outputs" / f"idm_gate_{cfg.env.output_prefix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = dict(
        env=cfg.env.name, n_test=n_test,
        mean_action_norm=float(np.linalg.norm(mean_action, axis=1).mean()),
        before=r_before, after_inferred_action=r_action,
        after_random_control=r_random, after_zero_control=r_zero,
        frac_improved_over_before=improved_over_before,
        frac_improved_over_random=improved_over_random,
    )
    out_path = out_dir / "e2c_env_verify_results.json"
    out_path.write_text(json.dumps(summary, indent=2))
    log(f"\nwrote {out_path}")
    log(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
