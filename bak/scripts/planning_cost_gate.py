"""E0 -- does LeWM's planning cost rank action candidates the way the real environment does?

LeWM plans with CEM. For each candidate action sequence it rolls the predictor forward in
latent space and scores the candidate by the squared latent distance between the predicted
TERMINAL embedding and the goal embedding (jepa.py:130 `criterion`). That single scalar is
the entire decision rule. Nothing in LeWM's training makes latent L2 correspond to "how much
work is left", and this repo has already measured that it does not: Spearman against a
privileged-state oracle is 0.22 on Push-T and 0.44 on Two-Room, versus 0.54 / 0.95 for the
graph geodesic (docs/graph-proposal/main.tex, sec:b5-results / sec:b0-results).

Those numbers were measured on random state pairs, which is NOT the job the cost actually
does. This script measures it in situ: given the candidates a planner really has to choose
between, how well does each cost rank them against what the simulator actually does?

Design
------
For each planning problem (a dataset row `t`, goal = the row `goal_offset` steps later in the
same episode -- the paper's App. F.1 protocol):

  1. Build K candidate action sequences spanning expert-quality to random.
  2. LATENT: roll the predictor forward over each candidate and score it with each cost.
  3. GROUND TRUTH: reset the simulator to the dataset start state, execute the candidate's
     raw actions, and record the true final state and the env's own success flag.
  4. Compare the ranking each cost induces against the ranking the simulator induces.

Three-way decomposition (the reason this is worth running rather than arguing about)
------------------------------------------------------------------------------------
  l2_pred  -- ||pred_terminal_emb - z_goal||^2. LeWM's deployed cost. Conflates two error
              sources: the predictor being wrong about where the candidate lands, and the
              cost being wrong about what "close" means.
  l2_exec  -- ||encode(true_final_frame) - z_goal||^2. The SAME cost function applied to the
              state the candidate genuinely reached. The predictor is removed from the loop
              entirely, so this isolates the cost function's own quality.
  true_dist -- privileged-state oracle distance. Ground truth.

So: if l2_exec ranks well but l2_pred does not, the predictor is the bottleneck and the cost
is fine. If l2_exec ALSO ranks poorly, the cost function itself is the defect, which is the
claim under test. If both rank well, there is no defect to fix and this line of work stops.

Horizon prediction
------------------
Latent L2 is sharply discriminative near zero and nearly flat far away (on Push-T one env
step moves the latent ~176x less than the spread between unrelated frames; on Two-Room only
~2.4x). A planner ranking candidates that ALL still sit far from the goal is working in the
flat region. So the defect should be mild when the goal is inside the planning horizon and
some candidate nearly reaches it, and should grow as the goal moves further away. Sweeping
--goal-offsets 25 50 100 against a fixed 25-env-step planning horizon tests exactly that.

Usage
-----
    python scripts/planning_cost_gate.py --env pusht --n-problems 60 --n-candidates 24 \
        --goal-offsets 25 50 100 --out outputs/planning_cost_gate

Needs PUSHT_H5_PATH (or the env's own h5 override) and the pretrained checkpoint; see
CLAUDE.md for the Ada staging paths.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from scipy import stats as sps

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.envs import ENV_MECHANICS
from common.lewm_loader import load_lewm
from common.log_util import log

DEV = "cuda" if torch.cuda.is_available() else "cpu"
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

SKIP = 5      # frame skip: one predictor step spans 5 env steps, and one action block is
              # the 5 real actions concatenated (2 dims x 5 = 10 = action_encoder input_dim).
              # See [[lewm-predictor-action-block-convention]] -- single-action zero-padding
              # is the superseded, wrong convention.
HISTORY = 3   # history blocks the predictor conditions on (config/train/lewm.yaml history_size)
HORIZON = 5   # planning blocks = 5 * SKIP = 25 env steps (config/eval/pusht.yaml plan_config)


# ------------------------------------------------------------------
# encoder / predictor
# ------------------------------------------------------------------

def make_encode_frame(model, batch=64):
    mean, std = IMAGENET_MEAN.to(DEV), IMAGENET_STD.to(DEV)

    def encode_frame(pixels_nhwc):
        """(n,H,W,3) uint8 -> (n,d) float32 numpy. Same normalization and CLS-token read as
        actor_rollout_utils.make_encoder, so embeddings are comparable to every other number
        this project has produced."""
        out = np.empty((len(pixels_nhwc), 192), dtype=np.float32)
        for lo in range(0, len(pixels_nhwc), batch):
            hi = min(lo + batch, len(pixels_nhwc))
            t = torch.from_numpy(np.ascontiguousarray(pixels_nhwc[lo:hi])).to(DEV)
            t = t.permute(0, 3, 1, 2).float() / 255.0
            t = (t - mean) / std
            with torch.no_grad():
                enc = model.encode({"pixels": t.unsqueeze(1)})
            out[lo:hi] = enc["emb"][:, 0].float().cpu().numpy()
        return out

    return encode_frame


def latent_rollout(model, z_hist, act_hist, act_future):
    """Autoregressive latent rollout, mirroring jepa.JEPA.rollout exactly.

    z_hist:     (B, HISTORY, D)          encoded real history frames, spaced SKIP apart
    act_hist:   (B, HISTORY, SKIP*A)     the action block taken AT each history frame -- the
                                         last one is the candidate's first block, the earlier
                                         ones are the real past actions
    act_future: (B, HORIZON-1, SKIP*A)   the candidate's remaining blocks

    Returns (B, HORIZON, D): the predicted embedding after each of the HORIZON blocks, i.e.
    z_{t+SKIP}, ..., z_{t+HORIZON*SKIP}.
    """
    emb = torch.as_tensor(z_hist, dtype=torch.float32, device=DEV)
    act = torch.as_tensor(act_hist, dtype=torch.float32, device=DEV)
    fut = torch.as_tensor(act_future, dtype=torch.float32, device=DEV)

    preds = []
    with torch.no_grad():
        for t in range(fut.shape[1] + 1):
            act_emb = model.action_encoder(act[:, -HISTORY:])
            pred = model.predict(emb[:, -HISTORY:], act_emb)[:, -1:]  # (B,1,D)
            preds.append(pred)
            if t < fut.shape[1]:
                emb = torch.cat([emb, pred], dim=1)
                act = torch.cat([act, fut[:, t:t + 1]], dim=1)
    return torch.cat(preds, dim=1).float().cpu().numpy()


# ------------------------------------------------------------------
# problem sampling
# ------------------------------------------------------------------

def sample_problems(h5_path, mech, goal_offset, n, seed):
    """Start rows with enough real history for the predictor and enough room for the goal.
    Same start distribution as the paper's protocol (any valid dataset row), restricted to
    step_idx >= (HISTORY-1)*SKIP so the history window exists."""
    min_step = (HISTORY - 1) * SKIP
    with h5py.File(h5_path, "r", swmr=True, rdcc_nbytes=256 * 1024 * 1024) as f:
        ep_offset = f["ep_offset"][:].astype(np.int64)
        ep_len = f["ep_len"][:].astype(np.int64)
        # The model consumes z-SCORED actions (train.py's get_column_normalizer, eval.py's
        # StandardScaler). The ENV consumes raw ones. Keep both: raw for stepping the
        # simulator, normalized for anything fed to the predictor.
        _a = f["action"][:].astype(np.float64)
        _a = _a[~np.isnan(_a).any(axis=1)]
        act_mean, act_std = _a.mean(0).astype(np.float32), _a.std(0).astype(np.float32)
        del _a

        # a start is valid if min_step <= step and step + goal_offset <= ep_len-1, and the
        # candidate's HORIZON*SKIP raw actions all exist inside the episode
        need_after = max(goal_offset, HORIZON * SKIP)
        counts = np.maximum(ep_len - need_after - min_step, 0)
        total = int(counts.sum())
        if total == 0:
            raise SystemExit(f"no valid starts at goal_offset={goal_offset}")
        rng = np.random.default_rng(seed)
        pick = np.sort(rng.choice(total, size=min(n, total), replace=False))
        cum = np.cumsum(counts)
        ep_pos = np.searchsorted(cum, pick, side="right")
        prev = np.where(ep_pos > 0, cum[np.maximum(ep_pos - 1, 0)], 0)
        step = pick - prev + min_step
        start_row = ep_offset[ep_pos] + step
        goal_row = start_row + goal_offset

        # every row we need to read, gathered once and read in sorted order (h5py fancy
        # indexing requires increasing indices)
        hist_rows = np.stack([start_row - k * SKIP for k in range(HISTORY - 1, -1, -1)], axis=1)
        want = np.unique(np.concatenate([hist_rows.ravel(), goal_row]))
        pix = f["pixels"][want]
        row_to_i = {int(r): i for i, r in enumerate(want)}

        start_state = mech.read_state_rows(f, start_row)
        goal_state = mech.read_state_rows(f, goal_row)
        # real past action blocks (the HISTORY-1 blocks before t) and the real future
        # actions from t onward, used as the expert candidate
        past_rows = np.stack([np.arange(r - (HISTORY - 1) * SKIP, r) for r in start_row])
        fut_rows = np.stack([np.arange(r, r + HORIZON * SKIP) for r in start_row])
        act_all = f["action"]
        past_act = np.stack([act_all[r[0]:r[-1] + 1] for r in past_rows]).astype(np.float32)
        real_act = np.stack([act_all[r[0]:r[-1] + 1] for r in fut_rows]).astype(np.float32)

    hist_pix = np.stack([[pix[row_to_i[int(r)]] for r in row] for row in hist_rows])
    goal_pix = np.stack([pix[row_to_i[int(r)]] for r in goal_row])
    return dict(ep=ep_pos, start_step=step, start_row=start_row, goal_row=goal_row,
                start_state=start_state, goal_state=goal_state,
                hist_pixels=hist_pix, goal_pixels=goal_pix,
                past_action=past_act, real_action=real_act,
                act_mean=act_mean, act_std=act_std)


def make_candidates(real_action, action_low, action_high, k, rng):
    """K candidate action sequences of HORIZON*SKIP raw actions, spanning expert quality to
    random. Candidate 0 is always the real dataset continuation.

    A planner's cost is only ever asked to rank the candidates the optimizer proposes, so
    the spread matters. CEM's early iterations look close to random and its late iterations
    cluster near a good solution; this mixture covers both regimes. If anything it is
    GENEROUS to the cost, since a spread this wide is easier to rank than a converged CEM
    population."""
    T, A = real_action.shape
    scale = (action_high - action_low)
    cands = [real_action]
    sigmas = [0.02, 0.05, 0.10, 0.25]
    per = max(1, (k - 1) // (len(sigmas) + 1))
    for s in sigmas:
        for _ in range(per):
            noise = rng.normal(0.0, s, size=(T, A)) * scale
            cands.append(np.clip(real_action + noise, action_low, action_high))
    while len(cands) < k:
        cands.append(rng.uniform(action_low, action_high, size=(T, A)))
    return np.stack(cands[:k]).astype(np.float32)


# ------------------------------------------------------------------
# main
# ------------------------------------------------------------------

def run(args):
    mech = ENV_MECHANICS[args.env]
    h5_path = mech.h5_path(Path(args.root))
    ckpt_dir = mech.ckpt_dir(Path(args.root))
    log(f"[setup] env={args.env} h5={h5_path}")
    log(f"[setup] ckpt={ckpt_dir} device={DEV}")

    model = load_lewm(ckpt_dir=Path(ckpt_dir), device=DEV)
    encode_frame = make_encode_frame(model)
    env = mech.make_env()
    action_low = np.asarray(env.action_space.low, dtype=np.float32)
    action_high = np.asarray(env.action_space.high, dtype=np.float32)
    log(f"[setup] action_space low={action_low} high={action_high}")

    results = {}
    for goal_offset in args.goal_offsets:
        log(f"\n===== goal_offset={goal_offset} =====")
        probs = sample_problems(h5_path, mech, goal_offset, args.n_problems, args.seed)
        n_prob = len(probs["start_row"])
        oracle = mech.build_true_distance_oracle(probs["goal_state"])
        rng = np.random.default_rng(args.seed + 977 * goal_offset)

        rows = []
        t0 = time.time()
        for i in range(n_prob):
            cands = make_candidates(probs["real_action"][i], action_low, action_high,
                                     args.n_candidates, rng)
            K = len(cands)

            # ---- latent side -------------------------------------------------
            z_hist = encode_frame(probs["hist_pixels"][i])                 # (HISTORY, D)
            z_goal = encode_frame(probs["goal_pixels"][i][None])[0]        # (D,)
            am, asd = probs["act_mean"], probs["act_std"]
            past_n = (probs["past_action"][i] - am) / asd
            cands_n = (cands - am) / asd          # predictor input: z-scored
            past = past_n.reshape(HISTORY - 1, SKIP * mech.action_dim)
            blocks = cands_n.reshape(K, HORIZON, SKIP * mech.action_dim)
            act_hist = np.concatenate(
                [np.repeat(past[None], K, axis=0), blocks[:, :1]], axis=1)  # (K,HISTORY,10)
            pred = latent_rollout(model, np.repeat(z_hist[None], K, axis=0),
                                   act_hist, blocks[:, 1:])                 # (K,HORIZON,D)

            d_pred = np.linalg.norm(pred - z_goal[None, None], axis=-1)     # (K,HORIZON)
            l2_pred = (d_pred[:, -1] ** 2)
            l2_pred_min = (d_pred.min(axis=1) ** 2)
            l2_pred_mean = (d_pred ** 2).mean(axis=1)
            gz = z_goal / (np.linalg.norm(z_goal) + 1e-8)
            pn = pred[:, -1] / (np.linalg.norm(pred[:, -1], axis=1, keepdims=True) + 1e-8)
            cos_pred = 1.0 - pn @ gz

            # ---- ground truth: execute each candidate in the simulator --------
            final_states, final_pixels, successes = [], [], []
            for k in range(K):
                _, info = env.reset(
                    seed=int((args.seed * 100_003 + i * 97 + k) % (2 ** 31 - 1)),
                    options=mech.reset_options(probs["start_state"][i], probs["goal_state"][i]),
                )
                success = False
                obs = None
                for a in cands[k]:
                    obs, reward, terminated, truncated, inf = env.step(a)
                    succ, _ = mech.step_result(obs, reward, terminated, truncated, inf)
                    if succ:
                        success = True
                        break
                final_states.append(np.asarray(obs["state"], dtype=np.float32).reshape(-1))
                final_pixels.append(env.render())
                successes.append(success)
            final_states = np.stack(final_states)
            final_pixels = np.stack(final_pixels)

            true_dist = oracle(final_states, probs["goal_state"][i][None])[:, 0]   # (K,)
            z_exec = encode_frame(final_pixels)                                    # (K,D)
            l2_exec = ((z_exec - z_goal[None]) ** 2).sum(axis=1)

            rows.append(dict(
                problem=i, l2_pred=l2_pred, l2_pred_min=l2_pred_min,
                l2_pred_mean=l2_pred_mean, cos_pred=cos_pred, l2_exec=l2_exec,
                true_dist=true_dist, success=np.asarray(successes),
                pred_drift=np.linalg.norm(pred[:, -1] - z_exec, axis=1),
            ))
            if i % 10 == 0:
                log(f"  problem {i}/{n_prob}  elapsed={time.time()-t0:.0f}s")

        results[str(goal_offset)] = summarize(rows, goal_offset)
        np.savez_compressed(
            Path(args.out) / f"raw_offset{goal_offset}.npz",
            **{k: np.stack([r[k] for r in rows]) for k in
               ("l2_pred", "l2_pred_min", "l2_pred_mean", "cos_pred", "l2_exec",
                "true_dist", "success", "pred_drift")},
            start_row=probs["start_row"], goal_row=probs["goal_row"],
        )

    env.close()
    out_json = Path(args.out) / "planning_cost_gate.json"
    out_json.write_text(json.dumps(results, indent=2))
    log(f"\n[done] wrote {out_json}")
    return results


COSTS = ("l2_pred", "l2_pred_min", "l2_pred_mean", "cos_pred", "l2_exec")


def summarize(rows, goal_offset):
    """Per-problem Spearman of each cost against the simulator's own outcome, plus the
    decision-relevant number: how much worse is the candidate the cost PICKS than the best
    candidate available, in true-distance units, normalized by the spread of that problem's
    candidates (0 = picked the best, 1 = picked the worst)."""
    out = {"goal_offset": goal_offset, "n_problems": len(rows)}
    for cost in COSTS:
        rhos, regrets = [], []
        for r in rows:
            td = r["true_dist"]
            if np.ptp(td) < 1e-9:
                continue
            rho = sps.spearmanr(r[cost], td).statistic
            if np.isfinite(rho):
                rhos.append(rho)
            picked = td[int(np.argmin(r[cost]))]
            regrets.append((picked - td.min()) / (td.max() - td.min()))
        rhos, regrets = np.asarray(rhos), np.asarray(regrets)
        out[cost] = dict(
            spearman_mean=float(rhos.mean()), spearman_median=float(np.median(rhos)),
            spearman_ci95=_boot_ci(rhos), frac_positive=float((rhos > 0).mean()),
            norm_regret_mean=float(regrets.mean()), norm_regret_median=float(np.median(regrets)),
        )
    out["oracle_success_rate"] = float(np.mean([r["success"].mean() for r in rows]))
    out["pred_drift_median"] = float(np.median(np.concatenate([r["pred_drift"] for r in rows])))
    out["true_dist_spread_median"] = float(np.median([np.ptp(r["true_dist"]) for r in rows]))
    return out


def _boot_ci(x, n=5000, seed=0):
    if len(x) < 2:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--env", default="pusht", choices=sorted(ENV_MECHANICS))
    p.add_argument("--n-problems", type=int, default=60)
    p.add_argument("--n-candidates", type=int, default=24)
    p.add_argument("--goal-offsets", type=int, nargs="+", default=[25, 50, 100])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--root", default=str(Path(__file__).resolve().parent.parent))
    p.add_argument("--out", default=None)
    args = p.parse_args()
    args.out = args.out or str(Path(args.root) / "outputs" / f"planning_cost_gate_{args.env}")
    Path(args.out).mkdir(parents=True, exist_ok=True)

    res = run(args)

    log("\n" + "=" * 78)
    log("SUMMARY -- Spearman(cost, true outcome) over candidates, mean over problems")
    log("=" * 78)
    log(f"{'offset':>7} {'cost':>14} {'spearman':>10} {'ci95':>18} {'regret':>8}")
    for off, r in res.items():
        for cost in COSTS:
            c = r[cost]
            ci = f"[{c['spearman_ci95'][0]:+.3f},{c['spearman_ci95'][1]:+.3f}]"
            log(f"{off:>7} {cost:>14} {c['spearman_mean']:>+10.3f} {ci:>18} "
                f"{c['norm_regret_mean']:>8.3f}")
    log("\nRead this as: l2_pred is LeWM's deployed cost. l2_exec is the same cost applied "
        "to the state\nthe candidate actually reached, so it removes the predictor and "
        "isolates the cost itself.\nIf l2_exec is high and l2_pred is low, the predictor is "
        "the bottleneck. If both are low, the\ncost function is the defect. If both are "
        "high, there is nothing here and this line stops.")


if __name__ == "__main__":
    main()
