"""B3 (scoped down): does graph-potential shaping help GCIQL value learning on Two-Room?

LIBRARY CODE ONLY -- do not run this file's own main() for results. Its original run
(no train/test split, fixed 5000 steps, eval pairs resampled from the same pool used
for training) produced a misleading comparison and has been retracted; see
tworoom_b3_convergence.py for the corrected methodology (held-out episode split, fixed
eval sets, evaluated to convergence). This file is kept because tworoom_b3_convergence.py
imports MLP, ema_update, build_her_tuples, and log from it.

Scope, as agreed: ONE data tier (the real expert dataset, no synthetic degradation),
ONE algorithm (GCIQL), baseline reward vs. baseline + graph-potential shaping. A
from-scratch, self-contained training loop -- NOT the full 955-line Lightning/Hydra
script from github.com/galilai-group/stable-worldmodel (that script is built around a
trainable DINO-style patch encoder with EMA teacher/student wrappers, multi-GPU sync,
and its own Hydra config tree we don't have; using it here would mean fighting a lot
of infrastructure irrelevant to this specific question). Reused directly from that
codebase: `stable_worldmodel.wm.gcrl.ExpectileLoss` -- the exact IQL expectile-regression
loss, verbatim.

Mechanism under test:
  reward(s,a,s',g)          = -1 if s' != g else 0            (standard sparse GCRL reward)
  shaped_reward(s,a,s',g)    = reward + gamma*Phi(s',g) - Phi(s,g)
  Phi(s,g)                   = -d_graph(s,g)                   (same graph as B0/B1/B2)

Ng, Harada & Russell (1999): shaped_reward has the SAME optimal policy as reward, for
any Phi. So this test is about optimization dynamics (does the value function get
accurate faster / more accurately), not about correctness.

Evaluation metric (not live rollouts -- deferred, see report): Spearman correlation
between the learned V(s,g) and TRUE wall-respecting distance (the same oracle from
B0/B1/B2), tracked over training. A good value function should rank goal-distances
correctly; this directly measures whether shaping gets there faster and/or better,
without requiring a full AWR actor + live gym rollouts.
"""

import json
import sys
import time
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401 -- see tworoom_b2_encoder_swap.py's note on this dependency
import numpy as np
import torch
import torch.nn as nn
from scipy import stats as sps
from scipy.sparse.csgraph import dijkstra

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.log_util import log  # noqa: F401
from common.training import MLP, build_her_tuples, ema_update  # noqa: F401
from common.training import EMA_TAU, GAMMA, HIDDEN, TAU_EXPECTILE  # noqa: F401 -- match common/training.py exactly
from tworoom_b0_graph_gate import (
    DEV, H5_PATH, build_true_distance_oracle, build_weighted_graph, find_graph_edges,
)

ROOT = Path(__file__).resolve().parent.parent
LANDMARKS = ROOT / "outputs" / "b0_tworoom" / "landmarks_ep50.npz"
OUT_DIR = ROOT / "outputs" / "b3_tworoom"
OUT_DIR.mkdir(parents=True, exist_ok=True)

Q_CALIB = 1e-3
# N_STEPS/EVAL_EVERY intentionally NOT imported from common/training.py: this file's own
# retracted run_condition/main() below used smaller original values (5000/250) predating
# the later held-out-eval methodology fix (tworoom_b3_convergence.py) that settled on
# 50000/50 for every other script -- these two stay local so this retracted historical
# script's exact original behavior is preserved if ever re-run for reference.
BATCH_SIZE = 256
N_STEPS = 5000
EVAL_EVERY = 250
SEED = 0


def run_condition(name, use_shaping, z, proprio, s_idx, next_idx, goal_idx, act_idx,
                   done, action, phi_dist, true_dist_oracle, d, seed=SEED):
    torch.manual_seed(seed)
    n_tuples = len(s_idx)
    z_t = torch.from_numpy(z).to(DEV)
    act_t = torch.from_numpy(action).float().to(DEV)

    v_net = MLP(2 * d, HIDDEN).to(DEV)
    v_target = MLP(2 * d, HIDDEN).to(DEV)
    v_target.load_state_dict(v_net.state_dict())
    q_net = MLP(2 * d + act_t.shape[1], HIDDEN).to(DEV)
    q_target = MLP(2 * d + act_t.shape[1], HIDDEN).to(DEV)
    q_target.load_state_dict(q_net.state_dict())

    from stable_worldmodel.wm.gcrl.module import ExpectileLoss  # not re-exported via wm.gcrl.__init__
    expectile_loss = ExpectileLoss(tau=TAU_EXPECTILE)

    opt = torch.optim.Adam(list(v_net.parameters()) + list(q_net.parameters()), lr=3e-4)

    rng = np.random.default_rng(seed + 1)
    history = []
    t0 = time.time()
    for step in range(1, N_STEPS + 1):
        batch = rng.integers(0, n_tuples, BATCH_SIZE)
        s, nx, g, a_i, dn = s_idx[batch], next_idx[batch], goal_idx[batch], act_idx[batch], done[batch]

        zs, zn, zg = z_t[s], z_t[nx], z_t[g]
        a = act_t[a_i]
        mask = (~torch.from_numpy(dn).to(DEV)).float().unsqueeze(-1)
        reward = -mask  # -1 if not done, 0 if done

        if use_shaping:
            phi_s = torch.from_numpy(-phi_dist[s, g]).float().to(DEV).unsqueeze(-1)
            phi_n = torch.from_numpy(-phi_dist[nx, g]).float().to(DEV).unsqueeze(-1)
            reward = reward + GAMMA * phi_n - phi_s

        sg = torch.cat([zs, zg], dim=-1)
        ng = torch.cat([zn, zg], dim=-1)
        sag = torch.cat([zs, a, zg], dim=-1)

        with torch.no_grad():
            q = q_target(sag)
        v = v_net(sg)
        value_loss = expectile_loss(v, q.detach())

        with torch.no_grad():
            next_v = v_target(ng)
            q_tgt = reward + GAMMA * mask * next_v
        q_pred = q_net(sag)
        critic_loss = ((q_pred - q_tgt) ** 2).mean()

        loss = value_loss + critic_loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        ema_update(v_target, v_net, EMA_TAU)
        ema_update(q_target, q_net, EMA_TAU)

        if step % EVAL_EVERY == 0 or step == 1:
            with torch.no_grad():
                n_eval = 500
                ei = rng.integers(0, z.shape[0], n_eval)
                ej = rng.integers(0, z.shape[0], n_eval)
                zsg = torch.cat([z_t[ei], z_t[ej]], dim=-1)
                v_eval = v_net(zsg).squeeze(-1).cpu().numpy()
            # one multi-source Dijkstra call for the whole eval batch, not one per pair
            uniq_src, src_row = np.unique(ei, return_inverse=True)
            D = true_dist_oracle(proprio[uniq_src], proprio[ej])  # (n_uniq_src, n_eval)
            true_d_arr = D[src_row, np.arange(n_eval)]
            finite = np.isfinite(true_d_arr)
            rho, _ = sps.spearmanr(true_d_arr[finite], (-v_eval[finite]))  # -V ~ distance
            history.append(dict(step=step, value_loss=float(value_loss.item()),
                                 critic_loss=float(critic_loss.item()), spearman_v_vs_true=float(rho)))
            if step % (EVAL_EVERY * 5) == 0 or step == 1:
                log(f"[{name}] step={step} value_loss={value_loss.item():.3f} "
                    f"critic_loss={critic_loss.item():.3f} spearman(-V,true_dist)={rho:.4f} "
                    f"elapsed={time.time()-t0:.1f}s")

    return history


def main():
    log(f"device={DEV}")
    d_land = np.load(LANDMARKS)
    z, ep_idx, step_idx, proprio = d_land["z"], d_land["ep_idx"], d_land["step_idx"], d_land["proprio"]
    n, d = z.shape
    log(f"landmarks: {n} frames, d={d}")

    f = h5py.File(H5_PATH, "r", swmr=True, rdcc_nbytes=256 * 1024 * 1024)
    ep_offset = f["ep_offset"][:]
    global_idx = ep_offset[ep_idx] + step_idx
    action = f["action"][:][global_idx].astype(np.float32)
    f.close()
    log(f"actions loaded: {action.shape}")

    log("building graph (transition + identification edges, q=1e-3, id_weight=1.0)...")
    order = np.lexsort((step_idx, ep_idx))
    ep_o, step_o = ep_idx[order], step_idx[order]
    adj = (ep_o[1:] == ep_o[:-1]) & (step_o[1:] == step_o[:-1] + 1)
    z_o = z[order]
    rho_hat = float(np.mean(np.sum(z_o[:-1][adj] * z_o[1:][adj], axis=1)) / d)
    eps2 = 2 * (1 - rho_hat) * sps.chi2.ppf(Q_CALIB, d)
    log(f"rho_hat={rho_hat:.4f} eps2={eps2:.2f}")

    trans_i, trans_j, id_i, id_j = find_graph_edges(z, ep_idx, step_idx, eps2)
    graph = build_weighted_graph(n, trans_i, trans_j, id_i, id_j, id_weight=1.0)
    log(f"graph: {len(trans_i)} transition edges, {len(id_i)} identification edges")

    log("precomputing full pairwise graph-distance matrix (Dijkstra from all landmarks)...")
    t0 = time.time()
    phi_dist = dijkstra(graph, indices=np.arange(n), directed=False)
    log(f"done in {time.time()-t0:.1f}s, shape={phi_dist.shape}, "
        f"finite fraction={np.isfinite(phi_dist).mean():.4f}")
    # replace any disconnected (inf) entries with a large-but-finite value so Phi stays usable
    finite_max = phi_dist[np.isfinite(phi_dist)].max()
    phi_dist = np.where(np.isfinite(phi_dist), phi_dist, finite_max * 2)

    log("building HER training tuples...")
    s_idx, next_idx, goal_idx, act_idx, done = build_her_tuples(ep_idx, step_idx, action, seed=SEED)
    log(f"n_tuples={len(s_idx)} (done_fraction={done.mean():.4f})")

    true_dist_oracle = build_true_distance_oracle()

    results = {}
    for use_shaping, name in [(False, "baseline"), (True, "shaped")]:
        log(f"\n{'='*60}\ntraining condition: {name}\n{'='*60}")
        history = run_condition(name, use_shaping, z, proprio, s_idx, next_idx, goal_idx,
                                 act_idx, done, action, phi_dist, true_dist_oracle, d)
        results[name] = history

    out_path = OUT_DIR / "b3_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    log(f"\nwrote {out_path}")

    base_final = results["baseline"][-1]["spearman_v_vs_true"]
    shaped_final = results["shaped"][-1]["spearman_v_vs_true"]
    log(f"\nFINAL: baseline spearman={base_final:.4f}  shaped spearman={shaped_final:.4f}")


if __name__ == "__main__":
    main()
