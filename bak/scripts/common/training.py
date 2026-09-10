"""Shared GCIQL training infrastructure -- environment-agnostic. Consolidated from
tworoom_b3_gciql_shaping.py (MLP/ema_update/build_her_tuples), tworoom_b3_mixed_auxphi.py
(run_condition_resumable -- mode-2 auxiliary-regression, the mechanism established as the
one that actually works, see [[tworoom-b3b4-auxphi-fix]]), and tworoom_b3_convergence.py
(precompute_eval_set). run_condition_resumable was already environment-agnostic before this
move -- Push-T's pusht_b3_datatiers.py/pusht_b3_worker.py already imported and called it
unchanged, proving the "setup dict in, everything else generic" design already worked.

Hyperparameters below are the exact values both tworoom_b3_datatiers.py and
pusht_b3_datatiers.py already independently defined identically -- consolidating them here
removes that duplication, not just the surrounding code's.

One deliberate behavior change from the original tworoom_b3_mixed_auxphi.run_condition_resumable:
`ckpt_dir` is now a required argument (no more silent fallback to a hardcoded Two-Room-specific
ckpt_path() import, which would have made this module depend on a specific environment's script
-- exactly the kind of coupling this consolidation removes). Every current caller (Two-Room's
datatiers/worker scripts, Push-T's) already passes ckpt_dir explicitly except
tworoom_b3_mixed_auxphi.py's own now-largely-historical main(), which was updated to pass it too.
"""

import time

import numpy as np
import torch
import torch.nn as nn
from scipy import stats as sps

from common.checkpoint_io import load_checkpoint, save_checkpoint
from common.log_util import log

DEV = "cuda" if torch.cuda.is_available() else "cpu"

GAMMA = 0.99
TAU_EXPECTILE = 0.9
EMA_TAU = 0.005
HIDDEN = 256
BATCH_SIZE = 256
N_STEPS = 50000
EVAL_EVERY = 50
GOALS_PER_TRANSITION = 4
GOAL_GAMMA = 0.9  # geometric future-goal sampling distribution for HER


class MLP(nn.Module):
    def __init__(self, in_dim, hidden, out_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def ema_update(target, source, tau):
    with torch.no_grad():
        for pt, ps in zip(target.parameters(), source.parameters()):
            pt.mul_(1 - tau).add_(ps, alpha=tau)


def build_her_tuples(ep_idx, step_idx, action, seed=0, allowed_episode_ids=None, goal_gamma=GOAL_GAMMA):
    """Hindsight relabeling: for each real transition (i -> i+1), sample K future
    goals within the same episode, geometric future-offset distribution. Returns
    arrays (s_idx, next_idx, goal_idx, action_of_s, done).

    allowed_episode_ids: if given, only build tuples from these episodes (a train/test
    split at the episode level -- landmark-row indices returned are still absolute,
    indexing into the full landmark arrays, so no remapping is needed downstream).
    goal_gamma: the geometric future-offset parameter (offset ~ Geom(1-goal_gamma), mean
    1/(1-goal_gamma) steps): 0.9 = the established default (mean 10-step goals); the LeWM
    paper's GCIQL uses 0.99 for its critic and evaluates on 25-step goals (Push-T)."""
    rng = np.random.default_rng(seed)
    order = np.lexsort((step_idx, ep_idx))
    ep_o = ep_idx[order]

    # group landmark-row indices by episode, in step order
    boundaries = np.nonzero(np.diff(ep_o))[0] + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [len(order)]))
    episodes = [order[s:e] for s, e in zip(starts, ends)]
    if allowed_episode_ids is not None:
        allowed = set(int(x) for x in allowed_episode_ids)
        episodes = [rows for rows in episodes if int(ep_idx[rows[0]]) in allowed]

    s_list, next_list, goal_list, action_list, done_list = [], [], [], [], []
    for ep_rows in episodes:
        L = len(ep_rows)
        for t in range(L - 1):  # need a real next-step
            s_row, next_row = ep_rows[t], ep_rows[t + 1]
            remaining = L - 1 - t  # number of valid future offsets (>=1)
            for _ in range(GOALS_PER_TRANSITION):
                # geometric offset in [1, remaining], biased toward small offsets
                offset = min(1 + rng.geometric(1 - goal_gamma) - 1, remaining)
                offset = max(1, offset)
                goal_row = ep_rows[t + offset]
                s_list.append(s_row); next_list.append(next_row); goal_list.append(goal_row)
                action_list.append(s_row)  # index into `action` array (action taken AT s)
                done_list.append(offset == 1)

    return (np.array(s_list), np.array(next_list), np.array(goal_list),
            np.array(action_list), np.array(done_list))


def build_stitch_her_tuples(ep_idx, step_idx, id_i, id_j, allowed_episode_ids=None,
                             goals_per_edge=GOALS_PER_TRANSITION, hop_gamma=GOAL_GAMMA, seed=0):
    """Cross-episode HER goals licensed directly by identification edges -- no graph
    distance, no Dijkstra, no value regression. An identification edge says "landmark row
    r1 (episode i, step t1) and row r2 (episode j, step t2) are approximately the same
    state." Given that, any row in episode i strictly BEFORE t1 can be hindsight-relabeled
    with any row in episode j AT OR AFTER t2 as its goal: follow i's own real trajectory up
    to the identified point, then (per the identification) continue along j's future. The
    start row's real action and real next-transition are completely unaffected -- it's
    strictly before t1, so the relabeling only changes (goal, reward, done), exactly like
    ordinary same-episode HER. Both directions of each edge are used (identification is
    symmetric); within each direction, time is one-directional (start must be strictly
    before its own endpoint, goal at-or-after the other endpoint) -- only the *edges*
    themselves are symmetric, not which side of one is "start" vs "goal".

    id_i/id_j: raw identification-edge row pairs (common.graph_lib.build_graph_edges_for_env).
    Same-episode identification edges (if any) are skipped -- ordinary build_her_tuples
    already covers within-episode goals.

    hop_gamma: geometric bias (hop ~ Geom(1-hop_gamma), mean 1/(1-hop_gamma)) on how far past
    the identified point t2 the goal is sampled, instead of uniformly across the whole
    remaining suffix. Measured directly (scripts/investigations/pusht_stitch_diagnosis/
    diagnose.py) on Push-T expert_100: true state-space distance between (s, g) grows
    monotonically with this hop (correlation 0.62) -- mean true-dist 0.88 at hop<2 (on par
    with ordinary same-episode HER's 1.74) vs 3.31 at hop>=40, because only the single
    identification-edge hop itself is grounded in real data; nothing past it has any
    relationship to the action recorded at s, so the farther past t2 the goal is drawn the
    less legitimate the (s, a, g) relabeling gets. Uniform sampling put most stitched goals
    in that far, illegitimate regime (over half beyond hop 40 in the same measurement).
    Deliberately NOT the same knob as her_goal_gamma (which sets the same-episode HER
    horizon to match the paper's training regime, mean ~100 steps) -- reusing that value
    here would barely differ from uniform, since Push-T episodes average ~125 steps.
    Defaults to GOAL_GAMMA=0.9 (mean 10-step hop), the original established same-episode
    HER default, chosen because the diagnosis found hop 5-15 already comparable to real HER.

    done is always False: unlike a same-episode HER goal, a stitched goal's real step-gap
    is unknown (that's the whole point of not using Dijkstra), so there's no "this step
    reached it" signal to emit -- only reward=-1-per-step is defined."""
    rng = np.random.default_rng(seed)
    order = np.lexsort((step_idx, ep_idx))
    ep_o, step_o = ep_idx[order], step_idx[order]
    n = len(ep_idx)

    # next_of[row] = the landmark row for the real next env step, within the same episode
    # (-1 at each episode's last row). Landmarks are every frame of their episode (load_
    # landmarks samples steps=np.arange(L)), so this is always a real, unskipped transition.
    next_of = np.full(n, -1, dtype=np.int64)
    same_ep = ep_o[1:] == ep_o[:-1]
    consec = step_o[1:] == step_o[:-1] + 1
    adj = np.nonzero(same_ep & consec)[0]
    next_of[order[adj]] = order[adj + 1]

    # per-episode landmark rows and their steps, both sorted by step
    boundaries = np.nonzero(np.diff(ep_o))[0] + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [n]))
    ep_rows, ep_steps = {}, {}
    for s, e in zip(starts, ends):
        rows = order[s:e]
        eid = int(ep_idx[rows[0]])
        ep_rows[eid] = rows
        ep_steps[eid] = step_idx[rows]

    allowed = None if allowed_episode_ids is None else set(int(x) for x in allowed_episode_ids)

    cross = ep_idx[id_i] != ep_idx[id_j]
    ei_arr, ej_arr = id_i[cross], id_j[cross]

    s_list, next_list, goal_list, action_list = [], [], [], []

    def stitch(r_from, r_to):
        i_ep, i_t = int(ep_idx[r_from]), int(step_idx[r_from])
        j_ep, j_t = int(ep_idx[r_to]), int(step_idx[r_to])
        if allowed is not None and (i_ep not in allowed or j_ep not in allowed):
            return
        prefix_end = np.searchsorted(ep_steps[i_ep], i_t, side="left")
        if prefix_end == 0:
            return  # r_from is its episode's first frame -- no earlier start to relabel
        prefix = ep_rows[i_ep][:prefix_end]
        suffix_start = np.searchsorted(ep_steps[j_ep], j_t, side="left")
        suffix = ep_rows[j_ep][suffix_start:]  # suffix[0] == r_to itself (hop 0)
        max_hop = len(suffix) - 1

        k = min(goals_per_edge, len(prefix))
        s_rows = rng.choice(prefix, size=k, replace=False)
        for s_row in s_rows:
            hop = min(rng.geometric(1 - hop_gamma) - 1, max_hop)
            g_row = suffix[hop]
            s_list.append(s_row); next_list.append(next_of[s_row]); goal_list.append(g_row)
            action_list.append(s_row)

    for r1, r2 in zip(ei_arr, ej_arr):
        stitch(r1, r2)
        stitch(r2, r1)

    m = len(s_list)
    log(f"[stitch-her] {len(ei_arr)} cross-episode identification edges -> {m} stitched HER tuples")
    return (np.array(s_list, dtype=np.int64), np.array(next_list, dtype=np.int64),
            np.array(goal_list, dtype=np.int64), np.array(action_list, dtype=np.int64),
            np.zeros(m, dtype=bool))


def precompute_eval_set(row_pool, proprio, true_dist_oracle, n_pairs, seed):
    """Fix a set of (s,g) landmark-row pairs drawn from row_pool, compute their true
    distances ONCE via a single batched multi-source Dijkstra call."""
    rng = np.random.default_rng(seed)
    ei = rng.choice(row_pool, size=n_pairs, replace=True)
    ej = rng.choice(row_pool, size=n_pairs, replace=True)
    keep = ei != ej
    ei, ej = ei[keep], ej[keep]
    uniq_src, src_row = np.unique(ei, return_inverse=True)
    D = true_dist_oracle(proprio[uniq_src], proprio[ej])  # (n_uniq_src, n_pairs)
    true_d = D[src_row, np.arange(len(ei))]
    finite = np.isfinite(true_d)
    return ei[finite], ej[finite], true_d[finite]


def run_condition_resumable(tier, name, aux_lambda, setup, seed, ckpt_dir,
                             distance_source="graph", dist_matrix=None):
    """Mode-2 auxiliary-regression training loop: baseline's plain -1/0 TD objective, plus
    (if aux_lambda>0) a non-bootstrapped regression loss pulling V(s,g) toward a
    Phi-derived pseudo-value pseudo_v(s,g) = -(1-gamma^Phi(s,g))/(1-gamma). Reward itself is
    untouched -- Phi enters ONLY via the aux loss, which is why this doesn't inherit the
    reward-shaping noise potential-based shaping (mode 3) did.

    distance_source selects what pseudo_v's underlying distance is:
      "graph"     -- setup["phi_dist"] (the graph geodesic; default)
      "oracle"    -- dist_matrix (a precomputed true-distance NxN matrix, ceiling condition)
      "euclidean" -- ||z_s - z_g|| computed on the fly (no matrix needed, known-bad proxy)
    dist_matrix is required (and used) only for "graph"/"oracle"; ignored for "euclidean".

    ckpt_dir: directory the checkpoint is written under, as f"{tier}__{name}__s{seed}.pt"
    -- required (see module docstring for why this replaced an implicit environment-specific
    default)."""
    path = ckpt_dir / f"{tier}__{name}__s{seed}.pt"
    z, action = setup["z"], setup["action"]
    if distance_source == "graph":
        dist_matrix = setup["phi_dist"]
    d = setup["d"]

    ckpt = load_checkpoint(path)
    if ckpt is not None and ckpt.get("done"):
        log(f"[{tier}/{name}/s{seed}] already complete ({ckpt['step']} steps) -- skipping")
        return ckpt["history"]

    torch.manual_seed(seed)
    z_t = torch.from_numpy(z).to(DEV)
    act_t = torch.from_numpy(action).float().to(DEV)

    v_net = MLP(2 * d, HIDDEN).to(DEV)
    v_target = MLP(2 * d, HIDDEN).to(DEV)
    q_net = MLP(2 * d + act_t.shape[1], HIDDEN).to(DEV)
    q_target = MLP(2 * d + act_t.shape[1], HIDDEN).to(DEV)
    v_target.load_state_dict(v_net.state_dict())
    q_target.load_state_dict(q_net.state_dict())

    from stable_worldmodel.wm.gcrl.module import ExpectileLoss
    expectile_loss = ExpectileLoss(tau=TAU_EXPECTILE)
    opt = torch.optim.Adam(list(v_net.parameters()) + list(q_net.parameters()), lr=3e-4)

    s_idx, next_idx, goal_idx, act_idx, done_arr = build_her_tuples(
        setup["ep_idx"], setup["step_idx"], action, seed=seed, allowed_episode_ids=setup["train_eps"]
    )
    n_tuples = len(s_idx)

    train_ei, train_ej, train_true_d = setup["train_eval"]
    test_ei, test_ej, test_true_d = setup["test_eval"]
    train_zsg = torch.cat([z_t[train_ei], z_t[train_ej]], dim=-1)
    test_zsg = torch.cat([z_t[test_ei], z_t[test_ej]], dim=-1)

    rng = np.random.default_rng(seed + 1)
    history = []
    start_step = 1

    if ckpt is not None:
        v_net.load_state_dict(ckpt["v_net"]); v_target.load_state_dict(ckpt["v_target"])
        q_net.load_state_dict(ckpt["q_net"]); q_target.load_state_dict(ckpt["q_target"])
        opt.load_state_dict(ckpt["opt"])
        history = ckpt["history"]
        start_step = ckpt["step"] + 1
        rng.bit_generator.state = ckpt["numpy_rng"]
        torch.set_rng_state(ckpt["torch_rng"])
        if torch.cuda.is_available() and ckpt.get("torch_cuda_rng") is not None:
            torch.cuda.set_rng_state(ckpt["torch_cuda_rng"])
        log(f"[{tier}/{name}/s{seed}] resuming from step {start_step}/{N_STEPS} "
            f"({len(history)} eval points so far)")
    else:
        log(f"[{tier}/{name}/s{seed}] starting fresh, n_tuples={n_tuples}, aux_lambda={aux_lambda}")

    if start_step > N_STEPS:
        return history

    t0 = time.time()
    for step in range(start_step, N_STEPS + 1):
        batch = rng.integers(0, n_tuples, BATCH_SIZE)
        s, nx, g, a_i, dn = s_idx[batch], next_idx[batch], goal_idx[batch], act_idx[batch], done_arr[batch]

        zs, zn, zg = z_t[s], z_t[nx], z_t[g]
        a = act_t[a_i]
        mask = (~torch.from_numpy(dn).to(DEV)).float().unsqueeze(-1)
        reward = -mask  # plain baseline reward, untouched -- Phi enters ONLY via the aux loss below

        sg = torch.cat([zs, zg], dim=-1)
        ng = torch.cat([zn, zg], dim=-1)
        sag = torch.cat([zs, a, zg], dim=-1)

        with torch.no_grad():
            q = q_target(sag)
        v = v_net(sg)
        value_loss = expectile_loss(v, q.detach())

        aux_loss = torch.tensor(0.0, device=DEV)
        if aux_lambda > 0:
            if distance_source == "euclidean":
                dist_sg = torch.norm(zs - zg, dim=-1, keepdim=True)
            else:
                dist_sg = torch.from_numpy(dist_matrix[s, g]).float().to(DEV).unsqueeze(-1)
            pseudo_v = -(1 - GAMMA ** dist_sg) / (1 - GAMMA)
            aux_loss = ((v - pseudo_v) ** 2).mean()

        with torch.no_grad():
            next_v = v_target(ng)
            q_tgt = reward + GAMMA * mask * next_v
        q_pred = q_net(sag)
        critic_loss = ((q_pred - q_tgt) ** 2).mean()

        loss = value_loss + critic_loss + aux_lambda * aux_loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        ema_update(v_target, v_net, EMA_TAU)
        ema_update(q_target, q_net, EMA_TAU)

        if step % EVAL_EVERY == 0 or step == 1:
            with torch.no_grad():
                v_train = v_net(train_zsg).squeeze(-1).cpu().numpy()
                v_test = v_net(test_zsg).squeeze(-1).cpu().numpy()
            rho_train, _ = sps.spearmanr(train_true_d, -v_train)
            rho_test, _ = sps.spearmanr(test_true_d, -v_test)
            history.append(dict(step=step, value_loss=float(value_loss.item()),
                                 critic_loss=float(critic_loss.item()), aux_loss=float(aux_loss.item()),
                                 spearman_train=float(rho_train), spearman_test=float(rho_test)))
            is_done = step == N_STEPS
            save_checkpoint(path, step, v_net, v_target, q_net, q_target, opt, history, rng, is_done)
            if step % (EVAL_EVERY * 20) == 0 or step == 1:
                log(f"[{tier}/{name}/s{seed}] step={step} vloss={value_loss.item():.3f} "
                    f"closs={critic_loss.item():.3f} auxloss={aux_loss.item():.3f} "
                    f"train_rho={rho_train:.4f} test_rho={rho_test:.4f} elapsed={time.time()-t0:.1f}s")

    return history


def peak_stats(history):
    i = max(range(len(history)), key=lambda k: history[k]["spearman_test"])
    return history[i]
