"""Hierarchical latent MPC on Push-T: LeWM's CEM planner with a GAS-style graph (TDR space,
TE-filtered TD-aware nodes, Dijkstra cost-to-go) supplying the objective instead of the
paper's terminal L2 to the final goal. The low-level controller is CEM itself; nothing is
trained beyond the TDR (scripts/gas_mpc_prepare.py, full 18,685-episode dataset).

The seam is jepa.JEPA.criterion(info_dict) (the same monkeypatch viability_live_rollout.py
uses): it sees the current latent (`emb`), every predicted block latent (`predicted_emb`)
and the goal latent (`goal_emb`), so each method below is a pure function of those plus
the graph -- stateless per CEM iteration, with one Dijkstra per (goal) cached per episode.

Methods (`+mpc.method=`):
    l2            LeWM's own MSE(z_T, z_goal)                                    -- baseline
    tdr           ||psi(z_T) - psi(z_goal)||                 (temporal metric, no graph)
    ctg           min( ||psi(z_T)-psi_g||, min_v ||psi(z_T)-psi_v|| + D_goal[v] )  graph cost-to-go
    subgoal       Alg. 1 node (with `lookahead` along the Dijkstra path) as the goal, cost
                  = LeWM's MSE to the node's medoid frame latent; true goal once within
                  `subgoal_threshold` in TDR space
    subgoal_tdr   same node, cost = ||psi(z_T) - centre_v||
    dir           -<psi(z_T) - psi(z_cur), h_dir>, h_dir = unit direction to the subgoal
                  (the paper's r^dir summed over the horizon); `tdr` once within threshold
    path          sum_k ||psi(z_k) - w_k|| over the predicted block latents, waypoints w_k
                  at cumulative path distance k * `step_units` toward the goal
    random        uniform action from the env's own action space at every env step (gymnasium
                  `action_space.sample()` via stable_worldmodel's RandomPolicy) -- no model, no
                  CEM, no graph; the floor every other method must clear. Ignores every other
                  `+mpc.*` add-on below (asserted). `mpc.seed` seeds the action draw.
Add-ons:
    +mpc.support_lambda=L   adds L * min_v ||psi(z_T) - psi_v||
    +mpc.retrieval=true     warm-starts CEM's mean with the logged 25-action block of the dataset
                            frame that starts nearest psi(z_cur) and lands nearest the Alg.-1
                            subgoal node 25 steps later (graph as an action library)
    +mpc.critic_beta=B      adds B * std(-log V(z_T, z_goal, h_rem)) to std(base cost), both
                            standardised by median/IQR over the CEM population (the viability
                            critic outputs/pusht/critic_training/critic_full_s0/critic.pt; h_rem follows the
                            plan-call clock as in viability_live_rollout.py)
    +mpc.receding=R         replan every R blocks (default 5 = the paper's open-loop 25 steps)
    +mpc.final_thresh=F     switch from the subgoal node to the TRUE goal once ||psi(z_cur)-psi_g|| <= F
                            TDR units (default = subgoal_threshold = h_td; 13.7 = one lookahead, i.e.
                            one CEM horizon, so the final 25 steps are planned straight at the goal)
    +mpc.final_metric=l2    in that final phase score with LeWM's own z-space L2 to the goal frame
                            instead of the TDR norm (`same` = keep the method's metric)
    +mpc.critic_cost=et     critic term = budget-capped expected hitting time in ENV STEPS,
                            SKIP * sum_{h in grid, h <= h_rem} (1 - V(z_T, z_goal, h)); ranks feasible
                            endpoints by how soon they arrive, infeasible ones at the cap. Hitting-time
                            heads compute the identical survival sum in one network evaluation. `etfull`
                            = the same sum over the critic's whole grid (budget-agnostic); `nlv` =
                            -log V(z_T, z_goal, h_rem) (the original)
    +mpc.compose=steps      NO per-population standardisation: base cost converted to env steps
                            through the empirical median-distance-by-step-gap table (TDR units or
                            z-space L2, outputs/pusht/gap_calib_s{seed}.json) + critic_beta * the
                            `et`/`etfull` term (already in steps). critic_beta=1 is the nominal
                            "same units, no weight" setting. `std` = the original robust_std sum
    +mpc.critic_filter=TAU  hard feasibility filter: candidates with V(z_T, z_goal, h_rem) < TAU rank
                            after every feasible candidate (ordered by V among themselves); feasible
                            ones keep the base/composed cost. Loads the critic even with beta=0
    +mpc.critic_final=false drop the critic term for an env that is in the FINAL phase (subgoal
                            methods with final_metric=l2: the base cost is then LeWM's own z-space L2
                            to the goal and nothing else). Default true = the critic term is added in
                            every phase. Tag `_nocf`. Motivation: on Reacher same25, B lost 6 points to
                            L2 entirely in the close-range second plan (2026-09-17)
    +mpc.critic_member=K   use only zero-based ensemble member K at inference (-1 = ensemble mean).
                            This is an ablation of whether averaging prevents critic exploitation.

Task sets: one fixed set of 200 (start, goal) tasks per protocol, drawn once from TASK_SEED
and shared by every method and every seed; pairs the env's success predicate already accepts
at t=0 are rejected at sampling. Both endpoints come from the TDR held-out episodes;
trajectory intervals never overlap within a protocol (pool `task200u`).
Older pools and results are preserved. A run evaluates the first `eval.num_eval` tasks;
`mpc.seed` changes only CEM's random draws.

Protocols (`+mpc.protocol=`): cross (the fixed cross-episode pairs, budget 250), same25 /
same50 / same100 (paper protocol at that goal offset, budget 2x offset capped at
250). `+mpc.budget=N` overrides the protocol's env-step budget (tag `_budN`) -- a diagnostic
knob, e.g. the same25 tasks with same50's 100-step budget to separate task difficulty from the
number of replans; keep such runs out of the report's OUT/eval (set GAS_MPC_OUT). Pairs are the shared files outputs/pusht/pairs/pairs_*_n{n}_s{seed}.json,
identical for every method. Rollouts run in chunks of `mpc.chunk` pairs; each chunk's result
is cached under outputs/pusht/eval/ and skipped on re-run, so any run can be resumed.

    python scripts/gas_mpc_eval.py +mpc.method=ctg +mpc.protocol=cross eval.num_eval=50
    python scripts/gas_mpc_eval.py +mpc.method=subgoal +mpc.lookahead=15 +mpc.protocol=same50
"""

import os

os.environ.setdefault("MUJOCO_GL", "egl")
# Required by torch.use_deterministic_algorithms(True) below for deterministic cuBLAS ops
# (matmuls in the predictor/CEM cost evaluation) -- must be set before CUDA context init.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import hashlib
import json
import pickle
import random
import sys
import time
from pathlib import Path

import hydra
import numpy as np
import stable_worldmodel as swm
import torch
from omegaconf import DictConfig, OmegaConf
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from sklearn import preprocessing

# The CEM solver's own candidate sampling is seeded (a dedicated torch.Generator), but
# candidate scoring runs the predictor through cuDNN conv/reduction kernels that are
# nondeterministic by default -- small float differences there can flip which candidate
# is elite, and that nudge compounds over replans into different executed trajectories
# (confirmed: identical +mpc.seed reran on the same machine gave 50.0% then 46.0% on
# same50/seed2). Forcing determinism trades some speed for genuinely reproducible runs.
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)

# Even with the above, 3 parallel identical +mpc.seed reps (same100, s3, n50, 2026-09-21)
# matched on success_rate but diverged on individual tasks' final_pos_err (20-24% of
# tasks, up to ~240px) -- no torch determinism warning fired, so the remaining suspect is
# CPU-side: multi-threaded BLAS (numpy/MKL/OpenBLAS) reductions are not order-deterministic
# across process launches even at a fixed thread count. For a genuinely bit-reproducible
# run, launch with OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 (slower) and
# re-verify with the same parallel-reps protocol before trusting it fixed the gap.

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from common.envs import ENV_MECHANICS
from common.gas import DEV
from common.lewm_loader import load_lewm
from common.log_util import log
from gas_mpc_prepare import ENV, MECH, OUT, graph_path, load_tdr, task_heldout_episodes
from viability_cross_episode_baseline import evaluate_pairs, img_transform, load_or_make_pairs

# Environment comes from GAS_MPC_ENV (gas_mpc_prepare.ENV). Push-T keeps every historical
# path; another env gets its own OUT (outputs/<env>/) with pairs, eval and critic
# under it, and its own config/eval/<env>.yaml (World env name + state/goal callables).
PAIRS_DIR = OUT / "pairs"
EVAL_DIR = OUT / "eval"
TASK_SEED = 20260915
SMOKE = os.environ.get("GAS_MPC_SMOKE") == "1"
if SMOKE and "pipeline_smoke" not in OUT.parts:
    raise ValueError("Smoke evaluation requires an isolated pipeline_smoke output directory")
TASKS = 2 if SMOKE else 200
# Separate held-out, nontrivial, disjoint pools from all historical results.  The archived
# unfiltered pool is selectable only for the old-vs-new task-pool comparison driver.
POOL = os.environ.get("GAS_MPC_POOL", f"task{TASKS}u")
if POOL not in (f"task{TASKS}", f"task{TASKS}u"):
    raise ValueError(f"GAS_MPC_POOL must be task{TASKS} or task{TASKS}u, got {POOL!r}")
CROSS_BUDGET = {"pusht": 250, "reacher": 200, "cube": 200}[ENV]   # env steps for arbitrary goals: Push-T's
                                                    # max episode; Reacher's dataset episodes are 201 steps;
                                                    # cube: same as Reacher (episode length unverified)

PROTOCOLS = {
    "cross": dict(pairing="cross_episode", offset=None, budget=CROSS_BUDGET),
    "same25": dict(pairing="same_episode", offset=25, budget=50),
    "same50": dict(pairing="same_episode", offset=50, budget=100),
    "same100": dict(pairing="same_episode", offset=100, budget=min(200, CROSS_BUDGET)),
}


# ----------------------------------------------------------------------------------------
# graph-side state: one Dijkstra per goal, everything else stateless
# ----------------------------------------------------------------------------------------

class GraphOracle:
    """Holds the node graph on the GPU and answers, for a batch of (z_cur, z_goal):
    dist-to-goal per node (Dijkstra with the goal attached as a temporary node, official
    GAS code), the Alg.-1 subgoal (optionally walked `lookahead` units further along the
    shortest path), and waypoint sequences. Dijkstra results are cached per goal latent."""

    def __init__(self, g, tdr, h_td, subgoal_threshold, lookahead, step_units, n_waypoints, final_thresh=None):
        self.tdr = tdr
        self.h_td = float(h_td)
        self.thresh = float(subgoal_threshold)
        self.final_thresh = float(subgoal_threshold if final_thresh is None else final_thresh)
        self.lookahead = float(lookahead)
        self.step_units = float(step_units)
        self.n_way = int(n_waypoints)
        self.centers = np.asarray(g["centers"], dtype=np.float32)
        self.C = torch.from_numpy(self.centers).to(DEV)
        self.node_z = torch.from_numpy(np.asarray(g["node_z"], dtype=np.float32)).to(DEV)
        self.graph = g["graph"].tocsr()
        self.n = len(self.centers)
        self.cache = {}
        self.stats = dict(n_goals=0, n_attached=[], n_reachable=[], n_select=0, n_fallback=0,
                          n_unreachable=0, n_final=0, n_walk=[])

    @torch.no_grad()
    def psi(self, z):
        return self.tdr.phi(z)

    def _key(self, hg):
        return hashlib.md5(np.round(hg, 4).tobytes()).hexdigest()

    def goal_info(self, hg):
        """hg: (dim,) numpy psi of the goal. Returns dict(dist (n,), pred (n,), dist_t)."""
        k = self._key(hg)
        if k in self.cache:
            return self.cache[k]
        d = np.linalg.norm(self.centers - hg, axis=1)
        thresh = max(self.h_td, 1.2 * float(d.min()))
        attach = np.nonzero(d <= thresh)[0]
        n = self.n
        g = self.graph.tocoo()
        rows = np.concatenate([g.row, np.full(len(attach), n), attach])
        cols = np.concatenate([g.col, attach, np.full(len(attach), n)])
        vals = np.concatenate([g.data, d[attach], d[attach]])
        aug = csr_matrix((vals, (rows, cols)), shape=(n + 1, n + 1))
        dist, pred = dijkstra(aug, directed=False, indices=n, return_predecessors=True)
        info = dict(dist=dist[:n], pred=pred[:n], dist_t=torch.from_numpy(dist[:n].astype(np.float32)).to(DEV),
                    hg=hg.copy(), n_attached=int(len(attach)), n_reachable=int(np.isfinite(dist[:n]).sum()))
        self.cache[k] = info
        self.stats["n_goals"] += 1
        self.stats["n_attached"].append(info["n_attached"]); self.stats["n_reachable"].append(info["n_reachable"])
        return info

    def select(self, hc, gi):
        """Alg. 1 from psi(z_cur)=hc: among nodes within `thresh` (all nodes if none), argmin
        dist_to_goal + ||hc - v||; then walk the shortest path toward the goal until the
        cumulative TDR distance from hc reaches `lookahead`. Returns (node or -1 for the
        goal itself, flags)."""
        d_goal = float(np.linalg.norm(gi["hg"] - hc))
        self.stats["n_select"] += 1
        if d_goal <= self.final_thresh:
            self.stats["n_final"] += 1
            return -1, dict(final=True)
        d = np.linalg.norm(self.centers - hc, axis=1)
        near = np.nonzero(d <= self.thresh)[0]
        fallback = len(near) == 0
        if fallback:
            self.stats["n_fallback"] += 1
            near = np.arange(self.n)
        score = gi["dist"][near] + d[near]
        if not np.isfinite(score).any():
            self.stats["n_unreachable"] += 1
            return int(near[np.argmin(d[near])]), dict(final=False, unreachable=True, fallback=fallback)
        v = int(near[np.argmin(score)])
        total = float(d[v])
        walked = 0
        while total < self.lookahead:
            nxt = int(gi["pred"][v])
            if nxt < 0 or nxt >= self.n:            # next hop is the goal node itself
                self.stats["n_walk"].append(walked)
                return -1, dict(final=True, walked=walked, fallback=fallback)
            total += float(np.linalg.norm(self.centers[nxt] - self.centers[v]))
            v = nxt
            walked += 1
        self.stats["n_walk"].append(walked)
        return v, dict(final=False, walked=walked, fallback=fallback)

    def waypoints(self, hc, gi):
        """Nodes along the shortest path from the Alg.-1 pick to the goal, one per predicted
        block at cumulative distance ~ k * step_units (the goal itself once the path ends).
        Returns (n_way, dim) psi targets."""
        d_goal = float(np.linalg.norm(gi["hg"] - hc))
        out = np.repeat(gi["hg"][None], self.n_way, axis=0)
        if d_goal <= self.final_thresh:
            return out
        d = np.linalg.norm(self.centers - hc, axis=1)
        near = np.nonzero(d <= self.thresh)[0]
        if len(near) == 0:
            near = np.arange(self.n)
        score = gi["dist"][near] + d[near]
        if not np.isfinite(score).any():
            return out
        v = int(near[np.argmin(score)])
        path, cum = [v], [float(d[v])]
        while True:
            nxt = int(gi["pred"][v])
            if nxt < 0 or nxt >= self.n:
                break
            cum.append(cum[-1] + float(np.linalg.norm(self.centers[nxt] - self.centers[v])))
            path.append(nxt)
            v = nxt
        cum = np.asarray(cum)
        for k in range(self.n_way):
            target = (k + 1) * self.step_units
            if target >= cum[-1] + self.step_units / 2:   # beyond the last node: the goal
                continue
            j = int(np.argmin(np.abs(cum - target)))
            out[k] = self.centers[path[j]]
        return out

    def summary(self):
        s = self.stats
        n = max(1, s["n_select"])
        return dict(n_goals=s["n_goals"], mean_attached=float(np.mean(s["n_attached"])) if s["n_attached"] else None,
                    mean_reachable=float(np.mean(s["n_reachable"])) if s["n_reachable"] else None,
                    n_select=s["n_select"], frac_final=s["n_final"] / n, frac_fallback=s["n_fallback"] / n,
                    frac_unreachable=s["n_unreachable"] / n,
                    mean_walk=float(np.mean(s["n_walk"])) if s["n_walk"] else None)


# ----------------------------------------------------------------------------------------
# retrieval warm-start, plan clock, critic
# ----------------------------------------------------------------------------------------

class Retriever:
    """Graph as an action library. For (psi(z_cur), target psi) pick the dataset row i
    minimising ||psi_i - psi_cur|| + ||psi_{i+L} - target|| over rows whose episode still has
    L steps left, and return its L logged actions (z-scored with the eval's own scaler) as
    the CEM initial mean, shaped (horizon, action_block * act_dim)."""

    def __init__(self, psi_all, actions, ep_offset, ep_len, scaler, horizon, block):
        self.L = int(horizon * block)
        self.horizon, self.block = int(horizon), int(block)
        n = len(psi_all)
        ep_end = np.empty(n, dtype=np.int64)
        for o, l in zip(ep_offset, ep_len):
            ep_end[o:o + l] = o + l - 1
        rows = np.arange(n)
        self.valid = rows[rows + self.L <= ep_end]
        self.P0 = torch.from_numpy(psi_all[self.valid]).to(DEV)
        self.PL = torch.from_numpy(psi_all[self.valid + self.L]).to(DEV)
        self.actions = actions
        self.scaler = scaler
        self.stats = dict(n=0, d_start=[], d_end=[])

    @torch.no_grad()
    def init_action(self, hc, target):
        """hc, target: (dim,) tensors on DEV -> (horizon, block*act_dim) float32 tensor."""
        d = torch.cdist(hc[None], self.P0)[0] + torch.cdist(target[None], self.PL)[0]
        j = int(d.argmin())
        i = int(self.valid[j])
        a = self.actions[i:i + self.L]                                   # (L, act_dim) raw
        a = self.scaler.transform(a).astype(np.float32)                   # z-scored, as CEM sees them
        self.stats["n"] += 1
        self.stats["d_start"].append(float(torch.linalg.norm(hc - self.P0[j])))
        self.stats["d_end"].append(float(torch.linalg.norm(target - self.PL[j])))
        return torch.from_numpy(a.reshape(self.horizon, -1))

    def summary(self):
        s = self.stats
        return dict(n_retrievals=s["n"], median_d_start=float(np.median(s["d_start"])) if s["d_start"] else None,
                    median_d_end=float(np.median(s["d_end"])) if s["d_end"] else None)


class PlanClock:
    """Plan-call counter (every solve advances all live envs by receding*block env steps), so
    the critic can be asked about the true remaining budget."""

    def __init__(self, budget, step_per_call, horizon_steps, h_cap):
        self.budget, self.step_per_call, self.horizon_steps, self.h_cap = budget, step_per_call, horizon_steps, h_cap
        self.n_calls = 0

    @property
    def h(self):
        return min(max(0, self.budget - self.n_calls * self.step_per_call - self.horizon_steps), self.h_cap)


GAP_CALIB_GAPS = [1, 2, 3, 5, 8, 10, 15, 20, 25, 30, 40, 50]


def gap_calibration(seed, per_gap=20000, rng_seed=0):
    """Median TDR distance and median z-space squared L2 between frames `g` env steps apart in the
    same episode, for g in GAP_CALIB_GAPS -- the table that converts a cost in either metric to
    env steps (`compose=steps`). Computed only from the training cache and cached as JSON."""
    from gas_mpc_prepare import TAG, ensure_psi, load_cache
    path = OUT / f"gap_calib_s{seed}{TAG}.json"
    if path.exists():
        saved = json.loads(path.read_text())
        if saved.get("scope") != "training_only":
            raise ValueError(f"{path} was calibrated using the full dataset; rebuild it")
        return saved
    cache = load_cache()
    z, psi = cache["z"], ensure_psi(seed)
    n = len(z)
    ep_end = np.empty(n, dtype=np.int64)
    for o, l in zip(cache["ep_offset"], cache["ep_len"]):
        ep_end[o:o + l] = o + l - 1
    rng = np.random.default_rng(rng_seed)
    tdr_med, l2_med = [], []
    for g in GAP_CALIB_GAPS:
        eligible = np.flatnonzero(np.arange(n) + g <= ep_end)
        if not len(eligible):
            raise ValueError(f"No training episode supports calibration gap {g}")
        i = rng.choice(eligible, size=per_gap)
        j = i + g
        tdr_med.append(float(np.median(np.linalg.norm(psi[i] - psi[j], axis=1))))
        l2_med.append(float(np.median(((z[i].astype(np.float32) - z[j].astype(np.float32)) ** 2).sum(1))))
    # enforce monotone tables so the inverse (distance -> steps) is well defined
    tdr_med = np.maximum.accumulate(np.asarray(tdr_med)).tolist()
    l2_med = np.maximum.accumulate(np.asarray(l2_med)).tolist()
    calib = dict(gaps=GAP_CALIB_GAPS, tdr=tdr_med, l2=l2_med, per_gap=per_gap, scope="training_only")
    path.write_text(json.dumps(calib, indent=1))
    log(f"[calib] wrote {path.name}: tdr={[round(v, 2) for v in tdr_med]} l2={[round(v, 1) for v in l2_med]}")
    return calib


class StepsConverter:
    """distance (TDR units or z-space squared L2) -> env steps by piecewise-linear interpolation
    of the gap-calibration table, linear extrapolation past the last gap."""

    def __init__(self, calib, device):
        self.tab = {}
        for k in ("tdr", "l2"):
            xs = torch.tensor([0.0] + list(calib[k]), dtype=torch.float32, device=device)
            ys = torch.tensor([0.0] + list(calib["gaps"]), dtype=torch.float32, device=device)
            keep = torch.cat([torch.tensor([True], device=device), xs[1:] > xs[:-1]])   # strictly increasing
            self.tab[k] = (xs[keep], ys[keep])

    def __call__(self, x, kind):
        xs, ys = self.tab[kind]
        idx = torch.searchsorted(xs, x.contiguous().float(), right=True).clamp(1, len(xs) - 1)
        x0, x1, y0, y1 = xs[idx - 1], xs[idx], ys[idx - 1], ys[idx]
        return y0 + (x - x0) * (y1 - y0) / (x1 - x0)


def robust_std(x):
    """(x - median) / IQR along the sample axis, per env."""
    q = torch.quantile(x, torch.tensor([0.25, 0.5, 0.75], device=x.device), dim=1)  # (3, B)
    iqr = (q[2] - q[0]).clamp_min(1e-6)
    return (x - q[1][:, None]) / iqr[:, None]


def _expand_for_cost(info_dict, s, device, dtype):
    """Mirror CEMSolver.solve's own info_dict -> (B, S, ...) expansion (stable_worldmodel/
    solver/cem.py), so a verification call to model.get_cost sees exactly the shapes/keys a
    real CEM candidate batch would, just with S=s instead of num_samples."""
    out = {}
    for k, v in info_dict.items():
        if torch.is_tensor(v):
            target_dtype = dtype if v.is_floating_point() else None
            out[k] = v.to(device=device, dtype=target_dtype).unsqueeze(1).expand(v.shape[0], s, *v.shape[1:])
        elif isinstance(v, np.ndarray):
            out[k] = np.repeat(v[:, None, ...], s, axis=1)
        else:
            out[k] = v
    return out


class MeanVarRecorder:
    """CEMSolver callback: snapshot the PRE-update (mean, var) -- the distribution actually
    used to sample that iteration's candidates -- at chosen 0-indexed steps (step=0 is
    iteration 1's sampling distribution, matching the fixed-var_scale log-P computed by hand
    for iteration 1 elsewhere in this project). Only the first solve() call's snapshots are
    kept (`done`), since the diagnostic wants one representative solve per task, not every
    replan of a full rollout."""

    def __init__(self, steps):
        self.steps = set(steps)
        self.parts = {s: [] for s in steps}   # step -> list of per-internal-batch (mean, var)
        self.snapshots = {}   # step -> concatenated (mean, var) cpu tensors, (B, horizon, action_dim)
        self.done = False
        self.history = []     # unused; CEMSolver.solve expects every callback to have one

    @property
    def output_key(self):
        return "MeanVarRecorder"

    def reset(self):
        pass

    def start_batch(self):
        pass

    def end_solve(self):
        # CEMSolver processes envs in internal batches of solver.batch_size < total_envs, each
        # its own start_batch()/step-loop/no separate end_solve(); this fires once, after every
        # internal batch of THIS solve() call has appended its slice for every requested step.
        if self.done or not all(self.parts[s] for s in self.steps):
            return
        for s in self.steps:
            means = torch.cat([p[0] for p in self.parts[s]], dim=0)
            vars_ = torch.cat([p[1] for p in self.parts[s]], dim=0)
            self.snapshots[s] = (means, vars_)
        self.done = True

    def __call__(self, **state):
        if self.done:
            return
        step = state["step"]
        if step in self.steps:
            self.parts[step].append((state["prev_mean"].detach().cpu().clone(),
                                     state["prev_var"].detach().cpu().clone()))


def attach_solve_hook(solver, model, oracle, retriever, clock, verify=False, logp_steps=None):
    """Wrap solver.solve: advance the clock, and (retrieval) encode the current frame + goal,
    pick the Alg.-1 node, and pass the retrieved logged block as init_action.

    `verify`: before trusting the retrieved block, score it against a zero-action candidate
    with the SAME model.get_cost the real CEM optimization uses (predictor rollout + this
    run's own configured criterion, critic included when active) and keep whichever of the
    two scores lower per env, falling back to CEM's own zero-mean default when retrieval
    loses. Motivated by the diagnostic in docs/gas-mpc/main.tex (Reacher's retrieved actions
    place CEM's iteration-1 search FURTHER from the true action than zero-init, on every held-
    out same25 task) -- this checks the same thing online, per live query, with no ground
    truth required, since it only compares the two candidates' own predicted cost.

    `logp_steps`: 0-indexed CEM iteration numbers (e.g. {0,4,9,19} for iterations 1,5,10,20)
    to snapshot the population (mean, var) at, via a MeanVarRecorder attached to
    solver.callbacks -- for the log-P-vs-iteration diagnostic (does CEM's own refinement close
    an initial-placement gap, or does an initial advantage/disadvantage persist?). Recorded
    from the first solve() call only; read back via solver.logp_recorder.snapshots."""
    orig = solver.solve
    stats = dict(n=0, n_kept=0)
    if logp_steps is not None:
        solver.logp_recorder = MeanVarRecorder(logp_steps)
        solver.callbacks.append(solver.logp_recorder)

    def solve(info_dict, init_action=None):
        if retriever is not None:
            with torch.no_grad():
                dev = next(model.parameters()).device
                pix = info_dict["pixels"].to(dev)                              # (B, T, C, H, W)
                z_cur = model.encode({"pixels": pix})["emb"][:, -1]             # (B, D)
                zg = model.encode({"pixels": info_dict["goal"].to(dev)})["emb"][:, -1]
                hc, hg = oracle.psi(z_cur), oracle.psi(zg)
                inits = []
                for b in range(len(hc)):
                    gi = oracle.goal_info(hg[b].cpu().numpy())
                    v, _ = oracle.select(hc[b].cpu().numpy(), gi)
                    target = hg[b] if v < 0 else oracle.C[v]
                    inits.append(retriever.init_action(hc[b], target))
                init_action = torch.stack(inits).to(device=solver.device, dtype=solver.dtype)

                if verify:
                    zero = torch.zeros_like(init_action)
                    candidates = torch.stack([init_action, zero], dim=1)        # (B, 2, horizon, blk*act)
                    expanded = _expand_for_cost(info_dict, 2, solver.device, solver.dtype)
                    costs = model.get_cost(expanded, candidates)                # (B, 2), lower is better
                    keep = costs[:, 0] <= costs[:, 1]
                    stats["n"] += len(keep)
                    stats["n_kept"] += int(keep.sum())
                    init_action = torch.where(keep[:, None, None].to(init_action.device), init_action, zero)
        out = orig(info_dict, init_action=init_action)
        if clock is not None:
            clock.n_calls += 1
        return out

    solver.solve = solve
    if verify:
        solver.retrieval_verify_stats = stats


# ----------------------------------------------------------------------------------------
# criterion factory
# ----------------------------------------------------------------------------------------

def make_criterion(method, oracle, support_lambda, horizon_blocks, l2_fn, critic=None, critic_beta=0.0,
                   clock=None, eps=1e-3, final_metric="same", critic_cost="nlv", compose="std",
                   critic_filter=0.0, steps=None, skip=5, critic_final=True):
    """Returns criterion(info_dict) -> (B, S) cost. `oracle` may be None for l2 / tdr.
    `steps`: StepsConverter, required for compose=steps."""
    shape_logged = [False]
    if compose == "steps":
        assert steps is not None, "compose=steps needs the gap calibration"
        assert critic is None or critic_beta == 0 or critic_cost in ("et", "etfull"), \
            "compose=steps needs a critic term in env steps (critic_cost=et|etfull)"
        assert method in ("l2", "tdr", "ctg", "subgoal", "subgoal_tdr"), f"compose=steps: {method} is not a distance"
    h_grid = list(range(0, int(critic.h_max) + 1, skip)) if critic is not None else None
    filt_stats = dict(n=0, feasible=0, none=0, envs=0)

    def split(info):
        pe = info["predicted_emb"]                            # (B, S, H_hist + n_pred, D)
        n_hist = info["emb"].shape[2]
        n_pred = pe.shape[2] - n_hist
        if not shape_logged[0]:
            log(f"[criterion] predicted_emb {tuple(pe.shape)} emb {tuple(info['emb'].shape)} "
                f"goal_emb {tuple(info['goal_emb'].shape)} -> {n_pred} predicted block latents")
            shape_logged[0] = True
        preds = pe[:, :, -n_pred:, :]                         # (B, S, n_pred, D)
        z_cur = info["emb"][:, 0, -1, :]                      # (B, D)
        z_goal = info["goal_emb"][..., -1, :]                 # (B, T_goal, D) -> (B, D); the
        if z_goal.ndim == 3:                                  # solver's batch_size=1 is what lets
            z_goal = z_goal[:, 0]                             # jepa's own expand_as work
        if oracle is not None:
            oracle.n_way = n_pred
        return preds, z_cur, z_goal

    def support(hT):
        return torch.cdist(hT.reshape(-1, hT.shape[-1]), oracle.C).min(dim=1).values.reshape(hT.shape[:-1])

    def v_at(zT, z_goal, h):
        """V(z_T, z_goal, h) for one scalar h -> (B, S)."""
        hh = torch.full(zT.shape[:-1], float(h), device=zT.device)
        v, _ = critic.prob(zT.reshape(-1, zT.shape[-1]), z_goal[:, None].expand_as(zT).reshape(-1, zT.shape[-1]),
                           hh.reshape(-1))
        return v.reshape(zT.shape[:-1])

    def critic_term(preds, z_goal):
        zT = preds[:, :, -1, :]
        if critic_cost == "nlv":
            return -torch.log(v_at(zT, z_goal, clock.h).clamp_min(eps))
        if hasattr(critic, "restricted_expected_steps"):
            h = critic.h_max if critic_cost == "etfull" else clock.h
            flat = zT.reshape(-1, zT.shape[-1])
            goals = z_goal[:, None].expand_as(zT).reshape(-1, zT.shape[-1])
            return critic.restricted_expected_steps(flat, goals, h).reshape(zT.shape[:-1])
        hs = h_grid if critic_cost == "etfull" else [h for h in h_grid if h <= clock.h]
        et = torch.zeros(zT.shape[:-1], device=zT.device)
        for h in hs:
            et += 1.0 - v_at(zT, z_goal, h)
        return skip * et

    def to_steps(base, is_l2):
        return torch.where(is_l2[:, None], steps(base, "l2"), steps(base, "tdr"))

    def criterion(info):
        base, is_l2 = base_cost(info)
        cost = base
        if critic is not None and critic_beta > 0:
            preds, _, z_goal = split(info)
            c = critic_term(preds, z_goal)
            if compose == "std":
                cost = robust_std(base) + critic_beta * robust_std(c)
                no_crit = robust_std(base)
            else:
                cost = to_steps(base, is_l2) + critic_beta * c
                no_crit = to_steps(base, is_l2)
            if not critic_final and method not in ("l2", "tdr"):
                # is_l2 marks the envs whose base is the final-phase z-space L2 (method=subgoal has
                # it always; subgoal_tdr/dir only after the final switch with final_metric=l2)
                cost = torch.where(is_l2[:, None], no_crit, cost)
        elif compose == "steps":
            cost = to_steps(base, is_l2)
        if critic is not None and critic_filter > 0:
            preds, _, z_goal = split(info)
            v = v_at(preds[:, :, -1, :], z_goal, clock.h)
            feasible = v >= critic_filter
            worst = cost.max(dim=1, keepdim=True).values
            cost = torch.where(feasible, cost, worst + 1.0 + (1.0 - v))
            filt_stats["n"] += feasible.numel(); filt_stats["feasible"] += int(feasible.sum())
            filt_stats["none"] += int((feasible.sum(dim=1) == 0).sum()); filt_stats["envs"] += feasible.shape[0]
        return cost

    def base_cost(info):
        """-> (cost (B, S), is_l2 (B,) bool: which envs' cost is a z-space squared L2 rather than a
        TDR-unit distance)."""
        if method == "l2":
            c = l2_fn(info)
            return c, torch.ones(c.shape[0], dtype=torch.bool, device=c.device)
        preds, z_cur, z_goal = split(info)
        is_l2 = torch.zeros(preds.shape[0], dtype=torch.bool, device=preds.device)
        if method == "subgoal":
            is_l2[:] = True
        B, S, P, D = preds.shape
        hT = oracle.psi(preds[:, :, -1, :])                   # (B, S, dim)
        hc = oracle.psi(z_cur)                                # (B, dim)
        hg = oracle.psi(z_goal)                               # (B, dim)
        if method == "tdr":
            cost = torch.linalg.norm(hT - hg[:, None], dim=-1)
        elif method == "ctg":
            cost = torch.empty(B, S, device=preds.device)
            for b in range(B):
                gi = oracle.goal_info(hg[b].cpu().numpy())
                dn = torch.cdist(hT[b], oracle.C) + gi["dist_t"][None]         # (S, V)
                direct = torch.linalg.norm(hT[b] - hg[b][None], dim=-1)       # (S,)
                cost[b] = torch.minimum(direct, dn.min(dim=1).values)
        elif method in ("subgoal", "subgoal_tdr", "dir"):
            cost = torch.empty(B, S, device=preds.device)
            for b in range(B):
                gi = oracle.goal_info(hg[b].cpu().numpy())
                v, flags = oracle.select(hc[b].cpu().numpy(), gi)
                if v < 0 and final_metric == "l2":          # final phase: LeWM's own z-space cost
                    cost[b] = ((preds[b, :, -1, :] - z_goal[b][None]) ** 2).sum(-1)
                    is_l2[b] = True
                elif method == "subgoal":
                    tgt = z_goal[b] if v < 0 else oracle.node_z[v]
                    cost[b] = ((preds[b, :, -1, :] - tgt[None]) ** 2).sum(-1)
                elif method == "subgoal_tdr":
                    tgt = hg[b] if v < 0 else oracle.C[v]
                    cost[b] = torch.linalg.norm(hT[b] - tgt[None], dim=-1)
                else:  # dir
                    if v < 0:
                        cost[b] = torch.linalg.norm(hT[b] - hg[b][None], dim=-1)
                    else:
                        h_dir = oracle.C[v] - hc[b]
                        h_dir = h_dir / h_dir.norm().clamp_min(1e-6)
                        cost[b] = -((hT[b] - hc[b][None]) * h_dir[None]).sum(-1)
        elif method == "path":
            hk = oracle.psi(preds.reshape(B * S * P, D)).reshape(B, S, P, -1)
            cost = torch.zeros(B, S, device=preds.device)
            for b in range(B):
                gi = oracle.goal_info(hg[b].cpu().numpy())
                W = torch.from_numpy(oracle.waypoints(hc[b].cpu().numpy(), gi)).to(preds.device)  # (P, dim)
                cost[b] = torch.linalg.norm(hk[b] - W[None], dim=-1).sum(-1)
        else:
            raise ValueError(method)
        if support_lambda > 0:
            cost = cost + support_lambda * support(hT)
        return cost, is_l2

    criterion.stats = filt_stats
    return criterion


# ----------------------------------------------------------------------------------------

def method_tag(m):
    tag = m.method
    if m.method in ("subgoal", "subgoal_tdr", "dir"):
        la = m.lookahead if str(m.lookahead) == "auto" else f"{float(m.lookahead):g}"
        tag += f"_la{la}_th{m.subgoal_threshold:g}"
    if m.method == "path":
        tag += f"_su{m.step_units if m.step_units is not None else 'gap5'}_th{m.subgoal_threshold:g}"
    if m.method not in ("l2", "tdr", "random"):
        tag += f"_htd{m.h_td:g}_te{m.te:g}"
    if m.support_lambda > 0:
        tag += f"_sup{m.support_lambda:g}"
    if m.receding != 5:
        tag += f"_rh{m.receding}"
    if m.retrieval:
        tag += "_ret"
    if m.retrieval_verify:
        tag += "_rv"
    if m.final_thresh is not None and (str(m.final_thresh) == "auto" or float(m.final_thresh) != float(m.subgoal_threshold)):
        tag += "_ftauto" if str(m.final_thresh) == "auto" else f"_ft{float(m.final_thresh):g}"
    if m.final_metric != "same":
        tag += f"_f{m.final_metric}"
    if m.critic_beta > 0:
        tag += f"_crit{m.critic_beta:g}"
    if (m.critic_beta > 0 or m.critic_filter > 0) and m.critic_cost != "nlv":
        tag += f"_cc{m.critic_cost}"
    if m.compose != "std":
        tag += f"_{m.compose}"
    if m.critic_filter > 0:
        tag += f"_flt{float(m.critic_filter):g}"
    if m.critic_beta > 0 and not bool(m.critic_final):
        tag += "_nocf"
    if int(m.critic_member) >= 0:
        tag += f"_cm{int(m.critic_member)}"
    if m.get("budget") is not None:
        tag += f"_bud{int(m.budget)}"
    if os.environ.get("GAS_MPC_TDR_TAG"):
        tag += os.environ["GAS_MPC_TDR_TAG"]
    return tag + (m.tag or "")


@hydra.main(version_base=None, config_path=str(ROOT / "config" / "eval"), config_name=MECH.eval_config_name)
def main(cfg: DictConfig):
    t0 = time.time()
    default_critic = OUT / "critic_s0_tdr_holdout" / "critic.pt"
    m = OmegaConf.create(dict(method="l2", protocol="cross", seed=0, chunk=25, h_td=8.0, te=0.9,
                              graph_seed=0, subgoal_threshold=None, lookahead=0.0, step_units=None,
                              support_lambda=0.0, receding=5, tag="", force=False, retrieval=False,
                              retrieval_verify=False, logp_steps="", critic_beta=0.0, critic=str(default_critic),
                              final_thresh=None, final_metric="same", critic_cost="nlv",
                              compose="std", critic_filter=0.0, budget=None, critic_final=True,
                              critic_member=-1))
    m = OmegaConf.merge(m, cfg.get("mpc", {}))
    if m.subgoal_threshold is None:
        m.subgoal_threshold = m.h_td
    if m.method == "random":
        assert not m.retrieval and m.critic_beta == 0 and m.critic_filter == 0 \
            and m.support_lambda == 0 and m.compose == "std", \
            "mpc.method=random takes no action from a model -- planning add-ons don't apply"
    proto = PROTOCOLS[m.protocol]
    if m.budget is not None:
        proto = dict(proto, budget=int(m.budget))
    n = int(cfg.eval.num_eval)
    seed = int(m.seed)
    cfg.plan_config.receding_horizon = int(m.receding)
    mech = MECH
    # lookahead / final_thresh "auto" = the TDR's own calibrated 25-step distance (one CEM
    # horizon), so the same setting transfers across environments and TDR seeds
    auto_keys = [k for k in ("lookahead", "final_thresh") if str(m.get(k)) == "auto"]
    if auto_keys and m.method in ("l2", "tdr"):
        raise ValueError(f"{auto_keys} = auto needs a graph method")
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    tag = method_tag(m)
    if "tasks" in cfg.get("mpc", {}):
        raise ValueError("mpc.tasks is no longer supported; evaluations always use the fixed 200-task pool")
    assert n <= TASKS, f"eval.num_eval={n} exceeds the fixed task set size {TASKS}"
    pool_suffix = ("__heldout_disjoint" if POOL.endswith("u") else "") + "__trainonly_assets"
    # chunk changes both the CEM seed a task gets (cfg.seed below is keyed off each chunk's
    # start index) and which cached per-chunk files are valid to reuse, so a non-default
    # chunk must not share a run_id with the historical chunk=25 runs. Suffix omitted at the
    # default so every existing archived result/cache filename is unaffected.
    chunk_tag = "" if int(m.chunk) == 25 else f"__ch{int(m.chunk)}"
    run_id = f"{tag}__{m.protocol}__{POOL}__s{seed}__n{n}{pool_suffix}{chunk_tag}"
    final_path = EVAL_DIR / f"{run_id}.json"
    if final_path.exists() and not m.force:
        r = json.loads(final_path.read_text())
        log(f"[done] {final_path.name} exists: success={r['success_rate']:.1f}% -- skipping")
        return
    log(f"=== gas_mpc_eval method={tag} protocol={m.protocol} pool={POOL} seed={seed} n={n} "
        f"chunk={m.chunk} budget={proto['budget']} plan={OmegaConf.to_container(cfg.plan_config)} ===")

    # ---- dataset / pairs (shared with every other evaluation) ----
    h5_path = str(mech.h5_path(ROOT))
    dataset = swm.data.HDF5Dataset(path=h5_path, keys_to_cache=list(cfg.dataset.keys_to_cache))
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    ep_col, step_col = dataset.get_col_data(col_name), dataset.get_col_data("step_idx")
    state_col = mech.state_column(dataset)
    pair_tag = "cross_episode" if proto["pairing"] == "cross_episode" else f"same_episode_off{proto['offset']}"
    pairs_path = PAIRS_DIR / f"pairs_{pair_tag}_{POOL}.json"
    if not POOL.endswith("u") and not pairs_path.exists():
        raise FileNotFoundError(f"Archived task pool is missing: {pairs_path}")
    pairs = load_or_make_pairs(pairs_path, ep_col=ep_col, step_col=step_col, state_col=state_col, n=TASKS,
                               seed=TASK_SEED, pairing=proto["pairing"], offset=proto["offset"] or 25,
                               reject=mech.goal_reached,
                               heldout_eps=task_heldout_episodes(ep_col, int(m.graph_seed)))
    pairs = {k: (v[:n] if k.startswith(("start_", "goal_")) and isinstance(v, list) else v)
             for k, v in pairs.items()}
    pairs["n"] = n

    process = {}
    # Reuse the pretrained action interface; fit other transforms on training episodes only.
    training_rows = ~np.isin(ep_col, pairs["heldout_eps"])
    with np.load(OUT / "cache_train.npz") as stats:
        action_mean, action_std = stats["act_mean"], stats["act_std"]
    for col in cfg.dataset.keys_to_cache:
        if col == "pixels":
            continue
        processor = preprocessing.StandardScaler()
        if col == "action":
            processor.mean_ = action_mean
            processor.scale_ = np.where(action_std > 0, action_std, 1.0)
            processor.var_ = action_std ** 2
            processor.n_features_in_ = len(action_mean)
            processor.n_samples_seen_ = int(training_rows.sum())
        else:
            col_data = dataset.get_col_data(col)[training_rows]
            col_data = col_data[~np.isnan(col_data).any(axis=1)]
            processor.fit(col_data)
        process[col] = processor
        if col != "action":
            process[f"goal_{col}"] = process[col]
    transform = {"pixels": img_transform(cfg), "goal": img_transform(cfg)}
    callables = OmegaConf.to_container(cfg.eval.get("callables"), resolve=True)

    # ---- graph assets ----
    oracle, tdr_hist, retriever, critic, g = None, None, None, None, None
    if m.method not in ("l2", "random") or m.retrieval:
        tdr, ck = load_tdr(int(m.graph_seed))
        tdr_hist = ck["history"][-1]
        if auto_keys:
            d25 = float(tdr_hist["median_by_gap"].get(25, tdr_hist["median_by_gap"].get("25")))
            for k in auto_keys:
                m[k] = round(d25, 2)
            log(f"[tdr] {auto_keys} = auto -> {round(d25, 2)} TDR units (calibrated 25-step distance)")
            tag = method_tag(m)
            run_id = f"{tag}__{m.protocol}__{POOL}__s{seed}__n{n}{pool_suffix}{chunk_tag}"
            final_path = EVAL_DIR / f"{run_id}.json"
            if final_path.exists() and not m.force:
                r = json.loads(final_path.read_text())
                log(f"[done] {final_path.name} exists: success={r['success_rate']:.1f}% -- skipping")
                return
        step_units = m.step_units
        if step_units is None:
            step_units = tdr_hist["median_by_gap"].get(5, tdr_hist["median_by_gap"].get("5"))
        if m.method != "tdr" or m.retrieval:
            with open(graph_path(int(m.graph_seed), m.h_td, m.te), "rb") as fh:
                g = pickle.load(fh)
            if not g.get("training_only"):
                raise ValueError("Graph lacks train-only preparation provenance; rebuild it")
            if np.isin(ep_col[g["kept_rows"]], pairs["heldout_eps"]).any():
                raise ValueError("Evaluation tasks overlap graph-building frames; rebuild "
                                 "the graph excluding the TDR held-out episodes")
            log(f"[graph] {g['stats']}")
        else:
            g = dict(centers=np.zeros((1, tdr.tdr_dim), np.float32), node_z=np.zeros((1, 192), np.float32),
                     graph=csr_matrix((1, 1)))
        oracle = GraphOracle(g, tdr, m.h_td, m.subgoal_threshold, m.lookahead, step_units,
                             n_waypoints=int(cfg.plan_config.horizon) - 3 + 1, final_thresh=m.final_thresh)
        log(f"[tdr] {tdr_hist}; step_units={step_units}")
    if m.retrieval:
        from gas_mpc_prepare import ensure_psi, load_cache
        cache = load_cache()
        psi_all = ensure_psi(int(m.graph_seed))
        retriever = Retriever(psi_all, cache["action"], cache["ep_offset"], cache["ep_len"], process["action"],
                              int(cfg.plan_config.horizon), int(cfg.plan_config.action_block))
        log(f"[retrieval] {len(retriever.valid)} candidate rows, block of {retriever.L} env steps")
        del cache
    if m.critic_beta > 0 or m.critic_filter > 0:
        from viability_eval_audit import load_critic
        provenance = torch.load(str(m.critic), map_location="cpu", weights_only=False)
        if not provenance["args"].get("training_only"):
            raise ValueError("Critic was not trained from a physically training-only cache")
        used = np.r_[provenance["train_episodes"], provenance["val_episodes"]]
        if np.intersect1d(used, pairs["heldout_eps"]).size:
            raise ValueError("Evaluation episodes overlap critic training/validation")
        del provenance
        critic, cargs = load_critic(str(m.critic))
        member = int(m.critic_member)
        if member >= 0:
            if not hasattr(critic, "members"):
                raise ValueError("mpc.critic_member requires an ensemble critic")
            if member >= len(critic.members):
                raise ValueError(f"critic member {member} is out of range for {len(critic.members)} members")
            critic.members = torch.nn.ModuleList([critic.members[member]])
            critic.n_members = 1
        log(f"[critic] {m.critic} h_max={cargs['h_max']} cost={m.critic_cost} compose={m.compose} "
            f"filter={m.critic_filter} member={'mean' if member < 0 else member}")
    steps = None
    if m.compose == "steps":
        calib = gap_calibration(int(m.graph_seed))
        steps = StepsConverter(calib, DEV)
        log(f"[calib] gaps={calib['gaps']} tdr={[round(v, 2) for v in calib['tdr']]} "
            f"l2={[round(v, 1) for v in calib['l2']]}")

    # ---- rollouts, in resumable chunks ----
    budget = int(proto["budget"])
    chunk = int(m.chunk)
    chunks = [(c, list(range(c, min(c + chunk, n)))) for c in range(0, n, chunk)]
    results = {}
    for ci, idx in chunks:
        cpath = EVAL_DIR / f"{run_id}__c{ci}.json"
        if cpath.exists():
            results[ci] = json.loads(cpath.read_text())
            if results[ci]["pair_idx"] != idx:
                raise ValueError(f"{cpath} covers pair_idx={results[ci]['pair_idx']}, expected {idx} "
                                  "-- stale cache from a different chunk size?")
            log(f"[chunk {ci}] cached: success={results[ci]['success_rate']:.1f}%")
            continue
        sub = {k: ([v[i] for i in idx] if isinstance(v, list) else v) for k, v in pairs.items()}
        sub["n"] = len(idx)
        cfg.seed = seed * 1000 + ci
        # Defense-in-depth for the reproducibility gap found 2026-09-21 (identical
        # +mpc.seed, identical task set: success_rate matched across 3 parallel reps but
        # 20-24% of individual tasks' final_pos_err differed by tens to ~240px). CEM's own
        # candidate sampling was already deterministically seeded (a dedicated
        # torch.Generator, reseeded per chunk above); this closes off any other library
        # silently consulting the *global* RNG state during env reset/step/render.
        random.seed(int(cfg.seed))
        np.random.seed(int(cfg.seed) % (2**32))
        if m.method == "random":
            # No model, no CEM: a fresh uniform action from the env's own action space every
            # env step (stable_worldmodel's RandomPolicy -> env.action_space.sample()).
            policy = swm.policy.RandomPolicy(seed=int(cfg.seed))
            world = swm.World(env_name=cfg.world.env_name, num_envs=len(idx), max_episode_steps=2 * budget,
                              image_shape=(224, 224), **mech.world_kwargs)
            world.set_policy(policy)
            t_run = time.time()
            first_hit, traj = evaluate_pairs(world, dataset, sub, budget, callables, mech=mech,
                                             seed=int(cfg.seed))
            dt = time.time() - t_run
            crit_stats, verify_stats = None, None
            world.close()
            del policy, world
        else:
            model = load_lewm(ckpt_dir=mech.ckpt_dir(ROOT), device="cuda")
            model.interpolate_pos_encoding = True
            clock = PlanClock(budget, int(cfg.plan_config.receding_horizon) * int(cfg.plan_config.action_block),
                              int(cfg.plan_config.horizon) * int(cfg.plan_config.action_block),
                              cargs["h_max"] if critic is not None else 0)
            if m.method != "l2" or critic is not None or m.compose != "std":
                l2_fn = model.criterion
                model.criterion = make_criterion(m.method, oracle, float(m.support_lambda),
                                                 int(cfg.plan_config.horizon), l2_fn, critic=critic,
                                                 critic_beta=float(m.critic_beta), clock=clock,
                                                 final_metric=str(m.final_metric), critic_cost=str(m.critic_cost),
                                                 compose=str(m.compose), critic_filter=float(m.critic_filter),
                                                 steps=steps, critic_final=bool(m.critic_final))
            config = swm.PlanConfig(**cfg.plan_config)
            logp_steps = [int(s) for s in str(m.logp_steps).split(",") if s.strip() != ""]
            solver = hydra.utils.instantiate(cfg.solver, model=model)
            if retriever is not None or critic is not None:
                attach_solve_hook(solver, model, oracle, retriever, clock, verify=bool(m.retrieval_verify),
                                  logp_steps=(logp_steps or None))
            policy = swm.policy.WorldModelPolicy(solver=solver, config=config, process=process, transform=transform)
            world = swm.World(env_name=cfg.world.env_name, num_envs=len(idx), max_episode_steps=2 * budget,
                              image_shape=(224, 224), **mech.world_kwargs)
            world.set_policy(policy)
            t_run = time.time()
            first_hit, traj = evaluate_pairs(world, dataset, sub, budget, callables, mech=mech,
                                             seed=int(cfg.seed))
            dt = time.time() - t_run
            crit_stats = dict(model.criterion.stats) if hasattr(model.criterion, "stats") else None
            verify_stats = dict(solver.retrieval_verify_stats) if hasattr(solver, "retrieval_verify_stats") else None
            if hasattr(solver, "logp_recorder") and solver.logp_recorder.snapshots:
                snap = solver.logp_recorder.snapshots
                np.savez_compressed(EVAL_DIR / f"{run_id}__c{ci}_logp.npz",
                                    steps=np.array(sorted(snap)),
                                    **{f"mean_{k}": v[0].numpy() for k, v in snap.items()},
                                    **{f"var_{k}": v[1].numpy() for k, v in snap.items()})
                log(f"[logp] wrote {run_id}__c{ci}_logp.npz steps={sorted(snap)}")
            world.close()
            del model, solver, policy, world
        torch.cuda.empty_cache()
        pos, ang = mech.goal_errors(traj, sub["goal_state"])
        pos0, ang0 = mech.goal_errors(np.array(sub["start_state"])[:, None], sub["goal_state"])
        rec = dict(pair_idx=idx, first_hit_step=first_hit.tolist(),
                   success_rate=float(np.mean(first_hit >= 0) * 100), secs=dt,
                   initial_pos_err=pos0[:, 0].tolist(), initial_ang_err=ang0[:, 0].tolist(),
                   min_pos_err=pos.min(1).tolist(), final_pos_err=pos[:, -1].tolist(),
                   final_ang_err=ang[:, -1].tolist(),
                   planner=oracle.summary() if oracle is not None else None,
                   retrieval=(dict(retriever.summary(), verify=verify_stats) if retriever is not None else None),
                   critic_filter=crit_stats)
        np.savez_compressed(EVAL_DIR / f"{run_id}__c{ci}_traj.npz",
                            traj=traj.astype(np.float32), first_hit=first_hit,
                            goal_state=np.array(sub["goal_state"], dtype=np.float32))
        cpath.write_text(json.dumps(rec, indent=1))
        results[ci] = rec
        log(f"[chunk {ci}] success={rec['success_rate']:.1f}% ({dt:.0f}s) first-hit: "
            f"{sorted(first_hit[first_hit >= 0].tolist())} planner={rec['planner']}")

    # ---- aggregate ----
    fh = np.concatenate([np.array(results[ci]["first_hit_step"]) for ci, _ in chunks])
    success_by_budget = {b: float(np.mean((fh >= 0) & (fh <= b)) * 100) for b in range(25, budget + 1, 25)}
    out = dict(method=tag, env=ENV, smoke_test=SMOKE, mpc=OmegaConf.to_container(m), protocol=m.protocol, pairing=proto["pairing"],
               offset=proto["offset"], budget=budget, n=n, seed=seed,
               pairs_file=str(pairs_path), tasks=TASKS, pool=POOL,
               success_rate=float(np.mean(fh >= 0) * 100), n_success=int((fh >= 0).sum()),
               success_by_budget=success_by_budget, first_hit_step=fh.tolist(),
               min_pos_err=sum((results[ci]["min_pos_err"] for ci, _ in chunks), []),
               final_pos_err=sum((results[ci]["final_pos_err"] for ci, _ in chunks), []),
               final_ang_err=sum((results[ci]["final_ang_err"] for ci, _ in chunks), []),
               planner=[results[ci]["planner"] for ci, _ in chunks], tdr_eval=tdr_hist,
               graph_stats=(g["stats"] if (g is not None and "stats" in g) else None),
               retrieval=[results[ci].get("retrieval") for ci, _ in chunks],
               solver=OmegaConf.to_container(cfg.solver), plan_config=OmegaConf.to_container(cfg.plan_config),
               secs=sum(results[ci]["secs"] for ci, _ in chunks), wall=time.time() - t0)
    final_path.write_text(json.dumps(out, indent=1))
    log(f"=== {tag} / {m.protocol} / s{seed}: success={out['success_rate']:.1f}% ({out['n_success']}/{n}) "
        f"by budget {success_by_budget}; min pos err median {np.median(out['min_pos_err']):.0f}px; "
        f"wrote {final_path.name} ({time.time() - t0:.0f}s) ===")


if __name__ == "__main__":
    main()
