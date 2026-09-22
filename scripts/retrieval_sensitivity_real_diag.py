"""Ground-truth version of retrieval_sensitivity_diag.py: instead of asking the FROZEN
PREDICTOR to imagine the outcome of running a training-bank node's action block from a
held-out query's latent, actually RUN those actions in the real simulator starting from the
query's real physical state, then re-encode the real resulting frame. No predictor involved
anywhere in this script.

For each (query, node) pair (node candidates span a wide range of initial latent distance
from the query, exactly as in the predictor version):
  d0_latent  = ||z_node_start - z_query_start||           (real encoded latents, as before)
  d0_state   = ||state_node_start - state_query_start||    (raw physical state, mech units)
  -- run the node's real logged 25-step RAW action block in the live simulator, starting
     from the query's real physical state (callables restore exact state; the "goal" passed
     to callables is the node's own real end state/frame, purely so termination -- if the env
     reaches it early -- freezes the trajectory there rather than corrupting it, matching real
     deployment; it plays no other role) --
  z_final_real    = frozen encoder applied to the REAL final frame after those 25 real steps
  state_final_real = the REAL final physical state after those 25 real steps
  d_final_latent = ||z_node_end - z_final_real||            (z_node_end = node's own real,
                                                              logged continuation)
  d_final_state  = ||state_node_end - state_final_real||    (state_node_end = node's own real,
                                                              logged continuation)

Run per env via GAS_MPC_ENV / hydra config_name. Needs the live env (MUJOCO_GL=egl for
cube/reacher; pusht is pure pymunk, no GL needed but harmless to set).
"""
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import sys
import json
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
import stable_worldmodel as swm

sys.path.insert(0, "scripts")
from common.lewm_loader import load_lewm
from planning_cost_gate import make_encode_frame
from gas_mpc_prepare import OUT, ENV, MECH, load_cache
from viability_cross_episode_baseline import extract_rows, evaluate_pairs

def state_dist(env, a, b):
    """Position-only distance in the env's own success-predicate space (matches
    docs/gas-mpc/main.tex tab:two-axes conventions), NOT a naive full-state-vector L2 --
    Push-T's/Reacher's raw state vectors mix position with other large-magnitude fields
    (angle needs wrap, Push-T also carries velocity-scale entries in dims 5:7 that would
    otherwise dominate) that have nothing to do with where the object/agent physically is."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if env == "pusht":
        return np.linalg.norm(a[..., :4] - b[..., :4], axis=-1)              # agent+block xy, px
    if env == "cube":
        return np.linalg.norm(a[..., :3] - b[..., :3], axis=-1)              # block xyz, m
    if env == "reacher":
        d0 = np.abs(a[..., 0] - b[..., 0])
        d0 = np.minimum(d0, 2 * np.pi - d0)                                   # shoulder wraps
        d1 = np.abs(a[..., 1] - b[..., 1])                                    # wrist, range-limited
        return np.maximum(d0, d1)                                             # rad, matches QPOS_THRESHOLD
    raise ValueError(env)


DEV = "cuda" if torch.cuda.is_available() else "cpu"
N_QUERIES = int(os.environ.get("DIAG_N_QUERIES", "50"))
PERCENTILES = [0, 5, 10, 25, 50, 75, 100]
BATCH = int(os.environ.get("DIAG_BATCH", "50"))
BUDGET = 25

print(f"[real-diag] env={ENV} N_QUERIES={N_QUERIES} percentiles={PERCENTILES} batch={BATCH}")

ROOT = Path("/home2/mayaank.ashok/lewm_research")


class ReplayPolicy:
    """Feeds a fixed, pre-recorded RAW action per env per call. `actions`: (n_envs, T, act_dim)."""

    def __init__(self, actions):
        self.actions = np.asarray(actions, dtype=np.float32)
        self.t = 0

    def set_env(self, envs):
        pass

    def get_action(self, infos):
        a = self.actions[:, self.t]
        self.t += 1
        return a


@hydra.main(version_base=None, config_path=str(ROOT / "config" / "eval"), config_name=ENV)
def main(cfg: DictConfig):
    h5_path, ckpt_dir = MECH.h5_path(ROOT), MECH.ckpt_dir(ROOT)
    model = load_lewm(Path(ckpt_dir), device=DEV)
    encode_frame = make_encode_frame(model)

    cache = load_cache()
    z_bank, state_bank, act_bank = cache["z"], cache["state"], cache["action"]
    ep_offset, ep_len, episode_id = cache["ep_offset"], cache["ep_len"], cache["episode_id"]
    Nc = len(z_bank)
    ep_start_of = np.empty(Nc, dtype=np.int64)
    ep_end_of = np.empty(Nc, dtype=np.int64)
    ep_k_of = np.empty(Nc, dtype=np.int64)
    for k, (o, l) in enumerate(zip(ep_offset, ep_len)):
        ep_start_of[o:o + l] = o
        ep_end_of[o:o + l] = o + l - 1
        ep_k_of[o:o + l] = k
    L = BUDGET
    fwd_ok = (np.arange(Nc) + L) <= ep_end_of
    cand_idx_all = np.where(fwd_ok)[0]
    rng = np.random.default_rng(0)
    MAX_CANDIDATES = 200_000
    if len(cand_idx_all) > MAX_CANDIDATES:
        cand_idx_all = np.sort(rng.choice(cand_idx_all, size=MAX_CANDIDATES, replace=False))
    cand_z = torch.from_numpy(z_bank[cand_idx_all]).to(DEV)
    print(f"[real-diag] candidate bank: {len(cand_idx_all)}/{Nc} rows usable")

    # ---- held-out queries: same pool/filter/seed as the predictor version, for comparability ----
    pairs_all = json.load(open(f"{OUT}/pairs/pairs_same_episode_off25_task200u.json"))
    start_row = np.asarray(pairs_all["start_row"])
    start_step = np.asarray(pairs_all["start_step"])
    start_ep = np.asarray(pairs_all["start_ep"])
    start_state = np.asarray(pairs_all["start_state"], dtype=np.float64)
    keep = start_step >= 10
    idx_pool = np.where(keep)[0]
    if len(idx_pool) > N_QUERIES:
        idx_pool = rng.choice(idx_pool, size=N_QUERIES, replace=False)
    print(f"[real-diag] {len(idx_pool)} held-out queries (of {keep.sum()} eligible / {len(keep)} total)")

    import h5py
    import hdf5plugin  # noqa: F401
    f = h5py.File(h5_path, "r", swmr=True, rdcc_nbytes=512 * 1024 * 1024)
    q_pix = np.stack([f["pixels"][int(r)] for r in start_row[idx_pool]])
    f.close()
    z_query = encode_frame(q_pix)                                    # (Nq, D)

    # ---- for each query, pick candidates at fixed percentiles of its own sorted distance list ----
    rows = []   # each: (query slot, cand row, d0_latent)
    z_query_t = torch.from_numpy(z_query).to(DEV)
    for qi in range(len(idx_pool)):
        d = torch.linalg.norm(cand_z - z_query_t[qi], dim=-1)
        order = torch.argsort(d)
        n = len(order)
        picked = sorted({min(n - 1, int(round(p / 100 * (n - 1)))) for p in PERCENTILES})
        for k in picked:
            ci = int(order[k].item())
            rows.append((qi, int(cand_idx_all[ci]), float(d[order[k]].item())))
    print(f"[real-diag] {len(rows)} (query, candidate) pairs")

    # ---- resolve each candidate's END row (r_c + L) back to (orig_ep, step) for the "goal" callable ----
    def resolve(row):
        k = ep_k_of[row]
        return int(episode_id[k]), int(row - ep_offset[k])

    dataset = swm.data.HDF5Dataset(path=str(h5_path), keys_to_cache=list(cfg.dataset.keys_to_cache))
    callables = OmegaConf.to_container(cfg.eval.get("callables"), resolve=True)

    d0_latent, d0_state, d_final_latent, d_final_state = [], [], [], []
    n_batches = (len(rows) + BATCH - 1) // BATCH
    world = None
    for b in range(n_batches):
        chunk = rows[b * BATCH:(b + 1) * BATCH]
        n = len(chunk)
        q_slots = [qi for qi, _, _ in chunk]
        cand_rows = [cr for _, cr, _ in chunk]
        d0s = [d for _, _, d in chunk]

        s_ep = start_ep[idx_pool[q_slots]]
        s_step = start_step[idx_pool[q_slots]]
        s_state = start_state[idx_pool[q_slots]]
        end_rows = [cr + L for cr in cand_rows]
        g_ep_step = [resolve(r) for r in end_rows]
        g_ep = np.array([e for e, _ in g_ep_step])
        g_step = np.array([s for _, s in g_ep_step])
        g_state = state_bank[end_rows]
        node_start_state = state_bank[cand_rows]
        node_actions_raw = np.stack([act_bank[cr:cr + L] for cr in cand_rows])   # (n, L, act_dim)

        pairs = dict(start_row=start_row[idx_pool[q_slots]], start_ep=s_ep, start_step=s_step,
                     goal_ep=g_ep, goal_step=g_step,
                     start_state=s_state.tolist(), goal_state=g_state.tolist())

        if world is None or world.num_envs != n:
            if world is not None:
                world.close()
            world = swm.World(env_name=cfg.world.env_name, num_envs=n, max_episode_steps=2 * BUDGET,
                              image_shape=(224, 224), **MECH.world_kwargs)
        world.set_policy(ReplayPolicy(node_actions_raw))
        _, traj = evaluate_pairs(world, dataset, pairs, BUDGET, callables, mech=MECH, seed=int(b))
        final_state_real = traj[:, -1]                                            # (n, state_dim)
        final_pix_real = world.infos["pixels"]
        if final_pix_real.ndim == 5:            # (n, T_ctx, H, W, C) -- take the latest frame
            final_pix_real = final_pix_real[:, -1]
        z_final_real = encode_frame(final_pix_real)

        z_node_end = z_bank[end_rows]
        for i in range(n):
            d0_latent.append(d0s[i])
            d0_state.append(float(state_dist(ENV, s_state[i], node_start_state[i])))
            d_final_latent.append(float(np.linalg.norm(z_node_end[i] - z_final_real[i])))
            d_final_state.append(float(state_dist(ENV, g_state[i], final_state_real[i])))
        print(f"[real-diag] batch {b + 1}/{n_batches} done ({n} pairs)")

    d0_latent = np.array(d0_latent); d0_state = np.array(d0_state)
    d_final_latent = np.array(d_final_latent); d_final_state = np.array(d_final_state)

    from scipy import stats as sstats
    print(f"\n=== {ENV}: REAL SIMULATOR replay, {len(d0_latent)} pairs ===")
    for name, d0, df in (("latent", d0_latent, d_final_latent), ("state", d0_state, d_final_state)):
        mask = (d0 > 1e-8) & (df > 1e-8)
        s, i, r, p, se = sstats.linregress(d0[mask], df[mask])
        sl, il, rl, pl, _ = sstats.linregress(np.log(d0[mask]), np.log(df[mask]))
        print(f"[{name}] linear slope={s:.4f} intercept={i:.4f} r={r:.4f}  |  "
              f"log-log slope={sl:.4f} (>1 divergent, <1 convergent) r={rl:.4f}")
        edges = np.quantile(d0, np.linspace(0, 1, 8))
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (d0 >= lo) & (d0 <= hi)
            if m.sum() == 0:
                continue
            print(f"    d0 in [{lo:8.3f}, {hi:8.3f}]  median d0={np.median(d0[m]):8.3f}  "
                  f"median d_final={np.median(df[m]):8.3f}  ratio={np.median(df[m]) / max(np.median(d0[m]), 1e-9):6.3f}  n={int(m.sum())}")

    np.savez(f"{OUT}/retrieval_sensitivity_real_diag.npz", d0_latent=d0_latent, d0_state=d0_state,
             d_final_latent=d_final_latent, d_final_state=d_final_state)
    print(f"\n[real-diag] saved outputs/{ENV}/retrieval_sensitivity_real_diag.npz")


if __name__ == "__main__":
    main()
