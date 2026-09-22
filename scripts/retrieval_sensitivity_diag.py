"""Does retrieval's core assumption hold: "same action sequence from a nearby latent lands
near the same result"?

Retrieval warm-starts CEM with a training-bank row's logged action block, then runs it (via
the frozen predictor's imagined rollout, inside CEM) from the QUERY's actual current latent
z_cur -- not from the training row's own real latent z_node. The implicit assumption is that
because z_cur is close to z_node in latent space, applying z_node's actions from z_cur lands
close to where they landed from z_node itself (near the intended target). This script tests
that directly and asks whether the error is CONVERGENT (bounded / shrinking relative to the
initial offset) or DIVERGENT (grows with, or faster than, the initial offset), per env.

For each of many held-out same-ep-25 query windows (start_row, with >=10 steps of real
history already in its own episode) and, for each query, several TRAINING-bank candidate
("node") rows spanning a wide range of initial latent distance from the query (nearest, a
few percentiles, and far/near-random), compute:
  d0       = ||z_node_start - z_query_cur||                          (real latents, both sides)
  z_node_end   = the training row's OWN real continuation, cache["z"][r_c + 25]   (ground truth)
  z_actual_end = predictor-imagined rollout of the NODE's action block, run from the QUERY's
                 own real history (context preserved, only the future action block is swapped)
  d_final  = ||z_node_end - z_actual_end||

Also, for candidates that themselves have >=10 steps of preceding history, a d0~0 baseline:
imagine the SAME node action block from the NODE's OWN real history (self-consistency) and
compare to z_node_end -- this is the predictor's floor error with no latent offset at all,
so the d0-vs-d_final trend can be read as excess error ON TOP of that floor, not just
predictor noise in general.

Run per env via GAS_MPC_ENV. No GPU-heavy loop: all (query, candidate) pairs for one env are
batched into a single `imagine` call.
"""
import json
import sys
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401
import numpy as np
import torch
from sklearn import preprocessing

sys.path.insert(0, "scripts")
from common.lewm_loader import load_lewm
from common.viability import imagine, HISTORY
from planning_cost_gate import make_encode_frame
from gas_mpc_prepare import OUT, ENV, MECH, load_cache

DEV = "cuda" if torch.cuda.is_available() else "cpu"
N_QUERIES = int(__import__("os").environ.get("DIAG_N_QUERIES", "150"))
PERCENTILES = [0, 0.5, 1, 2, 5, 10, 20, 35, 50, 70, 90, 99, 100]  # of the query's own sorted candidate-distance list

print(f"[sens-diag] env={ENV} N_QUERIES={N_QUERIES} percentiles={PERCENTILES}")

ROOT = Path("/home2/mayaank.ashok/lewm_research")
h5_path, ckpt_dir = MECH.h5_path(ROOT), MECH.ckpt_dir(ROOT)
model = load_lewm(Path(ckpt_dir), device=DEV)
encode_frame = make_encode_frame(model)

cache = load_cache()
z_bank = cache["z"]                                  # (Nc, D) training-only, already encoded
act_bank = cache["action"]                            # (Nc, act_dim) raw
ep_offset, ep_len = cache["ep_offset"], cache["ep_len"]
act_mean, act_std = cache["act_mean"], cache["act_std"]
scaler = preprocessing.StandardScaler()
scaler.mean_ = act_mean
scaler.scale_ = np.where(act_std > 0, act_std, 1.0)
scaler.var_ = act_std ** 2
scaler.n_features_in_ = len(act_mean)

Nc = len(z_bank)
ep_start_of = np.empty(Nc, dtype=np.int64)
ep_end_of = np.empty(Nc, dtype=np.int64)
for o, l in zip(ep_offset, ep_len):
    ep_start_of[o:o + l] = o
    ep_end_of[o:o + l] = o + l - 1
rng = np.random.default_rng(0)
L = 25
fwd_ok = (np.arange(Nc) + L) <= ep_end_of              # can extend 25 steps forward
back_ok = (np.arange(Nc) - ep_start_of) >= (HISTORY - 1) * 5  # >=10 steps of own history
cand_idx = np.where(fwd_ok)[0]                          # candidates only need forward room
MAX_CANDIDATES = int(__import__("os").environ.get("DIAG_MAX_CANDIDATES", "200000"))
if len(cand_idx) > MAX_CANDIDATES:
    cand_idx = rng.choice(cand_idx, size=MAX_CANDIDATES, replace=False)
    cand_idx.sort()
cand_z = torch.from_numpy(z_bank[cand_idx]).to(DEV)      # (Ncand, D)
print(f"[sens-diag] candidate bank: {len(cand_idx)}/{Nc} rows usable (fwd_ok); "
      f"{int((fwd_ok & back_ok).sum())} also have their own preceding history (for the d0~=0 baseline)")

BLOCK = 5   # env action steps per predictor block (frame-skip)
act_dim = act_bank.shape[-1]

def zscore_block(raw_LxA):
    """raw (L, act_dim) -> z-scored (L//BLOCK, BLOCK*act_dim), matching Retriever.init_action."""
    z = scaler.transform(raw_LxA).astype(np.float32)
    return z.reshape(len(raw_LxA) // BLOCK, -1)

# ---- held-out queries: same-ep-25 task pool, filtered to rows with >=10 steps of own history ----
pairs = json.load(open(f"{OUT}/pairs/pairs_same_episode_off25_task200u.json"))
start_row = np.asarray(pairs["start_row"])
start_step = np.asarray(pairs["start_step"])
keep = start_step >= 10
start_row, start_step = start_row[keep], start_step[keep]
if len(start_row) > N_QUERIES:
    sel = rng.choice(len(start_row), size=N_QUERIES, replace=False)
    start_row, start_step = start_row[sel], start_step[sel]
print(f"[sens-diag] {len(start_row)} held-out queries with >=10 steps of history "
      f"(of {keep.sum()} eligible / {len(keep)} total same25 pairs)")

f = h5py.File(h5_path, "r", swmr=True, rdcc_nbytes=512 * 1024 * 1024)
q_pix_t = np.stack([f["pixels"][int(r)] for r in start_row])
q_pix_t5 = np.stack([f["pixels"][int(r) - 5] for r in start_row])
q_pix_t10 = np.stack([f["pixels"][int(r) - 10] for r in start_row])
q_act_past2 = np.stack([f["action"][int(r) - 5:int(r)] for r in start_row])    # (N, 5, act_dim) raw, block [t-5, t)
q_act_past1 = np.stack([f["action"][int(r) - 10:int(r) - 5] for r in start_row])  # block [t-10, t-5)
f.close()

z_q_t = encode_frame(q_pix_t)
z_q_t5 = encode_frame(q_pix_t5)
z_q_t10 = encode_frame(q_pix_t10)
z_hist_q_all = np.stack([z_q_t10, z_q_t5, z_q_t], axis=1).astype(np.float32)   # (N, 3, D)
a_past1_z = np.stack([zscore_block(a)[0] for a in q_act_past1])                 # (N, block*act_dim)
a_past2_z = np.stack([zscore_block(a)[0] for a in q_act_past2])
a_past_q_all = np.stack([a_past1_z, a_past2_z], axis=1).astype(np.float32)      # (N, 2, block*act_dim)

# ---- for each query, pick candidates at fixed percentiles of ITS OWN sorted distance list ----
rows_d0, rows_qi, rows_ci = [], [], []
z_query_cur = torch.from_numpy(z_hist_q_all[:, -1]).to(DEV)                     # (N, D)
for qi in range(len(start_row)):
    d = torch.linalg.norm(cand_z - z_query_cur[qi], dim=-1)                     # (Ncand,)
    order = torch.argsort(d)
    n = len(order)
    picked = sorted({min(n - 1, int(round(p / 100 * (n - 1)))) for p in PERCENTILES})
    for k in picked:
        ci = int(order[k].item())
        rows_qi.append(qi)
        rows_ci.append(int(cand_idx[ci]))
        rows_d0.append(float(d[order[k]].item()))
rows_d0 = np.array(rows_d0)
rows_qi = np.array(rows_qi)
rows_ci = np.array(rows_ci)
M = len(rows_qi)
print(f"[sens-diag] {M} (query, candidate) pairs; d0 range [{rows_d0.min():.3f}, {rows_d0.max():.3f}], "
      f"median {np.median(rows_d0):.3f}")

# ---- batched imagine: node's actions, from the QUERY's own real history ----
z_hist_batch = torch.from_numpy(z_hist_q_all[rows_qi]).to(DEV)                  # (M, 3, D)
a_past_batch = torch.from_numpy(a_past_q_all[rows_qi]).to(DEV)                  # (M, 2, blk*A)
a_future_raw = np.stack([act_bank[c:c + L] for c in rows_ci])                    # (M, 25, A)
a_future_z = np.stack([zscore_block(a) for a in a_future_raw]).astype(np.float32)  # (M, 5, blk*A)
a_future_batch = torch.from_numpy(a_future_z).to(DEV)

BATCH = 1024
z_actual_end_list = []
with torch.no_grad():
    for s in range(0, M, BATCH):
        e = min(M, s + BATCH)
        z_img, _, _ = imagine(model, z_hist_batch[s:e], a_past_batch[s:e], a_future_batch[s:e])
        z_actual_end_list.append(z_img[:, -1].cpu().numpy())
z_actual_end = np.concatenate(z_actual_end_list, axis=0)                         # (M, D)
z_node_end = z_bank[rows_ci + L]                                                 # (M, D) real, ground truth
d_final = np.linalg.norm(z_node_end - z_actual_end, axis=-1)

# ---- d0~=0 self-consistency baseline: node's own actions from the node's OWN real history ----
own_hist_ok = back_ok[rows_ci]
bi = np.where(own_hist_ok)[0]
d_final_self = np.full(M, np.nan)
if len(bi) > 0:
    node_hist = np.stack([np.stack([z_bank[c - 10], z_bank[c - 5], z_bank[c]], axis=0)
                          for c in rows_ci[bi]]).astype(np.float32)              # (len(bi), 3, D)
    node_past_raw1 = np.stack([act_bank[c - 10:c - 5] for c in rows_ci[bi]])
    node_past_raw2 = np.stack([act_bank[c - 5:c] for c in rows_ci[bi]])
    node_a_past = np.stack([np.stack([zscore_block(a1)[0], zscore_block(a2)[0]], axis=0)
                            for a1, a2 in zip(node_past_raw1, node_past_raw2)]).astype(np.float32)
    node_hist_t, node_past_t, node_fut_t = (torch.from_numpy(node_hist).to(DEV),
                                             torch.from_numpy(node_a_past).to(DEV),
                                             a_future_batch[bi])
    out_self = []
    with torch.no_grad():
        for s in range(0, len(bi), BATCH):
            e = min(len(bi), s + BATCH)
            z_img, _, _ = imagine(model, node_hist_t[s:e], node_past_t[s:e], node_fut_t[s:e])
            out_self.append(z_img[:, -1].cpu().numpy())
    z_self_end = np.concatenate(out_self, axis=0)
    d_final_self[bi] = np.linalg.norm(z_bank[rows_ci[bi] + L] - z_self_end, axis=-1)
floor = float(np.nanmedian(d_final_self))
print(f"[sens-diag] predictor self-consistency floor (d0~=0 median d_final): {floor:.4f} "
      f"(n={len(bi)}/{M} candidates had their own history)")

# ---- report: is d_final convergent or divergent in d0? ----
from scipy import stats as sstats
slope, intercept, r, p, se = sstats.linregress(rows_d0, d_final)
print(f"\n[sens-diag] linear fit d_final ~ a*d0 + b: slope={slope:.4f} intercept={intercept:.4f} "
      f"r={r:.4f} p={p:.2e}")
mask_pos = (rows_d0 > 1e-6) & (d_final > 1e-6)
slope_log, intercept_log, rlog, plog, _ = sstats.linregress(np.log(rows_d0[mask_pos]), np.log(d_final[mask_pos]))
print(f"[sens-diag] log-log fit log(d_final) ~ s*log(d0) + c: s={slope_log:.4f} (>1 divergent, <1 convergent, "
      f"=1 isometric) r={rlog:.4f} p={plog:.2e}")

edges = np.quantile(rows_d0, np.linspace(0, 1, 11))
print(f"\n[sens-diag] {'d0 bucket':>20}  {'median d0':>10}  {'median d_final':>15}  {'ratio d_final/d0':>17}  "
      f"{'excess over floor':>18}  n")
for lo, hi in zip(edges[:-1], edges[1:]):
    m = (rows_d0 >= lo) & (rows_d0 <= hi)
    if m.sum() == 0:
        continue
    md0, mdf = np.median(rows_d0[m]), np.median(d_final[m])
    print(f"  [{lo:7.3f}, {hi:7.3f}]  {md0:10.3f}  {mdf:15.3f}  {mdf / max(md0, 1e-9):17.3f}  "
          f"{mdf - floor:18.3f}  {int(m.sum())}")

np.savez(f"{OUT}/retrieval_sensitivity_diag.npz", d0=rows_d0, d_final=d_final, d_final_self=d_final_self,
         query_idx=rows_qi, cand_idx=rows_ci, floor=floor)
print(f"\n[sens-diag] saved outputs/{ENV}/retrieval_sensitivity_diag.npz")
