"""Offline assets for hierarchical latent MPC on Push-T (GAS, arXiv:2506.07744, consumed by
LeWM's CEM planner instead of a learned low-level policy -- see scripts/gas_mpc_eval.py).

Preparation uses only non-evaluation episodes and is resumable: every stage
writes to outputs/pusht/ and is skipped when its output already exists.

    python scripts/gas_mpc_prepare.py encode              # training frames only, 500-episode parts
    python scripts/gas_mpc_prepare.py tdr   --seed 0      # Temporal Distance Representation
    python scripts/gas_mpc_prepare.py graph --seed 0 --h-td 8 --te 0.9   # TE filter + nodes + edges
    python scripts/gas_mpc_prepare.py all   --seed 0 --h-td 8 --te 0.9

Outputs (final evaluation episodes are absent from every active preparation cache):
    outputs/pusht/cache_train_parts/part_XXXX.npz     per-500-episode training encode parts
    outputs/pusht/cache_train.npz                     merged (z, state, action, ep_offset, ep_len,
                                                        episode_id, act_mean, act_std)
    outputs/pusht/tdr_full_s{seed}.pt                 TDR weights + eval history (resumable)
    outputs/pusht/psi_train_s{seed}.npy               psi(z) for non-evaluation frames only
    outputs/pusht/graph_full_s{seed}_htd{h}_te{te}.pkl   centers, csr graph, node medoid rows,
                                                        node_z (medoid frame latents), stats
"""

import argparse
import os
import pickle
import sys
import time
from pathlib import Path

import hdf5plugin  # noqa: F401 -- registers the Blosc filter the h5's `pixels` column uses
import h5py
import numpy as np
import torch
from scipy import stats as sps
from scipy.sparse import csr_matrix

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.envs import ENV_MECHANICS
from common.gas import (DEV, TDR, TDRSampler, _n_components, build_node_graph, ema_update, tdr_loss,
                        temporal_efficiency, waypoint_by_distance)
from common.log_util import log
from common.heldout_tasks import fixed_episode_split, source_rows, validate_training_cache
from common.nn_util import precompute_eval_set

ROOT = Path(__file__).resolve().parent.parent
# GAS_MPC_ENV selects the environment (pusht | reacher | tworoom); every output of this
# pipeline for a non-Push-T env lives under outputs/<env>/ so Push-T's paths and
# every existing result file are untouched.
ENV = os.environ.get("GAS_MPC_ENV", "pusht")
MECH = ENV_MECHANICS[ENV]
_default_out = ROOT / "outputs" / ENV
OUT = Path(os.environ.get("GAS_MPC_OUT", str(_default_out)))  # smoke tests redirect this
PART_EPISODES = 500


def atomic_save(path, payload):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as fh:
        if path.suffix == ".pt":
            torch.save(payload, fh)
        elif path.suffix == ".npy":
            np.save(fh, payload)
        elif path.suffix == ".npz":
            np.savez(fh, **payload)
        else:
            pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
        fh.flush(); os.fsync(fh.fileno())
    os.replace(tmp, path)


# ------------------------------------------------------------------
# stage 0: encode non-evaluation episodes (resumable in 500-episode parts)
# ------------------------------------------------------------------

def cmd_encode(args):
    from common.lewm_loader import load_lewm
    from planning_cost_gate import make_encode_frame
    mech = MECH
    h5_path, ckpt_dir = mech.h5_path(ROOT), mech.ckpt_dir(ROOT)
    storage = Path(os.environ.get("GAS_MPC_TRAIN_CACHE_DIR", str(OUT)))
    storage.mkdir(parents=True, exist_ok=True)
    merged = storage / "cache_train.npz"
    if merged.exists():
        with np.load(merged) as saved:
            validate_training_cache({k: saved[k] for k in (
                "training_only", "n_source_episodes", "heldout_frac", "episode_id", "heldout_episode_ids")})
            if float(saved["heldout_frac"]) != args.heldout_frac:
                raise ValueError("Existing training cache has a different holdout fraction")
        log(f"[encode] {merged} verified -- nothing to do")
        link = OUT / "cache_train.npz"
        if merged.resolve() != link.resolve() and not link.exists():
            link.symlink_to(merged.resolve())
        return
    parts_dir = storage / "cache_train_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(h5_path, "r", swmr=True, rdcc_nbytes=512 * 1024 * 1024) as f:
        ep_offset = f["ep_offset"][:].astype(np.int64)
        ep_len = f["ep_len"][:].astype(np.int64)
        n_ep = len(ep_len)
        if os.environ.get("GAS_MPC_MAX_EPISODES"):            # smoke tests: a prefix of the dataset
            n_ep = min(n_ep, int(os.environ["GAS_MPC_MAX_EPISODES"]))
            ep_offset, ep_len = ep_offset[:n_ep], ep_len[:n_ep]
            log(f"[encode] GAS_MPC_MAX_EPISODES={n_ep}")
        selected, held = fixed_episode_split(np.arange(n_ep), args.heldout_frac)
        # Preserve the frozen predictor's original normalization without reading episode actions.
        # These pre-existing aggregate statistics belong to the pretrained LeWM interface.
        with np.load(OUT / "cache_full.npz") as stats:
            act_mean, act_std = stats["act_mean"], stats["act_std"]
        n_parts = (len(selected) + PART_EPISODES - 1) // PART_EPISODES
        log(f"[encode] {len(selected)} training episodes, {int(ep_len[selected].sum())} frames, "
            f"{len(held)} final-holdout episodes excluded; {n_parts} parts of {PART_EPISODES}; "
            f"act_mean={act_mean} act_std={act_std}")
        model = None
        t0 = time.time()
        for p in range(n_parts):
            path = parts_dir / f"part_{p:04d}.npz"
            if model is None:
                model = load_lewm(Path(ckpt_dir), device=DEV)
                encode = make_encode_frame(model, batch=args.batch)
            eps = selected[p * PART_EPISODES:(p + 1) * PART_EPISODES]
            assert not np.intersect1d(eps, held).size
            total = int(ep_len[eps].sum())
            if path.exists():
                with np.load(path) as part:
                    if not np.array_equal(part["episodes"], eps):
                        raise ValueError(f"Stale encoding part: {path}")
                continue
            z = np.empty((total, 192), np.float32)
            state = np.empty((total, mech.state_dim), np.float32)
            action = np.empty((total, mech.action_dim), np.float32)
            pos = 0
            for e in eps:
                lo, hi = int(ep_offset[e]), int(ep_offset[e] + ep_len[e])
                z[pos:pos + hi - lo] = encode(f["pixels"][lo:hi])
                state[pos:pos + hi - lo] = mech.read_state_slice(f, lo, hi - lo)
                action[pos:pos + hi - lo] = f["action"][lo:hi]
                pos += hi - lo
            atomic_save(path, dict(z=z, state=state, action=action, episodes=eps))
            log(f"[encode] part {p + 1}/{n_parts} ({total} frames) elapsed={time.time() - t0:.0f}s")
    # merge
    zs, sts, acts, ids = [], [], [], []
    for p in range(n_parts):
        d = np.load(parts_dir / f"part_{p:04d}.npz")
        zs.append(d["z"]); sts.append(d["state"]); acts.append(d["action"]); ids.append(d["episodes"])
    episode_id = np.concatenate(ids)
    assert np.array_equal(episode_id, selected)
    new_len = ep_len[selected].copy()
    new_offset = np.concatenate([[0], np.cumsum(new_len)[:-1]]).astype(np.int64)
    atomic_save(merged, dict(z=np.concatenate(zs), state=np.concatenate(sts), action=np.concatenate(acts),
                             ep_offset=new_offset, ep_len=new_len, episode_id=episode_id,
                             act_mean=act_mean, act_std=act_std, h5_path=str(h5_path), ckpt_dir=str(ckpt_dir),
                             source_ep_offset=ep_offset[selected], n_source_episodes=n_ep,
                             heldout_frac=args.heldout_frac, heldout_episode_ids=held, training_only=True))
    link = OUT / "cache_train.npz"
    if merged.resolve() != link.resolve():
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(merged.resolve())
    log(f"[encode] wrote {merged} ({merged.stat().st_size / 1e9:.2f} GB)")


def load_cache():
    with np.load(OUT / "cache_train.npz") as d:
        metadata = {k: d[k] for k in ("training_only", "n_source_episodes", "heldout_frac", "episode_id", "heldout_episode_ids")}
        validate_training_cache(metadata)
        return {k: d[k] for k in d.files}


def episodes_of(cache):
    return [np.arange(o, o + l, dtype=np.int64) for o, l in zip(cache["ep_offset"], cache["ep_len"])]


def split_episodes(cache, heldout_frac, seed=0):
    eps = episodes_of(cache)
    perm = np.random.default_rng(seed).permutation(len(eps))
    n_test = max(1, int(len(eps) * heldout_frac))
    test = sorted(perm[:n_test].tolist())
    train = sorted(perm[n_test:].tolist())
    return [eps[i] for i in train], [eps[i] for i in test], np.array(train), np.array(test)


def task_heldout_episodes(ep_col, asset_seed=0):
    """Use precisely the saved TDR's episode split, without loading frame embeddings."""
    ck = torch.load(tdr_path(asset_seed), map_location="cpu", weights_only=False)
    with np.load(OUT / "cache_full.npz") as cache:
        episode_ids = cache["episode_id"]
    dataset_ids = np.unique(ep_col)
    if not np.array_equal(dataset_ids, episode_ids):
        raise ValueError("Encoded episode IDs do not match the evaluation dataset")
    n_test = max(1, int(len(episode_ids) * ck["cfg"]["heldout_frac"]))
    held = np.random.default_rng(0).permutation(len(episode_ids))[:n_test]
    return np.sort(episode_ids[held])


# ------------------------------------------------------------------
# stage 1: TDR (z stays on the CPU -- 1.8 GB does not fit next to a 4 GB GPU's model)
# ------------------------------------------------------------------

# GAS_MPC_TDR_TAG names an alternative TDR (e.g. "_e0.999" for the paper's state-based
# expectile); the TDR, its psi cache and every graph built from it carry the tag
TAG = os.environ.get("GAS_MPC_TDR_TAG", "")


def tdr_path(seed):
    return OUT / f"tdr_full_s{seed}{TAG}.pt"


def gap_pairs(episodes, gaps, per_gap, rng):
    out_i, out_j, out_g = [], [], []
    for g in gaps:
        cand = [(r[t], r[t + g]) for r in episodes for t in range(0, len(r) - g)]
        if not cand:
            continue
        pick = rng.choice(len(cand), size=min(per_gap, len(cand)), replace=False)
        for k in pick:
            out_i.append(cand[k][0]); out_j.append(cand[k][1]); out_g.append(g)
    return np.asarray(out_i), np.asarray(out_j), np.asarray(out_g)


def phi_rows(tdr, z_cpu, rows, bs=8192):
    out = []
    with torch.no_grad():
        for b in range(0, len(rows), bs):
            out.append(tdr.phi(torch.from_numpy(z_cpu[rows[b:b + bs]]).to(DEV)).cpu())
    return torch.cat(out)


def tdr_eval(tdr, z_cpu, ev, l2_ref):
    d_cross = torch.linalg.norm(phi_rows(tdr, z_cpu, ev["ci"]) - phi_rows(tdr, z_cpu, ev["cj"]), dim=-1).numpy()
    d_gap = torch.linalg.norm(phi_rows(tdr, z_cpu, ev["gi"]) - phi_rows(tdr, z_cpu, ev["gj"]), dim=-1).numpy()
    rho_cross = float(sps.spearmanr(d_cross, ev["ctrue"])[0])
    rho_gap = float(sps.spearmanr(d_gap, ev["gap"])[0])
    med = {int(g): float(np.median(d_gap[ev["gap"] == g])) for g in np.unique(ev["gap"])}
    return dict(rho_oracle=rho_cross, rho_gap=rho_gap, median_by_gap=med, l2_rho_oracle=l2_ref["oracle"],
                l2_rho_gap=l2_ref["gap"])


def cmd_tdr(args):
    path = tdr_path(args.seed)
    cache = load_cache()
    z = cache["z"]
    if float(cache["heldout_frac"]) != args.heldout_frac:
        raise ValueError("TDR holdout fraction differs from the training cache")
    train_eps, test_eps, train_idx, diagnostic_idx = split_episodes(cache, 0.1)
    tdr = TDR(z.shape[1], args.tdr_dim, args.tdr_hidden, args.tdr_layers).to(DEV)
    ck = torch.load(path, map_location="cpu", weights_only=False) if path.exists() else None
    if ck is not None and ck.get("done"):
        if ck["cfg"].get("diagnostic_scope") != "training_only":
            raise ValueError(f"{path} contains evaluation-holdout diagnostics; use the train-only preparation driver")
        log(f"[tdr] {path.name} finished ({ck['step']} steps): last eval {ck['history'][-1]}")
        return
    log(f"[tdr] {len(z)} rows, {len(train_eps)} train / {len(test_eps)} internal-validation episodes; final holdout absent, device={DEV}")
    torch.manual_seed(args.seed)
    target = TDR(z.shape[1], args.tdr_dim, args.tdr_hidden, args.tdr_layers).to(DEV)
    target.load_state_dict(tdr.state_dict())
    opt = torch.optim.Adam(tdr.parameters(), lr=args.tdr_lr)
    sampler = TDRSampler(train_eps, len(z), args.tdr_gamma, 0.625, seed=args.seed)

    mech = MECH
    oracle = mech.build_true_distance_oracle(cache["state"])
    test_rows = np.concatenate(test_eps)
    ci, cj, ctrue = precompute_eval_set(test_rows, cache["state"], oracle, 2000, seed=101)
    gi, gj, gap = gap_pairs(test_eps, (1, 2, 5, 10, 25, 50), 400, np.random.default_rng(7))
    ev = dict(ci=ci, cj=cj, ctrue=ctrue, gi=gi, gj=gj, gap=gap)
    l2c = np.linalg.norm(z[ci] - z[cj], axis=1)
    l2g = np.linalg.norm(z[gi] - z[gj], axis=1)
    l2_ref = dict(oracle=float(sps.spearmanr(l2c, ctrue)[0]), gap=float(sps.spearmanr(l2g, gap)[0]))
    log(f"[tdr] raw-L2 reference: rho_oracle={l2_ref['oracle']:.3f} rho_gap={l2_ref['gap']:.3f}")

    history, start = [], 1
    if ck is not None:
        tdr.load_state_dict(ck["tdr"]); target.load_state_dict(ck["target"]); opt.load_state_dict(ck["opt"])
        history, start = ck["history"], ck["step"] + 1
        sampler.rng.bit_generator.state = ck["rng"]
        log(f"[tdr] resuming at step {start}")
    z_t = torch.from_numpy(z)  # CPU
    t0 = time.time()
    for step in range(start, args.tdr_steps + 1):
        s, nx, g = sampler.sample(args.tdr_batch)
        zs, zn, zg = (z_t[s].to(DEV, non_blocking=True), z_t[nx].to(DEV, non_blocking=True),
                      z_t[g].to(DEV, non_blocking=True))
        loss, v_mean = tdr_loss(tdr, target, zs, zn, zg, args.tdr_gamma, args.tdr_expectile)
        opt.zero_grad(); loss.backward(); opt.step()
        ema_update(target, tdr, 0.005)
        if step % args.eval_every == 0 or step == args.tdr_steps:
            tdr.eval()
            rec = dict(step=step, loss=float(loss), v_mean=v_mean, **tdr_eval(tdr, z, ev, l2_ref))
            tdr.train()
            history.append(rec)
            log(f"[tdr] step={step} loss={rec['loss']:.4f} v_mean={v_mean:.2f} "
                f"rho_oracle={rec['rho_oracle']:.3f} (L2 {l2_ref['oracle']:.3f}) rho_gap={rec['rho_gap']:.3f} "
                f"(L2 {l2_ref['gap']:.3f}) median_by_gap={ {k: round(v, 1) for k, v in rec['median_by_gap'].items()} } "
                f"elapsed={time.time() - t0:.0f}s")
            done = step == args.tdr_steps
            atomic_save(path, dict(step=step, done=done, tdr=tdr.state_dict(),
                                   target=None if done else target.state_dict(),
                                   opt=None if done else opt.state_dict(), history=history,
                                   rng=sampler.rng.bit_generator.state,
                                   train_episode_ids=cache["episode_id"][train_idx],
                                   diagnostic_episode_ids=cache["episode_id"][diagnostic_idx],
                                   excluded_episode_ids=cache["heldout_episode_ids"],
                                   cfg=dict(tdr_dim=args.tdr_dim, tdr_hidden=args.tdr_hidden,
                                            tdr_layers=args.tdr_layers, tdr_expectile=args.tdr_expectile,
                                            tdr_gamma=args.tdr_gamma, heldout_frac=args.heldout_frac,
                                            seed=args.seed, diagnostic_scope="training_only")))
    log(f"[tdr] done: {history[-1]}")


def cmd_refresh_tdr(args):
    """Reuse verified weights; replace final-holdout diagnostics with training-only tables."""
    path = tdr_path(args.seed)
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if not ck.get("done"):
        raise ValueError("Only a completed TDR can have its diagnostic tables refreshed")
    cache = load_cache()
    if float(ck["cfg"]["heldout_frac"]) != float(cache["heldout_frac"]):
        raise ValueError("Checkpoint and training cache have different final holdouts")
    if ck["cfg"].get("diagnostic_scope") == "training_only":
        log(f"[refresh] {path.name} already training-only")
        return
    c = ck["cfg"]
    tdr = TDR(cache["z"].shape[1], c["tdr_dim"], c["tdr_hidden"], c["tdr_layers"]).to(DEV)
    tdr.load_state_dict(ck["tdr"]); tdr.eval()
    i, j, gap = gap_pairs(episodes_of(cache), (1, 2, 5, 10, 25, 50), 400, np.random.default_rng(7))
    distances = torch.linalg.norm(phi_rows(tdr, cache["z"], i) - phi_rows(tdr, cache["z"], j), dim=-1).numpy()
    rec = dict(step=ck["step"], scope="training_only", rho_gap=float(sps.spearmanr(distances, gap)[0]),
               median_by_gap={int(g): float(np.median(distances[gap == g])) for g in np.unique(gap)})
    archive = OUT / "legacy" / "holdout_diagnostics" / path.name
    archive.parent.mkdir(parents=True, exist_ok=True)
    if archive.exists():
        raise FileExistsError(f"Refusing to replace archived checkpoint {archive}")
    path.rename(archive)
    ck["history"] = [rec]
    ck["cfg"] = dict(c, diagnostic_scope="training_only")
    ck["train_episode_ids"] = cache["episode_id"]
    ck["diagnostic_episode_ids"] = cache["episode_id"]
    ck["excluded_episode_ids"] = cache["heldout_episode_ids"]
    ck["weights_reused"] = True
    atomic_save(path, ck)
    log(f"[refresh] {path.name}: preserved weights, replaced diagnostics with training-only measurements")


def load_tdr(seed, in_dim=192):
    ck = torch.load(tdr_path(seed), map_location="cpu", weights_only=False)
    assert ck.get("done"), f"TDR {tdr_path(seed)} not finished"
    c = ck["cfg"]
    if c.get("diagnostic_scope") != "training_only":
        raise ValueError(f"{tdr_path(seed)} lacks train-only diagnostic provenance")
    with np.load(OUT / "cache_train.npz") as cache:
        held = cache["heldout_episode_ids"]
        train = cache["episode_id"]
    used = np.r_[ck["train_episode_ids"], ck["diagnostic_episode_ids"]]
    if not np.array_equal(held, ck["excluded_episode_ids"]) or not np.isin(used, train).all():
        raise ValueError("TDR training/diagnostic provenance does not match the training cache")
    tdr = TDR(in_dim, c["tdr_dim"], c["tdr_hidden"], c["tdr_layers"]).to(DEV)
    tdr.load_state_dict(ck["tdr"]); tdr.eval()
    return tdr, ck


# ------------------------------------------------------------------
# stage 2: graph (GPU-chunked exact version of common.gas.td_aware_clustering)
# ------------------------------------------------------------------

def psi_path(seed):
    return OUT / f"psi_train_s{seed}{TAG}.npy"


def graph_path(seed, h_td, te):
    return OUT / f"graph_full_s{seed}{TAG}_htd{h_td:g}_te{te:g}.pkl"


def ensure_psi(seed):
    """psi(z) for training-cache frames only; final evaluation frames are absent."""
    pp = psi_path(seed)
    cache = load_cache()
    tdr, _ = load_tdr(seed, cache["z"].shape[1])
    if pp.exists():
        H = np.load(pp)
        if H.shape != (len(cache["z"]), tdr.tdr_dim):
            raise ValueError("Cached training features have the wrong row layout")
        return H
    H = phi_rows(tdr, cache["z"], np.arange(len(cache["z"]))).numpy().astype(np.float32)
    atomic_save(pp, H)
    log(f"[psi] wrote {pp.name} {H.shape}")
    return H


def td_aware_clustering_gpu(Hk, h_td, chunk=4096):
    """Same semantics as common.gas.td_aware_clustering (Alg. 2, sequential in dataset order:
    a state opens a new cluster iff it is > H_TD/2 from every centre that exists at that
    moment, else joins the nearest existing centre; centres are then reset to member means)
    but processed in chunks: distances to the centres that pre-date the chunk are one GPU
    matmul, and only the points that are far from all of them go through the sequential
    loop (which also sees the centres opened earlier in the same chunk)."""
    M, dim = Hk.shape
    half = h_td / 2.0
    Hk_t = torch.from_numpy(Hk).to(DEV)
    assign = np.empty(M, dtype=np.int64)
    assign[0] = 0
    n = 1
    C = Hk_t[0:1].clone()
    cb = 8192                                       # centre block: bounds every allocation to
    for ci_, a in enumerate(range(1, M, chunk)):    # chunk x cb floats, whatever n grows to
        X = Hk_t[a:a + chunk]
        dmin = torch.full((len(X),), float("inf"), device=DEV)
        jmin = torch.zeros(len(X), dtype=torch.long, device=DEV)
        for c0 in range(0, n, cb):                  # distances to pre-chunk centres, blocked
            d = torch.cdist(X, C[c0:c0 + cb])
            bd, bj = d.min(dim=1)
            better = bd < dmin
            dmin = torch.where(better, bd, dmin)
            jmin = torch.where(better, bj + c0, jmin)
            del d
        dmin_np, jmin_np = dmin.cpu().numpy(), jmin.cpu().numpy()
        far = np.nonzero(dmin_np > half)[0].tolist()
        X_np = Hk[a:a + chunk]
        new_idx, new_pos = [], []                   # centre rows opened in this chunk, and where
        Cn_np = np.empty((0, dim), dtype=np.float32)
        for i in far:                               # small sequential part, on the CPU
            best_d, best_j = float(dmin_np[i]), int(jmin_np[i])
            if len(new_idx):
                dn = np.linalg.norm(Cn_np - X_np[i], axis=1)
                k = int(dn.argmin())
                if dn[k] < best_d:
                    best_d, best_j = float(dn[k]), n + k
            if best_d > half:
                new_idx.append(a + i); new_pos.append(i)
                assign[a + i] = n + len(new_idx) - 1
                Cn_np = np.concatenate([Cn_np, X_np[i:i + 1]])
            else:
                assign[a + i] = best_j
        # near points: nearest among pre-chunk centres and the in-chunk centres opened
        # BEFORE them (exact sequential semantics)
        near = (dmin <= half).nonzero().flatten()
        if len(near):
            assign[a + near.cpu().numpy()] = jmin[near].cpu().numpy()
            if new_idx:
                Cn = Hk_t[new_idx]
                dn = torch.cdist(X[near], Cn)                        # (m_near, n_new)
                pos_new = torch.tensor(new_pos, device=DEV)
                mask = pos_new[None, :] > near[:, None]               # opened after the point
                dn = dn.masked_fill(mask, float("inf"))
                dn_min, kn = dn.min(dim=1)
                better = dn_min < dmin[near]
                if better.any():
                    sel = near[better].cpu().numpy()
                    assign[a + sel] = (n + kn[better]).cpu().numpy()
        if new_idx:
            C = torch.cat([C, Hk_t[new_idx]])
            n += len(new_idx)
        if ci_ % 25 == 0:
            torch.cuda.empty_cache()
            if ci_ % 100 == 0:
                log(f"[graph]   clustering {a}/{M} points -> {n} nodes")
    counts = np.bincount(assign, minlength=n)
    centers = np.zeros((n, dim), dtype=np.float64)
    np.add.at(centers, assign, Hk)
    centers = (centers / counts[:, None]).astype(np.float32)
    dist_to_center = np.linalg.norm(Hk - centers[assign], axis=1)
    order = np.lexsort((dist_to_center, assign))
    first = np.searchsorted(assign[order], np.arange(n))
    medoid = order[first]
    return centers, assign, medoid


def cmd_graph(args):
    gpath = graph_path(args.seed, args.h_td, args.te)
    if gpath.exists():
        with open(gpath, "rb") as fh:
            g = pickle.load(fh)
        if not g.get("training_only"):
            raise ValueError(f"{gpath} lacks train-only preparation provenance")
        log(f"[graph] {gpath.name} exists: {g['stats']}")
        return
    cache = load_cache()
    z = cache["z"]
    tdr, ck = load_tdr(args.seed, z.shape[1])
    H = ensure_psi(args.seed)
    train_eps = episodes_of(cache)
    t0 = time.time()
    way = waypoint_by_distance(H, train_eps, args.h_td)
    te = temporal_efficiency(H, train_eps, way, int(round(args.h_td)))  # s_reached = s_{t+H_TD}: a step count
    log(f"[graph] TE computed in {time.time() - t0:.0f}s")
    train_rows = np.concatenate(train_eps)
    kept_rows = train_rows[te[train_rows] >= args.te]
    log(f"[graph] TE >= {args.te}: kept {len(kept_rows)}/{len(train_rows)} "
        f"({100 * len(kept_rows) / len(train_rows):.1f}%)")
    t1 = time.time()
    centers, assign, medoid = td_aware_clustering_gpu(H[kept_rows], args.h_td)
    log(f"[graph] clustering: {len(centers)} nodes in {time.time() - t1:.0f}s")
    graph = build_node_graph(centers, args.h_td)
    n_comp = _n_components(graph)
    node_rows = kept_rows[medoid]
    g = dict(centers=centers, node_medoid_row=source_rows(cache, node_rows), node_z=z[node_rows].astype(np.float32),
             node_state=cache["state"][node_rows], graph=graph, way=way, te=te, kept_rows=source_rows(cache, kept_rows),
             training_only=True, excluded_episode_ids=cache["heldout_episode_ids"],
             assign=assign, h_td=float(args.h_td), te_threshold=float(args.te), seed=args.seed,
             stats=dict(n_train_states=int(len(train_rows)), n_kept=int(len(kept_rows)),
                        te_retention=float(len(kept_rows) / max(1, len(train_rows))),
                        n_nodes=int(len(centers)), n_edges=int(graph.nnz // 2),
                        mean_degree=float(graph.nnz / max(1, len(centers))), n_components=int(n_comp),
                        build_seconds=time.time() - t0))
    atomic_save(gpath, g)
    log(f"[graph] wrote {gpath.name}: {g['stats']}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=["encode", "tdr", "refresh", "graph", "all"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--heldout-frac", type=float, default=0.02,
                   help="fixed final evaluation exclusion; diagnostics validate within the remaining episodes")
    p.add_argument("--tdr-dim", type=int, default=32)
    p.add_argument("--tdr-hidden", type=int, default=512)
    p.add_argument("--tdr-layers", type=int, default=3)
    p.add_argument("--tdr-expectile", type=float, default=0.99)
    p.add_argument("--tdr-gamma", type=float, default=0.99)
    p.add_argument("--tdr-steps", type=int, default=50000)
    p.add_argument("--tdr-batch", type=int, default=1024)
    p.add_argument("--tdr-lr", type=float, default=3e-4)
    p.add_argument("--eval-every", type=int, default=2500)
    p.add_argument("--h-td", type=float, default=8.0)
    p.add_argument("--te", type=float, default=0.9)
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.stage in ("encode", "all"):
        cmd_encode(args)
    if args.stage in ("tdr", "all"):
        cmd_tdr(args)
    if args.stage == "refresh":
        cmd_refresh_tdr(args)
    if args.stage in ("graph", "all"):
        cmd_graph(args)


if __name__ == "__main__":
    main()
