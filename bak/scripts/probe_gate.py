"""E1 -- does a frozen LeWM latent retain what a temporal-distance head needs?

The kill criterion for the GAS-prior proposal (docs/gas-lewm-proposal.md). GAS learns its
Temporal Distance Representation with an Impala CNN trained end-to-end under an expectile
loss. We propose replacing that CNN with a frozen LeWM encoder plus an MLP head. That only
works if `z` is a sufficient statistic of the observation -- the head learns the metric, so
`z` does NOT need the right geometry (which Push-T already showed it does not have: raw
latent Euclidean scores 0.219 against the oracle), but it DOES need to have kept the
information.

Three probes, all on episode-level held-out data:
  A  state       z          -> projected privileged state   (is the state recoverable at all?)
  B  gap-free    [z_s, z_g] -> true same-episode step gap   (is temporal distance recoverable?)
  C  gap-metric  ||psi(z_s) - psi(z_g)|| -> same gap        (recoverable in METRIC form, which
                                                             is what GAS's graph actually needs)

Probe C is the one that matters for GAS. B is its unconstrained upper bound: if B works and C
does not, the information is there but a metric embedding cannot express it, which is a
different (and more interesting) problem than the prior being uninformative. Probe C also
reports edge precision/recall at a distance threshold, since GAS adds a graph edge exactly
when the predicted temporal distance falls under H_TD -- global Spearman is not what that
decision consumes.

Every probe reports TRAIN alongside TEST, so an underfit head is distinguishable from a
representation that genuinely carries less signal.

Feature arms, all with identical probe architecture/budget so the comparison is about the
representation and nothing else:
  lewm      frozen LeWM z (192-d)                        -- the proposal
  pixels    grid x grid x 3 mean-pooled raw frames       -- the floor: beat this or stop
  randproj  those pixels randomly projected to 192-d     -- dimensionality-matched floor
  state     projected privileged state (probe B/C only)  -- ceiling: how much of the gap is
                                                            predictable from ground truth at all

Scales to the full Push-T h5 (18,685 episodes / 2,336,736 frames): one streaming pass over the
h5 writes memmapped .npy caches, and the probes fetch/normalize per minibatch rather than
moving whole matrices to the GPU. Point PROBE_CACHE_DIR at node-local scratch -- the full-scale
cache is ~5.5 GB and must not land on /home2's 30 GB quota.

Run as:
    python scripts/probe_gate.py env=pusht                       # 300-episode subset
    python scripts/probe_gate.py env=pusht n_episodes=18685      # full dataset
"""

import json
import os
import sys
import time
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401 -- registers the Blosc (32001) filter the pusht h5's `pixels`
# dataset is written with; without it h5py raises "can't open directory (/usr/local/lib/plugin)"
# on the first pixel read. graph_gate.load_landmarks has the same latent dependency but normally
# never hits it, since it reads pixels only when its landmark cache is cold.
import hydra
import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig
from scipy import stats as sps

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.envs import ENV_MECHANICS
from common.graph_lib import DEV
from common.lewm_loader import load_lewm
from common.log_util import log

ROOT = Path(__file__).resolve().parent.parent
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


# ============================================================
# Streaming encode: one pass over the h5 -> memmapped caches
# ============================================================

def encode_dataset(mech, h5_path, cache_dir, n_episodes, ckpt_dir, seed, grid, batch=256):
    """Encode `n_episodes` episodes into memmapped .npy caches: LeWM `z`, mean-pooled pixels,
    privileged state, and episode/step indices.

    One pass, because at full scale the h5 is 46 GB and reading it twice (once for z, once for
    pixels) is the dominant cost. Caches are written to .tmp and renamed only once every array
    is complete -- an interrupted run must not leave a half-written cache that looks valid
    (this project has already been bitten by exactly that on a phi-distance cache).
    """
    tag = f"ep{n_episodes}_seed{seed}_g{grid}"
    names = ("z", "pooled", "state", "ep_idx", "step_idx")
    paths = {k: cache_dir / f"{k}_{tag}.npy" for k in names}
    if all(p.exists() for p in paths.values()):
        log(f"[encode] loading cached memmaps from {cache_dir} ({tag})")
        return {k: np.load(p, mmap_mode="r") for k, p in paths.items()}

    pool = 224 // grid
    if pool * grid != 224:
        raise ValueError(f"pixel_grid={grid} must divide 224 evenly")

    f = h5py.File(h5_path, "r", swmr=True, rdcc_nbytes=512 * 1024 * 1024)
    ep_offset, ep_len = f["ep_offset"][:], f["ep_len"][:]
    rng = np.random.default_rng(seed)
    chosen_ep = rng.choice(len(ep_offset), size=n_episodes, replace=False)
    chosen_ep.sort()
    total = int(ep_len[chosen_ep].sum())
    log(f"[encode] {n_episodes} episodes / {total} frames -> {cache_dir}")

    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp = {k: cache_dir / f"{k}_{tag}.npy.tmp" for k in names}
    spec = dict(z=(np.float32, (total, 192)), pooled=(np.float16, (total, grid * grid * 3)),
                state=(np.float32, (total, mech.state_dim)),
                ep_idx=(np.int64, (total,)), step_idx=(np.int64, (total,)))
    mm = {k: np.lib.format.open_memmap(tmp[k], mode="w+", dtype=d, shape=s)
          for k, (d, s) in spec.items()}

    model = load_lewm(ckpt_dir=ckpt_dir, device=DEV)
    mean, std = IMAGENET_MEAN.to(DEV), IMAGENET_STD.to(DEV)
    w, t0 = 0, time.time()
    for ei, e in enumerate(chosen_ep):
        s, L = int(ep_offset[e]), int(ep_len[e])
        pix = f["pixels"][s: s + L]                       # (L,224,224,3) uint8
        for b0 in range(0, L, batch):
            chunk = pix[b0: b0 + batch]
            t = torch.from_numpy(chunk).to(DEV).permute(0, 3, 1, 2).float() / 255.0
            t = (t - mean) / std
            with torch.no_grad():
                out = model.encode({"pixels": t.unsqueeze(1)})
            mm["z"][w + b0: w + b0 + len(chunk)] = out["emb"][:, 0].cpu().numpy()
        # (L,224,224,3) -> (L,grid,pool,grid,pool,3), mean over the two WITHIN-block axes.
        # Note the axis order: grid is the block index, pool the offset inside a block.
        pooled = pix.astype(np.float32).reshape(L, grid, pool, grid, pool, 3).mean(axis=(2, 4))
        mm["pooled"][w: w + L] = (pooled.reshape(L, -1) / 255.0).astype(np.float16)
        mm["state"][w: w + L] = mech.read_state_slice(f, s, L)
        mm["ep_idx"][w: w + L] = e
        mm["step_idx"][w: w + L] = np.arange(L)
        w += L
        if ei % 500 == 0:
            el = time.time() - t0
            rate = w / max(el, 1e-6)
            log(f"[encode] ep {ei}/{n_episodes} frames={w}/{total} {rate:.0f} f/s "
                f"eta={(total - w) / max(rate, 1e-6) / 60:.1f} min")
    f.close()
    if w != total:
        raise RuntimeError(f"wrote {w} frames, expected {total}")
    for k in names:
        mm[k].flush()
    del mm
    for k in names:
        os.replace(tmp[k], paths[k])
    log(f"[encode] done: {total} frames in {(time.time() - t0) / 60:.1f} min")
    return {k: np.load(p, mmap_mode="r") for k, p in paths.items()}


# ============================================================
# Arms: lazily fetched + normalized per minibatch
# ============================================================

class Arm:
    """A feature view over the cached arrays. Nothing is materialized: at full scale the
    pooled-pixel matrix alone is 7 GB as float32 and the GPU has 11 GB total."""

    def __init__(self, name, source, dim, transform=None):
        self.name, self.source, self.dim, self.transform = name, source, dim, transform
        self.mu = np.zeros((1, dim), dtype=np.float32)
        self.sd = np.ones((1, dim), dtype=np.float32)

    def raw(self, idx):
        a = np.asarray(self.source[idx], dtype=np.float32)
        return self.transform(a) if self.transform is not None else a

    def fit_norm(self, rows, seed, max_rows=200_000):
        """Standardization stats from a subsample -- 200k rows is ample and avoids a full pass.

        A plain `sd[sd < 1e-6] = 1.0` floor is wrong for the raw-pixel arms: mean-pooled frames
        have many near-constant background dims whose train sd is small but not tiny, so a small
        test-time deviation becomes a huge standardized activation and the arm collapses
        (measured: probe-A R^2 = -441 on the pixels arm before this fix). That would make the
        pixel FLOOR look broken and hand the kill criterion a free pass, so the floor is relative
        to the typical spread and the result is clipped -- identically for every arm."""
        sub = np.sort(np.random.default_rng(seed).choice(
            rows, size=min(len(rows), max_rows), replace=False))
        a = self.raw(sub)
        self.mu = a.mean(0, keepdims=True).astype(np.float32)
        sd = a.std(0, keepdims=True)
        self.sd = np.maximum(sd, max(1e-6, 0.01 * float(sd.mean()))).astype(np.float32)
        return self

    def get(self, idx, clip=10.0):
        return torch.from_numpy(
            np.clip((self.raw(idx) - self.mu) / self.sd, -clip, clip)).float().to(DEV)


def build_arms(cfg, cache, proj_state):
    arms = {}
    zdim = cache["z"].shape[1]
    pooled, pdim = cache["pooled"], cache["pooled"].shape[1]
    if "lewm" in cfg.arms:
        arms["lewm"] = Arm("lewm", cache["z"], zdim)
    if "pixels" in cfg.arms:
        arms["pixels"] = Arm("pixels", pooled, pdim)
    if "randproj" in cfg.arms:
        P = np.random.default_rng(cfg.seed).normal(size=(pdim, zdim)).astype(np.float32)
        P /= np.sqrt(pdim)
        arms["randproj"] = Arm("randproj", pooled, zdim, transform=lambda a: a @ P)
    if "state" in cfg.arms:
        arms["state"] = Arm("state", proj_state, proj_state.shape[1])
    return arms


# ============================================================
# Probe heads
# ============================================================

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden, depth):
        super().__init__()
        layers, d = [], in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.ReLU()]
            d = hidden
        layers += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def _train(net, step_fn, n_train, cfg, tag):
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)
    rng = np.random.default_rng(cfg.seed)
    t0 = time.time()
    for step in range(cfg.n_steps):
        loss = step_fn(np.sort(rng.integers(0, n_train, cfg.batch_size)))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % max(1, cfg.n_steps // 4) == 0:
            log(f"    [{tag}] step {step}/{cfg.n_steps} loss={loss.item():.4f} "
                f"({time.time() - t0:.0f}s)")
    return net


def _chunked(fn, n, chunk=16384):
    return np.concatenate([fn(np.arange(a, min(a + chunk, n))) for a in range(0, n, chunk)])


def probe_state(arm, rows_tr, rows_te, Y, cfg, tag):
    """Probe A: features -> projected privileged state. R^2 overall and per-dim."""
    Ytr, Yte = Y[rows_tr], Y[rows_te]
    ymu, ysd = Ytr.mean(0, keepdims=True), Ytr.std(0, keepdims=True)
    ysd[ysd < 1e-6] = 1.0
    ytr = torch.from_numpy((Ytr - ymu) / ysd).float().to(DEV)

    net = MLP(arm.dim, Y.shape[1], cfg.hidden, cfg.depth).to(DEV)
    _train(net, lambda b: ((net(arm.get(rows_tr[b])) - ytr[b]) ** 2).mean(),
           len(rows_tr), cfg, tag)

    def r2(rows, Ytrue):
        with torch.no_grad():
            pred = _chunked(lambda ix: net(arm.get(rows[ix])).cpu().numpy(), len(rows))
        pred = pred * ysd + ymu
        sse = ((pred - Ytrue) ** 2).sum(0)
        sst = ((Ytrue - Ytrue.mean(0, keepdims=True)) ** 2).sum(0)
        return (float(1.0 - sse.sum() / max(sst.sum(), 1e-12)),
                [float(v) for v in 1.0 - sse / np.maximum(sst, 1e-12)])

    te_r2, te_per_dim = r2(rows_te, Yte)
    k = min(len(rows_tr), cfg.n_train_eval)
    tr_r2, _ = r2(rows_tr[:k], Ytr[:k])
    return dict(r2=te_r2, r2_per_dim=te_per_dim, train_r2=tr_r2)


def probe_gap(arm, tr, te, cfg, tag, metric_head):
    """Probes B and C: recover the true same-episode step gap.

    metric_head=False -> free-form MLP on [f_s, f_g] (probe B, unconstrained upper bound)
    metric_head=True  -> ||psi(f_s) - psi(f_g)|| with psi an MLP (probe C, the form GAS needs)
    """
    (tr_s, tr_g, tr_gap), (te_s, te_g, te_gap) = tr, te
    # regress on log1p(gap): gaps are heavy-tailed and GAS thresholds at SMALL distances,
    # so relative error near zero matters far more than absolute error in the tail
    gtr = torch.from_numpy(np.log1p(tr_gap)).float().to(DEV).unsqueeze(-1)

    if metric_head:
        net = MLP(arm.dim, cfg.metric_dim, cfg.hidden, cfg.depth).to(DEV)
        pred_fn = lambda a, b: torch.linalg.norm(net(a) - net(b), dim=-1, keepdim=True)
    else:
        net = MLP(2 * arm.dim, 1, cfg.hidden, cfg.depth).to(DEV)
        pred_fn = lambda a, b: net(torch.cat([a, b], dim=-1))

    _train(net, lambda b: ((pred_fn(arm.get(tr_s[b]), arm.get(tr_g[b])) - gtr[b]) ** 2).mean(),
           len(tr_gap), cfg, tag)

    def predict(s_rows, g_rows):
        with torch.no_grad():
            p = _chunked(lambda ix: pred_fn(arm.get(s_rows[ix]), arm.get(g_rows[ix]))
                         .squeeze(-1).cpu().numpy(), len(s_rows))
        return np.expm1(np.clip(p, 0, 20))

    gap_hat = predict(te_s, te_g)
    k = min(len(tr_gap), cfg.n_train_eval)
    tr_hat = predict(tr_s[:k], tr_g[:k])

    buckets = {}
    for lo, hi in [(1, 5), (5, 15), (15, 40), (40, 10 ** 9)]:
        m = (te_gap >= lo) & (te_gap < hi)
        if m.sum() > 10:
            buckets[f"{lo}-{hi if hi < 10 ** 9 else 'inf'}"] = dict(
                n=int(m.sum()), median_true=float(np.median(te_gap[m])),
                median_pred=float(np.median(gap_hat[m])),
                mae=float(np.abs(gap_hat[m] - te_gap[m]).mean()))

    # What GAS's graph construction actually consumes: an edge is added iff the predicted
    # temporal distance falls under H_TD. Precision here is the fraction of proposed edges
    # that are genuinely within the threshold -- a false edge is a false shortcut in the graph.
    edges = {}
    for T in cfg.edge_thresholds:
        prop, true = gap_hat <= T, te_gap <= T
        edges[f"T{T}"] = dict(
            n_proposed=int(prop.sum()), n_true=int(true.sum()),
            precision=float((prop & true).sum() / max(prop.sum(), 1)),
            recall=float((prop & true).sum() / max(true.sum(), 1)))

    return dict(spearman=float(sps.spearmanr(gap_hat, te_gap).statistic),
                train_spearman=float(sps.spearmanr(tr_hat, tr_gap[:k]).statistic),
                mae=float(np.abs(gap_hat - te_gap).mean()),
                train_mae=float(np.abs(tr_hat - tr_gap[:k]).mean()),
                median_abs_err=float(np.median(np.abs(gap_hat - te_gap))),
                by_gap=buckets, edges=edges)


# ============================================================
# Pair construction
# ============================================================

def make_pairs(ep_idx, step_idx, episodes, n_per_episode, seed, max_gap):
    """Same-episode (s, g) row pairs with g strictly after s. Gap is the ground truth: on
    near-optimal expert data the recorded step gap IS the temporal distance (this project
    measured graph-vs-step-gap Spearman +0.9952 on Push-T expert data)."""
    rng = np.random.default_rng(seed)
    order = np.lexsort((step_idx, ep_idx))
    ep_o = ep_idx[order]
    bounds = np.nonzero(np.diff(ep_o))[0] + 1
    starts = np.concatenate(([0], bounds))
    ends = np.concatenate((bounds, [len(order)]))

    s_rows, g_rows, gaps = [], [], []
    for a, b in zip(starts, ends):
        rows = order[a:b]
        if int(ep_idx[rows[0]]) not in episodes or len(rows) < 2:
            continue
        L = len(rows)
        t = rng.integers(0, L - 1, n_per_episode)
        span = np.minimum(L - 1 - t, max_gap)
        gap = 1 + (rng.random(n_per_episode) * span).astype(np.int64)
        s_rows.append(rows[t])
        g_rows.append(rows[t + gap])
        gaps.append(gap)
    s = np.concatenate(s_rows)
    o = np.argsort(s, kind="stable")   # row locality, so memmap batches read nearby pages
    return s[o], np.concatenate(g_rows)[o], np.concatenate(gaps).astype(np.float64)[o]


@hydra.main(version_base=None, config_path="../config/graph", config_name="probe_gate")
def main(cfg: DictConfig):
    mech = ENV_MECHANICS[cfg.env.name]
    out_dir = ROOT / "outputs" / f"probe_gate_{cfg.env.output_prefix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(os.environ.get("PROBE_CACHE_DIR", str(out_dir / "cache")))

    log(f"=== E1: frozen-latent sufficiency probes (env={cfg.env.name}) ===")
    cache = encode_dataset(mech, mech.h5_path(ROOT), cache_dir, cfg.n_episodes,
                           mech.ckpt_dir(ROOT), cfg.seed, cfg.pixel_grid)
    ep_idx, step_idx = np.asarray(cache["ep_idx"]), np.asarray(cache["step_idx"])
    n = len(ep_idx)
    proj_state = np.asarray(cache["state"])
    if hasattr(mech, "_project_state"):
        proj_state = mech._project_state(proj_state)
    proj_state = proj_state.astype(np.float32)
    log(f"frames: {n}, episodes: {len(np.unique(ep_idx))}, z dim {cache['z'].shape[1]}, "
        f"pooled dim {cache['pooled'].shape[1]}")

    uniq = np.unique(ep_idx)
    shuffled = np.random.default_rng(cfg.seed).permutation(uniq)
    n_train_ep = int(len(shuffled) * cfg.train_frac)
    train_eps = set(map(int, shuffled[:n_train_ep]))
    test_eps = set(map(int, shuffled[n_train_ep:]))
    is_train = np.isin(ep_idx, shuffled[:n_train_ep])
    rows_tr, rows_te = np.nonzero(is_train)[0], np.nonzero(~is_train)[0]
    log(f"split: {len(train_eps)} train / {len(test_eps)} test episodes "
        f"({len(rows_tr)} / {len(rows_te)} frames)")

    tr = make_pairs(ep_idx, step_idx, train_eps, cfg.pairs_per_episode, cfg.seed, cfg.max_gap)
    te = make_pairs(ep_idx, step_idx, test_eps, cfg.pairs_per_episode, cfg.seed + 1, cfg.max_gap)
    log(f"pairs: {len(tr[2])} train / {len(te[2])} test "
        f"(gap median={np.median(te[2]):.0f} max={te[2].max():.0f})")

    arms = build_arms(cfg, cache, proj_state)
    results = {}
    for name, arm in arms.items():
        log(f"\n--- arm: {name} ({arm.dim}d) ---")
        arm.fit_norm(rows_tr, cfg.seed)
        r = {}
        if name != "state":
            log("  probe A: features -> privileged state")
            r["A_state"] = probe_state(arm, rows_tr, rows_te, proj_state, cfg, f"{name}/A")
            per_dim = [round(v, 3) for v in r["A_state"]["r2_per_dim"]]
            log(f"  [A] R^2={r['A_state']['r2']:.4f} (train {r['A_state']['train_r2']:.4f}) "
                f"per-dim={per_dim}")

        log("  probe B: [f_s, f_g] -> step gap (free-form)")
        r["B_gap_freeform"] = probe_gap(arm, tr, te, cfg, f"{name}/B", metric_head=False)
        log(f"  [B] spearman={r['B_gap_freeform']['spearman']:.4f} "
            f"(train {r['B_gap_freeform']['train_spearman']:.4f}) "
            f"median|err|={r['B_gap_freeform']['median_abs_err']:.2f} steps")

        log("  probe C: ||psi(f_s)-psi(f_g)|| -> step gap (metric form, what GAS needs)")
        r["C_gap_metric"] = probe_gap(arm, tr, te, cfg, f"{name}/C", metric_head=True)
        ep = {k: round(v["precision"], 3) for k, v in r["C_gap_metric"]["edges"].items()}
        log(f"  [C] spearman={r['C_gap_metric']['spearman']:.4f} "
            f"(train {r['C_gap_metric']['train_spearman']:.4f}) "
            f"median|err|={r['C_gap_metric']['median_abs_err']:.2f} steps  edge-precision={ep}")
        results[name] = r

    verdict = {}
    if "lewm" in results and "pixels" in results:
        floor = max(results[a]["C_gap_metric"]["spearman"]
                    for a in ("pixels", "randproj") if a in results)
        verdict = dict(
            lewm_A_r2=results["lewm"]["A_state"]["r2"],
            pixels_A_r2=results["pixels"]["A_state"]["r2"],
            lewm_C_spearman=results["lewm"]["C_gap_metric"]["spearman"],
            pixel_floor_C_spearman=floor,
            state_ceiling_C_spearman=(results["state"]["C_gap_metric"]["spearman"]
                                      if "state" in results else None),
            # E1's kill criterion (docs/gas-lewm-proposal.md): beat the raw-pixel floor on
            # BOTH probes, or the frozen prior carries nothing GAS's TDR head could use.
            passed=bool(results["lewm"]["A_state"]["r2"] > results["pixels"]["A_state"]["r2"]
                        and results["lewm"]["C_gap_metric"]["spearman"] > floor),
        )
        log(f"\n[E1] kill criterion (LeWM beats the raw-pixel floor on probe A AND probe C): "
            f"{'PASSED' if verdict['passed'] else 'FAILED'}")

    summary = dict(env=cfg.env.name, n_episodes=cfg.n_episodes, n_frames=int(n),
                   n_train_pairs=int(len(tr[2])), n_test_pairs=int(len(te[2])),
                   pixel_grid=int(cfg.pixel_grid), n_steps=int(cfg.n_steps),
                   arms=list(arms.keys()), probes=results, verdict=verdict)
    out_path = out_dir / f"e1_probe_results_ep{cfg.n_episodes}.json"
    out_path.write_text(json.dumps(summary, indent=2))
    log(f"\nwrote {out_path}")
    log(json.dumps(verdict, indent=2))


if __name__ == "__main__":
    main()
