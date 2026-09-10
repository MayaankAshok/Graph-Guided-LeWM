"""Latent displacement and predictor error, as distributions, across prediction horizons.

Produces, for each horizon h in {1, 5, 10, 25, 50, 80, 100} environment steps:

  * a histogram of the TRUE latent displacement ||z_t - z_{t+h}||, i.e. how far the encoder
    moves when the environment really advances h steps;
  * a histogram of the PREDICTOR ERROR ||z_{t+h} - pred(z_t, a_t..a_{t+h-1})||, i.e. how far
    the autoregressive latent rollout lands from where the episode really went.

h=1 gets displacement only: one predictor call spans SKIP=5 env steps by construction (the
action_encoder's 10 inputs are 5 concatenated 2-D actions, see
[[lewm-predictor-action-block-convention]]), so a 1-step forecast is not a question this
checkpoint was trained to answer.

Why both on the same axes: an error is only interpretable against the movement it is trying
to track. A prediction error of 13 is excellent if the state moved 100 and worthless if it
moved 15. The overlay makes that ratio visible directly, and the summary figure plots both
against horizon together with the cross-episode saturation level -- the distance between two
frames drawn from unrelated episodes, which is the ceiling any latent distance can report.

Efficiency: episodes are streamed one at a time (read pixels -> encode -> keep only the
(L,192) embeddings -> drop the pixels). Holding every sampled frame's pixels at once would be
several GB for no reason, and this repo has hit exactly that class of allocation failure
before.

Usage:
    python scripts/latent_horizon_histograms.py --env pusht --n-episodes 400 \
        --out outputs/latent_horizon_pusht
"""

import argparse
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.envs import ENV_MECHANICS
from common.lewm_loader import load_lewm
from common.log_util import log

DEV = "cuda" if torch.cuda.is_available() else "cpu"
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

SKIP = 5        # env steps per predictor call / actions per action block
HISTORY = 3     # history blocks the predictor conditions on
HORIZONS = [1, 5, 10, 25, 50, 80, 100]


def make_encode(model, batch=64):
    mean, std = IMAGENET_MEAN.to(DEV), IMAGENET_STD.to(DEV)

    def encode(pixels_nhwc):
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

    return encode


def encode_episodes(h5_path, mech, n_episodes, seed):
    """Stream-encode every frame of `n_episodes` uniformly-sampled episodes."""
    model = load_lewm(ckpt_dir=Path(mech.ckpt_dir(ROOT)), device=DEV)
    encode = make_encode(model)
    rng = np.random.default_rng(seed)

    with h5py.File(h5_path, "r", swmr=True, rdcc_nbytes=512 * 1024 * 1024) as f:
        ep_offset = f["ep_offset"][:].astype(np.int64)
        ep_len = f["ep_len"][:].astype(np.int64)
        all_len = ep_len.copy()

        # CRITICAL: the model was trained on z-SCORED actions. train.py applies
        # utils.get_column_normalizer to every non-pixel column (action included), and
        # eval.py fits a sklearn StandardScaler on the same column. Push-T's action std is
        # ~0.206, so feeding raw actions hands the predictor inputs ~5x too small, which
        # makes it behave like a no-motion baseline. Normalize with the dataset's own
        # statistics, exactly as both training and eval do.
        a_all = f["action"][:].astype(np.float64)
        a_all = a_all[~np.isnan(a_all).any(axis=1)]
        act_mean = a_all.mean(0).astype(np.float32)
        act_std = a_all.std(0).astype(np.float32)
        del a_all
        log(f"[action-norm] mean={act_mean} std={act_std}")
        eps = np.sort(rng.choice(len(ep_offset), size=min(n_episodes, len(ep_offset)),
                                  replace=False))
        episodes, t0 = [], time.time()
        for n, e in enumerate(eps):
            s, L = int(ep_offset[e]), int(ep_len[e])
            z = encode(f["pixels"][s:s + L])
            a = (f["action"][s:s + L].astype(np.float32) - act_mean) / act_std
            episodes.append(dict(ep=int(e), z=z, action=a, length=L))
            if n % 50 == 0:
                log(f"  encoded {n}/{len(eps)} episodes  elapsed={time.time()-t0:.0f}s")
    return model, episodes, all_len, (act_mean, act_std)


def latent_rollout(model, z_hist, act_hist, act_future, batch=1024):
    """Autoregressive rollout, identical in structure to jepa.JEPA.rollout.

    z_hist (N,HISTORY,D); act_hist (N,HISTORY,SKIP*A) where the last entry is the first block
    being rolled out; act_future (N,n_blocks-1,SKIP*A). Returns (N,D): the predicted
    embedding after all n_blocks blocks.
    """
    out = np.empty((len(z_hist), z_hist.shape[-1]), dtype=np.float32)
    for lo in range(0, len(z_hist), batch):
        hi = min(lo + batch, len(z_hist))
        emb = torch.from_numpy(z_hist[lo:hi]).float().to(DEV)
        act = torch.from_numpy(act_hist[lo:hi]).float().to(DEV)
        fut = torch.from_numpy(act_future[lo:hi]).float().to(DEV)
        with torch.no_grad():
            for t in range(fut.shape[1] + 1):
                act_emb = model.action_encoder(act[:, -HISTORY:])
                pred = model.predict(emb[:, -HISTORY:], act_emb)[:, -1:]
                if t < fut.shape[1]:
                    emb = torch.cat([emb, pred], dim=1)
                    act = torch.cat([act, fut[:, t:t + 1]], dim=1)
        out[lo:hi] = pred[:, 0].float().cpu().numpy()
    return out


def collect_horizon(model, episodes, h, max_samples, rng, action_dim):
    """Returns (displacement, pred_error, n_episodes_used). pred_error is None for h < SKIP."""
    disp, src = [], []
    for ei, ep in enumerate(episodes):
        z, L = ep["z"], ep["length"]
        if L <= h:
            continue
        d = np.linalg.norm(z[:L - h] - z[h:L], axis=1)
        disp.append(d)
        src.append(np.full(len(d), ei))
    if not disp:
        return np.array([]), None, 0
    disp = np.concatenate(disp)
    n_eps_used = len(set(np.concatenate(src).tolist()))
    if len(disp) > max_samples:
        disp = rng.choice(disp, size=max_samples, replace=False)

    if h < SKIP or h % SKIP != 0:
        return disp, None, n_eps_used

    # predictor arm: needs (HISTORY-1) real past blocks before t, and h/SKIP blocks after
    n_blocks = h // SKIP
    lead = (HISTORY - 1) * SKIP
    zh, ah, af, tgt = [], [], [], []
    for ep in episodes:
        z, a, L = ep["z"], ep["action"], ep["length"]
        if L <= lead + h:
            continue
        for t in range(lead, L - h):
            zh.append(z[[t - 2 * SKIP, t - SKIP, t]])
            blocks = [a[t - 2 * SKIP:t - SKIP], a[t - SKIP:t]]
            blocks += [a[t + k * SKIP:t + (k + 1) * SKIP] for k in range(n_blocks)]
            b = np.stack(blocks).reshape(len(blocks), SKIP * action_dim)
            ah.append(b[:HISTORY])
            af.append(b[HISTORY:])
            tgt.append(z[t + h])
    if not zh:
        return disp, None, n_eps_used
    if len(zh) > max_samples:
        idx = rng.choice(len(zh), size=max_samples, replace=False)
        zh = [zh[i] for i in idx]; ah = [ah[i] for i in idx]
        af = [af[i] for i in idx]; tgt = [tgt[i] for i in idx]
    zh = np.stack(zh).astype(np.float32)
    ah = np.stack(ah).astype(np.float32)
    af = (np.stack(af).astype(np.float32) if af[0].size
          else np.zeros((len(zh), 0, SKIP * action_dim), dtype=np.float32))
    tgt = np.stack(tgt).astype(np.float32)
    pred = latent_rollout(model, zh, ah, af)
    return disp, np.linalg.norm(pred - tgt, axis=1), n_eps_used


def cross_episode_reference(episodes, rng, n=200_000):
    """Median distance between frames from DIFFERENT episodes: the saturation ceiling."""
    starts = np.cumsum([0] + [len(e["z"]) for e in episodes])
    Z = np.concatenate([e["z"] for e in episodes])
    owner = np.concatenate([np.full(len(e["z"]), i) for i, e in enumerate(episodes)])
    i = rng.integers(0, len(Z), n)
    j = rng.integers(0, len(Z), n)
    keep = owner[i] != owner[j]
    d = np.linalg.norm(Z[i[keep]] - Z[j[keep]], axis=1)
    del starts
    return d


def make_figures(res, xref, ep_len, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    C_DISP, C_PRED, C_REF = "#3A4CC4", "#96590A", "#525A6C"
    hs = [h for h in HORIZONS if res[str(h)]["displacement"] is not None]

    # ---- Figure 1+2 combined: per-horizon overlay ----------------------------
    fig, axes = plt.subplots(2, 4, figsize=(15, 6.6))
    for ax, h in zip(axes.ravel(), hs):
        r = res[str(h)]
        d = r["displacement"]
        bins = np.linspace(0, max(float(np.percentile(d, 99.5)), 1.0) * 1.15, 60)
        ax.hist(d, bins=bins, color=C_DISP, alpha=0.65, label=f"true $\\|z_t-z_{{t+{h}}}\\|$")
        if r["pred_error"] is not None:
            ax.hist(r["pred_error"], bins=bins, color=C_PRED, alpha=0.6,
                    label="predictor error")
        ax.axvline(xref, color=C_REF, ls="--", lw=1.4, label="cross-episode ceiling")
        ax.set_title(f"h = {h} env steps", fontsize=11)
        ax.set_xlabel("latent distance"); ax.set_yticks([])
        ax.legend(fontsize=7, frameon=False)
    axes.ravel()[-1].axis("off")
    fig.suptitle("Latent displacement vs. predictor error, by horizon (Push-T)", fontsize=13)
    fig.tight_layout()
    fig.savefig(Path(out_dir) / "horizon_histograms.pdf")
    plt.close(fig)

    # ---- Figure 3: episode lengths -------------------------------------------
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    ax.hist(ep_len, bins=80, color=C_DISP, alpha=0.8)
    for h in [50, 80, 100]:
        ax.axvline(h + 11, color=C_PRED, ls="--", lw=1.2)
        ax.text(h + 13, ax.get_ylim()[1] * 0.85, f"h={h}", fontsize=8, color=C_PRED)
    ax.set_xlabel("episode length (env steps)"); ax.set_ylabel("episodes")
    ax.set_title(f"Push-T episode lengths (n={len(ep_len)}, median {int(np.median(ep_len))})",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(Path(out_dir) / "episode_lengths.pdf")
    plt.close(fig)

    # ---- Figure 4: the summary curve -----------------------------------------
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    med_d = [float(np.median(res[str(h)]["displacement"])) for h in hs]
    ax.plot(hs, med_d, "o-", color=C_DISP, label="median true displacement")
    ph = [h for h in hs if res[str(h)]["pred_error"] is not None]
    med_p = [float(np.median(res[str(h)]["pred_error"])) for h in ph]
    ax.plot(ph, med_p, "s-", color=C_PRED, label="median predictor error")
    ax.axhline(xref, color=C_REF, ls="--", lw=1.4, label="cross-episode ceiling")
    ax.set_xlabel("horizon h (env steps)"); ax.set_ylabel("latent distance")
    ax.set_title("The latent saturates; the predictor does not track it", fontsize=11)
    ax.legend(fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(Path(out_dir) / "horizon_summary.pdf")
    plt.close(fig)


ROOT = Path(__file__).resolve().parent.parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="pusht", choices=sorted(ENV_MECHANICS))
    p.add_argument("--n-episodes", type=int, default=400)
    p.add_argument("--max-samples", type=int, default=40_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    out = Path(args.out or ROOT / "outputs" / f"latent_horizon_{args.env}")
    out.mkdir(parents=True, exist_ok=True)

    mech = ENV_MECHANICS[args.env]
    log(f"[setup] env={args.env} device={DEV} n_episodes={args.n_episodes}")
    model, episodes, all_len, act_stats = encode_episodes(mech.h5_path(ROOT), mech,
                                                           args.n_episodes, args.seed)
    log(f"[setup] encoded {len(episodes)} episodes, "
        f"{sum(e['length'] for e in episodes)} frames")

    rng = np.random.default_rng(args.seed + 1)
    xref_samples = cross_episode_reference(episodes, rng)
    xref = float(np.median(xref_samples))
    log(f"[ref] cross-episode median distance (saturation ceiling) = {xref:.3f}")

    res, summary = {}, {}
    for h in HORIZONS:
        d, pe, n_eps = collect_horizon(model, episodes, h, args.max_samples,
                                        rng, mech.action_dim)
        res[str(h)] = dict(displacement=d, pred_error=pe)
        s = dict(h=h, n_disp=int(len(d)), n_episodes_used=int(n_eps),
                 disp_median=float(np.median(d)),
                 disp_q25=float(np.percentile(d, 25)), disp_q75=float(np.percentile(d, 75)),
                 disp_frac_of_ceiling=float(np.median(d) / xref))
        if pe is not None:
            s.update(n_pred=int(len(pe)), pred_median=float(np.median(pe)),
                     pred_q25=float(np.percentile(pe, 25)),
                     pred_q75=float(np.percentile(pe, 75)),
                     pred_over_disp=float(np.median(pe) / np.median(d)),
                     pred_frac_of_ceiling=float(np.median(pe) / xref))
        summary[str(h)] = s
        log(f"[h={h:>3}] disp med={s['disp_median']:6.2f} (n={s['n_disp']})"
            + (f"  pred med={s['pred_median']:6.2f} (n={s['n_pred']})"
               f"  pred/disp={s['pred_over_disp']:.3f}" if pe is not None else ""))

    ep_stats = dict(n_episodes_total=int(len(all_len)), n_frames_total=int(all_len.sum()),
                    mean=float(all_len.mean()), median=float(np.median(all_len)),
                    std=float(all_len.std()), min=int(all_len.min()), max=int(all_len.max()),
                    percentiles={str(q): float(np.percentile(all_len, q))
                                 for q in (1, 5, 10, 25, 50, 75, 90, 95, 99)},
                    usable_episodes={str(h): int((all_len >= h + 11).sum()) for h in HORIZONS})

    (out / "summary.json").write_text(json.dumps(
        dict(env=args.env, cross_episode_ceiling=xref, horizons=summary,
             action_mean=act_stats[0].tolist(), action_std=act_stats[1].tolist(),
             episode_stats=ep_stats), indent=2))
    np.savez_compressed(out / "raw.npz", cross_episode=xref_samples[:200_000],
                        ep_len=all_len,
                        **{f"disp_{h}": res[str(h)]["displacement"] for h in HORIZONS},
                        **{f"pred_{h}": res[str(h)]["pred_error"] for h in HORIZONS
                           if res[str(h)]["pred_error"] is not None})
    make_figures(res, xref, all_len, out)
    log(f"[done] wrote {out}/summary.json and 3 figures")


if __name__ == "__main__":
    main()
