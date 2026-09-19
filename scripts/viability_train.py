"""Train the latent viability critic -- docs/viability-proposal/viability-value.tex Sec. 4,
Alg. 1 -- on a cache produced by viability_cache.py. Frozen LeWM encoder + predictor; only
the critic ensemble is trained.

Per update (Alg. 1 lines 2-5), four batches feed three losses:

  hindsight   logged (z_t, z_{t+delta}, h), label 1[h >= tau]. tau is the FIRST step at
              which the trajectory satisfies the env's success predicate towards the goal
              frame (common.viability.first_hit_time), not delta itself.        -> L_viab
  cross-ep    logged (z_t, z_{t'}, h) from different episodes, label 0, optionally only
              when the block poses are far apart (--xneg-min-block-dist).       -> L_viab

Label source (--label-source). `state` (the original): tau and the cross-negative filter come
from the dataset's state column through the env's success predicate / block distance --
privileged supervision, not pixel-only. `tdr`: both come from the TDR psi(z) alone
(scripts/latent_label_substitute_diag.py measured the substitution on Push-T: tau labels
agree with the state rule on 97.7% of (pair, h) draws vs 94.9% for tau = delta; the state
column is then only read for held-out reporting):
  tau   = first k <= delta with ||psi_{t+k} - psi_{t+delta}|| < r, r = the median TDR
          distance at a --tdr-radius-gap-step gap (calibrated on the training episodes)
  cross (t, g) is a valid label-0 pair at budget h iff ||psi_t - psi_g|| >= --xneg-tdr-factor
          * d_cal(max(h, 5)), d_cal(k) = median TDR distance at a k-step gap
  imagined    roll the frozen predictor --imagine-blocks blocks from a logged history under
              logged / perturbed / random actions (proposal eq. 7). Rows driven by the LOGGED
              actions get hindsight labels (their true states are known); the others are
              label-free and enter only through the two consistency terms.      -> L_viab
  bellman     V(z,g,h) <- max_a V_tgt(F(hist,a), g, h-5) on logged and imagined pre-states
              (proposal eq. 12), EMA target ensemble, pessimistic aggregation.  -> L_bellman
  mono        relu(V(z,g,h) - V(z,g,h+5)) on all of the above (proposal eq. 11). -> L_mono

  L = L_viab + lambda_B * L_bellman + lambda_M * L_mono

The proposal's L_cal is NOT implemented: it is unspecified, BCE is already a proper scoring
rule, and calibration is measured (ECE on held-out episodes) rather than trained.

Usage:
    python scripts/viability_train.py --cache outputs/pusht/critic_training/cache_1000_s0.npz \
        --out outputs/pusht/critic_training/critic_v0 --steps 20000
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.envs import ENV_MECHANICS
from common.heldout_tasks import validate_training_cache
from common.gas import TDR
from common.lewm_loader import load_lewm
from common.log_util import log
from common.viability import (HISTORY, SKIP, ViabilityCritic, auroc, bellman_target, ece,
                              first_hit_time, history_at, imagine, sample_action_blocks)

DEV = "cuda" if torch.cuda.is_available() else "cpu"
MECH = ENV_MECHANICS["pusht"]          # overridden by --env in run()
GOAL_REACHED = MECH.goal_reached


# ------------------------------------------------------------------
# data
# ------------------------------------------------------------------

class LatentCache:
    """viability_cache.py output on the device, plus per-frame episode bookkeeping."""

    def __init__(self, path, device):
        d = np.load(path)
        split = {k: d[k] for k in ("training_only", "n_source_episodes", "heldout_frac", "episode_id", "heldout_episode_ids")}
        self.evaluation_eps = validate_training_cache(split)
        self.episode_ids = split["episode_id"]
        self.z = torch.tensor(d["z"], device=device)
        self.state = torch.tensor(d["state"], device=device)
        self.act_raw = torch.tensor(d["action"], device=device)
        self.act_mean = torch.tensor(d["act_mean"], device=device)
        self.act_std = torch.tensor(d["act_std"], device=device)
        self.act_norm = (self.act_raw - self.act_mean) / self.act_std
        self.ep_offset, self.ep_len = d["ep_offset"].astype(np.int64), d["ep_len"].astype(np.int64)
        n_frames, n_ep = len(self.z), len(self.ep_len)
        frame_ep = np.repeat(np.arange(n_ep), self.ep_len)
        self.frame_ep = torch.tensor(frame_ep, device=device)
        self.frame_step = torch.tensor(np.arange(n_frames) - self.ep_offset[frame_ep], device=device)
        self.frame_len = torch.tensor(self.ep_len[frame_ep], device=device)
        self.n_frames, self.n_ep, self.device = n_frames, n_ep, device
        self.meta = {k: (d[k].item() if (k in d.files and d[k].ndim == 0) else None)
                     for k in ("seed", "n_source_episodes", "h5_path", "ckpt_dir")}   # gas_mpc caches lack seed/n_source
        self.psi = None          # TDR psi(z) per frame, set by attach_tdr (label_source=tdr)
        self.d_cal = None        # (h_max+1,) median TDR distance at each same-episode step gap
        self.tdr_radius = None

    @torch.no_grad()
    def attach_tdr(self, path, bs=16384):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        assert ck.get("done"), f"TDR {path} not finished"
        c = ck["cfg"]
        if c.get("diagnostic_scope") != "training_only":
            raise ValueError("TDR has not been prepared with training-only diagnostics")
        used = np.r_[ck["train_episode_ids"], ck["diagnostic_episode_ids"]]
        if not np.array_equal(ck["excluded_episode_ids"], self.evaluation_eps) or not np.isin(used, self.episode_ids).all():
            raise ValueError("TDR split provenance differs from the critic training cache")
        tdr = TDR(self.z.shape[1], c["tdr_dim"], c["tdr_hidden"], c["tdr_layers"]).to(self.device)
        tdr.load_state_dict(ck["tdr"]); tdr.eval()
        self.psi = torch.cat([tdr.phi(self.z[b:b + bs]) for b in range(0, self.n_frames, bs)])
        return c

    @torch.no_grad()
    def calibrate_tdr(self, episodes, h_max, generator, per_gap=20000):
        """d_cal[k] = median ||psi_t - psi_{t+k}|| over same-episode pairs in `episodes`,
        k = 0..h_max (d_cal[0] = 0). Same statistic as gas_mpc_eval.gap_calibration, on the
        critic's own training split."""
        frames = torch.nonzero(self.frames_of(episodes)).squeeze(1)
        d_cal = torch.zeros(h_max + 1, device=self.device)
        for k in range(1, h_max + 1):
            t = frames[torch.randint(0, len(frames), (per_gap * 2,), device=self.device, generator=generator)]
            t = t[self.frame_step[t] + k <= self.frame_len[t] - 1][:per_gap]
            d_cal[k] = torch.linalg.norm(self.psi[t] - self.psi[t + k], dim=-1).median()
        self.d_cal = d_cal
        return d_cal

    def split_episodes(self, val_frac, seed):
        rng = np.random.default_rng(seed)
        perm = rng.permutation(self.n_ep)
        n_val = max(1, int(round(val_frac * self.n_ep)))
        return np.sort(perm[n_val:]), np.sort(perm[:n_val])

    def frames_of(self, episodes):
        ep_mask = torch.zeros(self.n_ep, dtype=torch.bool, device=self.device)
        ep_mask[torch.as_tensor(episodes, device=self.device)] = True
        return ep_mask[self.frame_ep]

    def anchors(self, episodes, min_step, min_after):
        """Frames in `episodes` with >= min_step frames of real history before them and
        >= min_after frames after them inside the episode."""
        ok = self.frames_of(episodes) & (self.frame_step >= min_step) & \
            (self.frame_step + min_after <= self.frame_len - 1)
        return torch.nonzero(ok).squeeze(1)


class Sampler:
    """Draws the four batch types of Alg. 1 from one episode split."""

    def __init__(self, cache, episodes, args, generator):
        self.c, self.a, self.g = cache, args, generator
        hist = (HISTORY - 1) * SKIP
        # min_after=SKIP: every anchor also serves as a Bellman pre-state, whose base action
        # block is the logged one taken at t, so those 5 actions must exist in the episode
        self.anchor_h = cache.anchors(episodes, min_step=hist, min_after=SKIP)
        self.anchor_i = cache.anchors(episodes, min_step=hist, min_after=SKIP * args.imagine_blocks)
        self.split_frames = torch.nonzero(cache.frames_of(episodes)).squeeze(1)
        self.h_grid = torch.arange(0, args.h_max + 1, SKIP, device=cache.device)
        self.dev = cache.device
        if len(self.anchor_h) == 0 or len(self.anchor_i) == 0:
            raise SystemExit("no valid anchors -- cache too small for these settings")
        # hit rule for tau: (per-frame features, predicate(frames, goal)) -- see module docstring
        if args.label_source == "tdr":
            r = cache.tdr_radius
            self.feat, self.reached = cache.psi, (lambda a, b: torch.linalg.norm(a - b, dim=-1) < r)
        else:
            self.feat, self.reached = cache.state, GOAL_REACHED

    # -- helpers ---------------------------------------------------------
    def _rint(self, n, high):
        return torch.randint(0, int(high), (n,), device=self.dev, generator=self.g)

    def _rand(self, n):
        return torch.rand(n, device=self.dev, generator=self.g)

    def rand_h(self, n, min_h=0):
        grid = self.h_grid[self.h_grid >= min_h]
        return grid[self._rint(n, len(grid))]

    def _goal_and_tau(self, t, delta_max):
        """Hindsight goal g = t + delta within the episode and the first-hit time tau."""
        c = self.c
        avail = c.frame_len[t] - 1 - c.frame_step[t]
        dmax = torch.minimum(avail, torch.full_like(avail, delta_max))
        delta = (self._rand(len(t)) * (dmax + 1).float()).long().clamp(max=dmax)
        g = t + delta
        L = delta_max + 1
        idx = t[:, None] + torch.arange(L, device=self.dev)[None]
        idx = torch.minimum(idx, (t + avail)[:, None])            # stay inside the episode
        tau = first_hit_time(self.feat[idx], self.feat[g], delta, self.reached)
        return g, delta, tau

    def tau_state(self, t, g, delta):
        """The state-predicate tau for already-drawn pairs (held-out reporting only)."""
        c = self.c
        avail = c.frame_len[t] - 1 - c.frame_step[t]
        idx = t[:, None] + torch.arange(self.a.delta_max + 1, device=self.dev)[None]
        idx = torch.minimum(idx, (t + avail)[:, None])
        return first_hit_time(c.state[idx], c.state[g], delta, GOAL_REACHED)

    def _history(self, t):
        """(z_hist (B,3,D), a_past (B,2,10)) for logged anchors t."""
        c = self.c
        zi = t[:, None] - torch.arange(HISTORY - 1, -1, -1, device=self.dev)[None] * SKIP
        ai = t[:, None] + torch.arange(-(HISTORY - 1) * SKIP, 0, device=self.dev)[None]
        return c.z[zi], c.act_norm[ai].reshape(len(t), HISTORY - 1, SKIP * c.act_raw.shape[1])

    def _raw_blocks(self, t, n):
        """Logged raw action blocks taken at t, t+5, ..., t+5(n-1): (B, n, 10)."""
        ai = t[:, None] + torch.arange(n * SKIP, device=self.dev)[None]
        return self.c.act_raw[ai].reshape(len(t), n, SKIP * self.c.act_raw.shape[1])

    # -- batches ---------------------------------------------------------
    def hindsight(self, B):
        t = self.anchor_h[self._rint(B, len(self.anchor_h))]
        g, delta, tau = self._goal_and_tau(t, self.a.delta_max)
        h = self.rand_h(B)
        label = (h >= tau).float()
        # optional band of ignored 'just beyond budget' negatives (trajectory-proxy noise)
        valid = ~((tau > h) & (tau - h <= self.a.neg_margin))
        return dict(t=t, g=g, h=h, label=label, tau=tau, delta=delta, valid=valid)

    def cross(self, B):
        t = self.anchor_h[self._rint(B, len(self.anchor_h))]
        g = self.split_frames[self._rint(B, len(self.split_frames))]
        c = self.c
        h = self.rand_h(B)
        valid = c.frame_ep[t] != c.frame_ep[g]
        if self.a.label_source == "tdr":
            d = torch.linalg.norm(c.psi[t] - c.psi[g], dim=-1)
            valid &= d >= self.a.xneg_tdr_factor * c.d_cal[h.clamp(min=SKIP)]
        elif self.a.xneg_min_block_dist > 0:
            block_d = MECH.xneg_distance(c.state[t], c.state[g])      # Push-T: block position (px)
            valid &= block_d >= self.a.xneg_min_block_dist
        return dict(t=t, g=g, h=h, label=torch.zeros(B, device=self.dev), valid=valid)

    def imagined(self, model, B):
        """Predictor rollouts from logged histories (proposal eq. 7)."""
        a, c, n = self.a, self.c, self.a.imagine_blocks
        t = self.anchor_i[self._rint(B, len(self.anchor_i))]
        z_hist, a_past = self._history(t)
        logged_raw = self._raw_blocks(t, n)                                    # (B, n, 10)
        # action mode per row: logged | gaussian(0.1) | gaussian(0.25) | uniform
        u = self._rand(B)
        mode = torch.where(u < a.p_logged, 0, 1 + ((u - a.p_logged) / (1 - a.p_logged) * 3).long().clamp(max=2))
        noise = torch.randn(logged_raw.shape, device=self.dev, generator=self.g)
        sig = torch.tensor([0.0, 0.1, 0.25, 0.0], device=self.dev)[mode][:, None, None] * 2.0
        raw = (logged_raw + noise * sig).clamp(-1, 1)
        uni = torch.rand(logged_raw.shape, device=self.dev, generator=self.g) * 2 - 1
        raw = torch.where((mode == 3)[:, None, None], uni, raw)
        A = c.act_raw.shape[1]
        fut = (raw - c.act_mean.repeat(SKIP)) / c.act_std.repeat(SKIP)
        z_img, z_seq, a_seq = imagine(model, z_hist, a_past, fut)              # (B, n, D)

        # hindsight labels for logged-action rows: imagined frame k sits at t + 5k, and its
        # time-to-goal is the first hit from there towards g = t + delta (needs 5k <= delta)
        g, delta, _ = self._goal_and_tau(t, a.delta_max)
        label = torch.zeros(B, n, device=self.dev)
        valid = torch.zeros(B, n, dtype=torch.bool, device=self.dev)
        h = self.rand_h(B * n).view(B, n)
        for k in range(1, n + 1):
            tk = t + k * SKIP
            dk = delta - k * SKIP
            ok = (mode == 0) & (dk >= 0)
            if not ok.any():
                continue
            avail = c.frame_len[tk] - 1 - c.frame_step[tk]
            L = a.delta_max + 1
            idx = tk[:, None] + torch.arange(L, device=self.dev)[None]
            idx = torch.minimum(idx, (tk + avail)[:, None])
            tau_k = first_hit_time(self.feat[idx], self.feat[g], dk.clamp(min=0), self.reached)
            label[:, k - 1] = (h[:, k - 1] >= tau_k).float()
            valid[:, k - 1] = ok
        return dict(t=t, g=g, delta=delta, mode=mode, z_img=z_img, z_seq=z_seq, a_seq=a_seq,
                    raw_future=raw, h=h, label=label, valid=valid)


# ------------------------------------------------------------------
# losses
# ------------------------------------------------------------------

def bce_members(logits, label, valid=None):
    """Mean BCE over ensemble members and (valid) rows; logits (K,B), label (B,)."""
    l = F.binary_cross_entropy_with_logits(logits, label[None].expand_as(logits), reduction="none")
    if valid is not None:
        if not valid.any():
            return logits.sum() * 0.0
        l = l[:, valid]
    return l.mean()


def bellman_prestates(S, hb, cb, ib, budget):
    """Assemble (z_hist, a_past, zg, h, base_raw_block) for the Bellman backup from the
    logged hindsight/cross anchors and the imagined frames, subsampled to `budget` rows."""
    c, dev = S.c, S.dev
    parts = []
    for b in (hb, cb):
        t = b["t"]
        zh, ap = S._history(t)
        parts.append((zh, ap, c.z[b["g"]], S._raw_blocks(t, 1)[:, 0]))
    # imagined frames: z_seq index i in [HISTORY, HISTORY-1+n] (skip i=HISTORY-1, the real anchor,
    # already covered above); the base block is the one the rollout actually took next there
    n = ib["z_img"].shape[1]
    B = ib["z_img"].shape[0]
    rows = torch.arange(B, device=dev).repeat_interleave(n)
    i_idx = (torch.arange(n, device=dev) + HISTORY).repeat(B)
    zi = i_idx[:, None] + torch.arange(-(HISTORY - 1), 1, device=dev)[None]
    ai = i_idx[:, None] + torch.arange(-(HISTORY - 1), 0, device=dev)[None]
    zh = ib["z_seq"][rows[:, None], zi]
    ap = ib["a_seq"][rows[:, None], ai]
    base = ib["raw_future"][rows, (i_idx - HISTORY + 1).clamp(max=n - 1)]
    parts.append((zh, ap, c.z[ib["g"]][rows], base))
    zh = torch.cat([p[0] for p in parts]); ap = torch.cat([p[1] for p in parts])
    zg = torch.cat([p[2] for p in parts]); base = torch.cat([p[3] for p in parts])
    if len(zh) > budget:
        keep = torch.randperm(len(zh), device=dev, generator=S.g)[:budget]
        zh, ap, zg, base = zh[keep], ap[keep], zg[keep], base[keep]
    h = S.rand_h(len(zh), min_h=SKIP)
    return zh, ap, zg, h, base


def train_step(model, critic, target, S, args, opt):
    c = S.c
    hb, cb = S.hindsight(args.batch_hindsight), S.cross(args.batch_cross)
    ib = S.imagined(model, args.batch_imagined)

    # -- L_viab: logged hindsight, cross-episode negatives, imagined-with-logged-actions
    l_h = bce_members(critic(c.z[hb["t"]], c.z[hb["g"]], hb["h"]), hb["label"], hb["valid"])
    l_x = bce_members(critic(c.z[cb["t"]], c.z[cb["g"]], cb["h"]), cb["label"], cb["valid"])
    n = ib["z_img"].shape[1]
    zg_i = c.z[ib["g"]][:, None].expand(-1, n, -1).reshape(-1, c.z.shape[1])
    l_i = bce_members(critic(ib["z_img"].reshape(-1, c.z.shape[1]), zg_i, ib["h"].reshape(-1)),
                      ib["label"].reshape(-1), ib["valid"].reshape(-1))
    l_viab = l_h + args.w_cross * l_x + args.w_imagined * l_i

    # -- L_bellman: model-based backup with the EMA target ensemble
    zh, ap, zg, h_b, base = bellman_prestates(S, hb, cb, ib, args.batch_bellman)
    cand = sample_action_blocks(base, args.bellman_actions, c.act_mean, c.act_std, generator=S.g)
    tgt = bellman_target(model, target, zh, ap, zg, h_b, cand, kappa=args.bellman_kappa, temp=args.bellman_temp)
    logits_b = critic(zh[:, -1], zg, h_b)
    l_bell = bce_members(logits_b, tgt)

    # -- L_mono: more budget can never lower viability
    z_all = torch.cat([c.z[hb["t"]], c.z[cb["t"]], ib["z_img"].reshape(-1, c.z.shape[1])])
    zg_all = torch.cat([c.z[hb["g"]], c.z[cb["g"]], zg_i])
    h_m = S.rand_h(len(z_all))
    h_m = torch.minimum(h_m, torch.full_like(h_m, args.h_max - SKIP))
    p_lo = torch.sigmoid(critic(z_all, zg_all, h_m))
    p_hi = torch.sigmoid(critic(z_all, zg_all, h_m + SKIP))
    l_mono = F.relu(p_lo - p_hi).mean()

    loss = l_viab + args.lambda_bellman * l_bell + args.lambda_mono * l_mono
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(critic.parameters(), args.grad_clip)
    opt.step()
    with torch.no_grad():
        for p_t, p in zip(target.parameters(), critic.parameters()):
            p_t.lerp_(p, 1.0 - args.ema)
    # BCE against a soft target has an entropy floor (~0.69 at target 0.5), so also report the
    # plain residual, which is what "consistency" means here
    bell_resid = (torch.sigmoid(logits_b).mean(0) - tgt).abs().mean().item()
    return dict(loss=loss.item(), viab_h=l_h.item(), viab_x=l_x.item(), viab_i=l_i.item(),
                bellman=l_bell.item(), bell_resid=bell_resid, mono=l_mono.item(),
                pos_frac=hb["label"][hb["valid"]].mean().item(),
                img_labeled=ib["valid"].float().mean().item(),
                bell_target=tgt.mean().item())


# ------------------------------------------------------------------
# evaluation (held-out episodes, fixed batches)
# ------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, critic, target, S, args):
    c = S.c
    S.g.manual_seed(args.seed + 12345)         # identical eval draws every call
    out = {}

    hb = S.hindsight(args.eval_n)
    p, _ = critic.prob(c.z[hb["t"]], c.z[hb["g"]], hb["h"])
    p, y, v = p.cpu().numpy(), hb["label"].cpu().numpy(), hb["valid"].cpu().numpy()
    out.update(hind_bce=float(F.binary_cross_entropy(torch.tensor(p[v]), torch.tensor(y[v])).item()),
               hind_auroc=auroc(p[v], y[v]), hind_ece=ece(p[v], y[v]),
               hind_acc=float(((p[v] > 0.5) == (y[v] > 0.5)).mean()), hind_pos_frac=float(y[v].mean()))
    # mean V by budget, split by whether the goal is within budget (tau <= h)
    if args.label_source == "tdr":
        # the same held-out pairs scored against the state predicate the tdr rule replaces
        ys = (hb["h"] >= S.tau_state(hb["t"], hb["g"], hb["delta"])).float().cpu().numpy()
        out.update(hind_auroc_state=auroc(p, ys), hind_acc_state=float(((p > 0.5) == (ys > 0.5)).mean()),
                   hind_label_agreement=float((y == ys).mean()))
    hh, tau = hb["h"].cpu().numpy(), hb["tau"].cpu().numpy()
    out["v_by_h"] = {int(h): [float(p[(hh == h) & (tau <= h)].mean()) if ((hh == h) & (tau <= h)).any() else None,
                              float(p[(hh == h) & (tau > h)].mean()) if ((hh == h) & (tau > h)).any() else None]
                     for h in np.unique(hh)}

    cb = S.cross(args.eval_n // 4)
    pc, _ = critic.prob(c.z[cb["t"]], c.z[cb["g"]], cb["h"])
    out["cross_mean_v"] = float(pc[cb["valid"]].mean())
    out["cross_valid_frac"] = float(cb["valid"].float().mean())

    ib = S.imagined(model, args.eval_n // 8)
    n = ib["z_img"].shape[1]
    zg_i = c.z[ib["g"]][:, None].expand(-1, n, -1).reshape(-1, c.z.shape[1])
    pi, si = critic.prob(ib["z_img"].reshape(-1, c.z.shape[1]), zg_i, ib["h"].reshape(-1))
    pi, yi, vi = pi.cpu().numpy(), ib["label"].reshape(-1).cpu().numpy(), ib["valid"].reshape(-1).cpu().numpy()
    out.update(imag_auroc=auroc(pi[vi], yi[vi]), imag_ece=ece(pi[vi], yi[vi]),
               imag_acc=float(((pi[vi] > 0.5) == (yi[vi] > 0.5)).mean()) if vi.any() else float("nan"),
               imag_std_logged=float(si.cpu().numpy()[vi].mean()),
               imag_std_offdata=float(si.cpu().numpy()[~vi].mean()) if (~vi).any() else float("nan"))

    # monotonicity violations across the whole grid on held-out logged pairs
    viol, tot = 0, 0
    for h in range(0, args.h_max - SKIP + 1, SKIP):
        hv = torch.full_like(hb["h"], h)
        lo, _ = critic.prob(c.z[hb["t"]], c.z[hb["g"]], hv)
        hi, _ = critic.prob(c.z[hb["t"]], c.z[hb["g"]], hv + SKIP)
        viol += int((lo > hi + 0.01).sum()); tot += len(lo)
    out["mono_violation_rate"] = viol / max(tot, 1)

    # Bellman residual on held-out logged anchors
    zh, ap = S._history(hb["t"])
    base = S._raw_blocks(hb["t"], 1)[:, 0]
    h_b = S.rand_h(len(zh), min_h=SKIP)
    cand = sample_action_blocks(base, args.bellman_actions, c.act_mean, c.act_std, generator=S.g)
    tgt = bellman_target(model, target, zh, ap, c.z[hb["g"]], h_b, cand, kappa=args.bellman_kappa, temp=args.bellman_temp)
    pb, _ = critic.prob(zh[:, -1], c.z[hb["g"]], h_b)
    out["bellman_abs_residual"] = float((pb - tgt).abs().mean())
    return out


# ------------------------------------------------------------------
# main
# ------------------------------------------------------------------

def run(args):
    global MECH, GOAL_REACHED
    MECH = ENV_MECHANICS[args.env]
    GOAL_REACHED = MECH.goal_reached
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    log(f"[setup] cache={args.cache} out={out} device={DEV}")
    cache = LatentCache(args.cache, DEV)
    args.training_only = True
    train_eps, val_eps = cache.split_episodes(args.val_frac, args.seed)
    evaluation_eps = cache.evaluation_eps
    if args.exclude_tdr_holdout:
        if not args.tdr:
            raise ValueError("--exclude-tdr-holdout requires --tdr")
        split_cfg = torch.load(args.tdr, map_location="cpu", weights_only=False)["cfg"]
        if split_cfg.get("diagnostic_scope") != "training_only":
            raise ValueError("A train-only TDR is required")
        log(f"[setup] excluded {len(evaluation_eps)} fixed TDR evaluation episodes")
    log(f"[setup] {cache.n_frames} frames / {cache.n_ep} episodes -> train {len(train_eps)} val {len(val_eps)} episodes")
    if args.label_source == "tdr":
        assert args.tdr, "--label-source tdr needs --tdr <tdr_full_s{seed}.pt>"
        tcfg = cache.attach_tdr(args.tdr)
        g_cal = torch.Generator(device=DEV); g_cal.manual_seed(args.seed + 7)
        d_cal = cache.calibrate_tdr(train_eps, args.h_max, g_cal)
        cache.tdr_radius = float(args.tdr_radius) if args.tdr_radius else float(d_cal[args.tdr_radius_gap_step])
        args.tdr_radius_resolved = cache.tdr_radius
        args.tdr_d_cal = [round(float(v), 4) for v in d_cal]
        args.tdr_cfg = tcfg
        log(f"[setup] tdr labels: {args.tdr} psi {tuple(cache.psi.shape)} | radius {cache.tdr_radius:.3f} "
            f"(gap {args.tdr_radius_gap_step}) | d_cal {[round(float(d_cal[k]), 2) for k in range(0, args.h_max + 1, SKIP)]} "
            f"| cross valid iff d >= {args.xneg_tdr_factor:g} * d_cal(max(h, {SKIP}))")

    ckpt_dir = cache.meta["ckpt_dir"] or str(MECH.ckpt_dir(Path(args.root)))
    model = load_lewm(Path(ckpt_dir), device=DEV)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    critic = ViabilityCritic(z_dim=cache.z.shape[1], h_max=args.h_max, n_members=args.members,
                             hidden=args.hidden, depth=args.depth).to(DEV)
    train_frames = cache.frames_of(train_eps)
    critic.set_input_stats(cache.z[train_frames].mean(0), cache.z[train_frames].std(0))
    target = ViabilityCritic(z_dim=cache.z.shape[1], h_max=args.h_max, n_members=args.members,
                             hidden=args.hidden, depth=args.depth).to(DEV)
    target.load_state_dict(critic.state_dict())
    for p in target.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(critic.parameters(), lr=args.lr, weight_decay=args.wd)
    log(f"[setup] critic: {sum(p.numel() for p in critic.parameters()) / 1e6:.2f}M params, "
        f"{args.members} members, h_max={args.h_max}, delta_max={args.delta_max}")

    g_train = torch.Generator(device=DEV); g_train.manual_seed(args.seed)
    g_val = torch.Generator(device=DEV); g_val.manual_seed(args.seed + 1)
    S_train = Sampler(cache, train_eps, args, g_train)
    S_val = Sampler(cache, val_eps, args, g_val)
    log(f"[setup] anchors: train hindsight={len(S_train.anchor_h)} imagined={len(S_train.anchor_i)} | "
        f"val hindsight={len(S_val.anchor_h)} imagined={len(S_val.anchor_i)}")

    history, t0, running = [], time.time(), {}
    for step in range(1, args.steps + 1):
        critic.train()
        m = train_step(model, critic, target, S_train, args, opt)
        for k, v in m.items():
            running[k] = running.get(k, 0.0) + v
        if step % args.log_every == 0:
            avg = {k: v / args.log_every for k, v in running.items()}
            running = {}
            log(f"[train] step {step}/{args.steps} loss={avg['loss']:.4f} viab(h/x/i)={avg['viab_h']:.3f}/"
                f"{avg['viab_x']:.3f}/{avg['viab_i']:.3f} bellman={avg['bellman']:.3f} (resid {avg['bell_resid']:.3f}) "
                f"mono={avg['mono']:.4f} "
                f"pos={avg['pos_frac']:.2f} tgt={avg['bell_target']:.2f} {(time.time() - t0) / step * 1000:.0f}ms/step")
        if step % args.eval_every == 0 or step == args.steps:
            critic.eval()
            ev = evaluate(model, critic, target, S_val, args)
            ev["step"] = step
            history.append(ev)
            log(f"[eval]  step {step}: hind auroc={ev['hind_auroc']:.3f} ece={ev['hind_ece']:.3f} "
                f"acc={ev['hind_acc']:.3f} | imag auroc={ev['imag_auroc']:.3f} ece={ev['imag_ece']:.3f} "
                f"| cross V={ev['cross_mean_v']:.3f} | mono viol={ev['mono_violation_rate']:.4f} "
                f"| bellman resid={ev['bellman_abs_residual']:.3f} "
                f"| std logged/offdata={ev['imag_std_logged']:.3f}/{ev['imag_std_offdata']:.3f}"
                + (f" | vs state labels: auroc={ev['hind_auroc_state']:.3f} acc={ev['hind_acc_state']:.3f} "
                   f"agree={ev['hind_label_agreement']:.3f}" if "hind_auroc_state" in ev else ""))
            torch.save(dict(critic=critic.state_dict(), target=target.state_dict(), args=vars(args), step=step,
                            train_episodes=cache.episode_ids[train_eps], val_episodes=cache.episode_ids[val_eps],
                            evaluation_episodes=evaluation_eps, cache=str(args.cache)),
                       out / "critic.pt")
            (out / "metrics.json").write_text(json.dumps(dict(args=vars(args), history=history), indent=2))
    log(f"[done] wrote {out / 'critic.pt'} and {out / 'metrics.json'}")
    return history


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=str(Path(__file__).resolve().parent.parent))
    p.add_argument("--env", default="pusht", choices=sorted(ENV_MECHANICS),
                   help="success predicate + cross-negative distance come from this env's mechanics")
    p.add_argument("--cache", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--exclude-tdr-holdout", action="store_true",
                   help="exclude the fixed TDR evaluation split from critic training and validation")
    # semantics
    p.add_argument("--h-max", type=int, default=50, help="largest budget (env steps); grid is multiples of 5")
    p.add_argument("--delta-max", type=int, default=60, help="largest hindsight offset; > h_max gives beyond-budget negatives")
    p.add_argument("--neg-margin", type=int, default=0, help="ignore hindsight negatives with 0 < tau-h <= margin")
    p.add_argument("--label-source", default="state", choices=("state", "tdr"),
                   help="state: env predicate / block distance on the dataset's state column (privileged); "
                        "tdr: TDR-distance ball and calibrated gap distances on psi(z) only (pixel-only)")
    p.add_argument("--tdr", default=None, help="tdr_full_s{seed}.pt from gas_mpc_prepare.py (label_source=tdr)")
    p.add_argument("--tdr-radius", type=float, default=None, help="override the tau ball radius in TDR units")
    p.add_argument("--tdr-radius-gap-step", type=int, default=2,
                   help="default radius = median TDR distance at this step gap (diag: best r sits at gap 2-3)")
    p.add_argument("--xneg-tdr-factor", type=float, default=1.25,
                   help="cross negative at budget h valid iff TDR distance >= factor * d_cal(max(h, 5)); 1.25 "
                        "reproduces the state filter's keep rate on Push-T (latent_label_substitute_diag)")
    p.add_argument("--xneg-min-block-dist", "--xneg-min-dist", dest="xneg_min_block_dist", type=float, default=100.0,
                   help="cross-episode negatives only when EnvMechanics.xneg_distance >= this (Push-T: block "
                        "position px; Reacher: joint-space rad); 0 = all pairs")
    # batches
    p.add_argument("--batch-hindsight", type=int, default=256)
    p.add_argument("--batch-cross", type=int, default=64)
    p.add_argument("--batch-imagined", type=int, default=64, help="rollouts per step")
    p.add_argument("--imagine-blocks", type=int, default=5, help="predictor blocks per rollout (5 = 25 env steps)")
    p.add_argument("--p-logged", type=float, default=0.5, help="fraction of rollouts driven by the logged actions (labelled)")
    p.add_argument("--batch-bellman", type=int, default=128, help="pre-states per step for the backup")
    p.add_argument("--bellman-actions", type=int, default=8, help="M sampled action blocks in the max")
    p.add_argument("--bellman-kappa", type=float, default=1.0, help="pessimism: target = max_a (mean - kappa*std)")
    p.add_argument("--bellman-temp", type=float, default=0.0, help="0 = hard max over actions, >0 = soft")
    # loss weights
    p.add_argument("--w-cross", type=float, default=1.0)
    p.add_argument("--w-imagined", type=float, default=1.0)
    p.add_argument("--lambda-bellman", type=float, default=1.0)
    p.add_argument("--lambda-mono", type=float, default=1.0)
    # model / optimisation
    p.add_argument("--members", type=int, default=5)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--wd", type=float, default=1e-5)
    p.add_argument("--ema", type=float, default=0.995)
    p.add_argument("--grad-clip", type=float, default=5.0)
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--eval-n", type=int, default=4096)
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
