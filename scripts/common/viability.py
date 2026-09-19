"""Latent viability critic -- docs/viability-proposal/viability-value.tex, Sec. 3-4, Alg. 1.

V_psi(z, z_g, h) in [0, 1] approximates P(reach goal g within h env steps | latent z) as an
ensemble of K MLP heads over the frozen LeWM CLS latent. This module holds what the training
script and (later) the planner share: the critic, horizon features, one-block latent dynamics
on top of the frozen predictor, candidate-action sampling for the Bellman backup, and the
hindsight label rules. Nothing here touches pixels -- `viability_cache.py` encodes frames once
and everything downstream is latent-only.

Conventions
-----------
* `h` is in ENV STEPS. The predictor advances SKIP=5 env steps per block, so the Bellman
  backup relates V(., h) to V(., h - SKIP). The critic accepts any integer h.
* Actions handed to the predictor are Z-SCORED with the FULL dataset's mean/std (CLAUDE.md,
  "Action normalization"). Raw [-1, 1] actions are only used to sample perturbations.
* The predictor is not Markov in z alone -- it conditions on the last HISTORY=3 latents and
  the action block taken at each. The critic reads a single z (as the proposal writes it);
  the dynamics helpers carry the history explicitly. See `history_at`.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

SKIP = 5      # env steps per predictor block; one action block = 5 raw actions = 10 dims
HISTORY = 3   # latents / action blocks the predictor conditions on
Z_DIM = 192


# ------------------------------------------------------------------
# critic
# ------------------------------------------------------------------

def horizon_features(h, h_max, n_freq=4):
    """(B,) env-step budgets -> (B, 1 + 2*n_freq) features: h/h_max plus sinusoids of it."""
    x = (h.float() / float(h_max)).unsqueeze(-1)
    feats = [x]
    for k in range(n_freq):
        w = math.pi * (2 ** k)
        feats += [torch.sin(w * x), torch.cos(w * x)]
    return torch.cat(feats, dim=-1)


def _mlp(in_dim, hidden, depth):
    layers, d = [], in_dim
    for _ in range(depth):
        layers += [nn.Linear(d, hidden), nn.SiLU()]
        d = hidden
    layers.append(nn.Linear(d, 1))
    return nn.Sequential(*layers)


class ViabilityCritic(nn.Module):
    """Ensemble of K independent heads, each mapping [z, z_g, z - z_g, phi(h)] -> logit.

    `z - z_g` is a representational convenience (lets a head recover the L2 signal cheaply);
    the function class is still V(z, z_g, h). Inputs are standardised per latent dim with
    statistics from the training cache (`set_input_stats`)."""

    def __init__(self, z_dim=Z_DIM, h_max=50, n_members=5, hidden=512, depth=3, n_freq=4):
        super().__init__()
        self.z_dim, self.h_max, self.n_members, self.n_freq = z_dim, h_max, n_members, n_freq
        in_dim = 3 * z_dim + 1 + 2 * n_freq
        self.members = nn.ModuleList([_mlp(in_dim, hidden, depth) for _ in range(n_members)])
        self.register_buffer("z_mean", torch.zeros(z_dim))
        self.register_buffer("z_std", torch.ones(z_dim))

    def set_input_stats(self, z_mean, z_std):
        self.z_mean.copy_(torch.as_tensor(z_mean, dtype=torch.float32))
        self.z_std.copy_(torch.as_tensor(z_std, dtype=torch.float32).clamp_min(1e-6))

    def features(self, z, zg, h):
        z = (z - self.z_mean) / self.z_std
        zg = (zg - self.z_mean) / self.z_std
        return torch.cat([z, zg, z - zg, horizon_features(h, self.h_max, self.n_freq)], dim=-1)

    def forward(self, z, zg, h):
        """-> (K, B) logits."""
        x = self.features(z, zg, h)
        return torch.stack([m(x).squeeze(-1) for m in self.members], dim=0)

    def prob(self, z, zg, h):
        """-> (mean, std) over members of sigmoid(logit), each (B,)."""
        p = torch.sigmoid(self.forward(z, zg, h))
        return p.mean(0), p.std(0, unbiased=False)


# ------------------------------------------------------------------
# latent dynamics on the frozen predictor
# ------------------------------------------------------------------

@torch.no_grad()
def predictor_step(model, z_hist, a_hist):
    """One block forward. z_hist (B, HISTORY, D); a_hist (B, HISTORY, SKIP*A) z-scored, where
    a_hist[:, i] is the block taken AT frame z_hist[:, i] (so the last one is the candidate
    block applied now). Returns the latent SKIP env steps after z_hist[:, -1], shape (B, D)."""
    return model.predict(z_hist, model.action_encoder(a_hist))[:, -1]


@torch.no_grad()
def imagine(model, z_hist, a_past, a_future):
    """Autoregressive rollout mirroring jepa.JEPA.rollout / planning_cost_gate.latent_rollout.

    z_hist   (B, HISTORY, D)       real latents at t-10, t-5, t
    a_past   (B, HISTORY-1, 10)    z-scored blocks taken at t-10 and t-5
    a_future (B, n, 10)            z-scored blocks taken at t, t+5, ..., t+5(n-1)

    Returns (z_img, z_seq, a_seq):
      z_img (B, n, D)              imagined latents at t+5, ..., t+5n
      z_seq (B, HISTORY+n, D)      [real history, imagined...] -- index i >= HISTORY-1 is the
                                   frame t+5(i-HISTORY+1); use `history_at(z_seq, a_seq, i)`
      a_seq (B, HISTORY-1+n, 10)   the block taken at each frame of z_seq except the last
    """
    n = a_future.shape[1]
    z_seq = z_hist
    a_seq = torch.cat([a_past, a_future[:, :1]], dim=1)
    outs = []
    for k in range(n):
        z_next = predictor_step(model, z_seq[:, -HISTORY:], a_seq[:, -HISTORY:])
        outs.append(z_next)
        z_seq = torch.cat([z_seq, z_next[:, None]], dim=1)
        if k + 1 < n:
            a_seq = torch.cat([a_seq, a_future[:, k + 1:k + 2]], dim=1)
    return torch.stack(outs, dim=1), z_seq, a_seq


def history_at(z_seq, a_seq, i):
    """Predictor context for frame index i of an `imagine` sequence (i >= HISTORY-1):
    (z_seq[:, i-2:i+1], a_seq[:, i-2:i]) -- the 3 latents ending at i and the 2 blocks taken
    at the two frames before it. A candidate block appended to the latter completes the input
    to `predictor_step`."""
    return z_seq[:, i - HISTORY + 1:i + 1], a_seq[:, i - HISTORY + 1:i]


def sample_action_blocks(base_raw, m, act_mean, act_std, sigmas=(0.1, 0.25), low=-1.0, high=1.0,
                         generator=None):
    """(B, 10) raw base block -> (B, m, 10) Z-SCORED candidate blocks.

    Slot 0 is the base block itself; then Gaussian perturbations of it at each sigma (raw
    units, clipped to the env's [low, high]); any remaining slots are uniform random. This
    mirrors planning_cost_gate.make_candidates -- a spread from on-data to random -- so the
    Bellman max sees both what the data did and what a planner might try."""
    B, A = base_raw.shape
    dev = base_raw.device
    blocks = [base_raw]
    per_sigma = max(1, (m - 1) // (len(sigmas) + 1))
    for s in sigmas:
        for _ in range(per_sigma):
            if len(blocks) >= m:
                break
            noise = torch.randn(B, A, device=dev, generator=generator) * s * (high - low)
            blocks.append((base_raw + noise).clamp(low, high))
    while len(blocks) < m:
        blocks.append(torch.rand(B, A, device=dev, generator=generator) * (high - low) + low)
    raw = torch.stack(blocks[:m], dim=1)                                   # (B, m, 10)
    mean = act_mean.repeat(A // act_mean.numel())
    std = act_std.repeat(A // act_std.numel())
    return (raw - mean) / std


@torch.no_grad()
def bellman_target(model, target_critic, z_hist, a_past, zg, h, cand_blocks, kappa=1.0, temp=0.0):
    """Model-based backup  V(z, g, h) <- max_a V_tgt(F(hist, a), g, h - SKIP)  (proposal eq. 12).

    z_hist (B, 3, D), a_past (B, 2, 10), zg (B, D), h (B,) with h >= SKIP,
    cand_blocks (B, M, 10) z-scored. The ensemble is aggregated pessimistically
    (mean - kappa*std, clipped to [0, 1]) BEFORE the max, so the max-over-actions cannot
    chase a single head's optimism on an off-manifold imagined state. temp > 0 replaces the
    hard max with temp*logsumexp(v/temp) - temp*log(M), which lies in [mean, max]."""
    B, M, A = cand_blocks.shape
    D = z_hist.shape[-1]
    zh = z_hist[:, None].expand(B, M, HISTORY, D).reshape(B * M, HISTORY, D)
    ah = torch.cat([a_past[:, None].expand(B, M, HISTORY - 1, A),
                    cand_blocks[:, :, None]], dim=2).reshape(B * M, HISTORY, A)
    z_next = predictor_step(model, zh, ah)
    zg_r = zg[:, None].expand(B, M, D).reshape(B * M, D)
    h_r = (h - SKIP).repeat_interleave(M)
    mean, std = target_critic.prob(z_next, zg_r, h_r)
    v = (mean - kappa * std).clamp(0.0, 1.0).view(B, M)
    if temp > 0:
        return temp * torch.logsumexp(v / temp, dim=1) - temp * math.log(M)
    return v.max(dim=1).values


# ------------------------------------------------------------------
# labels and metrics
# ------------------------------------------------------------------

def first_hit_time(states, goal, delta, goal_reached):
    """First k <= delta at which the logged trajectory already satisfies the goal predicate.

    states (B, L, S) the frames t, t+1, ..., t+L-1; goal (B, S) the frame t+delta; delta (B,)
    with delta <= L-1. Returns tau (B,) with tau <= delta -- k = delta always hits, since it
    IS the goal frame. tau replaces delta as the trajectory-derived time-to-goal: the expert
    may enter the env's success tolerance (Push-T: 20 px / pi/9) before the goal frame, and
    the label should be 1[h >= tau], not 1[h >= delta]."""
    B, L, _ = states.shape
    k = torch.arange(L, device=states.device)[None].expand(B, L)
    hit = goal_reached(states, goal[:, None].expand_as(states)) & (k <= delta[:, None])
    return hit.int().argmax(dim=1)


def ece(prob, label, n_bins=10):
    """Expected calibration error with equal-width bins; prob/label numpy (N,)."""
    bins = np.clip((prob * n_bins).astype(int), 0, n_bins - 1)
    out = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            out += m.mean() * abs(prob[m].mean() - label[m].mean())
    return float(out)


def auroc(score, label):
    """Rank-based AUROC (Mann-Whitney U with tie-averaged ranks); NaN if one class is missing."""
    pos = label > 0.5
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    s = score[order]
    ranks = np.empty(len(score))
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


# ------------------------------------------------------------------
# hitting-time distribution head (revised critic, proposal sec. "Revised Critic")
# ------------------------------------------------------------------

class HittingTimeHead(nn.Module):
    """p_psi(T = b | z, z_g) over predictor-step bins b in {0, ..., B_max} plus one '> B_max'
    class; the viability CDF is V(z, z_g, h) = sum_{b <= floor(h / SKIP)} p(b).

    Replaces the horizon-conditioned binary critic: no Bellman backup, no target network, no
    horizon embedding, no ensemble. Monotonicity in h and directedness in (z, z_g) hold by
    construction. Input is the proposal's ordered pair [l(z), l(z_g)] with l = parameter-free
    layer norm (applied after per-dim standardisation with training-cache statistics);
    `input_diff` optionally appends l(z) - l(z_g) (ablation, off by default).

    head='softmax': the proposal's 47-way softmax.
    head='hazard':  discrete-time survival parameterisation, lambda_b = P(T = b | T >= b) =
                    sigmoid(logit_b); p(b) = lambda_b prod_{b' < b}(1 - lambda_b'),
                    p(> B_max) = prod_b (1 - lambda_b). Same pmf interface, ordinal by
                    construction (ablation).

    Exposes the ensemble critic's `prob(z, zg, h) -> (V, std)` (std = 0) so
    viability_eval_audit.py and viability_live_rollout.py score it unchanged, and
    `h_max = B_max * SKIP` for their horizon grids."""

    def __init__(self, z_dim=Z_DIM, b_max=45, hidden=(512, 256), head="softmax", input_diff=False):
        super().__init__()
        assert head in ("softmax", "hazard"), head
        self.z_dim, self.b_max, self.head, self.input_diff = z_dim, b_max, head, input_diff
        self.n_classes = b_max + 2                       # 0..b_max and '> b_max'
        self.h_max = b_max * SKIP
        in_dim = (3 if input_diff else 2) * z_dim
        out_dim = self.n_classes if head == "softmax" else b_max + 1
        layers, d = [], in_dim
        for w in hidden:
            layers += [nn.Linear(d, w), nn.SiLU()]
            d = w
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)
        self.register_buffer("z_mean", torch.zeros(z_dim))
        self.register_buffer("z_std", torch.ones(z_dim))
        self.register_buffer("bins", torch.arange(self.n_classes, dtype=torch.float32))

    def set_input_stats(self, z_mean, z_std):
        self.z_mean.copy_(torch.as_tensor(z_mean, dtype=torch.float32))
        self.z_std.copy_(torch.as_tensor(z_std, dtype=torch.float32).clamp_min(1e-6))

    def _ell(self, z):
        return torch.nn.functional.layer_norm((z - self.z_mean) / self.z_std, (self.z_dim,))

    def features(self, z, zg):
        a, b = self._ell(z), self._ell(zg)
        return torch.cat([a, b, a - b], dim=-1) if self.input_diff else torch.cat([a, b], dim=-1)

    def log_pmf(self, z, zg):
        """-> (B, n_classes) log p(T = b | z, zg); last column is the '> B_max' class."""
        out = self.net(self.features(z, zg))
        if self.head == "softmax":
            return torch.log_softmax(out, dim=-1)
        log_lam = torch.nn.functional.logsigmoid(out)                 # log lambda_b
        log_surv = torch.nn.functional.logsigmoid(-out)               # log (1 - lambda_b)
        cum = torch.cumsum(log_surv, dim=-1)                          # sum_{b' <= b}
        prev = torch.cat([torch.zeros_like(cum[..., :1]), cum[..., :-1]], dim=-1)
        return torch.cat([log_lam + prev, cum[..., -1:]], dim=-1)

    def cdf(self, z, zg, h):
        """V(z, zg, h) = P(T <= floor(h / SKIP)); h (B,) in env steps."""
        p = self.log_pmf(z, zg).exp()
        m = torch.div(h.float(), SKIP, rounding_mode="floor").long().clamp(-1, self.b_max)
        c = torch.cumsum(p, dim=-1)
        c = torch.cat([torch.zeros_like(c[..., :1]), c], dim=-1)     # index m+1, so m=-1 -> 0
        return c.gather(-1, (m + 1)[..., None]).squeeze(-1).clamp(0.0, 1.0)

    def prob(self, z, zg, h):
        v = self.cdf(z, zg, h)
        return v, torch.zeros_like(v)

    def expected_bins(self, z, zg):
        """E[T] in predictor steps, counting '> B_max' as B_max + 1 (a lower bound)."""
        return (self.log_pmf(z, zg).exp() * self.bins).sum(-1)

    def forward(self, z, zg):
        return self.log_pmf(z, zg)


def hitting_time_bin(t_env, b_max):
    """Env-step hitting time -> class index. t_env < 0 encodes 'beyond the search bound /
    disconnected' -> the '> B_max' class. Otherwise ceil(t / SKIP), which makes
    P(bin <= h/SKIP) == P(t <= h) on the 5-step h grid; anything past B_max also lands in
    the last class."""
    t = torch.as_tensor(t_env)
    b = torch.div(t + SKIP - 1, SKIP, rounding_mode="floor")
    return torch.where(t < 0, torch.full_like(b, b_max + 1), b.clamp(max=b_max + 1)).long()


def hitting_time_loss(log_pmf, label_bin, censored, weight):
    """Proposal losses on one batch. label_bin (B,) class index; censored (B,) bool: True
    means the label is only 'T > c' with c = label_bin, i.e. loss = -log sum_{b > c} p(b)
    (the '> B_max' class is the censored case c = B_max, and is where the disconnected
    weight applies); False means exact, loss = -log p(label_bin). weight (B,) per-row.
    Returns the weighted mean."""
    B, C = log_pmf.shape
    label_bin = label_bin.clamp(max=C - 1)
    # 'T > c' with c >= B_max has the single-class tail {'> B_max'}: same as exact on it
    censored = censored & (label_bin < C - 1)
    exact = -log_pmf.gather(-1, label_bin[:, None]).squeeze(-1)
    b_idx = torch.arange(C, device=log_pmf.device)[None]
    tail_mask = b_idx > label_bin[:, None]
    tail = -torch.logsumexp(log_pmf.masked_fill(~tail_mask, float("-inf")), dim=-1)
    loss = torch.where(censored, tail, exact)
    return (loss * weight).sum() / weight.sum().clamp_min(1e-8)
