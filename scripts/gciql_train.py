"""GCIQL baseline trained with the LeWM paper's protocol, on frozen LeWM latents.

Port of stable-worldmodel's scripts/train/gciql.py (the recipe behind the paper's GCIQL
numbers): transformer V(s,g) and Q(s,a,g) heads with EMA targets, expectile 0.9, gamma 0.99,
sparse -1/0 reward (terminal iff the frame IS the goal frame), then AWR policy extraction
(alpha 10, advantage V(s')-V(s) under the phase-1 value net). Two phases, 100 epochs each,
batch 32, AdamW 3e-4, grad-clip 1.0, 3-frame history, 90/10 random clip split, upstream's
goal-sampling mixtures. Hyperparameters: config/gciql/gciql.yaml.

One model is trained for every goal type (critic: random / geometric-future / current goals;
actor: random / uniform-future goals), and the same checkpoint is evaluated on
same25/50/100 and cross by `gas_mpc_eval.py +mpc.method=gciql`.

Deviations from upstream:
  * encoder: the frozen pretrained LeWM encoder (192-d projected CLS latent) instead of
    DINOv2-small patches, so each frame is one token. Training reads the latents already in
    outputs/<env>/cache_train.npz (gas_mpc_prepare.py encode); evaluation encodes live frames
    with the same encoder and the same ImageNet normalisation.
  * data: that cache is physically training-only -- the final-evaluation episodes are not in
    it, so neither clips nor random goals can come from them. Every other episode is used.
  * actions: z-scored with the cache's act_mean/act_std (the stats gas_mpc_eval.py
    de-normalises with), not a fit over the full column.
  * plain PyTorch loop instead of Lightning/spt.Manager (same losses, optimiser and EMA);
    resumable per epoch.

    python scripts/gciql_train.py                                  # Push-T, paper config
    GAS_MPC_ENV=reacher python scripts/gciql_train.py precision=32
"""

import copy
import logging
import sys
import time
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from stable_worldmodel.wm.gcrl import module as gcrl_nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from gas_mpc_prepare import ENV, MECH, OUT, load_cache  # noqa: E402

OmegaConf.register_new_resolver("gas_mpc_out", lambda: str(OUT), replace=True)
logger = logging.getLogger(__name__)
CKPT = OUT / "gciql" / "checkpoints"


# ----------------------------------------------------------------------------------------
# data: clips and goals sampled straight from the cached latent table
# ----------------------------------------------------------------------------------------

class LatentClips:
    """Upstream's HDF5Dataset(num_steps=history+td_offset, frameskip=1) + GoalDataset, on rows
    of the cached latent table. A clip is rows s..s+3; the 'current' frame is s+2."""

    def __init__(self, cfg, device):
        cache = load_cache()                                  # validates training-only provenance
        store = device if cfg.latents_on_gpu else "cpu"
        self.z = torch.from_numpy(cache["z"]).to(store)
        act = (cache["action"] - cache["act_mean"]) / np.where(cache["act_std"] > 0, cache["act_std"], 1.0)
        self.action = torch.from_numpy(np.nan_to_num(act).astype(np.float32)).to(store)
        self.act_mean, self.act_std = cache["act_mean"], cache["act_std"]
        self.episode_id = cache["episode_id"]
        self.span = cfg.dinowm.history_size + cfg.dinowm.td_offset
        off, length = cache["ep_offset"].astype(np.int64), cache["ep_len"].astype(np.int64)
        starts, last = [], []
        for o, n in zip(off, length):
            if n >= self.span:
                starts.append(np.arange(o, o + n - self.span + 1))
                last.append(np.full(n - self.span + 1, o + n - 1))
        self.starts = torch.from_numpy(np.concatenate(starts)).to(device)
        self.ep_last = torch.from_numpy(np.concatenate(last)).to(device)   # last row of the clip's episode
        self.n_rows, self.device, self.H = len(self.z), device, cfg.dinowm.history_size
        gen = torch.Generator().manual_seed(cfg.seed)
        perm = torch.randperm(len(self.starts), generator=gen)
        n_train = int(round(cfg.train_split * len(perm)))
        self.split = {"train": perm[:n_train].to(device), "val": perm[n_train:].to(device)}
        self.gen = torch.Generator(device=device).manual_seed(cfg.seed)
        logger.info(f"[data] {len(self.episode_id)} training episodes, {self.n_rows} frames, "
                    f"{len(self.starts)} clips (train {n_train}, val {len(perm) - n_train}); "
                    f"latent dim {self.z.shape[1]}, action dim {self.action.shape[1]}")

    def batches(self, split, batch_size, shuffle, drop_last):
        idx = self.split[split]
        if shuffle:
            idx = idx[torch.randperm(len(idx), generator=self.gen, device=self.device)]
        stop = len(idx) - (len(idx) % batch_size if drop_last else 0)
        for lo in range(0, stop, batch_size):
            yield idx[lo:lo + batch_size]

    def sample(self, clip_idx, probs, gamma):
        """Latents (B,4,D), actions (B,4,A), goal latent (B,1,D), non-terminal mask (B,3,1)."""
        s = self.starts[clip_idx]
        rows = s[:, None] + torch.arange(self.span, device=self.device)
        cur = s + self.H - 1
        max_k = self.ep_last[clip_idx] - cur                             # >= 1 by construction
        B = len(s)
        u = torch.rand(B, generator=self.gen, device=self.device)
        edges = torch.tensor(np.cumsum(probs)[:-1], device=self.device, dtype=u.dtype)
        kind = torch.bucketize(u, edges, right=True)                     # 0 random, 1 geom, 2 uniform, 3 current
        v = torch.rand(B, generator=self.gen, device=self.device).clamp_min(1e-12)
        geo = torch.floor(torch.log(v) / np.log(gamma)).long() + 1        # Geom(1-gamma) on {1,2,...}
        uni = 1 + torch.minimum(torch.floor(v * max_k).long(), max_k - 1)
        rnd = torch.randint(0, self.n_rows, (B,), generator=self.gen, device=self.device)
        goal = torch.where(kind == 0, rnd,
               torch.where(kind == 1, cur + torch.minimum(geo, max_k),
               torch.where(kind == 2, cur + uni, cur)))
        get = lambda t, r: t[r.to(t.device)].to(self.device, non_blocking=True)  # noqa: E731
        masks = (rows[:, :self.H] != goal[:, None]).float().unsqueeze(-1)
        return get(self.z, rows), get(self.action, rows), get(self.z, goal)[:, None], masks


def goal_probs(p):
    return (p.random, p.geometric_future, p.uniform_future, p.current)


# ----------------------------------------------------------------------------------------
# networks (shared with gas_mpc_eval.py's gciql method)
# ----------------------------------------------------------------------------------------

def head_kwargs(cfg, dim):
    return dict(num_patches=1, num_frames=cfg.dinowm.history_size, dim=dim, **cfg.predictor)


def build_actor(cfg, dim, act_dim):
    return gcrl_nn.Predictor(out_dim=cfg.frameskip * act_dim, **head_kwargs(cfg, dim))


class GCIQLPolicy(torch.nn.Module):
    """Actionable model for swm.policy.FeedForwardPolicy: frozen LeWM encoder -> actor head,
    mean action of the last frame (upstream GCRL.get_action with sample=False)."""

    def __init__(self, lewm, actor):
        super().__init__()
        self.lewm, self.actor = lewm.eval().requires_grad_(False), actor

    @torch.no_grad()
    def get_action(self, info):
        z = self.lewm.encode({"pixels": info["pixels"].float()})["emb"]          # (B, T, D)
        zg = self.lewm.encode({"pixels": info["goal"].float()})["emb"][:, -1:]   # (B, 1, D)
        return self.actor(z, zg)[:, -1]


def load_policy(ckpt_dir, device="cuda"):
    """Returns (GCIQLPolicy, weights path, training episode ids)."""
    from common.lewm_loader import load_lewm
    weights = sorted(Path(ckpt_dir).glob("weights_epoch_*.pt"), key=lambda p: int(p.stem.rsplit("_", 1)[1]))[-1]
    ck = torch.load(weights, map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(ck["cfg"])
    actor = build_actor(cfg, ck["latent_dim"], ck["action_dim"])
    actor.load_state_dict(ck["actor"])
    lewm = load_lewm(ckpt_dir=MECH.ckpt_dir(ROOT), device=device)
    lewm.interpolate_pos_encoding = True
    return GCIQLPolicy(lewm, actor).to(device).eval(), weights, np.asarray(ck["train_episodes"])


# ----------------------------------------------------------------------------------------
# training
# ----------------------------------------------------------------------------------------

def expectile_loss(pred, target, tau):
    r = target - pred
    return (torch.abs(tau - (r.detach() < 0).float()) * r.pow(2)).mean()


@torch.no_grad()
def ema(target, source, tau):
    for t, s in zip(target.parameters(), source.parameters()):
        t.lerp_(s, 1 - tau)


class Phase:
    """One training phase: nets, optimisers, per-epoch resume state and checkpoints."""

    def __init__(self, cfg, name, nets, targets, data, meta, step_fn):
        self.cfg, self.name, self.nets, self.targets, self.data, self.meta = cfg, name, nets, targets, data, meta
        self.step_fn = step_fn
        self.opts = {k: torch.optim.AdamW(n.parameters(), lr=cfg.predictor_lr) for k, n in nets.items()}
        self.dir = CKPT / f"{cfg.output_model_name}_{name}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.dir / "resume.pt"
        self.epoch = 0
        if self.state_path.exists():
            st = torch.load(self.state_path, map_location=data.device, weights_only=False)
            for k in nets:
                nets[k].load_state_dict(st["nets"][k])
                self.opts[k].load_state_dict(st["opts"][k])
            for k in targets:
                targets[k].load_state_dict(st["targets"][k])
            data.gen.set_state(st["gen"].cpu())
            self.epoch = st["epoch"]
            logger.info(f"[{name}] resumed after epoch {self.epoch}")
        use_bf16 = cfg.precision == "bf16-mixed"
        if use_bf16 and not torch.cuda.is_bf16_supported():
            raise ValueError("this GPU has no bf16; pass precision=32")
        self.autocast = dict(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16)

    def done(self):
        return self.epoch >= self.cfg.max_epochs

    def run(self):
        cfg = self.cfg
        while not self.done():
            t0, tot, n = time.time(), {}, 0
            for idx in self.data.batches("train", cfg.batch_size, True, True):
                with torch.autocast(**self.autocast):
                    losses = self.step_fn(idx)
                for opt in self.opts.values():
                    opt.zero_grad(set_to_none=True)
                losses["loss"].backward()
                for k, opt in self.opts.items():
                    torch.nn.utils.clip_grad_norm_(self.nets[k].parameters(), cfg.gradient_clip_val)
                    opt.step()
                for k, tgt in self.targets.items():
                    ema(tgt, self.nets[k], cfg.value_ema_tau)
                for k, v in losses.items():
                    tot[k] = tot.get(k, 0.0) + float(v)
                n += 1
                if n % cfg.log_every == 0:
                    logger.info(f"[{self.name}] epoch {self.epoch + 1} step {n} "
                                + " ".join(f"{k}={v / n:.4f}" for k, v in tot.items())
                                + f" ({n / (time.time() - t0):.1f} it/s)")
            val, m = {}, 0
            with torch.no_grad(), torch.autocast(**self.autocast):
                for idx in self.data.batches("val", cfg.batch_size, False, False):
                    for k, v in self.step_fn(idx).items():
                        val[k] = val.get(k, 0.0) + float(v)
                    m += 1
            self.epoch += 1
            logger.info(f"[{self.name}] epoch {self.epoch}/{cfg.max_epochs} ({time.time() - t0:.0f}s) train "
                        + " ".join(f"{k}={v / n:.4f}" for k, v in tot.items()) + " | val "
                        + " ".join(f"{k}={v / max(m, 1):.4f}" for k, v in val.items()))
            if self.epoch % cfg.save_every == 0 or self.done():
                torch.save(dict(self.meta, epoch=self.epoch, **{k: v.state_dict() for k, v in self.nets.items()}),
                           self.dir / f"weights_epoch_{self.epoch}.pt")
            torch.save(dict(nets={k: v.state_dict() for k, v in self.nets.items()},
                            targets={k: v.state_dict() for k, v in self.targets.items()},
                            opts={k: v.state_dict() for k, v in self.opts.items()},
                            gen=self.data.gen.get_state(), epoch=self.epoch), self.state_path)


@hydra.main(version_base=None, config_path=str(ROOT / "config" / "gciql"), config_name="gciql")
def main(cfg):
    OmegaConf.resolve(cfg)
    dev = "cuda"
    torch.manual_seed(cfg.seed)
    data = LatentClips(cfg, dev)
    D, A, H = data.z.shape[1], data.action.shape[1], cfg.dinowm.history_size
    meta = dict(cfg=OmegaConf.to_container(cfg), latent_dim=D, action_dim=A, env=ENV,
                train_episodes=data.episode_id, act_mean=data.act_mean, act_std=data.act_std)
    logger.info(f"[gciql] env={ENV} out={CKPT}\n{OmegaConf.to_yaml(cfg)}")

    # ---- phase 1: V (expectile of Q) and Q (TD to the EMA value target) ----
    value = gcrl_nn.Predictor(out_dim=1, pool_type="mean", **head_kwargs(cfg, D)).to(dev)
    critic = gcrl_nn.QPredictor(action_dim=cfg.frameskip * A, pool_type="mean", **head_kwargs(cfg, D)).to(dev)
    value_t, critic_t = copy.deepcopy(value).requires_grad_(False), copy.deepcopy(critic).requires_grad_(False)
    probs = goal_probs(cfg.goal_probabilities)

    def critic_step(idx):
        z, act, zg, masks = data.sample(idx, probs, cfg.goal_gamma)
        emb, nxt, a = z[:, :H], z[:, cfg.dinowm.td_offset:], act[:, :H]
        with torch.no_grad():
            q = critic_t(emb, a, zg).float()
            q_target = -masks + cfg.discount * masks * value_t(nxt, zg).float()
        v = value(emb, zg).float()
        value_loss = expectile_loss(v, q, cfg.expectile)
        critic_loss = ((critic(emb, a, zg).float() - q_target) ** 2).mean()
        return dict(loss=value_loss + critic_loss, value_loss=value_loss, critic_loss=critic_loss,
                    v_mean=v.mean().detach(), goal_hit=1 - masks.mean())

    phase1 = Phase(cfg, "value", dict(value=value, critic=critic), dict(value=value_t, critic=critic_t),
                   data, meta, critic_step)
    if cfg.train_value:
        phase1.run()
    elif not phase1.done():
        raise ValueError(f"train_value=false but phase 1 has only {phase1.epoch}/{cfg.max_epochs} epochs")
    value.eval().requires_grad_(False)

    # ---- phase 2: AWR actor under the trained (student) value net ----
    actor = build_actor(cfg, D, A).to(dev)
    log_stds = torch.nn.Parameter(torch.zeros(cfg.frameskip * A, device=dev))
    actor_nets = dict(actor=actor, log_stds=torch.nn.ParameterList([log_stds]))
    aprobs = goal_probs(cfg.actor_goal_probabilities)

    def actor_step(idx):
        z, act, zg, _ = data.sample(idx, aprobs, cfg.goal_gamma)
        emb = z[:, :H]
        with torch.no_grad():
            adv = (value(z[:, cfg.dinowm.td_offset:], zg) - value(emb, zg)).float()
        mu = actor(emb, zg).float()
        ls = log_stds.clamp(-5.0, 2.0)                                   # GCRL's log-std bounds
        nll = ls + 0.5 * (act[:, :H] - mu) ** 2 / torch.exp(2 * ls)
        w = torch.exp(adv * cfg.awr_alpha).clamp(max=100.0)
        return dict(loss=(w * nll).mean(), nll=nll.mean().detach(), adv_mean=adv.mean(), exp_adv_mean=w.mean())

    Phase(cfg, "policy", actor_nets, {}, data, meta, actor_step).run()
    logger.info(f"[gciql] done; policy checkpoints in {CKPT / (cfg.output_model_name + '_policy')}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s | %(message)s")
    main()
