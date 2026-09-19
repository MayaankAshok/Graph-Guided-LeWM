"""Train the revised critic -- a hitting-time distribution head p_psi(T = b | z, z_g)
(viability-value.tex, sec. "Revised Critic") -- on a latent cache (viability_cache.py) and a
graph label bank (viability_graph_labels.py). Frozen LeWM encoder + predictor; only the head
is trained. No Bellman backup, no target network, no horizon input, no ensemble.

Per update, two half-batches (proposal: 50% logged endpoints, 50% predicted endpoints):

  logged     (z_start, z_goal) from the bank with its graph label T_G(start -> goal).
  predicted  the frozen predictor rolled --imagine-blocks blocks from a logged history under
             the LOGGED actions; imagined frame k stands for the real frame start + 5k and
             takes THAT frame's graph label T_G(start + 5k -> goal). This is how the head
             learns to tolerate the predictor's characteristic error without a simulator.
             Random / perturbed actions are NOT labelled (no known graph correspondent).

Loss: exact cross-entropy -log p(bin(T_G)) for finite paths; censored -log sum_{b > c} p(b)
otherwise. Pairs beyond the Dijkstra bound ('> B_max blocks', possibly just missing
coverage) get weight --w-disconnected. Checkpoint selection: lowest validation loss on
held-out episodes' pairs (offline; no oracle).

The saved critic.pt exposes the ensemble critic's interface (prob(z, zg, h) = CDF at h), so
viability_eval_audit.py and viability_live_rollout.py score it unchanged.

Usage:
    python scripts/viability_train_ht.py --cache outputs/pusht/critic_training/cache_1000_s0.npz \
        --labels outputs/pusht/critic_training/labels_1000_s0.npz --out outputs/pusht/critic_training/ht_1000_s0
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.envs import ENV_MECHANICS
from common.lewm_loader import load_lewm
from common.log_util import log
from common.viability import (HISTORY, SKIP, HittingTimeHead, auroc, hitting_time_bin,
                              hitting_time_loss, imagine)
from viability_train import LatentCache

DEV = "cuda" if torch.cuda.is_available() else "cpu"


class Bank:
    """One split of the label bank on the device."""

    def __init__(self, path, split, device, b_max):
        d = np.load(path)
        self.meta = json.loads(str(d["meta"]))
        m = d["split"] == split
        self.start = torch.tensor(d["start"][m], device=device)
        self.goal = torch.tensor(d["goal"][m], device=device)
        self.tg = torch.tensor(d["tg"][m], device=device)                 # env steps, -1 beyond bound
        self.tg_fwd = torch.tensor(d["tg_fwd"][m], device=device)         # (n, 5); -2 = no frame
        self.kind = torch.tensor(d["kind"][m].astype(np.int64), device=device)
        self.hist_ok = torch.nonzero(torch.tensor(d["hist_ok"][m], device=device)).squeeze(1)
        self.n, self.device, self.b_max = int(m.sum()), device, b_max
        if self.meta["b_max"] != b_max:
            log(f"[warn] bank was built with b_max={self.meta['b_max']}, head uses {b_max}")

    def labels(self, tg, w_disc):
        """env-step T_G (with -1 = beyond bound) -> (bin, censored, weight)."""
        b = hitting_time_bin(tg, self.b_max)
        beyond = tg < 0
        return b, beyond, torch.where(beyond, torch.full_like(tg, w_disc), torch.ones_like(tg))


class Sampler:
    def __init__(self, cache, bank, args, generator):
        self.c, self.b, self.a, self.g = cache, bank, args, generator

    def _rint(self, n, high):
        return torch.randint(0, int(high), (n,), device=self.c.device, generator=self.g)

    def logged(self, B):
        i = self._rint(B, self.b.n)
        return self.c.z[self.b.start[i]], self.c.z[self.b.goal[i]], self.b.tg[i], self.b.kind[i]

    def predicted(self, model, B):
        """Rollouts under the logged actions from bank starts with enough history/future.
        Returns z_img (B, n, D), z_goal (B, D), tg (B, n) with -2 = no label."""
        c, n = self.c, self.a.imagine_blocks
        i = self.b.hist_ok[self._rint(B, len(self.b.hist_ok))]
        t = self.b.start[i]
        zi = t[:, None] - torch.arange(HISTORY - 1, -1, -1, device=c.device)[None] * SKIP
        ai = t[:, None] + torch.arange(-(HISTORY - 1) * SKIP, 0, device=c.device)[None]
        z_hist = c.z[zi]
        a_past = c.act_norm[ai].reshape(B, HISTORY - 1, SKIP * c.act_raw.shape[1])
        fi = t[:, None] + torch.arange(n * SKIP, device=c.device)[None]
        fut = c.act_norm[fi].reshape(B, n, SKIP * c.act_raw.shape[1])
        z_img, _, _ = imagine(model, z_hist, a_past, fut)
        return z_img, c.z[self.b.goal[i]], self.b.tg_fwd[i, :n]


def batch_loss(head, S, model, args):
    z, zg, tg, kind = S.logged(args.batch_logged)
    b, cen, w = S.b.labels(tg, args.w_disconnected)
    l_log = hitting_time_loss(head.log_pmf(z, zg), b, cen, w)
    z_img, zg_p, tgf = S.predicted(model, args.batch_predicted)
    n, D = z_img.shape[1], z_img.shape[2]
    valid = (tgf > -2).reshape(-1)
    zq = z_img.reshape(-1, D)[valid]
    zg_q = zg_p[:, None].expand(-1, n, -1).reshape(-1, D)[valid]
    b, cen, w = S.b.labels(tgf.reshape(-1)[valid], args.w_disconnected)
    l_pred = hitting_time_loss(head.log_pmf(zq, zg_q), b, cen, w)
    return l_log, l_pred, dict(pred_labelled=float(valid.float().mean()),
                               logged_beyond=float((tg < 0).float().mean()))


@torch.no_grad()
def evaluate(head, S, model, args):
    """Held-out pairs: losses, plus CDF-vs-graph-label AUROC at a few budgets and the
    per-class accuracy, all against graph labels (no oracle)."""
    S.g.manual_seed(args.seed + 12345)
    out = {}
    z, zg, tg, kind = S.logged(args.eval_n)
    lp = head.log_pmf(z, zg)
    b, cen, w = S.b.labels(tg, args.w_disconnected)
    out["loss_logged"] = float(hitting_time_loss(lp, b, cen, w))
    out["top1_acc_exact"] = float((lp.argmax(-1) == b)[tg >= 0].float().mean())
    out["within1_acc_exact"] = float(((lp.argmax(-1) - b).abs() <= 1)[tg >= 0].float().mean())
    p = lp.exp()
    ebin = (p * head.bins).sum(-1)
    fin = tg >= 0
    if fin.sum() > 10:
        out["spearman_Ebins_vs_TG_exact"] = float(np.corrcoef(
            np.argsort(np.argsort(ebin[fin].cpu().numpy())), np.argsort(np.argsort(tg[fin].cpu().numpy())))[0, 1])
    out["p_beyond_mean_by_kind"] = {k: float(p[kind == k, -1].mean()) for k in (0, 1, 2) if (kind == k).any()}
    out["auroc_by_h"] = {}
    for h in (0, 25, 50, 100, 225):
        v = head.cdf(z, zg, torch.full_like(tg, float(h))).cpu().numpy()
        y = ((tg >= 0) & (tg <= h)).float().cpu().numpy()
        out["auroc_by_h"][h] = auroc(v, y)
    z_img, zg_p, tgf = S.predicted(model, args.eval_n // 8)
    n, D = z_img.shape[1], z_img.shape[2]
    valid = (tgf > -2).reshape(-1)
    zq = z_img.reshape(-1, D)[valid]
    zg_q = zg_p[:, None].expand(-1, n, -1).reshape(-1, D)[valid]
    tq = tgf.reshape(-1)[valid]
    b, cen, w = S.b.labels(tq, args.w_disconnected)
    lpq = head.log_pmf(zq, zg_q)
    out["loss_predicted"] = float(hitting_time_loss(lpq, b, cen, w))
    out["auroc_predicted_h25"] = auroc(head.cdf(zq, zg_q, torch.full_like(tq, 25.0)).cpu().numpy(),
                                       ((tq >= 0) & (tq <= 25)).float().cpu().numpy())
    out["loss"] = out["loss_logged"] + out["loss_predicted"]
    return out


def save(path, head, args, step, extra):
    torch.save(dict(kind="hitting_time", head=head.state_dict(),
                    head_kwargs=dict(z_dim=head.z_dim, b_max=head.b_max, hidden=tuple(args.hidden),
                                     head=args.head, input_diff=args.input_diff),
                    args=dict(vars(args), h_max=head.h_max), step=step, **extra), path)


def run(args):
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    cache = LatentCache(args.cache, DEV)
    bank_tr, bank_va = Bank(args.labels, 0, DEV, args.b_max), Bank(args.labels, 1, DEV, args.b_max)
    if Path(bank_tr.meta["cache"]).resolve() != Path(args.cache).resolve():
        log(f"[warn] labels were built on {bank_tr.meta['cache']}, training on {args.cache}")
    train_eps, val_eps = np.array(bank_tr.meta["train_episodes"]), np.array(bank_tr.meta["val_episodes"])
    log(f"[setup] cache {cache.n_frames} frames / {cache.n_ep} episodes; bank train {bank_tr.n} val {bank_va.n} pairs; "
        f"graph {bank_tr.meta['graph']}")

    ckpt_dir = cache.meta["ckpt_dir"] or str(ENV_MECHANICS["pusht"].ckpt_dir(Path(args.root)))
    model = load_lewm(Path(ckpt_dir), device=DEV)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    head = HittingTimeHead(z_dim=cache.z.shape[1], b_max=args.b_max, hidden=tuple(args.hidden),
                           head=args.head, input_diff=args.input_diff).to(DEV)
    tf = cache.frames_of(train_eps)
    head.set_input_stats(cache.z[tf].mean(0), cache.z[tf].std(0))
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps, eta_min=args.lr * 0.1)
    log(f"[setup] head={args.head} input_diff={args.input_diff} hidden={args.hidden} classes={head.n_classes} "
        f"params={sum(p.numel() for p in head.parameters()) / 1e6:.2f}M")

    g_tr = torch.Generator(device=DEV); g_tr.manual_seed(args.seed)
    g_va = torch.Generator(device=DEV); g_va.manual_seed(args.seed + 1)
    S_tr, S_va = Sampler(cache, bank_tr, args, g_tr), Sampler(cache, bank_va, args, g_va)

    history, best, t0, running = [], float("inf"), time.time(), {}
    for step in range(1, args.steps + 1):
        head.train()
        l_log, l_pred, m = batch_loss(head, S_tr, model, args)
        loss = l_log + args.w_predicted * l_pred
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), args.grad_clip)
        opt.step(); sched.step()
        for k, v in dict(loss=loss.item(), logged=l_log.item(), predicted=l_pred.item(), **m).items():
            running[k] = running.get(k, 0.0) + v
        if step % args.log_every == 0:
            avg = {k: v / args.log_every for k, v in running.items()}; running = {}
            log(f"[train] step {step}/{args.steps} loss={avg['loss']:.4f} logged={avg['logged']:.3f} "
                f"predicted={avg['predicted']:.3f} pred_labelled={avg['pred_labelled']:.2f} "
                f"{(time.time() - t0) / step * 1000:.0f}ms/step")
        if step % args.eval_every == 0 or step == args.steps:
            head.eval()
            ev = evaluate(head, S_va, model, args); ev["step"] = step
            history.append(ev)
            log(f"[eval]  step {step}: val loss={ev['loss']:.4f} (logged {ev['loss_logged']:.3f} / predicted "
                f"{ev['loss_predicted']:.3f}) top1={ev['top1_acc_exact']:.3f} within1={ev['within1_acc_exact']:.3f} "
                f"| auroc h0/25/50/100={ev['auroc_by_h'][0]:.3f}/{ev['auroc_by_h'][25]:.3f}/"
                f"{ev['auroc_by_h'][50]:.3f}/{ev['auroc_by_h'][100]:.3f} pred@25={ev['auroc_predicted_h25']:.3f} "
                f"| p(>Bmax) fwd/bwd/cross={'/'.join(f'{v:.2f}' for v in ev['p_beyond_mean_by_kind'].values())}")
            extra = dict(train_episodes=train_eps, val_episodes=val_eps, cache=str(args.cache),
                         labels=str(args.labels), val=ev)
            save(out / "critic_last.pt", head, args, step, extra)
            if ev["loss"] < best:
                best = ev["loss"]
                save(out / "critic.pt", head, args, step, dict(extra, selected_by="val_loss"))
                log(f"[select] step {step} is the new best (val loss {best:.4f}) -> critic.pt")
            (out / "metrics.json").write_text(json.dumps(dict(args=vars(args), history=history, best_val_loss=best), indent=2))
    log(f"[done] {out / 'critic.pt'} (best val loss {best:.4f}); last step in critic_last.pt")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=str(Path(__file__).resolve().parent.parent))
    p.add_argument("--cache", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    # head
    p.add_argument("--b-max", type=int, default=45)
    p.add_argument("--head", choices=["softmax", "hazard"], default="softmax")
    p.add_argument("--input-diff", action="store_true", help="append l(z) - l(z_g) to the input (ablation)")
    p.add_argument("--hidden", type=int, nargs="+", default=[512, 256])
    # batches / losses
    p.add_argument("--batch-logged", type=int, default=512)
    p.add_argument("--batch-predicted", type=int, default=128, help="rollouts per step (x imagine_blocks frames)")
    p.add_argument("--imagine-blocks", type=int, default=5)
    p.add_argument("--w-predicted", type=float, default=1.0)
    p.add_argument("--w-disconnected", type=float, default=0.25)
    # optimisation
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--log-every", type=int, default=200)
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--eval-n", type=int, default=8192)
    run(p.parse_args())


if __name__ == "__main__":
    main()
