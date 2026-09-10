"""Follow-up to diagnose.py. That run established:

  * Push-T's "true distance oracle" is NOT bogus (Spearman +0.90 vs real expert step-gaps).
  * The latent graph is EXCELLENT on Push-T (+0.9952 vs expert step-gaps; median phi_dist
    tracks the true gap almost exactly: 1,2,5,10,25,50,99 for gaps 1,2,5,10,25,50,100).
  * But the aux regression target is on a systematically different SCALE from the value the
    Bellman/expectile objective converges to: on the same HER pairs, pseudo_v median -5.85
    (p10 -15.7, min -54.8) vs the baseline's TD-learned V median -2.76 (p10 -8.1, min -22.6),
    and auxphi's V has been dragged to -4.98/-13.5/-47.4 -- i.e. toward pseudo_v, ~2x away
    from the TD fixed point.

Why the two disagree: pseudo_v is derived from graph geodesic distance, which faithfully
tracks how many steps the DATA took. IQL's expectile-0.9 V learns the OPTIMAL (best-action)
cost-to-go. On expert data those coincide; on this tier 79% of frames come from WeakPolicy
noisy rollouts that wander, so behavioural distance >> optimal distance and the aux term
fights the Bellman objective.

The consequence this script tests: the AWR actor weight is exp(beta * (Q - V)). Q is trained
purely by the Bellman objective while V is dragged down by the aux term, so the advantage
picks up a systematic positive bias -> weights saturate at awr_weight_clip -> AWR degenerates
into UNIFORM behaviour cloning over all data, including the 79% never-succeeding noisy data.

Run as:
    python scripts/investigations/pusht_lowscore/awr_weights.py env=pusht tier=mixed_large
"""

import sys
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, open_dict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from actor_train import ckpt_path
from common.checkpoint_io import load_checkpoint
from common.envs import ENV_MECHANICS
from common.graph_lib import DEV
from common.log_util import log
from common.training import HIDDEN, MLP, build_her_tuples
from datatiers import NOISY_EP_ID_OFFSET, build_tier_setups

ROOT = Path(__file__).resolve().parent.parent.parent.parent


@hydra.main(version_base=None, config_path="../../../config/graph", config_name="actor")
def main(cfg: DictConfig):
    mech = ENV_MECHANICS[cfg.env.name]
    with open_dict(cfg.env):
        cfg.env.ckpt_dir_resolved = str(mech.ckpt_dir(ROOT))
    setup = build_tier_setups(cfg, [cfg.tier])[cfg.tier]
    z, action, d = setup["z"], setup["action"], setup["d"]
    zt = torch.from_numpy(z).to(DEV)
    at = torch.from_numpy(action).float().to(DEV)

    s_idx, next_idx, goal_idx, act_idx, done = build_her_tuples(
        setup["ep_idx"], setup["step_idx"], action, seed=0,
        allowed_episode_ids=setup["train_eps"], goal_gamma=cfg.her_goal_gamma,
    )
    rng = np.random.default_rng(0)
    batch = rng.choice(len(s_idx), size=50000, replace=False)
    s, g, a_i = s_idx[batch], goal_idx[batch], act_idx[batch]
    is_noisy = setup["ep_idx"][s] >= NOISY_EP_ID_OFFSET

    log(f"\ntier={cfg.tier}  HER batch n={len(s)}  ({100*is_noisy.mean():.0f}% from noisy episodes)")
    log(f"beta={cfg.beta}  awr_weight_clip={cfg.awr_weight_clip}\n")
    log(f"{'variant':<10}{'adv mean':>10}{'adv p90':>9}{'w mean':>9}{'w median':>10}"
        f"{'% at clip':>11}{'w top1%/w med':>15}{'expert w / noisy w':>20}")

    for variant in ("baseline", "auxphi"):
        ck = load_checkpoint(ckpt_path(cfg, cfg.tier, variant, cfg.seed))
        if ck is None:
            log(f"{variant:<10}  no checkpoint -- skipped")
            continue
        v_net = MLP(2 * d, HIDDEN).to(DEV); v_net.load_state_dict(ck["v_net"]); v_net.eval()
        q_net = MLP(2 * d + action.shape[1], HIDDEN).to(DEV)
        q_net.load_state_dict(ck["q_net"]); q_net.eval()

        with torch.no_grad():
            sg = torch.cat([zt[s], zt[g]], dim=-1)
            sag = torch.cat([zt[s], at[a_i], zt[g]], dim=-1)
            adv = (q_net(sag) - v_net(sg)).squeeze(-1)
            w = torch.exp(cfg.beta * adv).clamp(max=cfg.awr_weight_clip)
        adv = adv.cpu().numpy(); w = w.cpu().numpy()

        at_clip = 100 * np.mean(w >= cfg.awr_weight_clip * 0.999)
        ratio = np.mean(w[w >= np.percentile(w, 99)]) / max(np.median(w), 1e-9)
        ew, nw = w[~is_noisy].mean(), w[is_noisy].mean()
        log(f"{variant:<10}{adv.mean():>10.3f}{np.percentile(adv, 90):>9.3f}{w.mean():>9.2f}"
            f"{np.median(w):>10.3f}{at_clip:>11.2f}{ratio:>15.1f}{ew/max(nw,1e-9):>20.2f}")

    log("\nReading: '% at clip' = share of samples whose AWR weight saturates (all such samples get\n"
        "identical weight -> no preference between them). 'w top1%/w med' = how strongly the actor\n"
        "still prefers its best samples over typical ones (1.0 = pure uniform behaviour cloning).\n"
        "'expert w / noisy w' = how much more the actor weights real expert data over WeakPolicy\n"
        "wandering data (1.0 = clones the never-succeeding noisy data just as hard as expert data).")


if __name__ == "__main__":
    main()
