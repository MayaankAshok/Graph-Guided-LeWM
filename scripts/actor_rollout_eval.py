"""Unified live-rollout evaluation for a trained actor checkpoint (actor_train.py) --
config-driven consolidation of tworoom_b3_actor_rollout_eval.py/pusht_b3_actor_rollout_eval.py.
Drives the real env from real (start, goal) state pairs and measures actual task success
rate under the protocol selected by cfg.eval_protocol (see actor_rollout_utils.py's module
docstring: `same_episode` = the LeWM paper's App. F.1 protocol, `cross_episode` = the
original arbitrary-held-out-pairs convention).

Run as:
    python scripts/actor_rollout_eval.py env=pusht tier=mixed_large variant=auxphi seed=1 select=success
    python scripts/actor_rollout_eval.py env=pusht tier=expert_25 policy=random   # the paper's Random baseline
"""

import json
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

sys.path.insert(0, str(Path(__file__).resolve().parent))
from actor_rollout_utils import rollout_eval
from actor_train import ckpt_path, get_setup, run_tag
from common.checkpoint_io import load_checkpoint
from common.envs import ENV_MECHANICS
from common.graph_lib import DEV
from common.log_util import log
from common.training import HIDDEN, MLP

ROOT = Path(__file__).resolve().parent.parent


@hydra.main(version_base=None, config_path="../config/graph", config_name="actor")
def main(cfg: DictConfig):
    log(f"device={DEV} env={cfg.env.name}")
    variant_name = f"{cfg.variant}{run_tag(cfg)}"   # e.g. baseline_hg96 for a run_tag=_hg96 run
    tag = f"{cfg.env.name}/{cfg.tier}/{variant_name}/s{cfg.seed}"
    setup = get_setup(cfg, cfg.tier)
    ckpt_dir = ENV_MECHANICS[cfg.env.name].ckpt_dir(ROOT)
    out_dir = ROOT / "outputs" / f"b3_{cfg.env.output_prefix}" / "actor" / "rollout_eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    if cfg.policy == "random":
        tag = f"{cfg.env.name}/{cfg.tier}/random"
        log(f"[{tag}] running {cfg.n_episodes} live rollout episodes with uniform-random actions...")
        result = rollout_eval(cfg, None, setup, cfg.n_episodes, cfg.eval_seed, ckpt_dir)
        meta = dict(env=cfg.env.name, tier=cfg.tier, variant="random", seed=None, select=None, peak_step=None,
                    peak_rho=None, peak_success_rate_in_training=None)
        out_path = out_dir / f"random__{cfg.tier}__{cfg.eval_protocol}.json"
    elif cfg.policy == "actor":
        path = ckpt_path(cfg, cfg.tier, cfg.variant, cfg.seed)
        ckpt = load_checkpoint(path)
        if ckpt is None:
            raise SystemExit(f"no checkpoint at {path} -- run actor_train.py first")

        if cfg.select == "rho":
            state_key, step_key = "peak_actor_state", "peak_step"
        elif cfg.select == "success":
            state_key, step_key = "peak_success_actor_state", "peak_success_step"
        else:  # "final" -- the raw actor as of the checkpoint's own last training step,
               # unconditioned on any peak tracking (e.g. to compare a specific mid-training
               # step against the peak-selected ones -- run with n_steps=<that step> and a
               # distinct run_tag so it doesn't overwrite the full run's checkpoint)
            state_key, step_key = "actor_net", "step"
        if ckpt.get(state_key) is None:
            raise SystemExit(f"{path} has no {state_key} yet -- training hasn't reached its first "
                              f"relevant eval checkpoint")
        peak_step = ckpt[step_key]
        trained_under = ckpt.get("eval_protocol", "cross_episode")
        log(f"[{tag}] loaded checkpoint: training step={ckpt['step']}, done={ckpt['done']}, "
            f"select={cfg.select} -> using step {peak_step} (peak_rho={ckpt.get('peak_rho')}, "
            f"peak_success_rate={ckpt.get('peak_success_rate')}, its rollout checks used "
            f"protocol={trained_under})")
        if cfg.select == "success" and trained_under != cfg.eval_protocol:
            log(f"[{tag}] WARNING: select=success but this checkpoint's success peak was tracked under "
                f"the '{trained_under}' protocol, not '{cfg.eval_protocol}' -- retrain to select on this protocol")

        d = setup["d"]
        act_dim = setup["action"].shape[1]
        actor_net = MLP(2 * d, HIDDEN, out_dim=act_dim).to(DEV)
        actor_net.load_state_dict(ckpt[state_key])
        actor_net.eval()

        log(f"[{tag}] running {cfg.n_episodes} live rollout episodes using the {cfg.select}-selected "
            f"actor (step {peak_step})...")
        result = rollout_eval(cfg, actor_net, setup, cfg.n_episodes, cfg.eval_seed, ckpt_dir)
        meta = dict(env=cfg.env.name, tier=cfg.tier, variant=variant_name, seed=cfg.seed, select=cfg.select,
                    peak_step=peak_step, peak_rho=ckpt.get("peak_rho"),
                    peak_success_rate_in_training=ckpt.get("peak_success_rate"),
                    training_rollout_protocol=trained_under)
        out_path = out_dir / f"{cfg.tier}__{variant_name}__s{cfg.seed}__sel-{cfg.select}__{cfg.eval_protocol}.json"
    else:
        raise ValueError(f"unknown policy '{cfg.policy}' (actor | random)")

    log(f"[{tag}] RESULT: success_rate={result['success_rate']:.3f} "
        f"({sum(e['success'] for e in result['episodes'])}/{result['n_episodes']}), "
        f"mean_steps_to_goal={result['mean_steps_to_goal']}, "
        f"mean_final_dist_on_failure={result['mean_final_dist_on_failure']}")

    protocol_meta = dict(eval_protocol=cfg.eval_protocol, eval_seed=cfg.eval_seed)
    if cfg.eval_protocol == "same_episode":
        protocol_meta.update(eval_pool=cfg.eval_pool, goal_offset=int(cfg.env.paper_goal_offset),
                             eval_budget=int(cfg.env.paper_eval_budget))
    out_path.write_text(json.dumps(dict(**meta, **protocol_meta, **result), indent=2))
    log(f"wrote {out_path}")


if __name__ == "__main__":
    main()
