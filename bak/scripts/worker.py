"""Per-combo CLI worker for the unified B3 sweep, one process per (env, tier, variant, seed)
-- config-driven consolidation of tworoom_b3_auxphi_worker.py/pusht_b3_worker.py, for
direct-SSH/multi-process Ada orchestration (ada_sweep.py). Each invocation rebuilds its
tier's setup independently (landmark load and phi_dist are cached, so this is a few
seconds of redundant work per seed, not per-step cost) and runs exactly one combo.

Run as:
    python scripts/worker.py env=pusht tier=mixed_large variant=auxphi seed=1
"""

import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.log_util import log
from common.training import run_condition_resumable
from datatiers import build_tier_setups, ckpt_path

NAME_TO_AUX_LAMBDA = {"baseline": 0.0}  # "auxphi" resolved from cfg.aux_lambda below


@hydra.main(version_base=None, config_path="../config/graph", config_name="datatiers")
def main(cfg: DictConfig):
    aux_lambda = NAME_TO_AUX_LAMBDA.get(cfg.variant, cfg.aux_lambda)
    log(f"[worker] env={cfg.env.name} tier={cfg.tier} variant={cfg.variant} seed={cfg.seed} setup starting...")
    setup = build_tier_setups(cfg, [cfg.tier])[cfg.tier]
    run_condition_resumable(
        cfg.tier, cfg.variant, aux_lambda, setup, cfg.seed,
        ckpt_dir=ckpt_path(cfg, cfg.tier, cfg.variant, cfg.seed).parent,
        distance_source="graph",
    )
    log(f"[{cfg.env.name}/{cfg.tier}/{cfg.variant}/s{cfg.seed}] worker finished")


if __name__ == "__main__":
    main()
