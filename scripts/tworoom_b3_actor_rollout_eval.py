"""Live-rollout evaluation for a trained actor checkpoint (tworoom_b3_actor_train.py):
drives the real TwoRoomEnv from real (start, goal) proprio pairs and measures actual task
success -- the metric every headline B3/B4 number has been a proxy for (Spearman vs. true
distance) but never measured directly.

Goal pairs are drawn from the SAME held-out test_eval set already used for the Spearman
metric (test_ei/test_ej, precomputed once per tier), not a fresh/disconnected sampling --
this extends the existing result rather than introducing a new evaluation methodology.
z_g is the tier's own cached landmark embedding for the goal row (exactly what the actor
was trained against), not a fresh re-render+re-encode of the goal.

Uses the PEAK-step actor snapshot from the checkpoint (peak_actor_state), matching the
peak-based comparison used everywhere else in this campaign -- the final-step actor is
already known (from V's own peak-then-decay pattern) to come from an overfit regime.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tworoom_b0_graph_gate import DEV, log
from tworoom_b3_actor_train import HIDDEN, get_setup, ckpt_path
from tworoom_b3_datatiers import load_checkpoint
from tworoom_b3_gciql_shaping import MLP
from tworoom_actor_rollout_utils import make_encoder, make_env, rollout_eval, run_episodes, summarize  # noqa: F401

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "outputs" / "b3_tworoom" / "actor" / "rollout_eval"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", required=True, choices=["expert_10", "expert_25", "expert_50",
                                                        "expert_100", "mixed", "mixed_large"])
    ap.add_argument("--variant", required=True, choices=["baseline", "auxphi"])
    ap.add_argument("--distance-source", default="graph",
                     help="only matters for --variant auxphi: 'graph' (default), 'transonly', "
                          "or 'k<N>' -- must match what tworoom_b3_actor_train.py was run with")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-episodes", type=int, default=50)
    ap.add_argument("--eval-seed", type=int, default=1000)
    ap.add_argument("--select", choices=["rho", "success"], default="rho",
                     help="which peak snapshot to evaluate: 'rho' = actor at V's peak-Spearman "
                          "step (original method); 'success' = actor at the step with the best "
                          "IN-TRAINING small-sample rollout success rate (tracked separately, "
                          "since there's no reason the two have to coincide).")
    args = ap.parse_args()

    log(f"device={DEV}")
    path = ckpt_path(args.tier, args.variant, args.seed, args.distance_source)
    ckpt = load_checkpoint(path)
    if ckpt is None:
        raise SystemExit(f"no checkpoint at {path} -- run tworoom_b3_actor_train.py first")

    if args.select == "rho":
        state_key, step_key = "peak_actor_state", "peak_step"
    else:
        state_key, step_key = "peak_success_actor_state", "peak_success_step"
    if ckpt.get(state_key) is None:
        raise SystemExit(f"{path} has no {state_key} yet -- training hasn't reached its first "
                          f"relevant eval checkpoint (rerun tworoom_b3_actor_train.py, or this "
                          f"checkpoint predates --select=success support and needs retraining)")
    peak_step = ckpt[step_key]
    log(f"[{args.tier}/{args.variant}/s{args.seed}] loaded checkpoint: training step={ckpt['step']}, "
        f"done={ckpt['done']}, select={args.select} -> using step {peak_step} "
        f"(peak_rho={ckpt.get('peak_rho')}, peak_success_rate={ckpt.get('peak_success_rate')})")

    setup = get_setup(args.tier)
    d = setup["d"]
    act_dim = setup["action"].shape[1]
    actor_net = MLP(2 * d, HIDDEN, out_dim=act_dim).to(DEV)
    actor_net.load_state_dict(ckpt[state_key])
    actor_net.eval()

    log(f"[{args.tier}/{args.variant}/s{args.seed}] running {args.n_episodes} live rollout episodes "
        f"using the {args.select}-selected actor (step {peak_step})...")
    result = rollout_eval(actor_net, setup, args.n_episodes, args.eval_seed)
    log(f"[{args.tier}/{args.variant}/s{args.seed}] RESULT: success_rate={result['success_rate']:.3f} "
        f"({sum(e['success'] for e in result['episodes'])}/{result['n_episodes']}), "
        f"mean_steps_to_goal={result['mean_steps_to_goal']}, "
        f"mean_final_dist_on_failure={result['mean_final_dist_on_failure']}")

    source_suffix = "" if (args.variant == "baseline" or args.distance_source == "graph") else f"_{args.distance_source}"
    out_path = OUT_DIR / f"{args.tier}__{args.variant}{source_suffix}__s{args.seed}__sel-{args.select}.json"
    out_path.write_text(json.dumps(dict(
        tier=args.tier, variant=args.variant, distance_source=args.distance_source, seed=args.seed,
        select=args.select, peak_step=peak_step,
        peak_rho=ckpt.get("peak_rho"), peak_success_rate_in_training=ckpt.get("peak_success_rate"),
        **result,
    ), indent=2))
    log(f"wrote {out_path}")


if __name__ == "__main__":
    main()
