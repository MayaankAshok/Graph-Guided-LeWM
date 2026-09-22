# Push-T failure-mode study (2026-09-22)

Where OUR method (`subgoal_tdr_la13.53_th8_htd8_te0.9_ft13.53_fl2_crit1_ccet`, main.tex's
headline **OUR** row: 88.4/70.0/46.4/42.4 on same25/same50/same100/cross) still fails, on
the fixed 50-task `task200u` prefix, all 4 protocols, 5 CEM seeds (n=250 episodes/protocol).
Uses the already-completed headline eval runs (`outputs/pusht/eval/subgoal_tdr_la13.53*`);
no new runs launched. Per-task `_traj.npz` state trajectories (not kept locally, ~2MB for
these 40 files) were pulled from Ada (`/home2/mayaank.ashok/lewm_research/outputs/pusht/eval/`)
to split the success predicate's combined agent+block position error back into agent-only
and block-only components, which the summary JSONs don't retain.

State layout (`PushTMechanics`): `[agent_x, agent_y, block_x, block_y, block_angle, ...]`.
Success predicate (`PushT.eval_state`): combined agent+block position error < 20px AND
block angle error < pi/9 (20 deg), simultaneously.

## Headline: failure taxonomy, pooled over 5 seeds

| Category | same25 (n=250) | same50 (n=250) | same100 (n=250) | cross (n=250) |
|---|---:|---:|---:|---:|
| success | 88.4% | 70.0% | 46.4% | 42.4% |
| close but not enough (never both conditions at once) | 10.0% | 21.6% | 38.4% | 47.6% |
| block never gets close (min block-pos-err > 60px) | 0.0% | 6.8% | 11.6% | 5.6% |
| orientation-only (position ok, angle never aligns) | 1.2% | 1.2% | 2.4% | 2.4% |
| agent abandons block (agent parks, block stays far) | 0.4% | 0.4% | 1.2% | 2.0% |

Takeaway: **failure is overwhelmingly "ran out of precision/time near the goal," not "gave
up on the block" or "wrong orientation."** The `orientation-only` bucket is small everywhere
(1-2%) -- when the method fails, the block's *position* is almost always still the blocker,
not its angle, even though Push-T requires matching both.

## What's actually the bottleneck at each episode's closest approach

For every failing episode, at its single best (minimum combined-error) timestep, which
component(s) were still over threshold:

| Bottleneck at closest approach | same25 (n=29) | same50 (n=75) | same100 (n=134) | cross (n=144) |
|---|---:|---:|---:|---:|
| block off, agent ok | 31.0% | 54.7% | 41.8% | 54.2% |
| both agent+block off | 27.6% | 26.7% | 38.8% | 25.0% |
| agent off, block ok | 10.3% | 8.0% | 8.2% | 10.4% |
| pos ok, angle off | 13.8% | 4.0% | 6.0% | 5.6% |
| looks like success (measurement artifact, see caveat) | 17.2% | 6.7% | 5.2% | 4.9% |

**"Block off, agent ok" is the single largest bucket everywhere (31-55%)**: the agent
reaches its own correct final pose, but the block itself isn't close enough to its target.
This is the classic Push-T difficulty -- the agent controls the block only indirectly
through contact, and precisely finishing a push is harder than getting the agent in
position. "Both off" (25-39%) is a genuine full miss (contact/timing never really
established). Angle-only misses stay under 14% throughout, confirming orientation is
rarely what fails when position also fails.

## Failures aren't monotonic progress -- many regress after getting close

Comparing each failing episode's best-ever combined error to its final-step error:

| Protocol | Episodes that got >=15px closer earlier, then ended worse | best-ever combined-error median |
|---|---:|---:|
| same25 | 34.5% (10/29) | 30px |
| same50 | 28.0% (21/75) | 38px |
| same100 | 57.5% (77/134) | 46px |
| cross | 45.1% (65/144) | 38px |

At same100 and cross, **over half of failing episodes were closer to the goal at some
earlier point than at episode end.** The planner approaches, then drifts or gets knocked
off again by a later re-plan or contact adjustment, and doesn't recover in the remaining
budget. This is a stronger signal at longer horizons, where there are more receding-horizon
replans (each replan is a fresh 25-step CEM solve that can re-perturb an already-good
block placement while chasing precision on the agent side, or transitioning between the
subgoal-tracking phase and the final L2-to-goal phase).

Also notable: the median best-ever error across failing episodes is 30-46px -- just 1.5-2.3x
the 20px success threshold. **Most failures are near misses, not gross misses**: only
11-21% of failing episodes never got the combined error below 100px (`same25` 3%, `same50`
20%, `same100` 21%, `cross` 11%).

## Difficulty correlates with (but isn't fully explained by) how far the task starts from the goal

Per-task success rate (fraction of 5 seeds succeeding) vs. that task's starting
agent+block position error and block angle error:

| Protocol | corr(success, initial_pos_err) | corr(success, initial_ang_err) |
|---|---:|---:|
| same25 | -0.39 | -0.30 |
| same50 | -0.37 | -0.35 |
| same100 | -0.27 | -0.26 |
| cross | -0.38 | -0.52 |

Moderate negative correlations everywhere -- larger required displacement/rotation does
predict more failure, most strongly for `cross`'s angle error (-0.52). But the correlation
is far from -1: tasks with modest starting error still fail sometimes (the near-miss/
regression story above), and a few large-displacement tasks still succeed. Both effects are
real and separate: **task difficulty (how far the goal is) sets a floor, but precision/
timing near the goal is what actually decides most individual failures.**

## Tasks that are structurally hard (fail in all 5 seeds)

| Protocol | count | task indices |
|---|---:|---|
| same25 | 2/50 | 17, 27 |
| same50 | 3/50 | 14, 15, 47 |
| same100 | 9/50 | 2, 3, 17, 21, 22, 23, 25, 44, 46 |
| cross | 15/50 | 2, 6, 8, 10, 14, 21, 23, 25, 30, 33, 37, 39, 40, 41, 47 |

These skew toward large initial angle error (many 90-180 deg off) combined with
above-median position error -- consistent with the correlation above, though not every
hard task fits that pattern (e.g. same100 task 46 has a 21deg initial angle error but
still fails all 5 seeds, and same100 task 25 has only a 1deg initial angle error).
Full per-task fail counts (0-5 out of 5 seeds) and category examples for every protocol are in
[pusht-failure-analysis.json](pusht-failure-analysis.json).

## Caveat: a small "looks like success but wasn't" artifact

5-17% of failing episodes (by protocol) have a timestep where recomputing the predicate
directly from the saved trajectory shows BOTH conditions met, yet `first_hit_step == -1`
in that run. This is most likely a resolution mismatch between the coarse per-replan state
snapshots stored in `_traj.npz` and the finer-grained step at which the live env actually
checks termination (`terminated` is env-internal, `states` here is only appended once per
`on_step` callback). It's not corrected for in the tables above; treat the smallest failure
percentages (especially `orientation-only` and the bottleneck table's small buckets) as
having a few points of measurement noise. It does not change the headline conclusions
(block-position precision and post-approach regression dominate).

## Source data and reproducing this

- Trajectory files: pulled from Ada (`ssh ada`, `/home2/mayaank.ashok/lewm_research/outputs/pusht/eval/*subgoal_tdr_la13.53*traj.npz`,
  40 files / ~2MB, one per (protocol, seed, chunk)) -- not committed, they're eval-run
  byproducts of the already-published `outputs/pusht/eval/subgoal_tdr_la13.53*` results in
  [eval-inventory-pusht.md](eval-inventory-pusht.md).
- Analysis scripts (ad hoc, not part of the tracked pipeline) reconstructed success/failure
  per task from `first_hit_step`, and recomputed agent-only / block-only / angle error by
  slicing the raw state vector, since the existing summary JSONs only store the combined
  4-dim norm. Local numpy-only, no torch/Ada compute needed once the trajectories were
  pulled down.

## Implications for next steps

- The dominant lever looks like **holding a good block placement through subsequent
  replans**, not reaching one in the first place -- most failures were closer earlier and
  regressed, especially at same100/cross. Worth checking whether `final_thresh`/`final_metric`
  phase transitions or the receding-horizon replan cadence are responsible for knocking an
  already-good block placement off course.
- Orientation is not where precision breaks down; **position precision on the block** is.
  Any second look at CEM population size, elite count, or the critic's role very close to
  the goal should focus on block-position precision, not angle.
- Task difficulty (large required displacement/rotation) is a real but partial driver --
  don't expect a fix targeted at "hard" tasks alone to close most of the gap, since a large
  share of failures are near-misses on moderate-difficulty tasks.
