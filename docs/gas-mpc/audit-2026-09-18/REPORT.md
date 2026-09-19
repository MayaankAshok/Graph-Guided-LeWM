# Push-T task-pool and table audit — 2026-09-18

The Table 14 old-task comparison is invalid: its `task200` rows actually evaluate the
same held-out tasks as its `task200u` rows. This is a confirmed task-loader bug, not a
standard-deviation explanation or evidence that the old critic changed performance on
the original task set. The original Table 1 runs remain traceable to the preserved
original pools.

## Evidence and scope

The audit snapshots Ada's pool files and relevant matrix logs into this directory.
It checks 103 completed 50-task results: 40 original L2/B runs (five seeds, four
protocols), 60 matrix runs for same25/same50/same100, and three completed cross runs
present in the snapshot. Cross is an incomplete snapshot, not a five-seed conclusion.

For every completed result, both 25-task chunk files match the aggregate first-hit
array, the success rate recomputes correctly, and all 50 initial position and angle
errors match a snapshot pool to tolerance 1e-8. Actual pool identities are compared
using SHA-256 over all 200 start/goal rows and states, independently of filenames.
All current held-out pools also pass checks for held-out endpoints, zero pre-solved
tasks, the prescribed offset, and disjoint inclusive intervals/endpoints.

`findings.json` contains the per-file matches, hashes, statistics and configuration
comparisons. `check.py` reruns the audit using the standard library and reproduces
the loader bug in a temporary directory. The existing
`scripts/test_heldout_tasks.py` also passes with NumPy.

## Confirmed root cause

`scripts/gas_mpc_eval.py:645–648` supplies `reject=mech.goal_reached` and
`heldout_eps=task_heldout_episodes(...)` for **both** pool labels.
`load_or_make_pairs` in `scripts/viability_cross_episode_baseline.py:153` then
archives an existing pool without `nonoverlap` metadata as `.full_dataset.json`
and writes the held-out draw back to the original path. It does not restrict
migration to `task200u`.

Ada logs record these migrations of `task200` itself:

- same25: hashed log `_ce1049ee__same25__s1__n50.log`, line 21.
- same50: `_f1121bf8__same50__s0__n50.log`, line 21.
- same100: `_f1121bf8__same100__s0__n50.log`, line 21.
- cross: `_ce1049ee__cross__s1__n50.log`, line 13.

For all four protocols, current `task200.json` and `task200u.json` have identical
200-pair/state signatures and share all 50 evaluation pairs. Every completed matrix
old-task row matches that held-out draw. Every original L2/B row matches the
`task200.full_dataset.json` archive.

For example, same25 task 0 originally has position/angle errors
150.588866292158 px / 0.970848679542542 rad. In both current labeled pools it has
107.586259032454 px / 0.0177875161170959 rad. These quantities precede control.

| Protocol | Original Table 1 B mean | Matrix labeled old tasks mean | Matrix new tasks, old critic mean | Actual matrix pools |
|---|---:|---:|---:|---|
| same25 | 90.8 | 87.6 | 88.0 | identical held-out tasks |
| same50 | 64.4 | 72.0 | 69.6 | identical held-out tasks |
| same100 | 50.4 | 54.4 | 54.4 | identical held-out tasks |

The manuscript statements that changing the pool causes +0.4 at same25, −2.4 at
same50, or no change at same100 are unsupported by this matrix.

## Statistics convention also differs

Table 1 uses population SD (`numpy.std`, ddof=0), as does
`scripts/gas_mpc_report.py`. Table 14 uses sample SD (ddof=1).

| L2 same25 | Per-seed percentages | Mean | Population SD | Sample SD |
|---|---|---:|---:|---:|
| Original tasks | 90, 92, 86, 94, 94 | 91.2 | 2.993 → 3.0 | 3.347 → 3.3 |
| Held-out tasks | 92, 92, 90, 92, 90 | 91.2 | 0.980 → 1.0 | 1.095 → 1.1 |

Thus both task identity and SD convention explain the two L2 entries. The earlier
assistant statement that both tables used sample SD was incorrect. Neither is a
standard error or confidence interval. Choose one SD convention for the manuscript.

## What remains usable, and what is unresolved

The original Table 1 results have internally consistent counts and original-task
provenance. The held-out L2 and new-critic rows match their intended held-out pools.
The held-out old-critic row likewise has the correct task identity. The matrix cannot
isolate an old-versus-new task-pool effect, however, because its alleged old-task
condition never used the original tasks.

Recorded MPC settings for paired matrix old-critic rows match for every available
seed after excluding the display tag/force flag. Seed-0 graph statistics match the
original log (20,559 nodes, 510,910 directed CSR entries); the old-task and new-task
seed-0 matrix logs show the same graph statistics and critic path. There is no
identified graph-size change explaining the main discrepancy.

There is a separate repeatability issue: duplicate old-critic runs on the same
held-out tasks and recorded seeds differ (e.g. same50 means 72.0 versus 69.6).
The installed CEM solver uses a private seeded Torch generator; the evaluator assigns
chunk seeds `1000 * seed + chunk_start`. This should not be described simply as
different CEM seeds. The saved files do not include asset hashes, solver configuration
or determinism settings sufficient to establish the remaining cause. Live environment
reset uses `seed=None`. Numerical nondeterminism, unrecorded runtime differences and
asset/code drift remain hypotheses requiring a controlled replay. No fresh control
experiment was needed to establish the pool-migration bug.

## Required repair before a valid rerun

1. For historical `task200`, load/verify the unfiltered pool with `reject=None` and
   `heldout_eps=None`; restrict held-out migration in the shared loader to explicitly
   eligible new-pool filenames. The task-making script also needs this distinction
   when `GAS_MPC_POOL=task200` is selected.
2. Restore original `task200` content from the preserved archives after fixing the
   loader, retaining both the original archive and the migrated-pool evidence.
3. Apply graph-heldout validation only to held-out pools. Historical pools have no
   `heldout_eps` key. Also preserve metadata lists when selecting task prefixes:
   the current generic `v[:n]` slicing truncates the 373-element `heldout_eps` list
   to 50, so graph validation checks only that subset of held-out episodes.
4. Rerun the historical-task condition under a new tag, or quarantine its existing
   mislabeled aggregates **and chunks**. `mpc.force=true` alone still reuses chunk
   caches. Cache checks currently use filenames, not task signatures, and the final
   cache is accepted before pool validation.
5. Record task signatures, asset hashes and solver settings; validate them before
   accepting either aggregate or chunk caches. Replay one duplicate held-out seed
   under fixed runtime settings to resolve the repeatability issue.
6. Remove the manuscript's pool-effect interpretations until the corrected comparison
   exists, and make the SD definition consistent.

This audit provides evidence and required repairs; it does not deploy changes to the
running Ada evaluation pipeline or replace historical experiment files.
