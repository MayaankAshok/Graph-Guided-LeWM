# Push-T eval inventory (2026-09-20)

As of 2026-09-20, `outputs/pusht/eval/` holds only the **5 current groups** (75
runs, plus one 2-task smoketest run, see below) that feed main.tex directly. The
**32 legacy groups** (102 runs) from the hyperparameter-exploration phase that
predates the `task200u` pool convention have been moved to
`outputs/pusht/bak/legacy_pretask200u/eval/` -- documented below for the record,
but no longer live. (This is in addition to the 154-file stale-pairs
`matrix_newtasks`/`matrix_oldcritic_newtasks`/`matrix_newcritic_newtasks` batch
already archived earlier to `bak/task200u_stale_pairs/`.)

Original full accounting before the move: 177 base runs total (75 current + 102
legacy), plus 1 stray `l2_rh20_h20_smoketest` run (n=2, a sanity check, not a real
result -- tagged `task200u` so it stayed in `eval/`, not counted in either group
above). Nothing was orphaned.

## Current groups (feed main.tex, verified exact match)

| Group (method tag) | Runs | Coverage | What it is |
|---|---:|---|---|
| `l2` (`l2_pipe_<hash>`) | 20 | all 4 protocols x 5 seeds | Plain LeWM L2-latent CEM baseline, no graph/critic. main.tex's headline **L2** row (91.2/40.8/13.2/14.4). |
| `subgoal_tdr_la13.53_th8_htd8_te0.9_ft13.53_fl2_crit1_ccet` (`_pipe_<hash>`) | 20 | all 4 protocols x 5 seeds | Graph/TDR method, critic on, final phase switches to L2 at 13.53 units. main.tex's headline **OUR** row (88.4/70.0/46.4/42.4), no retrieval. |
| `subgoal_tdr_la13.7_th8_htd8_te0.9_ret_ft13.7_fl2_crit1_ccet` | 20 | all 4 protocols x 5 seeds | OUR plus demonstration retrieval. main.tex's **OUR+retr** row (86.8/66.8/44.8/42.4) -- verified exact match. |
| `l2_rh10_h10` | 5 | same50 only, 5 seeds | L2 baseline, replan-frequency ablation: 2 open-loop chunks of 50 steps instead of 4 of 25 (`receding=10`). Matches main.tex's "long (2x50)" row (34.4). |
| `l2_rh20_h20` | 10 | same100+cross, 5 seeds | Same ablation at `receding=20` (2 chunks of 100 for same100, 3 of ~100 for cross). Matches main.tex's "long" rows (same100: 5.6, cross: 4.8). |

## Legacy groups (pre-`task200u`, exploratory hyperparameter search -- now in `bak/`)

102 runs across 32 distinct method-tag variants, none tagged with a `pool` (they
predate the pool convention), mostly single CEM seed, `n=50` (a few `n=100` with a
4-way split). These are the manual sweep that found the current `la`/`th`/`te`/`crit`
hyperparameters -- not part of any table in main.tex, superseded by the 5 current
groups above. **Moved to `outputs/pusht/bak/legacy_pretask200u/eval/`.**

| Method family | Runs | What was being explored |
|---|---:|---|
| `subgoal*` (`subgoal_tdr_...`, `subgoal_la...`) | 70 | The bulk of the search: lookahead (`la8`/`la13.7`/`la20`), threshold (`th8`/`th12`/`th16`), terminal-switch (`te0.9`/`te0.99`/`te0.999`), critic weight (`crit0.5`/`crit1`/`crit2`/`crit4`/`crit8`), receding-horizon (`rh1`/`rh2`), and early retrieval probes (`_ret`, `_ret_crit*`) -- one-off single-seed checks, not a systematic grid. |
| `l2` (untagged, not the current `l2_pipe_<hash>`) | 17 | Early L2 baseline checks, including `l2_crit1`/`l2_crit2`/`l2_ret` critic/retrieval probes and a couple of `n=100` single-pass runs (the ones with `__c0/25/50/75` splits and `_traj.npz` dumps). |
| `ctg_htd8_te0.9` | 5 | An early cost-to-go variant (`ctg`), including one retrieval probe (`ctg_htd8_te0.9_ret`). |
| `tdr` | 4 | TDR-distance-only planning cost, one seed across all 4 protocols -- an early, simpler alternative to the subgoal mechanism. |
| `dir_la13.7_th8_htd8_te0.9` | 3 | An early "direct" planning-cost variant. |
| `path_sugap5_th8_htd8_te0.9` | 3 | An early shortest-path-cost variant. |

## Notes

- Every current group's numbers were recomputed from the raw files and matched
  main.tex to the decimal.
- The `subgoal_tdr_la13.7_th8_htd8_te0.9_ret_ft13.7_fl2_crit1_ccet` retrieval
  variant (found while inventorying) turned out to already be in main.tex as the
  **OUR+retr** row of the main Push-T summary table (line 160) -- not a new/orphaned
  result.
- See [heldout-eval-verification-2026-09-18.md](../../heldout-eval-verification-2026-09-18.md)-style
  task-pool provenance checks: the 154 stale-pairs files that were here before are
  now archived separately (see chat history / `bak/task200u_stale_pairs/`), so
  every file counted above is consistent with the current pairs pool.

## Cross-reference
See also [eval-inventory-cube.md](eval-inventory-cube.md) and
[eval-inventory-reacher.md](eval-inventory-reacher.md) for the other two envs.
