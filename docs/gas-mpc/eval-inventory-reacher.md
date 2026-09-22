# Reacher eval inventory (2026-09-20)

Full accounting of `outputs/reacher/eval/` (live directory only, `bak/` excluded):
**85 runs** across 5 experiment groups.

The 4 `n=50` groups each have 20 runs, each run a base file plus two half-split
siblings (`__c0`, `__c25` -- first/last 25 of the 50 task pairs, same run, not
separate experiments), 3 files per run. The 5th group (`l2_full200audit`, n=200) has
5 runs, each splitting into 8 chunks of 25 instead of 2 (1 base + 8 chunk files per
run). Nothing is orphaned.

| Group (method tag) | Runs | Coverage | What it is |
|---|---:|---|---|
| `l2` (`l2_pipe_<hash>`) | 20 | all 4 protocols x 5 seeds | Plain LeWM L2-latent CEM baseline, no graph/critic. main.tex's headline **L2** row (77.2/90.8/82.8/29.2). |
| `subgoal_tdr_la5.83_th3.72_htd3.72_te0.9_ft5.83_fl2_crit1_ccet` | 20 | all 4 protocols x 5 seeds | Graph/TDR method, critic on, final phase switches to L2. main.tex's headline **OUR** row (76.0/98.0/97.6/65.6), no retrieval. |
| `subgoal_tdr_la5.83_..._ret_ft5.83_fl2_crit1_ccet` | 20 | all 4 protocols x 5 seeds | OUR plus demonstration retrieval. main.tex's **OUR+retr** row (62.0/88.4/94.0/63.6) -- verified exact match. Unlike Cube, retrieval **hurts** same25/same50 on Reacher (see main.tex "Retrieval is not noise on Push-T or Cube -- but it is on Reacher"). |
| `subgoal_tdr_la5.83_..._ret_rv_ft5.83_fl2_crit1_ccet` | 20 | all 4 protocols x 5 seeds | OUR+retrieval **plus predictor-verified retrieval** (`+mpc.retrieval_verify=true`, 2026-09-20 fix for the same25/same50 regression above). main.tex's **OUR+retr+verify** row (63.6/90.4/97.2/65.6) -- verified exact match. Fully recovers same100/cross to match OUR; same50 partially recovers; same25 is not fixed. Unlike Cube's equivalent sweep, this one is **complete** (full 5 seeds x 4 protocols, already written up). |
| `l2_full200audit` | 5 | same25 only, 5 seeds, n=200 (not 50) | A separate L2-reproduction audit checking whether the paper's 86% Reacher success reproduces on the new task200u draw. It doesn't: first-50 mean 77.2% (matches the `l2` group's same25 row exactly, a cross-check), full-200 mean ~80.9%. Not part of the headline table -- a standalone baseline-validity check. |

## Notes

- All 5 groups are accounted for and every number I spot-checked against main.tex
  matched exactly.
- Reacher's retrieval story differs from Cube's: retrieval alone is a net loss at
  short horizons (same25 -14.0, same50 -9.6) and the verify-gate fix only partially
  recovers it (same25 stays broken, `-12.4` and higher variance); main.tex proposes a
  stricter verification margin or partial-population warm start as the next fix,
  both untested.
- No incomplete/in-progress batches remain for Reacher (contrast with Cube, where the
  `ret_rv` sweep was only 6/20 done as of the last check).

## Cross-reference
See also [eval-inventory-cube.md](eval-inventory-cube.md) and
[eval-inventory-pusht.md](eval-inventory-pusht.md) for the other two envs.
