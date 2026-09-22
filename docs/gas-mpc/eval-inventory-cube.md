# Cube eval inventory (2026-09-20)

Full accounting of `outputs/cube/eval/` (live directory only, `bak/` excluded):
**96 runs** across 7 experiment groups.

Each run is a base file (`success_rate` etc.) plus two half-split siblings (`__c0`,
`__c25` — first/last 25 of the 50 task pairs from the same run, not separate
experiments), 3 files per run. Nothing is orphaned.

| Group (method tag) | Runs | Coverage | What it is |
|---|---:|---|---|
| `l2` (`l2_pipe_<hash>`) | 20 | all 4 protocols x 5 seeds | Plain LeWM L2-latent CEM baseline, no graph/critic. This is the **L2** row in main.tex's headline table (56.0/34.4/35.6/16.4). |
| `subgoal_tdr_la13.63_th7.25_htd7.25_te0.9_ft13.63_fl2_crit1_ccet` (`_pipe_<hash>`) | 20 | all 4 protocols x 5 seeds | Graph/TDR method, critic on (beta=1, ET cost), final phase switches to L2 at 13.63 units. main.tex's headline **OUR** row (62.0/49.6/50.4/27.6), no retrieval. |
| `subgoal_tdr_la13.63_th7.25_htd7.25_te0.9_ret_ft13.63_fl2_crit1_ccet` | 20 | all 4 protocols x 5 seeds | Same as OUR plus demonstration retrieval. main.tex's **OUR+retrieval** row (86.4/59.2/78.8/48.0) -- verified exact match against the raw files. |
| `subgoal_tdr_la13.63_th7.25_htd7.25_te0.9_ret_rv_ft13.63_fl2_crit1_ccet` | 6 | same100+cross only, seeds 0-2 | Newer "retrieval-verify" sweep, started 2026-09-20, **still running**. Not in main.tex yet. |
| `subgoal_tdr_la13.63_th7.25_htd7.25_te0.9_fl2_crit1_ccet` (no `ft` tag) | 10 | same100+cross, 5 seeds | The **"late"** tuning variant (tighter final-L2 switch, Table `tab:cube-variants`). main.tex only reports seeds 0-2 (confirmed exact match); seeds 3-4 exist but are unused in the writeup. |
| `subgoal_tdr_la13.63_th7.25_htd7.25_te0.9_ft13.63_fl2` (no `crit`) | 10 | same100+cross, 5 seeds | The **"nocrit"** variant -- critic removed entirely. Same story: only seeds 0-2 used in main.tex. |
| `subgoal_tdr_la13.63_th7.25_htd7.25_te0.9_ft13.63_fl2_crit2_ccet` | 10 | same100+cross, 5 seeds | The **"beta2"** variant -- critic weight doubled. Only seeds 0-2 used in main.tex. |

## Notes

- 4 of the 7 groups feed main.tex directly (`l2`, OUR, OUR+retrieval, and the 3
  tuning-sweep variants combined into Table `tab:cube-variants`).
- 1 group (`ret_rv`) is a new, incomplete follow-up not yet written up.
- The tuning-sweep variants (late/nocrit/beta2) actually have all 5 seeds available
  locally, not just the 3 main.tex reports -- Table `tab:cube-variants` could be
  extended to the full 5 seeds whenever convenient.
- `bak/` (superseded runs, old-`task200`-pool archive, stale-pairs archive) is excluded
  from this inventory by design -- it is not part of the live eval set.

## Cross-reference
See also [eval-inventory-reacher.md](eval-inventory-reacher.md) and
[eval-inventory-pusht.md](eval-inventory-pusht.md) for the other two envs.
