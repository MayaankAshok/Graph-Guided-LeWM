# Current asset isolation — 2026-09-19

Audited the actual Ada assets on gnode003, not only their provenance flags.

| Environment | Source episodes | Training-cache episodes | Final holdout | Training-cache frames |
|---|---:|---:|---:|---:|
| Push-T | 18,685 | 18,312 | 373 | 2,290,263 |
| Reacher | 10,000 | 9,800 | 200 | 1,969,800 |
| Cube | 10,000 | 9,800 | 200 | 1,969,800 |

For every environment:

- Training-cache episode IDs exactly match the fixed seed-0 non-evaluation split.
- Every cached source episode offset and length matches that episode in the HDF5
  dataset. Compact offsets and row counts are consistent with the filtered layout.
- All five TDR checkpoints exclude the final holdout from their recorded training
  and current diagnostic episode lists.
- Original training code uses the fixed seed-0 episode split for every TDR seed.
  TDRSampler draws both source and random-goal frames only from training rows.
  Future goals stay inside those episodes. Diagnostics run under no_grad on
  LayerNorm MLPs; they do not select a checkpoint. Training stops at a fixed step.
- Current TDR histories were refreshed using only the filtered training cache.
  Original holdout diagnostic histories are archived; TDR weights were reused.
- All five psi_train arrays have precisely the filtered training-cache row count.
- Every retained construction row and every node medoid in all five current graphs
  maps back to a non-evaluation source episode. This was checked directly using
  source HDF5 offsets, beyond graph metadata assertions.
- Current calibration code samples only cache_train/psi_train. Both global and
  per-seed gap calibrations have training_only scope; full-dataset tables are archived.
- Sixteen sampled training embeddings per environment match fresh encoding of
  their source pixels (maximum absolute difference 9.54e-7). For all five TDR seeds,
  the corresponding psi features match fresh TDR inference (maximum difference
  1.34e-5 across the audit).

Thus skipping these completed downstream preparation stages is justified.
Two qualifications prevent calling *every* pipeline artifact independent of held-out
episodes: task pools intentionally contain held-out starts/goals, and the frozen
authors' LeWM checkpoint plus inherited aggregate action-normalization statistics
come from the full dataset. Current preparation does not recompute those statistics
from held-out episode actions. cache_full remains an evaluation/legacy artifact.

The completed original TDR weights had held-out diagnostics, but those diagnostics
were reporting only. The current preparation tables are training-only; no claim is
made that the frozen pretrained world model itself never saw evaluation episodes.
