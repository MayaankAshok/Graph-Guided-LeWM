# One-pass long-horizon hitting-time critic study

## Why the current critic cannot provide long-range ET

The current `ViabilityCritic` estimates a separate binary quantity
`V(z, zg, h) = P(T <= h)` for a supplied horizon. Its expected-time planning cost is the
restricted survival sum

`ET_h = 5 * sum_{j=0,5,...,h} (1 - V(z, zg, j))`.

With `h_max=50`, this needs 11 critic evaluations and has a maximum cost of 55 environment
steps. More importantly, the planner clips the remaining budget to 50. At the first CEM call,
same100 has 175 post-plan steps left and cross has 225, but both are scored as if only 50 were
left. Endpoints whose true hitting times are 60 and 200 are therefore indistinguishable except
through classifier error. Robust standardisation can then expand that residual noise back to a
full vote in CEM.

The audit reports both logged and predictor-imagined held-out endpoints. For the legacy critic
it measures its deployed 0--50 restricted ET and separately probes the untrained CDF at 100,
150, and 225. This distinguishes the hard cap from accidental horizon-feature extrapolation.

## Alternatives and expected ranking

1. **Categorical hitting-time distribution (expected best).** One MLP emits a probability mass
   function over 5-step bins 0--225 plus a censored `>225` bin. A cumulative sum gives every
   viability budget and one survival reduction gives restricted ET. It is monotone by
   construction, handles multimodal route lengths, and uses stable cross-entropy. Expected
   failure modes: 47-way class imbalance, quantisation, a truncated tail, and overconfidence on
   predictor endpoints outside the logged support.

2. **Discrete hazard head.** One MLP emits conditional hit hazards for the same bins. It has the
   same expressivity and outputs as the categorical model but encodes survival ordering more
   directly. Expected failure modes: early hazard errors compound through every later survival
   probability; long-tail gradients pass through long products; label noise in short paths can
   distort the whole curve. It should be competitive but harder to optimise than categorical CE.

3. **Parametric Weibull survival head.** One MLP emits only scale and shape. It has an unbounded
   analytic CDF and mean, so it is the only candidate that genuinely extrapolates past the label
   horizon rather than assigning all tail mass to a final bin. Expected failure modes: Push-T has
   identity pairs, multiple routes, graph shortcuts, and disconnected/censored pairs, so a
   single unimodal two-parameter family is likely misspecified; its unconstrained mean may also
   give CEM exploitable tail values. It is ranked third despite being cheapest.

A direct scalar regressor was rejected before live evaluation: censored pairs do not have a
target value, it cannot recover `V(h)` for feasibility checks, and squared/Huber regression
collapses multimodal route lengths to a mean. The Weibull head is the similarly cheap alternative
with a proper censored likelihood.

## Implementation and checks

- `HittingTimeHead` implements `softmax`, `hazard`, and `weibull` heads.
- `restricted_expected_steps(z, zg, h)` exactly matches the legacy survival-grid definition but
  evaluates the network once.
- `gas_mpc_eval.py` automatically takes the one-pass path for these heads while retaining the
  old loop for legacy checkpoints.
- `test_hitting_time_head.py` checks PMF normalization, monotone CDFs, equality between one-pass
  and loop ET, and finite gradients.
- Training uses the physically filtered `cache_train.npz`; graph labels and internal validation
  contain no final `task200u` evaluation episode.

## Validation plan

1. Build a directed 225-step graph-label bank once from training-only Push-T frames.
2. Train all three heads with identical labels and predictor-imagined endpoint augmentation.
3. Rank them using held-out graph-label CDF calibration and ET-vs-hitting-time correlation,
   especially the `T > 50` subset and predictor-imagined endpoints.
4. Train the selected head for learned-asset seeds 0--2.
5. Replace only OUR's critic checkpoint and run `same25`, `same50`, `same100`, and `cross` on the
   first 50 tasks of the fixed `task200u` pool, paired CEM/asset seeds 0--2. All other OUR settings
   remain unchanged.

Results are appended after the training-only run completes.
