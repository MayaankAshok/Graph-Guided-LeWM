Yes. I went back through the relevant papers—including the papers that appeared **after LeWorldModel**, the August 2026 papers on decision-metric alignment, SCALE, AC-MTM, and the July 2026 SAGE paper—and I think there is a **genuine gap**.

The important conclusion is that your original idea should **not** be framed as “replace Euclidean distance with an asymmetric distance.” That territory is now crowded.

The stronger paper is:

# Proposed paper: **Viability-Aware Latent Planning for Irreversible World Models**

### Core idea

Instead of asking

$$
\text{“How close is my predicted endpoint to the goal?”}
$$

or even

$$
\text{“Was this endpoint reachable within }H\text{ steps?”},
$$

ask:

$$
\boxed{
\text{“From this predicted endpoint, what is the probability of still being able to reach the goal?}
\text{”}
}
$$

That leads to a **finite-horizon, directed, goal-conditioned viability value**

$$
V_\psi(z,g,h)
\approx
P(\text{reach }g\text{ within }h
\mid z).
$$

The planner then scores a candidate rollout ending at \(\hat z_H\) with

$$
C(a_{1:H})
=
-\log V_\psi(\hat z_H,z_g,H_{\rm rem}).
$$

I'd call this something like **Latent Viability Value (LVV)** or **Goal-Conditioned Viability Value (GCV)**.

The key novelty is not merely asymmetry. It is:

> **Learn a decision-theoretic, horizon-conditioned viability function that evaluates imagined states produced by the world model, rather than learning a symmetric distance over states observed in the offline dataset.**

That distinction is substantial.

---

# Why I think this is the actual gap

The current literature has converged on four different ways of attacking the problem.

| Work                       | What it fixes                                                                       | What remains                                                                                                        |
| -------------------------- | ----------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| **LeWM**                   | Predictive latent representation + Euclidean MPC                                    | Euclidean terminal distance can be a bad decision metric                                                            |
| **Temporal Straightening** | Reshapes representation so Euclidean distance better approximates geodesic distance | Still fundamentally a symmetric Euclidean goal metric; authors explicitly identify this limitation                  |
| **RC-aux**                 | Learns budget-conditioned reachability and trains multi-horizon prediction          | Reachability labels are trajectory-derived proxies; planner uses a relatively simple reachability gate              |
| **TRM**                    | Post-hoc learned terminal metric for candidate ranking                              | Its primary temporal supervision is a symmetric temporal-distance proxy, not a directed optimal goal-reaching value |
| **SCALE**                  | Makes task-relevant state affect latent Euclidean geometry                          | Requires privileged/task-specific state and remains symmetric                                                       |
| **DA-LeWM**                | Action-conditioned auxiliary objectives improve Euclidean-cost geometry             | Even improved variants show near-zero CEM elite-stage rank correlation                                              |
| **SAGE**                   | Improves proposal distribution using predicted reachable subgoals                   | Does not fundamentally replace the terminal decision metric                                                         |
| **QRL / Multistep QRL**    | Learns quasimetric goal-reaching values                                             | General GCRL framework, not an imagined-rollout terminal-value interface for JEPA MPC                               |
| **AGiLe**                  | Bidirectional latent planning + forward reachability critic + affordances           | Broader manipulation framework; not the specific model-based viability-value formulation below                      |

These distinctions are supported directly by the papers. RC-aux explicitly describes trajectory-derived finite-budget labels as proxies rather than true shortest-path reachability, while TRM explicitly characterizes its temporal label as a **symmetric scalar proxy rather than a directed goal-reaching value**. ([arXiv][1]) Temporal Straightening similarly says its current formulation uses a **symmetric Euclidean goal cost**. ([arXiv][2]) SCALE calibrates Euclidean geometry against task-state distance, rather than learning a directed control value. ([arXiv][3]) DA-LeWM improves global ranking, but its CEM elite-stage correlations remain close to zero, showing that the local terminal-selection problem is not solved by generic action-conditioned auxiliary supervision. ([arXiv][4])

And that gives us the opening.

---

# The missing object is not a distance

Consider two predicted endpoints \(z_A\) and \(z_B\).

Suppose

$$
\|z_A-z_g\| < \|z_B-z_g\|.
$$

Euclidean MPC prefers \(A\).

Now suppose:

$$
P(\text{goal}\mid z_A,H=25)=0.02
$$

because \(A\) has put the pusher on the wrong side of the object,

while

$$
P(\text{goal}\mid z_B,H=25)=0.75.
$$

Then \(B\) is clearly the better endpoint despite being farther away.

This is more fundamental than "distance is asymmetric."

The planner actually wants an approximation to

$$
\boxed{
V^*_H(z,g)
=
\max_{\pi}
P_\pi(
\tau_g\le H
\mid z_0=z
)
}
$$

or, if minimizing effort,

$$
\boxed{
J^*_H(z,g)
=
\min_{\pi}
E_\pi[
\text{cost to reach }g
\mid z_0=z].
}
$$

That is a **value function / viability function**, not a metric.

And because it is goal-conditioned:

$$
V_H(z,g)\neq V_H(g,z)
$$

in general.

Because it is finite-horizon:

$$
V_H(z,g)\neq V_{H+50}(z,g).
$$

And because it represents *future controllability*:

$$
V_H(z,g)
$$

can distinguish a state that is physically close to the target from a state that is **already a dead end**.

That is precisely what you were intuiting with PushT.

---

# But doesn't QRL already do this?

This is the most important novelty objection.

Yes, **Quasimetric RL** already establishes that optimal goal-reaching value functions have directed/quasimetric geometry. The 2025 NeurIPS work further unifies quasimetric representations with successor representations, and Multistep QRL extends this to long-horizon offline goal reaching. ([Proceedings of Machine Learning Research][5])

So I would **not** claim:

> “We introduce a directed goal-reaching value.”

That is definitely not novel.

The novelty should instead be:

## **A model-based viability interface for JEPA planners**

The important distinction is this:

QRL learns

$$
V(z,g)
$$

for states encountered in the GCRL learning problem.

Our planner produces

$$
\hat z_H =
F_\theta(z_0,a_{1:H})
$$

that may be **off the data manifold**.

The question is therefore:

$$
\boxed{
\text{Can a learned goal-reaching value be evaluated on counterfactual states generated by the world model?}
}
$$

That's a much more specific world-model question.

---

# The key technical innovation: train on both observed and imagined states

This is where I'd make the paper substantially different from TRM and RC-aux.

### Stage 1: Learn the usual LeWM

$$
z_t=E_\theta(o_t)
$$

$$
\hat z_{t+1}=F_\phi(\hat z_t,a_t).
$$

Keep the backbone unchanged initially.

### Stage 2: Learn a viability critic

Train

$$
V_\psi(z,g,h)
\in[0,1].
$$

Use hindsight pairs from offline trajectories.

For observed transitions:

$$
V_\psi(z_t,z_{t+\Delta},h)
$$

should be high when \(h\ge\Delta\).

But don't stop there.

Generate **counterfactual trajectories through the frozen world model**:

$$
\tilde z_{t+k}
=
F_\phi(\tilde z_{t+k-1},\tilde a_{t+k-1}).
$$

Now train the critic on both:

$$
(z_t,z_{t+\Delta},h)
$$

and

$$
(\tilde z_t,\tilde z_H,h).
$$

This addresses one of the big weaknesses of trajectory-pair methods: their labels only describe what happened in the recorded data.

RC-aux itself acknowledges that trajectory-derived labels are only a proxy for environment-level attainability. ([arXiv][1])

---

# Even more important: use Bellman-style viability consistency

This is the part I think can make the paper much stronger.

For deterministic dynamics:

$$
V^*_{h}(z,g)
=
\max_{a}
V^*_{h-1}(F(z,a),g)
$$

with

$$
V^*_0(z,g)
=
\mathbf 1[z\in G].
$$

Instead of merely teaching the network that two states are "close in time," teach it the recursive structure of the actual control problem.

Define

$$
Q_\psi(z,a,g,h)
=
V_\psi(F_\phi(z,a),g,h-1).
$$

Then

$$
V_\psi(z,g,h)
\approx
\operatorname{SoftMax}_a
Q_\psi(z,a,g,h).
$$

So the learned object has an explicit interpretation:

> **Does there exist a sequence of actions through the learned dynamics that can still reach the goal?**

This is much closer to what the planner actually needs.

---

# The planner then becomes very simple

LeWM currently does:

$$
a^*
=
\arg\min_a
\|
F_\phi(z_0,a)-z_g
\|^2.
$$

Replace it with:

$$
a^*
=
\arg\max_a
V_\psi(
F_\phi(z_0,a),
z_g,
H).
$$

Or use a hybrid:

$$
C(a)
=
\alpha
\|F_\phi(z_0,a)-z_g\|^2
-
\beta
\log
V_\psi(F_\phi(z_0,a),z_g,H).
$$

For PushT, I actually expect the hybrid to be the stronger version.

This follows an important empirical observation from TRM: on contact-rich PushT, replacing latent distance entirely is less reliable; a hybrid terminal cost can retain useful physical signal. ([arXiv][6])

---

# The really interesting part: add uncertainty

There's a second issue that the current literature exposes.

A world model can imagine a state that looks excellent according to its learned geometry but is actually a model hallucination.

So instead of

$$
V_\psi(z,g,H)
$$

use an ensemble

$$
\{V_{\psi_k}\}_{k=1}^K
$$

and define

$$
C_{\text{viab}}
=
-\mu_V
+
\lambda\sigma_V.
$$

Or probabilistically:

$$
C_{\text{viab}}
=
-\log
\left(
\mu_V-\lambda\sigma_V
\right).
$$

Now the planner asks:

> "Is the goal reachable from this imagined state, and are we confident about that judgment?"

That gives you:

$$
\boxed{
\text{Planning cost}
=
\text{goal progress}
+
\text{future viability}
+
\text{model uncertainty}
}
$$

rather than just geometric proximity.

---

# Why this is especially appropriate for PushT

PushT isn't merely an obstacle-navigation problem.

The important state can include:

$$
(x_{\rm object},y_{\rm object},\theta_{\rm object},
x_{\rm pusher},y_{\rm pusher},\ldots)
$$

and the effect of an action depends strongly on contact configuration.

So two states can be similarly close to the goal while having completely different future controllability.

This is exactly where a viability value should beat a static state distance.

LeWM itself reports that meaningful physical variables—including block position and angle—are decodable from its latent, so the proposed work would not need to argue that the world model necessarily lacks the relevant information. ([arXiv][7])

That distinction is important.

---

# There is an even stronger scientific hypothesis

I'd make the central hypothesis:

$$
\boxed{
\text{Predictive quality}
\neq
\text{decision quality}
\neq
\text{viability estimation quality}.
}
$$

Then test the three separately.

### Representation

Can we decode the physical state?

### Dynamics

Can the world model predict the outcome?

### Viability

Can we correctly estimate whether a predicted endpoint can still reach the goal?

### Planner

Does CEM select the right candidate?

This creates a very clean failure decomposition.

It also directly builds on the recent observation that **information sufficiency and decision-metric alignment are distinct**. ([arXiv][4])

---

# A particularly strong experiment

Suppose CEM produces 10,000 candidates.

For every candidate \(i\), record:

$$
d_i^{latent}
$$

$$
d_i^{TRM}
$$

$$
d_i^{SCALE}
$$

$$
V_i^{ours}
$$

and, because this is simulation, the oracle quantity:

$$
p_i^*
=
P(\text{reach goal within remaining budget}).
$$

Then measure:

$$
\rho(
d_{\rm latent},
p^*
)
$$

versus

$$
\rho(
d_{\rm TRM},
p^*
)
$$

versus

$$
\rho(
V_{\rm ours},
p^*
).
$$

But the most important measurement should be:

$$
\boxed{
\text{rank of the best feasible candidate}
}
$$

within the 10,000 candidates.

This directly measures what matters to CEM.

TRM already introduced the idea of auditing candidate ranking rather than relying solely on final success. ([arXiv][6])

Your paper can take that one step further by measuring **viability ranking**, not merely temporal distance ranking.

---

# The experiment that would make the paper hard to dismiss

Construct controlled PushT "irreversibility" cases.

Start with a state \(s\) and goal \(g\).

Find candidate endpoints \(z_1,z_2\) such that:

$$
\|z_1-z_g\|
<
\|z_2-z_g\|,
$$

but

$$
V^*(z_1,g,H)
<
V^*(z_2,g,H).
$$

Then measure the three planners.

### Euclidean LeWM

Chooses \(z_1\).

### TRM / RC-aux

May recognize temporal reachability.

### Viability planner

Chooses \(z_2\).

The important part is to create hundreds or thousands of these **metric inversion cases**.

Then you can claim something much stronger than:

> "Our method gets +8% success."

You can show:

> **Euclidean proximity systematically prefers states with lower future viability.**

That is a scientific phenomenon.

---

# Why I prefer this to another "better distance"

The problem with proposing

$$
d'(z,g)
$$

is that reviewers can reasonably say:

> "This is yet another learned distance. Why isn't TRM/SCALE/QRL sufficient?"

The answer becomes much stronger when you say:

$$
\boxed{
\text{We do not seek a better geometry.
We learn the finite-horizon control value induced by the dynamics.}
}
$$

Then the distinction is:

**SCALE**

$$
z\rightarrow\text{task-state geometry}
$$

**TRM**

$$
z,g\rightarrow\text{temporal distance}
$$

**RC-aux**

$$
z,g,h\rightarrow\text{trajectory-derived reachability}
$$

**Our method**

$$
\boxed{
z,g,h
\rightarrow
\text{future goal viability under the learned dynamics}
}
$$

---

# Where the novelty is strongest

I would explicitly define three properties:

### 1. Directed

$$
V(z,g,h)\neq V(g,z,h)
$$

### 2. Budgeted

$$
V(z,g,h_1)\le V(z,g,h_2)
\qquad h_1<h_2
$$

for a reachability-probability interpretation.

### 3. Counterfactual

The critic is evaluated on

$$
z=F_\phi(z_0,a_{1:t})
$$

generated by arbitrary candidate action sequences—not only states appearing in the dataset.

That third property is the part I would make the centerpiece.

---

# And there is a nice theoretical result available

Under an exact Markov latent representation and exact dynamics,

$$
V^*_0(z,g)
=
\mathbf 1[d(z,g)\le\epsilon]
$$

and

$$
V^*_{h}(z,g)
=
\max_a
V^*_{h-1}(F(z,a),g).
$$

Then define

$$
D_h(z,g)=-\log V^*_h(z,g).
$$

You immediately obtain a directed, horizon-dependent cost.

This gives you a bridge to quasimetric goal-reaching theory without claiming to reinvent quasimetric RL. QRL establishes the broader mathematical relationship between optimal goal-reaching values and quasimetric structure. ([Proceedings of Machine Learning Research][5])

Your contribution becomes the **world-model integration and counterfactual viability estimation**.

---

# What I would call the paper

My favorite:

## **"Latent Viability: Beyond Proximity for Planning with World Models"**

or, more technical:

## **"Viability-Aware Latent MPC for Irreversible Dynamics"**

The thesis sentence would be:

> **Latent world models should expose not a distance to the goal, but the remaining viability of reaching the goal under the available action budget.**

That's a crisp paper story.

---

# Conference-worthiness

I'd rate the idea approximately:

| Criterion                          |                                                            Assessment |
| ---------------------------------- | --------------------------------------------------------------------: |
| Genuine problem                    |                                                              **9/10** |
| Novel relative to LeWM             |                                                              **9/10** |
| Novel relative to 2026 literature  |                                                          **7.5–8/10** |
| Theoretical depth                  |                                                              **8/10** |
| Experimental tractability          |                                                              **9/10** |
| PushT suitability                  |                                                              **9/10** |
| Risk of being viewed as "just QRL" |                                                            **Medium** |
| Risk of being viewed as "just TRM" | **Low–medium**, provided imagined/counterfactual viability is central |
| Potential ICLR/NeurIPS/CoRL paper  |                                                               **Yes** |

The main novelty risk is **QRL/Multistep QRL**. You have to clearly establish that the contribution isn't merely "apply a quasimetric value head to LeWM." The counterfactual/model-generated-state aspect, CEM-interface formulation, uncertainty handling, and controlled irreversible-dynamics benchmark need to be central. ([NeurIPS Proceedings][8])

There is also an important reason to move away from a pure backward planner: **AGiLe already uses bidirectional latent planning with a backward planner and forward critic**, and SAGE already uses reachable subgoals to guide search. ([CVF Open Access][9])

So I would **not** make "bidirectional planning" the novelty.

---

# The paper I'd actually build

The minimal version is:

$$
\boxed{
\text{LeWM}
+
\text{finite-horizon viability critic}
+
\text{Bellman/hindsight training}
+
\text{CEM viability terminal cost}
}
$$

Then the experimental hierarchy is:

$$
\text{LeWM}
\rightarrow
\text{TRM}
\rightarrow
\text{RC-aux}
\rightarrow
\text{SCALE}
\rightarrow
\boxed{\text{Viability-LWM}}.
$$

And the headline result should ideally be something like:

> **When candidate trajectories exist, Euclidean and temporal-distance costs systematically select low-viability endpoints, while the proposed viability cost identifies endpoints with substantially higher probability of eventual goal completion.**

That would be much stronger than another incremental PushT success-rate improvement.

One final point: the field is moving *very* quickly. As of **September 9, 2026**, papers such as SCALE and Decision-Metric Alignment were only submitted in August 2026, so I would treat this as a **promising novelty hypothesis rather than a guaranteed first claim**. ([arXiv][3]) The exact contribution should be locked down against the latest arXiv/OpenReview versions immediately before writing.

[1]: https://arxiv.org/pdf/2605.07278 "Predictive but Not Plannable: RC-aux for Latent World Models"
[2]: https://arxiv.org/pdf/2603.12231 "Temporal Straightening for Latent Planning"
[3]: https://arxiv.org/pdf/2608.16287 "SCALE: State-Calibrated Latent Embeddings for JEPA Planning in the Right Geometry"
[4]: https://arxiv.org/pdf/2608.18746 "Decision-Metric Alignment in Latent World Models: Diagnostics and Action-Conditioned Objectives for MPC Planning"
[5]: https://proceedings.mlr.press/v202/wang23al.html?utm_source=chatgpt.com "Optimal Goal-Reaching Reinforcement Learning via Quasimetric Learning"
[6]: https://arxiv.org/pdf/2605.22164 "World Model Control by Trajectory Reachability Metrics"
[7]: https://arxiv.org/pdf/2603.19312 "LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels"
[8]: https://proceedings.neurips.cc/paper_files/paper/2025/hash/1c5956164472d6d8123d574aa75cd063-Abstract-Conference.html?utm_source=chatgpt.com "Offline Goal-conditioned Reinforcement Learning with Quasimetric Representations"
[9]: https://openaccess.thecvf.com/content/CVPR2026/html/Chen_AGiLe_Learning_Robust_Long-Horizon_Manipulation_via_Affordance-Grounded_Bidirectional_Latent_Planning_CVPR_2026_paper.html?utm_source=chatgpt.com "CVPR 2026 Open Access Repository"


---
# Experiments
Yes. And after looking at the latest LeWM-adjacent work, I would **not** use standard PushT success as the main experiment. Standard PushT is already too easy for LeWM: the original paper reports about **96% success at a 25-step goal with a 50-step execution budget**, and newer methods likewise get high short-horizon performance.

The right experiment is therefore to **manufacture and measure the specific failure mode**:

> Two predicted states can be similarly close to the goal in latent/task distance, while having radically different *future controllability*.
>
>

Then we ask whether our viability function detects that difference.

The nice thing is that the latest papers already give us several pieces of the evaluation machinery. TRM uses shared candidate pools and SASC-style rank audits; DA-LeWM explicitly measures Plan-Real and CEM-stage Spearman; RC-aux already studies finite-budget reachability; SAGE shows that long-horizon PushT is substantially harder than the standard short-horizon setting.

So I would build the experimental section around **five questions**.

---

# 1. First establish that the failure actually exists in PushT

This should be Experiment 1 and probably the most important figure in the paper.

## A. Generate lots of candidate terminal states

Take a real PushT state and a goal .

From , generate perhaps:

candidate action sequences using the same CEM candidate distribution as LeWM.

For every candidate , record:

and the corresponding **true simulator state**

Then compute three things:

### Latent cost

### Immediate task distance

For example,

### True remaining control cost

This is the crucial quantity.

From each , run a **privileged simulator planner** with access to the true state and true dynamics.

For instance:

or, even better,

Because PushT is deterministic, you can get a very strong approximation to this by running a much larger action search / MPPI / fine-grained CEM from the true state.

Then plot:

and

### What would be a failure?

You want to find many examples of

but

These are **metric inversions**.

Even stronger:

while

That is direct evidence that latent proximity is not the quantity a planner actually wants.

This is closely related to the candidate-ranking diagnosis used by TRM and DA-LeWM, but PushT is especially valuable because here the failure can arise from **contact configuration and recovery**, rather than simply a geometric obstacle. TRM's latest revision explicitly says PushT exposes layout, dynamics, and recovery limitations, while DA-LeWM finds that even action-conditioned training leaves CEM's elite-stage rank correlation near zero.

---

# 2. Make the "irreversibility" failure explicit

This is the experiment that connects directly to your original intuition.

The mistake would be to simply say:

> "PushT is irreversible."
>
>

Instead, construct **locally deceptive states**.

## State-pair experiment

Find pairs satisfying approximately

but with different pusher configurations.

For example:

### State A

The block is almost at the target, but the pusher is on the wrong side and must perform a large repositioning maneuver.

### State B

The block is slightly farther away, but the pusher is already in the correct contact configuration.

Then measure

and

The strongest cases are:

but

These are exactly the states where a **viability value** should differ from a geometric distance.

---

# 3. Build a "dead-end map" of PushT

This could become one of the paper's most compelling visualizations.

Instead of plotting just latent states, create a grid over:

For each state, ask:

Color the state according to probability of eventual success.

You should see regions like:

Then overlay:

The scientific question becomes:

> Does latent Euclidean geometry recognize the boundary between recoverable and unrecoverable configurations?
>
>

My expectation is that it will not do so cleanly.

Now overlay our

The desired result is that the viability boundary aligns much more closely with the true simulator boundary.

This is much stronger than "our success rate is 3% higher."

---

# 4. The most important experiment: candidate ranking

This should be the centerpiece of the paper.

Recent DA-LeWM work already argues that ordinary representation probes are insufficient and introduces **Plan-Real Spearman** and **CEM-stage Spearman**; importantly, on PushT it found that even stronger action-conditioned representations still had approximately zero elite-stage correlation.

We should adopt the same philosophy.

For each start/goal pair:

1. Sample one fixed candidate population.
2. Freeze it.
3. Score it using every method.
4. Compare each ranking against simulator ground truth.

For candidate :

Then compute:

But **do not stop at global Spearman**.

Measure it at:

Exactly this sort of audit has exposed a very interesting phenomenon in DA-LeWM: global ranking improves, but elite-stage correlations collapse toward zero.

### What we want

Our method should maintain a meaningful relationship at the elite stage:

and ideally

remains strongly positive.

That would be much more convincing than just higher closed-loop success.

---

# 5. Test whether the viability score actually means what we claim

This is where I'd distinguish our method from simply "another learned ranking head."

Suppose

claims to estimate probability of eventual success.

Then test whether

is actually **calibrated**.

Take thousands of states and bin them:

| Predicted viability | Empirical success |
| --- | --- |
| 0.0–0.1 | ? |
| 0.1–0.2 | ? |
| ... | ... |
| 0.9–1.0 | ? |

You want approximately:

Report:

and

for distinguishing reachable vs unreachable-within-budget states.

This would give our function an actual semantic meaning.

TRM deliberately describes its head as a pairwise distance surrogate rather than a formally defined metric; RC-aux predicts finite-budget reachability, but its own paper emphasizes that trajectory-derived supervision is an empirical reachability proxy.

So a **calibrated probability of future success** is a useful way of making our contribution conceptually distinct.

---

# 6. Horizon experiment: does viability behave correctly as the budget changes?

This is essential.

For every state/goal pair compute

for

True reachability should generally satisfy

So test whether the learned value follows this monotonic structure.

This directly distinguishes your method from an ordinary static distance.

A state could have:

but

That's useful information to the planner.

This is particularly motivated by RC-aux's finding that matching the supervision horizon matters significantly; TRM similarly shows that short-horizon training can fail badly when the planner operates at longer horizons.

---

# 7. The "near but doomed / far but viable" experiment

I would make this an explicit benchmark.

Construct four classes:

### A — Near + viable

### B — Near + doomed

### C — Far + viable

### D — Far + doomed

The two interesting quadrants are:

and

A Euclidean planner should systematically prefer the former.

A viability-aware planner should prefer the latter.

This gives you a very intuitive figure for reviewers.

---

# 8. Recovery experiment

This is where PushT becomes more than a generic navigation benchmark.

Take a successful trajectory and perturb it at time .

For example:

where changes:

- block position,
- block orientation,
- pusher position,
- relative pusher–block configuration.

Now ask:

And compare with

The strongest result would be that our method correctly detects:

> "This state is only 3 pixels closer to the goal, but you've lost the contact configuration and have very little recovery budget."
>
>

This directly addresses the original motivation much better than static start/goal evaluation.

---

# 9. Dataset coverage experiment

This is **absolutely necessary** because there is an important weakness in the current reachability literature.

TRM shows that coverage of the trajectory database is critical: insufficient doorway coverage can destroy the repair even when the model itself is fine. Its latest version explicitly calls coverage a boundary condition.

So take the PushT expert dataset and train our viability model with:

of the demonstrations.

Then evaluate on:

### interpolation

states similar to training states;

### extrapolation

novel starting configurations;

### contact extrapolation

different pusher–block configurations;

### goal extrapolation

different object orientations / target offsets.

The critical question is:

This matters because the planner does not only encounter training states.

---

# 10. Model-error experiment

This one could substantially strengthen the paper.

There are two things that can go wrong:

and

We need to separate them.

For every candidate endpoint, compare:

predicted by LeWM against

actually produced by the simulator.

Define model error:

Then plot:

You want to establish that when

Euclidean ranking can still fail.

That's crucial.

Otherwise reviewers can say:

> "The problem isn't the metric. LeWM simply predicts PushT poorly."
>
>

The TRM authors already found that contact-rich PushT remains a boundary case where improved candidate ranking does not always translate into equally large closed-loop improvements because dynamics/recovery errors remain.

Your experiments should explicitly quantify this.

---

# 11. The "shared candidates" ablation is non-negotiable

For each start/goal pair:

must be identical for every method.

Then compare:

and

Why?

Because SAGE, for example, explicitly improves **proposal generation**, while TRM changes **candidate ranking**.

Without shared candidates, a reviewer cannot tell whether our method is better because it scores candidates better or because it happened to generate better candidates.

---

# 12. Then do closed-loop control

Only after the diagnostic experiments.

Use several difficulty levels:

SAGE already establishes this as a useful PushT range and shows the base planner deteriorates dramatically at long offsets: its Base CEM is about 64%, 40%, 20%, 10%, 6%, 14% across the reported long-horizon settings for one seed, whereas SAGE substantially improves them.

So our central result shouldn't be:

> "We beat LeWM at ."
>
>

It should be:

> **The larger the gap between geometric proximity and future viability, the larger the benefit of viability-aware planning.**
>
>

That is a much more scientific prediction.

---

# The experiment I would make the headline

I'd create a new metric:

## **Viability–Proximity Inversion Rate**

For a candidate pair , define an inversion when

but

Then:

Now stratify by task difficulty.

You might discover something like:

but

and

I'm not claiming those numbers—they are exactly what we should measure.

But **if such a curve appears**, you've demonstrated the underlying phenomenon instead of simply optimizing a benchmark.

---

# What would convince me our method is suitable?

I'd establish **five go/no-go criteria**.

These are research targets, not claims about existing literature.

### 1. The failure must be measurable

There should be a substantial population of:

states.

If

on PushT, then this is the wrong environment for the proposed method.

---

### 2. Viability must predict ground truth

Something like:

should be strong, and classification performance for reachable/unreachable states should be substantially above trivial baselines.

I'd regard roughly

as encouraging and

as very strong, but these should be considered practical targets rather than universal thresholds.

---

### 3. It must improve candidate selection, not just offline prediction

The most important number is:

I'd want our value to outperform latent L2, TRM, and RC-aux substantially **on the same candidate pool**.

And ideally the oracle-best candidate should move from something like

toward

TRM demonstrates how informative this type of measurement is: on its TwoRoom diagnostic, the oracle best sampled candidate moved from roughly the 39th percentile under raw latent MSE to around the 4th percentile with TRM.

---

### 4. The gain should increase exactly where our hypothesis predicts

This is perhaps the strongest criterion.

Define an "inversion severity" for each task:

Then show:

In other words:

> **Our method should help most when Euclidean proximity is most misleading.**
>
>

That would be extremely compelling.

---

### 5. It must not destroy easy cases

On ordinary short-horizon PushT:

LeWM is already very strong.

Our method shouldn't turn a 95%-type easy task into 80%.

Ideally:

while

That's exactly the behavior we would want from a planner modification.

---

# The final experimental matrix I'd use

| Experiment | Question | What would support us? |
| --- | --- | --- |
| **Metric inversion** | Does latent closeness disagree with true future controllability? | Many near-but-doomed / far-but-viable cases |
| **Dead-end map** | Are there recoverability regions invisible to L2? | Viability boundary matches simulator |
| **Directed pairs** | Does direction/contact configuration matter? | empirically |
| **Horizon sweep** | Is future controllability budget-dependent? | Correct horizon monotonicity |
| **Calibration** | Does mean probability of success? | Good ECE/Brier/AUROC |
| **Shared-candidate SASC** | Does it rank better? | Higher Spearman + better oracle-best rank |
| **CEM-stage audit** | Does ranking survive optimizer concentration? | Strong elite-stage correlation |
| **Recovery perturbation** | Can it detect doomed states? | Accurate drop in viability |
| **Model-error control** | Is the issue metric rather than dynamics? | Metric failures even with accurate rollouts |
| **Dataset coverage** | Does viability generalize? | Robust under trajectory subsampling |
| **Long-horizon control** | Does this translate to real planning? | Larger gains at 75–150 steps |
| **Baseline comparison** | Is it better than existing fixes? | Beats L2/TRM/RC-aux/DA-LeWM under matched candidates |

---

# And there is a particularly nice paper structure

I would actually make the paper read like a scientific investigation:

### **Figure 1 — The failure**

Show two candidate PushT rollouts:

LeWM selects A.

Our method selects B.

---

### **Figure 2 — The phenomenon exists at scale**

Plot:

with many inversion cases.

---

### **Figure 3 — Viability map**

Show simulator reachability vs our prediction.

---

### **Figure 4 — Ranking**

Shared candidate pool:

Show Plan-Real and CEM-stage Spearman.

---

### **Figure 5 — Horizon**

and demonstrate that our advantage grows as the problem becomes more budget-constrained.

---

### **Figure 6 — Closed loop**

Success vs goal offset.

Critically, plot:

That would be my favorite result.

---

# The most important caveat

There is a real danger that these experiments reveal:

And that would actually be a **valuable result**.

The latest TRM paper already reports exactly this kind of boundary: on PushT, its task-state metrics substantially improve candidate ranking and selected final distance, but closed-loop success improves only modestly because contact, rollout, and recovery remain limiting factors.

So our method is suitable **only if the experiments show the following chain**:

If **all five** hold, I think you have the basis of a very credible conference paper.

If only the last one holds, it's probably just another planner tweak.

If the first three hold but the last two don't, then we have discovered that **PushT is a good diagnostic benchmark for the phenomenon but not necessarily a sufficient benchmark for demonstrating end-to-end benefit**. In that case, the natural next step would be to create a controlled family of irreversible/contact-rich tasks around PushT rather than relying on vanilla PushT alone.

The strongest immediate experiment, therefore, is **not training our method yet**. It is to take the frozen LeWM PushT checkpoint, generate ~10k candidate rollouts per start/goal pair, compute simulator-ground-truth remaining controllability, and measure the **nearer-but-less-viable inversion rate**. That one experiment will tell us whether the research hypothesis is actually present in the data before we invest in the viability model.