# HelixWM Final: implementation and falsifiable claims

This is an implementation-stage research design. It is not a completed
million-action experiment or evidence of improved reliability. The historical
V2 checkpoints, results and repaired evaluator remain separate and unchanged.

## Matched comparison

The two independently initialized systems use the same pinned author
DreamerV3 implementation (`e3f02248693a79dc8b0ebd62c93683888ddaccfe`), size12m,
Walker Walk, RGB64, camera0, six bounded actions and action repeat2.
Each formal arm receives exactly 1,000,000 actual action decisions / 2,000,000
DMC calls. Environment resets and observation-availability masks are paired;
different policies naturally generate different trajectories. Missing images
are zeros in both author learners, without hidden clean targets.

Both arms use the native black-image controller for their common bootstrap:
5,000 random actions, then 500 learned-policy actions and 125 author updates.
At action5501, Final uses a file-backed measurement of its current pinned model
and auxiliary predictors. A valid but inactive or stale gate disables fusion
and planning and explicitly falls back to the native prior actor. A corrupt
artifact is an error, not a reason to bypass verification. The native
arm never uses an availability flag to select its actions. Consequently the
primary comparison is a **complete system comparison**, not an equal-information
architecture-only ablation. Native-prior and cross-controller inference tests
separate some of these factors but do not replace independent training seeds.

## Real and persistent Dream strands

Let the author's RSSM state be `z=(h,s)`, where `h` has2048 entries and `s`
contains32 categorical variables of16 classes. For executed action `a[t-1]`:

```math
h_t^R = f_\theta(h_{t-1}^R,s_{t-1}^R,a_{t-1}),\quad
p_t^R = p_\theta(s_t\mid h_t^R).
```

An accepted image yields `q(s_t|h_t,o_t)` using only the posterior suffix;
the executed action is not advanced a second time. Without accepted visual
evidence, Real uses the prior and is explicitly labeled as predicted.
Up to8 persistent Dream hypotheses follow that same executed action prefix.
Actual hypotheses and prospective action plans are separately timestamped.
The persistent missing-age limit is100; an over-budget bank is discarded.

On visible recovery, score each *pre-observation* hypothesis by the author's
decoder energy `E_i = -score(z_i,o)/12288`. Only absolute-qualified finite
energies enter the normalized weights `w_i ∝ exp(-(E_i-min E)/T)`.
Accepting the relatively best candidate is insufficient. Distributional
entropy, JSD, ESS, accumulated uncertainty, independent-head disagreement and
observed temporal error have separate fitted bounds.

The constrained categorical correction is

```math
q_\lambda=(1-\lambda)q_R+\lambda\sum_iw_iq_i,
\quad 0\le\lambda\le0.25,\quad
\overline{D_{KL}(q_\lambda\Vert q_R)}\le0.02.
```

The tested finite lambda grid only changes the categorical belief; Real's
deterministic history is retained. A recovered-frame fused sample must also
have finite decoder energy no worse than the unfused observed posterior.
Otherwise the posterior is kept. This is an engineering consistency check,
not a proof of state accuracy or safety. Qualified recovery resets accumulated
risk only if the distribution and auxiliary gates also pass; one good
candidate with collapsed ESS does not suffice.

## Distinct uncertainty measurements

Categorical predictive entropy `H(sum w_i p_i)`, expected conditional entropy
`sum w_i H(p_i)`, their JSD, latent variance and ESS describe stochastic
hypotheses of **one** Dreamer model. They are not independent-model epistemic
uncertainty, and conditional entropy is not an identified measurement of
environmental aleatoric noise.

Three independently initialized bootstrap predictors add a limited model
uncertainty proxy. Each predicts16 coarse visible-image features, normalized
actual reward and continuation. Each has an independent random feature map,
fixed prior and Poisson(1) bootstrap weights:

```math
\phi_m(x,a)=[1,\cos(W_m[x,a]+b_m)]/\sqrt{64},
\quad \hat y_m=\phi_m(\beta_m+0.1P_m),
```

```math
\beta_m=\arg\min_\beta\sum_t k_{mt}
\|y_t-\phi_m(x_t,a_t)(\beta+0.1P_m)\|^2+\|\beta\|^2.
```

The online sufficient statistics are solved at pinning, then frozen for
inference. There are3456 additional fitted coefficients, plus fixed feature
maps/priors and sufficient statistics. Only adjacent actually visible
**training** frames are used. Calibration and test data never update these
heads or the author model. Their mean feature variance is calibrated as an
epistemic **proxy**, not a Bayesian posterior or three full Dreamer models.
Shared misspecification and coarse-feature blind spots remain possible.
The D arm's identical tracker is diagnostics-only for registered inference
ablations and never affects its collection policy or author updates; its
additional CPU/storage cost is reported separately.

The fixed prior and bootstrap choice is motivated by
[Randomized Prior Functions](https://arxiv.org/abs/1806.03335).
The broader use of independent predictors is related to
[Deep Ensembles](https://arxiv.org/abs/1612.01474).
This small ridge implementation is our engineering choice, not a reproduction
of either paper's reported performance.

## Bounded multiscale planning

Two first-action proposals are considered: the native actor and the previously
executed action (zero if none exists). A duplicate proposal is removed.
Subsequent prospective actions come from the same pinned actor and use the
author's action clipping boundary. Candidate rollouts use private RNG;
diagnostic rollouts use an additional isolated RNG.

For horizons H in{1,3,5}, reward utility is

```math
R_H={1\over2H}\sum_iw_i\sum_{k=1}^H
\left(\prod_{j<k}c_{ij}\right)\operatorname{clip}(r_{ik},0,2),
\quad C_H=\sum_iw_i(1-\prod_{k=1}^Hc_{ik}).
```

All raw reward predictions are retained, including extrapolations outside
the environment's support. Only the declared utility projection is clipped.
The denominator is the structural repeat2 reward scale, so zero-reward
initialization does not cause division by a measured zero quantile.
The fixed utility subtracts uncertainty, disagreement, termination risk and
normalized first-action slew. Horizon-specific gates try5, then3, then1;
the best feasible proposal is executed. Rejected plans do not enter the world
or the author replay. No feasible plan gives the registered bounded prior
actor fallback. Bounded actions do not certify physical safety.

## Unknown observation faults

Known-mask confirmation and unknown-fault studies remain separate strata.
The unknown-fault guard accepts only visible RGB and a model-bound prediction
created before the image arrived. It screens darkness/flatness, local
high-frequency changes, repeated frames combined with innovation, and model
innovation. It never receives hidden clean pixels or true fault labels.
The first suspect image is quarantined before posterior conditioning;
two suspect frames confirm and two clear frames recover. Static scenes alone
must not trigger an immediate fault. Initial anchoring currently requires an
accepted starting image; arbitrary semantic corruption is not solved.

Thresholds use fit-only clean data; separate labeled injected-fault validation
reports false positives/negatives, quarantine rate and recovery latency.
A plain self-hashed dictionary cannot activate the guard. Model/source/data
and externally trusted manifest identities are checked against retained files.
Detection performance still requires actual learned-model measurements.

## Calibration, data and resource accounting

The active v3 registration uses **one shared exogenous calibration bank** of
20 independent seeds, each with124 actual actions:2480 actions /4960 DMC calls
once, available identically to D and F. It is never author or auxiliary training
data. Historical closed-loop-v2 design files and engineering probes are retained
but are not the active training protocol.

For each pinned version, the same10 fit and10 heldout bank rollouts are replayed
offline under virtual gaps1/5/10/20/50/100. A window contains16 warmup action
prefixes, its gap and8 recovery prefixes:6600 model-only action prefixes per
version, with **zero extra environment calls**. The executed bank actions are
forced; they are not actions selected by the current controller. Fit-only
thresholds are sealed before the heldout replay; no repeated tuning on the
heldout set is permitted. Reusing the bank across model versions does not create
new independent samples.
Uniform pilot ESS is not reused as an automatic ESS lower bound: it is derived
from the retained fit energies under the fitted energy/temperature rule.
Before/after model, optimizer, normalizer, counter, host-RNG and head hashes
must match for every measurement. Independent fresh-process numeric replay is
a separate preflight and final-checkpoint audit, rather than a new environment
collection or full-model copy at every pin.

This is **off-policy empirical calibration**, not a closed-loop certificate or
a dynamics-error bound for novel plans. Every candidate and every proposed
depth must pass a fit-only feature/action support check: its third nearest
distinct fit seed must be inside the registered fit95th-percentile radius.
An unsupported plan is rejected; if no horizon survives, the native prior
fallback is used. This finite support test can still miss shared model errors.

The per-episode pinning design entails1990 F bank suites plus one D
cross-controller suite. Its registered interaction total is3,262,480 actions
including the shared bank, training, periodic evaluation and1520 final cases,
before engineering probes, failures or separate mechanism/unknown-fault studies.
The1991 offline suites can still be computationally expensive; their measured
wall time and complete storage requirements are explicit launch gates. Pin
cadence cannot be silently reduced during a formal run.

Trajectory records stream actual latent/action/decoded-image/reward/termination/
probability/error arrays into immutable SHA-bound files. Checkpoints contain
bounded reference cursors, not another in-memory pixel-history copy. Missing
and prospective errors remain null. All required files must survive archive
and recovery. Compression, transfer time and final ZIP space must be measured
before launch; memory boundedness does not imply disk usage is small.

## What the implementation can and cannot establish

Module tests establish mechanics: timestamps, RNG independence, input isolation,
bounded history, strict restoration, and rejection of tampered evidence.
GPU learning, actual offline bank calibration, independent recovery, final
evaluation and archival are separate gates. No synthetic test grants a
production certificate. A single training seed cannot establish multi-seed
generalization or prove superior real-world reliability. The existing V2
average-return advantage and its greater failure count remain historical
observations, not evidence that Final has already improved them.
