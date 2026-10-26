# DreamerV3 fidelity and the HelixWM intervention

Status: protocol documentation, 2026-10-26. No new comparison result is reported here.

## What “original DreamerV3” means in this comparison

The reference is the authors' DreamerV3 implementation at [`e3f02248693a79dc8b0ebd62c93683888ddaccfe`](https://github.com/danijar/dreamerv3/tree/e3f02248693a79dc8b0ebd62c93683888ddaccfe), associated with [Hafner et al., Nature (2025)](https://www.nature.com/articles/s41586-025-08744-2). This is a pinned author implementation of the published algorithm, not a claim to reproduce the initial 2023 repository revision or every paper setting.

Both arms use the official `size12m` preset. This matches capacity between arms; it is **not** the approximately 200M default preset. Task, action repeat, training budget, replay capacity, data collection, checkpointing and shared policy synchronization adaptations must be recorded separately from algorithm fidelity. No custom first-pilot learner substitutes for the official learner.

## Shared state-space model

With observation $x_t$ following action $a_{t-1}$:

$$h_t=f_\theta(h_{t-1},z_{t-1},a_{t-1}),\qquad
z_t\sim q_\theta(z_t\mid h_t,x_t),\qquad
\hat z_t\sim p_\theta(z_t\mid h_t).$$

The deterministic state $h_t$ and discrete stochastic state $z_t$ form the belief $s_t=(h_t,z_t)$. The encoder/posterior incorporates observations; the prior predicts without the current image. The actor uses this belief, not privileged simulator state.

The selected preset has a deterministic state of 2048, hidden width 256, 32 categorical variables with 16 classes, and eight recurrent blocks. Preserve straight-through categorical sampling, 1% uniform mixing, the author normalization and activation choices, and the native decoder/reward/continuation heads. RGB preprocessing is `uint8 / 255 - 0.5`; do not apply vector-observation symlog to RGB pixels.

## Shared world-model objective

Let $q_t=q_\theta(z_t\mid h_t,x_t)$, $p_t=p_\theta(z_t\mid h_t)$ and $\mathrm{sg}$ denote stop-gradient. The native model objective can be summarized as:

$$\mathcal L_{\rm WM}=\mathbb E_t\left[
\ell_{{\rm pred},t}
 + \max\left(1,D_{\rm KL}(\mathrm{sg}(q_t)\Vert p_t)\right)
 + 0.1\max\left(1,D_{\rm KL}(q_t\Vert\mathrm{sg}(p_t))\right)
\right].$$

Here $\ell_{\rm pred}$ is the sum of native observation reconstruction, reward prediction and continuation losses with their pinned configuration scales. Apply the one-nat free threshold at the same reduction level as the author code, not separately to every categorical class. Image reconstruction is the native squared-error head; the likelihood notation is shorthand for that implemented objective. Reward and value use the author 255-bin transformed two-hot representation.

The new matched protocol gives **both learners only the actually visible RGB stream**. A missing frame is a zero-valued RGB image before native preprocessing. Clean images hidden by the synthetic sensor cannot become replay targets. This changes the historical HelixWM data protocol, which retained clean supervision; the old checkpoint therefore remains a historical reference rather than one of the new primary arms.

## Shared imagination, actor and critic

For imagined states, let $d_t$ be the effective discount-continuation factor and $v_t$ the critic baseline. The lambda target has the familiar form

$$R_t^\lambda=r_{t+1}+d_{t+1}\left[(1-\lambda)v_{t+1}+\lambda R_{t+1}^\lambda\right],\qquad \lambda=0.95.$$

The selected implementation uses `contdisc=True`: the continuation target already incorporates the discount associated with horizon 333. The imagination loss consequently uses a unit additional discount. **Do not multiply the horizon discount twice.** Preserve the native indexing, terminal bootstrap and imagination weights by calling the author implementation rather than rewriting this schematic recursion.

The policy loss uses the native score-function objective, schematically

$$\mathcal L_\pi=-\mathbb E_t\left[w_t\left\{
\log\pi_\theta(a_t\mid s_t)\,\mathrm{sg}\left(\frac{R_t^\lambda-v_t}{\max(1,S)}\right)
 +\eta\,\mathcal H[\pi_\theta(\cdot\mid s_t)]\right\}\right],$$

where $S$ is the native smoothed 5th-to-95th-percentile return range and $\eta=3\times10^{-4}$. This continuous-action setting should not silently be replaced by a custom pathwise actor-gradient objective. The critic fits native two-hot lambda targets and retains slow-value regularization and the replay value loss. Preserve imagination horizon 15, the slow-value update, return normalization, adaptive gradient clipping and the author's optimizer configuration.

The continuous distribution is the author's bounded-normal parameterization (bounded mean and standard deviation), with the native environment action clipping. It is not interchangeable with sampling a normal and applying a new tanh transform. Evaluation samples from the pinned policy using private, recorded evaluation RNGs; it is not silently changed to mean-action evaluation.

## Where HelixWM differs

For the native reference, a missing image travels through the ordinary encoder/posterior. For HelixWM, known availability controls an external state-estimation adapter:

1. A valid image anchors an observation-conditioned state.
2. During a supported missing segment, candidate states branch from the same anchor and propagate through the **same learned prior**, using the actual executed actions.
3. When an image returns, decoder evidence weights the candidate histories. One complete ancestry is selected, including recurrent memory; that selected candidate becomes the current belief. Preserved V2 does not additionally apply a native posterior to that same return image. Ordinary observation filtering resumes subsequently; adding an extra posterior here would be a new intervention.
4. Unsupported long gaps use the existing bounded fallback and are reported separately. A model prediction never overwrites a recorded image or retroactively changes an action.

For candidate history $i$ and returning image $x_t$, the existing energy weighting is schematically

$$w_i=\frac{\exp[-e_i(x_t)/\tau]}{\sum_j\exp[-e_j(x_t)/\tau]},\qquad
\mathrm{ESS}=\frac{1}{\sum_iw_i^2}.$$

The normalized decoder reconstruction energy $e_i$ is **not** a calibrated observation likelihood. Candidates are already prior-proposed; multiplying another transition prior into these weights would count it twice. The fixed temperature is 0.01, candidate limit eight, and supported repair history at most five missing observations. Long-gap tests at 10, 20 and 50 exercise robustness and fallback, not a claim of a 50-frame repair algorithm.

This adds an inference mechanism, not a new model-loss formula or a demonstrated DNA replication theorem. Both arms retain the official learning objectives. The `prior_only` evaluation adapter is necessary to distinguish multi-candidate repair from the simpler benefit of recognizing a missing image and skipping its encoder.

## Execution adaptations and parity gates

Use identical replay semantics, recurrent context, sample distribution, update accounting, parameter initialization seed, compute dtype and shared scheduling in both new arms. Historical HelixWM used custom independent replay crops with 32-step burn-in and disabled replay context; those settings must not silently appear only in the Helix arm of the new experiment.

The two arms share episode-boundary inference snapshots, explicitly declared as an execution-layer adaptation. A candidate segment must never mix parameter versions. This is compatible with retaining native model/actor/critic formulas, but it is not a claim that the author training launcher and synchronization are unchanged.

The native driver's counter can include reset transitions. A wrapper must count completed action executions directly, exclude resets/evaluation, stop exactly at 1,000,000 actions per arm, and save the final state explicitly. A logger named `step` alone does not prove that budget. Checkpoints must include replay dependencies, model/optimizer/normalizers/target networks, training RNG and counters, and controller/scheduler state. Recovery evidence must state its supported boundary and software build.

Before any formal run: prove all-valid-observation adapter parity, no clean-frame access under missingness, one state transition per observation, fixed-seed reset behavior, matching action clipping/reward aggregation, finite actual-size learning, checkpoint strict reload and isolated evaluation RNG. Model, optimizer, normalizer and training-counter hashes must remain unchanged during frozen evaluation; private evaluation inference counters/RNG may advance and must not modify the training state.

## Code mapping

All author links below refer to the pinned commit, not a moving branch:

| Component | Author source |
|---|---|
| Model, policy, training losses and imagination | [dreamerv3/agent.py](https://github.com/danijar/dreamerv3/blob/e3f02248693a79dc8b0ebd62c93683888ddaccfe/dreamerv3/agent.py) |
| RSSM, encoder and decoder | [dreamerv3/rssm.py](https://github.com/danijar/dreamerv3/blob/e3f02248693a79dc8b0ebd62c93683888ddaccfe/dreamerv3/rssm.py) |
| Presets, losses and optimizer settings | [dreamerv3/configs.yaml](https://github.com/danijar/dreamerv3/blob/e3f02248693a79dc8b0ebd62c93683888ddaccfe/dreamerv3/configs.yaml) |
| Historical bounded Helix adapter | [helixwm_v2/dual_chain.py at 80cb355](https://github.com/iceezhu306-lgtm/HelixWM-for-world-model/blob/80cb3555586d01e0e7b0e8d362c27e540032d901/helixwm_v2/dual_chain.py) |
| Third-study inference variants | [helixwm_v3/variants.py at 22b93cd](https://github.com/iceezhu306-lgtm/HelixWM-for-world-model/blob/22b93cdb2b5a8643e8e14931dc2ffc5eae32a361/helixwm_v3/variants.py) |

Exact resolved configuration and source manifests, not this abbreviated mathematical description, define the executable experiment. Any required compatibility patch must be disclosed and tested in both arms before launch.
