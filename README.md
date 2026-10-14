# HelixWM

**Real-Dream Double-Helix World Model**

HelixWM is a world model research concept built around replicable, aligned Real and Dream trajectories. The clarified research target is **bidirectional, local segment repair**: reliable observations correct imagined predictions; model predictions help estimate state when observations are damaged; the corrected state supports further branching and future action selection. This DNA-inspired analogy describes computational branching and repair, not literal biological replication.

> Status, 2026-10-14: the first custom Walker seed0 pilot completed its budget but suffered a large performance regression. Final frozen evaluation: **69.38 mean return over 20 episodes**; best periodic mean: **570.98 over 5 episodes at 520k actions**. These are one training seed, not evidence of efficacy. The bounded replicated-chain mechanism now has a separate implementation in `helixwm_v2`; actual GPU, recovery and end-to-end launch gates are required before its single new experiment. There is no demonstrated performance improvement yet.

The [new experiment protocol](docs/dual_chain_experiment_protocol.md) pins the official DreamerV3 learner and explicitly adapts collection/replay for eight prior-proposed candidate histories, known missing-frame bursts, returning-evidence segment repair, and episode-boundary parameter synchronization. It runs **one HelixWM training seed only**; existing local DreamerV3 runs are not rerun or presented as matched pixel controls. The broader design remains a research proposal beyond the bounded mechanism described in that protocol.

The [frozen first experiment protocol](docs/first_experiment_protocol.md) specifies Walker Walk, seed 0, **1,000,000 action decisions**, action repeat 2 (**2,000,000 DMC calls**). Read the [results and failure analysis](docs/experiments/2026-10-13-walker-seed0.md) for the learning curve, compact numerical evidence, verified observations, unresolved causes, and design limitations. Its custom controller is not an official DreamerV3 baseline, and that run did not test the complete clarified repair architecture.

Suggested reading order: [Clarified dual-chain design and feasible alternatives](docs/DUAL_CHAIN_DESIGN.md) → [First experiment postmortem](docs/experiments/2026-10-13-walker-seed0.md) → [Implemented v0.1 equations](#2-time-indexing-and-the-two-paths) → [Historical reference design](docs/DESIGN.md) → [Validation record](docs/VALIDATION.md).

The sections below retain the historical v0.1 mathematical core and pilot. For the new implementation, use the separate experiment protocol above. Raw observations remain immutable evidence; repaired beliefs and reconstructed images must retain their predictive provenance. A genuine physical failure changes future control and learning, not the recorded past.

## 1. Research objective and minimal core

Real observations form the Real Strand; the model generates a Dream Strand from the same starting point. Two distinct mechanisms connect these paths:

- **Cross correction**: update the world model by supervising multistep open-loop predictions with observation posteriors from real sequences that use the same actions.
- **Reliability calibration**: combine previously observed prediction errors with model disagreement to estimate the risk of candidate imagined trajectories and select consecutive segments for policy learning.

The first version addresses how predictions are generated, corrected, and used for learning. Robotics applications are outside its current scope; context residuals and active acquisition remain optional extensions.

### Initial modeling choice

The original concept allows independent Real/Dream dynamics, two branches with a shared backbone, or fully shared dynamics. The first version adopts a **shared recurrent state-space model (RSSM)**: both paths share transition parameters, Real incorporates each new observation, and Dream evolves without future observations.

This is an initial design choice. It neither rules out independent models nor establishes that the shared architecture is optimal.

## 2. Time indexing and the two paths

After executing action $a_t$, the agent receives observation $o_{t+1}$, reward $r_{t+1}$, and continuation indicator $c_{t+1}$. Let the model state be $b_t=(h_t,z_t)$, where $h$ is deterministic memory and $z$ is a stochastic latent variable.

### Real Strand: incorporating observations after a transition

$$
h^R_{t+1}=f_\theta(h^R_t,z^R_t,a_t),\qquad
q^R_{t+1}=q_\phi(\cdot\mid h^R_{t+1},E_\phi(o_{t+1})).
$$

$$
z^R_{t+1}\sim q^R_{t+1},\qquad
b^R_{t+1}=(h^R_{t+1},z^R_{t+1}).
$$

The Real state is a latent estimate conditioned on observations, not the physical ground truth of the environment.

### Dream Strand: open-loop rollout from the same starting point

$$
\hat b_{t,0}=\operatorname{sg}(b^R_t),\qquad
\hat h_{t,k+1}=f_\theta(\hat h_{t,k},\hat z_{t,k},\tilde a_{t,k}),
$$

$$
p^D_{t,k+1}=p_\theta(\cdot\mid\hat h_{t,k+1}),\qquad
\hat z_{t,k+1}\sim p^D_{t,k+1}.
$$

$\operatorname{sg}$ denotes stop-gradient. Correction uses recorded actions $\tilde a_{t,k}=a_{t+k}$; policy learning uses candidate actions generated by the actor. Supervision labels must match the action sequence that generated the prediction.

Prediction heads output reward $\hat r_{t,k+1}$ and continuation probability $\hat c_{t,k+1}$. Model unreliability does not imply environment termination.

## 3. Model learning and cross correction

Let $p_i^1=p_\theta(\cdot\mid h_i^R)$ be the one-step prior before the current observation is incorporated. The base world model loss includes observation, reward, and continuation prediction, together with prior/posterior KL terms:

$$
\ell_{\mathrm{pred},i}
=-\log p_\beta(o_i\mid b_i^R)
-\log p_\beta(r_i\mid b_i^R)
-\log p_\beta(c_i\mid b_i^R),
$$

$$
\mathcal L_{\mathrm{base}}
=\mathbb E_i\left[
\ell_{\mathrm{pred},i}
+\lambda_d D_{\mathrm{KL}}(\operatorname{sg}[q_i^R]\Vert p_i^1)
+\lambda_r D_{\mathrm{KL}}(q_i^R\Vert\operatorname{sg}[p_i^1])
\right].
$$

Add multistep supervision using the same starting point and recorded actions:

$$
\mathcal L_{\mathrm{cc}}
=
\frac{
\sum_t\sum_{k=2}^{K}m_{t,k}w_k
\mathbb E_{\mathrm{roll}}\left[
D_{\mathrm{KL}}(\operatorname{sg}[q^R_{t+k}]\Vert p^D_{t,k})
\right]}
{\sum_t\sum_{k=2}^{K}m_{t,k}w_k+\epsilon},
$$

$$
\mathcal L_{\mathrm{WM}}
=\mathcal L_{\mathrm{base}}
+\lambda_{\mathrm{cc}}\mathcal L_{\mathrm{cc}}.
$$

- $m$ masks padding, paths crossing episode boundaries or resets, and invalid pairs.
- In the shared model, one-step correction is already covered by the base KL, so the additional term starts at two steps.
- Both the starting state and future posterior teacher targets are detached; correction gradients update the dynamics through the Dream path.
- This objective averages conditional KL over stochastic rollout paths. It is not the exact KL to a multistep marginal distribution.
- Multistep consistency alone does not guarantee more accurate physical predictions or higher environment returns.

## 4. Observed errors, disagreement, and risk

For grouped categorical latents, record the discrepancy between each one-step prediction and the posterior obtained from the subsequent observation:

$$
d_i^z=\frac1J\sum_{j=1}^J\operatorname{JS}(q^R_{i,j},p^{1,\mathrm{saved}}_{i,j}),
\qquad
d_i^r=\frac{|r_i-\hat r_i^{1,\mathrm{saved}}|}{s_r+\epsilon}.
$$

The error history $D_t=(d_{t-L+1},\ldots,d_t)$ may contain only information available by time $t$. Truncate or mask the history at episode starts; it must not cross resets. Save predictions before their actual outcomes arrive and associate them with a model version.

The code parameter `reward_scale` accepts a strictly positive effective denominator. Pass `s_r + epsilon` when using the stabilizing epsilon term above.

Independently initialized predictors trained with bootstrap supervision provide a disagreement proxy:

$$
u_{t,k}
=\frac1{MJ}\sum_{m,j}
D_{\mathrm{KL}}(P^{(m)}_{t,k,j}\Vert\bar P_{t,k,j}),
\qquad
\bar P_{t,k,j}=\frac1M\sum_m P^{(m)}_{t,k,j}.
$$

Disagreement is not an error probability; all members can share the same mistake.

The initial risk target follows the cumulative reward-prediction error event defined in the principles draft:

$$
Y_{t,k}=
\mathbf1\left[
\max_{1\le j\le k}
\frac{|r_{t+j}-\hat r_{t,j}|}{s_r+\epsilon}
>\varepsilon_{\mathrm{task}}
\right].
$$

$$
\hat p_{t,k}
=g_\omega\left(
\operatorname{sg}[b_t^R],
\tilde a_{t,0:k-1},
\operatorname{sg}[u_{t,1:k}],
\operatorname{sg}[D_t],k
\right).
$$

Optimize binary cross-entropy (BCE) on samples with valid future labels. Risk supervision uses open-loop reward predictions for the same actions; future one-step error histories cannot substitute for these multistep labels. Unknown labels and labels crossing resets must be masked, rather than treated as zero-valued supervision.

Future actual observations and rewards serve only as training targets and cannot enter the current risk estimator. BCE training does not guarantee calibration for a finite-capacity model; counterfactual actions and distribution shifts still require evaluation.

## 5. Trusted prefixes and value targets

Select the longest consecutive trusted prefix from the starting point:

$$
H_t=\max\{h\in\{0,\ldots,H_{\max}\}:
\hat p_{t,j}\le\delta,\ \forall\,1\le j\le h\}.
$$

Retain the value bootstrap at each trajectory's truncation boundary, then compute returns backward:

$$
G_{t,H_t}=V_{\bar\nu}(\hat b_{t,H_t}),
$$

$$
G_{t,i}=
\hat r_{t,i+1}
+\gamma\hat c_{t,i+1}
\left[(1-\lambda)V_{\bar\nu}(\hat b_{t,i+1})+\lambda G_{t,i+1}\right],
\quad i<H_t.
$$

Construct actor/critic targets only within the valid prefix. Computing returns from the full trajectory and masking only the final loss would allow rejected suffixes to contaminate prefix targets.

When $H=0$, the starting point produces no imagined policy-update samples. Skip the update if the entire batch has no valid samples.

The initial actor update uses a score-function surrogate on fixed gated samples. Sampled actions, states, gate weights, and advantages are detached along the required gradient paths. This is not claimed to be an unbiased gradient of the full adaptive-stopping objective. The pilot's actor/critic optimizers are implemented in [control.py](helixwm/control.py); their existence and finite gradients did not establish stable learning. See the [failure analysis](docs/experiments/2026-10-13-walker-seed0.md).

## 6. Overall workflow

~~~mermaid
flowchart TD
    A["Real interaction: observations, actions, rewards"] --> B["Real Strand: observation posterior"]
    B --> C["Same start + recorded actions: multistep Dream prediction"]
    B --> D["Detached future posterior targets"]
    C --> E["Cross correction"]
    D --> E
    E --> F["Update the world model"]
    C --> G["Compare predictions with actual outcomes"]
    A --> G
    G --> H["History of observed errors"]
    B --> I["Candidate actor actions: Dream imagination"]
    I --> J["Model disagreement + risk prediction"]
    H --> J
    J --> K["Consecutive trusted prefix"]
    K --> L["Value bootstrap at truncation"]
    L --> M["Actor / critic learning"]
    M --> A
    F --> C
    F --> I
~~~

This diagram describes the learning cycle. Future data required for training labels must already have been collected; decisions during real interaction cannot access outcomes that have not yet occurred.

## 7. Mapping equations to code

The files below implement the mathematical core. The first pilot additionally connected pixel observations, environment interaction, replay, actor/critic optimization, checkpoints and frozen evaluation. These engineering capabilities are separate from the proposed bidirectional segment-repair mechanism.

| Module | Code | Responsibility |
|---|---|---|
| Distribution operations | [distributions.py](helixwm/distributions.py) | Grouped KL/JS and straight-through sampling |
| Two-path model | [rssm.py](helixwm/rssm.py) | observe_step, imagine_step, and shared state representation |
| Model losses | [losses.py](helixwm/losses.py) | Base losses, multistep correction, and gradient isolation |
| Reliability | [reliability.py](helixwm/reliability.py) | Observed error history, bootstrap ensemble, causal risk head, and labels |
| Value targets | [returns.py](helixwm/returns.py) | Consecutive prefixes, lambda returns, and bootstrapping |
| Pixel model and controller | [pixels.py](helixwm/pixels.py), [control.py](helixwm/control.py) | Custom pilot architecture and imagined policy/value updates |
| Experiment pipeline | [train.py](helixwm/train.py), [replay.py](helixwm/replay.py), [checkpoint.py](helixwm/checkpoint.py), [evaluation.py](helixwm/evaluation.py) | Collection, replay, saved training state and frozen evaluation |
| Formula demonstration | [synthetic_trace.py](examples/synthetic_trace.py) | Synthetic inputs, backward-pass checks, and hand-computed return checks; zero optimizer steps |
| Correctness tests | [tests](tests/) | Causal timing, gradients, masks, return boundaries, and half-precision regressions |

The reference implementation uses Gaussian losses with fixed unit variance for observation/reward heads and a Bernoulli loss for continuation. It does not reproduce DreamerV3's distributional heads, free bits, normalization, or other training details. The synthetic examples verify mathematical behavior; the pilot's dimensions and thresholds are explicit, untuned first-run choices recorded in its protocol.

### Running locally

In an isolated Python 3.12 environment, run the following from the repository root:

```bash
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install numpy==2.2.6
python -m pip install -e .
python -m unittest discover -s tests -v
python -m examples.synthetic_trace
```

See the [official PyTorch instructions for previous versions](https://pytorch.org/get-started/previous-versions/) for CPU installation. These local checks cover the mathematical core on CPU. The separate [RTX 5090 environment runbook](docs/ENVIRONMENT_RTX5090.md) records the isolated GPU stack, bounded CUDA/Walker checks, and their scope. The package's declared version range does not mean that every combination has been tested.

The demonstration generates synthetic tensors and checks gradients. It does not connect over SSH, start experiments, or update model parameters. Its losses and risk probabilities come from untrained models and are not performance results.

Main checks:

1. The Dream interface does not read future observations.
2. Correction loss does not backpropagate through the posterior teacher or starting-state inputs, while dynamics receive finite gradients.
3. Invalid pairs crossing resets do not affect the previous episode's loss.
4. Risks [0.1, 0.8, 0.1] with threshold 0.5 yield a trusted length of 1; rejecting every step yields 0.
5. With two rewards of 1, discount 0.99, lambda=1, and endpoint bootstrap value 10, the starting target is 11.791. Incorrectly removing the boundary value would yield 1.99.

## 8. Environment and scope

The first candidate environment is **DeepMind Control Suite / Walker Walk, with pixel observations and continuous actions**.

Previous EAWM, DRAGO, and DrQ-v2 experiments inform environment selection and engineering practice only. Their scores, model weights, and acceptance records are not HelixWM results. Franka is outside the current scope.

The first version establishes the basic model, equations, and correctness tests before experimental design. Context residuals, active information acquisition, and comparisons with independent dual models remain outside the minimal core.

## 9. Current status and requirements for future experiments

| Stage | Status |
|---|---|
| Clarified dual-chain design | Replication, bidirectional local repair, replaceable modules and staged feasibility plan documented; not implemented |
| Reference implementation | v0.1 mathematical core, synthetic demonstration, and tests implemented |
| Historical reference tests | 30 CPU unit tests and the synthetic demonstration passed at the reference-validation stage; see its dated record |
| RTX 5090 environment | Isolated GPU environment and bounded CUDA/Walker smoke checks passed; see the environment runbook |
| CI | GitHub Actions was not configured or run for this version |
| First Walker pilot | Budget complete; final mean 69.38, substantial regression; single-seed artifact acceptance passed, efficacy not established |
| Bidirectional segment-repair experiments | Not started |
| Multiple seeds and publication-level comparisons | Not started |

Future experiments must follow these requirements:

- Record the source revision, dependencies, configuration, hardware, random seeds, and complete launch commands. Distinguish environment frames from control steps and document action repeat.
- Preserve each baseline author's original algorithm, budget, and evaluation protocol. Do not describe HelixWM modifications as an unmodified reproduction of an author's method.
- Start formal training only after environment compatibility, smoke tests, checkpoint reload, and frozen evaluation with zero parameter updates have passed.
- Use tmux for remote runs. Check every six hours as requested, and notify only for meaningful changes, completion, failure, or required user action. The completed pilot's monitoring is paused; this documentation does not start another run.
- Do not interrupt training, alter running source code or budgets, switch seeds, launch duplicate runs, or release instances without authorization.
- Document the model, optimizer, and other states included in each checkpoint. If replay, RNG, or environment state is missing, do not claim exact restoration of the complete training process.
- Archive results on the D drive according to the retained acceptance specification: models, raw data, logs, source/configuration, metadata, learning curves, and SHA256 checksums. Save one video and one learning curve for each actual experiment stage.
- Distinguish smoke checks, completion of a single seed, and publication-level evidence from multiple seeds. Synthetic demonstrations do not replace environment experiments.
- Keep code, documentation, small figures and reviewed numerical evidence on GitHub; store large checkpoints, replay and full logs in the experiment archive. SSH passwords, private keys, private machine paths and other credentials must not enter the repository.

## 10. Sources and research positioning

The research concept originates from the user-provided `HelixWM_Research_Summary_CN.txt`. This README organizes the first version in the requested order: principles, equations and workflows, then experiments.

Foundational references:

- [DreamerV3](https://arxiv.org/abs/2301.04104): world models and policy learning through imagination.
- [PlaNet / Learning Latent Dynamics for Planning from Pixels](https://proceedings.mlr.press/v97/hafner19a.html): RSSM and multistep latent overshooting.
- [Perceiving the Knowledge Boundary](https://ojs.aaai.org/index.php/AAAI/article/view/39576): related work on uncertainty-guided exploration and imagination.
- [dm_control](https://github.com/google-deepmind/dm_control): implementation of the candidate control environment.

Combining and renaming existing components does not establish novelty. The clarified questions are whether replicated, aligned trajectories can support evidence-anchored bidirectional segment repair, whether this improves state estimation and control over ordinary filtering and simpler alternatives, and whether the benefit survives matched data and compute budgets. The earlier error-calibration hypothesis remains a component-level question rather than the whole architecture.
