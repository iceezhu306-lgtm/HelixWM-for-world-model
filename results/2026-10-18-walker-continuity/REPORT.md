# Third HelixWM experiment: continuity under explicitly missing Walker observations

## Conclusion

The completed frozen-checkpoint study does **not establish superiority** for either proposed inference change. The predeclared held-out decision is `NOT_ESTABLISHED` for `locked_ancestor` and `posterior_return_once`. This result is compatible with small effects, no effect, or effects that vary by gap length; it is not an equivalence proof.

All six arms use the same learned weights from one training seed. There is no independently trained vanilla DreamerV3 arm in this study. In particular, `prior_only` and `black_frame` are inference controls applied to the HelixWM input model; renaming them DreamerV3 would change the meaning of the experiment.

## Question and intervention

The question is whether small changes to latent-state handling during a camera interruption improve control and recovery, while holding the learned world model and actor fixed.

| Arm | Input and inference intervention |
| --- | --- |
| `clean` | All real images remain available. |
| `black_frame` | A missing image is replaced with zeros and passed through the ordinary observation update as valid input. |
| `prior_only` | A missing image is not encoded; one latent prior is propagated using the actual executed action. |
| `v2_exact` | Preserved V2 bounded candidate repair: up to eight paths and five missing observations, with the original return-evidence reconciliation. Longer gaps trigger abstention and single-prior continuation. |
| `locked_ancestor` | During the bounded gap, control retains the first selected candidate lineage. The original selection draws are still consumed/discarded, and the original return reconciliation is retained. This can change actions and therefore later trajectories. |
| `posterior_return_once` | At the first valid return within the bounded gap, use one native posterior update from the pre-return control state instead of return-image energy resampling. This does not claim retrospective repair of candidate history. The original long-gap fallback is retained. |

The availability mask is explicitly supplied. No learned detector of unknown image corruption is evaluated. Candidate decoder energy is an uncalibrated score, not a probability of model correctness. Current simulator state and masked raw images are evaluator-only evidence, unavailable to the policy and repair branches.

## Fixed protocol and execution evidence

| Item | Completed study |
| --- | --- |
| Task | DeepMind Control `walker/walk`, RGB observations, 64 by 64 pixels |
| Input model | Training seed 0; earlier budget 1,000,000 actions / 2,000,000 DMC calls |
| New training in this study | 0 updates |
| Episode | 500 action decisions, action repeat 2, 1,000 DMC calls |
| Missing interval | One interval beginning at observation/action index 150; lengths 1, 5, 10, 20, 50 actions |
| Design split | Seeds 20000-20019: 20 clean + 5 arms x 5 lengths x 20 seeds = 520 episodes |
| Held-out split | Seeds 30000-30039: 40 clean + 5 arms x 5 lengths x 40 seeds = 1,040 episodes |
| Total evaluation | 1,560 episodes; 780,000 actions; 1,560,000 DMC calls |
| Technical failures | 0 in both archived analysis splits |
| Frozen/reload checks | Accepted source-bound strict reload, reference episode parity, model/optimizer/counter/host-RNG preservation |

`held_out` is the registered environment-seed split for this study, not evidence of generalization across independently trained agents. The input model had already been trained in the earlier HelixWM dual-chain experiment. All arms are paired by environment seed, fixed mask, and specified random-stream derivation; different actions can still produce different future trajectories.

Execution source commit: `22b93cdb2b5a8643e8e14931dc2ffc5eae32a361`.

Official Dreamer upstream commit: `e3f02248693a79dc8b0ebd62c93683888ddaccfe`.

Input checkpoint SHA-256: `8aba0d3c6a1e9116850f34d24caf2fcdc0c68232d849d5bebbb9ec6239c8816e`.

The public artifact is a subset of the accepted archive, not a new acceptance run. Every copied artifact was rehashed against that archive's `MANIFEST_SHA256.json` during preparation. The original archive manifest hash and the selected file bindings are retained in the publication provenance.

## Metric definitions

Let $r^{a,s}_t$ be the reward summed during action $t$, and $n^{a,s}_t$ its actual DMC call count. The normalized per-action reward is

$$
u^{a,s}_t = r^{a,s}_t / n^{a,s}_t.
$$

With missing-observation start $g=150$, gap length $L$, and recovery window $H=100$, the paired deficit for arm $a$ is

$$
d_{s,L,a}=\frac{1}{L+H}\sum_{t=g}^{g+L+H-1}
\left(u^{\mathrm{clean},s}_t-u^{a,s}_t\right).
$$

The primary seed-level outcome weights all five lengths equally:

$$
D_{s,a}=\frac{1}{5}\sum_{L\in\{1,5,10,20,50\}}d_{s,L,a},
\qquad
\Delta_{p,c}=\frac{1}{S}\sum_s(D_{s,p}-D_{s,c}).
$$

Negative $\Delta$ favors the proposal. This is a mean normalized reward deficit over the fault-plus-recovery window, not the raw full-episode return.

Functional failure is 20 consecutive normalized rewards below 0.2 inside that window. It does not establish a physical fall or a safety failure. The descriptive guard requires no increase in the observed aggregate failure count relative to either comparator; it is not a statistical noninferiority guarantee.

Recovery timing is eligible when mean reward in the 50 pre-fault actions is at least 0.75. After the image returns, a trailing 10-action mean must reach at least 90% of that episode's pre-fault mean for 10 consecutive checks. Recovery is censored at 100 post-return actions. The earliest qualifying elapsed time is therefore 19 actions. A recorded recovery does not erase an earlier functional failure or accumulated reward deficit.

The archived protocol zero-fills missing reward tails only for the primary technical-failure analysis and separately reports complete-case sensitivity; unknown failure flags are never imputed as false. No such zero-filling was required here because all recorded cases completed.

## Held-out results

The original archived panel is [PNG](plots/held_out_robustness_recovery.png) / [PDF](plots/held_out_robustness_recovery.pdf). The design split is [PNG](plots/design_robustness_recovery.png) / [PDF](plots/design_robustness_recovery.pdf) and remains descriptive.

Mean full-episode return by missing length (40 environment seeds per cell):

| Arm | 1 action | 5 actions | 10 actions | 20 actions | 50 actions |
| --- | ---: | ---: | ---: | ---: | ---: |
| Black frame | 948.6205 | 947.3529 | 933.8324 | 904.5893 | 848.4747 |
| Single prior | 948.6353 | 948.6964 | 945.9199 | 903.9993 | 842.8572 |
| Original V2 | 948.6084 | 948.4688 | 944.2371 | 906.3259 | 845.2080 |
| Locked ancestor | 948.6084 | 948.4864 | 945.1233 | 900.3988 | 844.1041 |
| Posterior return once | 948.6089 | 948.4103 | 944.2371 | 906.3259 | 845.2080 |

Clean return is 948.6745 on its 40 shared reference episodes. Larger interruptions reduce mean return in every faulted arm. Short-gap mean differences are small. The posterior-return modification only changes bounded return handling; its matching long-gap results should not be interpreted as successful repair of 10-50 missing frames.

The primary confirmatory contrasts use 40 whole environment-seed clusters, preserving all lengths and arms. The archived analysis uses 10,000 bootstrap resamples, 100,000 paired sign-randomization draws, and Holm adjustment over four predeclared one-sided tests. Sign randomization relies on within-pair exchangeability under its null.

| Proposal minus comparator | Mean deficit difference | Bootstrap 95% interval | Holm-adjusted p | Failure-count guard |
| --- | ---: | --- | ---: | --- |
| Locked ancestor - V2 | 0.004893 | [-0.010800, 0.020069] | 1.000000 | 59 vs 59: descriptive pass |
| Locked ancestor - single prior | 0.003178 | [-0.013040, 0.019094] | 1.000000 | 59 vs 54: observed increase |
| Posterior return once - V2 | -0.000111 | [-0.000314, 0.000084] | 0.583034 | 59 vs 59: descriptive pass |
| Posterior return once - single prior | -0.001826 | [-0.014222, 0.010688] | 1.000000 | 59 vs 54: observed increase |

Neither proposal improves over both comparators under the registered rule. The correct decision remains **`NOT_ESTABLISHED`**. The visible return ranking alone would be insufficient to change that decision. The cluster-level failure upper intervals in the JSON are not paired risk-difference bounds and do not certify safety.

## Video and historical learning curve

The only formal video is [held-out seed 30000, V2, gap 20](videos/held_out_seed30000_v2_exact_gap20.mp4). The preselected episode has 500 actions, 1,000 DMC calls, raw return 859.9601, and 501 frames at 20 fps. The original episode summary supplies the frame, mask, frozen-state and video-hash evidence. Raw rendered images are not proof of policy visibility: the availability mask controls what the policy actually receives.

The [historical learning curve](reference_training/historical_learning_curve.png) belongs to the earlier input-model training run and is included only for context. It must not be relabeled as training performed in this third experiment.

## Reproducibility and limitations

Run `python recompute.py` to verify publication hashes and independently recalculate the published group means, failure counts, per-seed means, and paired contrast means from the unchanged CSVs. The archived uncertainty estimates and recommendations are preserved, but this compact script does not regenerate the hypothesis tests or extract metrics from the omitted per-step traces. The public CSVs are analysis outputs, not the full raw simulator dataset.

The public summaries are visibly labeled derivatives because their two private absolute figure paths were replaced with relative public links. All other values are unchanged. Original CSVs, figures, the acceptance JSON, and the selected video remain byte-identical to their archive sources.

This is one trained policy on one locomotion task with explicit synthetic observation gaps. It does not establish retraining reliability, unknown-fault detection, calibration of candidate energy, real-world safety, or superiority to independently trained DreamerV3. The useful next question is narrower: whether a revised mechanism yields an effect that persists across separately trained seeds and a properly matched independently trained baseline. That would require a new registered experiment; none was performed for this publication.
