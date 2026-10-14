# Replicated dual-chain Walker experiment

Status: implementation and preflight preparation. A formal run is not launched until the recorded gates pass.

## Scope and units

One new HelixWM training seed, Walker Walk with 64 x 64 RGB, seed 0. The latest user instruction supersedes the earlier two-arm plan: do not rerun the existing DreamerV3 baseline. The exact budget is **1,000,000 executed action decisions**, repeat 2, **2,000,000 DMC calls**. Resets and evaluation calls are excluded. These are not MuJoCo integration substeps.

The historical local DreamerV3 experiments used proprioceptive input and repeat 1. They supply a tested source version and compatibility history, not matched numerical controls or pretrained pixel weights. This run starts from scratch and is not a paper-level reproduction or proof of improvement.

## Backbone and explicit adaptations

Use the unchanged official DreamerV3 commit `e3f02248693a79dc8b0ebd62c93683888ddaccfe`, `defaults + dmc_vision + size12m`, with its original model losses, bounded-normal actor, critic, return normalization, optimizer, and imagination objectives. The source is an external pinned checkout whose tracked files must be clean. The wrapper never invokes the first pilot's custom actor or learner.

The experiment configuration is [dual_chain_walker_seed0_1m.json](../configs/dual_chain_walker_seed0_1m.json). Size12m is an author-provided engineering scale choice, not the paper's default 200M setting. Actual GPU and checkpoint sizes must be measured before launch.

The collection and storage infrastructure differs explicitly from the author runner:

- One synchronous environment, direct counters of actual executed actions.
- An immutable disk episode archive and uniform sampling over eligible contiguous starts; capacity is the full one-million-action run. Thirty-two episodes may be cached in RAM.
- Each sampled crop has 32 observations of detached recurrent burn-in and 64 positions for the unchanged author loss, batch size 16. Replay context is disabled; independent crops do not carry hidden state from unrelated crops. The incoming action before a crop is unknown and starts at zero, so burn-in is a finite-context approximation.
- The first 5,000 actions collect data using the initialized native policy. Thereafter the scheduled training ratio is 256 loss positions per action, or one batch update per four actions. Burn-in inference adds compute and is reported separately.
- Inference parameters are pinned for each 500-action episode while optimizer updates continue on schedule. The next episode receives the current parameters. This explicit synchronization lag prevents a pending repair from combining different model versions; it is not claimed equivalent to the author's asynchronous scheduling.

## Implemented mechanism to validate

The Real chain and candidate Dream chains copy a common observation-qualified anchor, retain separate recurrent/stochastic states, and advance using the exact executed actions. Explicit sensor-validity metadata determines missingness; prediction disagreement alone never declares a physical fall to be sensor corruption.

For up to five missing observations, eight prior-proposed trajectories retain their complete state ancestry. Missing observations are not encoded as if black pixels were real. Dream supplies the current inferred belief. A returning valid observation reweights entire candidate histories by the fixed backbone decoder energy; the chosen complete state path reconnects the Real chain and anchors new Dream copies. Inferred past states carry the later evidence cutoff. Past actions, rewards and raw camera frames remain unchanged.

The image head provides negative squared-error energy, not calibrated observation likelihood. Scores are normalized per pixel and tempered by the predeclared value 0.01. Since proposals already follow the transition prior, weights add observation evidence only. Record candidate weights, effective sample size, selected ancestry, missingness, repair length, model version, abstention, and evidence time. Long or unsupported gaps trigger the documented bounded fallback rather than confident invented evidence.

The new mechanism is this inference/repair structure. The native model loss continues learning from actual recorded frames. No generated repair is silently promoted to a real training label, and no extra reciprocal distillation loss is claimed.

## Observation perturbation and evidence access

To exercise repair during this single experiment, the collector receives known synthetic missing-frame bursts: on eligible observations, burst-start probability 0.02 and a uniformly sampled length from 1 to 5. Initial and final episode observations are always valid. A dedicated seeded generator determines the schedule independently of policy and replay RNG.

The simulator's raw frames remain in the archive and may supervise the world model. During a masked collector step, the policy and repair engine receive no clean image. This is a declared synthetic sensor-denoising protocol with clean training targets; it is not an experiment where unobserved real-world frames are magically available for learning. Actual dropout fraction is logged. Native physical dynamics, reward, and task remain unchanged.

## Evaluation and preservation

Every 10,000 actions save a new immutable checkpoint and SHA256 sidecar. Every 20,000 actions evaluate five fixed environment seeds separately under clean and fixed seeded burst-dropout observation conditions. Evaluate the final strictly reloaded checkpoint on 20 fixed seeds, 10000 through 10019, for each condition, with model and optimizer hashes unchanged. Independent evaluation policy/dropout RNGs must not perturb the next training update or action.

Keep clean and missing-observation returns separate. The final frozen audit evaluates four conditions on the same trained checkpoint, each on the same 20 environment seeds (80 evaluation episodes total):

| Condition | Handling of a declared missing frame | Returning valid evidence |
|---|---|---|
| `clean` | No synthetic missingness | Ordinary backbone observation update |
| `dropout` | Bounded prior-proposed candidate trajectories supply the inferred belief | Evidence reweights complete candidate ancestry and reconnects the current belief |
| `dropout_disabled` | Encode an explicit black image with repair disabled | Ordinary backbone observation update |
| `dropout_prior_only` | Skip the observation encoder and propagate one ordinary prior state; no candidate selection | Ordinary backbone observation update; no ancestral segment repair |

The three dropout conditions use identical predeclared missingness schedules, environment seeds, and private policy seeds. Comparing repair with prior-only separates the proposed candidate/segment mechanism from the simpler benefit of avoiding an invalid image encoder input. These are frozen inference interventions on one trained model, not new training runs or matched independently trained baselines. A difference in their returns alone does not establish that retraining with each condition would produce the same ranking. Mechanism tests include whole-history repair, reset isolation, model-version alignment, causal information access, raw-data immutability and restore-next-action/update parity.

Save one formal MP4 using the first predeclared final clean evaluation seed; do not select the best video by reward. The two additional audit controls do not create additional videos. The learning curve includes labeled clean and dropout evaluation series and separately labeled final black-frame/prior-only audit points, with PNG and PDF copies. Save original per-episode CSV/JSONL and raw diagnostics, including observation-availability hashes to verify paired missingness. Episode SD from one trained policy is not multi-training-seed uncertainty.

Checkpoint payloads include official parameters and optimizer/normalizer/target state, explicit counters and RNGs, current pinned inference version, dual-chain state and pending ancestry, collector episode/environment integration state, replay index/RNG and required immutable files. Validate recovery on the same dependency build; do not promise cross-hardware bitwise recovery.

Formal checkpoints occur after completed episodes: the backbone is synchronized to the newest parameters, while the finished chain is retained as audit history. Recovery restores its independent RNG streams, then resets the environment and re-anchors under the saved current model. It never treats old-version latent state as compatible with new parameters. The explicit `--resume CHECKPOINT` option requires a new output directory, the identical source/configuration, and all SHA256-matching replay files. It is never invoked automatically. Earlier logs and checkpoints remain in the parent run and must be consolidated as verified lineage before accepting a recovered branch.

The actual-size preflight and completed 1,000-action smoke are bound to the exact source commit, upstream commit and full formal configuration hash. Smoke data is separate and does not count toward the formal budget. Launch requires both gates, including normal smoke exit, strict checkpoint reload and frozen evaluation; a gate from an older source/configuration is rejected.

Use tmux, a separate frozen source release and run directory, and one bounded inspection every six hours. Preserve the first pilot unchanged. Final acceptance requires complete budget, normal exit, strict reload, zero-update evidence, full MP4 decode, raw data/curves/config/source/dependency metadata, and verified D-drive archive/file SHA256. No automatic additional seeds, shutdown, release, or unrelated-server operations.
