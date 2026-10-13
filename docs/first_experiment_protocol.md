# First HelixWM experiment: frozen protocol

This is a new single-seed development experiment of the first HelixWM design.
It is not a reproduction of DreamerV3, a comparison with a trained baseline,
or evidence of a performance improvement. The mathematical reference has now
been extended with a pixel encoder/decoder, replay, a controller and an explicit
training loop. Previously collected baseline results only informed environment
selection. No Franka, residual dynamics, distribution shift, adaptation stage,
or additional training seed is included in this run.

## Budget and environment

The user selected **1,000,000 action decisions**, not one million frames.
DeepMind Control Walker Walk uses RGB 64 x 64, no state-vector policy input,
training seed 0 and action repeat 2. `env_steps` counts actual DMC `step` calls,
so a complete run has 2,000,000 such calls. These are not MuJoCo integration
substeps. Each 1,000-call time-limit episode contains 500 action decisions.
Repeated rewards are summed; the episode return remains on the usual
approximately 0-1,000 Walker scale. Native time-limit continuation discounts
are preserved. Evaluation interactions do not count toward training budget.

The executable source of all settings is `configs/walker_seed0_1m_actions.json`.
The smoke configuration is a separate, labeled 1,000-decision run with a
500-decision random prefill, 500-decision checkpoints/evaluations, and two
evaluation seeds. It uses the same model and batch sizes. Formal training
starts from scratch after the smoke, never from its checkpoint or replay.

## First-version algorithm and engineering choices

- Shared categorical RSSM: 256 deterministic units, 16 groups x 16 categories,
  256-dimensional image embedding, convolution depth 32. Images are normalized
  to [-0.5, 0.5]. Reconstruction is a fixed-variance Gaussian pixel sum.
- Base world-model loss retains dynamics KL weight 1 and representation KL
  weight 0.1. No free bits, symlog or undocumented Dreamer recipe is added.
- Multi-step correction uses horizons 2-5, weight 0.1, detached posterior
  teachers and detached starting beliefs, with episode/padding masks.
- Three independent bootstrapped prediction heads estimate disagreement.
  Reliability receives only the current belief, already-observed errors,
  candidate action prefix and disagreement. Outcome targets are separate.
  Reward error is normalized by 2 (the maximum repeat-summed reward). The
  risk event is any prefix error exceeding 0.25, not a sum of future errors.
- The risk head is learned jointly. A fixed threshold 0.5 selects the longest
  contiguous trusted prefix of at most 15 imagined decisions. A zero prefix
  skips both policy and value updates. There is no hidden fallback that bypasses
  the gate. No held-out calibration guarantee is assumed for this threshold.
- Policy optimization is a detached score-function objective with a tanh
  Gaussian actor. Original pre-tanh samples are retained so rounded saturated
  actions do not bias the score. World model and reliability parameters receive
  no policy gradients. Returns use gamma 0.99, lambda 0.95 and boundary value
  bootstrapping. The slow value target uses EMA 0.01.
- Random prefill: 5,000 action decisions. Thereafter one joint update every
  five decisions, batch 16, sequence 32 with eight observations of burn-in.
  All loss/gradient values must remain finite; gradient norms are recorded.
- The replay sampling set holds at most 200,000 decisions in complete episodes.
  Compressed episode files are append-only, including older inactive episodes.
  Up to 400 decompressed episodes are cached (about 2.5 GB for image arrays),
  bounded independently of the million-decision archive. This prevents repeated
  full archive loading and preserves older checkpoint references.

These are transparent first-run implementation choices, not tuned claims of
optimal hyperparameters. A later efficacy study needs an appropriate matched
baseline, ablations and at least three training seeds. In particular, extending
an official DreamerV3 backbone and the proposal's frozen calibration experiments
remain future work; this smaller implementation does not replace them.

## Checkpoints, evaluation and acceptance

Checkpoints are atomic and immutable every 10,000 decisions (20,000 DMC calls).
Each includes all model and target networks, all five optimizers, framework and
collector RNG states, replay index/RNG, counters, partial collector data and
Walker integration/task state. SHA256 sidecars and `latest.json` point to the
saved file. Replay files remain external and must accompany recovery backups.
`best.json` is a pointer selected using only the five fixed periodic evaluation
episodes; it is not a replacement for the final-budget checkpoint.

Periodic evaluation occurs every 20,000 decisions on fixed environment seeds
10000-10004. Final evaluation reloads the final checkpoint and uses seeds
10000-10019. Actor means and deterministic posterior categories are used.
Full learner and optimizer hashes must be unchanged before/after evaluation;
training RNG and module modes are restored. Raw episode results, population
standard deviation and one-step held-out reward RMSE are retained. Training
Brier scores do not demonstrate held-out calibration.

This run has **one actual stage, clean training**, so its final archive has one
stage video and one learning curve (PNG plus PDF), together with an additional
diagnostic plot. The video uses the first predeclared final evaluation seed;
there is no reward-based video selection. A single seed's episode variability
is never labeled uncertainty across independent training seeds.

Before formal launch: pass unit tests, actual GPU learner updates, Walker
snapshot-next-step equivalence, smoke budget, checkpoint reload, zero-update
frozen evaluation, decoded MP4 and rendered curves. Recovery probes perform
isolated identical next updates/collector steps from the same checkpoint;
they do not establish cross-hardware bitwise recovery or uninterrupted long-run
equivalence. Resume is explicit only, never triggered blindly by the monitor.

The six-hour monitor checks tmux/PID/log progress, units, loss/gradient finiteness,
memory/OOM/disk/GPU, new checkpoints and local backup hashes, then ends.
On completion it verifies the complete budget, evaluation/video/curve evidence,
preserves source/config/commands/dependencies/raw data/replay/checkpoints on D,
creates an integrity-checked final archive and pauses itself. It must not start
other seeds, change a running protocol, or shut down/release the instance.
Local scheduled checks require the user's computer and Codex app to be running.
