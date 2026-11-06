# HelixWM Final audit status — 2026-11-06

**Status: implementation and preflight; formal training has not started.**
The first fresh v3 GPU preflight failed before its first bank measurement
window; a corrected fresh attempt is now running in isolation. Historical
checkpoints, data and failed-attempt evidence are preserved.
There is no claim of improved performance, certified safety, or a completed
paper-scale reproduction.

## Implemented comparison

The registered comparison independently trains native DreamerV3 and HelixWM
Final on Walker Walk with the same author `size12m` backbone, environment,
RGB observations, reset seeds, training masks and update schedule. Each arm
has a budget of **1,000,000 actions / 2,000,000 DMC calls**. Different policies
naturally produce different trajectories. The complete-system comparison also
exposes availability to Final's controller; it is not an equal-information,
architecture-only ablation.

The implementation includes persistent Real/Dream hypotheses, constrained
categorical fusion, three auxiliary bootstrap predictors, temporal and
uncertainty gates, multiscale planning, empirical action-support rejection,
separate RNG streams, and immutable trajectory evidence. Unknown visual-fault
detection has a separate interface and is not silently included in this
known-mask experiment.

Both arms share one immutable **2,480-action independent bank**. Bank data do
not update the author learner or auxiliary heads. Per-version calibration
replays its recorded actions offline, with zero additional environment calls.
FIT and heldout evidence remain separate. These gates measure off-policy
empirical behavior, not counterfactual outcomes for newly proposed actions or
independent closed-loop safety. A valid inactive or stale gate disables
fusion/planning and uses an explicit native-prior fallback; corrupt evidence
fails verification.

The v3 checkpoint and archive interfaces bind source, protocol, environment,
bank, current/previous gates, replay dependencies, controller state and RNG.
The final evaluator checks independent D/F lineage, permits only an explicit
output-log-directory relocation during strict reload, preserves primary
failure evidence, and separates diagnostic video frames from policy inputs.
Its registered matrix is **1,280 core cases + 240 gap-100 stress cases**, kept
as separate statistical strata. This is a plan, not completed evaluation.

## Evidence and its limits

| Evidence | Observed result | What it does not establish |
|---|---|---|
| Full CPU regression snapshot after the bank-boundary fix | **420 tests passed**, zero failures, errors or skips; source hashes unchanged during the run. | Actual GPU integration, repair activation, or efficacy. The earlier418-test snapshot preceded the integration failure. |
| Earlier native-startup GPU probe | 5,500 actions, 11,000 DMC calls, 11 episodes, 125 author updates. | A completed v3 FullController episode or either formal training arm. |
| Earlier-format independent recovery probe | Short reference/restored-process comparison passed on the same hardware. | Complete current v3 checkpoint recovery, uninterrupted simulator continuation, or long-horizon/cross-hardware bitwise identity. |
| Actual independent-bank acquisition | 2,480 actions / 4,960 DMC calls; 20 data files; no model creation, learner updates or auxiliary updates; fixed manifest verified. | Current-model threshold validation or production activation. |
| Actual trajectory-cost probe | One learned gap-100 window, 124 actions; retained arrays and trajectory cursor verified. | Representative formal-run storage throughput or a passed capacity gate. |
| Fresh full v3 GPU preflight | **Failed** at 5,500 actions / 11,000 DMC calls / 11 episodes / 125 updates, before the first bank window. | The intended final 500-action integrated episode, active repair/planning, or a full v3 recovery checkpoint. |

The v3 failure was `ValueError: String dictionary keys required`: the immutable
bank uses integer seed keys, while the tree-hash interface required string
dictionary keys. The failure stopped the attempt. **96 archived files were
SHA-256 verified**, and source evidence was retained. A bank-specific canonical
digest now validates the exact integer seed set, sorts the seed records, and
hashes them without weakening the generic tree-hash contract. Added tests use
the actual verified bank to exercise the previous GPU-branch input check and
reject changed pixels or seed-type aliases. The420-test regression passed;
the corrected fresh GPU attempt remains in progress, not a reported pass.

An inactive gate can exercise controller wiring, auxiliary prediction,
trajectory writing and fallback. It cannot demonstrate activated fusion or
planning. The gate measured at action 5,500 controls the next episode; a gate
measured at 6,000 afterward cannot retroactively establish that activation.

## Storage and runtime cautions

- A structural estimate for per-pin bank measurement arrays is
  **168,625,080 raw bytes**. This is a theoretical uncompressed quantity, not
  a measured completed bank replay, archive size, or capacity guarantee.
- The measured trajectory window used approximately **219 KiB per actual
  record** for compressed NPZ arrays plus JSON metadata. It also deliberately
  generated 1,080 prospective diagnostic records for 124 actual records.
  Those extras are not a measured formal-training planning rate; extrapolating
  the whole diagnostic directory as ordinary training cost would be misleading.
- Full trajectory retention, all gate versions, checkpoints/replay, final
  evaluation, ZIP space and repeated prefix verification must be measured
  together. The current checkpoint dependency pass rescans growing history;
  low RAM usage does not imply low cumulative I/O cost. Retention requirements
  have not been reduced to make a capacity estimate pass.

## Remaining launch and acceptance gates

1. Complete the corrected fresh full v3 GPU preflight with honest
   activation/fallback measurements; the bank/hash fix and CPU regression passed.
2. Verify the complete v3 checkpoint in independent fresh processes, including
   the next replay sample/update, all required dependencies and private RNGs.
3. Complete the new archive/ACK transaction probe and measured compute,
   compressed-storage, transfer and final-package capacity checks.
4. Only after the launch gates pass, start the two registered formal arms.
   Then perform frozen evaluation, registered statistics, one prescribed
   video, plots, full integrity verification and package acceptance.

The historical matched-model final reevaluation has not been run as part of
this work. Historical results remain separate from this new architecture.
One training seed per arm and repeated evaluation seeds cannot establish
multi-seed robustness or scientific superiority.

Related design: [implementation and falsifiable claims](architecture.md).
