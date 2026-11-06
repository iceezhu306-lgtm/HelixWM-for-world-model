# HelixWM Final audit status — 2026-11-06

**Status: implementation and preflight; formal training has not started.**
The latest CPU regression passed **426/426 tests**. The corrected GPU probe
using the earlier runtime reached **6,000 actions / 250 author updates** and
was measuring its second bank window at **2026-11-06 11:29 UTC**. Its first
gate was inactive, so the intervening 500 controller actions used fallback.
The newly optimized runtime has not yet completed GPU verification. Historical
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
| Full CPU regression, `full_cpu_regression_20261106T112940Z` | **426/426 tests passed**, zero failures, errors or skips; tested source hashes unchanged during the run. | Actual GPU integration, repair activation, or efficacy. |
| CPU cache-optimization equivalence | **14 cases passed**: raw numerical arrays/rows, complete public snapshots, planning metadata and all private RNG states matched. | Production provenance, GPU equivalence or a GPU speed guarantee; both workers used a standardized synthetic context. |
| Earlier native-startup GPU probe | 5,500 actions, 11,000 DMC calls, 11 episodes, 125 author updates. | A completed v3 FullController episode or either formal training arm. |
| Earlier-format independent recovery probe | Short reference/restored-process comparison passed on the same hardware. | Complete current v3 checkpoint recovery, uninterrupted simulator continuation, or long-horizon/cross-hardware bitwise identity. |
| Actual independent-bank acquisition | 2,480 actions / 4,960 DMC calls; 20 data files; no model creation, learner updates or auxiliary updates; fixed manifest verified. | Current-model threshold validation or production activation. |
| Actual trajectory-cost probe | One learned gap-100 window, 124 actions; retained arrays and trajectory cursor verified. | Representative formal-run storage throughput or a passed capacity gate. |
| Fresh full v3 GPU preflight | **Failed** at 5,500 actions / 11,000 DMC calls / 11 episodes / 125 updates, before the first bank window. | The intended final 500-action integrated episode, active repair/planning, or a full v3 recovery checkpoint. |
| Corrected v3 GPU probe, earlier runtime | At 11:29 UTC: **6,000 actions / 250 author updates**, second bank measurement in progress. The gate at 5,500 was `active=false`; the next **500 FullController actions used fallback**. | Completed preflight acceptance, activated repair/planning, optimized-runtime GPU validation, or either formal training arm. |

The v3 failure was `ValueError: String dictionary keys required`: the immutable
bank uses integer seed keys, while the tree-hash interface required string
dictionary keys. The failure stopped the attempt. **96 archived files were
SHA-256 verified**, and source evidence was retained. A bank-specific canonical
digest now validates the exact integer seed set, sorts the seed records, and
hashes them without weakening the generic tree-hash contract. Added tests use
the actual verified bank to exercise the previous GPU-branch input check and
reject changed pixels or seed-type aliases. The subsequent full CPU snapshot
passed 426 tests; the corrected fresh GPU attempt remains in progress, not a
reported pass. The earlier failed attempt and its archived evidence remain
separate and are not overwritten by the fix.

An inactive gate can exercise controller wiring, auxiliary prediction,
trajectory writing and fallback. It cannot demonstrate activated fusion or
planning. The gate measured at action 5,500 controls the next episode; a gate
measured at 6,000 afterward cannot retroactively establish that activation.

The current v3 protocol permits inactive-gate fallback throughout the entire
1M-action budget; it has no maximum consecutive inactive-window count. Every
500-action pin requires a new measurement, and the strict final checkpoint
loader/evaluator also accepts a valid inactive gate. Consequently, finishing
the budget does **not** guarantee that fusion or planning was ever enabled.
Reporting must distinguish gate activation from accepted fusion/planning and
native fallback rather than describe budget completion as demonstrated dual-chain
repair. No activation threshold has been relaxed to obtain a pass.

**Training instrumentation remains incomplete:** `interactions.jsonl` currently
does not record `decision`, `fusion_lambda` or `gate_active` for each action.
Only a bounded controller history is retained in checkpoints every 10,000 actions;
those snapshots cannot reconstruct full-run action-level module rates. Retained
bank evidence supports complete active/inactive pin counts, and the final
evaluator can report its own module rates when it is run. Neither is a substitute
for complete training-time fusion/planning/fallback rates. Durable per-action
logging and coverage validation remain required work; this documentation update
does not modify the runtime or claim that these missing measurements exist.

### Optimization evidence is CPU-only

The frozen optimized runtime has SHA-256
`5430cccc774ae578c8569828bf37dd1ce9e1418d3a9e2e69b51076a986e2a57f`.
The running corrected probe uses the earlier runtime
`6fb927a9b5da8d3c1ad12c7d442d1f37c7e799baa71f4fd566563cbee5ef0a3a`;
its observations cannot be attributed to the optimized version.

The retained `optimized_equivalence_rng_20261106/equivalence.json` benchmark
compares 14 cases without environment interaction. In its gap-20 case using
actual FIT content under a **synthetic backend and standardized synthetic
provenance**, elapsed CPU time was approximately **8.101 seconds before /
0.381 seconds after** caching. Equality includes all recorded numerical
outputs, the complete public controller snapshot, planning metadata and every
private RNG stream. This is a mechanism-equivalence check, not authentication
of a production gate, an actual-model GPU test, or a guaranteed training speedup.

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

The retention audit `Final_remote_retention_capacity_audit_20261106T113334Z.txt`
marks capacity **BLOCKED**. Expanding the local archive drive alone does not
solve remote retention: current checkpoint dependencies include the complete
historical trajectory prefix, whose raw files are required for strict restore.
Deleting those files after a local copy would violate the current format.

The gate dependency is narrower: the latest checkpoint references its current
and just-completed previous gate, not every old gate. Nevertheless, the current
archive bridge recursively archives retained gates and implements **no pruning
transaction**. Old checkpoints still need their own gate dependencies. A
verified local copy therefore does not itself authorize remote deletion.

For scale only, one completed 5,500-action gate occupied about **142.389 MB**
across 244 files. Multiplying that single observation by the planned 1,991 gate
measurements gives approximately **264.027 GiB**, before trajectory, checkpoint,
replay and evaluation data. This is a linear projection, not a capacity
guarantee. Lossless hot/cold dependency handling, verified archive receipts,
explicit cleanup authorization and fresh recovery tests are required before
any reduced remote-retention design can pass. Existing evidence remains intact.

## Remaining launch and acceptance gates

1. Complete the earlier-runtime probe and a fresh optimized-runtime full v3 GPU
   preflight, with honest activation/fallback measurements. CPU regression and
   synthetic equivalence do not replace these gates.
2. Verify the complete v3 checkpoint in independent fresh processes, including
   the next replay sample/update, all required dependencies and private RNGs.
3. Resolve the remote historical-dependency retention blocker; complete the new
   archive/ACK transaction probe and measured compute, compressed-storage,
   transfer and final-package capacity checks for both remote and local storage.
4. Only after the launch gates pass, start the two registered formal arms.
   Then perform frozen evaluation, registered statistics, one prescribed
   video, plots, full integrity verification and package acceptance.

The historical matched-model final reevaluation has not been run as part of
this work. Historical results remain separate from this new architecture.
One training seed per arm and repeated evaluation seeds cannot establish
multi-seed robustness or scientific superiority.

The earlier-runtime probe remains in its existing tmux session. The requested
inspection cadence is six hours; no new scheduler has been created. This
documentation update does not continuously poll it or launch another probe.
Optimized GPU, recovery and archive work awaits a resolved capacity path.

Related design: [implementation and falsifiable claims](architecture.md).
