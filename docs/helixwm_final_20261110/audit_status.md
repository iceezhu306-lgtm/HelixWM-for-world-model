# HelixWM Final engineering validation — 2026-11-10

This is an engineering checkpoint, not a formal D/F performance result or a training launch authorization.

| Evidence | Status | Scope |
|---|---|---|
| CPU regression | PASS | 434 tests; frozen-source correspondence checked |
| Integrated preflight and raw logs | PASS | 6,000 actions, 12,000 DMC calls, 12 episodes, 250 updates |
| Evaluator patch CPU | PASS | 18 tests; external tool, separate provenance |
| Actual archive transaction | PASS | Engineering-only; no formal ACK, no remote deletions |
| Fresh-process recovery | PASS | Same-hardware, bounded next-state validation |
| Independent bank replay | PASS | Full FIT/HELDOUT numerical replay, zero calibration interactions/updates |
| Bounded evaluator audit | PASS | 1,500-action engineering scope; no formal data |
| Later integrated-runtime CPU | PASS | Separate source identity; does not replace original GPU provenance |

During the integrated preflight, the 500 Final-controller actions comprise 459 observed-actor decisions and 41 prior-only fallback decisions. Active-gate actions: 0; accepted fusion: 0; planning attempts/acceptances: 0/0.

Both measured gates are inactive. Structural invariants pass, but all six gap groups must pass the registered empirical coverage criteria. Consequently, this execution supplies no efficacy samples for accepted fusion or planning. Candidate computation and stored trajectories do not establish repair activation.

The independent bank is reused for action-aligned, off-policy calibration. It is not a closed-loop safety certificate for novel action sequences. No held-out threshold tuning is authorized by these observations.

Training runtime SHA256: `2331816df9627b5660f406ee2998338bfed9d4df09347425c9cdf7ed19a7c243`.

External evaluator patch SHA256: `9fb3357143201daf582868a2dc8859525b150f448eb82b1c85edf2f9f9ed3bab`.

The evaluator tool fixes the native action-call interface without changing the frozen training source. Its engineering validation is separate from the model identity.

Later integrated runtime: `984b57f68f2370063699921e2924cc35f54f439a2a20f3040c523cbea58967de`; CPU tests: 437. Historical GPU measurements remain bound to their original runtime.

All three engineering cases use the same F6000 checkpoint, seed 74000 and gap 20, with 500 actions each. They are not an independent D/F training comparison.

| Engineering controller | Episode return |
|---|---|
| black_frame_native | 61.359051465342375 |
| prior_only | 62.36988072904706 |
| helix_final | 62.36988072904706 |

The prior-only and Final action, reward, DMC and availability arrays are byte-identical in this one bounded case. This supports inactive-gate fallback behavior here, not equivalence across all cases.

The bounded Final case recorded 480 observed-actor decisions and 20 prior-only fallbacks; active-gate actions: 0, accepted fusion: 0, planning attempts/acceptances: 0/0.

**Fusion/planning: NOT_ACTIVATED. Module efficacy: NOT_TESTED.** These engineering returns do not establish a dual-chain improvement or D/F superiority.

The engineering archive contains 1,637 files / 599,012,549 bytes, with an independent full rehash and exact member-set check. Observed end-to-end time was 133.297 seconds (4.494 MB/s including verification and protocol overhead, not raw network bandwidth).

The initial independent-bank backup safely rejected 242 TAR hardlink entries. A separate ordinary-file export passed the unchanged integrity checks; the failed archive and partial extraction were preserved.

Two new one-million-action formal runs have not started. The registered 1,520-case final evaluation, designated video, statistical report and full archive acceptance have not run. Full-budget remote retention/storage capacity remains unresolved and blocks formal launch. Existing models and historical evidence remain preserved. No paid storage expansion or formal launch is authorized by this report.

These engineering cases use the registered known dropout masks. Unknown, unlabeled real observation faults are outside this evidence scope.

Evidence index: [validation evidence index](evidence/validation_evidence_index.json). Local machine paths, host addresses and credentials are intentionally omitted from this portable report.

## Inspectable evidence

- [cpu_training_runtime_434.json](evidence/cpu_training_runtime_434.json)
- [cpu_integrated_runtime_437.json](evidence/cpu_integrated_runtime_437.json)
- [evaluator_patch_cpu_18.json](evidence/evaluator_patch_cpu_18.json)
- [action_evidence_audit.json](evidence/action_evidence_audit.json)
- [archive_transaction.json](evidence/archive_transaction.json)
- [recovery_gate.json](evidence/recovery_gate.json)
- [independent_bank_replay.json](evidence/independent_bank_replay.json)
- [bounded_evaluator_completion.json](evidence/bounded_evaluator_completion.json)
- [validation_evidence_index.json](evidence/validation_evidence_index.json)

This folder publishes small, byte-preserved engineering evidence. Full checkpoints, replay, trajectories, logs, failed exports and stable backup manifests remain in the local E archive. Evidence paths inside the JSON index identify that local archive; they are not links to unpublished GitHub files.
