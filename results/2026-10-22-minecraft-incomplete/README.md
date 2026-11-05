# Minecraft navigation run stopped before completion

This compact historical evidence package describes the **failed formal run** `minecraft_visible_navigation_seed0_20261020`. The archived observation cutoff is **2026-10-22 20:18:04 UTC**. Publishing it on a later date is not a new live server inspection.

## What was actually completed

- A custom real Minecraft visible-target navigation task, not an equivalent reproduction of the official Navigate benchmark.
- Formal startup occurred on 2026-10-21. The prepared run name retains the earlier date.
- **194,030 of 1,000,000 action decisions**; **387,538 acknowledged native synchronous calls**; **47,257 learner updates**. Native calls are not all Java world ticks.
- The failure record reports an `episodes` field of 1022, while the stale running status reports 1021. Both original fields are disclosed in `status_summary.json`; no reconciliation is invented.
- **19 stable checkpoints** and **932 checkpoint-referenced replay files** were preserved in the full local archive. The latest checkpoint is at **190,037 actions**; complete recovery of the following **3,993 actions** is not established.
- **90 completed periodic evaluation cases**: nine progress rounds, five seeds for each of `clean` and `dropout`. The original CSV and JSONL are included without numeric changes.

## Failure and resource evidence

The run exited with code **1** after `TimeoutError('Minecraft worker timed out during reset')` at **2026-10-22 06:12:41.548986 UTC**. The configured worker timeout was 180 seconds. A stale `state: running` record does not override the exit, failure marker and absent process evidence.

Memory pressure/throttling was recorded before the timeout: the post-failure `memory.high` event count was **1,268,879**, with an 88 GiB high limit and a 90 GiB maximum. The last logged training memory sample was **94,488,363,008 bytes**. **OOM and OOM-kill counters were zero.** These observations do not establish which process caused the pressure or whether it caused the reset timeout. Do not label this an OOM kill, and do not rule out memory pressure merely because OOM counters were zero.

The reset path includes old-backend close, new-backend construction and engine reset. Existing evidence does not localize the blocked sub-operation. No automatic retry or recovery was performed. Any future recovery requires an explicitly authorized new run with checkpoint lineage; this publication starts no experiment.

## Scientific limitations

The 1M-action budget was not completed. The final 520-case evaluation, final-model acceptance and complete final package were **not completed**. Periodic scores are partial development data, not a completed six-arm comparison or proof that HelixWM improves Minecraft performance. Goal success must follow the native target-touch evidence, not high reward alone.

This historical Minecraft setup used unchanged real RGB learner targets, including frames hidden from the collector. It is not the later matched Walker protocol in which both learners use only visibly corrupted RGB. Its scores must not be presented as a matched missing-data training comparison.

Checkpoint support is software state at episode boundaries, with a fresh environment reset. Java engine state in the middle of an episode is not serialized or proven recoverable. The separate 2026-10-20 smoke package is an earlier validation artifact; its contemporaneous claim that formal training had not started is not the status of this later failed run.

## Package contents and verification

- `status_summary.json`: archived failure state and explicit completion boundaries.
- `failure.public.json`: original failure values with absolute-path traceback removed and source-relative stack locations retained.
- `config.public.json`: original scientific configuration with the absolute worker interpreter path omitted.
- `resource_evidence.json`: selected unchanged resource samples and bounded interpretation.
- `periodic_episodes.csv` and `periodic_episodes.jsonl`: byte-preserved raw 90-case periodic results. Relative `trace` paths refer to the full archive, not files included here.
- `periodic_data_validation.json`: row, case, round and finite-return checks; no trace-level reward audit is claimed.
- `backup_verification.json` and `failure_numeric_review.json`: archived verification records. They are historical evidence, not a new full-data validation.
- `latest_checkpoint_sidecar_summary.json`: unchanged selected sidecar values; no new model-payload rehash is claimed.
- `source_sha256.json` and `provenance.json`: source identities and archive-relative input hashes.
- `SHA256SUMS.json`: hashes for the public files other than this checksum file itself.

Full checkpoints, replay, environments, caches and private machine details are excluded from this small public subset. The original full archive remains unchanged. Trace payload omission and redactions are explicit; this subset is not a resumable training archive.

## Exact historical source snapshot

[Source snapshot and verification manifest](source_snapshot/README.md) preserves the actual d570bee Minecraft and V2 dependencies, configurations, runtime checks, tests and dependency lock. The repository-root matched-study tree contains a different historical Minecraft revision; do not mix files between these versions. These are byte-verified historical sources, not newly executed tests.
