# Research, code and result index

Updated 2026-11-05. This index separates received design requirements, implemented code, observed results and final-acceptance status.

| Study | Report and data | Code | Scope and outcome |
|---|---|---|---|
| First Walker pilot, 2026-10-13 | [Failure analysis](experiments/2026-10-13-walker-seed0.md), [raw evidence](experiments/evidence/walker_seed0_20261013/) | [Custom pilot](../helixwm/) | Budget complete; poor control performance; not an official DreamerV3 baseline |
| Bounded dual-chain Walker, 2026-10-14 | [Protocol/report](experiments/2026-10-14-dual-chain-walker-seed0.md), [results, plots and video](../results/2026-10-14-dual-chain-walker-seed0/) | [V2](../helixwm_v2/), [recomputation](../scripts/analyze_second_experiment.py) | One trained seed; bounded known-mask mechanism; general reliability not established |
| Continuity interventions, 2026-10-18 | [Report](../results/2026-10-18-walker-continuity/REPORT.md), [data, figures and video](../results/2026-10-18-walker-continuity/) | [V3 inference study](../helixwm_v3/), [recomputation](../results/2026-10-18-walker-continuity/recompute.py) | 520 design and 1,040 held-out frozen episodes; same trained checkpoint, not independent training controls |
| Minecraft navigation, started 2026-10-21 | [Failure evidence](../results/2026-10-22-minecraft-incomplete/README.md) | [Minecraft runtime](../helixwm_minecraft/) | Custom real-engine task; formal run stopped before budget; not an accepted 1M-action result |
| Matched DreamerV3 vs H, 2026-10-26 | [Report](../results/2026-10-29-matched-walker-development/REPORT.md), [data and plots](../results/2026-10-29-matched-walker-development/) | [Matched runtime](../helixwm_matched/), [analysis code](../results/2026-10-29-matched-walker-development/code/) | Both 1M-action trainings complete; development comparison published; final acceptance blocked |
| Defect review, 2026-11-05 | [Full Word/PDF and earlier report](../results/2026-11-05-defect-review/README.md) | Existing frozen source audited; no algorithm change | Numerical evaluator defect and mechanism limits distinguished from proposed improvements |
| HelixWM Final requirements, 2026-11-05 | [Received scope and original text](helixwm_final_20261105/README.md) | Not implemented by this publication | Attachment ends at section 7.2; omitted content was not reconstructed |

## Preservation policy

Historical experiment source commits, protocols and results are retained. Existing reports are not rewritten to suggest the new design was already tested. The matched study remains tied to runtime commit `97e049941bd755fab68e85e8bb1171911fa6cac8`; the earlier experiments retain their own provenance.

The repository contains code, configurations, analysis scripts, small raw score/trace data, report documents, figures and already-published compact videos. Full checkpoints, all versioned replay, complete logs and large local packages remain in the D/E experiment archives, as specified by the established publication policy. Credentials, private keys, environment caches and SSH operation scripts are not part of this upload.
