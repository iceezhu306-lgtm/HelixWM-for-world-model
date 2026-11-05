# HelixWM Final requirements received on 2026-11-05

Status: **requirements received; not implemented or experimentally validated**.

The [original Chinese source](user_requirements_original_zh.md) is preserved byte for byte. The supplied attachment ends in section 7.2 at `H2：Helix`. No omitted model groups, schedules or acceptance rules have been invented. Further source text can be added as a separately dated revision.

## Requested direction

The requested target is a Real–Dream architecture with bidirectional constrained correction, explicit predictive provenance, multiple candidate trajectories, calibrated uncertainty, rejection when all candidates are unreliable, bounded long-gap operation, and posterior correction when observations return. Each added module must specify its mathematics, tensor interface, gradient and parameter-sharing rules, compute costs, and a falsifiable ablation. New losses are hypotheses, not automatically necessary components.

The requested study levels are evaluator correctness, core confirmatory comparisons, mechanism ablations, and long-gap/unknown-fault generalization. Thresholds must be calibrated on independent data and frozen before testing. The source requests gaps 1/5/10/20/50/100 for the future study; this does not modify the existing registered comparison.

## Preserved experimental boundary

- The current V2 source, trained checkpoints and failed final-evaluation evidence remain baselines.
- No training, evaluator repair, evaluation retry, threshold tuning or new architecture was executed by this publication.
- The source explicitly requires an evaluator audit and approval before final evaluation. Uploading this requirement is not evidence that the gate has passed.
- Existing development gap20 results show higher mean return for H but 2/5 reward-defined functional failures versus D's 1/5. They do not establish reliability or long-horizon superiority.
- `size12m` is the official preset used by the matched experiment, not a claim of original-paper full-scale reproduction.

## Numerical wording clarification

Section A of the supplied source uses the shorthand “NumPy sum” for the differing value. The preserved diagnosis distinguishes Python built-in `sum` over NumPy scalar elements from `np.sum`: the failing check used the former, producing 989.9474279803873, while the saved return and the audited NumPy reduction produced 989.9474279803866. The 6-ULP summation-path mismatch remains the identified defect. The original source is retained without silent editing.

## Read with

- [Full defect report and editable document](../../results/2026-11-05-defect-review/README.md)
- [Matched development results and failure evidence](../../results/2026-10-29-matched-walker-development/REPORT.md)
- [Existing registered matched protocol](../experiments/2026-10-26-matched-dreamerv3-helixwm.md)
- [Research and publication index](../PUBLICATION_INDEX.md)
