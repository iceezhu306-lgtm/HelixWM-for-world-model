# Second experiment: public evidence bundle

This is a compact export of the completed **one-training-seed** HelixWM V2 Walker Walk development experiment. Training started on 2026-10-14 and completed on 2026-10-15. The source commit was `80cb3555586d01e0e7b0e8d362c27e540032d901`; the upstream DreamerV3 commit was `e3f02248693a79dc8b0ebd62c93683888ddaccfe`.

The budget was **1,000,000 executed training actions = 2,000,000 DMC step calls**, with action repeat 2 and RGB64 observations. DMC calls are not MuJoCo integration substeps. Evaluation interactions are excluded from the training budget. There was one trained policy, evaluated under four observation-handling interventions; these are not four independently trained agents.

## Recompute

From the repository root, using Python 3.10 or newer and NumPy:

```bash
python scripts/analyze_second_experiment.py
python -m unittest discover -s tests -p test_second_experiment_analysis.py -v
```

The analysis checks exported file SHA256 values, recomputes means and population/sample standard deviations, pairs returns by environment seed with schedule/policy metadata checks, computes 50,000-sample percentile bootstrap intervals, and recomputes ESS from all retained candidate weights. It recreates `summary.json` and `learning_curve.csv` deterministically; it does not connect to a server, load a model, or start an experiment. The exported statistics were computed with NumPy 2.2.6. The original experiment used NumPy 1.26.4; its environment packages are recorded in `metadata.json`.

## Files and statistical units

| File | Content and scope |
|---|---|
| `eval.csv` | Unmodified V2 per-episode evaluations, including every periodic and final row; `agent_steps` is training actions, `training_env_steps` is DMC calls, `evaluation_seed` is the pairing key. |
| `v1_eval.csv` | Unmodified V1 per-episode evaluation export; the seed column is `eval_seed`. V1 is a descriptive historical reference, not a controlled algorithm ablation. |
| `learning_curve.csv` | Condition/checkpoint/kind aggregates recomputed from `eval.csv`, with episode count and both definitions of standard deviation. No smoothing. |
| `repair_events.jsonl` | All 188 final-evaluation events: 184 repairs and 4 abstentions. Raw candidate weights and per-pixel scores are retained; redundant latent/image hashes are omitted. |
| `training_repair_weights.csv` | All 18,733 training events: 18,509 repairs and 224 abstentions. Eight original weights and stored ESS are present for each repair. Abstention weight columns are empty. |
| `training_metrics.csv` | All 12,000 original metric rows, projected to the fields used in the diagnostic figure. Loss/gradient values already average updates within each logged interval; no extra averaging was applied to this export. |
| `resource_samples.csv` | All 1,738 resource samples with explicit byte/KiB/MiB units; these are samples, not continuously measured peaks. |
| `config.json` / `backbone_config.json` | Actual driver configuration and public-safe projection of resolved backbone algorithm/compute settings. The custom driver controls the budget; unused upstream run-loop defaults are omitted. |
| `v1_config.json` | Actual V1 driver configuration, retained to expose comparison differences. |
| `metadata.json` | Source identity, package versions, completion/reload evidence, frozen-evaluation hash gate, and video identity. Internal paths and addresses were removed by explicit field projection. |
| `summary.json` | Recomputed statistics, conditional episode bootstrap intervals, missingness-schedule audit, and scope limits. |
| `provenance.json` | SHA256 and byte size of original source files and public exports, plus per-file transformation descriptions. It deliberately does not hash itself. |
| `plots/` | The original accepted learning-curve and training-diagnostics PNG/PDF files, copied without alteration. Their public source rows are included above. |
| `videos/clean_training_final.mp4` | The sole formal video: first final clean evaluation episode, not best-of-N; 501 frames at 20 fps, rendered at 256 x 256. This display resolution is not the RGB64 policy input resolution. |

Original full checkpoints, replay, logs, and environment artifacts remain in the retained local archive and are intentionally excluded from this compact public bundle. No new model evaluation was performed to prepare this export. The recorded checkpoint reload gate and frozen zero-update gate are separate checks: `checkpoint_reload_checked_here=false` in the evaluation report refers to that report's own scope, while the completion record separately confirms `checkpoint_reload_equal=true`.

## What the results support

The final 20-episode means are clean **946.8822**, dropout with repair **940.0385**, dropout with black-frame input **937.4552**, and dropout with prior-only propagation **946.0650**. All use the same trained policy and fixed environment/policy/missingness seeds. The paired repair-minus-black-frame mean is **+2.5833**, while repair-minus-prior-only is **-6.0265**. Their exploratory 95% episode bootstrap intervals both include zero; this experiment does not establish that repair improves returns over these controls.

V1's final mean was **69.3800**, so V2's clean final mean is higher by **877.5023**. This is an observed improvement between whole system versions. The backbone, optimization, replay settings, policy sampling, and training observation handling changed together, so it does not isolate the dual-chain mechanism's contribution. No inferential interval is reported for the V1/V2 whole-system comparison. The V2 intervention bootstrap intervals in `summary.json` resample evaluation episodes conditional on the trained seed, not independent training runs.

The configured sampled burst length was 1 to 5, but adjacent bursts can join. Reconstructing the final 20 schedules matches every saved availability SHA256: there were 563 missing observations, and the longest realized contiguous gap was **8**. Three contiguous gaps had length 7 and one had length 8. The four abstentions triggered when `missing_length` first reached 6, exceeding the repair budget of 5. A trigger value of 6 is not a maximum-gap measurement. The schedule audit records per-episode details.

Final repair ESS averaged **7.9478 out of 8** candidates, with 158 of 184 events above 7.9. Weights were therefore usually close to uniform. This measures candidate-weight diversity; it is not evidence that uncertainty is calibrated or that the selected repaired state was correct.

These remain one-seed development results. Evaluation episodes are not independent training seeds; the bootstrap intervals do not measure retraining variability, and the exploratory contrasts have no multiple-comparison adjustment. Known synthetic missingness does not test learned anomaly detection or general domain shift.
