# Defect review as of 2026-11-05

Evidence cutoff: **2026-11-05 09:38 UTC**. These documents report the preserved experiment state; they are not a successful final evaluation or a new experiment.

## Reports

- [Full Chinese PDF, 9 pages](HelixWM_defect_report_full_zh.pdf)
- [Editable Word version](HelixWM_defect_report_full_zh.docx)
- [Full searchable text](full_report_text_zh.txt)
- [Earlier 5-page defect PDF, public copy](HelixWM_defect_report_initial_zh_public.pdf)
- [Earlier report text, public copy](initial_report_text_zh_public.txt)
- [Earlier Chinese comparison report](../2026-10-29-matched-walker-development/HelixWM_DreamerV3_comparison_zh.pdf)

The full PDF and DOCX preserve the delivered bytes. The earlier PDF/text are separate public copies with personal local-machine paths omitted; experimental measurements and conclusions are unchanged. Original local copies are retained. The full report's reference to a local provenance file refers to the original local delivery; this public folder provides `provenance.json` and `SHA256SUMS.json` without private machine paths.

## Findings and limits

Both matched training arms completed 1M actions, 2M DMC calls, 2,000 episodes and 248,750 updates, with exit 0. Development gap20 means were D 923.28 and H 948.67 (+25.39, +2.75%), but reward-defined functional failures were D 1/5 and H 2/5. This is not evidence that H is more reliable.

The final evaluator stopped on a 6-ULP return-summation consistency mismatch in its first case. **0/1280 final cases have passed acceptance.** The failing case executed 500 actions; later freeze and environment gates were not reached. No repair or retry is implied by the diagnosis.

The review distinguishes copied latent states from independently trained models, known-mask input from automatic fault detection, the 5-missing-observation candidate bound from sustained 20-step repair, and relative candidate ranking from absolute trustworthiness. Rejection, calibrated uncertainty and return-frame posterior fusion remain proposed directions.

All numerical conclusions are single-training-seed findings. Large checkpoints, versioned replay and complete logs remain in the separate local experiment archive. This compact GitHub publication is not `FINAL_PACKAGE`.

## Sources

- [Raw development evidence and recomputation code](../2026-10-29-matched-walker-development/)
- [Registered protocol](../../configs/dreamerv3_helixwm_matched_20261026.json)
- [Frozen experimental source commit](https://github.com/iceezhu306-lgtm/HelixWM-for-world-model/tree/97e049941bd755fab68e85e8bb1171911fa6cac8)
- [New requirements received](../../docs/helixwm_final_20261105/README.md)
