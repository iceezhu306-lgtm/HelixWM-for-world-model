# Matched Walker development results

**[Read the comparison report](REPORT.md)** | [Learning curves](figures/development_learning_curves.png) | [Endpoint comparison](figures/latest_paired_returns.png) | [Fault-window traces](figures/latest_fault_recovery_trajectories.png)

Both independent trainings completed their 1M-action budgets. This folder publishes all 50 rounds of development evaluation per arm. The 1,280-case held-out confirmatory evaluation is blocked by a first-case return-consistency gate error (`evaluation.exit=1`); it was not automatically restarted. H has higher mean development gap20 episode return at 1M, but more reward-defined functional failures (2/5 versus 1/5). The report states these limits explicitly.

## Files and recomputation

- `data/development_episodes.csv`: all 1,000 raw per-episode results, keyed by arm, training actions, controller and seed.
- `data/D_development_eval.jsonl` and `H_development_eval.jsonl`: original compact evaluation logs.
- `data/development_round_summary.csv`: 200 arm/condition/checkpoint aggregates; SD is over five environment seeds.
- `data/latest_paired_outcomes.csv`: all five matched endpoint seed rows, including reward losses, window scores, failure flags and recovery observations.
- `data/latest_action_reward_traces.csv`: all 10,000 endpoint action rows (20 episodes × 500 actions).
- `data/latest_traces.zip`: 20 original endpoint NPZ traces and their 20 JSON case records; validated CRC and per-member SHA.
- `data/latest_traces_manifest.json`, `source_manifest_sha256.json`, `analysis_provenance.json`: provenance and integrity.
- `data/development_summary.json`: descriptive metrics and freeze/budget checks; final-case count used is zero.
- `data/registered_protocol.json`: unchanged registration; its DESIGN_ONLY flag is historical and does not describe today's execution status.
- `data/final_evaluation_blocker.json`: the separate failed final-evaluation gate and numerical evidence; this case is not mixed into the development statistics.
- `figures/`: three figures, each in PNG and PDF.
- `code/recompute.py`: reproduce statistics and figures from this public subset, without model weights or a server.
- `code/analyze_development.py`: full-evidence extraction/validation used to create the subset; it requires the separate full local evidence tree.

From this folder, with Python, NumPy and Matplotlib:

```bash
python code/recompute.py --output-dir ../recomputed
```

Do not run inference or retrain to recompute this report. The complete checkpoints, replay versions, and source evidence remain in the private E-drive archive; they are not silently omitted from that archive. This compact publication is not full experiment acceptance.
