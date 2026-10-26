# Data dictionary and reproducibility boundary

The eight CSV files under `data/design` and `data/held_out` are unchanged outputs from the accepted archive. No rows were removed, no outcomes were filled, and no numerical values were rounded in these files.

| File | Grain | Coverage |
| --- | --- | --- |
| `case_metrics.csv` | One environment seed x inference arm x missing length | 520 design rows; 1,040 held-out rows, including one clean reference per seed |
| `duration_metrics.csv` | One faulted arm x missing length | 25 rows per split; clean is in the case table |
| `seed_cluster_metrics.csv` | One environment seed x faulted arm | 100 design rows; 200 held-out rows |
| `paired_comparisons.csv` | One proposal x comparator | Four predeclared contrasts per split |
| `summary.public.json` | Complete archived split summary with sanitized figure links | All non-`plots` JSON values preserved |

## Consequential fields

- `stage`: `design` or `held_out`; do not merge these for a new confirmatory claim.
- `environment_seed`: reset/evaluation seed, **not** an independently trained agent.
- `missing_length`: unavailable observation/action indices; clean uses zero. Action repeat is two.
- `actual_eval_agent_steps` and `actual_eval_dmc_calls`: actually executed evaluation actions and DMC calls, excluded from the original training budget.
- `raw_observed_episode_return`: sum of action rewards across the complete episode; it is not normalized to the primary deficit scale.
- `paired_integrated_deficit`: mean clean-minus-arm normalized reward in the missing interval plus the next 100 actions. Despite its historical field name, the archived implementation takes a mean, not an unnormalized integral.
- `mean_paired_integrated_deficit`: equal mean of the five registered gap deficits for one seed/arm.
- `functional_failure`: 20 consecutive normalized rewards below 0.2 in the specified fault/recovery window. This is a performance proxy; it is not a measured fall classification.
- `posture_failure`: provisional evaluator-only height/upright threshold proxy. It does not establish safety; the full-episode flag uses a different time window and must not be confused with the fault-window flag.
- `recovery_status`: observed, censored, ineligible, or unknown after technical failure. An unknown/censored recovery is not an observed zero delay.
- `technical_failure`, `clean_reference_technical_failure`, `primary_zero_filled_action_count`: distinguish missing technical observations from actual failure. All cases in this completed archive are technical-success cases and have zero primary fill count.
- `availability_sha256`: hash of the registered input-availability schedule; same mask does not imply identical trajectories after actions diverge.
- `frozen_hash_gate_passed`: recorded per-case frozen-model audit status. The original acceptance record includes strict reload and reference-parity checks.
- `std_population`: population standard deviation of the finite recorded values, not the standard error or between-training-seed uncertainty.
- `bootstrap_ci95`: archived whole-environment-seed-cluster percentile interval.
- `holm_adjusted_p`: multiplicity-adjusted archived one-sided paired sign-randomization p-value. Negative deficit differences favor the proposal.
- `failure_guard`: descriptive observed-count check. It is not a confidence bound or noninferiority test.

The original `summary.json` files contained private absolute figure paths. Published `summary.public.json` files replace **only** the `plots` array with relative links. Source hashes and that transformation are documented in `publication_provenance.json`; these files are not advertised as byte-identical originals.

## What the portable script verifies

`recompute.py` uses only the Python standard library. It verifies hashes and sizes for every file listed in `publication_manifest.json`, checks the complete seed/arm/gap matrices, enforces the recorded completed-case contract, recalculates per-duration return/deficit means and failure counts, rebuilds seed-cluster means, and recomputes the four contrast means. It checks those arithmetic results against the archived tables.

The script deliberately does not fabricate the omitted per-step reward, image, posture, RNG or latent-state trajectories. It does not rerun bootstrap/randomization/Clopper-Pearson procedures, and it copies their archived values into the optional recomputed summary with an `archived_` prefix. The complete source archive and raw traces are required to repeat those deeper analyses and simulator checks. Neither running this script nor the original packaging acceptance establishes scientific superiority.
