"""Prespecified units, censoring, paired clusters and exported artifact tests."""
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np

from helixwm_v3.analysis import (analyze_records, analyze_study, bootstrap_mean_ci,
                                case_metrics, clopper_pearson_upper, holm_adjust,
                                sign_randomization_p)


ROOT = Path(__file__).resolve().parents[1]


def design():
    result = json.loads((ROOT/"configs/continuity_study_v3.json").read_text(encoding="utf-8"))
    result["analysis"]["bootstrap_replicates"] = 200
    result["analysis"]["paired_sign_randomization_draws"] = 1000
    return result


def fixture(arm="v2_exact", length=5, *, n=500, seed=40000, stage="smoke", low=None):
    if arm == "clean":
        length = 0
    mask = np.ones(501, bool)
    mask[150:150+length] = False
    reward = np.full(n, 2., np.float64)
    if low is not None:
        reward[low] = 0
    trace = dict(action_index=np.arange(n), reward=reward, env_steps=np.full(n, 2),
                 actions=np.zeros((n, 6), np.float32), availability=mask,
                 torso_height=np.ones(n), torso_upright=np.ones(n), latency_s=np.full(n, .001))
    summary = dict(case_id=f"{stage}-{seed}-{arm}-{length}", stage=stage, split=stage,
                   arm=arm, environment_seed=seed, missing_length=length, missing_start=150,
                   status="complete" if n == 500 else "technical_failure",
                   chain_root_seed=2000000+seed, inference_version=248750,
                   error=None if n == 500 else "synthetic failure", actual_eval_agent_steps=n,
                   actual_eval_dmc_calls=2*n, episode_return=float(reward.sum()),
                   availability_sha256=hashlib.sha256(mask.astype(np.uint8).tobytes()).hexdigest(),
                   hash_gate={"passed": True, "before": {"value": "unchanged"}, "after": {"value": "unchanged"}})
    return summary, trace


def metric(arm="v2_exact", length=5, **kwargs):
    s, t = fixture(arm, length, **kwargs)
    cs, ct = fixture("clean", seed=kwargs.get("seed", 40000), stage=kwargs.get("stage", "smoke"))
    return case_metrics(s, t, design(), ct, cs)


class ContinuityMetricsTests(unittest.TestCase):
    def test_action_index_not_returned_observation_index(self):
        row = metric(low=slice(149, 150))
        self.assertEqual(row["fault_mean_reward"], 1.)
        self.assertEqual(row["paired_integrated_deficit"], 0.)
        self.assertAlmostEqual(row["pre_fault_mean_reward"], .98)
        row = metric(low=slice(150, 155))
        self.assertEqual(row["fault_mean_reward"], 0.)
        self.assertAlmostEqual(row["paired_integrated_deficit"], 5/105)
        self.assertEqual(row["recovery_mean_reward"], 1.)
        self.assertEqual(row["actual_eval_dmc_calls"], 1000)

    def test_recovery_requires_full_trailing_windows_and_ten_checks(self):
        row = metric()
        self.assertEqual(row["recovery_time_actions"], 19)
        self.assertEqual(row["recovery_status"], "observed")
        row = metric(low=slice(155, 255))
        self.assertIsNone(row["recovery_time_actions"])
        self.assertEqual(row["recovery_status"], "right_censored")
        self.assertEqual(row["recovery_censor_at_actions"], 100)

    def test_technical_tail_imputed_only_for_primary_not_failure(self):
        row = metric(n=160, low=slice(150, 160))
        self.assertTrue(row["technical_failure"])
        self.assertEqual(row["primary_zero_filled_action_count"], 340)
        self.assertEqual(row["paired_integrated_deficit"], 1.)
        self.assertIsNone(row["functional_failure"])
        self.assertIsNone(row["complete_case_paired_deficit"])
        self.assertEqual(row["recovery_status"], "unknown_after_technical_failure")
        self.assertEqual(row["recovery_censor_at_actions"], 5)
        self.assertFalse(row["no_posture_failure_proxy"])
        # An already observed failure cannot disappear because the case stopped.
        row = metric(n=180, low=slice(150, 180))
        self.assertTrue(row["functional_failure"])

    def test_technical_failure_after_full_window_keeps_observed_false(self):
        row = metric(n=300)
        self.assertTrue(row["technical_failure"])
        self.assertFalse(row["functional_failure"])
        self.assertEqual(row["paired_integrated_deficit"], 0.)
        self.assertIsNone(row["complete_case_paired_deficit"])

    def test_executed_step_without_valid_reward_is_not_lost(self):
        s, t = fixture(n=160)
        cs, ct = fixture("clean")
        s.update(actual_eval_agent_steps=161, actual_eval_dmc_calls=322,
                 valid_reward_rows=160, actual_interaction_counters_complete=False)
        row = case_metrics(s, t, design(), ct, cs)
        self.assertEqual(row["actual_eval_agent_steps"], 161)
        self.assertEqual(row["valid_reward_rows"], 160)
        self.assertEqual(row["reward_rows_missing_after_execution"], 1)
        self.assertEqual(row["primary_zero_filled_action_count"], 340)
        self.assertTrue(row["issued_action_fraction_is_lower_bound"])
        self.assertIsNone(row["functional_failure"])

    def test_gap_zero_parity_is_smoke_only_and_pair_seeds_must_match(self):
        s, t = fixture("posterior_return_once", length=0)
        cs, ct = fixture("clean")
        self.assertEqual(case_metrics(s, t, design(), ct, cs)["paired_integrated_deficit"], 0.)
        s["chain_root_seed"] += 1
        with self.assertRaises(ValueError):
            case_metrics(s, t, design(), ct, cs)
        s, t = fixture("posterior_return_once", length=0, stage="design", seed=20000)
        cs, ct = fixture("clean", stage="design", seed=20000)
        with self.assertRaises(ValueError):
            case_metrics(s, t, design(), ct, cs)

    def test_low_prefault_performance_not_excluded_from_primary(self):
        row = metric(low=slice(100, 150))
        self.assertFalse(row["recovery_time_eligible"])
        self.assertEqual(row["recovery_status"], "ineligible")
        self.assertEqual(row["paired_integrated_deficit"], 0.)

    def test_posture_proxy_and_invalid_reference_are_separate(self):
        s, t = fixture()
        cs, ct = fixture("clean")
        t["torso_height"][150:160] = .5
        row = case_metrics(s, t, design(), ct, cs)
        self.assertTrue(row["posture_failure"])
        self.assertFalse(row["no_posture_failure_proxy"])
        t["torso_upright"][100:150] = .2
        row = case_metrics(s, t, design(), ct, cs)
        self.assertFalse(row["posture_reference_eligible"])
        self.assertIsNone(row["posture_failure"])
        self.assertIsNone(row["no_posture_failure_proxy"])

    def test_counter_mask_hash_finite_and_window_validation(self):
        cs, ct = fixture("clean")
        for failure in ("counter", "mask", "nan", "index", "hash"):
            s, t = fixture()
            if failure == "counter":
                s["actual_eval_dmc_calls"] = 999
            elif failure == "mask":
                t["availability"][149] = False
            elif failure == "nan":
                t["reward"][200] = np.nan
            elif failure == "index":
                t["action_index"][3] = 4
            else:
                s["availability_sha256"] = "incorrect"
            with self.subTest(failure=failure), self.assertRaises(ValueError):
                case_metrics(s, t, design(), ct, cs)

    def test_hash_flag_alone_does_not_prove_frozen_state(self):
        s, t = fixture()
        cs, ct = fixture("clean")
        s["hash_gate"]["after"] = {"value": "changed"}
        self.assertFalse(case_metrics(s, t, design(), ct, cs)["frozen_hash_gate_passed"])


class ContinuityStatisticsTests(unittest.TestCase):
    def test_bootstrap_sign_randomization_and_holm_known_cases(self):
        self.assertEqual(bootstrap_mean_ci([4]*5, draws=100), [4., 4.])
        self.assertEqual(bootstrap_mean_ci([1, 3, 9], seed=12, draws=100),
                         bootstrap_mean_ci([1, 3, 9], seed=12, draws=100))
        self.assertEqual(sign_randomization_p([0]*5, draws=100), 1.)
        p = sign_randomization_p([-1]*12, seed=42, draws=10000)
        self.assertGreater(p, 0.)
        self.assertLess(p, .01)
        np.testing.assert_allclose(holm_adjust([.01, .04, .03, .002]), [.03, .06, .06, .008])

    def test_exact_binomial_bound_handles_zero_and_all_events(self):
        self.assertAlmostEqual(clopper_pearson_upper(0, 40), 1-.05**(1/40), places=14)
        self.assertGreater(clopper_pearson_upper(0, 40), .05)
        self.assertEqual(clopper_pearson_upper(40, 40), 1.)
        self.assertAlmostEqual(clopper_pearson_upper(1, 2), math.sqrt(.95), places=13)
        with self.assertRaises(ValueError):
            clopper_pearson_upper(0, 0)

    def matrix(self):
        config = design()
        config["phase_a"]["splits"]["held_out_environment_seeds"]["count"] = 2
        rows = []
        values = {"black_frame": [.4, .8], "prior_only": [.3, .7], "v2_exact": [.2, .6],
                  "locked_ancestor": [.1, .1], "posterior_return_once": [0., .4]}
        for i, seed in enumerate([30000, 30001]):
            rows.append(metric("clean", seed=seed, stage="held_out"))
            for arm, deficits in values.items():
                for length in [1, 5, 10, 20, 50]:
                    row = metric(arm, length, seed=seed, stage="held_out")
                    row["paired_integrated_deficit"] = deficits[i]
                    row["complete_case_paired_deficit"] = deficits[i]
                    rows.append(row)
        return config, rows

    def test_bootstrap_unit_is_seed_cluster_not_duration_episode(self):
        config, rows = self.matrix()
        report = analyze_records(rows, config)["stages"]["held_out"]
        self.assertTrue(report["matrix_complete"])
        self.assertEqual(report["expected_cases"], 52)
        comparison = report["comparisons"][0]
        self.assertEqual(comparison["n"], 2)
        self.assertEqual(comparison["seed_clusters"], [30000, 30001])
        self.assertAlmostEqual(comparison["mean"], -.3)
        self.assertEqual(comparison["failure_guard"], "PASS_DESCRIPTIVE")
        self.assertEqual(comparison["bootstrap_ci95"], bootstrap_mean_ci([-.1, -.5], seed=4000000, draws=200))

    def test_technical_failure_kept_in_primary_and_guard_inconclusive(self):
        config, rows = self.matrix()
        damaged = next(x for x in rows if x["arm"] == "locked_ancestor" and x["environment_seed"] == 30000)
        damaged.update(technical_failure=True, status="technical_failure", functional_failure=None,
                       complete_case_paired_deficit=None)
        report = analyze_records(rows, config)["stages"]["held_out"]
        comparison = report["comparisons"][0]
        self.assertEqual(comparison["n"], 2)
        self.assertEqual(comparison["complete_case_sensitivity"]["n"], 1)
        self.assertEqual(comparison["failure_guard"], "INCONCLUSIVE")
        self.assertIsNone(report["failure_uncertainty"]["locked_ancestor"]["one_sided_95_upper"])

    def test_incomplete_matrix_cannot_claim_confirmation_and_duplicates_rejected(self):
        config, rows = self.matrix()
        report = analyze_records(rows[:-1], config)["stages"]["held_out"]
        self.assertFalse(report["matrix_complete"])
        self.assertEqual(report["analysis_scope"], "descriptive_only")
        self.assertTrue(all(x["failure_guard"] == "INCONCLUSIVE" for x in report["comparisons"]))
        with self.assertRaises(ValueError):
            analyze_records(rows+[copy.deepcopy(rows[0])], config)

    def test_file_export_reads_raw_data_and_does_not_change_case_files(self):
        with tempfile.TemporaryDirectory(dir=os.environ.get("HELIXWM_TEST_TMP")) as directory:
            root = Path(directory)
            files = []
            for arm in ("clean", "v2_exact"):
                summary, trace = fixture(arm)
                folder = root/"cases"/summary["case_id"]
                folder.mkdir(parents=True)
                (folder/"summary.json").write_text(json.dumps(summary), encoding="utf-8")
                np.savez_compressed(folder/"trace.npz", **trace)
                files.extend([folder/"summary.json", folder/"trace.npz"])
            before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
            result = analyze_study(root, design(), make_plots=True)
            self.assertFalse(result["stages"]["smoke"]["matrix_complete"])
            self.assertEqual(before, {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
            self.assertTrue((root/"ANALYSIS/smoke/case_metrics.csv").is_file())
            self.assertTrue((root/"ANALYSIS/smoke/robustness_recovery.pdf").read_bytes().startswith(b"%PDF-"))
            from PIL import Image
            with Image.open(root/"ANALYSIS/smoke/robustness_recovery.png") as image:
                self.assertGreater(image.width, 1000)
                self.assertGreater(np.asarray(image).std(), 1)
            saved = json.loads((root/"ANALYSIS/analysis.json").read_text(encoding="utf-8"))
            self.assertEqual(len(saved["source_artifacts"]), 4)


if __name__ == "__main__":
    unittest.main()
