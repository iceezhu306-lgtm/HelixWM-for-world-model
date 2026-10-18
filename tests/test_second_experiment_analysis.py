"""Independent aggregation, pairing, missingness and integrity checks."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.analyze_second_experiment import (
    audit_dropout_schedule, bootstrap_mean, effective_sample_size,
    paired_rows, parse_gpu_sample, read_csv, sha256, summarize, verify_provenance,
)


class SecondExperimentAnalysisTests(unittest.TestCase):
    def test_population_and_sample_spread_are_distinct(self):
        result = summarize([1, 2, 3, 4])
        self.assertEqual(result["n"], 4)
        self.assertEqual(result["mean"], 2.5)
        self.assertAlmostEqual(result["population_sd"], np.sqrt(1.25))
        self.assertAlmostEqual(result["sample_sd"], np.sqrt(5 / 3))
        with self.assertRaises(ValueError):
            summarize([1, float("nan")])

    def test_bootstrap_is_reproducible_and_constant_sample_is_exact(self):
        self.assertEqual(bootstrap_mean([7, 7, 7], samples=100), [7.0, 7.0])
        self.assertEqual(bootstrap_mean([1, 9, 4], seed=42, samples=100),
                         bootstrap_mean([1, 9, 4], seed=42, samples=100))

    @staticmethod
    def row(seed, value, condition="dropout"):
        return dict(evaluation_seed=seed, episode_return=value, condition=condition,
                    training_seed=0, evaluation_id="final_a", policy_seed=seed + 100,
                    dropout_seed=seed + 200, agent_steps=1000, availability_sha256="same")

    def test_pairing_uses_seed_not_row_order(self):
        left = [self.row(1, 10), self.row(2, 20)]
        right = [self.row(2, 18, "dropout_prior_only"), self.row(1, 7, "dropout_prior_only")]
        self.assertEqual(paired_rows(left, right), [3, 2])

    def test_pairing_rejects_duplicate_missing_and_incompatible_seeds(self):
        row = self.row(1, 10)
        for left, right in [([row, row], [row]), ([row], [self.row(2, 10)]),
                            ([row], [{**row, "availability_sha256": "changed"}]),
                            ([row], [{**row, "policy_seed": 999}])]:
            with self.assertRaises(ValueError):
                paired_rows(left, right)

    def test_ess_analytical_uniform_and_one_hot_cases(self):
        self.assertEqual(effective_sample_size([0.125] * 8), 8)
        self.assertEqual(effective_sample_size([1, 0, 0]), 1)
        self.assertAlmostEqual(effective_sample_size([0.75, 0.25]), 1.6)
        for invalid in [[0.2, 0.2], [-0.1, 1.1], [float("nan"), 1]]:
            with self.assertRaises(ValueError):
                effective_sample_size(invalid)

    def test_gpu_query_order_does_not_treat_temperature_as_utilization(self):
        # Actual first recorded source sample; train.py queries utilization first.
        self.assertEqual(parse_gpu_sample("0, 880, 32607, 35"),
                         {"gpu_utilization_percent": 0, "gpu_used_memory_mib": 880})
        self.assertEqual(parse_gpu_sample("70, 1024, 32768, 50")["gpu_utilization_percent"], 70)
        with self.assertRaises(ValueError):
            parse_gpu_sample("101, 1024, 32768, 50")

    def test_adjacent_bursts_can_exceed_sampled_maximum(self):
        # Probability one forces adjacent one-frame bursts to form a single gap.
        mask = bytes([0, 0, 0, 0, 1])
        row = dict(dropout_seed="7", actual_eval_agent_steps="5", evaluation_seed="3",
                   missing_observations="4", availability_sha256=hashlib.sha256(mask).hexdigest())
        config = {"observation_dropout": {"start_probability": 1, "min_length": 1, "max_length": 1},
                  "dual_chain": {"max_missing": 2}}
        result = audit_dropout_schedule([row], config)
        self.assertEqual(result["maximum_realized_contiguous_gap"], 4)
        self.assertEqual(result["episodes"][0]["contiguous_gaps_over_repair_budget"], [4])
        with self.assertRaises(ValueError):
            audit_dropout_schedule([{**row, "availability_sha256": "mismatch"}], config)

    def test_published_episode_contrast_matches_retained_analysis(self):
        path = Path(__file__).resolve().parents[1] / "results/2026-10-14-dual-chain-walker-seed0/eval.csv"
        rows = [row for row in read_csv(path) if row["kind"] == "final"]
        repair = [row for row in rows if row["condition"] == "dropout"]
        black = [row for row in rows if row["condition"] == "dropout_disabled"]
        differences = paired_rows(repair, black)
        self.assertEqual(len(differences), 20)
        self.assertAlmostEqual(summarize(differences)["mean"], 2.5832538719249043)
        interval = bootstrap_mean(differences)
        self.assertAlmostEqual(interval[0], -8.76362247, places=7)
        self.assertAlmostEqual(interval[1], 12.97986152, places=7)

    def test_provenance_checks_bytes_hash_and_path_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            item = root / "sample.csv"
            item.write_text("a,b\n1,2\n", encoding="utf-8")
            manifest = {"exports": {"sample.csv": {"bytes": item.stat().st_size, "sha256": sha256(item)}}}
            (root / "provenance.json").write_text(json.dumps(manifest), encoding="utf-8")
            verify_provenance(root)
            item.write_text("a,b\n1,3\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                verify_provenance(root)
            manifest["exports"] = {"../outside.csv": {"bytes": 0, "sha256": "0" * 64}}
            (root / "provenance.json").write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaises(ValueError):
                verify_provenance(root)


if __name__ == "__main__":
    unittest.main()
