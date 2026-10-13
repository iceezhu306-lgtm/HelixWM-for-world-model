"""Observed-data aggregation, missing evidence, and artifact integrity tests."""

import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from helixwm.artifacts import export_run


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        (self.root / "config.json").write_text(json.dumps({
            "seed": 0, "experiment_class": "single_seed_pilot",
            "budget_agent_steps": 1000000, "environment": {"action_repeat": 2},
        }), encoding="utf-8")

    def tearDown(self):
        self.temporary.cleanup()

    def _jsonl(self, name, rows):
        (self.root / name).write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    def _inputs(self):
        self._jsonl("metrics.jsonl", [
            {"agent_steps": 20, "env_steps": 40, "wall_time_s": 1.5, "episode_return": 8, "episode_length": 20},
            {"agent_steps": 20, "env_steps": 40, "update_steps": 1, "model_loss": 3.25, "trusted_horizon": 2, "risk_probability": 0.2},
        ])
        self._jsonl("evaluations.jsonl", [
            {"agent_steps": 20, "env_steps": 40, "seed": 0, "eval_seed": 10000, "episode_return": 10, "episode_length": 5, "kind": "periodic"},
            {"agent_steps": 20, "env_steps": 40, "seed": 0, "eval_seed": 10001, "episode_return": 20, "episode_length": 5, "kind": "periodic"},
            {"agent_steps": 30, "env_steps": 60, "seed": 0, "eval_seed": 10000, "episode_return": 30, "episode_length": 5, "kind": "final"},
        ])
        self._jsonl("resources.jsonl", [{"agent_steps": 20, "env_steps": 40, "gpu_util_percent": 45}])

    @staticmethod
    def _fake_plot(plot_dir, metrics, aggregates, title):
        paths = [plot_dir / name for name in ("learning_curve.png", "learning_curve.pdf", "diagnostics.png")]
        for path in paths:
            path.write_bytes(b"mock plot used only for exporter unit tests")
        return paths

    def _export(self):
        with patch("helixwm.artifacts._plot_run", side_effect=self._fake_plot):
            return export_run(self.root)

    def _csv(self, name):
        with (self.root / "RESULTS" / name).open(encoding="utf-8", newline="") as stream:
            return list(csv.DictReader(stream))

    def test_statistics_preserve_checkpoint_groups_and_raw_evaluation_rows(self):
        self._inputs()
        summary = self._export()
        self.assertEqual(summary["status"], "exported")
        self.assertEqual(summary["missing"], [])
        self.assertEqual(summary["errors"], [])
        self.assertEqual(summary["evaluation_rows"], 3)
        self.assertEqual(summary["aggregated_evaluation_rows"], 3)
        self.assertEqual(summary["evaluation_groups"], 2)
        raw = self._csv("eval_raw.csv")
        self.assertEqual([float(row["episode_return"]) for row in raw], [10, 20, 30])
        self.assertEqual([int(row["eval_seed"]) for row in raw], [10000, 10001, 10000])
        aggregate = self._csv("aggregate_results.csv")
        first, final = aggregate
        self.assertEqual(int(first["n"]), 2)
        self.assertEqual(float(first["mean"]), 15)
        self.assertEqual(float(first["std"]), 5)
        self.assertEqual(float(first["median"]), 15)
        self.assertEqual(float(first["min"]), 10)
        self.assertEqual(float(first["max"]), 20)
        self.assertEqual(int(first["std_ddof"]), 0)
        self.assertEqual(float(final["std"]), 0)
        stats = json.loads((self.root / "RESULTS" / "mean_std.json").read_text(encoding="utf-8"))
        self.assertEqual(stats["groups"][0]["mean"], 15)
        self.assertIn("not uncertainty across training seeds", stats["std_definition"])
        self.assertEqual(summary["acceptance_status"], "not_assessed_by_exporter")

    def test_raw_metrics_keep_sparse_columns_and_custom_values(self):
        self._inputs()
        self._jsonl("metrics.jsonl", [
            {"agent_steps": 1, "env_steps": 2, "episode_return": 7, "metadata": {"source": "test"}},
            {"agent_steps": 2, "env_steps": 4, "model_loss": 1.5, "risk_probability": 0.3},
        ])
        self._export()
        rows = self._csv("train_metrics.csv")
        self.assertEqual(rows[0]["model_loss"], "")
        self.assertEqual(rows[1]["episode_return"], "")
        self.assertEqual(rows[1]["model_loss"], "1.5")
        self.assertEqual(json.loads(rows[0]["metadata"]), {"source": "test"})

    def test_same_steps_with_different_identity_or_kind_are_not_merged(self):
        self._inputs()
        base = {"agent_steps": 20, "env_steps": 40, "seed": 0, "episode_return": 10, "episode_length": 5, "kind": "periodic"}
        self._jsonl("evaluations.jsonl", [
            dict(base, checkpoint_sha256="abc"),
            dict(base, checkpoint_sha256="def"),
            dict(base, checkpoint_sha256="abc", kind="final"),
        ])
        summary = self._export()
        self.assertEqual(summary["evaluation_groups"], 3)
        self.assertTrue(all(int(row["n"]) == 1 for row in self._csv("aggregate_results.csv")))

    def test_actual_checkpoint_and_invocation_ids_prevent_repeated_eval_pooling(self):
        self._inputs()
        base = {"agent_steps": 20, "env_steps": 40, "seed": 0, "eval_seed": 10000, "episode_return": 10, "episode_length": 5, "kind": "periodic"}
        self._jsonl("evaluations.jsonl", [
            dict(base, checkpoint_id="checkpoint_000000020.pt", evaluation_id="first"),
            dict(base, checkpoint_id="checkpoint_000000020.pt", evaluation_id="second"),
            dict(base, checkpoint_id="different_checkpoint.pt", evaluation_id="first"),
        ])
        summary = self._export()
        self.assertEqual(summary["evaluation_groups"], 3)
        rows = self._csv("aggregate_results.csv")
        self.assertTrue(all(int(row["n"]) == 1 for row in rows))
        self.assertEqual({row["evaluation_id"] for row in rows}, {"first", "second"})
        self.assertEqual({row["checkpoint_id"] for row in rows}, {"checkpoint_000000020.pt", "different_checkpoint.pt"})

    def test_missing_inputs_are_explicit_without_invented_statistics(self):
        (self.root / "config.json").unlink()
        summary = self._export()
        self.assertEqual(summary["status"], "exported_with_gaps")
        for filename in ("config.json", "metrics.jsonl", "evaluations.jsonl", "resources.jsonl"):
            self.assertIn(filename, summary["missing"])
        self.assertEqual(self._csv("eval_raw.csv"), [])
        self.assertEqual(self._csv("aggregate_results.csv"), [])
        self.assertEqual(summary["aggregated_evaluation_rows"], 0)

    def test_invalid_records_are_reported_and_readable_rows_remain_raw(self):
        self._inputs()
        path = self.root / "evaluations.jsonl"
        with path.open("a", encoding="utf-8") as stream:
            stream.write("{partial log row\n")
            stream.write(json.dumps({"agent_steps": 40, "env_steps": 80, "seed": 0, "episode_return": float("nan"), "episode_length": 5, "kind": "final"}) + "\n")
        summary = self._export()
        self.assertEqual(summary["evaluation_rows"], 4)
        self.assertEqual(summary["aggregated_evaluation_rows"], 3)
        self.assertEqual(len(self._csv("eval_raw.csv")), 4)
        self.assertTrue(any("evaluations.jsonl:4" in error for error in summary["errors"]))
        self.assertTrue(any("excluded from statistics" in error for error in summary["errors"]))

    def test_manifest_is_sorted_complete_and_valid_after_repeated_export(self):
        self._inputs()
        self._export()
        summary = self._export()
        lines = (self.root / "MANIFEST.sha256").read_text(encoding="utf-8").splitlines()
        records = [line.split("  ", 1) for line in lines]
        paths = [relative for digest, relative in records]
        self.assertEqual(paths, sorted(paths))
        self.assertNotIn("MANIFEST.sha256", paths)
        self.assertEqual(len(records), summary["manifest_file_count"])
        self.assertEqual(paths, summary["manifest_covered_files"])
        self.assertTrue(summary["manifest_included"])
        self.assertIn("+00:00", summary["snapshot_started_utc"])
        self.assertFalse(summary["writers_quiescent_verified"])
        self.assertEqual(set(paths), {path.relative_to(self.root).as_posix() for path in self.root.rglob("*") if path.is_file() and path.name != "MANIFEST.sha256"})
        for digest, relative in records:
            self.assertEqual(digest, hashlib.sha256((self.root / relative).read_bytes()).hexdigest())
        self.assertEqual(summary["count"], len(summary["files"]))
        self.assertTrue(all((self.root / relative).is_file() for relative in summary["files"]))

    def test_plot_failure_does_not_discard_raw_data(self):
        self._inputs()
        self._export()
        self.assertTrue((self.root / "PLOTS" / "learning_curve.png").is_file())
        with patch("helixwm.artifacts._plot_run", side_effect=ImportError("test plotting dependency absent")):
            summary = export_run(self.root)
        self.assertTrue(any("Plot export failed" in error for error in summary["errors"]))
        self.assertIn("current learning curve and diagnostic plots", summary["missing"])
        self.assertEqual(len(self._csv("eval_raw.csv")), 3)
        self.assertNotIn("PLOTS/learning_curve.png", summary["files"])
        self.assertFalse((self.root / "PLOTS" / "learning_curve.png").exists())
        self.assertFalse((self.root / "PLOTS" / "learning_curve.pdf").exists())
        self.assertFalse((self.root / "PLOTS" / "diagnostics.png").exists())

    def test_periodic_export_skips_all_hashing_and_removes_stale_manifest(self):
        self._inputs()
        self._export()
        self.assertTrue((self.root / "MANIFEST.sha256").exists())
        with patch("helixwm.artifacts._plot_run", side_effect=self._fake_plot), patch("helixwm.artifacts._hash_file", side_effect=AssertionError("periodic export must not hash archive")):
            summary = export_run(self.root, include_manifest=False)
        self.assertFalse(summary["manifest_included"])
        self.assertEqual(summary["manifest_file_count"], 0)
        self.assertEqual(summary["manifest_covered_files"], [])
        self.assertNotIn("MANIFEST.sha256", summary["files"])
        self.assertFalse((self.root / "MANIFEST.sha256").exists())
        self.assertEqual(summary["errors"], [])
        self.assertEqual(summary["status"], "exported")

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "optional plotting package is not installed")
    def test_real_plot_files_render_as_png_and_pdf(self):
        self._inputs()
        summary = export_run(self.root)
        self.assertEqual(summary["errors"], [])
        self.assertTrue((self.root / "PLOTS" / "learning_curve.png").read_bytes().startswith(b"\x89PNG"))
        self.assertTrue((self.root / "PLOTS" / "learning_curve.pdf").read_bytes().startswith(b"%PDF-"))
        self.assertTrue((self.root / "PLOTS" / "diagnostics.png").read_bytes().startswith(b"\x89PNG"))


if __name__ == "__main__":
    unittest.main()
