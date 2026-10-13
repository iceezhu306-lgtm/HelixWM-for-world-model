"""Read-only monitor checks with synthetic files, never a real training process."""

import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
import unittest

from examples.inspect_run import inspect_pid, inspect_run


class InspectRunTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name)
        self.run = self.base / "run"
        self.proc = self.base / "proc"
        self.cgroup = self.base / "cgroup"
        for path in (self.run, self.proc, self.cgroup):
            path.mkdir()
        self._json("config.json", {"seed": 0, "budget_agent_steps": 1000000, "checkpoint_every": 10000,
                                  "minimum_free_disk_bytes": 1, "environment": {"action_repeat": 2}})
        (self.run / "train.pid").write_text("321\n")
        (self.proc / "321").mkdir()
        self._cmdline(["python", "-m", "helixwm.train", "--run-dir", str(self.run.resolve())])
        (self.cgroup / "memory.current").write_text("1048576\n")
        (self.cgroup / "memory.max").write_text(str(90 * 1024**3))
        (self.cgroup / "memory.events").write_text("low 0\nhigh 0\nmax 0\noom 0\noom_kill 0\n")

    def tearDown(self):
        self.temporary.cleanup()

    def _json(self, name, value):
        path = self.run / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value) + "\n", encoding="utf-8")

    def _jsonl(self, name, rows):
        (self.run / name).write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    def _cmdline(self, args):
        (self.proc / "321" / "cmdline").write_bytes(b"\0".join(value.encode() for value in args) + b"\0")

    def _checkpoint(self, step, payload=b"trusted-test-payload"):
        name = f"checkpoint_{step:09d}"
        path = self.run / "CHECKPOINTS" / f"{name}.pt"
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(payload)
        os.utime(path, (time.time() - 10, time.time() - 10))
        digest = hashlib.sha256(payload).hexdigest()
        self._json(f"CHECKPOINTS/{name}.json", {
            "file": path.name, "bytes": len(payload), "sha256": digest,
            "counters": {"agent_steps": step, "env_steps": step * 2, "update_steps": step // 5},
        })
        return path, digest

    def _inspect(self, **kwargs):
        return inspect_run(self.run, proc_root=self.proc, cgroup_root=self.cgroup, run_commands=False, **kwargs)

    def test_units_budget_real_counters_checkpoint_sha_and_read_only(self):
        self._json("status.json", {"agent_steps": 19900, "env_steps": 39800, "update_steps": 3980})
        self._jsonl("metrics.jsonl", [{"agent_steps": 20000, "env_steps": 40000, "update_steps": 4000, "model_loss": 2.5}])
        self._jsonl("evaluations.jsonl", [{"agent_steps": 20000, "env_steps": 40000, "actual_eval_env_steps": 1000, "episode_return": 40}])
        path, digest = self._checkpoint(20000)
        (self.run / "REPLAY").mkdir()
        (self.run / "REPLAY" / "episode_00000000.npz").write_bytes(b"data")
        before = {item.relative_to(self.run).as_posix(): item.read_bytes() for item in self.run.rglob("*") if item.is_file()}
        report = self._inspect()
        after = {item.relative_to(self.run).as_posix(): item.read_bytes() for item in self.run.rglob("*") if item.is_file()}
        self.assertEqual(before, after)
        self.assertTrue(report["process"]["is_expected_training_process"])
        self.assertEqual(report["max_recorded_counters"], {"agent_steps": 20000, "env_steps": 40000, "update_steps": 4000})
        self.assertEqual(report["budget_fraction"], 0.02)
        self.assertEqual(report["units"]["action_repeat"], 2)
        self.assertIn("action decisions", report["units"]["agent_steps"])
        newest = report["checkpoints"]["newest_three"][0]
        self.assertEqual(newest["actual_sha256"], digest)
        self.assertTrue(newest["hash_verified_now"])
        self.assertEqual(report["checkpoints"]["newest_sha256_checked"], path.name)
        self.assertEqual(report["replay"]["archive_disk_bytes"], 4)
        self.assertEqual(report["replay"]["archive_file_count"], 1)
        self.assertEqual(report["alerts"], [])
        json.dumps(report, allow_nan=False)

    def test_nonfinite_and_malformed_lines_are_reported_but_output_is_valid_json(self):
        self._jsonl("metrics.jsonl", [
            {"agent_steps": 1, "env_steps": 2, "model_loss": float("nan")},
            {"agent_steps": 2, "env_steps": 4, "nested": {"gradient": float("inf")}},
        ])
        with (self.run / "metrics.jsonl").open("a") as stream:
            stream.write("{partial\n")
        report = self._inspect()
        self.assertEqual(report["jsonl_scan"]["nonfinite_values"], 2)
        self.assertEqual(report["jsonl_scan"]["malformed_rows"], 1)
        self.assertIn("nonfinite_values_recorded_in_jsonl", report["alerts"])
        self.assertEqual(report["max_recorded_counters"]["agent_steps"], 2)
        json.dumps(report, allow_nan=False)

    def test_pid_matches_exact_module_and_exact_resolved_run_path(self):
        self._cmdline(["python", "-m", "helixwm.train_debug", "--run-dir", str(self.run)])
        wrong_module = inspect_pid(321, self.run, self.proc)
        self.assertFalse(wrong_module["is_expected_training_process"])
        self.assertNotIn("argv", wrong_module)
        self._cmdline(["python", "-m", "helixwm.train", "--run-dir", str(self.run) + "-different"])
        wrong_path = self._inspect()
        self.assertFalse(wrong_path["process"]["matches_run_dir"])
        self.assertIn("train_pid_points_to_a_different_process", wrong_path["alerts"])
        self._cmdline(["python", "-m", "helixwm.train", "--run-dir=" + str(self.run.resolve())])
        self.assertTrue(inspect_pid(321, self.run, self.proc)["is_expected_training_process"])

    def test_completed_run_does_not_claim_failure_for_missing_live_pid(self):
        self._json("completion.json", {"state": "training_and_frozen_evaluation_complete", "budget_complete": True,
                                       "agent_steps": 1000000, "env_steps": 2000000, "update_steps": 199000})
        (self.run / "train.pid").write_text("999\n")
        (self.run / "train.exit").write_text("0\n")
        report = self._inspect()
        self.assertTrue(report["completion_marker_verified"])
        self.assertFalse(report["final_acceptance_verified"])
        self.assertFalse(any("expected_training_process_not_verified" in alert for alert in report["alerts"]))
        self.assertFalse(any("checkpoint_missing_or_stale" in alert for alert in report["alerts"]))

    def test_cumulative_oom_is_not_called_a_new_failure(self):
        (self.cgroup / "memory.events").write_text("oom 2\noom_kill 1\n")
        report = self._inspect()
        self.assertIsNone(report["cgroup"]["oom_delta"])
        self.assertEqual(report["cgroup"]["memory.events"]["oom_kill"], 1)
        self.assertTrue(any("cumulative" in alert and "unknown" in alert for alert in report["alerts"]))

    def test_missing_pairs_corrupt_latest_and_three_checkpoint_window(self):
        for step in (10000, 20000, 30000, 40000):
            path, _ = self._checkpoint(step, payload=str(step).encode())
        path.write_bytes(b"xxxxx")
        report = self._inspect()
        self.assertEqual(report["checkpoints"]["count"], 4)
        self.assertEqual(len(report["checkpoints"]["newest_three"]), 3)
        self.assertFalse(report["checkpoints"]["newest_three"][0]["hash_verified_now"])
        self.assertTrue(any("newest_checkpoint_not_sha256_verified" in alert for alert in report["alerts"]))
        path.unlink()
        report = self._inspect()
        self.assertFalse(report["checkpoints"]["newest_three"][0]["exists"])

    def test_stale_checkpoint_and_scan_limit_are_explicit(self):
        self._checkpoint(10000)
        self._json("status.json", {"agent_steps": 30000, "env_steps": 60000, "update_steps": 6000})
        self._jsonl("metrics.jsonl", [{"agent_steps": 30000, "env_steps": 60000}])
        report = self._inspect(max_jsonl_bytes=5)
        self.assertFalse(report["jsonl_scan"]["complete"])
        self.assertTrue(any("jsonl_scan_incomplete" in alert for alert in report["alerts"]))
        self.assertTrue(any("checkpoint_missing_or_stale" in alert for alert in report["alerts"]))


if __name__ == "__main__":
    unittest.main()
