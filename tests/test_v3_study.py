"""Transactional runner tests with synthetic data, never a trained model."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from helixwm_v3.study import (ROOT, availability, case_plan, execute_case, finalize_case,
                              verify_case, RunLock, validate_config, canonical_hash)


class FakeBackbone:
    action_dim = 6
    inference_version = 248750

    def __init__(self):
        self.agent = SimpleNamespace(n_updates=248750, n_batches=248750, n_actions=0)

    def initial(self, batch=1):
        return {"deter": np.zeros((batch, 1)), "stoch": np.zeros((batch, 1))}

    def observe(self, state, action, image, is_first=False, rng=None):
        return {"deter": state["deter"] + .01,
                "stoch": np.full((1, 1), np.asarray(image).mean() / 255) + rng.normal(size=(1, 1)) * .001}

    def prior(self, state, action, rng=None):
        return {"deter": state["deter"] + .01,
                "stoch": state["stoch"] + rng.normal(size=(1, 1)) * .001}

    def score(self, states, image):
        return -np.square(states["stoch"][:, 0] - np.asarray(image).mean() / 255)

    def act(self, state, deterministic=False, rng=None):
        return rng.normal(size=(1, 6)) * .01

    def parameter_hash(self, inference=False):
        return "b" * 64 if inference else "a" * 64


class FakeEnv:
    fail_at = None

    def __init__(self, **kwargs):
        self.t = 0
        self._env = SimpleNamespace(physics=self)

    def reset(self):
        self.t = 0
        return np.zeros((64, 64, 3), np.uint8)

    def step(self, action):
        self.t += 1
        return {"observation": np.full((64, 64, 3), self.t % 255, np.uint8),
                "reward": 1.8, "env_steps": 2, "done": self.t == (self.fail_at or 500)}

    def torso_height(self):
        return 1.2

    def torso_upright(self):
        return 1.

    def snapshot(self):
        return {"time": self.t}

    def close(self):
        pass


class EarlyEnd(FakeEnv):
    fail_at = 3


class MutatingBackbone(FakeBackbone):
    def act(self, *args, **kwargs):
        self.agent.n_actions += 1
        return super().act(*args, **kwargs)


class StudyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads((ROOT / "configs/continuity_study_v3.json").read_text())

    def test_design_only_cannot_execute(self):
        with self.assertRaises(ValueError):
            validate_config(self.config)
        config = copy.deepcopy(self.config)
        config.update(execution_authorized=True, runner_implemented=True)
        validate_config(config)

    def test_case_counts_unique_and_smoke_disjoint(self):
        plans = {stage: case_plan(self.config, stage) for stage in ("smoke", "design", "held_out")}
        self.assertEqual([len(plans[x]) for x in plans], [12, 520, 1040])
        ids = [x["case_id"] for values in plans.values() for x in values]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertTrue(all(x["environment_seed"] == 40000 for x in plans["smoke"]))

    def test_fixed_gap_indexing_and_exact_contiguous_lengths(self):
        for length in (0, 1, 5, 10, 20, 50):
            mask = availability(length)
            self.assertEqual(len(mask), 501)
            np.testing.assert_array_equal(np.where(~mask)[0], np.arange(150, 150 + length))
            self.assertTrue(mask[0] and mask[-1])

    def test_case_round_trip_manifest_and_mask_action_alignment(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            case = case_plan(self.config, "smoke")[3]
            path = Path(tmp) / "attempt"
            summary = execute_case(FakeBackbone(), self.config, case, path, FakeEnv, store_images=False)
            self.assertEqual(summary["status"], "complete")
            self.assertEqual(summary["actual_eval_agent_steps"], 500)
            self.assertTrue(summary["hash_gate"]["passed"])
            with np.load(path / "trace.npz") as data:
                self.assertTrue(data["input_available"][149])
                self.assertFalse(data["returned_available"][149])
                self.assertFalse(data["input_available"][150])
                self.assertEqual(len(data["frame_sha256"]), 501)
                self.assertEqual(data["policy_received_frame_sha256"][150], "UNAVAILABLE")
            final = Path(tmp) / case["case_id"]
            identity = canonical_hash({"test": True})
            record = finalize_case(path, final, identity)
            self.assertEqual(record["case_id"], case["case_id"])
            self.assertEqual(verify_case(final, identity), record)
            (final / "events.jsonl").write_text("changed")
            with self.assertRaises(ValueError):
                verify_case(final, identity)

    def test_technical_failure_preserves_real_rows_and_is_not_retried(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            case = case_plan(self.config, "smoke")[0]
            path = Path(tmp) / "attempt"
            summary = execute_case(FakeBackbone(), self.config, case, path, EarlyEnd, store_images=False)
            self.assertEqual(summary["status"], "technical_failure")
            self.assertEqual(summary["actual_eval_agent_steps"], 3)
            self.assertTrue(summary["hash_gate"]["passed"])
            with np.load(path / "trace.npz") as data:
                self.assertEqual(len(data["reward"]), 3)
                np.testing.assert_array_equal(data["reward"], np.full(3, 1.8))
            record = finalize_case(path, Path(tmp) / case["case_id"], "id")
            self.assertEqual(record["status"], "technical_failure")

    def test_case_restart_from_seed_preserves_raw_state_action_trajectory(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            case = case_plan(self.config, "smoke")[4]
            paths = [Path(tmp) / str(i) for i in range(2)]
            for p in paths:
                execute_case(FakeBackbone(), self.config, case, p, FakeEnv, store_images=False)
            with np.load(paths[0] / "trace.npz") as a, np.load(paths[1] / "trace.npz") as b:
                for key in ("actions", "frame_sha256", "reward", "state_sha256", "policy_received_frame_sha256"):
                    np.testing.assert_array_equal(a[key], b[key])

    def test_output_lock_rejects_concurrent_resume(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            path = Path(tmp) / "lock"
            with RunLock(path):
                with self.assertRaises(RuntimeError):
                    with RunLock(path):
                        pass
            with RunLock(path):
                pass

    def test_mutated_frozen_counter_is_not_accepted(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            case = case_plan(self.config, "smoke")[0]
            path = Path(tmp) / "attempt"
            with self.assertRaisesRegex(RuntimeError, "Frozen model"):
                execute_case(MutatingBackbone(), self.config, case, path, EarlyEnd, store_images=False)
            self.assertFalse(json.loads((path / "summary.json").read_text())["hash_gate"]["passed"])


if __name__ == "__main__":
    unittest.main()
