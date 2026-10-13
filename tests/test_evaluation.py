"""Frozen evaluator contracts using a deterministic lightweight environment."""

import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from helixwm.control import HelixLearner
from helixwm.evaluation import evaluate, state_sha256
from helixwm.rssm import RSSM


class FakeEnv:
    instances = []
    mutation = None

    def __init__(self, seed=0, action_repeat=2, image_size=64, episode_env_steps=3, camera_id=0):
        self.seed = seed
        self.action_dim = 2
        self.calls = 0
        self.closed = False
        self.actions = []
        self._env = self
        self.instances.append(self)

    def control_timestep(self):
        return .025

    @staticmethod
    def consume_rng():
        random.random()
        np.random.random()
        torch.rand(1)

    def reset(self):
        self.consume_rng()
        return np.full(5, self.seed % 256, dtype=np.uint8)

    def step(self, action):
        self.consume_rng()
        self.actions.append(np.asarray(action).copy())
        self.calls += 1
        if self.mutation is not None:
            self.mutation()
        return {"observation": np.full(5, (self.seed + self.calls) % 256, dtype=np.uint8),
                "reward": float(self.seed - 99), "continuation": 1.0,
                "done": self.calls == 2, "env_steps": 2 if self.calls == 1 else 1}

    def render(self, size=256):
        return np.full((size, size, 3), self.seed % 256, dtype=np.uint8)

    def close(self):
        self.closed = True


class FakeWriter:
    def __init__(self, path, fps):
        self.path, self.fps, self.frames, self.closed = Path(path), fps, [], False

    def append_data(self, frame):
        self.frames.append(np.asarray(frame).copy())

    def close(self):
        self.path.write_bytes(b"unit-test-writer-payload-not-a-real-video")
        self.closed = True


class EvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        random.seed(123)
        np.random.seed(123)
        torch.manual_seed(123)
        self.learner = HelixLearner(RSSM(5, 2, hidden_dim=8, groups=2, classes=3, embed_dim=8),
                                    {"actor_hidden": 16, "critic_hidden": 16, "reliability_hidden": 8})
        # Populate Adam moments and step counters so frozen-state checks cover
        # real optimizer tensors, not just five empty state dictionaries.
        for optimizer in self.learner.optimizers.values():
            parameters = [p for group in optimizer.param_groups for p in group["params"]]
            loss = sum(parameter.square().sum() for parameter in parameters) * .001
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        self.learner.train()
        self.learner.risk_head.eval()
        self.config = {"seed": 0, "environment": {"action_repeat": 2, "image_size": 64,
                                                   "episode_env_steps": 3, "camera_id": 0}}
        FakeEnv.instances = []
        FakeEnv.mutation = None

    def rng_hash(self):
        return state_sha256({"python": random.getstate(), "numpy": np.random.get_state(),
                             "torch": torch.get_rng_state(),
                             "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None})

    def optimizer_hash(self):
        return state_sha256({name: opt.state_dict() for name, opt in self.learner.optimizers.items()})

    def test_raw_returns_units_and_all_frozen_state_are_preserved(self):
        before_model = state_sha256(self.learner.state_dict())
        before_optimizer, before_rng = self.optimizer_hash(), self.rng_hash()
        before_modes = [module.training for module in self.learner.modules()]
        with tempfile.TemporaryDirectory() as temporary, patch("helixwm.evaluation.WalkerEnv", FakeEnv):
            report = evaluate(self.learner, self.config, [100, 101], Path(temporary),
                              1000, 2000, 7, "periodic", "checkpoint_1000")
            rows = [json.loads(line) for line in (Path(temporary) / "evaluations.jsonl").read_text().splitlines()]
            self.assertEqual([row["episode_return"] for row in rows], [2.0, 4.0])
            self.assertEqual([row["seed"] for row in rows], [0, 0])
            self.assertEqual([row["eval_seed"] for row in rows], [100, 101])
            self.assertEqual([row["episode_length"] for row in rows], [2, 2])
            self.assertEqual([row["actual_eval_env_steps"] for row in rows], [3, 3])
            self.assertTrue(all(row["agent_steps"] == 1000 and row["env_steps"] == 2000 for row in rows))
            self.assertEqual(report["episode_return_mean"], 3.0)
            self.assertEqual(report["episode_return_std"], 1.0)
            self.assertEqual(report["actual_eval_env_steps"], 6)
            self.assertEqual(report["reward_prediction_count"], 4)
            self.assertEqual(report["module_hash_before"], report["module_hash_after"])
            self.assertEqual(report["optimizer_hash_before"], report["optimizer_hash_after"])
            self.assertTrue(report["zero_update"])
            self.assertIsNone(report["video"])
            self.assertTrue(Path(report["report_path"]).is_file())
        self.assertEqual(before_model, state_sha256(self.learner.state_dict()))
        self.assertEqual(before_optimizer, self.optimizer_hash())
        self.assertEqual(before_rng, self.rng_hash())
        self.assertEqual(before_modes, [module.training for module in self.learner.modules()])
        self.assertEqual([env.seed for env in FakeEnv.instances], [100, 101])
        self.assertTrue(all(env.closed for env in FakeEnv.instances))

    def test_deterministic_policy_repeats_and_reports_do_not_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary, patch("helixwm.evaluation.WalkerEnv", FakeEnv):
            first = evaluate(self.learner, self.config, [100], Path(temporary), 20, 40, 1, "frozen")
            actions = np.stack(FakeEnv.instances[-1].actions)
            second = evaluate(self.learner, self.config, [100], Path(temporary), 20, 40, 1, "frozen")
            np.testing.assert_array_equal(actions, np.stack(FakeEnv.instances[-1].actions))
            self.assertNotEqual(first["report_path"], second["report_path"])
            self.assertEqual(len((Path(temporary) / "evaluations.jsonl").read_text().splitlines()), 2)

    def test_video_streams_only_first_declared_seed_at_physical_20fps(self):
        writers = []

        def make_writer(path, fps):
            writer = FakeWriter(path, fps)
            writers.append(writer)
            return writer

        with tempfile.TemporaryDirectory() as temporary, patch("helixwm.evaluation.WalkerEnv", FakeEnv), \
                patch("helixwm.evaluation._open_video_writer", make_writer):
            report = evaluate(self.learner, self.config, [100, 101], Path(temporary),
                              100, 200, 2, "final_frozen", video=True)
            self.assertEqual(len(writers), 1)
            self.assertEqual(writers[0].fps, 20)
            self.assertEqual(report["video"]["control_timestep_seconds"], .025)
            self.assertEqual(len(writers[0].frames), 3)
            self.assertTrue(all(frame.shape == (256, 256, 3) for frame in writers[0].frames))
            self.assertTrue(all((frame == 100).all() for frame in writers[0].frames))
            self.assertTrue(writers[0].closed)
            self.assertEqual(report["video"]["eval_seed"], 100)
            self.assertTrue(Path(report["video"]["path"]).is_file())
            self.assertEqual(len(list((Path(temporary) / "VIDEOS").glob("*.mp4"))), 1)
            with self.assertRaises(FileExistsError):
                evaluate(self.learner, self.config, [100], Path(temporary),
                         100, 200, 2, "final_frozen", video=True)

    def test_mutation_is_rejected_and_rng_modes_and_environment_are_restored(self):
        before_rng = self.rng_hash()
        modes = [module.training for module in self.learner.modules()]
        parameter = next(self.learner.critic.parameters())
        FakeEnv.mutation = staticmethod(lambda: parameter.add_(1.0))
        try:
            with tempfile.TemporaryDirectory() as temporary, patch("helixwm.evaluation.WalkerEnv", FakeEnv):
                with self.assertRaisesRegex(RuntimeError, "mutated module or optimizer"):
                    evaluate(self.learner, self.config, [100], Path(temporary), 20, 40, 1, "frozen")
                self.assertFalse((Path(temporary) / "evaluations.jsonl").exists())
        finally:
            FakeEnv.mutation = None
        self.assertEqual(before_rng, self.rng_hash())
        self.assertEqual(modes, [module.training for module in self.learner.modules()])
        self.assertTrue(all(env.closed for env in FakeEnv.instances))

    def test_hash_detects_optimizer_counter_and_dtype_changes(self):
        first = {"state": {0: {"step": torch.tensor(1.0), "moment": torch.ones(2)}},
                 "groups": [{"lr": 1e-3}]}
        second = {"state": {0: {"step": torch.tensor(2.0), "moment": torch.ones(2)}},
                  "groups": [{"lr": 1e-3}]}
        self.assertNotEqual(state_sha256(first), state_sha256(second))
        self.assertNotEqual(state_sha256(torch.ones(2)), state_sha256(torch.ones(2, dtype=torch.float64)))


if __name__ == "__main__":
    unittest.main()
