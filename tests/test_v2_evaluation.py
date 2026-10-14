"""Evaluation causality, paired conditions, no-update gates, and artifacts."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from helixwm_v2.evaluation import DropoutSchedule, evaluate, plot_learning_curve
from helixwm_v2.dual_chain import DualChain


class FakeEnv:
    def __init__(self, seed=0, action_repeat=2, image_size=64,
                 episode_env_steps=6, camera_id=0):
        self.seed, self.action_repeat = seed, action_repeat
        self.image_size, self.episode_env_steps = image_size, episode_env_steps
        self.step_count = 0
        self.closed = False

    def reset(self):
        self.step_count = 0
        return np.full(64 * 64 * 3, 64, np.uint8)

    def step(self, action):
        self.step_count += 1
        return {"observation": np.full(64 * 64 * 3, 64 + self.step_count, np.uint8),
                "reward": float(1 + np.mean(action)), "env_steps": 2,
                "done": self.step_count >= self.episode_env_steps // 2,
                "continuation": 1.0}

    def render(self, size=256):
        return np.full((size, size, 3), 64 + self.step_count, np.uint8)

    def close(self):
        self.closed = True


class FakeBackbone:
    action_dim = 6
    score_kind = "negative_fake_pixel_mse_energy"

    def __init__(self):
        self.params = {"model": 7., "optimizer": 9.}
        self.inference_params = {"model": 5.}
        self.inference_version = 3
        self.n_updates, self.n_batches, self.n_actions = 7, 7, 0
        self.observed_images = []
        self.mutable = False

    def parameter_hash(self, inference=False):
        payload = self.inference_params if inference else self.params
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def pin_inference(self):
        self.inference_params = {"model": self.params["model"]}
        self.inference_version = self.n_updates

    def initial(self, batch=1):
        return {"deter": np.zeros((batch, 1)), "stoch": np.zeros((batch, 1))}

    def observe(self, state, action, image, is_first=False, rng=None):
        mean = float(np.asarray(image).mean())
        self.observed_images.append(mean)
        return {"deter": state["deter"] + 1,
                "stoch": np.full((1, 1), mean / 255) + rng.normal(0, .001, (1, 1))}

    def prior(self, state, action, rng=None):
        return {"deter": state["deter"] + 1,
                "stoch": state["stoch"] + rng.normal(0, .1, state["stoch"].shape)}

    def score(self, states, image):
        return -np.square(states["stoch"][:, 0] - np.asarray(image).mean() / 255)

    def act(self, state, deterministic=False, rng=None):
        if self.mutable:
            self.n_updates += 1
        return np.tanh(np.repeat(state["stoch"], 6, axis=1) + rng.normal(0, .01, (1, 6)))


CONFIG = {"seed": 0, "environment": {"action_repeat": 2, "image_size": 64,
           "episode_env_steps": 6, "camera_id": 0},
          "observation_dropout": {"start_probability": 1., "min_length": 1, "max_length": 1},
          "dual_chain": {"max_candidates": 8, "max_missing": 5, "score_temperature": .01},
          "eval_seed_start": 10000, "video_fps": 20}


class EvaluationTests(unittest.TestCase):
    def test_missing_schedule_fixed_seed_endpoints_and_recovery(self):
        cfg = {"start_probability": 1., "min_length": 3, "max_length": 3}
        schedule = DropoutSchedule(4, cfg)
        self.assertTrue(schedule.valid(is_first=True))
        self.assertFalse(schedule.valid())
        clone = DropoutSchedule(999, cfg)
        clone.load_state_dict(schedule.state_dict())
        self.assertEqual([schedule.valid() for _ in range(8)], [clone.valid() for _ in range(8)])
        self.assertTrue(schedule.valid(is_last=True))
        self.assertEqual(schedule.remaining, 0)
        self.assertTrue(schedule.valid(is_first=True))
        self.assertEqual(schedule.remaining, 0)

    def test_schedule_never_uses_global_numpy_rng(self):
        np.random.seed(124)
        before = np.random.get_state()
        schedule = DropoutSchedule(8, CONFIG)
        for _ in range(20):
            schedule.valid()
        after = np.random.get_state()
        np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])

    @mock.patch("helixwm_v2.evaluation.WalkerEnv", FakeEnv)
    @mock.patch("helixwm_v2.evaluation.plot_learning_curve", return_value={"available": False, "test": True})
    def test_paired_conditions_freeze_and_restore_inference(self, _plot):
        backbone = FakeBackbone()
        before = (copy.deepcopy(backbone.inference_params), backbone.inference_version,
                  backbone.parameter_hash(), backbone.n_updates)
        with tempfile.TemporaryDirectory() as directory:
            report = evaluate(backbone, CONFIG, directory, 1000000, kind="final",
                              episodes=2, audit_disabled=True)
            rows = [json.loads(line) for line in (Path(directory) / "RESULTS/eval.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), 8)
            self.assertEqual(set(report["conditions"]),
                             {"clean", "dropout", "dropout_disabled", "dropout_prior_only"})
            self.assertTrue(report["hash_gate"]["passed"])
            self.assertEqual(report["training_env_steps"], 2000000)
            self.assertEqual(report["conditions"]["clean"]["missing_observations"], 0)
            for seed in (10000, 10001):
                masked = [row for row in rows if row["evaluation_seed"] == seed and row["condition"] != "clean"]
                self.assertEqual(len(masked), 3)
                self.assertEqual(len({row["availability_sha256"] for row in masked}), 1)
                for row in masked:
                    self.assertEqual(row["dropout_seed"], 100000 + seed)
                    self.assertEqual(row["policy_seed"], 200000 + seed)
                    self.assertEqual(row["actual_eval_env_steps"], 6)
                    self.assertEqual(row["missing_observations"], 2)
            self.assertTrue((Path(directory) / "RESULTS/eval.csv").exists())
            self.assertTrue(Path(report["report_path"]).exists())
        self.assertEqual(before, (backbone.inference_params, backbone.inference_version,
                                 backbone.parameter_hash(), backbone.n_updates))
        self.assertIn(0., backbone.observed_images)  # Disabled repair explicitly encodes black frames.

    @mock.patch("helixwm_v2.evaluation.WalkerEnv", FakeEnv)
    @mock.patch("helixwm_v2.evaluation.plot_learning_curve", return_value={"available": False})
    def test_prior_only_skips_missing_encoder_without_candidate_repair(self, _plot):
        calls = []

        class RecordingChain(DualChain):
            def step(self, observation, action, **kwargs):
                if "/dropout_prior_only/" in kwargs["episode_id"]:
                    calls.append({"mode": self.config["mode"], "valid": kwargs["valid"],
                                  "observation_absent": observation is None})
                return super().step(observation, action, **kwargs)

        with tempfile.TemporaryDirectory() as directory:
            with mock.patch("helixwm_v2.evaluation.DualChain", RecordingChain):
                report = evaluate(FakeBackbone(), CONFIG, directory, 1000000,
                                  kind="final", episodes=1, audit_disabled=True)
        self.assertEqual(calls, [{"mode": "disabled", "valid": False, "observation_absent": True},
                                {"mode": "disabled", "valid": False, "observation_absent": True},
                                {"mode": "disabled", "valid": True, "observation_absent": False}])
        prior = report["conditions"]["dropout_prior_only"]
        self.assertEqual(prior["repair_count"], 0)
        self.assertEqual(prior["abstention_count"], 0)
        self.assertEqual(prior["missing_observations"], 2)
        self.assertTrue(report["hash_gate"]["passed"])

    @mock.patch("helixwm_v2.evaluation.WalkerEnv", FakeEnv)
    @mock.patch("helixwm_v2.evaluation.plot_learning_curve", return_value={"available": False})
    def test_repair_condition_never_encodes_missing_image(self, _plot):
        backbone = FakeBackbone()
        with tempfile.TemporaryDirectory() as directory:
            report = evaluate(backbone, CONFIG, directory, 20000, episodes=1)
        self.assertNotIn(0., backbone.observed_images)
        self.assertEqual(report["conditions"]["dropout"]["repair_count"], 1)

    @mock.patch("helixwm_v2.evaluation.WalkerEnv", FakeEnv)
    @mock.patch("helixwm_v2.evaluation.plot_learning_curve", return_value={"available": False})
    def test_counter_mutation_fails_and_restores_collector_snapshot(self, _plot):
        backbone = FakeBackbone()
        backbone.mutable = True
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(RuntimeError):
                evaluate(backbone, CONFIG, directory, 20000, episodes=1)
            saved = next((Path(directory) / "RESULTS/evaluation_reports").glob("*.json"))
            report = json.loads(saved.read_text())
            self.assertFalse(report["hash_gate"]["passed"])
            self.assertEqual(report["status"], "failed")
        self.assertEqual(backbone.inference_version, 3)
        self.assertEqual(backbone.inference_params, {"model": 5.})

    @mock.patch("helixwm_v2.evaluation.WalkerEnv", FakeEnv)
    @mock.patch("helixwm_v2.evaluation.plot_learning_curve", return_value={"available": False})
    def test_repeated_evaluation_raw_episode_returns_reproduce(self, _plot):
        backbone = FakeBackbone()
        with tempfile.TemporaryDirectory() as directory:
            first = evaluate(backbone, CONFIG, directory, 20000, episodes=2)
            second = evaluate(backbone, CONFIG, directory, 20000, episodes=2)
            self.assertNotEqual(first["evaluation_id"], second["evaluation_id"])
            for condition in ("clean", "dropout"):
                self.assertEqual(first["conditions"][condition]["raw_episode_returns"],
                                 second["conditions"][condition]["raw_episode_returns"])
            self.assertEqual(len(list((Path(directory) / "RESULTS/evaluation_reports").glob("*.json"))), 2)

    @mock.patch("helixwm_v2.evaluation.WalkerEnv", FakeEnv)
    def test_plot_has_real_png_and_pdf(self):
        with tempfile.TemporaryDirectory() as directory:
            report = evaluate(FakeBackbone(), CONFIG, directory, 20000, episodes=1)
            png = Path(report["learning_curve"]["png"])
            pdf = Path(report["learning_curve"]["pdf"])
            self.assertEqual(png.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            self.assertTrue(pdf.read_bytes().startswith(b"%PDF"))

    @mock.patch("helixwm_v2.evaluation.WalkerEnv", FakeEnv)
    @mock.patch("helixwm_v2.evaluation.plot_learning_curve", return_value={"available": False})
    def test_single_formal_video_is_fully_decoded(self, _plot):
        with tempfile.TemporaryDirectory() as directory:
            report = evaluate(FakeBackbone(), CONFIG, directory, 1000000, kind="final", episodes=2, video=True)
            self.assertTrue(report["video"]["full_decode_pass"])
            self.assertEqual(report["video"]["frames"], 4)
            self.assertEqual(len(list(Path(directory).rglob("*.mp4"))), 1)
            with self.assertRaises(FileExistsError):
                evaluate(FakeBackbone(), CONFIG, directory, 1000000, kind="final", episodes=1, video=True)


if __name__ == "__main__":
    unittest.main()
