"""Mock-engine tests; never launch Minecraft or initialize a trained model."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from helixwm_minecraft.evaluation import evaluate, final_case_specs, _episode_diagnostics


ROOT = Path(__file__).resolve().parents[1]
CONFIG = {"seed": 0, "environment": {"worker_python": "unused-mock", "action_repeat": 2,
          "image_size": 64, "episode_env_steps": 1000}, "periodic_eval_episodes": 1,
          "final_eval_episodes": 20, "eval_seed_start": 10000, "video_fps": 20,
          "observation_dropout": {"start_probability": 1., "min_length": 1, "max_length": 1},
          "dual_chain": {"max_candidates": 8, "max_missing": 5, "score_temperature": .01}}


class Backend:
    action_dim = 6

    def __init__(self):
        self.agent = SimpleNamespace(n_updates=9, n_batches=9, n_actions=0)
        self.inference_params = {"value": 3}
        self.inference_version = 3
        self.observe_calls = 0

    def pin_inference(self):
        self.inference_params = {"value": 9}
        self.inference_version = 9

    def parameter_hash(self, inference=False):
        return hashlib.sha256(repr(self.inference_params if inference else {"weight": 9}).encode()).hexdigest()

    def initial(self, batch=1):
        return {"deter": np.zeros((batch, 1)), "stoch": np.zeros((batch, 1))}

    def observe(self, state, action, image, is_first=False, rng=None):
        self.observe_calls += 1
        return {"deter": state["deter"] + .1, "stoch": rng.normal(size=(1, 1)) * .01}

    def prior(self, state, action, rng=None):
        return {"deter": state["deter"] + .1, "stoch": state["stoch"] + rng.normal(size=(1, 1)) * .01}

    def score(self, states, image):
        return -np.square(states["stoch"][:, 0])

    def act(self, state, deterministic=False, rng=None):
        return rng.normal(size=(1, 6)) * .01


class Engine:
    live = 0
    maximum_live = 0

    def __init__(self, seed, **kwargs):
        self.seed = seed
        self.t = 0
        Engine.live += 1
        Engine.maximum_live = max(Engine.maximum_live, Engine.live)

    def reset(self, seed):
        if seed != self.seed:
            raise ValueError("seed was not passed explicitly")
        return np.zeros(12288, np.uint8)

    def step(self, action):
        self.t += 1
        return {"observation": np.full(12288, self.t, np.uint8), "reward": .25,
                "continuation": 0. if self.t == 3 else 1., "done": self.t == 3,
                "terminated": self.t == 3, "truncated": False,
                "env_steps": 1 if self.t == 3 else 2}

    def render(self, size=256):
        return np.zeros((size, size, 3), np.uint8)

    def close(self):
        Engine.live -= 1


class MutatingBackend(Backend):
    def act(self, *args, **kwargs):
        self.agent.n_actions += 1
        return super().act(*args, **kwargs)


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        Engine.live = Engine.maximum_live = 0
        self.plotter = mock.patch("helixwm_minecraft.evaluation.plot_learning_curve", return_value={"available": True})
        self.plotter.start()
        self.addCleanup(self.plotter.stop)
        self.comparison_plotter = mock.patch("helixwm_minecraft.evaluation.plot_final_comparison", return_value={"available": True})
        self.comparison_plotter.start()
        self.addCleanup(self.comparison_plotter.stop)

    def test_early_terminal_actual_ticks_private_rng_and_pinned_restore(self):
        backend = Backend()
        original_params = backend.inference_params
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            report = evaluate(backend, CONFIG, tmp, 100, kind="smoke", episodes=1, video=False,
                              env_factory=Engine, training_env_steps=193)
            self.assertEqual(report["status"], "complete")
            self.assertEqual(report["actual_eval_agent_steps"], 6)
            self.assertEqual(report["actual_eval_env_steps"], 10)
            self.assertEqual(report["training_env_steps"], 193)
            self.assertTrue(report["hash_gate"]["passed"])
            self.assertTrue(report["post_artifact_frozen_gate"]["passed"])
            self.assertTrue(report["collector_inference_restored"])
            self.assertIs(backend.inference_params, original_params)
            self.assertEqual(backend.inference_version, 3)
            self.assertEqual(Engine.live, 0)
            self.assertEqual(Engine.maximum_live, 1)
            self.assertEqual(backend.observe_calls, 5)  # four clean + initial dropout
            self.assertFalse((Path(tmp) / "VIDEOS").exists())
            self.assertTrue((Path(tmp) / "EVAL/episodes.csv").is_file())
            for path in (Path(tmp) / "EVAL/traces").glob("*/trace.npz"):
                with np.load(path) as arrays:
                    np.testing.assert_array_equal(arrays["env_steps"], [2, 2, 1])
                    self.assertEqual(arrays["full_availability"].shape, (501,))

    def test_final_rejects_small_seed_count_or_missing_tick_counter(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            with self.assertRaises(ValueError):
                evaluate(Backend(), CONFIG, tmp, 1000000, kind="final", episodes=1,
                         env_factory=Engine, training_env_steps=1999000)
            with self.assertRaises(ValueError):
                evaluate(Backend(), CONFIG, tmp, 1000000, kind="final", env_factory=Engine)

    def test_final_single_video_uses_actual_early_ending_frame_count(self):
        calls = []
        class Writer:
            def __init__(self, path):
                self.path, self.frames = Path(path), 0
                self.path.write_bytes(b"mock video")
            def append_data(self, frame):
                self.frames += 1
            def close(self):
                calls.append(self.frames)
        def decode(path, count, fps):
            return {"path": str(path), "frames": count, "fps": fps, "full_decode_pass": True}
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp, \
                mock.patch("imageio.v2.get_writer", side_effect=lambda path, **kw: Writer(path)), \
                mock.patch("helixwm_minecraft.evaluation._decode_video", side_effect=decode), \
                mock.patch("helixwm_minecraft.evaluation.plot_learning_curve", return_value={"available": True}):
            report = evaluate(Backend(), CONFIG, tmp, 1000000, kind="final", env_factory=Engine,
                              training_env_steps=1999000)
            self.assertEqual(report["conditions"]["clean"]["episodes"], 20)
            self.assertEqual(report["conditions"]["v2_exact"]["episodes"], 100)
            self.assertEqual(sum(x["episodes"] for x in report["conditions"].values()), 520)
            self.assertEqual(calls, [4])
            self.assertEqual(report["video"]["frames_expected"], 4)
            self.assertEqual(report["video"]["path"], "VIDEOS/v2_exact_seed10000_gap20_final.mp4")
            self.assertFalse(report["video"]["fault_exposed"])
            self.assertEqual(report["conditions"]["v2_exact"]["by_gap_length"]["20"]["early_ending_before_fault_episodes"], 20)
            self.assertEqual(len(list(Path(tmp).rglob("*.mp4"))), 1)

    def test_mutated_model_counter_fails_gate_and_preserves_raw_data(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            with self.assertRaisesRegex(RuntimeError, "changed frozen state"):
                evaluate(MutatingBackend(), CONFIG, tmp, 10, kind="smoke", episodes=1, video=False,
                         env_factory=Engine, training_env_steps=20)
            report_path = next((Path(tmp) / "EVAL/reports").glob("*.json"))
            report = json.loads(report_path.read_text())
            self.assertEqual(report["status"], "failed")
            self.assertFalse(report["hash_gate"]["passed"])
            self.assertTrue((Path(tmp) / "EVAL/episodes.jsonl").is_file())
            self.assertEqual(Engine.live, 0)

    def test_engine_failure_preserves_prefix_and_marks_unknown_final_tick(self):
        class FailingEngine(Engine):
            def step(self, action):
                if self.t == 1:
                    raise TimeoutError("Worker result unavailable")
                return super().step(action)
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            with self.assertRaisesRegex(RuntimeError, "Worker result unavailable"):
                evaluate(Backend(), CONFIG, tmp, 100, kind="smoke", episodes=1, video=False,
                         env_factory=FailingEngine, training_env_steps=193)
            row = json.loads((Path(tmp) / "EVAL/episodes.jsonl").read_text().splitlines()[-1])
            self.assertEqual(row["status"], "technical_failure")
            self.assertEqual(row["actual_eval_agent_steps"], 1)
            self.assertEqual(row["actual_eval_env_steps"], 2)
            self.assertEqual(row["valid_reward_rows"], 1)
            self.assertFalse(row["actual_interaction_counters_complete"])
            self.assertEqual(Engine.live, 0)

    def test_final_matrix_exact_and_no_historical_condition_selection(self):
        cases = final_case_specs(CONFIG)
        self.assertEqual(len(cases), 520)
        self.assertEqual(len({(c["arm"], c["seed"], c["gap_length"]) for c in cases}), 520)
        self.assertEqual(sum(c["arm"] == "clean" for c in cases), 20)
        self.assertEqual({c["gap_start"] for c in cases if c["arm"] != "clean"}, {10})
        for arm in {c["arm"] for c in cases} - {"clean"}:
            self.assertEqual({c["gap_length"] for c in cases if c["arm"] == arm}, {1, 5, 10, 20, 50})

    def test_smoke_video_is_separate_from_formal_artifacts(self):
        frames = []
        class Writer:
            def __init__(self, path):
                Path(path).write_bytes(b"mock smoke video")
            def append_data(self, frame):
                frames.append(frame.shape)
            def close(self):
                pass
        def decode(path, count, fps):
            return {"frames": count, "fps": fps, "full_decode_pass": True}
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp, \
                mock.patch("imageio.v2.get_writer", side_effect=lambda path, **kw: Writer(path)), \
                mock.patch("helixwm_minecraft.evaluation._decode_video", side_effect=decode):
            report = evaluate(Backend(), CONFIG, tmp, 100, kind="smoke", episodes=1,
                              env_factory=Engine, training_env_steps=193)
            self.assertEqual(len(frames), 4)
            self.assertTrue(report["video"]["path"].startswith("SMOKE_ARTIFACTS/"))
            self.assertFalse((Path(tmp) / "VIDEOS").exists())
            self.assertFalse((Path(tmp) / "PLOTS").exists())
            self.assertEqual(report["artifact_scope"], "SMOKE_ONLY_EXCLUDE_FROM_FORMAL_PACKAGE")

    def test_video_decode_failure_retains_episode_and_report(self):
        class Writer:
            def __init__(self, path):
                Path(path).write_bytes(b"invalid mocked video")
            def append_data(self, frame):
                pass
            def close(self):
                pass
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp, \
                mock.patch("imageio.v2.get_writer", side_effect=lambda path, **kw: Writer(path)), \
                mock.patch("helixwm_minecraft.evaluation._decode_video", side_effect=ValueError("decode failed")):
            with self.assertRaisesRegex(RuntimeError, "decode failed"):
                evaluate(Backend(), CONFIG, tmp, 1000000, kind="final", env_factory=Engine,
                         training_env_steps=1999000)
            row = json.loads((Path(tmp) / "EVAL/episodes.jsonl").read_text().splitlines()[-1])
            self.assertEqual(row["status"], "technical_failure")
            self.assertEqual(row["valid_reward_rows"], 3)
            self.assertEqual(row["error"]["operation"], "video_full_decode")
            self.assertTrue((Path(tmp) / row["trace"]).is_file())
            self.assertEqual(Engine.live, 0)

    def test_privileged_outcome_sidecar_is_post_episode_and_seed_tick_bound(self):
        rows = [{"event": "reset", "episode_seed": 10000, "episode_ticks": 0,
                 "privileged": {"position": [0, 4, 0], "goal_distance": 12., "path_length": 0., "success": False}},
                {"event": "native_step", "episode_seed": 10000, "episode_ticks": 1, "reward": 100.5,
                 "privileged": {"position": [0, 4, .5], "goal_distance": 11.5, "path_length": .5,
                                "success": True, "native_reward": 100., "progress_reward": .5}}]
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            path = Path(tmp) / "native.jsonl"
            path.write_text("\n".join(json.dumps(row) for row in rows))
            output = _episode_diagnostics(path, 10000, 1, 100.5)
            self.assertTrue(output["success"])
            self.assertFalse(output["policy_access"])
            self.assertTrue(output["consumed_only_after_episode_end"])
            with self.assertRaisesRegex(ValueError, "ticks/seed"):
                _episode_diagnostics(path, 10001, 1, 100.5)
            with self.assertRaisesRegex(ValueError, "reward"):
                _episode_diagnostics(path, 10000, 1, 0.)


if __name__ == "__main__":
    unittest.main()
