from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from helixwm_minecraft.backbone import BACKBONE_FORMAT, TASK_NAMESPACE, MinecraftBackbone
from helixwm_minecraft.storage import (FORMAT, load_checkpoint, restore_chain_boundary,
                                     validate_payload, tree_hash, Replay, sha256)
from helixwm_minecraft.train import (run, validate_config, verify_launch_gate, config_sha256,
                                     episode_outcome, verify_evaluation_report)
from helixwm_minecraft.preflight import software_probe
from helixwm_v2.dual_chain import DualChain


def configuration():
    return dict(task_namespace=TASK_NAMESPACE, seed=0, budget_agent_steps=11,
                checkpoint_every=5, eval_every=6, prefill_agent_steps=2,
                train_ratio=1., replay_capacity_steps=100, replay_cache_episodes=2,
                backbone=dict(seed=0, batch_size=1, batch_length=2, burnin=0),
                environment=dict(action_repeat=2, image_size=64, episode_env_steps=10),
                dual_chain=dict(mode="repair", max_candidates=2, max_missing=5),
                observation_dropout=dict(start_probability=.4, min_length=1, max_length=2),
                final_eval_episodes=20, periodic_eval_episodes=5, eval_seed_start=10000,
                final_arms=["clean", "black_frame", "prior_only", "v2_exact", "locked_ancestor", "posterior_return_once"],
                final_gap_lengths=[1, 5, 10, 20, 50], final_gap_start=10,
                final_video=dict(arm="v2_exact", seed=10000, gap_length=20))


class FakeBackbone:
    def __init__(self, config=None, *args):
        self.resolved_config = deepcopy(config or {})
        self.inference_version = self.updates = 0
        self.inference_params = {"weight": np.zeros(1, np.float32)}

    def initial(self, batch=1):
        return {"x": np.zeros((batch, 2), np.float32)}

    def observe(self, previous, action, image, is_first=False, rng=None):
        base = np.zeros_like(previous["x"]) if is_first else previous["x"]
        return {"x": base + np.float32(np.asarray(image).mean()/255.) + rng.normal(0, .01, base.shape).astype(np.float32)}

    def prior(self, previous, action, rng=None):
        return {"x": previous["x"] + rng.normal(0, .02, previous["x"].shape).astype(np.float32)}

    def score(self, states, image):
        return -np.square(states["x"]).sum(-1)

    def act(self, state, deterministic=False, rng=None):
        return rng.normal(0, .3, (len(state["x"]), 6)).astype(np.float32)

    def pin_inference(self):
        self.inference_version = self.updates
        self.inference_params = {"weight": np.asarray([self.updates], np.float32)}

    def train(self, batch):
        self.updates += 1
        return {"loss": .1}

    def parameter_hash(self):
        return tree_hash(self.updates)

    def snapshot(self):
        return {"format": BACKBONE_FORMAT, "task_namespace": TASK_NAMESPACE,
                "initialization": "random_seed_no_transfer",
                "backbone": {"training": {"counters": {"updates": self.updates, "batches": self.updates, "actions": 0},
                                           "params": {"weight": np.array([self.updates], np.float32)}},
                             "inference_version": self.inference_version}}

    def load(self, saved):
        if saved["format"] != BACKBONE_FORMAT:
            raise ValueError("wrong task")
        self.updates = saved["backbone"]["training"]["counters"]["updates"]
        self.inference_version = saved["backbone"]["inference_version"]


class FakeEnv:
    def __init__(self, **kwargs):
        self.closed = False

    def reset(self, seed):
        self.seed, self.t = seed, 0
        self.length = 3+seed % 2
        return np.full((12288,), seed, np.uint8)

    def step(self, action):
        assert np.asarray(action).shape == (6,)
        assert (np.abs(action) <= 1).all()
        self.t += 1
        done = self.t == self.length
        return dict(observation=np.full((12288,), self.seed+self.t, np.uint8),
                    reward=float(action.sum()), continuation=0. if done else 1.,
                    done=done, terminated=done, truncated=False, env_steps=1 if done else 2,
                    episode_seed=self.seed)

    def close(self):
        self.closed = True


def evaluate(backbone, config, run_dir, agent_steps, **kwargs):
    return {"status": "complete", "steps": agent_steps, "actual_ticks": kwargs["training_env_steps"],
            "hash_gate": {"passed": True}}


class MinecraftTrainingTests(unittest.TestCase):
    def execute(self, root, **kwargs):
        return run(configuration(), root, "unused", kind="test", env_factory=FakeEnv,
                   backbone_factory=FakeBackbone, evaluator=evaluate, identity={"test": True}, **kwargs)

    def test_variable_boundary_and_exact_final_budget(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)/"run"
            result = self.execute(root)
            self.assertEqual(result["agent_steps"], 11)
            self.assertEqual(result["env_steps"], 19)
            self.assertEqual(result["episodes"], 4)
            self.assertEqual(result["update_steps"], 4)
            paths = sorted((root/"CHECKPOINTS").glob("*.pkl"))
            self.assertEqual([int(p.stem.rsplit("_", 1)[1]) for p in paths], [7, 10, 11])
            final = load_checkpoint(paths[-1])
            self.assertTrue(final["collector"]["budget_truncated"])
            self.assertEqual(final["collector"]["episode_action_count"], 1)
            record = final["replay"]["archive"][-1]
            with np.load(root/"REPLAY"/record["file"]) as data:
                self.assertTrue(data["is_last"][-1])
                self.assertFalse(data["is_terminal"][-1])
                self.assertEqual(len(data["action"]), 2)
            events = [json.loads(line) for line in (root/"events.jsonl").read_text().splitlines()]
            self.assertEqual(events[0]["checkpoint_delay_actions"], 2)
            with self.assertRaises(FileExistsError):
                self.execute(root)

    def test_explicit_resume_preserves_software_and_next_episode_actions(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.execute(root/"first")
            checkpoint = root/"first/CHECKPOINTS/minecraft_checkpoint_000000007.pkl"
            result = self.execute(root/"resumed", resume=checkpoint)
            self.assertEqual(result["agent_steps"], 11)
            a = load_checkpoint(root/"first/CHECKPOINTS/minecraft_checkpoint_000000011.pkl")
            b = load_checkpoint(root/"resumed/CHECKPOINTS/minecraft_checkpoint_000000011.pkl")
            for name in ("backbone", "scheduler", "rng", "collector"):
                self.assertEqual(tree_hash(a[name]), tree_hash(b[name]), name)
            for record in a["replay"]["archive"]:
                with np.load(root/"first/REPLAY"/record["file"]) as left, np.load(root/"resumed/REPLAY"/record["file"]) as right:
                    np.testing.assert_array_equal(left["action"], right["action"])
                    np.testing.assert_array_equal(left["image"], right["image"])

    def test_boundary_validation_rejects_pending_and_bad_tick_count(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)/"run"
            self.execute(root)
            saved = load_checkpoint(root/"CHECKPOINTS/minecraft_checkpoint_000000007.pkl")
            for field, value in (("env_steps", 15), ("agent_steps", 8)):
                bad = deepcopy(saved)
                bad["counters"][field] = value
                with self.assertRaises(ValueError):
                    validate_payload(bad)
            for key, value in (("last_action", np.zeros(6)), ("pending", {}), ("timestamp", 500)):
                bad = deepcopy(saved)
                bad["collector"]["dual_chain"]["state"][key] = value
                with self.assertRaises(ValueError):
                    validate_payload(bad)

    def test_old_walker_checkpoint_rejected_before_loading(self):
        instance = object.__new__(MinecraftBackbone)
        with self.assertRaisesRegex(ValueError, "Walker"):
            instance.load({"schema_version": 1, "training": {}})

    def test_chain_generic_boundary_retains_rng_and_rejects_gap(self):
        backbone = FakeBackbone()
        chain = DualChain(backbone, seed=42)
        image = np.zeros((64, 64, 3), np.uint8)
        chain.reset(image, episode_id=0)
        for t in range(1, 4):
            action = chain.act()[0]
            chain.step(image, action, episode_id=0, timestamp=t)
        saved = chain.state_dict()
        backbone.updates = 1
        backbone.pin_inference()
        restored = DualChain(backbone, seed=999)
        restore_chain_boundary(restored, saved, 3)
        self.assertFalse(restored.initialized)
        self.assertEqual(restored.model_version, 1)
        self.assertEqual(tree_hash(saved["rngs"]), tree_hash(restored.state_dict()["rngs"]))
        bad = deepcopy(saved)
        bad["state"]["pending"] = {}
        with self.assertRaises(ValueError):
            restore_chain_boundary(restored, bad, 3)

    def test_formal_gates_fail_closed(self):
        cfg = configuration()
        with self.assertRaises(ValueError):
            validate_config(cfg, formal=True)
        cfg.update(budget_agent_steps=1_000_000, checkpoint_every=10_000, eval_every=20_000)
        validate_config(cfg, formal=True)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/"gate.json"
            base = dict(task_namespace=TASK_NAMESPACE, identity={"test": True},
                        config_sha256=config_sha256(cfg), passed=True, checks={})
            path.write_text(json.dumps(base))
            with self.assertRaisesRegex(ValueError, "missing live"):
                verify_launch_gate(path, cfg, {"test": True})
            base["config_sha256"] = "wrong"
            path.write_text(json.dumps(base))
            with self.assertRaisesRegex(ValueError, "identity/config"):
                verify_launch_gate(path, cfg, {"test": True})

    def test_worker_exception_retains_prefix_and_marks_unknown_tick_count(self):
        class FailingEnv(FakeEnv):
            def step(self, action):
                if self.t == 1:
                    raise RuntimeError("engine stopped after unknown partial action")
                return super().step(action)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)/"run"
            with self.assertRaisesRegex(RuntimeError, "engine stopped"):
                run(configuration(), root, "unused", kind="test", env_factory=FailingEnv,
                    backbone_factory=FakeBackbone, evaluator=evaluate, identity={"test": True})
            failure = json.loads((root/"failure.json").read_text())
            self.assertFalse(failure["actual_interaction_counters_complete"])
            self.assertEqual(failure["agent_steps"], 1)
            with np.load(root/"failed_episode_evidence.npz") as data:
                self.assertEqual(len(data["image"]), 2)
                self.assertEqual(len(data["reward"]), 2)
            self.assertFalse((root/"completion.json").exists())

    def test_inference_error_does_not_drop_acknowledged_transition(self):
        class FailingBackbone(FakeBackbone):
            def prior(self, *args, **kwargs):
                raise RuntimeError("inference failed")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)/"run"
            with self.assertRaisesRegex(RuntimeError, "inference failed"):
                run(configuration(), root, "unused", kind="test", env_factory=FakeEnv,
                    backbone_factory=FailingBackbone, evaluator=evaluate, identity={"test": True})
            failure = json.loads((root/"failure.json").read_text())
            self.assertTrue(failure["actual_interaction_counters_complete"])
            with np.load(root/"failed_episode_evidence.npz") as data:
                self.assertEqual(len(data["image"]), 2)
                self.assertEqual(len(data["reward"]), 2)
                self.assertTrue(np.isfinite(data["reward"]).all())

    def test_short_episode_no_sample_is_not_falsely_complete(self):
        class ShortEnv(FakeEnv):
            def reset(self, seed):
                result = super().reset(seed)
                self.length = 1
                return result
        cfg = configuration()
        cfg["backbone"]["batch_length"] = 30  # Even across real resets, only 22 real rows exist.
        cfg["train_ratio"] = 10
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)/"run"
            with self.assertRaisesRegex(RuntimeError, "Insufficient sampleable replay"):
                run(cfg, root, "unused", kind="test", env_factory=ShortEnv,
                    backbone_factory=FakeBackbone, evaluator=evaluate, identity={"test": True})
            failure = json.loads((root/"failure.json").read_text())
            self.assertEqual(failure["agent_steps"], 11)
            self.assertEqual(failure["update_steps"], 0)
            self.assertFalse((root/"completion.json").exists())

    def test_checkpoint_integrity_rejects_changed_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)/"run"
            self.execute(root)
            path = root/"CHECKPOINTS/minecraft_checkpoint_000000007.pkl"
            with path.open("ab") as file:
                file.write(b"unexpected")
            with self.assertRaisesRegex(ValueError, "SHA256/size"):
                load_checkpoint(path)

    def test_software_probe_matches_and_does_not_mutate_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)/"run"
            self.execute(root)
            checkpoint = root/"CHECKPOINTS/minecraft_checkpoint_000000007.pkl"
            payload = load_checkpoint(checkpoint)
            before = {path: sha256(path) for path in (checkpoint, root/"REPLAY/index.json")}
            image = np.full((64, 64, 3), 2, np.uint8)
            reports = []
            for _ in range(2):
                replay = Replay(root/"REPLAY", 100, seed=999, cacheepisodes=1)
                reports.append(software_probe(payload, FakeBackbone(), replay, image))
            self.assertEqual(reports[0], reports[1])
            self.assertTrue(reports[0]["full_checkpoint_reload_equal"])
            self.assertEqual(reports[0]["next_update_counters"]["updates"], payload["counters"]["update_steps"]+1)
            for path, expected in before.items():
                self.assertEqual(sha256(path), expected)

    def test_software_probe_rejects_corrupt_optimizer_restore(self):
        class BadRestore(FakeBackbone):
            def load(self, saved):
                super().load(saved)
                self.updates += 1
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)/"run"
            self.execute(root)
            payload = load_checkpoint(root/"CHECKPOINTS/minecraft_checkpoint_000000007.pkl")
            replay = Replay(root/"REPLAY", 100)
            with self.assertRaisesRegex(ValueError, "did not reload exactly"):
                software_probe(payload, BadRestore(), replay, np.zeros((64, 64, 3), np.uint8))

    def test_software_probe_rejects_wrong_first_frame_contract(self):
        with self.assertRaisesRegex(ValueError, "uint8 RGB64"):
            software_probe({"config": {}}, FakeBackbone(), None, np.zeros((64, 64, 3), np.float32))

    def test_native_outcome_is_postepisode_exact_sidecar_not_terminal_guess(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/"diagnostics.jsonl"
            self.assertIsNone(episode_outcome(path, 1, 5, required=False)["success"])
            with self.assertRaisesRegex(ValueError, "no native"):
                episode_outcome(path, 1, 5)
            row = dict(task_id="HelixWMVisibleNavigation-v0", event="native_step", episode_seed=1,
                       episode_ticks=5, privileged=dict(success=True))
            path.write_text(json.dumps(row)+"\n")
            self.assertTrue(episode_outcome(path, 1, 5)["success"])
            with self.assertRaisesRegex(ValueError, "do not match"):
                episode_outcome(path, 2, 5)
            row["privileged"]["success"] = 1
            path.write_text(json.dumps(row)+"\n")
            with self.assertRaisesRegex(ValueError, "do not match"):
                episode_outcome(path, 1, 5)

    def test_acceptance_rechecks_frozen_media_binding(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root/"SMOKE_ARTIFACTS").mkdir()
            (root/"EVAL").mkdir()
            (root/"SMOKE_ARTIFACTS/curve.png").write_bytes(b"\x89PNG\r\n\x1a\n")
            (root/"SMOKE_ARTIFACTS/curve.pdf").write_bytes(b"%PDF-1.7\n")
            (root/"EVAL/episodes.csv").write_text("condition,return\n")
            video = root/"SMOKE_ARTIFACTS/video.mp4"
            video.write_bytes(b"mock decoded fixture binding")
            report = dict(status="complete", hash_gate=dict(passed=True), post_artifact_frozen_gate=dict(passed=True),
                          collector_inference_restored=True, conditions={name: {"episodes": 5} for name in ("clean", "dropout")},
                          raw_csv="EVAL/episodes.csv", learning_curve=dict(available=True,
                          files=["SMOKE_ARTIFACTS/curve.png", "SMOKE_ARTIFACTS/curve.pdf"]),
                          video=dict(path="SMOKE_ARTIFACTS/video.mp4", full_decode_pass=True, frames=4,
                                     frames_expected=4, bytes=video.stat().st_size, sha256=sha256(video)))
            self.assertTrue(verify_evaluation_report(report, root, configuration(), smoke=True))
            bad = deepcopy(report)
            bad["post_artifact_frozen_gate"]["passed"] = False
            with self.assertRaisesRegex(ValueError, "gate failed"):
                verify_evaluation_report(bad, root, configuration(), smoke=True)
            video.write_bytes(b"altered after recorded decode")
            with self.assertRaisesRegex(ValueError, "decode proof"):
                verify_evaluation_report(report, root, configuration(), smoke=True)

    def test_acceptance_rejects_incomplete_final_matrix(self):
        report = dict(status="complete", hash_gate=dict(passed=True), post_artifact_frozen_gate=dict(passed=True),
                      collector_inference_restored=True,
                      conditions={name: {} for name in configuration()["final_arms"]}, actual_case_count=519)
        with self.assertRaisesRegex(ValueError, "520 cases"):
            verify_evaluation_report(report, ".", configuration(), final=True)

    def test_mock_training_real_media_pipeline_and_independent_acceptance(self):
        from helixwm_minecraft.evaluation import evaluate as actual_evaluate
        class IntegratedBackbone(FakeBackbone):
            @property
            def agent(self):
                return SimpleNamespace(n_updates=self.updates, n_batches=self.updates, n_actions=0)
            def parameter_hash(self, inference=False):
                return tree_hash(self.inference_params if inference else self.updates)
        class RenderEnv(FakeEnv):
            def reset(self, seed):
                self.seed, self.t, self.length = seed, 0, 3+seed % 2
                return np.full(12288, seed % 256, np.uint8)
            def step(self, action):
                self.t += 1
                done = self.t == self.length
                return dict(observation=np.full(12288, (self.seed+self.t) % 256, np.uint8),
                            reward=float(action.sum()), continuation=0. if done else 1.,
                            done=done, terminated=done, truncated=False, env_steps=1 if done else 2,
                            episode_seed=self.seed)
            def render(self, size=256):
                return np.full((size, size, 3), self.t % 255, np.uint8)
        cfg = configuration()
        cfg.update(budget_agent_steps=5, checkpoint_every=5, eval_every=5, periodic_eval_episodes=1, video_fps=20)
        def live_artifacts(backbone, config, root, steps, **kwargs):
            report = actual_evaluate(backbone, config, root, steps, env_factory=RenderEnv, **kwargs)
            verify_evaluation_report(report, root, config, smoke=True)
            return report
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)/"run"
            result = run(cfg, root, "unused", kind="test", env_factory=RenderEnv,
                         backbone_factory=IntegratedBackbone, evaluator=live_artifacts, identity={"test": True})
            self.assertEqual(result["agent_steps"], 5)
            report = result["evaluation_report"]
            self.assertTrue(report["video"]["full_decode_pass"])
            self.assertEqual(report["video"]["frames"], 4)
            self.assertEqual(len(list((root/"SMOKE_ARTIFACTS").rglob("*.mp4"))), 1)
            self.assertFalse((root/"VIDEOS").exists())


if __name__ == "__main__":
    unittest.main()
