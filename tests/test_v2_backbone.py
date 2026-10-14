"""Adapter boundaries and opt-in real JAX checkpoint/inference smoke.

Set HELIX_V2_UPSTREAM and HELIX_V2_TEST_DEVICE=cpu/cuda to run the real
official size1m smoke. This is a correctness check, never a reward benchmark.
"""
import importlib.util
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np

from helixwm_v2.backbone import OfficialDreamerBackbone, array_tree_hash, rng_key
from helixwm_v2.preflight import canonical_config_sha256, clean_git_identity


class AdapterBoundaryTests(unittest.TestCase):
    def test_gate_configuration_hash_is_canonical_and_rejects_nonfinite(self):
        one = {"budget": 1_000_000, "nested": {"seed": 0, "enabled": True}}
        reordered = {"nested": {"enabled": True, "seed": 0}, "budget": 1_000_000}
        self.assertEqual(canonical_config_sha256(one), canonical_config_sha256(reordered))
        altered = {**one, "budget": 999_999}
        self.assertNotEqual(canonical_config_sha256(one), canonical_config_sha256(altered))
        with self.assertRaises(ValueError):
            canonical_config_sha256({"temperature": float("nan")})

    def test_gate_rejects_both_unstaged_and_staged_source_edits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            def git(*args):
                return subprocess.run(["git", "-C", directory, *args], check=True,
                                      capture_output=True, text=True).stdout.strip()
            git("init")
            tracked = root / "model.py"
            tracked.write_text("version = 1\n")
            git("add", "model.py")
            git("-c", "user.name=Preflight Test", "-c", "user.email=test@example.invalid",
                "commit", "-m", "fixture")
            identity = clean_git_identity(root)
            self.assertEqual(identity["commit"], git("rev-parse", "HEAD"))
            self.assertTrue(identity["tracked_clean"])
            tracked.write_text("version = 2\n")
            with self.assertRaises(ValueError):
                clean_git_identity(root)
            git("add", "model.py")
            with self.assertRaises(ValueError):
                clean_git_identity(root)

    def test_explicit_rng_and_no_global_rng_consumption(self):
        np.random.seed(9)
        before = np.random.get_state()
        key = rng_key(np.random.default_rng(4))
        after = np.random.get_state()
        self.assertEqual(key.shape, (2,))
        self.assertEqual(key.dtype, np.uint32)
        np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])
        np.testing.assert_array_equal(rng_key(42), np.array([0, 42], np.uint32))
        for invalid in (None, -1, 2**32, [1, 2], np.array([1, 2], np.int64)):
            with self.assertRaises(ValueError):
                rng_key(invalid)

    def test_hash_detects_shape_dtype_and_value_changes(self):
        original = {"x": np.array([1, 2], np.float32)}
        h = array_tree_hash(original)
        self.assertEqual(h, array_tree_hash({"x": original["x"].copy()}))
        self.assertNotEqual(h, array_tree_hash({"x": original["x"].reshape(1, 2)}))
        self.assertNotEqual(h, array_tree_hash({"x": original["x"].astype(np.float64)}))
        self.assertNotEqual(h, array_tree_hash({"x": np.array([1, 3], np.float32)}))

    def test_image_contract_preserves_raw_uint8(self):
        image = np.arange(64 * 64 * 3, dtype=np.uint8).reshape(64, 64, 3)
        output = OfficialDreamerBackbone._image(image, 3)
        self.assertEqual(output.shape, (3, 64, 64, 3))
        np.testing.assert_array_equal(output[2], image)
        with self.assertRaises(ValueError):
            OfficialDreamerBackbone._image(image.astype(np.float32), 1)
        with self.assertRaises(ValueError):
            OfficialDreamerBackbone._image(np.zeros((32, 32, 3), np.uint8), 1)

    def test_action_is_not_silently_clipped(self):
        action = np.array([2, -2, 0, 0, 0, 0], np.float32)
        output = OfficialDreamerBackbone._action(action, 2)
        self.assertEqual(output.shape, (2, 6))
        self.assertEqual(float(output[0, 0]), 2.0)
        with self.assertRaises(ValueError):
            OfficialDreamerBackbone._action(np.full(6, np.nan), 1)

    def test_checkpoint_rejects_missing_dtype_shape_and_nonfinite_parameters(self):
        reference = {"weight": np.ones((2, 3), np.float32)}
        validate = OfficialDreamerBackbone._validate_loaded_parameters
        validate(reference, {"weight": reference["weight"].copy()})
        for bad in ({}, {"weight": np.ones((2, 3), np.float64)},
                    {"weight": np.ones((3, 2), np.float32)}):
            with self.assertRaises(ValueError):
                validate(reference, bad)
        with self.assertRaises(FloatingPointError):
            validate(reference, {"weight": np.full((2, 3), np.nan, np.float32)})


@unittest.skipUnless(os.environ.get("HELIX_V2_UPSTREAM") and importlib.util.find_spec("jax"),
                     "Opt-in test requires pinned upstream and a JAX environment")
class OfficialJaxSmokeTests(unittest.TestCase):
    def test_frozen_episode_parameters_and_strict_restore(self):
        with tempfile.TemporaryDirectory() as run:
            device = os.environ.get("HELIX_V2_TEST_DEVICE", "cpu")
            backbone = OfficialDreamerBackbone(
                {"size": "size1m", "batch_size": 1, "batch_length": 4,
                 "burnin": 2, "platform": device,
                 "compute_dtype": "bfloat16" if device == "cuda" else "float32"},
                os.environ["HELIX_V2_UPSTREAM"], run,
            )
            image = np.zeros((64, 64, 3), np.uint8)
            action = np.zeros(6, np.float32)
            before = backbone.parameter_hash()
            pinned = backbone.parameter_hash(inference=True)
            state = backbone.observe(backbone.initial(), action, image, True, rng=1)
            predicted = backbone.prior(state, action, rng=2)
            self.assertEqual(backbone.score(predicted, image).shape, (1,))
            sampled = backbone.act(state, rng=3)
            self.assertEqual(sampled.shape, (1, 6))
            np.testing.assert_array_equal(sampled, backbone.act(state, rng=3))
            self.assertEqual(before, backbone.parameter_hash())
            batch = {
                "image": np.zeros((1, 6, 64, 64, 3), np.uint8),
                "action": np.zeros((1, 6, 6), np.float32),
                "reward": np.zeros((1, 6), np.float32),
                "is_first": np.array([[True, False, False, False, False, False]]),
                "is_last": np.zeros((1, 6), bool),
                "is_terminal": np.zeros((1, 6), bool),
                "consec": np.zeros((1, 6), np.int32),
                "stepid": np.zeros((1, 6, 20), np.uint8),
            }
            # Exercise Ninjax's recurrent scan without allowing parameter or
            # training-RNG-counter mutation before the actual learner update.
            policy_params = {key: backbone.agent.params[key] for key in backbone.agent.policy_keys}
            prefix_args = (backbone.initial(), np.zeros((1, 6), np.float32),
                           batch["image"][:, :2], batch["action"][:, :2], batch["is_first"][:, :2])
            prefix_key = backbone.agent._seeds(0, backbone.agent.train_mirrored)
            prefix = backbone._host(backbone._prefix(policy_params, prefix_key, *backbone._device(prefix_args)))
            again = backbone._host(backbone._prefix(policy_params, prefix_key, *backbone._device(prefix_args)))
            self.assertEqual(array_tree_hash(prefix), array_tree_hash(again))
            self.assertTrue(all(np.isfinite(value).all() for value in prefix.values()))
            self.assertEqual(before, backbone.parameter_hash())
            self.assertEqual(int(backbone.agent.n_batches), 0)
            self.assertEqual(int(backbone.agent.n_updates), 0)
            metrics = backbone.train(batch)
            self.assertTrue(all(np.isfinite(v) for v in metrics.values()))
            self.assertNotEqual(before, backbone.parameter_hash())
            self.assertEqual(pinned, backbone.parameter_hash(inference=True))
            checkpoint = backbone.snapshot()
            self.assertFalse(checkpoint["inference_equals_training"])
            changed = backbone.parameter_hash()
            backbone.pin_inference()
            self.assertEqual(backbone.inference_version, 1)
            backbone.load(checkpoint)
            self.assertEqual(backbone.inference_version, 0)
            self.assertEqual(changed, backbone.parameter_hash())
            self.assertEqual(pinned, backbone.parameter_hash(inference=True))
            np.testing.assert_array_equal(sampled, backbone.act(state, rng=3))
            backbone.pin_inference()
            self.assertTrue(backbone.snapshot()["inference_equals_training"])


if __name__ == "__main__":
    unittest.main()
