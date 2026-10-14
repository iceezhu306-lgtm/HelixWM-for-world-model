"""NumPy-only V2 replay/checkpoint invariants and optional JAX dtype tests."""

import importlib.util
import json
from pathlib import Path
import pickle
import random
import tempfile
import unittest

import numpy as np

from helixwm_v2.storage import Replay, load_checkpoint, save_checkpoint, sha256, tree_hash


def episode(steps, marker=1):
    count = steps + 1
    images = np.zeros((count, 64, 64, 3), np.uint8)
    images[:, :, :, 0] = marker
    images[:, :, :, 1] = np.arange(count)[:, None, None]
    actions = np.repeat(np.arange(count, dtype=np.float32)[:, None], 6, axis=1)
    actions[-1] = 0
    return {"image": images, "action": actions, "reward": np.arange(count, dtype=np.float32),
            "is_first": np.arange(count) == 0, "is_last": np.arange(count) == steps,
            "is_terminal": np.zeros(count, np.bool_)}


class StorageV2Tests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name)
        self.replay = Replay(self.path / "REPLAY", capacity_steps=100, seed=17, cacheepisodes=2)

    def payload(self):
        return {"backbone": {"weight": np.arange(6, dtype=np.float32).reshape(2, 3),
                             "optimizer": {"step": np.int32(3), "moment": np.ones((2, 3), np.float32)}},
                "dualchain": {"risk": np.array([0.3, 0.2], np.float32)},
                "collector": {"episode": episode(3), "environment": {"time": 1.25}},
                "host_rng": {"python": random.getstate(), "numpy": np.random.get_state()},
                "replay": self.replay.snapshot(), "scheduler": {"updates": 3},
                "config": {"seed": 0}, "metadata": {"scope": "test"},
                "counters": {"agent_steps": 10, "env_steps": 20}}

    def test_alignment_exact_raw_flags_uniform_starts_and_no_crossing(self):
        self.replay.add_episode(episode(3, 1))
        self.replay.add_episode(episode(8, 2))
        batch = self.replay.sample(7000, 4)
        self.assertEqual(batch["image"].shape, (7000, 4, 64, 64, 3))
        self.assertEqual(batch["action"].shape, (7000, 4, 6))
        positions = batch["image"][:, :, 0, 0, 1]
        np.testing.assert_array_equal(np.diff(positions, axis=1), 1)
        np.testing.assert_array_equal(batch["is_first"], positions == 0)
        self.assertTrue((~batch["is_first"][:, 0]).any())
        np.testing.assert_array_equal(batch["reward"], positions)
        np.testing.assert_array_equal(batch["action"][:, :-1, 0], positions[:, :-1])
        markers = batch["image"][:, :, 0, 0, 0]
        np.testing.assert_array_equal(markers, np.repeat(markers[:, :1], 4, axis=1))
        self.assertLess(abs(np.mean(markers[:, 0] == 1) - 1 / 7), 0.02)

    def test_snapshot_rng_restore_is_read_only_and_exact(self):
        self.replay.add_episode(episode(10))
        snapshot = self.replay.snapshot()
        original_index = (self.path / "REPLAY" / "index.json").read_bytes()
        expected = self.replay.sample(12, 4)
        self.replay.restore(snapshot)
        actual = self.replay.sample(12, 4)
        for name in actual:
            np.testing.assert_array_equal(actual[name], expected[name])
        self.assertEqual((self.path / "REPLAY" / "index.json").read_bytes(), original_index)

    def test_eviction_retains_inactive_files_and_cache_is_bounded(self):
        replay = Replay(self.path / "bounded", capacity_steps=12, cacheepisodes=2)
        first = replay.add_episode(episode(4))
        for marker in range(2, 7):
            replay.add_episode(episode(4, marker))
        self.assertEqual(replay.total_steps, 12)
        self.assertEqual(replay.total_steps_seen, 24)
        self.assertEqual(len(replay.archive), 6)
        self.assertTrue((replay.directory / first["file"]).exists())
        replay.sample(100, 3)
        self.assertLessEqual(len(replay._cache), 2)
        self.assertFalse(replay.can_sample(6))

    def test_corrupt_archive_and_invalid_alignment_rejected(self):
        record = self.replay.add_episode(episode(5))
        snapshot = self.replay.snapshot()
        path = self.replay.directory / record["file"]
        original = path.read_bytes()
        path.write_bytes(b"x" * len(original))
        with self.assertRaisesRegex(ValueError, "SHA256"):
            self.replay.restore(snapshot)
        with self.assertRaisesRegex(ValueError, "SHA256"):
            self.replay.sample(1, 3)
        invalid = episode(4)
        invalid["action"][-1] = 1
        with self.assertRaisesRegex(ValueError, "placeholder"):
            self.replay.add_episode(invalid)

    def test_complete_payload_roundtrip_is_immutable_and_requires_replay(self):
        self.replay.add_episode(episode(5))
        payload = self.payload()
        checkpoint = save_checkpoint(self.path / "CHECKPOINTS", payload)
        restored = load_checkpoint(checkpoint)
        self.assertEqual(tree_hash(restored), tree_hash(payload))
        self.assertEqual(set(restored), set(payload))
        with self.assertRaises(FileExistsError):
            save_checkpoint(self.path / "CHECKPOINTS", payload)
        record = self.replay.archive[0]
        (self.replay.directory / record["file"]).unlink()
        with self.assertRaisesRegex(ValueError, "missing"):
            load_checkpoint(checkpoint)
        self.assertEqual(tree_hash(load_checkpoint(checkpoint, verify_replay=False)), tree_hash(payload))

    def test_checkpoint_file_hash_and_independent_tree_hash_detect_tampering(self):
        checkpoint = save_checkpoint(self.path / "CHECKPOINTS", self.payload())
        original = checkpoint.read_bytes()
        checkpoint.write_bytes(b"x" * len(original))
        with self.assertRaisesRegex(ValueError, "SHA256"):
            load_checkpoint(checkpoint)
        saved = pickle.loads(original)
        saved["payload"]["backbone"]["weight"][0, 0] = 999
        checkpoint.write_bytes(pickle.dumps(saved, protocol=pickle.HIGHEST_PROTOCOL))
        sidecar = json.loads(checkpoint.with_suffix(".json").read_text())
        sidecar.update({"bytes": checkpoint.stat().st_size, "sha256": sha256(checkpoint)})
        checkpoint.with_suffix(".json").write_text(json.dumps(sidecar))
        with self.assertRaisesRegex(ValueError, "tree hash"):
            load_checkpoint(checkpoint)

    def test_tree_hash_distinguishes_shape_dtype_and_structure(self):
        original = np.arange(4, dtype=np.float32)
        self.assertNotEqual(tree_hash(original), tree_hash(original.reshape(2, 2)))
        self.assertNotEqual(tree_hash(original), tree_hash(original.astype(np.float64)))
        self.assertNotEqual(tree_hash([original]), tree_hash((original,)))
        self.assertEqual(tree_hash({"a": 1, "b": original}), tree_hash({"b": original, "a": 1}))
        with self.assertRaises(TypeError):
            tree_hash(np.array([object()], dtype=object))

    @unittest.skipUnless(importlib.util.find_spec("ml_dtypes"), "ml_dtypes is optional locally")
    def test_bfloat16_values_preserve_dtype_identity(self):
        import ml_dtypes
        value = np.array([0.5, 1.5], dtype=ml_dtypes.bfloat16)
        self.assertNotEqual(tree_hash(value), tree_hash(value.view("V2")))
        payload = self.payload()
        payload["backbone"]["bfloat_weight"] = value
        checkpoint = save_checkpoint(self.path / "CHECKPOINTS", payload)
        restored = load_checkpoint(checkpoint)
        self.assertEqual(restored["backbone"]["bfloat_weight"].dtype.name, "bfloat16")
        self.assertEqual(tree_hash(restored), tree_hash(payload))

    @unittest.skipUnless(importlib.util.find_spec("jax"), "JAX is optional locally")
    def test_jax_array_and_typed_key_roundtrip(self):
        import jax
        import jax.numpy as jnp
        previous_guard = jax.config.jax_transfer_guard
        # Official-backbone tests can leave the process-wide transfer guard
        # at "disallow". This test intentionally creates device fixtures,
        # materializes them on the host, and reconstructs a typed device key.
        # Its contract is value/dtype preservation, not transfer scheduling.
        # Keep that policy local and restore the preceding test's guard.
        with jax.transfer_guard("allow"):
            payload = self.payload()
            payload["backbone"]["weight"] = jnp.arange(6, dtype=jnp.bfloat16).reshape(2, 3)
            payload["dualchain"]["key"] = jax.random.key(42)
            checkpoint = save_checkpoint(self.path / "CHECKPOINTS", payload)
            restored = load_checkpoint(checkpoint)
            self.assertEqual(tree_hash(restored), tree_hash(payload))
            np.testing.assert_array_equal(jax.random.key_data(restored["dualchain"]["key"]),
                                          jax.random.key_data(payload["dualchain"]["key"]))
        self.assertEqual(jax.config.jax_transfer_guard, previous_guard)


if __name__ == "__main__":
    unittest.main()
