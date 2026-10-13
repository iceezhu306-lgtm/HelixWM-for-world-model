"""Replay boundary, sampling, persistence and unit-accounting tests."""

import tempfile
from pathlib import Path
import unittest

import numpy as np

from helixwm.environment import WalkerEnv
from helixwm.replay import DiskReplay


def episode(steps, marker=0):
    n = steps + 1
    observations = np.zeros((n, 12), dtype=np.uint8)
    observations[:, 0] = marker
    observations[:, 1] = np.arange(n)
    return {
        "observations": observations,
        "actions": np.arange(steps * 2, dtype=np.float32).reshape(steps, 2),
        "rewards": np.arange(n, dtype=np.float32),
        "continuations": np.ones(n, dtype=np.float32),
        "errors": np.repeat(np.arange(n, dtype=np.float32)[:, None], 2, axis=1),
        "error_valid": np.array([False] + [True] * steps),
        "model_versions": np.arange(n, dtype=np.int64),
    }


class ReplayTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name)

    def test_sequences_preserve_transition_alignment_without_crossing_episodes(self):
        replay = DiskReplay(self.path, seed=7)
        replay.add_episode(episode(6, marker=3))
        replay.add_episode(episode(9, marker=9))
        sample = replay.sample(128, 3)
        self.assertEqual(sample["observations"].shape, (128, 4, 12))
        self.assertEqual(sample["actions"].shape, (128, 3, 2))
        self.assertEqual(sample["observations"].dtype, np.uint8)
        np.testing.assert_array_equal(np.diff(sample["observations"][:, :, 1], axis=1), 1)
        np.testing.assert_array_equal(sample["observations"][:, :, 0], np.repeat(sample["observations"][:, :1, 0], 4, axis=1))
        np.testing.assert_array_equal(sample["actions"][:, :, 0] / 2, sample["observations"][:, :-1, 1])
        np.testing.assert_array_equal(sample["rewards"], sample["observations"][:, :, 1])
        self.assertTrue(sample["is_first"][:, 0].all())
        self.assertFalse(sample["is_first"][:, 1:].any())
        self.assertTrue(sample["valid"].all())

    def test_sampling_is_uniform_over_starts_not_episodes(self):
        replay = DiskReplay(self.path, seed=11)
        replay.add_episode(episode(3, marker=1))
        replay.add_episode(episode(8, marker=2))
        # At L=3, the episodes contribute 1 and 6 valid starts respectively.
        sample = replay.sample(14000, 3)
        small_fraction = np.mean(sample["observations"][:, 0, 0] == 1)
        self.assertLess(abs(small_fraction - 1 / 7), 0.015)

    def test_eviction_preserves_files_for_older_checkpoints(self):
        replay = DiskReplay(self.path, capacity_steps=8)
        first = replay.add_episode(episode(5, marker=1))
        checkpoint = replay.snapshot()
        second = replay.add_episode(episode(5, marker=2))
        self.assertEqual(replay.total_steps, 5)
        self.assertEqual([record["id"] for record in replay.episodes], [second["id"]])
        self.assertTrue((self.path / first["file"]).is_file())
        self.assertEqual(len(replay.archived_episodes), 2)
        replay.restore(checkpoint)
        self.assertTrue(np.all(replay.sample(20, 2)["observations"][:, :, 0] == 1))
        third = replay.add_episode(episode(5, marker=3))
        self.assertGreater(third["id"], second["id"])
        self.assertTrue((self.path / second["file"]).is_file())

    def test_snapshot_restores_rng_and_index_and_reopens_from_disk(self):
        replay = DiskReplay(self.path, seed=11)
        replay.add_episode(episode(8))
        snapshot = replay.snapshot()
        expected = replay.sample(20, 3)
        replay.restore(snapshot)
        for name, value in replay.sample(20, 3).items():
            np.testing.assert_array_equal(value, expected[name])
        replay.restore(snapshot)
        reopened = DiskReplay(self.path, seed=999)
        for name, value in reopened.sample(20, 3).items():
            np.testing.assert_array_equal(value, expected[name])
        self.assertEqual(reopened.total_steps_seen, 8)
        self.assertGreater(reopened.disk_bytes, 0)

    def test_checksum_corruption_is_rejected_on_restore(self):
        replay = DiskReplay(self.path)
        record = replay.add_episode(episode(4))
        snapshot = replay.snapshot()
        path = self.path / record["file"]
        data = bytearray(path.read_bytes())
        data[len(data) // 2] ^= 1
        path.write_bytes(data)
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            replay.restore(snapshot)

    def test_cache_is_bounded_and_short_episodes_are_not_padded(self):
        replay = DiskReplay(self.path, seed=19)
        for marker in range(15):
            replay.add_episode(episode(3, marker=marker))
        replay.sample(100, 3)
        self.assertLessEqual(len(replay._cache), 8)
        self.assertFalse(replay.can_sample(4))
        with self.assertRaisesRegex(ValueError, "long enough"):
            replay.sample(1, 4)

    def test_invalid_episode_and_membership_rejected(self):
        replay = DiskReplay(self.path)
        invalid = episode(4)
        invalid["error_valid"][0] = True
        with self.assertRaises(ValueError):
            replay.add_episode(invalid)
        replay.add_episode(episode(4))
        snapshot = replay.snapshot()
        snapshot["active_ids"] = [987]
        with self.assertRaises(ValueError):
            replay.restore(snapshot)
        invalid = episode(4)
        invalid["actions"][0, 0] = np.nan
        with self.assertRaises(ValueError):
            replay.add_episode(invalid)


class _TimeStep:
    reward = 2.0
    discount = 1.0

    def __init__(self, last=False):
        self._last = last

    def last(self):
        return self._last


class _FakeDMC:
    def __init__(self):
        self.calls = 0

    def step(self, action):
        self.calls += 1
        return _TimeStep()


class EnvironmentAccountingTests(unittest.TestCase):
    def wrapper(self):
        wrapper = WalkerEnv.__new__(WalkerEnv)
        wrapper._env = _FakeDMC()
        wrapper._closed = False
        wrapper._done = False
        wrapper._episode_steps = 0
        wrapper.episode_env_steps = 3
        wrapper.action_repeat = 2
        wrapper.action_dim = 2
        wrapper.action_minimum = np.array([-1.0, -1.0])
        wrapper.action_maximum = np.array([1.0, 1.0])
        wrapper._observation = lambda: np.zeros(12, dtype=np.uint8)
        return wrapper

    def test_repeat_counts_actual_calls_and_handles_short_last_decision(self):
        wrapper = self.wrapper()
        first = wrapper.step(np.zeros(2))
        last = wrapper.step(np.zeros(2))
        self.assertEqual((first["env_steps"], first["reward"], first["done"]), (2, 4.0, False))
        self.assertEqual((last["env_steps"], last["reward"], last["done"]), (1, 2.0, True))
        self.assertEqual(wrapper._env.calls, 3)
        with self.assertRaises(RuntimeError):
            wrapper.step(np.zeros(2))

    def test_action_bounds_are_checked_without_silent_clipping(self):
        wrapper = self.wrapper()
        for action in (np.zeros(3), [1.001, 0], [np.nan, 0]):
            with self.assertRaises(ValueError):
                wrapper.step(action)
        self.assertEqual(wrapper._env.calls, 0)

    def test_native_terminal_stops_repeat_and_preserves_zero_discount(self):
        wrapper = self.wrapper()
        terminal = _TimeStep(last=True)
        terminal.discount = 0.0
        wrapper._env.step = lambda action: terminal
        result = wrapper.step(np.zeros(2))
        self.assertEqual(result["env_steps"], 1)
        self.assertEqual(result["reward"], 2.0)
        self.assertEqual(result["continuation"], 0.0)
        self.assertTrue(result["done"])

    def test_fractional_discounts_multiply_across_repeat(self):
        wrapper = self.wrapper()
        transition = _TimeStep()
        transition.discount = 0.5
        wrapper._env.step = lambda action: transition
        self.assertEqual(wrapper.step(np.zeros(2))["continuation"], 0.25)


if __name__ == "__main__":
    unittest.main()
