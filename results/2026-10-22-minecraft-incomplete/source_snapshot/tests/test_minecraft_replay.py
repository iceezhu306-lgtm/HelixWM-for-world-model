"""Regression checks for complete short episodes and real reset boundaries."""
from copy import deepcopy
import tempfile
import unittest

import numpy as np

from helixwm_minecraft.storage import Replay, SAMPLING_POLICY
from helixwm_v2.storage import FIELDS, Replay as V2Replay


def episode(steps, marker, success=False):
    n = steps + 1
    images = np.empty((n, 64, 64, 3), np.uint8)
    for t in range(n):
        images[t] = [marker, t % 256, t // 256]
    actions = np.arange(n * 6, dtype=np.float32).reshape(n, 6) / 1000
    actions[-1] = 0
    reward = np.arange(n, dtype=np.float32) / 100
    reward[-1] = 100.25 if success else -3.5
    first, last, terminal = (np.zeros(n, bool) for _ in range(3))
    first[0], last[-1], terminal[-1] = True, True, success
    return dict(image=images, action=actions, reward=reward,
                is_first=first, is_last=last, is_terminal=terminal)


class FixedStarts:
    """Make a regression window land exactly across both episode boundaries."""
    def __init__(self, starts):
        self.starts = np.asarray(starts)

    def integers(self, high, size):
        assert size == len(self.starts) and (self.starts < high).all()
        return self.starts


class MinecraftReplayTests(unittest.TestCase):
    def test_short_success_and_long_failure_all_fields_and_loss_region(self):
        with tempfile.TemporaryDirectory() as directory:
            replay = Replay(directory, seed=43)
            short, long = episode(4, 11, True), episode(120, 22)
            replay.add_episode(short, success=True)
            replay.add_episode(long, success=False)
            source = {key: np.concatenate([short[key], long[key]]) for key in FIELDS}
            # Both windows put the oldest short success after 32 burn-in rows.
            # A non-circular chronological sampler could never do this.
            starts = [80, 85]
            replay.rng = FixedStarts(starts)
            batch = replay.sample(2, 96, burnin=32)
            for b, start in enumerate(starts):
                positions = (np.arange(96) + start) % 126
                self.assertEqual(len(set(positions)), 96)
                for key in FIELDS:
                    np.testing.assert_array_equal(batch[key][b], source[key][positions])
                terminals = np.flatnonzero(batch["is_terminal"][b])
                self.assertEqual(len(terminals), 1)
                self.assertGreaterEqual(terminals[0], 32)
                self.assertEqual(batch["reward"][b, terminals[0]], np.float32(100.25))
                self.assertTrue(batch["is_last"][b, terminals[0]])
                self.assertTrue(batch["is_first"][b, terminals[0] + 1])
                np.testing.assert_array_equal(batch["action"][b, terminals[0]], np.zeros(6))
            stats = replay.sampling_statistics(96, burnin=32)
            self.assertEqual(stats["short_episode_fraction"], .5)
            self.assertEqual(stats["short_success_sampling_coverage"], 1.)
            self.assertEqual(stats["success_sampling_coverage"], 1.)
            self.assertEqual(stats["supervised_rows_delivered"], 128)

    def test_all_short_episodes_are_sampleable_without_padding(self):
        with tempfile.TemporaryDirectory() as directory:
            replay = Replay(directory)
            originals = [episode(1, marker, marker % 2 == 0) for marker in range(48)]
            for source in originals:
                replay.add_episode(source, success=bool(source["is_terminal"][-1]))
            self.assertTrue(replay.can_sample(96))
            self.assertFalse(replay.can_sample(97))
            batch = replay.sample(1, 96, burnin=32)
            self.assertEqual(len(set(batch["image"][0, :, 0, 0, 0])), 48)
            self.assertEqual(int(batch["is_first"].sum()), 48)
            self.assertEqual(int(batch["is_last"].sum()), 48)
            self.assertEqual(int(batch["is_terminal"].sum()), 24)
            for index in range(96):
                marker, t, _ = batch["image"][0, index, 0, 0]
                for key in FIELDS:
                    np.testing.assert_array_equal(batch[key][0, index], originals[int(marker)][key][int(t)])
            self.assertEqual(replay.total_steps, 48)
            self.assertEqual(replay.sampling_statistics(96, burnin=32)["short_episode_fraction"], 1.)

    def test_reset_and_outgoing_action_alignment_in_supervised_suffix(self):
        with tempfile.TemporaryDirectory() as directory:
            replay = Replay(directory)
            left, right = episode(2, 10, True), episode(4, 20)
            replay.add_episode(left, success=True)
            replay.add_episode(right, success=False)
            replay.rng = FixedStarts([0])
            batch = replay.sample(1, 8, burnin=2)
            # Adapter carries outgoing action[burnin-1] into official train,
            # then official agent shifts outgoing actions by one observation.
            incoming = np.concatenate([batch["action"][:, 1:2], batch["action"][:, 2:-1]], axis=1)
            first = batch["is_first"][:, 2:]
            np.testing.assert_array_equal(incoming[0, 0], left["action"][1])
            np.testing.assert_array_equal(incoming[0, 1], left["action"][-1])
            self.assertTrue(first[0, 1])
            np.testing.assert_array_equal(incoming[0, 2], right["action"][0])
            self.assertEqual(batch["reward"][0, 2], np.float32(100.25))
            self.assertEqual(batch["reward"][0, 3], 0.)
            self.assertTrue(batch["is_terminal"][0, 2])
            self.assertFalse(batch["is_terminal"][0, -1])  # time limit remains a truncation

    def test_snapshot_restore_repeats_sampling_and_all_counters(self):
        with tempfile.TemporaryDirectory() as directory:
            replay = Replay(directory, seed=123)
            replay.add_episode(episode(2, 1, True), success=True)
            replay.add_episode(episode(10, 2), success=False)
            replay.sample(3, 9, burnin=3)
            snapshot = replay.snapshot()
            expected = replay.sample(4, 9, burnin=3)
            expected_state = replay.snapshot()
            replay.restore(snapshot, persist=True)
            reopened = Replay(directory)
            actual = reopened.sample(4, 9, burnin=3)
            for key in FIELDS:
                np.testing.assert_array_equal(actual[key], expected[key])
            self.assertEqual(reopened.snapshot(), expected_state)
            self.assertEqual(reopened.snapshot()["sampling_policy"], SAMPLING_POLICY)

    def test_unknown_outcome_not_misreported_as_success(self):
        with tempfile.TemporaryDirectory() as directory:
            replay = Replay(directory)
            replay.add_episode(episode(4, 1, True))
            replay.sample(2, 5, burnin=1)
            stats = replay.sampling_statistics(5, burnin=1)
            self.assertEqual(stats["unknown_outcome_episodes"], 1)
            self.assertEqual(stats["known_success_episodes"], 0)
            self.assertIsNone(stats["success_sampling_coverage"])
            self.assertEqual(stats["episode_end_sampling_coverage"], 1.)

    def test_eviction_coverage_denominator_uses_active_episodes(self):
        with tempfile.TemporaryDirectory() as directory:
            replay = Replay(directory, capacity_steps=6)
            replay.add_episode(episode(2, 1, True), success=True)
            replay.sample(1, 3)
            replay.add_episode(episode(6, 2), success=False)
            stats = replay.sampling_statistics(6)
            self.assertEqual(stats["active_episodes"], 1)
            self.assertEqual(stats["known_success_episodes"], 0)
            self.assertIsNone(stats["success_sampling_coverage"])
            self.assertEqual(len(replay.archive), 2)
            replay.restore(replay.snapshot())

    def test_reject_old_sampler_and_corrupt_sampling_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            replay = Replay(directory)
            replay.add_episode(episode(3, 1))
            snapshot = replay.snapshot()
            for field in ("sampling_policy", "sampling_counters", "episode_success"):
                broken = deepcopy(snapshot)
                if field == "sampling_policy":
                    broken[field] = "old_sampler"
                elif field == "sampling_counters":
                    broken[field]["supervised_rows"] = 99
                else:
                    broken[field]["0"] = "yes"
                with self.assertRaises(ValueError):
                    replay.restore(broken)
                self.assertEqual(replay.snapshot(), snapshot)

    def test_insufficient_rows_and_invalid_parameters_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            replay = Replay(directory)
            replay.add_episode(episode(2, 1))
            with self.assertRaises(ValueError):
                replay.sample(1, 4)
            for batch, length, burnin in [(True, 2, 0), (1, 2, 2), (1, 2, -1), (1, True, 0)]:
                with self.assertRaises(ValueError):
                    replay.sample(batch, length, burnin=burnin)
            with self.assertRaises(ValueError):
                replay.add_episode(episode(2, 2), success=True)
            with self.assertRaises(ValueError):
                replay.add_episode(episode(2, 2), success=1)

    def test_v2_sampler_stays_episode_local(self):
        with tempfile.TemporaryDirectory() as directory:
            replay = V2Replay(directory)
            replay.add_episode(episode(2, 1, True))
            replay.add_episode(episode(2, 2))
            self.assertFalse(replay.can_sample(6))
            with self.assertRaises(ValueError):
                replay.sample(1, 6)


if __name__ == "__main__":
    unittest.main()
