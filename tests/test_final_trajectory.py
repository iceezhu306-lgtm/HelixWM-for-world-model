"""Synthetic archive engineering tests; no learned-model performance evidence."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from helixwm_final.trajectory import (TrajectoryArchive, make_record,
                                     validate_record, SCHEMA)

SOURCE, PROTOCOL, MODEL = 'a' * 64, 'b' * 64, 'c' * 64


def record(t=1, *, kind='actual', origin=None, k=2, observed=False):
    probabilities = np.broadcast_to(np.array([.2, .5, .3], np.float32), (k, 2, 3)).copy()
    state = {'deter': np.full((k, 4), t, np.float32),
             'stoch': np.broadcast_to(np.array([0., 1., 0.], np.float32), (k, 2, 3)).copy()}
    return make_record(kind=kind, episode_id=3, timestamp=t,
        origin_timestamp=t if kind == 'actual' else origin,
        depth=0 if kind == 'actual' else t-origin, model_version=7, model_sha256=MODEL,
        candidate_ids=['branch%d' % i for i in range(k)],
        ancestor_ids=['branch%d' % i for i in range(k)],
        state=state, real_state={key: value[:1].copy() for key, value in state.items()},
        actions=np.full((k, 6), .2, np.float32), weights=np.full(k, 1./k, np.float64),
        probabilities=probabilities,
        predicted_observation=np.full((k, 64, 64, 3), t/200, np.float32),
        predicted_reward=np.full(k, .7, np.float32),
        predicted_continuation=np.full(k, .99, np.float32),
        observation_available=observed,
        observed_error=np.full(k, .01, np.float64) if observed else None,
        prediction_rng_sha256='d'*64, executed_prefix_sha256='e'*64)


def snapshots_equal_array_tree(left, right):
    if isinstance(left, np.ndarray):
        return isinstance(right, np.ndarray) and left.dtype == right.dtype and np.array_equal(left, right)
    if isinstance(left, dict):
        return type(right) is dict and set(left) == set(right) and all(snapshots_equal_array_tree(v, right[k]) for k, v in left.items())
    if isinstance(left, list):
        return type(right) is list and len(left) == len(right) and all(snapshots_equal_array_tree(a, b) for a, b in zip(left, right))
    return left == right


class TrajectoryTests(unittest.TestCase):
    def setUp(self):
        # Never use the machine's C-drive default temporary directory.
        parent = Path(__file__).resolve().parents[2] / 'preflight' / 'trajectory_tests'
        parent.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=parent)
        self.root = Path(self.temp.name) / 'archive'
        self.opened = []

    def tearDown(self):
        for archive in self.opened:
            archive.close()
        self.temp.cleanup()

    def create(self, **kwargs):
        archive = TrajectoryArchive.create(self.root, source_sha256=SOURCE,
            protocol_sha256=PROTOCOL, **kwargs)
        self.opened.append(archive)
        return archive

    def restore(self, snapshot, **kwargs):
        archive = TrajectoryArchive.restore(self.root, snapshot,
            expected_source_sha256=SOURCE, expected_protocol_sha256=PROTOCOL, **kwargs)
        self.opened.append(archive)
        return archive

    def test_round_trip_retains_arrays_and_explicit_absence(self):
        original = record()
        archive = self.create()
        archive.append(original)
        loaded = archive.load_record(0)
        self.assertTrue(snapshots_equal_array_tree(original, loaded))
        self.assertIsNone(loaded['observed_error'])
        self.assertIsNone(loaded['calibration_confidence'])
        self.assertIsNone(loaded['metrics']['epistemic_disagreement'])
        original['state']['deter'][:] = 999
        loaded['predicted_observation'][:] = 0
        fresh = archive.load_record(0)
        self.assertEqual(float(fresh['state']['deter'][0, 0]), 1.)
        self.assertGreater(float(fresh['predicted_observation'][0, 0, 0, 0]), 0.)

    def test_full_100_step_segment_and_bounded_reference_memory(self):
        archive = self.create(ring_size=7)
        for timestamp in range(1, 101):
            archive.append(record(timestamp, observed=(timestamp == 100)))
        self.assertEqual(len(archive.recent_references), 7)
        self.assertTrue(all(type(x['metadata_path']) is str for x in archive.recent_references))
        self.assertFalse(any(isinstance(v, np.ndarray) for ref in archive.recent_references for v in ref.values()))
        snapshot = archive.snapshot()
        self.assertLess(len(json.dumps(snapshot)), 1100)
        restored = self.restore(snapshot)
        rows = list(restored.reconstruct(episode_id=3, candidate_id='branch1', start_timestamp=1, end_timestamp=100))
        self.assertEqual([x['timestamp'] for x in rows], list(range(1, 101)))
        self.assertTrue(all(float(x['state']['deter'][0, 0]) == x['timestamp'] for x in rows))
        self.assertTrue(all(x['observed_error'] is None for x in rows[:-1]))
        np.testing.assert_array_equal(rows[-1]['observed_error'], [.01])
        self.assertEqual(rows[0]['original_candidate_count'], 2)
        np.testing.assert_array_equal(rows[0]['weights'], [.5])

    def test_actual_and_overlapping_prospective_paths_are_separate(self):
        archive = self.create()
        archive.append(record(1))
        archive.append(record(2, kind='prospective', origin=1))
        archive.append(record(3, kind='prospective', origin=1))
        archive.append(record(2))
        archive.append(record(3, kind='prospective', origin=2))
        archive.append(record(3))
        rows = list(archive.reconstruct(episode_id=3, candidate_id='branch0', start_timestamp=1, end_timestamp=3))
        self.assertEqual([x['kind'] for x in rows], ['actual']*3)
        planned = list(archive.reconstruct(episode_id=3, candidate_id='branch0', kind='prospective',
            origin_timestamp=1, start_timestamp=2, end_timestamp=3))
        self.assertEqual([x['depth'] for x in planned], [1, 2])
        with self.assertRaises(ValueError):
            list(archive.reconstruct(episode_id=3, candidate_id='branch0', kind='prospective', start_timestamp=2, end_timestamp=3))

    def test_restore_can_read_prefix_but_cannot_overwrite_later_evidence(self):
        archive = self.create()
        archive.append(record(1)); prefix = archive.snapshot()
        archive.append(record(2)); full = archive.snapshot()
        archive.close()
        before = (self.root/'manifest.jsonl').read_bytes()
        restored = self.restore(prefix)
        self.assertEqual(restored.record_count, 1)
        with self.assertRaises(IndexError): restored.load_record(1)
        with self.assertRaises(RuntimeError): restored.append(record(3))
        with self.assertRaises(ValueError): self.restore(prefix, writable=True)
        self.assertEqual((self.root/'manifest.jsonl').read_bytes(), before)
        resumed = self.restore(full, writable=True)
        resumed.append(record(3))
        self.assertEqual(resumed.record_count, 3)

    def test_external_binding_and_snapshot_tampering_rejected(self):
        archive = self.create(); archive.append(record()); saved = archive.snapshot()
        altered = copy.deepcopy(saved); altered['record_count'] += 1
        with self.assertRaises(ValueError): self.restore(altered)
        with self.assertRaises(ValueError):
            TrajectoryArchive.restore(self.root, saved, expected_source_sha256='f'*64, expected_protocol_sha256=PROTOCOL)

    def test_changed_npz_is_rejected_on_load_and_restore(self):
        archive = self.create(); entry = archive.append(record()); saved = archive.snapshot()
        path = self.root/entry['data_path']
        with path.open('ab') as stream: stream.write(b'corruption')
        with self.assertRaises(ValueError): archive.load_record(0)
        with self.assertRaises(ValueError): self.restore(saved)

    def test_missing_dependency_and_metadata_corruption_rejected(self):
        archive = self.create(); entry = archive.append(record()); saved = archive.snapshot()
        path = self.root/entry['metadata_path']
        before = path.read_bytes()
        path.write_bytes(before + b' ')
        with self.assertRaises(ValueError): self.restore(saved)
        path.write_bytes(before)
        (self.root/entry['data_path']).unlink()
        with self.assertRaises(FileNotFoundError): self.restore(saved)

    def test_manifest_truncation_chain_order_and_changed_header_rejected(self):
        archive = self.create(); archive.append(record()); saved = archive.snapshot()
        path = self.root/'manifest.jsonl'; manifest = path.read_bytes()
        path.write_bytes(manifest[:-1])
        with self.assertRaises(ValueError): self.restore(saved)
        path.write_bytes(manifest.replace(b'"sequence":0', b'"sequence":1'))
        with self.assertRaises(ValueError): self.restore(saved)
        path.write_bytes(manifest)
        (self.root/'header.json').write_bytes((self.root/'header.json').read_bytes()+b' ')
        with self.assertRaises(ValueError): self.restore(saved)

    def test_single_writer_os_lock_and_create_no_overwrite(self):
        archive = self.create(); saved = archive.snapshot()
        with self.assertRaises(RuntimeError): self.restore(saved, writable=True)
        with self.assertRaises(FileExistsError): self.create()
        archive.close()
        resumed = self.restore(saved, writable=True)
        resumed.append(record())
        self.assertEqual(resumed.record_count, 1)

    def test_interrupted_append_preserves_orphan_and_publishes_no_manifest(self):
        archive = self.create()
        with patch('helixwm_final.trajectory.np.savez_compressed', side_effect=OSError('disk full fixture')):
            with self.assertRaises(OSError): archive.append(record())
        self.assertEqual(archive.record_count, 0)
        self.assertEqual((self.root/'manifest.jsonl').read_bytes(), b'')
        orphan = self.root/'records/000000/000000000.npz'
        self.assertTrue(orphan.exists())
        with self.assertRaises(FileExistsError): archive.append(record())
        self.assertTrue(orphan.exists())

    def test_absent_predictions_remain_null_not_reconstructed_ground_truth(self):
        data = record()
        for key in ('predicted_observation', 'predicted_reward', 'predicted_continuation', 'probabilities'):
            data[key] = None
        for key in ('predictive_entropy', 'expected_entropy', 'model_conditional_disagreement', 'uncertainty'):
            data['metrics'][key] = None
        self.assertTrue(validate_record(data))
        archive = self.create(); archive.append(data)
        self.assertIsNone(archive.load_record(0)['predicted_observation'])
        data['observed_error'] = np.zeros(2, np.float64)
        with self.assertRaises(ValueError): validate_record(data)

    def test_schema_rejects_bad_actions_metrics_units_and_fake_confidence(self):
        cases = []
        data = record(); data['actions'][1, 0] = .5; cases.append(data)
        data = record(); data['metrics']['ess_ratio'] = .1; cases.append(data)
        data = record(); data['metrics']['predictive_entropy'] = .1; cases.append(data)
        data = record(); data['predicted_continuation'][0] = 1.1; cases.append(data)
        data = record(); data['calibration_confidence'] = .99; cases.append(data)
        data = record(); data['state']['deter'][0, 0] = np.nan; cases.append(data)
        data = record(); data['model_version'] = True; cases.append(data)
        data = record(); data['missing_age'] = 101; cases.append(data)
        data = record(); data['real_state'] = None; cases.append(data)
        data = record(kind='prospective', t=2, origin=1); data['observation_available'] = True; cases.append(data)
        data = record(); data['weights'] = np.array([.4, .4]); cases.append(data)
        for data in cases:
            with self.subTest(data=data['kind']):
                with self.assertRaises(ValueError): validate_record(data)

    def test_missing_duplicate_segment_timestamp_is_not_padded(self):
        archive = self.create()
        archive.append(record(1)); archive.append(record(3))
        with self.assertRaises(ValueError):
            list(archive.reconstruct(episode_id=3, candidate_id='branch0', start_timestamp=1, end_timestamp=3))
        with self.assertRaises(ValueError):
            list(archive.reconstruct(episode_id=3, candidate_id='branch0', start_timestamp=1, end_timestamp=101))

    def test_manifest_prefix_load_never_reads_later_corrupted_tail(self):
        archive = self.create(); archive.append(record(1)); snapshot = archive.snapshot()
        archive.append(record(2)); archive.close()
        with (self.root/'manifest.jsonl').open('ab') as stream: stream.write(b'partial next record')
        restored = self.restore(snapshot)
        self.assertEqual(restored.load_record(0)['timestamp'], 1)
        with self.assertRaises(ValueError): self.restore(snapshot, writable=True)


if __name__ == '__main__':
    unittest.main()
