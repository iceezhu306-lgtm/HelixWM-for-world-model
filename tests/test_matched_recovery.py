"""CPU failure-path tests; these cannot satisfy the real GPU recovery gate."""
import copy
import hashlib
import random
from pathlib import Path
import tempfile
import unittest

import numpy as np

from helixwm_matched.recovery_probe import (
    COMPARISON_FIELDS, RESTORED_FIELDS, compare_results, jsonable,
    materialize_readonly_replay, sample_descriptor, tree_hash, validate_payload)


def fixture():
    cfg = {'training': {'actions_per_arm': 1000000}}
    identity = {'protocol_sha256': 'p', 'source_sha256': 's',
                'environment_lock_sha256': 'e', 'upstream_commit': 'u'}
    counts = {'agent_steps': 6000, 'env_steps': 12000, 'episodes': 12, 'update_steps': 250}
    payload = {'format': 'helixwm_matched_episode_boundary_v1', 'identity': identity,
               'next_operation': 'reset_new_episode_and_reanchor', 'counters': counts,
               'backbone': {'training': {'counters': {'updates': 250, 'batches': 250}},
                            'writeback_updates': 250, 'inference_version': 250},
               'replay': {'rows_added': 6012},
               'completed_controller': {'controller': 'helix_v2', 'chain': {
                   'state': {'episode_id': 11, 'timestamp': 500, 'anchor_time': 500,
                             'last_action': None, 'pending': None, 'missing_length': 0,
                             'abstaining': False},
                   'rngs': {k: np.random.default_rng(i).bit_generator.state
                            for i, k in enumerate(('real', 'actor', 'dream', 'repair'))}}},
               'prefill_rng': np.random.default_rng(810000).bit_generator.state,
               'numpy_global_rng': np.random.RandomState(0).get_state(),
               'python_rng': random.Random(0).getstate()}
    return payload, {'identity': identity, **counts}, cfg, identity


def worker_result():
    comparison = {k: 'typed-test-evidence' for k in COMPARISON_FIELDS}
    comparison['metrics'] = {'loss': 1.0}
    comparison['metrics_sha256'] = tree_hash(comparison['metrics'])
    return {'passed': True, 'all_devices_gpu': True, 'original_files_unchanged': True,
            'actual_counts_complete': True, 'probe_actions': 4, 'probe_dmc_calls': 8,
            'probe_updates': 1, 'restored_comparisons': {k: True for k in RESTORED_FIELDS},
            'comparison': comparison}


class PayloadTests(unittest.TestCase):
    def test_valid_boundary_without_touching_host_rng(self):
        payload, record, cfg, identity = fixture()
        before = tree_hash((np.random.get_state(), random.getstate()))
        self.assertEqual(validate_payload(payload, record, cfg, identity)['agent_steps'], 6000)
        self.assertEqual(before, tree_hash((np.random.get_state(), random.getstate())))

    def test_wrong_source_or_counter_rejected(self):
        for mutation in ('source', 'episodes', 'updates', 'rows', 'inference'):
            payload, record, cfg, identity = fixture()
            payload = copy.deepcopy(payload)
            if mutation == 'source': payload['identity']['source_sha256'] = 'different'
            elif mutation == 'episodes': payload['counters']['episodes'] = 11
            elif mutation == 'updates': payload['backbone']['writeback_updates'] = 249
            elif mutation == 'rows': payload['replay']['rows_added'] = 6000
            else: payload['backbone']['inference_version'] = 249
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate_payload(payload, record, cfg, identity)

    def test_pending_action_gap_and_rng_fail_closed(self):
        for mutation in ('action', 'gap', 'rng'):
            payload, record, cfg, identity = fixture()
            if mutation == 'action': payload['completed_controller']['chain']['state']['last_action'] = [0]*6
            elif mutation == 'gap': payload['completed_controller']['chain']['state']['pending'] = {'x': 1}
            else: del payload['completed_controller']['chain']['rngs']['repair']
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate_payload(payload, record, cfg, identity)

    def test_pretraining_and_final_checkpoints_not_misrepresented(self):
        for steps in (1000, 1000000):
            payload, record, cfg, identity = fixture(); payload['counters']['agent_steps'] = steps
            with self.assertRaises(ValueError): validate_payload(payload, record, cfg, identity)


class ComparisonTests(unittest.TestCase):
    def test_exact_fresh_process_match_only(self):
        a = worker_result()
        report = compare_results([a, copy.deepcopy(a)], [0, 0])
        self.assertTrue(report['passed'])
        self.assertEqual(report['actual_probe_actions'], 8)
        self.assertEqual(report['actual_probe_updates'], 2)
        self.assertFalse(report['uninterrupted_live_process_comparison_performed'])
        self.assertFalse(report['formal_training_resumed'])

    def test_tiny_metric_or_optimizer_hash_difference_fails(self):
        for field in ('after_training_tree_sha256', 'metrics', 'next_mask_sha256',
                      'restored_collector_rng_sha256', 'sample', 'after_global_rng'):
            a = worker_result(); b = copy.deepcopy(a)
            b['comparison'][field] = {'loss': 1.0000000000000002} if field == 'metrics' else 'different'
            report = compare_results([a, b], [0, 0])
            self.assertFalse(report['passed']); self.assertIn(field, report['mismatched_fields'])

    def test_cpu_nonzero_exit_or_partial_evidence_cannot_pass(self):
        for mutation in ('cpu', 'exit', 'schema', 'restore'):
            a = worker_result(); b = copy.deepcopy(a); exits = [0, 0]
            if mutation == 'cpu': b['all_devices_gpu'] = False
            elif mutation == 'exit': exits[1] = 1
            elif mutation == 'schema': del b['comparison']['prefill_rng_sha256']
            else: del b['restored_comparisons']['train_carry']
            with self.subTest(mutation=mutation):
                self.assertFalse(compare_results([a, b], exits)['passed'])

    def test_native_batch_schema(self):
        b = {'consec': np.zeros((16, 65), np.int32), 'stepid': np.zeros((16, 65, 20), np.uint8)}
        desc = sample_descriptor(b)
        self.assertEqual(desc['learning_positions'], 1024)
        b['stepid'] = b['stepid'][:, 1:]
        with self.assertRaises(ValueError): sample_descriptor(b)

    def test_tree_hash_retains_dtype_negative_zero_and_large_rng_integer(self):
        self.assertNotEqual(tree_hash(np.array([0], np.float32)), tree_hash(np.array([-0.], np.float32)))
        self.assertNotEqual(tree_hash(np.array([1], np.float32)), tree_hash(np.array([1], np.float64)))
        self.assertNotEqual(tree_hash(np.array(1, np.float32)), tree_hash(np.array([1], np.float32)))
        self.assertEqual(tree_hash(2**127), tree_hash(2**127))
        self.assertEqual(jsonable({'a': np.array([1.], np.float32)}), {'a': [1.0]})


class StorageTests(unittest.TestCase):
    def test_probe_reference_preserves_original_and_rejects_tampering(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); source = root/'original'; source.mkdir()
            raw = b'immutable-example'; name = 'object.npz'
            (source/name).write_bytes(raw)
            record = {'path': name, 'bytes': len(raw), 'file_sha256': hashlib.sha256(raw).hexdigest()}
            result = materialize_readonly_replay(source, root/'probe', [record])
            self.assertEqual(result['hardlinks'] + result['copies'], 1)
            self.assertEqual((source/name).read_bytes(), raw)
            self.assertEqual((root/'probe'/name).read_bytes(), raw)
            record['file_sha256'] = 'bad'
            with self.assertRaises(ValueError): materialize_readonly_replay(source, root/'bad_probe', [record])

    def test_parent_traversal_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(ValueError):
                materialize_readonly_replay(temp, Path(temp)/'probe', [{'path': '../outside.npz'}])


if __name__ == '__main__': unittest.main()
