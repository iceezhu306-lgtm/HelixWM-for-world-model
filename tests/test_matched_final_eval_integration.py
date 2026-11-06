"""CPU-only mocked evaluator-flow tests, never a real GPU/evaluation run.

The production run's persistence and fail-closed flow are exercised with a
two-cell synthetic plan. The second cell deliberately stops; the production
1280-cell completion guard remains unchanged. All model/environment/device
objects are mocks; the numerical fixture is the SHA-pinned historical trace.
"""
from __future__ import annotations

import copy
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from helixwm_matched import final_eval
from helixwm_matched.protocol import protocol_sha256
from test_matched_final_eval import ROOT, final_payload, original_case


class DeliberateSyntheticStop(RuntimeError):
    pass


class FakeBackbone:
    def __init__(self, *args, **kwargs):
        # Only a test double; no JAX module/device discovery is imported/run.
        self.jax = SimpleNamespace(devices=lambda: [SimpleNamespace(platform='gpu')])

    def load(self, state):
        self.state = copy.deepcopy(state)

    def snapshot(self):
        return copy.deepcopy(self.state)


class EvaluatorFlowTests(unittest.TestCase):
    def exercise(self, defect=None):
        np_state, py_state = np.random.get_state(), random.getstate()
        try:
            with tempfile.TemporaryDirectory(prefix='mock_final_eval_') as temporary:
                base = Path(temporary)
                cfg = ROOT/'configs/dreamerv3_helixwm_matched_20261026.json'
                lock = base/'environment_lock.json'
                lock.write_text(json.dumps({'test_fixture': 'mock_environment_only'}))
                args = SimpleNamespace(protocol=cfg, upstream=base/'unused_upstream',
                    checkpoint_d=base/'run_D'/'CHECKPOINTS'/'final.pkl',
                    checkpoint_h=base/'run_H'/'CHECKPOINTS'/'final.pkl',
                    environment_lock=lock, output_dir=base/'isolated_mock_output')
                cp_bytes = {}
                for arm, cp in (('D', args.checkpoint_d), ('H', args.checkpoint_h)):
                    cp.parent.mkdir(parents=True)
                    cp_bytes[arm] = ('not-a-real-checkpoint-'+arm).encode()
                    cp.write_bytes(cp_bytes[arm]); cp.with_suffix('.json').write_text('{}')
                row, first = original_case()
                first['record_video_frames'] = False
                second = copy.deepcopy(first)
                second.update(seed=60001, case_id='D_clean_native_seed60001_gap00')
                events = []
                original_return = row['episode_return']
                original_rewards = row['trace']['reward'].copy()

                def load_cp(path, config, identity, arm):
                    payload, record, _, _ = final_payload(arm)
                    record.update(sha256=hashlib.sha256(cp_bytes[arm]).hexdigest(),
                                  bytes=len(cp_bytes[arm]))
                    return payload, record

                def fake_episode(backbone, config, controller, seed, gap, **kwargs):
                    events.append('episode_'+str(seed))
                    if seed == 60001:
                        self.assertTrue((args.output_dir/'CASES'/first['case_id']/'summary.json').exists())
                        raise DeliberateSyntheticStop('intentional CPU-only test stop before second cell')
                    result = copy.deepcopy(row)
                    if defect == 'nan': result['trace']['reward'][0] = float('nan')
                    elif defect == 'inf': result['trace']['reward'][0] = float('inf')
                    elif defect == 'reward': result['trace']['reward'][0] += 1e-6
                    elif defect == 'freeze': backbone.state['training']['params']['model'][0] += np.float32(0.125)
                    return result

                def fake_lock(config):
                    events.append('environment_gate')
                    self.assertFalse((args.output_dir/'CASES'/first['case_id']/'summary.json').exists())
                    self.assertFalse((args.output_dir/'episodes.jsonl').exists())
                    return {'test_fixture': 'changed' if defect == 'environment' else 'mock_environment_only'}

                real_fingerprint = final_eval.frozen_fingerprint
                def observed_fingerprint(backbone):
                    events.append('freeze_gate')
                    return real_fingerprint(backbone)

                with ExitStack() as stack:
                    stack.enter_context(mock.patch.object(final_eval, 'build_plan', return_value=[first, second]))
                    stack.enter_context(mock.patch.object(final_eval, 'load_final_checkpoint', side_effect=load_cp))
                    stack.enter_context(mock.patch.object(final_eval, 'frozen_fingerprint', side_effect=observed_fingerprint))
                    stack.enter_context(mock.patch('helixwm_matched.backbone.MatchedDreamerBackbone', FakeBackbone))
                    episode_mock = stack.enter_context(mock.patch('helixwm_matched.evaluation.evaluate_episode', side_effect=fake_episode))
                    stack.enter_context(mock.patch('helixwm_matched.env_probe.runtime_lock', side_effect=fake_lock))
                    # If evaluation accidentally reaches a training entry point,
                    # fail without constructing an environment or learner.
                    stack.enter_context(mock.patch('helixwm_matched.environment.MatchedWalkerEnv', side_effect=AssertionError('real environment forbidden')))
                    stack.enter_context(mock.patch.object(final_eval, 'make_video', side_effect=AssertionError('real video work forbidden')))
                    if defect is None:
                        with self.assertRaises(DeliberateSyntheticStop): final_eval.run(args)
                    else:
                        with self.assertRaises((ValueError, RuntimeError)): final_eval.run(args)
                    calls = episode_mock.call_count
                output = args.output_dir
                failure = json.loads((output/'failure.json').read_text())
                case_dir = output/'CASES'/first['case_id']
                observed = json.loads((case_dir/'observed.json').read_text())
                with np.load(case_dir/'trace.npz', allow_pickle=False) as trace:
                    saved_rewards = trace['reward'].copy()
                result = {'failure': failure, 'observed': observed, 'saved_rewards': saved_rewards,
                    'original_rewards': original_rewards, 'original_return': original_return,
                    'events': events, 'episode_calls': calls,
                    'exit': (output/'evaluation.exit').read_text().strip(),
                    'has_completion': (output/'completion.json').exists(),
                    'has_summary': (case_dir/'summary.json').exists(),
                    'has_episodes': (output/'episodes.jsonl').exists(),
                    'return_audit': json.loads((case_dir/'return_validation.json').read_text())
                        if (case_dir/'return_validation.json').exists() else None,
                    'summary': json.loads((case_dir/'summary.json').read_text())
                        if (case_dir/'summary.json').exists() else None,
                    'original_checkpoint_bytes_preserved': all(cp.read_bytes() == cp_bytes[arm]
                        for arm, cp in (('D', args.checkpoint_d), ('H', args.checkpoint_h)))}
                return result
        finally:
            np.random.set_state(np_state); random.setstate(py_state)

    def test_historical_trace_accepted_only_after_frozen_and_environment_gates(self):
        result = self.exercise()
        self.assertEqual(result['episode_calls'], 2)
        self.assertEqual(result['events'][:5],
            ['freeze_gate', 'episode_60000', 'freeze_gate', 'environment_gate', 'episode_60001'])
        self.assertEqual(result['failure']['completed_episodes'], 1)
        self.assertEqual(result['failure']['acknowledged_actions'], 500)
        self.assertEqual(result['failure']['acknowledged_dmc_calls'], 1000)
        self.assertEqual(result['observed']['episode_return'], result['original_return'])
        np.testing.assert_array_equal(result['saved_rewards'], result['original_rewards'])
        self.assertTrue(result['return_audit']['passed'])
        self.assertEqual(result['return_audit'], result['summary']['return_validation'])
        self.assertTrue(result['summary']['frozen']['passed'])
        self.assertEqual(result['summary']['actual_environment_lock_sha256'],
                         protocol_sha256({'test_fixture': 'mock_environment_only'}))
        self.assertFalse(result['failure']['automatic_retry_or_resume'])
        self.assertEqual(result['failure']['failed_case'], 'D_clean_native_seed60001_gap00')
        self.assertEqual(result['exit'], '1')
        self.assertFalse(result['has_completion'])
        self.assertTrue(result['original_checkpoint_bytes_preserved'])

    def test_corrupt_rewards_fail_closed_before_acceptance_or_second_episode(self):
        for defect in ('nan', 'inf', 'reward'):
            with self.subTest(defect=defect):
                result = self.exercise(defect)
                self.assertEqual(result['episode_calls'], 1)
                self.assertEqual(result['failure']['completed_episodes'], 0)
                self.assertEqual(result['failure']['failed_case'], 'D_clean_native_seed60000_gap00')
                self.assertFalse(result['failure']['automatic_retry_or_resume'])
                self.assertNotIn('environment_gate', result['events'])
                self.assertFalse(result['has_summary'])
                self.assertFalse(result['has_episodes'])
                self.assertIsNone(result['return_audit'])
                self.assertEqual(result['exit'], '1')
                self.assertTrue(result['original_checkpoint_bytes_preserved'])

    def test_model_or_environment_change_fails_after_numeric_gate_before_acceptance(self):
        for defect in ('freeze', 'environment'):
            with self.subTest(defect=defect):
                result = self.exercise(defect)
                self.assertEqual(result['episode_calls'], 1)
                self.assertTrue(result['return_audit']['passed'])
                self.assertEqual(result['failure']['completed_episodes'], 0)
                self.assertEqual(result['failure']['acknowledged_actions'], 500)
                self.assertEqual(result['failure']['acknowledged_dmc_calls'], 1000)
                self.assertFalse(result['failure']['automatic_retry_or_resume'])
                self.assertFalse(result['has_summary'])
                self.assertFalse(result['has_episodes'])
                self.assertFalse(result['has_completion'])
                self.assertEqual(result['exit'], '1')
                if defect == 'freeze': self.assertNotIn('environment_gate', result['events'])
                else: self.assertIn('environment_gate', result['events'])

    def test_evaluator_source_manifest_binds_new_validation_helper(self):
        manifest = final_eval.source_manifest()
        root = Path(final_eval.__file__).resolve().parents[1]
        for name in ('helixwm_matched/final_eval.py', 'helixwm_matched/final_eval_validation.py'):
            self.assertEqual(manifest['files'][name], hashlib.sha256((root/name).read_bytes()).hexdigest())
        self.assertEqual(manifest['manifest_sha256'], protocol_sha256(manifest['files']))


if __name__ == '__main__': unittest.main()
