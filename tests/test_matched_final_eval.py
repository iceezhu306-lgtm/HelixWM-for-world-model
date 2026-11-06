"""CPU-only evaluator regression gates; no model, simulator, or training runs.

The historical episode is loaded from its SHA-pinned raw archive. Checkpoint
fixtures below are small synthetic dictionaries, never real model pickles.
These tests do not satisfy final GPU reload, freeze or 1280-case acceptance.
"""
from __future__ import annotations

import ast
import copy
from fractions import Fraction
import hashlib
import io
import json
import math
from pathlib import Path
import pickle
import random
import subprocess
import tempfile
import unittest
from unittest import mock
import zipfile

import numpy as np

from helixwm_matched import final_eval
from helixwm_matched.final_eval_validation import (
    reward_sum_audit, require_frozen_equal, validate_output_isolation,
    verify_file_integrity,
)
from helixwm_matched.protocol import fixed_gap_mask, mask_sha256
from helixwm_matched.recovery_probe import tree_hash

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / 'results/2026-10-29-matched-walker-development/data/failed_final_case.zip'
ARCHIVE_SHA = '8e642676ca55f764fb001b7a17384006c65b1547f06b245225ee3777f8dc9a08'
MEMBER_SHA = {
    'planned.json': 'b9136aca5795c0e47e66a86662b6b3c7d79ead479658910cd97ed91e26ec6364',
    'observed.json': 'e3efac6ca806aea81d7e1abd285aac62d69672ebee88d3bf306bbd70994daba9',
    'trace.npz': '2231a51d47919ba4467bfbff1f335328cc4d9891f4272a3e0466e81371d908ae',
}
FROZEN_COMMIT = '97e049941bd755fab68e85e8bb1171911fa6cac8'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def original_case():
    raw = ARCHIVE.read_bytes()
    if digest(raw) != ARCHIVE_SHA:
        raise AssertionError('Historical archive changed')
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        if archive.testzip() is not None or set(archive.namelist()) != set(MEMBER_SHA):
            raise AssertionError('Historical archive CRC/member set differs')
        members = {name: archive.read(name) for name in MEMBER_SHA}
    for name, value in members.items():
        if digest(value) != MEMBER_SHA[name]:
            raise AssertionError('Historical member changed: ' + name)
    row = json.loads(members['observed.json'])
    with np.load(io.BytesIO(members['trace.npz']), allow_pickle=False) as arrays:
        row['trace'] = {name: arrays[name].copy() for name in arrays.files}
    return row, json.loads(members['planned.json'])


def synthetic_case(gap=0):
    mask = np.ones(501, bool) if not gap else fixed_gap_mask(gap)
    controller = 'clean_native' if not gap else 'helix_v2'
    row = {
        'controller': controller, 'seed': 60000, 'gap': gap,
        'actions': 500, 'dmc_calls': 1000, 'episode_return': 500.0,
        'availability_sha256': mask_sha256(mask), 'repair_events': [],
        'trace': {
            'action': np.zeros((500, 6), np.float32),
            'reward': np.ones(500, np.float64),
            'dmc_calls': np.full(500, 2, np.int32),
            'availability_before_action': mask[:-1].copy(),
            'availability_outcome': mask[1:].copy(),
        },
    }
    planned = {
        'case_id': final_eval.case_id('H' if gap else 'D', controller, 60000, gap),
        'arm': 'H' if gap else 'D', 'controller': controller, 'seed': 60000,
        'gap': gap, 'actions': 500, 'dmc_calls': 1000,
        'availability_sha256': mask_sha256(mask),
        'missing_start': 150 if gap else None, 'record_video_frames': False,
    }
    return row, planned


def final_payload(arm='D'):
    """Synthetic final-shaped payload; no model parameters are loaded."""
    identity = {'protocol_sha256': 'test-protocol', 'source_sha256': 'test-source',
                'environment_lock_sha256': 'test-environment', 'upstream_commit': 'test-upstream'}
    counts = {'agent_steps': 1000000, 'env_steps': 2000000,
              'episodes': 2000, 'update_steps': 248750}
    payload = {
        'format': 'helixwm_matched_episode_boundary_v1',
        'identity': copy.deepcopy(identity), 'counters': copy.deepcopy(counts),
        'next_operation': 'reset_new_episode_and_reanchor',
        'backbone': {
            'training': {'counters': {'updates': 248750, 'batches': 248750},
                         'params': {'model': np.array([1.0], np.float32)}},
            'writeback_updates': 248750, 'inference_version': 248750,
            'resolved_config': {'seed': 0}, 'train_carry': np.zeros(2, np.float32),
            'inference_params': {'model': np.array([1.0], np.float32)},
        },
        'completed_controller': {
            'controller': 'black_frame_native' if arm == 'D' else 'helix_v2',
            'chain': {'state': {'episode_id': 1999, 'timestamp': 500,
                      'anchor_time': 500, 'last_action': None, 'pending': None,
                      'missing_length': 0, 'abstaining': False},
                      'rngs': {name: np.random.default_rng(i).bit_generator.state
                               for i, name in enumerate(('real', 'actor', 'dream', 'repair'))}},
        },
        'replay': {'rows_added': 1002000, 'sampler_rng': 'synthetic'},
        'prefill_rng': np.random.default_rng(810000).bit_generator.state,
        'numpy_global_rng': np.random.RandomState(0).get_state(),
        'python_rng': random.Random(0).getstate(),
    }
    record = {'identity': copy.deepcopy(identity), **counts, 'replay_files': []}
    config = json.loads((ROOT/'configs/dreamerv3_helixwm_matched_20261026.json').read_text())
    return payload, record, config, identity


def write_synthetic_checkpoint(root):
    payload, record, cfg, identity = final_payload()
    cp = root/'run'/'CHECKPOINTS'/'checkpoint_001000000.pkl'
    cp.parent.mkdir(parents=True)
    cp.write_bytes(pickle.dumps(payload))
    record.update(file='CHECKPOINTS/'+cp.name, bytes=cp.stat().st_size,
                  sha256=digest(cp.read_bytes()))
    cp.with_suffix('.json').write_text(json.dumps(record))
    (cp.parent.parent/'train.exit').write_text('0\n')
    (cp.parent.parent/'completion.json').write_text(json.dumps({
        'training_complete': True, 'stage': 'formal', 'arm': 'D',
        'identity': identity, 'final_checkpoint': {'sha256': record['sha256']},
        **payload['counters'],
    }))
    return cp, payload, record, cfg, identity


class HistoricalRegressionTests(unittest.TestCase):
    def test_original_raw_case_has_exact_provenance_and_reproduces_six_ulps(self):
        row, planned = original_case()
        old_source = subprocess.check_output(
            ['git', '-C', str(ROOT), 'show', FROZEN_COMMIT+':helixwm_matched/final_eval.py'],
            text=True, encoding='utf-8')
        function = next(n for n in ast.parse(old_source).body
                        if isinstance(n, ast.FunctionDef) and n.name == 'check_episode')
        namespace = {'np': np, 'fixed_gap_mask': fixed_gap_mask}
        exec(compile(ast.Module(body=[function], type_ignores=[]), '<frozen-check-only>', 'exec'), namespace)
        with self.assertRaisesRegex(ValueError, 'Episode return'):
            namespace['check_episode'](row, planned)
        old_sum = float(sum(row['trace']['reward']))
        self.assertEqual((old_sum-row['episode_return'])/math.ulp(row['episode_return']), 6.0)
        before = tree_hash(row)
        audit = final_eval.check_episode(row, planned)
        self.assertTrue(audit['passed'])
        self.assertEqual(audit['canonical_return'], 989.9474279803866)
        self.assertEqual(before, tree_hash(row))
        self.assertEqual(digest(ARCHIVE.read_bytes()), ARCHIVE_SHA)

    def test_both_legacy_reduction_results_are_within_declared_bound(self):
        row, planned = original_case()
        for recorded in (989.9474279803866, 989.9474279803873):
            with self.subTest(recorded=recorded):
                row['episode_return'] = recorded
                audit = final_eval.check_episode(row, planned)
                self.assertEqual(audit['recorded_return'], recorded)
                self.assertLessEqual(audit['absolute_error'], audit['absolute_error_bound'])

    def test_original_case_valid_range_reward_change_is_not_rounding(self):
        row, planned = original_case()
        row['trace']['reward'][0] += 1e-6
        self.assertLess(row['trace']['reward'][0], 2)
        with self.assertRaises(ValueError): final_eval.check_episode(row, planned)


class RewardSumTests(unittest.TestCase):
    def test_normal_zero_integer_fractional_and_boundary_arrays(self):
        arrays = (np.zeros(500, np.float64), np.ones(500, np.float64),
                  np.full(500, 2.0, np.float64),
                  np.linspace(0.0, 2.0, 500, dtype=np.float64))
        for rewards in arrays:
            expected = math.fsum(map(float, rewards))
            before = rewards.tobytes()
            audit = reward_sum_audit(rewards, expected)
            self.assertTrue(audit['passed'])
            self.assertEqual(audit['count'], 500)
            self.assertEqual(audit['canonical_return'], expected)
            self.assertEqual(before, rewards.tobytes())
            self.assertGreaterEqual(audit['absolute_error_bound'], 0)

    def test_nonfinite_reward_and_return_rejected(self):
        for invalid in (float('nan'), float('inf'), float('-inf')):
            with self.subTest(kind='reward', value=invalid):
                rewards = np.ones(500, np.float64); rewards[10] = invalid
                with self.assertRaises(ValueError): reward_sum_audit(rewards, 500.0)
            with self.subTest(kind='return', value=invalid):
                with self.assertRaises(ValueError): reward_sum_audit(np.ones(500, np.float64), invalid)

    def test_wrong_reward_schema_and_scalar_types_rejected(self):
        for rewards in (np.ones(500, np.float32), np.ones((500, 1), np.float64),
                        np.ones(500, np.complex128), np.ones(500, object),
                        np.empty(0, np.float64)):
            with self.subTest(dtype=str(rewards.dtype), shape=rewards.shape):
                with self.assertRaises((TypeError, ValueError)): reward_sum_audit(rewards, 500.0)
        for value in (True, '500.0', 500, np.array([500.0]), np.float32(500)):
            with self.subTest(type=type(value).__name__):
                with self.assertRaises((TypeError, ValueError)): reward_sum_audit(np.ones(500, np.float64), value)
        self.assertTrue(reward_sum_audit(np.ones(500, np.float64), np.float64(500))['passed'])

    def test_reward_bounds_and_material_return_error_rejected(self):
        for value in (-1e-12, 2.000000000001):
            rewards = np.ones(500, np.float64); rewards[10] = value
            with self.assertRaises(ValueError): reward_sum_audit(rewards, math.fsum(map(float, rewards)))
        with self.assertRaises(ValueError): reward_sum_audit(np.ones(500, np.float64), 500.000001)

    def test_aggregate_return_bounds_reject_even_sub_bound_errors(self):
        # A numerical consistency allowance cannot relax the task's range.
        scenarios = (
            (np.full(500, math.ulp(0.0), np.float64), -math.ulp(0.0)),
            (np.full(500, 2.0, np.float64), math.nextafter(1000.0, math.inf)),
        )
        for rewards, recorded in scenarios:
            with self.subTest(recorded=recorded), self.assertRaises(ValueError):
                reward_sum_audit(rewards, recorded)

    def test_actual_declared_bound_is_enforced_on_both_sides(self):
        rewards = np.ones(500, np.float64)
        audit = reward_sum_audit(rewards, 500.0)
        for direction in (-math.inf, math.inf):
            sign = -1 if direction < 0 else 1
            candidate = 500.0 + sign * audit['absolute_error_bound']
            while abs(candidate - 500.0) <= audit['absolute_error_bound']:
                candidate = math.nextafter(candidate, direction)
            with self.subTest(direction=direction), self.assertRaises(ValueError):
                reward_sum_audit(rewards, candidate)
            inside = math.nextafter(candidate, 500.0)
            self.assertLessEqual(abs(Fraction.from_float(inside) - 500),
                                 Fraction.from_float(audit['absolute_error_bound']))
            self.assertTrue(reward_sum_audit(rewards, inside)['passed'])

    def test_exact_fraction_oracle_for_subnormal_maximum_and_mixed_magnitudes(self):
        tiny = math.ulp(0.0)
        scenarios = {
            'subnormal': np.full(500, tiny, np.float64),
            'small_normal': np.full(500, np.finfo(np.float64).tiny, np.float64),
            'maximum': np.full(500, 2.0, np.float64),
            'mixed': np.asarray([2.0, tiny, 2.0**-50, 2.0**-500, 0.0]*100, np.float64),
            'historical': original_case()[0]['trace']['reward'],
        }
        for label, rewards in scenarios.items():
            with self.subTest(scenario=label):
                # Exact rational arithmetic is the independent sum oracle.
                exact = sum((Fraction.from_float(float(r)) for r in rewards), Fraction())
                audit = reward_sum_audit(rewards, float(exact))
                canonical_error = abs(Fraction.from_float(audit['canonical_return']) - exact)
                self.assertLessEqual(canonical_error,
                                     2 * Fraction.from_float(math.ulp(float(exact))))
                self.assertLess(audit['absolute_error_bound'], 1e-9)
                sequential = 0.0
                for value in rewards:
                    sequential += float(value)
                gamma = Fraction(499, 2**53 - 499)
                oracle_sequential_error_bound = gamma * exact + 500 * Fraction.from_float(tiny)
                self.assertLessEqual(abs(Fraction.from_float(sequential) - exact),
                                     oracle_sequential_error_bound)
                for recorded in (sequential, float(np.sum(rewards, dtype=np.float64))):
                    checked = reward_sum_audit(rewards, recorded)
                    exact_difference = abs(Fraction.from_float(recorded)
                                           - Fraction.from_float(checked['canonical_return']))
                    self.assertLessEqual(exact_difference,
                                         Fraction.from_float(checked['absolute_error_bound']))

    def test_cryptographic_layer_rejects_tiny_or_balanced_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'trusted_trace.npz'
            for mutation in ('tiny', 'balanced'):
                rewards = np.ones(500, np.float64)
                np.savez_compressed(path, reward=rewards)
                before_sha = digest(path.read_bytes()); before_size = path.stat().st_size
                verify_file_integrity(path, before_sha, before_size)
                if mutation == 'tiny': rewards[0] = np.nextafter(rewards[0], 2.0)
                else: rewards[0] += 0.125; rewards[1] -= 0.125
                # A sum is not a content identity check; this is expected.
                self.assertTrue(reward_sum_audit(rewards, 500.0)['passed'])
                np.savez_compressed(path, reward=rewards)
                with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                    verify_file_integrity(path, before_sha)


class EpisodeContractTests(unittest.TestCase):
    def test_all_registered_gaps_and_immutable_input(self):
        for gap in (0, 1, 5, 10, 20, 50):
            row, planned = synthetic_case(gap)
            before = tree_hash((row, planned))
            self.assertTrue(final_eval.check_episode(row, planned)['passed'])
            self.assertEqual(before, tree_hash((row, planned)))

    def test_identity_and_discrete_count_mismatches_rejected(self):
        changes = {'seed': 60001, 'controller': 'prior_only', 'gap': 5,
                   'actions': 499, 'dmc_calls': 999, 'availability_sha256': 'wrong'}
        for key, value in changes.items():
            row, planned = synthetic_case(); row[key] = value
            with self.subTest(field=key), self.assertRaises(ValueError):
                final_eval.check_episode(row, planned)

    def test_integer_identity_cannot_be_equal_float_or_bool(self):
        for field, value in (('seed',60000.0), ('gap',0.0), ('gap',False),
                             ('actions',500.0), ('dmc_calls',1000.0)):
            row, planned = synthetic_case(); row[field] = value
            with self.subTest(field=field, value=value), self.assertRaises((ValueError, TypeError)):
                final_eval.check_episode(row, planned)

    def test_truncation_extension_extra_missing_and_dtype_rejected(self):
        for kind in ('truncated', 'extended', 'missing', 'extra', 'dtype', 'action_shape'):
            row, planned = synthetic_case()
            if kind == 'truncated': row['trace']['reward'] = row['trace']['reward'][:-1]
            elif kind == 'extended': row['trace']['reward'] = np.ones(501, np.float64)
            elif kind == 'missing': del row['trace']['action']
            elif kind == 'extra': row['trace']['unexpected'] = np.zeros(500)
            elif kind == 'dtype': row['trace']['reward'] = row['trace']['reward'].astype(np.float32)
            else: row['trace']['action'] = np.zeros((500, 5), np.float32)
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                final_eval.check_episode(row, planned)

    def test_nonfinite_actions_and_reward_rejected(self):
        for field in ('action', 'reward'):
            for value in (float('nan'), float('inf'), float('-inf')):
                row, planned = synthetic_case(); row['trace'][field].flat[0] = value
                with self.subTest(field=field,value=value), self.assertRaises(ValueError):
                    final_eval.check_episode(row, planned)

    def test_actions_reward_and_native_call_bounds_rejected(self):
        for kind in ('action', 'reward_low', 'reward_high', 'calls'):
            row, planned = synthetic_case()
            if kind == 'action': row['trace']['action'][0, 0] = np.float32(1.0001)
            elif kind == 'reward_low': row['trace']['reward'][0] = -0.1
            elif kind == 'reward_high': row['trace']['reward'][0] = 2.1
            else: row['trace']['dmc_calls'][0] = 1
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                final_eval.check_episode(row, planned)

    def test_actual_mask_mismatch_and_one_step_shift_rejected(self):
        for field in ('availability_before_action', 'availability_outcome'):
            for shifted in (False, True):
                row, planned = synthetic_case(20)
                if shifted: row['trace'][field] = np.roll(row['trace'][field], 1)
                else: row['trace'][field][100] = False
                with self.subTest(field=field,shifted=shifted), self.assertRaises(ValueError):
                    final_eval.check_episode(row, planned)


class CheckpointGateTests(unittest.TestCase):
    def test_synthetic_final_payload_keeps_global_rng_and_supports_both_arms(self):
        for arm in ('D', 'H'):
            payload, record, cfg, identity = final_payload(arm)
            before = tree_hash((np.random.get_state(), random.getstate()))
            counts = final_eval.validate_checkpoint_payload(payload, record, cfg, identity, arm)
            self.assertEqual(counts['agent_steps'], 1000000)
            self.assertEqual(before, tree_hash((np.random.get_state(), random.getstate())))

    def test_wrong_checkpoint_identity_source_protocol_environment_rejected(self):
        for field in ('source_sha256','protocol_sha256','environment_lock_sha256','upstream_commit'):
            for location in ('payload', 'record'):
                payload, record, cfg, identity = final_payload()
                (payload if location=='payload' else record)['identity'][field] = 'wrong'
                with self.subTest(field=field,location=location), self.assertRaises(ValueError):
                    final_eval.validate_checkpoint_payload(payload, record, cfg, identity, 'D')

    def test_incomplete_budget_inference_wrong_arm_pending_and_replay_rejected(self):
        for kind in ('budget','native_updates','inference','wrong_arm','pending','replay','operation'):
            payload, record, cfg, identity = final_payload()
            if kind=='budget': payload['counters']['agent_steps'] = 990000
            elif kind=='native_updates': payload['backbone']['training']['counters']['updates'] -= 1
            elif kind=='inference': payload['backbone']['inference_version'] -= 1
            elif kind=='wrong_arm': payload['completed_controller']['controller']='helix_v2'
            elif kind=='pending': payload['completed_controller']['chain']['state']['pending']={'action':1}
            elif kind=='replay': payload['replay']['rows_added']-=1
            else: payload['next_operation']='continue_mid_episode'
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                final_eval.validate_checkpoint_payload(payload, record, cfg, identity, 'D')

    def test_checkpoint_seed_and_discrete_counter_types_rejected(self):
        for value in (1, False, 0.0):
            payload, record, cfg, identity = final_payload()
            payload['backbone']['resolved_config']['seed'] = value
            with self.subTest(seed=value, type=type(value).__name__), self.assertRaises((ValueError, TypeError)):
                final_eval.validate_checkpoint_payload(payload, record, cfg, identity, 'D')
        for location in ('payload', 'record', 'native', 'writeback', 'inference', 'replay', 'chain'):
            payload, record, cfg, identity = final_payload()
            if location == 'payload': payload['counters']['agent_steps'] = 1000000.0
            elif location == 'record': record['env_steps'] = 2000000.0
            elif location == 'native': payload['backbone']['training']['counters']['updates'] = 248750.0
            elif location == 'writeback': payload['backbone']['writeback_updates'] = 248750.0
            elif location == 'inference': payload['backbone']['inference_version'] = 248750.0
            elif location == 'replay': payload['replay']['rows_added'] = 1002000.0
            else: payload['completed_controller']['chain']['state']['timestamp'] = 500.0
            with self.subTest(location=location), self.assertRaises((ValueError, TypeError)):
                final_eval.validate_checkpoint_payload(payload, record, cfg, identity, 'D')
        payload, record, cfg, identity = final_payload('H')
        with self.assertRaises(ValueError):
            final_eval.validate_checkpoint_payload(payload, record, cfg, identity, 'UNKNOWN')

    def test_synthetic_checkpoint_load_and_completion_validation(self):
        with tempfile.TemporaryDirectory() as temporary:
            cp,payload,record,cfg,identity = write_synthetic_checkpoint(Path(temporary))
            with mock.patch('helixwm_matched.backbone.NativeReplay.referenced_files', return_value=[]):
                loaded, checked = final_eval.load_final_checkpoint(cp,cfg,identity,'D')
                self.assertEqual(tree_hash(loaded),tree_hash(payload))
                self.assertEqual(checked['sha256'],record['sha256'])
                completion=cp.parent.parent/'completion.json'
                bad=json.loads(completion.read_text());bad['final_checkpoint']['sha256']='wrong'
                completion.write_text(json.dumps(bad))
                with self.assertRaises(ValueError): final_eval.load_final_checkpoint(cp,cfg,identity,'D')

    def test_checkpoint_hash_and_size_fail_before_pickle_read(self):
        for kind in ('hash','size'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temporary:
                cp,payload,record,cfg,identity = write_synthetic_checkpoint(Path(temporary))
                if kind=='hash': record['sha256']='0'*64
                else: record['bytes']+=1
                cp.with_suffix('.json').write_text(json.dumps(record))
                with mock.patch('helixwm_matched.final_eval.pickle.load') as loader:
                    with self.assertRaises(ValueError): final_eval.load_final_checkpoint(cp,cfg,identity,'D')
                    loader.assert_not_called()

    def test_checksum_size_gate_and_original_file_preservation(self):
        with tempfile.TemporaryDirectory() as temporary:
            p=Path(temporary)/'evidence.bin';raw=b'unchanged-original-evidence';p.write_bytes(raw)
            verify_file_integrity(p,digest(raw),len(raw))
            for hash_value,size in ((digest(raw),len(raw)+1),('0'*64,len(raw))):
                with self.assertRaises(ValueError):verify_file_integrity(p,hash_value,size)
            self.assertEqual(p.read_bytes(),raw)


class FreezePlanIsolationTests(unittest.TestCase):
    def test_strict_frozen_fingerprint_rejects_each_state_change(self):
        payload,*_=final_payload()
        class FakeBackbone:
            def snapshot(self):return copy.deepcopy(payload['backbone'])
        before=final_eval.frozen_fingerprint(FakeBackbone())
        require_frozen_equal(before,copy.deepcopy(before))
        for key in before:
            after=copy.deepcopy(before)
            if isinstance(after[key],dict):after[key]['updates']+=1
            elif isinstance(after[key],int):after[key]+=1
            else:after[key]='changed'
            with self.subTest(field=key),self.assertRaises((ValueError,RuntimeError)):
                require_frozen_equal(before,after)
        with self.assertRaises((ValueError,RuntimeError)):
            require_frozen_equal({'metric':1.0},{'metric':np.nextafter(1.0,2.0)})

    def test_fixed_plan_counts_paired_masks_and_video(self):
        cfg=json.loads((ROOT/'configs/dreamerv3_helixwm_matched_20261026.json').read_text())
        plan=final_eval.build_plan(cfg)
        self.assertEqual(len(plan),1280)
        self.assertEqual(len({p['case_id']for p in plan}),1280)
        self.assertEqual(sum(p['actions']for p in plan),640000)
        self.assertEqual(sum(p['dmc_calls']for p in plan),1280000)
        self.assertEqual(sum(p['record_video_frames']for p in plan),3)
        for arm in ('D','H'):
            self.assertEqual(sum(p['arm']==arm for p in plan),640)
            self.assertEqual(sum(p['arm']==arm and p['controller']=='clean_native' for p in plan),40)
        for seed in range(60000,60040):
            for gap in (1,5,10,20,50):
                paired=[p for p in plan if p['seed']==seed and p['gap']==gap]
                self.assertEqual(len(paired),6)
                self.assertEqual(len({p['availability_sha256']for p in paired}),1)

    def test_output_isolation_rejects_existing_or_overlapping_locations(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            inputs={arm:root/('run_'+arm)/'CHECKPOINTS'/'final.pkl' for arm in ('D','H')}
            for p in inputs.values():p.parent.mkdir(parents=True);p.write_bytes(b'synthetic')
            validate_output_isolation(root/'fresh',inputs)
            (root/'existing').mkdir()
            for out in (root/'existing',root,root/'run_D',root/'run_H'/'new_child'):
                with self.subTest(path=out),self.assertRaises((ValueError,FileExistsError)):
                    validate_output_isolation(out,inputs)
            self.assertFalse((root/'fresh').exists())
            self.assertFalse((root/'run_H'/'new_child').exists())
            for invalid in ({'D': inputs['D'], 'H': inputs['D']}, {'D': inputs['D']}):
                with self.assertRaises(ValueError): validate_output_isolation(root/'fresh', invalid)


if __name__=='__main__':
    unittest.main()
