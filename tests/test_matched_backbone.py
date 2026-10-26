"""CPU contract tests; native Replay tests use the unmodified pinned source.

Set DREAMERV3_ARCHIVE to the verified e3f0224 source tar.gz to run source-native
Replay/Consec tests without importing JAX. These tests stub only elements utility
objects, not replay/chunk/selector/stream algorithms. A real GPU update and two
fresh-process restore are separate launch gates, never inferred from these tests.
"""
from contextlib import ContextDecorator, nullcontext
import copy
import os
from pathlib import Path
import sys
import tarfile
import threading
import types
import unittest
from unittest import mock
import uuid

import numpy as np

from helixwm_matched.backbone import (MatchedDreamerBackbone, NativeReplay,
                                    _array_id, _encode_array, _decode_array,
                                    _validate_tree, UPSTREAM_COMMIT)


def space(dtype, shape=()):
    return types.SimpleNamespace(dtype=np.dtype(dtype), shape=tuple(shape))


class CodecTests(unittest.TestCase):
    def test_exact_codec_roundtrip_including_negative_zero(self):
        a = np.array([0, -0.0, 1, -2.5, 0.125], np.float32)
        codec, packed = _encode_array(a)
        self.assertEqual(codec, 'float32_upper16_exact')
        b = _decode_array(codec, packed)
        np.testing.assert_array_equal(a.view(np.uint32), b.view(np.uint32))
        self.assertEqual(_array_id(a), _array_id(b))

    def test_non_bf16_value_never_rounded(self):
        a = np.array([0.1234567], np.float32)
        codec, packed = _encode_array(a)
        self.assertEqual(codec, 'identity')
        self.assertEqual(_array_id(a), _array_id(_decode_array(codec, packed)))

    def test_invalid_codec_dtype_and_checkpoint_dtype_rejected(self):
        with self.assertRaises(ValueError):
            _decode_array('float32_upper16_exact', np.ones(1, np.float32))
        with self.assertRaises(ValueError):
            _validate_tree({'x': np.zeros(1, np.float32)}, {'x': np.zeros(1, np.float64)})

    def test_replay_entries_are_real_state_copies(self):
        b = object.__new__(MatchedDreamerBackbone)
        b.config = types.SimpleNamespace(agent=types.SimpleNamespace(
            dyn=types.SimpleNamespace(rssm=types.SimpleNamespace(deter=2, stoch=1, classes=2))))
        state = {'deter': np.array([[1, 2]], np.float32),
                 'stoch': np.array([[[0, 1]]], np.float32)}
        entries = b.replay_entries(state)
        np.testing.assert_array_equal(entries['dyn/deter'], state['deter'][0])
        entries['dyn/deter'][0] = 99
        self.assertEqual(state['deter'][0, 0], 1)


class Counter:
    def __init__(self): self.value = 0
    def __int__(self): return self.value
    def increment(self): self.value += 1


class GuardedDeviceArray:
    """Reproduce the actual GPU failure: implicit host transfer is forbidden."""
    def __init__(self, shape=(1,), dtype=np.float32):
        self.shape, self.dtype = shape, np.dtype(dtype)
    def __array__(self, *args, **kwargs):
        raise AssertionError('Disallowed implicit device-to-host transfer')


class LoadTransferGuardTests(unittest.TestCase):
    def fixture(self):
        b = object.__new__(MatchedDreamerBackbone)
        b.resolved_config = {'seed': 0, 'logdir': 'new_probe_path'}
        b.agent = types.SimpleNamespace(
            params={'pol/w': GuardedDeviceArray(), 'opt/step': GuardedDeviceArray()},
            policy_keys=['pol/w'], train_sharded='train_sharded', train_mirrored='train_mirrored',
            pending_outs=None, pending_mets=None, load=mock.Mock())
        b.train_carry = ({}, {'deter': GuardedDeviceArray()}, {}, {'action': GuardedDeviceArray()})
        def explicit_get(value):
            if isinstance(value, GuardedDeviceArray): return np.zeros(value.shape, value.dtype)
            if isinstance(value, dict): return {k: explicit_get(v) for k, v in value.items()}
            if isinstance(value, tuple): return tuple(explicit_get(v) for v in value)
            raise AssertionError('Unexpected explicit transfer input')
        b.jax = types.SimpleNamespace(device_get=mock.Mock(side_effect=explicit_get),
                                     block_until_ready=lambda _: None)
        b.internal = types.SimpleNamespace(device_put=mock.Mock(side_effect=lambda value, sharding: value))
        payload = {'format': 'helixwm_matched_backbone_v1', 'upstream_commit': UPSTREAM_COMMIT,
                   'resolved_config': {'seed': 0, 'logdir': 'old_training_path'},
                   'training': {'params': {'pol/w': np.array([2.], np.float32),
                                           'opt/step': np.array([250.], np.float32)},
                                'counters': {'updates': 250, 'batches': 250, 'actions': 0}},
                   'writeback_updates': 250, 'inference_version': 250,
                   'inference_params': {'pol/w': np.array([2.], np.float32)},
                   'train_carry': ({}, {'deter': np.array([3.], np.float32)}, {},
                                   {'action': np.array([.1], np.float32)})}
        return b, payload

    def test_device_reference_uses_metadata_without_implicit_transfer(self):
        _validate_tree({'optimizer': GuardedDeviceArray()}, {'optimizer': np.zeros(1, np.float32)})

    def test_full_load_keeps_explicit_host_and_device_transfers(self):
        b, payload = self.fixture()
        b.load(payload)
        b.agent.load.assert_called_once_with(payload['training'])
        self.assertEqual(b.internal.device_put.call_args_list[0].args[1], 'train_sharded')
        self.assertEqual(b.internal.device_put.call_args_list[1].args[1], 'train_mirrored')
        self.assertIs(b.train_carry, payload['train_carry'])
        self.assertIs(b.inference_params, payload['inference_params'])
        self.assertEqual(b.inference_version, 250)

    def test_bad_host_checkpoint_rejected_before_model_mutation(self):
        for invalid in (np.zeros(2, np.float32), np.zeros(1, np.float64),
                        np.array([np.nan], np.float32), GuardedDeviceArray()):
            b, payload = self.fixture(); payload['training']['params']['opt/step'] = invalid
            with self.subTest(invalid=type(invalid).__name__), self.assertRaises((ValueError, FloatingPointError)):
                b.load(payload)
            b.agent.load.assert_not_called()


class TrainTests(unittest.TestCase):
    def fixture(self):
        b = object.__new__(MatchedDreamerBackbone)
        spaces = {'stepid': space(np.uint8, (20,)), 'consec': space(np.int32),
                  'action': space(np.float32, (6,)), 'dyn/deter': space(np.float32, (2,)),
                  'dyn/stoch': space(np.float32, (1, 2))}
        batch = {k: np.zeros((16, 65, *v.shape), v.dtype) for k, v in spaces.items()}
        batch['stepid'][:, :, 0] = np.arange(65)
        b.train_carry = ('ACTUAL_PERSISTENT_CARRY',)
        b.writeback_updates = 0
        a = types.SimpleNamespace(spaces=spaces, n_batches=Counter(),
                                  train_mirrored=None, train_sharded=None,
                                  pending_outs=None, pending_mets=None, params={})
        a._seeds = lambda counter, shard: counter
        a._take_outs = lambda value: value
        def train(carry, data):
            self.assertEqual(carry, ('ACTUAL_PERSISTENT_CARRY',))
            self.assertEqual(data.pop('seed'), 0)
            a.pending_outs = {'replay': {k: batch[k][:, 1:].copy()
                               for k in ('stepid', 'dyn/deter', 'dyn/stoch')}}
            a.pending_mets = {'loss': np.array(2., np.float32)}
            return ('UPDATED_CARRY',), {}, {}
        a.train = train
        b.agent = a
        b.internal = types.SimpleNamespace(device_put=lambda value, shard: dict(value))
        b.jax = types.SimpleNamespace(block_until_ready=lambda value: None)
        replay = types.SimpleNamespace(update=mock.Mock())
        return b, batch, replay

    def test_native_context_alignment_and_writeback(self):
        b, batch, replay = self.fixture()
        result = b.train(batch, replay)
        self.assertEqual(result['loss'], 2.)
        self.assertEqual(b.train_carry, ('UPDATED_CARRY',))
        np.testing.assert_array_equal(replay.update.call_args.args[0]['stepid'], batch['stepid'][:, 1:])
        self.assertEqual(b.writeback_updates, 1)
        self.assertIsNone(b.agent.pending_outs)

    def test_bad_context_or_schema_rejected_before_update(self):
        for field in ('consec', 'shape'):
            b, batch, replay = self.fixture()
            if field == 'consec': batch['consec'][0, 0] = 1
            else: batch['stepid'] = batch['stepid'][:, :-1]
            with self.assertRaises(ValueError): b.train(batch, replay)
            self.assertEqual(int(b.agent.n_batches), 0)
            replay.update.assert_not_called()

    def test_writeback_failure_cannot_be_silently_flushed(self):
        b, batch, replay = self.fixture()
        replay.update.side_effect = OSError('disk-independent writeback failure')
        with self.assertRaises(OSError): b.train(batch, replay)
        with self.assertRaises(RuntimeError): b.flush()
        self.assertEqual(b.writeback_updates, 0)


def native_source_modules(archive):
    """Use exact archived algorithms; emulate only lightweight utility imports."""
    elements = types.ModuleType('elements')
    class TimerSection(ContextDecorator):
        def __enter__(self): return self
        def __exit__(self, *args): return False
    elements.timer = types.SimpleNamespace(section=lambda *_: TimerSection())
    class UUID:
        def __init__(self, value=None):
            self.uuid = value.uuid if isinstance(value, UUID) else (
                uuid.uuid4() if value is None else uuid.UUID(bytes=value)
                if isinstance(value, bytes) else uuid.UUID(int=value)
                if isinstance(value, int) else uuid.UUID(str(value)))
        def __bytes__(self): return self.uuid.bytes
        def __str__(self): return str(self.uuid)
        def __hash__(self): return hash(self.uuid)
        def __eq__(self, other): return isinstance(other, UUID) and self.uuid == other.uuid
    elements.UUID = UUID
    elements.timestamp = lambda **_: '20260101T000000000'
    elements.RWLock = lambda: types.SimpleNamespace(reading=nullcontext(), writing=nullcontext())
    embodied = types.ModuleType('embodied'); embodied.__path__ = []
    core = types.ModuleType('embodied.core'); core.__path__ = []
    base = types.ModuleType('embodied.core.base'); base.Stream = object
    limiter = types.ModuleType('embodied.core.limiters')
    def wait(fn, message):
        if not fn(): raise RuntimeError(message)
    limiter.wait = wait
    portal = types.ModuleType('portal'); portal.Thread = threading.Thread
    modules = {'elements': elements, 'embodied': embodied, 'embodied.core': core,
               'embodied.core.base': base, 'embodied.core.limiters': limiter, 'portal': portal}
    with tarfile.open(archive) as tar:
        with mock.patch.dict(sys.modules, modules):
            for name in ('chunk', 'selectors', 'replay', 'streams'):
                mod = types.ModuleType('embodied.core.' + name)
                mod.__package__ = 'embodied.core'
                sys.modules[mod.__name__] = mod
                modules[mod.__name__] = mod
                setattr(core, name, mod)
                text = tar.extractfile('dreamerv3/embodied/core/' + name + '.py').read().decode()
                exec(compile(text, 'pinned-upstream/' + name + '.py', 'exec'), mod.__dict__)
    embodied.Replay = modules['embodied.core.replay'].Replay
    embodied.streams = modules['embodied.core.streams']
    return modules


@unittest.skipUnless(os.environ.get('DREAMERV3_ARCHIVE'), 'Set DREAMERV3_ARCHIVE for pinned native Replay tests')
class NativeReplayTests(unittest.TestCase):
    def setUp(self):
        import tempfile
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.module_patch = mock.patch.dict(sys.modules, native_source_modules(os.environ['DREAMERV3_ARCHIVE']))
        self.module_patch.start(); self.addCleanup(self.module_patch.stop)
        obs = {'image': space(np.uint8, (2, 2, 3)), 'reward': space(np.float32),
               'is_first': space(bool), 'is_last': space(bool), 'is_terminal': space(bool)}
        spaces = {**obs, 'action': space(np.float32, (6,)), 'dyn/deter': space(np.float32, (2,)),
                  'dyn/stoch': space(np.float32, (1, 2)), 'stepid': space(np.uint8, (20,)), 'consec': space(np.int32)}
        self.backbone = types.SimpleNamespace(agent=types.SimpleNamespace(obs_space=obs, spaces=spaces),
                       replay_entries=lambda state: {k: v.copy() for k, v in state.items()})
        self.replay = NativeReplay(self.backbone, self.temp.name, capacity=140, seed=7, chunksize=65)
        for i in range(205):
            self.replay.add({'image': np.full((2, 2, 3), i % 256, np.uint8),
                 'reward': np.float32(i), 'is_first': np.bool_(i % 50 == 0),
                 'is_last': np.bool_(i % 50 == 49), 'is_terminal': np.bool_(False)},
                 np.zeros(6, np.float32), {'dyn/deter': np.full(2, i, np.float32),
                                          'dyn/stoch': np.array([[0, 1]], np.float32)})

    def test_exact_native_next_batch_after_restore(self):
        next(self.replay.stream)
        saved = self.replay.snapshot()
        expected = next(self.replay.stream)
        fresh = NativeReplay(self.backbone, self.temp.name, capacity=140, seed=7, chunksize=65)
        fresh.load(saved)
        actual = next(fresh.stream)
        for key in expected: np.testing.assert_array_equal(expected[key], actual[key])
        self.assertEqual(actual['image'].shape[:2], (16, 65))
        self.assertFalse(actual['consec'].any())
        self.assertEqual(fresh.rows_added, 205)

    def test_writeback_new_snapshot_keeps_previous_version(self):
        batch = next(self.replay.stream)
        old = self.replay.snapshot()
        changes = {'stepid': batch['stepid'][:, 1:].copy(),
                   'dyn/deter': np.full((16, 64, 2), -999, np.float32)}
        self.replay.replay.update(changes)
        new = self.replay.snapshot()
        old_files = {r['path'] for r in NativeReplay.referenced_files(old)}
        new_files = {r['path'] for r in NativeReplay.referenced_files(new)}
        self.assertTrue(old_files & new_files)
        self.assertTrue(new_files - old_files)
        fresh = NativeReplay(self.backbone, self.temp.name, capacity=140, seed=7, chunksize=65)
        fresh.load(old)
        values = [c.data['dyn/deter'][:c.length] for c in fresh.replay.chunks.values() if c.length]
        self.assertFalse(any(np.any(x == -999) for x in values))

    def test_tampered_selector_and_object_rejected(self):
        saved = self.replay.snapshot()
        tampered = copy.deepcopy(saved); tampered['sampler_keys'].reverse()
        fresh = NativeReplay(self.backbone, self.temp.name, capacity=140, seed=7, chunksize=65)
        with self.assertRaises(ValueError): fresh.load(tampered)
        self.assertEqual(fresh.rows_added, 0)
        record = NativeReplay.referenced_files(saved)[0]
        path = Path(self.temp.name) / record['path']
        path.write_bytes(path.read_bytes() + b'tamper')
        with self.assertRaises(ValueError): fresh.load(saved)

    def test_configuration_mismatch_rejected(self):
        saved = self.replay.snapshot()
        fresh = NativeReplay(self.backbone, self.temp.name, capacity=141, seed=7, chunksize=65)
        with self.assertRaises(ValueError): fresh.load(saved)

    def test_reference_count_corruption_rejected_before_mutation(self):
        saved = self.replay.snapshot()
        saved['refs'][next(iter(saved['refs']))] += 1
        fresh = NativeReplay(self.backbone, self.temp.name, capacity=140, seed=7, chunksize=65)
        with self.assertRaises(ValueError): fresh.load(saved)
        self.assertEqual(fresh.rows_added, 0)

    def test_native_append_after_restore_retains_sampler_parity(self):
        saved = self.replay.snapshot()
        fresh = NativeReplay(self.backbone, self.temp.name, capacity=140, seed=7, chunksize=65)
        fresh.load(saved)
        for replay in (self.replay, fresh):
            replay.add({'image': np.zeros((2, 2, 3), np.uint8), 'reward': np.float32(0),
                        'is_first': np.bool_(True), 'is_last': np.bool_(False),
                        'is_terminal': np.bool_(False)}, np.zeros(6, np.float32),
                       {'dyn/deter': np.zeros(2, np.float32),
                        'dyn/stoch': np.array([[1, 0]], np.float32)})
        first, restored = next(self.replay.stream), next(fresh.stream)
        for key in first: np.testing.assert_array_equal(first[key], restored[key])


if __name__ == '__main__': unittest.main()
