"""Immutable, streaming evidence for actual and prospective Dream trajectories.

This archive records model predictions, not recovered ground truth. In
particular, an unavailable observation has ``observed_error=None``. The writer
keeps only bounded metadata references in memory; pixel/latent arrays live in
NPZ files with pickle disabled. Checkpoints contain a manifest cursor and hash,
not a second copy of trajectory pixels. A restored cursor cannot overwrite
later evidence. The storage cost is real and must be included in launch gates.
"""
from __future__ import annotations

from collections import deque
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import threading

import numpy as np

from .belief import uncertainty_metrics

SCHEMA = 'helixwm_final_trajectory_archive_v1'
RECORD_SCHEMA = 'helixwm_final_trajectory_record_v1'
ZERO_SHA = '0' * 64
METRIC_FIELDS = ('predictive_entropy', 'expected_entropy',
                 'model_conditional_disagreement', 'uncertainty', 'ess_ratio',
                 'epistemic_disagreement')
RECORD_FIELDS = {'schema', 'kind', 'episode_id', 'timestamp', 'origin_timestamp',
    'depth', 'missing_age', 'model_version', 'model_sha256', 'candidate_ids', 'ancestor_ids',
    'state', 'real_state', 'actions', 'predicted_observation', 'predicted_reward',
    'predicted_continuation', 'probabilities', 'weights', 'metrics',
    'observation_available', 'observed_error', 'calibration_sha256',
    'calibration_confidence', 'prediction_rng_sha256', 'executed_prefix_sha256'}


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      allow_nan=False, ensure_ascii=True).encode('utf8')


def _digest(value):
    return hashlib.sha256(_json(value)).hexdigest()


def _sha(value):
    return type(value) is str and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def _file_sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def _integer(value, name, minimum=0, maximum=None):
    if type(value) is not int or value < minimum or (maximum is not None and value > maximum):
        raise ValueError('Invalid integer: ' + name)


def _scalar(value, name, minimum=None, maximum=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (float, np.floating)) or not math.isfinite(value):
        raise ValueError('Expected finite scalar: ' + name)
    if minimum is not None and value < minimum or maximum is not None and value > maximum:
        raise ValueError('Scalar outside registered units: ' + name)


def _array(value, name, shape, dtypes=(np.float32, np.float64), minimum=None, maximum=None):
    if (not isinstance(value, np.ndarray) or value.dtype not in dtypes
            or value.shape != shape or not np.isfinite(value).all()):
        raise ValueError('Invalid finite array/dtype/shape: ' + name)
    if minimum is not None and np.any(value < minimum) or maximum is not None and np.any(value > maximum):
        raise ValueError('Array outside registered units: ' + name)


def _state(value, count, name):
    if type(value) is not dict or set(value) != {'deter', 'stoch'}:
        raise ValueError('Invalid latent keys: ' + name)
    h, s = value['deter'], value['stoch']
    if (not isinstance(h, np.ndarray) or h.ndim != 2 or h.shape[0] != count
            or not 1 <= h.shape[1] <= 65536 or not isinstance(s, np.ndarray)
            or s.ndim != 3 or s.shape[0] != count or not 1 <= s.shape[1] <= 1024
            or not 2 <= s.shape[2] <= 1024):
        raise ValueError('Invalid latent dimensions: ' + name)
    _array(h, name + '.deter', h.shape, (np.float32,))
    _array(s, name + '.stoch', s.shape, (np.float32,))


def validate_record(record):
    """Validate without coercing, clipping, normalizing or manufacturing data."""
    if type(record) is not dict or set(record) != RECORD_FIELDS or record['schema'] != RECORD_SCHEMA:
        raise ValueError('Unexpected/incomplete trajectory schema')
    if record['kind'] not in ('actual', 'prospective'):
        raise ValueError('Actual hypotheses and prospective plans must be distinct')
    for name in ('episode_id', 'timestamp', 'origin_timestamp', 'model_version'):
        _integer(record[name], name)
    _integer(record['depth'], 'depth', 0, 100)
    _integer(record['missing_age'], 'missing_age', 0, 100)
    if record['kind'] == 'actual':
        if record['depth'] != 0 or record['origin_timestamp'] != record['timestamp']:
            raise ValueError('Actual hypotheses must be timestamped at the actual outcome')
    elif record['depth'] == 0 or record['timestamp'] != record['origin_timestamp'] + record['depth']:
        raise ValueError('Prospective timestamp must equal origin plus positive depth')
    for name in ('model_sha256', 'prediction_rng_sha256', 'executed_prefix_sha256'):
        if not _sha(record[name]):
            raise ValueError('Missing external prediction/action-prefix identity: ' + name)
    ids, ancestors = record['candidate_ids'], record['ancestor_ids']
    if (type(ids) is not list or not 1 <= len(ids) <= 8 or len(set(ids)) != len(ids)
            or any(type(x) is not str or not 1 <= len(x) <= 128 for x in ids)
            or type(ancestors) is not list or len(ancestors) != len(ids)
            or any(x is not None and (type(x) is not str or not 1 <= len(x) <= 128) for x in ancestors)):
        raise ValueError('Candidate/ancestor identities must be explicit and bounded')
    k = len(ids)
    _state(record['state'], k, 'state')
    if record['real_state'] is not None:
        _state(record['real_state'], 1, 'real_state')
    elif record['kind'] == 'actual':
        raise ValueError('An actual record must retain its paired Real state')
    _array(record['actions'], 'actions', (k, 6), (np.float32,), -1., 1.)
    if record['kind'] == 'actual' and not np.array_equal(record['actions'], np.broadcast_to(record['actions'][0], (k, 6))):
        raise ValueError('Actual hypotheses must share the one executed action')
    _array(record['weights'], 'weights', (k,), (np.float64,), 0., 1.)
    if not math.isclose(float(record['weights'].sum()), 1., rel_tol=0., abs_tol=2e-12):
        raise ValueError('Weights must already be normalized')
    for name in ('predicted_reward', 'predicted_continuation', 'observed_error'):
        if record[name] is not None:
            _array(record[name], name, (k,))
    if record['predicted_continuation'] is not None:
        _array(record['predicted_continuation'], 'predicted_continuation', (k,), minimum=0., maximum=1.)
    if record['predicted_observation'] is not None:
        _array(record['predicted_observation'], 'predicted_observation', (k, 64, 64, 3), (np.float32,), 0., 1.)
    if type(record['observation_available']) is not bool:
        raise ValueError('Observation availability must be a strict boolean')
    if record['kind'] == 'actual' and record['observation_available'] and record['missing_age'] != 0:
        raise ValueError('A currently available actual observation must reset missing age')
    if (record['kind'] == 'prospective' or not record['observation_available']) and record['observed_error'] is not None:
        raise ValueError('An unavailable/future observation cannot have an observed error')
    if record['kind'] == 'prospective' and record['observation_available']:
        raise ValueError('A prospective prediction cannot claim a received observation')
    metrics = record['metrics']
    if type(metrics) is not dict or set(metrics) != set(METRIC_FIELDS):
        raise ValueError('Every uncertainty field must be explicit, using null when absent')
    for name, value in metrics.items():
        if value is not None:
            _scalar(value, name, 0., None if name == 'epistemic_disagreement' else 1.)
    expected_ess = float(1. / np.square(record['weights']).sum() / k)
    if metrics['ess_ratio'] is None or not math.isclose(metrics['ess_ratio'], expected_ess, abs_tol=2e-12, rel_tol=0.):
        raise ValueError('ESS does not match retained candidate weights')
    probs = record['probabilities']
    if probs is not None:
        _array(probs, 'probabilities', record['state']['stoch'].shape, minimum=0., maximum=1.)
        tolerance = 2e-6 if probs.dtype == np.float32 else 2e-12
        if not np.allclose(probs.sum(-1), 1., atol=tolerance, rtol=0.):
            raise ValueError('Probabilities must already be normalized')
        expected = uncertainty_metrics(probs, record['weights'])
        for name in ('predictive_entropy', 'expected_entropy', 'model_conditional_disagreement', 'uncertainty'):
            if metrics[name] is None or not math.isclose(metrics[name], expected[name], abs_tol=2e-7, rel_tol=0.):
                raise ValueError('Uncertainty does not match retained probabilities: ' + name)
    elif any(metrics[name] is not None for name in ('predictive_entropy', 'expected_entropy', 'model_conditional_disagreement', 'uncertainty')):
        raise ValueError('Categorical metrics require their actual probability arrays')
    if record['calibration_sha256'] is None:
        if record['calibration_confidence'] is not None:
            raise ValueError('Uncalibrated confidence must remain null')
    elif not _sha(record['calibration_sha256']):
        raise ValueError('Invalid calibration identity')
    if record['calibration_confidence'] is not None:
        _scalar(record['calibration_confidence'], 'calibration_confidence', 0., 1.)
    return True


def make_record(*, kind, episode_id, timestamp, model_version, model_sha256,
                candidate_ids, ancestor_ids, state, actions, weights,
                prediction_rng_sha256, executed_prefix_sha256, real_state=None,
                origin_timestamp=None, depth=0, missing_age=0, predicted_observation=None,
                predicted_reward=None, predicted_continuation=None, probabilities=None,
                observation_available=False, observed_error=None,
                calibration_sha256=None, calibration_confidence=None,
                epistemic_disagreement=None):
    """Construct a copied record; optional measurements remain explicitly null.

    ``actions`` is the transition INTO each retained latent (not the next
    outgoing action). For actual records every candidate uses the same executed
    action. For a plan it is the per-candidate prospective transition action.
    """
    values = locals().copy()
    values['schema'] = RECORD_SCHEMA
    values['candidate_ids'] = list(candidate_ids)
    values['ancestor_ids'] = list(ancestor_ids)
    values['origin_timestamp'] = timestamp if origin_timestamp is None else origin_timestamp
    values.pop('epistemic_disagreement')
    k = len(candidate_ids)
    if not isinstance(weights, np.ndarray) or weights.shape != (k,) or weights.dtype != np.float64:
        raise ValueError('Explicit float64 candidate weights required')
    metrics = {name: None for name in METRIC_FIELDS}
    metrics['ess_ratio'] = float(1. / np.square(weights).sum() / k)
    metrics['epistemic_disagreement'] = epistemic_disagreement
    if probabilities is not None:
        derived = uncertainty_metrics(probabilities, weights)
        for name in ('predictive_entropy', 'expected_entropy', 'model_conditional_disagreement', 'uncertainty'):
            metrics[name] = float(derived[name])
    values['metrics'] = metrics
    validate_record(values)
    return copy.deepcopy(values)


class _Lease:
    """An OS lock, released on process death; stale lock files are harmless."""
    def __init__(self, path):
        self.stream = open(path, 'a+b')
        if path.stat().st_size == 0:
            self.stream.write(b'0'); self.stream.flush()
        self.stream.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(self.stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            self.stream.close()
            raise RuntimeError('Another writer holds the trajectory archive') from error

    def close(self):
        if self.stream.closed:
            return
        self.stream.seek(0)
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(self.stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(self.stream.fileno(), fcntl.LOCK_UN)
        self.stream.close()


def _write_exclusive(path, content):
    with open(path, 'xb') as stream:
        stream.write(content); stream.flush(); os.fsync(stream.fileno())


def _path(root, relative):
    if type(relative) is not str or '\\' in relative or Path(relative).is_absolute():
        raise ValueError('Archive dependencies must be relative POSIX paths')
    path = root / relative
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError('Archive dependency escapes its root')
    return path


def _encode(value, arrays):
    if isinstance(value, np.ndarray):
        key = 'a%03d' % len(arrays)
        arrays[key] = value
        return {'__ndarray__': key}
    if isinstance(value, dict):
        return {key: _encode(item, arrays) for key, item in value.items()}
    if isinstance(value, list):
        return [_encode(item, arrays) for item in value]
    return value


def _decode(value, arrays):
    if isinstance(value, dict):
        if set(value) == {'__ndarray__'}:
            return arrays[value['__ndarray__']].copy()
        return {key: _decode(item, arrays) for key, item in value.items()}
    if isinstance(value, list):
        return [_decode(item, arrays) for item in value]
    return value


class TrajectoryArchive:
    """Append-only archive with O(ring_size) metadata memory and immutable data."""
    @classmethod
    def create(cls, directory, *, source_sha256, protocol_sha256, ring_size=100):
        if not _sha(source_sha256) or not _sha(protocol_sha256):
            raise ValueError('External source and protocol digests are required')
        _integer(ring_size, 'ring_size', 1, 100)
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=False)
        header = {'schema': SCHEMA, 'source_sha256': source_sha256,
                  'protocol_sha256': protocol_sha256, 'ring_size': ring_size,
                  'max_candidates': 8, 'max_gap': 100}
        _write_exclusive(root / 'header.json', _json(header))
        _write_exclusive(root / 'manifest.jsonl', b'')
        result = cls.__new__(cls)
        result._initialize(root, header, True)
        return result

    def _initialize(self, root, header, writable):
        self.root = root
        self.header = copy.deepcopy(header)
        self.header_sha256 = _file_sha(root / 'header.json')
        self._recent = deque(maxlen=header['ring_size'])
        self._count = self._prefix_bytes = 0
        self._chain = ZERO_SHA
        self._manifest_hash = hashlib.sha256()
        self._closed = False
        self._mutex = threading.RLock()
        self._lease = _Lease(root / '.writer.lock') if writable else None
        self.writable = writable

    @property
    def record_count(self):
        return self._count

    @property
    def recent_references(self):
        return copy.deepcopy(list(self._recent))

    def _require_open(self):
        if self._closed:
            raise RuntimeError('Trajectory archive has been closed')

    def append(self, record):
        """Persist a complete record before publishing its hash-chain entry."""
        with self._mutex:
            self._require_open()
            if not self.writable:
                raise RuntimeError('Read-only historical cursor cannot append')
            validate_record(record)
            if (self.root / 'manifest.jsonl').stat().st_size != self._prefix_bytes:
                raise ValueError('Manifest changed outside this writer; refusing overwrite')
            sequence = self._count
            stem = 'records/%06d/%09d' % (sequence // 1000, sequence)
            data_path, meta_path = self.root / (stem + '.npz'), self.root / (stem + '.json')
            data_path.parent.mkdir(parents=True, exist_ok=True)
            arrays = {}
            payload = _encode(record, arrays)
            # Exclusive final names preserve interrupted/unreferenced evidence.
            # A failed append never truncates an earlier file or manifest.
            with open(data_path, 'xb') as stream:
                np.savez_compressed(stream, **arrays)
                stream.flush(); os.fsync(stream.fileno())
            meta = {'schema': RECORD_SCHEMA, 'sequence': sequence,
                    'source_sha256': self.header['source_sha256'],
                    'protocol_sha256': self.header['protocol_sha256'],
                    'payload': payload,
                    'arrays': {name: {'dtype': a.dtype.str, 'shape': list(a.shape)}
                               for name, a in arrays.items()}}
            _write_exclusive(meta_path, _json(meta))
            entry = {'sequence': sequence, 'metadata_path': stem + '.json',
                     'metadata_sha256': _file_sha(meta_path), 'data_path': stem + '.npz',
                     'data_sha256': _file_sha(data_path), 'previous_chain_sha256': self._chain}
            entry['chain_sha256'] = _digest(entry)
            line = _json(entry) + b'\n'
            with open(self.root / 'manifest.jsonl', 'ab') as stream:
                stream.write(line); stream.flush(); os.fsync(stream.fileno())
            self._manifest_hash.update(line)
            self._prefix_bytes += len(line)
            self._count += 1
            self._chain = entry['chain_sha256']
            self._recent.append(entry)
            return copy.deepcopy(entry)

    def snapshot(self):
        """Small cursor only; every preceding record is a required dependency."""
        with self._mutex:
            self._require_open()
            result = {'schema': SCHEMA, 'header_sha256': self.header_sha256,
                'source_sha256': self.header['source_sha256'],
                'protocol_sha256': self.header['protocol_sha256'],
                'record_count': self._count, 'manifest_prefix_bytes': self._prefix_bytes,
                'manifest_prefix_sha256': self._manifest_hash.hexdigest(),
                'last_chain_sha256': self._chain}
            result['sha256'] = _digest(result)
            return result

    @classmethod
    def restore(cls, directory, snapshot, *, expected_source_sha256,
                expected_protocol_sha256, writable=False):
        """Verify ALL bound files, then open the exact saved prefix.

        Read-only restoration may coexist with a later tail. A writable restore
        requires the manifest to end exactly at the cursor, so it cannot replace
        later evidence. Unreferenced interrupted files remain untouched and will
        block reuse of their names; resolve them through an explicit new run.
        """
        fields = {'schema', 'header_sha256', 'source_sha256', 'protocol_sha256',
            'record_count', 'manifest_prefix_bytes', 'manifest_prefix_sha256',
            'last_chain_sha256', 'sha256'}
        if type(snapshot) is not dict or set(snapshot) != fields or snapshot['schema'] != SCHEMA:
            raise ValueError('Invalid trajectory snapshot schema')
        if snapshot['sha256'] != _digest({k: v for k, v in snapshot.items() if k != 'sha256'}):
            raise ValueError('Trajectory snapshot digest mismatch')
        for name in fields - {'schema', 'record_count', 'manifest_prefix_bytes'}:
            if not _sha(snapshot[name]):
                raise ValueError('Invalid trajectory snapshot digest: ' + name)
        for name in ('record_count', 'manifest_prefix_bytes'):
            _integer(snapshot[name], name)
        if (snapshot['source_sha256'] != expected_source_sha256
                or snapshot['protocol_sha256'] != expected_protocol_sha256):
            raise ValueError('Trajectory snapshot external context mismatch')
        root = Path(directory)
        header_bytes = (root / 'header.json').read_bytes()
        if hashlib.sha256(header_bytes).hexdigest() != snapshot['header_sha256']:
            raise ValueError('Trajectory header identity mismatch')
        header = json.loads(header_bytes)
        if (set(header) != {'schema', 'source_sha256', 'protocol_sha256', 'ring_size', 'max_candidates', 'max_gap'}
                or header['schema'] != SCHEMA or header['max_candidates'] != 8 or header['max_gap'] != 100
                or header['source_sha256'] != expected_source_sha256
                or header['protocol_sha256'] != expected_protocol_sha256):
            raise ValueError('Trajectory header context/schema mismatch')
        _integer(header['ring_size'], 'ring_size', 1, 100)
        result = cls.__new__(cls)
        result._initialize(root, header, writable)
        try:
            if writable and (root / 'manifest.jsonl').stat().st_size != snapshot['manifest_prefix_bytes']:
                raise ValueError('Writable restore cannot overwrite a later manifest tail')
            for entry, line in result._entries(snapshot['manifest_prefix_bytes']):
                result._read_entry(entry)
                result._manifest_hash.update(line)
                result._prefix_bytes += len(line)
                result._count += 1
                result._chain = entry['chain_sha256']
                result._recent.append(entry)
            if (result._count != snapshot['record_count'] or result._prefix_bytes != snapshot['manifest_prefix_bytes']
                    or result._manifest_hash.hexdigest() != snapshot['manifest_prefix_sha256']
                    or result._chain != snapshot['last_chain_sha256']):
                raise ValueError('Trajectory manifest prefix/cursor mismatch')
            return result
        except BaseException:
            result.close()
            raise

    def _entries(self, limit=None):
        limit = self._prefix_bytes if limit is None else limit
        position, expected_sequence, chain = 0, 0, ZERO_SHA
        with open(self.root / 'manifest.jsonl', 'rb') as stream:
            while position < limit:
                line = stream.readline(min(1024 * 1024, limit - position))
                if not line or not line.endswith(b'\n'):
                    raise ValueError('Truncated/incomplete manifest prefix')
                position += len(line)
                entry = json.loads(line)
                fields = {'sequence', 'metadata_path', 'metadata_sha256', 'data_path',
                          'data_sha256', 'previous_chain_sha256', 'chain_sha256'}
                if type(entry) is not dict or set(entry) != fields or type(entry['sequence']) is not int:
                    raise ValueError('Invalid manifest entry schema')
                if entry['sequence'] != expected_sequence or entry['previous_chain_sha256'] != chain:
                    raise ValueError('Manifest ordering or hash chain mismatch')
                if entry['chain_sha256'] != _digest({k: v for k, v in entry.items() if k != 'chain_sha256'}):
                    raise ValueError('Manifest entry hash mismatch')
                for name in ('metadata_sha256', 'data_sha256', 'previous_chain_sha256', 'chain_sha256'):
                    if not _sha(entry[name]):
                        raise ValueError('Invalid manifest file digest')
                # Canonical paths prevent duplicate names, traversal and aliases.
                stem = 'records/%06d/%09d' % (expected_sequence // 1000, expected_sequence)
                if entry['metadata_path'] != stem + '.json' or entry['data_path'] != stem + '.npz':
                    raise ValueError('Unexpected archive dependency path')
                expected_sequence += 1
                chain = entry['chain_sha256']
                yield entry, line

    def _read_entry(self, entry):
        paths = {kind: _path(self.root, entry[kind + '_path']) for kind in ('metadata', 'data')}
        for kind, path in paths.items():
            if _file_sha(path) != entry[kind + '_sha256']:
                raise ValueError('Trajectory dependency hash mismatch: ' + kind)
        meta = json.loads(paths['metadata'].read_bytes())
        if (type(meta) is not dict or set(meta) != {'schema', 'sequence', 'source_sha256', 'protocol_sha256', 'payload', 'arrays'}
                or meta['schema'] != RECORD_SCHEMA or meta['sequence'] != entry['sequence']
                or meta['source_sha256'] != self.header['source_sha256']
                or meta['protocol_sha256'] != self.header['protocol_sha256']):
            raise ValueError('Trajectory metadata identity mismatch')
        with np.load(paths['data'], allow_pickle=False) as archive:
            if set(archive.files) != set(meta['arrays']):
                raise ValueError('Trajectory NPZ member set mismatch')
            arrays = {}
            for key in archive.files:
                value = archive[key]
                expected = meta['arrays'][key]
                if value.dtype.hasobject or {'dtype': value.dtype.str, 'shape': list(value.shape)} != expected:
                    raise ValueError('Trajectory NPZ dtype/shape mismatch')
                arrays[key] = value.copy()
        result = _decode(meta['payload'], arrays)
        validate_record(result)
        return result

    def load_record(self, sequence):
        self._require_open()
        _integer(sequence, 'sequence', 0)
        if sequence >= self._count:
            raise IndexError('Record is outside this saved trajectory cursor')
        entry = next((x for x in self._recent if x['sequence'] == sequence), None)
        if entry is None:
            entry = next(x for x, _ in self._entries() if x['sequence'] == sequence)
        return self._read_entry(entry)

    def reconstruct(self, *, episode_id, candidate_id, start_timestamp, end_timestamp,
                    kind='actual', origin_timestamp=None):
        """Yield one candidate's real arrays in time order, with O(1) read memory.

        A segment is at most 100 transitions. Prospective segments additionally
        require an origin so overlapping plans cannot be mistaken for one path.
        Missing/duplicate timestamps are errors, never padded predictions.
        """
        self._require_open()
        for name, value in (('episode_id', episode_id), ('start_timestamp', start_timestamp), ('end_timestamp', end_timestamp)):
            _integer(value, name)
        if not 1 <= end_timestamp - start_timestamp + 1 <= 100 or kind not in ('actual', 'prospective'):
            raise ValueError('Reconstruction requires a registered 1..100 step segment')
        if kind == 'prospective' and (type(origin_timestamp) is not int or origin_timestamp < 0):
            raise ValueError('Prospective reconstruction requires its plan origin')
        expected = start_timestamp
        for entry, _ in self._entries():
            record = self._read_entry(entry)
            if (record['episode_id'] != episode_id or record['kind'] != kind
                    or candidate_id not in record['candidate_ids']
                    or not start_timestamp <= record['timestamp'] <= end_timestamp
                    or kind == 'prospective' and record['origin_timestamp'] != origin_timestamp):
                continue
            if record['timestamp'] != expected:
                raise ValueError('Trajectory segment has missing/duplicate/out-of-order timestamps')
            i = record['candidate_ids'].index(candidate_id)
            single = copy.deepcopy(record)
            single['candidate_ids'] = [candidate_id]
            single['ancestor_ids'] = [record['ancestor_ids'][i]]
            for key in ('actions', 'predicted_observation', 'predicted_reward', 'predicted_continuation', 'probabilities', 'weights', 'observed_error'):
                if record[key] is not None:
                    single[key] = record[key][i:i+1].copy()
            single['state'] = {key: value[i:i+1].copy() for key, value in record['state'].items()}
            # Original ensemble metrics and selected original weight are
            # retained; this extracted slice is NOT a new normalized ensemble.
            single['archive_sequence'] = entry['sequence']
            single['original_candidate_count'] = len(record['candidate_ids'])
            expected += 1
            yield single
        if expected != end_timestamp + 1:
            raise ValueError('Trajectory segment is incomplete at the saved cursor')

    def close(self):
        if not getattr(self, '_closed', True):
            if self._lease is not None:
                self._lease.close()
            self._closed = True

    def __enter__(self):
        self._require_open()
        return self

    def __exit__(self, *_):
        self.close()
