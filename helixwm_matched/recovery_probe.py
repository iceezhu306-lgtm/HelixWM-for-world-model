"""Two fresh-process, exact GPU episode-boundary recovery gate.

This is an isolated validation continuation, never automatic training recovery.
Each process restores the same trusted, SHA-verified boundary checkpoint, runs
the next four actual decisions and the next scheduled native learner update.
No original run, checkpoint, or immutable replay object is modified. An exact
mismatch is a failure, not a reason to introduce a numerical tolerance.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import random
import shutil
import struct
import subprocess
import sys
import time
import traceback

import numpy as np

from .protocol import (atomic_json, load_protocol, make_mask_bank, mask_sha256,
                       protocol_sha256, training_seed)

RESTORED_FIELDS = {'training', 'train_carry', 'inference_params',
                   'inference_version', 'writeback_updates'}
COMPARISON_FIELDS = {
    'identity', 'checkpoint_sha256', 'restored_training_tree_sha256',
    'restored_train_carry_sha256', 'restored_inference_sha256',
    'restored_collector_rng_sha256', 'prefill_rng_sha256', 'next_prefill_action_clone',
    'restored_global_rng', 'next_episode_index', 'next_environment_seed',
    'next_mask_sha256', 'reset', 'steps', 'sample', 'sampler_rng_after_sample_sha256',
    'after_training_tree_sha256', 'after_train_carry_sha256', 'after_inference_sha256',
    'after_collector_sha256', 'after_replay_contents_sha256', 'after_global_rng',
    'metrics', 'metrics_sha256', 'counters', 'gpu_devices'}


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def tree_hash(value):
    """Exact typed tree hash, including arbitrary-size NumPy RNG integers."""
    h = hashlib.sha256()
    def visit(x):
        if isinstance(x, np.ndarray) or isinstance(x, np.generic):
            a = np.asarray(x)
            if a.dtype.hasobject:
                raise TypeError('Object arrays are not recovery evidence')
            h.update(b'A' + a.dtype.str.encode() + str(a.shape).encode() + b':')
            h.update(a.tobytes(order='C'))
        elif isinstance(x, dict):
            h.update(b'D')
            for k in sorted(x):
                visit(k); visit(x[k])
        elif isinstance(x, (tuple, list)):
            h.update(b'T' if isinstance(x, tuple) else b'L')
            h.update(str(len(x)).encode() + b':')
            for item in x: visit(item)
        elif x is None: h.update(b'N')
        elif isinstance(x, bool): h.update(b'B1' if x else b'B0')
        elif isinstance(x, int): h.update(b'I' + str(x).encode() + b':')
        elif isinstance(x, float):
            if not math.isfinite(x): raise ValueError('Nonfinite recovery evidence')
            h.update(b'F' + struct.pack('!d', x))
        elif isinstance(x, str):
            raw = x.encode(); h.update(b'S' + str(len(raw)).encode() + b':' + raw)
        else: raise TypeError(f'Unsupported recovery evidence type {type(x).__name__}')
    visit(value)
    return h.hexdigest()


def jsonable(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, dict): return {k: jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [jsonable(v) for v in value]
    return value


def validate_payload(payload, record, config, identity):
    """Fail before initializing GPU/simulator on mismatched or partial state."""
    if payload.get('format') != 'helixwm_matched_episode_boundary_v1':
        raise ValueError('Wrong checkpoint format')
    if payload.get('next_operation') != 'reset_new_episode_and_reanchor':
        raise ValueError('Checkpoint is not an explicit episode boundary')
    if payload.get('identity') != identity or record.get('identity') != identity:
        raise ValueError('Source/protocol/environment identity mismatch')
    counts = payload.get('counters', {})
    if set(counts) != {'agent_steps', 'env_steps', 'episodes', 'update_steps'} or any(type(v) is not int or v < 0 for v in counts.values()):
        raise ValueError('Invalid checkpoint counters')
    a = counts['agent_steps']
    if not 5000 <= a < config['training']['actions_per_arm'] or a % 500:
        raise ValueError('Probe requires a post-prefill, non-final 500-action boundary')
    if counts['episodes'] != a // 500 or counts['env_steps'] != 2*a or counts['update_steps'] != (a-5000)//4:
        raise ValueError('Checkpoint budget/update counters disagree')
    if any(record.get(k) != v for k, v in counts.items()):
        raise ValueError('Checkpoint sidecar counters disagree')
    core = payload['backbone']
    if core['training']['counters']['updates'] != counts['update_steps'] or core['training']['counters']['batches'] != counts['update_steps'] or core['writeback_updates'] != counts['update_steps']:
        raise ValueError('Backbone/native-writeback counters disagree')
    if core['inference_version'] != counts['update_steps']:
        raise ValueError('Boundary must pin the next episode model version')
    if payload['replay']['rows_added'] != a + counts['episodes']:
        raise ValueError('Native replay must contain each action row plus episode final rows')
    controller = payload['completed_controller']
    if controller.get('controller') not in ('black_frame_native', 'helix_v2'):
        raise ValueError('Checkpoint is not a registered training controller')
    st = controller['chain']['state']
    if st['episode_id'] != counts['episodes']-1 or st['timestamp'] != 500 or st['anchor_time'] != 500:
        raise ValueError('Completed controller timestamp/episode differs')
    if st['last_action'] is not None or st['pending'] is not None or st['missing_length'] != 0 or st['abstaining']:
        raise ValueError('Checkpoint contains unresolved control state')
    # Validate private and host RNG payloads without changing any global RNG.
    generator = np.random.default_rng()
    generator.bit_generator.state = copy.deepcopy(payload['prefill_rng'])
    local_numpy = np.random.RandomState(); local_numpy.set_state(payload['numpy_global_rng'])
    local_python = random.Random(); local_python.setstate(payload['python_rng'])
    if set(controller['chain']['rngs']) != {'real', 'actor', 'dream', 'repair'}:
        raise ValueError('Controller RNG streams are incomplete')
    for state in controller['chain']['rngs'].values():
        np.random.default_rng().bit_generator.state = copy.deepcopy(state)
    return counts


def verify_and_load(checkpoint, config, environment_lock):
    from .train import source_digest
    checkpoint = Path(checkpoint).resolve()
    record = json.loads(checkpoint.with_suffix('.json').read_text())
    before = sha_file(checkpoint)
    if record.get('sha256') != before or record.get('bytes') != checkpoint.stat().st_size:
        raise ValueError('Checkpoint SHA256/size does not match its sidecar')
    # Checkpoints originate from this user's trusted training run. Never load an
    # arbitrary downloaded pickle. The SHA check does not make pickle sandboxed.
    with checkpoint.open('rb') as stream: payload = pickle.load(stream)
    identity = {'protocol_sha256': protocol_sha256(config),
                'source_sha256': source_digest()[0],
                'environment_lock_sha256': sha_file(environment_lock),
                'upstream_commit': config['official_core']['commit']}
    validate_payload(payload, record, config, identity)
    from .backbone import NativeReplay
    if record.get('replay_files') != NativeReplay.referenced_files(payload['replay']):
        raise ValueError('Checkpoint and sidecar replay inventories differ')
    if checkpoint.parent.name != 'CHECKPOINTS' or record.get('file') != 'CHECKPOINTS/' + checkpoint.name:
        raise ValueError('Unexpected checkpoint directory or relative filename')
    return payload, record, identity


def materialize_readonly_replay(source, destination, records):
    """Link immutable objects into a fresh probe pool; copy across filesystems."""
    source, destination = Path(source).resolve(), Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    linked = copied = total = 0
    for record in records:
        name = record['path']
        if Path(name).name != name or '/' in name or '\\' in name or not name.endswith('.npz'):
            raise ValueError('Unsafe replay reference')
        src, dst = source / name, destination / name
        if src.stat().st_size != record['bytes'] or sha_file(src) != record['file_sha256']:
            raise ValueError('Original replay object SHA256 mismatch')
        try: os.link(src, dst); linked += 1
        except OSError: shutil.copyfile(src, dst); copied += 1
        if sha_file(dst) != record['file_sha256']:
            raise ValueError('Probe replay reference SHA256 mismatch')
        total += record['bytes']
    return {'hardlinks': linked, 'copies': copied, 'referenced_bytes': total,
            'original_objects_opened_for_writing': False}


def sample_descriptor(batch):
    if any(np.asarray(v).shape[:2] != (16, 65) for v in batch.values()):
        raise ValueError('Expected actual native B16 x 65 sample')
    if batch['consec'].dtype != np.int32 or np.any(batch['consec'] != 0):
        raise ValueError('Native Consec annotation differs')
    return {'sha256': tree_hash(batch), 'fields': {k: {'shape': list(v.shape),
                'dtype': str(v.dtype), 'sha256': tree_hash(v)} for k, v in sorted(batch.items())},
            'learning_positions': 16*64, 'context_positions': 16}


def _replay_latent_hash(replay):
    # Existing UUID keys are preserved by restoration. We include only populated
    # chunks: a newly allocated empty chunk's OS-random UUID carries no data.
    values = {str(k): {name: v[:c.length] for name, v in c.data.items()}
              for k, c in replay.replay.chunks.items() if c.length}
    return tree_hash(values)


def run_worker(args):
    args._progress = {'probe_actions': 0, 'probe_dmc_calls': 0, 'probe_updates': 0,
                      'actual_counts_complete': True, 'phase': 'validating_checkpoint'}
    os.environ.setdefault('MUJOCO_GL', 'egl')
    os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
    os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
    from .backbone import MatchedDreamerBackbone, NativeReplay
    from .controllers import Controller
    from .environment import MatchedWalkerEnv
    cfg = load_protocol(args.protocol)
    payload, record, identity = verify_and_load(args.checkpoint, cfg, args.environment_lock)
    output = Path(args.output_dir).resolve(); output.mkdir(parents=True, exist_ok=False)
    source = Path(args.checkpoint).resolve().parent.parent / 'REPLAY'
    references = NativeReplay.referenced_files(payload['replay'])
    storage = materialize_readonly_replay(source, output/'REPLAY', references)
    args._progress['phase'] = 'initializing_and_restoring_gpu'
    backbone = MatchedDreamerBackbone({'seed': 0, 'size': 'size12m', 'platform': 'cuda',
                'compute_dtype': 'bfloat16', 'batch_size': 16, 'batch_length': 64}, args.upstream, output/'MODEL')
    backbone.load(payload['backbone'])
    restored = backbone.snapshot()
    restored_comparisons = {k: tree_hash(restored[k]) == tree_hash(payload['backbone'][k])
        for k in ('training', 'train_carry', 'inference_params', 'inference_version', 'writeback_updates')}
    if not all(restored_comparisons.values()): raise RuntimeError('Strict backbone read-back differs')
    rp = payload['replay']
    replay = NativeReplay(backbone, output/'REPLAY', capacity=rp['capacity'], seed=rp['seed'], chunksize=rp['chunksize'])
    replay.load(rp)
    saved_controller = payload['completed_controller']
    controller = Controller(backbone, saved_controller['controller'], seed=0)
    audit = controller.chain.restore_completed_episode(saved_controller['chain'])
    rng_loaded = controller.chain.state_dict()['rngs']
    if tree_hash(rng_loaded) != tree_hash(saved_controller['chain']['rngs']):
        raise RuntimeError('Collector RNG state did not restore exactly')
    prefill_rng = np.random.default_rng(); prefill_rng.bit_generator.state = copy.deepcopy(payload['prefill_rng'])
    np.random.set_state(payload['numpy_global_rng']); random.setstate(payload['python_rng'])
    before_host = {'numpy': tree_hash(np.random.get_state()), 'python': tree_hash(random.getstate())}
    prefill_copy = np.random.default_rng(); prefill_copy.bit_generator.state = copy.deepcopy(prefill_rng.bit_generator.state)
    prefill_next = prefill_copy.uniform(-1, 1, size=6).astype(np.float32)
    counts = copy.deepcopy(payload['counters'])
    episode = counts['episodes']; seed = training_seed(cfg, episode)
    masks = make_mask_bank(cfg); mask = masks[episode]
    original_bank = Path(args.checkpoint).resolve().parent.parent/'training_masks.npy'
    bank = np.load(original_bank, allow_pickle=False)
    if bank.dtype != np.bool_ or not np.array_equal(bank, masks):
        raise ValueError('Saved training mask bank differs from registered bank')
    env = MatchedWalkerEnv(cfg, diagnostics=True)
    steps = []
    try:
        args._progress['phase'] = 'next_episode_reset_and_four_decisions'
        image = env.reset(seed=seed, mask=mask)
        controller.reset(image, episode_id=episode)
        reset_evidence = {'image': tree_hash(image), 'state': tree_hash(controller.state),
                          'info': jsonable(env.last_info), 'controller': tree_hash(controller.snapshot())}
        obs = {'image': image.copy(), 'reward': np.float32(0), 'is_first': np.bool_(True),
               'is_last': np.bool_(False), 'is_terminal': np.bool_(False)}
        # At a registered 500-action post-prefill boundary, the next scheduled
        # learner update is after four actual decisions, exactly as train.py.
        for t in range(4):
            action = controller.act()
            state = controller.state
            try:
                next_image, reward, done, info = env.step(action)
            except BaseException:
                args._progress['actual_counts_complete'] = False
                raise
            args._progress.update(probe_actions=env.action_decisions, probe_dmc_calls=env.episode_env_steps)
            if done or info['env_steps_this_step'] != 2: raise RuntimeError('Unexpected probe episode termination')
            replay.add(obs, action, state)
            controller.step(next_image, action, info, episode_id=episode)
            counts['agent_steps'] += 1; counts['env_steps'] += 2
            steps.append({'action': action.tolist(), 'image': tree_hash(next_image),
                          'reward': float(reward), 'info': jsonable(info), 'info_sha256': tree_hash(info),
                          'state': tree_hash(controller.state)})
            obs = {'image': next_image.copy(), 'reward': np.float32(reward), 'is_first': np.bool_(False),
                   'is_last': np.bool_(False), 'is_terminal': np.bool_(False)}
            target = max(0, counts['agent_steps']-5000)//4
            if (target > counts['update_steps']) != (t == 3):
                raise RuntimeError('Recovered scheduler did not request exactly one next update at action four')
        batch = next(replay.stream)
        next_batch = sample_descriptor(batch)
        sample_rng_hash = tree_hash(replay.replay.sampler.rng.bit_generator.state)
        args._progress['phase'] = 'one_real_official_gpu_update'
        metrics = backbone.train(batch, replay.replay)
        args._progress['probe_updates'] = 1
        counts['update_steps'] += 1
        after = backbone.snapshot()
        if after['training']['counters']['updates'] != counts['update_steps']:
            raise RuntimeError('Actual official update count differs')
        if tree_hash(after['inference_params']) != tree_hash(restored['inference_params']) or after['inference_version'] != restored['inference_version']:
            raise RuntimeError('Episode-frozen inference copy changed during update')
        if tree_hash(prefill_rng.bit_generator.state) != tree_hash(payload['prefill_rng']):
            raise RuntimeError('Post-prefill continuation consumed the random-prefill RNG')
        comparison = {'identity': identity, 'checkpoint_sha256': record['sha256'],
            'restored_training_tree_sha256': tree_hash(restored['training']),
            'restored_train_carry_sha256': tree_hash(restored['train_carry']),
            'restored_inference_sha256': tree_hash(restored['inference_params']),
            'restored_collector_rng_sha256': tree_hash(rng_loaded),
            'prefill_rng_sha256': tree_hash(prefill_rng.bit_generator.state),
            'next_prefill_action_clone': prefill_next.tolist(), 'restored_global_rng': before_host,
            'next_episode_index': episode, 'next_environment_seed': seed,
            'next_mask_sha256': mask_sha256(mask), 'reset': reset_evidence, 'steps': steps,
            'sample': next_batch, 'sampler_rng_after_sample_sha256': sample_rng_hash,
            'after_training_tree_sha256': tree_hash(after['training']),
            'after_train_carry_sha256': tree_hash(after['train_carry']),
            'after_inference_sha256': tree_hash(after['inference_params']),
            'after_collector_sha256': tree_hash(controller.snapshot()),
            'after_replay_contents_sha256': _replay_latent_hash(replay),
            'after_global_rng': {'numpy': tree_hash(np.random.get_state()), 'python': tree_hash(random.getstate())},
            'metrics': metrics, 'metrics_sha256': tree_hash(metrics), 'counters': counts,
            'gpu_devices': [{'id': x.id, 'platform': x.platform,
                             'kind': x.device_kind} for x in backbone.jax.devices()]}
    finally:
        args._progress.update(probe_actions=env.action_decisions, probe_dmc_calls=env.episode_env_steps)
        env.close()
    # Rehash source objects, not merely their probe links, after the real update.
    for ref in references:
        if sha_file(source/ref['path']) != ref['file_sha256']:
            raise RuntimeError('Original immutable replay object changed')
    if sha_file(args.checkpoint) != record['sha256']:
        raise RuntimeError('Original checkpoint changed')
    result = {'format': 'helixwm_matched_recovery_worker_v1', 'passed': True,
              'gpu_devices': [str(x) for x in backbone.jax.devices()],
              'all_devices_gpu': all(x.platform == 'gpu' for x in backbone.jax.devices()),
              'restored_comparisons': restored_comparisons, 'boundary_audit': audit,
              'storage': storage, 'comparison': comparison, 'original_files_unchanged': True,
              'probe_actions': 4, 'probe_dmc_calls': 8, 'probe_updates': 1,
              'actual_counts_complete': True,
              'writes_to_original_training_run': False}
    if not result['all_devices_gpu']: raise RuntimeError('Recovery gate requires an actual GPU backend')
    atomic_json(output/'result.json', result)
    return result


def compare_results(results, exitcodes):
    if len(results) != 2 or len(exitcodes) != 2:
        raise ValueError('Exactly two fresh processes are required')
    required = ('passed', 'all_devices_gpu', 'original_files_unchanged', 'actual_counts_complete')
    worker_checks = [exitcodes[i] == 0 and all(r.get(k) is True for k in required)
                     and r.get('probe_actions') == 4 and r.get('probe_dmc_calls') == 8
                     and r.get('probe_updates') == 1
                     and set(r.get('restored_comparisons', {})) == RESTORED_FIELDS
                     and all(v is True for v in r['restored_comparisons'].values()) for i, r in enumerate(results)]
    left, right = [r.get('comparison', {}) for r in results]
    keys = sorted(set(left) | set(right))
    comparisons = {k: k in left and k in right and tree_hash(left[k]) == tree_hash(right[k]) for k in keys}
    complete = COMPARISON_FIELDS == set(left) and COMPARISON_FIELDS == set(right)
    passed = all(worker_checks) and complete and bool(comparisons) and all(comparisons.values())
    return {'format': 'helixwm_matched_fresh_process_recovery_v1', 'passed': passed,
            'process_exitcodes': exitcodes, 'worker_checks': worker_checks,
            'comparison_schema_complete': complete, 'exact_comparisons': comparisons,
            'mismatched_fields': [k for k, v in comparisons.items() if not v],
            'tolerance': 'none_exact_or_fail', 'actual_probe_actions': sum(r.get('probe_actions', 0) for r in results),
            'actual_probe_dmc_calls': sum(r.get('probe_dmc_calls', 0) for r in results),
            'actual_probe_updates': sum(r.get('probe_updates', 0) for r in results),
            'actual_counts_complete': all(r.get('actual_counts_complete') is True for r in results),
            'claim': 'two_fresh_processes_same_checkpoint_next_scheduled_update_parity',
            'uninterrupted_live_process_comparison_performed': False,
            'formal_training_resumed': False, 'automatic_recovery_authorized': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--upstream', type=Path, required=True)
    parser.add_argument('--environment-lock', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--timeout-seconds', type=int, default=1800)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    source_run = args.checkpoint.resolve().parent.parent
    if output == source_run or output.is_relative_to(source_run) or source_run.is_relative_to(output):
        raise ValueError('Recovery evidence must be outside the original training run')
    if output.exists(): raise FileExistsError(output)
    if args.worker:
        try: run_worker(args)
        except BaseException as exc:
            output.mkdir(parents=True, exist_ok=True)
            atomic_json(output/'result.json', {'passed': False, 'error': repr(exc),
                        'traceback': traceback.format_exc(), 'writes_to_original_training_run': False,
                        **getattr(args, '_progress', {'actual_counts_complete': False})})
            raise
        return
    output.mkdir(parents=True, exist_ok=False)
    results, codes = [], []
    for index in range(2):
        child = output/f'process_{index+1}'
        command = [sys.executable, '-B', '-m', 'helixwm_matched.recovery_probe', '--worker',
                   '--checkpoint', str(args.checkpoint.resolve()), '--protocol', str(args.protocol.resolve()),
                   '--upstream', str(args.upstream.resolve()), '--environment-lock', str(args.environment_lock.resolve()),
                   '--output-dir', str(child)]
        with (output/f'process_{index+1}.log').open('xb') as log:
            try:
                run = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                     timeout=args.timeout_seconds, check=False)
                code = run.returncode
            except subprocess.TimeoutExpired:
                code = 124
        codes.append(code)
        result_path = child/'result.json'
        results.append(json.loads(result_path.read_text()) if result_path.exists()
                       else {'passed': False, 'error': f'No result; process exit {code}'})
    report = compare_results(results, codes)
    report['workers'] = [{'result': f'process_{i+1}/result.json', 'exitcode': c} for i, c in enumerate(codes)]
    atomic_json(output/'recovery_gate.json', report)
    print(json.dumps({'passed': report['passed'], 'report': str(output/'recovery_gate.json')}), flush=True)
    if not report['passed']: raise SystemExit(1)


if __name__ == '__main__': main()
