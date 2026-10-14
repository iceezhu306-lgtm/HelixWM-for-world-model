"""One bounded, explicitly counted HelixWM dual-chain training experiment."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
import traceback

os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
import numpy as np

from helixwm.environment import WalkerEnv
from helixwm.run_storage import exclusive_run_lock, prepare_fresh_run
from .backbone import OfficialDreamerBackbone, UPSTREAM_COMMIT
from .dual_chain import DualChain
from .evaluation import DropoutSchedule, evaluate
from .storage import Replay, atomic_json, load_checkpoint, save_checkpoint, sha256, tree_hash
from .preflight import canonical_config_sha256, read_identity


def utc():
    return datetime.now(timezone.utc).isoformat()


def append(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(data, allow_nan=False, sort_keys=True) + '\n')
        stream.flush()


def command(args):
    return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT, timeout=30).strip()


def resources(run_dir, counters):
    row = {'utc': utc(), **counters, 'disk_free_bytes': shutil.disk_usage(run_dir).free}
    for name in ('memory.current', 'memory.max', 'memory.events'):
        path = Path('/sys/fs/cgroup') / name
        if path.exists():
            row['cgroup_' + name.replace('.', '_')] = path.read_text().strip()
    status = Path('/proc/self/status')
    if status.exists():
        row['process_memory'] = {line.split(':')[0]: line.split(':', 1)[1].strip()
                                 for line in status.read_text().splitlines()
                                 if line.startswith(('VmRSS:', 'VmHWM:'))}
    row['gpu'] = command(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu',
                          '--format=csv,noheader,nounits'])
    append(Path(run_dir)/'resources.jsonl', row)
    return row


def metadata(config, backbone):
    return {'created_utc': utc(), 'source_commit': command(['git', 'rev-parse', 'HEAD']),
            'source_status': command(['git', 'status', '--porcelain']),
            'upstream_commit': UPSTREAM_COMMIT, 'python': sys.version,
            'packages': {name: importlib.metadata.version(name) for name in
                         ('jax', 'jaxlib', 'numpy', 'dm-control', 'mujoco', 'elements', 'ninjax',
                          'imageio', 'imageio-ffmpeg', 'matplotlib')},
            'gpu': command(['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv,noheader']),
            'experiment_class': config['experiment_class'],
            'resolved_backbone_config': backbone.resolved_config,
            'units': {'agent_steps': 'executed actions, excluding reset and evaluation',
                      'env_steps': 'DMC calls, not MuJoCo integration substeps', 'action_repeat': 2},
            'observation_protocol': 'known synthetic collector missingness; immutable real-frame model targets',
            'baseline_rerun': False, 'paper_level_complete': False,
            'cross_hardware_bitwise_recovery_claimed': False}


def new_episode(image):
    return {'image': [image.copy()], 'action': [], 'reward': [0.0], 'is_first': [True],
            'is_last': [False], 'is_terminal': [False]}


def episode_arrays(episode):
    arrays = {key: np.asarray(value, dtype=(np.uint8 if key == 'image' else
              np.bool_ if key.startswith('is_') else np.float32)) for key, value in episode.items()}
    # The final observation has no executed outgoing action.
    arrays['action'] = np.concatenate([arrays['action'], np.zeros((1, 6), np.float32)])
    return arrays


def batch_for_backbone(replay, cfg):
    batch = replay.sample(cfg['batch_size'], cfg['batch_length'] + cfg['burnin'])
    shape = batch['reward'].shape
    batch['consec'] = np.zeros(shape, np.int32)
    # replay_context=0; identifiers are unused by the original loss.
    batch['stepid'] = np.zeros((*shape, 20), np.uint8)
    return batch


def validate_config(config):
    for key in ('budget_agent_steps', 'checkpoint_every', 'eval_every'):
        if not isinstance(config[key], int) or config[key] <= 0 or config[key] % 500:
            raise ValueError('Budget/checkpoint/evaluation boundaries must be full 500-action episodes')
    if config['eval_every'] % config['checkpoint_every']:
        raise ValueError('Every evaluation must have an immutable checkpoint')
    if config['environment']['action_repeat'] != 2 or config['environment']['episode_env_steps'] != 1000:
        raise ValueError('The declared Walker protocol requires repeat 2 and 1,000 DMC calls per episode')
    if not 0 <= config['prefill_agent_steps'] < config['budget_agent_steps']:
        raise ValueError('Invalid initial collection budget')
    if config['train_ratio'] <= 0:
        raise ValueError('Training ratio must be positive')


def verify_launch_gates(config, identity, gate_path, smoke_evidence):
    if gate_path is None or smoke_evidence is None:
        raise ValueError('Formal launch requires both preflight and completed smoke evidence')
    gate = json.loads(Path(gate_path).read_text())
    smoke = json.loads(Path(smoke_evidence).read_text())
    for record in (gate, smoke):
        if record.get('identity', {}).get('identity_sha256') != identity['identity_sha256']:
            raise ValueError('Launch evidence belongs to a different code/config/upstream identity')
    if not gate.get('passed') or identity['config_sha256'] != canonical_config_sha256(config):
        raise ValueError('Formal configuration did not pass its actual-size preflight')
    report = smoke.get('evaluation_report', {})
    if not (smoke.get('budget_complete') and smoke.get('agent_steps') == 1000
            and smoke.get('env_steps') == 2000 and smoke.get('checkpoint_reload_equal')
            and report.get('status') == 'complete' and report.get('hash_gate', {}).get('passed')):
        raise ValueError('End-to-end smoke and frozen checkpoint evaluation have not passed')
    if Path(smoke_evidence).with_name('train.exit').read_text().strip() != '0':
        raise ValueError('Smoke launcher did not exit successfully')
    return {'preflight_sha256': sha256(gate_path), 'smoke_completion_sha256': sha256(smoke_evidence)}


def restore_boundary(payload, backbone, replay, chain, dropout, env):
    """Restore a completed-episode checkpoint; the next operation is reset.

    The old chain is audit history, not an active state under new parameters.
    Its independent RNG streams are restored before a fresh episode re-anchor.
    """
    collector = payload['collector']
    counters, config = payload['counters'], payload['config']
    if collector['next_operation'] != 'reset_new_episode_and_reanchor':
        raise ValueError('This runner only resumes completed-episode checkpoints')
    steps = counters['agent_steps']
    if not isinstance(steps, int) or isinstance(steps, bool) or steps <= 0 or steps % 500:
        raise ValueError('Recovery action count is not a positive episode boundary')
    if (not collector['environment']['done'] or collector['environment']['episode_steps'] != 1000
            or counters['env_steps'] != 2*steps or counters['episodes'] != steps//500
            or payload['replay']['total_steps_seen'] != steps):
        raise ValueError('Recovery checkpoint is not an episode boundary')
    expected_updates = int(max(0, steps-config['prefill_agent_steps']) * config['train_ratio'] /
                           (config['backbone']['batch_size']*config['backbone']['batch_length']))
    official = payload['backbone']['training']['counters']
    if (counters['update_steps'] != expected_updates or official['updates'] != expected_updates
            or official['batches'] != expected_updates
            or payload['scheduler']['target_updates'] != expected_updates
            or payload['scheduler']['train_ratio'] != config['train_ratio']):
        raise ValueError('Recovery learner counters disagree with the frozen scheduler')
    backbone.load(payload['backbone'])
    replay.restore(payload['replay'], persist=True)
    env.restore(collector['environment'])
    chain.restore_completed_episode(collector['dual_chain'])
    dropout.load_state_dict(collector['dropout'])
    random.setstate(payload['rng']['python'])
    np.random.set_state(payload['rng']['numpy'])
    return payload['counters'].copy()


def copy_recovery_replay(checkpoint, payload, run_dir):
    source = Path(checkpoint).resolve().parent.parent / 'REPLAY'
    target = run_dir / 'REPLAY'
    target.mkdir(exist_ok=True)
    for row in payload['replay']['archive']:
        original, destination = source / row['file'], target / row['file']
        if original.stat().st_size != row['bytes'] or sha256(original) != row['sha256']:
            raise ValueError('Recovery replay does not match checkpoint SHA256')
        if destination.exists():
            raise FileExistsError(destination)
        try:
            os.link(original, destination)
        except OSError:
            shutil.copy2(original, destination)
        atomic_json(destination.with_suffix('.json'), row)


def _run(config, run_dir, upstream, *, smoke=False, gate_path=None, smoke_evidence=None,
         config_path=None, resume=None):
    validate_config(config)
    prepare_fresh_run(run_dir)
    identity = read_identity(upstream, config_path)
    gates = None
    if not smoke:
        gates = verify_launch_gates(config, identity, gate_path, smoke_evidence)
        if config['budget_agent_steps'] != 1_000_000:
            raise ValueError('Formal budget must remain one million actions')
    if command(['git', 'status', '--porcelain']):
        raise RuntimeError('Freeze a clean source release before a smoke or formal run')
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    atomic_json(run_dir/'config.json', config)
    atomic_json(run_dir/'command.json', {'argv': sys.argv, 'cwd': str(Path.cwd()), 'python': sys.executable})
    (run_dir/'train.pid').write_text(str(os.getpid())+'\n')
    (run_dir/'pip-freeze.txt').write_text(command([sys.executable, '-m', 'pip', 'freeze'])+'\n')
    backbone = OfficialDreamerBackbone(config['backbone'], upstream, run_dir)
    meta = metadata(config, backbone)
    meta['identity'] = identity
    meta['launch_gates'] = gates
    atomic_json(run_dir/'metadata.json', meta)
    atomic_json(run_dir/'resolved_backbone_config.json', backbone.resolved_config)
    replay = Replay(run_dir/'REPLAY', config['replay_capacity_steps'], config['seed']+101,
                    config['replay_cache_episodes'])
    chain = DualChain(backbone, seed=config['seed']+303, **config['dual_chain'])
    dropout = DropoutSchedule(config['seed']+7001, config['observation_dropout'])
    env = WalkerEnv(seed=config['seed'], **config['environment'])
    counters = {'agent_steps': 0, 'env_steps': 0, 'update_steps': 0, 'episodes': 0,
                'masked_observations': 0, 'wall_time_s': 0.0}
    if resume:
        payload = load_checkpoint(resume)
        if payload['config'] != config or payload['metadata']['source_commit'] != meta['source_commit']:
            raise ValueError('Recovery requires the exact frozen code and configuration')
        copy_recovery_replay(resume, payload, run_dir)
        counters = restore_boundary(payload, backbone, replay, chain, dropout, env)
        if counters['agent_steps'] >= config['budget_agent_steps']:
            raise ValueError('Completed training cannot be resumed as a new experiment')
        meta['recovery'] = {'source_checkpoint': str(Path(resume).resolve()),
                            'sha256': sha256(resume), 'restored_agent_steps': counters['agent_steps'],
                            'source_artifacts_preserved': True, 'automatic_resume': False,
                            'earlier_logs_remain_in_source_run': True}
        atomic_json(run_dir/'metadata.json', meta)
    start = time.monotonic()-counters['wall_time_s']
    last_resource = 0.0
    pending_means = defaultdict(list)
    episode = None
    checkpoint_path = None
    last_eval = None
    atomic_json(run_dir/'status.json', {'state': 'running', 'utc': utc(), **counters})
    print(json.dumps({'event': 'training_started', 'utc': utc(), 'pid': os.getpid(), **counters}), flush=True)

    def events():
        for event in chain.pop_events():
            append(run_dir/'dual_chain_events.jsonl', {'agent_steps': counters['agent_steps'], **event})

    try:
        while counters['agent_steps'] < config['budget_agent_steps']:
            # Episode-boundary parameter synchronization keeps each repair version coherent.
            backbone.pin_inference()
            observation = env.reset().reshape(64, 64, 3)
            dropout.valid(is_first=True)
            episode = new_episode(observation)
            chain.reset(observation, episode_id=counters['episodes'], timestamp=0,
                        model_version=backbone.inference_version)
            events()
            episode_masked = 0
            for t in range(1, 501):
                action = np.asarray(chain.act(deterministic=False)[0], np.float32)
                if not np.isfinite(action).all() or (np.abs(action) > 1).any():
                    raise FloatingPointError('Invalid executed action')
                result = env.step(action)
                observation = result['observation'].reshape(64, 64, 3)
                valid = dropout.valid(is_last=result['done'])
                # Do not pass the clean frame to inference during a declared sensor gap.
                chain.step(observation if valid else None, action, episode_id=counters['episodes'],
                           timestamp=t, model_version=backbone.inference_version, valid=valid)
                episode['action'].append(action.copy())
                episode['image'].append(observation.copy())
                episode['reward'].append(float(result['reward']))
                episode['is_first'].append(False)
                episode['is_last'].append(bool(result['done']))
                episode['is_terminal'].append(bool(result['continuation'] == 0))
                counters['agent_steps'] += 1
                counters['env_steps'] += result['env_steps']
                counters['masked_observations'] += int(not valid)
                episode_masked += int(not valid)
                events()
                if result['done']:
                    if t != 500:
                        raise RuntimeError('Unexpected episode length violates the frozen budget protocol')
                    replay.add_episode(episode_arrays(episode))
                available = max(0, counters['agent_steps'] - config['prefill_agent_steps'])
                positions = config['backbone']['batch_size'] * config['backbone']['batch_length']
                target_updates = int(available * config['train_ratio'] / positions)
                while counters['update_steps'] < target_updates:
                    metrics = backbone.train(batch_for_backbone(replay, config['backbone']))
                    if not all(np.isfinite(value) for value in metrics.values()):
                        raise FloatingPointError('Nonfinite official learner metric')
                    counters['update_steps'] += 1
                    for key, value in metrics.items():
                        pending_means[key].append(value)
                counters['wall_time_s'] = time.monotonic()-start
                if counters['agent_steps'] % 100 == 0:
                    append(run_dir/'metrics.jsonl', {'kind': 'train', 'utc': utc(), **counters,
                           **{key: float(np.mean(value)) for key, value in pending_means.items()}})
                    pending_means.clear()
                    atomic_json(run_dir/'status.json', {'state': 'running', 'utc': utc(), **counters})
                if time.monotonic()-last_resource >= 60:
                    resources(run_dir, counters)
                    last_resource = time.monotonic()
            counters['episodes'] += 1
            row = {'kind': 'episode', 'utc': utc(), **counters,
                   'episode_return': sum(episode['reward']), 'episode_length': 500,
                   'episode_masked_observations': episode_masked, 'replay_disk_bytes': replay.disk_bytes}
            append(run_dir/'metrics.jsonl', row)
            print(json.dumps(row), flush=True)

            if counters['agent_steps'] % config['checkpoint_every'] == 0:
                # The environment is at a completed episode. Next collector action starts a new episode.
                # Store the old chain/version too for audit; it is not reused under the new parameters.
                finished_chain = chain.state_dict()
                backbone.pin_inference()
                payload = {'schema_version': 2, 'config': config, 'metadata': meta, 'counters': counters.copy(),
                           'backbone': backbone.snapshot(), 'replay': replay.snapshot(),
                           'collector': {'environment': env.snapshot(), 'episode': episode_arrays(episode),
                                         'dual_chain': finished_chain, 'dropout': dropout.state_dict(),
                                         'next_operation': 'reset_new_episode_and_reanchor'},
                           'rng': {'python': random.getstate(), 'numpy': np.random.get_state()},
                           'scheduler': {'target_updates': target_updates, 'train_ratio': config['train_ratio']}}
                checkpoint_path = save_checkpoint(run_dir/'CHECKPOINTS', payload)
                atomic_json(run_dir/'CHECKPOINTS/latest.json', {'file': checkpoint_path.name,
                            'agent_steps': counters['agent_steps'], 'sha256': sha256(checkpoint_path)})
                append(run_dir/'events.jsonl', {'event': 'checkpoint_saved', 'utc': utc(),
                       'file': checkpoint_path.name, **counters})
            final = counters['agent_steps'] == config['budget_agent_steps']
            if counters['agent_steps'] % config['eval_every'] == 0 or final:
                if checkpoint_path is None:
                    raise RuntimeError('Evaluation requires a disk checkpoint')
                before = backbone.parameter_hash()
                loaded = load_checkpoint(checkpoint_path, verify_replay=False)
                backbone.load(loaded['backbone'])
                if backbone.parameter_hash() != before:
                    raise RuntimeError('Strict checkpoint reload changed the model/optimizer tree')
                last_eval = evaluate(backbone, config, run_dir, counters['agent_steps'],
                                     kind='final' if final else 'periodic',
                                     episodes=config['final_eval_episodes'] if final else config['periodic_eval_episodes'],
                                     video=final, audit_disabled=final)
                if backbone.parameter_hash() != before:
                    raise RuntimeError('Frozen evaluation changed model or optimizer state')
                append(run_dir/'events.jsonl', {'event': 'frozen_evaluation_complete', 'utc': utc(),
                       'agent_steps': counters['agent_steps'], 'model_optimizer_unchanged': True})
                if final:
                    atomic_json(run_dir/'RESULTS/final_summary.json', last_eval)
        expected = config['budget_agent_steps']*2
        if counters['env_steps'] != expected:
            raise RuntimeError('DMC-call count does not match action repeat')
        completion = {'completed_utc': utc(), 'budget_complete': True, 'identity': identity, **counters,
                      'expected_env_steps_match': True, 'checkpoint_reload_equal': True,
                      'final_checkpoint': checkpoint_path.name, 'checkpoint_sha256': sha256(checkpoint_path),
                      'evaluation_report': last_eval, 'single_seed_only': True,
                      'efficacy_demonstrated': False, 'paper_level_complete': False}
        atomic_json(run_dir/'completion.json', completion)
        atomic_json(run_dir/'status.json', {'state': 'completed', 'utc': utc(), **counters})
        resources(run_dir, counters)
        print(json.dumps({'event': 'completed', 'utc': utc(), **counters}), flush=True)
    except BaseException as exc:
        atomic_json(run_dir/'failure.json', {'utc': utc(), 'error': repr(exc), 'traceback': traceback.format_exc(), **counters})
        raise
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--upstream', required=True)
    parser.add_argument('--preflight')
    parser.add_argument('--smoke-evidence')
    parser.add_argument('--resume', help='Explicit checkpoint recovery into a NEW directory; never automatic')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    if args.smoke:
        config.update(experiment_class='smoke_only_not_formal', budget_agent_steps=1000,
                      prefill_agent_steps=500, checkpoint_every=500, eval_every=1000,
                      periodic_eval_episodes=1, final_eval_episodes=2)
    run_dir = Path(args.run_dir).resolve()
    with exclusive_run_lock(run_dir):
        _run(config, run_dir, Path(args.upstream), smoke=args.smoke, gate_path=args.preflight,
             smoke_evidence=args.smoke_evidence, config_path=Path(args.config), resume=args.resume)


if __name__ == '__main__':
    main()
