"""Fixed-action old/Final DMC equivalence gate; no model or learning imports.

Each worker runs exactly one 500-decision episode in a fresh process. This is
engineering evidence only, not a training or scientific performance result.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
from pathlib import Path
import struct
import subprocess
import sys
import traceback

import numpy as np

from helixwm_matched.env_probe import first_difference, runtime_lock
from helixwm_matched.environment import MatchedWalkerEnv
from helixwm_matched.protocol import array_sha256, canonical_json, gap_segments, validate_mask
from .environment import FinalWalkerEnv, original_environment_contract

SEED = 44000
ACTION_SEED = 944000
GAPS = ((10, 1), (30, 5), (60, 10), (100, 20), (160, 50), (270, 100))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def write_json(path, value):
    with Path(path).open('xb') as stream:
        stream.write(canonical_json(value) + b'\n')


def fixed_inputs():
    actions = np.random.default_rng(ACTION_SEED).uniform(-1, 1, (500, 6)).astype(np.float32)
    mask = np.ones(501, np.bool_)
    for start, length in GAPS:
        mask[start:start+length] = False
    return actions, validate_mask(mask)


def binary64(value):
    if type(value) is not float or not math.isfinite(value):
        raise ValueError('A finite native Python binary64 scalar is required')
    return {'value': value, 'hex': value.hex(), 'little_endian_bytes': struct.pack('<d', value).hex()}


def validate_row(row, index, action, mask):
    """Strict per-step budget, action and visibility contract, without tolerances."""
    if row['action_index'] != index or row['input_observation_timestamp'] != index:
        raise ValueError('Input timestamp mismatch')
    if row['observation_timestamp'] != index+1:
        raise ValueError('Output timestamp mismatch')
    if row['input_available'] != bool(mask[index]) or row['availability'] != bool(mask[index+1]):
        raise ValueError('Availability mask mismatch')
    if row['DMC_calls'] != 2 or row['episode_DMC_calls'] != 2*(index+1):
        raise ValueError('DMC call budget mismatch')
    if row['done'] != (index == 499) or row['terminated'] or row['truncated'] != (index == 499):
        raise ValueError('Walker time-limit semantics changed')
    if row['executed_action_sha256'] != array_sha256(action):
        raise ValueError('Executed action bytes differ')
    if row['reward'] != binary64(row['reward']['value']):
        raise ValueError('Reward scalar representation changed')
    if not 0 <= row['reward']['value'] <= 2:
        raise ValueError('Reward outside native repeated-action bounds')
    if len(row['subcalls']) != 2:
        raise ValueError('Missing native call evidence')
    for offset, sub in enumerate(row['subcalls'], 1):
        if sub['native_call'] != 2*index+offset or sub['last'] != (index == 499 and offset == 2):
            raise ValueError('Native subcall count/terminal mismatch')
        if array_sha256(np.asarray(sub['executed_action'], np.float32)) != array_sha256(action):
            raise ValueError('Native executed action differs')
        if not math.isfinite(sub['reward']) or not 0 <= sub['reward'] <= 1:
            raise ValueError('Invalid raw native reward')
    expected_reward = float(row['subcalls'][0]['reward']) + float(row['subcalls'][1]['reward'])
    if binary64(expected_reward) != row['reward']:
        raise ValueError('Reward is not the exact two native calls sum')
    if not row['availability'] and row['visible_RGB_sha256'] != array_sha256(np.zeros((64, 64, 3), np.uint8)):
        raise ValueError('Missing observation exposed a nonzero image')


def worker(kind, config_path, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = utc()
    actions, mask = fixed_inputs()
    config = json.loads(Path(config_path).read_bytes())
    cls = MatchedWalkerEnv if kind == 'old' else FinalWalkerEnv
    with cls(config, diagnostics=True) as env, (output/'trace.jsonl').open('xb') as trace:
        image = env.reset(SEED, mask)
        initial = {'seed': SEED, 'observation_timestamp': 0, 'availability': True,
                   'visible_RGB_sha256': array_sha256(image),
                   'physics_state_sha256': env.last_info['probe']['physics_state_sha256']}
        trace.write(canonical_json(initial)+b'\n'); trace.flush()
        for index, action in enumerate(actions):
            image, reward, done, info = env.step(action)
            row = {'action_index': index, 'input_observation_timestamp': info['input_observation_timestamp'],
                   'observation_timestamp': info['observation_timestamp'],
                   'input_available': info['input_available'], 'availability': info['availability'],
                   'visible_RGB_sha256': array_sha256(image), 'reward': binary64(reward),
                   'discount': binary64(info['discount']), 'done': done,
                   'terminated': info['terminated'], 'truncated': info['truncated'],
                   'DMC_calls': info['env_steps_this_step'], 'episode_DMC_calls': info['episode_env_steps'],
                   'executed_action': info['executed_action'].tolist(),
                   'executed_action_sha256': array_sha256(info['executed_action']),
                   'physics_state_sha256': info['probe']['physics_state_sha256'],
                   'subcalls': info['probe']['subcalls']}
            # Save the actual observation first, so a failure retains its evidence.
            trace.write(canonical_json(row)+b'\n'); trace.flush()
            validate_row(row, index, action, mask)
        lock = runtime_lock(original_environment_contract())
        if env.action_decisions != 500 or env.episode_env_steps != 1000:
            raise ValueError('Incomplete native episode budget')
    forbidden = [m for m in sys.modules if m == 'jax' or m.startswith(('jax.', 'helixwm_final.backbone', 'helixwm_matched.backbone'))]
    if forbidden:
        raise RuntimeError('Unexpected model/JAX import: '+str(forbidden))
    write_json(output/'environment_lock.json', lock)
    result = {'passed': True, 'kind': kind, 'started_utc': started, 'ended_utc': utc(),
              'pid': __import__('os').getpid(), 'seed': SEED, 'actions_seed': ACTION_SEED,
              'action_dtype': actions.dtype.str, 'actions_sha256': array_sha256(actions),
              'mask_sha256': array_sha256(mask), 'mask': mask.tolist(), 'gap_segments': gap_segments(mask),
              'action_decisions': 500, 'DMC_calls': 1000, 'trace_sha256': sha(output/'trace.jsonl'),
              'lock_sha256': sha(output/'environment_lock.json'),
              'training_updates': 0, 'model_imports': forbidden, 'old_checkpoints_read': False}
    write_json(output/'worker_result.json', result)
    return result


def compare_outputs(old_dir, final_dir, original_lock):
    a, b = Path(old_dir), Path(final_dir)
    left, right = (a/'trace.jsonl').read_bytes(), (b/'trace.jsonl').read_bytes()
    if left != right:
        for i, (x, y) in enumerate(zip(left.splitlines(), right.splitlines())):
            if x != y:
                raise ValueError('Exact trace mismatch at row '+str(i)+': '+str(first_difference(json.loads(x), json.loads(y))))
        raise ValueError('Exact trace length mismatch')
    if len(left.splitlines()) != 501:
        raise ValueError('Trace must contain reset plus 500 actions')
    locks = [json.loads((p/'environment_lock.json').read_bytes()) for p in (a, b)]
    for candidate in locks:
        difference = first_difference(original_lock, candidate)
        if difference:
            raise ValueError('Original runtime environment lock differs at '+difference)
    results = [json.loads((p/'worker_result.json').read_bytes()) for p in (a, b)]
    for key in ('seed', 'actions_seed', 'action_dtype', 'actions_sha256', 'mask_sha256', 'mask',
                'gap_segments', 'action_decisions', 'DMC_calls', 'trace_sha256', 'lock_sha256',
                'training_updates', 'model_imports', 'old_checkpoints_read'):
        if first_difference(results[0][key], results[1][key]):
            raise ValueError('Worker evidence differs: '+key)
    if results[0]['pid'] == results[1]['pid']:
        raise ValueError('Workers did not have independent process identities')
    return {'two_fresh_processes': True, 'exact_trace_bytes': True, 'original_runtime_lock_matches': True,
            'exact_501_visible_RGB_hashes': True, 'exact_1000_native_subcalls_per_arm': True,
            'all_six_gap_lengths': [length for _, length in GAPS], 'no_model_or_learning': True}


def run(old_config, final_config, original_lock_path, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    report = {'passed': False, 'started_utc': utc(), 'training_started': False,
              'formal_training_actions': 0, 'engineering_actions_expected': 1000,
              'engineering_DMC_calls_expected': 2000, 'checks': {}}
    try:
        original_lock = json.loads(Path(original_lock_path).read_bytes())
        write_json(output/'original_environment_lock.json', original_lock)
        for kind, config in [('old', old_config), ('final', final_config)]:
            argv = [sys.executable, '-B', '-u', '-m', 'helixwm_final.env_probe', '--worker', kind,
                    '--config', str(config), '--output', str(output/kind)]
            write_json(output/(kind+'_command.json'), {'argv': argv})
            with (output/(kind+'.console.log')).open('xb') as log:
                result = subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT, timeout=600)
            (output/(kind+'.exit')).write_text(str(result.returncode)+'\n')
            if result.returncode:
                raise RuntimeError(kind+' worker failed; preserved evidence; no automatic retry')
        report['checks'] = compare_outputs(output/'old', output/'final', original_lock)
        report.update(passed=True, engineering_actions_observed=1000, engineering_DMC_calls_observed=2000)
    except BaseException as exc:
        report.update(error=repr(exc), traceback=traceback.format_exc(),
                      partial_interaction_accounting='Retained trace JSONL records all returned actions before failure')
    report['ended_utc'] = utc()
    write_json(output/'environment_parity_report.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', choices=['old', 'final'])
    parser.add_argument('--config')
    parser.add_argument('--old-config')
    parser.add_argument('--final-config')
    parser.add_argument('--original-lock')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.worker:
        worker(args.worker, args.config, args.output)
        return 0
    result = run(args.old_config, args.final_config, args.original_lock, args.output)
    print(json.dumps(result, indent=2), flush=True)
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
