"""Independent fixed-action calibration images, excluded from learner replay.

The bank is collected once with shared deterministic random actions. It is not
an extra trained policy or an evaluation result. Hidden images may be used ONLY
for offline calibration diagnostics, never as a missing policy observation.
"""
from __future__ import annotations
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import numpy as np

FIT_SEEDS = tuple(range(55000, 55010))
VALIDATION_SEEDS = tuple(range(55010, 55020))
ACTIONS, IMAGE_SHAPE, ACTION_RNG_ROOT = 124, (64, 64, 3), 730000


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''): h.update(block)
    return h.hexdigest()


def fixed_actions(seed):
    if type(seed) is not int or seed not in FIT_SEEDS + VALIDATION_SEEDS:
        raise ValueError('Unregistered calibration seed')
    return np.random.default_rng(ACTION_RNG_ROOT + seed).uniform(-1, 1, (ACTIONS, 6)).astype(np.float32)


def _sha_format(value):
    return isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value)


def _image(value):
    if not isinstance(value,np.ndarray) or value.dtype!=np.uint8 or value.shape!=IMAGE_SHAPE:
        raise ValueError('Calibration raw image must already be uint8 RGB64')
    return value.copy()


def collect_bank(config, output, *, environment_lock_sha256, source_sha256, env_class=None):
    from .protocol import validate_protocol, protocol_sha256
    from .environment import FinalWalkerEnv
    config = validate_protocol(config)
    if not _sha_format(environment_lock_sha256) or not _sha_format(source_sha256):
        raise ValueError('Calibration collection requires source and environment binding')
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    factory = env_class or FinalWalkerEnv
    records = []; actions_count = calls = 0
    for seed in FIT_SEEDS + VALIDATION_SEEDS:
        env = factory(config, diagnostics=False)
        try:
            initial = env.reset(seed=seed, mask=np.ones(501, bool))
            frames = [_image(initial)]; rewards = []; call_counts = []
            actions = fixed_actions(seed)
            for t, action in enumerate(actions, 1):
                before_a, before_c = env.action_decisions, env.episode_env_steps
                try:
                    image, reward, done, info = env.step(action)
                finally:
                    actions_count += env.action_decisions - before_a
                    calls += env.episode_env_steps - before_c
                if done or not info['availability'] or info['observation_timestamp'] != t or info['env_steps_this_step'] != 2:
                    raise ValueError('Calibration bank environment contract differs')
                if not np.array_equal(info['executed_action'], action):
                    raise ValueError('Calibration fixed action changed')
                if type(reward) not in (float,np.float64) or not np.isfinite(reward) or not 0<=reward<=2:
                    raise ValueError('Calibration raw reward is invalid')
                frames.append(_image(image)); rewards.append(float(reward)); call_counts.append(2)
            path = output/f'seed{seed}.npz'
            with path.open('xb') as stream:
                np.savez_compressed(stream, image=np.asarray(frames, np.uint8), action=actions,
                    reward=np.asarray(rewards,np.float64), dmc_calls=np.asarray(call_counts,np.int32))
            records.append({'seed':seed,'split':'fit' if seed in FIT_SEEDS else 'validation',
                            'file':path.name,'sha256':sha256(path),'bytes':path.stat().st_size})
        except BaseException as exc:
            (output/'failure.json').write_text(json.dumps({'seed':seed,'error':repr(exc),
                'acknowledged_actions':actions_count,'acknowledged_dmc_calls':calls,
                'partial_bank_not_usable':True}),encoding='utf8')
            raise
        finally:
            env.close()
    if (actions_count,calls) != (2480,4960): raise ValueError('Calibration bank budget differs')
    manifest={'format':'helixwm_final_calibration_bank_v1','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'protocol_sha256':protocol_sha256(config),'fit_seeds':list(FIT_SEEDS),
        'environment_lock_sha256':environment_lock_sha256,'source_sha256':source_sha256,
        'validation_seeds':list(VALIDATION_SEEDS),'actions_per_seed':ACTIONS,'actions':actions_count,
        'dmc_calls':calls,'training_updates':0,'included_in_training_budget':False,
        'purpose':'independent_offline_calibration_only_never_learner_replay_or_missing_policy_input',
        'files':records}
    (output/'manifest.json').write_text(json.dumps(manifest,sort_keys=True,indent=2),encoding='utf8')
    return manifest


def load_bank(directory, expected_manifest_sha256, protocol_sha256_value, *, environment_lock_sha256, source_sha256):
    directory=Path(directory).resolve(); path=directory/'manifest.json'
    if sha256(path) != expected_manifest_sha256:
        raise ValueError('Calibration manifest differs from registered digest')
    manifest=json.loads(path.read_text())
    if any(type(manifest.get(k)) is not int for k in ('actions_per_seed','actions','dmc_calls','training_updates')):
        raise ValueError('Calibration counts require exact integer types')
    if (manifest['format']!='helixwm_final_calibration_bank_v1'
            or manifest['protocol_sha256']!=protocol_sha256_value
            or not _sha_format(environment_lock_sha256) or not _sha_format(source_sha256)
            or manifest['environment_lock_sha256']!=environment_lock_sha256 or manifest['source_sha256']!=source_sha256
            or manifest['fit_seeds']!=list(FIT_SEEDS) or manifest['validation_seeds']!=list(VALIDATION_SEEDS)
            or manifest['actions_per_seed']!=124 or manifest['actions']!=2480 or manifest['dmc_calls']!=4960
            or manifest['training_updates']!=0 or manifest['included_in_training_budget'] is not False
            or (directory/'failure.json').exists()):
        raise ValueError('Calibration manifest identity/count/split invalid')
    if [r['seed'] for r in manifest['files']] != list(FIT_SEEDS+VALIDATION_SEEDS):
        raise ValueError('Calibration file set differs')
    bank={}
    for record in manifest['files']:
        seed=record['seed']
        if record['file']!=f'seed{seed}.npz' or record['split']!=('fit' if seed in FIT_SEEDS else 'validation'):
            raise ValueError('Calibration file path or split changed')
        file=directory/record['file']
        if sha256(file)!=record['sha256'] or file.stat().st_size!=record['bytes']:
            raise ValueError('Calibration file integrity failure')
        with np.load(file,allow_pickle=False) as archive:
            values={k:archive[k].copy() for k in archive.files}
        schema={'image':((125,*IMAGE_SHAPE),np.uint8),'action':((124,6),np.float32),
                'reward':((124,),np.float64),'dmc_calls':((124,),np.int32)}
        if set(values)!=set(schema):raise ValueError('Calibration field set changed')
        for key,(shape,dtype) in schema.items():
            if values[key].shape!=shape or values[key].dtype!=dtype or not np.isfinite(values[key]).all():
                raise ValueError('Calibration schema invalid: '+key)
        if (not np.array_equal(values['action'],fixed_actions(seed)) or np.any(values['dmc_calls']!=2)
                or np.any(values['reward']<0) or np.any(values['reward']>2)):
            raise ValueError('Calibration action/reward/call evidence differs')
        bank[seed]=values
    return bank,manifest


def main():
    from .protocol import load_protocol
    from .provenance import source_digest
    from .environment import original_environment_contract
    from helixwm_matched.env_probe import runtime_lock,first_difference
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--environment-lock',required=True,type=Path)
    args=parser.parse_args()
    expected=json.loads(args.environment_lock.read_text())
    mismatch=first_difference(expected,runtime_lock(original_environment_contract()))
    if mismatch:raise ValueError('Calibration live environment lock differs: '+mismatch)
    source,files=source_digest()
    result=collect_bank(load_protocol(args.protocol),args.output,
        environment_lock_sha256=sha256(args.environment_lock),source_sha256=source)
    if source_digest()!=(source,files):
        (args.output/'failure.json').write_text(json.dumps({'error':'source_changed_during_collection'}),encoding='utf8')
        raise ValueError('Calibration source changed during collection')
    mismatch=first_difference(expected,runtime_lock(original_environment_contract()))
    if mismatch:
        (args.output/'failure.json').write_text(json.dumps({'error':'environment_lock_changed','detail':mismatch}),encoding='utf8')
        raise ValueError('Calibration live environment changed')
    print(json.dumps(result))


if __name__=='__main__':main()
