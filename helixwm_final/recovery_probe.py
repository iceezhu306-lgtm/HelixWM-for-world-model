"""Fresh-process recovery of ONLY the isolated 5500-action Final bootstrap.

Two sequential fresh GPU processes restore the same trusted checkpoint and
execute four new preflight decisions plus one scheduled author learner update.
The auxiliary bootstrap regressor receives the same eligible visible pairs.
This is not automatic/formal training recovery and does not compare against an
uninterrupted original process. No old experiment checkpoint format is accepted.
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
import re
import shutil
import struct
import subprocess
import sys
import traceback

import numpy as np

FORMAT='helixwm_final_bootstrap_probe_v1'
IDENTITY_KEYS={'source_sha256','protocol_sha256','environment_lock_sha256','stage','fresh_initialization',
               'formal_training_actions','old_checkpoint_loaded'}
PAYLOAD_KEYS={'format','identity','counts','backbone','auxiliary','heads','replay',
              'completed_native_controller','prefill_rng','numpy_rng','python_rng'}
EXPECTED_COUNTS={'agent_steps':5500,'env_steps':11000,'episodes':11,'update_steps':125}
RECOVERY_SEED=72100
RECOVERY_REGISTRATION={'environment_seed':RECOVERY_SEED,'mask_protocol':'only_observation3_missing',
    'selection_timing':'post_bootstrap_independent_engineering_probe',
    'performance_comparison':False,'formal_training_seed_modified':False,
    'per_process_actions':4,'per_process_DMC_calls':8,'per_process_author_updates':1}
RESTORED_FIELDS=('training','train_carry','inference_params','inference_version',
    'writeback_updates','final_auxiliary_heads','final_inference_bundle','final_upstream_api_sha256')
COMPARISON_FIELDS={'identity','checkpoint_sha256','environment_lock_sha256','verification_tool_sha256','probe_registration',
    'restored_backbone','restored_auxiliary','restored_replay','restored_collector_rng',
    'restored_host_rng','prefill_rng','next_prefill_action_clone','next_environment_seed',
    'next_mask_sha256','reset','steps','sample','sampler_rng_after_sample',
    'after_backbone','after_auxiliary','after_heads','after_replay','after_collector',
    'after_host_rng','metrics','counts','gpu_devices'}


def sha_file(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda:stream.read(1<<20),b''):h.update(part)
    return h.hexdigest()


def tree_hash(value):
    """Framed exact type/container hash, supporting NumPy scalar/RNG values."""
    h=hashlib.sha256()
    def frame(tag,raw):h.update(tag);h.update(len(raw).to_bytes(8,'big'));h.update(raw)
    def visit(x):
        if isinstance(x,(np.ndarray,np.generic)):
            a=np.asarray(x)
            if a.dtype.hasobject or not np.isfinite(a).all():raise ValueError('Finite non-object recovery arrays required')
            frame(b'A',json.dumps([a.dtype.str,list(a.shape)]).encode());frame(b'V',a.tobytes(order='C'))
        elif type(x) is dict:
            frame(b'D',str(len(x)).encode())
            for key in sorted(x):
                if type(key) is not str:raise ValueError('Recovery dictionary keys must be strings')
                frame(b'K',key.encode());visit(x[key])
        elif type(x) in (tuple,list):
            frame(b'T' if type(x) is tuple else b'L',str(len(x)).encode())
            for v in x:visit(v)
        elif x is None:frame(b'N',b'')
        elif type(x) is bool:frame(b'B',b'1' if x else b'0')
        elif type(x) is int:frame(b'I',str(x).encode())
        elif type(x) is float:
            if not math.isfinite(x):raise ValueError('Nonfinite recovery scalar')
            frame(b'F',struct.pack('!d',x))
        elif type(x) is str:frame(b'S',x.encode())
        else:raise TypeError('Unsupported recovery value '+type(x).__name__)
    visit(value)
    return h.hexdigest()


def _sha(value):
    return type(value) is str and bool(re.fullmatch('[0-9a-f]{64}',value))


def _json(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def _write_json(path,value):
    with Path(path).open('xb') as stream:stream.write(_json(value));stream.flush();os.fsync(stream.fileno())


def jsonable(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,dict):return {k:jsonable(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [jsonable(v) for v in value]
    return value


def verify_identity(identity,*,runtime_source_sha256,protocol_sha256,environment_lock_sha256):
    if (type(identity) is not dict or set(identity)!=IDENTITY_KEYS
            or not _sha(runtime_source_sha256) or not _sha(protocol_sha256) or not _sha(environment_lock_sha256)
            or identity['source_sha256']!=runtime_source_sha256
            or identity['protocol_sha256']!=protocol_sha256
            or identity['environment_lock_sha256']!=environment_lock_sha256
            or identity['stage']!='isolated_shared_native_bootstrap_preflight'
            or identity['fresh_initialization'] is not True
            or type(identity['formal_training_actions']) is not int or identity['formal_training_actions']!=0
            or identity['old_checkpoint_loaded'] is not False):
        raise ValueError('Not the trusted current-source fresh Final bootstrap identity')
    return copy.deepcopy(identity)


def _inventory(replay):
    records={}
    for chunk in replay['chunks']:
        for record in chunk['arrays'].values():
            old=records.setdefault(record['path'],record)
            if old!=record:raise ValueError('Conflicting replay dependency records')
    return [records[k] for k in sorted(records)]


def validate_payload(payload,sidecar,identity):
    """Validate boundary/RNG/state identities before creating GPU/environment."""
    if type(payload) is not dict or set(payload)!=PAYLOAD_KEYS or payload['format']!=FORMAT:
        raise ValueError('Only the new isolated Final bootstrap format is allowed')
    if _json(payload['identity'])!=_json(identity):raise ValueError('Checkpoint source/protocol identity differs')
    counts=payload['counts']
    if (type(counts) is not dict or set(counts)!=set(EXPECTED_COUNTS)
            or any(type(v) is not int for v in counts.values()) or counts!=EXPECTED_COUNTS
            or sidecar.get('counts')!=counts):raise ValueError('The exact 5500-action boundary accounting differs')
    core=payload['backbone']
    if core.get('format')!='helixwm_final_backbone_v1':raise ValueError('Old backbone format cannot enter Final recovery')
    counters=core['training']['counters']
    if any(type(counters.get(k)) is not int or counters[k]!=125 for k in ('updates','batches')):
        raise ValueError('Author update/batch count differs')
    if core['writeback_updates']!=125 or core['inference_version']!=125:
        raise ValueError('Author writeback and next-episode inference pin differ')
    if core['final_inference_bundle']['model_sha256']!=sidecar.get('model_sha256'):
        raise ValueError('Pinned model sidecar identity differs')
    replay=payload['replay']
    if (replay.get('format')!='helixwm_native_replay_exact_v1' or replay.get('rows_added')!=5511
            or replay.get('length')!=65 or replay.get('online') is not False
            or replay.get('capacity')!=1000000 or replay.get('chunksize')!=1024
            or type(replay.get('chunks')) is not list or not replay['chunks']):
        raise ValueError('Replay must include each reset/action/final row at this boundary')
    if any(type(c.get('length')) is not int or not 0<=c['length']<=1024 for c in replay['chunks']):
        raise ValueError('Invalid replay chunk lengths')
    if sum(c['length'] for c in replay['chunks'])!=5511 or not _inventory(replay):
        raise ValueError('Populated replay content does not account for every recorded row')
    if _inventory(replay)!=sidecar.get('replay_dependencies'):raise ValueError('Complete replay inventory differs')
    completed=payload['completed_native_controller']
    if completed.get('controller')!='black_frame_native':raise ValueError('Only native shared-bootstrap continuation is allowed')
    state=completed['chain']['state']
    if state.get('episode_id')!=10 or state.get('timestamp')!=500 or state.get('anchor_time')!=500:
        raise ValueError('Native completed collector is not the final episode boundary')
    if (state.get('last_action') is not None or state.get('pending') is not None
            or state.get('missing_length')!=0 or state.get('abstaining') is not False):
        raise ValueError('Native boundary has unresolved control state')
    if set(completed['chain']['rngs'])!={'real','actor','dream','repair'}:raise ValueError('Collector RNG streams incomplete')
    for s in completed['chain']['rngs'].values():np.random.default_rng().bit_generator.state=copy.deepcopy(s)
    np.random.default_rng().bit_generator.state=copy.deepcopy(payload['prefill_rng'])
    np.random.RandomState().set_state(payload['numpy_rng']);random.Random().setstate(payload['python_rng'])
    from helixwm_final.epistemic import BootstrapDynamics,FrozenBootstrap,tree_digest
    auxiliary=BootstrapDynamics(0,{k:identity[k] for k in ('source_sha256','protocol_sha256')})
    auxiliary.load(payload['auxiliary'])
    heads=FrozenBootstrap(payload['heads'])
    if (heads.sha256!=sidecar.get('epistemic_heads_sha256') or not heads.ready
            or tree_digest(auxiliary.pinned)!=tree_digest(payload['heads'])
            or auxiliary.last_transition_id!=5500 or auxiliary.accepted_pairs+auxiliary.skipped_pairs!=5500):
        raise ValueError('Auxiliary learning/pin/count identity differs')
    return copy.deepcopy(counts)


def validate_output_path(checkpoint,output):
    source=Path(checkpoint).resolve().parent;output=Path(output).resolve()
    if output==source or output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError('Recovery evidence must be isolated from the original run')
    if output.exists():raise FileExistsError(output)
    return output


def verify_runtime_manifest(runtime_root,manifest,expected_source_sha256):
    """Bind ORIGINAL runtime bytes; this independent tool has a separate hash."""
    runtime_root=Path(runtime_root).resolve()
    if (type(manifest) is not dict or not manifest or not _sha(expected_source_sha256)
            or hashlib.sha256(_json(manifest)).hexdigest()!=expected_source_sha256):
        raise ValueError('Original runtime manifest differs from the checkpoint source identity')
    for name,digest in manifest.items():
        if type(name) is not str or '\\' in name or Path(name).is_absolute() or not _sha(digest):
            raise ValueError('Invalid original runtime manifest member')
        path=runtime_root/name
        if (not path.resolve().is_relative_to(runtime_root) or path.is_symlink()
                or not path.is_file() or sha_file(path)!=digest):
            raise ValueError('Original runtime byte identity changed: '+name)
    for package in ('helixwm_final','helixwm_matched','helixwm_v2'):
        found={p.relative_to(runtime_root).as_posix() for p in (runtime_root/package).rglob('*.py')}
        registered={name for name in manifest if name.startswith(package+'/') and name.endswith('.py')}
        if found!=registered:raise ValueError('Original runtime Python member set changed: '+package)
    required={'helixwm_final/backbone.py','helixwm_final/epistemic.py',
        'helixwm_final/protocol_v2.py','helixwm_matched/backbone.py'}
    if not required.issubset(manifest):raise ValueError('Original runtime manifest omits required recovery APIs')
    return expected_source_sha256


def verify_and_load(args):
    """External identity and source hashes are checked BEFORE trusted pickle use."""
    if not _sha(args.tool_sha256) or sha_file(Path(__file__))!=args.tool_sha256:
        raise ValueError('Independent verification tool differs from external trusted digest')
    identity_path=Path(args.source_identity)
    if not _sha(args.source_identity_sha256) or sha_file(identity_path)!=args.source_identity_sha256:
        raise ValueError('Source identity differs from external trusted digest')
    identity=json.loads(identity_path.read_bytes())
    manifest=json.loads((Path(args.checkpoint).resolve().parent/'source_manifest.json').read_bytes())
    source=verify_runtime_manifest(args.runtime_source_root,manifest,identity.get('source_sha256'))
    from helixwm_final.protocol_v2 import load_protocol,protocol_sha256
    protocol=load_protocol(args.protocol)
    verify_identity(identity,runtime_source_sha256=source,protocol_sha256=protocol_sha256(protocol),
                    environment_lock_sha256=args.environment_lock_sha256)
    checkpoint=Path(args.checkpoint)
    if checkpoint.is_symlink() or checkpoint.name!='bootstrap_000005500.pkl':raise ValueError('Wrong bootstrap checkpoint path')
    raw=checkpoint.read_bytes()
    if not _sha(args.checkpoint_sha256) or hashlib.sha256(raw).hexdigest()!=args.checkpoint_sha256:
        raise ValueError('Checkpoint differs from external trusted SHA256')
    if not _sha(args.environment_lock_sha256) or sha_file(args.environment_lock)!=args.environment_lock_sha256:
        raise ValueError('Environment lock differs from external trusted SHA256')
    sidecar=json.loads((checkpoint.parent/'checkpoint.json').read_bytes())
    if (sidecar.get('path')!=checkpoint.name or sidecar.get('sha256')!=args.checkpoint_sha256
            or type(sidecar.get('bytes')) is not int or sidecar['bytes']!=checkpoint.stat().st_size):
        raise ValueError('Bootstrap sidecar SHA/size/path differs')
    # This is a user-owned trusted job checkpoint, not a sandboxed pickle parser.
    payload=pickle.loads(raw)
    del raw
    if sha_file(checkpoint)!=args.checkpoint_sha256:raise ValueError('Checkpoint changed during read')
    validate_payload(payload,sidecar,identity)
    return payload,sidecar,identity


def materialize_replay(source,destination,records):
    """Copy verified immutable objects into a new private write pool, no links."""
    source=Path(source).resolve();destination=Path(destination).resolve()
    if destination==source or destination.is_relative_to(source) or source.is_relative_to(destination):
        raise ValueError('Probe replay must be separate from source replay')
    names=set()
    for record in records:
        name=record.get('path')
        if (type(name) is not str or not re.fullmatch('[0-9a-f]{64}\\.npz',name)
                or name in names or type(record.get('bytes')) is not int or record['bytes']<1
                or not _sha(record.get('file_sha256'))):raise ValueError('Invalid/duplicate replay object reference')
        names.add(name)
        path=source/name
        if path.is_symlink() or not path.is_file() or path.stat().st_size!=record['bytes'] or sha_file(path)!=record['file_sha256']:
            raise ValueError('Source replay object SHA/size mismatch')
    destination.mkdir(parents=True,exist_ok=False)
    for record in records:
        source_path=source/record['path'];target=destination/record['path']
        with source_path.open('rb') as src,target.open('xb') as dst:shutil.copyfileobj(src,dst,1<<20)
        if sha_file(target)!=record['file_sha256']:raise ValueError('Copied replay SHA mismatch')
    return {'copies':len(records),'hardlinks':0,'bytes':sum(r['bytes'] for r in records),
            'original_files_opened_for_write':False}


def sample_descriptor(batch):
    if any(np.asarray(v).shape[:2]!=(16,65) for v in batch.values()):raise ValueError('Native sampler must return B16 x65')
    if batch['consec'].dtype!=np.int32 or np.any(batch['consec']!=0):raise ValueError('Native stream annotation differs')
    return {'sha256':tree_hash(batch),'fields':{k:{'shape':list(v.shape),'dtype':str(v.dtype),
        'sha256':tree_hash(v)} for k,v in sorted(batch.items())},'learning_positions':1024,'context_positions':16}


def _replay_contents(replay):
    return tree_hash({str(k):{name:v[:c.length] for name,v in c.data.items()}
        for k,c in replay.replay.chunks.items() if c.length})


def run_worker(args):
    args._progress={'probe_actions':0,'probe_dmc_calls':0,'probe_updates':0,
        'auxiliary_update_attempts':0,'auxiliary_eligible_updates':0,'actual_counts_complete':True}
    payload,record,identity=verify_and_load(args)
    output=validate_output_path(args.checkpoint,args.output_new);output.mkdir(parents=True,exist_ok=False)
    _write_json(output/'command.json',{'argv':sys.argv,'mode':args.mode,'probe_registration':RECOVERY_REGISTRATION})
    references=record['replay_dependencies'];source=Path(args.checkpoint).resolve().parent/'REPLAY'
    storage=materialize_replay(source,output/'REPLAY',references)
    os.environ.setdefault('MUJOCO_GL','egl');os.environ.setdefault('PYOPENGL_PLATFORM','egl')
    os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
    from helixwm_final.backbone import FinalBackbone
    from helixwm_final.environment import original_environment_contract
    from helixwm_final.epistemic import BootstrapDynamics,FrozenBootstrap
    from helixwm_matched.backbone import NativeReplay
    from helixwm_matched.controllers import Controller
    from helixwm_matched.environment import MatchedWalkerEnv
    from helixwm_matched.env_probe import runtime_lock
    from helixwm_matched.protocol import mask_sha256
    from helixwm_final.protocol_v2 import load_protocol,make_mask_bank
    backbone=FinalBackbone({'seed':0,'size':'size12m','platform':'cuda','compute_dtype':'bfloat16',
        'batch_size':16,'batch_length':64},args.upstream,output/'MODEL')
    backbone.load(payload['backbone']);restored=backbone.snapshot()
    restored_checks={k:tree_hash(restored[k])==tree_hash(payload['backbone'][k]) for k in RESTORED_FIELDS}
    if not all(restored_checks.values()):raise RuntimeError('Strict Final backbone restore differs')
    rp=payload['replay'];replay=NativeReplay(backbone,output/'REPLAY',capacity=rp['capacity'],seed=rp['seed'],chunksize=rp['chunksize'])
    replay.load(rp);replayed=replay.snapshot()
    if tree_hash(replayed)!=tree_hash(rp):raise RuntimeError('Native replay state did not restore exactly')
    auxiliary=BootstrapDynamics(0,{k:identity[k] for k in ('source_sha256','protocol_sha256')});auxiliary.load(payload['auxiliary'])
    if tree_hash(auxiliary.snapshot())!=tree_hash(payload['auxiliary']):raise RuntimeError('Auxiliary state did not restore exactly')
    saved=payload['completed_native_controller'];controller=Controller(backbone,'black_frame_native',seed=0)
    boundary=controller.chain.restore_completed_episode(saved['chain'])
    collector_rng=controller.snapshot()['chain']['rngs']
    if tree_hash(collector_rng)!=tree_hash(saved['chain']['rngs']):raise RuntimeError('Collector RNG did not restore exactly')
    prefill=np.random.default_rng();prefill.bit_generator.state=copy.deepcopy(payload['prefill_rng'])
    cloned=np.random.default_rng();cloned.bit_generator.state=copy.deepcopy(payload['prefill_rng'])
    next_prefill=cloned.uniform(-1,1,6).astype(np.float32)
    np.random.set_state(payload['numpy_rng']);random.setstate(payload['python_rng'])
    host_before={'numpy':tree_hash(np.random.get_state()),'python':tree_hash(random.getstate())}
    cfg=original_environment_contract();protocol=load_protocol(args.protocol)
    masks=make_mask_bank(protocol,partition='smoke')
    source_masks=np.load(Path(args.checkpoint).resolve().parent/'masks.npy',allow_pickle=False)
    if source_masks.dtype!=np.bool_ or not np.array_equal(source_masks,masks):raise ValueError('Original smoke mask bytes differ')
    # Independent engineering mask, fixed before this probe, not a formal or
    # heldout performance trajectory. It exercises both visible and missing data.
    mask=np.ones(501,np.bool_);mask[3]=False
    seed=RECOVERY_SEED;counts=copy.deepcopy(payload['counts'])
    env=MatchedWalkerEnv(cfg,diagnostics=True);steps=[]
    try:
        image=env.reset(seed=seed,mask=mask)
        actual_lock=runtime_lock(cfg);expected_lock=json.loads(Path(args.environment_lock).read_bytes())
        if _json(actual_lock)!=_json(expected_lock):raise ValueError('Live environment differs from the measured bootstrap lock')
        _write_json(output/'live_environment_lock.json',actual_lock)
        controller.reset(image,episode_id=11)
        reset={'image':tree_hash(image),'state':tree_hash(controller.state),'info':jsonable(env.last_info)}
        observation={'image':image.copy(),'reward':np.float32(0.),'is_first':np.bool_(True),
            'is_last':np.bool_(False),'is_terminal':np.bool_(False)}
        available=True
        for t in range(4):
            action=controller.act();state=controller.state
            before=(env.action_decisions,env.episode_env_steps)
            try:next_image,reward,done,info=env.step(action)
            finally:
                args._progress['probe_actions']=env.action_decisions
                args._progress['probe_dmc_calls']=env.episode_env_steps
                counts['agent_steps']+=env.action_decisions-before[0];counts['env_steps']+=env.episode_env_steps-before[1]
            if done or info['env_steps_this_step']!=2:raise RuntimeError('Unexpected next-episode probe outcome')
            replay.add(observation,action,state)
            outcome=auxiliary.update(image,action,next_image,float(reward),float(not info['terminated']),
                transition_id=counts['agent_steps'],previous_available=available,next_available=bool(info['availability']))
            args._progress['auxiliary_update_attempts']+=1
            args._progress['auxiliary_eligible_updates']+=int(outcome['updated'])
            controller.step(next_image,action,info,episode_id=11)
            steps.append({'action':action.tolist(),'reward':float(reward),'info':jsonable(info),
                'image_sha256':tree_hash(next_image),'state_sha256':tree_hash(controller.state),
                'auxiliary_sha256':tree_hash(auxiliary.snapshot())})
            image=next_image;available=bool(info['availability'])
            observation={'image':image.copy(),'reward':np.float32(reward),'is_first':np.bool_(False),
                'is_last':np.bool_(False),'is_terminal':np.bool_(info['terminated'])}
            target=max(0,counts['agent_steps']-5000)//4
            if (target>counts['update_steps'])!=(t==3):raise RuntimeError('Next scheduled update is not at fourth action')
        if args._progress['auxiliary_eligible_updates']<1:raise RuntimeError('Probe prefix did not exercise an eligible auxiliary update')
        batch=next(replay.stream);sample=sample_descriptor(batch)
        sampler_rng=tree_hash(replay.replay.sampler.rng.bit_generator.state)
        metrics=backbone.train(batch,replay.replay)
        args._progress['probe_updates']=1;counts['update_steps']+=1
        after=backbone.snapshot();next_heads=auxiliary.pin()
        if after['training']['counters']['updates']!=126 or after['writeback_updates']!=126:raise RuntimeError('One author update was not committed')
        for k in ('inference_params','inference_version','final_auxiliary_heads','final_inference_bundle'):
            if tree_hash(after[k])!=tree_hash(restored[k]):raise RuntimeError('Episode-frozen inference changed during update')
        if tree_hash(prefill.bit_generator.state)!=tree_hash(payload['prefill_rng']):raise RuntimeError('Post-prefill path consumed random-prefill RNG')
        comparison={'identity':identity,'checkpoint_sha256':record['sha256'],
            'environment_lock_sha256':args.environment_lock_sha256,
            'verification_tool_sha256':args.tool_sha256,
            'probe_registration':copy.deepcopy(RECOVERY_REGISTRATION),
            'restored_backbone':{k:tree_hash(restored[k]) for k in RESTORED_FIELDS},
            'restored_auxiliary':tree_hash(payload['auxiliary']),'restored_replay':tree_hash(replayed),
            'restored_collector_rng':tree_hash(collector_rng),'restored_host_rng':host_before,
            'prefill_rng':tree_hash(prefill.bit_generator.state),'next_prefill_action_clone':next_prefill.tolist(),
            'next_environment_seed':seed,'next_mask_sha256':mask_sha256(mask),'reset':reset,'steps':steps,
            'sample':sample,'sampler_rng_after_sample':sampler_rng,
            'after_backbone':{k:tree_hash(after[k]) for k in RESTORED_FIELDS},
            'after_auxiliary':tree_hash(auxiliary.snapshot()),'after_heads':next_heads.sha256,
            'after_replay':_replay_contents(replay),'after_collector':tree_hash(controller.snapshot()),
            'after_host_rng':{'numpy':tree_hash(np.random.get_state()),'python':tree_hash(random.getstate())},
            'metrics':metrics,'counts':counts,'gpu_devices':[{'id':x.id,'platform':x.platform,'kind':x.device_kind} for x in backbone.jax.devices()]}
    finally:
        args._progress.update(probe_actions=env.action_decisions,probe_dmc_calls=env.episode_env_steps)
        env.close()
    for ref in references:
        if sha_file(source/ref['path'])!=ref['file_sha256']:raise RuntimeError('Original replay object changed')
    if sha_file(args.checkpoint)!=args.checkpoint_sha256:raise RuntimeError('Original checkpoint changed')
    if sha_file(args.environment_lock)!=args.environment_lock_sha256 or sha_file(args.source_identity)!=args.source_identity_sha256:
        raise RuntimeError('Original input identity evidence changed')
    all_gpu=all(x.platform=='gpu' for x in backbone.jax.devices())
    if not all_gpu:raise RuntimeError('Probe did not run entirely on the registered GPU platform')
    result={'schema':'helixwm_final_bootstrap_recovery_worker_v1','passed':True,
        'all_devices_gpu':all_gpu,'original_files_unchanged':True,'restored_comparisons':restored_checks,
        'boundary_audit':boundary,'storage':storage,'comparison':comparison,**args._progress}
    _write_json(output/'result.json',result)
    print(json.dumps({'passed':True,'result':str(output/'result.json')}),flush=True)


def compare_results(results,exitcodes):
    if len(results)!=2 or len(exitcodes)!=2:raise ValueError('Exactly two independent fresh processes are required')
    worker_checks=[]
    for r,code in zip(results,exitcodes):
        flags=all(r.get(k) is True for k in ('passed','all_devices_gpu','original_files_unchanged','actual_counts_complete'))
        restored=r.get('restored_comparisons',{})
        worker_checks.append(code==0 and flags and r.get('probe_actions')==4 and r.get('probe_dmc_calls')==8
            and r.get('probe_updates')==1 and r.get('auxiliary_update_attempts')==4
            and type(r.get('auxiliary_eligible_updates')) is int and 1<=r['auxiliary_eligible_updates']<=4
            and set(restored)==set(RESTORED_FIELDS) and all(v is True for v in restored.values()))
    left,right=[r.get('comparison',{}) for r in results]
    exact={k:k in left and k in right and tree_hash(left[k])==tree_hash(right[k]) for k in sorted(COMPARISON_FIELDS)}
    complete=set(left)==COMPARISON_FIELDS and set(right)==COMPARISON_FIELDS
    passed=all(worker_checks) and complete and all(exact.values())
    return {'schema':'helixwm_final_bootstrap_recovery_gate_v1','passed':passed,'worker_checks':worker_checks,
        'comparison_schema_complete':complete,'exact_comparisons':exact,'exitcodes':exitcodes,
        'mismatched_fields':[k for k,v in exact.items() if not v],
        'actual_probe_actions':sum(r.get('probe_actions',0) for r in results),
        'actual_probe_dmc_calls':sum(r.get('probe_dmc_calls',0) for r in results),
        'actual_probe_updates':sum(r.get('probe_updates',0) for r in results),
        'tolerance':'none_exact_or_fail','claim':'two_fresh_processes_same_bootstrap_next_scheduled_update_parity',
        'uninterrupted_live_process_comparison_performed':False,'formal_training_resumed':False,
        'automatic_recovery_authorized':False,'cross_hardware_or_long_horizon_identity_proven':False,
        'probe_registration':copy.deepcopy(RECOVERY_REGISTRATION)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint','source-identity','protocol','environment-lock','upstream','output-new','runtime-source-root'):
        parser.add_argument('--'+name,type=Path,required=True)
    for name in ('checkpoint-sha256','source-identity-sha256','environment-lock-sha256','tool-sha256'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--mode',choices=('pair','reference','restored'),default='pair')
    parser.add_argument('--timeout-seconds',type=int,default=1800)
    args=parser.parse_args()
    # Standalone tool installation does not add/modify a file in frozen source.
    sys.path.insert(0,str(args.runtime_source_root.resolve()))
    if args.timeout_seconds<1:raise ValueError('Positive worker timeout required')
    output=validate_output_path(args.checkpoint,args.output_new)
    if args.mode!='pair':
        try:run_worker(args)
        except BaseException as exc:
            output.mkdir(parents=True,exist_ok=True)
            _write_json(output/'result.json',{'passed':False,'error':repr(exc),'traceback':traceback.format_exc(),
                'writes_to_original_run':False,**getattr(args,'_progress',{'actual_counts_complete':False})})
            raise
        return
    # Verify identities/checkpoint once before dispatch, then each child verifies independently.
    verify_and_load(args)
    output.mkdir(parents=True,exist_ok=False)
    results=[];codes=[]
    for index,mode in enumerate(('reference','restored')):
        command=[sys.executable,'-B',str(Path(__file__).resolve()),'--mode',mode]
        for name in ('checkpoint','source_identity','protocol','environment_lock','upstream','runtime_source_root'):
            command.extend(['--'+name.replace('_','-'),str(getattr(args,name).resolve())])
        for name in ('checkpoint_sha256','source_identity_sha256','environment_lock_sha256','tool_sha256'):
            command.extend(['--'+name.replace('_','-'),getattr(args,name)])
        command.extend(['--output-new',str(output/mode)])
        _write_json(output/(mode+'_command.json'),{'argv':command})
        with (output/(mode+'.log')).open('xb') as log:
            try:code=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=False,timeout=args.timeout_seconds).returncode
            except subprocess.TimeoutExpired:code=124
        codes.append(code);result_path=output/mode/'result.json'
        results.append(json.loads(result_path.read_bytes()) if result_path.exists() else {'passed':False,'error':'Worker produced no result'})
        if code or results[-1].get('passed') is not True:
            # Fail-stop: do not use another GPU run to retry a failed worker.
            if len(results)==1:results.append({'passed':False,'error':'Not started after first worker failure'});codes.append(None)
            break
    report=compare_results(results,codes)
    report['result_files']=['reference/result.json','restored/result.json']
    _write_json(output/'recovery_gate.json',report)
    print(json.dumps({'passed':report['passed'],'report':str(output/'recovery_gate.json')}),flush=True)
    if not report['passed']:raise SystemExit(1)


if __name__=='__main__':main()
