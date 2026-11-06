"""Read-only strict loading of new bank-v3 formal checkpoints.

External checkpoint AND sidecar digests are required. JSON metadata, frozen
source/environment/protocol, all replay files, trajectory dependencies, and
bank-gate bundles are verified before loading a user-owned trusted pickle.
This is not a sandbox for untrusted pickle and never resumes training.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import pickle
import random
import re
import stat

import numpy as np

from .recovery_probe import tree_hash

FORMAT = 'helixwm_final_bank_episode_boundary_v3'
PREFLIGHT_FORMAT = 'helixwm_final_bank_preflight_boundary_v3'
PREFLIGHT_FIELDS = {'probe_registration','formal_training_actions','fresh_initialization','old_checkpoint_loaded'}
UPSTREAM = 'e3f02248693a79dc8b0ebd62c93683888ddaccfe'
IDENTITY_FIELDS = {'run_id','arm','source_sha256','protocol_sha256',
                   'environment_lock_sha256','upstream_commit','bank_manifest_sha256'}
COUNTS_FIELDS = {'agent_steps','env_steps','episodes','update_steps'}
SHA = re.compile(r'^[0-9a-f]{64}$')


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def sha_file(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def safe_path(root,name):
    if (type(name) is not str or not name or '\\' in name or ':' in name or '\x00' in name
            or PurePosixPath(name).is_absolute() or any(x in ('','.','..') for x in name.split('/'))):
        raise ValueError('Dependency path must be canonical relative POSIX text')
    root=Path(root).absolute();path=root.joinpath(*name.split('/'))
    for p in (path,*path.parents):
        if p.exists() or p.is_symlink():
            s=p.lstat()
            if stat.S_ISLNK(s.st_mode) or getattr(s,'st_file_attributes',0)&0x400:
                raise ValueError('Dependency links/reparse points forbidden')
    if not path.resolve().is_relative_to(root.resolve()):raise ValueError('Dependency escapes run root')
    return path


def validate_identity(identity):
    if (type(identity) is not dict or set(identity)!=IDENTITY_FIELDS
            or identity['arm'] not in ('D','F') or identity['upstream_commit']!=UPSTREAM
            or type(identity['run_id']) is not str or not re.fullmatch(r'[A-Za-z0-9_-]+',identity['run_id'])
            or any(type(identity[k]) is not str or not SHA.fullmatch(identity[k]) for k in IDENTITY_FIELDS if k.endswith('sha256'))):
        raise ValueError('Complete bank-v3 run/arm/source/protocol/environment identity required')


def validate_counts(counts, *, require_final=False, preflight=False):
    if type(counts) is not dict or set(counts)!=COUNTS_FIELDS or any(type(x) is not int for x in counts.values()):
        raise ValueError('Exact integer formal budget counters required')
    n=counts['agent_steps']
    if preflight:
        if require_final or counts!={'agent_steps':6000,'env_steps':12000,'episodes':12,'update_steps':250}:
            raise ValueError('Only the exact isolated fresh 6000-action preflight boundary is permitted')
        return counts
    if (not 10000<=n<=1000000 or n%10000 or counts['env_steps']!=2*n
            or counts['episodes']!=n//500 or counts['update_steps']!=(n-5000)//4
            or require_final and n!=1000000):
        raise ValueError('Not a registered complete formal checkpoint boundary')
    return counts


def verify_descriptor(root,record, *, prefix=False):
    if (type(record) is not dict or set(record)!={'path','bytes','sha256'}
            or type(record['bytes']) is not int or record['bytes']<0
            or type(record['sha256']) is not str or not SHA.fullmatch(record['sha256'])):
        raise ValueError('Invalid dependency descriptor')
    path=safe_path(root,record['path']);before=path.stat()
    if not stat.S_ISREG(before.st_mode) or (before.st_size<record['bytes'] if prefix else before.st_size!=record['bytes']):
        raise ValueError('Dependency byte count/type differs')
    h=hashlib.sha256();left=record['bytes']
    with path.open('rb') as stream:
        while left:
            data=stream.read(min(left,4*1024*1024))
            if not data:raise ValueError('Dependency truncated during verification')
            left-=len(data);h.update(data)
    after=path.stat()
    if h.hexdigest()!=record['sha256'] or before.st_ino!=after.st_ino:
        raise ValueError('Dependency SHA256 differs')
    if not prefix and (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):
        raise ValueError('Immutable dependency changed during verification')
    return path


def verify_source(runtime_root, source_root, manifest, expected_sha256):
    if (type(manifest) is not dict or not manifest
            or hashlib.sha256(canonical(manifest)).hexdigest()!=expected_sha256):
        raise ValueError('Frozen runtime source manifest SHA differs')
    for name,digest in manifest.items():
        if type(digest) is not str or not SHA.fullmatch(digest):raise ValueError('Malformed runtime source digest')
        for root in (runtime_root,source_root):
            path=safe_path(root,name)
            if not path.is_file() or sha_file(path)!=digest:raise ValueError('Runtime/source archive bytes differ: '+name)
    for package in ('helixwm_final','helixwm_matched','helixwm_v2'):
        expected={x for x in manifest if x.startswith(package+'/') and x.endswith('.py')}
        for root in (Path(runtime_root),Path(source_root)):
            found={p.relative_to(root).as_posix() for p in (root/package).rglob('*.py')}
            if found!=expected:raise ValueError('Frozen Python member set differs: '+package)
    required={'helixwm_final/train.py','helixwm_final/protocol_v3.py','helixwm_final/backbone.py',
              'helixwm_final/full_controller.py','helixwm_final/bank_gate.py','helixwm_matched/backbone.py'}
    if not required.issubset(manifest):raise ValueError('Source manifest omits bank-v3 runtime dependencies')


def verify_dependency_manifest(root,descriptor):
    path=verify_descriptor(root,descriptor)
    rows={}
    with path.open('rb') as stream:
        for line in stream:
            row=json.loads(line)
            if (type(row) is not dict or set(row)!={'kind','path','bytes','sha256'}
                    or row['kind'] not in ('file','stable_byte_prefix')):
                raise ValueError('Formal dependency manifest requires explicit file/prefix rows')
            name=row['path'];key=name.casefold()
            if key in rows:raise ValueError('Duplicate/case-colliding dependency member')
            verify_descriptor(root,{k:v for k,v in row.items() if k!='kind'},prefix=row['kind']=='stable_byte_prefix')
            rows[key]=row
    if not rows:raise ValueError('Empty complete dependency manifest')
    return rows


def _covered(rows,name, *, sha256=None):
    item=rows.get(name.casefold())
    if item is None or sha256 is not None and item['sha256']!=sha256:
        raise ValueError('Required object omitted from pre-pickle dependency manifest: '+name)
    return item


def verify_replay_files(root, records):
    """Decode each immutable NPZ independently, never cache the whole replay."""
    from helixwm_matched.backbone import _decode_array,_array_id
    if type(records) is not list or not records:raise ValueError('Missing replay dependencies')
    seen={}
    fields={'path','value_sha256','codec','dtype','shape','file_sha256','bytes'}
    for record in records:
        if (type(record) is not dict or set(record)!=fields
                or type(record['value_sha256']) is not str or not SHA.fullmatch(record['value_sha256'])
                or record['path']!=record['value_sha256']+'.npz' or record['path'] in seen
                or type(record['shape']) is not list or not record['shape']
                or any(type(x) is not int or x<1 for x in record['shape'])):
            raise ValueError('Invalid/duplicate content-addressed replay object')
        path=verify_descriptor(root,{'path':'REPLAY/'+record['path'],'bytes':record['bytes'],'sha256':record['file_sha256']})
        with np.load(path,allow_pickle=False) as data:
            if data.files!=['value']:raise ValueError('Unexpected replay NPZ members')
            value=_decode_array(record['codec'],data['value'])
        if (value.dtype.hasobject or value.dtype.str!=record['dtype'] or list(value.shape)!=record['shape']
                or not np.isfinite(value).all() or _array_id(value)!=record['value_sha256']):
            raise ValueError('Replay decoded values/schema/content identity differ')
        seen[record['path']]=copy.deepcopy(record)
        del value
    return seen


def validate_replay_state(saved,counts, records):
    """Validate populated rows and selector/chunk graph without constructing JAX."""
    from helixwm_matched.backbone import NativeReplay
    required={'format','upstream_commit','capacity','chunksize','seed','length','online','rows_added','stream_index',
              'chunks','refs','items','fifo','itemid','current','streams','sampler_keys','sampler_indices','sampler_rng','metrics'}
    if (type(saved) is not dict or set(saved)!=required or saved['format']!='helixwm_native_replay_exact_v1'
            or saved['upstream_commit']!=UPSTREAM or saved['capacity']!=1000000 or saved['chunksize']!=1024
            or saved['seed']!=0 or saved['length']!=65 or saved['online'] is not False
            or type(saved['rows_added']) is not int or saved['rows_added']!=counts['agent_steps']+counts['episodes']
            or type(saved['stream_index']) is not int or saved['stream_index'] not in (0,1)):
        raise ValueError('Formal native replay accounting/configuration differs')
    if NativeReplay.referenced_files(saved)!=records:raise ValueError('Checkpoint replay inventory differs from verified sidecar')
    chunks={}
    for chunk in saved['chunks']:
        if (set(chunk)!={'uuid','succ','time','length','size','arrays'} or type(chunk['uuid']) is not str
                or chunk['uuid'] in chunks or type(chunk['length']) is not int
                or not 0<=chunk['length']<=1024 or chunk['size']!=1024):
            raise ValueError('Duplicate or invalid replay chunk')
        if bool(chunk['arrays'])!=bool(chunk['length']):raise ValueError('Replay empty/populated chunk mismatch')
        if any(ref['shape'][0]!=chunk['length'] for ref in chunk['arrays'].values()):
            raise ValueError('Actual decoded replay rows differ from chunk counters')
        chunks[chunk['uuid']]=chunk
    rows=sum(c['length'] for c in chunks.values())
    total=saved['rows_added']
    if not min(total,saved['capacity'])<=rows<=total:raise ValueError('Retained replay rows cannot account for declared history/capacity')
    items={k:(c,i) for k,c,i in saved['items']}
    if len(items)!=len(saved['items']) or len(items)>saved['capacity'] or not items:
        raise ValueError('Invalid replay sequence item count')
    if (sorted(saved['fifo'])!=sorted(items) or len(saved['sampler_keys'])!=len(items)
            or set(saved['sampler_keys'])!=set(items) or dict(saved['sampler_indices'])!={k:i for i,k in enumerate(saved['sampler_keys'])}
            or len(saved['sampler_indices'])!=len(items) or type(saved['itemid']) is not int or saved['itemid']<=max(items)):
        raise ValueError('Replay selector/FIFO order/counter differs')
    current={w:(c,i) for w,c,i in saved['current']};streams=dict(saved['streams'])
    if len(current)!=len(saved['current']) or len(streams)!=len(saved['streams']) or set(current)!=set(streams):
        raise ValueError('Replay worker/pending stream mappings differ')
    refs={c:0 for c in chunks}
    for c,i in current.values():
        if c not in chunks or i!=chunks[c]['length'] or not 0<=i<1024:raise ValueError('Replay current cursor differs')
        refs[c]+=1
    for pending in streams.values():
        if len(pending)>=65:raise ValueError('Replay pending sequence too long')
        for c,i in pending:
            if c not in chunks or type(i) is not int or not 0<=i<chunks[c]['length']:raise ValueError('Broken pending replay row')
            refs[c]+=1
    for c,i in items.values():
        if c not in chunks or type(i) is not int or not 0<=i<chunks[c]['length']:raise ValueError('Broken replay sample row')
        refs[c]+=1;remaining=65;walked=set()
        while remaining:
            if c not in chunks or c in walked:raise ValueError('Replay sequence missing successor/cyclic')
            walked.add(c);take=min(remaining,chunks[c]['length']-i)
            if take<1:raise ValueError('Replay sequence has no available rows')
            remaining-=take;c=chunks[c]['succ'];i=0
    for chunk in chunks.values():
        if chunk['succ'] in refs:refs[chunk['succ']]+=1
    if saved['refs']!=refs:raise ValueError('Replay exact reference counts differ')
    np.random.default_rng().bit_generator.state=copy.deepcopy(saved['sampler_rng'])
    return {'rows_added':total,'retained_rows':rows,'retained_sequences':len(items)}


def validate_payload(payload,record,identity, *, allow_preflight=False, protocol_config=None):
    fields={'format','identity','counts','backbone','auxiliary','pinned_heads','replay','completed_controller_audit',
        'completed_controller_is_previous_episode_pin','bank_gate','bank_reference','bank_gate_reference',
        'completed_previous_gate_reference','trajectory','prefill_rng','numpy_rng','python_rng','next_operation',
        'environment_restore_scope','auto_resume_allowed'}
    preflight=record.get('format')==PREFLIGHT_FORMAT
    if preflight:fields=fields|PREFLIGHT_FIELDS
    if (type(payload) is not dict or set(payload)!=fields or payload['format']!=record['format']
            or record['format'] not in ((FORMAT,PREFLIGHT_FORMAT) if allow_preflight else (FORMAT,))):
        raise ValueError('Only complete new bank-v3 formal payloads are accepted')
    if preflight:
        if (identity['arm']!='F' or not identity['run_id'].startswith('preflight_')
                or payload['formal_training_actions']!=0 or type(payload['formal_training_actions']) is not int
                or payload['fresh_initialization'] is not True or payload['old_checkpoint_loaded'] is not False
                or protocol_config is None or payload['probe_registration']!=protocol_config['full_integration_preflight']
                or any(payload[k]!=record[k] for k in PREFLIGHT_FIELDS)):
            raise ValueError('Fresh integrated preflight lineage/registration differs')
    if payload['identity']!=identity or payload['counts']!=record['counts']:raise ValueError('Payload identity/counters differ from trusted sidecar')
    c=validate_counts(payload['counts'],preflight=preflight);updates=c['update_steps']
    if (payload['completed_controller_is_previous_episode_pin'] is not True or payload['auto_resume_allowed'] is not False
            or payload['next_operation']!='current_pin_bank_gate_or_native_fallback_then_reset_new_episode'
            or payload['environment_restore_scope']!='completed_episode_boundary_new_registered_reset_not_mid_episode'):
        raise ValueError('Checkpoint recovery scope differs; no automatic or mid-episode resume')
    from .backbone import (UPSTREAM_API_SHA256,inference_bundle_hash,host_parameter_schema,AUX_FORMAT)
    from helixwm_matched.backbone import array_tree_hash
    core=payload['backbone'];aux_core=core.get('final_auxiliary_heads',{})
    if core.get('format')!='helixwm_final_backbone_v1' or core.get('upstream_commit')!=UPSTREAM or core.get('final_upstream_api_sha256')!=UPSTREAM_API_SHA256:
        raise ValueError('Formal backbone/upstream identity differs')
    counters=core['training']['counters']
    if (set(counters)!={'updates','batches','actions'} or any(type(v) is not int or v<0 for v in counters.values())
            or counters['updates']!=updates or counters['batches']!=updates
            or type(core['writeback_updates']) is not int or core['writeback_updates']!=updates
            or type(core['inference_version']) is not int or core['inference_version']!=updates):
        raise ValueError('Model/optimizer/writeback/current inference counters differ')
    if (aux_core.get('format')!=AUX_FORMAT or aux_core.get('model_version')!=updates
            or aux_core.get('params_sha256')!=array_tree_hash(aux_core['params'])
            or aux_core.get('parameter_schema')!=host_parameter_schema(aux_core['params'])):
        raise ValueError('Pinned reward/continuation head schema/SHA differs')
    model=inference_bundle_hash(core['inference_params'],aux_core['params'],updates)
    bundle={'model_version':updates,'model_sha256':model,'policy_sha256':array_tree_hash(core['inference_params']),
            'auxiliary_sha256':array_tree_hash(aux_core['params']),'upstream_commit':UPSTREAM}
    if core.get('final_inference_bundle')!=bundle or record['pinned_model_sha256']!=model:
        raise ValueError('Current actual model parameter hash differs from sidecar')
    for values in (core['inference_params'],aux_core['params']):
        if any(k not in core['training']['params'] for k in values) or array_tree_hash(values)!=array_tree_hash({k:core['training']['params'][k] for k in values}):
            raise ValueError('Current episode-boundary pin does not match latest learner parameters')
    tree_hash(core['training']);tree_hash(core['train_carry'])
    from .epistemic import BootstrapDynamics,FrozenBootstrap,tree_digest
    auxiliary=BootstrapDynamics(0,{k:identity[k] for k in ('source_sha256','protocol_sha256')});auxiliary.load(payload['auxiliary'])
    heads=FrozenBootstrap(payload['pinned_heads'])
    if (not heads.ready or heads.sha256!=record['heads_sha256'] or tree_digest(auxiliary.pinned)!=tree_digest(payload['pinned_heads'])
            or auxiliary.last_transition_id!=c['agent_steps'] or auxiliary.accepted_pairs+auxiliary.skipped_pairs!=c['agent_steps']
            or auxiliary.version!=c['episodes']+1):
        raise ValueError('Auxiliary optimizer/pinned heads/true transition counts differ')
    replay_audit=validate_replay_state(payload['replay'],c,record['replay_files'])
    for field in ('trajectory','bank_gate_reference','completed_previous_gate_reference','bank_reference'):
        if payload[field]!=record[field]:raise ValueError('Payload dependency reference differs: '+field)
    completed=payload['completed_controller_audit']
    if identity['arm']=='D':
        if completed.get('controller')!='black_frame_native':raise ValueError('D controller is not author native black-frame')
        state=completed['chain']['state'];rngs=completed['chain']['rngs']
        if (state.get('episode_id')!=c['episodes']-1 or state.get('timestamp')!=500 or state.get('anchor_time')!=500
                or state.get('last_action') is not None or state.get('pending') is not None
                or state.get('missing_length')!=0 or set(rngs)!={'real','actor','dream','repair'}):
            raise ValueError('D completed collector is not a resolved episode boundary')
    else:
        from types import SimpleNamespace
        from .controller import STREAMS,_state,FinalController
        state=completed['state'];rngs=completed['rngs'];integration=completed['integration']
        if (completed.get('format')!='helixwm_final_integrated_bank_controller_v3' or state.get('episode_id')!=c['episodes']-1
                or state.get('timestamp')!=500 or state.get('missing_age')!=0 or state.get('issued_action') is not None
                or state.get('initialized') is not True or state.get('model_version')!=updates-125
                or state.get('evidence_kind') not in ('OBSERVED_POSTERIOR','RECONNECTED_POSTERIOR')
                or completed.get('expected_source_sha256')!=identity['source_sha256'] or set(rngs)!=set(STREAMS)):
            raise ValueError('F previous-pin collector boundary/source/RNG differs')
        _state(state['real'],1)
        if state['bank'] is not None:_state(state['bank'],8)
        weights=np.asarray(state['weights'])
        if weights.shape!=(8,) or not np.isfinite(weights).all() or np.any(weights<0) or not np.isclose(weights.sum(),1.,rtol=0,atol=1e-12):
            raise ValueError('Completed candidate weights are invalid')
        if integration.get('trajectory')!=payload['trajectory']:raise ValueError('Completed collector/trajectory cursor differs')
        # The base validator checks all latent/probability/action/history fields
        # on a disposable CPU shell; it calls no learned function or environment.
        validator=FinalController(SimpleNamespace(inference_version=updates),
            expected_source_sha256=identity['source_sha256'],expected_model_sha256=model)
        validator.FORMAT='helixwm_final_integrated_bank_controller_v3'
        validator.restore_completed_episode(completed)
        required_extra={'context','execution','heads_sha256','auxiliary_state','current_epistemic',
            'current_temporal_error','executed_prefix_sha256','screen_rng','guard','trajectory'}
        if set(integration)!=required_extra or integration['guard'] is not None or integration['execution']!='bank_production':
            raise ValueError('Incomplete or incompatible integrated collector state')
        for key in ('current_epistemic','current_temporal_error'):
            v=integration[key]
            if v is None and key=='current_temporal_error':continue
            if not isinstance(v,(float,np.float64)) or not np.isfinite(v) or v<0:raise ValueError('Nonfinite integrated collector metric')
        aux_state=integration['auxiliary_state']
        if (not isinstance(aux_state,np.ndarray) or aux_state.shape!=(3,1,16) or aux_state.dtype!=np.float64
                or not np.isfinite(aux_state).all() or not SHA.fullmatch(integration['executed_prefix_sha256'])):
            raise ValueError('Integrated auxiliary collector or action-prefix identity invalid')
        np.random.default_rng().bit_generator.state=copy.deepcopy(integration['screen_rng'])
    for value in rngs.values():np.random.default_rng().bit_generator.state=copy.deepcopy(value)
    np.random.default_rng().bit_generator.state=copy.deepcopy(payload['prefill_rng'])
    np.random.RandomState().set_state(payload['numpy_rng']);random.Random().setstate(payload['python_rng'])
    return {'counts':copy.deepcopy(c),'replay':replay_audit,'model_sha256':model,'heads_sha256':heads.sha256}


def load_checkpoint(path, *, expected_sha256, expected_record_sha256, expected_identity,
                    protocol_config, require_final=False, dependency_root=None, runtime_source_root=None,
                    allow_preflight=False):
    """Return (payload, trusted sidecar, current VerifiedBankGate); never write input."""
    validate_identity(expected_identity)
    path=Path(path).absolute();root=Path(dependency_root).absolute() if dependency_root else path.parent.parent
    relative=path.relative_to(root).as_posix();safe_path(root,relative)
    if not re.fullmatch(r'CHECKPOINTS/checkpoint_[0-9]{9}\.pkl',relative):raise ValueError('Only new formal CP paths accepted')
    if not SHA.fullmatch(expected_sha256) or not SHA.fullmatch(expected_record_sha256):raise ValueError('External checkpoint and sidecar SHA256 are required')
    record_path=safe_path(root,str(PurePosixPath(relative).with_suffix('.json')))
    if sha_file(record_path)!=expected_record_sha256:raise ValueError('Sidecar differs from external trusted SHA256')
    record=json.loads(record_path.read_bytes())
    required={'format','identity','counts','checkpoint','replay_files','pinned_model_sha256','heads_sha256',
              'bank_gate_sha256','trajectory','dependencies_manifest','bank_reference','bank_gate_reference','completed_previous_gate_reference'}
    preflight=record.get('format')==PREFLIGHT_FORMAT
    if preflight:required=required|PREFLIGHT_FIELDS
    if (type(record) is not dict or set(record)!=required
            or record['format'] not in ((FORMAT,PREFLIGHT_FORMAT) if allow_preflight else (FORMAT,))
            or record['identity']!=expected_identity):
        raise ValueError('Not the complete trusted bank-v3 sidecar identity')
    validate_counts(record['counts'],require_final=require_final,preflight=preflight)
    if relative!=f"CHECKPOINTS/checkpoint_{record['counts']['agent_steps']:09d}.pkl":
        raise ValueError('Checkpoint filename and actual action boundary differ')
    if preflight:
        if (expected_identity['arm']!='F' or not expected_identity['run_id'].startswith('preflight_')
                or record['formal_training_actions']!=0 or type(record['formal_training_actions']) is not int
                or record['fresh_initialization'] is not True or record['old_checkpoint_loaded'] is not False
                or record['probe_registration']!=protocol_config['full_integration_preflight']):
            raise ValueError('Unregistered/reused preflight lineage before pickle')
    if record['checkpoint']['path']!=relative or record['checkpoint']['sha256']!=expected_sha256:
        raise ValueError('Checkpoint path/digest differs from trusted sidecar')
    verify_descriptor(root,record['checkpoint'])
    from . import protocol_v3
    protocol_v3.validate_protocol(protocol_config)
    if protocol_v3.protocol_sha256(protocol_config)!=expected_identity['protocol_sha256']:
        raise ValueError('Registered v3 protocol differs')
    environment=safe_path(root,'environment_lock.json')
    if sha_file(environment)!=expected_identity['environment_lock_sha256']:raise ValueError('Environment lock SHA differs')
    source=json.loads(safe_path(root,'source_manifest.json').read_bytes())
    verify_source(runtime_source_root or Path(__file__).resolve().parents[1],root/'SOURCE',source,expected_identity['source_sha256'])
    rows=verify_dependency_manifest(root,record['dependencies_manifest'])
    for name,digest in source.items():_covered(rows,'SOURCE/'+name,sha256=digest)
    bank_reference=record['bank_reference']
    if (bank_reference!={'path':'BANK_SHARED','manifest_sha256':expected_identity['bank_manifest_sha256']}):
        raise ValueError('Shared original bank reference differs from training identity')
    bank_manifest=safe_path(root,'BANK_SHARED/manifest.json')
    if sha_file(bank_manifest)!=bank_reference['manifest_sha256']:raise ValueError('Shared bank external manifest SHA differs')
    bank_metadata=json.loads(bank_manifest.read_bytes())
    from .bank_replay import load_original_bank
    bank,origin=load_original_bank(root/'BANK_SHARED',bank_reference['manifest_sha256'],
        environment_lock_sha256=expected_identity['environment_lock_sha256'],
        origin_source_sha256=bank_metadata['source_sha256'],origin_protocol_sha256=bank_metadata['protocol_sha256'])
    del bank
    for child in (root/'BANK_SHARED').rglob('*'):
        if child.is_file():_covered(rows,child.relative_to(root).as_posix(),sha256=sha_file(child))
    verify_replay_files(root,record['replay_files'])
    for replay in record['replay_files']:_covered(rows,'REPLAY/'+replay['path'],sha256=replay['file_sha256'])
    if record['trajectory'] is not None:
        from .trajectory import TrajectoryArchive
        archive=TrajectoryArchive.restore(root/'TRAJECTORIES',record['trajectory'],
            expected_source_sha256=expected_identity['source_sha256'],expected_protocol_sha256=expected_identity['protocol_sha256'],writable=False)
        archive.close()
        _covered(rows,'TRAJECTORIES/header.json',sha256=record['trajectory']['header_sha256'])
        _covered(rows,'TRAJECTORIES/manifest.jsonl',sha256=record['trajectory']['manifest_prefix_sha256'])
        with (root/'TRAJECTORIES/manifest.jsonl').open('rb') as stream:
            left=record['trajectory']['manifest_prefix_bytes']
            while left:
                line=stream.readline(left)
                if not line or not line.endswith(b'\n'):raise ValueError('Trajectory prefix changed while checking dependency coverage')
                left-=len(line);entry=json.loads(line)
                for field in ('metadata','data'):_covered(rows,'TRAJECTORIES/'+entry[field+'_path'],sha256=entry[field+'_sha256'])
    elif expected_identity['arm']=='F':raise ValueError('F formal checkpoints must preserve complete trajectory archive')
    gates={}
    from .bank_gate import load_bank_gate
    for name in ('bank_gate_reference','completed_previous_gate_reference'):
        reference=record[name]
        if reference is None:
            gates[name]=None;continue
        if set(reference)!={'path','manifest_sha256','artifact_sha256','context'}:raise ValueError('Incomplete bound bank gate reference')
        folder=safe_path(root,reference['path'])
        if not reference['path'].startswith('BANK_GATES/'):raise ValueError('Bank gate outside immutable gate root')
        for child in folder.rglob('*'):
            if child.is_file():_covered(rows,child.relative_to(root).as_posix(),sha256=sha_file(child))
        gate=load_bank_gate(folder,reference['manifest_sha256'],reference['context'],require_active=False,
                           bank_directory=root/'BANK_SHARED')
        if gate['artifact_sha256']!=reference['artifact_sha256']:raise ValueError('Gate artifact identity differs')
        gates[name]=gate
    gate=gates['bank_gate_reference']
    if gate is None and record['bank_gate_sha256'] is not None:raise ValueError('Sidecar claims a missing gate artifact')
    if (expected_identity['arm']=='F' or require_final) and gate is None:raise ValueError('Current model-bound bank gate is required')
    if gate is not None:
        context=record['bank_gate_reference']['context']
        bound={'source_sha256':expected_identity['source_sha256'],'protocol_sha256':expected_identity['protocol_sha256'],
            'bank_manifest_sha256':expected_identity['bank_manifest_sha256'],'model_sha256':record['pinned_model_sha256'],
            'model_version':record['counts']['update_steps'],'epistemic_heads_sha256':record['heads_sha256']}
        if any(context.get(k)!=v for k,v in bound.items()) or record['bank_gate_sha256']!=gate['artifact_sha256']:
            raise ValueError('Current bank gate does not bind this formal pin/heads')
    # Read exact verified bytes once. Changed data cannot be executed via pickle.
    raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=expected_sha256:raise ValueError('Checkpoint changed after dependency verification')
    payload=pickle.loads(raw);del raw
    validate_payload(payload,record,expected_identity,allow_preflight=allow_preflight,protocol_config=protocol_config)
    if payload['bank_gate'] is None:
        if gate is not None:raise ValueError('Payload omits verified bank gate')
    elif gate is None or dict(payload['bank_gate'])!=dict(gate):raise ValueError('Payload bank gate differs from independently verified artifact')
    if expected_identity['arm']=='F':
        previous=gates['completed_previous_gate_reference'];completed=payload['completed_controller_audit']
        if previous is None or completed['calibration']!=dict(previous):raise ValueError('Completed F collector gate is not the preserved prior-pin evidence')
        context=record['completed_previous_gate_reference']['context']
        if (completed['integration']['context']!=context or completed['integration']['execution']!='bank_production'
                or completed['integration']['heads_sha256']!=context['epistemic_heads_sha256']
                or completed['expected_model_sha256']!=context['model_sha256']
                or context['model_version']!=record['counts']['update_steps']-125
                or any(context[k]!=expected_identity[k] for k in ('source_sha256','protocol_sha256','bank_manifest_sha256'))):
            raise ValueError('Completed collector prior-pin context differs from dependency evidence')
    if sha_file(path)!=expected_sha256 or sha_file(record_path)!=expected_record_sha256:raise ValueError('Input mutated while loading')
    return payload,record,gate
