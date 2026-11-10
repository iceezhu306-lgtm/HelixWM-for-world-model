"""Guarded fresh independent-bank Final training; no implicit resume or approval.

The CLI refuses absent/unbound preflight evidence. It uses the unchanged
author learner and shared native bootstrap. Per-pin calibration only replays
one shared 2480-action bank; no new environment interaction is permitted.
Inactive empirical bank gates disable candidate fusion/planning. They do not
claim independent closed-loop certification. Archive receipts must bind the
new source/run/arm/checkpoint and all referenced bank/trajectory dependencies.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path, PureWindowsPath
import pickle
import random
import shutil
import time
import traceback

import numpy as np

from . import protocol_v3 as protocol
from helixwm_matched.protocol import canonical_json
from helixwm_matched.recovery_probe import tree_hash

IDENTITY_FIELDS = {'run_id', 'arm', 'source_sha256', 'protocol_sha256',
                   'environment_lock_sha256', 'upstream_commit', 'bank_manifest_sha256'}
GATE_SCHEMA = 'helixwm_final_bank_formal_launch_gate_v3'
REQUEST_SCHEMA = 'helixwm_final_bank_archive_request_v3'
RECEIPT_SCHEMA = 'helixwm_final_bank_archive_receipt_v3'
CHECKPOINT_FORMAT = 'helixwm_final_bank_episode_boundary_v3'
RESERVE_BYTES = 2 * 1024**3


def _sha_value(value):
    return type(value) is str and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(1024*1024), b''): h.update(data)
    return h.hexdigest()


def json_sha(value):
    return hashlib.sha256(canonical_json(value)).hexdigest()


def now():
    return __import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat()


def write_json_exclusive(path, value):
    # Reject nonfinite values before creating even an empty evidence file.
    raw = canonical_json(value) + b'\n'
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    return descriptor(path, path.parent)


def _status(path, value):
    from helixwm_matched.protocol import atomic_json
    atomic_json(path, value)


def _append(path, value):
    raw = canonical_json(value) + b'\n'
    with Path(path).open('ab') as stream:
        stream.write(raw); stream.flush()


def descriptor(path, root):
    path, root = Path(path).resolve(), Path(root).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError('Evidence must be a regular file within its declared root')
    return {'path': path.relative_to(root).as_posix(), 'bytes': path.stat().st_size,
            'sha256': sha_file(path)}


def _inside(root, relative):
    if type(relative) is not str or not relative or '\\' in relative or Path(relative).is_absolute():
        raise ValueError('Evidence path must be relative POSIX text')
    path = Path(root)/relative
    if path.is_symlink() or not path.resolve().is_relative_to(Path(root).resolve()):
        raise ValueError('Evidence dependency escapes its root')
    return path


def verified_descriptor(root, item):
    if (type(item) is not dict or set(item) != {'path', 'bytes', 'sha256'}
            or type(item['bytes']) is not int or item['bytes'] < 0 or not _sha_value(item['sha256'])):
        raise ValueError('Invalid evidence file descriptor')
    path = _inside(root, item['path'])
    if not path.is_file() or path.stat().st_size != item['bytes'] or sha_file(path) != item['sha256']:
        raise ValueError('Evidence file SHA/length mismatch: '+item['path'])
    return path


def validate_identity(identity):
    if type(identity) is not dict or set(identity) != IDENTITY_FIELDS:
        raise ValueError('Exact new run/arm/source/protocol/environment identity is required')
    if identity['arm'] not in ('D', 'F') or type(identity['run_id']) is not str:
        raise ValueError('Only the new D/F arms are registered')
    if not identity['run_id'] or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in identity['run_id']):
        raise ValueError('Run ID must be an explicit path-free identifier')
    for key in ('source_sha256', 'protocol_sha256', 'environment_lock_sha256', 'bank_manifest_sha256'):
        if not _sha_value(identity[key]): raise ValueError('Missing runtime identity: '+key)
    if identity['upstream_commit'] != protocol.registered_protocol()['official_core']['commit']:
        raise ValueError('Author commit differs from the frozen baseline')
    return copy.deepcopy(identity)


def validate_counts(counts, *, boundary=False):
    if (type(counts) is not dict or set(counts) != {'agent_steps', 'env_steps', 'episodes', 'update_steps'}
            or any(type(v) is not int or v < 0 for v in counts.values())):
        raise ValueError('Counters must be complete nonnegative strict integers')
    actions = counts['agent_steps']
    if actions > 1000000 or counts['env_steps'] != 2*actions:
        raise ValueError('Actual action/DMC budget differs')
    if counts['update_steps'] != max(0, actions-5000)//4:
        raise ValueError('Author learner update schedule differs')
    if boundary and (actions % 500 or counts['episodes'] != actions//500):
        raise ValueError('An exact complete episode boundary is required')
    return True


def validate_launch_gate(path, expected_sha256, identity, config, run_dir):
    """Verify the externally pinned launch bundle; no self-issued PASS shortcut.

    This is a transport/identity validator of a trusted audited gate. It cannot
    establish real GPU execution from an arbitrary maliciously authored JSON;
    the external gate digest must come from the actual completed preflight job.
    """
    validate_identity(identity); protocol.validate_protocol(config)
    if identity['protocol_sha256']!=protocol.protocol_sha256(config):
        raise ValueError('Runtime identity does not bind the full registered protocol')
    path = Path(path)
    if not _sha_value(expected_sha256) or sha_file(path) != expected_sha256:
        raise ValueError('Launch gate differs from externally supplied audit SHA')
    gate = json.loads(path.read_bytes())
    fields = {'schema', 'identity', 'authorized_runs', 'evidence', 'capacity',
              'archive_service_source_sha256', 'archive_bridge_probe', 'measurement_kind'}
    if type(gate) is not dict or set(gate) != fields or gate['schema'] != GATE_SCHEMA:
        raise ValueError('Not a complete Full Final launch gate')
    common = {k:v for k,v in identity.items() if k not in ('run_id','arm')}
    if gate['identity'] != common or gate['measurement_kind'] != 'actual_gpu_preflight':
        raise ValueError('Launch gate source or actual execution provenance differs')
    runs = gate['authorized_runs']
    if type(runs) is not dict or set(runs) != {'D','F'}:
        raise ValueError('Both explicit new run roots must be authorized')
    if any(type(value) is not str for value in runs.values()) or Path(runs['D']).resolve()==Path(runs['F']).resolve():
        raise ValueError('The independent arms require two distinct explicit run roots')
    if Path(runs[identity['arm']]).resolve() != Path(run_dir).resolve() or Path(run_dir).name != identity['run_id']:
        raise ValueError('Launch gate is for a different new run directory')
    expected = set(config['launch_gate_requirements'])
    if type(gate['evidence']) is not dict or set(gate['evidence']) != expected:
        raise ValueError('Incomplete registered launch evidence')
    for name, item in gate['evidence'].items():
        report = json.loads(verified_descriptor(path.parent, item).read_bytes())
        keys = {'schema','gate_name','identity','measurement_kind','result','raw_evidence'}
        if (type(report) is not dict or set(report) != keys
                or report['schema'] != 'helixwm_final_audited_gate_evidence_v1'
                or report['gate_name'] != name or report['identity'] != common
                or report['measurement_kind'] != 'actual_gpu_preflight' or report['result'] != 'PASS'
                or type(report['raw_evidence']) is not list or not report['raw_evidence']):
            raise ValueError('Missing measured audited gate: '+name)
        for raw in report['raw_evidence']: verified_descriptor(path.parent, raw)
    cap = gate['capacity']
    if (type(cap) is not dict or set(cap) != {'remote_required_bytes_before_start','remote_checkpoint_peak_factor',
            'remote_episode_write_peak_bytes',
            'remote_model_anchor_reserve_bytes','E_required_full_retention_and_ZIP_bytes','E_verified_free_bytes',
            'includes_all_calibration_versions_and_full_trajectory_arrays','prediction_is_not_guarantee'}
            or any(type(cap[k]) is not int or cap[k] <= 0 for k in ('remote_required_bytes_before_start',
                'remote_model_anchor_reserve_bytes','remote_episode_write_peak_bytes',
                'E_required_full_retention_and_ZIP_bytes','E_verified_free_bytes'))
            or type(cap['remote_checkpoint_peak_factor']) is not float or cap['remote_checkpoint_peak_factor'] < 1.2
            or cap['E_verified_free_bytes'] < cap['E_required_full_retention_and_ZIP_bytes']
            or cap['includes_all_calibration_versions_and_full_trajectory_arrays'] is not True
            or cap['prediction_is_not_guarantee'] is not True):
        raise ValueError('Complete measured archive/ZIP/trajectory/calibration capacity is required')
    if not _sha_value(gate['archive_service_source_sha256']):
        raise ValueError('Archive implementation must have a pinned source digest')
    bridge = json.loads(verified_descriptor(path.parent, gate['archive_bridge_probe']).read_bytes())
    if (bridge.get('schema') != 'helixwm_final_archive_bridge_probe_v1'
            or bridge.get('source_sha256') != gate['archive_service_source_sha256']
            or bridge.get('identity') != common or bridge.get('result') != 'PASS'
            or bridge.get('measured_roundtrip') is not True or bridge.get('old_ACK_rejected') is not True):
        raise ValueError('New identity-bound archive bridge has not passed its real probe')
    return gate


def validate_archive_receipt(receipt, request, *, archive_source_sha256):
    """Old checkpoint-only ACKs cannot release a new run or another arm."""
    fields = {'schema','identity','request_sha256','object_manifest_sha256','checkpoint_sha256',
              'archive_service_source_sha256','local_root','remote_before_manifest_sha256',
              'remote_after_manifest_sha256','local_verified_manifest_sha256','transaction_complete',
              'latest_remote_dependencies_retained','local_unique_objects_deleted'}
    if type(receipt) is not dict or set(receipt) != fields or receipt['schema'] != RECEIPT_SCHEMA:
        raise ValueError('Incomplete new archive receipt; legacy ACK is forbidden')
    required_request={'schema','identity','counts','checkpoint_sha256','object_manifest',
        'archive_service_source_sha256','must_expand_recursive_dependency_kinds',
        'delete_local_unique_data_allowed','latest_remote_checkpoint_and_all_dependencies_must_remain'}
    if type(request) is not dict or set(request)!=required_request:
        raise ValueError('Incomplete archive request identity/dependency contract')
    validate_identity(request['identity']);validate_counts(request['counts'],boundary=True)
    manifest=request['object_manifest']
    if (type(manifest) is not dict or set(manifest)!={'path','bytes','sha256'}
            or type(manifest['bytes']) is not int or manifest['bytes']<0
            or not _sha_value(manifest['sha256']) or type(manifest['path']) is not str):
        raise ValueError('Archive request is missing the exact dependency manifest')
    if (request['must_expand_recursive_dependency_kinds'] is not True
            or request['delete_local_unique_data_allowed'] is not False
            or request['latest_remote_checkpoint_and_all_dependencies_must_remain'] is not True
            or request['archive_service_source_sha256']!=archive_source_sha256
            or not _sha_value(request['checkpoint_sha256'])):
        raise ValueError('Archive request would relax verified dependency preservation')
    if (request.get('schema') != REQUEST_SCHEMA
            or receipt['identity'] != request['identity']
            or receipt['request_sha256'] != json_sha(request)
            or receipt['checkpoint_sha256'] != request['checkpoint_sha256']
            or receipt['object_manifest_sha256'] != request['object_manifest']['sha256']
            or receipt['archive_service_source_sha256'] != archive_source_sha256):
        raise ValueError('Archive receipt does not bind this run/arm/source/checkpoint/request')
    for key in ('remote_before_manifest_sha256','remote_after_manifest_sha256','local_verified_manifest_sha256'):
        if receipt[key] != request['object_manifest']['sha256']:
            raise ValueError('Archive before/after/local SHA manifests differ')
    local=PureWindowsPath(receipt['local_root']) if type(receipt['local_root']) is str else None
    if (local is None or not local.is_absolute() or local.drive.upper()!='E:' or '..' in local.parts
            or receipt['transaction_complete'] is not True
            or receipt['latest_remote_dependencies_retained'] is not True
            or receipt['local_unique_objects_deleted'] is not False):
        raise ValueError('Archive transaction/E-drive/dependency preservation incomplete')
    return True


def wait_for_json(path, *, status_path, stage, identity, counts, timeout_seconds=0.,
                  poll_seconds=15., clock=time.monotonic, sleep=time.sleep):
    """Only wait for a pending file. Malformed or failed evidence is not retried."""
    if (not math.isfinite(timeout_seconds) or timeout_seconds < 0
            or not math.isfinite(poll_seconds) or not 0 < poll_seconds <= 60):
        raise ValueError('Invalid bounded poll/deadline settings')
    started = clock()
    while True:
        if Path(path).exists():
            value = json.loads(Path(path).read_bytes())
            _status(status_path, {'waiting':False,'stage':stage,'identity':identity,'counts':dict(counts),'utc':now()})
            return value
        elapsed = clock()-started
        _status(status_path, {'waiting':True,'stage':stage,'identity':identity,'counts':dict(counts),
            'awaited_path':str(path),'elapsed_s':elapsed,'timeout_seconds':timeout_seconds,'utc':now(),
            'training_interactions_advance_while_waiting':False})
        if timeout_seconds and elapsed >= timeout_seconds:
            raise TimeoutError('Required '+stage+' evidence was not supplied before its deadline')
        remaining = timeout_seconds-elapsed if timeout_seconds else poll_seconds
        sleep(min(poll_seconds, max(0., remaining)))


def _finite_evidence(value):
    """Detach diagnostic values without sampling, inference or state mutation."""
    if value is None or type(value) in (str, bool, int): return value
    if isinstance(value, np.bool_): return bool(value)
    if isinstance(value, np.integer): return int(value)
    if isinstance(value, (float, np.floating)):
        if not math.isfinite(float(value)): raise FloatingPointError('Nonfinite module evidence')
        return float(value)
    if isinstance(value, np.ndarray): return _finite_evidence(value.tolist())
    if isinstance(value, (list, tuple)): return [_finite_evidence(x) for x in value]
    if isinstance(value, dict) and all(type(k) is str for k in value):
        return {k:_finite_evidence(v) for k,v in value.items()}
    raise ValueError('Module evidence must contain JSON-compatible detached values')


def _module_gate_status(controller):
    # FullController.calibration_status and the bound-handle verifier only read
    # immutable identities/readiness/version; neither calls a model nor an RNG.
    active, reason = controller.calibration_status()
    if type(active) is not bool or type(reason) is not str:
        raise ValueError('Invalid module gate diagnostic')
    return active, reason


def _module_selection(controller, *, timestamp, available, external):
    """Capture a_t after act, using only state/observation o_t (not o_{t+1})."""
    from .full_controller import FullController
    full = isinstance(controller, FullController)
    native_name = getattr(controller, 'name', None)
    evidence = {'phase':'after_act_before_environment_step',
        'conditioning_observation_timestamp':timestamp-1,
        'conditioning_observation_available':available,
        'full_controller':full, 'execution':controller.execution if full else None,
        'controller_name':'helixwm_final' if full else native_name,
        'action_origin':'external_prefill' if external else ('final_controller' if full else 'native_or_test_controller'),
        'final_action_modules_applicable':full and not external,
        'model_version':controller.model_version if full else None,
        'model_sha256':controller.context['model_sha256'] if full else None,
        'bank_gate_artifact_sha256':(controller._calibration.get('artifact_sha256')
            if full and controller._calibration is not None else None),
        'gate_active':None, 'gate_status':None, 'missing_age':None,
        'decision':{'kind':'external_prefill' if external else 'native_policy', 'planning':None},
        'planning_attempted':None, 'planning_accepted':None,
        'prior_only_fallback':None, 'fallback_reason':None}
    if full:
        if controller.timestamp != timestamp-1: raise ValueError('Selection diagnostic timestamp differs')
        decision = _finite_evidence(controller.last_action_decision)
        if type(decision) is not dict or decision.get('kind') not in (
                'external_prefill','candidate_control','prior_only_fallback','observed_actor'):
            raise ValueError('Missing exact issued-action decision')
        if (decision['kind']=='external_prefill') != external:
            raise ValueError('Issued-action prefill provenance differs')
        planning = decision.get('planning')
        if planning is False: attempted = accepted = False
        elif (type(planning) is dict and type(planning.get('used')) is bool
                and type(planning.get('reason')) is str):
            attempted = planning['used']; accepted = planning['reason']=='accepted_plan'
            if accepted and not attempted: raise ValueError('Accepted plan without a planning attempt')
        else: raise ValueError('Missing issued-action planning diagnostic')
        active, reason = _module_gate_status(controller)
        fallback = decision['kind']=='prior_only_fallback'
        evidence.update(gate_active=active, gate_status=reason, missing_age=controller.missing_age,
            decision=decision, planning_attempted=attempted, planning_accepted=accepted,
            prior_only_fallback=fallback, fallback_reason=(
                planning['reason'] if isinstance(planning,dict) and planning['reason']=='all_plans_rejected'
                else decision.get('calibration_status')) if fallback else None)
    return _finite_evidence(evidence)


def _module_observation(controller, *, timestamp, available):
    """Capture assimilation of o_{t+1}; fusion here did not choose a_t."""
    from .full_controller import FullController
    evidence = {'phase':'after_controller_step_before_learner_updates',
        'observation_timestamp':timestamp, 'observation_available':available,
        'full_controller':isinstance(controller,FullController),
        'gate_active':None, 'gate_status':None, 'fusion_lambda':None,
        'fusion_accepted':None, 'fusion_kl':None, 'mode':None,
        'evidence_kind':None, 'missing_age':None, 'observation_restored':None,
        'candidate_reconnection_accepted':None}
    if evidence['full_controller']:
        if controller.timestamp != timestamp: raise ValueError('Observation diagnostic timestamp differs')
        metrics = controller.last_metrics
        # Do not interpret previous-episode history or a gate's existence as use.
        fusion = metrics['fusion_lambda']; kl = metrics['fusion_kl']
        if (isinstance(fusion,(bool,np.bool_)) or not math.isfinite(float(fusion))
                or not math.isfinite(float(kl))):
            raise FloatingPointError('Nonfinite/invalid fusion evidence')
        if not 0 <= float(fusion) <= 1 or float(kl) < 0: raise ValueError('Invalid fusion evidence range')
        active, reason = _module_gate_status(controller)
        if 'bank_gate_active' in metrics and metrics['bank_gate_active'] is not active:
            raise ValueError('Post-observation bank diagnostic differs')
        evidence.update(gate_active=active, gate_status=reason,
            fusion_lambda=float(fusion), fusion_accepted=bool(fusion>0), fusion_kl=float(kl),
            mode=controller.mode, evidence_kind=metrics['evidence_kind'],
            missing_age=metrics['missing_age'], observation_restored=metrics['observation_restored'],
            candidate_reconnection_accepted=metrics['candidate_reconnection_accepted'])
    return _finite_evidence(evidence)


def run_episode(*, env, controller, replay, auxiliary, prefill_rng, counts, episode,
                seed, mask, on_reset=None, on_interaction=None, on_metric=None, arm=None):
    """One exact author-learning episode; auxiliary output never chooses action.

    Dependencies are explicit for CPU orchestration tests. Only the guarded CLI
    constructs the actual GPU author learner. Actual acknowledged simulator
    advances are counted even if a later rendering/inference/update fails.
    """
    validate_counts(counts, boundary=True)
    if arm not in (None,'D','F'): raise ValueError('Interaction arm must be explicitly D/F or unspecified engineering scope')
    if type(episode) is not int or episode != counts['episodes'] or counts['agent_steps']+500 > 1000000:
        raise ValueError('Wrong episode or action budget would be exceeded')
    if not isinstance(mask,np.ndarray) or mask.dtype != np.bool_ or mask.shape != (501,) or not mask[0] or not mask[-1]:
        raise ValueError('Exact boundary-valid paired mask required')
    image = env.reset(seed=seed,mask=mask)
    if on_reset is not None: on_reset()
    controller.reset(image,episode_id=episode)
    observation = {'image':image.copy(),'reward':np.float32(0.),'is_first':np.bool_(True),
                   'is_last':np.bool_(False),'is_terminal':np.bool_(False)}
    available = True; rewards=[]
    for timestamp in range(1,501):
        external = prefill_rng.uniform(-1,1,6).astype(np.float32) if counts['agent_steps'] < 5000 else None
        action = np.asarray(controller.act(external),np.float32).reshape(6)
        if not np.isfinite(action).all() or np.any(np.abs(action)>1): raise ValueError('Controller issued invalid action')
        selection = (_module_selection(controller,timestamp=timestamp,available=available,external=external is not None)
                     if on_interaction is not None else None)
        current_state = controller.state
        a0,c0 = env.action_decisions,env.episode_env_steps
        stage='environment_step'; post=None; validated_reward=None; validated_available=None
        try:
            try: next_image,reward,done,info = env.step(action)
            finally:
                counts['agent_steps'] += env.action_decisions-a0
                counts['env_steps'] += env.episode_env_steps-c0
            stage='environment_result_validation'
            if (done != (timestamp==500) or info.get('observation_timestamp') != timestamp
                    or type(info.get('availability')) is not bool or info['availability'] != bool(mask[timestamp])
                    or info.get('env_steps_this_step') != 2
                    or env.action_decisions-a0 != 1 or env.episode_env_steps-c0 != 2):
                raise RuntimeError('Actual environment horizon/mask/call accounting differs')
            if not math.isfinite(float(reward)) or not -1e-12 <= float(reward) <= 2+1e-12:
                raise FloatingPointError('Invalid returned DMC reward')
            if (not isinstance(next_image,np.ndarray) or next_image.dtype!=np.uint8 or next_image.shape!=(64,64,3)
                    or (not info['availability'] and np.any(next_image))):
                raise ValueError('Learner must receive exactly the visible/corrupted RGB frame')
            validated_reward=float(reward); validated_available=info['availability']
            stage='replay_and_auxiliary_update'
            replay.add(observation,action,current_state)
            auxiliary.update(image,action,next_image,float(reward),float(not info['terminated']),
                transition_id=counts['agent_steps'],previous_available=available,next_available=info['availability'])
            stage='controller_observation_update'
            controller.step(next_image,action,info,episode_id=episode)
            stage='observation_evidence_validation'
            if on_interaction is not None:post=_module_observation(controller,timestamp=timestamp,available=info['availability'])
            image=next_image; available=info['availability']; rewards.append(float(reward))
            observation={'image':image.copy(),'reward':np.float32(reward),'is_first':np.bool_(False),
                         'is_last':np.bool_(done),'is_terminal':np.bool_(info['terminated'])}
            target=max(0,counts['agent_steps']-5000)//4
            stage='learner_update'
            while counts['update_steps']<target:
                metrics=replay.train_once()
                if any(not np.isfinite(v) for v in metrics.values()): raise FloatingPointError('Nonfinite author loss/gradient')
                counts['update_steps']+=1
                if on_metric is not None:on_metric(dict(counts),metrics)
            stage='interaction_log'
            if on_interaction is not None:
                on_interaction({'episode':episode,'timestamp':timestamp,**counts,'reward':float(reward),
                    'availability':available,'action':action.tolist(),'DMC_calls':2,
                    'module_evidence':{'schema':'helixwm_final_action_module_evidence_v1','arm':arm,
                        'action_selection':selection,'observation_update':post,'interaction_complete':True}})
        except BaseException as exc:
            # Preserve acknowledged actions even if inference or an author
            # update fails. Missing outcomes stay null, never invented. A log
            # failure must not replace the original execution exception.
            if on_interaction is not None and env.action_decisions>a0 and stage!='interaction_log':
                try:
                    on_interaction({'episode':episode,'timestamp':timestamp,**counts,'reward':validated_reward,
                        'availability':validated_available,'action':action.tolist(),'DMC_calls':env.episode_env_steps-c0,
                        'module_evidence':{'schema':'helixwm_final_action_module_evidence_v1','arm':arm,
                            'action_selection':selection,'observation_update':post,'interaction_complete':False,
                            'failure_stage':stage,'error':repr(exc)}})
                except BaseException as secondary:
                    exc.add_note('Secondary interaction evidence write failed: '+repr(secondary))
            raise
    replay.add(observation,np.zeros(6,np.float32),controller.state)
    counts['episodes']+=1
    validate_counts(counts,boundary=True)
    return {'episode':episode,'environment_seed':seed,'return':math.fsum(rewards),
            'mask_sha256':protocol.mask_sha256(mask),**counts}


def _wait_space(root, required_bytes, counts, identity, *, timeout_seconds=0):
    start=time.monotonic()
    while True:
        free=shutil.disk_usage(root).free
        _status(root/'capacity_wait.json',{'waiting':free<required_bytes,'identity':identity,'counts':dict(counts),
            'free_bytes':free,'required_bytes':required_bytes,'utc':now(),'prediction_is_not_guarantee':True})
        if free>=required_bytes:return
        if timeout_seconds and time.monotonic()-start>=timeout_seconds:
            raise TimeoutError('Measured write peak lacks capacity; no interaction or deletion performed')
        time.sleep(15)


def _checkpoint_dependencies(root, path, replay_state, trajectory_cursor,
                             bank_reference, gate_reference, previous_gate_reference):
    """Stream an immutable inventory; never retain a million-record index in RAM."""
    from helixwm_matched.backbone import NativeReplay
    with path.open('xb') as stream:
        def emit(item):stream.write(canonical_json(item)+b'\n')
        def file_entry(file,expected=None):
            row=descriptor(file,root)
            if expected is not None and row['sha256']!=expected:raise ValueError('Dependency changed: '+row['path'])
            emit({'kind':'file',**row})
        for name in ('identity.json','protocol.json','source_manifest.json','environment_lock.json'):
            file_entry(root/name)
        for item in NativeReplay.referenced_files(replay_state):
            file_entry(_inside(root,'REPLAY/'+item['path']),item['file_sha256'])
        trees=['SOURCE',bank_reference['path']]
        for ref in (gate_reference,previous_gate_reference):
            if ref is not None and ref['path'] not in trees:trees.append(ref['path'])
        for relative in trees:
            directory=_inside(root,relative)
            if not directory.is_dir():raise ValueError('Missing checkpoint dependency tree: '+relative)
            for file in sorted(directory.rglob('*')):
                if file.is_symlink():raise ValueError('Symlink checkpoint dependency is forbidden')
                if file.is_file():file_entry(file)
        if trajectory_cursor is not None:
            folder=root/'TRAJECTORIES';file_entry(folder/'header.json',trajectory_cursor['header_sha256'])
            manifest=folder/'manifest.jsonl';remaining=trajectory_cursor['manifest_prefix_bytes'];digest=hashlib.sha256();records=0
            with manifest.open('rb') as source:
                while remaining:
                    line=source.readline()
                    if not line.endswith(b'\n') or len(line)>remaining:raise ValueError('Incomplete trajectory dependency prefix')
                    remaining-=len(line);digest.update(line);entry=json.loads(line);records+=1
                    file_entry(_inside(folder,entry['metadata_path']),entry['metadata_sha256'])
                    file_entry(_inside(folder,entry['data_path']),entry['data_sha256'])
            if records!=trajectory_cursor['record_count'] or digest.hexdigest()!=trajectory_cursor['manifest_prefix_sha256']:
                raise ValueError('Trajectory dependency cursor changed')
            emit({'kind':'stable_byte_prefix','path':'TRAJECTORIES/manifest.jsonl',
                  'bytes':trajectory_cursor['manifest_prefix_bytes'],'sha256':digest.hexdigest()})
        stream.flush();os.fsync(stream.fileno())
    return descriptor(path,root)


def _checkpoint(root, backbone, replay, auxiliary, heads, completed_controller, prefill_rng,
                counts, identity, bank_gate, trajectory, bank_reference, bank_gate_reference,
                completed_previous_gate_reference):
    from helixwm_matched.backbone import NativeReplay
    validate_counts(counts,boundary=True)
    if counts['agent_steps'] == 0 or counts['agent_steps'] % 10000:
        raise ValueError('Only 100 exact registered formal checkpoints are allowed')
    replay_state=replay.snapshot()
    payload={'format':CHECKPOINT_FORMAT,'identity':identity,'counts':dict(counts),
        'backbone':backbone.snapshot(),'auxiliary':auxiliary.snapshot(),'pinned_heads':heads.snapshot(),
        'replay':replay_state,'completed_controller_audit':completed_controller,
        'completed_controller_is_previous_episode_pin':True,
        'bank_gate':None if bank_gate is None else dict(bank_gate),
        'bank_reference':copy.deepcopy(bank_reference),'bank_gate_reference':copy.deepcopy(bank_gate_reference),
        'completed_previous_gate_reference':copy.deepcopy(completed_previous_gate_reference),
        'trajectory':None if trajectory is None else trajectory.snapshot(),
        'prefill_rng':copy.deepcopy(prefill_rng.bit_generator.state),'numpy_rng':np.random.get_state(),
        'python_rng':random.getstate(),'next_operation':'current_pin_bank_gate_or_native_fallback_then_reset_new_episode',
        'environment_restore_scope':'completed_episode_boundary_new_registered_reset_not_mid_episode',
        'auto_resume_allowed':False}
    folder=root/'CHECKPOINTS';folder.mkdir(exist_ok=True)
    target=folder/f"checkpoint_{counts['agent_steps']:09d}.pkl"
    dependencies=_checkpoint_dependencies(root,target.with_suffix('.dependencies.jsonl'),replay_state,
        payload['trajectory'],bank_reference,bank_gate_reference,completed_previous_gate_reference)
    with target.open('xb') as stream:
        pickle.dump(payload,stream,protocol=5);stream.flush();os.fsync(stream.fileno())
    result={'format':CHECKPOINT_FORMAT,'identity':identity,'counts':dict(counts),
        'checkpoint':descriptor(target,root),'replay_files':NativeReplay.referenced_files(replay_state),
        'pinned_model_sha256':backbone.inference_bundle_sha256(),'heads_sha256':heads.sha256,
        'bank_gate_sha256':None if bank_gate is None else bank_gate['artifact_sha256'],
        'bank_reference':payload['bank_reference'],'bank_gate_reference':payload['bank_gate_reference'],
        'completed_previous_gate_reference':payload['completed_previous_gate_reference'],
        'trajectory':payload['trajectory'],'dependencies_manifest':dependencies}
    write_json_exclusive(target.with_suffix('.json'),result)
    _status(root/'latest_checkpoint.json',result)
    return result


def _archive_checkpoint(root, record, counts, identity, gate, *, timeout_seconds):
    """Request an immutable recursive dependency transaction; never prune here."""
    index=counts['agent_steps'];folder=root/'ARCHIVE_REQUESTS';folder.mkdir(exist_ok=True)
    manifest=folder/f'{index:09d}.objects.jsonl'
    snapshot_dir=folder/f'{index:09d}_log_snapshot';snapshot_dir.mkdir(exist_ok=False)
    stable_json=[]
    # Mutable status/WAIT JSON is copied to exclusive snapshot paths. Treating
    # an overwritten JSON as an append-only prefix would invalidate its SHA
    # as soon as this function writes the new archive_wait status.
    for path in sorted(root.glob('*.json')):
        target=snapshot_dir/path.name
        with path.open('rb') as src,target.open('xb') as dst:
            shutil.copyfileobj(src,dst);dst.flush();os.fsync(dst.fileno())
        stable_json.append(descriptor(target,root))
    with manifest.open('xb') as stream:
        def emit(value):stream.write(canonical_json(value)+b'\n')
        emit({'kind':'identity','identity':identity,'counts':dict(counts)})
        emit({'kind':'file',**record['checkpoint']})
        emit({'kind':'file',**descriptor(root/record['checkpoint']['path'].replace('.pkl','.json'),root)})
        emit({'kind':'file',**record['dependencies_manifest']})
        for item in record['replay_files']:
            emit({'kind':'file','path':'REPLAY/'+item['path'],'bytes':item['bytes'],'sha256':item['file_sha256']})
        if record['trajectory'] is not None:
            emit({'kind':'trajectory_prefix_recursive','path':'TRAJECTORIES','cursor':record['trajectory'],
                  'all_prefix_NPZ_JSON_dependencies_required':True})
        emit({'kind':'immutable_tree_recursive','path':'BANK_SHARED','include_shared_independent_bank':True})
        emit({'kind':'immutable_tree_recursive','path':'BANK_GATES','include_all_pin_gates':True})
        emit({'kind':'immutable_tree_recursive','path':'SOURCE','include_all_runtime_files':True})
        emit({'kind':'immutable_tree_recursive','path':'EVAL','include_all_existing_rounds':True})
        for item in stable_json:emit({'kind':'file',**item})
        for path in sorted(folder.glob('*.accepted_receipt.json')):
            emit({'kind':'file',**descriptor(path,root)})
        for path in sorted(root.iterdir()):
            if path.is_file() and path.suffix in ('.jsonl','.npy','.log'):
                emit({'kind':'stable_byte_prefix',**descriptor(path,root)})
        stream.flush();os.fsync(stream.fileno())
    request={'schema':REQUEST_SCHEMA,'identity':identity,'counts':dict(counts),
        'checkpoint_sha256':record['checkpoint']['sha256'],'object_manifest':descriptor(manifest,root),
        'archive_service_source_sha256':gate['archive_service_source_sha256'],
        'must_expand_recursive_dependency_kinds':True,'delete_local_unique_data_allowed':False,
        'latest_remote_checkpoint_and_all_dependencies_must_remain':True}
    write_json_exclusive(folder/f'{index:09d}.request.json',request)
    response=wait_for_json(root/'ARCHIVE_ACK'/f'{index:09d}.json',status_path=root/'archive_wait.json',
        stage='archive',identity=identity,counts=counts,timeout_seconds=timeout_seconds)
    validate_archive_receipt(response,request,archive_source_sha256=gate['archive_service_source_sha256'])
    write_json_exclusive(folder/f'{index:09d}.accepted_receipt.json',response)


def _stage_shared_bank(root, directory, manifest_sha256, environment_lock_sha256):
    """Copy and verify the same externally pinned bank; no bank collection here."""
    from .bank_replay import load_original_bank
    directory=Path(directory).resolve();manifest_path=directory/'manifest.json'
    if not _sha_value(manifest_sha256) or sha_file(manifest_path)!=manifest_sha256:
        raise ValueError('Shared independent bank manifest differs from the approved digest')
    manifest=json.loads(manifest_path.read_bytes())
    _,origin=load_original_bank(directory,manifest_sha256,environment_lock_sha256=environment_lock_sha256,
        origin_source_sha256=manifest['source_sha256'],origin_protocol_sha256=manifest['protocol_sha256'])
    target=root/'BANK_SHARED';target.mkdir(exist_ok=False)
    for relative in ['manifest.json']+[r['file'] for r in origin['files']]:
        source=_inside(directory,relative);destination=_inside(target,relative)
        destination.parent.mkdir(parents=True,exist_ok=True)
        with source.open('rb') as src,destination.open('xb') as dst:
            shutil.copyfileobj(src,dst);dst.flush();os.fsync(dst.fileno())
    bank,local_origin=load_original_bank(target,manifest_sha256,environment_lock_sha256=environment_lock_sha256,
        origin_source_sha256=origin['source_sha256'],origin_protocol_sha256=origin['protocol_sha256'])
    return bank,local_origin,{'path':'BANK_SHARED','manifest_sha256':manifest_sha256}


def _bank_gate_at_boundary(root, backbone, auxiliary, heads, counts, identity, bank, bank_origin):
    from .bank_replay import collector_sha256,measure_and_write
    from .bank_gate import load_bank_gate
    context={'source_sha256':identity['source_sha256'],'protocol_sha256':identity['protocol_sha256'],
        'model_sha256':backbone.inference_bundle_sha256(),'model_version':backbone.inference_version,
        'epistemic_heads_sha256':heads.sha256,'collector_sha256':collector_sha256(),
        'bank_manifest_sha256':identity['bank_manifest_sha256'],'measurement_kind':'actual_gpu_independent_bank'}
    folder=root/'BANK_GATES'/f"{counts['agent_steps']:09d}"
    frozen={'backbone':backbone.snapshot(),'auxiliary':auxiliary.snapshot(),'heads':heads.snapshot(),
            'numpy_rng':np.random.get_state(),'python_rng':random.getstate(),'counts':dict(counts)}
    frozen_sha=tree_hash(frozen)
    def measure_frozen():
        state=backbone.snapshot();training=state['training'];params=tree_hash(training['params'])
        return {'model':backbone.inference_bundle_sha256(),'optimizer':params,'normalizers':params,
            'counters':tree_hash({'author':training['counters'],'actual':counts}),
            'training_rng':tree_hash({'author_counter_keys':training['counters'],'carry':state['train_carry'],
                'python':random.getstate(),'numpy':np.random.get_state(),'auxiliary':auxiliary.snapshot()}),
            'epistemic_heads':heads.sha256}
    _status(root/'bank_gate_status.json',{'stage':'offline_current_pin_bank_replay','identity':identity,
        'counts':dict(counts),'environment_actions':0,'scope':'off_policy_empirical_not_closed_loop_certificate','utc':now()})
    devices=backbone.jax.devices()
    result=measure_and_write(backbone,heads,context,bank,directory=folder,bank_origin=bank_origin,
        frozen_state_hashes=measure_frozen,device='cuda',gpu=str(devices[0].device_kind))
    if result['environment_actions']!=0:raise RuntimeError('Bank replay cannot consume extra environment interactions')
    gate=load_bank_gate(folder,result['manifest_sha256'],context,require_active=False)
    after={'backbone':backbone.snapshot(),'auxiliary':auxiliary.snapshot(),'heads':heads.snapshot(),
           'numpy_rng':np.random.get_state(),'python_rng':random.getstate(),'counts':dict(counts)}
    if tree_hash(after)!=frozen_sha:raise RuntimeError('Offline bank replay mutated live learner/heads/RNG')
    reference={'path':folder.relative_to(root).as_posix(),'manifest_sha256':result['manifest_sha256'],
        'artifact_sha256':gate['artifact_sha256'],'context':context}
    _append(root/'bank_gate_ledger.jsonl',{'identity':identity,'counts':dict(counts),'reference':reference,
        'active':gate['active'],'environment_actions':0,'utc':now(),'scope':gate['evidence_scope']})
    return gate,context,reference


def _periodic_evaluation(backbone, heads, context, bank_gate, config, root, arm, counts):
    """Registered ten frozen cases using private controllers; no learner data."""
    from .full_controller import FullController
    from .environment import original_environment_contract
    from .trajectory import TrajectoryArchive
    from helixwm_matched.controllers import Controller
    from helixwm_matched.environment import MatchedWalkerEnv
    folder=root/'EVAL'/f"{counts['agent_steps']:09d}";folder.mkdir(parents=True,exist_ok=False)
    before=tree_hash(backbone.snapshot());head_before=heads.sha256
    rng_before=tree_hash((np.random.get_state(),random.getstate()));rows=[];ledger={'actions':0,'DMC_calls':0}
    try:
        for name,gap in config['periodic_evaluation']['conditions_by_arm'][arm]:
            for seed in config['periodic_evaluation']['environment_seeds']:
                key=f'{name}_seed{seed}_gap{gap}';trace=None
                if name=='helix_final':
                    trace=TrajectoryArchive.create(folder/(key+'_trajectory'),
                        source_sha256=context['source_sha256'],protocol_sha256=context['protocol_sha256'])
                    control=FullController(backbone,heads,context,seed=seed,bank_gate=bank_gate,
                        execution='bank_production',trajectory=trace)
                else:control=Controller(backbone,name,seed=seed)
                env=MatchedWalkerEnv(original_environment_contract())
                mask=np.ones(501,np.bool_) if gap==0 else protocol.fixed_gap_mask(gap)
                rewards=[];actions=[];available=[]
                try:
                    initial=env.reset(seed=seed,mask=mask)
                    if (not isinstance(initial,np.ndarray) or initial.dtype!=np.uint8
                            or initial.shape!=(64,64,3) or env.action_decisions!=0 or env.episode_env_steps!=0):
                        raise RuntimeError('Periodic evaluation reset image/counters differ')
                    control.reset(initial,episode_id=seed)
                    for timestamp in range(1,501):
                        a=np.asarray(control.act(),np.float32).reshape(6)
                        if not np.isfinite(a).all() or np.any(np.abs(a)>1):
                            raise FloatingPointError('Periodic evaluation issued a nonfinite or unbounded action')
                        a0,c0=env.action_decisions,env.episode_env_steps
                        try:image,reward,done,info=env.step(a)
                        finally:
                            ledger['actions']+=env.action_decisions-a0;ledger['DMC_calls']+=env.episode_env_steps-c0
                        if (type(done) is not bool or done!=(timestamp==500) or type(info) is not dict
                                or type(info.get('observation_timestamp')) is not int
                                or info['observation_timestamp']!=timestamp
                                or type(info.get('availability')) is not bool
                                or info['availability']!=bool(mask[timestamp])
                                or type(info.get('env_steps_this_step')) is not int or info['env_steps_this_step']!=2
                                or env.action_decisions-a0!=1 or env.episode_env_steps-c0!=2
                                or env.action_decisions!=timestamp or env.episode_env_steps!=2*timestamp):
                            raise RuntimeError('Periodic evaluation horizon/mask/timestamp/actual-call mismatch')
                        if not math.isfinite(float(reward)) or not -1e-12<=float(reward)<=2+1e-12:
                            raise FloatingPointError('Periodic evaluation returned a nonfinite or out-of-range reward')
                        if (not isinstance(image,np.ndarray) or image.dtype!=np.uint8 or image.shape!=(64,64,3)
                                or (not info['availability'] and np.any(image))):
                            raise ValueError('Periodic evaluation must expose only the visible/corrupted RGB image')
                        control.step(image,a,info,episode_id=seed)
                        rewards.append(float(reward));actions.append(a);available.append(info['availability'])
                    with (folder/(key+'.npz')).open('xb') as stream:
                        np.savez_compressed(stream,reward=np.asarray(rewards,np.float64),action=np.stack(actions),
                                            availability=np.asarray(available,np.bool_))
                    row={'arm':arm,'training_actions':counts['agent_steps'],'controller':name,'seed':seed,'gap':gap,
                        'return':math.fsum(rewards),'actions':500,'DMC_calls':1000,'availability_sha256':protocol.mask_sha256(mask)}
                    rows.append(row);write_json_exclusive(folder/(key+'.json'),row);_append(root/'eval.jsonl',row)
                finally:
                    env.close()
                    if trace is not None:trace.close()
        if len(rows)!=10 or ledger!={'actions':5000,'DMC_calls':10000}:raise RuntimeError('Periodic evaluation cardinality differs')
        after=tree_hash(backbone.snapshot())
        if before!=after or head_before!=heads.sha256 or rng_before!=tree_hash((np.random.get_state(),random.getstate())):
            raise RuntimeError('Periodic evaluation changed frozen model/heads/global RNG')
        write_json_exclusive(folder/'frozen_report.json',{'rows':rows,'ledger':ledger,'training_updates':0,
            'model_before_sha256':before,'model_after_sha256':after,'heads_sha256':head_before,'host_rng_sha256':rng_before})
    except BaseException as exc:
        write_json_exclusive(folder/'failure.json',{'error':repr(exc),'actual_evaluation_interactions':ledger,'completed_cases':len(rows)})
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--upstream',type=Path,required=True)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--arm',choices=['D','F'],required=True)
    parser.add_argument('--environment-lock',type=Path,required=True)
    parser.add_argument('--bank-dir',type=Path,required=True,help='Existing shared 2480-action independent bank; never collected by this CLI')
    parser.add_argument('--bank-manifest-sha256',required=True)
    parser.add_argument('--launch-gate',type=Path,required=True)
    parser.add_argument('--launch-gate-sha256',required=True)
    parser.add_argument('--wait-timeout-seconds',type=float,default=0.,help='0 waits at a boundary until valid external evidence; never retries failures')
    args=parser.parse_args()
    config=protocol.load_protocol(args.protocol)
    from .provenance import source_digest
    source,files=source_digest()
    root=args.run_dir.resolve()
    identity={'run_id':root.name,'arm':args.arm,'source_sha256':source,'protocol_sha256':protocol.protocol_sha256(config),
              'environment_lock_sha256':sha_file(args.environment_lock),'upstream_commit':config['official_core']['commit'],
              'bank_manifest_sha256':args.bank_manifest_sha256}
    gate=validate_launch_gate(args.launch_gate,args.launch_gate_sha256,identity,config,root)
    if root.exists():raise FileExistsError('Fresh new run required; automatic resume is forbidden')
    if not math.isfinite(args.wait_timeout_seconds) or args.wait_timeout_seconds<0:raise ValueError('Invalid evidence wait deadline')
    root.mkdir(parents=True,exist_ok=False)
    (root/'BANK_GATES').mkdir()
    (root/'EVAL').mkdir()
    for name,value in [('identity.json',identity),('protocol.json',config),('source_manifest.json',files),
                       ('command.json',{'argv':__import__('sys').argv}),('launch_gate.json',gate)]:write_json_exclusive(root/name,value)
    runtime_root=Path(__file__).resolve().parents[1]
    for relative,expected in files.items():
        source_path=_inside(runtime_root,relative);target=_inside(root,'SOURCE/'+relative)
        target.parent.mkdir(parents=True,exist_ok=True)
        with source_path.open('rb') as src,target.open('xb') as dst:
            shutil.copyfileobj(src,dst);dst.flush();os.fsync(dst.fileno())
        if sha_file(target)!=expected:raise ValueError('Runtime source changed during staging: '+relative)
    shutil.copyfile(args.environment_lock,root/'environment_lock.json')
    (root/'train.pid').write_text(str(os.getpid())+'\n')
    counts={'agent_steps':0,'env_steps':0,'episodes':0,'update_steps':0};env=trajectory=None
    started=time.monotonic();records=[]
    try:
        _wait_space(root,gate['capacity']['remote_required_bytes_before_start'],counts,identity,timeout_seconds=args.wait_timeout_seconds)
        bank,bank_origin,bank_reference=_stage_shared_bank(root,args.bank_dir,args.bank_manifest_sha256,identity['environment_lock_sha256'])
        write_json_exclusive(root/'bank_origin.json',bank_origin)
        os.environ.setdefault('MUJOCO_GL','egl');os.environ.setdefault('PYOPENGL_PLATFORM','egl')
        os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
        from .backbone import FinalBackbone
        from .epistemic import BootstrapDynamics
        from .environment import original_environment_contract
        from .full_controller import FullController
        from .trajectory import TrajectoryArchive
        from helixwm_matched.backbone import NativeReplay
        from helixwm_matched.controllers import Controller
        from helixwm_matched.environment import MatchedWalkerEnv
        from helixwm_matched.env_probe import runtime_lock,first_difference
        from helixwm_matched.train import resources
        np.random.seed(0);random.seed(0)
        masks=protocol.make_mask_bank(config)
        with (root/'training_masks.npy').open('xb') as stream:np.save(stream,masks,allow_pickle=False)
        backbone=FinalBackbone({'seed':0,'size':'size12m','platform':'cuda','compute_dtype':'bfloat16','batch_size':16,'batch_length':64},args.upstream,root)
        write_json_exclusive(root/'resolved_backbone.json',backbone.resolved_config)
        replay=NativeReplay(backbone,directory=root/'REPLAY',capacity=1000000,seed=0)
        auxiliary=BootstrapDynamics(0,{k:identity[k] for k in ('source_sha256','protocol_sha256')})
        prefill=np.random.default_rng(810000);env=MatchedWalkerEnv(original_environment_contract())
        backbone.pin_inference();heads=auxiliary.pin()
        native=Controller(backbone,'black_frame_native',seed=0);controller=native
        current_gate=context=previous_full_snapshot=gate_reference=None
        def verify_live_lock():
            if counts['agent_steps']==0:
                memory_limit=Path('/sys/fs/cgroup/memory.max')
                if not memory_limit.is_file() or memory_limit.read_text().strip()!=str(90*1024**3):
                    raise RuntimeError('The registered 90GiB cgroup memory limit is not verified')
                live=runtime_lock(original_environment_contract());write_json_exclusive(root/'live_environment_lock.json',live)
                mismatch=first_difference(json.loads(args.environment_lock.read_bytes()),live)
                if mismatch:raise RuntimeError('Live environment differs from approved lock at '+mismatch)
        def interaction(row):
            _append(root/'interactions.jsonl',row)
            if row['agent_steps']%100==0:_status(root/'status.json',{'stage':'formal_training','identity':identity,'counts':dict(counts),'utc':now()})
        def metric(c,m):
            if c['update_steps']<=3 or c['update_steps']%10==0:_append(root/'metrics.jsonl',{'utc':now(),**c,**m})
        for episode in range(2000):
            _wait_space(root,gate['capacity']['remote_episode_write_peak_bytes']+RESERVE_BYTES,
                counts,identity,timeout_seconds=args.wait_timeout_seconds)
            if args.arm=='F' and counts['agent_steps']>=5500:
                if context is None:raise RuntimeError('Final bank control requires the exact current-pin context')
                if trajectory is None:trajectory=TrajectoryArchive.create(root/'TRAJECTORIES',source_sha256=source,protocol_sha256=identity['protocol_sha256'])
                controller=FullController(backbone,heads,context,seed=0,bank_gate=current_gate,
                    execution='bank_production',trajectory=trajectory)
                if previous_full_snapshot is not None:controller.restore_completed_episode(previous_full_snapshot)
            else:controller=native
            episode_gate_reference=gate_reference if isinstance(controller,FullController) else None
            row=run_episode(env=env,controller=controller,replay=replay,auxiliary=auxiliary,prefill_rng=prefill,
                counts=counts,episode=episode,seed=protocol.training_seed(config,episode),mask=masks[episode],
                on_reset=verify_live_lock,on_interaction=interaction,on_metric=metric,arm=args.arm)
            _append(root/'episodes.jsonl',row);resources(root,counts)
            completed=controller.snapshot()
            if isinstance(controller,FullController):previous_full_snapshot=completed
            # Exactly one next pin per episode boundary, also the final pin.
            backbone.pin_inference();heads=auxiliary.pin()
            cap=gate['capacity'];reserve=cap['remote_model_anchor_reserve_bytes']+RESERVE_BYTES
            _wait_space(root,reserve,counts,identity,timeout_seconds=args.wait_timeout_seconds)
            if protocol.calibration_required_at(config,args.arm,counts['agent_steps']):
                current_gate,context,gate_reference=_bank_gate_at_boundary(root,backbone,auxiliary,heads,counts,identity,bank,bank_origin)
            if counts['agent_steps']%10000==0:
                peak=int(cap['remote_checkpoint_peak_factor']*replay.statistics()['ram_bytes'])+reserve
                _wait_space(root,peak,counts,identity,timeout_seconds=args.wait_timeout_seconds)
                cp=_checkpoint(root,backbone,replay,auxiliary,heads,completed,prefill,counts,identity,current_gate,trajectory,
                    bank_reference,gate_reference,episode_gate_reference)
                records.append(cp)
                _archive_checkpoint(root,cp,counts,identity,gate,timeout_seconds=args.wait_timeout_seconds)
            if counts['agent_steps']%20000==0:
                before=tree_hash({'controller':completed,'auxiliary':auxiliary.snapshot(),'prefill':prefill.bit_generator.state,
                    'sampler':replay.replay.sampler.rng.bit_generator.state,'stream_index':int(replay.stream.index),'counts':counts})
                _periodic_evaluation(backbone,heads,context,current_gate,config,root,args.arm,counts)
                after=tree_hash({'controller':controller.snapshot(),'auxiliary':auxiliary.snapshot(),'prefill':prefill.bit_generator.state,
                    'sampler':replay.replay.sampler.rng.bit_generator.state,'stream_index':int(replay.stream.index),'counts':counts})
                if before!=after:raise RuntimeError('Periodic evaluation mutated collector/replay/auxiliary training state')
            print(json.dumps({'utc':now(),'arm':args.arm,**counts,'return':row['return']},allow_nan=False),flush=True)
        validate_counts(counts,boundary=True)
        if counts['agent_steps']!=1000000 or len(records)!=100:raise RuntimeError('Formal budget or 100-checkpoint requirement incomplete')
        env.close();env=None
        if trajectory is not None:trajectory.close();trajectory=None
        write_json_exclusive(root/'completion.json',{'training_complete':True,'final_acceptance_complete':False,
            'identity':identity,'counts':counts,'checkpoint_count':100,'final_checkpoint':records[-1]['checkpoint'],
            'periodic_evaluation_rounds':50,'final_evaluation_started':False,'wall_s':time.monotonic()-started,
            'shared_bank_original_actions':2480,'shared_bank_collected_by_this_run':False,
            'per_pin_extra_environment_actions':0,'bank_gate_scope':'off_policy_empirical_not_closed_loop_certificate'})
        (root/'train.exit').write_text('0\n')
    except BaseException as exc:
        write_json_exclusive(root/'failure.json',{'identity':identity,'counts':counts,'error':repr(exc),
            'traceback':traceback.format_exc(),'automatic_retry_or_resume':False})
        (root/'train.exit').write_text('1\n');raise
    finally:
        if env is not None:env.close()
        if trajectory is not None:trajectory.close()


if __name__=='__main__':main()
