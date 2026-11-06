"""Fresh 6000-action integrated v3 engineering preflight, not formal training.

Reuses the actual formal episode and bank-boundary implementation. It exercises
the common native startup, a complete Final episode, two current bank gates and
the full dependency-backed checkpoint schema. No launch gate is fabricated and
the registered 1M formal budget is never changed. Inactive gates are reported;
native fallback execution is not evidence that the repair/planner was active.
"""
from __future__ import annotations
import argparse
import copy
import json
import os
from pathlib import Path
import pickle
import random
import shutil
import time
import traceback
import numpy as np

from . import protocol_v3 as protocol
from . import train

FORMAT='helixwm_final_bank_preflight_boundary_v3'


def probe_fields(config):
    protocol.validate_protocol(config)
    return {'probe_registration':copy.deepcopy(config['full_integration_preflight']),
            'formal_training_actions':0,'fresh_initialization':True,'old_checkpoint_loaded':False}


def validate_probe_identity(identity,config,counts=None):
    train.validate_identity(identity)
    if identity['arm']!='F' or not identity['run_id'].startswith('preflight_'):
        raise ValueError('Integrated preflight must have its explicit F preflight identity')
    if identity['protocol_sha256']!=protocol.protocol_sha256(config):raise ValueError('Preflight protocol differs')
    if counts is not None and counts!=config['full_integration_preflight']['counts']:
        raise ValueError('Full preflight must finish exactly 6000 actions/12000 DMC/12 episodes/250 updates')


def integration_loop(*,config,backbone,auxiliary,native,env,replay,prefill_rng,counts,
                     measure_gate,make_final,on_reset=None,on_interaction=None,on_metric=None,on_episode=None):
    """Actual counted episode engine with injectable CPU orchestration edges."""
    protocol.validate_protocol(config)
    if counts!={'agent_steps':0,'env_steps':0,'episodes':0,'update_steps':0}:raise ValueError('Fresh zero-count startup required')
    masks=protocol.full_preflight_masks(config)
    backbone.pin_inference();heads=auxiliary.pin()
    gate=context=reference=previous_reference=None;completed=None;controller=native
    ledger=[]
    for episode in range(12):
        if episode==11:
            if context is None or reference is None:raise RuntimeError('Current 5500 pin bank measurement is missing')
            controller=make_final(heads,context,gate)
            previous_reference=reference
        row=train.run_episode(env=env,controller=controller,replay=replay,auxiliary=auxiliary,
            prefill_rng=prefill_rng,counts=counts,episode=episode,seed=72000+episode,mask=masks[episode],
            on_reset=on_reset,on_interaction=on_interaction,on_metric=on_metric)
        completed=controller.snapshot()
        backbone.pin_inference();heads=auxiliary.pin()
        if counts['agent_steps'] in (5500,6000):
            gate,context,reference=measure_gate(heads)
            ledger.append({'agent_steps':counts['agent_steps'],'model_version':context['model_version'],
                           'active':bool(gate['active']),'reference':copy.deepcopy(reference)})
        if on_episode is not None:on_episode(row)
    if counts!=config['full_integration_preflight']['counts']:raise RuntimeError('Integrated actual count mismatch')
    return {'heads':heads,'gate':gate,'context':context,'gate_reference':reference,
            'previous_gate_reference':previous_reference,'completed_controller':completed,
            'bank_gate_ledger':ledger,'controller':controller,'masks':masks}


def save_checkpoint(root,*,config,identity,counts,backbone,replay,auxiliary,heads,completed_controller,
                    prefill_rng,gate,trajectory,bank_reference,gate_reference,previous_gate_reference):
    """Full formal payload layout, with an explicit non-formal format/registration.

    train._checkpoint intentionally permits only registered 10k formal saves;
    do not spoof counts or relax that guard for this isolated 6000-action probe.
    Its dependency inventory and descriptors are shared unchanged.
    """
    validate_probe_identity(identity,config,counts)
    from helixwm_matched.backbone import NativeReplay
    root=Path(root);folder=root/'CHECKPOINTS';folder.mkdir(exist_ok=False)
    checkpoint=folder/'checkpoint_000006000.pkl';replay_state=replay.snapshot()
    cursor=None if trajectory is None else trajectory.snapshot()
    payload={'format':FORMAT,'identity':copy.deepcopy(identity),'counts':dict(counts),**probe_fields(config),
        'backbone':backbone.snapshot(),'auxiliary':auxiliary.snapshot(),'pinned_heads':heads.snapshot(),
        'replay':replay_state,'completed_controller_audit':completed_controller,
        'completed_controller_is_previous_episode_pin':True,'bank_gate':dict(gate),
        'bank_reference':copy.deepcopy(bank_reference),'bank_gate_reference':copy.deepcopy(gate_reference),
        'completed_previous_gate_reference':copy.deepcopy(previous_gate_reference),'trajectory':cursor,
        'prefill_rng':copy.deepcopy(prefill_rng.bit_generator.state),'numpy_rng':np.random.get_state(),
        'python_rng':random.getstate(),'next_operation':'current_pin_bank_gate_or_native_fallback_then_reset_new_episode',
        'environment_restore_scope':'completed_episode_boundary_new_registered_reset_not_mid_episode',
        'auto_resume_allowed':False}
    dependencies=train._checkpoint_dependencies(root,checkpoint.with_suffix('.dependencies.jsonl'),replay_state,cursor,
        bank_reference,gate_reference,previous_gate_reference)
    with checkpoint.open('xb') as stream:
        pickle.dump(payload,stream,protocol=5);stream.flush();os.fsync(stream.fileno())
    record={'format':FORMAT,'identity':copy.deepcopy(identity),'counts':dict(counts),**probe_fields(config),
        'checkpoint':train.descriptor(checkpoint,root),'replay_files':NativeReplay.referenced_files(replay_state),
        'pinned_model_sha256':backbone.inference_bundle_sha256(),'heads_sha256':heads.sha256,
        'bank_gate_sha256':gate['artifact_sha256'],'bank_reference':payload['bank_reference'],
        'bank_gate_reference':payload['bank_gate_reference'],
        'completed_previous_gate_reference':payload['completed_previous_gate_reference'],
        'trajectory':cursor,'dependencies_manifest':dependencies}
    train.write_json_exclusive(checkpoint.with_suffix('.json'),record)
    train.write_json_exclusive(root/'latest_checkpoint.json',record)
    return record


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--upstream',type=Path,required=True)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--environment-lock',type=Path,required=True)
    parser.add_argument('--bank-dir',type=Path,required=True)
    parser.add_argument('--bank-manifest-sha256',required=True)
    args=parser.parse_args();config=protocol.load_protocol(args.protocol)
    from .provenance import source_digest
    source,files=source_digest();root=args.run_dir.resolve()
    identity={'run_id':'preflight_'+root.parent.name+'_'+root.name,'arm':'F','source_sha256':source,
        'protocol_sha256':protocol.protocol_sha256(config),'environment_lock_sha256':train.sha_file(args.environment_lock),
        'upstream_commit':config['official_core']['commit'],'bank_manifest_sha256':args.bank_manifest_sha256}
    validate_probe_identity(identity,config)
    root.mkdir(parents=True,exist_ok=False)
    counts={'agent_steps':0,'env_steps':0,'episodes':0,'update_steps':0};env=trajectory=None
    started=time.monotonic()
    try:
        if shutil.disk_usage(root).free<8*1024**3:raise RuntimeError('Preflight requires at least 8 GiB free before model/replay writes')
        for name,value in [('identity.json',identity),('protocol.json',config),('source_manifest.json',files),
                           ('command.json',{'argv':__import__('sys').argv}),('probe_registration.json',probe_fields(config))]:
            train.write_json_exclusive(root/name,value)
        runtime=Path(__file__).resolve().parents[1]
        for relative,expected in files.items():
            target=root/'SOURCE'/relative;target.parent.mkdir(parents=True,exist_ok=True)
            with (runtime/relative).open('rb') as src,target.open('xb') as dst:shutil.copyfileobj(src,dst)
            if train.sha_file(target)!=expected:raise ValueError('Source changed while staging preflight')
        shutil.copyfile(args.environment_lock,root/'environment_lock.json')
        (root/'train.pid').write_text(str(os.getpid())+'\n')
        bank,origin,bank_reference=train._stage_shared_bank(root,args.bank_dir,args.bank_manifest_sha256,identity['environment_lock_sha256'])
        train.write_json_exclusive(root/'bank_origin.json',origin)
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
        backbone=FinalBackbone({'seed':0,'size':'size12m','platform':'cuda','compute_dtype':'bfloat16','batch_size':16,'batch_length':64},args.upstream,root)
        train.write_json_exclusive(root/'resolved_backbone.json',backbone.resolved_config)
        replay=NativeReplay(backbone,directory=root/'REPLAY',capacity=1000000,seed=0)
        auxiliary=BootstrapDynamics(0,{k:identity[k] for k in ('source_sha256','protocol_sha256')})
        prefill=np.random.default_rng(config['full_integration_preflight']['prefill_action_rng_seed'])
        native=Controller(backbone,'black_frame_native',seed=0);env=MatchedWalkerEnv(original_environment_contract())
        with (root/'training_masks.npy').open('xb') as stream:np.save(stream,protocol.full_preflight_masks(config),allow_pickle=False)
        def reset_gate():
            if counts['agent_steps']==0:
                limit=Path('/sys/fs/cgroup/memory.max')
                if not limit.is_file() or limit.read_text().strip()!=str(90*1024**3):raise RuntimeError('Expected actual 90GiB cgroup memory limit')
                live=runtime_lock(original_environment_contract());train.write_json_exclusive(root/'live_environment_lock.json',live)
                difference=first_difference(json.loads(args.environment_lock.read_bytes()),live)
                if difference:raise RuntimeError('Original environment lock differs after actual render: '+difference)
        def new_final(heads,context,gate):
            nonlocal trajectory
            trajectory=TrajectoryArchive.create(root/'TRAJECTORIES',source_sha256=source,protocol_sha256=identity['protocol_sha256'])
            return FullController(backbone,heads,context,seed=0,bank_gate=gate,execution='bank_production',trajectory=trajectory)
        def measure_gate(heads):return train._bank_gate_at_boundary(root,backbone,auxiliary,heads,counts,identity,bank,origin)
        def interaction(row):
            train._append(root/'interactions.jsonl',row)
            if row['agent_steps']%100==0:
                train._status(root/'status.json',{'stage':'isolated_full_preflight','counts':dict(counts),'formal_training_actions':0})
        def metric(c,m):train._append(root/'metrics.jsonl',{**c,**m})
        def episode(row):
            train._append(root/'episodes.jsonl',row);resources(root,counts)
            print(json.dumps({'stage':'isolated_full_preflight',**row,'wall_s':time.monotonic()-started}),flush=True)
        resources(root,counts)
        result=integration_loop(config=config,backbone=backbone,auxiliary=auxiliary,native=native,env=env,replay=replay,
            prefill_rng=prefill,counts=counts,measure_gate=measure_gate,make_final=new_final,on_reset=reset_gate,
            on_interaction=interaction,on_metric=metric,on_episode=episode)
        if source_digest()!=(source,files):raise RuntimeError('Preflight runtime source changed')
        difference=first_difference(json.loads(args.environment_lock.read_bytes()),runtime_lock(original_environment_contract()))
        if difference:raise RuntimeError('Environment runtime changed during preflight: '+difference)
        checkpoint=save_checkpoint(root,config=config,identity=identity,counts=counts,backbone=backbone,replay=replay,
            auxiliary=auxiliary,heads=result['heads'],completed_controller=result['completed_controller'],prefill_rng=prefill,
            gate=result['gate'],trajectory=trajectory,bank_reference=bank_reference,gate_reference=result['gate_reference'],
            previous_gate_reference=result['previous_gate_reference'])
        env.close();env=None
        if trajectory is not None:trajectory.close();trajectory=None
        train.write_json_exclusive(root/'completion.json',{'engineering_execution_completed':True,**probe_fields(config),
            'counts':counts,'bank_gate_ledger':result['bank_gate_ledger'],'checkpoint':checkpoint,
            'final_acceptance':False,'formal_training_started':False,'independent_recovery_verified':False,
            'inactive_gate_execution_is_native_fallback_not_repair_activation':True,'wall_s':time.monotonic()-started})
        (root/'train.exit').write_text('0\n')
    except BaseException as exc:
        train.write_json_exclusive(root/'failure.json',{'identity':identity,'counts':dict(counts),**probe_fields(config),
            'error':repr(exc),'traceback':traceback.format_exc(),'automatic_retry_or_resume':False})
        (root/'train.exit').write_text('1\n');raise
    finally:
        if env is not None:env.close()
        if trajectory is not None:trajectory.close()


if __name__=='__main__':main()
