"""Two fresh-process next-action/sample/update checks for bank-v3 full state.

This tool never resumes a formal job. It preserves inputs, uses a new isolated
replay/trajectory output, and counts its 4 actions/8 DMC/1 update per process as
engineering work. It proves neither uninterrupted continuation nor cross-hardware
or long-horizon identity. Bootstrap-v1 checkpoints are rejected by the loader.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import traceback

import numpy as np

from . import final_checkpoint as checkpoint
from .recovery_probe import (tree_hash as _plain_tree_hash,materialize_replay,sample_descriptor,
                             jsonable,_replay_contents)

REGISTRATION={'environment_seed':72120,'missing_observation_indices':[3],
    'selection_timing':'registered_before_formal_or_recovery_execution','performance_comparison':False,
    'per_process_actions':4,'per_process_DMC_calls':8,'per_process_author_updates':1,
    'formal_training_actions':0,'two_fresh_processes':True}
RESTORED_FIELDS=('training','train_carry','inference_params','inference_version','writeback_updates',
                 'final_auxiliary_heads','final_inference_bundle','final_upstream_api_sha256')
COMPARISON_FIELDS={'identity','checkpoint_format','source_counts','source_checkpoint_sha256','source_record_sha256','tool_bundle_sha256','registration',
    'restored_backbone','restored_auxiliary','restored_replay','restored_private_rng','restored_host_rng',
    'restored_prefill_rng','parent_trajectory_cursor','current_gate_artifact_sha256','boundary_audit',
    'reset','steps','sample','sampler_rng_after_sample','after_backbone','after_auxiliary','after_replay',
    'after_collector','after_host_rng','metrics','isolated_counts','gpu_devices'}


def tree_hash(value):
    """Hash immutable verified mappings by content, never by capability identity."""
    def ordinary(x):
        if isinstance(x,dict):return {k:ordinary(v) for k,v in x.items()}
        if type(x) is list:return [ordinary(v) for v in x]
        if type(x) is tuple:return tuple(ordinary(v) for v in x)
        return x
    return _plain_tree_hash(ordinary(value))


def tool_digest():
    files={name:checkpoint.sha_file(Path(__file__).with_name(name))
           for name in ('final_checkpoint.py','final_recovery_probe.py')}
    return hashlib.sha256(checkpoint.canonical(files)).hexdigest()


def write_json(path,value):
    with Path(path).open('xb') as stream:
        stream.write(checkpoint.canonical(jsonable(value))+b'\n');stream.flush();os.fsync(stream.fileno())


def isolated_output(source_root,output):
    source=Path(source_root).resolve();target=Path(output).resolve()
    if target==source or target.is_relative_to(source) or source.is_relative_to(target):
        raise ValueError('Recovery output must be isolated from every original run object')
    if target.exists():raise FileExistsError('Never overwrite earlier recovery evidence')
    checkpoint.safe_path(target.parent,target.name)
    return target


def load_inputs(args):
    if args.tool_bundle_sha256!=tool_digest():raise ValueError('External independent recovery tool SHA differs')
    if checkpoint.sha_file(args.identity_json)!=args.identity_sha256:raise ValueError('External identity JSON SHA differs')
    identity=json.loads(Path(args.identity_json).read_bytes())
    from .protocol_v3 import load_protocol
    config=load_protocol(args.protocol)
    payload,record,gate=checkpoint.load_checkpoint(args.checkpoint,expected_sha256=args.checkpoint_sha256,
        expected_record_sha256=args.record_sha256,expected_identity=identity,protocol_config=config,
        dependency_root=args.dependency_root,runtime_source_root=args.runtime_source_root,
        allow_preflight=args.preflight_checkpoint)
    root=Path(args.dependency_root).resolve() if args.dependency_root else Path(args.checkpoint).resolve().parent.parent
    return payload,record,gate,identity,config,root


def restored_private_rng(controller,arm):
    snap=controller.snapshot()
    if arm=='D':return {'native':snap['chain']['rngs']}
    return {'final':snap['rngs'],'screen':snap['integration']['screen_rng']}


def run_worker(args):
    args._progress={'probe_actions':0,'probe_dmc_calls':0,'probe_updates':0,'auxiliary_update_attempts':0,
                    'auxiliary_eligible_updates':0,'actual_counts_complete':True,'formal_training_actions':0}
    payload,record,gate,identity,config,source=load_inputs(args)
    output=isolated_output(source,args.output_new);output.mkdir(parents=True,exist_ok=False)
    write_json(output/'command.json',{'argv':sys.argv,'registration':REGISTRATION,'mode':args.mode})
    replay_bytes=sum(x['bytes'] for x in record['replay_files'])
    if shutil.disk_usage(output).free<replay_bytes+2*1024**3:
        raise RuntimeError('Insufficient isolated replay copy space plus 2 GiB reserve; original inputs preserved')
    storage=materialize_replay(source/'REPLAY',output/'REPLAY',record['replay_files'])
    os.environ.setdefault('MUJOCO_GL','egl');os.environ.setdefault('PYOPENGL_PLATFORM','egl')
    os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
    from .backbone import FinalBackbone
    from .epistemic import BootstrapDynamics,FrozenBootstrap
    from .environment import original_environment_contract
    from .full_controller import FullController
    from .trajectory import TrajectoryArchive
    from helixwm_matched.backbone import NativeReplay
    from helixwm_matched.controllers import Controller
    from helixwm_matched.environment import MatchedWalkerEnv
    from helixwm_matched.env_probe import runtime_lock,first_difference

    backbone=FinalBackbone({'seed':0,'size':'size12m','platform':'cuda','compute_dtype':'bfloat16',
        'batch_size':16,'batch_length':64},args.upstream,output/'MODEL')
    backbone.load(payload['backbone']);restored=backbone.snapshot()
    restored_checks={k:tree_hash(restored[k])==tree_hash(payload['backbone'][k]) for k in RESTORED_FIELDS}
    if not all(restored_checks.values()):raise RuntimeError('Strict complete Final backbone reload differs')
    rp=payload['replay'];replay=NativeReplay(backbone,output/'REPLAY',capacity=rp['capacity'],seed=rp['seed'],chunksize=rp['chunksize'])
    replay.load(rp);replayed=replay.snapshot()
    if tree_hash(replayed)!=tree_hash(rp):raise RuntimeError('Exact native replay and sampler state differ')
    auxiliary=BootstrapDynamics(0,{k:identity[k] for k in ('source_sha256','protocol_sha256')})
    auxiliary.load(payload['auxiliary']);heads=FrozenBootstrap(payload['pinned_heads'])
    if tree_hash(auxiliary.snapshot())!=tree_hash(payload['auxiliary']):raise RuntimeError('Auxiliary optimizer/RNG restore differs')
    old=payload['completed_controller_audit'];parent_trace=payload['trajectory'];trace=None
    if identity['arm']=='D':
        controller=Controller(backbone,'black_frame_native',seed=0)
        boundary=controller.chain.restore_completed_episode(old['chain'])
        expected_rng={'native':old['chain']['rngs']}
    else:
        origin=TrajectoryArchive.restore(source/'TRAJECTORIES',parent_trace,
            expected_source_sha256=identity['source_sha256'],expected_protocol_sha256=identity['protocol_sha256'],writable=False)
        controller=FullController(backbone,heads,record['bank_gate_reference']['context'],seed=0,
                                  bank_gate=gate,execution='bank_production',trajectory=origin)
        boundary=controller.restore_completed_episode(old)
        expected_rng={'final':old['rngs'],'screen':old['integration']['screen_rng']}
        # This is an isolated engineering branch. Keep the authenticated parent
        # cursor as lineage, then write NEW probe arrays in a separate archive.
        origin.close()
        trace=TrajectoryArchive.create(output/'PROBE_TRAJECTORY',source_sha256=identity['source_sha256'],
                                        protocol_sha256=identity['protocol_sha256'])
        write_json(output/'trajectory_parent.json',{'parent_cursor':parent_trace,'parent_root':str(source/'TRAJECTORIES'),
            'new_independent_probe_archive':True,'original_archive_opened_for_write':False})
        controller.trajectory=trace
    private=restored_private_rng(controller,identity['arm'])
    if tree_hash(private)!=tree_hash(expected_rng):raise RuntimeError('Completed collector private RNG streams differ')
    prefill=np.random.default_rng();prefill.bit_generator.state=copy.deepcopy(payload['prefill_rng'])
    np.random.set_state(payload['numpy_rng']);random.setstate(payload['python_rng'])
    host_before={'numpy':tree_hash(np.random.get_state()),'python':tree_hash(random.getstate())}
    env=MatchedWalkerEnv(original_environment_contract(),diagnostics=True)
    mask=np.ones(501,np.bool_);mask[3]=False;steps=[]
    base=payload['counts'];isolated=dict(base)
    try:
        image=env.reset(seed=REGISTRATION['environment_seed'],mask=mask)
        actual=runtime_lock(original_environment_contract());expected=json.loads((source/'environment_lock.json').read_bytes())
        mismatch=first_difference(expected,actual)
        if mismatch:raise ValueError('Live registered environment changed: '+mismatch)
        write_json(output/'live_environment_lock.json',actual)
        controller.reset(image,episode_id=base['episodes'])
        reset={'image':tree_hash(image),'state':tree_hash(controller.state),'info':jsonable(env.last_info)}
        observation={'image':image.copy(),'reward':np.float32(0.),'is_first':np.bool_(True),
                     'is_last':np.bool_(False),'is_terminal':np.bool_(False)}
        available=True
        for t in range(4):
            action=controller.act();state=copy.deepcopy(controller.state)
            before=(env.action_decisions,env.episode_env_steps)
            try:next_image,reward,done,info=env.step(action)
            finally:
                args._progress.update(probe_actions=env.action_decisions,probe_dmc_calls=env.episode_env_steps)
                isolated['agent_steps']+=env.action_decisions-before[0];isolated['env_steps']+=env.episode_env_steps-before[1]
            if done or info['env_steps_this_step']!=2:raise RuntimeError('Unexpected engineering environment boundary')
            if not bool(info['availability']) and np.any(next_image):raise RuntimeError('Hidden image entered missing-observation probe')
            replay.add(observation,action,state)
            result=auxiliary.update(image,action,next_image,float(reward),float(not info['terminated']),
                transition_id=isolated['agent_steps'],previous_available=available,next_available=bool(info['availability']))
            args._progress['auxiliary_update_attempts']+=1;args._progress['auxiliary_eligible_updates']+=int(result['updated'])
            controller.step(next_image,action,info,episode_id=base['episodes'])
            steps.append({'action':action.tolist(),'reward':float(reward),'info':jsonable(info),
                'image_sha256':tree_hash(next_image),'state_sha256':tree_hash(controller.state),
                'auxiliary_sha256':tree_hash(auxiliary.snapshot())})
            image=next_image;available=bool(info['availability'])
            observation={'image':image.copy(),'reward':np.float32(reward),'is_first':np.bool_(False),
                         'is_last':np.bool_(False),'is_terminal':np.bool_(info['terminated'])}
            target=(isolated['agent_steps']-5000)//4
            if (target>isolated['update_steps'])!=(t==3):raise RuntimeError('Next scheduled update must be at action four')
        batch=next(replay.stream);sample=sample_descriptor(batch)
        sampler_rng=tree_hash(replay.replay.sampler.rng.bit_generator.state)
        metrics=backbone.train(batch,replay.replay)
        args._progress['probe_updates']=1;isolated['update_steps']+=1
        if not all(np.isfinite(x).all() for x in metrics.values()):raise FloatingPointError('Nonfinite restored learner metrics')
        after=backbone.snapshot()
        if after['training']['counters']['updates']!=base['update_steps']+1 or after['writeback_updates']!=base['update_steps']+1:
            raise RuntimeError('Next actual learner/writeback update not committed exactly once')
        for field in ('inference_params','inference_version','final_auxiliary_heads','final_inference_bundle'):
            if tree_hash(after[field])!=tree_hash(restored[field]):raise RuntimeError('Episode-frozen current inference mutated')
        if tree_hash(prefill.bit_generator.state)!=tree_hash(payload['prefill_rng']):raise RuntimeError('Post-prefill path consumed prefill RNG')
        devices=[{'id':x.id,'platform':x.platform,'kind':x.device_kind} for x in backbone.jax.devices()]
        if not devices or any(x['platform']!='gpu' for x in devices):raise RuntimeError('Actual GPU execution not verified')
        comparison={'identity':identity,'checkpoint_format':record['format'],'source_counts':base,
            'source_checkpoint_sha256':args.checkpoint_sha256,'source_record_sha256':args.record_sha256,
            'tool_bundle_sha256':args.tool_bundle_sha256,'registration':copy.deepcopy(REGISTRATION),
            'restored_backbone':{k:tree_hash(restored[k]) for k in RESTORED_FIELDS},
            'restored_auxiliary':tree_hash(payload['auxiliary']),'restored_replay':tree_hash(replayed),
            'restored_private_rng':tree_hash(private),'restored_host_rng':host_before,
            'restored_prefill_rng':tree_hash(prefill.bit_generator.state),'parent_trajectory_cursor':parent_trace,
            'current_gate_artifact_sha256':None if gate is None else gate['artifact_sha256'],'boundary_audit':boundary,
            'reset':reset,'steps':steps,'sample':sample,'sampler_rng_after_sample':sampler_rng,
            'after_backbone':{k:tree_hash(after[k]) for k in RESTORED_FIELDS},'after_auxiliary':tree_hash(auxiliary.snapshot()),
            'after_replay':_replay_contents(replay),'after_collector':tree_hash(controller.snapshot()),
            'after_host_rng':{'numpy':tree_hash(np.random.get_state()),'python':tree_hash(random.getstate())},
            'metrics':metrics,'isolated_counts':isolated,'gpu_devices':devices}
    finally:
        args._progress.update(probe_actions=env.action_decisions,probe_dmc_calls=env.episode_env_steps)
        env.close()
        if trace is not None:trace.close()
    # Independently reload every original dependency again after the probe.
    load_inputs(args)
    result={'schema':'helixwm_final_bank_recovery_worker_v3','passed':True,'all_devices_gpu':True,
        'original_files_unchanged':True,'restored_comparisons':restored_checks,'storage':storage,
        'comparison':comparison,**args._progress}
    write_json(output/'result.json',result)
    return result


def compare_results(results,exitcodes):
    if len(results)!=2 or len(exitcodes)!=2:raise ValueError('Exactly two independent process results required')
    checks=[]
    for result,code in zip(results,exitcodes):
        restored=result.get('restored_comparisons',{})
        consistent=False
        try:
            comparison=result['comparison'];scope=comparison['checkpoint_format']
            checkpoint.validate_identity(comparison['identity'])
            if scope not in (checkpoint.FORMAT,checkpoint.PREFLIGHT_FORMAT):raise ValueError('Wrong recovery scope')
            start=checkpoint.validate_counts(comparison['source_counts'],preflight=scope==checkpoint.PREFLIGHT_FORMAT)
            expected=dict(start);expected['agent_steps']+=4;expected['env_steps']+=8;expected['update_steps']+=1
            if comparison['isolated_counts']!=expected or len(comparison['steps'])!=4:raise ValueError('Actual isolated counts differ')
            for index,step in enumerate(comparison['steps'],1):
                action=np.asarray(step['action']);info=step['info']
                if (action.shape!=(6,) or not np.isfinite(action).all() or np.any(np.abs(action)>1)
                        or not np.isfinite(step['reward']) or info['observation_timestamp']!=index
                        or bool(info['availability'])!=(index!=3) or info['env_steps_this_step']!=2):
                    raise ValueError('Actual executed probe action/outcome evidence differs')
            if not comparison['gpu_devices'] or any(x['platform']!='gpu' for x in comparison['gpu_devices']):raise ValueError('No actual GPU devices')
            consistent=True
        except (KeyError,ValueError,TypeError):pass
        checks.append(code==0 and result.get('schema')=='helixwm_final_bank_recovery_worker_v3'
            and all(result.get(x) is True for x in ('passed','all_devices_gpu','original_files_unchanged','actual_counts_complete'))
            and result.get('formal_training_actions')==0 and result.get('probe_actions')==4
            and result.get('probe_dmc_calls')==8 and result.get('probe_updates')==1
            and result.get('auxiliary_update_attempts')==4 and type(result.get('auxiliary_eligible_updates')) is int
            and 1<=result['auxiliary_eligible_updates']<=4 and set(restored)==set(RESTORED_FIELDS)
            and all(x is True for x in restored.values()) and consistent)
    left,right=[x.get('comparison',{}) for x in results]
    exact={}
    for key in sorted(COMPARISON_FIELDS):
        try:exact[key]=key in left and key in right and tree_hash(left[key])==tree_hash(right[key])
        except (ValueError,TypeError):exact[key]=False
    complete=set(left)==COMPARISON_FIELDS and set(right)==COMPARISON_FIELDS
    registration=left.get('registration')==REGISTRATION and right.get('registration')==REGISTRATION
    passed=all(checks) and complete and registration and all(exact.values())
    return {'schema':'helixwm_final_bank_full_recovery_gate_v3','passed':passed,'worker_checks':checks,
        'complete_schema':complete,'registered_probe':registration,'exact_comparisons':exact,'exitcodes':exitcodes,
        'mismatched_fields':[k for k,v in exact.items() if not v],
        'actual_probe_actions':sum(x.get('probe_actions',0) for x in results),
        'actual_probe_dmc_calls':sum(x.get('probe_dmc_calls',0) for x in results),
        'actual_probe_updates':sum(x.get('probe_updates',0) for x in results),'formal_training_actions':0,
        'claim':'same_hardware_two_fresh_processes_next_actions_native_sample_and_scheduled_update',
        'uninterrupted_process_reference':False,'automatic_formal_resume':False,'cross_hardware_or_long_horizon_identity':False,
        'source_checkpoint_format':left.get('checkpoint_format'),
        'formal_checkpoint_recovery_proven':bool(passed and left.get('checkpoint_format')==checkpoint.FORMAT),
        'integrated_preflight_checkpoint_recovery_proven':bool(passed and left.get('checkpoint_format')==checkpoint.PREFLIGHT_FORMAT),
        'tolerance':'exact_or_fail','registration':copy.deepcopy(REGISTRATION)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint','identity-json','protocol','runtime-source-root','upstream','output-new'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--dependency-root',type=Path)
    parser.add_argument('--preflight-checkpoint',action='store_true',help='Only the explicitly registered fresh 6000-action full preflight format')
    for name in ('checkpoint-sha256','record-sha256','identity-sha256','tool-bundle-sha256'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--mode',choices=('pair','reference','restored'),default='pair')
    parser.add_argument('--timeout-seconds',type=int,default=3600)
    args=parser.parse_args()
    if args.timeout_seconds<1:raise ValueError('Positive bounded worker timeout required')
    source=args.dependency_root or args.checkpoint.resolve().parent.parent
    output=isolated_output(source,args.output_new)
    if args.mode!='pair':
        try:run_worker(args)
        except BaseException as exc:
            output.mkdir(parents=True,exist_ok=True)
            if not (output/'result.json').exists():write_json(output/'result.json',{'passed':False,'error':repr(exc),
                'traceback':traceback.format_exc(),'original_inputs_opened_for_write':False,
                **getattr(args,'_progress',{'actual_counts_complete':False})})
            raise
        return
    load_inputs(args);output.mkdir(parents=True,exist_ok=False)
    results=[];codes=[]
    for mode in ('reference','restored'):
        command=[sys.executable,'-B','-m','helixwm_final.final_recovery_probe','--mode',mode]
        for name in ('checkpoint','identity_json','protocol','runtime_source_root','upstream'):
            command+=['--'+name.replace('_','-'),str(getattr(args,name).resolve())]
        for name in ('checkpoint_sha256','record_sha256','identity_sha256','tool_bundle_sha256'):
            command+=['--'+name.replace('_','-'),getattr(args,name)]
        if args.dependency_root:command+=['--dependency-root',str(args.dependency_root.resolve())]
        if args.preflight_checkpoint:command+=['--preflight-checkpoint']
        command+=['--output-new',str(output/mode)]
        write_json(output/(mode+'_command.json'),{'argv':command})
        with (output/(mode+'.log')).open('xb') as log:
            try:code=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,cwd=args.runtime_source_root,
                check=False,timeout=args.timeout_seconds).returncode
            except subprocess.TimeoutExpired:code=124
        codes.append(code);path=output/mode/'result.json'
        results.append(json.loads(path.read_bytes()) if path.exists() else {'passed':False,'error':'Worker produced no result'})
        if code or results[-1].get('passed') is not True:
            if len(results)==1:results.append({'passed':False,'error':'Not started after first failure'});codes.append(None)
            break
    report=compare_results(results,codes);write_json(output/'recovery_gate.json',report)
    print(json.dumps({'passed':report['passed'],'report':str(output/'recovery_gate.json')}),flush=True)
    if not report['passed']:raise SystemExit(1)


if __name__=='__main__':main()
