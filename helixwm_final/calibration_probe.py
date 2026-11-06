"""Frozen learned-model deployment measurements; no training or auto activation."""
from __future__ import annotations
import argparse,copy,hashlib,json,os,pickle,random,subprocess,time,traceback
from pathlib import Path
os.environ.setdefault('MUJOCO_GL','egl')
os.environ.setdefault('PYOPENGL_PLATFORM','egl')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
import numpy as np
from .backbone import FinalBackbone
from .environment import original_environment_contract
from .epistemic import FrozenBootstrap,tree_digest
from .provenance import source_digest
from .protocol_v2 import load_protocol, validate_measurement_parent_protocol
from . import deployment_collect as collect
from . import deployment_calibration as calibration
from .belief import canonical_json
from helixwm_matched.environment import MatchedWalkerEnv
from helixwm_matched.backbone import array_tree_hash
from helixwm_matched.env_probe import runtime_lock,first_difference
from helixwm_matched.train import sha,resources


def write(path,value):
    with Path(path).open('x',encoding='utf8') as f:json.dump(value,f,indent=2,allow_nan=False)


def verify_environment_runtime(config, expected, *, env_factory=None, lock_fn=None):
    """Instantiate and render once BEFORE collecting the shared runtime lock.

    EGL/NVIDIA libraries are loaded lazily by the first RGB render. A package
    import or JAX allocation alone does not establish the renderer identity.
    This isolated reset consumes no training/calibration actions or DMC calls.
    """
    factory=env_factory or (lambda:MatchedWalkerEnv(config))
    lock_fn=lock_fn or runtime_lock
    env=factory()
    try:
        frame=env.reset(seed=72099,mask=np.ones(501,np.bool_))
        if not isinstance(frame,np.ndarray) or frame.dtype!=np.uint8 or frame.shape!=(64,64,3):
            raise ValueError('Runtime gate did not produce native visible RGB64')
        if (env.action_decisions,env.episode_env_steps)!=(0,0):
            raise ValueError('Environment reset unexpectedly consumed actions/native calls')
        live=lock_fn(config)
        difference=first_difference(expected,live)
        if difference:raise RuntimeError('Measurement environment lock differs: '+difference)
        return {'passed':True,'live_environment_lock':live,'initial_rgb_sha256':tree_digest(frame),
                'initialization_seed':72099,'resets':1,'agent_steps':0,'env_steps':0,
                'included_in_training_budget':False}
    finally:
        env.close()


def capture_frozen_state(backbone,heads):
    """Hash overlapping complete parameter supersets plus the full snapshot.

    The optimizer/normalizer labels intentionally mean the complete official
    parameter tree, including all slots. They are stronger overlapping hashes,
    not claims that undocumented key prefixes identify only those components.
    """
    state=backbone.snapshot();training=state['training']
    complete_parameter_hash=array_tree_hash(training['params'])
    values={'model':backbone.inference_bundle_sha256(),'optimizer':complete_parameter_hash,
            'normalizers':complete_parameter_hash,'counters':tree_digest(training['counters']),
            'training_rng':tree_digest({'counter_keys':training['counters'],'python':random.getstate(),
                'numpy':np.random.get_state(),'carry_sha256':array_tree_hash(state['train_carry'])}),
            'epistemic_heads':tree_digest(heads.snapshot())}
    return values,array_tree_hash(state)


def validate_parent(payload,parent,metadata):
    expected={'agent_steps':5500,'env_steps':11000,'episodes':11,'update_steps':125}
    if (payload.get('format')!='helixwm_final_bootstrap_probe_v1' or payload.get('identity')!=parent
            or parent.get('stage')!='isolated_shared_native_bootstrap_preflight'
            or parent.get('fresh_initialization') is not True or parent.get('old_checkpoint_loaded') is not False
            or type(parent.get('formal_training_actions')) is not int or parent['formal_training_actions']!=0):
        raise ValueError('Expected isolated fresh bootstrap parent, not a historical training model')
    if payload.get('counts')!=expected or metadata.get('counts')!=expected:
        raise ValueError('Parent bootstrap incomplete')
    if (any(type(v) is not int for v in payload['counts'].values())
            or any(type(v) is not int for v in metadata['counts'].values())):
        raise ValueError('Parent action counts must be actual integers')
    binding={k:parent[k] for k in ('source_sha256','protocol_sha256')}
    if payload['heads']['binding']!=binding or payload['auxiliary']['binding']!=binding:
        raise ValueError('Parent prediction heads have different source/protocol lineage')


def acknowledged_counts(root, exception, fallback):
    """Read retained phase counters; do not report only the last failed cell."""
    root=Path(root)
    total_actions=total_calls=0;seen=False
    for phase in ('fit','validation'):
        failure=root/(phase+'.failure.json')
        if failure.is_file():
            record=json.loads(failure.read_text())
            total_actions+=record['acknowledged_actions'];total_calls+=record['acknowledged_dmc_calls'];seen=True
        else:
            for path in root.glob(phase+'_seed*_gap*.counts.json'):
                record=json.loads(path.read_text())
                total_actions+=record['actions'];total_calls+=record['dmc_calls'];seen=True
    if seen:return {'agent_steps':total_actions,'env_steps':total_calls,'update_steps':0}
    if hasattr(exception,'acknowledged_actions'):
        return {'agent_steps':exception.acknowledged_actions,'env_steps':exception.acknowledged_dmc_calls,'update_steps':0}
    return copy.deepcopy(fallback)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--checkpoint',type=Path,required=True)
    ap.add_argument('--checkpoint-sha256',required=True)
    ap.add_argument('--parent-identity',type=Path,required=True)
    ap.add_argument('--parent-identity-sha256',required=True)
    ap.add_argument('--checkpoint-metadata',type=Path,required=True)
    ap.add_argument('--checkpoint-metadata-sha256',required=True)
    ap.add_argument('--protocol',type=Path,required=True)
    ap.add_argument('--parent-protocol',type=Path,required=True)
    ap.add_argument('--parent-protocol-sha256',required=True)
    ap.add_argument('--run-dir',type=Path,required=True)
    ap.add_argument('--upstream',type=Path,required=True)
    ap.add_argument('--environment-lock',type=Path,required=True)
    ap.add_argument('--mode',choices=('window','suite'),required=True)
    ap.add_argument('--trajectory-cost',action='store_true',
        help='Retain full actual/prospective arrays for one fit window; capacity evidence only')
    args=ap.parse_args()
    if args.trajectory_cost and args.mode!='window':
        raise ValueError('Full-array cost probe is a separate one-window engineering measurement')
    if (sha(args.checkpoint)!=args.checkpoint_sha256 or sha(args.parent_identity)!=args.parent_identity_sha256
            or sha(args.checkpoint_metadata)!=args.checkpoint_metadata_sha256
            or sha(args.parent_protocol)!=args.parent_protocol_sha256):
        raise ValueError('Trusted parent evidence identity mismatch; refusing deserialization')
    parent=json.loads(args.parent_identity.read_text())
    protocol_migration=validate_measurement_parent_protocol(load_protocol(args.protocol),
        json.loads(args.parent_protocol.read_text()),trusted_parent_sha256=parent['protocol_sha256'])
    metadata=json.loads(args.checkpoint_metadata.read_text())
    if (metadata.get('sha256')!=args.checkpoint_sha256 or metadata.get('bytes')!=args.checkpoint.stat().st_size
            or metadata.get('path')!=args.checkpoint.name):
        raise ValueError('External checkpoint metadata differs from retained parent bytes')
    if sha(args.environment_lock)!=parent.get('environment_lock_sha256'):
        raise ValueError('Measurement must use the exact original matched environment lock')
    root=args.run_dir.resolve();root.mkdir(parents=True,exist_ok=False)
    source,files=source_digest();write(root/'source_manifest.json',files)
    write(root/'command.json',{'argv':__import__('sys').argv})
    (root/'probe.pid').write_text(str(os.getpid())+'\n')
    start=time.perf_counter();counts={'agent_steps':0,'env_steps':0,'update_steps':0}
    try:
        with args.checkpoint.open('rb') as f:payload=pickle.load(f)
        validate_parent(payload,parent,metadata)
        cfg=original_environment_contract()
        environment_gate=verify_environment_runtime(cfg,json.loads(args.environment_lock.read_text()))
        write(root/'environment_gate.json',environment_gate)
        bcfg={'seed':0,'size':'size12m','platform':'cuda','compute_dtype':'bfloat16','batch_size':16,'batch_length':64}
        backbone=FinalBackbone(bcfg,args.upstream,root)
        backbone.load(payload['backbone']);heads=FrozenBootstrap(payload['heads'])
        if not heads.ready:raise ValueError('Independent heads have no visible-pair training')
        if (backbone.inference_bundle_sha256()!=metadata['model_sha256'] or heads.sha256!=metadata['epistemic_heads_sha256']):
            raise ValueError('Strictly loaded inference model/heads differ from retained parent metadata')
        np.random.set_state(payload['numpy_rng']);random.setstate(payload['python_rng'])
        context={'source_sha256':source,'model_sha256':backbone.inference_bundle_sha256(),
            'bank_manifest_sha256':collect.scenario_bank_sha256(),
            'protocol_sha256':protocol_migration['measurement_protocol_sha256'],
            'collector_sha256':collect.collector_sha256(),'epistemic_heads_sha256':heads.sha256,
            'model_version':backbone.inference_version,'measurement_kind':'actual_gpu_closed_loop'}
        # Probe implementation is a separately identified measurement runtime;
        # this is NOT a training resume or a rewrite of the parent's identity.
        write(root/'lineage.json',{'parent_checkpoint_sha256':args.checkpoint_sha256,
            'parent_identity':parent,'parent_identity_sha256':args.parent_identity_sha256,
            'parent_checkpoint_metadata_sha256':args.checkpoint_metadata_sha256,
            'measurement_context':context,'training_resume':False,
            'protocol_migration':protocol_migration,'parent_protocol_file_sha256':args.parent_protocol_sha256,
            'heads_training_source_sha256':payload['heads']['binding']['source_sha256']})
        write(root/'scenario_bank.json',collect.scenario_bank())
        def frozen_hashes():
            return capture_frozen_state(backbone,heads)[0]
        write(root/'freeze_hash_scope.json',{'optimizer_and_normalizers':'both hash complete author training parameter tree including optimizer/normalizer slots; overlapping supersets',
            'training_rng':'official counter-key schedule plus host Python/NumPy global RNG and train carry'})
        before,whole_before=capture_frozen_state(backbone,heads);resources(root,counts)
        if args.mode=='window':
            trajectory=None
            if args.trajectory_cost:
                from .trajectory import TrajectoryArchive
                trajectory=TrajectoryArchive.create(root/'TRAJECTORIES',
                    source_sha256=source,protocol_sha256=context['protocol_sha256'])
            try:
                row,trace,audit=collect.collect_window(backbone,heads,context,
                    env_factory=lambda:MatchedWalkerEnv(cfg),seed=55000,gap=100,phase='fit',
                    anchor_sha256=args.checkpoint_sha256,trajectory=trajectory)
                cursor=None if trajectory is None else trajectory.snapshot()
            finally:
                if trajectory is not None:trajectory.close()
            if args.trajectory_cost:
                restored=TrajectoryArchive.restore(root/'TRAJECTORIES',cursor,
                    expected_source_sha256=source,expected_protocol_sha256=context['protocol_sha256'])
                try:
                    if restored.snapshot()!=cursor:raise RuntimeError('Full-array cursor replay differs')
                finally:restored.close()
                costs={}
                for meta_path in (root/'TRAJECTORIES/records').rglob('*.json'):
                    meta=json.loads(meta_path.read_bytes());kind=meta['payload']['kind']
                    item=costs.setdefault(kind,{'records':0,'uncompressed_array_bytes':0,'compressed_NPZ_bytes':0,'metadata_bytes':0})
                    with np.load(meta_path.with_suffix('.npz'),allow_pickle=False) as arrays:
                        item['uncompressed_array_bytes']+=sum(arrays[k].nbytes for k in arrays.files)
                    item['records']+=1;item['compressed_NPZ_bytes']+=meta_path.with_suffix('.npz').stat().st_size
                    item['metadata_bytes']+=meta_path.stat().st_size
                write(root/'trajectory_cost.json',{'measurement':'one_learned_gap100_fit_window_only',
                    'actions':audit['actions'],'record_counts_and_bytes_by_kind':costs,'verified_cursor':cursor,
                    'all_files_bytes':sum(p.stat().st_size for p in (root/'TRAJECTORIES').rglob('*') if p.is_file()),
                    'includes_prospective_diagnostic_rollouts':True,'compression_is_not_capacity_guarantee':True,
                    'formal_training':False,'capacity_gate_passed':False})
            calibration.write_measurement_window(row,root)
            with (root/'raw_trace.json').open('xb') as stream:stream.write(canonical_json(trace))
            if sha(root/'raw_trace.json')!=row['trace_sha256']:raise RuntimeError('Raw trace byte commitment differs')
            write(root/'window_counts.json',audit)
            counts={'agent_steps':audit['actions'],'env_steps':audit['dmc_calls'],'update_steps':0}
            result={'scope':'one_actual_learned_gpu_gap100_fit_window_only',
                'production_active':False,'full_deployment_calibration_passed':False}
        else:
            anchor=root/'retained_bootstrap_anchor.pkl'
            # Immutable hard link; no mutation or duplicate complete replay.
            os.link(args.checkpoint,anchor)
            gpu=subprocess.run(['nvidia-smi','--query-gpu=name','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout.strip()
            result=collect.collect_sequence(backbone,heads,context,env_factory=lambda:MatchedWalkerEnv(cfg),
                directory=root,anchor_sha256=args.checkpoint_sha256,anchor_descriptor=calibration.file_descriptor(anchor,root),
                frozen_state_hashes=frozen_hashes,device='cuda',gpu=gpu)
            counts={'agent_steps':result['counts']['actions'],'env_steps':result['counts']['dmc_calls'],'update_steps':0}
        after,whole_after=capture_frozen_state(backbone,heads)
        if before!=after or whole_before!=whole_after:raise RuntimeError('Frozen complete backbone/model/optimizer/normalizer/counter/host-RNG changed')
        if sha(args.checkpoint)!=args.checkpoint_sha256:raise RuntimeError('Parent checkpoint changed')
        if sha(args.parent_identity)!=args.parent_identity_sha256 or sha(args.checkpoint_metadata)!=args.checkpoint_metadata_sha256:
            raise RuntimeError('Parent metadata changed during measurement')
        if sha(args.parent_protocol)!=args.parent_protocol_sha256:
            raise RuntimeError('Retained parent protocol bytes changed during measurement')
        if source_digest()!=(source,files):raise RuntimeError('Measurement source changed during execution')
        difference=first_difference(environment_gate['live_environment_lock'],runtime_lock(cfg))
        if difference:raise RuntimeError('Measurement runtime changed during execution: '+difference)
        resources(root,counts)
        write(root/'probe_result.json',{'engineering_execution_passed':True,'counts':counts,
            'seconds':time.perf_counter()-start,'freeze_before':before,'freeze_after':after,'result':result,
            'full_backbone_snapshot_sha256_before':whole_before,'full_backbone_snapshot_sha256_after':whole_after,
            'measurement_source_sha256':source,
            'formal_training_started':False,'final_acceptance':False})
        (root/'probe.exit').write_text('0\n')
    except BaseException as exc:
        counts=acknowledged_counts(root,exc,counts)
        write(root/'failure.json',{'error':repr(exc),'traceback':traceback.format_exc(),'counts':counts})
        (root/'probe.exit').write_text('1\n');raise

if __name__=='__main__':main()
