"""Frozen D/Final evaluation; new v3 inputs only, no learner or auto retry.

The registered core and gap100 stress are separate strata. Frames for the one
prespecified video are evaluator diagnostics and never enter model inputs.
"""
from __future__ import annotations
import argparse,copy,csv,hashlib,json,math,os,random,time,traceback
from pathlib import Path
import numpy as np
from . import protocol_v3 as protocol
from .epistemic import tree_digest,FrozenBootstrap
from .provenance import source_digest
from helixwm_matched.final_eval_validation import reward_sum_audit
from helixwm_matched.final_eval import FrameCollector
from helixwm_matched.protocol import atomic_json,fixed_gap_mask

VIDEO_PANELS=(('D','black_frame_native'),('F','helix_final'),('D','prior_only'))
VIDEO_SEED,VIDEO_GAP,VIDEO_FPS=60000,20,20

def sha_file(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda:f.read(1024*1024),b''):h.update(data)
    return h.hexdigest()

def write(path,value):
    data=json.dumps(value,sort_keys=True,indent=2,allow_nan=False).encode()
    with Path(path).open('xb') as f:f.write(data)

def case_id(arm,controller,seed,gap):return f'{arm}_{controller}_seed{seed}_gap{gap:02d}'

def build_plan(config):
    protocol.validate_protocol(config)
    plan=protocol.build_eval_plan(config,'core')+protocol.build_eval_plan(config,'long_gap100_stress')
    if len(plan)!=1520 or len({x['case_id'] for x in plan})!=1520:
        raise ValueError('Final matrix is not the registered 1280 core plus 240 stress cases')
    if sum(x['record_video_frames'] for x in plan)!=3:
        raise ValueError('Exactly the three registered panel cases must supply one MP4')
    return plan

def validate_inputs(inputs,config,*,runtime_source_sha256,environment_lock_sha256,output_new):
    """Bind manifest keys to independent final lineages before deserialization."""
    from .train import validate_identity,_sha_value
    protocol.validate_protocol(config)
    if type(inputs) is not dict or set(inputs)!={'D','F'}:
        raise ValueError('Both independently trained final arms are required')
    output=Path(output_new).resolve();run_roots=[]
    for arm,item in inputs.items():
        if type(item) is not dict or set(item)!={'path','checkpoint_sha256','record_sha256','identity'}:
            raise ValueError('Final input requires exact path/checkpoint/sidecar/identity fields')
        validate_identity(item['identity'])
        if item['identity']['arm']!=arm:raise ValueError('Manifest arm key differs from checkpoint lineage')
        if not _sha_value(item['checkpoint_sha256']) or not _sha_value(item['record_sha256']):
            raise ValueError('Externally audited checkpoint and sidecar digests are required')
        if type(item['path']) is not str or not Path(item['path']).is_absolute():
            raise ValueError('Final checkpoint must have an explicit absolute path')
        path=Path(item['path']).resolve();run_root=path.parent.parent;run_roots.append(run_root)
        if output==run_root or output.is_relative_to(run_root) or run_root.is_relative_to(output):
            raise ValueError('Final output must be isolated from each entire input run')
    if inputs['D']['identity']['run_id']==inputs['F']['identity']['run_id']:
        raise ValueError('Independent D/F run lineage must differ')
    if (run_roots[0]==run_roots[1] or run_roots[0].is_relative_to(run_roots[1])
            or run_roots[1].is_relative_to(run_roots[0])
            or inputs['D']['checkpoint_sha256']==inputs['F']['checkpoint_sha256']
            or inputs['D']['record_sha256']==inputs['F']['record_sha256']):
        raise ValueError('D/F checkpoints and run roots must be independent')
    common=('source_sha256','protocol_sha256','environment_lock_sha256','upstream_commit','bank_manifest_sha256')
    if any(inputs['D']['identity'][k]!=inputs['F']['identity'][k] for k in common):
        raise ValueError('D/F source/environment/budget/bank identities differ')
    identity=inputs['D']['identity']
    if identity['source_sha256']!=runtime_source_sha256 or identity['protocol_sha256']!=protocol.protocol_sha256(config):
        raise ValueError('Frozen runtime is not the trained source and registered protocol')
    if identity['environment_lock_sha256']!=environment_lock_sha256:
        raise ValueError('Final environment identity differs')
    return copy.deepcopy(inputs)


def load_final_input(item,config):
    from .final_checkpoint import load_checkpoint
    return load_checkpoint(item['path'],expected_sha256=item['checkpoint_sha256'],
        expected_record_sha256=item['record_sha256'],expected_identity=item['identity'],
        protocol_config=config,require_final=True)


def strict_reload_backbone(backbone,saved,*,output_dir):
    """Only the explicitly isolated log directory may differ after loading."""
    from helixwm_matched.backbone import array_tree_hash
    backbone.load(saved);actual=backbone.snapshot();expected=copy.deepcopy(saved)
    for value in (expected,actual):
        config=value.get('resolved_config')
        if type(config) is not dict or type(config.get('logdir')) is not str or not Path(config['logdir']).is_absolute():
            raise ValueError('Strict reload requires the original complete resolved configuration')
    if Path(actual['resolved_config']['logdir']).resolve()!=Path(output_dir).resolve():
        raise ValueError('Reload log directory is not the isolated evaluator model directory')
    old_logdir=expected['resolved_config']['logdir'];new_logdir=actual['resolved_config']['logdir']
    expected['resolved_config']['logdir']=new_logdir
    if array_tree_hash(actual)!=array_tree_hash(expected):
        raise ValueError('Strict final backbone reload differs beyond declared logdir relocation')
    return {'exact_state_and_config_except_logdir':True,'original_logdir':old_logdir,
            'evaluation_logdir':new_logdir,'normalized_state_sha256':array_tree_hash(expected)}


def make_case_controller(backbone,heads,gate,planned,*,output_dir,source_sha256):
    from .full_controller import FullController
    from .trajectory import TrajectoryArchive
    from helixwm_matched.controllers import Controller
    if planned['controller']!='helix_final':
        return Controller(backbone,planned['controller'],seed=planned['seed']),None
    if gate is None:raise ValueError('Final controller requires a verified current bank artifact')
    context=gate['context']
    if context['source_sha256']!=source_sha256:raise ValueError('Case bank context differs from frozen runtime')
    trajectory=TrajectoryArchive.create(Path(output_dir)/'TRAJECTORIES',source_sha256=source_sha256,
        protocol_sha256=context['protocol_sha256'])
    try:
        controller=FullController(backbone,heads,context,seed=planned['seed'],bank_gate=gate,
            execution='bank_production',trajectory=trajectory)
    except BaseException:
        try:trajectory.close()
        except BaseException:pass
        raise
    return controller,trajectory


def record_failure(root,error,counts,*,active=None,frames=None,trajectory=None):
    """Diagnostic saving must never replace the primary failure."""
    secondary=list(getattr(error,'evaluation_cleanup_errors',[]))
    def attempt(label,fn):
        try:fn()
        except BaseException as failure:secondary.append({'operation':label,'error':repr(failure)})
    if trajectory is not None:attempt('close_trajectory',trajectory.close)
    if frames is not None and active is not None:
        def partial_video():
            if frames.frames and not (active/'video_frames.npz').exists():frames.save(active,complete=False)
        attempt('save_partial_video',partial_video)
    def exit_file():
        with (root/'evaluation.exit').open('x') as stream:stream.write('1\n')
    attempt('write_failure_exit',exit_file)
    report={'error':repr(error),'traceback':''.join(traceback.format_exception(type(error),error,error.__traceback__)),
        'case':None if active is None else active.name,'partial':getattr(error,'evaluation_partial',None),
        **counts,'automatic_retry':False,'cleanup_errors':secondary}
    attempt('write_failure_report',lambda:write(root/'failure.json',report))
    return report


def _frame(image,available):
    if not isinstance(image,np.ndarray) or image.shape!=(64,64,3) or image.dtype!=np.uint8:
        raise ValueError('Evaluation observation is not native RGB64')
    if not available and np.any(image):raise ValueError('Missing policy input contains hidden pixels')
    return image

def evaluate_case(controller,planned,env,*,render_callback=None):
    """One already prescribed case; partial errors retain true interaction counts."""
    from .train import _module_selection, _module_observation, _finite_evidence
    actions=[];rewards=[];available=[];events=[];counts={'actions':0,'dmc_calls':0};primary_error=None
    selection=None
    try:
        mask=np.ones(501,np.bool_) if planned['gap']==0 else fixed_gap_mask(planned['gap'])
        if protocol.mask_sha256(mask)!=planned['availability_sha256']:
            raise ValueError('Planned mask commitment differs')
        image=_frame(env.reset(seed=planned['seed'],mask=mask),True)
        if (env.action_decisions,env.episode_env_steps)!=(0,0):raise ValueError('Reset consumed actions')
        controller.reset(image,episode_id=planned['seed'])
        if render_callback is not None:render_callback(env.render_diagnostic(256),0,{'availability':True})
        for timestamp in range(1,501):
            # Native Controller.act() and FullController.act() share this
            # no-argument stochastic interface. Do not pass a Final-only kwarg
            # to the native adapter or consume another sample for diagnostics.
            action=np.asarray(controller.act(),np.float32).reshape(6)
            if not np.isfinite(action).all() or np.any(np.abs(action)>1):raise ValueError('Invalid policy action')
            selection=_module_selection(controller,timestamp=timestamp,
                available=bool(mask[timestamp-1]),external=False)
            a0,c0=env.action_decisions,env.episode_env_steps
            try:image,reward,done,info=env.step(action)
            finally:
                counts['actions']+=env.action_decisions-a0;counts['dmc_calls']+=env.episode_env_steps-c0
            if (env.action_decisions-a0!=1 or env.episode_env_steps-c0!=2
                or env.action_decisions!=timestamp or env.episode_env_steps!=2*timestamp
                or type(done) is not bool or done!=(timestamp==500)
                or type(info.get('availability')) is not bool or info['availability']!=bool(mask[timestamp])
                or type(info.get('observation_timestamp')) is not int or info['observation_timestamp']!=timestamp
                or type(info.get('env_steps_this_step')) is not int or info['env_steps_this_step']!=2
                or not np.array_equal(info.get('executed_action'),action)):
                raise ValueError('Evaluation actual action/mask/time/native-call contract differs')
            if (isinstance(reward,(bool,np.bool_)) or not isinstance(reward,(float,np.floating))
                or not math.isfinite(float(reward)) or not 0<=float(reward)<=2):
                raise ValueError('Evaluation reward nonfinite or outside native units')
            _frame(image,info['availability'])
            controller.step(image,action,info,episode_id=planned['seed'])
            actions.append(action.copy());rewards.append(float(reward));available.append(info['availability'])
            # Selection uses o_t; fusion below assimilates o_{t+1}. Preserve the
            # complete issued decision, including attempted/rejected planning,
            # and never assign the latter fusion to the preceding action.
            observation=_module_observation(controller,timestamp=timestamp,available=info['availability'])
            decision=selection['decision']
            events.append({'timestamp':timestamp,'kind':decision['kind'],
                'calibration_status':decision.get('calibration_status'),
                'metrics':_finite_evidence(controller.last_metrics) if hasattr(controller,'last_metrics') else None,
                'module_evidence':{'schema':'helixwm_final_action_module_evidence_v1','arm':planned['arm'],
                    'action_selection':selection,'observation_update':observation,'interaction_complete':True}})
            selection=None
            if render_callback is not None:
                render_callback(env.render_diagnostic(256),timestamp,{'availability':info['availability']})
        if counts!={'actions':500,'dmc_calls':1000}:raise ValueError('Final case budget differs')
        trace={'action':np.asarray(actions,np.float32),'reward':np.asarray(rewards,np.float64),
            'dmc_calls':np.full(500,2,np.int32),'availability_outcome':np.asarray(available,np.bool_),
            'availability_before_action':mask[:-1].copy()}
        total=math.fsum(rewards)
        reduction=reward_sum_audit(trace['reward'],total)
        return {'episode_return':total,'trace':trace,'counts':counts,'events':events,'return_validation':reduction}
    except BaseException as error:
        primary_error=error
        error.evaluation_partial={'counts':counts,'recorded_reward_prefix':rewards,
            'recorded_action_prefix':[a.tolist() for a in actions],'automatic_retry':False,
            'module_events_prefix':events,'pending_action_selection':selection}
        raise
    finally:
        try:env.close()
        except BaseException as cleanup:
            if primary_error is not None:
                primary_error.evaluation_cleanup_errors=[{'operation':'close_environment','error':repr(cleanup)}]
            else:
                cleanup.evaluation_partial={'counts':counts,'recorded_reward_prefix':rewards,
                    'recorded_action_prefix':[a.tolist() for a in actions],'automatic_retry':False,
                    'module_events_prefix':events,'pending_action_selection':selection}
                raise

def make_video(root, panel_records):
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw, ImageFont
    root = Path(root); out = root/'VIDEOS'; out.mkdir(exist_ok=False)
    arrays, sources = [], []
    for arm, controller in VIDEO_PANELS:
        cid = case_id(arm, controller, VIDEO_SEED, VIDEO_GAP)
        record = panel_records[cid]
        path = root/'CASES'/cid/record['file']
        if sha_file(path) != record['sha256']: raise ValueError('Video source frames changed')
        with np.load(path, allow_pickle=False) as data:
            frames, available, times = data['frames'], data['availability'], data['timestamp']
        if frames.shape != (501, 256, 256, 3) or frames.dtype != np.uint8 or not np.array_equal(times, np.arange(501)):
            raise ValueError('Formal panel frame contract differs')
        if not np.array_equal(available, fixed_gap_mask(20)):
            raise ValueError('Formal panel availability differs')
        arrays.append((frames, available))
        sources.append({'case_id': cid, 'source_frames_sha256': record['sha256']})
    labels = ['D: black-frame', 'F: Helix Final', 'D: prior-only']
    try: font = ImageFont.truetype('DejaVuSans.ttf', 14)
    except OSError: font = ImageFont.load_default()
    partial = out/'final_walker_comparison.partial.mp4'
    writer = imageio.get_writer(str(partial), fps=VIDEO_FPS, codec='libx264',
                                macro_block_size=16, ffmpeg_log_level='error')
    try:
        for t in range(501):
            canvas = Image.new('RGB', (768, 304), (18, 22, 29))
            draw = ImageDraw.Draw(canvas)
            for p, (frames, available) in enumerate(arrays):
                x = p*256; canvas.paste(Image.fromarray(frames[t]), (x, 48))
                draw.text((x+6, 4), labels[p]+' | seed 60000', fill='white', font=font)
                draw.text((x+6, 24), f't={t:03d}/500 | '+('OBSERVED' if available[t] else 'MISSING'),
                          fill=(140, 240, 170) if available[t] else (255, 170, 100), font=font)
            writer.append_data(np.asarray(canvas))
    finally: writer.close()
    reader = imageio.get_reader(str(partial)); decoded = 0
    try:
        metadata = reader.get_meta_data()
        for frame in reader:
            if frame.shape != (304, 768, 3) or frame.dtype != np.uint8:
                raise ValueError('Decoded formal MP4 shape/dtype mismatch')
            decoded += 1
    finally: reader.close()
    if decoded != 501 or abs(float(metadata['fps'])-VIDEO_FPS) > 1e-8:
        raise ValueError('Formal MP4 did not fully decode at its declared playback rate')
    target = out/'final_walker_comparison.mp4'; partial.rename(target)
    if len(list(root.rglob('*.mp4'))) != 1: raise ValueError('Expected exactly one formal MP4')
    return {'file': target.relative_to(root).as_posix(), 'sha256': sha_file(target),
            'bytes': target.stat().st_size, 'full_decode_pass': True, 'frames': decoded,
            'dimensions': [768, 304], 'fps': VIDEO_FPS, 'fps_semantics': 'presentation_playback_rate',
            'sources': sources, 'extra_evaluation_episodes': 0,
            'synchronized_by': 'action_timestamp_not_identical_trajectory',
            'real_scene_not_policy_input': True, 'availability_labels': ['OBSERVED', 'MISSING']}


def run(args):
    os.environ.setdefault('MUJOCO_GL','egl');os.environ.setdefault('PYOPENGL_PLATFORM','egl')
    os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
    from .backbone import FinalBackbone
    from .calibration_probe import capture_frozen_state,verify_environment_runtime
    from .environment import original_environment_contract
    from helixwm_matched.environment import MatchedWalkerEnv
    cfg=protocol.load_protocol(args.protocol);plan=build_plan(cfg)
    if sha_file(args.input_manifest)!=args.input_manifest_sha256:
        raise ValueError('Final input manifest differs from externally audited SHA')
    raw_inputs=json.loads(Path(args.input_manifest).read_bytes())
    source,files=source_digest();root=Path(args.output_new).resolve()
    inputs=validate_inputs(raw_inputs,cfg,runtime_source_sha256=source,
        environment_lock_sha256=sha_file(args.environment_lock),output_new=root)
    root.mkdir(parents=True,exist_ok=False);(root/'CASES').mkdir()
    (root/'evaluation.pid').write_text(str(os.getpid())+'\n')
    counts={'completed_cases':0,'actions':0,'dmc_calls':0};rows=[];panels={};started=time.perf_counter()
    active=None;frames=None;trajectory=None
    try:
        write(root/'input_manifest.json',inputs);write(root/'protocol.json',cfg);write(root/'plan.json',plan)
        write(root/'source_manifest.json',{'source_sha256':source,'files':files})
        expected=json.loads(Path(args.environment_lock).read_bytes())
        write(root/'environment_gate.json',verify_environment_runtime(original_environment_contract(),expected))
        for arm in ('D','F'):
            item=inputs[arm]
            p,_,_=load_final_input(item,cfg)
            del p
        backbone=FinalBackbone({'seed':0,'platform':'cuda','size':'size12m','compute_dtype':'bfloat16',
            'batch_size':16,'batch_length':64},args.upstream,root/'MODEL')
        for arm in ('D','F'):
            item=inputs[arm]
            payload,record,gate=load_final_input(item,cfg)
            reload_audit=strict_reload_backbone(backbone,payload['backbone'],output_dir=root/'MODEL')
            write(root/f'strict_reload_{arm}.json',reload_audit)
            heads=FrozenBootstrap(payload['pinned_heads'])
            np.random.set_state(payload['numpy_rng']);random.setstate(payload['python_rng'])
            before=capture_frozen_state(backbone,heads)
            for planned in (x for x in plan if x['arm']==arm):
                active=root/'CASES'/planned['case_id'];active.mkdir(exist_ok=False)
                write(active/'planned.json',planned)
                atomic_json(root/'status.json',{'phase':'frozen_evaluation','case':planned,**counts})
                frames=FrameCollector() if planned['record_video_frames'] else None
                controller,trajectory=make_case_controller(backbone,heads,gate,planned,
                    output_dir=active,source_sha256=source)
                outcome=evaluate_case(controller,planned,MatchedWalkerEnv(original_environment_contract(),diagnostics=frames is not None),
                    render_callback=frames)
                counts['actions']+=outcome['counts']['actions'];counts['dmc_calls']+=outcome['counts']['dmc_calls']
                with (active/'trace.npz').open('xb') as f:np.savez_compressed(f,**outcome.pop('trace'))
                write(active/'events.json',outcome.pop('events'))
                after=capture_frozen_state(backbone,heads)
                if before!=after:raise RuntimeError('Evaluation changed model/optimizer/normalizers/counters/host RNG/heads')
                if trajectory is not None:
                    write(active/'trajectory_cursor.json',trajectory.snapshot());trajectory.close();trajectory=None
                frame_record=frames.save(active,complete=True) if frames is not None else None
                if frame_record is not None:panels[planned['case_id']]=frame_record
                row={**planned,'episode_return':outcome['episode_return'],'checkpoint_sha256':item['checkpoint_sha256'],
                    'trace_sha256':sha_file(active/'trace.npz'),'gate_active':bool(gate['active']) if planned['controller']=='helix_final' else None,
                    'scope':'single_training_seed_frozen_inference','frozen':True,'training_updates':0}
                write(active/'summary.json',{**row,'return_validation':outcome['return_validation'],'freeze_before':before,'freeze_after':after})
                with (root/'episodes.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
                rows.append(row);counts['completed_cases']+=1;active=frames=None
            if sha_file(item['path'])!=item['checkpoint_sha256']:raise RuntimeError('Final checkpoint changed')
            write(root/f'frozen_{arm}.json',{'before':before,'after':capture_frozen_state(backbone,heads),'passed':True})
            del payload
        if counts!={'completed_cases':1520,'actions':760000,'dmc_calls':1520000}:raise RuntimeError('Final matrix incomplete')
        artifact_before=capture_frozen_state(backbone,heads)
        video=make_video(root,panels)
        if capture_frozen_state(backbone,heads)!=artifact_before:
            raise RuntimeError('Video artifact generation changed frozen state')
        from helixwm_matched.env_probe import runtime_lock,first_difference
        mismatch=first_difference(expected,runtime_lock(original_environment_contract()))
        if mismatch:raise RuntimeError('Final environment runtime changed: '+mismatch)
        with (root/'episodes.csv').open('x',newline='',encoding='utf8') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        if source_digest()!=(source,files):raise RuntimeError('Frozen source changed')
        write(root/'completion.json',{'passed':True,**counts,'core_cases':1280,'gap100_stress_cases':240,
            'video':video,'training_updates':0,'seconds':time.perf_counter()-started,
            'statistical_analysis_complete':False,'local_full_video_decode_and_visual_inspection_required':True,
            'scientific_superiority_established':False,'final_package_acceptance':False})
        (root/'evaluation.exit').write_text('0\n')
    except BaseException as exc:
        partial=getattr(exc,'evaluation_partial',None)
        if partial:
            counts['actions']+=partial['counts']['actions'];counts['dmc_calls']+=partial['counts']['dmc_calls']
        record_failure(root,exc,counts,active=active,frames=frames,trajectory=trajectory)
        raise

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol',type=Path,required=True);p.add_argument('--upstream',type=Path,required=True)
    p.add_argument('--input-manifest',type=Path,required=True);p.add_argument('--input-manifest-sha256',required=True)
    p.add_argument('--environment-lock',type=Path,required=True);p.add_argument('--output-new',type=Path,required=True)
    run(p.parse_args())

if __name__=='__main__':main()
