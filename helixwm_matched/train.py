"""Exactly counted, sequentially deployed matched arms. No automatic recovery."""
from __future__ import annotations
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import pickle
import random
import shutil
import subprocess
import time
import traceback
os.environ.setdefault('MUJOCO_GL','egl')
os.environ.setdefault('PYOPENGL_PLATFORM','egl')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
import numpy as np
from .protocol import load_protocol, protocol_sha256, make_mask_bank, mask_sha256, training_seed, atomic_json
from .environment import MatchedWalkerEnv
from .controllers import Controller
from .backbone import MatchedDreamerBackbone, NativeReplay, array_tree_hash
from .evaluation import periodic_evaluation
from .env_probe import runtime_lock, first_difference

def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def append(path,row):
    with Path(path).open('a',encoding='utf-8',newline='\n') as f:
        f.write(json.dumps(row,allow_nan=False)+'\n'); f.flush()

def resources(root,counters):
    row={'utc':now(),**counters,'disk_free_bytes':shutil.disk_usage(root).free}
    for k in ['memory.current','memory.max','memory.events']:
        p=Path('/sys/fs/cgroup')/k
        if p.exists():row[k]=p.read_text().strip()
    result=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,memory.total','--format=csv,noheader,nounits'],capture_output=True,text=True)
    row['gpu']=result.stdout.strip();row['gpu_query_exit']=result.returncode
    append(root/'resources.jsonl',row)
    return row

def source_digest():
    root=Path(__file__).resolve().parent.parent
    # Bind the complete training/evaluation runtime. Independent gate tools do
    # not execute in training and have their own source manifests.
    files=[root/'helixwm_matched'/name for name in
           ('__init__.py','backbone.py','controllers.py','environment.py',
            'evaluation.py','protocol.py','env_probe.py','train.py')]+[root/'helixwm_v2/dual_chain.py']
    records={str(p.relative_to(root)):sha(p) for p in sorted(files)}
    return hashlib.sha256(json.dumps(records,sort_keys=True).encode()).hexdigest(),records

def checkpoint(root,backbone,replay,controller,prefill_rng,counters,identity):
    """A complete episode-boundary snapshot with versioned replay dependencies."""
    if counters['agent_steps']%500: raise ValueError('Checkpoint must be at episode boundary')
    # Pin next episode parameters; completed controller is audit history only.
    finished=controller.snapshot()
    backbone.pin_inference()
    replay_state=replay.snapshot()
    payload={'format':'helixwm_matched_episode_boundary_v1','identity':identity,
             'backbone':backbone.snapshot(),'replay':replay_state,
             'completed_controller':finished,'prefill_rng':prefill_rng.bit_generator.state,
             'numpy_global_rng':np.random.get_state(),'python_rng':random.getstate(),
             'counters':dict(counters),'next_operation':'reset_new_episode_and_reanchor'}
    folder=root/'CHECKPOINTS';folder.mkdir(exist_ok=True)
    target=folder/f"checkpoint_{counters['agent_steps']:09d}.pkl"
    if target.exists():raise FileExistsError(target)
    tmp=target.with_suffix('.pkl.part')
    with tmp.open('xb') as f:
        pickle.dump(payload,f,protocol=5);f.flush();os.fsync(f.fileno())
    tmp.rename(target)
    record={'file':str(target.relative_to(root)),'bytes':target.stat().st_size,
            'sha256':sha(target),'utc':now(),**counters,'identity':identity,
            'replay_files':NativeReplay.referenced_files(replay_state)}
    atomic_json(target.with_suffix('.json'),record)
    atomic_json(root/'latest_checkpoint.json',record)
    return record

def host_fingerprint(controller,prefill_rng,replay,counts):
    state=(controller.snapshot(),prefill_rng.bit_generator.state,np.random.get_state(),
           random.getstate(),counts,replay.replay.sampler.rng.bit_generator.state,
           replay.stream.index)
    return hashlib.sha256(pickle.dumps(state,protocol=5)).hexdigest()

def await_verified_archive(root,record,counts):
    """Backpressure at a completed boundary; never prune or resume by guessing."""
    ack=root/'ARCHIVE_ACK'/f"checkpoint_{counts['agent_steps']:09d}.json"
    atomic_json(root/'archive_wait.json',{'waiting':True,'utc':now(),**counts,
        'checkpoint_sha256':record['sha256'],'checkpoint':record['file']})
    while True:
        if ack.exists():
            receipt=json.loads(ack.read_text())
            if receipt.get('passed') is not True or receipt.get('checkpoint_sha256')!=record['sha256']:
                raise RuntimeError('Archive acknowledgement does not match this checkpoint')
            if not receipt.get('verification_sha256') or not receipt.get('local_root'):
                raise RuntimeError('Archive acknowledgement lacks verification evidence')
            atomic_json(root/'archive_wait.json',{'waiting':False,'utc':now(),**counts,
                'ack':str(ack.relative_to(root)),'checkpoint_sha256':record['sha256']})
            return
        time.sleep(15)

def await_checkpoint_space(root,replay,counts):
    """Conservative measured write-peak guard, separate from the action budget."""
    previous=root/'latest_checkpoint.json'
    if previous.exists():
        old=json.loads(previous.read_text())
        ratio=counts['agent_steps']/max(1,old['agent_steps'])
        dependencies=sum(x['bytes'] for x in old['replay_files'])
        need=int(1.5*ratio*(dependencies+old['bytes'])+512*1024**2+2*1024**3)
        method='previous_full_dependency_bytes_scaled_to_current_rows_times_1.5_plus_write_and_reserve'
    else:
        need=int(1.2*replay.statistics()['ram_bytes']+3*1024**3)
        method='first_checkpoint_uncompressed_replay_plus_model_and_reserve'
    while True:
        free=shutil.disk_usage(root).free
        atomic_json(root/'capacity_wait.json',{'waiting':free<need,'utc':now(),**counts,
            'free_bytes':free,'required_write_peak_bytes':need,'method':method,
            'prediction_is_not_a_guarantee':True})
        if free>=need:return
        time.sleep(30)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--protocol',type=Path,required=True)
    ap.add_argument('--upstream',type=Path,required=True)
    ap.add_argument('--run-dir',type=Path,required=True)
    ap.add_argument('--arm',choices=['D','H'],required=True)
    ap.add_argument('--stage',choices=['smoke','formal'],required=True)
    ap.add_argument('--smoke-actions',type=int,default=6000)
    ap.add_argument('--environment-lock',type=Path,required=True)
    ap.add_argument('--launch-gate',type=Path)
    ap.add_argument('--archive-required',action='store_true')
    args=ap.parse_args()
    cfg=load_protocol(args.protocol)
    source_hash,source_files=source_digest()
    identity={'protocol_sha256':protocol_sha256(cfg),'source_sha256':source_hash,
              'environment_lock_sha256':sha(args.environment_lock),'upstream_commit':cfg['official_core']['commit']}
    budget=cfg['training']['actions_per_arm'] if args.stage=='formal' else args.smoke_actions
    if budget<500 or budget%500:raise ValueError('Only whole 500-action episodes permitted')
    if args.stage=='formal':
        if not args.archive_required:raise ValueError('Formal runs require verified E-drive archive acknowledgements')
        if args.launch_gate is None:raise ValueError('Formal launch requires a verified gate')
        gate=json.loads(args.launch_gate.read_text())
        if not gate.get('passed') or gate.get('identity')!=identity:
            raise ValueError('Gate missing, failed or bound to different source/config/environment')
        if not all(gate.get(k) for k in ['both_smokes_passed','recovery_passed','capacity_passed','environment_exact_match']):
            raise ValueError('Not all required launch gates passed')
    root=args.run_dir.resolve();root.mkdir(parents=True,exist_ok=False)
    atomic_json(root/'identity.json',identity)
    atomic_json(root/'source_manifest.json',source_files)
    atomic_json(root/'protocol.json',cfg)
    shutil.copyfile(args.environment_lock,root/'environment_lock.json')
    atomic_json(root/'command.json',{'argv':__import__('sys').argv,'stage':args.stage,'arm':args.arm})
    (root/'train.pid').write_text(str(os.getpid())+'\n')
    counts={'agent_steps':0,'env_steps':0,'episodes':0,'update_steps':0}
    env=None;started=time.monotonic()
    try:
        np.random.seed(0);random.seed(0)
        masks=make_mask_bank(cfg)
        np.save(root/'training_masks.npy',masks,allow_pickle=False)
        atomic_json(root/'mask_bank.json',{'sha256':sha(root/'training_masks.npy'),'episodes':len(masks),'same_for_both_arms':True})
        bcfg={'seed':0,'size':'size12m','platform':'cuda','compute_dtype':'bfloat16','batch_size':16,'batch_length':64}
        backbone=MatchedDreamerBackbone(bcfg,args.upstream,root)
        atomic_json(root/'backbone_config.json',backbone.resolved_config)
        replay=NativeReplay(backbone,directory=root/'REPLAY',capacity=1000000,seed=0)
        controller=Controller(backbone,'black_frame_native' if args.arm=='D' else 'helix_v2',seed=0)
        prefill_rng=np.random.default_rng(810000)
        env=MatchedWalkerEnv(cfg)
        checkpoint_records=[]
        resources(root,counts)
        for episode in range(budget//500):
            backbone.pin_inference()
            image=env.reset(seed=training_seed(cfg,episode),mask=masks[episode])
            if episode==0:
                live_lock=runtime_lock(cfg)
                mismatch=first_difference(json.loads(args.environment_lock.read_text()),live_lock)
                atomic_json(root/'live_environment_lock.json',live_lock)
                if mismatch:raise RuntimeError('Live environment differs from exact gate at '+mismatch)
            controller.reset(image,episode_id=episode)
            observation={'image':image.copy(),'reward':np.float32(0.),'is_first':np.bool_(True),'is_last':np.bool_(False),'is_terminal':np.bool_(False)}
            ep_return=0.;masked=0
            for t in range(500):
                action=controller.act(prefill_rng.uniform(-1,1,size=6).astype(np.float32) if counts['agent_steps']<5000 else None)
                state=controller.state
                before_actions,before_calls=env.action_decisions,env.episode_env_steps
                try:
                    next_image,reward,done,info=env.step(action)
                finally:
                    # Count acknowledged interactions even if rendering, replay,
                    # or inference fails after the simulator advanced.
                    counts['agent_steps']+=env.action_decisions-before_actions
                    counts['env_steps']+=env.episode_env_steps-before_calls
                if done!=(t==499):raise RuntimeError('Unexpected training horizon')
                ep_return+=float(reward);masked+=not bool(info['availability'])
                append(root/'interactions.jsonl',{'episode':episode,'t':t,'agent_steps':counts['agent_steps'],
                    'reward':float(reward),'dmc_calls':int(info['env_steps_this_step']),
                    'available_before_action':bool(masks[episode,t]),'available_outcome':bool(info['availability']),
                    'action':action.tolist()})
                replay.add(observation,action,state)
                controller.step(next_image,action,info,episode_id=episode)
                observation={'image':next_image.copy(),'reward':np.float32(reward),'is_first':np.bool_(False),
                    'is_last':np.bool_(done),'is_terminal':np.bool_(info['terminated'])}
                target=max(0,counts['agent_steps']-5000)//4
                while counts['update_steps']<target:
                    metrics=replay.train_once()
                    if any(not np.isfinite(v) for v in metrics.values()):raise FloatingPointError('Nonfinite official learning metric')
                    counts['update_steps']+=1
                    if counts['update_steps']%10==0 or counts['update_steps']<=3:
                        append(root/'metrics.jsonl',{'utc':now(),**counts,**metrics})
                if counts['agent_steps']%100==0:
                    atomic_json(root/'status.json',{'stage':args.stage,'arm':args.arm,'utc':now(),**counts,'wall_s':time.monotonic()-started})
            replay.add(observation,np.zeros(6,np.float32),controller.state)
            counts['episodes']+=1
            append(root/'episodes.jsonl',{'utc':now(),**counts,'return':ep_return,'masked_observations':masked,
                  'mask_sha256':mask_sha256(masks[episode]),'environment_seed':training_seed(cfg,episode)})
            resources(root,counts)
            print(json.dumps({'utc':now(),**counts,'return':ep_return}),flush=True)
            save_due=counts['agent_steps']%10000==0 or counts['agent_steps']==budget
            if args.stage=='smoke' and counts['agent_steps'] in (1000,5000,5500):save_due=True
            if save_due:
                if args.archive_required:await_checkpoint_space(root,replay,counts)
                checkpoint_records.append(checkpoint(root,backbone,replay,controller,prefill_rng,counts,identity))
                if args.archive_required:await_verified_archive(root,checkpoint_records[-1],counts)
            eval_due=(args.stage=='formal' and counts['agent_steps']%20000==0) or (args.stage=='smoke' and counts['agent_steps']==budget)
            if eval_due:
                before_host=host_fingerprint(controller,prefill_rng,replay,counts)
                periodic_evaluation(backbone,cfg,root,args.arm,counts['agent_steps'],
                    seeds=None if args.stage=='formal' else [50000])
                after_host=host_fingerprint(controller,prefill_rng,replay,counts)
                if before_host!=after_host:raise RuntimeError('Evaluation mutated host/controller/replay RNG or training counters')
                atomic_json(root/'EVAL'/f"{counts['agent_steps']:09d}"/'host_zero_update.json',
                    {'passed':True,'before_sha256':before_host,'after_sha256':after_host})
        if counts['env_steps']!=2*budget or counts['update_steps']!=max(0,budget-5000)//4:
            raise RuntimeError('Budget/update accounting mismatch')
        closing_env=env;env=None
        closing_env.close()
        atomic_json(root/'completion.json',{'training_complete':True,'final_acceptance_complete':False,
              'stage':args.stage,'arm':args.arm,'utc':now(),**counts,'identity':identity,
              'final_checkpoint':checkpoint_records[-1],'wall_s':time.monotonic()-started})
        (root/'train.exit').write_text('0\n')
    except BaseException as e:
        atomic_json(root/'failure.json',{'utc':now(),**counts,'error':repr(e),'traceback':traceback.format_exc()})
        (root/'train.exit').write_text('1\n')
        raise
    finally:
        if env is not None:
            try:env.close()
            except BaseException as exc:
                atomic_json(root/'cleanup_failure.json',{'utc':now(),'error':repr(exc)})
                (root/'train.exit').write_text('1\n')

if __name__=='__main__':main()
