"""Isolated, fresh 5500-action learning preflight; never a formal launch.

Both proposed arms share the native controller during this bootstrap. This
job tests that path and the independent auxiliary updates. It does not prove
the subsequent calibrated Final collector or two completed training runs.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import pickle
import random
import time
import traceback

os.environ.setdefault('MUJOCO_GL','egl')
os.environ.setdefault('PYOPENGL_PLATFORM','egl')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
import numpy as np

from .backbone import FinalBackbone
from .environment import original_environment_contract
from .epistemic import BootstrapDynamics
from .provenance import source_digest
from helixwm_matched.backbone import NativeReplay
from helixwm_matched.controllers import Controller
from helixwm_matched.environment import MatchedWalkerEnv
from helixwm_matched.env_probe import runtime_lock,first_difference
from helixwm_matched.protocol import atomic_json
from helixwm_matched.train import append,resources,sha


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--run-dir',type=Path,required=True)
    ap.add_argument('--upstream',type=Path,required=True)
    ap.add_argument('--protocol',type=Path,required=True)
    ap.add_argument('--environment-lock',type=Path,required=True)
    args=ap.parse_args()
    from .protocol_v2 import load_protocol,protocol_sha256,make_mask_bank,training_seed
    protocol=load_protocol(args.protocol)
    source,files=source_digest()
    root=args.run_dir.resolve();root.mkdir(parents=True,exist_ok=False)
    identity={'source_sha256':source,'protocol_sha256':protocol_sha256(protocol),
              'stage':'isolated_shared_native_bootstrap_preflight','fresh_initialization':True,
              'formal_training_actions':0,'old_checkpoint_loaded':False,
              'environment_lock_sha256':sha(args.environment_lock)}
    atomic_json(root/'identity.json',identity);atomic_json(root/'source_manifest.json',files)
    atomic_json(root/'command.json',{'argv':__import__('sys').argv})
    (root/'train.pid').write_text(str(os.getpid())+'\n')
    counts={'agent_steps':0,'env_steps':0,'episodes':0,'update_steps':0}
    env=None;start=time.monotonic()
    try:
        np.random.seed(0);random.seed(0)
        cfg=original_environment_contract();masks=make_mask_bank(protocol,partition='smoke')
        np.save(root/'masks.npy',masks[:11],allow_pickle=False)
        bcfg={'seed':0,'size':'size12m','platform':'cuda','compute_dtype':'bfloat16','batch_size':16,'batch_length':64}
        backbone=FinalBackbone(bcfg,args.upstream,root)
        atomic_json(root/'resolved_backbone.json',backbone.resolved_config)
        replay=NativeReplay(backbone,directory=root/'REPLAY',capacity=1000000,seed=0)
        auxiliary=BootstrapDynamics(protocol['training']['auxiliary_predictor']['rng_seed_root'],
                                    {k:identity[k] for k in ('source_sha256','protocol_sha256')})
        controller=Controller(backbone,'black_frame_native',seed=0)
        prefill=np.random.default_rng(1810000);env=MatchedWalkerEnv(cfg)
        resources(root,counts)
        for episode in range(11):
            backbone.pin_inference()
            image=env.reset(seed=training_seed(protocol,episode,partition='smoke'),mask=masks[episode])
            if episode==0:
                live=runtime_lock(cfg);atomic_json(root/'live_environment_lock.json',live)
                mismatch=first_difference(json.loads(args.environment_lock.read_text()),live)
                if mismatch:raise RuntimeError('Live environment differs from saved matched lock at '+mismatch)
            controller.reset(image,episode_id=episode)
            observation={'image':image.copy(),'reward':np.float32(0.),'is_first':np.bool_(True),
                         'is_last':np.bool_(False),'is_terminal':np.bool_(False)}
            available=True;total=0.
            for t in range(500):
                action=controller.act(prefill.uniform(-1,1,6).astype(np.float32) if counts['agent_steps']<5000 else None)
                before_actions,before_calls=env.action_decisions,env.episode_env_steps
                state=controller.state
                try:next_image,reward,done,info=env.step(action)
                finally:
                    counts['agent_steps']+=env.action_decisions-before_actions
                    counts['env_steps']+=env.episode_env_steps-before_calls
                if done!=(t==499):raise RuntimeError('Unexpected Walker horizon')
                replay.add(observation,action,state)
                auxiliary.update(image,action,next_image,float(reward),float(not info['terminated']),
                    transition_id=counts['agent_steps'],previous_available=available,next_available=bool(info['availability']))
                controller.step(next_image,action,info,episode_id=episode)
                image=next_image;available=bool(info['availability']);total+=float(reward)
                observation={'image':image.copy(),'reward':np.float32(reward),'is_first':np.bool_(False),
                    'is_last':np.bool_(done),'is_terminal':np.bool_(info['terminated'])}
                target=max(0,counts['agent_steps']-5000)//4
                while counts['update_steps']<target:
                    metrics=replay.train_once()
                    if any(not np.isfinite(v) for v in metrics.values()):raise FloatingPointError('Nonfinite author learning metric')
                    counts['update_steps']+=1
                    append(root/'metrics.jsonl',{**counts,**metrics})
                append(root/'interactions.jsonl',{'episode':episode,'timestamp':t+1,**counts,
                    'reward':float(reward),'available':available,'action':action.tolist()})
                if counts['agent_steps']%100==0:
                    atomic_json(root/'status.json',{'stage':'shared_bootstrap',**counts,'wall_s':time.monotonic()-start})
            replay.add(observation,np.zeros(6,np.float32),controller.state)
            counts['episodes']+=1
            append(root/'episodes.jsonl',{**counts,'return':total,'environment_seed':72000+episode})
            resources(root,counts)
            print(json.dumps({**counts,'return':total,'wall_s':time.monotonic()-start}),flush=True)
        if counts!={'agent_steps':5500,'env_steps':11000,'episodes':11,'update_steps':125}:
            raise RuntimeError('Bootstrap accounting mismatch')
        backbone.pin_inference();heads=auxiliary.pin()
        if not heads.ready:raise RuntimeError('No usable visible-transition auxiliary fit')
        replay_state=replay.snapshot()
        payload={'format':'helixwm_final_bootstrap_probe_v1','identity':identity,'counts':counts,
            'backbone':backbone.snapshot(),'auxiliary':auxiliary.snapshot(),'heads':heads.snapshot(),
            'replay':replay_state,'completed_native_controller':controller.snapshot(),
            'prefill_rng':prefill.bit_generator.state,'numpy_rng':np.random.get_state(),'python_rng':random.getstate()}
        checkpoint=root/'bootstrap_000005500.pkl'
        with checkpoint.open('xb') as f:
            pickle.dump(payload,f,protocol=5);f.flush();os.fsync(f.fileno())
        record={'path':checkpoint.name,'sha256':sha(checkpoint),'bytes':checkpoint.stat().st_size,
            'model_sha256':backbone.inference_bundle_sha256(),'epistemic_heads_sha256':heads.sha256,
            'replay_dependencies':NativeReplay.referenced_files(replay_state),'counts':counts}
        atomic_json(root/'checkpoint.json',record)
        env.close();env=None
        atomic_json(root/'completion.json',{'passed':True,'scope':'fresh_native_bootstrap_and_auxiliary_learning_only',
            'final_controller_calibrated':False,'formal_training_started':False,'counts':counts,
            'checkpoint':record,'wall_s':time.monotonic()-start})
        (root/'train.exit').write_text('0\n')
    except BaseException as exc:
        atomic_json(root/'failure.json',{'error':repr(exc),'traceback':traceback.format_exc(),'counts':counts})
        (root/'train.exit').write_text('1\n');raise
    finally:
        if env is not None:env.close()

if __name__=='__main__':main()
