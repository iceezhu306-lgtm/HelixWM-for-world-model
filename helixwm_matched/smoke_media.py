"""Explicit short-run media gate; not a formal result or a training continuation."""
from pathlib import Path
import argparse,os,json,pickle,hashlib
os.environ.setdefault('MUJOCO_GL','egl');os.environ.setdefault('PYOPENGL_PLATFORM','egl')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
import numpy as np
from .protocol import load_protocol,atomic_json
from .recovery_probe import verify_and_load,tree_hash
from .backbone import MatchedDreamerBackbone
from .evaluation import evaluate_episode

def main():
    p=argparse.ArgumentParser();p.add_argument('--protocol',required=True);p.add_argument('--upstream',required=True)
    p.add_argument('--checkpoint',required=True);p.add_argument('--environment-lock',required=True);p.add_argument('--output-dir',required=True)
    a=p.parse_args();cfg=load_protocol(a.protocol);payload,_,_=verify_and_load(a.checkpoint,cfg,a.environment_lock)
    out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=False)
    model=MatchedDreamerBackbone({'seed':0,'size':'size12m','platform':'cuda','compute_dtype':'bfloat16','batch_size':16,'batch_length':64},a.upstream,out/'MODEL')
    model.load(payload['backbone']);before=tree_hash(model.snapshot())
    frames=[];availability=[]
    def callback(frame,t,info):
        if t!=len(frames):raise RuntimeError('Nonconsecutive video frames')
        frames.append(frame.copy());availability.append(info['availability'])
    row=evaluate_episode(model,cfg,payload['completed_controller']['controller'],72000,20,
        partial_report=out/'failure.json',render_callback=callback)
    trace=row.pop('trace');events=row.pop('repair_events')
    after=tree_hash(model.snapshot())
    if before!=after or len(frames)!=501:raise RuntimeError('Smoke media gate mismatch')
    np.savez_compressed(out/'frames.npz',frames=np.stack(frames),availability=np.array(availability,bool))
    np.savez_compressed(out/'trace.npz',**trace)
    atomic_json(out/'report.json',{'passed':True,'kind':'smoke_media_only','training_updates':0,
        'actions':500,'dmc_calls':1000,'frames':501,'before_hash':before,'after_hash':after,'episode':row,'repair_events':events})
    print(json.dumps({'passed':True,'output':str(out),'kind':'smoke_media_only'}),flush=True)
if __name__=='__main__':main()
