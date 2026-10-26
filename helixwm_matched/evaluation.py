"""Frozen evaluations of the common environment; no training or replay access."""
from __future__ import annotations
from pathlib import Path
import copy
import hashlib
import json
import numpy as np
from .controllers import Controller
from .environment import MatchedWalkerEnv
from .protocol import fixed_gap_mask, mask_sha256, atomic_json
from .backbone import array_tree_hash

def evaluate_episode(backbone, config, controller_name, seed, gap=0, *, partial_report=None, render_callback=None):
    mask = np.ones(501, bool) if gap == 0 else fixed_gap_mask(gap)
    if controller_name == 'clean_native' and gap:
        raise ValueError('Clean condition must use clean observations')
    chain = Controller(backbone, controller_name, seed=int(seed))
    # Diagnostic images are evaluator-only video output and never enter policy.
    env = MatchedWalkerEnv(config,diagnostics=render_callback is not None)
    actions, rewards, calls, availability = [], [], [], []
    try:
        image = env.reset(seed=int(seed), mask=mask)
        chain.reset(image, episode_id=int(seed))
        if render_callback is not None:
            render_callback(env.render_diagnostic(256),0,{'availability':True})
        for i in range(500):
            action = chain.act()
            image, reward, done, info = env.step(action)
            actions.append(action.copy()); rewards.append(float(reward))
            calls.append(int(info['env_steps_this_step']))
            availability.append(bool(info['availability']))
            chain.step(image, action, info, episode_id=int(seed))
            if render_callback is not None:
                render_callback(env.render_diagnostic(256),i+1,{'availability':bool(info['availability'])})
            if done != (i == 499):
                raise RuntimeError('Unexpected evaluation horizon')
        assert sum(calls) == 1000
        return {'controller':controller_name,'seed':int(seed),'gap':gap,
                'episode_return':float(sum(rewards)), 'actions':500,'dmc_calls':sum(calls),
                'availability_sha256':mask_sha256(mask),
                'trace': {'action':np.asarray(actions,np.float32),
                          'reward':np.asarray(rewards,np.float64),
                          'dmc_calls':np.asarray(calls,np.int32),
                          'availability_outcome':np.asarray(availability,bool),
                          'availability_before_action':mask[:-1].copy()},
                'repair_events':list(chain.chain.events)}
    except BaseException as exc:
        if partial_report is not None:
            atomic_json(partial_report,{'passed':False,'controller':controller_name,
                'seed':int(seed),'gap':gap,'error':repr(exc),
                'acknowledged_actions':env.action_decisions,
                'acknowledged_dmc_calls':env.episode_env_steps,
                'complete_rewards':rewards,'complete_actions':[a.tolist() for a in actions]})
        raise
    finally:
        try:env.close()
        except BaseException as exc:
            if partial_report is not None:
                atomic_json(partial_report,{'passed':False,'error':'environment_close: '+repr(exc),
                    'acknowledged_actions':env.action_decisions,'acknowledged_dmc_calls':env.episode_env_steps})
            raise

def periodic_evaluation(backbone, config, run_dir, arm, decisions, *, seeds=None):
    """Private controllers and environments leave backbone and training RNG intact."""
    root=Path(run_dir)/'EVAL'/f'{decisions:09d}'
    root.mkdir(parents=True,exist_ok=False)
    seeds=list(range(50000,50005)) if seeds is None else list(seeds)
    before=array_tree_hash(backbone.snapshot())
    controller='black_frame_native' if arm=='D' else 'helix_v2'
    rows=[]
    completed_actions=completed_calls=0;case_complete=False;partial=None
    try:
        for condition,gap in [('clean_native',0),(controller,20)]:
            for seed in seeds:
                case_complete=False
                partial=root/f'{condition}_seed{seed}_gap{gap}_failure.json'
                row=evaluate_episode(backbone,config,condition,seed,gap,partial_report=partial)
                case_complete=True;completed_actions+=row['actions'];completed_calls+=row['dmc_calls']
                atomic_json(root/'interaction_ledger.json',{'passed':False,'stage':'in_progress',
                    'acknowledged_actions':completed_actions,'acknowledged_dmc_calls':completed_calls})
                trace=row.pop('trace'); events=row.pop('repair_events')
                name=f'{condition}_seed{seed}_gap{gap}'
                np.savez_compressed(root/(name+'.npz'),**trace)
                atomic_json(root/(name+'.json'),{**row,'repair_events':events})
                row.update(arm=arm,training_actions=decisions)
                rows.append(row)
                with (Path(run_dir)/'eval.jsonl').open('a',encoding='utf-8',newline='\n') as f:
                    f.write(json.dumps(row,allow_nan=False)+'\n')
        after=array_tree_hash(backbone.snapshot())
        if before!=after:
            raise RuntimeError('Frozen evaluation changed official training state')
    except BaseException as exc:
        if not case_complete and partial is not None and partial.exists():
            failed=json.loads(partial.read_text())
            completed_actions+=failed['acknowledged_actions'];completed_calls+=failed['acknowledged_dmc_calls']
        atomic_json(root/'interaction_ledger.json',{'passed':False,'stage':'failed','error':repr(exc),
            'acknowledged_actions':completed_actions,'acknowledged_dmc_calls':completed_calls,
            'completed_artifact_rows':len(rows)})
        raise
    report={'passed':True,'training_actions':decisions,'episodes':len(rows),
            'actual_evaluation_actions':500*len(rows),'actual_evaluation_dmc_calls':1000*len(rows),
            'training_updates':0,'before_hash':before,'after_hash':after,'rows':rows}
    atomic_json(root/'frozen_report.json',report)
    atomic_json(root/'interaction_ledger.json',{'passed':True,'stage':'complete',
        'acknowledged_actions':completed_actions,'acknowledged_dmc_calls':completed_calls})
    return report
