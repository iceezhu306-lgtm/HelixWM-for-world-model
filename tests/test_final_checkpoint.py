"""CPU-only strict formal checkpoint contracts; never GPU recovery evidence."""
import copy
import hashlib
import json
from pathlib import Path
import pickle
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from helixwm_final import final_checkpoint as c
from helixwm_final import protocol_v3
from helixwm_final.backbone import AUX_FORMAT,UPSTREAM_API_SHA256,inference_bundle_hash,host_parameter_schema
from helixwm_final.epistemic import BootstrapDynamics
from helixwm_matched.backbone import _array_id,array_tree_hash,NativeReplay


def put(root,name,data):
    path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
    return {'path':name,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}


def encoded(root,value):
    name=_array_id(value)+'.npz';path=root/'REPLAY'/name;path.parent.mkdir(exist_ok=True)
    if not path.exists():
        with path.open('xb') as stream:np.savez_compressed(stream,value=value)
    return {'path':name,'value_sha256':_array_id(value),'codec':'identity','dtype':value.dtype.str,
            'shape':list(value.shape),'file_sha256':c.sha_file(path),'bytes':path.stat().st_size}


def fixture(root,steps=10000):
    root.mkdir(parents=True,exist_ok=True);runtime=root.parent/'runtime';runtime.mkdir(exist_ok=True)
    names=['helixwm_final/train.py','helixwm_final/protocol_v3.py','helixwm_final/backbone.py',
           'helixwm_final/full_controller.py','helixwm_final/bank_gate.py','helixwm_matched/backbone.py']
    source={}
    for name in names:
        data=b'# CPU fixture source never executed\n';put(runtime,name,data);put(root,'SOURCE/'+name,data)
        source[name]=hashlib.sha256(data).hexdigest()
    put(root,'source_manifest.json',c.canonical(source));env=put(root,'environment_lock.json',b'{"cpu_fixture_only":true}')
    bank=put(root,'BANK_SHARED/manifest.json',c.canonical({'source_sha256':'a'*64,'protocol_sha256':'b'*64}))
    config=protocol_v3.registered_protocol()
    identity={'run_id':root.name,'arm':'D','source_sha256':hashlib.sha256(c.canonical(source)).hexdigest(),
        'protocol_sha256':protocol_v3.protocol_sha256(config),'environment_lock_sha256':env['sha256'],
        'upstream_commit':c.UPSTREAM,'bank_manifest_sha256':bank['sha256']}
    counts={'agent_steps':steps,'env_steps':2*steps,'episodes':steps//500,'update_steps':(steps-5000)//4}
    updates=counts['update_steps']
    policy={'dyn/x':np.zeros((2,),np.float32)};auxparams={'rew/x':np.ones(2,np.float32),'con/x':np.ones(2,np.float32)}
    bundle={'model_version':updates,'model_sha256':inference_bundle_hash(policy,auxparams,updates),
        'policy_sha256':array_tree_hash(policy),'auxiliary_sha256':array_tree_hash(auxparams),'upstream_commit':c.UPSTREAM}
    core={'format':'helixwm_final_backbone_v1','upstream_commit':c.UPSTREAM,'final_upstream_api_sha256':UPSTREAM_API_SHA256,
        'training':{'counters':{'updates':updates,'batches':updates,'actions':0},'params':{**policy,**auxparams,'opt/state':np.zeros(3,np.float32)}},
        'train_carry':{'fixture':np.zeros(2,np.float32)},'inference_params':policy,'writeback_updates':updates,'inference_version':updates,
        'final_auxiliary_heads':{'format':AUX_FORMAT,'model_version':updates,'params':auxparams,
            'params_sha256':array_tree_hash(auxparams),'parameter_schema':host_parameter_schema(auxparams)},'final_inference_bundle':bundle}
    auxiliary=BootstrapDynamics(0,{k:identity[k] for k in ('source_sha256','protocol_sha256')})
    image=np.zeros((64,64,3),np.uint8)
    for i in range(1,41):auxiliary.update(image,np.zeros(6),image,0.,1.,transition_id=i,previous_available=True,next_available=True)
    for i in range(41,steps+1):auxiliary.update(None,None,None,None,None,transition_id=i,previous_available=False,next_available=False)
    auxiliary.version=counts['episodes'];heads=auxiliary.pin()
    chunks=[]
    nrows=steps+counts['episodes'];lengths=[1024]*(nrows//1024)+[nrows%1024]
    for i,n in enumerate(lengths):
        chunks.append({'uuid':'chunk'+str(i),'succ':'chunk'+str(i+1),'time':'fixture','length':n,'size':1024,
                       'arrays':{'image':encoded(root,np.zeros((n,1),np.uint8))}})
    refs={x['uuid']:1 for x in chunks};last=chunks[-1]['uuid'];length=chunks[-1]['length'];refs[last]=66
    rng=copy.deepcopy(np.random.default_rng(9).bit_generator.state)
    replay={'format':'helixwm_native_replay_exact_v1','upstream_commit':c.UPSTREAM,'capacity':1000000,'chunksize':1024,
        'seed':0,'length':65,'online':False,'rows_added':nrows,'stream_index':1,'chunks':chunks,'refs':refs,
        'items':[[0,'chunk0',0]],'fifo':[0],'itemid':1,'current':[[0,last,length]],
        'streams':[[0,[[last,i] for i in range(length-64,length)]]],'sampler_keys':[0],'sampler_indices':[[0,0]],
        'sampler_rng':rng,'metrics':{}}
    completed={'controller':'black_frame_native','chain':{'state':{'episode_id':counts['episodes']-1,'timestamp':500,'anchor_time':500,
        'last_action':None,'pending':None,'missing_length':0},'rngs':{k:copy.deepcopy(rng) for k in ('real','actor','dream','repair')}}}
    payload={'format':c.FORMAT,'identity':identity,'counts':counts,'backbone':core,'auxiliary':auxiliary.snapshot(),
        'pinned_heads':heads.snapshot(),'replay':replay,'completed_controller_audit':completed,
        'completed_controller_is_previous_episode_pin':True,'bank_gate':None,
        'bank_reference':{'path':'BANK_SHARED','manifest_sha256':bank['sha256']},'bank_gate_reference':None,
        'completed_previous_gate_reference':None,'trajectory':None,'prefill_rng':rng,
        'numpy_rng':np.random.RandomState(2).get_state(),'python_rng':random.Random(2).getstate(),
        'next_operation':'current_pin_bank_gate_or_native_fallback_then_reset_new_episode',
        'environment_restore_scope':'completed_episode_boundary_new_registered_reset_not_mid_episode','auto_resume_allowed':False}
    cp=put(root,f'CHECKPOINTS/checkpoint_{steps:09d}.pkl',pickle.dumps(payload,protocol=5))
    refs=NativeReplay.referenced_files(replay)
    rows=[]
    for name in ['SOURCE/'+x for x in names]+['BANK_SHARED/manifest.json']+['REPLAY/'+x['path'] for x in refs]:
        path=root/name;rows.append({'kind':'file','path':name,'bytes':path.stat().st_size,'sha256':c.sha_file(path)})
    dep=put(root,f'CHECKPOINTS/checkpoint_{steps:09d}.dependencies.jsonl',b''.join(c.canonical(x)+b'\n' for x in rows))
    record={'format':c.FORMAT,'identity':identity,'counts':counts,'checkpoint':cp,'replay_files':refs,
        'pinned_model_sha256':bundle['model_sha256'],'heads_sha256':heads.sha256,'bank_gate_sha256':None,
        'trajectory':None,'dependencies_manifest':dep,'bank_reference':payload['bank_reference'],
        'bank_gate_reference':None,'completed_previous_gate_reference':None}
    side=put(root,f'CHECKPOINTS/checkpoint_{steps:09d}.json',c.canonical(record))
    options={'expected_sha256':cp['sha256'],'expected_record_sha256':side['sha256'],'expected_identity':identity,
             'protocol_config':config,'runtime_source_root':runtime}
    return payload,record,options


def as_final_collector(payload,record):
    from types import SimpleNamespace
    from helixwm_final.controller import FinalController
    identity=payload['identity'];identity['arm']='F';record['identity']=copy.deepcopy(identity)
    old_version=payload['counts']['update_steps']-125
    shell=FinalController(SimpleNamespace(inference_version=old_version),
        expected_source_sha256=identity['source_sha256'],expected_model_sha256='f'*64)
    shell.FORMAT='helixwm_final_integrated_bank_controller_v3'
    old=shell.snapshot();old['state'].update(initialized=True,episode_id=payload['counts']['episodes']-1,
        timestamp=500,missing_age=0,evidence_kind='OBSERVED_POSTERIOR',real={'deter':np.zeros((1,2),np.float32),
        'stoch':np.zeros((1,1,2),np.float32)},bank=None)
    cursor={'cpu_fixture_dependency_cursor':True}
    old['integration']={'context':{},'execution':'bank_production','heads_sha256':'e'*64,
        'auxiliary_state':np.zeros((3,1,16),np.float64),'current_epistemic':0.,'current_temporal_error':None,
        'executed_prefix_sha256':'e'*64,'screen_rng':copy.deepcopy(payload['prefill_rng']),'guard':None,'trajectory':cursor}
    payload['completed_controller_audit']=old;payload['trajectory']=cursor;record['trajectory']=cursor


class FormalCheckpointTests(unittest.TestCase):
    def setUp(self):
        directory=Path(__file__).resolve().parents[2]/'preflight'/'formal_checkpoint_cpu';directory.mkdir(parents=True,exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(dir=directory);self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)/'formal_D';self.payload,self.record,self.options=fixture(self.root)
        self.path=self.root/'CHECKPOINTS/checkpoint_000010000.pkl'

    def test_valid_cpu_metadata_and_state_do_not_mutate_host_rng(self):
        before=c.tree_hash((np.random.get_state(),random.getstate()))
        audit=c.validate_payload(self.payload,self.record,self.options['expected_identity'])
        self.assertEqual(audit['replay']['rows_added'],10020)
        self.assertEqual(audit['replay']['retained_rows'],10020)
        self.assertEqual(before,c.tree_hash((np.random.get_state(),random.getstate())))

    def test_readonly_loader_with_mocked_bank_authority_preserves_input_files(self):
        # Unit seam only: the production loader is tested by bank-specific tests.
        before={p.relative_to(self.root).as_posix():c.sha_file(p) for p in self.root.rglob('*') if p.is_file()}
        with patch('helixwm_final.bank_replay.load_original_bank',return_value=({},{})):
            payload,record,gate=c.load_checkpoint(self.path,**self.options)
        self.assertIsNone(gate);self.assertEqual(record,self.record);self.assertEqual(c.tree_hash(payload),c.tree_hash(self.payload))
        after={p.relative_to(self.root).as_posix():c.sha_file(p) for p in self.root.rglob('*') if p.is_file()}
        self.assertEqual(before,after)

    def test_require_final_and_old_bootstrap_format_rejected(self):
        with self.assertRaises(ValueError):c.load_checkpoint(self.path,**self.options,require_final=True)
        for name in ('helixwm_final_bootstrap_probe_v1','helixwm_final_full_episode_boundary_v1'):
            bad=copy.deepcopy(self.payload);bad['format']=name
            with self.assertRaises(ValueError):c.validate_payload(bad,self.record,self.options['expected_identity'])

    def test_metadata_cp_source_environment_protocol_errors_precede_pickle(self):
        variants=[dict(self.options,expected_record_sha256='f'*64),dict(self.options,expected_sha256='f'*64),
            dict(self.options,expected_identity={**self.options['expected_identity'],'arm':'F'}),
            dict(self.options,protocol_config={})]
        for opts in variants:
            with patch.object(c.pickle,'loads') as load:
                with self.assertRaises(ValueError):c.load_checkpoint(self.path,**opts)
                load.assert_not_called()
        (self.root/'environment_lock.json').write_bytes(b'changed')
        with patch.object(c.pickle,'loads') as load:
            with self.assertRaises(ValueError):c.load_checkpoint(self.path,**self.options)
            load.assert_not_called()

    def test_full_replay_corruption_blocks_before_pickle(self):
        item=self.record['replay_files'][0];(self.root/'REPLAY'/item['path']).write_bytes(b'corrupt')
        with patch.object(c.pickle,'loads') as load:
            with self.assertRaises(ValueError):c.load_checkpoint(self.path,**self.options)
            load.assert_not_called()

    def test_stable_prefix_dependency_accepts_tail_but_not_prefix_mutation(self):
        desc=put(self.root,'console.log',b'original')
        with (self.root/'console.log').open('ab') as stream:stream.write(b'extra')
        c.verify_descriptor(self.root,desc,prefix=True)
        with self.assertRaises(ValueError):c.verify_descriptor(self.root,desc)
        (self.root/'console.log').write_bytes(b'corrupt!extra')
        with self.assertRaises(ValueError):c.verify_descriptor(self.root,desc,prefix=True)

    def test_actual_replay_rows_selectors_refs_and_graph_are_checked(self):
        for change in ('rows','shape','refs','sampler','cycle'):
            bad=copy.deepcopy(self.payload)
            if change=='rows':bad['replay']['rows_added']-=1
            if change=='shape':bad['replay']['chunks'][0]['length']-=1
            if change=='refs':bad['replay']['refs']['chunk0']+=1
            if change=='sampler':bad['replay']['sampler_indices']=[[0,1]]
            if change=='cycle':bad['replay']['items']=[[0,'chunk0',1000]];bad['replay']['chunks'][0]['succ']='chunk0'
            with self.assertRaises(ValueError):c.validate_payload(bad,self.record,self.options['expected_identity'])

    def test_budget_author_counter_pin_and_optimizer_finite_state_checked(self):
        for change in ('env','updates','pin','opt','fresh'):
            bad=copy.deepcopy(self.payload)
            if change=='env':bad['counts']['env_steps']-=1
            if change=='updates':bad['backbone']['training']['counters']['updates']-=1
            if change=='pin':bad['backbone']['inference_version']-=1
            if change=='opt':bad['backbone']['training']['params']['opt/state'][0]=np.nan
            if change=='fresh':bad['backbone']['training']['params']['dyn/x']=np.ones(2,np.float32)
            with self.assertRaises(ValueError):c.validate_payload(bad,self.record,self.options['expected_identity'])

    def test_heads_auxiliary_rng_collector_and_scope_are_checked(self):
        for change in ('heads','aux','rng','collector','scope'):
            bad=copy.deepcopy(self.payload)
            if change=='heads':bad['pinned_heads']['coefficients'][0,0,0]+=1
            if change=='aux':bad['auxiliary']['state']['last_transition_id']-=1
            if change=='rng':bad['prefill_rng']={}
            if change=='collector':bad['completed_controller_audit']['chain']['state']['pending']={'x':1}
            if change=='scope':bad['auto_resume_allowed']=True
            with self.assertRaises(ValueError):c.validate_payload(bad,self.record,self.options['expected_identity'])

    def test_source_addition_and_dependency_path_escape_are_rejected(self):
        put(self.options['runtime_source_root'],'helixwm_final/unregistered.py',b'# additional code')
        with patch.object(c.pickle,'loads') as load:
            with self.assertRaises(ValueError):c.load_checkpoint(self.path,**self.options)
            load.assert_not_called()
        for name in ('../outside','/root/file','a/../b','x\\y','a:b'):
            with self.assertRaises(ValueError):c.safe_path(self.root,name)

    def test_integrated_collector_checks_all_latents_history_and_seventh_rng(self):
        payload=copy.deepcopy(self.payload);record=copy.deepcopy(self.record);as_final_collector(payload,record)
        c.validate_payload(payload,record,payload['identity'])
        for change in ('screen','history','aux','probabilities','guard','issued'):
            bad=copy.deepcopy(payload);old=bad['completed_controller_audit']
            if change=='screen':old['integration']['screen_rng']={}
            if change=='history':old['action_history']=[{}]*101
            if change=='aux':old['integration']['auxiliary_state'][0,0,0]=np.nan
            if change=='probabilities':old['state']['real_probs']=np.zeros((1,1,2),np.float32)
            if change=='guard':old['integration']['guard']={'unexpected':True}
            if change=='issued':old['state']['issued_action']=np.zeros(6,np.float32)
            with self.assertRaises((ValueError,TypeError)):c.validate_payload(bad,record,payload['identity'])

    def test_fresh_full_preflight_is_explicit_registered_and_never_formal(self):
        root=Path(self.temp.name)/'preflight_fixture';payload,record,options=fixture(root,6000)
        as_final_collector(payload,record)
        fields={'probe_registration':options['protocol_config']['full_integration_preflight'],
            'formal_training_actions':0,'fresh_initialization':True,'old_checkpoint_loaded':False}
        payload.update(format=c.PREFLIGHT_FORMAT,**fields);record.update(format=c.PREFLIGHT_FORMAT,**fields)
        with self.assertRaises(ValueError):c.validate_payload(payload,record,payload['identity'])
        audit=c.validate_payload(payload,record,payload['identity'],allow_preflight=True,protocol_config=options['protocol_config'])
        self.assertEqual(audit['counts'],{'agent_steps':6000,'env_steps':12000,'episodes':12,'update_steps':250})
        for key,value in [('old_checkpoint_loaded',True),('formal_training_actions',6000),('fresh_initialization',False)]:
            bad=copy.deepcopy(payload);bad[key]=value
            with self.assertRaises(ValueError):c.validate_payload(bad,record,payload['identity'],allow_preflight=True,protocol_config=options['protocol_config'])


if __name__=='__main__':unittest.main()
