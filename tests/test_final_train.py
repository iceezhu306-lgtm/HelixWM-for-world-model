"""CPU orchestration/identity tests ONLY; no GPU, SSH or formal launch."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from helixwm_final import train as t
from helixwm_final import protocol_v3 as p


def counts(actions=0):
    return {'agent_steps':actions,'env_steps':2*actions,'episodes':actions//500,
            'update_steps':max(0,actions-5000)//4}


def identity():
    return {'run_id':'fresh_D','arm':'D','source_sha256':'a'*64,
            'protocol_sha256':p.protocol_sha256(p.registered_protocol()),
            'environment_lock_sha256':'b'*64,'upstream_commit':p.registered_protocol()['official_core']['commit'],
            'bank_manifest_sha256':'9'*64}


class FakeEnv:
    def __init__(self, *, raise_after=None, early_done=None, hidden_image=False):
        self.raise_after=raise_after;self.early_done=early_done;self.hidden_image=hidden_image
        self.action_decisions=self.episode_env_steps=0;self.actions=[]
    def reset(self, *, seed, mask):
        self.seed=seed;self.mask=mask;self.action_decisions=self.episode_env_steps=0
        return np.full((64,64,3),30,np.uint8)
    def step(self, action):
        self.action_decisions+=1;self.episode_env_steps+=2;self.actions.append(action.copy())
        n=self.action_decisions
        if n==self.raise_after:raise RuntimeError('Synthetic post-step rendering failure')
        valid=bool(self.mask[n]);image=np.full((64,64,3),30 if valid or self.hidden_image else 0,np.uint8)
        return image,1.,n==500 or n==self.early_done,{'observation_timestamp':n,'availability':valid,
            'env_steps_this_step':2,'terminated':False}


class FakeController:
    def reset(self,image,episode_id):self.timestamp=0;self.image=image.copy()
    @property
    def state(self):return {'at':self.timestamp}
    def act(self, external=None):return np.full(6,.25,np.float32) if external is None else external.copy()
    def step(self,image,action,info,episode_id):self.timestamp+=1;self.image=image.copy()


class FakeReplay:
    def __init__(self, fail_metric=False):self.rows=[];self.updates=0;self.fail_metric=fail_metric
    def add(self,observation,action,state):self.rows.append((copy.deepcopy(observation),action.copy(),copy.deepcopy(state)))
    def train_once(self):
        self.updates+=1
        return {'loss':float('nan') if self.fail_metric else .5,'gradient_norm':1.}


class FakeAuxiliary:
    def __init__(self, value=1e9):self.rows=[];self.value=value
    def update(self,previous_image,action,next_image,reward,continuation,**kwargs):
        self.rows.append({'previous':previous_image.copy(),'next':next_image.copy(),**kwargs})
        return {'untrusted_action_proposal':self.value}


class FinalTrainTests(unittest.TestCase):
    def setUp(self):
        parent=Path(__file__).resolve().parents[2]/'preflight'/'train_contract_tests'
        parent.mkdir(parents=True,exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(dir=parent);self.root=Path(self.temp.name)
        self.cfg=p.registered_protocol()
    def tearDown(self):self.temp.cleanup()

    def episode(self, *, start=5000, env=None, replay=None, auxiliary=None, mask=None):
        c=counts(start);env=FakeEnv() if env is None else env
        replay=FakeReplay() if replay is None else replay
        aux=FakeAuxiliary() if auxiliary is None else auxiliary
        mask=np.ones(501,np.bool_) if mask is None else mask
        result=t.run_episode(env=env,controller=FakeController(),replay=replay,auxiliary=aux,
            prefill_rng=np.random.default_rng(810000),counts=c,episode=start//500,seed=100000+start//500,mask=mask)
        return result,c,env,replay,aux

    def test_shared_native_bootstrap_real_count_schedule_and_terminal_replay_row(self):
        result,c,env,replay,aux=self.episode()
        self.assertEqual(c,counts(5500));self.assertEqual(replay.updates,125)
        self.assertEqual(len(replay.rows),501);self.assertEqual(len(aux.rows),500)
        self.assertEqual(result['return'],500.)
        self.assertTrue(replay.rows[0][0]['is_first']);self.assertFalse(replay.rows[1][0]['is_first'])
        self.assertTrue(replay.rows[-1][0]['is_last'])
        np.testing.assert_array_equal(replay.rows[-1][1],np.zeros(6,np.float32))
        self.assertEqual(replay.rows[0][2],{'at':0});self.assertEqual(replay.rows[1][2],{'at':1})
        self.assertEqual(replay.rows[-1][2],{'at':500})
        self.assertEqual([x['transition_id'] for x in aux.rows],list(range(5001,5501)))

    def test_D_auxiliary_outputs_cannot_select_actions_or_change_update_count(self):
        a=self.episode(auxiliary=FakeAuxiliary(-1e9))
        b=self.episode(auxiliary=FakeAuxiliary(1e9))
        np.testing.assert_array_equal(a[2].actions,b[2].actions)
        self.assertEqual(a[1],b[1]);self.assertEqual(a[3].updates,b[3].updates)

    def test_random_prefill_consumes_external_actions_and_no_author_updates(self):
        result,c,env,replay,aux=self.episode(start=0)
        self.assertEqual(c,counts(500));self.assertEqual(replay.updates,0)
        expected=np.random.default_rng(810000).uniform(-1,1,(500,6)).astype(np.float32)
        np.testing.assert_array_equal(np.stack(env.actions),expected)

    def test_zero_corrupted_frames_go_to_learner_and_visibility_to_auxiliary(self):
        mask=np.ones(501,np.bool_);mask[20:25]=False
        result,c,env,replay,aux=self.episode(mask=mask)
        for index in range(20,25):self.assertFalse(replay.rows[index][0]['image'].any())
        self.assertTrue(aux.rows[18]['previous_available']);self.assertTrue(aux.rows[18]['next_available'])
        self.assertTrue(aux.rows[19]['previous_available']);self.assertFalse(aux.rows[19]['next_available'])
        self.assertFalse(aux.rows[24]['previous_available']);self.assertTrue(aux.rows[24]['next_available'])

    def test_hidden_frame_injection_and_early_termination_fail(self):
        mask=np.ones(501,np.bool_);mask[20]=False
        with self.assertRaises(ValueError):self.episode(env=FakeEnv(hidden_image=True),mask=mask)
        with self.assertRaises(RuntimeError):self.episode(env=FakeEnv(early_done=20))

    def test_failure_preserves_acknowledged_environment_counts(self):
        c=counts(5000);env=FakeEnv(raise_after=7)
        with self.assertRaises(RuntimeError):
            t.run_episode(env=env,controller=FakeController(),replay=FakeReplay(),auxiliary=FakeAuxiliary(),
                prefill_rng=np.random.default_rng(0),counts=c,episode=10,seed=100010,mask=np.ones(501,np.bool_))
        self.assertEqual(c['agent_steps'],5007);self.assertEqual(c['env_steps'],10014)
        self.assertEqual(c['update_steps'],1);self.assertEqual(c['episodes'],10)

    def test_nonfinite_author_metric_stops_at_first_bad_update(self):
        replay=FakeReplay(fail_metric=True);env=FakeEnv()
        with self.assertRaises(FloatingPointError):self.episode(env=env,replay=replay)
        self.assertEqual(env.action_decisions,4);self.assertEqual(replay.updates,1)

    def test_environment_lock_failure_prevents_the_first_action(self):
        env=FakeEnv();c=counts()
        def reject():raise RuntimeError('Synthetic changed environment lock')
        with self.assertRaises(RuntimeError):
            t.run_episode(env=env,controller=FakeController(),replay=FakeReplay(),auxiliary=FakeAuxiliary(),
                prefill_rng=np.random.default_rng(0),counts=c,episode=0,seed=100000,
                mask=np.ones(501,np.bool_),on_reset=reject)
        self.assertEqual(env.actions,[]);self.assertEqual(c,counts())

    def test_counts_and_exact_budget_cannot_be_relaxed(self):
        self.assertTrue(t.validate_counts(counts(1000000),boundary=True))
        for bad in ({**counts(10000),'env_steps':10000},{**counts(5500),'update_steps':126},
                    {**counts(5000),'episodes':True},counts(1000500)):
            with self.assertRaises(ValueError):t.validate_counts(bad,boundary=True)
        with self.assertRaises(ValueError):self.episode(start=1000000)

    def request_receipt(self):
        request={'schema':t.REQUEST_SCHEMA,'identity':identity(),'counts':counts(10000),
            'checkpoint_sha256':'c'*64,'object_manifest':{'path':'ARCHIVE_REQUESTS/10000.objects.jsonl','bytes':100,'sha256':'d'*64},
            'archive_service_source_sha256':'e'*64,'must_expand_recursive_dependency_kinds':True,
            'delete_local_unique_data_allowed':False,'latest_remote_checkpoint_and_all_dependencies_must_remain':True}
        receipt={'schema':t.RECEIPT_SCHEMA,'identity':identity(),'request_sha256':t.json_sha(request),
            'object_manifest_sha256':'d'*64,'checkpoint_sha256':'c'*64,'archive_service_source_sha256':'e'*64,
            'local_root':'E:/VerifiedArchives/fresh_D',
            'remote_before_manifest_sha256':'d'*64,'remote_after_manifest_sha256':'d'*64,
            'local_verified_manifest_sha256':'d'*64,'transaction_complete':True,
            'latest_remote_dependencies_retained':True,'local_unique_objects_deleted':False}
        return request,receipt

    def test_archive_receipt_binds_all_new_identities_and_no_legacy_ack(self):
        request,receipt=self.request_receipt()
        self.assertTrue(t.validate_archive_receipt(receipt,request,archive_source_sha256='e'*64))
        mutations=[('checkpoint_sha256','f'*64),('request_sha256','f'*64),('object_manifest_sha256','f'*64),
                   ('archive_service_source_sha256','f'*64),('local_verified_manifest_sha256','f'*64),
                   ('remote_after_manifest_sha256','f'*64),('transaction_complete',False),
                   ('latest_remote_dependencies_retained',False),('local_unique_objects_deleted',True),
                   ('local_root','D:/old'),('local_root','E:/../C:/unsafe')]
        for key,value in mutations:
            bad=copy.deepcopy(receipt);bad[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):
                t.validate_archive_receipt(bad,request,archive_source_sha256='e'*64)
        for key,value in [('arm','F'),('run_id','old_run'),('source_sha256','f'*64),('protocol_sha256','f'*64)]:
            bad=copy.deepcopy(receipt);bad['identity'][key]=value
            with self.assertRaises(ValueError):t.validate_archive_receipt(bad,request,archive_source_sha256='e'*64)
        with self.assertRaises(ValueError):t.validate_archive_receipt({'passed':True,'checkpoint_sha256':'c'*64},request,archive_source_sha256='e'*64)

    def test_wait_deadline_and_malformed_evidence_never_advance_or_retry_training(self):
        clock=[0.];slept=[];c=counts(5500);before=copy.deepcopy(c)
        def sleep(amount):slept.append(amount);clock[0]+=amount
        with self.assertRaises(TimeoutError):
            t.wait_for_json(self.root/'absent.json',status_path=self.root/'wait.json',stage='calibration',
                identity=identity(),counts=c,timeout_seconds=20.,poll_seconds=15.,clock=lambda:clock[0],sleep=sleep)
        self.assertEqual(slept,[15.,5.]);self.assertEqual(c,before)
        saved=json.loads((self.root/'wait.json').read_bytes())
        self.assertTrue(saved['waiting']);self.assertFalse(saved['training_interactions_advance_while_waiting'])
        (self.root/'invalid.json').write_text('{bad')
        with self.assertRaises(json.JSONDecodeError):
            t.wait_for_json(self.root/'invalid.json',status_path=self.root/'wait.json',stage='calibration',
                identity=identity(),counts=c,sleep=lambda _:self.fail('Malformed evidence was retried'))

    def test_archive_snapshot_freezes_overwritten_status_JSON_and_includes_evaluation(self):
        for name in ('CHECKPOINTS','BANK_SHARED','BANK_GATES','SOURCE','EVAL'):(self.root/name).mkdir()
        checkpoint=self.root/'CHECKPOINTS/checkpoint_000010000.pkl'
        checkpoint.write_bytes(b'synthetic checkpoint bytes - not a model')
        checkpoint.with_suffix('.json').write_text('{"unit_fixture":true}')
        (self.root/'archive_wait.json').write_text('{"old_wait":true}')
        deps=checkpoint.with_suffix('.dependencies.jsonl');deps.write_text('{}\n')
        record={'checkpoint':t.descriptor(checkpoint,self.root),'replay_files':[],'trajectory':None,
            'dependencies_manifest':t.descriptor(deps,self.root)}
        requests=[]
        def complete(path, **kwargs):
            # Simulate the real wait-status overwrite after the dependency
            # snapshot has been sealed. It must not invalidate that snapshot.
            (self.root/'archive_wait.json').write_text('{"new_wait":true}')
            request=json.loads((self.root/'ARCHIVE_REQUESTS/000010000.request.json').read_bytes())
            requests.append(request)
            manifest=request['object_manifest']['sha256']
            return {'schema':t.RECEIPT_SCHEMA,'identity':request['identity'],
                'request_sha256':t.json_sha(request),'object_manifest_sha256':manifest,
                'checkpoint_sha256':request['checkpoint_sha256'],'archive_service_source_sha256':'e'*64,
                'local_root':'E:/archive/fresh_D','remote_before_manifest_sha256':manifest,
                'remote_after_manifest_sha256':manifest,'local_verified_manifest_sha256':manifest,
                'transaction_complete':True,'latest_remote_dependencies_retained':True,'local_unique_objects_deleted':False}
        with patch.object(t,'wait_for_json',side_effect=complete):
            t._archive_checkpoint(self.root,record,counts(10000),identity(),
                {'archive_service_source_sha256':'e'*64},timeout_seconds=1.)
        manifest_path=t.verified_descriptor(self.root,requests[0]['object_manifest'])
        rows=[json.loads(x) for x in manifest_path.read_text().splitlines()]
        snapshot=next(x for x in rows if x.get('path','').endswith('_log_snapshot/archive_wait.json'))
        path=t.verified_descriptor(self.root,{k:v for k,v in snapshot.items() if k!='kind'})
        self.assertEqual(json.loads(path.read_bytes()),{'old_wait':True})
        self.assertTrue(any(x.get('path')=='EVAL' and x['kind']=='immutable_tree_recursive' for x in rows))
        self.assertTrue(any(x.get('path')=='BANK_SHARED' and x.get('include_shared_independent_bank') is True for x in rows))
        self.assertTrue(any(x.get('path')=='BANK_GATES' and x.get('include_all_pin_gates') is True for x in rows))
        self.assertTrue(any(x.get('path')==record['dependencies_manifest']['path'] and x['kind']=='file' for x in rows))
        self.assertTrue((self.root/'ARCHIVE_REQUESTS/000010000.accepted_receipt.json').is_file())

    def gate_fixture(self):
        common={k:v for k,v in identity().items() if k not in ('run_id','arm')}
        raw=self.root/'synthetic_unit_test_only_raw.txt';raw.write_text('Unit test fixture, not GPU evidence.')
        raw_descriptor=t.descriptor(raw,self.root)
        evidence={}
        for i,name in enumerate(self.cfg['launch_gate_requirements']):
            path=self.root/f'gate_{i}.json'
            t.write_json_exclusive(path,{'schema':'helixwm_final_audited_gate_evidence_v1','gate_name':name,
                'identity':common,'measurement_kind':'actual_gpu_preflight','result':'PASS','raw_evidence':[raw_descriptor]})
            evidence[name]=t.descriptor(path,self.root)
        bridge=self.root/'bridge.json'
        t.write_json_exclusive(bridge,{'schema':'helixwm_final_archive_bridge_probe_v1','identity':common,
            'source_sha256':'e'*64,'result':'PASS','measured_roundtrip':True,'old_ACK_rejected':True})
        gate={'schema':t.GATE_SCHEMA,'identity':common,'authorized_runs':{'D':str(self.root/'fresh_D'),'F':str(self.root/'fresh_F')},
            'evidence':evidence,'measurement_kind':'actual_gpu_preflight','archive_service_source_sha256':'e'*64,
            'archive_bridge_probe':t.descriptor(bridge,self.root),
            'capacity':{'remote_required_bytes_before_start':10**10,'remote_checkpoint_peak_factor':1.2,
                'remote_model_anchor_reserve_bytes':10**9,'remote_episode_write_peak_bytes':10**9,
                'E_required_full_retention_and_ZIP_bytes':10**12,'E_verified_free_bytes':2*10**12,
                'includes_all_calibration_versions_and_full_trajectory_arrays':True,'prediction_is_not_guarantee':True}}
        path=self.root/'gate.json';t.write_json_exclusive(path,gate)
        return path,gate

    def test_launch_gate_external_hash_raw_dependencies_and_scope_are_required(self):
        # Tests only the verifier's byte/identity contract. This fixture must
        # never be used to claim that a real experiment passed GPU preflight.
        path,gate=self.gate_fixture()
        validated=t.validate_launch_gate(path,t.sha_file(path),identity(),self.cfg,self.root/'fresh_D')
        self.assertEqual(validated,gate)
        with self.assertRaises(ValueError):t.validate_launch_gate(path,'f'*64,identity(),self.cfg,self.root/'fresh_D')
        with self.assertRaises(ValueError):t.validate_launch_gate(path,t.sha_file(path),identity(),self.cfg,self.root/'old_D')
        (self.root/'synthetic_unit_test_only_raw.txt').write_text('Changed raw evidence')
        with self.assertRaises(ValueError):t.validate_launch_gate(path,t.sha_file(path),identity(),self.cfg,self.root/'fresh_D')

    def test_launch_gate_rejects_incomplete_preflight_capacity_and_synthetic_marker(self):
        path,gate=self.gate_fixture()
        for mutate in (lambda x:x['capacity'].update(E_verified_free_bytes=1),
                       lambda x:x.update(measurement_kind='synthetic_test_fixture'),
                       lambda x:x['evidence'].pop(next(iter(x['evidence']))),
                       lambda x:x['authorized_runs'].update(F=x['authorized_runs']['D'])):
            bad=copy.deepcopy(gate);mutate(bad);path.write_text(json.dumps(bad))
            with self.assertRaises(ValueError):t.validate_launch_gate(path,t.sha_file(path),identity(),self.cfg,self.root/'fresh_D')

    def periodic_fixture(self, root, *, fault=None, environments=None, controllers=None):
        """Exercise the real periodic orchestrator with explicit CPU fakes only."""
        environments=[] if environments is None else environments
        controllers=[] if controllers is None else controllers
        class EvalEnv(FakeEnv):
            def __init__(self, *args, **kwargs):
                super().__init__(raise_after=3 if fault=='raises_after_advance' else None)
                self.closed=False;environments.append(self)
            def step(self, action):
                image,reward,done,info=super().step(action)
                if self.action_decisions==3:
                    if fault in ('nan','inf','negative_inf'):reward={'nan':float('nan'),'inf':float('inf'),'negative_inf':float('-inf')}[fault]
                    if fault=='high_reward':reward=2.001
                    if fault=='negative_reward':reward=-.001
                    if fault=='timestamp':info['observation_timestamp']=4
                    if fault=='mask':info['availability']=not info['availability']
                    if fault=='nonboolean_mask':info['availability']=1
                    if fault=='reported_calls':info['env_steps_this_step']=1
                    if fault=='actual_calls':self.episode_env_steps+=1
                    if fault=='actual_actions':self.action_decisions+=1
                    if fault=='early_done':done=True
                    if fault=='image_dtype':image=image.astype(np.float32)
                    if fault=='image_shape':image=image[:63]
                if fault=='hidden_missing_image' and not info['availability']:image=np.full((64,64,3),20,np.uint8)
                return image,reward,done,info
            def close(self):self.closed=True
        class EvalController(FakeController):
            def __init__(self,*args,**kwargs):controllers.append(self)
            def act(self,external=None):
                if fault=='nan_action':return np.full(6,np.nan,np.float32)
                if fault=='unbounded_action':return np.full(6,1.01,np.float32)
                return super().act(external)
        backbone=SimpleNamespace(snapshot=lambda:{'frozen':np.asarray([1.],np.float32)})
        heads=SimpleNamespace(sha256='a'*64)
        with patch('helixwm_matched.environment.MatchedWalkerEnv',EvalEnv),patch('helixwm_matched.controllers.Controller',EvalController):
            t._periodic_evaluation(backbone,heads,None,None,self.cfg,root,'D',counts(20000))
        return environments,controllers

    def test_periodic_frozen_cases_have_exact_counts_and_matching_masks(self):
        environments,controllers=self.periodic_fixture(self.root)
        report=json.loads((self.root/'EVAL/000020000/frozen_report.json').read_bytes())
        self.assertEqual(report['ledger'],{'actions':5000,'DMC_calls':10000})
        self.assertEqual(report['training_updates'],0);self.assertEqual(len(report['rows']),10)
        self.assertEqual(report['model_before_sha256'],report['model_after_sha256'])
        self.assertTrue(all(x.closed for x in environments));self.assertEqual(len(controllers),10)
        self.assertTrue(all(row['return']==500. for row in report['rows']))

    def test_periodic_rejects_nonfinite_or_invalid_reward_immediately(self):
        for fault in ('nan','inf','negative_inf','high_reward','negative_reward'):
            with self.subTest(fault=fault):
                root=self.root/fault;environments=[];controllers=[]
                with self.assertRaises(FloatingPointError):
                    self.periodic_fixture(root,fault=fault,environments=environments,controllers=controllers)
                self.assertEqual(environments[0].action_decisions,3)
                self.assertEqual(controllers[0].timestamp,2)
                report=json.loads((root/'EVAL/000020000/failure.json').read_bytes())
                self.assertEqual(report['actual_evaluation_interactions'],{'actions':3,'DMC_calls':6})
                self.assertEqual(report['completed_cases'],0)
                self.assertFalse((root/'EVAL/000020000/frozen_report.json').exists())
                self.assertFalse((root/'eval.jsonl').exists());self.assertTrue(environments[0].closed)

    def test_periodic_rejects_per_step_mask_timestamp_and_budget_offsets(self):
        for fault in ('timestamp','mask','nonboolean_mask','reported_calls','actual_calls','actual_actions','early_done','raises_after_advance'):
            with self.subTest(fault=fault):
                root=self.root/fault;environments=[];controllers=[]
                with self.assertRaises(RuntimeError):
                    self.periodic_fixture(root,fault=fault,environments=environments,controllers=controllers)
                self.assertEqual(controllers[0].timestamp,2)
                report=json.loads((root/'EVAL/000020000/failure.json').read_bytes())
                self.assertEqual(report['actual_evaluation_interactions'],{
                    'actions':4 if fault=='actual_actions' else 3,
                    'DMC_calls':7 if fault=='actual_calls' else 6})
                self.assertFalse((root/'EVAL/000020000/frozen_report.json').exists())
                self.assertTrue(environments[0].closed)

    def test_periodic_rejects_bad_action_before_environment_advance(self):
        for fault in ('nan_action','unbounded_action'):
            with self.subTest(fault=fault):
                root=self.root/fault;environments=[]
                with self.assertRaises(FloatingPointError):self.periodic_fixture(root,fault=fault,environments=environments)
                self.assertEqual(environments[0].action_decisions,0)
                report=json.loads((root/'EVAL/000020000/failure.json').read_bytes())
                self.assertEqual(report['actual_evaluation_interactions'],{'actions':0,'DMC_calls':0})

    def test_periodic_rejects_invalid_or_hidden_image_before_posterior(self):
        for fault in ('image_dtype','image_shape','hidden_missing_image'):
            with self.subTest(fault=fault):
                root=self.root/fault;environments=[];controllers=[]
                with self.assertRaises(ValueError):
                    self.periodic_fixture(root,fault=fault,environments=environments,controllers=controllers)
                report=json.loads((root/'EVAL/000020000/failure.json').read_bytes())
                self.assertFalse((root/'EVAL/000020000/frozen_report.json').exists())
                if fault=='hidden_missing_image':
                    self.assertEqual(report['completed_cases'],5)
                    self.assertEqual(environments[-1].action_decisions,150)
                    self.assertEqual(controllers[-1].timestamp,149)
                else:self.assertEqual(controllers[-1].timestamp,2)

    def test_all_JSON_writers_reject_nan_and_preserve_existing_evidence(self):
        for value in (float('nan'),float('inf'),float('-inf')):
            with self.subTest(value=value):
                exclusive=self.root/'invalid.json';journal=self.root/'invalid.jsonl';status=self.root/'status.json'
                status.write_text('{"preserved":true}')
                for writer,path in ((t.write_json_exclusive,exclusive),(t._append,journal),(t._status,status)):
                    with self.assertRaises(ValueError):writer(path,{'nested':[value]})
                self.assertFalse(exclusive.exists());self.assertFalse(journal.exists())
                self.assertEqual(status.read_text(),'{"preserved":true}')
        journal=self.root/'kept.jsonl';journal.write_bytes(b'{"preserved":true}\n')
        with self.assertRaises(ValueError):t._append(journal,{'bad':float('nan')})
        self.assertEqual(journal.read_bytes(),b'{"preserved":true}\n')

    def test_v3_identity_and_registration_reject_closed_loop_v2_launch(self):
        from helixwm_final import protocol_v2
        self.assertEqual(t.CHECKPOINT_FORMAT,'helixwm_final_bank_episode_boundary_v3')
        with self.assertRaises(ValueError):t.protocol.validate_protocol(protocol_v2.registered_protocol())
        for bad in ({k:v for k,v in identity().items() if k!='bank_manifest_sha256'},
                    {**identity(),'bank_manifest_sha256':'not-a-digest'}):
            with self.assertRaises(ValueError):t.validate_identity(bad)
        self.assertEqual(self.cfg['planned_interaction_accounting']['per_pin_calibration_environment_actions'],0)
        self.assertEqual(self.cfg['planned_interaction_accounting']['one_shared_calibration_bank_actions'],2480)

    def test_checkpoint_dependency_inventory_binds_both_gates_shared_bank_and_exact_trajectory_prefix(self):
        for name in ('SOURCE','BANK_SHARED','BANK_GATES/current','BANK_GATES/previous','REPLAY','TRAJECTORIES/records'):
            (self.root/name).mkdir(parents=True,exist_ok=True)
        for name in ('identity.json','protocol.json','source_manifest.json','environment_lock.json',
                     'SOURCE/source.py','BANK_SHARED/manifest.json','BANK_SHARED/seed55000.npz',
                     'BANK_GATES/current/bank_gate_manifest.json','BANK_GATES/previous/bank_gate_manifest.json',
                     'TRAJECTORIES/header.json','TRAJECTORIES/records/000.json','TRAJECTORIES/records/000.npz','REPLAY/array.npz'):
            (self.root/name).write_bytes(('CPU immutable bytes for '+name).encode())
        trajectory=self.root/'TRAJECTORIES';entry={'metadata_path':'records/000.json','data_path':'records/000.npz',
            'metadata_sha256':t.sha_file(trajectory/'records/000.json'),'data_sha256':t.sha_file(trajectory/'records/000.npz')}
        prefix=t.canonical_json(entry)+b'\n';(trajectory/'manifest.jsonl').write_bytes(prefix+b'not-in-this-checkpoint-tail\n')
        cursor={'record_count':1,'manifest_prefix_bytes':len(prefix),'manifest_prefix_sha256':hashlib.sha256(prefix).hexdigest(),
            'header_sha256':t.sha_file(trajectory/'header.json')}
        replay={'chunks':[{'arrays':{'image':{'path':'array.npz','file_sha256':t.sha_file(self.root/'REPLAY/array.npz')}}}]}
        path=self.root/'deps.jsonl'
        record=t._checkpoint_dependencies(self.root,path,replay,cursor,{'path':'BANK_SHARED'},
            {'path':'BANK_GATES/current'},{'path':'BANK_GATES/previous'})
        self.assertEqual(record,t.descriptor(path,self.root))
        rows=[json.loads(x) for x in path.read_text().splitlines()];indexed={r['path']:r for r in rows}
        self.assertEqual(len(indexed),len(rows))
        for name in ('REPLAY/array.npz','SOURCE/source.py','BANK_SHARED/seed55000.npz',
                     'BANK_GATES/current/bank_gate_manifest.json','BANK_GATES/previous/bank_gate_manifest.json'):
            self.assertEqual(indexed[name]['sha256'],t.sha_file(self.root/name))
        saved=indexed['TRAJECTORIES/manifest.jsonl']
        self.assertEqual(saved['kind'],'stable_byte_prefix');self.assertEqual(saved['bytes'],len(prefix))
        self.assertEqual(saved['sha256'],hashlib.sha256(prefix).hexdigest())
        (trajectory/'records/000.npz').write_bytes(b'changed')
        with self.assertRaises(ValueError):
            t._checkpoint_dependencies(self.root,self.root/'rejected.jsonl',replay,cursor,{'path':'BANK_SHARED'},
                {'path':'BANK_GATES/current'},{'path':'BANK_GATES/previous'})


if __name__=='__main__':unittest.main()
