"""CPU case/matrix contracts; no author model, GPU, or actual final evaluation."""
import copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from helixwm_final import final_eval as e
from helixwm_final import protocol_v3 as p

class Policy:
    def __init__(self,bad_action=False):self.bad_action=bad_action;self.received=[]
    def reset(self,image,episode_id):self.received.append(image.copy())
    def act(self,deterministic=False):return np.full(6,np.nan if self.bad_action else 0,np.float32)
    def step(self,image,action,info,episode_id):self.received.append(image.copy())

class Env:
    def __init__(self,defect=None):self.defect=defect;self.closed=False;self.rendered=0
    def reset(self,seed,mask):
        self.action_decisions=self.episode_env_steps=0;self.mask=mask.copy()
        return np.full((64,64,3),80,np.uint8)
    def step(self,action):
        self.action_decisions+=1;self.episode_env_steps+=2
        t=self.action_decisions;available=bool(self.mask[t])
        image=np.full((64,64,3),80 if available else 0,np.uint8)
        reward=1.;done=t==500
        info={'availability':available,'observation_timestamp':t,'env_steps_this_step':2,'executed_action':action.copy()}
        if self.defect=='reward_nan':reward=float('nan')
        if self.defect=='reward_inf':reward=float('inf')
        if self.defect=='reward_range':reward=2.1
        if self.defect=='mask':info['availability']=not available
        if self.defect=='time':info['observation_timestamp']+=1
        if self.defect=='calls':self.episode_env_steps+=1
        if self.defect=='action':info['executed_action'][0]=1
        if self.defect=='done':done=True
        if self.defect=='hidden' and not available:image[:]=80
        return image,reward,done,info
    def render_diagnostic(self,size):self.rendered+=1;return np.full((size,size,3),200,np.uint8)
    def close(self):self.closed=True

class FinalEvaluatorTests(unittest.TestCase):
    def setUp(self):
        self.config=p.registered_protocol()
        self.planned=next(x for x in e.build_plan(self.config) if x['arm']=='F' and x['gap']==20)
    def test_fixed_plan_cardinality_strata_video_and_pairing(self):
        plan=e.build_plan(self.config)
        self.assertEqual(len(plan),1520)
        self.assertEqual(sum(x['actions'] for x in plan),760000)
        self.assertEqual(sum(x['stratum']=='core' for x in plan),1280)
        self.assertEqual(sum(x['stratum']=='long_gap100_stress' for x in plan),240)
        self.assertEqual({(x['arm'],x['controller']) for x in plan if x['record_video_frames']},set(e.VIDEO_PANELS))
        for gap in (1,5,10,20,50,100):
            self.assertEqual(len({x['availability_sha256'] for x in plan if x['gap']==gap}),1)
    def test_complete_episode_retains_original_reward_bytes_and_counts(self):
        env=Env();out=e.evaluate_case(Policy(),self.planned,env)
        self.assertEqual(out['counts'],{'actions':500,'dmc_calls':1000})
        self.assertEqual(out['episode_return'],500.)
        self.assertEqual(out['trace']['reward'].dtype,np.float64)
        self.assertTrue(env.closed);self.assertEqual(env.rendered,0)
    def test_invalid_outcome_stops_before_policy_update(self):
        for defect in ('reward_nan','reward_inf','reward_range','mask','time','calls','action','done'):
            with self.subTest(defect=defect):
                env=Env(defect);policy=Policy()
                with self.assertRaises(ValueError) as caught:e.evaluate_case(policy,self.planned,env)
                self.assertEqual(len(policy.received),1)
                self.assertEqual(caught.exception.evaluation_partial['counts']['actions'],1)
                self.assertEqual(caught.exception.evaluation_partial['counts']['dmc_calls'],3 if defect=='calls' else 2)
                self.assertTrue(env.closed)
    def test_nonfinite_action_consumes_no_environment_calls(self):
        env=Env()
        with self.assertRaises(ValueError) as caught:e.evaluate_case(Policy(True),self.planned,env)
        self.assertEqual(caught.exception.evaluation_partial['counts'],{'actions':0,'dmc_calls':0})
    def test_hidden_missing_pixels_rejected(self):
        env=Env('hidden');policy=Policy()
        with self.assertRaises(ValueError):e.evaluate_case(policy,self.planned,env)
        self.assertEqual(env.action_decisions,150)
        self.assertEqual(len(policy.received),150)
    def test_video_diagnostics_never_reach_model(self):
        env=Env();policy=Policy();frames=[]
        e.evaluate_case(policy,self.planned,env,render_callback=lambda frame,t,info:frames.append((frame,t,info)))
        self.assertEqual(len(frames),501)
        self.assertEqual(env.rendered,501)
        self.assertTrue(all(np.max(x)<=80 for x in policy.received))
        self.assertTrue(np.all(policy.received[150]==0));self.assertTrue(np.all(frames[150][0]==200))
    def test_mask_commitment_tampering_consumes_no_actions(self):
        row=copy.deepcopy(self.planned);row['availability_sha256']='0'*64
        env=Env()
        with self.assertRaises(ValueError) as caught:e.evaluate_case(Policy(),row,env)
        self.assertTrue(env.closed)
        self.assertEqual(caught.exception.evaluation_partial['counts'],{'actions':0,'dmc_calls':0})

    def test_environment_close_cannot_hide_original_failure_or_true_counts(self):
        class BadClose(Env):
            def close(self):raise OSError('secondary close failure')
        with self.assertRaisesRegex(ValueError,'reward nonfinite') as caught:
            e.evaluate_case(Policy(),self.planned,BadClose('reward_nan'))
        self.assertEqual(caught.exception.evaluation_partial['counts'],{'actions':1,'dmc_calls':2})
        self.assertEqual(caught.exception.evaluation_cleanup_errors[0]['operation'],'close_environment')
        with self.assertRaisesRegex(OSError,'secondary close failure') as caught:
            e.evaluate_case(Policy(),self.planned,BadClose())
        self.assertEqual(caught.exception.evaluation_partial['counts'],{'actions':500,'dmc_calls':1000})


class FinalEvaluatorWiringTests(unittest.TestCase):
    """Actual constructor/file wiring with a synthetic backend, not GPU proof."""
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='final_eval_')
        self.root=Path(self.temp.name);self.config=p.registered_protocol()
        self.inputs={arm:{'path':str(self.root/arm/'CHECKPOINTS'/'checkpoint_001000000.pkl'),
            'checkpoint_sha256':a*64,'record_sha256':b*64,'identity':{
                'run_id':'new_'+arm,'arm':arm,'source_sha256':'1'*64,
                'protocol_sha256':p.protocol_sha256(self.config),'environment_lock_sha256':'2'*64,
                'upstream_commit':self.config['official_core']['commit'],'bank_manifest_sha256':'3'*64}}
            for arm,a,b in (('D','4','5'),('F','6','7'))}
    def tearDown(self):self.temp.cleanup()
    def validate(self,value,output=None):
        return e.validate_inputs(value,self.config,runtime_source_sha256='1'*64,
            environment_lock_sha256='2'*64,output_new=output or self.root/'evaluation')
    def test_manifest_binds_arm_distinct_lineage_and_entire_input_run_isolation(self):
        self.assertEqual(self.validate(self.inputs),self.inputs)
        mutations=(lambda x:x['D']['identity'].update(arm='F'),
            lambda x:x['F']['identity'].update(run_id=x['D']['identity']['run_id']),
            lambda x:x['F'].update(path=x['D']['path']),
            lambda x:x['F'].update(checkpoint_sha256=x['D']['checkpoint_sha256']),
            lambda x:x['F'].update(record_sha256=x['D']['record_sha256']),
            lambda x:x['F']['identity'].update(bank_manifest_sha256='f'*64),
            lambda x:x['F'].update(path='relative/checkpoint.pkl'),
            lambda x:x['F'].pop('record_sha256'))
        for mutate in mutations:
            broken=copy.deepcopy(self.inputs);mutate(broken)
            with self.subTest(mutate=mutate),self.assertRaises(ValueError):self.validate(broken)
        for output in (self.root/'D',self.root/'D'/'EVAL',self.root/'F'/'TRAJECTORIES',self.root):
            with self.subTest(output=output),self.assertRaises(ValueError):self.validate(self.inputs,output)
    def test_loader_passes_exact_trusted_sidecar_identity_and_final_budget_gate(self):
        sentinel=object()
        with patch('helixwm_final.final_checkpoint.load_checkpoint',return_value=sentinel) as loader:
            result=e.load_final_input(self.inputs['F'],self.config)
        self.assertIs(result,sentinel)
        item=self.inputs['F']
        loader.assert_called_once_with(item['path'],expected_sha256=item['checkpoint_sha256'],
            expected_record_sha256=item['record_sha256'],expected_identity=item['identity'],
            protocol_config=self.config,require_final=True)
    def test_strict_reload_allows_only_declared_logdir_relocation(self):
        saved={'resolved_config':{'logdir':str(self.root/'original'),'seed':0,'batch_size':16},
            'training':{'parameter':np.arange(4,dtype=np.float32)},'counter':8}
        original=copy.deepcopy(saved);target=self.root/'evaluation'/'MODEL'
        class Backend:
            def __init__(self,change=None):self.change=change
            def load(self,state):
                self.state=copy.deepcopy(state);self.state['resolved_config']['logdir']=str(target)
                if self.change:self.change(self.state)
            def snapshot(self):return copy.deepcopy(self.state)
        result=e.strict_reload_backbone(Backend(),saved,output_dir=target)
        self.assertTrue(result['exact_state_and_config_except_logdir'])
        self.assertEqual(saved['resolved_config'],original['resolved_config'])
        np.testing.assert_array_equal(saved['training']['parameter'],original['training']['parameter'])
        for change in (lambda x:x['resolved_config'].update(seed=9),
                       lambda x:x.update(counter=9),lambda x:x['training']['parameter'].__setitem__(0,99),
                       lambda x:x['resolved_config'].update(logdir=str(self.root/'wrong')),
                       lambda x:x['resolved_config'].pop('batch_size')):
            with self.subTest(change=change),self.assertRaises(ValueError):
                e.strict_reload_backbone(Backend(change),saved,output_dir=target)
    def test_actual_bank_constructor_path_rejects_dict_and_accepts_verified_inactive_handle(self):
        from test_final_full_controller import heads_fixture,context,ImageBackend,IMAGE
        from test_final_bank_gate import fixture_row,bundle
        from helixwm_final import bank_gate as g
        heads=heads_fixture();ctx=context(heads)
        fit=[fixture_row(seed,gap,'fit',ctx) for seed in g.FIT_SEEDS for gap in g.GAPS]
        candidate=g.fit_candidate(fit,ctx)
        held=[fixture_row(seed,gap,'validation',ctx,candidate['sha256']) for seed in g.VALIDATION_SEEDS for gap in g.GAPS]
        bundle_dir=self.root/'gate';bundle_dir.mkdir()
        manifest,_=bundle(bundle_dir,ctx,fit,candidate,held)
        gate=g.load_bank_gate(bundle_dir,manifest['sha256'],ctx,require_active=False)
        planned={'controller':'helix_final','seed':60000};active=self.root/'case';active.mkdir()
        controller,trajectory=e.make_case_controller(ImageBackend(),heads,gate,planned,
            output_dir=active,source_sha256=ctx['source_sha256'])
        try:
            self.assertEqual(controller.execution,'bank_production')
            self.assertFalse(controller.calibration_status()[0])
            controller.reset(IMAGE,60000)
            self.assertEqual(controller.act().shape,(6,))
        finally:trajectory.close()
        bad=self.root/'bad_case';bad.mkdir()
        with self.assertRaises(ValueError):e.make_case_controller(ImageBackend(),heads,dict(gate),planned,
            output_dir=bad,source_sha256=ctx['source_sha256'])
    def test_secondary_video_and_cleanup_failure_preserves_primary_failure_record(self):
        class Broken:
            frames=[1]
            def save(self,*args,**kwargs):raise OSError('partial video failed')
            def close(self):raise OSError('trajectory close failed')
        case=self.root/'case';case.mkdir();primary=ValueError('original invalid reward')
        primary.evaluation_partial={'counts':{'actions':3,'dmc_calls':6}}
        with self.assertRaises(ValueError) as caught:
            try:raise primary
            except BaseException as error:
                e.record_failure(self.root,error,{'completed_cases':0,'actions':3,'dmc_calls':6},
                    active=case,frames=Broken(),trajectory=Broken())
                raise
        self.assertIs(caught.exception,primary)
        report=json.loads((self.root/'failure.json').read_bytes())
        self.assertIn('original invalid reward',report['error'])
        self.assertEqual({r['operation'] for r in report['cleanup_errors']},{'save_partial_video','close_trajectory'})
        self.assertEqual(report['actions'],3);self.assertEqual((self.root/'evaluation.exit').read_text(),'1\n')

if __name__=='__main__':unittest.main()
