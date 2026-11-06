"""Synthetic integrated mechanics only; no production calibration or GPU proof."""
import copy
import hashlib
from pathlib import Path
import tempfile
import shutil
import unittest
from unittest.mock import patch

import numpy as np

from helixwm_final.full_controller import FullController
from helixwm_final.epistemic import BootstrapDynamics, FrozenBootstrap, tree_digest
from helixwm_final import deployment_calibration as d
from helixwm_final import observation_guard as og
from helixwm_final.trajectory import TrajectoryArchive
from helixwm_final.belief import canonical_json
from test_final_controller import FakeBackend, calibration, SOURCE, MODEL

IMAGE=np.full((64,64,3),64,np.uint8)


class ImageBackend(FakeBackend):
    def predict_heads(self,state,with_image=False):
        result=super().predict_heads(state,with_image)
        if with_image:result['image_mean']=np.broadcast_to(IMAGE.astype(np.float32)/255,
            (len(state['deter']),64,64,3)).copy()
        return result


def heads_fixture(constant=None):
    model=BootstrapDynamics(734,{'source_sha256':SOURCE,'protocol_sha256':'4'*64})
    for i in range(40):model.update(IMAGE,np.zeros(6),IMAGE,1.,1.,transition_id=i,
        previous_available=True,next_available=True)
    head=model.pin()
    if constant is not None:
        record=head.snapshot();record['coefficients'][:]=0.
        for m,value in enumerate(constant):record['coefficients'][m,0,:16]=value*8.
        head=FrozenBootstrap(record)
    return head


def context(head):
    return dict(source_sha256=SOURCE,model_sha256=MODEL,bank_manifest_sha256='3'*64,
        protocol_sha256='4'*64,collector_sha256='5'*64,epistemic_heads_sha256=head.sha256,
        model_version=3,measurement_kind='synthetic_test_fixture')


def proposal(ctx,**changes):
    # Explicit CPU mechanism proposal, not fitted evidence and not production.
    table=calibration()['thresholds_by_gap']
    for row in table.values():
        row.update(max_epistemic_disagreement=10.,max_temporal_prediction_error=10.,
            max_energy=1.,action_amplitude=1.,slew_limit=2.,kl_limit=.02)
        row.update(changes)
    limits={str(g):{str(h):dict(uncertainty=1.,disagreement=1.,termination_risk=1.,
        epistemic_disagreement=10.) for h in d.HORIZONS} for g in d.GAPS}
    result=dict(schema=d.SCHEMA,schema_version=2,context=copy.deepcopy(ctx),active=False,
        fit_attempt=1,fit_seeds=list(d.FIT_SEEDS),validation_seeds=list(d.VALIDATION_SEEDS),
        planning_operator_version=d.OPERATOR_VERSION,reward_projection=d.REWARD_PROJECTION,
        coverage_revision=copy.deepcopy(d.COVERAGE_REVISION),
        utility_weights=copy.deepcopy(d.UTILITY_WEIGHTS),thresholds_by_gap=table,
        planning_limits_by_gap_horizon=limits)
    result['sha256']=hashlib.sha256(canonical_json(result)).hexdigest()
    return result


def guard_fixture(mode='unknown_observation'):
    binding=dict(source_sha256=SOURCE,model_sha256=MODEL,model_version=3,protocol_sha256='4'*64,
        fit_data_sha256='6'*64,heldout_data_sha256='7'*64)
    record=dict(schema=og.SCHEMA,binding=binding,rules=copy.deepcopy(og.RULES),
        thresholds=dict(dark_mean_max=0.,flat_std_max=0.,noise_high_frequency_min=1.,
            innovation_mse_min=.01,freeze_repeat_min=3),fit_statistics={},heldout_report=None,
        measurement_kind='synthetic_test_fixture')
    return og.ObservationGuard(record,expected_binding=binding,
        expected_calibration_sha256=og.calibration_sha256(record),allow_synthetic_for_tests=True,mode=mode)


def make(*,head=None,guard=None,archive=None,execution='calibration_validation',**limits):
    head=head or heads_fixture();ctx=context(head);backend=ImageBackend()
    candidate=proposal(ctx,**limits) if execution=='calibration_validation' else None
    controller=FullController(backend,head,ctx,seed=17,execution=execution,
        fitted_candidate=candidate,observation_guard=guard,trajectory=archive)
    controller.reset(IMAGE,9)
    return controller,backend


def step(controller,valid,image=IMAGE):
    action=controller.act(external_action=np.zeros(6,np.float32))
    return controller.step(image,action,dict(availability=valid,observation_timestamp=controller.timestamp+1),9)


class FullControllerTests(unittest.TestCase):
    def test_plain_proposal_cannot_activate_production(self):
        head=heads_fixture();ctx=context(head);candidate=proposal(ctx)
        with self.assertRaises(ValueError):FullController(ImageBackend(),head,ctx,deployment=candidate)
        for execution in ('bootstrap','calibration_fit'):
            with self.assertRaises(ValueError):FullController(ImageBackend(),head,ctx,
                fitted_candidate=candidate,execution=execution)
        controller,backend=make()
        self.assertEqual(controller.execution,'calibration_validation')
        self.assertFalse(controller.calibration['active'])
        self.assertTrue(controller.calibration_status()[0])

    def test_actual_backbone_and_auxiliary_identity_checked(self):
        head=heads_fixture();ctx=context(head)
        for key,value in [('model_sha256','f'*64),('model_version',4),('epistemic_heads_sha256','f'*64)]:
            broken=copy.deepcopy(ctx);broken[key]=value
            with self.assertRaises(ValueError):FullController(ImageBackend(),head,broken,execution='bootstrap')

    def test_known_mask_missing_pixels_never_inspected(self):
        class Poison:
            def __array__(self,*args,**kwargs):raise AssertionError('Missing image accessed')
        controller,backend=make();step(controller,False,Poison())
        self.assertEqual(controller.evidence_kind,'PREDICTED_PRIOR')
        self.assertIsNone(controller.current_temporal_error)
        self.assertIsNone(controller.last_prediction['prediction_error'])
        self.assertEqual(len(backend.posteriors),0)

    def test_epistemic_gate_blocks_fusion_before_observation(self):
        controller,backend=make(head=heads_fixture((0.,.5,1.)),max_epistemic_disagreement=.01)
        step(controller,False)
        self.assertGreater(controller.current_epistemic,.01)
        self.assertEqual(controller.last_metrics['fusion_lambda'],0.)
        self.assertIn('independent_model_disagreement',controller.last_metrics['calibration_status'])

    def test_temporal_error_gate_rejects_reconnection_before_fusion(self):
        controller,backend=make(head=heads_fixture((0.,0.,0.)),max_temporal_prediction_error=.01)
        step(controller,False);risk=controller.cumulative_risk
        step(controller,True,np.full_like(IMAGE,255))
        self.assertGreater(controller.current_temporal_error,.01)
        self.assertEqual(controller.last_metrics['fusion_lambda'],0.)
        self.assertGreaterEqual(controller.cumulative_risk,risk)
        self.assertFalse(controller.last_metrics['risk_reset'])
        self.assertEqual(controller.real['deter'][0,0],2.)

    def test_low_ess_does_not_reset_accumulated_risk(self):
        controller,backend=make(min_ess_ratio=.9,temperature=.001)
        step(controller,False);risk=controller.cumulative_risk
        original=backend.score
        def score(states,image):
            if len(states['deter'])==8:return -np.asarray([.01]+[.5]*7)*12288
            return original(states,image)
        backend.score=score
        step(controller,True)
        self.assertGreaterEqual(controller.cumulative_risk,risk)
        self.assertFalse(controller.last_metrics['risk_reset'])
        self.assertEqual(controller.last_metrics['fusion_lambda'],0.)

    def test_gap100_and_recovery_have_no_double_executed_transition(self):
        controller,backend=make()
        for _ in range(100):step(controller,False)
        self.assertEqual(controller.missing_age,100);self.assertIsNotNone(controller.bank)
        self.assertEqual(controller.real['deter'][0,0],100.)
        step(controller,True)
        self.assertEqual(controller.real['deter'][0,0],101.)
        self.assertEqual(controller.bank['deter'][0,0],101.)
        self.assertEqual(controller.timestamp,101)
        self.assertEqual(controller.missing_age,0)
        self.assertTrue(all(np.all(x[:,0]==101) for x in backend.posteriors))

    def test_gap101_expires_candidates(self):
        controller,backend=make()
        for _ in range(101):step(controller,False)
        self.assertIsNone(controller.bank)
        self.assertEqual(controller.real['deter'][0,0],101.)
        self.assertEqual(controller.last_metrics['fusion_lambda'],0.)

    def test_independent_scratch_probe_preserves_all_controller_rngs_and_beliefs(self):
        controller,backend=make();step(controller,False)
        before=tree_digest(controller.snapshot())
        plans,diversity=controller.scratch_plans(np.zeros(6,np.float32),rng=np.random.default_rng(991))
        self.assertEqual({p['measurement']['horizon'] for p in plans},{1,3,5})
        self.assertEqual(before,tree_digest(controller.snapshot()))
        self.assertTrue(all(p['measurement']['scratch_only'] for p in plans))

    def test_snapshot_restores_next_action_and_transition(self):
        controller,backend=make();step(controller,False);snapshot=controller.snapshot()
        restored,_=make();restored.load(snapshot)
        a=controller.act();b=restored.act();np.testing.assert_array_equal(a,b)
        for c,action in ((controller,a),(restored,b)):
            c.step(IMAGE,action,dict(availability=True,observation_timestamp=2),9)
        self.assertEqual(tree_digest(controller.snapshot()),tree_digest(restored.snapshot()))

    def test_unknown_guard_quarantine_happens_before_posterior(self):
        # Detector is a synthetic mechanism fixture; production calibration is not claimed.
        controller,backend=make(guard=guard_fixture(),execution='bootstrap')
        step(controller,True,np.zeros_like(IMAGE))
        self.assertFalse(controller.last_guard['allow_posterior'])
        self.assertEqual(controller.last_guard['status'],'WARNING')
        self.assertEqual(len(backend.posteriors),0)
        self.assertEqual(controller.evidence_kind,'PREDICTED_PRIOR')
        self.assertIsNone(controller.current_temporal_error)
        step(controller,True,IMAGE)
        self.assertEqual(len(backend.posteriors),0)
        step(controller,True,IMAGE)
        self.assertGreater(len(backend.posteriors),0)
        self.assertEqual(controller.real['deter'][0,0],3.)

    def test_guard_snapshot_restores_next_screening_decision(self):
        controller,backend=make(guard=guard_fixture(),execution='bootstrap')
        step(controller,True,np.zeros_like(IMAGE));snapshot=controller.snapshot()
        restored,_=make(guard=guard_fixture(),execution='bootstrap');restored.load(snapshot)
        step(controller,True,IMAGE);step(restored,True,IMAGE)
        self.assertEqual(tree_digest(controller.snapshot()),tree_digest(restored.snapshot()))

    def test_known_guard_passthrough_cannot_replace_mask_or_touch_missing_pixels(self):
        class Poison:
            def __array__(self,*args,**kwargs):raise AssertionError('Masked image inspected by passthrough guard')
        controller,backend=make(guard=guard_fixture('known_mask_passthrough'),execution='bootstrap')
        step(controller,False,Poison())
        self.assertEqual(controller.evidence_kind,'PREDICTED_PRIOR')
        self.assertEqual(len(backend.posteriors),0)
        self.assertEqual(controller.observation_guard.snapshot()['state']['counts']['frames'],0)
        self.assertFalse(controller.action_history[-1]['availability'])

    def test_unknown_detector_does_not_require_privileged_availability_flag(self):
        controller,backend=make(guard=guard_fixture(),execution='bootstrap')
        action=controller.act(external_action=np.zeros(6,np.float32))
        controller.step(np.zeros_like(IMAGE),action,dict(observation_timestamp=1),9)
        self.assertFalse(controller.last_guard['allow_posterior'])
        self.assertEqual(len(backend.posteriors),0)

    def test_invalid_outcome_is_atomic_including_auxiliary_state(self):
        controller,backend=make();action=controller.act(external_action=np.zeros(6,np.float32))
        before=tree_digest(controller.snapshot())
        for info in (dict(availability=1,observation_timestamp=1),dict(availability=True,observation_timestamp=True)):
            with self.assertRaises(ValueError):controller.step(IMAGE,action,info,9)
            self.assertEqual(before,tree_digest(controller.snapshot()))

    def test_nonfinite_integrated_snapshot_rejected_atomically(self):
        controller,backend=make();step(controller,False);before=tree_digest(controller.snapshot())
        for name in ('current_epistemic','current_temporal_error'):
            record=controller.snapshot();record['integration'][name]=float('nan')
            with self.assertRaises(ValueError):controller.load(record)
            self.assertEqual(before,tree_digest(controller.snapshot()))

    def test_archive_keeps_actual_missing_errors_null_and_full_predictions(self):
        root=Path(tempfile.mkdtemp(prefix='full_controller_'))
        archive=TrajectoryArchive.create(root/'trajectory',source_sha256=SOURCE,protocol_sha256='4'*64)
        try:
            controller,backend=make(archive=archive)
            step(controller,False)
            controller.scratch_plans(np.zeros(6,np.float32),rng=np.random.default_rng(48),archive=True)
            self.assertGreaterEqual(archive.record_count,6)
            # Read-back surface rather than file existence is the acceptance gate.
            rows=[archive.load_record(i) for i in range(archive.record_count)]
            actual=[r for r in rows if r['kind']=='actual']
            self.assertEqual(len(actual),1)
            self.assertIsNone(actual[0]['observed_error'])
            self.assertEqual(actual[0]['predicted_observation'].shape,(8,64,64,3))
            self.assertEqual(actual[0]['state']['deter'].shape,(8,4))
            self.assertTrue(all(r['observed_error'] is None for r in rows if r['kind']=='prospective'))
        finally:
            archive.close();shutil.rmtree(root)


class BankControllerTests(unittest.TestCase):
    """Offline-bank synthetic mechanics, never actual GPU deployment proof."""
    @classmethod
    def setUpClass(cls):
        from helixwm_final import bank_gate as g
        from test_final_bank_gate import fixture_row,bundle
        cls.head=heads_fixture();cls.ctx=context(cls.head)
        fit=[fixture_row(s,gap,'fit',cls.ctx) for s in g.FIT_SEEDS for gap in g.GAPS]
        cls.candidate=g.fit_candidate(fit,cls.ctx)
        held=[fixture_row(s,gap,'validation',cls.ctx,cls.candidate['sha256']) for s in g.VALIDATION_SEEDS for gap in g.GAPS]
        cls.temp=tempfile.TemporaryDirectory(prefix='bank_controller_')
        cls.root=Path(cls.temp.name);description,_=bundle(cls.root,cls.ctx,fit,cls.candidate,held)
        cls.handle=g.load_bank_gate(cls.root,description['sha256'],cls.ctx,require_active=False)

    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def controller(self,*,execution='bank_production',gate=None,ctx=None):
        backend=ImageBackend();context_value=self.ctx if ctx is None else ctx
        controller=FullController(backend,self.head,context_value,seed=17,execution=execution,
            bank_gate=gate,fitted_bank_candidate=copy.deepcopy(self.candidate) if execution=='bank_validation' else None)
        controller.reset(IMAGE,9)
        return controller,backend

    def test_bank_namespace_rejects_old_certificate_raw_dict_and_unknown_guard(self):
        with self.assertRaises(ValueError):self.controller(gate=dict(self.handle))
        with self.assertRaises(ValueError):FullController(ImageBackend(),self.head,self.ctx,
            execution='bank_production',deployment=proposal(self.ctx))
        with self.assertRaises(ValueError):FullController(ImageBackend(),self.head,self.ctx,
            execution='bank_production',observation_guard=guard_fixture())
        with self.assertRaises(ValueError):FullController(ImageBackend(),self.head,self.ctx,
            execution='production',bank_gate=self.handle)
        controller,_=self.controller(execution='bank_validation')
        self.assertTrue(controller.calibration_status()[0])
        self.assertIn('not_closed_loop',controller.calibration_status()[1])
        self.assertEqual(controller.snapshot()['format'],'helixwm_final_integrated_bank_controller_v3')

    def test_missing_inactive_and_valid_stale_gate_use_native_prior_without_hidden_pixels(self):
        class Hidden:
            def __array__(self,*args,**kwargs):raise AssertionError('Hidden missing frame was read')
        for gate,ctx in ((None,self.ctx),(self.handle,self.ctx),(self.handle,{**self.ctx,'bank_manifest_sha256':'f'*64})):
            with self.subTest(gate=gate is not None,stale=ctx!=self.ctx):
                controller,backend=self.controller(gate=gate,ctx=ctx)
                self.assertFalse(controller.calibration_status()[0])
                action=controller.act(external_action=np.zeros(6,np.float32))
                controller.step(Hidden(),action,{'availability':False,'observation_timestamp':1},9)
                self.assertEqual(controller.last_metrics['fusion_lambda'],0.)
                action=controller.act()
                self.assertEqual(controller.last_action_decision['kind'],'prior_only_fallback')
                self.assertFalse(controller.last_action_decision['planning'])
                self.assertEqual(backend.posteriors,[])

    def test_modified_handle_fails_instead_of_becoming_silent_fallback(self):
        controller,_=self.controller(gate=self.handle)
        controller._calibration['active']=True
        with self.assertRaises(ValueError):controller.act()
        self.assertIsNone(controller.issued_action)

    def test_support_queries_match_each_pre_action_head_state(self):
        from helixwm_final.bank_gate import support_query
        controller,_=self.controller(execution='bank_validation')
        before=tree_digest(controller.snapshot())
        plans,_=controller.scratch_plans(np.zeros(6,np.float32),rng=np.random.default_rng(3))
        for plan in plans:
            row=plan['measurement'];queries=row['support_queries']
            aux=np.repeat(controller.auxiliary_state,8,axis=1)
            for depth,action in enumerate(row['actions']):
                np.testing.assert_array_equal(queries[depth],support_query(aux,action))
                aux,_=controller.heads.advance(aux,action)
        self.assertEqual(before,tree_digest(controller.snapshot()))

    def test_one_unsupported_particle_or_depth_rejects_whole_plan(self):
        controller,_=self.controller(execution='bank_validation');controller.missing_age=1
        stats=dict(uncertainty=0.,disagreement=0.,termination_risk=0.,epistemic_disagreement=0.,utility=1.)
        limits=controller._calibration['thresholds_by_gap']['1']
        for out_of_support in (False,True):
            queries=np.zeros((5,8,22),np.float64)
            if out_of_support:queries[-1,-1,0]=100.
            row={'measurement':{'horizon':5,'proposal_id':0,'support_queries':queries},
                'statistics':dict(stats),'first_action':np.full(6,.2,np.float32)}
            with patch.object(controller,'_trusted',return_value=True),patch.object(controller,'scratch_plans',return_value=([row],1.)):
                _,decision=controller._plan_action(np.zeros(6,np.float32),limits)
            if out_of_support:self.assertEqual(decision['reason'],'all_plans_rejected')
            else:
                self.assertEqual(decision['decision_label'],'EXPERIMENTAL_SUPPORTED_PLAN')
                self.assertFalse(decision['physical_safety_proven'])

    def test_completed_boundary_restores_all_seven_rngs_and_reanchors_next_episode(self):
        first,_=self.controller()
        for _ in range(500):step(first,True)
        first.screen_rng.random(13)
        record=first.snapshot()
        left,left_backend=self.controller();right,right_backend=self.controller()
        for controller in (left,right):
            audit=controller.restore_completed_episode(copy.deepcopy(record))
            self.assertEqual(audit['rng_streams_restored'][-1],'screen')
            self.assertFalse(controller.initialized);self.assertIsNone(controller.auxiliary_state)
            self.assertEqual(tree_digest(controller.screen_rng.bit_generator.state),tree_digest(first.screen_rng.bit_generator.state))
            controller.reset(IMAGE,10)
        np.testing.assert_array_equal(left.act(),right.act())
        self.assertEqual(tree_digest(left.snapshot()),tree_digest(right.snapshot()))

    def test_completed_boundary_rejects_incomplete_foreign_integrated_state_atomically(self):
        first,_=self.controller()
        for _ in range(500):step(first,True)
        record=first.snapshot();receiver,_=self.controller();before=tree_digest(receiver.snapshot())
        variants=[]
        for mutate in (lambda x:x.pop('integration'),lambda x:x['integration'].pop('screen_rng'),
                       lambda x:x['integration']['context'].update(protocol_sha256='f'*64),
                       lambda x:x['integration'].update(current_epistemic=float('nan'))):
            broken=copy.deepcopy(record);mutate(broken);variants.append(broken)
        for broken in variants:
            with self.assertRaises(ValueError):receiver.restore_completed_episode(broken)
            self.assertEqual(before,tree_digest(receiver.snapshot()))


if __name__=='__main__':unittest.main()
