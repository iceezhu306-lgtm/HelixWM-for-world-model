"""CPU observation-only logging checks, not GPU or production activation proof."""
import copy
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from helixwm_final import train as t
from helixwm_matched.controllers import Controller
from helixwm_matched.recovery_probe import tree_hash
from test_final_train import FakeEnv, FakeReplay, FakeAuxiliary, FakeController, counts
from test_final_full_controller import make, ImageBackend


class NativeBackend(ImageBackend):
    """Mechanism fixture satisfying the unchanged native collector interface."""
    def observe(self, previous, action, image, is_first=False, rng=None):
        if is_first: return super().observe(previous,action,image,is_first=True,rng=rng)
        prior,_ = self.advance_prior(previous,action,rng=rng)
        posterior,_ = self.posterior_from_deter(prior['deter'],image,rng=rng)
        return posterior


class ActionEvidenceTests(unittest.TestCase):
    def setUp(self):
        parent=Path(__file__).resolve().parents[2]/'preflight'/'action_evidence_contract_tests'
        parent.mkdir(parents=True,exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(dir=parent);self.root=Path(self.temp.name)
    def tearDown(self): self.temp.cleanup()

    def execute(self, controller, *, logging, start=5500, arm='F', mask=None, replay=None, env=None, sink=None):
        env=FakeEnv() if env is None else env
        replay=FakeReplay() if replay is None else replay
        auxiliary=FakeAuxiliary();c=counts(start);rng=np.random.default_rng(810000)
        mask=np.ones(501,np.bool_) if mask is None else mask
        rows=[];path=self.root/f'actions_{len(list(self.root.glob("actions_*")))}.jsonl'
        def append(row):
            t._append(path,row); rows.append(copy.deepcopy(row))
            if sink is not None:sink(row)
        random.seed(20261109);np.random.seed(20261109)
        result=t.run_episode(env=env,controller=controller,replay=replay,auxiliary=auxiliary,
            prefill_rng=rng,counts=c,episode=start//500,seed=100000+start//500,mask=mask,
            on_interaction=append if logging else None,arm=arm)
        snapshot=controller.snapshot() if hasattr(controller,'snapshot') else controller.__dict__
        backend=getattr(controller,'backend',getattr(getattr(controller,'chain',None),'backend',None))
        state={'controller':snapshot,'prefill_rng':rng.bit_generator.state,
            'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),
            'actions':env.actions,'replay_rows':replay.rows,'auxiliary_rows':auxiliary.rows,
            'counts':c,'updates':replay.updates,'result':result,
            'backend_calls':None if backend is None else {'actor_calls':backend.actions,
                'prior_inputs':backend.transitions,'posterior_inputs':backend.posteriors}}
        return state,rows,path

    def compare_logging(self,factory,**kwargs):
        plain=self.execute(factory(),logging=False,**kwargs)
        logged=self.execute(factory(),logging=True,**kwargs)
        # All six controller streams plus screen_rng are part of the full
        # snapshot; this also checks replay/auxiliary inputs and every action.
        self.assertEqual(tree_hash(plain[0]),tree_hash(logged[0]))
        self.assertEqual(len(logged[1]),500)
        parsed=[json.loads(x) for x in logged[2].read_text().splitlines()]
        self.assertEqual(parsed,logged[1])
        self.assertEqual([r['agent_steps'] for r in parsed],list(range(kwargs.get('start',5500)+1,kwargs.get('start',5500)+501)))
        self.assertEqual([r['timestamp'] for r in parsed],list(range(1,501)))
        for row in parsed:
            m=row['module_evidence'];self.assertTrue(m['interaction_complete'])
            self.assertEqual(m['schema'],'helixwm_final_action_module_evidence_v1')
            self.assertEqual(m['action_selection']['conditioning_observation_timestamp'],row['timestamp']-1)
            self.assertEqual(m['observation_update']['observation_timestamp'],row['timestamp'])
        return logged

    def test_full_inactive_bank_gap100_logging_preserves_all_actions_counts_and_rng(self):
        mask=np.ones(501,np.bool_);mask[20:120]=False
        state,rows,_=self.compare_logging(lambda:make(execution='bank_production')[0],mask=mask)
        for row in rows:
            before=row['module_evidence']['action_selection'];after=row['module_evidence']['observation_update']
            self.assertTrue(before['full_controller']);self.assertFalse(before['gate_active'])
            self.assertFalse(before['planning_attempted']);self.assertFalse(before['planning_accepted'])
            self.assertEqual(before['conditioning_observation_available'],bool(mask[row['timestamp']-1]))
            self.assertEqual(before['prior_only_fallback'],not bool(mask[row['timestamp']-1]))
            self.assertEqual(after['fusion_lambda'],0.);self.assertFalse(after['fusion_accepted'])
            self.assertFalse(after['gate_active'])
        # The return-to-visible action was selected while still blind. Its
        # posterior belongs to the subsequent observation, not that selection.
        recovery=rows[119]['module_evidence']
        self.assertTrue(recovery['action_selection']['prior_only_fallback'])
        self.assertTrue(recovery['observation_update']['observation_restored'])
        self.assertTrue(recovery['observation_update']['observation_available'])
        self.assertEqual(state['counts'],counts(6000))

    def test_full_candidate_mechanism_logs_planning_and_fusion_without_rng_changes(self):
        # This permissive synthetic calibration mode cannot authenticate a
        # production bank. It exercises actually executed planning branches.
        mask=np.ones(501,np.bool_);mask[20:25]=False
        _,rows,_=self.compare_logging(lambda:make(max_temporal_prediction_error=100.,
            max_epistemic_disagreement=100.)[0],mask=mask)
        attempted=[r for r in rows if r['module_evidence']['action_selection']['planning_attempted']]
        self.assertTrue(attempted)
        self.assertTrue(any(r['module_evidence']['action_selection']['planning_accepted'] for r in rows))
        self.assertTrue(any(r['module_evidence']['observation_update']['fusion_accepted'] for r in rows))
        for row in attempted:
            selection=row['module_evidence']['action_selection']
            self.assertEqual(selection['execution'],'calibration_validation')
            self.assertEqual(selection['planning_accepted'],selection['decision']['planning']['reason']=='accepted_plan')
        for row in rows:
            after=row['module_evidence']['observation_update']
            self.assertEqual(after['fusion_accepted'],after['fusion_lambda']>0)

    def test_native_D_fields_are_not_final_activation_and_prefill_remains_external(self):
        mask=np.ones(501,np.bool_);mask[20:25]=False
        for start in (0,5000):
            with self.subTest(start=start):
                _,rows,_=self.compare_logging(lambda:Controller(NativeBackend(),'black_frame_native',seed=0),
                    arm='D',start=start,mask=mask)
                for row in rows:
                    m=row['module_evidence'];selection=m['action_selection'];post=m['observation_update']
                    self.assertEqual(m['arm'],'D');self.assertFalse(selection['full_controller'])
                    self.assertFalse(selection['final_action_modules_applicable'])
                    self.assertEqual(selection['controller_name'],'black_frame_native')
                    for name in ('gate_active','planning_attempted','planning_accepted','prior_only_fallback'):
                        self.assertIsNone(selection[name])
                    self.assertIsNone(post['fusion_lambda']);self.assertIsNone(post['fusion_accepted'])
                    self.assertEqual(selection['decision']['kind'],'external_prefill' if start==0 else 'native_policy')

    def test_detached_decision_record_does_not_alias_live_controller(self):
        controller,_=make();controller.reset(np.full((64,64,3),64,np.uint8),11)
        controller.act()
        before=tree_hash(controller.snapshot())
        record=t._module_selection(controller,timestamp=1,available=True,external=False)
        record['decision']['kind']='altered in an external consumer'
        self.assertEqual(tree_hash(controller.snapshot()),before)

    def test_nonfinite_action_decision_stops_before_environment_action(self):
        for bad in (float('nan'),float('inf'),np.array([0.,float('-inf')])):
            controller,_=make();env=FakeEnv();original=controller.act
            def injected(*args,**kwargs):
                result=original(*args,**kwargs);controller.last_action_decision['diagnostic']=bad;return result
            with patch.object(controller,'act',side_effect=injected),self.assertRaises(FloatingPointError):
                self.execute(controller,logging=True,env=env)
            self.assertEqual(env.action_decisions,0)

    def test_invalid_post_observation_diagnostic_records_partial_not_fabricated_fusion(self):
        for bad in (float('nan'),float('inf')):
            controller,_=make();env=FakeEnv();original=controller.step;rows=[]
            def injected(*args,**kwargs):
                result=original(*args,**kwargs);controller.last_metrics['fusion_lambda']=bad;return result
            with patch.object(controller,'step',side_effect=injected),self.assertRaises(FloatingPointError):
                self.execute(controller,logging=True,env=env,sink=rows.append)
            self.assertEqual(env.action_decisions,1);self.assertEqual(len(rows),1)
            evidence=rows[0]['module_evidence']
            self.assertFalse(evidence['interaction_complete']);self.assertIsNone(evidence['observation_update'])
            self.assertEqual(evidence['failure_stage'],'observation_evidence_validation')

    def test_learner_failure_preserves_executed_action_and_original_error_if_logging_fails(self):
        rows=[];env=FakeEnv();replay=FakeReplay(fail_metric=True)
        def sink(row):
            rows.append(row)
            if not row['module_evidence']['interaction_complete']:raise OSError('synthetic evidence disk failure')
        with self.assertRaisesRegex(FloatingPointError,'Nonfinite author loss') as caught:
            self.execute(FakeController(),logging=True,env=env,replay=replay,sink=sink)
        self.assertEqual(env.action_decisions,4);self.assertEqual(len(rows),4)
        self.assertEqual(rows[-1]['module_evidence']['failure_stage'],'learner_update')
        self.assertIsNotNone(rows[-1]['module_evidence']['observation_update'])
        self.assertTrue(any('Secondary interaction evidence write failed' in x for x in caught.exception.__notes__))

    def test_environment_and_observe_failures_preserve_only_known_outcome_scope(self):
        for source in ('environment','observe'):
            with self.subTest(source=source):
                env=FakeEnv(raise_after=2 if source=='environment' else None);controller=FakeController();rows=[]
                original=controller.step
                def step(*args,**kwargs):
                    if args[2]['observation_timestamp']==2:raise RuntimeError('synthetic observe failure')
                    return original(*args,**kwargs)
                with patch.object(controller,'step',side_effect=step if source=='observe' else original):
                    with self.assertRaises(RuntimeError):
                        self.execute(controller,logging=True,env=env,sink=rows.append)
                self.assertEqual(env.action_decisions,2);self.assertEqual(len(rows),2)
                row=rows[-1];self.assertIsNone(row['module_evidence']['observation_update'])
                self.assertFalse(row['module_evidence']['interaction_complete'])
                if source=='environment':
                    self.assertIsNone(row['reward']);self.assertIsNone(row['availability'])
                else:
                    self.assertEqual(row['reward'],1.);self.assertTrue(row['availability'])


if __name__=='__main__':unittest.main()
