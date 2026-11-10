"""Drop-in repo CPU adapter regression; synthetic backend, no GPU performance claim.

This file is staged outside the repo until the separately audited evaluator is
adopted. It does not depend on the old broken evaluator or its keyword bug.
"""
import copy
import json
import random
import unittest
import numpy as np
from helixwm_final import final_eval as e,protocol_v3 as protocol
from helixwm_matched.controllers import Controller
from helixwm_matched.recovery_probe import tree_hash
from test_final_action_evidence import NativeBackend
from test_final_full_controller import make
from test_final_evaluator import Env


class Backend(NativeBackend):
    def prior(self,previous,action,rng=None):return self.advance_prior(previous,action,rng=rng)[0]


class ActualEvaluatorContractTests(unittest.TestCase):
    def planned(self,name,gap,arm):
        return next(x for x in e.build_plan(protocol.registered_protocol())
                    if (x['controller'],x['gap'],x['arm'])==(name,gap,arm))

    def compare_direct_policy(self,factory,planned):
        # Independent direct rollout contains no diagnostic calls. Equality of
        # actual snapshots/private RNG proves evaluation observations are passive.
        reference=factory();env=Env();mask=(np.ones(501,np.bool_) if planned['gap']==0
                                          else protocol.fixed_gap_mask(planned['gap']))
        initial=env.reset(seed=planned['seed'],mask=mask);reference.reset(initial,episode_id=planned['seed'])
        actions=[];rewards=[]
        for _ in range(500):
            action=reference.act();image,reward,done,info=env.step(action)
            reference.step(image,action,info,episode_id=planned['seed'])
            actions.append(action);rewards.append(reward)
        env.close()
        actual=factory();host=(copy.deepcopy(random.getstate()),copy.deepcopy(np.random.get_state()))
        result=e.evaluate_case(actual,planned,Env())
        self.assertEqual(tree_hash(reference.snapshot()),tree_hash(actual.snapshot()))
        self.assertEqual(tree_hash(host),tree_hash((random.getstate(),np.random.get_state())))
        np.testing.assert_array_equal(result['trace']['action'],np.asarray(actions,np.float32))
        np.testing.assert_array_equal(result['trace']['reward'],np.asarray(rewards,np.float64))
        self.assertEqual(result['counts'],{'actions':500,'dmc_calls':1000})
        self.assertEqual(len(result['events']),500);json.dumps(result['events'],allow_nan=False)
        return result

    def test_actual_native_adapter_clean_black_and_prior_complete_500_actions(self):
        for name,gap in (('clean_native',0),('black_frame_native',20),('prior_only',100)):
            with self.subTest(controller=name):
                result=self.compare_direct_policy(lambda:Controller(Backend(),name,seed=60000),self.planned(name,gap,'D'))
                for event in result['events']:
                    m=event['module_evidence'];self.assertFalse(m['action_selection']['full_controller'])
                    self.assertIsNone(m['observation_update']['fusion_lambda'])

    def test_actual_full_adapter_inactive_and_candidate_complete_500_actions(self):
        for mode in ('bank_production','calibration_validation'):
            with self.subTest(mode=mode):
                result=self.compare_direct_policy(lambda:make(execution=mode,max_epistemic_disagreement=100.,
                    max_temporal_prediction_error=100.)[0],self.planned('helix_final',20,'F'))
                for event in result['events']:
                    m=event['module_evidence'];selection=m['action_selection'];post=m['observation_update']
                    self.assertTrue(selection['full_controller']);self.assertTrue(selection['model_sha256'])
                    self.assertEqual(selection['conditioning_observation_timestamp'],event['timestamp']-1)
                    self.assertEqual(post['observation_timestamp'],event['timestamp'])
                    self.assertIn('planning',selection['decision'])
                if mode=='calibration_validation':
                    self.assertTrue(any(x['module_evidence']['action_selection']['planning_accepted'] for x in result['events']))

    def test_real_native_adapter_invalid_reward_stops_before_posterior(self):
        backend=Backend();controller=Controller(backend,'black_frame_native',seed=60000)
        with self.assertRaisesRegex(ValueError,'reward') as caught:
            e.evaluate_case(controller,self.planned('black_frame_native',20,'D'),Env('reward_nan'))
        self.assertEqual(caught.exception.evaluation_partial['counts'],{'actions':1,'dmc_calls':2})
        self.assertEqual(backend.posteriors,[])


if __name__=='__main__':unittest.main()
