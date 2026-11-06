"""Pure CPU numerical/operator checks, never evidence of model performance."""
import copy
import unittest
from fractions import Fraction

import numpy as np

from helixwm_final import planning as p


def inputs(h=3, k=8):
    return dict(rewards=np.ones((h,k), np.float64),
        continuations=np.full((h,k), .5, np.float64),
        uncertainty=np.full(h,.3,np.float64), disagreement=np.full(h,.1,np.float64),
        actions=np.zeros((h,k,6), np.float64), previous_action=np.zeros(6,np.float64),
        weights=np.full(k,1/k,np.float64))


class PlanningTests(unittest.TestCase):
    def test_survival_before_reward_and_two_call_horizon_units(self):
        value = p.rollout_statistics(**inputs())
        exact = sum(Fraction(1,2)**h for h in range(3))
        self.assertEqual(value['predicted_return'], float(exact))
        self.assertEqual(value['reward_normalized'], float(exact/Fraction(6)))
        self.assertEqual(value['termination_risk'], float(1-Fraction(1,2)**3))

    def test_zero_reward_is_finite_and_uses_no_fitted_denominator(self):
        raw=inputs(5); raw['rewards'][:]=0; raw['continuations'][:]=1
        value=p.rollout_statistics(**raw)
        self.assertEqual(value['reward_normalized'],0.)
        self.assertTrue(np.isfinite(value['utility']))
        self.assertEqual(value['utility'],-.25*.3-.25*.1)

    def test_full_reward_normalizes_to_one_for_every_horizon(self):
        for h in (1,3,5):
            raw=inputs(h); raw['rewards'][:]=2; raw['continuations'][:]=1
            self.assertEqual(p.rollout_statistics(**raw)['reward_normalized'],1.)

    def test_candidate_weights_apply_after_per_candidate_survival(self):
        raw=inputs(3,2);raw['weights'][:]=[.75,.25]
        raw['continuations'][:]=[[0.,1.],[0.,1.],[0.,1.]]
        raw['rewards'][:]=[[2.,1.],[2.,1.],[2.,1.]]
        result=p.rollout_statistics(**raw)
        self.assertEqual(result['predicted_return'],2.25)
        self.assertEqual(result['termination_risk'],.75)

    def test_first_action_slew_and_full_scratch_path_are_distinct(self):
        raw=inputs();raw['previous_action'][:]=-1;raw['actions'][:]=1
        result=p.rollout_statistics(**raw)
        self.assertEqual(result['slew_normalized'],1.)
        self.assertAlmostEqual(result['path_slew_normalized'],1/3)
        self.assertEqual(result['action_slew'],2.)

    def test_future_candidate_actions_can_differ_but_first_action_cannot(self):
        raw=inputs();raw['actions'][1,0]=.5
        p.rollout_statistics(**raw)
        raw['actions'][0,0]=.5
        with self.assertRaises(ValueError):p.rollout_statistics(**raw)

    def test_nonfinite_predictions_or_invalid_probability_action_reject(self):
        for field,value in [('rewards',np.nan),
            ('continuations',1.01),('continuations',-1e-9),('actions',1.01),
            ('uncertainty',np.inf),('weights',-1.)]:
            raw=inputs();raw[field].flat[0]=value
            with self.subTest(field=field,value=value),self.assertRaises(ValueError):
                p.rollout_statistics(**raw)

    def test_reward_extrapolation_is_retained_and_explicitly_projected_for_utility_only(self):
        raw=inputs(1,2);raw['rewards'][:]=[-2.,4.]
        before=raw['rewards'].copy();result=p.rollout_statistics(**raw)
        np.testing.assert_array_equal(raw['rewards'],before)
        self.assertEqual(result['raw_predicted_return'],1.)
        self.assertEqual(result['predicted_return'],1.)
        self.assertEqual(result['reward_out_of_support_fraction'],1.)
        self.assertEqual(result['reward_projection'],p.REWARD_PROJECTION)
        raw['rewards'][:]=[-2.,-4.]
        result=p.rollout_statistics(**raw)
        self.assertEqual(result['raw_predicted_return'],-3.)
        self.assertEqual(result['reward_normalized'],0.)

    def test_epistemic_proxy_is_separate_and_cannot_be_invented_from_entropy(self):
        raw=inputs(3);raw['epistemic_disagreement']=np.array([.001,.02,.01],np.float64)
        result=p.rollout_statistics(**raw)
        self.assertEqual(result['epistemic_disagreement'],.02)
        limits=dict(uncertainty=1.,disagreement=1.,termination_risk=1.,epistemic_disagreement=.01)
        self.assertFalse(p.plan_accepted(result,limits))
        raw.pop('epistemic_disagreement')
        missing=p.rollout_statistics(**raw)
        self.assertIsNone(missing['epistemic_disagreement'])
        self.assertFalse(p.plan_accepted(missing,limits))
        limits['epistemic_disagreement']=.03
        self.assertTrue(p.plan_accepted(result,limits))

    def test_invalid_epistemic_proxy_rejects(self):
        for values in (np.array([-.1,.1,.1],np.float64),np.array([0.,np.nan,0.],np.float64),
                       np.zeros(3,np.float32)):
            raw=inputs(3);raw['epistemic_disagreement']=values
            with self.assertRaises(ValueError):p.rollout_statistics(**raw)

    def test_no_implicit_dtype_shape_weight_or_horizon_coercion(self):
        for field in inputs():
            raw=inputs();raw[field]=raw[field].astype(np.float32)
            with self.subTest(field=field),self.assertRaises(ValueError):p.rollout_statistics(**raw)
        raw=inputs();raw['weights']*=.5
        with self.assertRaises(ValueError):p.rollout_statistics(**raw)
        with self.assertRaises(ValueError):p.rollout_statistics(**inputs(2))

    def test_sufficient_statistics_keep_jsd_independently_gateable(self):
        probs=np.zeros((8,2,2),np.float64);probs[:4,:,0]=1;probs[4:,:,1]=1
        statistics=p.categorical_statistics(probs,np.full(8,1/8,np.float64))
        self.assertEqual(statistics['predictive_entropy'],1.)
        self.assertEqual(statistics['expected_entropy'],0.)
        self.assertEqual(p.uncertainty_from_sufficient(1.,0.),(1.,1.))
        raw=inputs(1);raw['uncertainty'][:]=1;raw['disagreement'][:]=1
        stats=p.rollout_statistics(**raw)
        self.assertFalse(p.plan_accepted(stats,dict(uncertainty=1.,disagreement=.5,termination_risk=1.)))

    def test_impossible_entropy_or_jsd_is_rejected(self):
        with self.assertRaises(ValueError):p.uncertainty_from_sufficient(.2,.3)
        with self.assertRaises(ValueError):p.uncertainty_from_sufficient(True,0.)
        raw=inputs();raw['disagreement'][:]=.4
        with self.assertRaises(ValueError):p.rollout_statistics(**raw)

    def test_diversity_in_registered_units_with_legitimate_duplicates(self):
        self.assertEqual(p.proposal_diversity(np.zeros((2,6),np.float64)),0.)
        self.assertEqual(p.proposal_diversity(np.array([[-1]*6,[1]*6],np.float64)),1.)
        self.assertEqual(p.proposal_diversity(np.zeros((1,6),np.float64)),0.)

    def test_operator_preserves_all_inputs_and_global_rng(self):
        raw=inputs();before=copy.deepcopy(raw);rng=copy.deepcopy(np.random.get_state())
        p.rollout_statistics(**raw)
        for key in raw:np.testing.assert_array_equal(before[key],raw[key])
        after=np.random.get_state()
        self.assertEqual(rng[0],after[0]);np.testing.assert_array_equal(rng[1],after[1])
        self.assertEqual(rng[2:],after[2:])


if __name__=='__main__':unittest.main()
