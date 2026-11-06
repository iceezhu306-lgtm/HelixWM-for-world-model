"""Immutable calibration hot-path tests; no production authority is issued."""
import copy
import pickle
import unittest
from unittest.mock import patch
import numpy as np
from helixwm_final import bank_gate as g,bank_replay as r
from helixwm_final.full_controller import FullController
from test_final_full_controller import ImageBackend,heads_fixture,context,IMAGE
from test_final_bank_gate import fixture_row


class BankOptimizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.heads=heads_fixture();cls.ctx=context(cls.heads);cls.ctx['collector_sha256']=r.collector_sha256()
        rows=[fixture_row(s,gap,'fit',cls.ctx) for s in g.FIT_SEEDS for gap in g.GAPS]
        cls.candidate=g.fit_candidate(rows,cls.ctx)

    def controller(self):
        c=FullController(ImageBackend(),self.heads,self.ctx,seed=17,execution='bank_validation',
                         fitted_bank_candidate=self.candidate)
        c.reset(IMAGE,episode_id=55010)
        return c

    def test_sealing_keeps_exact_json_and_deepcopy_without_rehashing_library(self):
        sealed=g.seal_bank_candidate(self.candidate,self.ctx)
        self.assertIs(copy.deepcopy(sealed),sealed)
        self.assertEqual(g.canonical_json(sealed),g.canonical_json(self.candidate))
        with patch.object(g,'digest',side_effect=AssertionError('hot validation must not rehash')):
            for _ in range(100):self.assertTrue(g.validate_fitted_candidate(sealed,self.ctx))
        self.assertEqual(g.candidate_plain_data(sealed,self.ctx),self.candidate)
        self.assertIs(type(g.candidate_plain_data(sealed,self.ctx)['fit_seeds']),list)

    def test_corruption_context_and_unvalidated_constructor_rejected(self):
        with self.assertRaises(ValueError):g.SealedBankCandidate(self.candidate)
        bad=copy.deepcopy(self.candidate);bad['support_by_gap']['1']['radius']+=1
        with self.assertRaises(ValueError):g.seal_bank_candidate(bad,self.ctx)
        sealed=g.seal_bank_candidate(self.candidate,self.ctx)
        with self.assertRaises(TypeError):sealed['support_by_gap']['1']['radius']=99
        with self.assertRaises(TypeError):sealed['active']=True
        for key in self.ctx:
            wrong=dict(self.ctx);wrong[key]=wrong[key]+1 if key=='model_version' else (
                'actual_gpu_independent_bank' if key=='measurement_kind' else 'f'*64)
            with self.subTest(key=key),self.assertRaises(ValueError):g.verify_sealed_candidate(sealed,wrong)
        # Bypassing normal dict mutation is still detected by bound identities.
        dict.__setitem__(sealed,'active',True)
        with self.assertRaises(ValueError):g.verify_sealed_candidate(sealed,self.ctx)

    def test_candidate_never_gets_production_or_pickle_authority(self):
        sealed=g.seal_bank_candidate(self.candidate,self.ctx)
        with self.assertRaises(ValueError):g.verify_bank_gate(sealed,self.ctx,require_active=False)
        with self.assertRaises(ValueError):
            FullController(ImageBackend(),self.heads,self.ctx,execution='bank_production',bank_gate=sealed)
        data=pickle.loads(pickle.dumps(sealed))
        self.assertIs(type(data),dict);self.assertEqual(data,self.candidate)
        with self.assertRaises(ValueError):g.verify_sealed_candidate(data,self.ctx)

    def test_full_snapshot_export_stays_exact_plain_data(self):
        controller=self.controller();snapshot=controller.snapshot()
        self.assertIs(type(snapshot['calibration']),dict)
        self.assertEqual(snapshot['calibration'],self.candidate)
        snapshot['calibration']['support_by_gap']['1']['radius']=100
        self.assertEqual(controller.snapshot()['calibration'],self.candidate)
        self.assertIsInstance(controller._calibration,g.SealedBankCandidate)

    def test_scratch_guard_detects_all_mutable_state_categories(self):
        mutations={
            'real':lambda c:c.real['deter'].__setitem__((0,0),c.real['deter'][0,0]+1),
            'dream':lambda c:c.bank['deter'].__setitem__((0,0),c.bank['deter'][0,0]+1),
            'weights':lambda c:c.weights.__setitem__(0,.3),
            'private_rng':lambda c:c.rngs['planning'].random(),
            'screen_rng':lambda c:c.screen_rng.random(),
            'events':lambda c:c.events.append({'event':'unexpected'}),
            'action_history':lambda c:c.action_history.append({'action':'unexpected'}),
            'auxiliary':lambda c:c.auxiliary_state.__setitem__((0,0,0),123.),
            'metrics':lambda c:c.last_metrics.update(unexpected=True),
            'last_prediction':lambda c:setattr(c,'last_prediction',{'unexpected':True}),
            'last_plans':lambda c:c.last_plans.append({'unexpected':True}),
            'last_guard':lambda c:setattr(c,'last_guard',{'unexpected':True}),
            'recovery':lambda c:setattr(c,'last_recovery_qualified',True),
            'prediction_rng_before':lambda c:setattr(c,'prediction_rng_before','e'*64),
        }
        for name,mutate in mutations.items():
            c=self.controller();before=c.scratch_state_digest();mutate(c)
            with self.subTest(category=name):self.assertNotEqual(c.scratch_state_digest(),before)
        c=self.controller();before=c.scratch_state_digest()
        dict.__setitem__(c._calibration,'active',True)
        with self.assertRaises(ValueError):c.scratch_state_digest()

    def test_scratch_accidental_mutation_is_rejected_by_real_window_runner(self):
        bank={'image':np.broadcast_to(IMAGE,(125,64,64,3)).copy(),'action':r.original.fixed_actions(55010),
              'reward':np.zeros(124,np.float64),'dmc_calls':np.full(124,2,np.int32)}
        original=FullController.scratch_plans
        def corrupt(controller,*a,**kw):
            result=original(controller,*a,**kw)
            controller.rngs['actor'].random()
            return result
        with patch.object(FullController,'scratch_plans',corrupt),self.assertRaisesRegex(ValueError,'Scratch planning changed'):
            r.measure_window(ImageBackend(),self.heads,self.ctx,bank,seed=55010,gap=1,
                             phase='validation',candidate=self.candidate)


if __name__=='__main__':unittest.main()
