"""Independent-bank CPU fixtures; they never authorize production."""
import copy
import json
import pickle
from pathlib import Path
import tempfile
import unittest
import numpy as np

from helixwm_final import bank_gate as g
from helixwm_final.belief import canonical_json
from test_final_deployment_calibration import row_fixture,context


def fixture_row(seed,gap,phase,ctx,candidate=None):
    row=row_fixture(seed,gap,phase,ctx,candidate)
    row.update(measurement_protocol=g.MEASUREMENT_PROTOCOL,execution_mode='forced_recorded_actions',
        planning_actions_are_counterfactual=True,counterfactual_actual_outcomes_available=False)
    row.pop('replay_anchor_sha256');row.pop('trace_sha256')
    row['support_query']=np.concatenate((np.zeros((gap+8,16)),row['actions']),1)
    for plan in row['plans']:
        plan['support_queries']=np.concatenate((np.zeros((plan['horizon'],8,16)),plan['actions']),-1)
    return row


def fixtures():
    ctx=context();fit=[fixture_row(s,gap,'fit',ctx) for s in g.FIT_SEEDS for gap in g.GAPS]
    candidate=g.fit_candidate(fit,ctx)
    heldout=[fixture_row(s,gap,'validation',ctx,candidate['sha256']) for s in g.VALIDATION_SEEDS for gap in g.GAPS]
    return ctx,fit,candidate,heldout


def bundle(root,ctx,fit,candidate,heldout,pid=1):
    root=Path(root);root.mkdir(exist_ok=True)
    (root/'candidate.json').write_bytes(canonical_json(candidate))
    windows=[g.write_window(r,root) for r in fit+heldout]
    frozen={k:'a'*64 for k in g.numeric.FREEZE_COMPONENTS}
    frozen.update(model=ctx['model_sha256'],epistemic_heads=ctx['epistemic_heads_sha256'])
    record={'schema':g.SCHEMA,'context':ctx,'bank_origin':{'manifest_sha256':ctx['bank_manifest_sha256']},
        'environment_actions':0,'author_updates':0,'auxiliary_updates':0,'process_id':pid,
        'freeze_before':frozen,'freeze_after':frozen,'windows':windows,'rows_sha256':g.rows_digest(fit+heldout),
        'candidate':g.numeric.file_descriptor(root/'candidate.json',root),
        'candidate_sealed_ns':1,'heldout_started_ns':2,'completed_ns':3,'device':'synthetic','gpu':'none'}
    (root/'bank_measurement.json').write_bytes(canonical_json(record))
    sha=g.numeric.file_descriptor(root/'bank_measurement.json',root)['sha256']
    descriptor=g.seal_measurement(root,sha,ctx)
    return descriptor,record


class BankGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ctx,cls.fit,cls.candidate,cls.heldout=fixtures()
        cls.tmp=tempfile.TemporaryDirectory();cls.root=Path(cls.tmp.name)
        cls.descriptor,_=bundle(cls.root,cls.ctx,cls.fit,cls.candidate,cls.heldout)
    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()

    def test_zero_rewards_and_finite_empirical_thresholds(self):
        report=g.evaluate_candidate(self.candidate,self.heldout,self.ctx)
        self.assertTrue(report['passed']);self.assertEqual(report['scope'],g.SCOPE)
        self.assertFalse(report['novel_actions_have_observed_counterfactual_outcomes'])

    def test_fit_only_seed_library_and_leave_own_seed_out_rule(self):
        for table in self.candidate['support_by_gap'].values():
            self.assertEqual(set(table['seeds']),set(g.FIT_SEEDS))
            self.assertEqual(table['rule'],g.SUPPORT_RULE)
        rows=copy.deepcopy(self.fit);rows[0]['seed']=55010
        with self.assertRaises(ValueError):g.fit_candidate(rows,self.ctx)

    def test_every_plan_depth_and_candidate_must_be_supported(self):
        q=np.zeros((5,8,22),np.float64)
        self.assertTrue(g.plan_support(self.candidate,100,q)['supported'])
        q[-1,-1,0]=100.
        self.assertFalse(g.plan_support(self.candidate,100,q)['supported'])
        q[-1,-1,0]=np.nan
        with self.assertRaises(ValueError):g.plan_support(self.candidate,100,q)

    def test_support_query_uses_pre_action_head_mean_and_executed_actions(self):
        state=np.stack([np.zeros((2,16)),np.ones((2,16)),np.full((2,16),2.)])
        actions=np.ones((2,6))*.2
        q=g.support_query(state,actions)
        np.testing.assert_array_equal(q[:,:16],np.ones((2,16)))
        np.testing.assert_array_equal(q[:,-6:],actions)

    def test_counterfactual_reward_assignment_is_rejected(self):
        rows=copy.deepcopy(self.fit);rows[0]['counterfactual_actual_outcomes_available']=True
        with self.assertRaisesRegex(ValueError,'ground truth'):g.fit_candidate(rows,self.ctx)

    def test_new_plan_actions_must_match_support_query_actions(self):
        rows=copy.deepcopy(self.fit);rows[0]['plans'][0]['support_queries'][0,0,-1]=.2
        with self.assertRaisesRegex(ValueError,'proposed actions'):g.fit_candidate(rows,self.ctx)

    def test_wrong_masks_nonfinite_and_missing_cells_rejected(self):
        for field in ('energy','support_query'):
            rows=copy.deepcopy(self.fit);rows[0][field].flat[0]=np.nan
            with self.subTest(field=field),self.assertRaises(ValueError):g.fit_candidate(rows,self.ctx)
        with self.assertRaises(ValueError):g.fit_candidate(self.fit[:-1],self.ctx)
        rows=copy.deepcopy(self.fit);rows[0]['availability'][0]=True
        with self.assertRaises(ValueError):g.fit_candidate(rows,self.ctx)

    def test_one_eighth_failure_pools_without_loosening_reset_structure(self):
        rows=copy.deepcopy(self.heldout);rows[0]['temporal_prediction_error'][-1]=1.
        report=g.evaluate_candidate(self.candidate,rows,self.ctx)
        self.assertFalse(report['passed'])
        rows[0]['risk_reset'][-1]=False
        self.assertTrue(g.evaluate_candidate(self.candidate,rows,self.ctx)['passed'])

    def test_heldout_failure_does_not_change_fit_or_support(self):
        before=copy.deepcopy(self.candidate);rows=copy.deepcopy(self.heldout)
        rows[0]['epistemic_disagreement'][:]=100.
        self.assertFalse(g.evaluate_candidate(self.candidate,rows,self.ctx)['passed'])
        self.assertEqual(before,self.candidate)

    def test_file_backed_synthetic_stays_inactive_and_cannot_be_forged(self):
        handle=g.load_bank_gate(self.root,self.descriptor['sha256'],self.ctx,require_active=False)
        self.assertIsInstance(handle,g.VerifiedBankGate);self.assertFalse(handle['active'])
        self.assertTrue(handle['raw_array_numeric_recomputation_verified'])
        self.assertFalse(handle['independent_recomputation_verified'])
        with self.assertRaises(ValueError):g.verify_bank_gate(handle,self.ctx)
        with self.assertRaises(ValueError):g.VerifiedBankGate(dict(handle))
        restored=copy.deepcopy(handle);g.verify_bank_gate(restored,self.ctx,require_active=False)
        restored['active']=True
        with self.assertRaises(ValueError):g.verify_bank_gate(restored,self.ctx,require_active=False)

    def test_current_version_model_heads_source_bank_and_protocol_binding(self):
        handle=g.load_bank_gate(self.root,self.descriptor['sha256'],self.ctx,require_active=False)
        for key in ('source_sha256','model_sha256','bank_manifest_sha256','protocol_sha256','epistemic_heads_sha256'):
            ctx=dict(self.ctx);ctx[key]='f'*64
            with self.subTest(key=key),self.assertRaises(ValueError):g.verify_bank_gate(handle,ctx,require_active=False)
        with self.assertRaises(ValueError):g.load_bank_gate(self.root,'f'*64,self.ctx,require_active=False)

    def test_nested_support_is_immutable_and_checkpoint_does_not_restore_authority(self):
        handle=g.load_bank_gate(self.root,self.descriptor['sha256'],self.ctx,require_active=False)
        with self.assertRaises(TypeError):handle['support_by_gap']['1']['radius']=100.
        restored=pickle.loads(pickle.dumps(handle))
        self.assertIs(type(restored),dict)
        self.assertEqual(g.digest(restored),g.digest(dict(handle)))
        with self.assertRaises(ValueError):g.verify_bank_gate(restored,self.ctx,require_active=False)

    def test_modified_raw_bytes_fail_before_recompute(self):
        import shutil
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)/'bundle';shutil.copytree(self.root,root)
            target=next(root.glob('fit*.npz'));target.write_bytes(target.read_bytes()+b'changed')
            with self.assertRaisesRegex(ValueError,'mismatch'):g.load_bank_gate(root,self.descriptor['sha256'],self.ctx,require_active=False)

    def test_split_label_and_schema_cannot_claim_closed_loop(self):
        rows=copy.deepcopy(self.fit);rows[0]['execution_mode']='closed_loop'
        with self.assertRaises(ValueError):g.fit_candidate(rows,self.ctx)
        bad=dict(self.ctx);bad['measurement_kind']='actual_gpu_closed_loop'
        with self.assertRaises(ValueError):g.context_valid(bad)


if __name__=='__main__':unittest.main()
