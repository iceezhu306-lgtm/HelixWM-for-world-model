"""CPU bank replay contracts: synthetic data and no environment access."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

from helixwm_final import bank_replay as r,bank_gate as g
from test_final_full_controller import ImageBackend,heads_fixture,context,IMAGE
from test_final_bank_gate import fixture_row


def bank_fixture():
    return {s:{'image':np.broadcast_to(IMAGE,(125,64,64,3)).copy(),'action':r.original.fixed_actions(s),
               'reward':np.zeros(124,np.float64),'dmc_calls':np.full(124,2,np.int32)}
            for s in g.FIT_SEEDS+g.VALIDATION_SEEDS}


class BankReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.bank=bank_fixture();cls.heads=heads_fixture()
    def ctx(self):
        result=context(self.heads);result['collector_sha256']=r.collector_sha256();return result

    def test_actual_controller_replay_forces_recorded_actions_and_has_no_env_calls(self):
        backend=ImageBackend();ctx=self.ctx();record=self.bank[55000]
        with patch('helixwm_matched.environment.MatchedWalkerEnv',side_effect=AssertionError('no env allowed')):
            row=r.measure_window(backend,self.heads,ctx,record,seed=55000,gap=5,phase='fit')
        np.testing.assert_array_equal(row['actions'],record['action'][16:29].astype(np.float64))
        np.testing.assert_array_equal(row['actual_rewards'],record['reward'][16:29])
        self.assertEqual(row['execution_mode'],'forced_recorded_actions')
        self.assertFalse(row['counterfactual_actual_outcomes_available'])
        # Only warmup and recovered images reach a real posterior; five hidden
        # images never do. Candidate posterior calls add at visible frames only.
        self.assertTrue(all(p['support_queries'].shape==(p['horizon'],8,22) for p in row['plans']))
        self.assertEqual(row['energy'].shape,(8,8));self.assertEqual(row['temporal_prediction_error'].shape,(8,))

    def test_exogenous_action_tape_and_complete_independent_seeds_are_mandatory(self):
        with self.assertRaises(ValueError):r._check_bank({k:v for k,v in self.bank.items() if k!=55010})
        bad=copy.deepcopy(self.bank);bad[55000]['action'][0,0]=0.
        with self.assertRaisesRegex(ValueError,'exogenous'):r._check_bank(bad)

    def test_bank_digest_preserves_exact_seeds_order_and_array_contents(self):
        expected=r._bank_content_hash(self.bank)
        reversed_bank=dict(reversed(list(self.bank.items())))
        self.assertEqual(r._bank_content_hash(reversed_bank),expected)
        changed=copy.deepcopy(self.bank)
        changed[55000]['image'][0,0,0,0]^=np.uint8(1)
        self.assertNotEqual(r._bank_content_hash(changed),expected)
        for alias in (55000.,np.int64(55000),'55000'):
            changed={alias if k==55000 else k:v for k,v in self.bank.items()}
            with self.subTest(alias_type=type(alias).__name__),self.assertRaisesRegex(ValueError,'integer seed'):
                r._bank_content_hash(changed)
        # A string/int pair remains distinguishable and cannot bypass the exact
        # partition by a downstream int() conversion.
        changed=dict(self.bank);changed['55000']=changed[55000]
        with self.assertRaisesRegex(ValueError,'integer seed'):r._bank_content_hash(changed)

    def test_actual_branch_reloads_real_verified_original_bank_before_measurement(self):
        root=(Path(__file__).resolve().parents[2]/'preflight'/'shared_bank_20261106T102644756146Z'
              /'VERIFIED_BACKUP'/'run'/'BANK')
        if not root.is_dir():self.skipTest('Real independent bank is a local external acceptance fixture')
        manifest_sha='4d25f2755c8258f8c0ca0b7233fa8c4bb2aab2b529e391f8c7127ae3d8af0f42'
        metadata=json.loads((root/'manifest.json').read_bytes())
        bank,origin=r.load_original_bank(root,manifest_sha,
            environment_lock_sha256=metadata['environment_lock_sha256'],
            origin_source_sha256=metadata['source_sha256'],origin_protocol_sha256=metadata['protocol_sha256'])
        ctx=self.ctx();ctx.update(measurement_kind='actual_gpu_independent_bank',bank_manifest_sha256=manifest_sha)
        frozen={k:'a'*64 for k in g.numeric.FREEZE_COMPONENTS}
        frozen.update(model=ctx['model_sha256'],epistemic_heads=ctx['epistemic_heads_sha256'])
        class ReachedActualMeasurement(RuntimeError):pass
        with tempfile.TemporaryDirectory() as td:
            with patch('helixwm_final.provenance.source_digest',return_value=(ctx['source_sha256'],{})), \
                 patch.object(r,'measure_window',side_effect=ReachedActualMeasurement('CPU stops before GPU work')) as measure:
                with self.assertRaises(ReachedActualMeasurement):
                    r.measure_and_write(None,None,ctx,bank,directory=Path(td)/'verified',bank_origin=origin,
                        frozen_state_hashes=lambda:copy.deepcopy(frozen),device='cuda',gpu='CPU guarded branch fixture')
                self.assertEqual(measure.call_count,1)
                self.assertFalse((Path(td)/'verified'/'bank_gate_manifest.json').exists())
                bad=copy.deepcopy(bank);bad[55000]['image'][0,0,0,0]^=np.uint8(1)
                measure.reset_mock()
                with self.assertRaisesRegex(ValueError,'Supplied arrays differ'):
                    r.measure_and_write(None,None,ctx,bad,directory=Path(td)/'tampered',bank_origin=origin,
                        frozen_state_hashes=lambda:copy.deepcopy(frozen),device='cuda',gpu='CPU guarded branch fixture')
                measure.assert_not_called()
                self.assertFalse((Path(td)/'tampered').exists())

    def test_collector_identity_and_phase_leakage_reject_before_control(self):
        ctx=self.ctx();ctx['collector_sha256']='e'*64
        with self.assertRaisesRegex(ValueError,'source mismatch'):
            r.measure_window(ImageBackend(),self.heads,ctx,self.bank[55000],seed=55000,gap=1,phase='fit')
        with self.assertRaises(ValueError):
            r.measure_window(ImageBackend(),self.heads,self.ctx(),self.bank[55010],seed=55010,gap=1,phase='fit')

    def test_synchronous_fit_seal_heldout_order_and_synthetic_inactivity(self):
        ctx=self.ctx();events=[];frozen={k:'a'*64 for k in g.numeric.FREEZE_COMPONENTS}
        frozen.update(model=ctx['model_sha256'],epistemic_heads=ctx['epistemic_heads_sha256'])
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)/'new_measurement'
            def measured(*args,seed,gap,phase,candidate=None,**kw):
                events.append(phase)
                if phase=='validation':self.assertTrue((root/'candidate.json').is_file())
                return fixture_row(seed,gap,phase,ctx,None if candidate is None else candidate['sha256'])
            with patch.object(r,'measure_window',side_effect=measured):
                result=r.measure_and_write(None,None,ctx,self.bank,directory=root,
                    bank_origin={'manifest_sha256':ctx['bank_manifest_sha256']},
                    frozen_state_hashes=lambda:copy.deepcopy(frozen),device='synthetic',gpu='none')
            self.assertEqual(events,['fit']*60+['validation']*60)
            self.assertFalse(result['active']);self.assertEqual(result['environment_actions'],0)
            handle=g.load_bank_gate(root,result['manifest_sha256'],ctx,require_active=False)
            self.assertFalse(handle['active'])
            with self.assertRaises(FileExistsError):
                r.measure_and_write(None,None,ctx,self.bank,directory=root,
                    bank_origin={'manifest_sha256':ctx['bank_manifest_sha256']},
                    frozen_state_hashes=lambda:frozen,device='synthetic',gpu='none')

    def test_freeze_change_blocks_heldout_and_preserves_fit(self):
        ctx=self.ctx();frozen={k:'a'*64 for k in g.numeric.FREEZE_COMPONENTS}
        frozen.update(model=ctx['model_sha256'],epistemic_heads=ctx['epistemic_heads_sha256'])
        calls=[];phases=[]
        def freeze():
            result=copy.deepcopy(frozen)
            if calls:result['optimizer']='f'*64
            calls.append(1);return result
        def measured(*args,seed,gap,phase,candidate=None,**kw):
            phases.append(phase);return fixture_row(seed,gap,phase,ctx)
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)/'failed'
            with patch.object(r,'measure_window',side_effect=measured),self.assertRaisesRegex(ValueError,'HELDOUT not started'):
                r.measure_and_write(None,None,ctx,self.bank,directory=root,
                    bank_origin={'manifest_sha256':ctx['bank_manifest_sha256']},frozen_state_hashes=freeze,device='synthetic',gpu='none')
            self.assertEqual(phases,['fit']*60);self.assertTrue((root/'candidate.json').is_file())
            self.assertTrue((root/'failure.json').is_file());self.assertFalse((root/'bank_gate_manifest.json').exists())


if __name__=='__main__':unittest.main()
