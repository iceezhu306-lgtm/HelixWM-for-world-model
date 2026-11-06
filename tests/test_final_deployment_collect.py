"""Synthetic end-to-end collector contracts; no GPU or training evidence."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from helixwm_final import deployment_collect as c
from helixwm_final import deployment_calibration as d
from helixwm_final.belief import canonical_json
from helixwm_final.epistemic import BootstrapDynamics, tree_digest
from test_final_controller import FakeBackend, SOURCE, MODEL
from test_final_deployment_calibration import row_fixture


class Backend(FakeBackend):
    def predict_heads(self,state,with_image=False):
        result=super().predict_heads(state,with_image)
        if with_image:
            result['image_mean']=np.zeros((len(state['deter']),64,64,3),np.float32)
        return result


class Environment:
    def __init__(self,break_at=None):
        self.action_decisions=self.episode_env_steps=0
        self.break_at=break_at;self.closed=False
    def reset(self,seed,mask):
        self.mask=mask.copy();self.seed=seed
        return self.image()
    def image(self):
        return (np.full((64,64,3),(self.seed+self.action_decisions)%256,np.uint8)
                if self.mask[self.action_decisions] else np.zeros((64,64,3),np.uint8))
    def step(self,action):
        self.action_decisions+=1;self.episode_env_steps+=2
        if self.action_decisions==self.break_at:
            raise RuntimeError('injected native failure after acknowledged calls')
        info=dict(availability=bool(self.mask[self.action_decisions]),
            observation_timestamp=self.action_decisions,env_steps_this_step=2,
            executed_action=action.copy())
        return self.image(),1.,False,info
    def close(self):self.closed=True
    def render_diagnostic(self,*args):raise AssertionError('Hidden render must never be requested')


def setup():
    heads=BootstrapDynamics(17,{'source_sha256':SOURCE,'protocol_sha256':'4'*64}).pin()
    ctx=dict(source_sha256=SOURCE,model_sha256=MODEL,model_version=3,
        protocol_sha256='4'*64,bank_manifest_sha256=c.scenario_bank_sha256(),
        collector_sha256=c.collector_sha256(),epistemic_heads_sha256=heads.sha256,
        measurement_kind='synthetic_test_fixture')
    return Backend(),heads,ctx


class CollectorTests(unittest.TestCase):
    def test_optional_complete_trajectory_archive_keeps_control_and_rows_identical(self):
        from helixwm_final.trajectory import TrajectoryArchive
        plain_args = setup()
        plain,_,_ = c.collect_window(*plain_args, env_factory=Environment,
            seed=55000,gap=1,phase='fit',anchor_sha256='8'*64)
        with tempfile.TemporaryDirectory() as td:
            args=setup();ctx=args[2]
            archive=TrajectoryArchive.create(Path(td)/'records',
                source_sha256=ctx['source_sha256'],protocol_sha256=ctx['protocol_sha256'])
            try:
                archived,_,_=c.collect_window(*args,env_factory=Environment,
                    seed=55000,gap=1,phase='fit',anchor_sha256='8'*64,trajectory=archive)
                cursor=archive.snapshot()
            finally:archive.close()
            restored=TrajectoryArchive.restore(Path(td)/'records',cursor,
                expected_source_sha256=ctx['source_sha256'],
                expected_protocol_sha256=ctx['protocol_sha256'])
            try:self.assertEqual(restored.snapshot(),cursor)
            finally:restored.close()
            self.assertGreater(cursor['record_count'],25)
            self.assertEqual(d.diagnostic_content_sha256(plain),d.diagnostic_content_sha256(archived))

    def synthetic_window(self,backbone,heads,context,*,env_factory,seed,gap,phase,anchor_sha256,candidate=None):
        row=row_fixture(seed,gap,phase,context,None if candidate is None else candidate['sha256'])
        row['replay_anchor_sha256']=anchor_sha256
        n=gap+8
        trace=dict(schema=d.SCHEMA,context=context,phase=phase,seed=seed,gap=gap,
            execution_mode='closed_loop',diagnostics_sha256=d.diagnostic_content_sha256(row),
            replay_anchor_sha256=anchor_sha256,observation_sha256=['a'*64]*n,
            controller_state_sha256=['b'*64]*n,rng_sha256=['c'*64]*n)
        row['trace_sha256']=hashlib.sha256(canonical_json(trace)).hexdigest()
        return row,trace,dict(actions=24+gap,dmc_calls=2*(24+gap),process_id=123,
            phase=phase,seed=seed,gap=gap,model_updates=0,included_in_training_budget=False)

    def sequence_args(self,td):
        backbone,heads,ctx=setup()
        anchor=Path(td)/'synthetic_anchor.json';anchor.write_bytes(b'{"synthetic":true}')
        descriptor=d.file_descriptor(anchor,td)
        hashes={k:'d'*64 for k in d.FREEZE_COMPONENTS}
        hashes['model']=ctx['model_sha256'];hashes['epistemic_heads']=ctx['epistemic_heads_sha256']
        kwargs=dict(env_factory=Environment,directory=td,anchor_sha256=descriptor['sha256'],
            anchor_descriptor=descriptor,frozen_state_hashes=lambda:copy.deepcopy(hashes),
            device='synthetic',gpu='NO GPU EXECUTED')
        return (backbone,heads,ctx),kwargs,hashes

    def test_real_controller_short_window_counts_mask_and_error_alignment(self):
        backbone,heads,ctx=setup();env=Environment()
        head_before=tree_digest(heads.snapshot())
        row,trace,audit=c.collect_window(backbone,heads,ctx,env_factory=lambda:env,
            seed=55000,gap=1,phase='fit',anchor_sha256='8'*64)
        self.assertEqual((audit['actions'],audit['dmc_calls']),(25,50))
        self.assertTrue(env.closed)
        self.assertEqual(row['timestamp'].tolist(),list(range(17,26)))
        self.assertEqual(row['availability'].tolist(),[False]+[True]*8)
        self.assertEqual(row['energy'].shape,(8,8))
        self.assertEqual(row['temporal_prediction_error'].shape,(8,))
        self.assertEqual({p['horizon'] for p in row['plans']},{1,3,5})
        self.assertEqual(row['trace_sha256'],hashlib.sha256(canonical_json(trace)).hexdigest())
        self.assertEqual(trace['diagnostics_sha256'],d.diagnostic_content_sha256(row))
        self.assertEqual(head_before,tree_digest(heads.snapshot()))
        self.assertFalse(row['hidden_target_to_controller'])

    def test_fresh_collectors_reproduce_trace_without_copying_result(self):
        first=[]
        for _ in range(2):
            backbone,heads,ctx=setup()
            row,trace,_=c.collect_window(backbone,heads,ctx,env_factory=Environment,
                seed=55000,gap=5,phase='fit',anchor_sha256='8'*64)
            first.append((d.records_sha256([row]),trace))
        self.assertEqual(first[0],first[1])

    def test_full100gap_is_bounded_and_has_no_hidden_error(self):
        backbone,heads,ctx=setup()
        row,trace,audit=c.collect_window(backbone,heads,ctx,env_factory=Environment,
            seed=55000,gap=100,phase='fit',anchor_sha256='8'*64)
        self.assertEqual((audit['actions'],audit['dmc_calls']),(124,248))
        self.assertEqual(row['temporal_prediction_error'].shape,(8,))
        self.assertFalse(row['availability'][:100].any())
        self.assertEqual(len(trace['observation_sha256']),108)

    def test_unbounded_author_Normal_action_is_clipped_at_shared_action_boundary(self):
        class ExtrapolatingBackend(Backend):
            def act(self,state,deterministic=False,rng=None):
                return np.broadcast_to(np.array([-3.,2.5,-1.2,4.,.2,.3],np.float32),
                                       (len(state['deter']),6)).copy()
        _,heads,ctx=setup()
        row,_,_=c.collect_window(ExtrapolatingBackend(),heads,ctx,env_factory=Environment,
            seed=55000,gap=1,phase='fit',anchor_sha256='8'*64)
        np.testing.assert_array_equal(row['actions'][0],np.array([-1.,1.,-1.,1.,.2,.3],np.float32).astype(np.float64))
        for plan in row['plans']:
            self.assertLessEqual(float(np.abs(plan['actions']).max()),1.)

    def test_native_failure_preserves_actual_counters_and_closes_env(self):
        backbone,heads,ctx=setup();env=Environment(break_at=18)
        with self.assertRaises(c.WindowCollectionError) as caught:
            c.collect_window(backbone,heads,ctx,env_factory=lambda:env,
                seed=55000,gap=5,phase='fit',anchor_sha256='8'*64)
        self.assertEqual(caught.exception.acknowledged_actions,18)
        self.assertEqual(caught.exception.acknowledged_dmc_calls,36)
        self.assertTrue(env.closed)

    def test_wrong_code_schedule_phase_final_seed_or_anchor_reject_before_env(self):
        for key,bad in [('collector_sha256','f'*64),('bank_manifest_sha256','f'*64)]:
            backbone,heads,ctx=setup();ctx[key]=bad
            with self.assertRaises(ValueError):c.collect_window(backbone,heads,ctx,env_factory=lambda:self.fail('env created'),
                seed=55000,gap=1,phase='fit',anchor_sha256='8'*64)
        for args in (dict(seed=60000,gap=1,phase='fit',anchor_sha256='8'*64),
                     dict(seed=55010,gap=1,phase='fit',anchor_sha256='8'*64),
                     dict(seed=55000,gap=101,phase='fit',anchor_sha256='8'*64),
                     dict(seed=55000,gap=1,phase='fit',anchor_sha256='bad')):
            backbone,heads,ctx=setup()
            with self.assertRaises(ValueError):c.collect_window(backbone,heads,ctx,env_factory=lambda:self.fail('env created'),**args)

    def test_scenario_bank_is_explicitly_not_old_open_loop_data(self):
        bank=c.scenario_bank()
        self.assertEqual(bank['total_actions'],6600)
        self.assertEqual(bank['total_DMC_calls'],13200)
        self.assertFalse(bank['included_in_training_budget'])
        self.assertFalse(bank['policy_deterministic'])

    def test_partial_phase_claim_and_failure_are_retained_no_retry_overwrite(self):
        backbone,heads,ctx=setup()
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(c.WindowCollectionError):
                c.collect_phase(backbone,heads,ctx,env_factory=lambda:Environment(break_at=1),
                    phase='fit',directory=td,anchor_sha256='8'*64)
            failure=json.loads((Path(td)/'fit.failure.json').read_text())
            self.assertEqual(failure['acknowledged_actions'],1)
            self.assertFalse(failure['usable_for_deployment'])
            before=(Path(td)/'fit.failure.json').read_bytes()
            with self.assertRaises(FileExistsError):
                c.collect_phase(backbone,heads,ctx,env_factory=Environment,
                    phase='fit',directory=td,anchor_sha256='8'*64)
            self.assertEqual(before,(Path(td)/'fit.failure.json').read_bytes())

    def test_sequence_seals_candidate_before_holdout_but_cannot_invent_replay_gate(self):
        with tempfile.TemporaryDirectory() as td:
            args,kwargs,_=self.sequence_args(td)
            def measured(*a,**kw):
                if kw['phase']=='validation':
                    self.assertTrue((Path(td)/'candidate.json').is_file())
                    self.assertEqual(json.loads((Path(td)/'candidate.json').read_text())['sha256'],kw['candidate']['sha256'])
                return self.synthetic_window(*a,**kw)
            # A frozen/coarse wall clock reproduced the former intermittent
            # failure: adjacent valid causal events had identical time_ns.
            # Event ordering must remain strict and use a monotonic clock.
            with (patch.object(c,'collect_window',side_effect=measured),
                  patch('time.time_ns',return_value=17),patch('time.monotonic_ns',return_value=17)):
                result=c.collect_sequence(*args,**kwargs)
            self.assertFalse(result['active'])
            self.assertFalse(result['independent_replay_completed'])
            self.assertEqual(result['counts'],dict(actions=6600,dmc_calls=13200))
            self.assertFalse((Path(td)/'deployment_manifest.json').exists())
            ledger=json.loads((Path(td)/'ledger.json').read_text())
            self.assertEqual(ledger['event_clock'],'monotonic_perf_counter_ns')
            self.assertLess(ledger['candidate_sealed_ns'],ledger['validation_started_ns'])
            claims={k:json.loads((Path(td)/v['path']).read_text()) for k,v in result['phase_claims'].items()}
            candidate=json.loads((Path(td)/'candidate.json').read_text())
            d._ledger(ledger,claims,args[2],candidate,result['validation_audit'])

    def test_changed_optimizer_stops_before_heldout_and_preserves_fit(self):
        with tempfile.TemporaryDirectory() as td:
            args,kwargs,hashes=self.sequence_args(td)
            calls=[]
            def frozen():
                result=copy.deepcopy(hashes)
                if calls:result['optimizer']='e'*64
                calls.append(1)
                return result
            kwargs['frozen_state_hashes']=frozen
            with patch.object(c,'collect_window',side_effect=self.synthetic_window),self.assertRaises(ValueError):
                c.collect_sequence(*args,**kwargs)
            self.assertTrue((Path(td)/'candidate.json').is_file())
            self.assertTrue((Path(td)/'freeze.failure.json').is_file())
            self.assertFalse(list(Path(td).glob('*.validation.claim.json')))

    def test_external_retained_anchor_cannot_be_missing_or_tampered(self):
        with tempfile.TemporaryDirectory() as td:
            args,kwargs,_=self.sequence_args(td)
            (Path(td)/'synthetic_anchor.json').write_bytes(b'changed')
            with self.assertRaises(ValueError):c.collect_sequence(*args,**kwargs)
            self.assertFalse(list(Path(td).glob('*.claim.json')))


if __name__=='__main__':unittest.main()
