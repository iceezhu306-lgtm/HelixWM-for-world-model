"""Synthetic CPU fixtures only: no GPU measurement or production activation."""
import copy
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np

from helixwm_final import deployment_calibration as d
from helixwm_final.belief import canonical_json


def context():
    return dict(source_sha256='1'*64,model_sha256='2'*64,bank_manifest_sha256='3'*64,
        protocol_sha256='4'*64,collector_sha256='5'*64,epistemic_heads_sha256='a'*64,model_version=7,
        measurement_kind='synthetic_test_fixture')


def row_fixture(seed,gap,phase,ctx,candidate=None):
    n=gap+8
    risk=0.;risks=[]
    for t in range(n):
        if t<gap:risk+=.4
        risks.append(risk)
        if t>=gap:risk=0.
    plans=[]
    for h in (1,3,5):
        plans.append(dict(timestamp=17,proposal_id=0,horizon=h,state_sha256='6'*64,
            rng_sha256='7'*64,scratch_only=True,rewards=np.zeros((h,8),np.float64),
            continuations=np.full((h,8),.99,np.float64),actions=np.zeros((h,8,6),np.float64),
            predictive_entropy=np.full(h,.4,np.float64),expected_entropy=np.full(h,.3,np.float64),
            epistemic_disagreement=np.full(h,.005,np.float64),
            weights=np.full(8,1/8,np.float64),previous_action=np.zeros(6,np.float64)))
    return dict(seed=seed,gap=gap,phase=phase,context=copy.deepcopy(ctx),
        measurement_protocol=d.MEASUREMENT_PROTOCOL,candidate_sha256=candidate,
        model_updates=0,hidden_target_to_controller=False,hidden_target_to_learner=False,
        replay_anchor_sha256='8'*64,trace_sha256='9'*64,plans=plans,
        timestamp=np.arange(17,17+n,dtype=np.int32),availability=np.arange(n)>=gap,
        actions=np.zeros((n,6),np.float64),actual_rewards=np.zeros(n,np.float64),
        predictive_entropy=np.full(n,.4,np.float64),expected_entropy=np.full(n,.3,np.float64),
        candidate_weights=np.full((n,8),1/8,np.float64),cumulative_risk=np.array(risks,np.float64),
        risk_reset=np.arange(n)>=gap,energy=np.broadcast_to(.1+.01*np.arange(8),(8,8)).copy(),
        fallback=np.arange(n)<gap,fusion_lambda=np.zeros(n,np.float64),
        fusion_kl=np.zeros(n,np.float64),previous_action=np.zeros(6,np.float64),
        epistemic_disagreement=np.full(n,.005,np.float64),
        temporal_prediction_error=np.full(8,.02,np.float64))


def fixtures():
    ctx=context()
    fit=[row_fixture(s,g,'fit',ctx) for s in d.FIT_SEEDS for g in d.GAPS]
    candidate=d.fit_candidate(fit,ctx)
    valid=[row_fixture(s,g,'validation',ctx,candidate['sha256']) for s in d.VALIDATION_SEEDS for g in d.GAPS]
    return fit,valid,ctx,candidate


def write_json(path,value):
    path.write_bytes(canonical_json(value))
    return path


def write_fixture_bundle(root):
    """Full transport schema exercising validation, explicitly synthetic."""
    root=Path(root);root.mkdir(parents=True,exist_ok=True);ctx=context()
    anchor=write_json(root/'synthetic_anchor.json',{'fixture_only':True,'no_model_or_gpu':True})
    anchor_desc=d.file_descriptor(anchor,root)
    traces=[];replay_cells=[];windows=[]
    def finalise(row):
        row['replay_anchor_sha256']=anchor_desc['sha256']
        n=row['gap']+8
        trace=dict(schema=d.SCHEMA,context=ctx,phase=row['phase'],seed=row['seed'],gap=row['gap'],
            execution_mode='closed_loop',diagnostics_sha256=d.diagnostic_content_sha256(row),
            replay_anchor_sha256=row['replay_anchor_sha256'],observation_sha256=['a'*64]*n,
            controller_state_sha256=['b'*64]*n,rng_sha256=['c'*64]*n)
        name=f"{row['phase']}_{row['seed']}_{row['gap']}"
        tp=write_json(root/(name+'.trace.json'),trace);td=d.file_descriptor(tp,root)
        row['trace_sha256']=td['sha256'];traces.append(td)
        rp=root/(name+'.synthetic_replayed.json');rp.write_bytes(tp.read_bytes())
        replay_cells.append(dict(phase=row['phase'],seed=row['seed'],gap=row['gap'],
            replay_anchor_sha256=row['replay_anchor_sha256'],reference_trace_sha256=row['trace_sha256'],
            replayed_trace=d.file_descriptor(rp,root),reference_process_id=100,replay_process_id=101))
        windows.append(d.write_measurement_window(row,root))
        return row
    fit=[finalise(row_fixture(s,g,'fit',ctx)) for s in d.FIT_SEEDS for g in d.GAPS]
    candidate=d.fit_candidate(fit,ctx)
    cp=write_json(root/'candidate.json',candidate)
    valid=[finalise(row_fixture(s,g,'validation',ctx,candidate['sha256'])) for s in d.VALIDATION_SEEDS for g in d.GAPS]
    audit=d.evaluate_candidate(candidate,valid,ctx)
    hashes={name:('d'*64) for name in d.FREEZE_COMPONENTS};hashes['model']=ctx['model_sha256']
    hashes['epistemic_heads']=ctx['epistemic_heads_sha256']
    freeze=dict(schema=d.SCHEMA,context=ctx,device='synthetic',gpu='NO GPU EXECUTED',model_updates=0,
        before=hashes,after=copy.deepcopy(hashes),fit_records_sha256=candidate['fit_records_sha256'],
        validation_records_sha256=audit['validation_records_sha256'])
    fp=write_json(root/'freeze.json',freeze)
    claims={}
    for phase,when in [('fit',1),('validation',4)]:
        item=dict(schema=d.SCHEMA,context=ctx,phase=phase,attempt=1,event_clock=d.EVENT_CLOCK,claimed_ns=when,
            candidate_sha256=None if phase=='fit' else candidate['sha256'])
        claims[phase]=d.file_descriptor(write_json(root/(phase+'.claim.json'),item),root)
    ledger=dict(schema=d.SCHEMA,context=ctx,event_clock=d.EVENT_CLOCK,fit_attempts=1,validation_attempts=1,
        holdout_threshold_revisions=0,candidate_sha256=candidate['sha256'],
        fit_records_sha256=candidate['fit_records_sha256'],validation_records_sha256=audit['validation_records_sha256'],
        fit_completed_ns=2,candidate_sealed_ns=3,validation_started_ns=5,validation_completed_ns=6)
    lp=write_json(root/'ledger.json',ledger)
    rp=write_json(root/'replay.json',dict(schema=d.SCHEMA,context=ctx,cells=replay_cells))
    manifest=dict(schema=d.SCHEMA,measurement_protocol=d.MEASUREMENT_PROTOCOL,context=ctx,
        candidate=d.file_descriptor(cp,root),windows=windows,freeze_evidence=d.file_descriptor(fp,root),
        replay_evidence=d.file_descriptor(rp,root),phase_ledger=d.file_descriptor(lp,root),
        phase_claims=claims,replay_anchors=[anchor_desc],raw_traces=traces)
    mp=write_json(root/'deployment_manifest.json',manifest)
    return d.file_descriptor(mp,root)['sha256'],ctx


class DeploymentCalibrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fit,cls.valid,cls.ctx,cls.candidate=fixtures()
        cls.bundle_tmp=tempfile.TemporaryDirectory()
        cls.bundle=Path(cls.bundle_tmp.name)/'bundle'
        cls.manifest_sha,cls.bundle_ctx=write_fixture_bundle(cls.bundle)

    @classmethod
    def tearDownClass(cls):cls.bundle_tmp.cleanup()

    def test_zero_reward_fit_and_holdout_are_valid_engineering_statistics(self):
        report=d.evaluate_candidate(self.candidate,self.valid,self.ctx)
        self.assertTrue(report['validation']['passed']);self.assertFalse(report['active'])
        for gap in d.GAPS:
            self.assertEqual(self.candidate['thresholds_by_gap'][str(gap)]['reward_weight'],1.)
        self.assertTrue(report['validation']['per_cell']['55010:100']['functional_failure_20_actions'])
        self.assertFalse(report['physical_safety_proven'])

    def test_all_horizons_have_their_own_cumulative_termination_limits(self):
        limits=self.candidate['planning_limits_by_gap_horizon']['100']
        self.assertLess(limits['1']['termination_risk'],limits['3']['termination_risk'])
        self.assertLess(limits['3']['termination_risk'],limits['5']['termination_risk'])
        self.assertEqual(self.candidate['planning_operator_version'],'helixwm_final_planning_statistics_v2')

    def test_fit_does_not_accept_holdout_or_final_seeds(self):
        for seed in (55010,60000,True):
            rows=copy.deepcopy(self.fit);rows[0]['seed']=seed
            with self.subTest(seed=seed),self.assertRaises(ValueError):d.fit_candidate(rows,self.ctx)

    def test_missing_duplicate_and_truncated_cells_reject(self):
        with self.assertRaises(ValueError):d.fit_candidate(self.fit[:-1],self.ctx)
        rows=copy.deepcopy(self.fit);rows[-1]=copy.deepcopy(rows[0])
        with self.assertRaises(ValueError):d.fit_candidate(rows,self.ctx)
        rows=copy.deepcopy(self.fit);rows[0]['energy']=rows[0]['energy'][:-1]
        with self.assertRaises(ValueError):d.fit_candidate(rows,self.ctx)

    def test_nonfinite_dtype_and_unit_failures_are_not_imputed(self):
        for field,bad in [('actual_rewards',np.nan),('predictive_entropy',np.inf),('fusion_kl',-1.)]:
            rows=copy.deepcopy(self.fit);rows[0][field].flat[0]=bad
            with self.subTest(field=field),self.assertRaises(ValueError):d.fit_candidate(rows,self.ctx)
        rows=copy.deepcopy(self.fit);rows[0]['actions']=rows[0]['actions'].astype(np.float32)
        with self.assertRaises(ValueError):d.fit_candidate(rows,self.ctx)

    def test_wrong_mask_risk_reset_count_or_raw_entropy_reject(self):
        rows=copy.deepcopy(self.fit);rows[0]['availability'][0]=True
        with self.assertRaises(ValueError):d.fit_candidate(rows,self.ctx)
        rows=copy.deepcopy(self.fit);rows[0]['cumulative_risk'][0]=0.
        with self.assertRaises(ValueError):d.fit_candidate(rows,self.ctx)
        rows=copy.deepcopy(self.fit);rows[0]['expected_entropy'][0]=.5
        with self.assertRaises(ValueError):d.fit_candidate(rows,self.ctx)
        rows=copy.deepcopy(self.fit);rows[0]['model_updates']=False
        with self.assertRaises(ValueError):d.fit_candidate(rows,self.ctx)

    def test_holdout_failed_disagreement_cannot_change_fitted_thresholds(self):
        rows=copy.deepcopy(self.valid);rows[0]['expected_entropy'][:]=0.
        before=copy.deepcopy(self.candidate)
        result=d.evaluate_candidate(self.candidate,rows,self.ctx)
        self.assertEqual(self.candidate,before);self.assertFalse(result['active'])
        self.assertFalse(result['validation']['passed'])
        self.assertIn('disagreement_upper',result['validation']['per_cell']['55010:1']['failed_checks'])

    def test_wrong_candidate_source_model_version_and_hidden_data_reject(self):
        for key,value in [('candidate_sha256','f'*64),('hidden_target_to_controller',True),
            ('hidden_target_to_learner',True),('measurement_protocol','open_loop_random_action_bank_v1')]:
            rows=copy.deepcopy(self.valid);rows[0][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):d.evaluate_candidate(self.candidate,rows,self.ctx)
        rows=copy.deepcopy(self.valid);rows[0]['context']['model_version']=True
        with self.assertRaises(ValueError):d.evaluate_candidate(self.candidate,rows,self.ctx)

    def test_missing_horizon_or_duplicate_plan_reject(self):
        rows=copy.deepcopy(self.fit);rows[0]['plans'].pop()
        with self.assertRaises(ValueError):d.fit_candidate(rows,self.ctx)
        rows=copy.deepcopy(self.fit);rows[0]['plans'].append(copy.deepcopy(rows[0]['plans'][0]))
        with self.assertRaises(ValueError):d.fit_candidate(rows,self.ctx)

    def test_fusion_trust_violation_fails_holdout_even_if_coverage_passes(self):
        rows=copy.deepcopy(self.valid);rows[0]['fusion_lambda'][0]=.3
        result=d.evaluate_candidate(self.candidate,rows,self.ctx)
        self.assertFalse(result['validation']['passed'])
        self.assertIn('fusion_lambda_cap',result['validation']['per_cell']['55010:1']['failed_checks'])

    def test_eight_point_cell_diagnostic_does_not_require_perfect_pooled_coverage(self):
        rows=copy.deepcopy(self.valid)
        rows[0]['temporal_prediction_error'][-1]=1.
        rows[0]['energy'][-1,:]=1.
        rows[0]['risk_reset'][-1]=False
        result=d.evaluate_candidate(self.candidate,rows,self.ctx)['validation']
        cell=result['per_cell']['55010:1']
        self.assertEqual(cell['coverage']['energy_upper'],7/8)
        self.assertFalse(cell['coverage_diagnostic_passed'])
        self.assertTrue(cell['structural_invariants_passed'])
        self.assertAlmostEqual(result['by_gap']['1']['coverage']['energy_upper'],79/80)
        self.assertTrue(result['passed'])
        self.assertEqual(self.candidate['coverage_revision'],d.COVERAGE_REVISION)

    def test_single_unqualified_reset_fails_even_when_pooled_coverage_passes(self):
        rows=copy.deepcopy(self.valid)
        rows[0]['temporal_prediction_error'][-1]=1.
        result=d.evaluate_candidate(self.candidate,rows,self.ctx)['validation']
        self.assertTrue(result['by_gap']['1']['coverage_passed'])
        self.assertFalse(result['all_structural_invariants_passed'])
        self.assertFalse(result['passed'])
        self.assertIn('risk_reset_has_qualified_temporal_error',result['per_cell']['55010:1']['failed_checks'])

    def test_pooled_fraction_is_equal_seed_mean_and_does_not_refit(self):
        rows=copy.deepcopy(self.valid)
        for row in rows:
            if row['gap']==1:
                row['temporal_prediction_error'][-1]=1.
                row['risk_reset'][-1]=False
        original=copy.deepcopy(self.candidate)
        result=d.evaluate_candidate(self.candidate,rows,self.ctx)['validation']
        self.assertEqual(result['by_gap']['1']['coverage']['temporal_prediction_error_upper'],7/8)
        self.assertFalse(result['by_gap']['1']['coverage_passed'])
        self.assertFalse(result['passed'])
        self.assertEqual(original,self.candidate)

    def test_file_backed_synthetic_pass_is_still_inactive_and_default_rejected(self):
        report=d.load_deployment_calibration(self.bundle,self.manifest_sha,self.bundle_ctx,require_active=False)
        self.assertTrue(report['validation']['passed']);self.assertFalse(report['active'])
        self.assertIn('synthetic_fixture_not_production',report['inactive_reasons'])
        self.assertTrue(report['provenance']['file_bytes_verified'])
        with self.assertRaises(ValueError):d.load_deployment_calibration(self.bundle,self.manifest_sha,self.bundle_ctx)

    def test_external_manifest_sha_and_external_context_cannot_be_self_declared(self):
        with self.assertRaises(ValueError):d.load_deployment_calibration(self.bundle,'0'*64,self.bundle_ctx,require_active=False)
        wrong=copy.deepcopy(self.bundle_ctx);wrong['measurement_kind']='actual_gpu_closed_loop'
        with self.assertRaises(ValueError):d.load_deployment_calibration(self.bundle,self.manifest_sha,wrong,require_active=False)

    def test_manual_passed_dictionary_and_modified_handle_cannot_authorize(self):
        with self.assertRaises(ValueError):d.verify_deployment_handle(dict(active=True,passed=True),self.ctx)
        with self.assertRaises(ValueError):d.VerifiedDeployment(dict(active=True))
        report=d.load_deployment_calibration(self.bundle,self.manifest_sha,self.bundle_ctx,require_active=False)
        report['active']=True
        with self.assertRaises(ValueError):d.verify_deployment_handle(report,self.ctx)
        with self.assertRaises(ValueError):d.verify_deployment_handle(dict(report),self.ctx)

    def test_verified_synthetic_handle_deepcopy_preserves_inactivity(self):
        report=d.load_deployment_calibration(self.bundle,self.manifest_sha,self.bundle_ctx,require_active=False)
        restored=copy.deepcopy(report)
        self.assertIsInstance(restored,d.VerifiedDeployment)
        with self.assertRaises(ValueError):d.verify_deployment_handle(restored,self.ctx)

    def test_bootstrap_does_not_permit_uncalibrated_training_past_5500(self):
        self.assertEqual(d.training_calibration_gate(4999),'SHARED_RANDOM_PREFILL')
        self.assertEqual(d.training_calibration_gate(5000),'REGISTERED_FIRST_LEARNED_BOOTSTRAP_EPISODE')
        self.assertEqual(d.training_calibration_gate(5499),'REGISTERED_FIRST_LEARNED_BOOTSTRAP_EPISODE')
        for count in (5500,5501,1000000):
            with self.subTest(count=count),self.assertRaises(RuntimeError):d.training_calibration_gate(count)
        with self.assertRaises(RuntimeError):d.training_calibration_gate(5500,dict(active=True),self.ctx)
        with self.assertRaises(ValueError):d.training_calibration_gate(True)

    def test_exclusive_phase_claims_and_window_files_do_not_overwrite(self):
        with tempfile.TemporaryDirectory() as td:
            first=d.claim_phase(td,self.ctx,'fit')
            original=first.read_bytes()
            with self.assertRaises(FileExistsError):d.claim_phase(td,self.ctx,'fit')
            self.assertEqual(first.read_bytes(),original)
            d.claim_phase(td,self.ctx,'validation',self.candidate)
            with self.assertRaises(FileExistsError):d.claim_phase(td,self.ctx,'validation',self.candidate)
            d.write_measurement_window(self.fit[0],td)
            with self.assertRaises(FileExistsError):d.write_measurement_window(self.fit[0],td)

    def test_mutated_npz_or_missing_raw_anchor_fails_byte_verification(self):
        for mode in ('npz','anchor'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as td:
                dest=Path(td)/'bundle';shutil.copytree(self.bundle,dest)
                target=next(dest.glob('*.npz')) if mode=='npz' else dest/'synthetic_anchor.json'
                target.write_bytes(target.read_bytes()+b'corrupted')
                with self.assertRaises(ValueError):d.load_deployment_calibration(dest,self.manifest_sha,self.bundle_ctx,require_active=False)

    def _changed_json_bundle(self,filename,mutator,manifest_key):
        temp=tempfile.TemporaryDirectory();dest=Path(temp.name)/'bundle';shutil.copytree(self.bundle,dest)
        path=dest/filename;value=json.loads(path.read_text());mutator(value);write_json(path,value)
        mp=dest/'deployment_manifest.json';manifest=json.loads(mp.read_text());manifest[manifest_key]=d.file_descriptor(path,dest)
        write_json(mp,manifest)
        return temp,dest,d.file_descriptor(mp,dest)['sha256']

    def test_resigned_freeze_mutation_rejects_without_trusting_pass_flag(self):
        temp,dest,sha=self._changed_json_bundle('freeze.json',lambda x:x['after'].update(optimizer='f'*64),'freeze_evidence')
        with temp,self.assertRaises(ValueError):d.load_deployment_calibration(dest,sha,self.bundle_ctx,require_active=False)

    def test_resigned_refit_ledger_and_validation_before_fit_reject(self):
        for mutate in (lambda x:x.update(fit_attempts=2),lambda x:x.update(holdout_threshold_revisions=1),
                       lambda x:x.update(validation_started_ns=2),lambda x:x.update(validation_started_ns=x['candidate_sealed_ns']),
                       lambda x:x.update(event_clock='time_ns')):
            temp,dest,sha=self._changed_json_bundle('ledger.json',mutate,'phase_ledger')
            with temp,self.assertRaises(ValueError):d.load_deployment_calibration(dest,sha,self.bundle_ctx,require_active=False)

    def test_resigned_non_independent_or_wrong_replay_transcript_reject(self):
        for mutate in (lambda x:x['cells'][0].update(replay_process_id=100),
                       lambda x:x['cells'][0].update(reference_trace_sha256='f'*64)):
            temp,dest,sha=self._changed_json_bundle('replay.json',mutate,'replay_evidence')
            with temp,self.assertRaises(ValueError):d.load_deployment_calibration(dest,sha,self.bundle_ctx,require_active=False)

    def test_input_rows_unchanged_by_fit_and_validation(self):
        before=d.records_sha256(self.fit);vb=d.records_sha256(self.valid)
        d.fit_candidate(self.fit,self.ctx);d.evaluate_candidate(self.candidate,self.valid,self.ctx)
        self.assertEqual(before,d.records_sha256(self.fit));self.assertEqual(vb,d.records_sha256(self.valid))

    def test_epistemic_and_temporal_error_are_independently_calibrated_and_gate_reset(self):
        limits=self.candidate['thresholds_by_gap']['20']
        self.assertEqual(limits['max_epistemic_disagreement'],.005)
        self.assertEqual(limits['max_temporal_prediction_error'],.02)
        for field,failure in [('epistemic_disagreement','risk_reset_has_qualified_epistemic'),
                              ('temporal_prediction_error','risk_reset_has_qualified_temporal_error')]:
            rows=copy.deepcopy(self.valid);rows[0][field][:]=.9
            report=d.evaluate_candidate(self.candidate,rows,self.ctx)
            self.assertFalse(report['validation']['passed'])
            self.assertIn(failure,report['validation']['per_cell']['55010:1']['failed_checks'])

    def test_missing_interval_temporal_error_cannot_be_fabricated(self):
        rows=copy.deepcopy(self.fit);rows[0]['temporal_prediction_error']=np.zeros(9,np.float64)
        with self.assertRaises(ValueError):d.fit_candidate(rows,self.ctx)
        rows=copy.deepcopy(self.fit);rows[0]['temporal_prediction_error'][0]=np.nan
        with self.assertRaises(ValueError):d.fit_candidate(rows,self.ctx)

    def test_independent_planning_epistemic_bound_is_not_particle_jsd(self):
        rows=copy.deepcopy(self.valid)
        rows[0]['plans'][-1]['epistemic_disagreement'][:]=.9
        report=d.evaluate_candidate(self.candidate,rows,self.ctx)
        self.assertIn('plan_h5_epistemic_disagreement_upper',
            report['validation']['per_cell']['55010:1']['failed_checks'])

    def test_calibration_only_candidate_validation_is_not_deployment_authority(self):
        self.assertTrue(d.validate_candidate_for_calibration(self.candidate,self.ctx))
        with self.assertRaises(ValueError):d.verify_deployment_handle(self.candidate,self.ctx)
        wrong=copy.deepcopy(self.ctx);wrong['epistemic_heads_sha256']='f'*64
        with self.assertRaises(ValueError):d.validate_candidate_for_calibration(self.candidate,wrong)
        candidate=copy.deepcopy(self.candidate);candidate['fit_attempt']=True
        candidate['sha256']=d._digest({k:v for k,v in candidate.items() if k!='sha256'})
        with self.assertRaises(ValueError):d.validate_candidate_for_calibration(candidate,self.ctx)

    def test_raw_reward_extrapolation_is_retained_in_calibration_and_reported(self):
        rows=copy.deepcopy(self.valid);rows[0]['plans'][0]['rewards'][:]=-2.
        before=d.records_sha256(rows)
        result=d.evaluate_candidate(self.candidate,rows,self.ctx)
        self.assertEqual(result['validation']['per_cell']['55010:1']['planning_reward_out_of_support_fraction'],1/3)
        self.assertEqual(before,d.records_sha256(rows))
        self.assertTrue(result['validation']['passed'])

    def test_plan_outcome_clock_prevents_previous_action_off_by_one(self):
        rows=copy.deepcopy(self.fit);rows[0]['plans'][0]['previous_action'][0]=.2
        with self.assertRaises(ValueError):d.fit_candidate(rows,self.ctx)

    def test_resigned_epistemic_freeze_changed_or_wrong_binding_rejects(self):
        for mutate in (lambda x:x['after'].update(epistemic_heads='f'*64),
                       lambda x:(x['before'].update(epistemic_heads='f'*64),x['after'].update(epistemic_heads='f'*64))):
            temp,dest,sha=self._changed_json_bundle('freeze.json',mutate,'freeze_evidence')
            with temp,self.assertRaises(ValueError):d.load_deployment_calibration(dest,sha,self.bundle_ctx,require_active=False)

    def test_uniform_pilot_weights_do_not_force_impossible_ESS_one(self):
        before=d.records_sha256(self.fit)
        self.assertTrue(np.all(self.fit[0]['candidate_weights']==1/8))
        self.assertEqual(self.candidate['thresholds_by_gap']['1']['min_ess_ratio'],1/8)
        self.assertEqual(before,d.records_sha256(self.fit))
        self.assertEqual(self.candidate['ess_fit_derivation']['1']['distribution_count'],80)
        rows=copy.deepcopy(self.valid)
        for row in rows:
            row['candidate_weights'][row['gap']:]=np.array([1.,0.,0.,0.,0.,0.,0.,0.])
        self.assertTrue(d.evaluate_candidate(self.candidate,rows,self.ctx)['validation']['passed'])

    def test_fit_reweight_keeps_all_rejection_explicit(self):
        energies=np.array([[.1,.2,.3,.4,.5,.6,.7,.8],[2.,2.,2.,2.,2.,2.,2.,2.]],np.float64)
        ess,rejected=d.fit_reweighted_ess(energies,.1,.1)
        self.assertEqual(ess.tolist(),[1/8]);self.assertEqual(rejected,1)
        with self.assertRaises(ValueError):d.fit_reweighted_ess(energies,.01,.1)


if __name__=='__main__':unittest.main()
