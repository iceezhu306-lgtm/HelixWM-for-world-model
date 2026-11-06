"""Budget-matched independent-bank registration; preserves full-v2 history."""
from __future__ import annotations
import copy
import json
from pathlib import Path
from . import protocol_v2 as previous
from . import protocol as original
from .bank_gate import SCHEMA,MEASUREMENT_PROTOCOL,SCOPE,SUPPORT_RULE
from helixwm_matched.protocol import canonical_json,protocol_sha256,fixed_gap_mask,mask_sha256

PROTOCOL_ID='dreamerv3_vs_helixwm_final_budget_matched_walker_20261106'
CORE_GAPS=previous.CORE_GAPS
CALIBRATION_GAPS=previous.CALIBRATION_GAPS
CONTROLLERS=previous.CONTROLLERS
FIRST_CALIBRATION_ACTION=5500
VIDEO_PANELS=previous.VIDEO_PANELS


def registered_protocol():
    p=previous.registered_protocol()
    p.update(schema_version=3,protocol_id=PROTOCOL_ID,
        scope='two_fresh_D_and_Final_runs_same_Walker_training_and_one_shared_independent_bank')
    p['registration_provenance']['preserved_full_v2_protocol_id']=previous.PROTOCOL_ID
    p['registration_provenance']['budget_revision']='restore_one_shared_2480_action_exogenous_bank_no_per_pin_environment_sampling'
    p['architecture']['implementation_stage']='Final_budget_matched_bank_v3'
    p['architecture']['calibration_claim']=SCOPE
    p['training']['bootstrap']['Final_5501_requires_new_pin_and_verified_model_heads_calibration']=False
    p['training']['bootstrap']['Final_5501_requires_current_bank_measurement_or_explicit_native_fallback']=True
    p['training']['bootstrap']['uncalibrated_formal_continuation_allowed']=True
    p['training']['bootstrap']['uncalibrated_continuation_scope']='native_prior_only_no_fusion_no_planning_logged_each_action'
    p['training']['bootstrap']['candidate_trajectory_archive_starts_at_Final_action']=5501
    base=previous._base_protocol()['calibration']
    p['calibration']={**copy.deepcopy(base),'schema':SCHEMA,'measurement_protocol':MEASUREMENT_PROTOCOL,
        'evidence_scope':SCOPE,'bank_origin_protocol_id':original.PROTOCOL_ID,
        'bank_origin_source_protocol_environment_manifest_bound_separately':True,
        'bank_available_identically_to_D_and_F':True,'independent_bank_not_author_or_auxiliary_training_data':True,
        'per_pin_additional_environment_actions':0,'per_pin_execution':'synchronous_model_only_forced_recorded_action_replay',
        'current_model_heads_source_protocol_bank_binding_required':True,
        'fit_upper_quantile':.99,'fit_ESS_lower_quantile':.05,'planning_horizons':[1,3,5],
        'coverage_scope':'equal_seed_mean_within_gap_empirical_not_IID',
        'required_upper_coverage_per_gap':.95,'required_lower_coverage_per_gap':.90,
        'support_rule':copy.deepcopy(SUPPORT_RULE),'planning_support_fit_only':True,
        'novel_plan_counterfactual_real_outcomes_available':False,
        'unsupported_plans':'reject_then_shorter_H_then_native_prior_fallback',
        'missing_inactive_or_stale_gate':'native_prior_fallback_fusion_planning_disabled',
        'invalid_integrity_or_nonfinite_evidence':'fail_and_preserve',
        'heldout_driven_threshold_retry_allowed':False,'all_failed_attempts_retained':True,
        'thresholds_frozen_before_corresponding_tests':True,'final_test_may_update_thresholds':False,
        'reused_bank_across_versions_is_new_independent_sample':False,
        'same_checkpoint_gate_numeric_recomputation_required':True,
        'fresh_process_replay_each_pin_required':False,
        'fresh_process_replay_required_at':['actual_GPU_preflight','final_checkpoint_audit'],
        'per_pin_full_model_anchor_duplication_required':False,
        'F_first_boundary_actions':5500,'F_subsequent_pin_interval_actions':500,
        'F_training_pins':1989,'F_final_evaluation_pin':1,'D_final_cross_controller_pin':1,
        'minimum_registered_model_version_suites':1991,
        'GPU_cost_and_complete_storage_gate_required':True}
    p['calibration']['sustained_uncalibrated_formal_launch_allowed']=True
    p['calibration']['version_mismatch_disables']=['fusion','planning']
    p['calibration']['shared_open_loop_bank_can_replace_closed_loop_certificate']=False
    p['planning']['action_support_required_every_candidate_and_depth']=True
    p['planning']['support_scope']='empirical_OOD_rejection_no_dynamics_error_or_physical_safety_bound'
    p['planning']['accepted_plan_label']='EXPERIMENTAL_SUPPORTED_PLAN'
    p['smoke']['post5500_calibration_and_controller_probes_have_separate_actual_counts']=True
    p['formal_recovery_probe']={'scope':'isolated_same_hardware_fresh_process_next_sample_update',
        'environment_seed':72120,'missing_observation_indices':[3],'actions_per_process':4,
        'DMC_calls_per_process':8,'author_updates_per_process':1,'fresh_processes':2,
        'total_actions':8,'total_DMC_calls':16,'total_author_updates':2,
        'counts_as_formal_training':False,'separate_output_directory_required':True,
        'cross_hardware_or_long_horizon_bitwise_claim':False,
        'proves_uninterrupted_environment_continuation':False}
    p['full_integration_preflight']={
        'scope':'isolated_fresh_native_then_bank_Final_integrated_learning_preflight',
        'environment_seeds':{'start':72000,'count':12},'mask_rng_seed':1710000,
        'mask_rule':'same_smoke_generator_0.02_start_uniform_gap1to5_boundary_visible',
        'first_eleven_masks_equal_preserved_smoke':True,'prefill_action_rng_seed':1810000,
        'counts':{'agent_steps':6000,'env_steps':12000,'episodes':12,'update_steps':250},
        'native_bootstrap_actions':5500,'bank_production_actions':500,
        'current_pin_bank_replay_boundaries':[5500,6000],'bank_replay_environment_actions':0,
        'formal_training_actions':0,'fresh_initialization':True,'old_checkpoint_loaded':False,
        'checkpoint_format':'helixwm_final_bank_preflight_boundary_v3','run_id_prefix':'preflight_',
        'original_D_and_F_shared_native_startup_exercised':True,
        'proves_two_complete_formal_runs':False,'inactive_bank_gate_uses_explicit_native_fallback':True}
    p['final_evaluation']['controller_namespace']='helixwm_final_budget_bank_v3_not_closed_loop_v2'
    p['final_evaluation']['final_threshold_artifact_frozen_before_any_test']=True
    p['launch_gate_requirements']=[
        'new_v3_source_protocol_and_original_environment_frozen',
        'one_original_2480_action_independent_bank_SHA_identical_for_both_arms',
        'identical_training_masks_seeds_budget_and_author_update_schedule',
        'actual_size_bootstrap_GPU_and_independent_recovery_evidence',
        'actual_GPU_bank_replay_zero_environment_calls_zero_training_updates_and_all_frozen_hashes',
        'FIT_sealed_before_HELDOUT_no_bank_data_to_author_or_auxiliary_training',
        'actual_GPU_preflight_independent_bank_numeric_recomputation',
        'action_support_and_inactive_current_version_native_fallback_verified',
        '100step_full_array_trajectory_reconstruction_and_measured_compressed_capacity',
        'strict_checkpoint_reload_and_fresh_process_next_sample_update_recovery',
        'new_run_archive_ACK_transaction_and_live_capacity',
        'measured_per_pin_offline_compute_and_complete_archive_capacity']
    p['planned_interaction_accounting']={
        'training_actions':2000000,'training_DMC_calls':4000000,'periodic_evaluation_actions':500000,
        'known_mask_final_evaluation_actions':760000,'one_shared_calibration_bank_actions':2480,
        'one_shared_calibration_bank_DMC_calls':4960,'per_pin_calibration_environment_actions':0,
        'total_actions_excluding_smoke_failed_attempts_and_recovery_probes':3262480,
        'total_DMC_calls_excluding_smoke_failed_attempts_and_recovery_probes':6524960,
        'shared_bank_information_available_to_both_arms_identically':True,
        'offline_model_compute_is_reported_separately_not_free':True,
        'unknown_fault_stage_D_not_automatically_added':True}
    return p


def validate_protocol(config):
    if type(config) is not dict or canonical_json(config)!=canonical_json(registered_protocol()):
        raise ValueError('Full budget-matched v3 registration differs')
    return copy.deepcopy(config)


def load_protocol(path):return validate_protocol(json.loads(Path(path).read_text(encoding='utf-8-sig')))


def _delegate(name,config,*args,**kwargs):
    validate_protocol(config)
    return getattr(previous,name)(previous.registered_protocol(),*args,**kwargs)


def base_environment_protocol(config):return _delegate('base_environment_protocol',config)
def _base_protocol():return previous._base_protocol()
def seed_partitions(config):return _delegate('seed_partitions',config)
def training_seed(config,episode_index,*,partition='formal'):return _delegate('training_seed',config,episode_index,partition=partition)
def make_mask_bank(config,*,partition='formal'):return _delegate('make_mask_bank',config,partition=partition)
def learner_updates_at(config,completed_actions):return _delegate('learner_updates_at',config,completed_actions)
def calibration_required_at(config,arm,completed_actions):return _delegate('calibration_required_at',config,arm,completed_actions)


def phase_at(config,arm,completed_actions):
    phase=_delegate('phase_at',config,arm,completed_actions)
    return 'FINAL_CURRENT_BANK_GATE_OR_EXPLICIT_NATIVE_FALLBACK' if phase=='FINAL_REQUIRES_VERIFIED_CURRENT_PIN' else phase


def build_calibration_plan(config):
    validate_protocol(config)
    # This plan denotes a single shared acquisition, NOT 120 new windows/pin.
    return original.build_calibration_plan(previous._base_protocol())


def full_preflight_masks(config):
    validate_protocol(config)
    import numpy as np
    rng=np.random.default_rng(config['full_integration_preflight']['mask_rng_seed'])
    masks=np.ones((12,501),np.bool_)
    for row in masks:
        t=1
        while t<500:
            if rng.random()<.02:
                gap=int(rng.integers(1,6))
                if t+gap<=500:
                    row[t:t+gap]=False;t+=gap+1;continue
            t+=1
    if not np.array_equal(masks[:11],previous.make_mask_bank(previous.registered_protocol(),partition='smoke')):
        raise RuntimeError('Integrated preflight changed preserved native smoke masks')
    return masks


def build_eval_plan(config,stratum='core'):
    rows=_delegate('build_eval_plan',config,stratum)
    for row in rows:row['protocol_id']=PROTOCOL_ID
    return rows
