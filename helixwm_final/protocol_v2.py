"""Full Final registration; pure plans, never training or deployment authority.

The preserved C1 registration is a validated input for the unchanged official
learner/environment contract. Its identity is NOT reused for this new method.
Only explicit overrides below define Full Final. The emitted JSON and this
source must both enter the new runtime/source manifest. Unknown-fault stage D
cannot be silently added to the known-mask confirmatory stage B.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np

from . import protocol as c1
from .deployment_calibration import COVERAGE_REVISION
from helixwm_matched.protocol import (array_sha256, canonical_json, fixed_gap_mask,
    gap_segments, mask_sha256, protocol_sha256)

PROTOCOL_ID = 'dreamerv3_vs_helixwm_final_full_walker_20261106'
BASE_CONFIG_SHA256 = '8132c0e25d8e783afcdeced58d57022c3eb2aa82b0ec5cd4085a951c43f25514'
CORE_GAPS = (1, 5, 10, 20, 50)
CALIBRATION_GAPS = (1, 5, 10, 20, 50, 100)
CONTROLLERS = ('clean_native', 'black_frame_native', 'prior_only', 'helix_final')
FIRST_CALIBRATION_ACTION = 5500
VIDEO_PANELS = (('D', 'black_frame_native'), ('F', 'helix_final'), ('D', 'prior_only'))


def _base_protocol():
    path = Path(__file__).resolve().parents[1] / 'configs/helixwm_final_matched_20261105.json'
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != BASE_CONFIG_SHA256:
        raise ValueError('Preserved C1 registration bytes changed; no implicit protocol migration')
    return c1.validate_protocol(json.loads(raw))


def registered_protocol():
    """Return the exact new registration; runtime PASS status is separate."""
    p = _base_protocol()
    p.update(schema_version=2, protocol_id=PROTOCOL_ID,
        scope='two_fresh_independent_D_and_Full_Final_runs_same_Walker_environment_and_visible_data_rules')
    p['registration_provenance'] = {
        'user_document_truncated_after': 'section_7_2_H2_Helix',
        'full_final_details': 'engineering_registration_20261106_not_verbatim_user_document',
        'preserved_C1_protocol_id': c1.PROTOCOL_ID,
        'preserved_C1_config_sha256': BASE_CONFIG_SHA256,
        'prior_experimental_protocol': 'dreamerv3_vs_helixwm_walker_20261026',
        'new_protocol_identity_required': True, 'old_results_preserved': True}
    p['training']['arms'] = [
        {'id': 'D', 'name': 'native_image_DreamerV3', 'controller': 'black_frame_native',
         'consumes_availability_flag': False},
        {'id': 'F', 'name': 'HelixWM_Final_full', 'controller': 'helix_final',
         'consumes_availability_flag': True}]
    p['training']['additional_trainable_parameters'] = 3456
    p['training']['additional_loss_terms'] = []
    p['training']['additional_parameters_scope'] = 'three_auxiliary_ridge_predictors_not_author_Dreamer_parameters'
    p['training']['auxiliary_by_arm'] = {
        'D': {'fitted_coefficients': 3456, 'mode': 'diagnostic_only_for_frozen_inference_ablations',
              'may_affect_training_action': False, 'may_affect_author_updates': False,
              'report_extra_CPU_memory_storage_separately': True},
        'F': {'fitted_coefficients': 3456, 'mode': 'visible_training_dynamics_uncertainty_proxy',
              'may_affect_training_action_after_bound_calibration': True,
              'may_affect_author_updates': False}}
    p['training']['bootstrap'] = {
        'both_arms_use_native_black_frame_controller_through_actions': 5500,
        'random_external_action_decisions': 5000,
        'native_actor_learning_bootstrap_action_decisions': 500,
        'author_learner_updates_at_5000': 0,
        'author_learner_updates_at_5500': 125,
        'all_Final_fusion_planning_unknown_guard_gates_inactive': True,
        'auxiliary_heads_may_fit_adjacent_visible_training_pairs': True,
        'Final_first_calibration_boundary_actions': 5500,
        'Final_5501_requires_new_pin_and_verified_model_heads_calibration': True,
        'uncalibrated_formal_continuation_allowed': False,
        'candidate_trajectory_archive_starts_at_Final_action': 5501,
        'earlier_native_state_and_action_evidence': 'native_controller_checkpoint_and_replay',
        'fabricated_precalibration_candidate_trajectories_allowed': False,
        'FullController_bootstrap_mode_is_isolated_probe_only': True}
    p['training']['auxiliary_predictor'] = {
        'schema': 'helixwm_visible_bootstrap_dynamics_v1',
        'members': 3, 'input_features': 22, 'random_features_including_bias': 64,
        'image_features': 16, 'outputs': 18, 'fitted_coefficients': 3456,
        'features': '4x4_means_of_RGB_channel_mean_divided_by_255_plus_executed_action6',
        'targets': 'next_visible_image_features16_actual_repeat2_reward_divided_by2_actual_continuation',
        'feature_map': '[1,cos(W_m[x,a]+b_m)]/sqrt(64)',
        'feature_projection_distribution': 'independent_standard_normal_per_member',
        'feature_phase_distribution': 'independent_uniform_minus_pi_plus_pi_per_member',
        'fixed_prior_distribution': 'independent_standard_normal_64x18_per_member',
        'fixed_prior_scale': 0.1, 'ridge_lambda': 1.0,
        'bootstrap_weights': 'independent_Poisson_1_per_member_per_visible_training_transition',
        'objective': 'sum_bootstrap_weight_squared_prediction_error_plus_squared_coefficient_norm',
        'solve': 'separate_ridge_sufficient_statistics_solve_at_pin',
        'rng_seed_root': 0,
        'streams': 'SeedSequence(seed).spawn(6):first3_maps_last3_bootstrap',
        'update_partitions': ['training'],
        'both_adjacent_observations_must_be_actually_available': True,
        'hidden_clean_or_calibration_or_eval_data_allowed': False,
        'extra_author_gradients_or_loss_terms': [],
        'minimum_ready_visible_pairs': 32,
        'each_member_requires_positive_bootstrap_count': True,
        'frozen_per_inference_version': True,
        'pin_SHA_and_complete_statistics_RNG_in_checkpoint_required': True,
        'uncertainty_scope': 'independent_bootstrap_coarse_visible_dynamics_proxy_not_Dreamer_ensemble',
        'claims_calibrated_Bayesian_posterior': False,
        'disagreement_raw_predictions_clipped_before_variance': False}
    p['architecture'] = {
        'implementation_stage': 'Final_full_v2', 'controller': 'helix_final',
        'max_candidates': 8, 'maximum_supported_gap_actions': 100,
        'known_mask_stage_B_availability_used_by_Final': True,
        'unknown_fault_detector_stage_D_only': True,
        'unknown_fault_detector_claim': 'visible_image_engineering_anomaly_screen_not_semantic_or_physical_failure_oracle',
        'independent_auxiliary_predictors': 3, 'independent_Dreamer_backbones': 1,
        'auxiliary_epistemic_proxy_available_only_after_measured_validation': True,
        'candidate_variance_is_epistemic_uncertainty': False,
        'original_dreamer_loss_unchanged': True, 'additional_trainable_parameters': 3456,
        'additional_author_loss_terms': [],
        'candidate_rejection_required': True,
        'real_posterior_cannot_be_overwritten_by_candidate': True,
        'single_action_advance_required': True,
        'recovery_requires_observed_error_and_all_uncertainty_ESS_gates': True,
        'prediction_does_not_equal_observed_state': True,
        'unbounded_missingness_or_physical_safety_claim_allowed': False}
    p['planning'] = {
        'operator_version': 'helixwm_final_planning_statistics_v2',
        'proposals': ['native_actor_first_action', 'hold_previous_executed_action'],
        'hold_before_first_executed_action': 'zeros6',
        'identical_first_action_proposals_deduplicated': True,
        'horizons_measured': [1, 3, 5], 'horizon_preference_per_proposal': [5, 3, 1],
        'shorter_horizon_only_if_longer_fails_own_gate': True,
        'candidate_selection': 'highest_fixed_utility_among_longest_accepted_horizon_per_proposal',
        'reward_units': 'repeat2_DMC_reward_in_0_2',
        'raw_model_reward_predictions_retained': True,
        'utility_reward_projection': 'clip_raw_reward_prediction_to_0_2_only_for_utility',
        'utility_reward_normalizer': '2_times_H',
        'utility_reward_formula': 'sum_survival_before_step_times_projected_reward_divided_by_2H',
        'termination_risk': 'weighted_1_minus_product_of_continuations',
        'uncertainty_aggregation': 'maximum_over_measured_horizon',
        'utility_weights': {'reward': 1.0, 'uncertainty': 0.25, 'disagreement': 0.25,
                            'termination': 1.0, 'slew': 0.1},
        'zero_reward_calibration_supported': True,
        'scratch_plans_may_change_executed_history_or_actor_RNG': False,
        'actual_control_and_calibration_share_same_operator': True,
        'fallback': 'bounded_native_prior_actor_with_registered_amplitude_and_slew',
        'fallback_constraint_conflict': 'explicit_failure_no_action_not_silent_relaxation',
        'physical_safety_proven': False}
    p['calibration'] = {
        'registration_source': 'engineering_choice_not_verbatim_user_document',
        'schema': 'helixwm_final_closed_loop_deployment_v2',
        'measurement_protocol': 'frozen_candidate_closed_loop_sufficient_statistics_v2',
        'fit_environment_seeds': {'start': 55000, 'count': 10},
        'validation_environment_seeds': {'start': 55010, 'count': 10},
        'missing_lengths_actions': list(CALIBRATION_GAPS),
        'warmup_actions': 16, 'recovery_actions': 8, 'missing_observation_start': 17,
        'actions_per_cell_formula': '16_plus_gap_plus_8',
        'cells_per_split': 60, 'cells_per_model_version': 120,
        'actual_actions_per_model_version': 6600, 'DMC_calls_per_model_version': 13200,
        'shared_open_loop_bank_can_replace_closed_loop_evidence': False,
        'fit_controller': 'registered_Final_measurement_pilot_with_gates_inactive',
        'validation_controller': 'exact_fitted_candidate_sealed_before_heldout_execution',
        'fit_upper_quantile': 0.99, 'fit_ESS_lower_quantile': 0.05,
        'required_upper_coverage_per_gap': 0.95, 'required_lower_coverage_per_gap': 0.90,
        'coverage_revision': copy.deepcopy(COVERAGE_REVISION),
        'structural_caps': {'lambda_max': 0.25, 'kl_limit': 0.02,
                            'action_amplitude': 1.0, 'slew_limit': 2.0},
        'fit_once_before_independent_validation': True,
        'heldout_driven_threshold_retry_allowed': False,
        'synthetic_fixture_can_activate_production': False,
        'self_asserted_pass_can_activate_production': False,
        'external_bindings': ['source_sha256', 'model_sha256', 'bank_manifest_sha256',
                             'protocol_sha256', 'collector_sha256', 'epistemic_heads_sha256', 'model_version'],
        'strict_file_reload_and_independent_replay_transcripts_required': True,
        'all_registered_gaps_required_per_version': True,
        'recompute_every_pinned_inference_version': True,
        'F_first_boundary_actions': 5500, 'F_subsequent_pin_interval_actions': 500,
        'F_training_pins': 1989, 'F_final_evaluation_pin': 1, 'D_final_cross_controller_pin': 1,
        'minimum_registered_model_version_suites': 1991,
        'versions_reusing_same_seeds_are_not_independent_holdouts': True,
        'gap5_threshold_extrapolation_to_gap100_allowed': False,
        'model_or_heads_version_mismatch': 'block_production_interaction_until_current_evidence_passes',
        'bank_actions_or_images_to_learner_allowed': False,
        'hidden_clean_target_to_policy_allowed': False, 'final_test_seeds_allowed': False,
        'calibration_is_model_weight_update': False,
        'calibration_interactions_in_training_budget': False,
        'compressed_sufficient_statistics_require_bound_raw_action_RNG_anchor_and_replay_verification': True,
        'GPU_cost_and_complete_storage_gate_required': True,
        'failed_attempts_preserved_and_counted_separately': True,
        'changing_pin_cadence_requires_new_explicit_registration': True}
    p['trajectory_evidence'] = {
        'schema': 'helixwm_final_trajectory_archive_v1',
        'record_schema': 'helixwm_final_trajectory_record_v1',
        'actual_and_prospective_records_separate': True,
        'maximum_candidates': 8, 'maximum_missing_age': 100, 'maximum_ring_references': 100,
        'retain_true_latent_action_predicted_image_reward_continuation_probability_arrays': True,
        'retain_candidate_ancestor_model_source_RNG_and_executed_prefix_identities': True,
        'retain_entropy_JSD_ESS_weights_observed_errors_and_calibration_identity': True,
        'missing_observation_error': None,
        'unknown_or_unmeasured_value_may_be_fabricated': False,
        'epistemic_proxy_is_separate_from_same_model_particle_JSD': True,
        'immutable_NPZ_JSON_manifest_SHA_and_no_pickle_required': True,
        'checkpoint_contains_cursor_and_verified_dependencies_not_duplicated_pixels': True,
        'writable_restore_cannot_overwrite_later_tail': True,
        'streaming_100step_reconstruction_gate_required': True,
        'measured_full_array_storage_and_transfer_cost_required_before_launch': True}
    p['unknown_observation_faults'] = {
        'stage': 'D_separate_unknown_fault_inference_stress',
        'enabled_during_known_mask_training_or_stage_B': False,
        'ground_truth_fault_mask_or_clean_pixels_to_detector_allowed': False,
        'visible_features': ['mean', 'std', 'high_frequency', 'temporal_l1', 'repeat_run', 'model_innovation_MSE'],
        'pre_observation_prediction_identity_and_timing_required': True,
        'labels_allowed_only_in_offline_fit_and_evaluation': True,
        'fault_families': ['black', 'freeze', 'noise'],
        'confirm_frames': 2, 'recover_frames': 2, 'minimum_repeat_run': 3,
        'static_frame_alone_can_instantly_trigger_fault': False,
        'fit_environment_seeds': {'start': 55000, 'count': 10},
        'validation_environment_seeds': {'start': 55010, 'count': 10},
        'file_backed_current_model_source_data_protocol_binding_required': True,
        'synthetic_only_evidence_can_activate_production': False,
        'quarantined_image_may_update_posterior': False,
        'metrics': ['false_positive_rate', 'false_negative_rate', 'quarantine_latency', 'recovery_latency'],
        'claims_universal_semantic_fault_recognition': False,
        'distinct_frozen_stage_D_matrix_and_resource_gate_required_before_execution': True,
        'unknown_fault_results_may_be_pooled_into_stage_B_primary_statistics': False}
    p['smoke'] = {
        'partition': 'isolated_actual_size_learning_smoke_not_formal_training',
        'arms': ['D', 'F'], 'actions_per_arm': 5500, 'DMC_calls_per_arm': 11000,
        'episodes_per_arm': 11, 'total_paired_episode_executions': 22,
        'environment_seeds': {'start': 72000, 'count': 11, 'step': 1},
        'mask_rng_seed': 1710000, 'prefill_action_rng_seed': 1810000,
        'author_updates_per_arm': 125, 'formal_seed_reuse_allowed': False,
        'fresh_process_reload_and_next_update_probe_required': True,
        'post5500_calibration_and_controller_probes_have_separate_actual_counts': True,
        'no_formal_data_initialization_from_smoke': True}
    p['final_evaluation']['controller_namespace'] = 'helixwm_final_full_v2_not_C1_or_old_V2'
    p['final_evaluation']['same_checkpoint_helix_inference_requires_its_own_heads_and_calibration'] = True
    p['final_evaluation']['stage_B_unknown_fault_detector_enabled'] = False
    p['final_evaluation']['stage_C_mechanism_and_stage_D_unknown_require_separate_frozen_manifests'] = True
    p['final_evaluation']['known_mask_B_and_long_gap_stress_only_in_this_matrix'] = True
    p['operations']['full_array_trajectory_and_calibration_artifacts_must_enter_capacity_estimate'] = True
    p['operations']['old_342GiB_estimate_is_sufficient_for_full_Final_launch'] = False
    p['operations']['formal_launch_requires_actual_measured_gate_not_registration_flags'] = True
    p['operations']['additional_automatic_unregistered_evaluation_stages_allowed'] = False
    p['operations']['checkpoint_includes_auxiliary_statistics_RNG_pinned_heads_and_trajectory_dependencies'] = True
    p['launch_gate_requirements'] = [
        'new_full_v2_source_protocol_and_environment_frozen',
        'D_and_F_fresh_initialization_and_same_native_first5500',
        'identical_environment_fixed_action_trace_and_formal_mask_bank',
        'both_actual_size_5500_action_125update_GPU_smokes',
        'D_auxiliary_tracker_never_changes_native_actions_or_author_updates',
        'no_hidden_data_to_policy_author_learner_or_auxiliary_heads',
        '5500_boundary_current_model_heads_closed_loop_calibration_and_independent_replay',
        'all6gaps_sufficient_statistics_and_shared_planning_operator_validated',
        '100step_full_array_trajectory_reconstruction_and_storage_measurement',
        'strict_checkpoint_reload_and_fresh_process_next_sample_update_recovery',
        'new_run_archive_ACK_transaction_and_live_capacity',
        'measured_all_versions_calibration_trajectory_replay_checkpoint_ZIP_capacity',
        'no_sustained_uncalibrated_formal_actions_after5500']
    p['planned_interaction_accounting'] = {
        'training_actions': 2000000, 'periodic_evaluation_actions': 500000,
        'known_mask_final_evaluation_actions': 760000,
        'closed_loop_calibration_actions_per_version': 6600,
        'minimum_registered_calibration_model_versions': 1991,
        'minimum_closed_loop_calibration_actions': 13140600,
        'minimum_known_mask_total_actions_excluding_extra_replays_smoke_failures_and_C_D': 16400600,
        'minimum_known_mask_total_DMC_calls_excluding_extra_replays_smoke_failures_and_C_D': 32801200,
        'two_isolated_learning_smokes_actions': 11000,
        'shared_open_loop_bank_can_substitute_for_these_actions': False,
        'independent_replay_cost_is_additional_and_must_be_measured': True,
        'per_version_calibration_environment_actions_are_not_training_actions': True,
        'extra_C_D_smoke_failed_attempts_and_recovery_probes_need_separate_actual_counts': True}
    return p


def validate_protocol(config):
    """Strict registration validation, not an executable launch approval."""
    if type(config) is not dict:
        raise ValueError('Full Final protocol must be a dictionary')
    canonical_json(config)
    c1._equal(config, registered_protocol(), 'Full_Final_v2')
    return copy.deepcopy(config)


def load_protocol(path):
    return validate_protocol(json.loads(Path(path).read_text(encoding='utf-8-sig')))


def validate_measurement_parent_protocol(current, parent, *, trusted_parent_sha256):
    """Permit only the registered prospective coverage aggregation correction.

    This authorizes frozen measurements under a new protocol identity, never a
    training resume. Parent bytes and its auxiliary-head bindings stay intact.
    The caller must obtain parent JSON from the verified frozen source archive.
    """
    validate_protocol(current)
    parent_sha = protocol_sha256(parent)
    if parent_sha != trusted_parent_sha256:
        raise ValueError('Parent protocol differs from trusted bootstrap identity')
    current_sha = protocol_sha256(current)
    if parent_sha == current_sha:
        mode = 'same_protocol'
        changed = []
    else:
        old = copy.deepcopy(current)
        calibration = old['calibration']
        del calibration['coverage_revision']
        calibration['required_upper_coverage_per_cell'] = calibration.pop('required_upper_coverage_per_gap')
        calibration['required_lower_coverage_per_cell'] = calibration.pop('required_lower_coverage_per_gap')
        if canonical_json(old) != canonical_json(parent):
            raise ValueError('Parent migration changed fields beyond registered coverage aggregation')
        mode = 'prospective_preheldout_coverage_aggregation_only'
        changed = ['calibration.required_upper_coverage_per_cell->required_upper_coverage_per_gap',
                   'calibration.required_lower_coverage_per_cell->required_lower_coverage_per_gap',
                   'calibration.coverage_revision']
    return {'mode': mode, 'parent_protocol_sha256': parent_sha,
            'measurement_protocol_sha256': current_sha, 'changed_fields': changed,
            'coverage_revision': copy.deepcopy(COVERAGE_REVISION),
            'environment_training_masks_seeds_threshold_fit_and_budgets_unchanged': True,
            'parent_identity_rewritten': False, 'training_resume_authorized': False}


def base_environment_protocol(config):
    """Explicit validated C1 contract for its unchanged environment factory."""
    validate_protocol(config)
    return _base_protocol()


def seed_partitions(config):
    validate_protocol(config)
    return {'training': list(range(100000, 102000)),
            'periodic': list(range(50000, 50005)),
            'calibration_fit': list(range(55000, 55010)),
            'calibration_validation': list(range(55010, 55020)),
            'final_known_mask': list(range(60000, 60040)),
            'isolated_smoke': list(range(72000, 72011))}


def training_seed(config, episode_index, *, partition='formal'):
    validate_protocol(config)
    if partition not in ('formal', 'smoke'):
        raise ValueError('Unknown training reset partition')
    count, start = (2000, 100000) if partition == 'formal' else (11, 72000)
    if type(episode_index) is not int or not 0 <= episode_index < count:
        raise ValueError('Episode index is outside its registered partition')
    return start + episode_index


def make_mask_bank(config, *, partition='formal'):
    validate_protocol(config)
    if partition == 'formal':
        # No mutation/projection of the old validator's global constants.
        return c1.make_mask_bank(_base_protocol())
    if partition != 'smoke':
        raise ValueError('Unknown mask partition')
    rng = np.random.default_rng(1710000)
    masks = np.ones((11, 501), np.bool_)
    for row in masks:
        t = 1
        while t < 500:
            if rng.random() < .02:
                gap = int(rng.integers(1, 6))
                if t + gap <= 500:
                    row[t:t+gap] = False
                    t += gap + 1
                    continue
            t += 1
    return masks


def learner_updates_at(config, completed_actions):
    validate_protocol(config)
    if type(completed_actions) is not int or not 0 <= completed_actions <= 1000000:
        raise ValueError('Action count must be an integer within the formal budget')
    return max(0, completed_actions - 5000) // 4


def phase_at(config, arm, completed_actions):
    """Name the NEXT action's phase. This never grants deployment authority."""
    learner_updates_at(config, completed_actions)
    if arm not in ('D', 'F'):
        raise ValueError('Only D and F are registered')
    if completed_actions == 1000000:
        return 'TRAINING_BUDGET_COMPLETE'
    if completed_actions < 5000:
        return 'NATIVE_RANDOM_PREFILL'
    if completed_actions < 5500:
        return 'NATIVE_LEARNING_BOOTSTRAP'
    return 'NATIVE_DREAMER' if arm == 'D' else 'FINAL_REQUIRES_VERIFIED_CURRENT_PIN'


def calibration_required_at(config, arm, completed_actions):
    learner_updates_at(config, completed_actions)
    if arm not in ('D', 'F'):
        raise ValueError('Only D and F are registered')
    return completed_actions == 1000000 if arm == 'D' else (
        completed_actions >= 5500 and completed_actions % 500 == 0)


def build_calibration_plan(config):
    """120 REAL closed-loop windows per pin, never one shared action tape."""
    validate_protocol(config)
    rows = []
    for phase, seeds in (('fit', range(55000, 55010)), ('validation', range(55010, 55020))):
        for seed in seeds:
            for gap in CALIBRATION_GAPS:
                window = c1.calibration_subwindow(gap)
                rows.append({'phase': phase, 'seed': seed, 'gap': gap,
                    'actions': window['actions'], 'DMC_calls': 2*window['actions'],
                    'observations': window['observations'], 'missing_start': 17,
                    'first_recovery_observation': 17+gap,
                    'availability_sha256': window['availability_sha256'],
                    'action_source': 'registered_pilot' if phase == 'fit' else 'sealed_fitted_candidate',
                    'training_updates': 0, 'counts_as_training': False})
    return rows


def build_eval_plan(config, stratum='core'):
    validate_protocol(config)
    if stratum not in ('core', 'long_gap100_stress'):
        raise ValueError('Unknown/C/D stage needs a separate frozen manifest')
    spec = config['final_evaluation']['strata'][stratum]
    rows = []
    for arm in ('D', 'F'):
        for controller in CONTROLLERS:
            gaps = [0] if controller == 'clean_native' and spec['include_clean'] else (
                [] if controller == 'clean_native' else spec['gaps'])
            for gap in gaps:
                mask = np.ones(501, np.bool_) if gap == 0 else fixed_gap_mask(gap)
                for seed in range(60000, 60040):
                    rows.append({'case_id': f'{arm}_{controller}_seed{seed}_gap{gap:02d}',
                        'protocol_id': PROTOCOL_ID, 'stratum': stratum,
                        'arm': arm, 'controller': controller, 'seed': seed, 'gap': gap,
                        'missing_start': 150 if gap else None,
                        'actions': 500, 'dmc_calls': 1000, 'availability_sha256': mask_sha256(mask),
                        'unknown_fault_guard_enabled': False,
                        'record_video_frames': (arm, controller) in VIDEO_PANELS and seed == 60000 and gap == 20})
    if len(rows) != spec['episodes'] or len({row['case_id'] for row in rows}) != len(rows):
        raise RuntimeError('Frozen evaluation cardinality differs')
    return rows
