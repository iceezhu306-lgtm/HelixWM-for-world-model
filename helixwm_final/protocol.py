"""Final-C1 registration and pure data plans; never launch or train anything.

The old protocol is not rewritten, projected, or monkeypatched. Only its pure
hash/mask/file utilities are reused and must enter the new runtime manifest.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np

from helixwm_matched.protocol import (
    UPSTREAM_COMMIT, array_sha256, atomic_json, canonical_json, fixed_gap_mask,
    gap_segments, mask_sha256, protocol_sha256, validate_mask,
)

PROTOCOL_ID = 'dreamerv3_vs_helixwm_final_walker_20261105'
CORE_GAPS = [1, 5, 10, 20, 50]
CALIBRATION_GAPS = [1, 5, 10, 20, 50, 100]
CONTROLLERS = ['clean_native', 'black_frame_native', 'prior_only', 'helix_final']
VIDEO_PANELS = [('D', 'black_frame_native'), ('F', 'helix_final'), ('D', 'prior_only')]


def _equal(actual, expected, name):
    if type(actual) is not type(expected):
        raise ValueError(f'Protocol type mismatch: {name}')
    if isinstance(expected, dict):
        if set(actual) != set(expected):
            raise ValueError(f'Protocol keys differ: {name}')
        for key, value in expected.items():
            _equal(actual[key], value, name + '.' + key)
    elif isinstance(expected, list):
        if len(actual) != len(expected):
            raise ValueError(f'Protocol length differs: {name}')
        for index, (left, right) in enumerate(zip(actual, expected)):
            _equal(left, right, f'{name}[{index}]')
    elif actual != expected:
        raise ValueError(f'Protocol value differs: {name}')


def _fields(actual, expected, name):
    for key, value in expected.items():
        if key not in actual:
            raise ValueError(f'Missing protocol field: {name}.{key}')
        _equal(actual[key], value, name + '.' + key)


def seed_partitions(config):
    """Environment-reset partitions, excluding model/mask/action RNG roots."""
    tr = config['environment']['identical_environment_contract']['training_episode_reset_seeds']
    cal = config['calibration']
    ev = config['final_evaluation']['environment_seeds']
    def span(spec):
        if type(spec['start']) is not int or type(spec['count']) is not int or spec['count'] < 1:
            raise ValueError('Seed ranges require integer start and positive count')
        return list(range(spec['start'], spec['start'] + spec['count']))
    return {'training': span(tr), 'periodic': list(config['periodic_evaluation']['environment_seeds']),
            'calibration_fit': span(cal['fit_environment_seeds']),
            'calibration_validation': span(cal['validation_environment_seeds']),
            'final': span(ev), 'preflight': list(config['final_evaluation']['smoke_seeds'])}


def validate_protocol(config):
    """Validate the declared engineering design, not a live launch gate."""
    if type(config) is not dict:
        raise ValueError('Protocol must be a dictionary')
    canonical_json(config)
    try:
        _fields(config, {'schema_version': 1, 'protocol_id': PROTOCOL_ID,
                        'design_status': 'REGISTERED_ENGINEERING_DESIGN_PENDING_PREFLIGHT',
                        'launch_status_at_registration': 'NOT_LAUNCHED',
                        'implementation_complete_at_registration': False}, 'root')
        _fields(config['official_core'], {'commit': UPSTREAM_COMMIT,
                'presets': ['defaults', 'dmc_vision', 'size12m'],
                'modify_loss_optimizer_actor_or_critic_formula': False}, 'official_core')
        _fields(config['environment'], {'domain': 'walker', 'task': 'walk',
                'observation': 'RGB_only', 'image_size': 64, 'camera_id': 0,
                'action_dimensions': 6, 'action_repeat': 2, 'episode_action_decisions': 500,
                'episode_dmc_calls': 1000, 'dynamics_changed': False,
                'privileged_current_state_to_policy': False}, 'environment')
        _equal(config['environment']['identical_environment_contract']['training_episode_reset_seeds'],
               {'start': 100000, 'count': 2000, 'step': 1}, 'training_reset_seeds')
        tr = config['training']
        _fields(tr, {'actions_per_arm': 1000000, 'dmc_calls_per_arm': 2000000,
                'combined_training_actions': 2000000, 'combined_training_dmc_calls': 4000000,
                'episodes_per_arm': 2000, 'paired_training_seeds': [0],
                'training_seed_count_per_arm': 1, 'batch_size': 16, 'batch_length': 64,
                'replay_context': 1, 'custom_burnin': 0, 'replay_capacity_transitions': 1000000,
                'train_ratio_positions_per_decision': 256, 'prefill_actions': 5000,
                'prefill_action_rng_seed': 810000, 'replay_sampler_rng_seed': 0,
                'nominal_updates_at_budget': 248750, 'compute_dtype': 'bfloat16',
                'checkpoint_every_actions': 10000, 'environments_per_arm': 1,
                'learner_image_source': 'corrupted_visible_RGB_only_for_both_arms',
                'learner_missing_image': 'uint8_zeros', 'learner_hidden_clean_targets_allowed': False,
                'arm_specific_masked_loss_removal_allowed': False,
                'synthetic_candidate_images_to_learner_allowed': False,
                'fresh_initialization_required': True,
                'historical_checkpoint_or_replay_initialization_allowed': False,
                'additional_trainable_parameters': 0, 'additional_loss_terms': []}, 'training')
        _equal(tr['arms'], [
            {'id': 'D', 'name': 'native_image_DreamerV3', 'controller': 'black_frame_native', 'consumes_availability_flag': False},
            {'id': 'F', 'name': 'HelixWM_Final_C1', 'controller': 'helix_final', 'consumes_availability_flag': True}], 'training.arms')
        _fields(tr['policy_parameter_sync'], {'rule': 'copy_at_each_episode_boundary_then_hold_for_500_actions',
                'shared_by_both_arms': True, 'official_core_untouched': True}, 'policy_sync')
        _fields(tr['training_missingness'], {'mask_seed': 710000,
                'start_probability_at_eligible_timestamp': 0.02,
                'burst_length_uniform_inclusive': [1, 5],
                'minimum_valid_observations_between_bursts': 1, 'actual_contiguous_gap_max': 5,
                'burst_must_fit_before_final_observation': True,
                'initial_and_final_observations_valid': True, 'same_mask_per_paired_episode': True,
                'masks_apply_during_prefill': True, 'policy_reward_dependent': False}, 'training_missingness')
        _fields(config['architecture'], {'implementation_stage': 'Final_C1', 'controller': 'helix_final',
                'max_candidates': 8, 'maximum_supported_gap_actions': 100,
                'known_availability_required': True, 'automatic_unknown_fault_detector_implemented': False,
                'true_epistemic_available': False, 'ensemble_independent_models': 0,
                'candidate_variance_is_epistemic_uncertainty': False,
                'original_dreamer_loss_unchanged': True, 'additional_trainable_parameters': 0,
                'additional_loss_terms': [], 'candidate_rejection_required': True,
                'real_posterior_cannot_be_overwritten_by_candidate': True,
                'single_action_advance_required': True}, 'architecture')
        _fields(config['calibration'], {'fit_environment_seeds': {'start': 55000, 'count': 10},
                'validation_environment_seeds': {'start': 55010, 'count': 10},
                'window_actions': 124, 'window_observations': 125, 'warmup_actions': 16,
                'maximum_gap_actions': 100, 'recovery_actions': 8, 'missing_observation_start': 17,
                'missing_lengths_actions': CALIBRATION_GAPS, 'action_rng_base': 730000,
                'action_rng_seed_formula': '730000+environment_seed', 'windows_per_seed': 1,
                'total_bank_actions': 2480, 'total_bank_dmc_calls': 4960,
                'bank_collection_repeated_per_model': False, 'bank_collection_in_training_budget': False,
                'bank_actions_or_images_to_learner_allowed': False,
                'hidden_clean_target_to_policy_allowed': False, 'final_test_seeds_allowed': False,
                'recompute_every_pinned_inference_version': True,
                'all_registered_gaps_required_per_version': True,
                'gap_specific_thresholds_required': True,
                'gap5_threshold_extrapolation_to_gap100_allowed': False,
                'current_model_and_bank_hash_binding_required': True,
                'calibration_is_model_weight_update': False,
                'version_mismatch_disables': ['fusion', 'planning'],
                'sustained_uncalibrated_formal_launch_allowed': False,
                'gpu_cost_and_coverage_gate_required': True,
                'threshold_fit_formula_frozen_in_source_before_launch': True}, 'calibration')
        _fields(config['periodic_evaluation'], {'every_actions': 20000,
                'environment_seeds': [50000, 50001, 50002, 50003, 50004],
                'conditions_by_arm': {'D': [['clean_native', 0], ['black_frame_native', 20]],
                                      'F': [['clean_native', 0], ['helix_final', 20]]},
                'training_updates': 0, 'total_actions_both_arms': 500000,
                'total_dmc_calls_both_arms': 1000000}, 'periodic_evaluation')
        ev = config['final_evaluation']
        _fields(ev, {'checkpoints': ['D_final_1000000', 'F_final_1000000'],
                'controllers': CONTROLLERS, 'environment_seeds': {'start': 60000, 'count': 40},
                'missing_observation_start': 150, 'missing_lengths_actions': CORE_GAPS,
                'training_updates': 0, 'evaluation_time_adaptation_allowed': False,
                'unavailable_clean_frame_access_allowed': False,
                'historical_seed_exposure_disclosed': True,
                'known_historically_observed_final_seeds': [60000],
                'claims_fully_unseen_blind_test': False,
                'smoke_seeds': [72000, 72001], 'core_total_episodes': 1280,
                'core_total_actions': 640000, 'core_total_dmc_calls': 1280000,
                'total_episodes': 1520, 'total_actions': 760000, 'total_dmc_calls': 1520000}, 'final_evaluation')
        _equal(ev['strata'], {'core': {'gaps': CORE_GAPS, 'include_clean': True,
                'episodes': 1280, 'actions': 640000, 'dmc_calls': 1280000},
                'long_gap100_stress': {'gaps': [100], 'include_clean': False,
                'episodes': 240, 'actions': 120000, 'dmc_calls': 240000,
                'outside_training_gap_distribution': True, 'inside_calibration_gap_support': True,
                'claims_uncalibrated_horizon_ood': False}}, 'strata')
        _fields(config['operations'], {'tmux_required': True, 'inspection_interval_hours': 6,
                'auto_restart_or_resume': False, 'local_output_drive': 'E:',
                'archive_before_ack_required': True, 'checkpoint_count_per_arm': 100,
                'complete_checkpoint_and_all_replay_versions_required': True,
                'archive_receipt_binds_run_arm_source_protocol_and_checkpoint': True,
                'latest_checkpoint_dependencies_retained_remotely': True,
                'old_experiment_cleanup_allowed': False,
                'automatic_E_unique_data_deletion_allowed': False,
                'new_training_namespace_manifest_required': True,
                'old_protocol_global_monkeypatch_allowed': False}, 'operations')
        _equal(config['operations']['final_video_selection'], {'environment_seed': 60000,
                'gap_actions': 20, 'panels': ['D_final_1000000/black_frame_native',
                'F_final_1000000/helix_final', 'D_final_1000000/prior_only'],
                'formal_mp4_count': 1}, 'video')
        _fields(config['metrics'], {'primary_absolute_score': 'mean(arm_u) over [150,150+gap+100), then equal mean over five gaps per seed',
                'functional_failure': {'normalized_reward_below': 0.2, 'consecutive_actions': 20,
                    'window': 'gap_plus_100_recovery_actions', 'certifies_fall_or_safety': False},
                'unresolved_technical_or_hash_failure_blocks_positive_acceptance': True}, 'metrics')
        _fields(config['analysis'], {'holm_primary_tests': 2, 'alpha': 0.05,
                'bootstrap_replicates': 10000, 'paired_sign_randomization_draws': 100000,
                'claim_multi_training_seed_generalization': False,
                'long_gap100_is_separate_secondary_stratum': True,
                'core_confirmatory_gaps_remain': CORE_GAPS}, 'analysis')
        _fields(config['analysis']['clean_performance_guard'], {'ratio_minimum': 0.95,
                'rule': 'F_mean >= 0.95 * D_mean'}, 'clean_performance_guard')
        _fields(config['planned_interaction_accounting'], {'training_actions': 2000000,
                'periodic_evaluation_actions': 500000, 'one_shared_calibration_bank_actions': 2480,
                'final_evaluation_actions': 760000,
                'total_actions_excluding_smoke_failed_attempts_and_recovery_probes': 3262480,
                'total_dmc_calls_excluding_smoke_failed_attempts_and_recovery_probes': 6524960,
                'per_version_calibration_bank_replay_adds_environment_actions': False}, 'accounting')
        parts = seed_partitions(config)
        seen = set()
        for name, seeds in parts.items():
            if any(type(seed) is not int for seed in seeds) or len(set(seeds)) != len(seeds) or seen.intersection(seeds):
                raise ValueError('Repeated, invalid, or overlapping seed partition: ' + name)
            seen.update(seeds)
    except (KeyError, TypeError) as exc:
        raise ValueError('Incomplete Final-C1 protocol: ' + str(exc)) from exc
    return copy.deepcopy(config)


def load_protocol(path):
    return validate_protocol(json.loads(Path(path).read_text(encoding='utf-8-sig')))


def training_seed(config, episode_index):
    spec = config['environment']['identical_environment_contract']['training_episode_reset_seeds']
    if type(episode_index) is not int or not 0 <= episode_index < spec['count']:
        raise ValueError('Training episode index must be an integer in [0,2000)')
    return spec['start'] + episode_index * spec['step']


def make_mask_bank(config):
    validate_protocol(config)
    rng = np.random.default_rng(config['training']['training_missingness']['mask_seed'])
    bank = np.ones((2000, 501), np.bool_)
    for row in bank:
        t = 1
        while t < 500:
            if rng.random() < 0.02:
                length = int(rng.integers(1, 6))
                if t + length <= 500:
                    row[t:t + length] = False
                    t += length + 1
                    continue
            t += 1
    return bank


def calibration_subwindow(gap):
    """x0 reset; actions 1..16 warm up; x17 starts the missing interval.

    For gap100, x17..x116 are missing and x117..x124 recover. Thus
    the one bank window has 124 actions and 125 observations, including x0.
    """
    if type(gap) is not int or gap not in CALIBRATION_GAPS:
        raise ValueError('Unregistered calibration gap')
    actions = 16 + gap + 8
    mask = np.ones(actions + 1, np.bool_)
    mask[17:17 + gap] = False
    return {'gap': gap, 'actions': actions, 'observations': actions + 1,
            'missing_start': 17, 'first_recovery_observation': 17 + gap,
            'recovery_observations': 8, 'availability_sha256': array_sha256(mask),
            'availability': mask}


def build_calibration_plan(config):
    validate_protocol(config)
    parts = seed_partitions(config)
    return [{'split': split, 'seed': seed, 'action_seed': 730000 + seed,
             'actions': 124, 'dmc_calls': 248, 'observations': 125,
             'gaps': list(CALIBRATION_GAPS)}
            for split in ('calibration_fit', 'calibration_validation') for seed in parts[split]]


def calibration_actions(environment_seed):
    """The one shared bank's bounded action tape; never an actor rollout."""
    if type(environment_seed) is not int or not 55000 <= environment_seed <= 55019:
        raise ValueError('Only registered independent calibration seeds are allowed')
    return np.random.default_rng(730000 + environment_seed).uniform(
        -1.0, 1.0, size=(124, 6)).astype(np.float32)


def build_eval_plan(config, stratum='core'):
    validate_protocol(config)
    if stratum not in config['final_evaluation']['strata']:
        raise ValueError('Unregistered evaluation stratum')
    spec = config['final_evaluation']['strata'][stratum]
    rows = []
    for arm in ('D', 'F'):
        for controller in CONTROLLERS:
            gaps = [0] if controller == 'clean_native' and spec['include_clean'] else (
                [] if controller == 'clean_native' else spec['gaps'])
            for gap in gaps:
                mask = np.ones(501, bool) if gap == 0 else fixed_gap_mask(gap)
                for seed in range(60000, 60040):
                    rows.append({'case_id': f'{arm}_{controller}_seed{seed}_gap{gap:02d}',
                        'stratum': stratum, 'arm': arm, 'controller': controller,
                        'seed': seed, 'gap': gap, 'missing_start': 150 if gap else None,
                        'actions': 500, 'dmc_calls': 1000, 'availability_sha256': mask_sha256(mask),
                        'record_video_frames': (arm, controller) in VIDEO_PANELS and seed == 60000 and gap == 20})
    if len(rows) != spec['episodes'] or len({row['case_id'] for row in rows}) != len(rows):
        raise RuntimeError('Evaluation cardinality differs from registration')
    return rows
