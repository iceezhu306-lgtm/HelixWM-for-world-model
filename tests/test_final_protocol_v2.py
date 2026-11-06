"""Full-v2 registration tests; no launch, real training or acceptance claim."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import pickle
import unittest

import numpy as np

from helixwm_final import protocol as c1
from helixwm_final import protocol_v2 as p
from helixwm_final import epistemic, planning, deployment_calibration
from helixwm_matched import protocol as old

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT/'configs/helixwm_final_full_20261106.json'


class FullFinalProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.registration = json.loads(CONFIG.read_text(encoding='utf-8'))

    def setUp(self):
        self.config = copy.deepcopy(self.registration)

    def test_exact_new_registration_does_not_claim_completion_or_launch(self):
        validated = p.load_protocol(CONFIG)
        self.assertEqual(validated, p.registered_protocol())
        self.assertNotEqual(validated['protocol_id'], c1.PROTOCOL_ID)
        self.assertFalse(validated['implementation_complete_at_registration'])
        self.assertEqual(validated['launch_status_at_registration'], 'NOT_LAUNCHED')
        self.assertNotEqual(p.protocol_sha256(validated), c1.protocol_sha256(p.base_environment_protocol(validated)))
        validated['training']['arms'][0]['id'] = 'tamper'
        self.assertEqual(p.registered_protocol()['training']['arms'][0]['id'], 'D')

    def test_old_C1_bytes_and_original_validators_preserved(self):
        raw = (ROOT/'configs/helixwm_final_matched_20261105.json').read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), p.BASE_CONFIG_SHA256)
        base = json.loads(raw)
        self.assertEqual(c1.validate_protocol(base), base)
        original = json.loads((ROOT/'configs/dreamerv3_helixwm_matched_20261026.json').read_text())
        self.assertEqual(old.validate_protocol(original), original)
        with self.assertRaises(ValueError): c1.validate_protocol(self.config)
        with self.assertRaises(ValueError): old.validate_protocol(self.config)
        with self.assertRaises(ValueError): p.validate_protocol(base)

    def test_frozen_parent_migration_only_allows_prospective_coverage_aggregation(self):
        parent=copy.deepcopy(self.config)
        gate=parent['calibration']
        del gate['coverage_revision']
        gate['required_upper_coverage_per_cell']=gate.pop('required_upper_coverage_per_gap')
        gate['required_lower_coverage_per_cell']=gate.pop('required_lower_coverage_per_gap')
        old_sha=p.protocol_sha256(parent)
        self.assertEqual(old_sha,'10f3cfdc653d5778dfda69715ac71ec11185b00064d4a909a4f70e5029dde02f')
        result=p.validate_measurement_parent_protocol(self.config,parent,trusted_parent_sha256=old_sha)
        self.assertEqual(result['mode'],'prospective_preheldout_coverage_aggregation_only')
        self.assertFalse(result['training_resume_authorized'])
        self.assertNotEqual(result['parent_protocol_sha256'],result['measurement_protocol_sha256'])
        for section,field,value in [('calibration','fit_upper_quantile',.98),
                                    ('calibration','warmup_actions',17),
                                    ('environment','action_repeat',3)]:
            bad=copy.deepcopy(parent);bad[section][field]=value
            with self.subTest(field=field),self.assertRaisesRegex(ValueError,'beyond registered'):
                p.validate_measurement_parent_protocol(self.config,bad,trusted_parent_sha256=p.protocol_sha256(bad))
        with self.assertRaisesRegex(ValueError,'trusted bootstrap'):
            p.validate_measurement_parent_protocol(self.config,parent,trusted_parent_sha256='f'*64)
        self.assertEqual(p.validate_measurement_parent_protocol(self.config,self.config,
            trusted_parent_sha256=p.protocol_sha256(self.config))['mode'],'same_protocol')

    def test_exact_environment_formal_masks_and_reset_seeds_match_previous_run(self):
        previous = json.loads((ROOT/'configs/dreamerv3_helixwm_matched_20261026.json').read_text())
        for key in ('domain', 'task', 'image_size', 'camera_id', 'action_repeat', 'action_dimensions',
                    'episode_action_decisions', 'episode_dmc_calls', 'privileged_current_state_to_policy'):
            self.assertEqual(self.config['environment'][key], previous['environment'][key])
        before = pickle.dumps(np.random.get_state())
        full = p.make_mask_bank(self.config)
        original = old.make_mask_bank(previous)
        np.testing.assert_array_equal(full, original)
        self.assertEqual(p.array_sha256(full), old.array_sha256(original))
        self.assertEqual(full.shape, (2000, 501))
        self.assertEqual(p.training_seed(self.config, 0), 100000)
        self.assertEqual(p.training_seed(self.config, 1999), 101999)
        self.assertEqual(before, pickle.dumps(np.random.get_state()))

    def test_smoke_is_same_environment_but_has_disjoint_reset_and_rng_partition(self):
        masks = p.make_mask_bank(self.config, partition='smoke')
        self.assertEqual(masks.shape, (11, 501))
        self.assertTrue(masks[:, 0].all() and masks[:, -1].all())
        self.assertTrue(all(length <= 5 for row in masks for _, length in p.gap_segments(row)))
        self.assertEqual([p.training_seed(self.config, i, partition='smoke') for i in range(11)], list(range(72000, 72011)))
        np.testing.assert_array_equal(masks, p.make_mask_bank(self.config, partition='smoke'))
        self.assertFalse(np.array_equal(masks, p.make_mask_bank(self.config)[:11]))
        self.assertEqual(self.config['smoke']['total_paired_episode_executions'], 22)
        parts = p.seed_partitions(self.config)
        seen = set()
        for name, seeds in parts.items():
            self.assertFalse(seen.intersection(seeds), name)
            seen.update(seeds)
        for invalid in (-1, 11, True, 0.0):
            with self.assertRaises(ValueError): p.training_seed(self.config, invalid, partition='smoke')

    def test_native_bootstrap_125_update_and_first_Final_action_boundary(self):
        counts = {0: 0, 4999: 0, 5000: 0, 5003: 0, 5004: 1,
                  5499: 124, 5500: 125, 1000000: 248750}
        for actions, updates in counts.items():
            self.assertEqual(p.learner_updates_at(self.config, actions), updates)
        for arm in ('D', 'F'):
            self.assertEqual(p.phase_at(self.config, arm, 4999), 'NATIVE_RANDOM_PREFILL')
            self.assertEqual(p.phase_at(self.config, arm, 5000), 'NATIVE_LEARNING_BOOTSTRAP')
            self.assertEqual(p.phase_at(self.config, arm, 5499), 'NATIVE_LEARNING_BOOTSTRAP')
        self.assertEqual(p.phase_at(self.config, 'F', 5500), 'FINAL_REQUIRES_VERIFIED_CURRENT_PIN')
        self.assertEqual(p.phase_at(self.config, 'D', 5500), 'NATIVE_DREAMER')
        self.assertTrue(p.calibration_required_at(self.config, 'F', 5500))
        self.assertTrue(p.calibration_required_at(self.config, 'F', 6000))
        self.assertFalse(p.calibration_required_at(self.config, 'F', 5501))
        self.assertFalse(p.calibration_required_at(self.config, 'D', 5500))
        self.assertTrue(p.calibration_required_at(self.config, 'D', 1000000))
        self.assertEqual(self.config['training']['bootstrap']['candidate_trajectory_archive_starts_at_Final_action'], 5501)
        for invalid in (-1, 1000001, True, 5000.0):
            with self.assertRaises(ValueError): p.learner_updates_at(self.config, invalid)

    def test_closed_loop_per_version_cost_replaces_shared_bank_accounting(self):
        rows = p.build_calibration_plan(self.config)
        self.assertEqual(len(rows), 120)
        self.assertEqual(len({(r['phase'], r['seed'], r['gap']) for r in rows}), 120)
        self.assertEqual(sum(r['actions'] for r in rows), 6600)
        self.assertEqual(sum(r['DMC_calls'] for r in rows), 13200)
        self.assertTrue(all(r['actions'] == 16+r['gap']+8 for r in rows))
        self.assertTrue(all(r['training_updates'] == 0 and not r['counts_as_training'] for r in rows))
        self.assertEqual({r['action_source'] for r in rows if r['phase'] == 'validation'}, {'sealed_fitted_candidate'})
        self.assertFalse({r['seed'] for r in rows if r['phase'] == 'fit'} &
                         {r['seed'] for r in rows if r['phase'] == 'validation'})
        suites = len(range(5500, 1000001, 500)) + 1
        self.assertEqual(suites, 1991)
        accounting = self.config['planned_interaction_accounting']
        calibration_actions = suites * sum(r['actions'] for r in rows)
        self.assertEqual(accounting['minimum_closed_loop_calibration_actions'], calibration_actions)
        self.assertEqual(accounting['minimum_known_mask_total_actions_excluding_extra_replays_smoke_failures_and_C_D'],
                         2000000 + 500000 + 760000 + calibration_actions)
        self.assertNotIn('one_shared_calibration_bank_actions', accounting)

    def test_auxiliary_registration_matches_actual_dimensions_and_no_D_control_effect(self):
        aux = self.config['training']['auxiliary_predictor']
        self.assertEqual((aux['members'], aux['random_features_including_bias'], aux['outputs']),
                         (epistemic.MEMBERS, epistemic.FEATURES, epistemic.OUTPUTS))
        self.assertEqual(aux['fitted_coefficients'], 3 * 64 * 18)
        self.assertEqual(aux['ridge_lambda'], epistemic.RIDGE)
        self.assertEqual(aux['fixed_prior_scale'], epistemic.PRIOR_SCALE)
        d = self.config['training']['auxiliary_by_arm']['D']
        self.assertFalse(d['may_affect_training_action'])
        self.assertFalse(d['may_affect_author_updates'])
        self.assertTrue(d['report_extra_CPU_memory_storage_separately'])
        self.assertEqual(aux['update_partitions'], ['training'])
        self.assertTrue(aux['both_adjacent_observations_must_be_actually_available'])
        self.assertFalse(aux['hidden_clean_or_calibration_or_eval_data_allowed'])
        self.assertFalse(aux['claims_calibrated_Bayesian_posterior'])

    def test_shared_planning_and_current_heads_deployment_bindings_match_modules(self):
        plan = self.config['planning']
        self.assertEqual(plan['operator_version'], planning.OPERATOR_VERSION)
        self.assertEqual(plan['horizons_measured'], list(planning.HORIZONS))
        self.assertEqual(plan['utility_weights'], planning.UTILITY_WEIGHTS)
        self.assertTrue(plan['raw_model_reward_predictions_retained'])
        self.assertTrue(plan['zero_reward_calibration_supported'])
        cal = self.config['calibration']
        self.assertEqual(cal['schema'], deployment_calibration.SCHEMA)
        self.assertEqual(cal['measurement_protocol'], deployment_calibration.MEASUREMENT_PROTOCOL)
        self.assertEqual(set(cal['external_bindings']), set(deployment_calibration.CONTEXT_FIELDS) - {'measurement_kind'})
        self.assertEqual(cal['structural_caps'], deployment_calibration.STRUCTURAL_CAPS)

    def test_known_matrix_has_same_1280_core_240_stress_and_three_video_sources(self):
        core = p.build_eval_plan(self.config)
        stress = p.build_eval_plan(self.config, 'long_gap100_stress')
        self.assertEqual((len(core), len(stress)), (1280, 240))
        self.assertEqual(sum(x['actions'] for x in core + stress), 760000)
        self.assertEqual(sum(x['record_video_frames'] for x in core + stress), 3)
        self.assertFalse(any(x['unknown_fault_guard_enabled'] for x in core + stress))
        for gap in p.CALIBRATION_GAPS:
            paired = [x for x in core + stress if x['gap'] == gap and x['seed'] == 60000]
            self.assertEqual(len(paired), 6)
            self.assertEqual(len({x['availability_sha256'] for x in paired}), 1)
        for unknown in ('D_unknown', 'mechanism', 'unknown_fault'):
            with self.assertRaises(ValueError): p.build_eval_plan(self.config, unknown)

    def test_no_faked_unknown_fault_or_archive_capability_enters_known_test(self):
        guard = self.config['unknown_observation_faults']
        self.assertFalse(guard['enabled_during_known_mask_training_or_stage_B'])
        self.assertFalse(guard['ground_truth_fault_mask_or_clean_pixels_to_detector_allowed'])
        self.assertFalse(guard['synthetic_only_evidence_can_activate_production'])
        self.assertFalse(guard['quarantined_image_may_update_posterior'])
        self.assertTrue(guard['distinct_frozen_stage_D_matrix_and_resource_gate_required_before_execution'])
        evidence = self.config['trajectory_evidence']
        self.assertIsNone(evidence['missing_observation_error'])
        self.assertTrue(evidence['retain_true_latent_action_predicted_image_reward_continuation_probability_arrays'])
        self.assertTrue(evidence['measured_full_array_storage_and_transfer_cost_required_before_launch'])
        self.assertFalse(self.config['operations']['old_342GiB_estimate_is_sufficient_for_full_Final_launch'])

    def test_training_budget_information_and_algorithm_cannot_be_silently_changed(self):
        changes = [
            ('training', 'actions_per_arm', 999999), ('training', 'paired_training_seeds', [1]),
            ('training', 'nominal_updates_at_budget', 248751),
            ('training', 'learner_hidden_clean_targets_allowed', True),
            ('training', 'additional_trainable_parameters', 0),
            ('environment', 'action_repeat', 1), ('environment', 'image_size', 32),
            ('environment', 'camera_id', True),
            ('calibration', 'heldout_driven_threshold_retry_allowed', True),
            ('calibration', 'F_first_boundary_actions', 5000),
            ('calibration', 'F_subsequent_pin_interval_actions', 10000),
            ('unknown_observation_faults', 'enabled_during_known_mask_training_or_stage_B', True),
            ('operations', 'old_experiment_cleanup_allowed', True),
            ('operations', 'archive_before_ack_required', False)]
        for section, field, value in changes:
            bad = copy.deepcopy(self.config); bad[section][field] = value
            with self.subTest(section=section, field=field), self.assertRaises(ValueError):
                p.validate_protocol(bad)
        for value in (float('nan'), float('inf'), 5000.0, True):
            bad = copy.deepcopy(self.config); bad['training']['prefill_actions'] = value
            with self.assertRaises(ValueError): p.validate_protocol(bad)
        bad = copy.deepcopy(self.config); bad['self_asserted_all_gates_passed'] = True
        with self.assertRaises(ValueError): p.validate_protocol(bad)


if __name__ == '__main__':
    unittest.main()
