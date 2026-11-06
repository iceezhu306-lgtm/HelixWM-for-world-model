"""Pure CPU registration tests; no simulator, model, SSH, or launch."""
import copy
import json
from pathlib import Path
import pickle
import unittest

import numpy as np

from helixwm_final import protocol as p
from helixwm_matched import protocol as old

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'configs/helixwm_final_matched_20261105.json'


class FinalProtocolTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(CONFIG.read_text(encoding='utf-8'))

    def test_design_validated_without_claiming_launch_and_defensive_copy(self):
        result = p.load_protocol(CONFIG)
        self.assertEqual(result['launch_status_at_registration'], 'NOT_LAUNCHED')
        self.assertFalse(result['implementation_complete_at_registration'])
        self.assertFalse(result['architecture']['true_epistemic_available'])
        result['training']['arms'][0]['id'] = 'changed'
        self.assertEqual(self.config['training']['arms'][0]['id'], 'D')
        self.assertEqual(p.validate_protocol(self.config), self.config)

    def test_core_hyperparameters_and_no_third_arm_are_strict(self):
        mutations = [('actions_per_arm', 999999), ('batch_size', 8), ('batch_length', 32),
                     ('prefill_actions', 0), ('train_ratio_positions_per_decision', 128),
                     ('nominal_updates_at_budget', 248751), ('paired_training_seeds', [1]),
                     ('compute_dtype', 'float32'), ('additional_trainable_parameters', 1),
                     ('additional_loss_terms', ['new_loss']), ('fresh_initialization_required', False),
                     ('historical_checkpoint_or_replay_initialization_allowed', True)]
        for field, value in mutations:
            config = copy.deepcopy(self.config); config['training'][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError): p.validate_protocol(config)
        config = copy.deepcopy(self.config)
        config['training']['arms'].append({'id': 'H2'})
        with self.assertRaises(ValueError): p.validate_protocol(config)

    def test_bool_float_and_nonfinite_cannot_change_discrete_contract(self):
        for section, field, value in [('training', 'batch_size', 16.0),
                ('training', 'environments_per_arm', True), ('architecture', 'max_candidates', 8.0),
                ('calibration', 'window_actions', 124.0), ('architecture', 'true_epistemic_available', 0)]:
            config = copy.deepcopy(self.config); config[section][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError): p.validate_protocol(config)
        for value in (float('nan'), float('inf'), float('-inf')):
            config = copy.deepcopy(self.config); config['unused_nonfinite'] = value
            with self.assertRaises(ValueError): p.validate_protocol(config)

    def test_no_hidden_data_or_mismatched_masks_allowed(self):
        for section, field in [('training', 'learner_hidden_clean_targets_allowed'),
                ('training', 'synthetic_candidate_images_to_learner_allowed'),
                ('calibration', 'bank_actions_or_images_to_learner_allowed'),
                ('calibration', 'hidden_clean_target_to_policy_allowed'),
                ('final_evaluation', 'unavailable_clean_frame_access_allowed')]:
            config = copy.deepcopy(self.config); config[section][field] = True
            with self.subTest(field=field), self.assertRaises(ValueError): p.validate_protocol(config)
        config = copy.deepcopy(self.config)
        config['training']['training_missingness']['mask_seed'] += 1
        with self.assertRaises(ValueError): p.validate_protocol(config)

    def test_old_validator_is_unchanged_and_rejects_new_registration(self):
        original = json.loads((ROOT / 'configs/dreamerv3_helixwm_matched_20261026.json').read_text())
        self.assertEqual(old.validate_protocol(original), original)
        with self.assertRaises(ValueError): old.validate_protocol(self.config)
        with self.assertRaises(ValueError): p.validate_protocol(original)

    def test_mask_bank_exactly_matches_old_algorithm_without_global_rng_changes(self):
        original = json.loads((ROOT / 'configs/dreamerv3_helixwm_matched_20261026.json').read_text())
        before = pickle.dumps(np.random.get_state())
        actual = p.make_mask_bank(self.config)
        expected = old.make_mask_bank(original)
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(p.array_sha256(actual), old.array_sha256(expected))
        self.assertEqual(actual.dtype, np.bool_)
        self.assertEqual(actual.shape, (2000, 501))
        self.assertTrue(actual[:, 0].all() and actual[:, -1].all())
        self.assertTrue(all(length <= 5 for row in actual for _, length in p.gap_segments(row)))
        self.assertEqual(before, pickle.dumps(np.random.get_state()))

    def test_training_reset_mapping_and_index_rejection(self):
        self.assertEqual(p.training_seed(self.config, 0), 100000)
        self.assertEqual(p.training_seed(self.config, 1999), 101999)
        for bad in (-1, 2000, True, 0.0):
            with self.assertRaises(ValueError): p.training_seed(self.config, bad)

    def test_seed_partitions_are_disjoint_and_calibration_never_uses_test_seeds(self):
        parts = p.seed_partitions(self.config)
        seen = set()
        for seeds in parts.values():
            self.assertFalse(seen.intersection(seeds)); seen.update(seeds)
        self.assertEqual(parts['calibration_fit'], list(range(55000, 55010)))
        self.assertEqual(parts['calibration_validation'], list(range(55010, 55020)))
        for key, start in [('fit_environment_seeds', 60000), ('validation_environment_seeds', 55000)]:
            config = copy.deepcopy(self.config); config['calibration'][key]['start'] = start
            with self.assertRaises(ValueError): p.validate_protocol(config)

    def test_one_shared_calibration_bank_has_exact_registered_budget(self):
        plan = p.build_calibration_plan(self.config)
        self.assertEqual(len(plan), 20)
        self.assertEqual(sum(x['actions'] for x in plan), 2480)
        self.assertEqual(sum(x['dmc_calls'] for x in plan), 4960)
        self.assertEqual(len({x['seed'] for x in plan}), 20)
        self.assertTrue(all(x['action_seed'] == 730000 + x['seed'] for x in plan))
        self.assertTrue(all(x['observations'] == 125 for x in plan))
        self.assertTrue(all(x['gaps'] == [1, 5, 10, 20, 50, 100] for x in plan))

    def test_calibration_masks_have_correct_warmup_missing_and_recovery_indexing(self):
        for gap in p.CALIBRATION_GAPS:
            row = p.calibration_subwindow(gap); mask = row['availability']
            self.assertEqual(mask.shape, (16 + gap + 8 + 1,))
            self.assertTrue(mask[:17].all())
            self.assertFalse(mask[17:17 + gap].any())
            self.assertEqual(len(mask[17 + gap:]), 8)
            self.assertTrue(mask[17 + gap:].all())
            self.assertEqual(row['availability_sha256'], p.array_sha256(mask))
        row = p.calibration_subwindow(100)
        self.assertEqual((row['actions'], row['observations'], row['first_recovery_observation']), (124, 125, 117))
        for bad in (0, 4, 101, True, 5.0):
            with self.assertRaises(ValueError): p.calibration_subwindow(bad)

    def test_calibration_action_tapes_are_bounded_deterministic_and_rng_isolated(self):
        before = pickle.dumps(np.random.get_state())
        a = p.calibration_actions(55000); b = p.calibration_actions(55000)
        np.testing.assert_array_equal(a, b)
        self.assertEqual(a.shape, (124, 6)); self.assertEqual(a.dtype, np.float32)
        self.assertTrue(np.isfinite(a).all() and (np.abs(a) <= 1).all())
        self.assertFalse(np.array_equal(a, p.calibration_actions(55001)))
        self.assertEqual(before, pickle.dumps(np.random.get_state()))
        for bad in (60000, 54999, 55020, True, 55000.0):
            with self.assertRaises(ValueError): p.calibration_actions(bad)

    def test_every_gap_requires_current_version_calibration_no_extrapolation(self):
        for field, value in [('missing_lengths_actions', [1, 5]), ('missing_observation_start', 16),
                ('recompute_every_pinned_inference_version', False),
                ('current_model_and_bank_hash_binding_required', False),
                ('gap5_threshold_extrapolation_to_gap100_allowed', True),
                ('sustained_uncalibrated_formal_launch_allowed', True)]:
            config = copy.deepcopy(self.config); config['calibration'][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError): p.validate_protocol(config)

    def test_core_and_stress_matrix_identity_counts_pairing_and_one_video(self):
        core = p.build_eval_plan(self.config)
        stress = p.build_eval_plan(self.config, 'long_gap100_stress')
        self.assertEqual((len(core), len(stress)), (1280, 240))
        all_rows = core + stress
        self.assertEqual(len({r['case_id'] for r in all_rows}), 1520)
        self.assertEqual(sum(r['actions'] for r in all_rows), 760000)
        self.assertEqual(sum(r['dmc_calls'] for r in all_rows), 1520000)
        self.assertEqual(sum(r['record_video_frames'] for r in core), 3)
        self.assertFalse(any(r['record_video_frames'] for r in stress))
        self.assertEqual({r['arm'] for r in all_rows}, {'D', 'F'})
        self.assertEqual({r['gap'] for r in stress}, {100})
        for gap in p.CALIBRATION_GAPS:
            for seed in range(60000, 60040):
                paired = [r for r in all_rows if r['seed'] == seed and r['gap'] == gap]
                self.assertEqual(len(paired), 6)
                self.assertEqual(len({r['availability_sha256'] for r in paired}), 1)
        with self.assertRaises(ValueError): p.build_eval_plan(self.config, 'unknown')

    def test_historical_seed_exposure_and_calibrated_gap100_must_be_disclosed(self):
        config = copy.deepcopy(self.config); config['final_evaluation']['claims_fully_unseen_blind_test'] = True
        with self.assertRaises(ValueError): p.validate_protocol(config)
        config = copy.deepcopy(self.config)
        config['final_evaluation']['strata']['long_gap100_stress']['claims_uncalibrated_horizon_ood'] = True
        with self.assertRaises(ValueError): p.validate_protocol(config)

    def test_metrics_thresholds_archive_and_budget_cannot_be_relaxed(self):
        for section, field, value in [('analysis', 'holm_primary_tests', 1), ('analysis', 'alpha', 0.1),
                ('operations', 'archive_before_ack_required', False),
                ('operations', 'old_experiment_cleanup_allowed', True),
                ('operations', 'checkpoint_count_per_arm', 1),
                ('operations', 'old_protocol_global_monkeypatch_allowed', True)]:
            config = copy.deepcopy(self.config); config[section][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError): p.validate_protocol(config)


if __name__ == '__main__':
    unittest.main()
