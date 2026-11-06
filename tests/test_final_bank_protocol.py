"""Shared independent-bank budget registration, not a launch approval."""
import copy
import json
from pathlib import Path
import unittest
import numpy as np
from helixwm_final import protocol_v3 as p,protocol_v2 as old


class BankProtocolTests(unittest.TestCase):
    def test_checked_in_registration_exact(self):
        path=Path(__file__).resolve().parents[1]/'configs/helixwm_final_budget_matched_20261106.json'
        self.assertEqual(p.load_protocol(path),p.registered_protocol())

    def test_original_environment_masks_seeds_updates_and_budgets_unchanged(self):
        current=p.registered_protocol();before=old.registered_protocol()
        self.assertEqual(current['environment'],before['environment'])
        for name in ('actions_per_arm','dmc_calls_per_arm','policy_parameter_sync'):
            self.assertEqual(current['training'][name],before['training'][name])
        np.testing.assert_array_equal(p.make_mask_bank(current),old.make_mask_bank(before))
        self.assertEqual(p.training_seed(current,1999),101999)
        self.assertEqual(p.learner_updates_at(current,1000000),248750)

    def test_one_shared_bank_not_1991_new_environment_suites(self):
        cfg=p.registered_protocol();plan=p.build_calibration_plan(cfg)
        self.assertEqual(len(plan),20);self.assertEqual(sum(x['actions'] for x in plan),2480)
        self.assertEqual(cfg['planned_interaction_accounting']['per_pin_calibration_environment_actions'],0)
        self.assertTrue(cfg['calibration']['bank_available_identically_to_D_and_F'])
        self.assertFalse(cfg['calibration']['bank_actions_or_images_to_learner_allowed'])

    def test_unknown_fault_stage_and_final_frozen_thresholds_stay_separate(self):
        cfg=p.registered_protocol()
        self.assertFalse(cfg['unknown_observation_faults']['enabled_during_known_mask_training_or_stage_B'])
        self.assertFalse(cfg['calibration']['final_test_may_update_thresholds'])
        self.assertFalse(cfg['calibration']['heldout_driven_threshold_retry_allowed'])
        self.assertTrue(cfg['final_evaluation']['final_threshold_artifact_frozen_before_any_test'])

    def test_gate_failure_is_explicit_native_not_old_threshold_reuse(self):
        cfg=p.registered_protocol()
        self.assertEqual(p.phase_at(cfg,'F',5500),'FINAL_CURRENT_BANK_GATE_OR_EXPLICIT_NATIVE_FALLBACK')
        self.assertEqual(cfg['calibration']['missing_inactive_or_stale_gate'],'native_prior_fallback_fusion_planning_disabled')
        self.assertFalse(cfg['calibration']['fresh_process_replay_each_pin_required'])

    def test_unregistered_threshold_or_action_budget_changes_reject(self):
        cfg=p.registered_protocol();cfg['calibration']['required_upper_coverage_per_gap']=.9
        with self.assertRaises(ValueError):p.validate_protocol(cfg)
        cfg=p.registered_protocol();cfg['training']['actions_per_arm']=2000000
        with self.assertRaises(ValueError):p.validate_protocol(cfg)


if __name__=='__main__':unittest.main()
