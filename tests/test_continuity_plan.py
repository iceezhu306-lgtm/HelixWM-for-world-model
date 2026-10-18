"""Design-only regression guards; no model, simulator, SSH, or training imports."""
import copy
import importlib.util
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("continuity_planner", ROOT / "scripts/plan_continuity_study.py")
planner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(planner)


class ContinuityPlanTests(unittest.TestCase):
    def setUp(self):
        self.design = json.loads((ROOT / "configs/continuity_study_v3.json").read_text(encoding="utf-8"))

    def test_registered_cost_and_unique_cases(self):
        plan = planner.build_plan(self.design)
        self.assertEqual(plan["summary"]["total_episodes"], 1560)
        self.assertEqual(plan["summary"]["total_action_decisions"], 780000)
        self.assertEqual(plan["summary"]["total_dmc_calls"], 1560000)
        ids = [case["case_id"] for case in plan["cases"]]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(plan, planner.build_plan(copy.deepcopy(self.design)))

    def test_exact_contiguous_masks_without_joined_bursts(self):
        plan = planner.build_plan(self.design)
        for length in [0, 1, 5, 10, 20, 50]:
            mask = plan["schedules"][str(length)]["available"]
            self.assertEqual(len(mask), 501)
            self.assertTrue(mask[0] and mask[-1])
            self.assertEqual([i for i, available in enumerate(mask) if not available],
                             list(range(150, 150 + length)))
        self.assertNotEqual(plan["schedules"]["5"]["sha256"], plan["schedules"]["10"]["sha256"])

    def test_historical_and_held_out_leakage_rejected(self):
        for split, start in [("design_environment_seeds", 10000),
                             ("held_out_environment_seeds", 20000)]:
            changed = copy.deepcopy(self.design)
            changed["phase_a"]["splits"][split]["start"] = start
            with self.assertRaises(ValueError):
                planner.build_plan(changed)

    def test_hidden_budget_and_reference_changes_rejected(self):
        for section, key, value in [(None, "training_update_budget", 1),
                                    ("environment", "privileged_current_state_to_policy", True),
                                    ("environment", "action_repeat", 1)]:
            changed = copy.deepcopy(self.design)
            (changed if section is None else changed[section])[key] = value
            with self.assertRaises(ValueError):
                planner.build_plan(changed)
        self.design["phase_a"]["candidate_reference"]["max_missing"] = 50
        with self.assertRaises(ValueError):
            planner.build_plan(self.design)

    def test_double_evidence_and_missing_identity_rejected(self):
        changed = copy.deepcopy(self.design)
        changed["identity"]["checkpoint_sha256"] = None
        with self.assertRaises(ValueError):
            planner.build_plan(changed)
        for arm in self.design["phase_a"]["arms"]:
            if arm["id"] == "posterior_return_once":
                arm["score_return_frame_before_posterior"] = True
        with self.assertRaises(ValueError):
            planner.build_plan(self.design)

    def test_wrong_cost_and_out_of_episode_window_rejected(self):
        changed = copy.deepcopy(self.design)
        changed["phase_a"]["evaluation_cost_upper_bound"]["total_dmc_calls"] = 780000
        with self.assertRaises(ValueError):
            planner.build_plan(changed)
        self.design["phase_a"]["schedule"]["missing_observation_start"] = 480
        with self.assertRaises(ValueError):
            planner.build_plan(self.design)


if __name__ == "__main__":
    unittest.main()
