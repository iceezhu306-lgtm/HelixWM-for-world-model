"""Shared environment/mask contract without a GPU or installed simulator.

These tests are not the real two-process environment gate. That gate is run by
helixwm_matched.env_probe on the intended simulator/runtime.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from helixwm_matched.environment import MatchedWalkerEnv
from helixwm_matched.env_probe import first_difference, check_fault_visibility
from helixwm_matched.protocol import (load_protocol, validate_protocol, protocol_sha256,
    make_mask_bank, save_mask_bank, gap_segments, fixed_gap_mask, training_seed,
    mask_sha256, array_sha256, validate_mask)


CONFIG = Path(__file__).resolve().parents[1]/"configs/dreamerv3_helixwm_matched_20261026.json"


class Step:
    def __init__(self, first=False, last=False, reward=0.5, discount=1.0):
        self._first, self._last = first, last
        self.reward, self.discount = reward, discount
    def first(self): return self._first
    def last(self): return self._last


class FakeNative:
    def __init__(self, seed, early=None, invalid_discount=None):
        self.seed, self.early, self.invalid_discount = seed, early, invalid_discount
        self.calls = self.renders = 0
        self.closed = False
        self.physics = self
        self.actions = []
    def action_spec(self):
        return type("Spec", (), {"shape": (6,), "minimum": -np.ones(6), "maximum": np.ones(6)})()
    def reset(self):
        self.calls = 0
        return Step(first=True)
    def step(self, action):
        self.calls += 1
        self.actions.append(np.asarray(action).copy())
        return Step(last=self.calls == (self.early or 1000),
                    reward=0.25 if self.calls % 2 else 0.5,
                    discount=np.nan if self.calls == self.invalid_discount else 1.0)
    def render(self, *, height, width, camera_id):
        if camera_id != 0: raise ValueError("Wrong camera")
        self.renders += 1
        return np.full((height, width, 3), (self.seed+self.calls) % 251+1, np.uint8)
    def diagnostic_state(self):
        return np.array([self.seed, self.calls, sum(float(a.sum()) for a in self.actions)], np.float64)
    def close(self): self.closed = True


class SharedContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_protocol(CONFIG)
        cls.bank = make_mask_bank(cls.config)

    def env(self, **kwargs):
        return MatchedWalkerEnv(self.config, env_factory=kwargs.pop("env_factory", FakeNative), **kwargs)

    def test_mask_bank_deterministic_and_global_rng_untouched(self):
        np.random.seed(123)
        before = np.random.get_state()
        other = make_mask_bank(self.config)
        after = np.random.get_state()
        np.testing.assert_array_equal(self.bank, other)
        self.assertEqual(before[0], after[0])
        np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])
        self.assertEqual(self.bank.shape, (2000, 501))
        self.assertEqual(self.bank.dtype, np.bool_)

    def test_actual_gaps_bounded_and_reset_terminal_valid(self):
        self.assertTrue(self.bank[:, 0].all() and self.bank[:, -1].all())
        lengths = set()
        for row in self.bank:
            for start, length in gap_segments(row):
                lengths.add(length)
                self.assertTrue(1 <= length <= 5)
                self.assertTrue(row[start-1] and row[start+length])
        self.assertEqual(lengths, {1,2,3,4,5})

    def test_forced_separator_even_when_rng_always_starts(self):
        rng = mock.Mock()
        rng.random.return_value = 0.0
        rng.integers.return_value = 5
        with mock.patch("helixwm_matched.protocol.np.random.default_rng", return_value=rng):
            bank = make_mask_bank(self.config)
        self.assertFalse(bank[0, 1:6].any())
        self.assertTrue(bank[0, 6])
        self.assertFalse(bank[0, 7:12].any())
        self.assertTrue(all(n == 5 for row in bank for _, n in gap_segments(row)))

    def test_protocol_mutations_fail_closed(self):
        for section, field, value in [
            ("training", "replay_context", 0),
            ("training", "actions_per_arm", 999999),
            ("training", "learner_hidden_clean_targets_allowed", True),
            ("environment", "camera_id", 1),
            ("environment", "action_repeat", True),
        ]:
            changed = copy.deepcopy(self.config); changed[section][field] = value
            with self.assertRaises(ValueError): validate_protocol(changed)
        changed = copy.deepcopy(self.config); changed["anything"] = float("nan")
        with self.assertRaises(ValueError): validate_protocol(changed)

    def test_reset_seed_mapping_no_boolean_or_out_of_range(self):
        self.assertEqual(training_seed(self.config, 0), 100000)
        self.assertEqual(training_seed(self.config, 1999), 101999)
        for bad in (-1, 2000, True, 1.0):
            with self.assertRaises(ValueError): training_seed(self.config, bad)

    def test_mask_bank_persistence_identity_and_tamper_rejection(self):
        with tempfile.TemporaryDirectory() as temp:
            report = save_mask_bank(self.config, temp)
            self.assertEqual(report, save_mask_bank(self.config, temp))
            self.assertEqual(len(report["row_sha256"]), 2000)
            self.assertEqual(report["content_sha256"], array_sha256(self.bank))
            altered = self.bank.copy(); altered[0, 2] = ~altered[0, 2]
            np.save(Path(temp)/"mask_bank.npy", altered)
            with self.assertRaises(ValueError): save_mask_bank(self.config, temp)

    def test_fixed_eval_gaps_exact_with_valid_boundaries(self):
        for length in (1,5,10,20,50):
            self.assertEqual(gap_segments(fixed_gap_mask(length)), [(150, length)])
        self.assertEqual(gap_segments(fixed_gap_mask(0)), [])
        for mask in (np.ones(501, np.uint8), np.ones(500, bool)):
            with self.assertRaises(ValueError): validate_mask(mask)
        bad = np.ones(501, bool); bad[-1] = False
        with self.assertRaises(ValueError): validate_mask(bad)
        with self.assertRaises(ValueError): fixed_gap_mask(501)
        with self.assertRaises(ValueError): fixed_gap_mask(True)

    def test_no_hidden_frame_rendered_or_returned(self):
        instances = []
        def factory(seed):
            instance = FakeNative(seed); instances.append(instance); return instance
        mask = fixed_gap_mask(5, start=1)
        with self.env(env_factory=factory) as env:
            first = env.reset(72000, mask)
            self.assertTrue(first.any())
            image, reward, done, info = env.step(np.zeros(6))
            self.assertFalse(image.any())
            self.assertEqual(instances[0].renders, 1)  # Only valid reset rendered.
            self.assertNotIn("probe", info)
            self.assertFalse(any("physics" in k or "raw" in k or "clean" in k for k in info))
            self.assertFalse(env.render().any())
            self.assertFalse(info["availability"])
            self.assertTrue(info["input_available"])
            self.assertEqual((info["input_observation_timestamp"], info["observation_timestamp"]), (0,1))
            with self.assertRaises(PermissionError): env.render_diagnostic()

    def test_mask_and_images_are_defensive_copies(self):
        mask = fixed_gap_mask(1, start=1)
        with self.env() as env:
            image = env.reset(72000, mask)
            image[:] = 0; mask[:] = True
            self.assertTrue(env.render().any())
            image, _, _, info = env.step(np.zeros(6))
            self.assertFalse(image.any())
            info["executed_action"][:] = 1
            self.assertFalse(env.last_info["executed_action"].any())

    def test_clip_repeat_reward_and_discount_contract(self):
        with self.env() as env:
            env.reset(72000)
            _, reward, done, info = env.step(np.array([-2, -1, 0, .5, 1, 2], np.float64))
            np.testing.assert_array_equal(info["executed_action"], [-1, -1, 0, .5, 1, 1])
            self.assertEqual(info["executed_action"].dtype, np.float32)
            self.assertEqual(reward, .75)
            self.assertEqual(info["env_steps_this_step"], 2)
            self.assertEqual(info["discount"], 1.0)
            self.assertFalse(done)

    def test_exact_horizon_and_new_seed_reconstruction(self):
        instances = []
        def factory(seed):
            instance = FakeNative(seed); instances.append(instance); return instance
        with self.env(env_factory=factory) as env:
            env.reset(100000)
            for i in range(500):
                image, _, done, info = env.step(np.zeros(6))
                self.assertEqual(done, i == 499)
            self.assertEqual(env.action_decisions, 500)
            self.assertEqual(env.episode_env_steps, 1000)
            self.assertTrue(info["truncated"])
            self.assertFalse(info["terminated"])
            with self.assertRaises(RuntimeError): env.step(np.zeros(6))
            env.reset(100001)
            self.assertTrue(instances[0].closed)
            self.assertEqual(instances[-1].seed, 100001)
            self.assertEqual(env.action_decisions, 0)

    def test_mid_episode_reset_is_rejected(self):
        with self.env() as env:
            env.reset(100000)
            with self.assertRaises(RuntimeError): env.reset(100001)
            self.assertEqual(env.seed, 100000)

    def test_invalid_actions_make_no_native_calls(self):
        with self.env() as env:
            env.reset(1)
            for action in (np.full(6, np.nan), np.zeros(7), np.ones(6, complex), np.ones(6, bool)):
                with self.assertRaises((ValueError, FloatingPointError)): env.step(action)
                self.assertEqual(env.episode_env_steps, 0)

    def test_partial_native_failure_retains_actual_counts_and_no_autoreset(self):
        with self.env(env_factory=lambda seed: FakeNative(seed, early=3)) as env:
            env.reset(1)
            env.step(np.zeros(6))
            with self.assertRaises(RuntimeError): env.step(np.zeros(6))
            self.assertEqual(env.episode_env_steps, 3)
            self.assertEqual(env.action_decisions, 2)
            with self.assertRaises(RuntimeError): env.step(np.zeros(6))
            with self.assertRaises(RuntimeError): env.reset(2)

    def test_nonfinite_native_discount_fails_and_preserves_call_count(self):
        with self.env(env_factory=lambda seed: FakeNative(seed, invalid_discount=1)) as env:
            env.reset(1)
            with self.assertRaises(FloatingPointError): env.step(np.zeros(6))
            self.assertEqual(env.episode_env_steps, 1)

    def test_diagnostic_state_is_explicit_and_subcalls_exposed_only_there(self):
        with self.env(diagnostics=True) as env:
            env.reset(1, fixed_gap_mask(1, start=1))
            self.assertIn("probe", env.last_info)
            image, _, _, info = env.step(np.zeros(6))
            self.assertFalse(image.any())
            self.assertEqual(len(info["probe"]["subcalls"]), 2)
            self.assertTrue(env.render_diagnostic().any())

    def test_seed_validation_and_closed_instance(self):
        for seed in (True, -1, 2**32, .2):
            env = self.env()
            with self.assertRaises(ValueError): env.reset(seed)
            env.close()
        env = self.env(); env.close(); env.close()
        with self.assertRaises(RuntimeError): env.reset(1)

    def test_strict_evidence_comparison_does_not_relax_float_tolerance(self):
        self.assertIsNone(first_difference({"a": [1, 0.5]}, {"a": [1, 0.5]}))
        self.assertIsNotNone(first_difference({"a": 1.0}, {"a": 1.0+1e-14}))
        self.assertIsNotNone(first_difference(1, 1.0))
        self.assertIsNotNone(first_difference(float("nan"), float("nan")))

    def test_probe_fault_visibility_rejects_leak_or_physics_difference(self):
        clean = {"seed": 1, "fixed_requested_actions_sha256": "same", "physics_state_sha256": ["p0", "p1"],
                 "rows": [{"input_available": True, "reward": 1.0}],
                 "visible_RGB_sha256": ["initial", "actual"], "availability": [True, True]}
        fault = copy.deepcopy(clean)
        fault["availability"][1] = False
        fault["visible_RGB_sha256"][1] = array_sha256(np.zeros((64,64,3), np.uint8))
        check_fault_visibility(clean, fault)
        fault["visible_RGB_sha256"][1] = "actual"
        with self.assertRaises(ValueError): check_fault_visibility(clean, fault)
        fault["physics_state_sha256"][1] = "different"
        with self.assertRaises(ValueError): check_fault_visibility(clean, fault)


if __name__ == "__main__":
    unittest.main()
