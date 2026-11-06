"""Final-C1 CPU prototype tests; no real learned model/environment is run."""
from __future__ import annotations

import copy
import unittest
from unittest.mock import patch

import numpy as np

from helixwm_final.belief import (GAP_BINS, accumulate_risk, calibration_sha256,
    constrained_fusion, sample_categorical, uncertainty_metrics, validate_calibration,
    _validate_calibration_structure)
from helixwm_final.controller import FinalController, STREAMS, FallbackConstraintConflict

SOURCE, MODEL = 'a'*64, 'b'*64
IMAGE = np.zeros((64, 64, 3), np.uint8)


def calibration(version=3):
    thresholds = dict(max_energy=.05, max_uncertainty=1.0, max_disagreement=1.0, max_cumulative_risk=100.0,
        lambda_max=.25, kl_limit=1.0, min_ess_ratio=0.0, planning_horizon=3,
        action_amplitude=.25, slew_limit=.1, reward_weight=1.0,
        uncertainty_weight=.1, termination_weight=1.0, slew_weight=.1,
        temperature=.01, max_termination_risk=.95)
    result = dict(schema_version=1, frozen=True, active=True, validation={'passed': True},
        bank_manifest_sha256='d'*64, calibration_id='synthetic-calibration',
        source_sha256=SOURCE, model_sha256=MODEL, model_version=version,
        fit_seeds=[55000], validation_seeds=[55010],
        thresholds_by_gap={str(g): copy.deepcopy(thresholds) for g in GAP_BINS})
    result['sha256'] = calibration_sha256(result)
    return result


class FakeBackend:
    inference_version = 3
    action_dim = 6

    def __init__(self):
        self.transitions = []
        self.posteriors = []
        self.energies = .01
        self.continuation = .99
        self.actions = 0

    def inference_bundle_sha256(self): return MODEL

    def initial(self, batch=1):
        return {'deter': np.zeros((batch, 4), np.float32),
                'stoch': np.zeros((batch, 2, 3), np.float32)}

    def _probs(self, batch, posterior=False):
        p = np.array([.2, .7, .1] if posterior else [.8, .15, .05], np.float32)
        return np.broadcast_to(p, (batch, 2, 3)).copy()

    def observe(self, previous, action, image, is_first=False, rng=None):
        if not is_first: raise AssertionError('Final must use posterior_from_deter after a prior')
        state = self.initial(len(previous['deter']))
        state['stoch'] = sample_categorical(self._probs(len(state['deter']), True), rng)
        return state

    def advance_prior(self, previous, action, rng=None):
        batch = len(previous['deter'])
        self.transitions.append((previous['deter'].copy(), np.asarray(action).copy()))
        state = copy.deepcopy(previous); state['deter'][:, 0] += 1
        p = self._probs(batch)
        state['stoch'] = sample_categorical(p, rng)
        return state, {'probs': p, 'entropy': np.zeros(batch, np.float32), 'model_version': self.inference_version}

    def posterior_from_deter(self, deter, image, rng=None):
        self.posteriors.append(deter.copy())
        p = self._probs(len(deter), True)
        return {'deter': deter.copy(), 'stoch': sample_categorical(p, rng)}, {
            'probs': p, 'entropy': np.zeros(len(deter), np.float32), 'model_version': self.inference_version}

    def score(self, states, image):
        return np.full(len(states['deter']), -self.energies*12288, np.float64)

    def act(self, state, deterministic=False, rng=None):
        self.actions += 1
        return (np.full((len(state['deter']), 6), .8, np.float32) if deterministic else
                rng.uniform(.75, .85, (len(state['deter']), 6)).astype(np.float32))

    def predict_heads(self, state, with_image=False):
        batch = len(state['deter'])
        return {'reward': np.ones(batch, np.float32),
                'continuation': np.full(batch, self.continuation, np.float32)}


class MechanismTestController(FinalController):
    """CPU test double ONLY: hypothetical gates, never a deployable artifact."""
    def calibration_status(self):
        return _validate_calibration_structure(self._calibration, self.expected_source_sha256,
            self.model_version, self.expected_model_sha256)


def make_controller(cal=True, **kwargs):
    backend = FakeBackend()
    c = MechanismTestController(backend, seed=7, calibration=calibration() if cal else None,
        expected_source_sha256=SOURCE, expected_model_sha256=MODEL, **kwargs)
    c.reset(IMAGE, episode_id=9)
    return c, backend


def outcome(c, valid, image=IMAGE, external=True):
    action = c.act(external_action=np.zeros(6, np.float32)) if external else c.act()
    return c.step(image, action, {'availability': valid, 'observation_timestamp': c.timestamp+1}, 9)


def assert_tree_equal(test, a, b):
    test.assertIs(type(a), type(b))
    if isinstance(a, dict):
        test.assertEqual(set(a), set(b))
        for key in a: assert_tree_equal(test, a[key], b[key])
    elif isinstance(a, (list, tuple)):
        test.assertEqual(len(a), len(b))
        for x, y in zip(a, b): assert_tree_equal(test, x, y)
    elif isinstance(a, np.ndarray):
        test.assertEqual(a.dtype, b.dtype); np.testing.assert_array_equal(a, b)
    else: test.assertEqual(a, b)


class BeliefTests(unittest.TestCase):
    def test_uncertainty_units_and_epistemic_limitation(self):
        same = np.repeat(np.array([[[1., 0.]]]), 8, axis=0)
        metrics = uncertainty_metrics(same)
        self.assertEqual(metrics['uncertainty'], 0)
        self.assertEqual(metrics['model_conditional_disagreement'], 0)
        self.assertFalse(metrics['epistemic_available'])
        opposing = same.copy(); opposing[4:] = [[[0., 1.]]]
        mixed = uncertainty_metrics(opposing)
        self.assertAlmostEqual(mixed['predictive_entropy'], 1)
        self.assertAlmostEqual(mixed['expected_entropy'], 0)
        self.assertAlmostEqual(mixed['model_conditional_disagreement'], 1)
        self.assertFalse(mixed['epistemic_available'])

    def test_risk_is_monotonic_and_bounded_without_new_evidence(self):
        risk = 0
        for i in range(250):
            after = accumulate_risk(risk, .75)
            self.assertGreaterEqual(after, risk); self.assertLessEqual(after, 100)
            risk = after
        self.assertEqual(risk, 100)
        for bad in (float('nan'), float('inf'), -1, 1.01):
            with self.assertRaises(ValueError): accumulate_risk(0, bad)

    def test_fusion_is_anchored_and_respects_support_and_kl(self):
        real = np.array([[[.9, .1]]]); bank = np.array([[[.1, .9]], [[.2, .8]]])
        fused, evidence = constrained_fusion(real, bank, None, .5, .001)
        self.assertLessEqual(evidence['lambda'], .5)
        self.assertLessEqual(evidence['kl_to_real'], .001)
        self.assertGreater(fused[0, 0, 0], .8)
        strict, evidence = constrained_fusion(np.array([[[1., 0.]]]), bank, None, .5, 1)
        self.assertEqual(evidence['lambda'], 0)
        np.testing.assert_array_equal(strict, [[[1., 0.]]])

    def test_calibration_requires_independent_version_source_model_hash_and_seed_binding(self):
        good = calibration()
        self.assertEqual(_validate_calibration_structure(good, SOURCE, 3, MODEL), (True, 'calibrated'))
        self.assertEqual(validate_calibration(good, SOURCE, 3, MODEL),
                         (False, 'deployment_calibration_unverified'))
        self.assertFalse(validate_calibration(None, SOURCE, 3, MODEL)[0])
        self.assertFalse(validate_calibration(good, SOURCE, 4, MODEL)[0])
        self.assertFalse(validate_calibration(good, 'c'*64, 3, MODEL)[0])
        self.assertFalse(validate_calibration(good, SOURCE, 3, 'c'*64)[0])
        self.assertFalse(validate_calibration(good, SOURCE, 3, None)[0])
        for kind in ('hash', 'overlap', 'test_seed', 'negative_threshold', 'missing_threshold',
                     'inactive', 'coverage_failed', 'missing_bank_sha', 'malformed_bank_sha'):
            bad = copy.deepcopy(good)
            if kind == 'hash': bad['sha256'] = '0'*64
            elif kind == 'overlap': bad['validation_seeds'] = bad['fit_seeds']
            elif kind == 'test_seed': bad['fit_seeds'] = [60000]
            elif kind == 'negative_threshold': bad['thresholds_by_gap']['1']['max_energy'] = -1
            elif kind == 'inactive': bad['active'] = False
            elif kind == 'coverage_failed': bad['validation']['passed'] = False
            elif kind == 'missing_bank_sha': del bad['bank_manifest_sha256']
            elif kind == 'malformed_bank_sha': bad['bank_manifest_sha256'] = 'not-a-sha'
            else: del bad['thresholds_by_gap']['1']['slew_limit']
            if kind != 'hash': bad['sha256'] = calibration_sha256(bad)
            with self.subTest(kind=kind): self.assertFalse(validate_calibration(bad, SOURCE, 3, MODEL)[0])


class ControllerTests(unittest.TestCase):
    def test_production_controller_cannot_activate_a_handmade_test_dictionary(self):
        c = FinalController(FakeBackend(), calibration=calibration(),
            expected_source_sha256=SOURCE, expected_model_sha256=MODEL)
        c.reset(IMAGE, episode_id=9)
        outcome(c, False)
        self.assertEqual(c.mode, 'UNCALIBRATED')
        self.assertFalse(c.calibration_status()[0])
        self.assertEqual(c.last_metrics['fusion_lambda'], 0)
        c.act()
        self.assertFalse(c.last_action_decision['planning'])

    def test_reconnection_rejects_fused_sample_that_worsens_real_observation_fit(self):
        c, backend = make_controller()
        outcome(c, False)
        original_post = backend.posterior_from_deter
        def posterior(deter, image, rng=None):
            state, info = original_post(deter, image, rng)
            state['stoch'][:] = 0
            state['stoch'][:, :, 1] = 1
            return state, info
        backend.posterior_from_deter = posterior
        def score(states, image):
            bad = np.all(states['stoch'][:, :, 2] == 1, axis=1)
            return -np.where(bad, .4, .01) * 12288
        backend.score = score
        def bad_fusion(probs, rng):
            stoch = np.zeros(np.asarray(probs).shape, np.float32)
            stoch[:, :, 2] = 1
            return stoch
        with patch('helixwm_final.controller.sample_categorical', bad_fusion):
            outcome(c, True)
        check = c.last_metrics['fusion_observation_check']
        self.assertTrue(check['checked'])
        self.assertFalse(check['accepted'])
        self.assertGreater(check['fused_energy'], check['posterior_energy'])
        self.assertEqual(c.last_metrics['fusion_lambda'], 0)
        self.assertTrue(np.all(c.real['stoch'][:, :, 1] == 1))
        self.assertEqual(c.real['deter'][0, 0], 2)

    def test_candidate_disagreement_rejects_even_when_entropy_gate_passes(self):
        c, backend = make_controller()
        for row in c._calibration['thresholds_by_gap'].values():
            row['max_disagreement'] = .05
        c._calibration['sha256'] = calibration_sha256(c._calibration)
        original = backend._probs
        def opposing(batch, posterior=False):
            if batch != 8 or posterior:
                return original(batch, posterior)
            probs = np.zeros((8, 2, 3), np.float32)
            probs[:4, :, 0] = 1
            probs[4:, :, 1] = 1
            return probs
        backend._probs = opposing
        outcome(c, False)
        self.assertLess(c.last_metrics['uncertainty'], 1.0)
        self.assertGreater(c.last_metrics['model_conditional_disagreement'], .05)
        self.assertEqual(c.mode, 'PRIOR_ONLY_FALLBACK')
        self.assertEqual(c.last_metrics['fusion_lambda'], 0)
        c.act()
        self.assertEqual(c.last_action_decision['kind'], 'prior_only_fallback')
        self.assertFalse(c.last_action_decision['planning'])

    def test_incompatible_fallback_constraints_issue_no_action(self):
        c, backend = make_controller()
        c.last_executed_action = np.full(6, .8, np.float32)
        c.missing_age = 1
        c.absolute_quality_ok = False
        timestamp = c.timestamp
        with self.assertRaises(FallbackConstraintConflict):
            c.act(deterministic=True)
        self.assertIsNone(c.issued_action)
        self.assertEqual(c.timestamp, timestamp)
        self.assertEqual(len(backend.transitions), 0)
        self.assertEqual(c.events[-1]['event'], 'fallback_constraint_conflict')
        self.assertFalse(c.events[-1]['action_issued'])

    def test_fallback_projects_to_joint_feasible_interval(self):
        c, _ = make_controller()
        thresholds = calibration()['thresholds_by_gap']['1']
        c.last_executed_action = np.array([.2, -.2, 0, .3, -.3, .1], np.float32)
        action = c._fallback(np.array([.8, -.8, .8, -.8, .8, -.8], np.float32), thresholds)
        self.assertEqual(action.dtype, np.float32)
        self.assertTrue(np.all(np.abs(action) <= .25 + 1e-7))
        self.assertTrue(np.all(np.abs(action-c.last_executed_action) <= .1 + 1e-7))
        np.testing.assert_allclose(action, [.25, -.25, .1, .2, -.2, 0], rtol=0, atol=2e-8)

    def test_real_and_bank_advance_actual_action_once_and_reconnect_without_extra_transition(self):
        c, b = make_controller()
        for _ in range(5): outcome(c, False)
        self.assertEqual(c.real['deter'][0, 0], 5)
        self.assertEqual(len(b.transitions), 10)
        self.assertEqual(len(b.posteriors), 0)
        self.assertIsNone(c.diagnostics()['prediction_error'])
        outcome(c, True)
        self.assertEqual(c.real['deter'][0, 0], 6)
        self.assertEqual(len(b.transitions), 12)
        self.assertEqual(len(b.posteriors), 2)
        self.assertEqual(c.evidence_kind, 'RECONNECTED_POSTERIOR')
        self.assertLessEqual(c.last_metrics['fusion_lambda'], .25)
        self.assertEqual(c.cumulative_risk, 0)
        np.testing.assert_array_equal(b.posteriors[0], c.real['deter'])

    def test_all_gaps_up_to_100_have_bounded_bank_and_over_budget_abstains(self):
        for gap in (1, 5, 10, 20, 50, 100):
            c, b = make_controller()
            for _ in range(gap): outcome(c, False)
            with self.subTest(gap=gap):
                self.assertEqual(c.bank['deter'].shape[0], 8)
                self.assertEqual(c.real['deter'][0, 0], gap)
                self.assertLessEqual(len(c.action_history), 100)
                self.assertFalse(c.last_metrics['epistemic_available'])
            if gap == 100:
                outcome(c, False)
                self.assertIsNone(c.bank)
                self.assertEqual(c.mode, 'PRIOR_ONLY_FALLBACK')
                self.assertEqual(c.last_metrics['fusion_lambda'], 0)
                self.assertEqual(len(c.action_history), 100)
                next_action = c.act()
                self.assertLessEqual(float(np.max(np.abs(next_action))), .100001)

    def test_missing_pixels_are_not_read_and_prediction_error_is_unknown(self):
        c, b = make_controller()
        class ForbiddenPixels:
            def __array__(self, *args, **kwargs): raise AssertionError('Missing image was accessed')
        outcome(c, False, image=ForbiddenPixels())
        self.assertIsNone(c.last_metrics['prediction_error'])
        self.assertEqual(c.evidence_kind, 'PREDICTED_PRIOR')

    def test_absolute_quality_rejects_every_candidate_even_uniform_ess(self):
        c, b = make_controller()
        outcome(c, False)
        b.energies = 10
        self.assertEqual(1/np.square(c.weights).sum(), 8)
        outcome(c, True)
        self.assertFalse(c.absolute_quality_ok)
        self.assertEqual(c.last_metrics['fusion_lambda'], 0)
        self.assertEqual(c.evidence_kind, 'RECONNECTED_POSTERIOR')
        self.assertEqual(c.real['deter'][0, 0], 2)

    def test_no_or_stale_calibration_disables_fusion_and_new_planning(self):
        for kind in ('missing', 'version', 'model', 'hash'):
            c, b = make_controller(cal=False)
            if kind != 'missing':
                artifact = calibration()
                if kind == 'version': artifact['model_version'] = 2
                elif kind == 'model': artifact['model_sha256'] = 'c'*64
                else: artifact['sha256'] = '0'*64
                if kind != 'hash': artifact['sha256'] = calibration_sha256(artifact)
                c = MechanismTestController(b, calibration=artifact, expected_source_sha256=SOURCE,
                                    expected_model_sha256=MODEL)
                c.reset(IMAGE, 9)
            outcome(c, False)
            with self.subTest(kind=kind):
                self.assertEqual(c.last_metrics['fusion_lambda'], 0)
                self.assertEqual(c.mode, 'UNCALIBRATED')
                before = len(b.transitions)
                c.act()
                self.assertEqual(len(b.transitions), before)
                self.assertFalse(c.last_action_decision['planning'])

    def test_prefill_is_recorded_and_exactly_one_outcome_is_required(self):
        c, b = make_controller()
        action = c.act(external_action=np.full(6, .3, np.float32))
        self.assertEqual(b.actions, 0)
        with self.assertRaises(RuntimeError): c.act()
        with self.assertRaises(ValueError): c.step(IMAGE, action+.1, {'availability': True, 'observation_timestamp': 1}, 9)
        with self.assertRaises(ValueError): c.step(IMAGE, action, {'availability': True, 'observation_timestamp': 2}, 9)
        c.step(IMAGE, action, {'availability': True, 'observation_timestamp': 1}, 9)
        self.assertEqual(c.action_history[-1]['provenance']['kind'], 'external_prefill')
        np.testing.assert_array_equal(c.action_history[-1]['action'], action)
        with self.assertRaises(RuntimeError): c.step(IMAGE, action, {'availability': True, 'observation_timestamp': 2}, 9)

    def test_planning_is_scratch_only_and_consumes_only_planning_stream(self):
        c, b = make_controller()
        outcome(c, False)
        before = c.snapshot()
        thresholds, _ = c._thresholds()
        _, report = c._plan_action(np.full(6, .8, np.float32), thresholds)
        after = c.snapshot()
        self.assertTrue(report['used'])
        self.assertLessEqual(len(report['proposals']), 2)
        for proposal in report['proposals']:
            self.assertEqual(proposal['horizon'], 3)
            self.assertEqual(len(proposal['action_sequence']), 3)
            self.assertFalse(proposal['prospective_actions_executed'])
        assert_tree_equal(self, before['state'], after['state'])
        assert_tree_equal(self, before['action_history'], after['action_history'])
        for name in STREAMS:
            if name != 'planning': assert_tree_equal(self, before['rngs'][name], after['rngs'][name])
        self.assertNotEqual(before['rngs']['planning'], after['rngs']['planning'])

    def test_all_future_plans_can_be_rejected_without_optimistic_selection(self):
        c, b = make_controller(); outcome(c, False)
        b.continuation = .01
        action, report = c._plan_action(np.full(6, .8, np.float32), c._thresholds()[0])
        self.assertEqual(report['reason'], 'all_plans_rejected')
        self.assertIsNone(report['selected'])
        self.assertLessEqual(float(np.max(np.abs(action))), .100001)
        self.assertFalse(report['physical_safety_proven'])

    def test_degenerate_ess_disables_missing_fusion_and_planning(self):
        c, b = make_controller()
        artifact = calibration()
        for row in artifact['thresholds_by_gap'].values(): row['min_ess_ratio'] = .5
        artifact['sha256'] = calibration_sha256(artifact)
        c = MechanismTestController(b, calibration=artifact, expected_source_sha256=SOURCE,
                            expected_model_sha256=MODEL)
        c.reset(IMAGE, 9)
        c.weights = np.array([1., 0, 0, 0, 0, 0, 0, 0])
        outcome(c, False)
        self.assertEqual(c.last_metrics['fusion_lambda'], 0)
        self.assertEqual(c.mode, 'PRIOR_ONLY_FALLBACK')
        before = len(b.transitions)
        c.act()
        self.assertEqual(len(b.transitions), before)
        self.assertFalse(c.last_action_decision['planning'])

    def test_snapshot_recovers_pending_action_rng_bank_and_next_outcome(self):
        c, b = make_controller(); outcome(c, False)
        issued = c.act()
        record = c.snapshot()
        clone, other = make_controller()
        clone.load(record)
        assert_tree_equal(self, clone.snapshot(), record)
        info = {'availability': True, 'observation_timestamp': 2}
        c.step(IMAGE, issued, info, 9); clone.step(IMAGE, issued, info, 9)
        assert_tree_equal(self, c.snapshot(), clone.snapshot())
        np.testing.assert_array_equal(c.act(), clone.act())

    def test_invalid_restore_is_rejected_before_replacing_live_state(self):
        c, _ = make_controller(); before = c.snapshot()
        for kind in ('rng', 'version', 'source', 'model', 'calibration', 'missing_field', 'risk'):
            bad = copy.deepcopy(before)
            if kind == 'rng': bad['rngs']['dream'] = {'invalid': True}
            elif kind == 'version': bad['state']['model_version'] += 1
            elif kind == 'source': bad['expected_source_sha256'] = 'c'*64
            elif kind == 'model': bad['expected_model_sha256'] = 'c'*64
            elif kind == 'calibration': bad['calibration']['thresholds_by_gap']['1']['max_energy'] *= 2
            elif kind == 'missing_field': del bad['state']['real']
            else: bad['state']['cumulative_risk'] = float('nan')
            with self.subTest(kind=kind), self.assertRaises((ValueError, TypeError, KeyError)):
                c.load(bad)
            assert_tree_equal(self, c.snapshot(), before)

    def test_versions_and_image_contract_are_fail_closed(self):
        c, b = make_controller(); b.inference_version += 1
        with self.assertRaises(ValueError): c.act()
        c, b = make_controller(); a = c.act(external_action=np.zeros(6, np.float32))
        with self.assertRaises(ValueError): c.step(np.zeros((64, 64, 3), np.float32), a,
            {'availability': True, 'observation_timestamp': 1}, 9)

    def test_completed_boundary_restore_preserves_rngs_but_discards_old_version_latents(self):
        c, b = make_controller()
        for _ in range(500): outcome(c, True)
        saved = c.snapshot()
        later_backend = FakeBackend(); later_backend.inference_version = 4
        later = MechanismTestController(later_backend, calibration=calibration(version=4),
                                expected_source_sha256=SOURCE, expected_model_sha256=MODEL)
        transition_count = len(later_backend.transitions)
        audit = later.restore_completed_episode(saved)
        self.assertFalse(audit['old_latent_state_reused'])
        self.assertFalse(audit['old_calibration_reused'])
        self.assertEqual(later.model_version, 4)
        self.assertFalse(later.initialized)
        self.assertIsNone(later.real); self.assertIsNone(later.bank)
        self.assertEqual(len(later_backend.transitions), transition_count)
        assert_tree_equal(self, saved['rngs'], later.snapshot()['rngs'])
        with self.assertRaises(RuntimeError): later.act()
        later.reset(IMAGE, 10)
        self.assertEqual(later.timestamp, 0)
        self.assertEqual(later.calibration_status(), (True, 'calibrated'))
        for key, value in (('timestamp', 499), ('missing_age', 1), ('issued_action', np.zeros(6, np.float32))):
            bad = copy.deepcopy(saved); bad['state'][key] = value
            with self.subTest(field=key), self.assertRaises(ValueError): later.restore_completed_episode(bad)


if __name__ == '__main__': unittest.main()
