"""Synthetic CPU fixtures only: never claim a production calibration PASS."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from helixwm_final import calibration as c
from helixwm_final.belief import accumulate_risk, calibration_sha256, uncertainty_metrics


def fixture():
    binding = {'source_sha256': '1'*64, 'bank_manifest_sha256': '2'*64,
               'model_sha256': '3'*64, 'model_version': 1, 'protocol_sha256': '4'*64,
               'measurement_kind': 'synthetic_test_fixture', 'measurement_protocol': c.MEASUREMENT_PROTOCOL,
               'diagnostics_manifest_sha256': '5'*64, 'frozen_gpu_zero_update_verified': False,
               'frozen_gpu_report_sha256': None}
    rows = []
    probs = np.asarray([[[.65 + .01*k, .25 - .005*k, .10 - .005*k]]*2 for k in range(8)], np.float64)
    weights = np.full(8, 1/8, np.float64)
    metrics = uncertainty_metrics(probs, weights)
    for seed in c.FIT_SEEDS + c.VALIDATION_SEEDS:
        variant = (seed - 55000) % 10
        for gap in c.GAP_BINS:
            count = gap + 8
            risk, risks = 0.0, []
            for t in range(count):
                if t < gap:
                    risk = accumulate_risk(risk, metrics['uncertainty'], 100)
                risks.append(risk)
            rows.append({'seed': seed, 'gap': gap,
                'binding': {key: binding[key] for key in c.IDENTITY_FIELDS},
                'epistemic_available': False, 'model_updates': 0,
                'timestamp': np.arange(17, 17 + count, dtype=np.int32),
                'energy': np.broadcast_to(.1 + .001*variant + .01*np.arange(8), (count, 8)).copy(),
                'candidate_probs': np.broadcast_to(probs, (count, *probs.shape)).copy(),
                'candidate_weights': np.broadcast_to(weights, (count, 8)).copy(),
                'uncertainty': np.full(count, metrics['uncertainty'], np.float64),
                'disagreement': np.full(count, metrics['model_conditional_disagreement'], np.float64),
                'cumulative_risk': np.asarray(risks, np.float64),
                'ess_ratio': np.ones(count, np.float64),
                'reward_pred': np.full(count, 1.0 + .01*variant, np.float64),
                'termination_risk': np.full(count, .03 + .001*variant, np.float64),
                'action_amplitude': np.full(count, .7 + .01*variant, np.float64),
                'action_slew': np.full(count, .2 + .005*variant, np.float64),
                'slew_squared_mean': np.full(count, .01 + .001*variant, np.float64)})
    return rows, binding


class CalibrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows, cls.binding = fixture()
        cls.audit = c.fit_from_diagnostics(cls.rows, cls.binding)

    def test_complete_fit_is_frozen_but_synthetic_fixture_never_active(self):
        self.assertTrue(self.audit['frozen'])
        self.assertTrue(self.audit['thresholds_complete'])
        self.assertTrue(self.audit['validation']['passed'])
        self.assertFalse(self.audit['active'])
        self.assertIn('synthetic_test_fixture_not_production', self.audit['inactive_reasons'])
        self.assertIn('independent_frozen_gpu_measurement_not_verified', self.audit['inactive_reasons'])
        self.assertFalse(self.audit['provenance']['measurement_executed_by_fitter'])
        self.assertFalse(self.audit['provenance']['external_manifest_bytes_verified_by_fitter'])
        self.assertEqual(self.audit['sha256'], calibration_sha256(self.audit))

    def test_heldout_cannot_change_thresholds_and_failure_disables_activation(self):
        rows = copy.deepcopy(self.rows)
        altered = next(r for r in rows if r['seed'] == 55010 and r['gap'] == 100)
        altered['energy'][:] = 1.0
        result = c.fit_from_diagnostics(rows, self.binding)
        self.assertEqual(result['thresholds_by_gap'], self.audit['thresholds_by_gap'])
        self.assertFalse(result['validation']['passed'])
        self.assertFalse(result['active'])
        report = result['validation']['per_gap']['100']
        self.assertEqual(report['per_seed_coverage']['55010']['energy_upper'], 0.0)
        self.assertAlmostEqual(report['mean_seed_coverage']['energy_upper'], .9)
        self.assertIn('energy_upper', report['failed_metrics'])
        self.assertIn('independent_validation_failed', result['inactive_reasons'])

    def test_fit_quantiles_structural_caps_and_horizon_have_explicit_units(self):
        for gap, row in self.audit['thresholds_by_gap'].items():
            self.assertAlmostEqual(row['max_energy'], .109)
            self.assertAlmostEqual(row['max_termination_risk'], .039)
            self.assertEqual(row['max_disagreement'], float(self.rows[0]['disagreement'][0]))
            self.assertAlmostEqual(row['action_amplitude'], .79)
            self.assertLessEqual(row['lambda_max'], .25)
            self.assertEqual(row['kl_limit'], .02)
            self.assertIn(row['planning_horizon'], (1, 3, 5))
            self.assertLessEqual(row['planning_horizon'], int(gap))
        self.assertEqual(self.audit['fitting']['method'], 'higher')
        self.assertEqual(self.audit['fitting']['upper_quantile'], .99)
        self.assertEqual(self.audit['fitting']['lower_quantile'], .05)
        self.assertEqual(self.audit['validation']['upper_min_empirical_coverage'], .95)
        self.assertEqual(self.audit['validation']['lower_min_empirical_coverage'], .90)
        self.assertFalse(self.audit['fitting']['structural_caps_proven_safe_by_calibration'])

    def test_disagreement_holdout_gate_is_independent_of_entropy_gate(self):
        rows = copy.deepcopy(self.rows)
        altered = next(r for r in rows if r['seed'] == 55010 and r['gap'] == 20)
        sharp = np.zeros_like(altered['candidate_probs'])
        sharp[:, :4, :, 0] = 1.0
        sharp[:, 4:, :, 1] = 1.0
        altered['candidate_probs'] = sharp
        metrics = uncertainty_metrics(sharp[0], altered['candidate_weights'][0])
        altered['uncertainty'][:] = metrics['uncertainty']
        altered['disagreement'][:] = metrics['model_conditional_disagreement']
        cumulative = 0.0
        for t in range(len(altered['timestamp'])):
            if t < altered['gap']:
                cumulative = accumulate_risk(cumulative, metrics['uncertainty'], 100)
            altered['cumulative_risk'][t] = cumulative
        result = c.fit_from_diagnostics(rows, self.binding)
        self.assertEqual(result['thresholds_by_gap'], self.audit['thresholds_by_gap'])
        report = result['validation']['per_gap']['20']
        self.assertEqual(report['mean_seed_coverage']['uncertainty_upper'], 1.0)
        self.assertAlmostEqual(report['mean_seed_coverage']['disagreement_upper'], .9)
        self.assertEqual(report['failed_metrics'], ['disagreement_upper'])
        self.assertFalse(result['validation']['passed'])
        self.assertFalse(result['active'])

    def test_zero_scale_is_not_implemented_instead_of_default_threshold(self):
        rows = copy.deepcopy(self.rows)
        for row in rows:
            row['termination_risk'][:] = 0
        result = c.fit_from_diagnostics(rows, self.binding)
        self.assertFalse(result['thresholds_complete'])
        self.assertFalse(result['validation']['passed'])
        self.assertFalse(result['active'])
        self.assertTrue(result['not_implemented'])
        self.assertNotIn('termination_weight', result['thresholds_by_gap']['100'])
        self.assertIn('NOT_IMPLEMENTED_incomplete_data_derived_thresholds', result['inactive_reasons'])

    def test_zero_measured_candidate_spread_cannot_invent_temperature(self):
        rows = copy.deepcopy(self.rows)
        for row in rows:
            row['energy'][:] = .1
        result = c.fit_from_diagnostics(rows, self.binding)
        self.assertFalse(result['active'])
        self.assertFalse(result['thresholds_complete'])
        self.assertIn('positive_measured_candidate_energy_spread_for_temperature', result['not_implemented']['20'])

    def test_missing_duplicate_and_test_seed_cells_rejected(self):
        mutations = []
        mutations.append(self.rows[:-1])
        duplicate = copy.deepcopy(self.rows); duplicate[-1] = copy.deepcopy(duplicate[0]); mutations.append(duplicate)
        for field, value in [('seed', 60000), ('gap', 4), ('seed', 55000.0), ('gap', True)]:
            rows = copy.deepcopy(self.rows); rows[0][field] = value; mutations.append(rows)
        for rows in mutations:
            with self.assertRaises(ValueError): c.fit_from_diagnostics(rows, self.binding)

    def test_required_measured_fields_are_not_silently_filled(self):
        rows = copy.deepcopy(self.rows); del rows[0]['reward_pred']
        with self.assertRaisesRegex(ValueError, 'NOT_IMPLEMENTED'):
            c.fit_from_diagnostics(rows, self.binding)

    def test_wrong_dtype_shape_truncation_and_nonfinite_rejected(self):
        for kind in ('dtype', 'shape', 'timestamp', 'nan', 'inf', 'negative_inf'):
            rows = copy.deepcopy(self.rows)
            if kind == 'dtype': rows[0]['energy'] = rows[0]['energy'].astype(np.float32)
            elif kind == 'shape': rows[0]['reward_pred'] = rows[0]['reward_pred'][:-1]
            elif kind == 'timestamp': rows[0]['timestamp'] -= 1
            else: rows[0]['energy'][0, 0] = {'nan': np.nan, 'inf': np.inf, 'negative_inf': -np.inf}[kind]
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                c.fit_from_diagnostics(rows, self.binding)

    def test_probability_uncertainty_risk_and_ess_must_match_formulas(self):
        for key in ('candidate_probs', 'uncertainty', 'disagreement', 'ess_ratio', 'cumulative_risk'):
            rows = copy.deepcopy(self.rows); rows[0][key].flat[0] += .01
            with self.subTest(field=key), self.assertRaises(ValueError):
                c.fit_from_diagnostics(rows, self.binding)

    def test_source_bank_model_version_mismatch_fails_closed(self):
        for key in ('source_sha256', 'bank_manifest_sha256', 'model_sha256', 'protocol_sha256'):
            binding = copy.deepcopy(self.binding); binding[key] = 'e'*64
            with self.subTest(field=key), self.assertRaises(ValueError): c.fit_from_diagnostics(self.rows, binding)
        rows = copy.deepcopy(self.rows); rows[0]['binding']['model_version'] = True
        with self.assertRaises(ValueError): c.fit_from_diagnostics(rows, self.binding)
        for field, value in [('model_version', True), ('model_sha256', None),
                ('measurement_kind', 'invented'), ('diagnostics_manifest_sha256', '')]:
            binding = copy.deepcopy(self.binding); binding[field] = value
            with self.assertRaises(ValueError): c.fit_from_diagnostics(self.rows, binding)

    def test_input_arrays_and_global_rng_remain_unchanged(self):
        before = c.records_sha256(self.rows)
        state_before = copy.deepcopy(np.random.get_state())
        c.fit_from_diagnostics(self.rows, self.binding)
        self.assertEqual(before, c.records_sha256(self.rows))
        state_after = np.random.get_state()
        self.assertEqual(state_before[0], state_after[0])
        np.testing.assert_array_equal(state_before[1], state_after[1])
        self.assertEqual(state_before[2:], state_after[2:])

    def test_inactive_audit_seal_no_overwrite_and_default_load_rejects(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'synthetic_inactive_audit.json'
            receipt = c.seal_calibration(self.audit, path, self.binding)
            self.assertFalse(receipt['active'])
            with self.assertRaises(FileExistsError): c.seal_calibration(self.audit, path, self.binding)
            with self.assertRaises(ValueError): c.load_calibration(path, receipt['sha256'], self.binding)
            loaded = c.load_calibration(path, receipt['sha256'], self.binding, require_active=False)
            self.assertEqual(loaded, self.audit)
            path.write_bytes(path.read_bytes() + b' ')
            with self.assertRaisesRegex(ValueError, 'file digest'):
                c.load_calibration(path, receipt['sha256'], self.binding, require_active=False)

    def test_record_tampering_and_false_fixture_activation_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'must_not_exist.json'
            record = copy.deepcopy(self.audit); record['thresholds_by_gap']['1']['max_energy'] += .01
            with self.assertRaises(ValueError): c.seal_calibration(record, path, self.binding)
            record = copy.deepcopy(self.audit); record['active'] = True; record['inactive_reasons'] = []
            record['sha256'] = calibration_sha256(record)
            with self.assertRaises(ValueError): c.seal_calibration(record, path, self.binding)
            self.assertFalse(path.exists())

    def test_resigned_semantic_coverage_and_trust_cap_tampering_rejected(self):
        for kind in ('coverage', 'lambda', 'kl', 'criterion'):
            record = copy.deepcopy(self.audit)
            if kind == 'coverage': record['validation']['per_gap']['1']['per_seed_coverage']['55010']['energy_upper'] = 0.0
            elif kind == 'lambda': record['thresholds_by_gap']['1']['lambda_max'] = .3
            elif kind == 'kl': record['thresholds_by_gap']['1']['kl_limit'] = .03
            else: record['validation']['upper_min_empirical_coverage'] = .5
            record['sha256'] = calibration_sha256(record)
            with tempfile.TemporaryDirectory() as tmp, self.subTest(kind=kind), self.assertRaises(ValueError):
                c.seal_calibration(record, Path(tmp)/'invalid.json', self.binding)

    def test_load_requires_current_model_and_trusted_byte_digest(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'synthetic_inactive_audit.json'
            receipt = c.seal_calibration(self.audit, path, self.binding)
            binding = copy.deepcopy(self.binding); binding['model_version'] += 1
            with self.assertRaises(ValueError): c.load_calibration(path, receipt['sha256'], binding, require_active=False)
            with self.assertRaises(ValueError): c.load_calibration(path, '0'*64, self.binding, require_active=False)

    def test_no_real_production_artifact_is_created_by_these_tests(self):
        self.assertEqual(self.binding['measurement_kind'], 'synthetic_test_fixture')
        self.assertEqual(self.audit['measurement_kind'], 'synthetic_test_fixture')
        self.assertFalse(self.audit['active'])

    def test_fixture_cannot_claim_frozen_gpu_production_evidence(self):
        binding = copy.deepcopy(self.binding)
        binding['frozen_gpu_zero_update_verified'] = True
        binding['frozen_gpu_report_sha256'] = '6'*64
        with self.assertRaises(ValueError): c.fit_from_diagnostics(self.rows, binding)
        self.assertFalse(self.audit['applicability']['closed_loop_policy_coverage_guaranteed'])
        self.assertFalse(self.audit['applicability']['multi_step_planning_risk_calibrated'])

    def test_open_loop_prototype_cannot_self_certify_deployment(self):
        self.assertEqual(self.audit['deployment']['status'], 'BLOCKED_DEPLOYMENT_CALIBRATION')
        self.assertIn('BLOCKED_DEPLOYMENT_CALIBRATION', self.audit['inactive_reasons'])
        self.assertFalse(self.audit['deployment']['frozen_gpu_provenance_alone_unlocks_deployment'])
        record = copy.deepcopy(self.audit)
        record['applicability']['multi_step_planning_risk_calibrated'] = True
        record['sha256'] = calibration_sha256(record)
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(ValueError):
            c.seal_calibration(record, Path(tmp)/'false_deployment.json', self.binding)


if __name__ == '__main__':
    unittest.main()
