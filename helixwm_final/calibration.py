"""Fit frozen engineering gates from complete, independently measured traces.

This module never runs a model or creates measurements. A production caller
must verify its measurement-file manifest before supplying its trusted digest.
Test fixtures remain explicitly synthetic and can NEVER activate the module.
The current open-loop protocol is a prototype only: even independently frozen
GPU measurements cannot activate deployment until the online fused candidate
distribution, H-step proposal utility/risk and fallback outcomes are matched.

Per gap, E(t)=min_i energy(t,i), U is predictive entropy, J is the separate
model-conditional disagreement from belief.uncertainty_metrics, and R is
belief.accumulate_risk. Upper gates use fit-only Q_.99(method='higher'); the
ESS lower gate uses fit-only Q_.05(method='higher'). Temperature is Q_.99 of
mean_i(energy_i-min_i energy_i), not a fabricated default. Reconnection ESS
is then recomputed from the measured energies with that frozen temperature
and absolute energy gate. The reported proposal ESS remains separate.

lambda_max=.25*Q_.05(reconnection_ESS/K)*(1-Q_.99(U)), bounded by .25;
KL=.02 is a structural engineering trust region, NOT a fitted safety claim.
Planning horizon is the largest h in {1,3,5}, h<=gap, whose fit-only first-h
prediction diagnostics meet the fit energy/U/risk/termination upper gates.
It calibrates fixed-action prediction prefixes, not arbitrary action safety.
Objective weights are inverse positive fit Q_.99 scales for absolute reward,
uncertainty, termination risk and squared-mean slew. A zero scale or no valid
prefix is NOT_IMPLEMENTED/inactive; no plausible-looking default is inserted.

Holdout seeds NEVER change thresholds. Their per-gap empirical upper coverage
must be >=.95, and ESS lower coverage >=.90. These engineering activation
checks are not IID confidence guarantees; per-seed coverage is retained.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np

from .belief import (GAP_BINS, THRESHOLD_FIELDS, accumulate_risk,
                     calibration_sha256, canonical_json, normalized_weights,
                     uncertainty_metrics, validate_thresholds)

FIT_SEEDS = tuple(range(55000, 55010))
VALIDATION_SEEDS = tuple(range(55010, 55020))
MEASUREMENT_PROTOCOL = 'open_loop_random_action_bank_v1'
IDENTITY_FIELDS = ('source_sha256', 'bank_manifest_sha256', 'model_sha256',
                   'model_version', 'protocol_sha256', 'measurement_kind', 'measurement_protocol')
BINDING_ONLY_FIELDS = ('diagnostics_manifest_sha256', 'frozen_gpu_zero_update_verified',
                       'frozen_gpu_report_sha256')
METRICS = ('uncertainty', 'disagreement', 'cumulative_risk', 'ess_ratio',
           'reward_pred', 'termination_risk', 'action_amplitude', 'action_slew',
           'slew_squared_mean')
ARRAY_FIELDS = ('timestamp', 'energy', 'candidate_probs', 'candidate_weights') + METRICS
ROW_FIELDS = {'seed', 'gap', 'binding', 'epistemic_available', 'model_updates'} | set(ARRAY_FIELDS)
UPPER_QUANTILE, LOWER_QUANTILE = .99, .05
UPPER_COVERAGE, LOWER_COVERAGE = .95, .90
DEPLOYMENT_BLOCKERS = (
    'online_fusion_candidate_distribution_not_matched',
    'H_step_cumulative_termination_uncertainty_and_utility_not_measured',
    'fallback_action_amplitude_and_slew_outcomes_not_validated',
)


def _sha(value):
    return type(value) is str and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def _binding(binding):
    if type(binding) is not dict or set(binding) != set(IDENTITY_FIELDS) | set(BINDING_ONLY_FIELDS):
        raise ValueError('Calibration binding fields differ')
    for key in ('source_sha256', 'bank_manifest_sha256', 'model_sha256',
                'protocol_sha256', 'diagnostics_manifest_sha256'):
        if not _sha(binding[key]):
            raise ValueError('Missing or invalid trusted binding SHA: ' + key)
    if type(binding['model_version']) is not int or binding['model_version'] < 0:
        raise ValueError('Model version must be a nonnegative integer')
    if binding['measurement_kind'] not in ('measured_offline_model', 'synthetic_test_fixture'):
        raise ValueError('No actual or explicitly synthetic measurement provenance')
    if binding['measurement_protocol'] != MEASUREMENT_PROTOCOL:
        raise ValueError('Unregistered measurement distribution')
    if type(binding['frozen_gpu_zero_update_verified']) is not bool:
        raise ValueError('Independent frozen GPU verification must be explicit')
    if binding['frozen_gpu_zero_update_verified']:
        if (binding['measurement_kind'] != 'measured_offline_model'
                or not _sha(binding['frozen_gpu_report_sha256'])):
            raise ValueError('A synthetic or unbound frozen GPU report cannot activate calibration')
    elif binding['frozen_gpu_report_sha256'] is not None:
        raise ValueError('Unverified GPU evidence cannot be attached as verified provenance')
    return {key: copy.deepcopy(binding[key]) for key in IDENTITY_FIELDS}


def records_sha256(records):
    """Content digest separate from the external manifest's own byte digest."""
    digest = hashlib.sha256()
    for row in sorted(records, key=lambda r: (r['seed'], r['gap'])):
        metadata = {key: value for key, value in row.items() if key not in ARRAY_FIELDS}
        digest.update(canonical_json(metadata))
        for key in ARRAY_FIELDS:
            arr = np.asarray(row[key])
            if arr.dtype.hasobject:
                raise ValueError('Object diagnostic arrays are forbidden')
            digest.update(canonical_json({'field': key, 'dtype': arr.dtype.str, 'shape': list(arr.shape)}))
            digest.update(np.ascontiguousarray(arr).tobytes())
    return digest.hexdigest()


def _close(actual, expected, name):
    if not np.allclose(actual, expected, rtol=0, atol=2e-12):
        raise ValueError('Measured diagnostic contradicts the registered formula: ' + name)


def _validate_records(records, binding):
    identity = _binding(binding)
    if not isinstance(records, (list, tuple)) or len(records) != 120:
        raise ValueError('Exactly 20 seeds by 6 measured gap cells are required')
    seen, latent_shape = set(), None
    for row in records:
        if type(row) is not dict or set(row) != ROW_FIELDS:
            raise ValueError('NOT_IMPLEMENTED: incomplete or unexpected measured diagnostic fields')
        if (type(row['seed']) is not int or row['seed'] not in FIT_SEEDS + VALIDATION_SEEDS
                or type(row['gap']) is not int or row['gap'] not in GAP_BINS):
            raise ValueError('Unregistered calibration seed or gap; final-test seeds are forbidden')
        cell = (row['seed'], row['gap'])
        if cell in seen:
            raise ValueError('Repeated calibration cell')
        seen.add(cell)
        if (type(row['binding']) is not dict or set(row['binding']) != set(identity)
                or any(type(row['binding'][key]) is not type(value) or row['binding'][key] != value
                       for key, value in identity.items())):
            raise ValueError('Diagnostic source/bank/model/version/protocol provenance differs')
        if row['epistemic_available'] is not False or type(row['model_updates']) is not int or row['model_updates'] != 0:
            raise ValueError('Offline zero-update, non-epistemic diagnostic contract violated')
        count = row['gap'] + 8
        for key in ARRAY_FIELDS:
            value = row[key]
            dtype = np.int32 if key == 'timestamp' else np.float64
            if not isinstance(value, np.ndarray) or value.dtype != dtype or not np.isfinite(value).all():
                raise ValueError('Invalid diagnostic ndarray/dtype/finite values: ' + key)
            expected = (count, 8) if key in ('energy', 'candidate_weights') else (count,)
            if key == 'candidate_probs':
                if value.ndim != 4 or value.shape[:2] != (count, 8) or value.shape[2] < 1 or value.shape[3] < 2:
                    raise ValueError('Candidate probabilities require float64[T,8,S,C]')
                if latent_shape is None:
                    latent_shape = value.shape[2:]
                if value.shape[2:] != latent_shape:
                    raise ValueError('Latent dimensions changed within one model calibration')
            elif value.shape != expected:
                raise ValueError('Diagnostic length/shape differs: ' + key)
        if not np.array_equal(row['timestamp'], np.arange(17, 17 + count, dtype=np.int32)):
            raise ValueError('Calibration must begin at observation timestamp 17')
        if np.any(row['energy'] < 0):
            raise ValueError('Energy cannot be negative')
        for key in ('uncertainty', 'disagreement', 'ess_ratio', 'termination_risk', 'action_amplitude'):
            if np.any(row[key] < 0) or np.any(row[key] > 1 + 2e-12):
                raise ValueError('Normalized diagnostic outside [0,1]: ' + key)
        for key, upper in (('cumulative_risk', 100), ('action_slew', 2), ('slew_squared_mean', 4)):
            if np.any(row[key] < 0) or np.any(row[key] > upper):
                raise ValueError('Diagnostic outside its physical/numerical units: ' + key)
        if np.any(row['slew_squared_mean'] > row['action_slew']**2 + 2e-12):
            raise ValueError('Mean squared slew exceeds maximum squared slew')
        cumulative = 0.0
        for t in range(count):
            weights = row['candidate_weights'][t]
            if np.any(weights < 0) or abs(float(weights.sum()) - 1.0) > 2e-12:
                raise ValueError('Measured candidate weights must be normalized')
            normalized = normalized_weights(weights, 8)
            metrics = uncertainty_metrics(row['candidate_probs'][t], normalized)
            _close(row['uncertainty'][t], metrics['uncertainty'], 'uncertainty')
            _close(row['disagreement'][t], metrics['model_conditional_disagreement'], 'disagreement')
            _close(row['ess_ratio'][t], 1 / float(np.square(normalized).sum()) / 8, 'proposal ESS ratio')
            if t < row['gap']:
                cumulative = accumulate_risk(cumulative, metrics['uncertainty'], 100)
                _close(row['cumulative_risk'][t], cumulative, 'missing cumulative risk')
            elif t == row['gap']:
                _close(row['cumulative_risk'][t], cumulative, 'first restored pre-correction risk')
            # Later recovery values are actual pre-correction measurements;
            # resetting risk depends on quality and is not invented here.
    expected = {(seed, gap) for seed in FIT_SEEDS + VALIDATION_SEEDS for gap in GAP_BINS}
    if seen != expected:
        raise ValueError('Incomplete seed/gap Cartesian product')
    return identity


def _quantile(values, q):
    return float(np.quantile(np.asarray(values, np.float64), q, method='higher'))


def _energy_ess(energy, max_energy, temperature):
    values = []
    for scores in energy:
        accepted = scores <= max_energy
        if not accepted.any():
            values.append(0.0)
            continue
        weights = np.zeros(8, np.float64)
        weights[accepted] = np.exp(-(scores[accepted] - scores[accepted].min()) / temperature)
        weights /= weights.sum()
        values.append(1 / float(np.square(weights).sum()) / 8)
    return np.asarray(values)


def _fit_gap(rows, gap):
    energy = np.concatenate([r['energy'] for r in rows])
    pooled = {key: np.concatenate([r[key] for r in rows]) for key in METRICS}
    q = lambda x: _quantile(x, UPPER_QUANTILE)
    thresholds = {'max_energy': q(energy.min(axis=1)),
        'max_uncertainty': q(pooled['uncertainty']),
        'max_disagreement': q(pooled['disagreement']),
        'max_cumulative_risk': q(pooled['cumulative_risk']),
        'action_amplitude': q(pooled['action_amplitude']),
        'slew_limit': q(pooled['action_slew']),
        'max_termination_risk': q(pooled['termination_risk']),
        'temperature': q((energy - energy.min(axis=1, keepdims=True)).mean(axis=1)),
        'kl_limit': .02}
    missing = []
    if thresholds['temperature'] <= 0:
        missing.append('positive_measured_candidate_energy_spread_for_temperature')
    scales = {'reward_weight': q(np.abs(pooled['reward_pred'])),
              'uncertainty_weight': q(pooled['uncertainty']),
              'termination_weight': q(pooled['termination_risk']),
              'slew_weight': q(pooled['slew_squared_mean'])}
    for key, value in scales.items():
        if value <= 0 or not math.isfinite(1.0 / value):
            missing.append('positive_measured_scale_for_' + key)
        else:
            thresholds[key] = 1.0 / value
    if thresholds['temperature'] > 0:
        ess = _energy_ess(energy, thresholds['max_energy'], thresholds['temperature'])
        thresholds['min_ess_ratio'] = _quantile(ess, LOWER_QUANTILE)
        thresholds['lambda_max'] = min(.25, .25 * thresholds['min_ess_ratio'] * (1 - thresholds['max_uncertainty']))
    horizons = []
    for horizon in (1, 3, 5):
        if horizon > gap:
            continue
        prefix = lambda key: np.concatenate([r[key][:horizon] for r in rows])
        if (q(prefix('energy').min(axis=1)) <= thresholds['max_energy']
                and q(prefix('uncertainty')) <= thresholds['max_uncertainty']
                and q(prefix('disagreement')) <= thresholds['max_disagreement']
                and q(prefix('cumulative_risk')) <= thresholds['max_cumulative_risk']
                and q(prefix('termination_risk')) <= thresholds['max_termination_risk']):
            horizons.append(horizon)
    if not horizons:
        missing.append('qualified_measured_fixed_action_planning_prefix')
    else:
        thresholds['planning_horizon'] = max(horizons)
    return thresholds, missing


def _coverage(row, thresholds):
    energy = row['energy'].min(axis=1)
    return {'energy_upper': float(np.mean(energy <= thresholds['max_energy'])),
            'uncertainty_upper': float(np.mean(row['uncertainty'] <= thresholds['max_uncertainty'])),
            'disagreement_upper': float(np.mean(row['disagreement'] <= thresholds['max_disagreement'])),
            'cumulative_risk_upper': float(np.mean(row['cumulative_risk'] <= thresholds['max_cumulative_risk'])),
            'termination_risk_upper': float(np.mean(row['termination_risk'] <= thresholds['max_termination_risk'])),
            'action_amplitude_upper': float(np.mean(row['action_amplitude'] <= thresholds['action_amplitude'])),
            'action_slew_upper': float(np.mean(row['action_slew'] <= thresholds['slew_limit'])),
            'ess_lower': float(np.mean(_energy_ess(row['energy'], thresholds['max_energy'],
                                  thresholds['temperature']) >= thresholds['min_ess_ratio']))}


def fit_from_diagnostics(records, binding):
    """Return an immutable-by-hash audit artifact; never repair bad evidence.

    Row binding has IDENTITY_FIELDS only. The top-level trusted external
    diagnostics_manifest_sha256 is excluded from its own underlying file
    content; records_sha256 independently binds the supplied arrays. This
    function does not pretend to execute or authenticate the measurement job.
    """
    _validate_records(records, binding)
    table, unavailable, validation = {}, {}, {}
    for gap in GAP_BINS:
        fit = [row for row in records if row['gap'] == gap and row['seed'] in FIT_SEEDS]
        heldout = [row for row in records if row['gap'] == gap and row['seed'] in VALIDATION_SEEDS]
        thresholds, missing = _fit_gap(fit, gap)
        table[str(gap)] = thresholds
        if missing:
            unavailable[str(gap)] = missing
            validation[str(gap)] = {'passed': False, 'status': 'NOT_IMPLEMENTED', 'reasons': missing}
            continue
        per_seed = {str(row['seed']): _coverage(row, thresholds) for row in heldout}
        averages = {name: float(np.mean([values[name] for values in per_seed.values()]))
                    for name in next(iter(per_seed.values()))}
        failed = [name for name, value in averages.items()
                  if value < (LOWER_COVERAGE if name == 'ess_lower' else UPPER_COVERAGE)]
        validation[str(gap)] = {'passed': not failed, 'per_seed_coverage': per_seed,
                               'mean_seed_coverage': averages, 'failed_metrics': failed,
                               'seed_count': 10, 'observations_per_seed': gap + 8}
    complete = not unavailable
    if complete:
        validate_thresholds(table)
    passed = complete and all(row['passed'] for row in validation.values())
    reasons = ['BLOCKED_DEPLOYMENT_CALIBRATION']
    if unavailable:
        reasons.append('NOT_IMPLEMENTED_incomplete_data_derived_thresholds')
    if not passed:
        reasons.append('independent_validation_failed')
    if binding['measurement_kind'] != 'measured_offline_model':
        reasons.append('synthetic_test_fixture_not_production')
    if not binding['frozen_gpu_zero_update_verified']:
        reasons.append('independent_frozen_gpu_measurement_not_verified')
    result = {'schema_version': 1, 'frozen': True, 'active': not reasons,
        'calibration_id': 'final_c1_v' + str(binding['model_version']) + '_' + records_sha256(records)[:16],
        **copy.deepcopy(binding), 'records_sha256': records_sha256(records),
        'fit_seeds': list(FIT_SEEDS), 'validation_seeds': list(VALIDATION_SEEDS),
        'thresholds_by_gap': table, 'thresholds_complete': complete,
        'not_implemented': unavailable, 'inactive_reasons': reasons,
        'deployment': {'status': 'BLOCKED_DEPLOYMENT_CALIBRATION',
            'blockers': list(DEPLOYMENT_BLOCKERS),
            'frozen_gpu_provenance_alone_unlocks_deployment': False},
        'validation': {'passed': passed, 'upper_min_empirical_coverage': UPPER_COVERAGE,
            'lower_min_empirical_coverage': LOWER_COVERAGE, 'per_gap': validation,
            'holdout_changes_thresholds': False, 'iid_or_safety_guarantee': False},
        'fitting': {'upper_quantile': .99, 'lower_quantile': .05, 'method': 'higher',
            'fit_only': True, 'lambda_rule': 'min(.25,.25*fit_Q05(reconnection_ESS/K)*(1-fit_Q99(U)))',
            'planning_rule': 'largest qualified fit fixed-action missing prefix in {1,3,5}, capped by gap',
            'objective_weights': 'inverse_positive_fit_Q99_scales',
            'structural_caps': {'lambda_max': .25, 'kl_limit': .02},
            'structural_caps_proven_safe_by_calibration': False,
            'proposal_ess_distinct_from_energy_reweighted_ess': True},
        'applicability': {'measurement_protocol': MEASUREMENT_PROTOCOL,
            'state_action_distribution': 'no_fusion_prior_candidates_under_fixed_random_executed_actions',
            'online_fused_actor_distribution_coverage': 'NOT_VERIFIED',
            'closed_loop_policy_coverage_guaranteed': False,
            'planning_threshold_statistics': 'single_step_risk_and_fixed_action_missing_prefix_diagnostics_not_H_step_proposal_rollout_risk',
            'multi_step_planning_risk_calibrated': False,
            'loss_free_calibration_guarantees_policy_coverage': False},
        'provenance': {'model_updates': 0, 'measurement_executed_by_fitter': False,
            'external_manifest_bytes_verified_by_fitter': False,
            'independent_frozen_gpu_report_verified_by_fitter': False,
            'external_manifest_digest_must_be_verified_by_measurement_loader': True,
            'true_epistemic_available': False, 'hidden_clean_targets_to_policy_or_learner': False}}
    result['sha256'] = calibration_sha256(result)
    return result


def _validate_artifact(record, binding, require_active):
    _binding(binding)
    if (type(record) is not dict or type(record.get('schema_version')) is not int
            or record.get('schema_version') != 1 or record.get('frozen') is not True):
        raise ValueError('Calibration artifact schema or freeze differs')
    if not _sha(record.get('sha256')) or record['sha256'] != calibration_sha256(record):
        raise ValueError('Calibration artifact digest mismatch')
    if any(type(record.get(key)) is not type(value) or record.get(key) != value for key, value in binding.items()):
        raise ValueError('Calibration artifact does not match the current trusted binding')
    if record.get('fit_seeds') != list(FIT_SEEDS) or record.get('validation_seeds') != list(VALIDATION_SEEDS):
        raise ValueError('Calibration seed partition changed')
    if not _sha(record.get('records_sha256')) or type(record.get('calibration_id')) is not str or not record['calibration_id']:
        raise ValueError('Calibration data identity missing')
    if type(record.get('active')) is not bool or type(record.get('thresholds_complete')) is not bool:
        raise ValueError('Calibration activation state requires booleans')
    if record['thresholds_complete'] is not (not bool(record.get('not_implemented'))):
        raise ValueError('Calibration completeness contradicts missing measurements')
    deployment = record.get('deployment', {})
    applicability = record.get('applicability', {})
    if (deployment.get('status') != 'BLOCKED_DEPLOYMENT_CALIBRATION'
            or deployment.get('blockers') != list(DEPLOYMENT_BLOCKERS)
            or deployment.get('frozen_gpu_provenance_alone_unlocks_deployment') is not False
            or 'BLOCKED_DEPLOYMENT_CALIBRATION' not in record.get('inactive_reasons', [])
            or applicability.get('measurement_protocol') != MEASUREMENT_PROTOCOL
            or applicability.get('online_fused_actor_distribution_coverage') != 'NOT_VERIFIED'
            or applicability.get('closed_loop_policy_coverage_guaranteed') is not False
            or applicability.get('multi_step_planning_risk_calibrated') is not False):
        raise ValueError('Open-loop prototype cannot self-certify deployment applicability')
    validation = record.get('validation')
    if not isinstance(validation, dict) or type(validation.get('passed')) is not bool:
        raise ValueError('Calibration validation evidence missing')
    if (validation.get('upper_min_empirical_coverage') != UPPER_COVERAGE
            or validation.get('lower_min_empirical_coverage') != LOWER_COVERAGE
            or validation.get('holdout_changes_thresholds') is not False
            or validation.get('iid_or_safety_guarantee') is not False
            or set(validation.get('per_gap', {})) != {str(g) for g in GAP_BINS}):
        raise ValueError('Calibration validation criterion or gap coverage changed')
    gap_passes = []
    for gap, report in validation['per_gap'].items():
        if type(report.get('passed')) is not bool:
            raise ValueError('Missing per-gap validation outcome')
        gap_passes.append(report['passed'])
        if report.get('status') == 'NOT_IMPLEMENTED':
            if report['passed'] or not report.get('reasons'):
                raise ValueError('Unavailable calibration cannot pass')
            continue
        per_seed = report.get('per_seed_coverage', {})
        if set(per_seed) != {str(seed) for seed in VALIDATION_SEEDS}:
            raise ValueError('Missing per-seed holdout coverage')
        names = {'energy_upper', 'uncertainty_upper', 'disagreement_upper', 'cumulative_risk_upper',
                 'termination_risk_upper', 'action_amplitude_upper', 'action_slew_upper', 'ess_lower'}
        for values in per_seed.values():
            if set(values) != names or any(type(v) not in (float, int) or not math.isfinite(v)
                                           or not 0 <= v <= 1 for v in values.values()):
                raise ValueError('Invalid holdout coverage values')
        means = {key: float(np.mean([r[key] for r in per_seed.values()])) for key in names}
        if report.get('mean_seed_coverage') != means:
            raise ValueError('Holdout aggregate does not match per-seed evidence')
        failed = {key for key, value in means.items()
                  if value < (LOWER_COVERAGE if key == 'ess_lower' else UPPER_COVERAGE)}
        if set(report.get('failed_metrics', [])) != failed or report['passed'] is not (not failed):
            raise ValueError('Holdout pass flag contradicts coverage')
        if (type(report.get('seed_count')) is not int or report['seed_count'] != 10
                or type(report.get('observations_per_seed')) is not int
                or report['observations_per_seed'] != int(gap) + 8):
            raise ValueError('Holdout coverage count differs')
    if validation['passed'] is not (record['thresholds_complete'] and all(gap_passes)):
        raise ValueError('Overall validation contradicts per-gap evidence')
    fitting = record.get('fitting', {})
    if (fitting.get('upper_quantile') != .99 or fitting.get('lower_quantile') != .05
            or fitting.get('method') != 'higher' or fitting.get('fit_only') is not True
            or fitting.get('structural_caps') != {'lambda_max': .25, 'kl_limit': .02}):
        raise ValueError('Registered fitting or structural caps changed')
    if record['thresholds_complete']:
        validate_thresholds(record['thresholds_by_gap'])
        if any(row['lambda_max'] > .25 or row['kl_limit'] > .02 for row in record['thresholds_by_gap'].values()):
            raise ValueError('Structural trust-region caps exceeded')
    # This schema covers only the open-loop engineering prototype. A future
    # deployment protocol requires new matched measurements and registration,
    # not flipping this flag or trusting a GPU-provenance hash as applicability.
    expected_active = False
    if record['active'] is not expected_active:
        raise ValueError('Calibration active flag contradicts evidence')
    if require_active and not record['active']:
        raise ValueError('Calibration is inactive: ' + ','.join(record.get('inactive_reasons', [])))


def seal_calibration(record, path, binding):
    """Persist a complete active result OR a clearly inactive audit, no overwrite."""
    _validate_artifact(record, binding, require_active=False)
    path = Path(path)
    raw = canonical_json(record) + b'\n'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    return {'file': str(path), 'sha256': hashlib.sha256(raw).hexdigest(),
            'bytes': len(raw), 'record_sha256': record['sha256'], 'active': record['active']}


def load_calibration(path, expected_file_sha256, binding, *, require_active=True):
    """Only a trusted byte digest plus current model/source binding can load."""
    if not _sha(expected_file_sha256):
        raise ValueError('A trusted calibration file SHA256 is required')
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_file_sha256:
        raise ValueError('Calibration file digest mismatch')
    def pairs(items):
        value = {}
        for key, item in items:
            if key in value:
                raise ValueError('Duplicate calibration JSON key')
            value[key] = item
        return value
    def nonfinite(value):
        raise ValueError('Nonfinite calibration JSON: ' + value)
    record = json.loads(raw, object_pairs_hook=pairs, parse_constant=nonfinite)
    _validate_artifact(record, binding, require_active)
    return copy.deepcopy(record)
