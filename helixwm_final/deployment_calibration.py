"""Closed-loop deployment calibration v2, separate from open-loop prototype.

COLLECTOR CONTRACT
==================
For each pinned model: collect exactly seeds 55000..55009 x gaps 1/5/10/20/50/100
using a preregistered pilot; fit once, persist the candidate before validation.
Then actually execute that frozen candidate on seeds 55010..55019 x all gaps.
Each window has 16 warm-up actions, a gap beginning at observation 17 and eight
recovery actions. Records below cover observations 17..16+gap+8, while the
replay anchor retains warm-up and all RNG state. Every decision is closed-loop;
an open-loop clean bank cannot be relabeled as a closed-loop validation window.

At model output time use planning.categorical_statistics, retaining normalized
predictive/expected entropy, candidate weights and per-state/RNG digests. Retain
all executed actions/rewards/masks and real planning tensors, NOT the potentially
285-GiB full probability sequence. This is an explicitly versioned sufficient-
statistics format; a byte-verified replay checkpoint + RNG/environment/controller
anchor and independent replay verification remain mandatory. Existing raw files
must not be deleted. Entropy identities can be recomputed; their authenticity
requires independently replaying the bound checkpoint, not a passed boolean.

Each row uses ARRAY_FIELDS, plus plans (one or more measured planning branches
per H=1/3/5, with actual timestamps/proposal identities and HxK predictions).
Extra scratch probes must have independent RNG, never execute in the world or
change controller state. Counterfactual ground-truth rewards are not fabricated.
Model predicted rewards are distinguished from actual returned env rewards.

Fit Q99 upper/Q05 ESS uses only fit data. Fixed utility units and structural
caps are engineering registration, not fitted guarantees. Validation uses the
actual frozen candidate and cannot change it. Coverage is averaged with equal
weight per seed within each gap; every cell must satisfy action/fusion invariants. Seeds
are few and correlated samples are NOT an IID coverage guarantee. Functional
failure is reported, not relabeled as physical safety. Failed attempts are kept;
reusing the same held-out windows to tune until PASS is disallowed.

Only load_deployment_calibration can promote actual file-backed evidence to
active. It verifies externally supplied manifest/context byte identities,
recomputes fit+validation, verifies freeze and independent replay transcripts,
and rejects synthetic provenance. The loader does not authenticate hardware
against a malicious author of the entire evidence bundle: trusted context must
come from the actual collection job. In-memory dictionaries cannot authorize
production. CPU tests never produce production evidence.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

from .belief import accumulate_risk, canonical_json
from .planning import (HORIZONS, OPERATOR_VERSION, UTILITY_WEIGHTS,
                       REWARD_PROJECTION, rollout_statistics, uncertainty_from_sufficient)

SCHEMA = 'helixwm_final_closed_loop_deployment_v2'
MEASUREMENT_PROTOCOL = 'frozen_candidate_closed_loop_sufficient_statistics_v2'
FIT_SEEDS = tuple(range(55000, 55010))
VALIDATION_SEEDS = tuple(range(55010, 55020))
GAPS = (1, 5, 10, 20, 50, 100)
CONTEXT_FIELDS = ('source_sha256', 'model_sha256', 'bank_manifest_sha256',
                  'protocol_sha256', 'collector_sha256', 'epistemic_heads_sha256',
                  'model_version', 'measurement_kind')
SHA_FIELDS = CONTEXT_FIELDS[:6]
ARRAY_FIELDS = ('timestamp', 'availability', 'actions', 'actual_rewards',
                'predictive_entropy', 'expected_entropy', 'candidate_weights',
                'cumulative_risk', 'risk_reset', 'energy', 'fallback',
                'fusion_lambda', 'fusion_kl', 'previous_action',
                'epistemic_disagreement', 'temporal_prediction_error')
PLAN_ARRAY_FIELDS = ('rewards', 'continuations', 'actions', 'predictive_entropy',
                     'expected_entropy', 'weights', 'previous_action', 'epistemic_disagreement')
UPPER_COVERAGE = .95
LOWER_COVERAGE = .90
COVERAGE_REVISION = {
    'id': 'equal_seed_per_gap_coverage_20261106_preheldout_v1',
    'registered_before_actual_heldout_collection': True,
    'aggregation': 'arithmetic_mean_of_ten_seed_level_coverages_within_each_gap',
    'per_seed_coverage_failures': 'retained_diagnostics_not_individual_acceptance_gates',
    'structural_invariants': 'every_cell_must_pass_without_aggregation',
    'interpretation': 'empirical_engineering_gate_not_IID_coverage_guarantee',
    'threshold_fit_or_heldout_tuning_changed': False,
}
STRUCTURAL_CAPS = {'lambda_max': .25, 'kl_limit': .02,
                   'action_amplitude': 1., 'slew_limit': 2.}
FREEZE_COMPONENTS = ('model', 'optimizer', 'normalizers', 'counters', 'training_rng', 'epistemic_heads')
EVENT_CLOCK = 'monotonic_perf_counter_ns'
_FILE_VERIFIED_TOKEN = object()


class VerifiedDeployment(dict):
    """In-process evidence handle; deserialized dictionaries lose this authority."""
    def __init__(self, value, *, _token=None):
        if _token is not _FILE_VERIFIED_TOKEN:
            raise ValueError('Only the file-backed loader can issue deployment authority')
        super().__init__(value)

    def __deepcopy__(self, memo):
        return VerifiedDeployment(copy.deepcopy(dict(self), memo), _token=_FILE_VERIFIED_TOKEN)


def verify_deployment_handle(record, context):
    """Fast use-time check after expensive load, bound to CURRENT external context."""
    _context(context)
    if not isinstance(record, VerifiedDeployment):
        raise ValueError('A manual/deserialized dict is not verified deployment evidence')
    _equal(record.get('context'), context, 'current deployment context')
    if record.get('sha256') != _digest({k:v for k,v in record.items() if k!='sha256'}):
        raise ValueError('Verified deployment handle was modified after verification')
    if record.get('active') is not True or record.get('status') != 'ENGINEERING_CALIBRATED':
        raise ValueError('Inactive/synthetic deployment handle cannot run the controller')
    return True


def _sha(value):
    return type(value) is str and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def _digest(value):
    return hashlib.sha256(canonical_json(value)).hexdigest()


def _context(context):
    if type(context) is not dict or set(context) != set(CONTEXT_FIELDS):
        raise ValueError('External deployment context keys differ')
    if any(not _sha(context[name]) for name in SHA_FIELDS):
        raise ValueError('External source/model/bank/protocol/collector/epistemic-heads SHA is required')
    if type(context['model_version']) is not int or context['model_version'] < 0:
        raise ValueError('Model version must be a strict nonnegative integer')
    if context['measurement_kind'] not in ('synthetic_test_fixture', 'actual_gpu_closed_loop'):
        raise ValueError('Open-loop or unregistered provenance cannot deploy')
    return copy.deepcopy(context)


def _equal(left, right, name):
    if type(left) is not type(right) or canonical_json(left) != canonical_json(right):
        raise ValueError('Strict binding mismatch: ' + name)


def _array(value, dtype, shape, name, lower=None, upper=None):
    if not isinstance(value, np.ndarray) or value.dtype != dtype or value.shape != shape or not np.isfinite(value).all():
        raise ValueError('Invalid measured ndarray/dtype/shape: ' + name)
    if lower is not None and np.any(value < lower) or upper is not None and np.any(value > upper):
        raise ValueError('Measured values outside registered units: ' + name)
    return value


def _scalar(value, name, lower=None, upper=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (float, np.float64)) or not math.isfinite(value):
        raise ValueError('Expected finite measured floating point: ' + name)
    if lower is not None and value < lower or upper is not None and value > upper:
        raise ValueError('Measured scalar outside registered units: ' + name)
    return float(value)


def _array_hash(value):
    return hashlib.sha256(canonical_json({'dtype': value.dtype.str, 'shape': list(value.shape)})
                          + np.ascontiguousarray(value).tobytes()).hexdigest()


def records_sha256(rows):
    """Order-independent row digest; includes every retained raw byte and ID."""
    hashes = []
    for row in sorted(rows, key=lambda x: (x['seed'], x['gap'])):
        item = {key: copy.deepcopy(value) for key, value in row.items()
                if key not in ARRAY_FIELDS and key != 'plans'}
        item['arrays'] = {key: _array_hash(row[key]) for key in ARRAY_FIELDS}
        item['plans'] = []
        for plan in row['plans']:
            p = {key: value for key, value in plan.items() if key not in PLAN_ARRAY_FIELDS}
            p['arrays'] = {key: _array_hash(plan[key]) for key in PLAN_ARRAY_FIELDS}
            item['plans'].append(p)
        hashes.append(_digest(item))
    return _digest(hashes)


def diagnostic_content_sha256(row):
    """Bind actual diagnostic arrays to a replay trace without a hash cycle."""
    copy_row = dict(row)
    copy_row['trace_sha256'] = '0'*64
    return records_sha256([copy_row])


def _validate_rows(rows, context, phase, candidate_sha256=None):
    _context(context)
    seeds = FIT_SEEDS if phase == 'fit' else VALIDATION_SEEDS
    if type(rows) not in (tuple, list) or len(rows) != 60:
        raise ValueError('Exactly ten seeds by six gap cells are required per phase')
    seen, measured = set(), {}
    required = set(ARRAY_FIELDS) | {'seed', 'gap', 'phase', 'context', 'measurement_protocol',
               'candidate_sha256', 'model_updates', 'hidden_target_to_controller',
               'hidden_target_to_learner', 'replay_anchor_sha256', 'trace_sha256', 'plans'}
    for row in rows:
        if type(row) is not dict or set(row) != required:
            raise ValueError('Incomplete or unexpected deployment diagnostic fields')
        seed, gap = row['seed'], row['gap']
        if type(seed) is not int or seed not in seeds or type(gap) is not int or gap not in GAPS:
            raise ValueError('Unregistered seed/gap, phase overlap or final-test leakage')
        if (seed, gap) in seen:
            raise ValueError('Duplicate deployment calibration cell')
        seen.add((seed, gap))
        _equal(row['context'], context, 'current source/model/version')
        if row['phase'] != phase or row['measurement_protocol'] != MEASUREMENT_PROTOCOL:
            raise ValueError('Not the registered closed-loop measurement phase')
        if row['candidate_sha256'] != candidate_sha256:
            raise ValueError('Validation did not execute the exact sealed fitted candidate')
        if type(row['model_updates']) is not int or row['model_updates'] != 0:
            raise ValueError('Calibration must perform zero learner updates')
        if row['hidden_target_to_controller'] is not False or row['hidden_target_to_learner'] is not False:
            raise ValueError('Hidden clean target leakage is forbidden')
        if not _sha(row['replay_anchor_sha256']) or not _sha(row['trace_sha256']):
            raise ValueError('Replay anchor and raw execution trace must be bound')
        n = gap + 8
        expected_timestamp = np.arange(17, 17+n, dtype=np.int32)
        if not np.array_equal(_array(row['timestamp'], np.int32, (n,), 'timestamp'), expected_timestamp):
            raise ValueError('Gap/recovery timestamp alignment differs')
        expected_mask = np.arange(n) >= gap
        if not np.array_equal(_array(row['availability'], np.bool_, (n,), 'availability'), expected_mask):
            raise ValueError('Gap/recovery availability differs')
        _array(row['actions'], np.float64, (n, 6), 'actions', -1., 1.)
        _array(row['actual_rewards'], np.float64, (n,), 'actual_rewards', 0., 2.)
        _array(row['previous_action'], np.float64, (6,), 'previous_action', -1., 1.)
        for name in ('predictive_entropy', 'expected_entropy', 'fusion_lambda'):
            _array(row[name], np.float64, (n,), name, 0., 1.)
        _array(row['fusion_kl'], np.float64, (n,), 'fusion_kl', 0.)
        _array(row['cumulative_risk'], np.float64, (n,), 'cumulative_risk', 0., 100.)
        _array(row['energy'], np.float64, (8, 8), 'recovery_energy', 0.)
        _array(row['epistemic_disagreement'], np.float64, (n,), 'epistemic_disagreement', 0.)
        # Errors exist only for the eight observed recovery outcomes. No value
        # is fabricated for the missing interval, even in offline diagnostics.
        _array(row['temporal_prediction_error'], np.float64, (8,), 'observed_temporal_prediction_error', 0.)
        for name in ('risk_reset', 'fallback'):
            _array(row[name], np.bool_, (n,), name)
        weights = _array(row['candidate_weights'], np.float64, (n, 8), 'candidate_weights', 0., 1.)
        if not np.allclose(weights.sum(1), 1., rtol=0., atol=2e-12):
            raise ValueError('Candidate weights must be normalized')
        us, js, risk = [], [], 0.
        for t in range(n):
            u, j = uncertainty_from_sufficient(row['predictive_entropy'][t], row['expected_entropy'][t])
            us.append(u); js.append(j)
            if not row['availability'][t]:
                risk = accumulate_risk(risk, u)
            if not math.isclose(float(row['cumulative_risk'][t]), risk, rel_tol=0, abs_tol=2e-12):
                raise ValueError('Cumulative risk differs from the shared operator')
            if row['risk_reset'][t]:
                if not row['availability'][t]:
                    raise ValueError('Predicted frame cannot reset cumulative risk')
                risk = 0.
        if type(row['plans']) is not list or not row['plans']:
            raise ValueError('Actual multi-horizon planning measurements are missing')
        plan_stats, plan_ids = [], set()
        for plan in row['plans']:
            _validate_plan(plan, row, plan_ids)
            pu, pj = zip(*(uncertainty_from_sufficient(p, e) for p, e in zip(
                plan['predictive_entropy'], plan['expected_entropy'])))
            stats = rollout_statistics(plan['rewards'], plan['continuations'], np.array(pu, np.float64),
                np.array(pj, np.float64), plan['actions'], plan['previous_action'], plan['weights'],
                plan['epistemic_disagreement'])
            plan_stats.append(stats)
        if {s['horizon'] for s in plan_stats} != set(HORIZONS):
            raise ValueError('Measured H=1,3,5 planning statistics are all required')
        measured[(seed, gap)] = {'uncertainty': np.array(us), 'disagreement': np.array(js),
            'ess_ratio': 1./np.square(weights).sum(1)/8., 'plans': plan_stats,
            'slew': np.abs(np.diff(np.vstack((row['previous_action'], row['actions'])), axis=0)).max(1)}
    if seen != {(seed, gap) for seed in seeds for gap in GAPS}:
        raise ValueError('Incomplete deployment Cartesian product')
    return measured


def _validate_plan(plan, row, seen):
    fields = set(PLAN_ARRAY_FIELDS) | {'timestamp', 'proposal_id', 'horizon', 'state_sha256', 'rng_sha256', 'scratch_only'}
    if type(plan) is not dict or set(plan) != fields:
        raise ValueError('Planning raw diagnostic fields differ')
    if (type(plan['timestamp']) is not int or plan['timestamp'] not in row['timestamp']
            or type(plan['proposal_id']) is not int or plan['proposal_id'] < 0
            or type(plan['horizon']) is not int or plan['horizon'] not in HORIZONS):
        raise ValueError('Planning identity or horizon differs')
    identity = (plan['timestamp'], plan['proposal_id'], plan['horizon'])
    if identity in seen:
        raise ValueError('Duplicate planning diagnostic')
    seen.add(identity)
    if plan['scratch_only'] is not True or not _sha(plan['state_sha256']) or not _sha(plan['rng_sha256']):
        raise ValueError('Scratch state/RNG provenance missing')
    h = plan['horizon']
    for name in ('predictive_entropy', 'expected_entropy'):
        _array(plan[name], np.float64, (h,), name, 0., 1.)
    _array(plan['rewards'], np.float64, (h, 8), 'raw plan reward predictions')
    _array(plan['epistemic_disagreement'], np.float64, (h,), 'plan epistemic proxy', 0.)
    _array(plan['continuations'], np.float64, (h, 8), 'plan continuations', 0., 1.)
    _array(plan['actions'], np.float64, (h, 8, 6), 'plan actions', -1., 1.)
    _array(plan['weights'], np.float64, (8,), 'plan weights', 0., 1.)
    _array(plan['previous_action'], np.float64, (6,), 'plan previous_action', -1., 1.)
    row_index = int(plan['timestamp']) - 17
    if not np.array_equal(plan['previous_action'], row['actions'][row_index]):
        raise ValueError('Plan starts after its observation outcome; previous_action must be that incoming action')


def _q(values, level=.99):
    return float(np.quantile(np.asarray(values, np.float64), level, method='higher'))


def fit_reweighted_ess(energy, max_energy, temperature):
    """Fit-only counterfactual ESS under the fitted recovery weight operator.

    The uncalibrated pilot necessarily has different weights from a fitted
    controller. Do not mislabel its uniform weights as an observed ESS lower
    limit of one. Reapply the exact accept/softmax operation to RETAINED fit
    energies only. All-rejected rows are separately counted, not imputed.
    Independent closed-loop validation still measures actual deployed weights.
    """
    if not isinstance(energy, np.ndarray) or energy.ndim != 2 or energy.shape[1] != 8:
        raise ValueError('Expected retained fit energies[N,8]')
    _array(energy, np.float64, energy.shape, 'fit recovery energy', 0.)
    _scalar(max_energy, 'fitted absolute energy bound', 0.)
    _scalar(temperature, 'fitted positive temperature', 1e-300)
    ess, rejected = [], 0
    for values in energy:
        accepted = values <= max_energy
        if not accepted.any():
            rejected += 1
            continue
        logits = np.full(8, -np.inf)
        logits[accepted] = -(values[accepted]-values[accepted].min()) / temperature
        weights = np.exp(logits)
        weights /= weights.sum()
        ess.append(float(1. / np.square(weights).sum() / 8.))
    if not ess:
        raise ValueError('Fit set provides no absolutely accepted candidate distribution')
    return np.asarray(ess, np.float64), rejected


def fit_candidate(rows, context):
    """Fit once on fit-only data; result is never a production authorization."""
    metrics = _validate_rows(rows, context, 'fit')
    thresholds, horizon_limits, ess_fits = {}, {}, {}
    for gap in GAPS:
        selected = [r for r in rows if r['gap'] == gap]
        m = [metrics[(r['seed'], gap)] for r in selected]
        concat = lambda name: np.concatenate([r[name] for r in m])
        energy = np.concatenate([r['energy'] for r in selected])
        # Temperature is a structural positive scale when measured spread is
        # zero; this is declared rather than pretending it was learned.
        spread = _q((energy-energy.min(1, keepdims=True)).mean(1))
        max_energy, temperature = _q(energy.min(1)), max(spread, 1e-6)
        reweighted_ess, rejected = fit_reweighted_ess(energy, max_energy, temperature)
        ess_fits[str(gap)] = {'rule': 'fit_energy_reweight_once_before_holdout',
            'distribution_count': len(reweighted_ess), 'all_rejected_count': rejected,
            'retained_pilot_weights_unchanged': True,
            'counterfactual_fit_ess_is_not_observed_closed_loop_ess': True}
        thresholds[str(gap)] = {'max_energy': max_energy,
            'max_uncertainty': _q(concat('uncertainty')),
            'max_disagreement': _q(concat('disagreement')),
            'max_epistemic_disagreement': _q(np.concatenate([r['epistemic_disagreement'] for r in selected])),
            'max_temporal_prediction_error': _q(np.concatenate([r['temporal_prediction_error'] for r in selected])),
            'max_cumulative_risk': _q(np.concatenate([r['cumulative_risk'] for r in selected])),
            'min_ess_ratio': _q(reweighted_ess, .05),
            'temperature': temperature, 'temperature_floor_engineering': 1e-6,
            **STRUCTURAL_CAPS, 'planning_horizon': 5,
            'reward_weight': 1., 'uncertainty_weight': .25, 'termination_weight': 1.,
            'slew_weight': .10, 'disagreement_weight': .25}
        horizon_limits[str(gap)] = {}
        for h in HORIZONS:
            plans = [p for item in m for p in item['plans'] if p['horizon'] == h]
            horizon_limits[str(gap)][str(h)] = {key: _q([p[key] for p in plans])
                                                for key in ('uncertainty', 'disagreement', 'termination_risk', 'epistemic_disagreement')}
        thresholds[str(gap)]['max_termination_risk'] = horizon_limits[str(gap)]['5']['termination_risk']
    candidate = {'schema': SCHEMA, 'schema_version': 2, 'measurement_protocol': MEASUREMENT_PROTOCOL,
        'context': _context(context), 'fit_seeds': list(FIT_SEEDS), 'validation_seeds': list(VALIDATION_SEEDS),
        'fit_attempt': 1, 'fit_records_sha256': records_sha256(rows),
        'thresholds_by_gap': thresholds, 'planning_limits_by_gap_horizon': horizon_limits,
        'ess_fit_derivation': ess_fits,
        'coverage_revision': copy.deepcopy(COVERAGE_REVISION),
        'planning_operator_version': OPERATOR_VERSION, 'utility_weights': copy.deepcopy(UTILITY_WEIGHTS),
        'reward_projection': REWARD_PROJECTION,
        'active': False, 'physical_safety_proven': False,
        'threshold_rule': 'fit_Q99_higher_upper_Q05_higher_fit_reweighted_ESS_once',
        'structural_caps_are_fitted': False,
        'action_limits_are_environment_bounds_not_safety_calibration': True,
        'planning_horizon_rule': 'registered_max5_with_independent_horizon_limits',
        'epistemic_available': True,
        'epistemic_scope': 'independently_initialized_bootstrap_prediction_heads_proxy_not_full_Dreamer_ensemble_or_Bayesian_posterior'}
    candidate['sha256'] = _digest(candidate)
    return candidate


def _candidate(candidate, context):
    if not isinstance(candidate, dict) or candidate.get('sha256') != _digest({k:v for k,v in candidate.items() if k != 'sha256'}):
        raise ValueError('Candidate commitment hash mismatch')
    if candidate.get('schema') != SCHEMA or candidate.get('active') is not False:
        raise ValueError('Candidate is not a frozen v2 engineering proposal')
    _equal(candidate.get('context'), context, 'candidate context')


def validate_candidate_for_calibration(candidate, context):
    """Validate an inactive fitted proposal for calibration-only closed-loop use.

    This does not return VerifiedDeployment and cannot authorize training.
    The caller must bind the exact candidate in its validation claim, retain
    all traces and never tune it after seeing heldout outcomes. Production
    loading also recomputes the candidate from independently bound fit rows.
    """
    _context(context)
    _candidate(candidate, context)
    if (type(candidate.get('fit_attempt')) is not int or candidate['fit_attempt'] != 1
            or type(candidate.get('schema_version')) is not int or candidate['schema_version'] != 2
            or candidate.get('fit_seeds') != list(FIT_SEEDS)
            or candidate.get('validation_seeds') != list(VALIDATION_SEEDS)
            or candidate.get('planning_operator_version') != OPERATOR_VERSION
            or candidate.get('reward_projection') != REWARD_PROJECTION
            or candidate.get('coverage_revision') != COVERAGE_REVISION
            or candidate.get('utility_weights') != UTILITY_WEIGHTS):
        raise ValueError('Candidate does not match the registered calibration-only protocol')
    if set(candidate.get('thresholds_by_gap', {})) != {str(g) for g in GAPS}:
        raise ValueError('Candidate has incomplete gap thresholds')
    if set(candidate.get('planning_limits_by_gap_horizon', {})) != {str(g) for g in GAPS}:
        raise ValueError('Candidate has incomplete planning thresholds')
    from .belief import validate_thresholds
    validate_thresholds(candidate['thresholds_by_gap'])
    for gap in GAPS:
        thresholds = candidate['thresholds_by_gap'][str(gap)]
        for name in ('max_epistemic_disagreement', 'max_temporal_prediction_error'):
            _scalar(thresholds.get(name), name, 0.)
        for name, value in STRUCTURAL_CAPS.items():
            if thresholds.get(name) != value:
                raise ValueError('Candidate changed registered structural bounds')
        horizons = candidate['planning_limits_by_gap_horizon'][str(gap)]
        if set(horizons) != {str(h) for h in HORIZONS}:
            raise ValueError('Candidate planning horizon set differs')
        for limits in horizons.values():
            if set(limits) != {'uncertainty', 'disagreement', 'termination_risk', 'epistemic_disagreement'}:
                raise ValueError('Candidate planning limit fields differ')
            for name, value in limits.items():
                _scalar(value, name, 0., None if name == 'epistemic_disagreement' else 1.)
    return True


def evaluate_candidate(candidate, rows, context):
    """Holdout measurements cannot refit a threshold; in-memory result inactive."""
    validate_candidate_for_calibration(candidate, context)
    metrics = _validate_rows(rows, context, 'validation', candidate['sha256'])
    cells, structural_passed = {}, True
    for row in rows:
        seed, gap = row['seed'], row['gap']
        m, limits = metrics[(seed, gap)], candidate['thresholds_by_gap'][str(gap)]
        coverage = {'energy_upper': float(np.mean(row['energy'].min(1) <= limits['max_energy'])),
            'uncertainty_upper': float(np.mean(m['uncertainty'] <= limits['max_uncertainty'])),
            'disagreement_upper': float(np.mean(m['disagreement'] <= limits['max_disagreement'])),
            'epistemic_disagreement_upper': float(np.mean(row['epistemic_disagreement'] <= limits['max_epistemic_disagreement'])),
            'temporal_prediction_error_upper': float(np.mean(row['temporal_prediction_error'] <= limits['max_temporal_prediction_error'])),
            'cumulative_risk_upper': float(np.mean(row['cumulative_risk'] <= limits['max_cumulative_risk'])),
            'ess_lower': float(np.mean(m['ess_ratio'] >= limits['min_ess_ratio']))}
        for h in HORIZONS:
            plans = [p for p in m['plans'] if p['horizon'] == h]
            hs = candidate['planning_limits_by_gap_horizon'][str(gap)][str(h)]
            for name in ('uncertainty', 'disagreement', 'termination_risk', 'epistemic_disagreement'):
                coverage[f'plan_h{h}_{name}_upper'] = float(np.mean([p[name] <= hs[name] for p in plans]))
        invariants = {'fusion_lambda_cap': bool(np.all(row['fusion_lambda'] <= STRUCTURAL_CAPS['lambda_max'])),
                      'fusion_kl_cap': bool(np.all(row['fusion_kl'] <= STRUCTURAL_CAPS['kl_limit'])),
                      'action_amplitude_cap': bool(np.all(np.abs(row['actions']) <= 1.)),
                      'action_slew_cap': bool(np.all(m['slew'] <= 2.)),
                      'risk_reset_has_visible_evidence': bool(np.all(~row['risk_reset'] | row['availability']))}
        # A reset must actually have a candidate accepted by the fixed energy gate.
        resets = row['risk_reset'][gap:]
        invariants['risk_reset_has_qualified_energy'] = bool(np.all(~resets | (row['energy'].min(1) <= limits['max_energy'])))
        for name, values, limit, is_lower in (
            ('uncertainty', m['uncertainty'][gap:], limits['max_uncertainty'], False),
            ('disagreement', m['disagreement'][gap:], limits['max_disagreement'], False),
            ('epistemic', row['epistemic_disagreement'][gap:], limits['max_epistemic_disagreement'], False),
            ('temporal_error', row['temporal_prediction_error'], limits['max_temporal_prediction_error'], False),
            ('ess', m['ess_ratio'][gap:], limits['min_ess_ratio'], True)):
            qualified = values >= limit if is_lower else values <= limit
            invariants['risk_reset_has_qualified_' + name] = bool(np.all(~resets | qualified))
        coverage_failures = [key for key, value in coverage.items() if value < (LOWER_COVERAGE if key == 'ess_lower' else UPPER_COVERAGE)]
        invariant_failures = [key for key, value in invariants.items() if not value]
        failures = coverage_failures + invariant_failures
        run, longest = 0, 0
        for value in row['actual_rewards']/2.:
            run = run+1 if value < .2 else 0
            longest = max(longest, run)
        cells[f'{seed}:{gap}'] = {'coverage': coverage, 'invariants': invariants,
            'failed_checks': failures, 'passed': not failures,
            'passed_scope': 'per_seed_diagnostic_not_overall_acceptance',
            'coverage_diagnostic_passed': not coverage_failures,
            'coverage_diagnostic_failed_checks': coverage_failures,
            'structural_invariants_passed': not invariant_failures,
            'normalized_actual_return': float(row['actual_rewards'].mean()/2.),
            'fallback_actions': int(row['fallback'].sum()),
            'longest_low_reward_run': longest, 'functional_failure_20_actions': bool(longest >= 20),
            'functional_failure_is_physical_safety': False}
        cells[f'{seed}:{gap}']['planning_reward_out_of_support_fraction'] = float(np.mean([
            p['reward_out_of_support_fraction'] for p in m['plans']]))
        structural_passed = structural_passed and not invariant_failures
    by_gap = {}
    for gap in GAPS:
        selected = [cells[f'{seed}:{gap}'] for seed in VALIDATION_SEEDS]
        # Equal seed weighting matters when distinct seeds measure different
        # numbers of planning proposals. Do not pool raw proposal counts.
        coverage = {name: float(np.mean([cell['coverage'][name] for cell in selected]))
                    for name in selected[0]['coverage']}
        failures = [name for name, value in coverage.items()
                    if value < (LOWER_COVERAGE if name == 'ess_lower' else UPPER_COVERAGE)]
        by_gap[str(gap)] = {'coverage': coverage, 'seed_count': len(selected),
            'seed_weights': [1./len(selected)]*len(selected),
            'failed_checks': failures, 'coverage_passed': not failures,
            'all_structural_invariants_passed': all(cell['structural_invariants_passed'] for cell in selected)}
    all_passed = structural_passed and all(item['coverage_passed'] for item in by_gap.values())
    return {'schema': SCHEMA, 'candidate_sha256': candidate['sha256'],
        'validation_records_sha256': records_sha256(rows), 'validation_attempt': 1,
        'validation': {'passed': bool(all_passed), 'per_cell': cells,
                       'by_gap': by_gap, 'all_structural_invariants_passed': bool(structural_passed),
                       'coverage_revision': copy.deepcopy(COVERAGE_REVISION),
                       'thresholds_changed_using_holdout': False,
                       'coverage_scope': 'empirical_equal_seed_per_gap_engineering_gate_not_IID_guarantee'},
        'active': False, 'status': 'FILE_BACKED_PROVENANCE_REQUIRED',
        'physical_safety_proven': False, 'epistemic_available': True,
        'epistemic_scope': candidate['epistemic_scope']}


def claim_phase(registry_directory, context, phase, candidate=None):
    """Exclusive per-model attempt claim; preserve claims after any failure.

    The runner must use one registered persistent directory, not evade the
    once-only policy by choosing a new directory after looking at holdout.
    """
    _context(context)
    if phase not in ('fit', 'validation'):
        raise ValueError('Unknown calibration phase')
    if phase == 'validation':
        validate_candidate_for_calibration(candidate, context)
    elif candidate is not None:
        raise ValueError('Fit cannot receive a previously fitted candidate')
    root = Path(registry_directory)
    root.mkdir(parents=True, exist_ok=True)
    identity = _digest({k: v for k, v in context.items() if k != 'measurement_kind'})
    path = root / f'{identity}.{phase}.claim.json'
    record = {'schema': SCHEMA, 'context': _context(context), 'phase': phase,
              'attempt': 1, 'event_clock': EVENT_CLOCK, 'claimed_ns': time.perf_counter_ns(),
              'candidate_sha256': None if candidate is None else candidate['sha256']}
    with path.open('xb') as stream:
        stream.write(canonical_json(record))
    return path


def _read_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('Duplicate JSON field')
            result[key] = value
        return result
    def invalid(value):
        raise ValueError('Nonfinite JSON value: ' + value)
    return json.loads(Path(path).read_text(encoding='utf8'), object_pairs_hook=pairs, parse_constant=invalid)


def file_descriptor(path, root):
    """Bind relative file bytes, not an unverified producer status flag."""
    root, path = Path(root).resolve(), Path(path).resolve()
    relative = path.relative_to(root)
    if not path.is_file():
        raise ValueError('Evidence file missing')
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return {'path': relative.as_posix(), 'sha256': h.hexdigest(), 'size': path.stat().st_size}


def _verified_file(root, descriptor):
    if (type(descriptor) is not dict or set(descriptor) != {'path', 'sha256', 'size'}
            or type(descriptor['path']) is not str or not _sha(descriptor['sha256'])
            or type(descriptor['size']) is not int or descriptor['size'] < 0):
        raise ValueError('Malformed external file descriptor')
    base = Path(root).resolve()
    candidate = Path(descriptor['path'])
    if candidate.is_absolute() or '..' in candidate.parts or '\\' in descriptor['path']:
        raise ValueError('Evidence path traversal is forbidden')
    path = (base/candidate).resolve()
    try:
        path.relative_to(base)
    except ValueError as exc:
        raise ValueError('Evidence file escapes bundle root') from exc
    if file_descriptor(path, base) != descriptor:
        raise ValueError('Evidence SHA/size/path mismatch: ' + descriptor['path'])
    return path


def write_measurement_window(row, directory):
    """Write compact raw arrays without overwriting; not a validation shortcut."""
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    name = f"{row['phase']}_seed{row['seed']}_gap{row['gap']}"
    meta_path, array_path = root/(name+'.json'), root/(name+'.npz')
    if meta_path.exists() or array_path.exists():
        raise FileExistsError('Measured window already exists; retain the prior attempt')
    arrays = {name: row[name] for name in ARRAY_FIELDS}
    metadata = {name: copy.deepcopy(value) for name, value in row.items()
                if name not in ARRAY_FIELDS and name != 'plans'}
    metadata['plans'] = []
    for index, plan in enumerate(row['plans']):
        item = {name: value for name, value in plan.items() if name not in PLAN_ARRAY_FIELDS}
        for name in PLAN_ARRAY_FIELDS:
            arrays[f'plan_{index}_{name}'] = plan[name]
        metadata['plans'].append(item)
    with array_path.open('xb') as stream:
        np.savez_compressed(stream, **arrays)
    with meta_path.open('xb') as stream:
        stream.write(canonical_json(metadata))
    return {'metadata': file_descriptor(meta_path, root), 'arrays': file_descriptor(array_path, root)}


def _load_window(root, descriptor):
    if type(descriptor) is not dict or set(descriptor) != {'metadata', 'arrays'}:
        raise ValueError('Window file descriptor differs')
    row = _read_json(_verified_file(root, descriptor['metadata']))
    archive = _verified_file(root, descriptor['arrays'])
    if type(row.get('plans')) is not list:
        raise ValueError('Missing planning metadata')
    with np.load(archive, allow_pickle=False) as data:
        expected = set(ARRAY_FIELDS) | {f'plan_{i}_{name}' for i in range(len(row['plans'])) for name in PLAN_ARRAY_FIELDS}
        if set(data.files) != expected:
            raise ValueError('NPZ measured fields are missing or unexpected')
        for name in ARRAY_FIELDS:
            row[name] = data[name].copy()
        for i, plan in enumerate(row['plans']):
            for name in PLAN_ARRAY_FIELDS:
                plan[name] = data[f'plan_{i}_{name}'].copy()
    return row


def _frozen_evidence(record, context, fit_digest, validation_digest):
    required = {'schema', 'context', 'device', 'gpu', 'model_updates', 'before', 'after',
                'fit_records_sha256', 'validation_records_sha256'}
    if type(record) is not dict or set(record) != required or record['schema'] != SCHEMA:
        raise ValueError('Missing independent frozen-state evidence')
    _equal(record['context'], context, 'freeze context')
    if record['fit_records_sha256'] != fit_digest or record['validation_records_sha256'] != validation_digest:
        raise ValueError('Freeze evidence does not bind current measurement data')
    if type(record['model_updates']) is not int or record['model_updates'] != 0:
        raise ValueError('Calibration changed learner update counters')
    for side in ('before', 'after'):
        if (type(record[side]) is not dict or set(record[side]) != set(FREEZE_COMPONENTS)
                or any(not _sha(value) for value in record[side].values())):
            raise ValueError('Incomplete frozen model/optimizer/normalizer/counters/RNG hashes')
    _equal(record['before'], record['after'], 'all frozen content hashes')
    if record['before']['model'] != context['model_sha256']:
        raise ValueError('Frozen measured model differs from deployed model')
    if record['before']['epistemic_heads'] != context['epistemic_heads_sha256']:
        raise ValueError('Frozen independent prediction heads differ from calibrated heads')
    if context['measurement_kind'] == 'actual_gpu_closed_loop':
        if record['device'] != 'cuda' or type(record['gpu']) is not str or not record['gpu'].strip():
            raise ValueError('Actual GPU provenance is missing')
    elif record['device'] != 'synthetic':
        raise ValueError('Synthetic fixture cannot claim actual GPU measurement')


def _ledger(record, claims, context, candidate, audit):
    keys = {'schema', 'context', 'fit_attempts', 'validation_attempts', 'holdout_threshold_revisions',
            'candidate_sha256', 'fit_records_sha256', 'validation_records_sha256',
            'fit_completed_ns', 'candidate_sealed_ns', 'validation_started_ns', 'validation_completed_ns',
            'event_clock'}
    if type(record) is not dict or set(record) != keys or record['schema'] != SCHEMA:
        raise ValueError('Incomplete once-only phase ledger')
    _equal(record['context'], context, 'phase ledger context')
    # These fields are causal event times in one host boot, never UTC times.
    # Wall-clock resolution/adjustments cannot establish strict phase ordering.
    # Some Windows Python builds also implement monotonic_ns with the coarse
    # GetTickCount64 clock. perf_counter_ns uses monotonic QPC at high resolution.
    if record['event_clock'] != EVENT_CLOCK:
        raise ValueError('Phase ledger requires the declared monotonic event clock')
    for name, expected in (('fit_attempts', 1), ('validation_attempts', 1), ('holdout_threshold_revisions', 0)):
        if type(record[name]) is not int or record[name] != expected:
            raise ValueError('Refitting or reused holdout attempt is forbidden')
    if record['candidate_sha256'] != candidate['sha256'] or record['fit_records_sha256'] != candidate['fit_records_sha256'] or record['validation_records_sha256'] != audit['validation_records_sha256']:
        raise ValueError('Phase ledger identity differs')
    times = [record[name] for name in ('fit_completed_ns', 'candidate_sealed_ns', 'validation_started_ns', 'validation_completed_ns')]
    if any(type(value) is not int or value < 0 for value in times) or not times[0] <= times[1] < times[2] <= times[3]:
        raise ValueError('Candidate must be sealed before independent validation begins')
    for phase, claim in claims.items():
        if type(claim) is not dict or set(claim) != {'schema','context','phase','attempt','claimed_ns','candidate_sha256','event_clock'}:
            raise ValueError('Missing exclusive phase claim')
        _equal(claim['context'], context, 'phase claim context')
        if (claim['schema'] != SCHEMA or claim['phase'] != phase or type(claim['attempt']) is not int
                or claim['attempt'] != 1 or type(claim['claimed_ns']) is not int
                or claim['event_clock'] != EVENT_CLOCK):
            raise ValueError('Invalid once-only phase claim')
        expected = None if phase == 'fit' else candidate['sha256']
        if claim['candidate_sha256'] != expected or not 0 <= claim['claimed_ns'] <= (times[0] if phase == 'fit' else times[2]):
            raise ValueError('Phase claim must precede the bound measurement')
        if phase == 'validation' and claim['claimed_ns'] < times[1]:
            raise ValueError('Validation may only be claimed after the candidate is sealed')


def _replay_evidence(root, record, context, rows, anchors, traces):
    if type(record) is not dict or set(record) != {'schema', 'context', 'cells'} or record['schema'] != SCHEMA:
        raise ValueError('Independent fresh-process replay evidence missing')
    _equal(record['context'], context, 'independent replay context')
    expected = {(r['phase'], r['seed'], r['gap']): r for r in rows}
    if type(record['cells']) is not list or len(record['cells']) != len(expected):
        raise ValueError('Independent replay must cover every recorded window')
    seen = set()
    for cell in record['cells']:
        keys = {'phase', 'seed', 'gap', 'replay_anchor_sha256', 'reference_trace_sha256',
                'replayed_trace', 'reference_process_id', 'replay_process_id'}
        if type(cell) is not dict or set(cell) != keys:
            raise ValueError('Independent replay cell fields differ')
        if type(cell['seed']) is not int or type(cell['gap']) is not int:
            raise ValueError('Replay seed/gap must be strict integers')
        identity = (cell['phase'], cell['seed'], cell['gap'])
        if identity not in expected or identity in seen:
            raise ValueError('Missing, duplicate or wrong independent replay cell')
        seen.add(identity); row = expected[identity]
        if cell['replay_anchor_sha256'] != row['replay_anchor_sha256'] or cell['reference_trace_sha256'] != row['trace_sha256']:
            raise ValueError('Replay anchor/raw trace identity mismatch')
        if row['replay_anchor_sha256'] not in anchors or row['trace_sha256'] not in traces:
            raise ValueError('Required checkpoint/RNG anchor/raw trace not retained')
        trace = _read_json(traces[row['trace_sha256']])
        trace_fields = {'schema','context','phase','seed','gap','execution_mode',
            'diagnostics_sha256','replay_anchor_sha256','observation_sha256',
            'controller_state_sha256','rng_sha256'}
        if type(trace) is not dict or set(trace) != trace_fields or trace['schema'] != SCHEMA:
            raise ValueError('Raw replay trace fields differ')
        _equal(trace['context'],context,'raw trace context')
        if (trace['execution_mode']!='closed_loop' or trace['phase']!=row['phase']
                or type(trace['seed']) is not int or trace['seed']!=row['seed']
                or type(trace['gap']) is not int or trace['gap']!=row['gap']
                or trace['replay_anchor_sha256']!=row['replay_anchor_sha256']
                or trace['diagnostics_sha256']!=diagnostic_content_sha256(row)):
            raise ValueError('Raw trace does not bind the retained closed-loop diagnostics')
        for name in ('observation_sha256','controller_state_sha256','rng_sha256'):
            if type(trace[name]) is not list or len(trace[name])!=row['gap']+8 or any(not _sha(v) for v in trace[name]):
                raise ValueError('Per-observation/RNG/state replay evidence is incomplete')
        for key in ('reference_process_id', 'replay_process_id'):
            if type(cell[key]) is not int or cell[key] <= 0:
                raise ValueError('Independent process identity missing')
        if cell['reference_process_id'] == cell['replay_process_id']:
            raise ValueError('Replay was not a fresh independent process')
        _verified_file(root, cell['replayed_trace'])
        if cell['replayed_trace']['sha256'] != row['trace_sha256']:
            raise ValueError('Independent replay content does not match retained diagnostics')


def load_deployment_calibration(directory, expected_manifest_sha256, context, *, require_active=True):
    """Recompute all acceptance from externally bound file bytes.

    No field such as manifest.passed, artifact.active or producer validation
    result is read. No pickle or executable model is loaded by this validator.
    """
    _context(context)
    if not _sha(expected_manifest_sha256):
        raise ValueError('Trusted external manifest SHA is mandatory')
    root = Path(directory).resolve()
    manifest_path = root/'deployment_manifest.json'
    if file_descriptor(manifest_path, root)['sha256'] != expected_manifest_sha256:
        raise ValueError('External deployment manifest byte digest differs')
    manifest = _read_json(manifest_path)
    keys = {'schema', 'measurement_protocol', 'context', 'candidate', 'windows', 'freeze_evidence',
            'replay_evidence', 'phase_ledger', 'phase_claims', 'replay_anchors', 'raw_traces'}
    if type(manifest) is not dict or set(manifest) != keys or manifest['schema'] != SCHEMA or manifest['measurement_protocol'] != MEASUREMENT_PROTOCOL:
        raise ValueError('Not a complete v2 deployment evidence manifest')
    _equal(manifest['context'], context, 'external deployment context')
    if type(manifest['windows']) is not list or len(manifest['windows']) != 120:
        raise ValueError('All 120 fit/validation windows must be retained')
    rows = [_load_window(root, item) for item in manifest['windows']]
    fit = [r for r in rows if r['phase'] == 'fit']
    validation = [r for r in rows if r['phase'] == 'validation']
    candidate = fit_candidate(fit, context)
    _equal(_read_json(_verified_file(root, manifest['candidate'])), candidate, 'recomputed fit-only candidate')
    audit = evaluate_candidate(candidate, validation, context)
    frozen = _read_json(_verified_file(root, manifest['freeze_evidence']))
    _frozen_evidence(frozen, context, candidate['fit_records_sha256'], audit['validation_records_sha256'])
    if type(manifest['phase_claims']) is not dict or set(manifest['phase_claims']) != {'fit', 'validation'}:
        raise ValueError('Both exclusive phase claims are required')
    claims = {phase: _read_json(_verified_file(root, item)) for phase, item in manifest['phase_claims'].items()}
    _ledger(_read_json(_verified_file(root, manifest['phase_ledger'])), claims, context, candidate, audit)
    references = {}
    for name in ('replay_anchors', 'raw_traces'):
        if type(manifest[name]) is not list or not manifest[name]:
            raise ValueError('Replay anchors and traces must be retained')
        for item in manifest[name]:
            _verified_file(root, item)
        references[name] = {item['sha256']: _verified_file(root,item) for item in manifest[name]}
    _replay_evidence(root, _read_json(_verified_file(root, manifest['replay_evidence'])), context, rows,
                     references['replay_anchors'], references['raw_traces'])
    active = context['measurement_kind'] == 'actual_gpu_closed_loop' and audit['validation']['passed']
    result = {**candidate, **audit, 'schema_version': 2, 'frozen': True, 'active': bool(active),
              'context': copy.deepcopy(context), 'manifest_sha256': expected_manifest_sha256,
              'status': 'ENGINEERING_CALIBRATED' if active else 'DEPLOYMENT_REJECTED',
              'provenance': {'file_bytes_verified': True, 'fit_and_holdout_recomputed': True,
                             'independent_replay_transcripts_verified': True,
                             'hardware_authenticated_by_loader': False},
              'inactive_reasons': ([] if active else
                   (['synthetic_fixture_not_production'] if context['measurement_kind'] == 'synthetic_test_fixture' else [])
                   + ([] if audit['validation']['passed'] else ['independent_closed_loop_validation_failed']))}
    result.pop('sha256', None)
    result['sha256'] = _digest(result)
    if require_active and not active:
        raise ValueError('Deployment not active: ' + ','.join(result['inactive_reasons']))
    return VerifiedDeployment(result, _token=_FILE_VERIFIED_TOKEN)


def training_calibration_gate(actions_completed, calibration=None, context=None):
    """Shared 5000-action random prefill; first learned boundary is 5500.

    The interval 5000..5499 is an explicitly registered bootstrap episode,
    not permission to run indefinitely in UNCAlIBRATED mode. The runner must
    block before issuing action 5501 when no matched calibration is active.
    """
    if type(actions_completed) is not int or actions_completed < 0:
        raise ValueError('Training gate requires a strict action count')
    if actions_completed < 5000:
        return 'SHARED_RANDOM_PREFILL'
    if actions_completed < 5500:
        return 'REGISTERED_FIRST_LEARNED_BOOTSTRAP_EPISODE'
    try:
        verify_deployment_handle(calibration,context)
    except (ValueError,TypeError) as exc:
        raise RuntimeError('BLOCKED: first post-learning boundary requires matched deployment calibration') from exc
    return 'MATCHED_CALIBRATION_REQUIRED_FOR_CURRENT_PINNED_VERSION'
