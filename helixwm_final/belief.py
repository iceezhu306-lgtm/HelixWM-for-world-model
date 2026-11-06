"""Non-learning categorical belief and calibration contracts for Final-C1.

One-model particle disagreement is not epistemic ensemble uncertainty.
Calibration and control import these exact functions to keep units aligned.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from numbers import Real

import numpy as np

GAP_BINS = (1, 5, 10, 20, 50, 100)
THRESHOLD_FIELDS = (
    'max_energy', 'max_uncertainty', 'max_disagreement', 'max_cumulative_risk', 'lambda_max',
    'kl_limit', 'min_ess_ratio', 'planning_horizon', 'action_amplitude',
    'slew_limit', 'reward_weight', 'uncertainty_weight', 'termination_weight',
    'slew_weight', 'temperature', 'max_termination_risk')


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      allow_nan=False, ensure_ascii=True).encode('utf8')


def calibration_sha256(record):
    return hashlib.sha256(canonical_json({k: v for k, v in record.items()
                                         if k != 'sha256'})).hexdigest()


def _sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def validate_thresholds(table):
    if not isinstance(table, dict) or set(table) != {str(g) for g in GAP_BINS}:
        raise ValueError('Calibration requires the six registered gap bins')
    for row in table.values():
        if not isinstance(row, dict) or any(k not in row for k in THRESHOLD_FIELDS):
            raise ValueError('Incomplete calibrated thresholds')
        for key in THRESHOLD_FIELDS:
            value = row[key]
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
                raise ValueError('Calibration thresholds must be finite numbers')
            if value < 0:
                raise ValueError('Calibration thresholds cannot be negative')
        for key in ('max_uncertainty', 'max_disagreement', 'lambda_max', 'min_ess_ratio', 'action_amplitude', 'max_termination_risk'):
            if row[key] > 1:
                raise ValueError('Normalized threshold outside [0,1]')
        if row['lambda_max'] > 0.5:
            raise ValueError('Final-C1 cannot give Dream a majority fusion weight')
        if row['max_cumulative_risk'] > 100 or row['slew_limit'] > 2:
            raise ValueError('Risk/slew threshold outside defined units')
        if type(row['planning_horizon']) is not int or row['planning_horizon'] not in (1, 3, 5):
            raise ValueError('Planning horizon must be 1, 3, or 5')
        if row['temperature'] <= 0:
            raise ValueError('Energy temperature must be positive')
    return copy.deepcopy(table)


def _validate_calibration_structure(record, source_sha256, model_version, model_sha256=None):
    """Validate structure only; this is NOT production deployment approval."""
    if record is None:
        return False, 'calibration_missing'
    try:
        if not isinstance(record, dict) or record.get('schema_version') != 1 or record.get('frozen') is not True:
            return False, 'calibration_schema_or_freeze'
        if not _sha(record.get('sha256')) or record['sha256'] != calibration_sha256(record):
            return False, 'calibration_hash_mismatch'
        if record.get('active') is not True:
            return False, 'calibration_inactive'
        if not isinstance(record.get('validation'), dict) or record['validation'].get('passed') is not True:
            return False, 'calibration_validation_failed'
        if not _sha(record.get('bank_manifest_sha256')):
            return False, 'calibration_bank_binding_missing'
        if not _sha(source_sha256) or record.get('source_sha256') != source_sha256:
            return False, 'calibration_source_mismatch'
        if type(record.get('model_version')) is not int or record['model_version'] != model_version:
            return False, 'calibration_version_mismatch'
        if not _sha(model_sha256):
            return False, 'model_binding_unavailable'
        if record.get('model_sha256') != model_sha256:
            return False, 'calibration_model_mismatch'
        if not isinstance(record.get('calibration_id'), str) or not record['calibration_id']:
            return False, 'calibration_id_missing'
        fit, valid = record.get('fit_seeds'), record.get('validation_seeds')
        if not all(isinstance(v, list) and len(v) > 0 and all(type(x) is int for x in v)
                   and len(v) == len(set(v)) for v in (fit, valid)):
            return False, 'calibration_seed_schema'
        if set(fit) & set(valid) or any(60000 <= x <= 60039 for x in fit + valid):
            return False, 'calibration_seed_overlap'
        validate_thresholds(record.get('thresholds_by_gap'))
    except (TypeError, ValueError, OverflowError):
        return False, 'calibration_invalid_values'
    return True, 'calibrated'


def validate_calibration(record, source_sha256, model_version, model_sha256=None):
    """Require the full sealed-artifact contract, even for direct dict inputs.

    Current open-loop artifacts cannot deploy. CPU mechanism tests explicitly
    override this method in a test-only controller; real callers cannot enable
    fusion merely by constructing an active=True dictionary.
    """
    valid, reason = _validate_calibration_structure(record, source_sha256,
                                                  model_version, model_sha256)
    if not valid:
        return valid, reason
    try:
        from .calibration import IDENTITY_FIELDS, BINDING_ONLY_FIELDS, _validate_artifact
        binding = {key: record[key] for key in IDENTITY_FIELDS + BINDING_ONLY_FIELDS}
        _validate_artifact(record, binding, require_active=True)
    except (KeyError, TypeError, ValueError, OverflowError):
        return False, 'deployment_calibration_unverified'
    return True, 'calibrated'


def thresholds_for_gap(table, gap):
    if type(gap) is not int or gap < 0 or gap > 100:
        raise ValueError('Gap outside the registered 0..100 control range')
    return copy.deepcopy(table[str(next(g for g in GAP_BINS if g >= max(1, gap)))])


def probabilities(value):
    p = np.asarray(value, dtype=np.float64)
    if p.ndim != 3 or p.shape[0] < 1 or p.shape[1] < 1 or p.shape[2] < 2:
        raise ValueError('Categorical probabilities must be [B,S,C]')
    if not np.isfinite(p).all() or np.any(p < 0) or np.any(p > 1):
        raise ValueError('Invalid categorical probability values')
    sums = p.sum(axis=-1, keepdims=True)
    if not np.allclose(sums, 1.0, rtol=0, atol=2e-5):
        raise ValueError('Categorical probabilities must sum to one')
    return p / sums


def normalized_weights(value, count):
    w = np.full(count, 1.0/count) if value is None else np.asarray(value, np.float64)
    if w.shape != (count,) or not np.isfinite(w).all() or np.any(w < 0) or w.sum() <= 0:
        raise ValueError('Invalid candidate weights')
    return w / w.sum()


def uncertainty_metrics(probs, weights=None):
    p = probabilities(probs)
    w = normalized_weights(weights, len(p))
    marginal = np.einsum('k,ksc->sc', w, p)
    entropy = lambda a: -np.sum(a * np.log(np.maximum(a, 1e-300)), axis=-1)
    scale = math.log(p.shape[-1])
    expected = float(np.dot(w, entropy(p).mean(axis=-1)) / scale)
    predictive = float(entropy(marginal).mean() / scale)
    disagreement = max(0.0, predictive - expected)
    variance = float(np.einsum('k,ksc->sc', w, (p-marginal)**2).mean())
    return {'predictive_entropy': min(1.0, max(0.0, predictive)),
            'expected_entropy': min(1.0, max(0.0, expected)),
            'model_conditional_disagreement': disagreement,
            'latent_predictive_variance': variance,
            # JSD <= predictive entropy, so max(entropy,JSD) was redundant.
            # Keep entropy and disagreement as independently gated metrics.
            'uncertainty': min(1.0, max(0.0, predictive)),
            'epistemic_available': False}


def accumulate_risk(previous, uncertainty, max_gap=100):
    if not all(math.isfinite(float(v)) for v in (previous, uncertainty)) or previous < 0 or not 0 <= uncertainty <= 1:
        raise ValueError('Invalid cumulative-risk input')
    if type(max_gap) is not int or not 1 <= max_gap <= 100:
        raise ValueError('Invalid cumulative-risk bound')
    return min(float(max_gap), float(previous) + float(uncertainty))


def constrained_fusion(base, candidates, weights, lambda_max, kl_limit):
    """Fixed finite lambda grid, anchored to the Real categorical belief."""
    b, p = probabilities(base), probabilities(candidates)
    if b.shape[0] != 1 or b.shape[1:] != p.shape[1:]:
        raise ValueError('Belief fusion shape mismatch')
    if not 0 <= lambda_max <= .5 or not math.isfinite(kl_limit) or kl_limit < 0:
        raise ValueError('Invalid belief-fusion trust region')
    w = normalized_weights(weights, len(p))
    marginal = np.einsum('k,ksc->sc', w, p)[None]
    for fraction in (1.0, .5, .25, .125, 0.0):
        lam = float(lambda_max) * fraction
        mixed = (1-lam)*b + lam*marginal
        # Positive mass outside base support is infinite KL and must reject.
        if np.any((mixed > 0) & (b == 0)):
            kl = math.inf
        else:
            positive = mixed > 0
            terms = np.zeros_like(mixed)
            terms[positive] = mixed[positive]*np.log(mixed[positive]/b[positive])
            kl = max(0.0, float(terms.sum(axis=-1).mean()))
        if kl <= kl_limit:
            return mixed.astype(np.float32), {'lambda': lam, 'kl_to_real': kl}
    raise AssertionError('The zero-fusion baseline must always be feasible')


def sample_categorical(probs, rng):
    p = probabilities(probs)
    draw = rng.random(p.shape[:-1])
    indices = (draw[..., None] > np.cumsum(p, axis=-1)).sum(axis=-1)
    indices = np.minimum(indices, p.shape[-1]-1)
    return np.eye(p.shape[-1], dtype=np.float32)[indices]
