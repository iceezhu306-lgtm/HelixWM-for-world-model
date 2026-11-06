"""Shared, dimensioned planning statistics for control and calibration.

No learned reward-scale denominator is used. A zero-reward model is valid.
These are model predictions and engineering scores, not safety guarantees.
The host arrays deliberately retain every candidate at every planning depth.
"""
from __future__ import annotations

import math
import numpy as np

from .belief import uncertainty_metrics

HORIZONS = (1, 3, 5)
OPERATOR_VERSION = 'helixwm_final_planning_statistics_v2'
REWARD_PROJECTION = 'planning_only_project_to_registered_DMC_repeat2_support_0_2'
UTILITY_WEIGHTS = {'reward': 1.0, 'uncertainty': .25,
                   'disagreement': .25, 'termination': 1.0, 'slew': .10}


def _array(value, shape, name, lower=None, upper=None):
    if (not isinstance(value, np.ndarray) or value.dtype != np.float64
            or value.shape != shape or not np.isfinite(value).all()):
        raise ValueError('Expected finite float64 ' + name + ' ' + str(shape))
    if lower is not None and np.any(value < lower):
        raise ValueError(name + ' below its registered units')
    if upper is not None and np.any(value > upper):
        raise ValueError(name + ' above its registered units')
    return value


def _weights(weights, count):
    w = _array(weights, (count,), 'weights', 0., 1.)
    if not math.isclose(float(w.sum()), 1., rel_tol=0., abs_tol=2e-12):
        raise ValueError('Candidate weights must already be normalized')
    return w


def categorical_statistics(probs, weights):
    """Compute compact sufficient statistics before discarding transient probs.

    The retained distribution/RNG/checkpoint digest must permit independent
    replay. Sufficient statistics alone do not authenticate the model output.
    """
    if not isinstance(probs, np.ndarray) or probs.dtype != np.float64 or probs.ndim != 3:
        raise ValueError('Expected float64[K,S,C] categorical probabilities')
    k, s, c = probs.shape
    if not 1 <= k <= 8 or s < 1 or c < 2:
        raise ValueError('Invalid categorical dimensions')
    _array(probs, (k, s, c), 'categorical probabilities', 0., 1.)
    _weights(weights, k)
    if not np.allclose(probs.sum(-1), 1., rtol=0., atol=2e-12):
        raise ValueError('Categorical probabilities must already be normalized')
    metrics = uncertainty_metrics(probs, weights)
    return {**metrics, 'ess_ratio': float(1. / np.square(weights).sum() / k),
            'candidate_count': k, 'categorical_variables': s, 'categorical_classes': c}


def uncertainty_from_sufficient(predictive_entropy, expected_entropy):
    """Recompute normalized entropy and JSD from retained sufficient statistics."""
    values = (predictive_entropy, expected_entropy)
    if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (float, np.float64))
           or not math.isfinite(v) or not 0 <= v <= 1 for v in values):
        raise ValueError('Normalized entropies must be finite floating-point scalars')
    if expected_entropy > predictive_entropy + 2e-12:
        raise ValueError('Expected entropy exceeds predictive entropy')
    return float(predictive_entropy), max(0., float(predictive_entropy - expected_entropy))


def rollout_statistics(rewards, continuations, uncertainty, disagreement,
                       actions, previous_action, weights, epistemic_disagreement=None):
    """Aggregate float64 rewards/continuations[H,K], actions[H,K,6].

    Reward at depth h is weighted by survival BEFORE that transition. DMC's
    registered two-call action has reward units [0,2]. Learned reward heads may
    extrapolate outside that support. Retain those raw values, return and rate
    of extrapolation, and explicitly project ONLY the planning objective onto
    the known support. This changes neither model training nor evidence bytes.
    Only depth-zero action is executed; future actions remain scratch plans.
    """
    if not isinstance(rewards, np.ndarray) or rewards.ndim != 2:
        raise ValueError('Expected rewards[H,K]')
    h, k = rewards.shape
    if h not in HORIZONS or not 1 <= k <= 8:
        raise ValueError('Unregistered horizon or candidate count')
    _array(rewards, (h, k), 'raw predicted rewards')
    _array(continuations, (h, k), 'continuations', 0., 1.)
    _array(uncertainty, (h,), 'uncertainty', 0., 1.)
    _array(disagreement, (h,), 'disagreement', 0., 1.)
    if np.any(disagreement > uncertainty + 2e-12):
        raise ValueError('JSD cannot exceed predictive entropy')
    _array(actions, (h, k, 6), 'actions', -1., 1.)
    _array(previous_action, (6,), 'previous_action', -1., 1.)
    w = _weights(weights, k)
    if epistemic_disagreement is not None:
        _array(epistemic_disagreement, (h,), 'epistemic proxy disagreement', 0.)
    if not np.array_equal(actions[0], np.broadcast_to(actions[0, 0], (k, 6))):
        raise ValueError('One executed first action must be shared by all candidates')
    survival = np.ones(k, np.float64)
    returns = np.zeros(k, np.float64)
    raw_returns = np.zeros(k, np.float64)
    objective_rewards = np.clip(rewards, 0., 2.)
    for depth in range(h):
        raw_returns += survival * rewards[depth]
        returns += survival * objective_rewards[depth]
        survival *= continuations[depth]
    if not np.isfinite(raw_returns).all():
        raise ValueError('Raw predicted return overflowed despite finite individual predictions')
    previous = np.concatenate((np.broadcast_to(previous_action, (1, k, 6)), actions[:-1]), axis=0)
    delta = actions - previous
    first_slew = float(np.square(delta[0, 0]).mean() / 4.)
    path_slew = float(np.dot(w, np.square(delta).mean(axis=(0, 2))) / 4.)
    result = {'operator_version': OPERATOR_VERSION, 'horizon': h, 'candidate_count': k,
              'predicted_return': float(np.dot(w, returns)),
              'raw_predicted_return': float(np.dot(w, raw_returns)),
              'reward_projection': REWARD_PROJECTION,
              'reward_out_of_support_fraction': float(np.mean((rewards < 0.) | (rewards > 2.))),
              'reward_normalized': float(np.dot(w, returns) / (2. * h)),
              'termination_risk': float(np.dot(w, 1. - survival)),
              'uncertainty': float(uncertainty.max()), 'disagreement': float(disagreement.max()),
              'slew_normalized': first_slew, 'path_slew_normalized': path_slew,
              'action_amplitude': float(np.abs(actions[0, 0]).max()),
              'action_slew': float(np.abs(delta[0, 0]).max()),
              'ess_ratio': float(1. / np.square(w).sum() / k)}
    result['epistemic_disagreement'] = (None if epistemic_disagreement is None
                                       else float(epistemic_disagreement.max()))
    result['utility'] = utility(result)
    return result


def utility(stats):
    """Fixed dimensionless engineering objective, valid at zero reward."""
    terms = {'reward': stats['reward_normalized'], 'uncertainty': stats['uncertainty'],
             'disagreement': stats['disagreement'], 'termination': stats['termination_risk'],
             'slew': stats['slew_normalized']}
    if any(isinstance(v, bool) or not isinstance(v, (float, np.float64))
           or not math.isfinite(v) or not 0 <= v <= 1 for v in terms.values()):
        raise ValueError('Utility terms must use normalized [0,1] units')
    return float(sum((1 if name == 'reward' else -1) * UTILITY_WEIGHTS[name] * value
                     for name, value in terms.items()))


def proposal_diversity(first_actions):
    """Mean pairwise action mean-square /4; zero diversity is explicitly legal."""
    if not isinstance(first_actions, np.ndarray) or first_actions.ndim != 2:
        raise ValueError('Expected proposal actions[P,6]')
    p = len(first_actions)
    if p < 1:
        raise ValueError('At least one proposal is required')
    _array(first_actions, (p, 6), 'proposal actions', -1., 1.)
    values = [float(np.square(first_actions[i]-first_actions[j]).mean()/4.)
              for i in range(p) for j in range(i)]
    return float(np.mean(values)) if values else 0.


def plan_accepted(stats, limits):
    """Apply horizon-specific prediction gates, never a physical safety claim."""
    names = ('uncertainty', 'disagreement', 'termination_risk')
    if any(name not in limits for name in names):
        raise ValueError('Missing horizon-specific planning limits')
    for name in names:
        limit = limits[name]
        if isinstance(limit, bool) or not isinstance(limit, (float, np.float64)) or not math.isfinite(limit) or not 0 <= limit <= 1:
            raise ValueError('Invalid horizon-specific limit')
    if 'epistemic_disagreement' in limits:
        limit, value = limits['epistemic_disagreement'], stats.get('epistemic_disagreement')
        if (isinstance(limit, bool) or not isinstance(limit, (float, np.float64))
                or not math.isfinite(limit) or limit < 0):
            raise ValueError('Invalid independently fitted epistemic proxy limit')
        if (value is None or isinstance(value, bool) or not isinstance(value, (float, np.float64))
                or not math.isfinite(value) or value < 0):
            return False
        if value > limit:
            return False
    return all(stats[name] <= limits[name] for name in names)
