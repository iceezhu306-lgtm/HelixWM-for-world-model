"""CPU-only validation for the isolated final evaluator.

Arithmetic consistency is not file integrity. A bound cannot detect balanced
reward changes or every sub-bound perturbation; trusted SHA256 evidence remains
independent and exact. No training code or checkpoint identity is changed here.
"""
from __future__ import annotations

from fractions import Fraction
import hashlib
import math
from pathlib import Path

import numpy as np


def reward_sum_audit(rewards, episode_return):
    """Check binary64 reward reduction without changing either input.

    For round-to-nearest binary64 additions, gamma_(n-1) times the absolute
    sum bounds sequential reduction error (no overflow). Two reference ULPs
    conservatively cover fsum rounding, including its documented occasional
    double rounding. The absolute sum is inflated by the same allowance.
    n * min_subnormal also covers gradual-underflow absolute rounding error.
    The final bound is rounded upward using exact rational bound arithmetic.
    """
    if not isinstance(rewards, np.ndarray) or rewards.dtype != np.float64 or rewards.ndim != 1 or not rewards.size:
        raise ValueError('Rewards must be a nonempty one-dimensional float64 array')
    if not np.isfinite(rewards).all() or np.any(rewards < 0) or np.any(rewards > 2):
        raise ValueError('Rewards must be finite and within registered Walker bounds')
    if type(episode_return) not in (float, np.float64) or not math.isfinite(float(episode_return)):
        raise ValueError('Episode return must be a finite binary64 scalar')
    values = [float(value) for value in rewards]
    canonical = math.fsum(values)
    recorded = float(episode_return)
    n = len(values)
    if recorded < 0 or recorded > 2 * n:
        raise ValueError('Episode return falls outside registered aggregate reward bounds')
    if n - 1 >= 2**53:
        raise ValueError('Reward reduction length exceeds error-bound assumptions')
    absolute_sum = math.fsum(abs(value) for value in values)
    gamma = Fraction(n - 1, 2**53 - (n - 1))
    reference_allowance = 2 * Fraction.from_float(math.ulp(canonical))
    absolute_sum_upper = (Fraction.from_float(absolute_sum)
                          + 2 * Fraction.from_float(math.ulp(absolute_sum)))
    exact_bound = (gamma * absolute_sum_upper + reference_allowance
                   + n * Fraction.from_float(math.ulp(0.0)))
    bound = float(exact_bound)
    if Fraction.from_float(bound) < exact_bound:
        bound = math.nextafter(bound, math.inf)
    # An all-zero trace has no reduction ambiguity at all.
    if not any(values):
        bound = 0.0
    exact_error = abs(Fraction.from_float(recorded) - Fraction.from_float(canonical))
    if exact_error > Fraction.from_float(bound):
        raise ValueError('Episode return exceeds the justified reward-reduction error bound')
    return {'passed': True, 'count': n, 'canonical_return': canonical,
            'recorded_return': recorded, 'absolute_error': float(exact_error),
            'absolute_error_bound': bound,
            'reduction': 'math.fsum(explicit_python_float_binary64_rewards)',
            'bound_version': 'binary64_gamma_n_minus_1_fsum_2ulp_v1',
            'bound_formula': 'gamma_(n-1)*(fsum(abs(r))+2*ulp(fsum(abs(r))))+2*ulp(fsum(r))+n*min_subnormal; rounded upward',
            'unit_roundoff': 2.0**-53,
            'assumptions': 'IEEE754 binary64 round-to-nearest; finite rewards in [0,2]; no overflow',
            'file_integrity_proven_by_this_check': False}


def verify_file_integrity(path, expected_sha256, expected_size=None):
    """Require an independently recorded digest, before any deserialization."""
    path = Path(path)
    if (not isinstance(expected_sha256, str) or len(expected_sha256) != 64
            or any(char not in '0123456789abcdef' for char in expected_sha256)):
        raise ValueError('Missing or invalid expected SHA256')
    if expected_size is not None and (type(expected_size) is not int or expected_size < 0
                                     or path.stat().st_size != expected_size):
        raise ValueError('File size differs from recorded evidence')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    if digest.hexdigest() != expected_sha256:
        raise ValueError('File SHA256 differs from recorded evidence')


def require_frozen_equal(before, after):
    if before != after:
        raise RuntimeError('Frozen evaluation changed model/optimizer/counter/host RNG')


def validate_output_isolation(root, inputs):
    root = Path(root).resolve()
    if root.exists():
        raise FileExistsError('Final evaluation output already exists; never overwrite or resume')
    checkpoints = {arm: Path(path).resolve() for arm, path in inputs.items()}
    if set(checkpoints) != {'D', 'H'} or checkpoints['D'] == checkpoints['H']:
        raise ValueError('D and H require distinct final checkpoints')
    for cp in checkpoints.values():
        source = cp.parent.parent
        if root == source or root.is_relative_to(source) or source.is_relative_to(root):
            raise ValueError('Final evaluation must be isolated from both original training runs')
