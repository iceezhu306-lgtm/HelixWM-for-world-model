"""Trusted-prefix selection and correctly truncated lambda-return targets.

These are mathematical helpers, not a complete actor/critic implementation.
All supplied arrays must be finite, including unused suffixes. Unused *finite*
suffix values never affect accepted targets. Returned targets remain
differentiable; detach them explicitly when using them as critic labels.
"""

import math

import torch
from torch import Tensor


def _unit_interval(value: float, name: str) -> None:
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]")


def _finite(tensor: Tensor, name: str) -> None:
    if not torch.is_floating_point(tensor) or not torch.isfinite(tensor).all():
        raise ValueError(f"{name} must be a finite floating-point tensor")


def trusted_horizon(risk: Tensor, threshold: float) -> Tensor:
    """Return the longest consecutive accepted prefix, including length zero.

    Args:
        risk: Probabilities with shape ``[batch, horizon]``.
        threshold: A step is accepted when its risk is <= this threshold.

    NaN/Inf or out-of-range probabilities raise ValueError rather than being
    silently classified. Output is an integer tensor of shape ``[batch]``.
    """
    _unit_interval(threshold, "threshold")
    _finite(risk, "risk")
    if risk.ndim != 2 or ((risk < 0) | (risk > 1)).any():
        raise ValueError("risk must have shape [B, K] and values in [0, 1]")
    return (risk <= threshold).to(torch.long).cumprod(dim=1).sum(dim=1)


def lambda_returns(
    rewards: Tensor,
    continuation: Tensor,
    values: Tensor,
    horizons: Tensor,
    gamma: float = 0.99,
    lambda_: float = 0.95,
) -> tuple[Tensor, Tensor]:
    """Compute each sample's returns using only its accepted prefix.

    ``rewards[:, i]`` and ``continuation[:, i]`` belong to the transition
    from state i to state i+1. ``values[:, i]`` belongs to state i. Shapes
    are ``[B,K]``, ``[B,K]``, ``[B,K+1]`` and integer ``[B]`` respectively.
    Each trajectory starts backward recursion at ``V(state_horizons)``.
    Environment termination is represented by continuation=0; reliability
    truncation retains the boundary value bootstrap.

    Returns ``(targets[B,K], valid[B,K])``. Entries outside the prefix are
    exactly zero; horizons=0 yields no update samples. A caller should skip
    an update when the entire mask is false.
    """
    _unit_interval(gamma, "gamma")
    _unit_interval(lambda_, "lambda_")
    for name, tensor in (("rewards", rewards), ("continuation", continuation),
                         ("values", values)):
        _finite(tensor, name)
    if rewards.ndim != 2:
        raise ValueError("rewards must have shape [B, K]")
    batch, steps = rewards.shape
    if continuation.shape != rewards.shape or values.shape != (batch, steps + 1):
        raise ValueError("continuation must be [B,K] and values must be [B,K+1]")
    if horizons.shape != (batch,) or horizons.dtype not in (
        torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8,
    ):
        raise ValueError("horizons must be an integer tensor of shape [B]")
    if ((horizons < 0) | (horizons > steps)).any():
        raise ValueError("horizons must be between zero and K")
    if ((continuation < 0) | (continuation > 1)).any():
        raise ValueError("continuation must have values in [0, 1]")
    if any(t.device != rewards.device for t in (continuation, values, horizons)):
        raise ValueError("all inputs must be on the same device")
    if continuation.dtype != rewards.dtype or values.dtype != rewards.dtype:
        raise ValueError("rewards, continuation and values must share a dtype")

    valid = torch.arange(steps, device=rewards.device)[None, :] < horizons[:, None]
    running = values.gather(1, horizons.to(torch.long)[:, None]).squeeze(1)
    reverse_targets = []
    for index in range(steps - 1, -1, -1):
        active = valid[:, index]
        # Sanitize inactive cells *before* arithmetic. Even very large finite
        # rejected suffixes cannot overflow and pollute a valid target/gradient.
        r = torch.where(active, rewards[:, index], 0.0)
        c = torch.where(active, continuation[:, index], 0.0)
        next_value = torch.where(active, values[:, index + 1], 0.0)
        next_return = torch.where(active, running, 0.0)
        candidate = r + gamma * c * ((1.0 - lambda_) * next_value + lambda_ * next_return)
        running = torch.where(active, candidate, running)
        reverse_targets.append(torch.where(active, running, 0.0))
    if not reverse_targets:
        return rewards.clone(), valid
    return torch.stack(list(reversed(reverse_targets)), dim=1), valid
