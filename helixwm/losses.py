"""Explicit reference losses; no Dreamer free bits or distributional heads."""

import torch
from torch import Tensor
from torch.nn import functional as F

from .distributions import categorical_kl
from .rssm import ObservationRollout, RSSM


def _check_mask(mask: Tensor, shape: tuple[int, int], name: str) -> None:
    if mask.shape != shape or mask.dtype != torch.bool:
        raise ValueError(f"{name} must be a bool tensor with shape {shape}")


def _masked_mean(value: Tensor, mask: Tensor) -> Tensor:
    return torch.where(mask, value, torch.zeros_like(value)).sum() / mask.sum().clamp_min(1)


def world_model_loss(
    model: RSSM,
    rollout: ObservationRollout,
    observations: Tensor,
    rewards: Tensor,
    continuations: Tensor,
    valid: Tensor,
    is_first: Tensor | None = None,
    *,
    dynamics_scale: float = 1.0,
    representation_scale: float = 0.1,
) -> dict[str, Tensor]:
    """Base loss, with targets all aligned to observation positions [B,N].

    Observation and reward heads use fixed-unit-variance Gaussian NLL up to
    constants (half squared error); continuation uses Bernoulli NLL. Reward
    and continuation at position zero and reset positions have no preceding
    in-episode transition and are excluded. Keep the reward/continuation on
    the terminal observation before reset, if present in the collected data.
    Pass the same ``is_first`` used by ``observe`` for multi-episode sequences.
    """
    if observations.ndim != 3 or observations.shape[-1] != model.obs_dim:
        raise ValueError("observations must have shape [B,N,obs_dim]")
    shape = observations.shape[:2]
    if len(rollout.states) != shape[1]:
        raise ValueError("rollout length must match observations")
    if rewards.shape != shape or continuations.shape != shape:
        raise ValueError("rewards and continuations must have shape [B,N]")
    _check_mask(valid, shape, "valid")
    if is_first is None:
        is_first = torch.zeros_like(valid)
    _check_mask(is_first, shape, "is_first")
    if dynamics_scale < 0 or representation_scale < 0:
        raise ValueError("KL scales must be nonnegative")
    transition_valid = valid & ~is_first
    transition_valid = transition_valid.clone()
    transition_valid[:, 0] = False
    # Both ends of the transition must exist; padded observations are not
    # sources of reward or continuation supervision.
    transition_valid[:, 1:] &= valid[:, :-1]
    predictions = [model.decode(state) for state in rollout.states]
    predicted_obs = torch.stack([prediction["observation"] for prediction in predictions], 1)
    predicted_reward = torch.stack([prediction["reward"] for prediction in predictions], 1)
    continuation_logits = torch.stack(
        [prediction["continuation_logits"] for prediction in predictions], 1
    )
    observation_loss = _masked_mean(0.5 * (predicted_obs - observations).square().sum(-1), valid)
    reward_loss = _masked_mean(0.5 * (predicted_reward - rewards).square(), transition_valid)
    continuation_loss = _masked_mean(
        F.binary_cross_entropy_with_logits(continuation_logits, continuations, reduction="none"),
        transition_valid,
    )
    dynamics_kl = _masked_mean(
        categorical_kl(rollout.posterior_logits.detach(), rollout.prior_logits), valid
    )
    representation_kl = _masked_mean(
        categorical_kl(rollout.posterior_logits, rollout.prior_logits.detach()), valid
    )
    total = (
        observation_loss + reward_loss + continuation_loss
        + dynamics_scale * dynamics_kl + representation_scale * representation_kl
    )
    return {
        "total": total,
        "observation": observation_loss,
        "reward": reward_loss,
        "continuation": continuation_loss,
        "dynamics_kl": dynamics_kl,
        "representation_kl": representation_kl,
    }


def cross_correction_loss(
    model: RSSM,
    observation_rollout: ObservationRollout,
    actions: Tensor,
    is_first: Tensor,
    valid: Tensor,
    max_horizon: int = 3,
    *,
    sample: bool = True,
) -> Tensor:
    """KL posterior-teacher -> Dream-prior from horizons 2 through K.

    Starts and teacher distributions stop gradients; Dream dynamics retain
    gradients. The same recorded actions must produce the observation rollout.
    All positions in a pair's path must be valid and no reset may occur after
    its start. Each valid (batch,start,horizon) pair has equal weight.

    One sampled Dream path gives a Monte Carlo estimate of expected conditional
    KL, NOT the exact KL to a multi-step marginal. The straight-through sampler
    additionally supplies a biased surrogate gradient through discrete states.
    """
    if max_horizon < 1:
        raise ValueError("max_horizon must be positive")
    posterior = observation_rollout.posterior_logits
    if posterior.ndim != 4:
        raise ValueError("posterior logits must have shape [B,N,groups,classes]")
    batch_size, num_observations = posterior.shape[:2]
    shape = (batch_size, num_observations)
    _check_mask(valid, shape, "valid")
    _check_mask(is_first, shape, "is_first")
    if len(observation_rollout.states) != num_observations:
        raise ValueError("state count must match posterior logits")
    if actions.shape != (batch_size, num_observations - 1, model.action_dim):
        raise ValueError("actions must have shape [B,N-1,action_dim]")
    # An all-masked batch remains a differentiable zero without inventing pairs.
    numerator = next(model.prior.parameters()).sum() * 0.0
    denominator = posterior.new_zeros(())
    for start in range(num_observations - 2):
        horizon = min(max_horizon, num_observations - start - 1)
        if horizon < 2:
            continue
        imagined = model.imagine(
            observation_rollout.states[start].detach(),
            actions[:, start:start + horizon],
            sample=sample,
        )
        path_valid = valid[:, start].clone()
        for step in range(1, horizon + 1):
            end = start + step
            path_valid = path_valid & valid[:, end] & ~is_first[:, end]
            if step < 2:
                continue
            pair_kl = categorical_kl(posterior[:, end].detach(), imagined.prior_logits[:, step - 1])
            numerator = numerator + torch.where(path_valid, pair_kl, torch.zeros_like(pair_kl)).sum()
            denominator = denominator + path_valid.sum()
    return numerator / denominator.clamp_min(1)
