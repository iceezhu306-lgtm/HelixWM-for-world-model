"""Causal reliability components for the HelixWM mathematical prototype.

An ensemble disagreement is a proxy, not an error probability. The risk head
is an uncalibrated estimator until fitted and evaluated on held-out data. No
environment interactions or online training loops are implemented here.
"""

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _finite(tensor: Tensor, name: str) -> None:
    if not torch.is_floating_point(tensor) or not torch.isfinite(tensor).all():
        raise ValueError(f"{name} must be a finite floating-point tensor")


def _mask(mask: Tensor, shape: tuple[int, ...], name: str) -> None:
    if mask.shape != shape or mask.dtype != torch.bool:
        raise ValueError(f"{name} must be boolean with shape {shape}")


def _scale(reward_scale: float) -> None:
    if not math.isfinite(reward_scale) or reward_scale <= 0:
        raise ValueError("reward_scale must be finite and strictly positive")


def _calculation_dtype(tensor: Tensor) -> Tensor:
    """Keep half-precision differences/reductions from overflowing early."""
    return tensor.float() if tensor.dtype in (torch.float16, torch.bfloat16) else tensor


def ensemble_disagreement(logits: Tensor) -> Tensor:
    """Grouped mixture JS, averaging members and groups, in natural-log units.

    Input is explicitly ``[members, batch, horizon, groups, classes]``;
    output is ``[batch, horizon]``. These are independently predicted
    categorical marginals evaluated on the same inputs, not a calibrated
    probability that a trajectory is wrong. Half/bfloat16 logits are promoted
    to float32 before subtraction and divergence arithmetic.
    """
    _finite(logits, "logits")
    if logits.ndim != 5 or any(size == 0 for size in logits.shape):
        raise ValueError("logits must have nonempty shape [M,B,K,J,C]")
    log_p = F.log_softmax(_calculation_dtype(logits), dim=-1)
    log_mixture = torch.logsumexp(log_p, dim=0) - math.log(logits.shape[0])
    member_group_kl = (log_p.exp() * (log_p - log_mixture)).sum(dim=-1)
    return member_group_kl.mean(dim=0).mean(dim=-1).clamp_min(0.0)


class PredictionEnsemble(nn.Module):
    """Independent small predictors with caller-supplied bootstrap selection.

    Features ``[B,K,D]`` must describe the same prediction inputs for all
    members (for example Dream hidden states). Each member has its own
    parameters. Inputs and posterior targets are detached, so this auxiliary
    supervision cannot update the encoder/dynamics producing them. The caller
    supplies bootstrap masks to make the sampling scheme inspectable.
    """

    def __init__(self, input_dim: int, groups: int, classes: int,
                 members: int = 3, hidden_dim: int = 64) -> None:
        super().__init__()
        if min(input_dim, groups, classes, members, hidden_dim) <= 0:
            raise ValueError("ensemble dimensions must be positive")
        self.input_dim, self.groups, self.classes = input_dim, groups, classes
        self.members = nn.ModuleList([
            nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.SiLU(),
                          nn.Linear(hidden_dim, groups * classes))
            for _ in range(members)
        ])

    def forward(self, features: Tensor) -> Tensor:
        _finite(features, "features")
        if features.ndim != 3 or features.shape[-1] != self.input_dim:
            raise ValueError("features must have shape [B,K,input_dim]")
        frozen = features.detach()
        return torch.stack([
            member(frozen).reshape(*features.shape[:2], self.groups, self.classes)
            for member in self.members
        ], dim=0)

    def bootstrap_loss(self, features: Tensor, target_logits: Tensor,
                       bootstrap_mask: Tensor, valid: Tensor | None = None) -> Tensor:
        """Posterior-to-member KL, summed over groups and averaged selected cells.

        Target logits have shape ``[B,K,J,C]``, selection mask ``[M,B,K]``.
        A completely empty selection gives a differentiable zero; callers
        should skip the optimizer step in that case (including weight decay).
        """
        prediction = self(features)
        _finite(target_logits, "target_logits")
        if target_logits.shape != prediction.shape[1:]:
            raise ValueError("target_logits must have shape [B,K,J,C]")
        _mask(bootstrap_mask, prediction.shape[:3], "bootstrap_mask")
        selected = bootstrap_mask
        if valid is not None:
            _mask(valid, features.shape[:2], "valid")
            selected = selected & valid.unsqueeze(0)
        log_q = F.log_softmax(_calculation_dtype(target_logits.detach()), dim=-1).unsqueeze(0)
        log_p = F.log_softmax(_calculation_dtype(prediction), dim=-1)
        kl = (log_q.exp() * (log_q - log_p)).sum(dim=(-1, -2))
        return torch.where(selected, kl, 0.0).sum() / selected.sum().clamp_min(1)


def observed_prediction_errors(
    posterior_logits: Tensor, saved_prior_logits: Tensor,
    actual_rewards: Tensor, saved_rewards: Tensor, reward_scale: float = 1.0,
) -> Tensor:
    """Compute already-observed latent JS and normalized reward residual.

    Logits shape is ``[...,J,C]``, rewards shape ``[...]``, result ``[...,2]``.
    The saved predictions must have been made before their matching actual
    outcome and carry a model-version identifier in the caller's data store.
    This stateless helper cannot verify that provenance. Output is detached.
    ``reward_scale`` is a strictly positive denominator (use scale+epsilon
    externally if required), not estimated from unseen future data.
    """
    _scale(reward_scale)
    for name, value in (("posterior_logits", posterior_logits),
                        ("saved_prior_logits", saved_prior_logits),
                        ("actual_rewards", actual_rewards), ("saved_rewards", saved_rewards)):
        _finite(value, name)
    if posterior_logits.ndim < 2 or posterior_logits.shape != saved_prior_logits.shape:
        raise ValueError("matching posterior and saved-prior shapes [...,J,C] required")
    if min(posterior_logits.shape[-2:]) <= 0:
        raise ValueError("groups and classes must be nonempty")
    if actual_rewards.shape != posterior_logits.shape[:-2] or saved_rewards.shape != actual_rewards.shape:
        raise ValueError("reward shapes must match the leading logit dimensions")
    log_q = F.log_softmax(_calculation_dtype(posterior_logits.detach()), dim=-1)
    log_p = F.log_softmax(_calculation_dtype(saved_prior_logits.detach()), dim=-1)
    log_mix = torch.logaddexp(log_q, log_p) - math.log(2.0)
    js = 0.5 * (log_q.exp() * (log_q - log_mix) + log_p.exp() * (log_p - log_mix)).sum(-1).mean(-1)
    residual = (_calculation_dtype(actual_rewards.detach())
                - _calculation_dtype(saved_rewards.detach())).abs() / reward_scale
    return torch.stack((js.clamp_min(0.0), residual), dim=-1)


def causal_error_history(errors: Tensor, valid: Tensor, is_first: Tensor,
                         window: int) -> tuple[Tensor, Tensor]:
    """Build each time's history from errors observed by that time, within episode.

    ``errors[B,T,E]`` is a chronologically indexed array of *already observed*
    residuals. Boolean ``valid[B,T]`` states availability; ``is_first[B,T]``
    marks the first index of a new episode. Return ``history[B,T,L,E]`` and
    ``mask[B,T,L]``, ordered oldest to newest with left padding. At t, no
    index after t is read. Mark a reset observation invalid when no matching
    within-episode prediction exists; never pair it with the previous episode.
    """
    _finite(errors, "errors")
    if errors.ndim != 3 or not isinstance(window, int) or isinstance(window, bool) or window <= 0:
        raise ValueError("errors must be [B,T,E] and window a positive integer")
    batch, steps, features = errors.shape
    _mask(valid, (batch, steps), "valid")
    _mask(is_first, (batch, steps), "is_first")
    histories = errors.new_zeros((batch, steps, window, features))
    masks = torch.zeros((batch, steps, window), dtype=torch.bool, device=errors.device)
    buffer = errors.new_zeros((batch, window, features))
    buffer_mask = torch.zeros((batch, window), dtype=torch.bool, device=errors.device)
    frozen = errors.detach()
    for index in range(steps):
        reset = is_first[:, index]
        buffer = torch.where(reset[:, None, None], 0.0, buffer)
        buffer_mask = buffer_mask & ~reset[:, None]
        entry = torch.where(valid[:, index, None], frozen[:, index], 0.0)
        buffer = torch.cat((buffer[:, 1:], entry[:, None]), dim=1)
        buffer_mask = torch.cat((buffer_mask[:, 1:], valid[:, index, None]), dim=1)
        histories[:, index], masks[:, index] = buffer, buffer_mask
    return histories, masks


def make_risk_targets(
    actual_rewards: Tensor, dreamed_rewards: Tensor, valid: Tensor,
    reward_scale: float = 1.0, threshold: float = 0.25,
) -> tuple[Tensor, Tensor]:
    """Label whether any same-action rollout error has exceeded the threshold.

    Arrays have shape ``[B,K]``. Prefix validity ends at the first unavailable
    or cross-reset label. Invalid labels are stored as zero placeholders *and
    must be masked*: they are unknown, not evidence of safety. The target is
    maximum individual normalized reward residual, not the sum of residuals.
    All arrays must be finite, even at masked positions; fill absent rewards
    with a finite placeholder and set valid=False. Labels are detached.
    """
    _scale(reward_scale)
    if not math.isfinite(threshold) or threshold < 0:
        raise ValueError("threshold must be finite and nonnegative")
    _finite(actual_rewards, "actual_rewards")
    _finite(dreamed_rewards, "dreamed_rewards")
    if actual_rewards.ndim != 2 or dreamed_rewards.shape != actual_rewards.shape:
        raise ValueError("reward arrays must have matching [B,K] shapes")
    _mask(valid, actual_rewards.shape, "valid")
    prefix_valid = valid.to(torch.long).cumprod(dim=1).bool()
    errors = (_calculation_dtype(actual_rewards.detach())
              - _calculation_dtype(dreamed_rewards.detach())).abs() / reward_scale
    ever_exceeded = (errors > threshold).to(torch.long).cummax(dim=1).values.bool()
    labels = (ever_exceeded & prefix_valid).to(actual_rewards.dtype)
    return labels, prefix_valid


class RiskHead(nn.Module):
    """Predict cumulative-event logits from available information only.

    The caller supplies the belief at rollout start, an already-observed
    masked history, candidate actions and matching ensemble disagreements.
    A forward-only GRU means risk at step k can use action/disagreement
    prefix <= k, never later candidates. This head consumes no future actual
    rewards/observations. Every input is detached: fitting risk cannot update
    the world model, ensemble or actor. Outputs need sigmoid before gating.
    """

    def __init__(self, belief_dim: int, action_dim: int, history_dim: int = 2,
                 hidden_dim: int = 64) -> None:
        super().__init__()
        if min(belief_dim, action_dim, history_dim, hidden_dim) <= 0:
            raise ValueError("risk-head dimensions must be positive")
        self.belief_dim, self.action_dim = belief_dim, action_dim
        self.history_dim, self.hidden_dim = history_dim, hidden_dim
        self.history_cell = nn.GRUCell(history_dim, hidden_dim)
        self.initial = nn.Sequential(nn.Linear(belief_dim + hidden_dim, hidden_dim), nn.Tanh())
        # Prefix input includes log(1+k), independent of the future horizon.
        self.prefix_cell = nn.GRUCell(action_dim + 2, hidden_dim)
        self.output = nn.Linear(hidden_dim, 1)

    def forward(self, belief: Tensor, history: Tensor, history_mask: Tensor,
                actions: Tensor, disagreements: Tensor) -> Tensor:
        for name, value in (("belief", belief), ("history", history),
                            ("actions", actions), ("disagreements", disagreements)):
            _finite(value, name)
        if belief.ndim != 2 or belief.shape[-1] != self.belief_dim:
            raise ValueError("belief must have shape [B,belief_dim]")
        batch = belief.shape[0]
        if history.ndim != 3 or history.shape[0] != batch or history.shape[-1] != self.history_dim:
            raise ValueError("history must have shape [B,L,history_dim]")
        _mask(history_mask, history.shape[:2], "history_mask")
        if actions.ndim != 3 or actions.shape[0] != batch or actions.shape[-1] != self.action_dim:
            raise ValueError("actions must have shape [B,K,action_dim]")
        if disagreements.shape != actions.shape[:2] or (disagreements < 0).any():
            raise ValueError("disagreements must be nonnegative with shape [B,K]")
        frozen_history = history.detach()
        encoded = belief.new_zeros((batch, self.hidden_dim))
        for index in range(history.shape[1]):
            usable = history_mask[:, index, None]
            entry = torch.where(usable, frozen_history[:, index], 0.0)
            candidate = self.history_cell(entry, encoded)
            encoded = torch.where(usable, candidate, encoded)
        state = self.initial(torch.cat((belief.detach(), encoded), dim=-1))
        logits = []
        for index in range(actions.shape[1]):
            step = belief.new_full((batch, 1), math.log1p(index + 1))
            entry = torch.cat((actions[:, index].detach(),
                               disagreements[:, index, None].detach(), step), dim=-1)
            state = self.prefix_cell(entry, state)
            logits.append(self.output(state).squeeze(-1))
        if not logits:
            return belief.new_empty((batch, 0))
        return torch.stack(logits, dim=1)


def risk_bce_loss(logits: Tensor, labels: Tensor, valid: Tensor) -> Tensor:
    """Masked risk-head BCE; skip the optimizer step if no label is valid."""
    _finite(logits, "logits")
    _finite(labels, "labels")
    if logits.ndim != 2 or labels.shape != logits.shape or ((labels < 0) | (labels > 1)).any():
        raise ValueError("matching [B,K] logits and labels in [0,1] required")
    _mask(valid, logits.shape, "valid")
    loss = F.binary_cross_entropy_with_logits(logits, labels.detach(), reduction="none")
    return torch.where(valid, loss, 0.0).sum() / valid.sum().clamp_min(1)
