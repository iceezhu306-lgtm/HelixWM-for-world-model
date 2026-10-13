"""Grouped categorical operations used by the HelixWM math prototype.

The final axes are ``[groups, classes]``. KL sums independent groups, whereas
JS averages groups so its natural-log scale stays between zero and log(2).
"""

import math

import torch
from torch import Tensor
from torch.nn import functional as F


def _log_probs(logits: Tensor) -> Tensor:
    if logits.ndim < 2 or min(logits.shape[-2:]) < 1:
        raise ValueError("logits must end in nonempty [groups, classes] axes")
    if not logits.is_floating_point():
        raise TypeError("logits must have a floating-point dtype")
    # Avoid half-precision overflow in differences of large finite logits.
    if logits.dtype in (torch.float16, torch.bfloat16):
        logits = logits.float()
    return F.log_softmax(logits, dim=-1)


def _paired_log_probs(q_logits: Tensor, p_logits: Tensor) -> tuple[Tensor, Tensor]:
    if q_logits.shape != p_logits.shape:
        raise ValueError("q_logits and p_logits must have identical shapes")
    return _log_probs(q_logits), _log_probs(p_logits)


def categorical_kl(q_logits: Tensor, p_logits: Tensor) -> Tensor:
    """Return KL(q || p), summing classes and independent latent groups."""
    log_q, log_p = _paired_log_probs(q_logits, p_logits)
    return (log_q.exp() * (log_q - log_p)).sum(dim=(-1, -2))


def categorical_js(q_logits: Tensor, p_logits: Tensor) -> Tensor:
    """Return symmetric Jensen-Shannon divergence, averaged over groups."""
    log_q, log_p = _paired_log_probs(q_logits, p_logits)
    log_m = torch.logaddexp(log_q, log_p) - math.log(2.0)
    per_group = 0.5 * (
        (log_q.exp() * (log_q - log_m)).sum(-1)
        + (log_p.exp() * (log_p - log_m)).sum(-1)
    )
    return per_group.mean(-1)


def straight_through_sample(logits: Tensor, *, deterministic: bool = False) -> Tensor:
    """One-hot forward values with softmax surrogate gradients.

    This is a biased straight-through estimator, not an exact categorical
    reparameterization. ``deterministic=True`` uses argmax for reproducible
    mathematical checks, rather than sampling from the categorical law.
    """
    probabilities = _log_probs(logits).exp()
    if deterministic:
        indices = probabilities.argmax(-1)
    else:
        flat = probabilities.reshape(-1, probabilities.shape[-1])
        indices = torch.multinomial(flat, 1).reshape(probabilities.shape[:-1])
    one_hot = F.one_hot(indices, num_classes=probabilities.shape[-1]).to(probabilities)
    return one_hot + (probabilities - probabilities.detach())
