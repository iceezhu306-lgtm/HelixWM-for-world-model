"""Pixel capacity choices and batched forms of the v1 reference equations.

Images are flattened NHWC RGB arrays, normalized to [-0.5, 0.5]. The CNNs
and head widths are explicit engineering choices for the first experiment;
they do not make this implementation a reproduction of another agent.
"""

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .distributions import categorical_kl, straight_through_sample
from .losses import _check_mask, _masked_mean
from .rssm import ObservationRollout, RSSM, RSSMState


class PixelEncoder(nn.Module):
    def __init__(self, depth: int, embed_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, depth, 4, 2), nn.ELU(),
            nn.Conv2d(depth, depth * 2, 4, 2), nn.ELU(),
            nn.Conv2d(depth * 2, depth * 4, 4, 2), nn.ELU(),
            nn.Conv2d(depth * 4, depth * 8, 4, 2), nn.ELU(),
            nn.Flatten(), nn.Linear(depth * 8 * 2 * 2, embed_dim), nn.ELU(),
        )

    def forward(self, observation: Tensor) -> Tensor:
        image = observation.reshape(-1, 64, 64, 3).permute(0, 3, 1, 2).contiguous()
        return self.network(image)


class PixelDecoder(nn.Module):
    def __init__(self, feature_dim: int, depth: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(feature_dim, depth * 8 * 2 * 2), nn.ELU(),
            nn.Unflatten(1, (depth * 8, 2, 2)),
            nn.ConvTranspose2d(depth * 8, depth * 4, 4, 2), nn.ELU(),
            nn.ConvTranspose2d(depth * 4, depth * 2, 4, 2), nn.ELU(),
            nn.ConvTranspose2d(depth * 2, depth, 4, 2), nn.ELU(),
            nn.ConvTranspose2d(depth, 3, 6, 2),
        )

    def forward(self, features: Tensor) -> Tensor:
        image = self.network(features)
        # An unconstrained mean is consistent with the fixed-variance Gaussian
        # reconstruction loss. There is no sigmoid or output clipping.
        return image.permute(0, 2, 3, 1).reshape(features.shape[0], 64 * 64 * 3)


class PixelRSSM(RSSM):
    """Shared categorical RSSM with 64x64 RGB input and batched encoding."""

    def __init__(
        self,
        action_dim: int = 6,
        hidden_dim: int = 256,
        groups: int = 16,
        classes: int = 16,
        embed_dim: int = 256,
        cnn_depth: int = 32,
        image_size: int = 64,
    ) -> None:
        if image_size != 64:
            raise ValueError("the v1 pixel model requires image_size=64")
        if cnn_depth < 1:
            raise ValueError("cnn_depth must be positive")
        super().__init__(image_size * image_size * 3, action_dim, hidden_dim, groups, classes, embed_dim)
        self.embed_dim = embed_dim
        self.cnn_depth = cnn_depth
        self.image_size = image_size
        self.constructor_config = {
            "action_dim": action_dim,
            "hidden_dim": hidden_dim,
            "groups": groups,
            "classes": classes,
            "embed_dim": embed_dim,
            "cnn_depth": cnn_depth,
            "image_size": image_size,
        }
        feature_dim = hidden_dim + groups * classes
        self.encoder = PixelEncoder(cnn_depth, embed_dim)
        self.decoder = PixelDecoder(feature_dim, cnn_depth)
        self.reward_head = self._scalar_head(feature_dim)
        self.continuation_head = self._scalar_head(feature_dim)

    @staticmethod
    def _scalar_head(feature_dim: int) -> nn.Module:
        return nn.Sequential(
            nn.Linear(feature_dim, 256), nn.ELU(),
            nn.Linear(256, 256), nn.ELU(), nn.Linear(256, 1),
        )

    def observe(
        self,
        observations: Tensor,
        actions: Tensor,
        is_first: Tensor | None = None,
        *,
        sample: bool = True,
    ) -> ObservationRollout:
        """Encode [B,N,O] once, retaining the reference recurrent/reset order."""
        if observations.ndim != 3 or observations.shape[-1] != self.obs_dim:
            raise ValueError("observations must have shape [B,T+1,obs_dim]")
        batch_size, num_observations, _ = observations.shape
        if batch_size < 1 or num_observations < 1:
            raise ValueError("observation batch and sequence must be nonempty")
        if actions.shape != (batch_size, num_observations - 1, self.action_dim):
            raise ValueError("actions must have shape [B,T,action_dim]")
        if is_first is None:
            is_first = torch.zeros(
                batch_size, num_observations, device=observations.device, dtype=torch.bool
            )
            is_first[:, 0] = True
        _check_mask(is_first, (batch_size, num_observations), "is_first")
        embeddings = self.encoder(observations.reshape(-1, self.obs_dim)).reshape(
            batch_size, num_observations, self.embed_dim
        )
        state = self.initial(batch_size)
        states, priors, posteriors = [], [], []
        for index in range(num_observations):
            reset = is_first[:, index]
            previous_action = (
                actions[:, index - 1]
                if index > 0
                else actions.new_zeros(batch_size, self.action_dim)
            )
            state = RSSMState(
                torch.where(reset[:, None], torch.zeros_like(state.h), state.h),
                torch.where(reset[:, None, None], torch.zeros_like(state.z), state.z),
            )
            previous_action = torch.where(
                reset[:, None], torch.zeros_like(previous_action), previous_action
            )
            h = self._transition(state, previous_action)
            prior = self.prior(h).reshape(batch_size, self.groups, self.classes)
            posterior = self.posterior(torch.cat((h, embeddings[:, index]), -1)).reshape(
                batch_size, self.groups, self.classes
            )
            state = RSSMState(h, straight_through_sample(posterior, deterministic=not sample))
            states.append(state)
            priors.append(prior)
            posteriors.append(posterior)
        return ObservationRollout(tuple(states), torch.stack(priors, 1), torch.stack(posteriors, 1))


def pixel_world_model_loss(
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
    """Reference base loss, decoding all B*N states in one CNN call.

Reconstruction sums all pixels, rather than averaging them. Reward and
continuation exclude position zero, reset positions, and invalid transitions.
There are no free bits or other changes to the v1 loss equations.
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
    transition_valid = (valid & ~is_first).clone()
    transition_valid[:, 0] = False
    transition_valid[:, 1:] &= valid[:, :-1]
    flat_state = RSSMState(
        torch.stack([state.h for state in rollout.states], 1).reshape(-1, model.hidden_dim),
        torch.stack([state.z for state in rollout.states], 1).reshape(-1, model.groups, model.classes),
    )
    prediction = model.decode(flat_state)
    predicted_obs = prediction["observation"].reshape(*shape, model.obs_dim)
    predicted_reward = prediction["reward"].reshape(shape)
    continuation_logits = prediction["continuation_logits"].reshape(shape)
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


def fast_cross_correction_loss(
    model: RSSM,
    observation_rollout: ObservationRollout,
    actions: Tensor,
    is_first: Tensor,
    valid: Tensor,
    max_horizon: int = 5,
    *,
    sample: bool = True,
) -> Tensor:
    """Batch sequence start positions; keep only a loop over Dream horizon.

Every valid (batch,start,horizon>=2) pair has equal weight, exactly as in
``cross_correction_loss``. Starts and posterior teachers are detached, while
Dream dynamics and supplied actions retain gradients. Stochastic sampling
has the same law, but batching changes RNG draw order relative to reference
calls; only deterministic-mode numerical equivalence is expected.
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
    numerator = next(model.prior.parameters()).sum() * 0.0
    denominator = posterior.new_zeros(())
    num_starts = num_observations - 2
    if num_starts < 1 or max_horizon < 2:
        return numerator
    h = torch.stack([state.h.detach() for state in observation_rollout.states[:num_starts]], 1)
    z = torch.stack([state.z.detach() for state in observation_rollout.states[:num_starts]], 1)
    path_valid = valid[:, :num_starts]
    for step in range(1, min(max_horizon, num_observations - 1) + 1):
        # Starts near the sequence end naturally have shorter Dream paths.
        count = min(num_starts, num_observations - step)
        state = RSSMState(
            h[:, :count].reshape(-1, model.hidden_dim),
            z[:, :count].reshape(-1, model.groups, model.classes),
        )
        action = actions[:, step - 1:step - 1 + count].reshape(-1, model.action_dim)
        imagined, prior = model.imagine_step(state, action, sample=sample)
        h = imagined.h.reshape(batch_size, count, model.hidden_dim)
        z = imagined.z.reshape(batch_size, count, model.groups, model.classes)
        path_valid = (
            path_valid[:, :count]
            & valid[:, step:step + count]
            & ~is_first[:, step:step + count]
        )
        if step < 2:
            continue
        pair_kl = categorical_kl(
            posterior[:, step:step + count].detach(),
            prior.reshape(batch_size, count, model.groups, model.classes),
        )
        numerator = numerator + torch.where(path_valid, pair_kl, torch.zeros_like(pair_kl)).sum()
        denominator = denominator + path_valid.sum()
    return numerator / denominator.clamp_min(1)
