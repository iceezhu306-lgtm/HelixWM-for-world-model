"""Small shared RSSM for synthetic vector observations.

This is a reference implementation of the Real/Dream equations, not DreamerV3
or a pixel-control training agent. An action at t leads to observation t+1.
"""

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .distributions import straight_through_sample


@dataclass(frozen=True)
class RSSMState:
    h: Tensor
    z: Tensor

    def detach(self) -> "RSSMState":
        return RSSMState(self.h.detach(), self.z.detach())

    def features(self) -> Tensor:
        return torch.cat((self.h, self.z.flatten(start_dim=-2)), dim=-1)


@dataclass(frozen=True)
class ObservationRollout:
    """N=T+1 posterior states; logits have shape [B,N,groups,classes]."""

    states: tuple[RSSMState, ...]
    prior_logits: Tensor
    posterior_logits: Tensor


@dataclass(frozen=True)
class ImaginationRollout:
    """K+1 states including the start; logits have shape [B,K,groups,classes]."""

    states: tuple[RSSMState, ...]
    prior_logits: Tensor


class RSSM(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 32,
        groups: int = 4,
        classes: int = 4,
        embed_dim: int = 32,
    ) -> None:
        super().__init__()
        if min(obs_dim, action_dim, hidden_dim, groups, classes, embed_dim) < 1:
            raise ValueError("all model dimensions must be positive")
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.groups = groups
        self.classes = classes
        self.encoder = nn.Sequential(nn.Linear(obs_dim, embed_dim), nn.ELU())
        self.recurrent = nn.GRUCell(groups * classes + action_dim, hidden_dim)
        self.prior = nn.Linear(hidden_dim, groups * classes)
        self.posterior = nn.Linear(hidden_dim + embed_dim, groups * classes)
        feature_dim = hidden_dim + groups * classes
        self.decoder = nn.Linear(feature_dim, obs_dim)
        self.reward_head = nn.Linear(feature_dim, 1)
        self.continuation_head = nn.Linear(feature_dim, 1)

    def initial(self, batch_size: int) -> RSSMState:
        reference = next(self.parameters())
        return RSSMState(
            reference.new_zeros(batch_size, self.hidden_dim),
            reference.new_zeros(batch_size, self.groups, self.classes),
        )

    def _transition(self, state: RSSMState, action: Tensor) -> Tensor:
        batch_size = state.h.shape[0]
        if state.h.shape != (batch_size, self.hidden_dim):
            raise ValueError("state.h must have shape [B,hidden_dim]")
        if state.z.shape != (batch_size, self.groups, self.classes):
            raise ValueError("state.z must have shape [B,groups,classes]")
        if action.shape != (batch_size, self.action_dim):
            raise ValueError("action must have shape [B,action_dim]")
        inputs = torch.cat((state.z.flatten(1), action), dim=-1)
        return self.recurrent(inputs, state.h)

    def observe_step(
        self,
        previous: RSSMState,
        previous_action: Tensor,
        observation: Tensor,
        is_first: Tensor,
        *,
        sample: bool = True,
    ) -> tuple[RSSMState, Tensor, Tensor]:
        """Reset previous memory and action BEFORE assimilating a new episode.

        Returns posterior state, one-step prior logits, posterior logits.
        The initial observation uses a zero previous state and zero action.
        """
        batch_size = previous.h.shape[0]
        if observation.shape != (batch_size, self.obs_dim):
            raise ValueError("observation must have shape [B,obs_dim]")
        if is_first.shape != (batch_size,) or is_first.dtype != torch.bool:
            raise ValueError("is_first must be a bool tensor with shape [B]")
        previous = RSSMState(
            torch.where(is_first[:, None], torch.zeros_like(previous.h), previous.h),
            torch.where(is_first[:, None, None], torch.zeros_like(previous.z), previous.z),
        )
        previous_action = torch.where(
            is_first[:, None], torch.zeros_like(previous_action), previous_action
        )
        h = self._transition(previous, previous_action)
        prior_logits = self.prior(h).reshape(batch_size, self.groups, self.classes)
        embedded = self.encoder(observation)
        posterior_logits = self.posterior(torch.cat((h, embedded), -1)).reshape(
            batch_size, self.groups, self.classes
        )
        z = straight_through_sample(posterior_logits, deterministic=not sample)
        return RSSMState(h, z), prior_logits, posterior_logits

    def imagine_step(
        self, previous: RSSMState, action: Tensor, *, sample: bool = True
    ) -> tuple[RSSMState, Tensor]:
        """Pure prior step: deliberately accepts no future observation."""
        h = self._transition(previous, action)
        logits = self.prior(h).reshape(h.shape[0], self.groups, self.classes)
        z = straight_through_sample(logits, deterministic=not sample)
        return RSSMState(h, z), logits

    def observe(
        self,
        observations: Tensor,
        actions: Tensor,
        is_first: Tensor | None = None,
        *,
        sample: bool = True,
    ) -> ObservationRollout:
        """Observe [B,T+1,O] using actions [B,T,A] and reset flags [B,T+1]."""
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
        if is_first.shape != (batch_size, num_observations) or is_first.dtype != torch.bool:
            raise ValueError("is_first must be a bool tensor with shape [B,T+1]")
        state = self.initial(batch_size)
        states, priors, posteriors = [], [], []
        for index in range(num_observations):
            previous_action = (
                actions[:, index - 1]
                if index > 0
                else actions.new_zeros(batch_size, self.action_dim)
            )
            state, prior, posterior = self.observe_step(
                state, previous_action, observations[:, index], is_first[:, index], sample=sample
            )
            states.append(state)
            priors.append(prior)
            posteriors.append(posterior)
        return ObservationRollout(tuple(states), torch.stack(priors, 1), torch.stack(posteriors, 1))

    def imagine(
        self, start_state: RSSMState, actions: Tensor, *, sample: bool = True
    ) -> ImaginationRollout:
        """Prior-only rollout with actions [B,K,A].

        This primitive preserves caller gradients. Pass ``start_state.detach()``
        at the cross-correction boundary, as done in ``cross_correction_loss``.
        """
        if actions.ndim != 3 or actions.shape[0] != start_state.h.shape[0]:
            raise ValueError("actions must have shape [B,K,action_dim]")
        if actions.shape[-1] != self.action_dim:
            raise ValueError("actions have an incorrect action dimension")
        states, priors = [start_state], []
        for action in actions.unbind(1):
            state, prior = self.imagine_step(states[-1], action, sample=sample)
            states.append(state)
            priors.append(prior)
        logits = (
            torch.stack(priors, 1)
            if priors
            else start_state.h.new_empty(start_state.h.shape[0], 0, self.groups, self.classes)
        )
        return ImaginationRollout(tuple(states), logits)

    def decode(self, state: RSSMState) -> dict[str, Tensor]:
        features = state.features()
        return {
            "observation": self.decoder(features),
            "reward": self.reward_head(features).squeeze(-1),
            "continuation_logits": self.continuation_head(features).squeeze(-1),
        }
