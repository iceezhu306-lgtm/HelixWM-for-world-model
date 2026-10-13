"""First-experiment policy learning with strictly gated imagined returns.

This is the first HelixWM research controller, not an implementation of the
Dreamer authors' training recipe. Actions for the score-function objective
are detached; future observed outcomes are auxiliary targets only. A zero
trusted horizon really produces no policy or value optimizer step.
"""

import copy
import math

import torch
from torch import Tensor, nn
from torch.distributions import Normal
from torch.nn import functional as F

from .pixels import pixel_world_model_loss, fast_cross_correction_loss
from .reliability import (
    PredictionEnsemble, RiskHead, causal_error_history, ensemble_disagreement,
    make_risk_targets, risk_bce_loss,
)
from .returns import lambda_returns, trusted_horizon
from .rssm import RSSMState


class TanhGaussianActor(nn.Module):
    """Diagonal Gaussian followed by tanh; log density sums action dimensions.

    ``sample`` returns detached actions and differentiable score-function log
    densities. ``entropy_estimate`` instead uses a fresh reparameterized draw
    to estimate the entropy gradient of the bounded action distribution.
    """

    def __init__(self, feature_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.action_dim = action_dim
        self.network = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ELU(),
        )
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

    def _distribution(self, features: Tensor) -> Normal:
        hidden = self.network(features)
        return Normal(self.mean(hidden), self.log_std(hidden).clamp(-5.0, 2.0).exp(), validate_args=False)

    @staticmethod
    def _log_density(distribution: Normal, pre_tanh: Tensor) -> Tensor:
        # log(1 - tanh(x)^2), computed without cancellation at large |x|.
        log_jacobian = 2.0 * (math.log(2.0) - pre_tanh - F.softplus(-2.0 * pre_tanh))
        return (distribution.log_prob(pre_tanh) - log_jacobian).sum(-1)

    def sample(self, features: Tensor, deterministic: bool = False) -> tuple[Tensor, Tensor]:
        actions, log_probability, _ = self.sample_with_pre_tanh(features, deterministic)
        return actions, log_probability

    def sample_with_pre_tanh(
        self, features: Tensor, deterministic: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return bounded action, exact sample log density, and detached draw.

        Keep the original Gaussian draw for later score-function updates.
        Float32 tanh can round to +/-1, making inverse reconstruction lossy.
        """
        distribution = self._distribution(features)
        pre_tanh = distribution.mean.detach() if deterministic else distribution.sample()
        actions = pre_tanh.tanh().detach()
        return actions, self._log_density(distribution, pre_tanh), pre_tanh.detach()

    def log_prob_from_pre_tanh(self, features: Tensor, pre_tanh: Tensor) -> Tensor:
        """Exact score density from a retained, fixed original Gaussian draw."""
        if pre_tanh.shape != (*features.shape[:-1], self.action_dim):
            raise ValueError("pre_tanh must match feature leading dimensions and action_dim")
        if not torch.isfinite(pre_tanh).all():
            raise ValueError("pre_tanh must be finite")
        return self._log_density(self._distribution(features), pre_tanh.detach())

    def log_prob(self, features: Tensor, actions: Tensor) -> Tensor:
        """Approximate inverse density when only rounded actions are available.

        Do not use this helper for on-policy score updates: saturation loses
        the original draw. Use ``log_prob_from_pre_tanh`` for those updates.
        """
        if actions.shape != (*features.shape[:-1], self.action_dim):
            raise ValueError("actions must match feature leading dimensions and action_dim")
        # Float32 tanh may round to an endpoint. Clamp only for the inverse,
        # never to conceal a nonfinite sample or to change environment actions.
        if not torch.isfinite(actions).all() or (actions.abs() > 1).any():
            raise ValueError("actions must be finite and inside [-1,1]")
        bounded = actions.detach().clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        return self._log_density(self._distribution(features), torch.atanh(bounded))

    def entropy_estimate(self, features: Tensor) -> Tensor:
        distribution = self._distribution(features)
        return -self._log_density(distribution, distribution.rsample())


class Value(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ELU(), nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: Tensor) -> Tensor:
        return self.network(features).squeeze(-1)


def same_episode_prefix_mask(valid: Tensor, is_first: Tensor, rows: Tensor,
                             starts: Tensor, horizon: int) -> Tensor:
    """Available transitions after sampled starts; no reset or missing suffix.

    The source itself must exist. Once one endpoint is missing or begins a
    new episode, all following endpoints are unavailable for that rollout.
    """
    if valid.ndim != 2 or valid.dtype != torch.bool or is_first.shape != valid.shape:
        raise ValueError("valid and is_first must be matching [B,N] boolean arrays")
    if is_first.dtype != torch.bool or horizon < 1:
        raise ValueError("is_first must be boolean and horizon positive")
    endpoints = starts[:, None] + torch.arange(1, horizon + 1, device=valid.device)
    exists = endpoints < valid.shape[1]
    bounded = endpoints.clamp_max(valid.shape[1] - 1)
    available = exists & valid[rows[:, None], bounded] & ~is_first[rows[:, None], bounded]
    available &= valid[rows, starts, None]
    return available.long().cumprod(1).bool()


class HelixLearner(nn.Module):
    """World-model, causal reliability, and score-function controller updates.

    Batch fields follow the reference observation-aligned convention: images
    ``observations[B,N,O]``, ``actions[B,N-1,A]``, and rewards/continuations/
    valid/is_first/errors/error_valid/model_versions at N observation indices.
    ``errors[B,N,2]`` must already have been computed from predictions saved
    before the corresponding real outcomes, with their collection-time model
    version. This module cannot reconstruct that provenance retrospectively.

    ``state_dict`` includes every module, including the slow target critic.
    The caller must also checkpoint each entry in ``optimizers``, replay,
    collector state, counters and RNG state for a training recovery attempt.
    """

    DEFAULTS = {
        "burnin": 8, "horizon": 15, "cc_horizon": 5, "cc_scale": 0.1,
        "grad_clip": 100.0, "risk_threshold": 0.5, "risk_error_threshold": 0.25,
        "reward_scale": 2.0, "ensemble_members": 3, "ensemble_bootstrap_prob": 0.8,
        "target_tau": 0.01, "gamma": 0.99, "lambda_": 0.95, "entropy_scale": 0.001,
        "imagination_batch": 128, "aux_batch": 128, "actor_hidden": 256,
        "critic_hidden": 256, "reliability_hidden": 64, "history_window": 8,
        "model_lr": 3e-4, "ensemble_lr": 3e-4, "risk_lr": 3e-4,
        "actor_lr": 8e-5, "critic_lr": 8e-5,
    }

    def __init__(self, model: nn.Module, config: dict):
        super().__init__()
        self.config = {**self.DEFAULTS, **config}
        c = self.config
        for key in ("horizon", "cc_horizon", "history_window", "imagination_batch", "aux_batch"):
            if not isinstance(c[key], int) or isinstance(c[key], bool) or c[key] < 1:
                raise ValueError(f"{key} must be a positive integer")
        if c["burnin"] < 0 or c["reward_scale"] <= 0 or c["grad_clip"] <= 0:
            raise ValueError("burnin must be nonnegative; reward_scale and grad_clip positive")
        for key in ("risk_threshold", "ensemble_bootstrap_prob", "target_tau", "gamma", "lambda_"):
            if not math.isfinite(c[key]) or not 0 <= c[key] <= 1:
                raise ValueError(f"{key} must be finite and in [0,1]")
        self.model = model
        features = model.hidden_dim + model.groups * model.classes
        self.actor = TanhGaussianActor(features, model.action_dim, c["actor_hidden"])
        self.critic = Value(features, c["critic_hidden"])
        self.target_critic = copy.deepcopy(self.critic).requires_grad_(False)
        self.ensemble = PredictionEnsemble(
            model.hidden_dim, model.groups, model.classes, c["ensemble_members"],
            c["reliability_hidden"],
        )
        self.risk_head = RiskHead(features, model.action_dim, hidden_dim=c["reliability_hidden"])
        # Respect the supplied model's device before optimizers capture tensors.
        self.to(next(model.parameters()).device)
        self.optimizers = {
            name: torch.optim.Adam(getattr(self, name if name != "risk" else "risk_head").parameters(),
                                   lr=c[f"{name}_lr"])
            for name in ("model", "ensemble", "risk", "actor", "critic")
        }

    def _step(self, name: str, loss: Tensor) -> float:
        optimizer = self.optimizers[name]
        optimizer.zero_grad(set_to_none=True)
        if loss.ndim != 0 or not torch.isfinite(loss).item():
            raise FloatingPointError(f"nonfinite or nonscalar {name} loss")
        loss.backward()
        parameters = [p for group in optimizer.param_groups for p in group["params"] if p.grad is not None]
        if not parameters:
            raise RuntimeError(f"{name} objective produced no parameter gradients")
        if not torch.stack([torch.isfinite(p.grad).all() for p in parameters]).all().item():
            optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError(f"nonfinite {name} gradients")
        norm = nn.utils.clip_grad_norm_(parameters, self.config["grad_clip"], error_if_nonfinite=True)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        return float(norm.detach())

    @staticmethod
    def _select(mask: Tensor, count: int) -> tuple[Tensor, Tensor]:
        candidates = mask.nonzero(as_tuple=False)
        if len(candidates) > count:
            candidates = candidates[torch.randperm(len(candidates), device=mask.device)[:count]]
        return candidates[:, 0], candidates[:, 1]

    def _start_mask(self, valid: Tensor, is_first: Tensor) -> Tensor:
        # Each replay chunk also starts without its true earlier recurrent
        # state. Burn-in is repeated after any reset or unavailable position.
        age = torch.full_like(valid[:, 0], -1, dtype=torch.long)
        usable = []
        for index in range(valid.shape[1]):
            age = torch.where(is_first[:, index], 0, age + 1)
            age = torch.where(valid[:, index], age, -1)
            usable.append(valid[:, index] & (age >= self.config["burnin"]))
        return torch.stack(usable, 1)

    @staticmethod
    def _gather_state(rollout, rows: Tensor, indices: Tensor) -> RSSMState:
        h = torch.stack([s.h for s in rollout.states], 1)
        z = torch.stack([s.z for s in rollout.states], 1)
        return RSSMState(h[rows, indices].detach(), z[rows, indices].detach())

    def _auxiliary_update(self, rollout, batch: dict, history: Tensor,
                          history_mask: Tensor, starts_valid: Tensor) -> dict[str, float]:
        candidates = starts_valid.clone()
        candidates[:, -1] = False
        rows, starts = self._select(candidates, self.config["aux_batch"])
        empty = {"ensemble_loss": 0.0, "ensemble_grad_norm": 0.0, "risk_loss": 0.0,
                 "risk_grad_norm": 0.0, "risk_brier": 0.0, "risk_event_prevalence": 0.0,
                 "aux_disagreement": 0.0, "risk_labels": 0.0, "aux_skipped": 1.0}
        if len(rows) == 0:
            return empty
        horizon = self.config["horizon"]
        offsets = torch.arange(horizon, device=rows.device)
        action_indices = (starts[:, None] + offsets).clamp_max(batch["actions"].shape[1] - 1)
        endpoint_indices = (starts[:, None] + offsets + 1).clamp_max(batch["valid"].shape[1] - 1)
        actions = batch["actions"][rows[:, None], action_indices].detach()
        prefix_valid = same_episode_prefix_mask(
            batch["valid"], batch["is_first"], rows, starts, horizon,
        )
        if not prefix_valid.any().item():
            return empty
        with torch.no_grad():
            state = self._gather_state(rollout, rows, starts)
            dream = self.model.imagine(state, actions)
            hidden = torch.stack([s.h for s in dream.states[1:]], 1)
            features = torch.stack([s.features() for s in dream.states[1:]], 1)
            dreamed_rewards = self.model.reward_head(features).squeeze(-1)
            target_logits = rollout.posterior_logits[rows[:, None], endpoint_indices].detach()
            actual_rewards = batch["rewards"][rows[:, None], endpoint_indices]
            labels, prefix_valid = make_risk_targets(
                actual_rewards, dreamed_rewards, prefix_valid,
                reward_scale=self.config["reward_scale"], threshold=self.config["risk_error_threshold"],
            )
        bootstrap = torch.rand(
            len(self.ensemble.members), len(rows), horizon, device=rows.device,
        ) < self.config["ensemble_bootstrap_prob"]
        ensemble_loss = self.ensemble.bootstrap_loss(hidden, target_logits, bootstrap, prefix_valid)
        ensemble_grad = 0.0
        if (bootstrap & prefix_valid[None]).any().item():
            ensemble_grad = self._step("ensemble", ensemble_loss)
        with torch.no_grad():
            disagreement = ensemble_disagreement(self.ensemble(hidden))
        logits = self.risk_head(state.features(), history[rows, starts], history_mask[rows, starts],
                                actions, disagreement)
        risk_loss = risk_bce_loss(logits, labels, prefix_valid)
        risk_grad = self._step("risk", risk_loss)
        with torch.no_grad():
            brier = (logits.sigmoid() - labels).square()[prefix_valid].mean()
        return {
            "ensemble_loss": float(ensemble_loss.detach()), "ensemble_grad_norm": ensemble_grad,
            "risk_loss": float(risk_loss.detach()), "risk_grad_norm": risk_grad,
            "risk_brier": float(brier), "risk_event_prevalence": float(labels[prefix_valid].mean()),
            "aux_disagreement": float(disagreement[prefix_valid].mean()),
            "risk_labels": float(prefix_valid.sum()), "aux_skipped": 0.0,
        }

    def control_update(self, start: RSSMState, history: Tensor,
                       history_mask: Tensor) -> dict[str, float]:
        """Policy/value optimization only; no gradients enter model or risk.

        This public boundary makes gradient isolation and zero-prefix behavior
        inspectable independently of the world-model and auxiliary updates.
        """
        c = self.config
        with torch.no_grad():
            state = start.detach()
            states, actions, pre_tanh_samples = [state], [], []
            for _ in range(c["horizon"]):
                action, _, pre_tanh = self.actor.sample_with_pre_tanh(state.features())
                state, _ = self.model.imagine_step(state, action)
                actions.append(action)
                pre_tanh_samples.append(pre_tanh)
                states.append(state)
            features = torch.stack([s.features() for s in states], 1)
            actions = torch.stack(actions, 1)
            pre_tanh_samples = torch.stack(pre_tanh_samples, 1)
            hidden = torch.stack([s.h for s in states[1:]], 1)
            disagreement = ensemble_disagreement(self.ensemble(hidden))
            risks = self.risk_head(start.features(), history, history_mask, actions, disagreement).sigmoid()
            horizons = trusted_horizon(risks, c["risk_threshold"])
            rewards = self.model.reward_head(features[:, 1:]).squeeze(-1)
            continuation = self.model.continuation_head(features[:, 1:]).squeeze(-1).sigmoid()
            values = self.target_critic(features)
            targets, accepted = lambda_returns(rewards, continuation, values, horizons,
                                               gamma=c["gamma"], lambda_=c["lambda_"])
            weights = torch.cat((torch.ones_like(continuation[:, :1]),
                                 (c["gamma"] * continuation[:, :-1]).cumprod(1)), 1)
            weights = weights * accepted
            advantages = targets - values[:, :-1]
        metrics = {
            "trusted_horizon": float(horizons.float().mean()),
            "trusted_horizon_zero_fraction": float((horizons == 0).float().mean()),
            "risk_probability": float(risks.mean()), "imagination_disagreement": float(disagreement.mean()),
            "accepted_imagination_steps": float(accepted.sum()), "actor_loss": 0.0,
            "critic_loss": 0.0, "actor_grad_norm": 0.0, "critic_grad_norm": 0.0,
            "actor_entropy_mc": 0.0, "actor_skipped": 1.0, "critic_skipped": 1.0,
        }
        if not accepted.any().item():
            return metrics
        # Gather accepted cells before objective arithmetic. Rejected tails
        # must not leak through a 0 * overflowing-finite-expression product.
        policy_features = features[:, :-1][accepted].detach()
        policy_pre_tanh = pre_tanh_samples[accepted].detach()
        policy_weights = weights[accepted]
        log_prob = self.actor.log_prob_from_pre_tanh(policy_features, policy_pre_tanh)
        entropy = self.actor.entropy_estimate(policy_features)
        # Normalize by the accepted sample count; continuation discounting
        # remains in the weights instead of cancelling in their normalization.
        denominator = accepted.sum().clamp_min(1)
        actor_loss = -(policy_weights * (log_prob * advantages[accepted].detach()
                                        + c["entropy_scale"] * entropy)).sum() / denominator
        actor_grad = self._step("actor", actor_loss)
        prediction = self.critic(policy_features)
        critic_loss = (policy_weights * 0.5 * (prediction - targets[accepted].detach()).square()).sum() / denominator
        critic_grad = self._step("critic", critic_loss)
        with torch.no_grad():
            for target, current in zip(self.target_critic.parameters(), self.critic.parameters()):
                target.lerp_(current, c["target_tau"])
        metrics.update({"actor_loss": float(actor_loss.detach()), "critic_loss": float(critic_loss.detach()),
                        "actor_grad_norm": actor_grad, "critic_grad_norm": critic_grad,
                        "actor_entropy_mc": float(entropy.detach().mean()),
                        "actor_skipped": 0.0, "critic_skipped": 0.0})
        return metrics

    def update(self, batch: dict[str, Tensor], update_step: int) -> dict[str, float]:
        """One update; returns finite scalar metrics, raising on invalid data."""
        required = {"observations", "actions", "rewards", "continuations", "valid", "is_first",
                    "errors", "error_valid", "model_versions"}
        if required - batch.keys():
            raise ValueError(f"missing replay fields: {sorted(required - batch.keys())}")
        if batch["model_versions"].shape != batch["valid"].shape:
            raise ValueError("model_versions must align with observation indices")
        if batch["model_versions"].dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
            raise ValueError("model_versions must contain integer collection-time versions")
        if (batch["model_versions"][batch["error_valid"]] < 0).any().item():
            raise ValueError("available observed errors require collection-time model versions")
        starts_valid = self._start_mask(batch["valid"], batch["is_first"])
        if not starts_valid.any().item():
            raise ValueError("replay sequence has no valid observations after burn-in")
        history, history_mask = causal_error_history(
            batch["errors"], batch["error_valid"] & batch["valid"], batch["is_first"],
            window=self.config["history_window"],
        )
        rollout = self.model.observe(batch["observations"], batch["actions"], batch["is_first"])
        losses = pixel_world_model_loss(
            self.model, rollout, batch["observations"], batch["rewards"], batch["continuations"],
            starts_valid, batch["is_first"],
        )
        correction = fast_cross_correction_loss(
            self.model, rollout, batch["actions"], batch["is_first"], starts_valid,
            max_horizon=self.config["cc_horizon"],
        )
        total = losses["total"] + self.config["cc_scale"] * correction
        metrics = {f"model_{key}": float(value.detach()) for key, value in losses.items()}
        metrics.update({"model_loss": float(total.detach()), "cross_correction_loss": float(correction.detach()),
                        "model_grad_norm": self._step("model", total), "update_step": float(update_step)})
        # Refresh detached beliefs after the model update so auxiliary labels
        # and dreamed trajectories use the same current parameter version.
        with torch.no_grad():
            rollout = self.model.observe(batch["observations"], batch["actions"], batch["is_first"])
        metrics.update(self._auxiliary_update(rollout, batch, history, history_mask, starts_valid))
        policy_starts = starts_valid & (batch["continuations"] > 0)
        rows, indices = self._select(policy_starts, self.config["imagination_batch"])
        if len(rows):
            start = self._gather_state(rollout, rows, indices)
            metrics.update(self.control_update(start, history[rows, indices], history_mask[rows, indices]))
        else:
            metrics.update({"actor_skipped": 1.0, "critic_skipped": 1.0,
                            "trusted_horizon": 0.0, "accepted_imagination_steps": 0.0})
        if not all(math.isfinite(value) for value in metrics.values()):
            raise FloatingPointError("nonfinite update metrics")
        return metrics
