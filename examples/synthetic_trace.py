"""Run the formula pipeline on synthetic tensors; no environment or training.

From the repository root: python -m examples.synthetic_trace
No optimizer step is taken and no weights, videos or experiment results are saved.
"""

import json

import torch

from helixwm.losses import cross_correction_loss, world_model_loss
from helixwm.reliability import (
    PredictionEnsemble, RiskHead, causal_error_history, ensemble_disagreement,
    make_risk_targets, observed_prediction_errors, risk_bce_loss,
)
from helixwm.returns import lambda_returns, trusted_horizon
from helixwm.rssm import RSSM


def main() -> None:
    torch.manual_seed(0)
    torch.set_num_threads(1)
    batch, steps, obs_dim, action_dim = 2, 6, 5, 2
    model = RSSM(obs_dim, action_dim)
    observations = torch.randn(batch, steps + 1, obs_dim)
    actions = torch.tanh(torch.randn(batch, steps, action_dim))
    rewards = torch.randn(batch, steps + 1)
    continuations = torch.ones_like(rewards)
    valid = torch.ones_like(rewards, dtype=torch.bool)
    is_first = torch.zeros_like(valid)
    is_first[:, 0] = True
    real = model.observe(observations, actions, is_first, sample=False)

    base = world_model_loss(model, real, observations, rewards, continuations, valid, is_first)
    correction = cross_correction_loss(model, real, actions, is_first, valid, max_horizon=3)

    # Replay reconstruction of prospective predictions from each prefix only.
    # All predictions and targets below belong to this unchanged model version.
    saved_priors, saved_rewards = [], []
    with torch.no_grad():
        for t in range(steps):
            one_step = model.imagine(real.states[t].detach(), actions[:, t:t+1], sample=False)
            saved_priors.append(one_step.prior_logits[:, 0])
            saved_rewards.append(model.decode(one_step.states[1])["reward"])
    errors = observed_prediction_errors(
        real.posterior_logits[:, 1:], torch.stack(saved_priors, 1),
        rewards[:, 1:], torch.stack(saved_rewards, 1), reward_scale=1.0,
    )
    errors = torch.cat((torch.zeros(batch, 1, 2), errors), dim=1)
    error_valid = valid.clone()
    error_valid[:, 0] = False  # A reset has no within-episode transition error.
    history, history_mask = causal_error_history(errors, error_valid, is_first, window=3)

    start, length = 2, 3
    candidate_actions = actions[:, start:start+length]
    dream = model.imagine(real.states[start].detach(), candidate_actions, sample=False)
    dream_hidden = torch.stack([state.h for state in dream.states[1:]], dim=1)
    dream_rewards = torch.stack([model.decode(state)["reward"] for state in dream.states[1:]], 1)
    ensemble = PredictionEnsemble(model.hidden_dim, model.groups, model.classes)
    selection = torch.rand(3, batch, length) < 0.8  # Independent member bootstrap selections.
    ensemble_loss = ensemble.bootstrap_loss(
        dream_hidden, real.posterior_logits[:, start+1:start+length+1], selection,
    )
    disagreement = ensemble_disagreement(ensemble(dream_hidden))
    belief = real.states[start].features()
    risk_head = RiskHead(belief.shape[-1], action_dim)
    logits = risk_head(belief, history[:, start], history_mask[:, start], candidate_actions, disagreement)
    # Labels use the SAME recorded action sequence; never pair with another policy.
    labels, label_mask = make_risk_targets(
        rewards[:, start+1:start+length+1], dream_rewards,
        torch.ones(batch, length, dtype=torch.bool), threshold=0.25,
    )
    calibration_loss = risk_bce_loss(logits, labels, label_mask)
    total = base["total"] + 0.1 * correction + ensemble_loss + calibration_loss
    total.backward()  # Gradient check only; no optimizer exists in this demo.
    gradients = [p.grad for net in (model, ensemble, risk_head) for p in net.parameters() if p.grad is not None]
    if not gradients or not all(torch.isfinite(g).all() for g in gradients):
        raise RuntimeError("nonfinite or missing gradients")

    # A hand-specified risk sequence illustrates the gate independently of an
    # untrained risk head. The rejected third reward/value cannot enter targets.
    horizon = trusted_horizon(torch.tensor([[0.1, 0.2, 0.9]]), threshold=0.5)
    targets, target_mask = lambda_returns(
        torch.tensor([[1., 1., 9999.]], dtype=torch.float64),
        torch.ones(1, 3, dtype=torch.float64),
        torch.tensor([[0., 0., 10., 9999.]], dtype=torch.float64),
        horizon, gamma=0.99, lambda_=1.0,
    )
    torch.testing.assert_close(targets[0, 0], torch.tensor(11.791, dtype=torch.float64))
    print(json.dumps({
        "scope": "synthetic math verification only; no RL experiment",
        "prediction_model_version": "unchanged_synthetic_initialization_seed0",
        "torch": torch.__version__,
        "base_loss": float(base["total"].detach()),
        "cross_correction_loss": float(correction.detach()),
        "ensemble_loss": float(ensemble_loss.detach()),
        "risk_bce": float(calibration_loss.detach()),
        "all_gradients_finite": True,
        "untrained_risk_probabilities": logits.detach().sigmoid().tolist(),
        "example_horizon": horizon.tolist(),
        "example_returns": targets.tolist(),
        "example_valid_mask": target_mask.tolist(),
        "optimizer_steps": 0,
    }, indent=2))


if __name__ == "__main__":
    main()
