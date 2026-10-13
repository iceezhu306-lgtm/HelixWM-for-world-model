"""Controller isolation, terminal boundaries, and an actual small update."""

import copy
import math
import unittest

import torch

from helixwm.control import HelixLearner, TanhGaussianActor, same_episode_prefix_mask
from helixwm.pixels import PixelRSSM
from helixwm.rssm import RSSM, RSSMState


class ControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        torch.manual_seed(142)

    @staticmethod
    def learner(pixel=False, **overrides):
        constructor = PixelRSSM if pixel else RSSM
        model_kwargs = {"action_dim": 2, "hidden_dim": 8, "groups": 2, "classes": 3, "embed_dim": 8}
        model_kwargs.update({"cnn_depth": 2} if pixel else {"obs_dim": 5})
        model = constructor(**model_kwargs)
        config = {"burnin": 1, "horizon": 3, "cc_horizon": 3, "history_window": 2,
                  "actor_hidden": 16, "critic_hidden": 16, "reliability_hidden": 8,
                  "aux_batch": 6, "imagination_batch": 6, **overrides}
        return HelixLearner(model, config)

    @staticmethod
    def fix_risk(learner, high):
        with torch.no_grad():
            learner.risk_head.output.weight.zero_()
            learner.risk_head.output.bias.fill_(20.0 if high else -20.0)

    @staticmethod
    def batch(learner):
        batch_size, count = 2, 6
        valid = torch.ones(batch_size, count, dtype=torch.bool)
        first = torch.zeros_like(valid)
        first[:, 0] = True
        error_valid = valid & ~first
        return {
            "observations": torch.rand(batch_size, count, learner.model.obs_dim) - .5,
            "actions": torch.rand(batch_size, count - 1, learner.model.action_dim) * 2 - 1,
            "rewards": torch.rand(batch_size, count) * 2,
            "continuations": torch.ones(batch_size, count),
            "valid": valid, "is_first": first, "errors": torch.rand(batch_size, count, 2),
            "error_valid": error_valid,
            "model_versions": torch.where(error_valid, 0, -1),
        }

    def test_sample_uses_score_function_without_action_path_gradient(self):
        actor = TanhGaussianActor(5, 2, 16)
        features = torch.randn(4, 5)
        actions, log_prob, pre_tanh = actor.sample_with_pre_tanh(features)
        self.assertFalse(actions.requires_grad)
        self.assertFalse(pre_tanh.requires_grad)
        self.assertEqual(log_prob.shape, (4,))
        torch.testing.assert_close(log_prob, actor.log_prob_from_pre_tanh(features, pre_tanh), rtol=0, atol=0)
        loss = -(log_prob * torch.tensor([1.0, -2.0, .5, 3.0])).mean()
        loss.backward()
        gradients = [p.grad for p in actor.parameters() if p.grad is not None]
        self.assertGreater(sum(g.abs().sum().item() for g in gradients), 0)
        self.assertTrue(all(torch.isfinite(g).all().item() for g in gradients))
        # A caller cannot accidentally create an action-path derivative by
        # handing a differentiable stored action to log_prob.
        differentiable_action = actions.clone().requires_grad_()
        actor.zero_grad(set_to_none=True)
        actor.log_prob(features, differentiable_action).sum().backward()
        self.assertIsNone(differentiable_action.grad)

    def test_saturated_samples_preserve_original_gaussian_score(self):
        actor = TanhGaussianActor(1, 1, 2)
        with torch.no_grad():
            for parameter in actor.parameters():
                parameter.zero_()
            actor.log_std.bias.fill_(1.9)
        features = torch.zeros(100000, 1)
        actions, log_probability, pre_tanh = actor.sample_with_pre_tanh(features)
        # This scale regularly saturates the old inverse clamp. The unchanged
        # original draw must supply the density despite tanh rounding.
        self.assertGreater(float((actions.abs() > 1 - 1e-6).float().mean()), 0.2)
        reconstructed = actor.log_prob_from_pre_tanh(features, pre_tanh)
        torch.testing.assert_close(log_probability, reconstructed, rtol=0, atol=0)
        distribution = actor._distribution(features)
        original_normal_score = torch.autograd.grad(
            distribution.log_prob(pre_tanh).sum(-1).mean(), actor.log_std.bias,
        )[0]
        actual_score = torch.autograd.grad(reconstructed.mean(), actor.log_std.bias)[0]
        # With a detached draw, the Jacobian has no parameter gradient: the
        # transformed score must equal the exact original Normal sample score.
        torch.testing.assert_close(actual_score, original_normal_score, atol=0, rtol=0)
        self.assertLess(float(actual_score.abs().max()), 0.02)
        # The previous inverse-clamp algorithm creates a large negative score.
        approximate = actor.log_prob(features, actions)
        old_score = torch.autograd.grad(approximate.mean(), actor.log_std.bias)[0]
        self.assertLess(float(old_score), -0.3)

    def test_retained_draw_is_detached_for_later_score_updates(self):
        actor = TanhGaussianActor(5, 2, 16)
        features = torch.randn(4, 5)
        pre_tanh = torch.randn(4, 2, requires_grad=True)
        actor.log_prob_from_pre_tanh(features, pre_tanh).sum().backward()
        self.assertIsNone(pre_tanh.grad)
        self.assertTrue(any(parameter.grad is not None for parameter in actor.parameters()))

    def test_control_update_never_inverts_rounded_actions(self):
        learner = self.learner()
        self.fix_risk(learner, high=False)

        def forbidden_inverse(*args, **kwargs):
            raise AssertionError("controller attempted an inverse-tanh score reconstruction")

        learner.actor.log_prob = forbidden_inverse
        metrics = learner.control_update(learner.model.initial(3), torch.zeros(3, 2, 2),
                                         torch.ones(3, 2, dtype=torch.bool))
        self.assertEqual(metrics["actor_skipped"], 0)
        self.assertTrue(math.isfinite(metrics["actor_loss"]))

    def test_deterministic_actions_and_entropy_are_finite(self):
        actor = TanhGaussianActor(5, 2, 16)
        features = torch.randn(10, 5)
        first, _ = actor.sample(features, deterministic=True)
        second, _ = actor.sample(features, deterministic=True)
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        torch.testing.assert_close(first, actor._distribution(features).mean.tanh())
        entropy = actor.entropy_estimate(features)
        self.assertTrue(torch.isfinite(entropy).all())
        (-entropy.mean()).backward()
        self.assertGreater(sum(p.grad.abs().sum().item() for p in actor.parameters()), 0)

    def test_boundary_mask_ends_at_reset_missing_or_sequence_end(self):
        valid = torch.tensor([[True, True, True, True, True, True],
                              [True, True, False, True, True, True]])
        first = torch.tensor([[True, False, False, True, False, False],
                              [True, False, False, False, False, False]])
        result = same_episode_prefix_mask(valid, first, torch.tensor([0, 0, 1, 0]),
                                          torch.tensor([0, 3, 0, 5]), 4)
        expected = torch.tensor([[True, True, False, False], [True, True, False, False],
                                 [True, False, False, False], [False, False, False, False]])
        torch.testing.assert_close(result, expected)

    def test_burnin_restarts_after_episode_boundary_and_padding(self):
        learner = self.learner(burnin=2)
        valid = torch.tensor([[True, True, True, True, True, True, True, True, True]])
        first = torch.tensor([[False, False, False, True, False, False, False, False, False]])
        valid[0, 6] = False
        result = learner._start_mask(valid, first)
        torch.testing.assert_close(result, torch.tensor([[False, False, True, False, False,
                                                        True, False, False, False]]))

    def test_control_step_updates_only_actor_critic_and_target(self):
        learner = self.learner()
        self.fix_risk(learner, high=False)
        protected = {name: copy.deepcopy(getattr(learner, name).state_dict())
                     for name in ("model", "ensemble", "risk_head")}
        actor_before = copy.deepcopy(learner.actor.state_dict())
        start = RSSMState(torch.randn(4, 8, requires_grad=True),
                          torch.randn(4, 2, 3, requires_grad=True))
        history = torch.zeros(4, 2, 2, requires_grad=True)
        metrics = learner.control_update(start, history, torch.ones(4, 2, dtype=torch.bool))
        self.assertEqual(metrics["actor_skipped"], 0)
        self.assertGreater(metrics["actor_grad_norm"], 0)
        self.assertGreater(metrics["critic_grad_norm"], 0)
        self.assertEqual(metrics["trusted_horizon"], 3)
        self.assertIsNone(start.h.grad)
        self.assertIsNone(start.z.grad)
        self.assertIsNone(history.grad)
        self.assertTrue(any(not torch.equal(value, learner.actor.state_dict()[name])
                            for name, value in actor_before.items()))
        for name, previous in protected.items():
            module = getattr(learner, name)
            for key, value in module.state_dict().items():
                torch.testing.assert_close(value, previous[key], atol=0, rtol=0)
            self.assertTrue(all(p.grad is None for p in module.parameters()))
        self.assertTrue(all(p.grad is None for p in learner.target_critic.parameters()))

    def test_zero_horizon_skips_optimizer_and_target_state_changes(self):
        learner = self.learner()
        self.fix_risk(learner, high=True)
        before = copy.deepcopy(learner.state_dict())
        metrics = learner.control_update(learner.model.initial(3), torch.zeros(3, 2, 2),
                                         torch.ones(3, 2, dtype=torch.bool))
        self.assertEqual(metrics["trusted_horizon"], 0)
        self.assertEqual(metrics["trusted_horizon_zero_fraction"], 1)
        self.assertEqual(metrics["actor_skipped"], 1)
        self.assertEqual(metrics["critic_skipped"], 1)
        self.assertEqual(len(learner.optimizers["actor"].state), 0)
        self.assertEqual(len(learner.optimizers["critic"].state), 0)
        for key, value in learner.state_dict().items():
            torch.testing.assert_close(value, before[key], atol=0, rtol=0)

    def test_actual_pixel_update_is_finite_and_checkpoint_includes_all_modules(self):
        learner = self.learner(pixel=True)
        self.fix_risk(learner, high=False)
        metrics = learner.update(self.batch(learner), update_step=1)
        self.assertTrue(all(math.isfinite(value) for value in metrics.values()))
        self.assertGreater(metrics["model_grad_norm"], 0)
        self.assertGreater(metrics["ensemble_grad_norm"], 0)
        self.assertGreater(metrics["risk_grad_norm"], 0)
        self.assertGreater(metrics["risk_labels"], 0)
        self.assertEqual(metrics["actor_skipped"], 0)
        restored = self.learner(pixel=True)
        restored.load_state_dict(learner.state_dict())
        for name, optimizer in learner.optimizers.items():
            restored.optimizers[name].load_state_dict(optimizer.state_dict())
            self.assertGreater(len(restored.optimizers[name].state), 0)
        for key, value in learner.state_dict().items():
            torch.testing.assert_close(value, restored.state_dict()[key], atol=0, rtol=0)

    def test_missing_error_provenance_is_rejected(self):
        learner = self.learner()
        batch = self.batch(learner)
        batch["model_versions"][0, 1] = -1
        with self.assertRaisesRegex(ValueError, "model versions"):
            learner.update(batch, 1)

    def test_nonfinite_gradient_is_rejected_without_parameter_step(self):
        learner = self.learner()
        parameter = next(learner.actor.parameters())
        before = parameter.detach().clone()
        handle = parameter.register_hook(lambda grad: grad * float("nan"))
        with self.assertRaisesRegex(FloatingPointError, "actor gradients"):
            learner._step("actor", parameter.square().mean())
        handle.remove()
        torch.testing.assert_close(before, parameter, atol=0, rtol=0)
        self.assertEqual(len(learner.optimizers["actor"].state), 0)


if __name__ == "__main__":
    unittest.main()
