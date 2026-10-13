"""Pixel capacity, reset alignment, and reference-equation parity checks."""

import copy
import unittest

import torch

from helixwm.losses import cross_correction_loss, world_model_loss
from helixwm.pixels import PixelRSSM, fast_cross_correction_loss, pixel_world_model_loss
from helixwm.rssm import ObservationRollout, RSSM, RSSMState


class PixelModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        torch.manual_seed(17)

    def _pixels(self):
        return PixelRSSM(action_dim=2, hidden_dim=8, groups=2, classes=3, embed_dim=8, cnn_depth=2)

    def _vector(self):
        return RSSM(3, 2, hidden_dim=8, groups=2, classes=3, embed_dim=8).double()

    def _compare_gradients(self, first, second):
        for (name, left), (other_name, right) in zip(first.named_parameters(), second.named_parameters()):
            self.assertEqual(name, other_name)
            self.assertEqual(left.grad is None, right.grad is None, name)
            if left.grad is not None:
                torch.testing.assert_close(left.grad, right.grad, rtol=1e-9, atol=1e-11, msg=name)

    def test_pixel_shapes_and_finite_backward(self):
        model = self._pixels()
        observations = torch.rand(2, 4, model.obs_dim) - 0.5
        actions = torch.randn(2, 3, 2)
        first = torch.zeros(2, 4, dtype=torch.bool)
        first[:, 0] = True
        valid = torch.ones_like(first)
        rollout = model.observe(observations, actions, first, sample=True)
        self.assertEqual(rollout.posterior_logits.shape, (2, 4, 2, 3))
        self.assertEqual(model.decode(rollout.states[-1])["observation"].shape, (2, 64 * 64 * 3))
        losses = pixel_world_model_loss(
            model, rollout, observations, torch.randn(2, 4), torch.ones(2, 4), valid, first
        )
        correction = fast_cross_correction_loss(model, rollout, actions, first, valid, sample=True)
        total = losses["total"] + correction
        self.assertTrue(torch.isfinite(total).item())
        total.backward()
        self.assertTrue(all(parameter.grad is not None for parameter in model.parameters()))
        self.assertTrue(all(torch.isfinite(parameter.grad).all().item() for parameter in model.parameters()))
        self.assertGreater(sum(parameter.grad.abs().sum().item() for parameter in model.encoder.parameters()), 0)
        clone = PixelRSSM(**model.constructor_config)
        clone.load_state_dict(model.state_dict(), strict=True)

    def test_preencoded_observe_matches_reference_and_step_interface(self):
        model = self._pixels()
        observations = torch.rand(2, 4, model.obs_dim) - 0.5
        actions = torch.randn(2, 3, 2)
        first = torch.zeros(2, 4, dtype=torch.bool)
        first[:, 0] = True
        first[1, 2] = True
        fast = model.observe(observations, actions, first, sample=False)
        reference = RSSM.observe(model, observations, actions, first, sample=False)
        torch.testing.assert_close(fast.prior_logits, reference.prior_logits, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(fast.posterior_logits, reference.posterior_logits, rtol=1e-5, atol=1e-6)
        for actual, expected in zip(fast.states, reference.states):
            torch.testing.assert_close(actual.h, expected.h, rtol=1e-5, atol=1e-6)
            torch.testing.assert_close(actual.z, expected.z, rtol=0, atol=0)

    def test_pixel_reset_discards_history_and_transition_action(self):
        model = self._pixels()
        observations = torch.rand(2, 4, model.obs_dim) - 0.5
        actions = torch.randn(2, 3, 2)
        first = torch.zeros(2, 4, dtype=torch.bool)
        first[:, [0, 2]] = True
        original = model.observe(observations, actions, first, sample=False)
        observations[:, :2] += 50
        actions[:, :2] -= 100
        changed = model.observe(observations, actions, first, sample=False)
        torch.testing.assert_close(original.prior_logits[:, 2:], changed.prior_logits[:, 2:], rtol=0, atol=0)
        torch.testing.assert_close(original.posterior_logits[:, 2:], changed.posterior_logits[:, 2:], rtol=0, atol=0)

    def test_batched_base_loss_matches_reference_values_and_gradients(self):
        model = self._vector()
        fast_model = copy.deepcopy(model)
        observations = torch.randn(2, 6, 3, dtype=torch.float64)
        actions = torch.randn(2, 5, 2, dtype=torch.float64)
        rewards = torch.randn(2, 6, dtype=torch.float64)
        continuations = torch.randint(2, (2, 6)).double()
        first = torch.zeros(2, 6, dtype=torch.bool)
        first[:, 0] = True
        first[1, 3] = True
        valid = torch.ones_like(first)
        valid[0, 4:] = False
        reference = world_model_loss(
            model, model.observe(observations, actions, first, sample=False),
            observations, rewards, continuations, valid, first
        )
        actual = pixel_world_model_loss(
            fast_model, fast_model.observe(observations, actions, first, sample=False),
            observations, rewards, continuations, valid, first
        )
        for name in reference:
            torch.testing.assert_close(actual[name], reference[name], rtol=1e-10, atol=1e-12)
        reference["total"].backward()
        actual["total"].backward()
        self._compare_gradients(model, fast_model)

    def test_fast_correction_matches_reference_values_and_gradients(self):
        for horizon in (1, 2, 3, 5, 10):
            with self.subTest(horizon=horizon):
                model = self._vector()
                fast_model = copy.deepcopy(model)
                observations = torch.randn(3, 7, 3, dtype=torch.float64)
                actions = torch.randn(3, 6, 2, dtype=torch.float64, requires_grad=True)
                fast_actions = actions.detach().clone().requires_grad_()
                first = torch.zeros(3, 7, dtype=torch.bool)
                first[:, 0] = True
                first[1, 3] = True
                valid = torch.ones_like(first)
                valid[0, 5:] = False
                valid[2, 1] = False
                reference = cross_correction_loss(
                    model, model.observe(observations, actions, first, sample=False),
                    actions, first, valid, max_horizon=horizon, sample=False
                )
                actual = fast_cross_correction_loss(
                    fast_model, fast_model.observe(observations, fast_actions, first, sample=False),
                    fast_actions, first, valid, max_horizon=horizon, sample=False
                )
                torch.testing.assert_close(actual, reference, rtol=1e-10, atol=1e-12)
                reference.backward()
                actual.backward()
                self._compare_gradients(model, fast_model)
                self.assertEqual(actions.grad is None, fast_actions.grad is None)
                if actions.grad is not None:
                    torch.testing.assert_close(actions.grad, fast_actions.grad, rtol=1e-9, atol=1e-11)

    def test_fast_correction_stops_teacher_and_start_gradients(self):
        model = self._vector()
        states = tuple(
            RSSMState(
                torch.randn(2, 8, dtype=torch.float64, requires_grad=True),
                torch.randn(2, 2, 3, dtype=torch.float64, requires_grad=True),
            )
            for _ in range(6)
        )
        teacher = torch.randn(2, 6, 2, 3, dtype=torch.float64, requires_grad=True)
        rollout = ObservationRollout(states, torch.zeros_like(teacher), teacher)
        first = torch.zeros(2, 6, dtype=torch.bool)
        first[:, 0] = True
        loss = fast_cross_correction_loss(
            model, rollout, torch.randn(2, 5, 2, dtype=torch.float64),
            first, torch.ones_like(first), sample=False
        )
        loss.backward()
        self.assertIsNone(teacher.grad)
        for state in states:
            self.assertIsNone(state.h.grad)
            self.assertIsNone(state.z.grad)
        self.assertGreater(sum(parameter.grad.abs().sum().item() for parameter in model.prior.parameters()), 0)

    def test_fast_correction_all_invalid_and_short_sequences(self):
        model = self._vector()
        for length in (1, 2, 5):
            observations = torch.randn(2, length, 3, dtype=torch.float64)
            actions = torch.randn(2, length - 1, 2, dtype=torch.float64)
            first = torch.zeros(2, length, dtype=torch.bool)
            first[:, 0] = True
            rollout = model.observe(observations, actions, first, sample=False)
            loss = fast_cross_correction_loss(model, rollout, actions, first, torch.zeros_like(first))
            self.assertEqual(loss.item(), 0)
            self.assertTrue(loss.requires_grad)
            loss.backward()

    def test_pixel_base_loss_preserves_reset_and_padding_masks(self):
        model = self._pixels()
        observations = torch.rand(2, 5, model.obs_dim) - 0.5
        actions = torch.randn(2, 4, 2)
        first = torch.zeros(2, 5, dtype=torch.bool)
        first[:, [0, 3]] = True
        valid = torch.ones_like(first)
        valid[:, 4] = False
        rewards = torch.randn(2, 5)
        continuations = torch.ones(2, 5)
        rollout = model.observe(observations, actions, first, sample=False)
        before = pixel_world_model_loss(model, rollout, observations, rewards, continuations, valid, first)
        observations[:, 4] = 100
        rewards[:, [0, 3, 4]] = 100
        continuations[:, [0, 3, 4]] = 0
        after = pixel_world_model_loss(model, rollout, observations, rewards, continuations, valid, first)
        for name in before:
            torch.testing.assert_close(before[name], after[name], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
