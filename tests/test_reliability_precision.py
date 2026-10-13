"""Regression tests for finite half-precision inputs that overflow subtraction."""

import math
import unittest

import torch

from helixwm.reliability import (
    PredictionEnsemble,
    ensemble_disagreement,
    make_risk_targets,
    observed_prediction_errors,
)


class ReliabilityPrecisionTests(unittest.TestCase):
    def test_ensemble_js_promotes_finite_extreme_half_logits(self):
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                logits = torch.tensor(
                    [[60000.0, -60000.0], [-60000.0, 60000.0]], dtype=dtype,
                ).reshape(2, 1, 1, 1, 2)
                actual = ensemble_disagreement(logits)
                self.assertEqual(actual.dtype, torch.float32)
                self.assertTrue(torch.isfinite(actual).all())
                torch.testing.assert_close(actual, torch.full((1, 1), math.log(2.0)))

    def test_observed_errors_promote_before_js_and_reward_subtraction(self):
        logits = torch.tensor([[[[60000.0, -60000.0]]]], dtype=torch.float16)
        reward = torch.tensor([[60000.0]], dtype=torch.float16)
        errors = observed_prediction_errors(logits, -logits, reward, -reward, 60000.0)
        self.assertEqual(errors.dtype, torch.float32)
        self.assertTrue(torch.isfinite(errors).all())
        torch.testing.assert_close(errors, torch.tensor([[[math.log(2.0), 2.0]]]))

    def test_risk_target_normalization_occurs_without_half_overflow(self):
        reward = torch.tensor([[60000.0]], dtype=torch.float16)
        valid = torch.ones((1, 1), dtype=torch.bool)
        labels, mask = make_risk_targets(reward, -reward, valid, 60000.0, 3.0)
        self.assertEqual(labels.item(), 0.0)  # true normalized error is 2, not infinity
        self.assertTrue(mask.item())
        labels, _ = make_risk_targets(reward, -reward, valid, 60000.0, 1.0)
        self.assertEqual(labels.item(), 1.0)

    def test_bootstrap_kl_promotes_half_logits_and_has_finite_gradient(self):
        ensemble = PredictionEnsemble(1, 1, 2, members=2, hidden_dim=2).half()
        with torch.no_grad():
            for member in ensemble.members:
                for parameter in member.parameters():
                    parameter.zero_()
                member[-1].bias.copy_(torch.tensor([60000.0, -60000.0]))
        features = torch.zeros((1, 1, 1), dtype=torch.float16, requires_grad=True)
        teacher = torch.tensor([[[[-60000.0, 60000.0]]]],
                               dtype=torch.float16, requires_grad=True)
        mask = torch.ones((2, 1, 1), dtype=torch.bool)
        loss = ensemble.bootstrap_loss(features, teacher, mask)
        self.assertEqual(loss.dtype, torch.float32)
        self.assertAlmostEqual(loss.item(), 120000.0, places=2)
        loss.backward()
        self.assertIsNone(features.grad)
        self.assertIsNone(teacher.grad)
        for parameter in ensemble.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())


if __name__ == "__main__":
    unittest.main()
