"""Analytic distribution oracles, not policy-performance tests."""
import unittest
import torch
from helixwm.distributions import categorical_kl, categorical_js, straight_through_sample


class DistributionTests(unittest.TestCase):
    def test_kl_direction_and_sum_over_groups(self):
        q = torch.tensor([[[.75, .25], [.75, .25]]], dtype=torch.float64).log()
        p = torch.tensor([[[.5, .5], [.5, .5]]], dtype=torch.float64).log()
        self.assertAlmostEqual(categorical_kl(q, p).item(), .261624071882274, places=10)
        torch.testing.assert_close(categorical_kl(q, q), torch.zeros(1, dtype=q.dtype), atol=1e-12, rtol=0)
        torch.testing.assert_close(categorical_kl(q + 500, p - 500), categorical_kl(q, p), atol=1e-12, rtol=0)

    def test_js_is_symmetric_and_mean_over_groups(self):
        q = torch.tensor([[[.9, .1], [.9, .1]]], dtype=torch.float64).log()
        p = torch.tensor([[[.1, .9], [.1, .9]]], dtype=torch.float64).log()
        self.assertAlmostEqual(categorical_js(q, p).item(), .368064207168497, places=10)
        torch.testing.assert_close(categorical_js(q, p), categorical_js(p, q))
        torch.testing.assert_close(categorical_js(q, q), torch.zeros(1, dtype=q.dtype), atol=1e-12, rtol=0)
        extreme = torch.tensor([[[1000., -1000.]]])
        self.assertTrue(torch.isfinite(categorical_js(extreme, -extreme)).all().item())

    def test_straight_through_is_hard_forward_with_softmax_gradient(self):
        torch.manual_seed(31)
        logits = torch.zeros(1, 1, 3, dtype=torch.float64, requires_grad=True)
        sample = straight_through_sample(logits)
        self.assertTrue(((sample == 0) | (sample == 1)).all().item())
        torch.testing.assert_close(sample.sum(-1), torch.ones(1, 1, dtype=sample.dtype))
        (sample * torch.tensor([1., 2., 4.], dtype=sample.dtype)).sum().backward()
        torch.testing.assert_close(logits.grad, torch.tensor([[[-4/9, -1/9, 5/9]]], dtype=logits.dtype), atol=1e-12, rtol=0)


if __name__ == '__main__':
    unittest.main()
