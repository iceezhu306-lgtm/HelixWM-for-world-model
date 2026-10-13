"""Hand-computed finite-horizon return and rejection boundary tests."""
import unittest
import torch
from helixwm.returns import trusted_horizon, lambda_returns


class ReturnTests(unittest.TestCase):
    def test_horizon_is_contiguous_prefix_not_pass_count(self):
        risks = torch.tensor([[.1,.2,.1],[.1,.9,.1],[.9,.1,.1],[.1,.1,.9]])
        actual = trusted_horizon(risks, threshold=.2)
        torch.testing.assert_close(actual, torch.tensor([3,1,0,2]))
        self.assertEqual(actual.dtype, torch.long)

    def test_zero_one_and_two_step_hand_oracles(self):
        r = torch.tensor([[1.,2.,999999.]], dtype=torch.float64).repeat(3,1)
        c = torch.ones_like(r)
        v = torch.tensor([[10.,20.,30.,999999.]], dtype=torch.float64).repeat(3,1)
        targets, valid = lambda_returns(r,c,v,torch.tensor([0,1,2]),gamma=.9,lambda_=.8)
        expected = torch.tensor([[0.,0.,0.],[19.,0.,0.],[25.48,29.,0.]], dtype=r.dtype)
        torch.testing.assert_close(targets,expected,atol=1e-10,rtol=0)
        torch.testing.assert_close(valid,torch.tensor([[False,False,False],[True,False,False],[True,True,False]]))

    def test_rejected_tail_has_no_effect(self):
        r = torch.tensor([[1.,2.,1e12]], dtype=torch.float64)
        c = torch.tensor([[1.,1.,1.]], dtype=torch.float64)
        v = torch.tensor([[10.,20.,30.,1e12]], dtype=torch.float64)
        first, _ = lambda_returns(r,c,v,torch.tensor([2]),gamma=.9,lambda_=.8)
        r[0,2], c[0,2], v[0,3] = -1e12, 0., -1e12
        second, _ = lambda_returns(r,c,v,torch.tensor([2]),gamma=.9,lambda_=.8)
        torch.testing.assert_close(first,second,atol=0,rtol=0)

    def test_continuation_lambda_and_discount_boundaries(self):
        r = torch.tensor([[1.,2.,3.]], dtype=torch.float64)
        v = torch.tensor([[10.,20.,30.,40.]], dtype=torch.float64)
        cases = [
            ([0.,1.,1.], .9, .8, [1.,29.,0.]),
            ([1.,0.,1.], .9, .8, [6.04,2.,0.]),
            ([1.,1.,1.], .9, 0., [19.,29.,0.]),
            ([1.,1.,1.], .9, 1., [27.1,29.,0.]),
            ([1.,1.,1.], 0., .8, [1.,2.,0.]),
        ]
        for cont,gamma,lam,expected in cases:
            with self.subTest(cont=cont,gamma=gamma,lam=lam):
                actual,_ = lambda_returns(r,torch.tensor([cont],dtype=r.dtype),v,torch.tensor([2]),gamma=gamma,lambda_=lam)
                torch.testing.assert_close(actual,torch.tensor([expected],dtype=r.dtype),atol=1e-10,rtol=0)

    def test_nonfinite_even_in_rejected_tail_is_explicit_error(self):
        with self.assertRaises(ValueError):
            lambda_returns(torch.tensor([[1.,float('nan')]]),torch.ones(1,2),torch.zeros(1,3),torch.tensor([1]))


if __name__ == '__main__':
    unittest.main()
