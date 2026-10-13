"""Causal features, detached auxiliary supervision, and prefix-risk oracles."""
import unittest
import torch
from helixwm.reliability import (
    ensemble_disagreement, PredictionEnsemble, RiskHead, make_risk_targets,
    observed_prediction_errors, causal_error_history, risk_bce_loss,
)


class ReliabilityTests(unittest.TestCase):
    def test_observed_error_uses_saved_prediction_and_known_scale(self):
        q=torch.tensor([[[.9,.1],[.9,.1]]],dtype=torch.float64).log().requires_grad_()
        p=torch.tensor([[[.1,.9],[.1,.9]]],dtype=torch.float64).log().requires_grad_()
        errors=observed_prediction_errors(q,p,torch.tensor([2.],dtype=q.dtype),torch.tensor([1.],dtype=q.dtype),reward_scale=2.)
        torch.testing.assert_close(errors,torch.tensor([[.368064207168497,.5]],dtype=q.dtype),atol=1e-12,rtol=0)
        self.assertFalse(errors.requires_grad)

    def test_history_contains_received_error_but_no_future_or_previous_episode(self):
        errors=torch.tensor([[[1.,10.],[2.,20.],[3.,30.],[4.,40.]]])
        valid=torch.tensor([[True,True,False,True]])
        resets=torch.tensor([[True,False,True,False]])
        history,mask=causal_error_history(errors,valid,resets,window=2)
        expected=torch.tensor([[[[0.,0.],[1.,10.]],[[1.,10.],[2.,20.]],[[0.,0.],[0.,0.]],[[0.,0.],[4.,40.]]]])
        expected_mask=torch.tensor([[[False,True],[True,True],[False,False],[False,True]]])
        torch.testing.assert_close(history,expected,atol=0,rtol=0)
        torch.testing.assert_close(mask,expected_mask)
        altered=errors.clone(); altered[:,2:]+=9999.
        second,_=causal_error_history(altered,valid,resets,window=2)
        torch.testing.assert_close(history[:,:2],second[:,:2],atol=0,rtol=0)

    def test_unknown_labels_do_not_contribute_to_risk_loss(self):
        logits=torch.zeros(1,3,requires_grad=True)
        labels=torch.tensor([[0.,1.,0.]])
        valid=torch.tensor([[True,False,False]])
        loss=risk_bce_loss(logits,labels,valid)
        self.assertAlmostEqual(loss.item(),.6931471805599453,places=6)
        loss.backward()
        torch.testing.assert_close(logits.grad,torch.tensor([[.5,0.,0.]]),atol=0,rtol=0)

    def test_ensemble_js_hand_oracle(self):
        probs=torch.tensor([[[[[.9,.1],[.9,.1]]]],[[[[.1,.9],[.1,.9]]]]],dtype=torch.float64)
        actual=ensemble_disagreement(probs.log())
        torch.testing.assert_close(actual,torch.tensor([[.368064207168497]],dtype=probs.dtype),atol=1e-12,rtol=0)
        equal=probs[:1].expand(3,-1,-1,-1,-1)
        torch.testing.assert_close(ensemble_disagreement(equal.log()),torch.zeros(1,1,dtype=probs.dtype),atol=1e-12,rtol=0)

    def test_risk_targets_use_prefix_max_not_sum(self):
        actual=torch.zeros(2,3)
        predicted=torch.tensor([[.15,.15,.15],[.1,.3,.1]])
        labels,valid=make_risk_targets(actual,predicted,torch.ones(2,3,dtype=torch.bool),threshold=.25)
        torch.testing.assert_close(labels,torch.tensor([[0.,0.,0.],[0.,1.,1.]]))
        self.assertTrue(valid.all().item())

    def test_unknown_prefix_never_becomes_valid_again(self):
        observed=torch.tensor([[True,False,True,True]])
        actual=torch.zeros(1,4)
        dreamed=torch.tensor([[.1,900.,900.,900.]])
        labels,valid=make_risk_targets(actual,dreamed,observed,threshold=.25)
        torch.testing.assert_close(valid,torch.tensor([[True,False,False,False]]))
        torch.testing.assert_close(labels,torch.zeros(1,4))
        dreamed[0,1:]=-999.
        again,again_valid=make_risk_targets(actual,dreamed,observed,threshold=.25)
        torch.testing.assert_close(labels,again)
        torch.testing.assert_close(valid,again_valid)

    def test_prediction_ensemble_only_trains_auxiliary_parameters(self):
        torch.manual_seed(14)
        net=PredictionEnsemble(5,groups=2,classes=3,members=3,hidden_dim=8)
        features=torch.randn(2,3,5,requires_grad=True)
        teacher=torch.randn(2,3,2,3,requires_grad=True)
        mask=torch.ones(3,2,3,dtype=torch.bool)
        valid=torch.tensor([[True,True,False],[True,False,False]])
        loss=net.bootstrap_loss(features,teacher,mask,valid)
        self.assertTrue(torch.isfinite(loss).item())
        loss.backward()
        self.assertIsNone(features.grad)
        self.assertIsNone(teacher.grad)
        grads=[p.grad for p in net.parameters() if p.grad is not None]
        self.assertTrue(grads)
        self.assertTrue(all(torch.isfinite(g).all().item() for g in grads))
        self.assertGreater(sum(g.abs().sum().item() for g in grads),0.)

    def test_risk_head_cannot_use_future_actions_or_disagreement(self):
        torch.manual_seed(9)
        head=RiskHead(4,2,history_dim=2,hidden_dim=8)
        belief=torch.randn(2,4)
        history=torch.randn(2,3,2)
        history_mask=torch.tensor([[True,True,False],[True,False,False]])
        actions=torch.randn(2,4,2)
        disagreement=torch.rand(2,4)
        original=head(belief,history,history_mask,actions,disagreement)
        changed_actions=actions.clone(); changed_actions[:,2:]+=100.
        changed_disagreement=disagreement.clone(); changed_disagreement[:,2:]+=.7
        changed=head(belief,history,history_mask,changed_actions,changed_disagreement)
        torch.testing.assert_close(original[:,:2],changed[:,:2],atol=0,rtol=0)
        padded_history=history.clone(); padded_history[~history_mask]=9999.
        padded=head(belief,padded_history,history_mask,actions,disagreement)
        torch.testing.assert_close(original,padded,atol=0,rtol=0)

    def test_risk_head_cannot_change_its_input_features_by_backprop(self):
        torch.manual_seed(8)
        head=RiskHead(4,2,history_dim=2,hidden_dim=8)
        inputs=[torch.randn(2,4,requires_grad=True),torch.randn(2,3,2,requires_grad=True),torch.randn(2,4,2,requires_grad=True),torch.rand(2,4,requires_grad=True)]
        belief,history,actions,disagreement=inputs
        logits=head(belief,history,torch.ones(2,3,dtype=torch.bool),actions,disagreement)
        logits.sum().backward()
        for value in inputs:
            self.assertIsNone(value.grad)
        grads=[p.grad for p in head.parameters() if p.grad is not None]
        self.assertTrue(grads)
        self.assertGreater(sum(g.abs().sum().item() for g in grads),0.)


if __name__=='__main__':
    unittest.main()
