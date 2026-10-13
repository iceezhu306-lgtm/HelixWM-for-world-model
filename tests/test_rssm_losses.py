"""Time-index, reset, action alignment, and stop-gradient checks."""
import unittest
import torch
from helixwm.rssm import RSSM,RSSMState,ObservationRollout
from helixwm.losses import cross_correction_loss,world_model_loss


class RSSMTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.model=RSSM(3,2,hidden_dim=8,groups=2,classes=3,embed_dim=8)
        self.obs=torch.randn(2,5,3)
        self.actions=torch.randn(2,4,2)

    def test_next_observation_changes_posterior_but_not_prior(self):
        first=self.model.observe(self.obs,self.actions,sample=False)
        changed_obs=self.obs.clone(); changed_obs[:,2]+=17.
        second=self.model.observe(changed_obs,self.actions,sample=False)
        torch.testing.assert_close(first.prior_logits[:,:3],second.prior_logits[:,:3],atol=0,rtol=0)
        torch.testing.assert_close(first.posterior_logits[:,:2],second.posterior_logits[:,:2],atol=0,rtol=0)
        self.assertFalse(torch.allclose(first.posterior_logits[:,2],second.posterior_logits[:,2]))
        self.assertEqual(len(first.states),5)

    def test_action_t_first_affects_prior_t_plus_one(self):
        first=self.model.observe(self.obs,self.actions,sample=False)
        changed_actions=self.actions.clone(); changed_actions[:,1]+=13.
        second=self.model.observe(self.obs,changed_actions,sample=False)
        torch.testing.assert_close(first.prior_logits[:,:2],second.prior_logits[:,:2],atol=0,rtol=0)
        self.assertFalse(torch.allclose(first.prior_logits[:,2],second.prior_logits[:,2]))

    def test_imagination_uses_supplied_action_prefix(self):
        start=self.model.observe(self.obs,self.actions,sample=False).states[0].detach()
        dream=self.model.imagine(start,self.actions,sample=False)
        changed_actions=self.actions.clone(); changed_actions[:,1]+=19.
        changed=self.model.imagine(start,changed_actions,sample=False)
        self.assertEqual(len(dream.states),5)
        torch.testing.assert_close(dream.states[0].h,start.h,atol=0,rtol=0)
        torch.testing.assert_close(dream.states[0].z,start.z,atol=0,rtol=0)
        torch.testing.assert_close(dream.prior_logits[:,0],changed.prior_logits[:,0],atol=0,rtol=0)
        self.assertFalse(torch.allclose(dream.prior_logits[:,1],changed.prior_logits[:,1]))

    def test_reset_discards_old_state_and_transition_action(self):
        is_first=torch.zeros(2,5,dtype=torch.bool); is_first[:,0]=True; is_first[:,3]=True
        first=self.model.observe(self.obs,self.actions,is_first,sample=False)
        obs=self.obs.clone(); obs[:,:3]+=500.
        actions=self.actions.clone(); actions[:,:3]-=100.
        second=self.model.observe(obs,actions,is_first,sample=False)
        torch.testing.assert_close(first.prior_logits[:,3:],second.prior_logits[:,3:],atol=0,rtol=0)
        torch.testing.assert_close(first.posterior_logits[:,3:],second.posterior_logits[:,3:],atol=0,rtol=0)
        for index in (3,4):
            torch.testing.assert_close(first.states[index].h,second.states[index].h,atol=0,rtol=0)
            torch.testing.assert_close(first.states[index].z,second.states[index].z,atol=0,rtol=0)

    def test_cross_correction_trains_dynamics_not_teacher_encoder(self):
        rollout=self.model.observe(self.obs,self.actions,sample=False)
        first=torch.zeros(2,5,dtype=torch.bool); first[:,0]=True
        loss=cross_correction_loss(self.model,rollout,self.actions,first,torch.ones(2,5,dtype=torch.bool),max_horizon=3,sample=False)
        loss.backward()
        for module in (self.model.encoder,self.model.posterior):
            self.assertTrue(all(p.grad is None or not p.grad.any().item() for p in module.parameters()))
        dynamic_grads=[p.grad for module in (self.model.recurrent,self.model.prior) for p in module.parameters() if p.grad is not None]
        self.assertTrue(dynamic_grads)
        self.assertTrue(all(torch.isfinite(g).all().item() for g in dynamic_grads))
        self.assertGreater(sum(g.abs().sum().item() for g in dynamic_grads),0.)

    def _leaf_rollout(self,n=6):
        states=[]
        for _ in range(n):
            h=torch.randn(2,8,requires_grad=True)
            z=torch.nn.functional.one_hot(torch.randint(3,(2,2)),3).float().requires_grad_()
            states.append(RSSMState(h,z))
        teacher=torch.randn(2,n,2,3,requires_grad=True)
        rollout=ObservationRollout(states=tuple(states),prior_logits=torch.zeros_like(teacher),posterior_logits=teacher)
        return rollout,teacher

    def test_cross_correction_detaches_start_states_and_teacher_targets(self):
        rollout,teacher=self._leaf_rollout()
        first=torch.zeros(2,6,dtype=torch.bool); first[:,0]=True
        actions=torch.randn(2,5,2)
        loss=cross_correction_loss(self.model,rollout,actions,first,torch.ones(2,6,dtype=torch.bool),max_horizon=3,sample=False)
        loss.backward()
        self.assertIsNone(teacher.grad)
        for state in rollout.states:
            self.assertIsNone(state.h.grad)
            self.assertIsNone(state.z.grad)

    def test_cross_correction_excludes_reset_crossings_and_padding(self):
        rollout,_=self._leaf_rollout()
        first=torch.zeros(2,6,dtype=torch.bool); first[:,0]=True; first[:,3]=True
        valid=torch.ones(2,6,dtype=torch.bool); valid[:,5]=False
        actions=torch.randn(2,5,2)
        # k>=2: the only admissible pair is observation 0 -> observation 2.
        before=cross_correction_loss(self.model,rollout,actions,first,valid,max_horizon=2,sample=False)
        modified_states=list(rollout.states)
        for i in range(3,6):
            modified_states[i]=RSSMState(modified_states[i].h+1000.,modified_states[i].z)
        changed_teacher=rollout.posterior_logits.detach().clone(); changed_teacher[:,3:]+=torch.tensor([100.,-100.,50.])
        changed=ObservationRollout(states=tuple(modified_states),prior_logits=rollout.prior_logits,posterior_logits=changed_teacher)
        altered_actions=actions.clone(); altered_actions[:,2:]+=1000.
        after=cross_correction_loss(self.model,changed,altered_actions,first,valid,max_horizon=2,sample=False)
        torch.testing.assert_close(before,after,atol=0,rtol=0)
        self.assertGreater(before.item(),0.)

    def test_all_invalid_correction_is_finite_zero(self):
        rollout=self.model.observe(self.obs,self.actions,sample=False)
        first=torch.zeros(2,5,dtype=torch.bool)
        loss=cross_correction_loss(self.model,rollout,self.actions,first,torch.zeros(2,5,dtype=torch.bool),max_horizon=3,sample=False)
        self.assertTrue(torch.isfinite(loss).item())
        self.assertEqual(loss.item(),0.)

    def test_world_model_ignores_reset_reward_and_padded_targets(self):
        first=torch.zeros(2,5,dtype=torch.bool); first[:,0]=True; first[:,3]=True
        rollout=self.model.observe(self.obs,self.actions,first,sample=False)
        valid=torch.ones(2,5,dtype=torch.bool); valid[:,4]=False
        rewards=torch.randn(2,5); continuation=torch.ones(2,5)
        before=world_model_loss(self.model,rollout,self.obs,rewards,continuation,valid,is_first=first)
        rewards2=rewards.clone(); rewards2[:,[0,3,4]]=10000.
        continuation2=continuation.clone(); continuation2[:,[0,3,4]]=0.
        obs2=self.obs.clone(); obs2[:,4]=10000.
        after=world_model_loss(self.model,rollout,obs2,rewards2,continuation2,valid,is_first=first)
        for key in before:
            torch.testing.assert_close(before[key],after[key],atol=0,rtol=0)


if __name__=='__main__':
    unittest.main()
