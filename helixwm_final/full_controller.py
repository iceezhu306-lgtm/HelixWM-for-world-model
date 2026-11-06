"""Integrated Final controller with bounded histories and auxiliary uncertainty.

Production, pre-learning bootstrap and isolated calibration have different
entry points. A fitted proposal cannot masquerade as a production certificate.
Known-mask control is unchanged by the optional unknown-observation detector.
"""
from __future__ import annotations

import copy
import numpy as np

from .controller import FinalController, _state
from .belief import thresholds_for_gap, uncertainty_metrics
from .deployment_calibration import (verify_deployment_handle,
                                     validate_candidate_for_calibration)
from .epistemic import FrozenBootstrap, tree_digest
from .planning import rollout_statistics, plan_accepted, proposal_diversity


class FullController(FinalController):
    FORMAT = 'helixwm_final_integrated_controller_v2'

    def __init__(self, backbone, heads, context, *, seed=0, deployment=None,
                 trajectory=None, observation_guard=None, execution='production',
                 fitted_candidate=None, bank_gate=None, fitted_bank_candidate=None):
        if execution not in ('production', 'bootstrap', 'calibration_fit', 'calibration_validation',
                             'bank_production', 'bank_fit', 'bank_validation'):
            raise ValueError('Unknown controller execution phase')
        if not isinstance(heads,FrozenBootstrap): raise ValueError('Independently pinned auxiliary heads required')
        if context.get('epistemic_heads_sha256') != heads.sha256:
            raise ValueError('External auxiliary-head identity mismatch')
        if context.get('model_sha256') != backbone.inference_bundle_sha256() or context.get('model_version') != backbone.inference_version:
            raise ValueError('External current author-model identity mismatch')
        if execution=='production':
            verify_deployment_handle(deployment,context)
            if not heads.ready: raise ValueError('Production auxiliary heads have not been trained')
        bank_mode=execution.startswith('bank_')
        if bank_mode:
            if deployment is not None or fitted_candidate is not None or observation_guard is not None:
                raise ValueError('Independent-bank control cannot consume closed-loop certificates or unknown-fault guards')
            self.FORMAT='helixwm_final_integrated_bank_controller_v3'
            from .bank_gate import VerifiedBankGate, verify_bank_gate, validate_fitted_candidate
            if execution=='bank_production' and bank_gate is not None:
                if not isinstance(bank_gate,VerifiedBankGate):
                    raise ValueError('Bank production requires a file-authenticated independent-bank handle')
                # Authenticate even stale/inactive evidence before permitting a
                # conservative fallback. A handmade dictionary never qualifies.
                verify_bank_gate(bank_gate,bank_gate['context'],require_active=False)
            elif bank_gate is not None:
                raise ValueError('Only bank production accepts a verified bank handle')
            if execution=='bank_validation':validate_fitted_candidate(fitted_bank_candidate,context)
            elif fitted_bank_candidate is not None:
                raise ValueError('Fitted bank proposals are isolated validation inputs only')
        elif bank_gate is not None or fitted_bank_candidate is not None:
            raise ValueError('Independent-bank evidence must use its own explicit execution mode')
        if execution=='calibration_validation':
            validate_candidate_for_calibration(fitted_candidate,context)
        elif fitted_candidate is not None:
            raise ValueError('Fitted proposals are only usable in isolated calibration validation')
        if execution.startswith('calibration') and observation_guard is not None:
            raise ValueError('Known-mask deployment calibration cannot silently add unknown-fault screening')
        self.context=copy.deepcopy(context);self.execution=execution
        self.heads=heads;self.trajectory=trajectory;self.observation_guard=observation_guard
        self.auxiliary_state=None;self.current_epistemic=0.;self.current_temporal_error=None
        self.last_prediction=None;self.last_plans=[];self.last_guard=None
        self.last_recovery_qualified=False
        self.prediction_rng_before=None
        self.executed_prefix_sha256='0'*64
        self.screen_rng=np.random.default_rng(np.random.SeedSequence([seed,918713]))
        record=(bank_gate if execution=='bank_production' else fitted_bank_candidate) if bank_mode else (
            deployment if execution=='production' else fitted_candidate)
        super().__init__(backbone,seed=seed,calibration=record,
            expected_source_sha256=context['source_sha256'],expected_model_sha256=context['model_sha256'])

    def calibration_status(self):
        try:
            if self.execution=='bank_production':
                if self._calibration is None:return False,'independent_bank_gate_missing_native_prior_fallback'
                from .bank_gate import verify_bank_gate
                verify_bank_gate(self._calibration,self.context,require_active=True)
                if not self.heads.ready:return False,'independent_bank_heads_not_ready_native_prior_fallback'
            elif self.execution=='bank_validation':
                from .bank_gate import validate_fitted_candidate
                validate_fitted_candidate(self._calibration,self.context)
            elif self.execution=='production':verify_deployment_handle(self._calibration,self.context)
            elif self.execution=='calibration_validation':validate_candidate_for_calibration(self._calibration,self.context)
            else:return False,'isolated_'+self.execution+'_gates_inactive'
            if self.heads.sha256!=self.context['epistemic_heads_sha256']:raise ValueError('Heads changed')
            if self.backend.inference_version!=self.context['model_version']:raise ValueError('Pinned model changed')
            if self.execution.startswith('bank_'):
                return True,'independent_bank_off_policy_empirical_not_closed_loop_certificate'
            return True,'verified_production' if self.execution=='production' else 'calibration_candidate_only'
        except (ValueError,KeyError,TypeError) as exc:return False,str(exc)

    def _thresholds(self,gap=None):
        t,reason=super()._thresholds(gap)
        if t is not None:
            if self.current_epistemic > t['max_epistemic_disagreement']:
                return None,'independent_model_disagreement_rejected'
            if self.current_temporal_error is not None and self.current_temporal_error>t['max_temporal_prediction_error']:
                return None,'observed_temporal_prediction_error_rejected'
        return t,reason

    def reset(self,image,episode_id,**kwargs):
        # A new pinned version needs a NEW explicit evidence context/controller.
        if kwargs:raise ValueError('Construct a new bound controller for a new model version')
        self.auxiliary_state=self.heads.initialize(image)
        self.current_epistemic=0.;self.current_temporal_error=None
        self.last_prediction=None;self.last_plans=[];self.last_guard=None
        self.last_recovery_qualified=False
        self.executed_prefix_sha256=tree_digest({'episode_id':episode_id,'model_sha256':self.context['model_sha256']})
        if self.observation_guard is not None:self.observation_guard.reset()
        return super().reset(image,episode_id)

    def _on_candidate_prediction(self,state,probs,action,timestamp,image,weights):
        p=np.asarray(probs,np.float64);p/=p.sum(-1,keepdims=True)
        heads=self.backend.predict_heads(copy.deepcopy(state),with_image=True)
        error=None if image is None else self._energies(state,image)
        self.last_prediction={'timestamp':timestamp,'latent':copy.deepcopy(state),'probs':p.copy(),
            'actions':np.broadcast_to(action,(len(p),6)).copy(),
            'image_mean':np.asarray(heads['image_mean']).copy(),
            'reward':np.asarray(heads['reward'],np.float64).copy(),
            'continuation':np.asarray(heads['continuation'],np.float64).copy(),
            'weights':weights.copy(),'prediction_error':None if error is None else error.copy(),
            'metrics':uncertainty_metrics(p,weights),'epistemic_disagreement':self.current_epistemic,
            'prediction_rng_sha256':self.prediction_rng_before}

    def scratch_plans(self,first_action,*,rng=None,horizons=(1,3,5),archive=False):
        """Observe the exact shared planning operator; never issue an env action.

        Independent collector RNG means diagnostic probes cannot change the
        production actor/planning streams. Future ensemble features recursively
        follow each independent head, not identical copies of one prediction.
        """
        if self.bank is None:raise ValueError('Candidate budget exhausted; no fabricated plan')
        if tuple(horizons)!=(1,3,5):raise ValueError('Registered multiscale planning horizons required')
        rng=self.rngs['planning'] if rng is None else rng
        first=self._action(first_action)
        hold=np.zeros(6,np.float32) if self.last_executed_action is None else self.last_executed_action.copy()
        proposals=[first,hold]
        if np.array_equal(first,hold):proposals=proposals[:1]
        all_plans=[];previous=np.zeros(6,np.float64) if self.last_executed_action is None else self.last_executed_action.astype(np.float64)
        for proposal_id,proposal in enumerate(proposals):
            state=copy.deepcopy(self.bank);k=len(state['deter'])
            aux=np.repeat(self.auxiliary_state,k,axis=1)
            anchor_hash=tree_digest({'latent':state,'auxiliary':aux});rng_hash=tree_digest(rng.bit_generator.state)
            rewards=[];continuations=[];actions=[];entropies=[];expected=[];epistemic=[];states=[];images=[];probabilities=[];support_queries=[]
            for depth in range(5):
                a=(np.broadcast_to(proposal,(k,6)).copy() if depth==0 else
                   np.asarray(self.backend.act(copy.deepcopy(state),deterministic=False,rng=rng),np.float32))
                if a.shape!=(k,6) or not np.isfinite(a).all():raise ValueError('Invalid scratch action')
                # Match the author's ClipAction environment boundary. Its
                # Normal policy can legitimately sample outside [-1,1].
                a=np.clip(a,-1.,1.).astype(np.float32)
                if self.execution.startswith('bank_'):
                    from .bank_gate import support_query
                    support_queries.append(support_query(aux,a))
                state,info=self.backend.advance_prior(state,a,rng=rng);state=_state(state,k)
                p=np.asarray(self._info(state,info),np.float64);p/=p.sum(-1,keepdims=True)
                pred=self.backend.predict_heads(copy.deepcopy(state),with_image=archive)
                aux,aux_info=self.heads.advance(aux,a)
                metrics=uncertainty_metrics(p,self.weights)
                rewards.append(np.asarray(pred['reward'],np.float64));continuations.append(np.asarray(pred['continuation'],np.float64))
                actions.append(a.astype(np.float64));entropies.append(metrics['predictive_entropy'])
                expected.append(metrics['expected_entropy']);epistemic.append(float(aux_info['epistemic_disagreement'].max()))
                if archive:
                    states.append(copy.deepcopy(state));images.append(np.asarray(pred['image_mean']).copy());probabilities.append(p.copy())
                h=depth+1
                if h in horizons:
                    row={'timestamp':self.timestamp,'proposal_id':proposal_id,'horizon':h,
                        'scratch_only':True,'state_sha256':anchor_hash,'rng_sha256':rng_hash,
                        'rewards':np.stack(rewards),'continuations':np.stack(continuations),
                        'actions':np.stack(actions),'predictive_entropy':np.asarray(entropies,np.float64),
                        'expected_entropy':np.asarray(expected,np.float64),'weights':self.weights.copy(),
                        'previous_action':previous.copy(),'epistemic_disagreement':np.asarray(epistemic,np.float64)}
                    if self.execution.startswith('bank_'):row['support_queries']=np.stack(support_queries)
                    u=row['predictive_entropy'];d=np.maximum(0.,u-row['expected_entropy'])
                    stats=rollout_statistics(row['rewards'],row['continuations'],u,d,row['actions'],previous,self.weights,
                                             epistemic_disagreement=row['epistemic_disagreement'])
                    all_plans.append({'measurement':row,'statistics':stats,'first_action':proposal.copy()})
            if archive and self.trajectory is not None:
                self._archive_prospective(proposal_id,states,images,probabilities,actions,rewards,continuations,epistemic,rng_hash)
        return all_plans,proposal_diversity(np.asarray(proposals,np.float64))

    def _plan_action(self,base,thresholds):
        if not self._trusted(thresholds) or self.missing_age==0:
            return base.copy(),{'used':False,'reason':'planning_not_eligible'}
        records,diversity=self.scratch_plans(base,archive=self.trajectory is not None)
        limits=self._calibration['planning_limits_by_gap_horizon']
        gap=next(g for g in (1,5,10,20,50,100) if self.missing_age<=g)
        for row in records:
            row['accepted']=plan_accepted(row['statistics'],limits[str(gap)][str(row['measurement']['horizon'])])
            if self.execution.startswith('bank_'):
                from .bank_gate import plan_support
                row['support_audit']=plan_support(self._calibration,gap,row['measurement']['support_queries'])
                row['accepted']=row['accepted'] and row['support_audit']['supported']
        # Reduce depth only when the longer prediction fails its own gate.
        feasible=[]
        for proposal in sorted({r['measurement']['proposal_id'] for r in records}):
            accepted=[r for r in records if r['measurement']['proposal_id']==proposal and r['accepted']]
            if accepted:feasible.append(max(accepted,key=lambda r:r['measurement']['horizon']))
        self.last_plans=records
        if not feasible:
            return self._fallback(base,thresholds),{'used':True,'reason':'all_plans_rejected','selected':None,
                'proposal_diversity':diversity,'physical_safety_proven':False}
        chosen=max(feasible,key=lambda r:r['statistics']['utility'])
        return chosen['first_action'].copy(),{'used':True,'reason':'accepted_plan',
            'selected':chosen['measurement']['proposal_id'],'horizon':chosen['measurement']['horizon'],
            'statistics':copy.deepcopy(chosen['statistics']),'proposal_diversity':diversity,'physical_safety_proven':False,
            **({'decision_label':'EXPERIMENTAL_SUPPORTED_PLAN','scope':'off_policy_empirical_not_closed_loop_certificate',
                'support_audit':copy.deepcopy(chosen['support_audit'])} if self.execution.startswith('bank_') else {})}

    def act(self,external_action=None,deterministic=False):
        if self.execution=='production' and not self.calibration_status()[0]:
            raise RuntimeError('Production lost its exact pinned calibration')
        if self.execution=='bank_production' and self._calibration is not None:
            from .bank_gate import verify_bank_gate
            # Corruption is not an ordinary inactive/stale gate. Authenticate
            # its own sealed identity before the current-context fallback test.
            verify_bank_gate(self._calibration,self._calibration['context'],require_active=False)
        return super().act(external_action,deterministic)

    def step(self,visible_image,action,info,episode_id):
        # Validate identity BEFORE auxiliary/detector state is advanced.
        unknown=self.observation_guard is not None and self.observation_guard.mode=='unknown_observation'
        if (type(info) is not dict or type(info.get('observation_timestamp')) is not int
                or type(episode_id) is not int or (not unknown and type(info.get('availability')) is not bool)
                or self.issued_action is None or episode_id!=self.episode_id
                or info.get('observation_timestamp')!=self.timestamp+1
                or not np.array_equal(np.asarray(action,np.float32).reshape(-1),self.issued_action)):
            raise ValueError('Outcome does not match the outstanding action/episode/timestamp')
        self._version()
        aux,diagnostic=self.heads.advance(self.auxiliary_state,np.asarray(action,np.float64).reshape(1,6))
        self.current_epistemic=float(diagnostic['epistemic_disagreement'][0])
        decision_info=dict(info)
        if unknown:
            # Scratch prediction is explicitly pre-image and uses its own RNG.
            prior,_=self.backend.advance_prior(self.state,np.asarray(action,np.float32).reshape(1,6),rng=self.screen_rng)
            predicted=self.backend.predict_heads(prior,with_image=True)['image_mean'][0]
            prediction={'image_mean':predicted,'target_timestamp':self.timestamp+1,
                'conditioning_timestamp':self.timestamp,'model_sha256':self.context['model_sha256'],
                'model_version':self.model_version}
            screened=self.observation_guard.step(visible_image,timestamp=self.timestamp+1,prior_prediction=prediction)
            self.last_guard={k:v for k,v in screened.items() if k!='accepted_image'}
            decision_info['availability']=bool(screened['allow_posterior'])
            visible_image=screened['accepted_image']
        valid=decision_info.get('availability')
        if type(valid) is not bool:raise ValueError('Known-mask path requires a bool availability')
        self.current_temporal_error=float(self.heads.observed_error(aux,visible_image)[0]) if valid else None
        old_risk=self.cumulative_risk
        self.prediction_rng_before=tree_digest(self.rngs['dream'].bit_generator.state)
        self.executed_prefix_sha256=tree_digest({'previous':self.executed_prefix_sha256,
            'episode':episode_id,'timestamp':self.timestamp+1,'action':np.asarray(action,np.float32)})
        self.last_prediction=None
        state=super().step(visible_image,action,decision_info,episode_id)
        self.auxiliary_state=self.heads.initialize(visible_image) if valid else aux
        # The step evaluated the pre-recovery gap's bounds. Rechecking here
        # would use age0/gap1 after recovery and could falsify a real reset.
        self.last_recovery_qualified=bool(valid and self.absolute_quality_ok
            and self.cumulative_risk==0. and (old_risk>0 or self.last_metrics['observation_restored']))
        self.last_metrics.update(epistemic_available=bool(self.heads.ready),
            epistemic_disagreement=self.current_epistemic,temporal_prediction_error=self.current_temporal_error,
            epistemic_scope='bootstrap_visible_dynamics_proxy',
            unknown_fault_detector_implemented=self.observation_guard is not None,
            unknown_fault_detector_active=unknown,
            risk_reset=self.last_recovery_qualified,execution=self.execution)
        if self.execution.startswith('bank_'):
            self.last_metrics.update(bank_gate_scope='off_policy_empirical_not_closed_loop_certificate',
                closed_loop_certified=False,bank_gate_active=self.calibration_status()[0])
        self.last_metrics['pre_reset_cumulative_risk']=(old_risk if valid else
            min(float(self.config['max_gap']),old_risk+self.last_metrics['uncertainty']))
        if self.trajectory is not None and self.last_prediction is not None:self._archive_actual(valid)
        return state

    def _archive_actual(self,visible):
        from .trajectory import make_record
        # Adapter is implemented against the immutable trajectory writer; the
        # sink receives only legitimate observed prediction errors.
        p=self.last_prediction
        ids=[f'e{self.episode_id}:c{i}' for i in range(8)]
        record=make_record(kind='actual',episode_id=self.episode_id,model_version=self.model_version,
            model_sha256=self.context['model_sha256'],timestamp=self.timestamp,
            calibration_sha256=self._evidence_sha256(),
            candidate_ids=ids,ancestor_ids=ids,state=p['latent'],real_state=self.state,actions=p['actions'].astype(np.float32),
            predicted_observation=p['image_mean'],predicted_reward=p['reward'],predicted_continuation=p['continuation'],
            probabilities=p['probs'],weights=p['weights'],observed_error=p['prediction_error'],
            epistemic_disagreement=p['epistemic_disagreement'],calibration_confidence=None,
            observation_available=visible,missing_age=self.missing_age,
            prediction_rng_sha256=p['prediction_rng_sha256'],executed_prefix_sha256=self.executed_prefix_sha256)
        self.trajectory.append(record)

    def _archive_prospective(self,proposal,states,images,probs,actions,rewards,continuations,epistemic,rng_hash):
        from .trajectory import make_record
        for depth,state in enumerate(states):
            record=make_record(kind='prospective',episode_id=self.episode_id,model_version=self.model_version,
                model_sha256=self.context['model_sha256'],timestamp=self.timestamp+depth+1,
                calibration_sha256=self._evidence_sha256(),
                origin_timestamp=self.timestamp,depth=depth+1,missing_age=self.missing_age,
                candidate_ids=[f'e{self.episode_id}:t{self.timestamp}:p{proposal}:c{i}' for i in range(8)],
                ancestor_ids=[f'e{self.episode_id}:c{i}' for i in range(8)],state=state,actions=np.asarray(actions[depth],np.float32),
                predicted_observation=images[depth],predicted_reward=rewards[depth],predicted_continuation=continuations[depth],
                probabilities=probs[depth],weights=self.weights,observed_error=None,
                epistemic_disagreement=epistemic[depth],calibration_confidence=None,observation_available=False,
                prediction_rng_sha256=rng_hash,executed_prefix_sha256=self.executed_prefix_sha256)
            self.trajectory.append(record)

    def _evidence_sha256(self):
        if self._calibration is None:return None
        return self._calibration['artifact_sha256' if self.execution=='bank_production' else 'sha256']

    def snapshot(self):
        out=super().snapshot()
        out['integration']={'context':copy.deepcopy(self.context),'execution':self.execution,
            'heads_sha256':self.heads.sha256,'auxiliary_state':copy.deepcopy(self.auxiliary_state),
            'current_epistemic':self.current_epistemic,'current_temporal_error':self.current_temporal_error,
            'executed_prefix_sha256':self.executed_prefix_sha256,
            'screen_rng':copy.deepcopy(self.screen_rng.bit_generator.state),
            'guard':None if self.observation_guard is None else self.observation_guard.snapshot(),
            'trajectory':None if self.trajectory is None else self.trajectory.snapshot()}
        return out

    def restore_completed_episode(self,record):
        """Carry private RNGs across a pin while discarding all old latents.

        This is not an environment continuation. The next operation must reset
        a new registered episode under the new context and independently loaded
        bank gate. Unknown-fault guard lineage is a separate execution design.
        """
        extra=record.get('integration')
        if type(extra) is not dict or set(extra)!=set(self.snapshot()['integration']):
            raise ValueError('Completed episode is missing integrated provenance/RNG')
        old_context=extra['context']
        if type(old_context) is not dict or set(old_context)!=set(self.context) or extra['execution']!=self.execution:
            raise ValueError('Completed episode execution/context schema differs')
        changing={'model_sha256','model_version','epistemic_heads_sha256'}
        if any(old_context[k]!=self.context[k] for k in self.context if k not in changing):
            raise ValueError('Completed episode source/protocol/bank/collector differs')
        if (old_context['model_sha256']!=record.get('expected_model_sha256')
                or old_context['model_version']!=record['state']['model_version']
                or old_context['epistemic_heads_sha256']!=extra['heads_sha256']):
            raise ValueError('Completed episode pin/head identities are internally inconsistent')
        if type(old_context['model_version']) is not int or not 0<=old_context['model_version']<=self.context['model_version']:
            raise ValueError('Completed inference pin cannot come from the future')
        old_evidence=record.get('calibration')
        if old_evidence is not None and old_evidence.get('context')!=old_context:
            raise ValueError('Completed calibration context differs from its collector')
        for key in ('model_sha256','epistemic_heads_sha256'):
            value=old_context[key]
            if type(value) is not str or len(value)!=64 or any(x not in '0123456789abcdef' for x in value):
                raise ValueError('Completed episode digest is invalid')
        if (extra['guard'] is not None or self.observation_guard is not None):
            raise ValueError('Unknown-fault guard boundary continuation requires a separate explicit design')
        if (extra['trajectory'] is None)!=(self.trajectory is None):
            raise ValueError('Completed trajectory dependency is missing')
        if self.trajectory is not None and self.trajectory.snapshot()!=extra['trajectory']:
            raise ValueError('Completed trajectory cursor must match before new-episode append')
        for key in ('current_epistemic','current_temporal_error'):
            value=extra[key]
            if key=='current_temporal_error' and value is None:continue
            if type(value) not in (float,np.float64) or not np.isfinite(value) or value<0:
                raise ValueError('Completed integrated uncertainty is invalid')
        value=extra['auxiliary_state']
        if not isinstance(value,np.ndarray) or value.shape!=(3,1,16) or value.dtype!=np.float64 or not np.isfinite(value).all():
            raise ValueError('Completed integrated state is invalid')
        prefix=extra['executed_prefix_sha256']
        if type(prefix) is not str or len(prefix)!=64 or any(x not in '0123456789abcdef' for x in prefix):
            raise ValueError('Completed executed-prefix digest is invalid')
        screen=np.random.default_rng();screen.bit_generator.state=copy.deepcopy(extra['screen_rng'])
        # Base validation is atomic and checks 500 actual outcomes/no issued
        # action/final available posterior before committing any six RNGs.
        audit=super().restore_completed_episode(record)
        self.screen_rng=screen;self.auxiliary_state=None
        self.current_epistemic=0.;self.current_temporal_error=None
        self.last_prediction=None;self.last_plans=[];self.last_guard=None
        self.last_recovery_qualified=False;self.prediction_rng_before=None
        self.executed_prefix_sha256='0'*64
        audit['rng_streams_restored']=list(audit['rng_streams_restored'])+['screen']
        audit['integrated_old_latent_state_reused']=False
        return audit

    def load(self,record):
        extra=record['integration']
        if type(extra) is not dict or set(extra)!=set(self.snapshot()['integration']):
            raise ValueError('Incomplete integrated snapshot')
        if extra['context']!=self.context or extra['execution']!=self.execution or extra['heads_sha256']!=self.heads.sha256:
            raise ValueError('Integrated controller snapshot binding differs')
        for key in ('current_epistemic','current_temporal_error'):
            value=extra[key]
            if key=='current_temporal_error' and value is None:continue
            if type(value) not in (float,np.float64) or not np.isfinite(value) or value<0:
                raise ValueError('Invalid integrated uncertainty scalar')
        prefix=extra['executed_prefix_sha256']
        if type(prefix) is not str or len(prefix)!=64 or any(c not in '0123456789abcdef' for c in prefix):
            raise ValueError('Invalid executed action-prefix identity')
        f=extra['auxiliary_state']
        if f is not None and (not isinstance(f,np.ndarray) or f.shape!=(3,1,16) or f.dtype!=np.float64 or not np.isfinite(f).all()):
            raise ValueError('Invalid auxiliary collector state')
        if (extra['guard'] is None)!=(self.observation_guard is None) or (extra['trajectory'] is None)!=(self.trajectory is None):
            raise ValueError('Snapshot dependencies are missing')
        if self.trajectory is not None and self.trajectory.snapshot()!=extra['trajectory']:
            raise ValueError('Restore the matching immutable trajectory cursor first')
        rng=np.random.default_rng();rng.bit_generator.state=copy.deepcopy(extra['screen_rng'])
        guard=copy.deepcopy(self.observation_guard)
        if guard is not None:guard.load(extra['guard'],expected_snapshot_sha256=extra['guard']['sha256'])
        super().load(record)
        self.auxiliary_state=copy.deepcopy(f);self.current_epistemic=extra['current_epistemic']
        self.current_temporal_error=extra['current_temporal_error'];self.screen_rng=rng;self.observation_guard=guard
        self.executed_prefix_sha256=extra['executed_prefix_sha256']
        self.last_prediction=None;self.last_plans=[];self.last_guard=None;self.last_recovery_qualified=False
