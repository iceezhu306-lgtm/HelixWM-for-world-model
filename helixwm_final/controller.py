"""Final-C1 non-learning controller prototype, not formal acceptance evidence.

Real and current Dream hypotheses share only executed action prefixes. Short
prospective plans are disposable. Prediction disagreement is conditional on
one model and never represented as epistemic ensemble uncertainty or safety.
"""
from __future__ import annotations

from collections import deque
import copy
import hashlib
import math

import numpy as np

from .belief import (accumulate_risk, calibration_sha256, canonical_json,
                     constrained_fusion, normalized_weights, probabilities,
                     sample_categorical, thresholds_for_gap, uncertainty_metrics,
                     validate_calibration)

STREAMS = ('real', 'actor', 'dream', 'fusion', 'planning', 'diagnostic')


class FallbackConstraintConflict(RuntimeError):
    """No action satisfies both registered constraints; issue no action."""


def _copy(value):
    return copy.deepcopy(value)


def _repeat(state, count):
    return {k: np.repeat(v, count, axis=0) for k, v in state.items()}


def _state(state, batch=None):
    if not isinstance(state, dict) or set(state) != {'deter', 'stoch'}:
        raise ValueError('State must contain only deter and stoch')
    h, s = np.asarray(state['deter']), np.asarray(state['stoch'])
    if (h.ndim != 2 or s.ndim != 3 or h.shape[0] != s.shape[0]
            or (batch is not None and h.shape[0] != batch)
            or h.dtype != np.float32 or s.dtype != np.float32
            or not np.isfinite(h).all() or not np.isfinite(s).all()):
        raise ValueError('Invalid finite float32 latent schema')
    return {'deter': h.copy(), 'stoch': s.copy()}


class FinalController:
    FORMAT = 'helixwm_final_c1_controller_v1'

    def __init__(self, backbone, seed=0, *, calibration=None,
                 expected_source_sha256=None, expected_model_sha256=None,
                 max_candidates=8, max_gap=100, score_normalization=12288.0,
                 max_events=512):
        if type(max_candidates) is not int or not 1 <= max_candidates <= 8:
            raise ValueError('Candidate count must be 1..8')
        if type(max_gap) is not int or not 1 <= max_gap <= 100:
            raise ValueError('Missing-age bound must be 1..100')
        if not math.isfinite(score_normalization) or score_normalization <= 0:
            raise ValueError('Score normalization must be positive')
        if type(seed) is not int or seed < 0 or type(max_events) is not int or max_events < 1:
            raise ValueError('Invalid seed/event budget')
        self.backend = backbone
        self.config = dict(max_candidates=max_candidates, max_gap=max_gap,
                           score_normalization=float(score_normalization), max_events=max_events)
        self.expected_source_sha256 = expected_source_sha256
        self.expected_model_sha256 = expected_model_sha256
        self._calibration = _copy(calibration)
        self.model_version = int(backbone.inference_version)
        self.rngs = {name: np.random.default_rng(stream) for name, stream in
                     zip(STREAMS, np.random.SeedSequence(seed).spawn(len(STREAMS)))}
        self.events = deque(maxlen=max_events)
        self.event_counter = 0
        self.initialized = False
        self.episode_id = self.timestamp = None
        self.real = self.bank = self.real_probs = self.bank_probs = None
        self.weights = np.full(max_candidates, 1/max_candidates, np.float64)
        self.issued_action = self.last_executed_action = None
        self.missing_age = 0
        self.cumulative_risk = 0.0
        self.absolute_quality_ok = False
        self.evidence_kind = 'UNINITIALIZED'
        self.mode = 'UNCALIBRATED'
        self.action_history = deque(maxlen=max_gap)
        self.last_metrics = {'prediction_error': None, 'epistemic_available': False}
        self.last_action_decision = None

    @property
    def state(self):
        if not self.initialized:
            raise RuntimeError('Reset required before state access')
        return _copy(self.real)

    @property
    def calibration(self):
        return _copy(self._calibration)

    def calibration_status(self):
        return validate_calibration(self._calibration, self.expected_source_sha256,
                                    self.model_version, self.expected_model_sha256)

    def _thresholds(self, gap=None):
        valid, reason = self.calibration_status()
        if not valid:
            return None, reason
        age = self.missing_age if gap is None else gap
        if age > self.config['max_gap']:
            return None, 'missing_age_budget_exceeded'
        return thresholds_for_gap(self._calibration['thresholds_by_gap'], age), reason

    def _version(self):
        if self.backend.inference_version != self.model_version:
            raise ValueError('Inference version changed inside an episode')

    def _event(self, kind, **data):
        self.event_counter += 1
        self.events.append({'event': kind, 'event_id': self.event_counter,
            'episode_id': self.episode_id, 'timestamp': self.timestamp,
            'model_version': self.model_version, **data})

    def pop_events(self):
        result = _copy(list(self.events)); self.events.clear(); return result

    def reset(self, image, episode_id, *, model_version=None, calibration=None,
              expected_model_sha256=None):
        if self.issued_action is not None:
            raise RuntimeError('Cannot reset while an issued action awaits an outcome')
        version = int(self.backend.inference_version) if model_version is None else model_version
        if type(version) is not int or version != self.backend.inference_version:
            raise ValueError('Reset inference version mismatch')
        if calibration is not None:
            self._calibration = _copy(calibration)
        if expected_model_sha256 is not None:
            self.expected_model_sha256 = expected_model_sha256
        elif hasattr(self.backend, 'inference_bundle_sha256'):
            self.expected_model_sha256 = self.backend.inference_bundle_sha256()
        self.model_version = version
        image = self._image(image)
        initial = self.backend.initial(batch=1)
        zero = np.zeros((1, 6), np.float32)
        self.real = _state(self.backend.observe(initial, zero, image, is_first=True,
                                               rng=self.rngs['real']), 1)
        self.bank = _repeat(self.real, self.config['max_candidates'])
        self.real_probs = self.bank_probs = None
        self.weights = np.full(self.config['max_candidates'], 1/self.config['max_candidates'])
        self.initialized, self.episode_id, self.timestamp = True, episode_id, 0
        self.issued_action = self.last_executed_action = None
        self.missing_age, self.cumulative_risk = 0, 0.0
        self.action_history.clear()
        self.evidence_kind = 'OBSERVED_POSTERIOR'
        thresholds, reason = self._thresholds(0)
        energies = self._energies(self.bank, image)
        self.absolute_quality_ok = bool(thresholds is not None and np.any(
            np.isfinite(energies) & (energies <= thresholds['max_energy'])))
        self.mode = 'OBSERVED' if thresholds else 'UNCALIBRATED'
        self.last_metrics = {'prediction_error': float(np.min(energies)) if np.isfinite(energies).all() else None,
            'epistemic_available': False, 'calibration_status': reason,
            'absolute_quality_ok': self.absolute_quality_ok, 'fusion_lambda': 0.0}
        self.last_action_decision = None
        self._event('reset', calibration_status=reason, epistemic_available=False)
        return self.state

    @staticmethod
    def _image(image):
        image = np.asarray(image)
        if image.dtype != np.uint8 or image.shape != (64, 64, 3):
            raise ValueError('Visible image must be uint8[64,64,3]')
        return image.copy()

    @staticmethod
    def _action(action):
        a = np.asarray(action, np.float32)
        if a.shape == (1, 6): a = a[0]
        if a.shape != (6,) or not np.isfinite(a).all():
            raise ValueError('Action must be finite float32[6]')
        return np.clip(a, -1, 1).copy()

    def _energies(self, states, image):
        values = np.asarray(self.backend.score(_copy(states), image.copy()), np.float64).reshape(-1)
        if values.shape != (len(states['deter']),):
            raise ValueError('Candidate energy count differs')
        return -values/self.config['score_normalization']

    def _info(self, state, info):
        if not isinstance(info, dict) or info.get('model_version') != self.model_version:
            raise ValueError('Prior/posterior distribution version mismatch')
        p = probabilities(info['probs'])
        if p.shape != state['stoch'].shape:
            raise ValueError('State/distribution shape mismatch')
        return p.astype(np.float32)

    def _trusted(self, thresholds):
        return bool(thresholds is not None and self.bank is not None and self.absolute_quality_ok
            and self.missing_age <= self.config['max_gap']
            and self.last_metrics.get('uncertainty', 0) <= thresholds['max_uncertainty']
            and self.last_metrics.get('model_conditional_disagreement', 0) <= thresholds['max_disagreement']
            and self.cumulative_risk <= thresholds['max_cumulative_risk']
            and (1/np.square(self.weights).sum())/len(self.weights) >= thresholds['min_ess_ratio'])

    def _on_candidate_prediction(self, state, probs, action, timestamp, image, weights):
        """Optional immutable diagnostic sink; no influence on model RNG/state."""
        return None

    def _fallback(self, base, thresholds):
        if thresholds is None:
            # No invented calibration constants. The ordinary prior actor is
            # explicitly uncalibrated and has no claimed physical safety.
            return base.copy()
        amplitude = float(thresholds['action_amplitude'])
        lower = np.full(6, -amplitude, np.float64)
        upper = np.full(6, amplitude, np.float64)
        if self.last_executed_action is not None:
            previous = self.last_executed_action.astype(np.float64)
            lower = np.maximum(lower, previous-float(thresholds['slew_limit']))
            upper = np.minimum(upper, previous+float(thresholds['slew_limit']))
        if np.any(lower > upper):
            # Sequential clipping used to silently violate the amplitude
            # limit. Do not invent a priority between incompatible controls.
            # Production calibration must exclude this configuration before
            # launch; this prototype does not claim safe continuous control.
            self._event('fallback_constraint_conflict', lower=lower.tolist(),
                        upper=upper.tolist(), action_issued=False)
            raise FallbackConstraintConflict('Amplitude and slew intervals do not intersect')
        return np.clip(np.asarray(base, np.float64), lower, upper).astype(np.float32)

    def _plan_action(self, base, thresholds):
        """Scratch-only bounded planning. Only its dedicated RNG is consumed."""
        if not self._trusted(thresholds) or self.missing_age == 0:
            return base.copy(), {'used': False, 'reason': 'planning_not_eligible'}
        proposals = [base.copy(), self._fallback(base, thresholds)]
        if np.array_equal(proposals[0], proposals[1]): proposals = proposals[:1]
        horizon = thresholds['planning_horizon']
        records = []
        for proposal in proposals:
            scratch = _copy(self.bank)
            cumulative_reward = np.zeros(len(scratch['deter']), np.float64)
            survival = np.ones_like(cumulative_reward)
            worst_u = 0.0
            worst_disagreement = 0.0
            action_sequence, reward_path, continuation_path, latent_hashes = [], [], [], []
            for depth in range(horizon):
                actions = (np.broadcast_to(proposal, (len(survival), 6)).copy() if depth == 0 else
                    np.clip(np.asarray(self.backend.act(_copy(scratch), deterministic=False,
                        rng=self.rngs['planning']), np.float32), -1, 1))
                if actions.shape != (len(survival), 6) or not np.isfinite(actions).all():
                    raise ValueError('Invalid scratch policy actions')
                scratch, info = self.backend.advance_prior(scratch, actions, rng=self.rngs['planning'])
                scratch = _state(scratch, len(survival))
                p = self._info(scratch, info)
                heads = self.backend.predict_heads(_copy(scratch), with_image=False)
                reward, cont = np.asarray(heads['reward'], np.float64), np.asarray(heads['continuation'], np.float64)
                if (reward.shape != survival.shape or cont.shape != survival.shape
                        or not np.isfinite(reward).all() or not np.isfinite(cont).all()
                        or np.any(cont < 0) or np.any(cont > 1)):
                    raise ValueError('Invalid scratch prediction heads')
                cumulative_reward += survival*reward
                survival *= cont
                uncertainty = uncertainty_metrics(p, self.weights)
                worst_u = max(worst_u, uncertainty['uncertainty'])
                worst_disagreement = max(worst_disagreement, uncertainty['model_conditional_disagreement'])
                action_sequence.append(actions.copy())
                reward_path.append(reward.copy()); continuation_path.append(cont.copy())
                latent_hashes.append(hashlib.sha256(scratch['deter'].tobytes()+scratch['stoch'].tobytes()).hexdigest())
            risk = float(np.dot(self.weights, 1-survival))
            slew = float(np.square(proposal-(self.last_executed_action if self.last_executed_action is not None else base)).mean())
            accepted = (worst_u <= thresholds['max_uncertainty']
                and worst_disagreement <= thresholds['max_disagreement']
                and risk <= thresholds['max_termination_risk'])
            utility = (thresholds['reward_weight']*float(np.dot(self.weights, cumulative_reward))
                - thresholds['uncertainty_weight']*worst_u
                - thresholds['termination_weight']*risk - thresholds['slew_weight']*slew)
            if not math.isfinite(utility):
                accepted = False
                utility = None
            records.append({'action': proposal.copy(), 'utility': utility, 'accepted': accepted,
                'uncertainty': worst_u, 'termination_risk': risk, 'horizon': horizon,
                'model_conditional_disagreement': worst_disagreement,
                'action_sequence': action_sequence, 'predicted_rewards': reward_path,
                'predicted_continuations': continuation_path, 'latent_path_sha256': latent_hashes,
                'prediction_error': None, 'image_prediction_recorded': False,
                'prospective_actions_executed': False})
        feasible = [r for r in records if r['accepted']]
        if not feasible:
            return self._fallback(base, thresholds), {'used': True, 'reason': 'all_plans_rejected',
                'selected': None, 'proposals': records, 'physical_safety_proven': False}
        selected = max((i for i, r in enumerate(records) if r['accepted']), key=lambda i: records[i]['utility'])
        chosen = records[selected]
        return chosen['action'].copy(), {'used': True, 'reason': 'accepted_plan',
            'selected': selected, 'proposals': records, 'physical_safety_proven': False}

    def act(self, external_action=None, deterministic=False):
        if not self.initialized or self.issued_action is not None:
            raise RuntimeError('Act requires an initialized state without an outstanding action')
        self._version()
        if external_action is not None:
            action = self._action(external_action)
            decision = {'kind': 'external_prefill', 'planning': False}
        else:
            base = self._action(self.backend.act(self.state, deterministic=deterministic, rng=self.rngs['actor']))
            thresholds, reason = self._thresholds()
            if self.missing_age and self._trusted(thresholds):
                action, planning = self._plan_action(base, thresholds)
                decision = {'kind': 'prior_only_fallback' if planning['reason'] == 'all_plans_rejected'
                            else 'candidate_control', 'calibration_status': reason, 'planning': planning}
            elif self.missing_age:
                fallback_thresholds = thresholds
                if reason == 'missing_age_budget_exceeded' and self.calibration_status()[0]:
                    fallback_thresholds = thresholds_for_gap(self._calibration['thresholds_by_gap'], self.config['max_gap'])
                action = self._fallback(base, fallback_thresholds)
                decision = {'kind': 'prior_only_fallback', 'calibration_status': reason,
                            'planning': False, 'physical_safety_proven': False}
            else:
                action, decision = base, {'kind': 'observed_actor', 'planning': False}
        self.issued_action = action.copy()
        self.last_action_decision = _copy(decision)
        return action.copy()

    def step(self, visible_image, action, info, episode_id):
        if not self.initialized or self.issued_action is None:
            raise RuntimeError('Step requires one previously issued action')
        self._version()
        if episode_id != self.episode_id:
            raise ValueError('Episode identity changed')
        timestamp = info.get('observation_timestamp')
        if type(timestamp) is not int or timestamp != self.timestamp+1:
            raise ValueError('Outcome timestamp must advance exactly once')
        if type(info.get('availability')) is not bool:
            raise ValueError('Availability must be an explicit boolean')
        raw = np.asarray(action, np.float32).reshape(-1)
        if raw.shape != (6,) or not np.array_equal(raw, self.issued_action):
            raise ValueError('Executed action differs from the action actually issued')
        valid = info['availability']
        image = self._image(visible_image) if valid else None
        previous_missing = self.missing_age
        new_age = 0 if valid else previous_missing+1
        thresholds, reason = self._thresholds(previous_missing if valid else new_age)
        # The prior diagnostic on an observed step must not consume the Real
        # posterior stream. Its current deter is reused without another action.
        prior, prior_info = self.backend.advance_prior(self.state, raw[None],
            rng=self.rngs['diagnostic'] if valid else self.rngs['real'])
        prior = _state(prior, 1); prior_probs = self._info(prior, prior_info)
        bank, bank_probs = None, None
        if self.bank is not None and (valid or new_age <= self.config['max_gap']):
            bank, bank_info = self.backend.advance_prior(_copy(self.bank),
                np.broadcast_to(raw, (self.config['max_candidates'], 6)).copy(), rng=self.rngs['dream'])
            bank = _state(bank, self.config['max_candidates']); bank_probs = self._info(bank, bank_info)
        u = uncertainty_metrics(bank_probs if bank_probs is not None else prior_probs,
                                self.weights if bank_probs is not None else None)
        if bank is not None:
            self._on_candidate_prediction(bank, bank_probs, raw, timestamp, image, self.weights)
        cumulative = self.cumulative_risk if valid else accumulate_risk(
            self.cumulative_risk, u['uncertainty'], self.config['max_gap'])
        fusion = {'lambda': 0.0, 'kl_to_real': 0.0}
        fusion_observation = {'checked': False, 'accepted': None,
                              'posterior_energy': None, 'fused_energy': None}
        quality, energy, ess = False, None, None
        new_weights = self.weights.copy()
        if valid:
            real, posterior_info = self.backend.posterior_from_deter(prior['deter'].copy(), image,
                                                                   rng=self.rngs['real'])
            real = _state(real, 1); base_probs = self._info(real, posterior_info)
            if not np.array_equal(real['deter'], prior['deter']):
                raise ValueError('Posterior correction must not advance or replace Real deter')
            if bank is not None:
                energies = self._energies(bank, image)
                finite = np.isfinite(energies) & (energies >= 0)
                energy = float(np.min(energies)) if finite.all() else None
                accepted = finite & (energies <= thresholds['max_energy']) if thresholds else np.zeros(len(energies), bool)
                quality = bool(np.any(accepted))
                if quality:
                    logits = np.full(len(energies), -np.inf)
                    with np.errstate(over='ignore'):
                        logits[accepted] = -(energies[accepted]-np.min(energies[accepted]))/thresholds['temperature']
                    new_weights = np.exp(logits); new_weights /= new_weights.sum()
                    ess = float(1/np.square(new_weights).sum())
                    post_bank, post_info = self.backend.posterior_from_deter(bank['deter'].copy(), image,
                                                                           rng=self.rngs['dream'])
                    post_bank = _state(post_bank, self.config['max_candidates'])
                    post_probs = self._info(post_bank, post_info)
                    if not np.array_equal(post_bank['deter'], bank['deter']):
                        raise ValueError('Candidate posterior must not advance its action again')
                    if (previous_missing > 0 and u['uncertainty'] <= thresholds['max_uncertainty']
                            and u['model_conditional_disagreement'] <= thresholds['max_disagreement']
                            and cumulative <= thresholds['max_cumulative_risk']
                            and ess/len(new_weights) >= thresholds['min_ess_ratio']):
                        mixed, fusion = constrained_fusion(base_probs, post_probs, new_weights,
                            thresholds['lambda_max'], thresholds['kl_limit'])
                        if fusion['lambda'] > 0:
                            proposed = _copy(real)
                            proposed['stoch'] = sample_categorical(mixed, self.rngs['fusion'])
                            baseline_energy = float(self._energies(real, image)[0])
                            proposed_energy = float(self._energies(proposed, image)[0])
                            finite = (math.isfinite(baseline_energy) and baseline_energy >= 0
                                and math.isfinite(proposed_energy) and proposed_energy >= 0)
                            accepted_fusion = bool(finite
                                and proposed_energy <= thresholds['max_energy']
                                and proposed_energy <= baseline_energy)
                            fusion_observation = {'checked': True, 'accepted': accepted_fusion,
                                'posterior_energy': baseline_energy if math.isfinite(baseline_energy) else None,
                                'fused_energy': proposed_energy if math.isfinite(proposed_energy) else None}
                            if accepted_fusion:
                                real, base_probs = proposed, mixed
                            else:
                                # Different candidate deter histories can make
                                # marginal categorical fusion inconsistent at
                                # Real's deter. Keep its actual posterior if
                                # the proposed sample worsens observed fit.
                                fusion = {'lambda': 0.0, 'kl_to_real': 0.0}
                    bank, bank_probs = post_bank, post_probs
                    # An individually good candidate is insufficient when the
                    # distribution is degenerate or highly uncertain. An actual
                    # observed posterior is still accepted independently.
                    if (u['uncertainty'] <= thresholds['max_uncertainty']
                            and u['model_conditional_disagreement'] <= thresholds['max_disagreement']
                            and ess/len(new_weights) >= thresholds['min_ess_ratio']):
                        cumulative = 0.0
            if not quality:
                bank, bank_probs = _repeat(real, self.config['max_candidates']), np.repeat(base_probs, self.config['max_candidates'], axis=0)
                new_weights = np.full(self.config['max_candidates'], 1/self.config['max_candidates'])
            evidence = 'RECONNECTED_POSTERIOR' if previous_missing else 'OBSERVED_POSTERIOR'
            mode = ('RECONNECTED' if previous_missing else 'OBSERVED') if thresholds else 'UNCALIBRATED'
        else:
            real, base_probs = prior, prior_probs
            quality = self.absolute_quality_ok
            if (thresholds is not None and bank is not None and quality
                    and u['uncertainty'] <= thresholds['max_uncertainty']
                    and u['model_conditional_disagreement'] <= thresholds['max_disagreement']
                    and cumulative <= thresholds['max_cumulative_risk']
                    and (1/np.square(new_weights).sum())/len(new_weights) >= thresholds['min_ess_ratio']):
                mixed, fusion = constrained_fusion(base_probs, bank_probs, new_weights,
                    thresholds['lambda_max'], thresholds['kl_limit'])
                if fusion['lambda'] > 0:
                    real['stoch'] = sample_categorical(mixed, self.rngs['fusion'])
                    base_probs = mixed
            evidence = 'PREDICTED_PRIOR'
            mode = ('MISSING_TRUSTED' if thresholds and bank is not None and quality
                and u['uncertainty'] <= thresholds['max_uncertainty']
                and u['model_conditional_disagreement'] <= thresholds['max_disagreement']
                and cumulative <= thresholds['max_cumulative_risk']
                and (1/np.square(new_weights).sum())/len(new_weights) >= thresholds['min_ess_ratio'] else 'PRIOR_ONLY_FALLBACK')
            if thresholds is None and reason != 'missing_age_budget_exceeded': mode = 'UNCALIBRATED'
        self.real, self.real_probs = real, base_probs
        self.bank, self.bank_probs, self.weights = bank, bank_probs, new_weights
        self.missing_age, self.cumulative_risk = new_age, cumulative
        self.absolute_quality_ok = quality
        self.evidence_kind, self.mode = evidence, mode
        self.timestamp = timestamp
        self.last_executed_action = raw.copy()
        self.action_history.append({'timestamp': timestamp, 'action': raw.copy(),
                                    'availability': valid, 'provenance': _copy(self.last_action_decision)})
        self.issued_action = None
        self.last_metrics = {**u, 'cumulative_risk': cumulative, 'missing_age': new_age,
            'prediction_error': energy if valid else None, 'absolute_quality_ok': quality,
            'effective_sample_size': ess, 'fusion_lambda': fusion['lambda'],
            'fusion_kl': fusion['kl_to_real'], 'calibration_status': reason,
            'fusion_observation_check': fusion_observation,
            'evidence_kind': evidence, 'physical_safety_proven': False,
            'observation_restored': bool(valid and previous_missing),
            'candidate_reconnection_accepted': bool(valid and previous_missing and fusion['lambda'] > 0),
            'unknown_fault_detector_implemented': False}
        event_kind = ('reconnected' if fusion['lambda'] > 0 else 'reconnection_posterior_only') if valid and previous_missing else 'step'
        self._event(event_kind, mode=mode, **self.last_metrics)
        return self.state

    def diagnostics(self):
        return {'mode': self.mode, 'model_version': self.model_version,
                'timestamp': self.timestamp, **_copy(self.last_metrics)}

    def snapshot(self):
        fields = ('model_version', 'initialized', 'episode_id', 'timestamp', 'real', 'bank',
                  'real_probs', 'bank_probs', 'weights', 'issued_action', 'last_executed_action',
                  'missing_age', 'cumulative_risk', 'absolute_quality_ok', 'evidence_kind',
                  'mode', 'last_metrics', 'last_action_decision', 'event_counter')
        return {'format': self.FORMAT, 'config': _copy(self.config),
            'expected_source_sha256': self.expected_source_sha256,
            'expected_model_sha256': self.expected_model_sha256, 'calibration': self.calibration,
            'state': {key: _copy(getattr(self, key)) for key in fields},
            'rngs': {key: _copy(rng.bit_generator.state) for key, rng in self.rngs.items()},
            'events': _copy(list(self.events)), 'action_history': _copy(list(self.action_history))}

    def _validated_record(self, record, *, completed_boundary=False):
        if record.get('format') != self.FORMAT or record.get('config') != self.config:
            raise ValueError('Final controller snapshot format/config differs')
        if record.get('expected_source_sha256') != self.expected_source_sha256:
            raise ValueError('Controller snapshot source differs')
        state = _copy(record['state'])
        if (set(state) != set(self.snapshot()['state']) or type(state['model_version']) is not int
                or (not completed_boundary and state['model_version'] != self.backend.inference_version)):
            raise ValueError('Controller snapshot state/version differs')
        if not completed_boundary and record.get('expected_model_sha256') != self.expected_model_sha256:
            raise ValueError('Controller snapshot model binding differs')
        if not completed_boundary and canonical_json(record.get('calibration')) != canonical_json(self._calibration):
            raise ValueError('Controller snapshot calibration differs')
        if set(record.get('rngs', {})) != set(STREAMS):
            raise ValueError('Incomplete controller RNG snapshot')
        generators = {}
        for name in STREAMS:
            generators[name] = np.random.default_rng()
            generators[name].bit_generator.state = _copy(record['rngs'][name])
        if type(state['initialized']) is not bool or type(state['absolute_quality_ok']) is not bool:
            raise ValueError('Invalid snapshot state flags')
        if state['initialized']:
            _state(state['real'], 1)
            if state['bank'] is not None: _state(state['bank'], self.config['max_candidates'])
            for name, reference in (('real_probs', state['real']), ('bank_probs', state['bank'])):
                if state[name] is not None:
                    p = probabilities(state[name])
                    if reference is None or p.shape != reference['stoch'].shape:
                        raise ValueError('Invalid saved belief shape')
            normalized_weights(state['weights'], self.config['max_candidates'])
            if not np.isclose(np.sum(state['weights']), 1.0, rtol=0, atol=1e-12):
                raise ValueError('Saved candidate weights are not normalized')
            for key in ('timestamp', 'missing_age', 'event_counter'):
                if type(state[key]) is not int or state[key] < 0: raise ValueError('Invalid snapshot counter')
            if not math.isfinite(state['cumulative_risk']) or not 0 <= state['cumulative_risk'] <= self.config['max_gap']:
                raise ValueError('Invalid snapshot cumulative risk')
            if state['issued_action'] is not None:
                if not np.array_equal(self._action(state['issued_action']), state['issued_action']):
                    raise ValueError('Invalid saved issued action')
        if not isinstance(record['events'], list) or len(record['events']) > self.config['max_events']:
            raise ValueError('Invalid saved event history')
        if not isinstance(record['action_history'], list) or len(record['action_history']) > self.config['max_gap']:
            raise ValueError('Invalid saved action history')
        return state, generators

    def load(self, record):
        state, generators = self._validated_record(record)
        # Commit only after validating all RNGs and state; restore never takes
        # an action, resets an environment, or calls a learned function.
        for key, value in state.items(): setattr(self, key, value)
        self.rngs = generators
        self.events = deque(_copy(record['events']), maxlen=self.config['max_events'])
        self.action_history = deque(_copy(record['action_history']), maxlen=self.config['max_gap'])

    def restore_completed_episode(self, record):
        """Restore private RNGs at the 500-action boundary, discard old latents.

        A newly pinned backbone/calibration must never consume old-version
        latent states. This operation takes no action and samples no model.
        It does not restore an environment or prove cross-hardware replay.
        """
        state, generators = self._validated_record(record, completed_boundary=True)
        if (state['initialized'] is not True or state['timestamp'] != 500
                or state['missing_age'] != 0 or state['issued_action'] is not None
                or state['evidence_kind'] not in ('OBSERVED_POSTERIOR', 'RECONNECTED_POSTERIOR')):
            raise ValueError('Completed recovery requires 500 actions and a final observed posterior')
        audit = {'next_operation': 'reset_new_episode_and_reanchor',
            'completed_episode_id': state['episode_id'], 'completed_timestamp': state['timestamp'],
            'completed_model_version': state['model_version'],
            'next_inference_model_version': self.backend.inference_version,
            'old_latent_state_reused': False, 'rng_streams_restored': list(STREAMS),
            'old_calibration_reused': False}
        self.rngs = generators
        self.event_counter = state['event_counter']
        self.events = deque(_copy(record['events']), maxlen=self.config['max_events'])
        self.model_version = int(self.backend.inference_version)
        self.initialized = False
        self.episode_id = self.timestamp = None
        self.real = self.bank = self.real_probs = self.bank_probs = None
        self.issued_action = self.last_executed_action = None
        self.weights = np.full(self.config['max_candidates'], 1/self.config['max_candidates'])
        self.missing_age, self.cumulative_risk = 0, 0.0
        self.absolute_quality_ok = False
        self.evidence_kind, self.mode = 'UNINITIALIZED', 'UNCALIBRATED'
        self.action_history.clear()
        self.last_metrics = {'prediction_error': None, 'epistemic_available': False}
        self.last_action_decision = None
        return audit

