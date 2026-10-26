"""Shared controller interface, including explicitly external prefill actions."""
from __future__ import annotations
import copy
import numpy as np
from helixwm_v2.dual_chain import DualChain

CONTROLLERS = ('clean_native', 'black_frame_native', 'prior_only', 'helix_v2')

class Controller:
    def __init__(self, backbone, name, seed=0):
        if name not in CONTROLLERS:
            raise ValueError('Unregistered controller')
        self.name = name
        self.chain = DualChain(backbone, seed=seed,
            model_version=backbone.inference_version,
            mode='repair' if name == 'helix_v2' else 'disabled',
            max_candidates=8, max_missing=5, score_temperature=.01,
            score_normalization=12288.)

    def reset(self, image, episode_id):
        return self.chain.reset(image, episode_id=episode_id,
                                model_version=self.chain.backend.inference_version)

    @property
    def state(self):
        return self.chain.state

    def act(self, external_action=None):
        if external_action is None:
            return self.chain.act(deterministic=False).reshape(6)
        if not self.chain.initialized or self.chain.last_action is not None:
            raise RuntimeError('An external action requires an initialized, unissued state')
        action = np.asarray(external_action, dtype=np.float32)
        if action.shape != (6,) or not np.isfinite(action).all():
            raise ValueError('External action must be finite float32[6]')
        self.chain.last_action = np.clip(action, -1., 1.).reshape(1,6).copy()
        return self.chain.last_action.reshape(6).copy()

    def step(self, visible_image, action, info, episode_id):
        valid = bool(info['availability'])
        if self.name in ('clean_native', 'black_frame_native'):
            if self.name == 'clean_native' and not valid:
                raise ValueError('clean_native cannot receive a missing observation')
            valid = True
        return self.chain.step(visible_image if valid else None, action,
            valid=valid, episode_id=episode_id,
            timestamp=int(info['observation_timestamp']),
            model_version=self.chain.backend.inference_version)

    def snapshot(self):
        return {'controller': self.name, 'chain': self.chain.state_dict()}

    def load(self, record):
        if record['controller'] != self.name:
            raise ValueError('Controller changed across restore')
        self.chain.load_state_dict(copy.deepcopy(record['chain']))
