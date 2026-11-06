"""Independent bootstrap predictors of visible coarse-image dynamics.

Three independently randomized feature maps and fixed prior functions are
fitted by separate Poisson-bootstrap ridge regressions. They are auxiliary
predictors, NOT three Dreamer backbones or a calibrated Bayesian posterior.
Disagreement is a model-uncertainty proxy; common blind spots remain possible.
The author learner and its gradients are untouched. Only adjacent available
training observations enter these heads. No hidden clean image, calibration
or evaluation transition may update their sufficient statistics.
"""
from __future__ import annotations

import copy
import hashlib
import json

import numpy as np

SCHEMA = 'helixwm_visible_bootstrap_dynamics_v1'
MEMBERS, FEATURES, IMAGE_FEATURES, OUTPUTS = 3, 64, 16, 18
RIDGE, PRIOR_SCALE = 1., .1


def tree_digest(value):
    h = hashlib.sha256()
    def framed(tag,raw):
        h.update(tag);h.update(len(raw).to_bytes(8,'big'));h.update(raw)
    def visit(x):
        if isinstance(x, np.ndarray):
            if x.dtype.hasobject or not np.isfinite(x).all():
                raise ValueError('Finite non-object arrays required')
            framed(b'A',json.dumps([x.dtype.str, list(x.shape)]).encode())
            framed(b'V',np.ascontiguousarray(x).tobytes())
        elif isinstance(x, dict):
            framed(b'D',str(len(x)).encode())
            for key in sorted(x):
                if type(key) is not str:raise ValueError('String dictionary keys required')
                framed(b'K',json.dumps(key).encode()); visit(x[key])
        elif isinstance(x, (list, tuple)):
            framed(b'L' if isinstance(x,list) else b'T',str(len(x)).encode())
            for item in x: visit(item)
        else:
            framed(b'J',json.dumps(x, sort_keys=True, allow_nan=False).encode())
    visit(value)
    return h.hexdigest()


def image_features(image):
    x = np.asarray(image)
    if x.dtype != np.uint8 or x.shape[-3:] != (64, 64, 3):
        raise ValueError('Only actually visible uint8 RGB64 images are accepted')
    x = x.astype(np.float64).mean(-1) / 255.
    return x.reshape((*x.shape[:-2], 4, 16, 4, 16)).mean(axis=(-3, -1)).reshape((*x.shape[:-2], 16))


def _binding(value):
    if (type(value) is not dict or set(value) != {'source_sha256', 'protocol_sha256'}
            or any(type(v) is not str or len(v) != 64 or any(c not in '0123456789abcdef' for c in v)
                   for v in value.values())):
        raise ValueError('External source and protocol identities required')
    return copy.deepcopy(value)


class BootstrapDynamics:
    """Small fixed-feature regressors, trained online and pinned per episode.

    For member m: phi_m(x,a)=[1, cos(W_m[x,a]+b_m)]/sqrt(64),
    f_m=phi_m(theta_m + .1*prior_m).  theta_m minimizes the independently
    bootstrap-weighted squared error plus ||theta_m||^2. Targets contain
    16 coarse image means, actual repeat2 reward /2 and continuation.
    Raw predictions may extrapolate. No clipping conceals their disagreement.
    """
    def __init__(self, seed, binding):
        if type(seed) is not int or seed < 0: raise ValueError('Nonnegative seed required')
        self.binding = _binding(binding)
        streams = np.random.SeedSequence(seed).spawn(MEMBERS*2)
        self.bootstrap_rng = [np.random.default_rng(s) for s in streams[MEMBERS:]]
        maps = [np.random.default_rng(s) for s in streams[:MEMBERS]]
        self.projection = np.stack([r.normal(size=(22, FEATURES-1)) for r in maps])
        self.phase = np.stack([r.uniform(-np.pi, np.pi, FEATURES-1) for r in maps])
        self.prior = np.stack([r.normal(size=(FEATURES, OUTPUTS)) for r in maps])
        self.gram = np.repeat((RIDGE*np.eye(FEATURES))[None], MEMBERS, 0)
        self.rhs = np.zeros((MEMBERS, FEATURES, OUTPUTS), np.float64)
        self.theta = np.zeros_like(self.rhs)
        self.bootstrap_counts = np.zeros(MEMBERS, np.int64)
        self.accepted_pairs = self.skipped_pairs = self.version = 0
        self.last_transition_id = -1
        self.pinned = None

    @property
    def additional_trainable_parameters(self): return MEMBERS*FEATURES*OUTPUTS

    @staticmethod
    def _phi(features, action, projection, phase):
        x = np.concatenate((features, action), axis=-1)
        if not np.isfinite(x).all(): raise ValueError('Nonfinite predictor inputs')
        return np.concatenate((np.ones((*x.shape[:-1], 1)), np.cos(x@projection+phase)), -1)/np.sqrt(FEATURES)

    def update(self, previous_image, action, next_image, reward, continuation, *,
               transition_id, previous_available, next_available, partition='training'):
        if partition != 'training': raise ValueError('Calibration/evaluation data cannot fit auxiliary heads')
        if type(transition_id) is not int or transition_id <= self.last_transition_id:
            raise ValueError('Repeated/out-of-order transition')
        if type(previous_available) is not bool or type(next_available) is not bool:
            raise ValueError('Explicit training availability required')
        if not previous_available or not next_available:
            # Never inspect a missing frame, even to validate its dtype.
            self.skipped_pairs += 1; self.last_transition_id = transition_id
            return {'updated': False, 'reason': 'adjacent_visible_pair_unavailable'}
        before, after = image_features(previous_image), image_features(next_image)
        a = np.asarray(action, np.float64)
        if before.shape != (16,) or after.shape != (16,) or a.shape != (6,) or not np.isfinite(a).all() or np.any(np.abs(a)>1):
            raise ValueError('One visible transition with bounded six-dimensional action required')
        if type(reward) not in (float, int) or not np.isfinite(reward) or not 0 <= reward <= 2:
            raise ValueError('Actual repeat2 reward must lie in [0,2]')
        if type(continuation) not in (float, int) or continuation not in (0., 1.):
            raise ValueError('Actual terminal flag required')
        y = np.concatenate((after, [reward/2., continuation]))
        for m in range(MEMBERS):
            phi = self._phi(before, a, self.projection[m], self.phase[m])
            w = int(self.bootstrap_rng[m].poisson(1.))
            residual = y - PRIOR_SCALE*(phi@self.prior[m])
            self.gram[m] += w*np.outer(phi, phi)
            self.rhs[m] += w*np.outer(phi, residual)
            self.bootstrap_counts[m] += w
        self.accepted_pairs += 1; self.last_transition_id = transition_id
        return {'updated': True, 'accepted_pairs': self.accepted_pairs}

    def pin(self):
        self.theta = np.stack([np.linalg.solve(self.gram[m], self.rhs[m]) for m in range(MEMBERS)])
        if not np.isfinite(self.theta).all(): raise FloatingPointError('Nonfinite bootstrap solution')
        self.version += 1
        self.pinned = {'schema': SCHEMA, 'binding': copy.deepcopy(self.binding), 'version': self.version,
            'accepted_pairs': self.accepted_pairs, 'additional_trainable_parameters': self.additional_trainable_parameters,
            'projection': self.projection.copy(), 'phase': self.phase.copy(),
            'coefficients': self.theta.copy()+PRIOR_SCALE*self.prior,
            'independent_bootstrap_counts': self.bootstrap_counts.copy(), 'calibrated_uncertainty': False}
        return FrozenBootstrap(self.pinned)

    def snapshot(self):
        fields = ('projection', 'phase', 'prior', 'gram', 'rhs', 'theta', 'bootstrap_counts',
                  'accepted_pairs', 'skipped_pairs', 'version', 'last_transition_id', 'pinned')
        out = {'schema': SCHEMA, 'binding': copy.deepcopy(self.binding),
               'state': {k:copy.deepcopy(getattr(self,k)) for k in fields},
               'rngs':[copy.deepcopy(r.bit_generator.state) for r in self.bootstrap_rng]}
        out['sha256'] = tree_digest(out)
        return out

    def load(self, record):
        r = copy.deepcopy(record)
        digest = r.pop('sha256', None)
        if digest != tree_digest(r) or r.get('schema') != SCHEMA or r.get('binding') != self.binding:
            raise ValueError('Auxiliary snapshot hash/schema/binding mismatch')
        expected = self.snapshot()['state']
        if set(r['state']) != set(expected) or len(r['rngs']) != MEMBERS: raise ValueError('Incomplete auxiliary state')
        for name, value in r['state'].items():
            if isinstance(expected[name], np.ndarray):
                if not isinstance(value, np.ndarray) or value.shape != expected[name].shape or value.dtype != expected[name].dtype:
                    raise ValueError('Auxiliary array schema mismatch: '+name)
            elif name != 'pinned' and (type(value) is not int or value < (-1 if name=='last_transition_id' else 0)):
                raise ValueError('Invalid auxiliary counter')
        for gram in r['state']['gram']:
            if not np.allclose(gram,gram.T,rtol=0,atol=1e-12): raise ValueError('Asymmetric bootstrap Gram')
            np.linalg.cholesky(gram)
        if np.any(r['state']['bootstrap_counts']<0):raise ValueError('Negative bootstrap multiplicity')
        if r['state']['pinned'] is not None:
            p=r['state']['pinned'];FrozenBootstrap(p)
            if (p['binding']!=self.binding or p['version']!=r['state']['version']
                    or p['accepted_pairs']>r['state']['accepted_pairs']
                    or np.any(p['independent_bootstrap_counts']>r['state']['bootstrap_counts'])):
                raise ValueError('Frozen heads do not belong to the saved learner')
            for name in ('projection','phase'):
                if not np.array_equal(p[name],r['state'][name]):raise ValueError('Pinned feature map differs from learner')
            if not np.array_equal(p['coefficients'],r['state']['theta']+PRIOR_SCALE*r['state']['prior']):
                raise ValueError('Pinned fitted coefficients differ from learner')
        rngs = [np.random.default_rng() for _ in range(MEMBERS)]
        for generator, state in zip(rngs,r['rngs']): generator.bit_generator.state = state
        for name,value in r['state'].items(): setattr(self,name,value)
        self.bootstrap_rng = rngs


class FrozenBootstrap:
    def __init__(self, record):
        r = copy.deepcopy(record)
        required={'schema','binding','version','accepted_pairs','additional_trainable_parameters',
                  'projection','phase','coefficients','independent_bootstrap_counts','calibrated_uncertainty'}
        if type(r) is not dict or set(r)!=required:raise ValueError('Frozen bootstrap schema differs')
        _binding(r['binding'])
        if r['schema'] != SCHEMA or type(r['version']) is not int or r['version'] < 1:
            raise ValueError('Bad frozen bootstrap identity')
        if (type(r['accepted_pairs']) is not int or r['accepted_pairs']<0
                or type(r['additional_trainable_parameters']) is not int
                or r['additional_trainable_parameters']!=MEMBERS*FEATURES*OUTPUTS
                or r['calibrated_uncertainty'] is not False):
            raise ValueError('Invalid frozen bootstrap counters/claims')
        counts=r['independent_bootstrap_counts']
        if not isinstance(counts,np.ndarray) or counts.shape!=(MEMBERS,) or counts.dtype!=np.int64 or np.any(counts<0):
            raise ValueError('Invalid independent bootstrap counts')
        for key,shape in [('projection',(MEMBERS,22,FEATURES-1)),('phase',(MEMBERS,FEATURES-1)),
                          ('coefficients',(MEMBERS,FEATURES,OUTPUTS))]:
            if not isinstance(r[key],np.ndarray) or r[key].dtype!=np.float64 or r[key].shape!=shape or not np.isfinite(r[key]).all():
                raise ValueError('Bad frozen bootstrap array '+key)
        self._record = r
        self.sha256 = tree_digest(r)
        self.version = r['version']
        self.ready = r['accepted_pairs'] >= 32 and np.all(r['independent_bootstrap_counts'] > 0)

    def snapshot(self): return copy.deepcopy(self._record)

    def initialize(self,image,batch=1):
        if type(batch) is not int or batch < 1: raise ValueError('Positive batch required')
        f = image_features(image)
        if f.shape != (16,): raise ValueError('Single visible anchor required')
        return np.broadcast_to(f,(MEMBERS,batch,16)).copy()

    def advance(self, features, actions):
        f,a = np.asarray(features,np.float64), np.asarray(actions,np.float64)
        if f.ndim!=3 or f.shape[0]!=MEMBERS or f.shape[2]!=16 or a.shape!=(f.shape[1],6) or np.any(np.abs(a)>1):
            raise ValueError('Expected independent feature states[M,B,16] and actions[B,6]')
        predictions = np.stack([BootstrapDynamics._phi(f[m],a,self._record['projection'][m],self._record['phase'][m])
                                @self._record['coefficients'][m] for m in range(MEMBERS)])
        if not np.isfinite(predictions).all(): raise FloatingPointError('Auxiliary rollout became nonfinite')
        disagreement = predictions[:,:,:16].var(axis=0).mean(-1)
        return predictions[:,:,:16].copy(), {'epistemic_disagreement':disagreement,
            'raw_reward_predictions':2.*predictions[:,:,16], 'raw_continuation_predictions':predictions[:,:,17],
            'heads_sha256':self.sha256, 'trained':bool(self.ready),
            'scope':'independent_bootstrap_coarse_visible_dynamics_proxy_not_Dreamer_ensemble'}

    def observed_error(self, predicted_features, image):
        f = np.asarray(predicted_features,np.float64)
        if f.ndim!=3 or f.shape[0]!=MEMBERS or f.shape[2]!=16 or not np.isfinite(f).all():
            raise ValueError('Actual pre-observation prediction required')
        return np.square(f.mean(0)-image_features(image)).mean(-1)
