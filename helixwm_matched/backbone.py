"""Official e3f0224 core with native replay-context learning.

Inference uses shared explicit latent/actor keys for both matched arms. This is
a declared harness RNG adaptation, not bitwise original Agent.policy scheduling.
Replay entries are the actual current controller carry; the official learner
refreshes them through its native replay.update output. No custom burn-in.
"""
from __future__ import annotations

import copy
import io
import json
from collections import defaultdict, deque
import hashlib
from pathlib import Path
import subprocess
import sys

import numpy as np


UPSTREAM_COMMIT = "e3f02248693a79dc8b0ebd62c93683888ddaccfe"
SCORE_KIND = "negative_official_pixel_mse_energy"


def validate_upstream(path: str | Path) -> Path:
    path = Path(path).resolve()
    def git(*args):
        return subprocess.run(["git", "-C", str(path), *args], check=True,
                              capture_output=True, text=True).stdout.strip()
    if git("rev-parse", "HEAD") != UPSTREAM_COMMIT:
        raise ValueError("The backbone must use the approved upstream commit")
    if git("status", "--porcelain", "--untracked-files=no"):
        raise ValueError("Tracked upstream files must remain unchanged")
    for name in ("dreamerv3/agent.py", "dreamerv3/rssm.py", "embodied/jax/agent.py"):
        if not (path / name).is_file():
            raise ValueError(f"Missing official backbone source: {name}")
    return path


def rng_key(rng) -> np.ndarray:
    """Make an explicit JAX key; never touch NumPy's global random generator."""
    if isinstance(rng, np.random.Generator):
        return rng.integers(0, 2**32, size=2, dtype=np.uint32)
    if isinstance(rng, (int, np.integer)):
        seed = int(rng)
        if not 0 <= seed < 2**32:
            raise ValueError("Integer RNG seeds must fit uint32")
        return np.array([0, seed], np.uint32)
    value = np.asarray(rng)
    if value.shape != (2,) or value.dtype != np.uint32:
        raise ValueError("Pass an explicit seed, uint32[2] key, or Generator")
    return value.copy()


def array_tree_hash(tree) -> str:
    """Stable hash for nested checkpoint arrays, preserving shapes and dtypes."""
    digest = hashlib.sha256()
    def visit(value):
        if isinstance(value, dict):
            digest.update(b"dict")
            for key in sorted(value):
                digest.update(str(key).encode("utf-8") + b"\0")
                visit(value[key])
        elif isinstance(value, (tuple, list)):
            digest.update(type(value).__name__.encode())
            for item in value:
                visit(item)
        elif value is None:
            digest.update(b"none")
        else:
            array = np.asarray(value)
            if array.dtype.hasobject:
                raise TypeError("Object arrays cannot be hashed as model state")
            digest.update(str(array.dtype).encode() + str(array.shape).encode())
            digest.update(np.ascontiguousarray(array).tobytes())
    visit(tree)
    return digest.hexdigest()


class MatchedDreamerBackbone:
    """Pinned size12m/B16/64/BF16 learner; episode-fixed inference copy.

    Config accepts seed, platform and no algorithm overrides. Replay context is
    exactly one. Caller owns the collector's four explicit RNG streams and must
    checkpoint them alongside this object's snapshot and NativeReplay snapshot.
    """

    score_kind = SCORE_KIND

    def __init__(self, config: dict, upstream_path: str | Path,
                 run_dir: str | Path):
        self.upstream_path = validate_upstream(upstream_path)
        for package in ("dreamerv3", "embodied"):
            module = sys.modules.get(package)
            if module is not None and not Path(module.__file__).resolve().is_relative_to(self.upstream_path):
                raise RuntimeError(f"Another {package} checkout is already imported")
        sys.path.insert(0, str(self.upstream_path))
        import elements
        import jax
        import jax.numpy as jnp
        import ninjax as nj
        from ruamel.yaml import YAML
        from dreamerv3.agent import Agent
        from embodied.jax import internal
        self.jax, self.jnp, self.nj, self.internal = jax, jnp, nj, internal
        self.requested_config = copy.deepcopy(config)
        presets = YAML(typ="safe").load((self.upstream_path / "dreamerv3/configs.yaml").read_text())
        size = config.get("size", "size12m")
        if size != "size12m":
            raise ValueError("Matched study requires the official size12m preset")
        for key, expected in {"batch_size": 16, "batch_length": 64,
                              "compute_dtype": "bfloat16", "replay_context": 1,
                              "burnin": 0}.items():
            if key in config and config[key] != expected:
                raise ValueError(f"Matched study requires {key}={expected!r}")
        if config.get("overrides"):
            raise ValueError("Algorithm overrides are not allowed in the matched study")
        cfg = elements.Config(presets["defaults"])
        cfg = cfg.update(presets["dmc_vision"]).update(presets[size])
        cfg = cfg.update({
            "seed": int(config.get("seed", 0)), "logdir": str(Path(run_dir).resolve()),
            "task": "dmc_walker_walk", "env.dmc.repeat": 2,
            "batch_size": int(config.get("batch_size", 16)),
            "batch_length": int(config.get("batch_length", 64)),
            "replay_context": 1, "jax.prealloc": False,
            "jax.platform": config.get("platform", "cuda"),
            "jax.compute_dtype": config.get("compute_dtype", "bfloat16"),
            "jax.enable_policy": False,
        })
        if config.get("overrides"):
            cfg = cfg.update(config["overrides"])
        cfg_dict = dict(cfg)
        cfg_dict["jax"] = {**dict(cfg.jax), "precompile": False, "profiler": False}
        cfg = elements.Config(cfg_dict)
        if cfg.replay_context != 1 or cfg.env.dmc.proprio or not cfg.env.dmc.image:
            raise ValueError("Adapter requires pixel-only observations and replay_context=1")
        if tuple(cfg.env.dmc.size) != (64, 64) or cfg.env.dmc.repeat != 2:
            raise ValueError("The approved observation contract is RGB64 and repeat2")
        self.batch_size, self.batch_length = cfg.batch_size, cfg.batch_length
        self.burnin = 0
        self.config = cfg
        self.resolved_config = dict(cfg)
        obs_space = {
            "image": elements.Space(np.uint8, (64, 64, 3)),
            "reward": elements.Space(np.float32),
            "is_first": elements.Space(bool), "is_last": elements.Space(bool),
            "is_terminal": elements.Space(bool),
        }
        act_space = {"action": elements.Space(np.float32, (6,), -1, 1)}
        agent_cfg = elements.Config(
            **cfg.agent, logdir=cfg.logdir, seed=cfg.seed, jax=cfg.jax,
            batch_size=cfg.batch_size, batch_length=cfg.batch_length,
            replay_context=1, report_length=cfg.report_length,
            replica=0, replicas=1,
        )
        self.agent = Agent(obs_space, act_space, agent_cfg)
        self.model = self.agent.model
        if self.model.enc.entry_space or self.model.dec.entry_space:
            raise RuntimeError("This adapter only supports stateless encoder and decoder")
        self.inference_params = None
        self.inference_version = -1
        self._last_flush_metrics = {}
        self._build_inference()
        self.pin_inference()
        self.train_carry = self.agent.init_train(self.batch_size)
        self.writeback_updates = 0

    def _build_inference(self):
        jax, jnp, nj, model = self.jax, self.jnp, self.nj, self.model
        from embodied.jax import nets as nn

        def observe(state, actions, images, reset):
            _, _, tokens = model.enc({}, {"image": images}, reset, training=False, single=True)
            state, _, _ = model.dyn.observe(state, tokens, {"action": actions}, reset,
                                           training=False, single=True)
            return state

        def prior(state, actions):
            # RSSM.observe casts its input carry; imagine(single=True) expects
            # the already-cast carry normally supplied by official imagination.
            state, _ = model.dyn.imagine(nn.cast(state), {"action": nn.cast(actions)}, length=1,
                                         training=False, single=True)
            return state

        def score(state, images):
            reset = jnp.zeros(state["deter"].shape[:1], bool)
            _, _, recon = model.dec({}, state, reset, training=False, single=True)
            return -recon["image"].loss(images.astype(jnp.float32) / 255.0)

        def act(state, deterministic):
            distribution = model.pol(model.feat2tensor(state), bdims=1)["action"]
            return distribution.pred() if deterministic else distribution.sample(nj.seed())

        def wrap(fn, static=(), foldin=None, scan_bookkeeping=False):
            pure = nj.pure(fn)
            def call(params, key, *args):
                if foldin is not None:
                    key = jax.random.fold_in(key, foldin)
                if scan_bookkeeping:
                    # Ninjax 3.5.1 _prerun returns one set with modify=False,
                    # but scan requires (accessed, modified). Permit its scan
                    # bookkeeping, then reject ANY actual state write while
                    # tracing. No parameter updates escape this pure call.
                    _, output, _, modified, created = pure(
                        params, *args, seed=key, create=False, modify=True, track=True)
                    if modified or created:
                        raise RuntimeError("Read-only prefix attempted to modify model state")
                else:
                    _, output = pure(params, *args, seed=key, create=False, modify=False)
                return output
            return jax.jit(call, static_argnums=static)

        self._observe = wrap(observe)
        self._prior = wrap(prior)
        self._score = wrap(score)
        self._act = wrap(act, static=(3,))

    def initial(self, batch=1):
        if not isinstance(batch, int) or batch < 1:
            raise ValueError("batch must be a positive integer")
        c = self.config.agent.dyn.rssm
        return {"deter": np.zeros((batch, c.deter), np.float32),
                "stoch": np.zeros((batch, c.stoch, c.classes), np.float32)}

    def _state(self, state):
        if set(state) != {"deter", "stoch"}:
            raise ValueError("State must contain deter and stoch only")
        state = {k: np.asarray(v) for k, v in state.items()}
        batch = state["deter"].shape[0]
        expected = self.initial(batch)
        if any(state[k].shape != expected[k].shape for k in state):
            raise ValueError("State does not match the pinned latent schema")
        if any(not np.isfinite(v).all() for v in state.values()):
            raise FloatingPointError("Nonfinite latent state")
        return {k: np.asarray(v, np.float32) for k, v in state.items()}

    @staticmethod
    def _image(image, batch):
        image = np.asarray(image)
        if image.dtype != np.uint8:
            raise ValueError("Images must be uint8; preprocessing belongs to upstream")
        if image.shape == (64, 64, 3):
            image = np.broadcast_to(image, (batch, 64, 64, 3)).copy()
        if image.shape != (batch, 64, 64, 3):
            raise ValueError("Image shape must be [64,64,3] or [B,64,64,3]")
        return image

    @staticmethod
    def _action(action, batch):
        action = np.asarray(action, np.float32)
        if action.shape == (6,):
            action = np.broadcast_to(action, (batch, 6)).copy()
        if action.shape != (batch, 6) or not np.isfinite(action).all():
            raise ValueError("Action must be finite [6] or [B,6]")
        return action

    def _device(self, value):
        return self.internal.device_put(value, self.agent.train_mirrored)

    def _host(self, value):
        result = self.jax.device_get(value)
        def convert(x):
            x = np.asarray(x)
            return x.astype(np.float32) if str(x.dtype) == "bfloat16" else x.copy()
        return self.jax.tree.map(convert, result)

    def observe(self, previous, action, image, is_first=False, rng=None):
        state = self._state(previous)
        batch = len(state["deter"])
        reset = np.broadcast_to(np.asarray(is_first, bool), (batch,)).copy()
        args = (state, self._action(action, batch), self._image(image, batch), reset)
        return self._host(self._observe(self.inference_params, self._device(rng_key(rng)),
                                        *self._device(args)))

    def prior(self, previous, action, rng=None):
        state = self._state(previous)
        args = (state, self._action(action, len(state["deter"])))
        return self._host(self._prior(self.inference_params, self._device(rng_key(rng)),
                                      *self._device(args)))

    imagine = prior

    def score(self, states, image):
        state = self._state(states)
        args = (state, self._image(image, len(state["deter"])))
        output = self._score(self.inference_params, self._device(rng_key(0)),
                             *self._device(args))
        return self._host(output).astype(np.float64)

    score_candidates = score

    def act(self, state, deterministic=False, rng=None):
        # Original bounded-mean Normal can sample outside action bounds.
        # The environment clips, as in official ClipAction; retain that fact.
        state = self._state(state)
        key = rng_key(0 if deterministic and rng is None else rng)
        return self._host(self._act(self.inference_params, self._device(key),
                                    self._device(state), bool(deterministic)))

    def pin_inference(self):
        """Copy current policy/model parameters once, normally at episode reset."""
        with self.agent.train_lock:
            params = {k: self.agent.params[k].copy() for k in self.agent.policy_keys}
            self.jax.block_until_ready(params)
            self.inference_params = params
            self.inference_version = int(self.agent.n_updates)
        return self.inference_version

    def replay_entries(self, state):
        """The current x_t-filtered controller state, paired with outgoing a_t.

        H stores its actual repaired/prior state, D its native image posterior.
        These are causal carries, not hidden clean-image posteriors. Native
        replay-context learning subsequently refreshes both via replay.update.
        """
        state = self._state(state)
        if len(state['deter']) != 1:
            raise ValueError('Collector replay entries require batch size one')
        return {'dyn/' + k: v[0].copy() for k, v in state.items()}

    def train(self, batch, replay):
        """One synchronous official update, including immediate native writeback."""
        if set(batch) != set(self.agent.spaces):
            raise ValueError('Native replay batch keys differ from agent.spaces')
        for key, space in self.agent.spaces.items():
            value = np.asarray(batch[key])
            if value.shape != (16, 65, *space.shape) or value.dtype != np.dtype(space.dtype):
                raise ValueError(f'Invalid native replay schema: {key}')
            if np.issubdtype(value.dtype, np.floating) and not np.isfinite(value).all():
                raise FloatingPointError(f'Nonfinite replay field: {key}')
        if np.any(batch['consec'] != 0):
            raise ValueError('This protocol requires consec_train=1')
        if self.agent.pending_outs is not None or self.agent.pending_mets is not None:
            raise RuntimeError('Unconsumed previous update output')
        counter = int(self.agent.n_batches)
        self.agent.n_batches.increment()
        data = self.internal.device_put(batch, self.agent.train_sharded)
        data['seed'] = self.agent._seeds(counter, self.agent.train_mirrored)
        self.train_carry, stale_outs, stale_metrics = self.agent.train(self.train_carry, data)
        if stale_outs or stale_metrics:
            raise RuntimeError('Unexpected delayed output in synchronous learner')
        self.jax.block_until_ready(self.agent.params)
        outs = self.agent._take_outs(self.agent.pending_outs)
        metrics = self.agent._take_outs(self.agent.pending_mets)
        # Keep pending outputs intact if validation or writeback fails: such an
        # update cannot be checkpointed as successfully committed.
        if set(outs) != {'replay'}:
            raise RuntimeError('Official replay-context update did not return entries')
        updates = outs['replay']
        required = {'stepid', 'dyn/deter', 'dyn/stoch'}
        if set(updates) != required or any(v.shape[:2] != (16, 64) for v in updates.values()):
            raise RuntimeError('Native replay writeback shape or keys differ')
        if not np.array_equal(updates['stepid'], batch['stepid'][:, 1:]):
            raise RuntimeError('Native replay writeback timestamp alignment differs')
        if any(not np.isfinite(v).all() for v in updates.values()):
            raise FloatingPointError('Nonfinite replay writeback')
        replay.update({k: np.asarray(v).copy() for k, v in updates.items()})
        self.writeback_updates += 1
        self.agent.pending_outs = self.agent.pending_mets = None
        scalars = {k: float(v) for k, v in metrics.items() if np.asarray(v).ndim == 0}
        if any(not np.isfinite(v) for v in scalars.values()):
            raise FloatingPointError('Nonfinite official training metric')
        return scalars

    def flush(self):
        self.jax.block_until_ready(self.agent.params)
        if self.agent.pending_outs is not None or self.agent.pending_mets is not None:
            raise RuntimeError('Cannot discard pending native replay updates')

    def snapshot(self):
        self.flush()
        training = self.agent.save()
        counters = training['counters']
        if counters['batches'] != counters['updates'] or self.writeback_updates != counters['updates']:
            raise RuntimeError('Training/writeback counters are not synchronized')
        return {'format': 'helixwm_matched_backbone_v1', 'upstream_commit': UPSTREAM_COMMIT,
                'resolved_config': copy.deepcopy(self.resolved_config), 'training': training,
                'train_carry': self.jax.device_get(self.train_carry),
                'writeback_updates': self.writeback_updates,
                'inference_version': self.inference_version,
                'inference_params': self.jax.device_get(self.inference_params),
                'rng_semantics': 'official_training_counter_keys; caller_owned_split_policy_keys'}

    def load(self, snapshot):
        if snapshot.get('format') != 'helixwm_matched_backbone_v1' or snapshot.get('upstream_commit') != UPSTREAM_COMMIT:
            raise ValueError('Incompatible matched backbone checkpoint')
        old, new = copy.deepcopy(snapshot['resolved_config']), copy.deepcopy(self.resolved_config)
        old.pop('logdir', None)
        new.pop('logdir', None)
        if old != new:
            raise ValueError('Resolved backbone configuration differs')
        counters = snapshot['training']['counters']
        if set(counters) != {'updates', 'batches', 'actions'} or any(type(v) is not int or v < 0 for v in counters.values()):
            raise ValueError('Invalid official training counters')
        if counters['updates'] != counters['batches'] or snapshot['writeback_updates'] != counters['updates']:
            raise ValueError('Incomplete native replay writeback checkpoint')
        if type(snapshot['inference_version']) is not int or not 0 <= snapshot['inference_version'] <= counters['updates']:
            raise ValueError('Invalid pinned inference version')
        self._validate_loaded_parameters(self.agent.params, snapshot['training']['params'])
        self._validate_loaded_parameters({k: self.agent.params[k] for k in self.agent.policy_keys}, snapshot['inference_params'])
        _validate_tree(self.jax.device_get(self.train_carry), snapshot['train_carry'])
        self.flush()
        self.agent.load(snapshot['training'])
        self.train_carry = self.internal.device_put(snapshot['train_carry'], self.agent.train_sharded)
        self.inference_params = self._device(snapshot['inference_params'])
        self.inference_version = snapshot['inference_version']
        self.writeback_updates = snapshot['writeback_updates']

    @staticmethod
    def _validate_loaded_parameters(reference, saved):
        _validate_tree(reference, saved)

    def parameter_hash(self, inference=False):
        params = self.inference_params if inference else self.agent.save()['params']
        return array_tree_hash(self.jax.device_get(params))


def _validate_tree(reference, saved):
    if isinstance(reference, dict):
        if not isinstance(saved, dict) or reference.keys() != saved.keys():
            raise ValueError('Checkpoint tree keys differ')
        for key in reference:
            _validate_tree(reference[key], saved[key])
    elif isinstance(reference, (tuple, list)):
        if type(saved) is not type(reference) or len(reference) != len(saved):
            raise ValueError('Checkpoint tree sequence differs')
        for left, right in zip(reference, saved):
            _validate_tree(left, right)
    else:
        # A live reference can be a JAX device array under transfer_guard='disallow'.
        # Schema validation needs metadata only; np.asarray(reference) would
        # perform a forbidden implicit device-to-host transfer. Saved leaves
        # must already be host arrays from the serialized checkpoint.
        if not hasattr(reference, 'shape') or not hasattr(reference, 'dtype'):
            raise ValueError('Checkpoint reference must have array shape/dtype metadata')
        if not isinstance(saved, (np.ndarray, np.generic)):
            raise ValueError('Saved checkpoint arrays must reside on the host')
        value = np.asarray(saved)
        if tuple(reference.shape) != value.shape or np.dtype(reference.dtype) != value.dtype or value.dtype.hasobject:
            raise ValueError('Checkpoint array schema differs')
        if not np.isfinite(value).all():
            raise FloatingPointError('Nonfinite checkpoint array')


def _array_id(value):
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(value.dtype.str.encode() + b'\0')
    digest.update(json.dumps(list(value.shape)).encode() + b'\0')
    digest.update(value.tobytes())
    return digest.hexdigest()


def _encode_array(value):
    """Lossless codec only: BF16-valued f32 gets reversible bit packing."""
    value = np.ascontiguousarray(value)
    if value.dtype == np.float32 and not np.any(value.view(np.uint32) & 0xffff):
        return 'float32_upper16_exact', (value.view(np.uint32) >> 16).astype(np.uint16)
    return 'identity', value


def _decode_array(codec, stored):
    if codec == 'float32_upper16_exact':
        if stored.dtype != np.uint16:
            raise ValueError('Invalid exact BF16 codec input')
        return (stored.astype(np.uint32) << 16).view(np.float32)
    if codec != 'identity' or stored.dtype.hasobject:
        raise ValueError('Unknown or unsafe replay codec')
    return stored


class NativeReplay:
    """Official Replay add/sample/update with exact, versioned persistence.

    The author's save/load cannot preserve mutable latent writebacks or selector
    RNG/order. We persist those explicitly without replacing the sampler or the
    native Consec stream. Chunk fields are immutable content-addressed objects;
    same pixels across checkpoint versions are stored once. Versioned recurrent
    entries can still be large: a measured storage forecast must gate launch.
    Use only from the synchronous collector/learner thread, with online=False.
    """

    def __init__(self, backbone, directory, capacity=1_000_000, seed=0, chunksize=1024):
        import embodied
        from embodied.core import chunk, selectors
        self.backbone = backbone
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.capacity, self.seed, self.chunksize = int(capacity), int(seed), int(chunksize)
        if self.capacity < 65 or self.chunksize < 65:
            raise ValueError('Replay capacity/chunksize must hold a full 65-position sample')
        self._chunk_class = chunk.Chunk
        self._uniform_class = selectors.Uniform
        self._embodied = embodied
        self.replay = embodied.Replay(length=65, capacity=self.capacity, chunksize=self.chunksize,
                                      directory=None, online=False, seed=self.seed)
        self.rows_added = 0
        self._make_stream()

    def _make_stream(self):
        source = self._embodied.streams.Stateless(lambda: self.replay.sample(16, 'train'))
        self.stream = iter(self._embodied.streams.Consec(source, length=64, consec=1,
                            prefix=1, strict=True, contiguous=True))

    def __len__(self):
        return len(self.replay)

    def add(self, observation, outgoing_action, state):
        """Add x_t, incoming r_t, actual outgoing a_t, and current state_t.

        Include reset and terminal observations (terminal outgoing action zero).
        The caller must supply only the visible image, including known blackout
        zeros, and update its actual-interaction counter separately from rows.
        """
        expected = set(self.backbone.agent.obs_space)
        if set(observation) != expected:
            raise ValueError('Unexpected replay observation keys')
        row = {k: np.asarray(v) for k, v in observation.items()}
        action = np.asarray(outgoing_action)
        if action.dtype != np.float32 or action.shape != (6,) or not np.isfinite(action).all() or np.any(np.abs(action) > 1):
            raise ValueError('Store the actual executed clipped float32 action')
        if bool(row['is_last']) and np.any(action != 0):
            raise ValueError('Final observation has no outgoing environment action')
        row['action'] = action.copy()
        row.update(self.backbone.replay_entries(state))
        for key, value in row.items():
            space = self.backbone.agent.spaces[key]
            if value.shape != tuple(space.shape) or value.dtype != np.dtype(space.dtype):
                raise ValueError(f'Invalid replay row schema: {key}')
            if np.issubdtype(value.dtype, np.floating) and not np.isfinite(value).all():
                raise FloatingPointError(f'Nonfinite replay row: {key}')
        self.replay.add(row, worker=0)
        self.rows_added += 1

    def train_once(self, backbone=None):
        if backbone is not None and backbone is not self.backbone:
            raise ValueError('Replay is bound to a different backbone')
        if not len(self):
            raise RuntimeError('Replay has no complete 65-position sequence')
        return self.backbone.train(next(self.stream), self.replay)

    def _put_array(self, value):
        value = np.ascontiguousarray(value)
        identity = _array_id(value)
        path = self.directory / (identity + '.npz')
        codec, encoded = _encode_array(value)
        if not path.exists():
            temporary = path.with_suffix('.npz.partial')
            with temporary.open('xb') as stream:
                np.savez_compressed(stream, value=encoded)
            temporary.replace(path)
        # Persist actual storage hash as well as decoded value hash. Never
        # overwrite a same-named object if external modification is detected.
        record = {'path': path.name, 'value_sha256': identity, 'codec': codec,
                  'dtype': value.dtype.str, 'shape': list(value.shape),
                  'file_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                  'bytes': path.stat().st_size}
        if not np.array_equal(self._get_array(record).view(np.uint8), value.view(np.uint8)):
            raise ValueError('Stored replay object does not match its content identity')
        return record

    def _get_array(self, record):
        name = record['path']
        if name != record['value_sha256'] + '.npz' or Path(name).name != name:
            raise ValueError('Invalid replay object path')
        path = self.directory / name
        raw = path.read_bytes()
        if len(raw) != record['bytes'] or hashlib.sha256(raw).hexdigest() != record['file_sha256']:
            raise ValueError('Replay object storage SHA256 mismatch')
        with np.load(io.BytesIO(raw), allow_pickle=False) as data:
            if data.files != ['value']:
                raise ValueError('Unexpected replay object members')
            value = _decode_array(record['codec'], data['value'])
        if value.dtype.str != record['dtype'] or list(value.shape) != record['shape'] or _array_id(value) != record['value_sha256']:
            raise ValueError('Replay object decoded SHA256/schema mismatch')
        return value

    def snapshot(self):
        replay = self.replay
        if replay.online or type(replay.sampler) is not self._uniform_class:
            raise RuntimeError('Only pinned native Uniform/online=False is supported')
        chunks = []
        with replay.rwlock.writing:
            for key, chunk in replay.chunks.items():
                arrays = {} if not chunk.length else {k: self._put_array(v[:chunk.length]) for k, v in chunk.data.items()}
                chunks.append({'uuid': str(key), 'succ': str(chunk.succ), 'time': chunk.time,
                               'length': chunk.length, 'size': chunk.size, 'arrays': arrays})
            sampler = replay.sampler
            return {'format': 'helixwm_native_replay_exact_v1', 'upstream_commit': UPSTREAM_COMMIT,
                    'capacity': self.capacity, 'chunksize': self.chunksize, 'seed': self.seed,
                    'length': 65, 'online': False, 'rows_added': self.rows_added,
                    'stream_index': int(self.stream.index), 'chunks': chunks,
                    'refs': {str(k): v for k, v in replay.refs.items()},
                    'items': [[k, str(c), i] for k, (c, i) in replay.items.items()],
                    'fifo': list(replay.fifo), 'itemid': replay.itemid,
                    'current': [[w, str(c), i] for w, (c, i) in replay.current.items()],
                    'streams': [[w, [[str(c), i] for c, i in st]] for w, st in replay.streams.items()],
                    'sampler_keys': list(sampler.keys),
                    'sampler_indices': [[k, i] for k, i in sampler.indices.items()],
                    'sampler_rng': copy.deepcopy(sampler.rng.bit_generator.state),
                    'metrics': copy.deepcopy(replay.metrics)}

    def load(self, saved):
        import elements
        identity = ('format', 'upstream_commit', 'capacity', 'chunksize', 'seed', 'length', 'online')
        required = ('helixwm_native_replay_exact_v1', UPSTREAM_COMMIT, self.capacity, self.chunksize, self.seed, 65, False)
        if tuple(saved.get(k) for k in identity) != required or len(self) or self.rows_added:
            raise ValueError('Replay restore requires matching configuration and an empty receiver')
        if saved['stream_index'] not in (0, 1) or type(saved['rows_added']) is not int or saved['rows_added'] < 0:
            raise ValueError('Invalid stream/row counters')
        staged = self._embodied.Replay(length=65, capacity=self.capacity, chunksize=self.chunksize,
                                       directory=None, online=False, seed=self.seed)
        UUID = elements.UUID
        for info in saved['chunks']:
            key = UUID(info['uuid'])
            if key in staged.chunks or not 0 <= info['length'] <= info['size'] or info['size'] != self.chunksize:
                raise ValueError('Duplicate or invalid replay chunk')
            chunk = self._chunk_class(info['size'])
            chunk.uuid, chunk.succ, chunk.time = key, UUID(info['succ']), info['time']
            chunk.length, chunk.saved = info['length'], False
            chunk.data = None
            if chunk.length:
                expected = set(self.backbone.agent.spaces) - {'consec'}
                if set(info['arrays']) != expected:
                    raise ValueError('Replay chunk fields differ')
                chunk.data = {}
                for name, record in info['arrays'].items():
                    value = self._get_array(record)
                    space = self.backbone.agent.spaces[name]
                    if value.shape != (chunk.length, *space.shape) or value.dtype != np.dtype(space.dtype):
                        raise ValueError('Replay chunk dtype/shape mismatch')
                    chunk.data[name] = np.empty((chunk.size, *value.shape[1:]), value.dtype)
                    chunk.data[name][:chunk.length] = value
            elif info['arrays']:
                raise ValueError('Empty replay chunk contains arrays')
            staged.chunks[key] = chunk
        staged.refs = {UUID(k): int(v) for k, v in saved['refs'].items()}
        staged.items = {int(k): (UUID(c), int(i)) for k, c, i in saved['items']}
        staged.fifo = deque(saved['fifo'])
        staged.itemid = int(saved['itemid'])
        staged.current = {w: (UUID(c), int(i)) for w, c, i in saved['current']}
        staged.streams = defaultdict(deque, {w: deque((UUID(c), int(i)) for c, i in st) for w, st in saved['streams']})
        sampler = staged.sampler
        sampler.keys = list(saved['sampler_keys'])
        sampler.indices = dict(saved['sampler_indices'])
        sampler.rng.bit_generator.state = copy.deepcopy(saved['sampler_rng'])
        staged.metrics = copy.deepcopy(saved['metrics'])
        if set(staged.refs) != set(staged.chunks) or any(v < 0 for v in staged.refs.values()):
            raise ValueError('Replay reference mapping differs')
        if len(staged.items) != len(saved['items']) or len(staged.items) > self.capacity:
            raise ValueError('Replay item count invalid')
        if sorted(staged.fifo) != sorted(staged.items) or set(sampler.keys) != set(staged.items) or len(sampler.keys) != len(staged.items):
            raise ValueError('Replay FIFO/selector items differ')
        if sampler.indices != {k: i for i, k in enumerate(sampler.keys)}:
            raise ValueError('Native selector order differs')
        if staged.items and staged.itemid <= max(staged.items):
            raise ValueError('Replay item counter would reuse an ID')
        if set(staged.current) != set(staged.streams):
            raise ValueError('Replay worker maps differ')
        for c, i in staged.current.values():
            if c not in staged.chunks or i != staged.chunks[c].length or i >= self.chunksize:
                raise ValueError('Invalid active chunk cursor')
        for st in staged.streams.values():
            if len(st) >= 65:
                raise ValueError('Invalid pending sequence length')
            for c, i in st:
                if c not in staged.chunks or not 0 <= i < staged.chunks[c].length:
                    raise ValueError('Invalid pending sequence reference')
        expected_refs = {key: 0 for key in staged.chunks}
        for c, _ in staged.current.values():
            expected_refs[c] += 1
        for st in staged.streams.values():
            for c, _ in st:
                expected_refs[c] += 1
        for c, _ in staged.items.values():
            if c not in expected_refs:
                raise ValueError('Replay item references a missing chunk')
            expected_refs[c] += 1
        for chunk in staged.chunks.values():
            if chunk.succ in expected_refs:
                expected_refs[chunk.succ] += 1
        if staged.refs != expected_refs:
            raise ValueError('Native chunk reference counts differ')
        # Check all native sequence chains, not only the next random sample.
        for c, i in staged.items.values():
            try:
                if staged._getseq(c, i, ['stepid'])['stepid'].shape != (65, 20):
                    raise ValueError('Invalid native replay sequence length')
            except (KeyError, AssertionError) as exc:
                raise ValueError('Broken native replay sequence') from exc
        self.replay, self.rows_added = staged, saved['rows_added']
        self._make_stream()
        self.stream.index = saved['stream_index']

    @staticmethod
    def referenced_files(saved):
        """Portable object inventory required by this particular checkpoint."""
        files = {}
        for chunk in saved['chunks']:
            for record in chunk['arrays'].values():
                previous = files.setdefault(record['path'], record)
                if previous != record:
                    raise ValueError('Conflicting content-addressed object metadata')
        return [files[k] for k in sorted(files)]

    def statistics(self):
        return {'native_sequences': len(self), 'observation_rows': self.rows_added,
                'ram_bytes': sum(x.nbytes for x in self.replay.chunks.values()),
                'sample_positions': 65, 'learning_positions_per_update': 1024,
                'sampler': 'official_Uniform', 'online': False}


def initialize(config, upstream_path, run_dir):
    return MatchedDreamerBackbone(config, upstream_path, run_dir)


NativeReplayAdapter = NativeReplay

