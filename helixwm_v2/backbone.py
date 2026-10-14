"""Pinned official DreamerV3 adapter; no replacement learner or controller.

The upstream loss, optimizer, actor distribution and return calculations are
untouched. Sampling infrastructure differs: independent replay crops have an
explicit 32-observation filtering prefix, followed by the official 64-position
loss. Inference parameters are copied at episode boundaries and remain fixed
throughout that episode. Decoder scores are energies, not calibrated likelihoods.
"""
from __future__ import annotations

import copy
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


class OfficialDreamerBackbone:
    """Single-device adapter with explicit, independently seeded inference.

    Configuration options: seed, platform (cuda/cpu), compute_dtype,
    batch_size (16), batch_length (64), burnin (32), size (size12m).
    ``overrides`` accepts explicit upstream configuration overrides. The
    resolved configuration is retained for inspection and checkpoint identity.
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
        if size not in ("size1m", "size12m", "size25m", "size50m", "size100m", "size200m"):
            raise ValueError("Choose an existing official model-size preset")
        cfg = elements.Config(presets["defaults"])
        cfg = cfg.update(presets["dmc_vision"]).update(presets[size])
        cfg = cfg.update({
            "seed": int(config.get("seed", 0)), "logdir": str(Path(run_dir).resolve()),
            "task": "dmc_walker_walk", "env.dmc.repeat": 2,
            "batch_size": int(config.get("batch_size", 16)),
            "batch_length": int(config.get("batch_length", 64)),
            "replay_context": 0, "jax.prealloc": False,
            "jax.platform": config.get("platform", "cuda"),
            "jax.compute_dtype": config.get("compute_dtype", "bfloat16"),
            "jax.enable_policy": False,
        })
        if config.get("overrides"):
            cfg = cfg.update(config["overrides"])
        cfg_dict = dict(cfg)
        cfg_dict["jax"] = {**dict(cfg.jax), "precompile": False, "profiler": False}
        cfg = elements.Config(cfg_dict)
        if cfg.replay_context != 0 or cfg.env.dmc.proprio or not cfg.env.dmc.image:
            raise ValueError("Adapter requires pixel-only observations and replay_context=0")
        if tuple(cfg.env.dmc.size) != (64, 64) or cfg.env.dmc.repeat != 2:
            raise ValueError("The approved observation contract is RGB64 and repeat2")
        self.batch_size, self.batch_length = cfg.batch_size, cfg.batch_length
        self.burnin = int(config.get("burnin", 32))
        if self.burnin < 0:
            raise ValueError("burnin cannot be negative")
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
            replay_context=0, report_length=cfg.report_length,
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

        def prefix(state, previous_action, images, actions, resets):
            # Outgoing action[i] belongs to image[i]. Shift to obtain the
            # incoming action for each observation, matching official loss.
            incoming = jnp.concatenate((previous_action[:, None], actions[:, :-1]), 1)
            _, _, tokens = model.enc({}, {"image": images}, resets, training=False)
            state, _, _ = model.dyn.observe(state, tokens, {"action": incoming}, resets,
                                           training=False)
            return state

        def wrap(fn, static=(), foldin=None):
            pure = nj.pure(fn)
            def call(params, key, *args):
                if foldin is not None:
                    key = jax.random.fold_in(key, foldin)
                _, output = pure(params, *args, seed=key, create=False, modify=False)
                return output
            return jax.jit(call, static_argnums=static)

        self._observe = wrap(observe)
        self._prior = wrap(prior)
        self._score = wrap(score)
        self._act = wrap(act, static=(3,))
        self._prefix = wrap(prefix, foldin=0x48454C49)

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

    def train(self, batch: dict, previous_action=None):
        """Filter a 32-position prefix then call the official 64-position update.

        All batch actions are outgoing actions. Random crops use finite history;
        unlike the original replay-context path, no stale stored latent enters.
        """
        expected = set(self.agent.spaces)
        if set(batch) != expected:
            raise ValueError(f"Expected exactly {sorted(expected)}, got {sorted(batch)}")
        total = self.burnin + self.batch_length
        for key, space in self.agent.spaces.items():
            value = np.asarray(batch[key])
            if value.shape != (self.batch_size, total, *space.shape):
                raise ValueError(f"Invalid replay shape for {key}: {value.shape}")
            if value.dtype != np.dtype(space.dtype):
                raise ValueError(f"Invalid replay dtype for {key}: {value.dtype}")
            if np.issubdtype(value.dtype, np.floating) and not np.isfinite(value).all():
                raise FloatingPointError(f"Nonfinite replay field: {key}")
        B = self.batch_size
        previous_action = np.zeros((B, 6), np.float32) if previous_action is None else self._action(previous_action, B)
        counter = int(self.agent.n_batches)
        self.agent.n_batches.increment()
        train_seed = self.agent._seeds(counter, self.agent.train_mirrored)
        carry = self.agent.init_train(B)
        if self.burnin:
            prefix = self.burnin
            # Isolated folded key; official update key remains unchanged.
            args = (self.initial(B), previous_action, batch["image"][:, :prefix],
                    batch["action"][:, :prefix], batch["is_first"][:, :prefix])
            policy_params = {k: self.agent.params[k] for k in self.agent.policy_keys}
            state = self._prefix(policy_params, train_seed, *self._device(args))
            carry = ({}, state, {}, {"action": self._device(batch["action"][:, prefix - 1])})
        else:
            carry = (carry[0], carry[1], carry[2], {"action": self._device(previous_action)})
        trimmed = {k: np.asarray(v[:, self.burnin:]) for k, v in batch.items()}
        data = self.internal.device_put(trimmed, self.agent.train_sharded)
        data["seed"] = train_seed
        _, _, delayed = self.agent.train(carry, data)
        # Synchronous extraction avoids pending metrics/prefetch RNG ambiguity.
        metrics = self.flush()
        if not metrics:
            metrics = delayed
        return {k: float(v) for k, v in metrics.items() if np.asarray(v).ndim == 0}

    def flush(self):
        self.jax.block_until_ready(self.agent.params)
        metrics = {}
        if self.agent.pending_outs is not None:
            self.agent._take_outs(self.agent.pending_outs)
            self.agent.pending_outs = None
        if self.agent.pending_mets is not None:
            metrics = self.agent._take_outs(self.agent.pending_mets)
            self.agent.pending_mets = None
        self._last_flush_metrics = metrics
        return metrics

    def snapshot(self):
        self.flush()
        training = self.agent.save()
        if training["counters"]["batches"] != training["counters"]["updates"]:
            raise RuntimeError("Synchronous adapter has an unconsumed training batch")
        same = self.inference_version == int(self.agent.n_updates)
        inference = None if same else self.jax.device_get(self.inference_params)
        return {"schema_version": 1, "upstream_commit": UPSTREAM_COMMIT,
                "resolved_config": self.resolved_config, "burnin": self.burnin,
                "training": training, "inference_version": self.inference_version,
                "inference_equals_training": same, "inference_params": inference,
                "state_semantics": "separate caller-owned per-episode dual-chain state"}

    def load(self, snapshot):
        if snapshot["upstream_commit"] != UPSTREAM_COMMIT or snapshot["schema_version"] != 1:
            raise ValueError("Incompatible backbone checkpoint")
        if snapshot["burnin"] != self.burnin:
            raise ValueError("Checkpoint replay prefix differs")
        # logdir is an output destination; all algorithm fields must agree.
        old, new = copy.deepcopy(snapshot["resolved_config"]), copy.deepcopy(self.resolved_config)
        old.pop("logdir", None)
        new.pop("logdir", None)
        if old != new:
            raise ValueError("Resolved backbone configuration differs from checkpoint")
        counters = snapshot["training"]["counters"]
        if set(counters) != {"updates", "batches", "actions"} or any(
                not isinstance(v, int) or isinstance(v, bool) or v < 0
                for v in counters.values()):
            raise ValueError("Invalid official training counters")
        if counters["updates"] != counters["batches"]:
            raise ValueError("Synchronous checkpoint must not contain prefetched batches")
        if not 0 <= snapshot["inference_version"] <= counters["updates"]:
            raise ValueError("Invalid frozen inference version")
        if snapshot["inference_equals_training"] != (
                snapshot["inference_version"] == counters["updates"]):
            raise ValueError("Inference deduplication flag disagrees with update counter")
        self._validate_loaded_parameters(self.agent.params, snapshot["training"]["params"])
        if not snapshot["inference_equals_training"]:
            reference = {k: self.agent.params[k] for k in self.agent.policy_keys}
            self._validate_loaded_parameters(reference, snapshot["inference_params"])
        self.flush()
        self.agent.load(snapshot["training"])
        if snapshot["inference_equals_training"]:
            self.pin_inference()
        else:
            self.inference_params = self._device(snapshot["inference_params"])
            self.inference_version = int(snapshot["inference_version"])
        if self.inference_version != int(snapshot["inference_version"]):
            raise ValueError("Inference parameter version was not restored")

    @staticmethod
    def _validate_loaded_parameters(reference, saved):
        if not isinstance(saved, dict) or set(reference) != set(saved):
            raise ValueError("Checkpoint parameter keys differ")
        for key in reference:
            value, expected = np.asarray(saved[key]), reference[key]
            if value.shape != expected.shape or value.dtype != expected.dtype:
                raise ValueError(f"Checkpoint parameter schema differs: {key}")
            if not np.isfinite(value).all():
                raise FloatingPointError(f"Checkpoint contains nonfinite parameter: {key}")

    def parameter_hash(self, inference=False):
        params = self.inference_params if inference else self.agent.save()["params"]
        return array_tree_hash(self.jax.device_get(params))


def initialize(config: dict, upstream_path, run_dir):
    return OfficialDreamerBackbone(config, upstream_path, run_dir)
