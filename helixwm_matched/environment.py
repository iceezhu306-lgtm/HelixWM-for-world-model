"""One shared Walker environment; only visible RGB leaves its normal interface.

reset(seed, mask) -> uint8[64,64,3]
step(action) -> (visible_rgb, repeat_reward_sum, done, info)

All diagnostics that expose physical state or unmasked renders are opt-in and
reserved for the environment probe/evaluator. No JAX or simulator import occurs
until a real environment is reset. A new simulator is created for every explicit
episode seed; neither arm inherits an undocumented reset-RNG trajectory.
"""
from __future__ import annotations

import copy
import os
import numpy as np

from .protocol import array_sha256, validate_mask, validate_protocol


def _native_factory(seed: int):
    os.environ.setdefault("MUJOCO_GL", "egl")
    from dm_control import suite
    env = suite.load("walker", "walk", task_kwargs={"random": seed},
                     environment_kwargs={"flat_observation": False})
    env._step_limit = 1000
    return env


def _physics_hash(env) -> str:
    # Mock-only hook allows unit tests without importing a physics runtime.
    if hasattr(env, "diagnostic_state"):
        return array_sha256(env.diagnostic_state())
    import mujoco
    physics = env.physics
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    state = np.empty(mujoco.mj_stateSize(physics.model.ptr, spec), np.float64)
    mujoco.mj_getState(physics.model.ptr, physics.data.ptr, state, spec)
    if not np.isfinite(state).all():
        raise FloatingPointError("Nonfinite physics diagnostic state")
    return array_sha256(state)


class MatchedWalkerEnv:
    def __init__(self, config: dict, *, diagnostics: bool = False, env_factory=None):
        self.config = validate_protocol(config)
        self._diagnostics = bool(diagnostics)
        self._factory = env_factory or _native_factory
        self._env = None
        self._closed = False
        self._failed = False
        self._done = True
        self._mask = None
        self._visible = None
        self._info = {}
        self.seed = None
        self.action_decisions = 0
        self.episode_env_steps = 0
        self.action_dim = 6

    @property
    def availability(self) -> bool:
        if self._mask is None:
            raise RuntimeError("Environment has not been reset")
        return bool(self._mask[self.action_decisions])

    @property
    def last_info(self):
        return copy.deepcopy(self._info)

    def _dispose(self):
        if self._env is not None:
            env, self._env = self._env, None
            if callable(getattr(env, "close", None)):
                env.close()
            elif getattr(env, "physics", None) is not None:
                env.physics.free()

    def _image(self) -> np.ndarray:
        if not self.availability:
            # Do not render and then accidentally retain a hidden clean frame.
            return np.zeros((64, 64, 3), np.uint8)
        frame = np.asarray(self._env.physics.render(height=64, width=64, camera_id=0))
        if frame.shape != (64, 64, 3) or frame.dtype != np.uint8:
            raise ValueError("Simulator violated the uint8 RGB64 contract")
        return frame.copy()

    def reset(self, seed: int, mask=None) -> np.ndarray:
        if self._closed:
            raise RuntimeError("Cannot reset a closed environment")
        if self._failed:
            raise RuntimeError("A failed environment cannot silently resume or reset")
        if not self._done:
            raise RuntimeError("Reset is permitted only at an episode boundary")
        if not isinstance(seed, (int, np.integer)) or isinstance(seed, (bool, np.bool_)) or not 0 <= int(seed) < 2**32:
            raise ValueError("Explicit environment seed must be a uint32 integer")
        chosen = validate_mask(np.ones(501, np.bool_) if mask is None else mask)
        self._dispose()
        self.seed = int(seed)
        self._env = self._factory(self.seed)
        try:
            spec = self._env.action_spec()
            low, high = np.asarray(spec.minimum), np.asarray(spec.maximum)
            if tuple(spec.shape) != (6,) or not np.all(low == -1) or not np.all(high == 1):
                raise ValueError("Expected six Walker actions in [-1, 1]")
            first = self._env.reset()
            if not first.first():
                raise ValueError("Native reset did not return FIRST")
            self._mask = chosen
            self.action_decisions = self.episode_env_steps = 0
            self._done = False
            self._visible = self._image()
            self._info = {"availability": True, "observation_timestamp": 0,
                          "env_steps_this_step": 0, "episode_env_steps": 0,
                          "discount": 1.0, "continuation": 1.0,
                          "is_first": True, "is_last": False,
                          "terminated": False, "truncated": False}
            if self._diagnostics:
                self._info["probe"] = {"physics_state_sha256": _physics_hash(self._env), "subcalls": []}
            return self._visible.copy()
        except BaseException:
            self._done = True
            self._dispose()
            raise

    def step(self, action):
        if self._closed or self._env is None or self._done:
            raise RuntimeError("A live, unfinished episode is required")
        requested = np.asarray(action)
        if requested.shape != (6,) or requested.dtype.kind not in "iuf":
            raise ValueError("Action must be a numeric vector of length six")
        if not np.isfinite(requested).all():
            raise FloatingPointError("Nonfinite action")
        executed = np.clip(requested, -1, 1).astype(np.float32)
        input_timestamp = self.action_decisions
        reward, discount, count, subcalls = 0.0, 1.0, 0, []
        try:
            for _ in range(2):
                timestep = self._env.step(executed.copy())
                count += 1
                self.episode_env_steps += 1
                if count == 1:
                    self.action_decisions += 1
                r = float(timestep.reward)
                d = float(timestep.discount)
                if not np.isfinite(r) or not np.isfinite(d) or not 0 <= d <= 1:
                    raise FloatingPointError("Nonfinite/invalid native reward or discount")
                reward += r
                discount *= d
                last = bool(timestep.last())
                if self._diagnostics:
                    subcalls.append({"native_call": self.episode_env_steps, "reward": r,
                        "discount": d, "last": last, "executed_action": executed.tolist(),
                        "physics_state_sha256": _physics_hash(self._env)})
                if last != (self.episode_env_steps == 1000):
                    raise RuntimeError("Unexpected native termination or time-limit semantics")
            self._done = self.episode_env_steps == 1000
            if self._done != (self.action_decisions == 500):
                raise RuntimeError("Action/DMC-call budget mismatch")
            self._visible = self._image()
            terminated = bool(self._done and discount == 0)
            self._info = {"availability": self.availability,
                          "input_available": bool(self._mask[input_timestamp]),
                          "input_observation_timestamp": input_timestamp,
                          "observation_timestamp": self.action_decisions,
                          "executed_action": executed.copy(),
                          "env_steps_this_step": count, "env_steps": count,
                          "episode_env_steps": self.episode_env_steps,
                          "discount": discount, "continuation": discount,
                          "is_first": False, "is_last": self._done,
                          "terminated": terminated, "truncated": bool(self._done and not terminated)}
            if self._diagnostics:
                self._info["probe"] = {"physics_state_sha256": _physics_hash(self._env), "subcalls": subcalls}
            return self._visible.copy(), reward, self._done, self.last_info
        except BaseException:
            # Actual counters already include acknowledged native calls. A
            # failed transition cannot be retried on this poisoned episode.
            self._done = True
            self._failed = True
            raise

    def render(self, size: int = 256):
        """Render only the visible observation, including zeros during gaps."""
        if self._visible is None or self._closed:
            raise RuntimeError("No visible observation")
        if type(size) is not int or size < 64 or size % 64:
            raise ValueError("Visible render size must be a positive multiple of 64")
        return np.repeat(np.repeat(self._visible, size//64, axis=0), size//64, axis=1)

    def render_diagnostic(self, size: int = 256):
        """Evaluator-only real scene; never call from a learner/collector."""
        if not self._diagnostics:
            raise PermissionError("Raw scene rendering requires an explicit diagnostic environment")
        if self._closed or self._env is None:
            raise RuntimeError("No active simulator")
        return np.asarray(self._env.physics.render(height=size, width=size, camera_id=0)).copy()

    def close(self):
        self._dispose()
        self._closed, self._done = True, True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
