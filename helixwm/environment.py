"""Pixel-only DeepMind Control Walker wrapper with explicit unit accounting.

One agent decision repeats one bounded action for ``action_repeat`` DMC calls.
``env_steps`` counts those calls, not MuJoCo integration substeps. Privileged
physics state is exposed only through checkpoint snapshots, never observations.
"""

from __future__ import annotations

import importlib.metadata
import os
from typing import Any

import numpy as np


class WalkerEnv:
    """Walker Walk with RGB observations and undiscounted repeat reward sums."""

    def __init__(
        self,
        seed: int = 0,
        action_repeat: int = 2,
        image_size: int = 64,
        episode_env_steps: int = 1000,
        camera_id: int = 0,
    ) -> None:
        if min(action_repeat, image_size, episode_env_steps) < 1:
            raise ValueError("action_repeat, image_size and episode_env_steps must be positive")
        # Set this before importing dm_control or MuJoCo. An explicitly selected
        # backend is respected, allowing controlled tests on non-EGL machines.
        os.environ.setdefault("MUJOCO_GL", "egl")
        from dm_control import suite

        self.seed = int(seed)
        self.action_repeat = int(action_repeat)
        self.image_size = int(image_size)
        self.episode_env_steps = int(episode_env_steps)
        self.camera_id = int(camera_id)
        self._env = suite.load(
            "walker", "walk", task_kwargs={"random": self.seed},
            environment_kwargs={"flat_observation": False},
        )
        # Use the same step limit in the underlying environment so its native
        # terminal discount/time-limit semantics remain available.
        self._env._step_limit = self.episode_env_steps
        spec = self._env.action_spec()
        self.action_dim = int(np.prod(spec.shape))
        self.observation_dim = self.image_size * self.image_size * 3
        self.action_minimum = np.asarray(spec.minimum, dtype=np.float64).copy()
        self.action_maximum = np.asarray(spec.maximum, dtype=np.float64).copy()
        self._episode_steps = 0
        self._done = True
        self._closed = False

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("environment has been closed")

    def _observation(self) -> np.ndarray:
        return self.render(self.image_size).reshape(-1).copy()

    def reset(self) -> np.ndarray:
        self._check_open()
        self._env.reset()
        self._episode_steps = 0
        self._done = False
        return self._observation()

    def step(self, action: np.ndarray) -> dict[str, Any]:
        self._check_open()
        if self._done:
            raise RuntimeError("reset is required before stepping a new episode")
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (self.action_dim,):
            raise ValueError(f"action must have shape ({self.action_dim},)")
        if not np.isfinite(action).all():
            raise ValueError("action must be finite")
        if np.any(action < self.action_minimum) or np.any(action > self.action_maximum):
            raise ValueError("action is outside the environment action bounds")
        reward, continuation, count = 0.0, 1.0, 0
        for _ in range(min(self.action_repeat, self.episode_env_steps - self._episode_steps)):
            time_step = self._env.step(action)
            reward += float(time_step.reward or 0.0)
            continuation *= float(1.0 if time_step.discount is None else time_step.discount)
            count += 1
            self._episode_steps += 1
            if time_step.last():
                self._done = True
                break
        self._done = self._done or self._episode_steps >= self.episode_env_steps
        if not np.isfinite(reward) or not np.isfinite(continuation):
            raise RuntimeError("environment returned nonfinite reward or discount")
        return {
            "observation": self._observation(), "reward": reward,
            "continuation": continuation, "done": self._done,
            "env_steps": count,
        }

    def render(self, size: int = 256) -> np.ndarray:
        self._check_open()
        if size < 1:
            raise ValueError("render size must be positive")
        return np.asarray(self._env.physics.render(
            height=int(size), width=int(size), camera_id=self.camera_id,
        ), dtype=np.uint8).copy()

    def _configuration(self) -> dict[str, int]:
        return {name: getattr(self, name) for name in (
            "seed", "action_repeat", "image_size", "episode_env_steps", "camera_id",
        )}

    def snapshot(self) -> dict[str, Any]:
        """Serialize integration state, wrapper counters and task PRNG.

        This targets the pinned Walker task on the same MuJoCo build. It does
        not promise cross-version/cross-hardware bitwise reproducibility. Agent,
        replay and framework RNG states must be saved separately by the trainer.
        """
        self._check_open()
        import mujoco

        physics = self._env.physics
        state_spec = mujoco.mjtState.mjSTATE_INTEGRATION
        state = np.empty(mujoco.mj_stateSize(physics.model.ptr, state_spec), dtype=np.float64)
        mujoco.mj_getState(physics.model.ptr, physics.data.ptr, state, state_spec)
        rng = self._env.task.random.get_state()
        return {
            "format_version": 1, "configuration": self._configuration(),
            "mujoco_version": mujoco.__version__,
            "dm_control_version": importlib.metadata.version("dm_control"),
            "state_spec": int(state_spec), "integration_state": state.tolist(),
            "task_random": [rng[0], rng[1].tolist(), int(rng[2]), int(rng[3]), float(rng[4])],
            "episode_steps": self._episode_steps, "done": self._done,
            "dmc_step_count": int(self._env._step_count),
            "dmc_reset_next_step": bool(self._env._reset_next_step),
        }

    def restore(self, snapshot: dict[str, Any]) -> None:
        """Restore a same-version snapshot without taking an environment step."""
        self._check_open()
        import mujoco

        if snapshot.get("format_version") != 1 or snapshot["configuration"] != self._configuration():
            raise ValueError("environment snapshot format or configuration does not match")
        if snapshot["mujoco_version"] != mujoco.__version__ or snapshot["dm_control_version"] != importlib.metadata.version("dm_control"):
            raise ValueError("environment snapshot dependency versions do not match")
        physics = self._env.physics
        state_spec = mujoco.mjtState.mjSTATE_INTEGRATION
        if snapshot["state_spec"] != int(state_spec):
            raise ValueError("unsupported MuJoCo state specification")
        state = np.asarray(snapshot["integration_state"], dtype=np.float64)
        if state.shape != (mujoco.mj_stateSize(physics.model.ptr, state_spec),) or not np.isfinite(state).all():
            raise ValueError("invalid MuJoCo integration state")
        rng = snapshot["task_random"]
        task_rng = (rng[0], np.asarray(rng[1], dtype=np.uint32), int(rng[2]), int(rng[3]), float(rng[4]))
        mujoco.mj_setState(physics.model.ptr, physics.data.ptr, state, state_spec)
        physics.forward()
        # forward recomputes derived quantities; restore every integration field
        # again so solver warm-start and externally applied forces are retained.
        mujoco.mj_setState(physics.model.ptr, physics.data.ptr, state, state_spec)
        self._env.task.random.set_state(task_rng)
        self._episode_steps = int(snapshot["episode_steps"])
        self._done = bool(snapshot["done"])
        self._env._step_count = int(snapshot["dmc_step_count"])
        self._env._reset_next_step = bool(snapshot["dmc_reset_next_step"])

    def close(self) -> None:
        if not self._closed:
            self._env.physics.free()
            self._closed = True

    def __enter__(self) -> "WalkerEnv":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
