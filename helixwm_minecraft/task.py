"""A named custom task in real MineRL/Minecraft; live validation is pending.

MineRL is imported only when constructing the real backend. This module never
substitutes a fake simulator or Crafter for Minecraft.
"""
import hashlib
import json
import math

from .protocol import TASK_ID, integer


def task_manifest(seed, episode_env_steps=1000):
    integer(seed, "seed")
    integer(episode_env_steps, "episode_env_steps", 1, 6000)
    digest = hashlib.sha256((TASK_ID + ":" + str(seed)).encode("ascii")).digest()
    goal_x = int.from_bytes(digest[0:2], "big") % 13 - 6
    goal_z = 12 + int.from_bytes(digest[2:4], "big") % 9
    # A three-block-high diamond pillar in the initial forward field of view.
    # Rendering visibility is an empirical gate, not a guarantee from geometry.
    layout = {
        "task_id": TASK_ID, "episode_seed": seed,
        "engine": "Minecraft 1.11.2 via MineRL 0.4.4 / Malmo 0.37.0",
        "flat_generator": "1;7,2x3,2;1",
        "spawn": {"x": 0.5, "y": 4.0, "z": 0.5, "yaw": 0.0, "pitch": 0.0},
        "goal": {"x": goal_x, "z": goal_z, "y_min": 4, "y_max": 6, "block": "diamond_block"},
        "goal_position_for_reward": [goal_x + 0.5, goal_z + 0.5],
        "max_native_steps": episode_env_steps, "nominal_native_step_seconds": 0.05,
        "native_counter_unit": "acknowledged_MineRL_synchronous_step_calls",
        "total_engine_world_ticks_measured": False,
        "weather": "clear", "time": 6000, "mobs": False,
        "inventory": "empty_no_compass", "observation": "RGB64_only_to_policy",
        "reward": "previous_horizontal_goal_distance_minus_current_distance_plus_native_touch_reward_100",
        "native_benchmark_equivalence": False,
    }
    layout["manifest_sha256"] = hashlib.sha256(json.dumps(layout, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    return layout


def make_spec(seed, episode_env_steps=1000):
    from minerl.herobraine.env_specs.simple_embodiment import SimpleEmbodimentEnvSpec
    from minerl.herobraine.hero import handlers

    manifest = task_manifest(seed, episode_env_steps)

    class VisibleNavigationSpec(SimpleEmbodimentEnvSpec):
        def __init__(self):
            self.task_manifest = manifest
            super().__init__(TASK_ID, max_episode_steps=episode_env_steps, resolution=(64, 64))

        def create_observables(self):
            # Location is ONLY for reward/evaluator inside the separate worker.
            return super().create_observables() + [handlers.ObservationFromCurrentLocation()]

        def create_rewardables(self):
            return [handlers.RewardForTouchingBlockType([{"type": "diamond_block", "behaviour": "onceOnly", "reward": 100.0}])]

        def create_agent_start(self):
            return [handlers.AgentStartPlacement(**manifest["spawn"]), handlers.SimpleInventoryAgentStart([])]

        def create_agent_handlers(self):
            return [handlers.AgentQuitFromTouchingBlockType(["diamond_block"])]

        def create_server_world_generators(self):
            return [handlers.FlatWorldGenerator(force_reset=True, generatorString=manifest["flat_generator"])]

        def create_server_decorators(self):
            g = manifest["goal"]
            return [handlers.DrawingDecorator('<DrawCuboid x1="{x}" y1="4" z1="{z}" x2="{x}" y2="6" z2="{z}" type="diamond_block"/>'.format(**g))]

        def create_server_initial_conditions(self):
            return [handlers.TimeInitialCondition(allow_passage_of_time=False, start_time=6000), handlers.WeatherInitialCondition("clear"), handlers.SpawningInitialCondition("false")]

        def create_server_quit_producers(self):
            return [handlers.ServerQuitFromTimeUp(episode_env_steps * 50), handlers.ServerQuitWhenAnyAgentFinishes()]

        def is_from_folder(self, folder):
            return False

        def determine_success_from_rewards(self, rewards):
            # Native touch reward only; shaped reward is added outside MineRL.
            return any(float(reward) >= 100.0 for reward in rewards)

        def get_docstring(self):
            return "Custom real-Minecraft visual target navigation; not the native MineRLNavigate benchmark."

    return VisibleNavigationSpec()


class VisibleNavigationBackend:
    def __init__(self, seed, episode_env_steps):
        import minerl  # Registers the real MineRL backend, never a fake entrypoint.
        import importlib.metadata
        import platform
        version = importlib.metadata.version("minerl")
        if version != "0.4.4":
            raise RuntimeError("This task requires MineRL 0.4.4, observed " + version)
        self.runtime_metadata = {"python": platform.python_version(), "minerl": version,
                                 "gym": importlib.metadata.version("gym"),
                                 "numpy": importlib.metadata.version("numpy")}
        self.manifest = task_manifest(seed, episode_env_steps)
        self.env = make_spec(seed, episode_env_steps).make(fake=False)
        self.env.seed(seed)
        self.previous_distance = None
        self.previous_position = None
        self.path_length = 0.0
        self.last_diagnostics = {}

    def _position(self, observation):
        values = observation["location_stats"]
        p = tuple(float(values[key]) for key in ("xpos", "ypos", "zpos"))
        if not all(math.isfinite(value) for value in p):
            raise ValueError("Nonfinite privileged reward position")
        return p

    def _distance(self, position):
        goal_x, goal_z = self.manifest["goal_position_for_reward"]
        return math.hypot(position[0] - goal_x, position[2] - goal_z)

    def reset(self):
        observation = self.env.reset()
        self.previous_position = self._position(observation)
        self.previous_distance = self._distance(self.previous_position)
        self.last_diagnostics = {"position": list(self.previous_position), "goal_distance": self.previous_distance, "success": False, "path_length": 0.0}
        return observation

    def step(self, action):
        import numpy as np
        native = self.env.action_space.noop()
        for key, value in action.items():
            if key in native:
                native[key] = np.asarray(value, dtype=np.float32) if key == "camera" else value
        observation, native_reward, done, info = self.env.step(native)
        # MineRL 0.4.4 can turn a client socket failure into a random sampled
        # observation plus info['error']. Its server-send failure also sets
        # env.done without necessarily updating the returned done value.
        # Neither path is a valid environment transition or terminal success.
        if not isinstance(info, dict) or "error" in info:
            raise RuntimeError("MineRL reported a native step error; rejecting synthetic fallback observation")
        underlying = getattr(self.env, "unwrapped", self.env)
        if bool(getattr(underlying, "done", done)) and not bool(done):
            raise RuntimeError("MineRL internal/returned termination mismatch; possible failed server step")
        position = self._position(observation)
        distance = self._distance(position)
        progress_reward = self.previous_distance - distance
        self.path_length += math.hypot(position[0] - self.previous_position[0], position[2] - self.previous_position[2])
        success = float(native_reward) >= 100.0
        self.last_diagnostics = {"position": list(position), "goal_distance": distance, "path_length": self.path_length, "native_reward": float(native_reward), "progress_reward": progress_reward, "success": success}
        self.previous_distance, self.previous_position = distance, position
        return observation, float(native_reward) + progress_reward, bool(done), {"success": success}

    def close(self):
        self.env.close()
