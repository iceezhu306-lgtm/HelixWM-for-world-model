"""Prove same-build Walker snapshot restoration using real EGL pixel steps.

Run from the repository root:
    python -m examples.check_walker_snapshot --output /path/to/report.json

This is an environment recovery test, not agent training or policy evaluation.
The test uses exact equality, including pixels and MuJoCo integration states.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import traceback

import numpy as np

from helixwm.environment import WalkerEnv


def digest(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def trajectory(env: WalkerEnv, actions: np.ndarray) -> dict[str, np.ndarray]:
    records = []
    for action in actions:
        result = env.step(action)
        records.append({**result, "physics_time": float(env._env.physics.time()),
                        "dmc_step_count": int(env._env._step_count)})
    return {name: np.asarray([record[name] for record in records]) for name in records[0]}


def comparison(expected: dict[str, np.ndarray], actual: dict[str, np.ndarray]) -> dict:
    result = {}
    for name, reference in expected.items():
        value = actual[name]
        equal = reference.shape == value.shape and np.array_equal(reference, value)
        result[name] = {
            "exact_equal": bool(equal), "shape": list(value.shape),
            "expected_sha256": digest(reference), "actual_sha256": digest(value),
            "max_absolute_difference": float(np.max(np.abs(
                reference.astype(np.float64) - value.astype(np.float64),
            ))) if reference.shape == value.shape else None,
        }
    result["all_exact_equal"] = all(value["exact_equal"] for value in result.values())
    return result


def run(output: Path) -> dict:
    rng = np.random.default_rng(7321)
    report = {
        "scope": "environment-only same-build exact snapshot restoration",
        "seed": 0, "action_repeat": 2, "warmup_action_decisions": 17,
        "comparison_action_decisions": 4, "tolerance": "exact equality",
        "training_updates": 0,
    }
    with WalkerEnv(seed=0, action_repeat=2) as env:
        observation = env.reset()
        warmup_actions = rng.uniform(-0.8, 0.8, size=(17, env.action_dim))
        for action in warmup_actions:
            observation = env.step(action)["observation"]
        # A JSON round trip checks portability of the snapshot representation.
        snapshot = json.loads(json.dumps(env.snapshot()))
        report["snapshot"] = {
            "configuration": snapshot["configuration"],
            "mujoco_version": snapshot["mujoco_version"],
            "dm_control_version": snapshot["dm_control_version"],
            "dmc_step_count": snapshot["dmc_step_count"],
            "episode_env_steps_elapsed": snapshot["episode_steps"],
            "physics_time": float(env._env.physics.time()),
            "observation_sha256": digest(observation),
            "integration_state_values": len(snapshot["integration_state"]),
        }
        actions = rng.uniform(-0.8, 0.8, size=(4, env.action_dim))
        expected = trajectory(env, actions)
        expected_after = json.loads(json.dumps(env.snapshot()))
        env.restore(snapshot)
        report["same_environment_restored_observation_exact"] = bool(np.array_equal(
            observation, env.render(env.image_size).reshape(-1),
        ))
        same = trajectory(env, actions)
        report["same_environment"] = comparison(expected, same)
        report["same_environment_final_snapshot_exact"] = expected_after == env.snapshot()
        with WalkerEnv(seed=0, action_repeat=2) as fresh:
            # Deliberately restore before reset: no extra RNG draw or episode
            # initialization is needed to reconstruct a captured live state.
            fresh.restore(snapshot)
            report["fresh_environment_restored_observation_exact"] = bool(np.array_equal(
                observation, fresh.render(fresh.image_size).reshape(-1),
            ))
            restored = trajectory(fresh, actions)
            report["fresh_environment"] = comparison(expected, restored)
            report["fresh_environment_final_snapshot_exact"] = expected_after == fresh.snapshot()
        sidecar = output.with_suffix(".trajectory.npz")
        np.savez_compressed(sidecar, actions=actions, warmup_actions=warmup_actions,
                            snapshot_observation=observation, **expected)
        report["reference_trajectory"] = {
            "file": sidecar.name,
            "sha256": hashlib.sha256(sidecar.read_bytes()).hexdigest(),
            "dmc_step_counts": expected["dmc_step_count"].tolist(),
            "env_steps_per_decision": expected["env_steps"].tolist(),
            "physics_times": expected["physics_time"].tolist(),
            "rewards": expected["reward"].tolist(),
            "continuations": expected["continuation"].tolist(),
        }
    report["passed"] = all((
        report["same_environment_restored_observation_exact"],
        report["fresh_environment_restored_observation_exact"],
        report["same_environment_final_snapshot_exact"],
        report["fresh_environment_final_snapshot_exact"],
        report["same_environment"]["all_exact_equal"],
        report["fresh_environment"]["all_exact_equal"],
    ))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        report = run(args.output)
    except Exception as error:
        report = {"passed": False, "error": str(error), "traceback": traceback.format_exc()}
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "report": str(args.output),
                      "error": report.get("error")}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
