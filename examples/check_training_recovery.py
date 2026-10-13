"""Read-only source-run probe of same-build checkpoint recovery.

Run only after the smoke run has completed:
    python -m examples.check_training_recovery --checkpoint /run/CHECKPOINTS/checkpoint_000001000.pt --output /external/recovery.json --device cuda

Two disposable learners each perform one isolated optimizer update from the
same checkpoint and replay/RNG snapshot. Two fresh environments separately
perform one stochastic actor decision from the saved collector state. No
episode, replay index, log or checkpoint is written into the source run.
This proves repeatable short recovery under the current build, not equality
to an independently recorded uninterrupted long run or another GPU/build.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import json
import math
import os
from pathlib import Path
import traceback
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np
import torch

from helixwm.checkpoint import (
    atomic_json, capture_rng, load_checkpoint, restore_learner, restore_rng, sha256,
)
from helixwm.control import HelixLearner
from helixwm.environment import WalkerEnv
from helixwm.evaluation import state_sha256
from helixwm.pixels import PixelRSSM
from helixwm.replay import DiskReplay
from helixwm.rssm import RSSMState
from helixwm.train import image_tensor, setup, tensor_batch


def _cpu_copy(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, dict):
        return {key: _cpu_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_copy(item) for item in value)
    return value


def _tensor_manifest(value: Any, path: str = "") -> dict:
    result = {}
    if isinstance(value, (torch.Tensor, np.ndarray)):
        result[path] = {"shape": list(value.shape), "dtype": str(value.dtype),
                        "sha256": state_sha256(value)}
    elif isinstance(value, dict):
        for key, item in value.items():
            result.update(_tensor_manifest(item, f"{path}/{key}"))
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            result.update(_tensor_manifest(item, f"{path}/{index}"))
    return result


def _scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return repr(value)
    return value


def _differences(expected: Any, actual: Any, path: str = "") -> list[dict]:
    """Report differing fields and exact first differing elements, not tolerance."""
    if isinstance(expected, torch.Tensor) and isinstance(actual, torch.Tensor):
        if expected.shape != actual.shape or expected.dtype != actual.dtype:
            return [{"path": path, "reason": "tensor_shape_or_dtype",
                     "expected_shape": list(expected.shape), "actual_shape": list(actual.shape),
                     "expected_dtype": str(expected.dtype), "actual_dtype": str(actual.dtype)}]
        if torch.equal(expected, actual):
            return []
        a, b = expected.reshape(-1), actual.reshape(-1)
        indices = torch.nonzero(a != b).flatten()
        count = len(indices)
        selected = indices[:16].tolist()
        delta = (a.double() - b.double()).abs()
        return [{"path": path, "reason": "tensor_values", "different_elements": count,
                 "max_absolute_difference": _scalar(float(delta.max())) if delta.numel() else 0.0,
                 "first_flat_indices": selected,
                 "expected_values": [_scalar(a[index].item()) for index in selected],
                 "actual_values": [_scalar(b[index].item()) for index in selected]}]
    if isinstance(expected, np.ndarray) and isinstance(actual, np.ndarray):
        if expected.shape == actual.shape and expected.dtype == actual.dtype and np.array_equal(expected, actual):
            return []
        if expected.shape != actual.shape or expected.dtype != actual.dtype:
            return [{"path": path, "reason": "array_shape_or_dtype",
                     "expected_shape": list(expected.shape), "actual_shape": list(actual.shape),
                     "expected_dtype": str(expected.dtype), "actual_dtype": str(actual.dtype)}]
        a, b = expected.reshape(-1), actual.reshape(-1)
        indices = np.flatnonzero(a != b)
        selected = indices[:16].tolist()
        return [{"path": path, "reason": "array_values", "different_elements": int(len(indices)),
                 "max_absolute_difference": _scalar(float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))),
                 "first_flat_indices": selected,
                 "expected_values": [_scalar(a[index]) for index in selected],
                 "actual_values": [_scalar(b[index]) for index in selected]}]
    if isinstance(expected, dict) and isinstance(actual, dict):
        differences = []
        if expected.keys() != actual.keys():
            differences.append({"path": path, "reason": "dictionary_keys",
                                "only_expected": sorted(map(str, expected.keys() - actual.keys())),
                                "only_actual": sorted(map(str, actual.keys() - expected.keys()))})
        for key in expected.keys() & actual.keys():
            differences.extend(_differences(expected[key], actual[key], f"{path}/{key}"))
        return differences
    if isinstance(expected, (tuple, list)) and type(expected) is type(actual):
        if len(expected) != len(actual):
            return [{"path": path, "reason": "sequence_length", "expected": len(expected), "actual": len(actual)}]
        differences = []
        for index, (a, b) in enumerate(zip(expected, actual)):
            differences.extend(_differences(a, b, f"{path}/{index}"))
        return differences
    if type(expected) is not type(actual) or expected != actual:
        return [{"path": path, "reason": "scalar_or_type", "expected": _scalar(expected), "actual": _scalar(actual)}]
    return []


def _run_manifest(directory: Path) -> dict:
    return {path.relative_to(directory).as_posix(): {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in sorted(directory.rglob("*")) if path.is_file()}


def _optimizer_state(learner) -> dict:
    return {name: optimizer.state_dict() for name, optimizer in learner.optimizers.items()}


def _new_learner(saved: dict, device: str) -> HelixLearner:
    config = saved["config"]
    learner = HelixLearner(PixelRSSM(**config["model"]).to(device), config["learner"])
    restore_learner(learner, saved)
    learner.train()
    return learner


def _one_update(checkpoint: Path, replay_dir: Path, device: str) -> dict:
    # Read a fresh object each time: optimizer.load_state_dict may alias some
    # CPU step tensors. Reusing one loaded object could mutate trial two's start.
    saved = load_checkpoint(checkpoint)
    config = saved["config"]
    learner = _new_learner(saved, device)
    replay = DiskReplay(replay_dir, config["replay_capacity_steps"], config["seed"] + 101)
    replay.max_cached_episodes = config["replay_cache_episodes"]
    replay.restore(saved["replay"], persist=False)
    restore_rng(saved["rng"])
    before = {"modules": state_sha256(learner.state_dict()),
              "optimizers": state_sha256(_optimizer_state(learner)),
              "replay": state_sha256(replay.snapshot()), "rng": state_sha256(capture_rng())}
    arrays = replay.sample(config["batch_size"], config["sequence_length"])
    batch = tensor_batch(arrays, device)
    metrics = learner.update(batch, int(saved["counters"]["update_steps"]))
    if device == "cuda":
        torch.cuda.synchronize()
    trace = {"initial_hashes": before, "batch": _cpu_copy(batch),
             "modules": _cpu_copy(learner.state_dict()),
             "optimizers": _cpu_copy(_optimizer_state(learner)),
             "metrics": metrics, "replay_after_sampling": replay.snapshot(),
             "rng_after_update": _cpu_copy(capture_rng())}
    del learner, batch, replay, saved
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return trace


def _collector_step(checkpoint: Path, device: str) -> dict:
    saved = load_checkpoint(checkpoint)
    config, collector = saved["config"], saved["collector"]
    learner = _new_learner(saved, device)
    module_before = state_sha256(learner.state_dict())
    optimizer_before = state_sha256(_optimizer_state(learner))
    if collector["environment"]["done"]:
        raise ValueError("collector snapshot is terminal; probe will not invent an extra reset/episode")
    collector_rng = np.random.default_rng()
    collector_rng.bit_generator.state = collector["rng"]
    with WalkerEnv(seed=config["seed"], **config["environment"]) as env:
        env.restore(collector["environment"])
        initial_observation = env.render(env.image_size).reshape(-1)
        if not np.array_equal(initial_observation, collector["observation"]):
            raise AssertionError("restored collector RGB differs from its saved observation")
        state = RSSMState(collector["h"].to(device), collector["z"].to(device))
        restore_rng(saved["rng"])
        with torch.no_grad():
            action, log_prob = learner.actor.sample(state.features(), deterministic=False)
            dreamed, prior = learner.model.imagine_step(state, action)
            predicted_reward = learner.model.reward_head(dreamed.features()).squeeze(-1)
            result = env.step(action[0].cpu().numpy())
            next_state, _, posterior = learner.model.observe_step(
                state, action, image_tensor(result["observation"], device),
                torch.zeros(1, dtype=torch.bool, device=device),
            )
        if device == "cuda":
            torch.cuda.synchronize()
        trace = {
            "initial_observation": initial_observation, "action": action.cpu().clone(),
            "action_log_probability": log_prob.cpu().clone(), "saved_prior": prior.cpu().clone(),
            "saved_predicted_reward": predicted_reward.cpu().clone(),
            "next_posterior": posterior.cpu().clone(),
            "next_h": next_state.h.cpu().clone(), "next_z": next_state.z.cpu().clone(),
            "outcome": _cpu_copy(result), "physics_time": float(env._env.physics.time()),
            "dmc_step_count": int(env._env._step_count), "environment_after": env.snapshot(),
            "collector_rng_after": collector_rng.bit_generator.state,
            "rng_after": _cpu_copy(capture_rng()),
            "modules_unchanged": state_sha256(learner.state_dict()) == module_before,
            "optimizers_unchanged": state_sha256(_optimizer_state(learner)) == optimizer_before,
        }
    del learner, saved
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return trace


def _trace_report(trace: dict) -> dict:
    return {"sha256": state_sha256(trace),
            "field_sha256": {key: state_sha256(value) for key, value in trace.items()},
            "tensor_sha256_manifest": _tensor_manifest(trace)}


def run(checkpoint: Path, output: Path, device: str) -> dict:
    checkpoint, output = checkpoint.resolve(), output.resolve()
    run_dir = checkpoint.parent.parent
    if output.is_relative_to(run_dir):
        raise ValueError("recovery report must be outside the original run directory")
    if not (run_dir / "REPLAY").is_dir():
        raise FileNotFoundError("existing source-run REPLAY directory is required")
    saved = load_checkpoint(checkpoint)
    if saved["config"].get("experiment_class") != "smoke":
        raise ValueError("this bounded gate is restricted to a completed smoke run")
    completion = run_dir / "completion.json"
    if not completion.is_file() or json.loads(completion.read_text())["state"] != "training_and_frozen_evaluation_complete":
        raise RuntimeError("source smoke run must complete before the recovery gate")
    setup(saved["config"], device)
    original_rng = capture_rng()
    before_manifest = _run_manifest(run_dir)
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "checkpoint": str(checkpoint),
        "source_run": str(run_dir), "device": device, "checkpoint_counters": saved["counters"],
        "checkpoint_sha256_before": sha256(checkpoint), "source_run_manifest_before": before_manifest,
        "checkpoint_source_commit": saved["metadata"].get("source_commit"),
        "probe_source_sha256": sha256(Path(__file__)),
        "torch_version": torch.__version__, "torch_cuda": torch.version.cuda,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "tolerance": "exact equality; no allclose acceptance", "passed": False,
        "scope": ["repeatable saved-replay sample and one full learner update from two independent loads",
                  "repeatable stochastic actor action and one real Walker step from two saved collector restorations",
                  "original checkpoint, replay, index and run artifacts remain byte-identical"],
        "not_established": ["equivalence to a separately recorded uninterrupted long training run",
                            "cross-version or cross-GPU bitwise equivalence",
                            "formal task performance or multi-seed reproduction"],
        "planned_isolated_optimizer_updates": 2, "completed_isolated_optimizer_updates": 0,
        "planned_isolated_collector_action_decisions": 2, "completed_isolated_collector_action_decisions": 0,
        "additional_complete_episodes": 0, "writes_to_original_run_authorized": False,
        "collector_probe_policy": "stochastic actor; does not use random-prefill branch",
        "saved_checkpoint_is_past_prefill": saved["counters"]["agent_steps"] >= saved["config"]["prefill_agent_steps"],
    }
    try:
        first = _one_update(checkpoint, run_dir / "REPLAY", device)
        report["completed_isolated_optimizer_updates"] += 1
        second = _one_update(checkpoint, run_dir / "REPLAY", device)
        report["completed_isolated_optimizer_updates"] += 1
        update_differences = _differences(first, second, "update")
        report["update_recovery"] = {"exact_equal": not update_differences,
                                     "differences": update_differences,
                                     "first": _trace_report(first), "second": _trace_report(second),
                                     "first_metrics": first["metrics"], "second_metrics": second["metrics"]}
        del first, second
        first = _collector_step(checkpoint, device)
        report["completed_isolated_collector_action_decisions"] += 1
        second = _collector_step(checkpoint, device)
        report["completed_isolated_collector_action_decisions"] += 1
        collector_differences = _differences(first, second, "collector")
        collector_unchanged = all(trace["modules_unchanged"] and trace["optimizers_unchanged"]
                                  for trace in (first, second))
        report["collector_recovery"] = {
            "exact_equal": not collector_differences, "frozen_learner_unchanged": collector_unchanged,
            "differences": collector_differences, "first": _trace_report(first), "second": _trace_report(second),
            "action": first["action"].tolist(), "reward": first["outcome"]["reward"],
            "env_steps": first["outcome"]["env_steps"], "done": first["outcome"]["done"],
            "physics_time": first["physics_time"], "dmc_step_count": first["dmc_step_count"],
        }
        report["passed"] = not update_differences and not collector_differences and collector_unchanged
    except Exception as error:
        report.update({"passed": False, "error": repr(error), "traceback": traceback.format_exc()})
    finally:
        restore_rng(original_rng)
        after_manifest = _run_manifest(run_dir)
        report["source_run_manifest_after"] = after_manifest
        report["source_run_manifest_differences"] = _differences(before_manifest, after_manifest, "source_run")
        report["source_run_unchanged"] = before_manifest == after_manifest
        report["checkpoint_sha256_after"] = sha256(checkpoint)
        report["checkpoint_unchanged"] = report["checkpoint_sha256_before"] == report["checkpoint_sha256_after"]
        report["passed"] = report["passed"] and report["source_run_unchanged"] and report["checkpoint_unchanged"]
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    checkpoint, output = args.checkpoint.resolve(), args.output.resolve()
    if output.is_relative_to(checkpoint.parent.parent):
        parser.error("--output must be outside the checkpoint's original run directory")
    if output.exists():
        parser.error("--output already exists; choose a fresh external report path")
    try:
        report = run(checkpoint, output, args.device)
    except Exception as error:
        report = {"passed": False, "error": repr(error), "traceback": traceback.format_exc(),
                  "checkpoint": str(checkpoint)}
    atomic_json(output, report)
    print(json.dumps({"passed": report["passed"], "output": str(output),
                      "source_run_unchanged": report.get("source_run_unchanged"),
                      "error": report.get("error")}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
