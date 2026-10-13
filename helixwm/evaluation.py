"""Frozen deterministic Walker evaluation with raw episodes and state hashes.

Evaluation seeds are independent environment seeds, not additional training
seeds. One-step reward prediction errors are held-out diagnostics; they do
not by themselves establish calibrated trajectory-risk probabilities.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import struct
import time
from typing import Any

import numpy as np
import torch

from .environment import WalkerEnv


def state_sha256(value: Any) -> str:
    """Hash nested module/optimizer state by values, shapes, types and keys.

    Torch serialization can encode storage identifiers; hashing contiguous
    value bytes directly avoids confusing serialization details with mutation.
    No tensor, optimizer state or RNG stream is modified by this operation.
    """
    digest = hashlib.sha256()

    def block(data: bytes) -> None:
        digest.update(struct.pack("!Q", len(data)))
        digest.update(data)

    def visit(item: Any) -> None:
        if isinstance(item, torch.Tensor):
            if item.layout != torch.strided:
                raise TypeError("state hashing supports dense tensors only")
            block(b"tensor")
            block(str(item.dtype).encode("ascii"))
            visit(tuple(item.shape))
            array = item.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy()
            block(array.tobytes())
        elif isinstance(item, np.ndarray):
            block(b"ndarray")
            block(item.dtype.str.encode("ascii"))
            visit(tuple(item.shape))
            block(np.ascontiguousarray(item).tobytes())
        elif isinstance(item, dict):
            block(b"dict")
            visit(len(item))
            for key in sorted(item, key=lambda k: (type(k).__name__, repr(k))):
                visit(key)
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            block(type(item).__name__.encode("ascii"))
            visit(len(item))
            for entry in item:
                visit(entry)
        elif isinstance(item, np.generic):
            visit(item.item())
        elif item is None:
            block(b"none")
        elif isinstance(item, bool):
            block(b"bool:true" if item else b"bool:false")
        elif isinstance(item, int):
            block(b"int")
            block(str(item).encode("ascii"))
        elif isinstance(item, float):
            block(b"float")
            block(struct.pack("!d", item))
        elif isinstance(item, str):
            block(b"str")
            block(item.encode("utf-8"))
        elif isinstance(item, bytes):
            block(b"bytes")
            block(item)
        else:
            raise TypeError(f"unsupported checkpoint value for hashing: {type(item).__name__}")

    visit(value)
    return digest.hexdigest()


def _optimizer_state(learner) -> dict:
    return {name: optimizer.state_dict() for name, optimizer in learner.optimizers.items()}


def _open_video_writer(path: Path, fps: float):
    import imageio.v2 as imageio
    return imageio.get_writer(str(path), format="FFMPEG", mode="I", fps=fps,
                              codec="libx264", quality=7, macro_block_size=16)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _observation_tensor(observation: np.ndarray, model) -> torch.Tensor:
    reference = next(model.parameters())
    array = np.asarray(observation)
    if array.dtype != np.uint8 or array.size != model.obs_dim:
        raise ValueError("evaluation requires flattened uint8 RGB matching model.obs_dim")
    return torch.as_tensor(array.copy(), device=reference.device, dtype=reference.dtype).reshape(1, -1) / 255.0 - 0.5


def evaluate(learner, config: dict, seeds: list[int], output_dir: Path,
             agent_steps: int, env_steps: int, update_steps: int,
             kind: str, checkpoint_id: str = "", video: bool = False) -> dict:
    """Evaluate one frozen snapshot on every requested seed, without selection.

    ``agent_steps``/``env_steps`` are training counters. Raw episode_length is
    the number of action decisions, and actual_eval_env_steps is the number
    of DMC calls made solely for that evaluation episode. Evaluation time and
    interaction are never added to the recorded training budget.

    A requested video uses only seeds[0], selected before any return is known.
    All module tensors and optimizer state are checked exactly. Global Python,
    NumPy, CPU/CUDA Torch RNG and every original module mode are restored even
    if evaluation fails. The caller performs checkpoint reconstruction before
    entering this function; frozen evaluation does not itself prove reload.
    """
    if not seeds or any(not isinstance(seed, int) or isinstance(seed, bool) for seed in seeds):
        raise ValueError("seeds must be a nonempty list of integer environment seeds")
    if len(set(seeds)) != len(seeds):
        raise ValueError("evaluation seeds must be unique")
    if any(not isinstance(value, int) or value < 0 for value in (agent_steps, env_steps, update_steps)):
        raise ValueError("training counters must be nonnegative integers")
    if not kind or not re.fullmatch(r"[A-Za-z0-9_-]+", kind):
        raise ValueError("kind must be a nonempty filesystem-safe label")
    environment_config = dict(config["environment"])
    training_seed = int(config["seed"])
    output_dir = Path(output_dir)
    reports_dir = output_dir / "evaluation_reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    evaluation_id = f"{agent_steps:09d}_{kind}_{time.time_ns()}"
    report_path = reports_dir / f"{evaluation_id}.json"
    video_path = output_dir / "VIDEOS" / "clean_training_final.mp4"
    partial_video_path = output_dir / "VIDEOS" / f".{evaluation_id}.partial.mp4"
    if video and video_path.exists():
        raise FileExistsError(f"refusing to overwrite the saved fixed-seed video: {video_path}")
    if video:
        video_path.parent.mkdir(parents=True, exist_ok=True)

    python_rng = random.getstate()
    numpy_rng = np.random.get_state()
    torch_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    original_modes = [(module, module.training) for module in learner.modules()]
    module_hash_before = state_sha256(learner.state_dict())
    optimizer_hash_before = state_sha256(_optimizer_state(learner))
    rows, video_metadata = [], None
    total_squared_error, total_predictions = 0.0, 0
    try:
        learner.eval()
        with torch.no_grad():
            for episode_index, eval_seed in enumerate(seeds):
                env, writer = None, None
                frame_count, episode_return, decisions, evaluation_env_steps = 0, 0.0, 0, 0
                reward_squared_error = 0.0
                try:
                    env = WalkerEnv(seed=eval_seed, **environment_config)
                    if env.action_dim != learner.model.action_dim:
                        raise ValueError("evaluation environment/model action dimensions do not match")
                    observation = env.reset()
                    observation = _observation_tensor(observation, learner.model)
                    reference = next(learner.model.parameters())
                    zero_action = reference.new_zeros(1, env.action_dim)
                    first = torch.ones(1, dtype=torch.bool, device=reference.device)
                    state, _, _ = learner.model.observe_step(
                        learner.model.initial(1), zero_action, observation, first, sample=False,
                    )
                    if video and episode_index == 0:
                        control_timestep = float(env._env.control_timestep())
                        if not math.isfinite(control_timestep) or control_timestep <= 0:
                            raise ValueError("environment control timestep must be finite and positive")
                        fps = 1.0 / (control_timestep * environment_config["action_repeat"])
                        writer = _open_video_writer(partial_video_path, fps)
                        writer.append_data(env.render(256))
                        frame_count += 1
                    decision_limit = math.ceil(environment_config["episode_env_steps"] /
                                               environment_config["action_repeat"])
                    done = False
                    while not done:
                        if decisions >= decision_limit:
                            raise RuntimeError("evaluation exceeded its declared episode decision limit")
                        action, _ = learner.actor.sample(state.features(), deterministic=True)
                        imagined, _ = learner.model.imagine_step(state, action, sample=False)
                        predicted_reward = float(learner.model.reward_head(imagined.features()).squeeze().cpu())
                        outcome = env.step(action.squeeze(0).cpu().numpy())
                        reward = float(outcome["reward"])
                        count = int(outcome["env_steps"])
                        if not math.isfinite(reward) or not math.isfinite(predicted_reward):
                            raise FloatingPointError("nonfinite evaluation reward or reward prediction")
                        if not 1 <= count <= environment_config["action_repeat"]:
                            raise ValueError("invalid evaluation environment-step count")
                        episode_return += reward
                        reward_squared_error += (reward - predicted_reward) ** 2
                        decisions += 1
                        evaluation_env_steps += count
                        done = bool(outcome["done"])
                        state, _, _ = learner.model.observe_step(
                            state, action, _observation_tensor(outcome["observation"], learner.model),
                            torch.zeros_like(first), sample=False,
                        )
                        if writer is not None:
                            writer.append_data(env.render(256))
                            frame_count += 1
                    row = {
                        "evaluation_id": evaluation_id, "seed": training_seed,
                        "training_seed": training_seed, "eval_seed": eval_seed,
                        "episode_index": episode_index, "kind": kind, "checkpoint_id": checkpoint_id,
                        "agent_steps": agent_steps, "env_steps": env_steps, "update_steps": update_steps,
                        "episode_return": episode_return, "episode_length": decisions,
                        "actual_eval_env_steps": evaluation_env_steps,
                        "reward_prediction_rmse": math.sqrt(reward_squared_error / decisions),
                        "reward_prediction_count": decisions, "zero_update": True,
                        "policy": "deterministic_tanh_mean", "posterior_sampling": False,
                    }
                    rows.append(row)
                    total_squared_error += reward_squared_error
                    total_predictions += decisions
                    if writer is not None:
                        video_metadata = {"path": str(video_path), "eval_seed": eval_seed,
                                          "selection": "first_predeclared_evaluation_seed",
                                          "width": 256, "height": 256, "fps": fps,
                                          "control_timestep_seconds": control_timestep,
                                          "action_repeat": environment_config["action_repeat"],
                                          "frame_count": frame_count,
                                          "duration_seconds": frame_count / fps,
                                          "includes_initial_frame": True}
                finally:
                    try:
                        if writer is not None:
                            writer.close()
                    finally:
                        if env is not None:
                            env.close()

        module_hash_after = state_sha256(learner.state_dict())
        optimizer_hash_after = state_sha256(_optimizer_state(learner))
        if module_hash_after != module_hash_before or optimizer_hash_after != optimizer_hash_before:
            raise RuntimeError("frozen evaluation mutated module or optimizer state")
        if video:
            if not partial_video_path.is_file() or partial_video_path.stat().st_size == 0:
                raise RuntimeError("evaluation video writer produced no nonempty file")
            partial_video_path.replace(video_path)
            video_metadata["sha256"] = _file_sha256(video_path)
            video_metadata["bytes"] = video_path.stat().st_size
        returns = np.asarray([row["episode_return"] for row in rows], dtype=np.float64)
        lengths = np.asarray([row["episode_length"] for row in rows], dtype=np.float64)
        report = {
            "format_version": 1, "evaluation_id": evaluation_id, "kind": kind,
            "checkpoint_id": checkpoint_id, "training_seed": training_seed,
            "evaluation_seeds": list(seeds), "episode_count": len(rows),
            "agent_steps": agent_steps, "env_steps": env_steps, "update_steps": update_steps,
            "episode_return_mean": float(returns.mean()), "episode_return_std": float(returns.std(ddof=0)),
            "std_definition": "population_std_over_fixed_evaluation_episodes",
            "episode_length_mean": float(lengths.mean()),
            "actual_eval_env_steps": sum(row["actual_eval_env_steps"] for row in rows),
            "reward_prediction_rmse": math.sqrt(total_squared_error / total_predictions),
            "reward_prediction_count": total_predictions,
            "reward_prediction_definition": "one_step_prior_reward_under_fixed_real_evaluation_actions",
            "zero_update": True, "module_hash_before": module_hash_before,
            "module_hash_after": module_hash_after, "optimizer_hash_before": optimizer_hash_before,
            "optimizer_hash_after": optimizer_hash_after,
            "rng_restored": True, "module_modes_restored": True,
            "policy": "deterministic_tanh_mean", "posterior_sampling": False,
            "video": video_metadata, "episodes": rows,
            "report_path": str(report_path), "raw_episodes_path": str(output_dir / "evaluations.jsonl"),
        }
        with report_path.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        with (output_dir / "evaluations.jsonl").open("a", encoding="utf-8", newline="\n") as handle:
            handle.write("".join(json.dumps(row, ensure_ascii=True, allow_nan=False) + "\n" for row in rows))
            handle.flush()
            os.fsync(handle.fileno())
        return report
    finally:
        for module, mode in original_modes:
            module.training = mode
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
        torch.set_rng_state(torch_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
        if partial_video_path.exists():
            partial_video_path.unlink()
