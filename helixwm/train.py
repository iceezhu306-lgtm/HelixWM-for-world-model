"""Single-seed pixel Walker training, with explicit budget and artifact gates."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
import traceback

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np
import torch

from .checkpoint import atomic_json, capture_rng, load_checkpoint, restore_learner, restore_rng, save_checkpoint
from .control import HelixLearner
from .environment import WalkerEnv
from .pixels import PixelRSSM
from .reliability import observed_prediction_errors
from .replay import DiskReplay
from .rssm import RSSMState
from .run_storage import exclusive_run_lock, prepare_fresh_run, prepare_resume_branch


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_json(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()


def command(args: list[str]) -> str:
    return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT, timeout=30).strip()


def setup(config: dict, device: str) -> None:
    torch.set_num_threads(config["torch_threads"])
    torch.set_num_interop_threads(1)
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        torch.cuda.init()
        torch.cuda.manual_seed_all(config["seed"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = config["deterministic_algorithms"]
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(config["deterministic_algorithms"])


def tensor_batch(arrays: dict, device: str) -> dict:
    batch = {key: torch.from_numpy(value).to(device) for key, value in arrays.items()}
    batch["observations"] = batch["observations"].float() / 255.0 - 0.5
    for key in ("actions", "rewards", "continuations", "errors"):
        batch[key] = batch[key].float()
    return batch


def image_tensor(observation: np.ndarray, device: str) -> torch.Tensor:
    return torch.as_tensor(observation, device=device).float().unsqueeze(0) / 255.0 - 0.5


def new_episode(observation: np.ndarray) -> dict:
    return {"observations": [observation], "actions": [], "rewards": [0.0],
            "continuations": [1.0], "errors": [np.zeros(2, np.float32)],
            "error_valid": [False], "model_versions": [-1]}


def episode_arrays(episode: dict) -> dict:
    types = {"observations": np.uint8, "actions": np.float32, "rewards": np.float32,
             "continuations": np.float32, "errors": np.float32, "error_valid": np.bool_,
             "model_versions": np.int64}
    return {key: np.asarray(value, dtype=types[key]) for key, value in episode.items()}


def metadata(config: dict) -> dict:
    packages = {name: importlib.metadata.version(name) for name in
                ("torch", "numpy", "dm-control", "mujoco", "imageio", "imageio-ffmpeg", "matplotlib")}
    return {"created_utc": utc(), "source_commit": command(["git", "rev-parse", "HEAD"]),
            "source_status": command(["git", "status", "--porcelain"]),
            "python": sys.version, "packages": packages, "torch_cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "gpu": command(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"]),
            "units": {"agent_steps": "action decisions", "env_steps": "actual DMC step calls, not MuJoCo integration substeps",
                      "action_repeat": config["environment"]["action_repeat"],
                      "evaluation_interactions_in_training_budget": False},
            "experiment_class": config["experiment_class"], "stages": ["clean_training"],
            "deterministic_algorithms": config["deterministic_algorithms"],
            "cross_hardware_bitwise_recovery_claimed": False}


def resources(run_dir: Path, counters: dict) -> dict:
    row = {"utc": utc(), **counters, "disk_free_bytes": shutil.disk_usage(run_dir).free}
    for name in ("memory.current", "memory.max", "memory.events"):
        path = Path("/sys/fs/cgroup") / name
        if path.exists():
            row["cgroup_" + name.replace(".", "_")] = path.read_text().strip()
    status = Path("/proc/self/status")
    if status.exists():
        row["process_memory"] = {line.split(":")[0]: line.split(":", 1)[1].strip()
                                  for line in status.read_text().splitlines() if line.startswith(("VmRSS:", "VmHWM:"))}
    try:
        row["gpu"] = command(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu", "--format=csv,noheader,nounits"])
    except (subprocess.SubprocessError, OSError) as exc:
        row["gpu_query_error"] = str(exc)
    if torch.cuda.is_available():
        row["torch_peak_memory_allocated_bytes"] = torch.cuda.max_memory_allocated()
    append_json(run_dir / "resources.jsonl", row)
    return row


def validate_config(config: dict) -> None:
    for key in ("budget_agent_steps", "batch_size", "sequence_length", "train_every", "checkpoint_every", "eval_every"):
        if not isinstance(config[key], int) or config[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if config["environment"]["action_repeat"] != 2 or config["environment"]["episode_env_steps"] != 1000:
        raise ValueError("this frozen first-experiment protocol requires repeat=2 and 1000 DMC calls/episode")
    if config["budget_agent_steps"] % 500 or config["checkpoint_every"] % 500 or config["eval_every"] % 500:
        raise ValueError("budget, checkpoint and evaluation boundaries must be full 500-action episodes")
    if config["eval_every"] % config["checkpoint_every"]:
        raise ValueError("each evaluation boundary must have an immutable checkpoint")
    if config["sequence_length"] <= config["learner"]["burnin"]:
        raise ValueError("sequence length must exceed burn-in")


def _run(config: dict, run_dir: Path, device: str, resume: Path | None = None) -> None:
    from .evaluation import evaluate
    from .artifacts import export_run

    validate_config(config)
    run_dir.mkdir(parents=True, exist_ok=True)
    setup(config, device)
    learner = HelixLearner(PixelRSSM(**config["model"]).to(device), config["learner"])
    replay = DiskReplay(run_dir / "REPLAY", config["replay_capacity_steps"], config["seed"] + 101)
    replay.max_cached_episodes = config["replay_cache_episodes"]
    collector_rng = np.random.default_rng(config["seed"] + 202)
    env = WalkerEnv(seed=config["seed"], **config["environment"])
    meta = metadata(config)
    counters = {"agent_steps": 0, "env_steps": 0, "update_steps": 0, "episodes": 0, "wall_time_s": 0.0}
    start_wall = time.monotonic()
    elapsed_prior = 0.0
    observation = env.reset()
    with torch.no_grad():
        state, _, _ = learner.model.observe_step(learner.model.initial(1), torch.zeros(1, 6, device=device),
                                                 image_tensor(observation, device), torch.ones(1, dtype=torch.bool, device=device))
    episode = new_episode(observation)
    if resume:
        saved = load_checkpoint(resume)
        if saved["config"] != config:
            raise ValueError("resume must use the exact checkpoint configuration")
        if saved["metadata"]["source_commit"] != meta["source_commit"] or saved["metadata"]["packages"] != meta["packages"]:
            raise ValueError("resume source commit and package versions must match")
        restore_learner(learner, saved)
        replay.restore(saved["replay"])
        counters = saved["counters"].copy()
        elapsed_prior = counters["wall_time_s"]
        collector = saved["collector"]
        env.restore(collector["environment"])
        collector_rng.bit_generator.state = collector["rng"]
        observation, episode = collector["observation"], collector["episode"]
        state = RSSMState(collector["h"].to(device), collector["z"].to(device))
        restore_rng(saved["rng"])
        append_json(run_dir / "events.jsonl", {"utc": utc(), "event": "explicit_resume", "checkpoint": str(resume), **counters})
    atomic_json(run_dir / "config.json", config)
    atomic_json(run_dir / "metadata.json", meta)
    (run_dir / "pip-freeze.txt").write_text(command([sys.executable, "-m", "pip", "freeze"]) + "\n")
    atomic_json(run_dir / "command.json", {"argv": sys.argv, "cwd": str(Path.cwd()), "python": sys.executable})
    (run_dir / "train.pid").write_text(str(os.getpid()) + "\n")
    averages = defaultdict(list)
    last_resource = 0.0
    checkpoint_path = resume
    best_mean = -float("inf")
    best_pointer = run_dir / "CHECKPOINTS" / "best.json"
    if resume and best_pointer.exists():
        best_mean = json.loads(best_pointer.read_text())["eval_mean"]
    print(json.dumps({"event": "training_started", "utc": utc(), "pid": os.getpid(), "budget_agent_steps": config["budget_agent_steps"], **counters}), flush=True)
    try:
        while counters["agent_steps"] < config["budget_agent_steps"]:
            step_start = time.monotonic()
            with torch.no_grad():
                if counters["agent_steps"] < config["prefill_agent_steps"]:
                    action = torch.as_tensor(collector_rng.uniform(-1, 1, 6).astype(np.float32), device=device).unsqueeze(0)
                else:
                    action, _ = learner.actor.sample(state.features())
                # Save the prior and reward BEFORE observing the real outcome.
                dreamed, saved_prior = learner.model.imagine_step(state, action)
                saved_reward = learner.model.reward_head(dreamed.features()).squeeze(-1)
                version = counters["update_steps"]
                result = env.step(action[0].cpu().numpy())
                next_observation = result["observation"]
                next_state, _, posterior = learner.model.observe_step(state, action, image_tensor(next_observation, device),
                                                                       torch.zeros(1, dtype=torch.bool, device=device))
                error = observed_prediction_errors(posterior, saved_prior,
                    torch.tensor([result["reward"]], device=device), saved_reward, config["learner"]["reward_scale"])
            for key, value in (("observations", next_observation), ("actions", action[0].cpu().numpy()),
                               ("rewards", result["reward"]), ("continuations", result["continuation"]),
                               ("errors", error[0].cpu().numpy()), ("error_valid", True), ("model_versions", version)):
                episode[key].append(value)
            observation, state = next_observation, next_state
            counters["agent_steps"] += 1
            counters["env_steps"] += result["env_steps"]
            averages["collection_seconds"].append(time.monotonic() - step_start)
            if result["done"]:
                replay.add_episode(episode_arrays(episode))
                counters["episodes"] += 1
                row = {"kind": "episode", "utc": utc(), **counters, "seed": config["seed"],
                       "episode_return": float(sum(episode["rewards"])), "episode_length": len(episode["actions"]),
                       "replay_active_steps": replay.total_steps, "replay_disk_bytes": replay.disk_bytes}
                append_json(run_dir / "metrics.jsonl", row)
                print(json.dumps(row), flush=True)
                observation = env.reset()
                with torch.no_grad():
                    state, _, _ = learner.model.observe_step(learner.model.initial(1), torch.zeros(1, 6, device=device),
                                                             image_tensor(observation, device), torch.ones(1, dtype=torch.bool, device=device))
                episode = new_episode(observation)
            # No optimizer can change the parameters between saved prediction
            # and observed error. Real collection data never receives gradients.
            if counters["agent_steps"] >= config["prefill_agent_steps"] and counters["agent_steps"] % config["train_every"] == 0 and replay.can_sample(config["sequence_length"]):
                update_start = time.monotonic()
                batch = tensor_batch(replay.sample(config["batch_size"], config["sequence_length"]), device)
                sample_seconds = time.monotonic() - update_start
                update = learner.update(batch, counters["update_steps"])
                counters["update_steps"] += 1
                update.update({"sample_seconds": sample_seconds, "update_seconds": time.monotonic() - update_start})
                for key, value in update.items():
                    averages[key].append(value)
            counters["wall_time_s"] = elapsed_prior + time.monotonic() - start_wall
            if counters["agent_steps"] % config["log_every"] == 0:
                row = {"kind": "training", "utc": utc(), **counters, "seed": config["seed"],
                       **{key: float(np.mean(value)) for key, value in averages.items()},
                       "fps_agent": counters["agent_steps"] / max(counters["wall_time_s"], 1)}
                append_json(run_dir / "metrics.jsonl", row)
                atomic_json(run_dir / "status.json", {"state": "training", **row})
                averages.clear()
            if time.monotonic() - last_resource >= config["resource_every_seconds"]:
                resource = resources(run_dir, counters)
                last_resource = time.monotonic()
                if resource["disk_free_bytes"] < config["minimum_free_disk_bytes"]:
                    raise RuntimeError("free disk below fixed safety threshold; retain prior verified checkpoints")
            if counters["agent_steps"] % config["checkpoint_every"] == 0 or counters["agent_steps"] == config["budget_agent_steps"]:
                collector = {"environment": env.snapshot(), "rng": collector_rng.bit_generator.state,
                             "observation": observation, "episode": episode, "h": state.h.cpu(), "z": state.z.cpu()}
                checkpoint_path = save_checkpoint(run_dir / "CHECKPOINTS", learner, config, counters, replay, collector, meta)
                append_json(run_dir / "events.jsonl", {"event": "checkpoint", "utc": utc(), "file": checkpoint_path.name, **counters})
                print(json.dumps({"event": "checkpoint", "file": checkpoint_path.name, **counters}), flush=True)
            if counters["agent_steps"] % config["eval_every"] == 0 and counters["agent_steps"] < config["budget_agent_steps"]:
                assert checkpoint_path is not None
                report = evaluate(learner, config, config["eval_seeds"], run_dir, counters["agent_steps"], counters["env_steps"],
                                  counters["update_steps"], "periodic", checkpoint_path.name, video=False)
                # Raw episode rows are the authoritative result; no best-seed selection.
                rows = [json.loads(line) for line in (run_dir / "evaluations.jsonl").read_text().splitlines()]
                values = [row["episode_return"] for row in rows if row["agent_steps"] == counters["agent_steps"] and row["kind"] == "periodic"]
                mean = float(np.mean(values))
                if mean > best_mean:
                    best_mean = mean
                    atomic_json(best_pointer, {"file": checkpoint_path.name, "eval_mean": mean, "agent_steps": counters["agent_steps"], "eval_seeds": config["eval_seeds"]})
                export_run(run_dir, include_manifest=False)
        assert checkpoint_path is not None
        # Final acceptance evaluates a separately constructed, disk-reloaded model.
        atomic_json(run_dir / "status.json", {"state": "final_evaluation", **counters})
        frozen = load_checkpoint(checkpoint_path)
        reloaded = HelixLearner(PixelRSSM(**config["model"]).to(device), config["learner"])
        restore_learner(reloaded, frozen)
        for name, value in learner.state_dict().items():
            if not torch.equal(value, reloaded.state_dict()[name]):
                raise AssertionError(f"checkpoint reload mismatch: {name}")
        report = evaluate(reloaded, config, config["final_eval_seeds"], run_dir, counters["agent_steps"], counters["env_steps"],
                          counters["update_steps"], "final", checkpoint_path.name, video=True)
        summary = export_run(run_dir, include_manifest=False)
        result = {"state": "training_and_frozen_evaluation_complete", "utc": utc(), **counters,
                  "checkpoint": checkpoint_path.name, "checkpoint_reload_equal": True,
                  "budget_complete": counters["agent_steps"] == config["budget_agent_steps"],
                  "expected_env_steps_match": counters["env_steps"] == config["budget_agent_steps"] * 2,
                  "experiment_class": config["experiment_class"], "paper_level_complete": False,
                  "evaluation_report": report, "export": summary,
                  "d_drive_acceptance": "pending external transfer and independent integrity/video verification"}
        atomic_json(run_dir / "completion.json", result)
        atomic_json(run_dir / "status.json", {key: result[key] for key in ("state", "utc", "agent_steps", "env_steps", "update_steps")})
        print(json.dumps({"event": "completed", **counters}), flush=True)
    except BaseException as exc:
        failure = {"state": "failed", "utc": utc(), **counters, "exception": repr(exc), "traceback": traceback.format_exc(),
                   "automatic_resume": False, "last_checkpoint": checkpoint_path.name if checkpoint_path else None}
        atomic_json(run_dir / "failure.json", failure)
        atomic_json(run_dir / "status.json", failure)
        raise
    finally:
        env.close()


def run(config: dict, run_dir: Path, device: str, resume: Path | None = None) -> None:
    # The lock is held before any run mutation; a second launch cannot share
    # replay, output files or checkpoint pointers with a live process.
    with exclusive_run_lock(run_dir):
        if resume:
            saved = load_checkpoint(resume)
            if saved["config"] != config:
                raise ValueError("resume configuration differs from saved configuration")
            resume = prepare_resume_branch(run_dir, resume, saved)
        else:
            prepare_fresh_run(run_dir)
        try:
            _run(config, run_dir, device, resume)
        except BaseException as exc:
            # Initialization failures are also visible without relying on tmux.
            if not (run_dir / "failure.json").exists():
                failure = {"state": "initialization_failed", "utc": utc(), "exception": repr(exc),
                           "traceback": traceback.format_exc(), "automatic_resume": False}
                atomic_json(run_dir / "failure.json", failure)
                atomic_json(run_dir / "status.json", failure)
            raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--resume", type=Path, help="Explicit trusted checkpoint; never selected automatically")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    run(config, args.run_dir.resolve(), args.device, args.resume)


if __name__ == "__main__":
    main()
