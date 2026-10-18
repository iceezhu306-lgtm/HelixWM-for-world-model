"""One task-locked Minecraft training run with explicit boundary-only recovery.

This runner is deliberately separate from the frozen Walker pipeline. Only
training actions consume the budget; acknowledged native calls are counted from the
worker. Episode-ending checks may pass a nominal checkpoint/evaluation threshold
by at most one episode; the last training action never passes the budget.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
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

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import numpy as np

from helixwm.run_storage import exclusive_run_lock, prepare_fresh_run
from helixwm_v2.dual_chain import DualChain
from helixwm_v2.evaluation import DropoutSchedule
from .backbone import MinecraftBackbone, TASK_NAMESPACE, UPSTREAM_COMMIT
from .protocol import TASK_ID
from .storage import (FORMAT, Replay, atomic_json, sha256, tree_hash, save_checkpoint,
                      load_checkpoint, restore_chain_boundary, validate_payload)


def utc():
    return datetime.now(timezone.utc).isoformat()


def append(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(row, allow_nan=False, sort_keys=True)+"\n")
        file.flush()


def runtime_metadata():
    packages = {}
    for name in ("jax", "jaxlib", "numpy", "elements", "ninjax", "imageio", "imageio-ffmpeg", "matplotlib"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    try:
        gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                                       "--format=csv,noheader"], text=True, timeout=10).strip()
    except (OSError, subprocess.SubprocessError) as error:
        gpu = {"unavailable": type(error).__name__}
    return {"python": sys.version, "executable": sys.executable, "packages": packages,
            "gpu": gpu, "worker_runtime": "isolated process; actual handshake and task manifest in diagnostics"}


def resources(run_dir, counters):
    row = {"utc": utc(), **counters, "disk_free_bytes": shutil.disk_usage(run_dir).free}
    for name in ("memory.current", "memory.max", "memory.events"):
        path = Path("/sys/fs/cgroup")/name
        if path.is_file():
            row["cgroup_"+name.replace(".", "_")] = path.read_text().strip()
    try:
        row["gpu"] = subprocess.check_output(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
                                               "--format=csv,noheader,nounits"], text=True, timeout=10).strip()
    except (OSError, subprocess.SubprocessError) as error:
        row["gpu_unavailable"] = type(error).__name__
    append(Path(run_dir)/"resources.jsonl", row)
    return row


def config_sha256(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def source_identity(root=None):
    root = Path(root or Path(__file__).resolve().parents[1])
    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()
    if git("status", "--porcelain"):
        raise ValueError("Minecraft launch requires a clean committed source release")
    required = [str(path.relative_to(root)).replace("\\", "/")
                for name in ("helixwm_minecraft", "helixwm_v2")
                for path in (root/name).glob("*.py")]
    git("ls-files", "--error-unmatch", "--", *required)
    return {"commit": git("rev-parse", "HEAD"), "tree": git("rev-parse", "HEAD^{tree}"),
            "upstream_commit": UPSTREAM_COMMIT, "task_namespace": TASK_NAMESPACE}


def validate_config(config, *, formal=False):
    if config.get("task_namespace") != TASK_NAMESPACE or config.get("seed") != 0:
        raise ValueError("Only the new Minecraft seed0 task is authorized")
    if config.get("task_id", TASK_ID) != TASK_ID:
        raise ValueError("Training configuration does not match the actual Minecraft task")
    for name in ("budget_agent_steps", "checkpoint_every", "eval_every", "replay_capacity_steps"):
        if type(config[name]) is not int or config[name] <= 0:
            raise ValueError("Action budgets and schedules must be positive integers")
    if formal and (config["budget_agent_steps"] != 1_000_000 or config["checkpoint_every"] != 10_000
                   or config["eval_every"] != 20_000 or config["final_eval_episodes"] != 20):
        raise ValueError("Formal Minecraft budget/schedule must remain 1M actions, 10k checkpoints, 20k eval, final20")
    if formal and (config.get("periodic_eval_episodes") != 5 or
                   config.get("final_arms") != ["clean", "black_frame", "prior_only", "v2_exact", "locked_ancestor", "posterior_return_once"] or
                   config.get("final_gap_lengths") != [1, 5, 10, 20, 50] or config.get("final_gap_start") != 10 or
                   config.get("eval_seed_start") != 10000 or
                   config.get("final_video") != {"arm": "v2_exact", "seed": 10000, "gap_length": 20}):
        raise ValueError("Formal frozen evaluation must retain the approved paired 520-case protocol")
    env = config["environment"]
    if env["action_repeat"] != 2 or env["image_size"] != 64 or not 2 <= env["episode_env_steps"] <= 6000:
        raise ValueError("Minecraft requires RGB64/repeat2 and an explicit bounded tick horizon")
    if "seed" in env or "diagnostics_path" in env or "failure_log_path" in env:
        raise ValueError("Episode seeds and output paths are owned by this runner")
    if (type(config["prefill_agent_steps"]) is not int or
            not 0 <= config["prefill_agent_steps"] < config["budget_agent_steps"] or
            not np.isfinite(config["train_ratio"]) or config["train_ratio"] <= 0):
        raise ValueError("Invalid collection/update schedule")
    for key in ("batch_size", "batch_length", "burnin"):
        value = config["backbone"][key]
        if type(value) is not int or value < (0 if key == "burnin" else 1):
            raise ValueError("Invalid replay batch contract")
    if config["backbone"].get("seed", 0) != config["seed"]:
        raise ValueError("Learner initialization must use the task training seed")


def verify_launch_gate(path, config, identity):
    if path is None:
        raise ValueError("Formal Minecraft launch requires its own live preflight/smoke/restore gate")
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    if (report.get("task_namespace") != TASK_NAMESPACE or report.get("identity") != identity or
            report.get("config_sha256") != config_sha256(config) or report.get("passed") is not True):
        raise ValueError("Minecraft gate identity/config mismatch")
    required = ("real_minecraft_environment", "visible_goal_confirmed", "finite_official_update",
                "checkpoint_full_reload", "fresh_process_boundary_restore", "smoke_budget_exact",
                "frozen_evaluation", "video_full_decode", "learning_curve_files", "capacity_sufficient",
                "short_episode_sampling_coverage")
    if any(report.get("checks", {}).get(key) is not True for key in required):
        raise ValueError("Minecraft formal gate has missing live acceptance checks")
    throughput = report.get("throughput", {})
    if (throughput.get("measured_seconds", 0) < 60 or throughput.get("actual_agent_steps", 0) <= 0
            or not np.isfinite(throughput.get("actions_per_second", np.nan))
            or throughput["actions_per_second"] <= 0):
        raise ValueError("A real 60-second throughput/capacity measurement is required")
    return {"file": str(Path(path).resolve()), "sha256": sha256(path)}


def new_episode(image):
    return {"image": [image.copy()], "action": [], "reward": [0.], "is_first": [True],
            "is_last": [False], "is_terminal": [False]}


def episode_arrays(episode):
    arrays = {key: np.asarray(value, dtype=(np.uint8 if key == "image" else bool
              if key.startswith("is_") else np.float32)) for key, value in episode.items()}
    actions = arrays["action"].reshape(-1, 6)
    arrays["action"] = np.concatenate((actions, np.zeros((1, 6), np.float32)))
    return arrays


def target_updates(config, actions):
    return int(max(0, actions-config["prefill_agent_steps"]) * config["train_ratio"] /
               (config["backbone"]["batch_size"]*config["backbone"]["batch_length"]))


def batch_for_backbone(replay, config):
    batch = replay.sample(config["batch_size"], config["batch_length"]+config["burnin"], burnin=config["burnin"])
    shape = batch["reward"].shape
    batch["consec"] = np.zeros(shape, np.int32)
    batch["stepid"] = np.zeros((*shape, 20), np.uint8)
    return batch


def episode_outcome(path, seed, ticks, *, required=True):
    """Postepisode audit only. Privileged values never enter policy or learner."""
    path = Path(path)
    if not path.is_file():
        if required:
            raise ValueError("Minecraft episode has no native outcome diagnostics")
        return {"success": None, "available": False}
    with path.open("rb") as file:
        file.seek(max(0, path.stat().st_size-65536))
        lines = file.read().splitlines()
    if not lines:
        raise ValueError("Minecraft outcome diagnostics are empty")
    raw = lines[-1]
    row = json.loads(raw)
    success = row.get("privileged", {}).get("success")
    if (row.get("task_id") != TASK_ID or row.get("event") != "native_step" or
            row.get("episode_seed") != seed or row.get("episode_ticks") != ticks or type(success) is not bool):
        raise ValueError("Minecraft outcome diagnostics do not match the completed episode")
    return {"success": success, "available": True, "native_diagnostics_row_sha256": hashlib.sha256(raw).hexdigest()}


def next_threshold(threshold, interval, current):
    while threshold <= current:
        threshold += interval
    return threshold


def verify_evaluation_report(report, run_dir, config, *, final=False, smoke=False):
    """Check persisted artifact binding independently of the evaluator return."""
    root = Path(run_dir).resolve()
    if (report.get("status") != "complete" or report.get("hash_gate", {}).get("passed") is not True or
            report.get("post_artifact_frozen_gate", {}).get("passed") is not True or
            report.get("collector_inference_restored") is not True):
        raise ValueError("Frozen evaluation or collector restoration gate failed")
    conditions = report.get("conditions", {})
    arms = config["final_arms"] if final else ["clean", "dropout"]
    if set(conditions) != set(arms):
        raise ValueError("Frozen evaluation condition matrix is incomplete")
    if final:
        if report.get("actual_case_count") != 520:
            raise ValueError("Final Minecraft evaluation must contain all 520 cases")
        for arm in arms:
            expected_gaps = [0] if arm == "clean" else config["final_gap_lengths"]
            groups = conditions[arm].get("by_gap_length", {})
            if set(groups) != {str(gap) for gap in expected_gaps}:
                raise ValueError("Final Minecraft gap matrix is incomplete")
            for group in groups.values():
                if (group.get("episodes") != 20 or group.get("evaluation_seeds") != list(range(10000, 10020)) or
                        group.get("unknown_goal_outcomes") != 0):
                    raise ValueError("Final fixed-seed outcome evidence is incomplete")
    else:
        expected = config["periodic_eval_episodes"]
        if any(value.get("episodes") != expected for value in conditions.values()):
            raise ValueError("Periodic/smoke evaluation episode count differs from config")
    def file(relative):
        path = root/relative
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root):
            raise ValueError("Evaluation artifact is missing or escaped its run directory")
        return path
    curve = report.get("learning_curve", {})
    if curve.get("available") is not True:
        raise ValueError("Learning curve was not generated")
    paths = [file(relative) for relative in curve.get("files", [])]
    if {path.suffix.lower() for path in paths} != {".png", ".pdf"}:
        raise ValueError("Learning curve needs PNG and PDF")
    for path in paths:
        with path.open("rb") as stream:
            header = stream.read(8)
        if not (header.startswith(b"%PDF-") if path.suffix.lower() == ".pdf" else header == b"\x89PNG\r\n\x1a\n"):
            raise ValueError("Learning curve file header is invalid")
    file(report["raw_csv"])
    if final or smoke:
        proof = report.get("video", {})
        if proof.get("full_decode_pass") is not True or proof.get("frames") != proof.get("frames_expected"):
            raise ValueError("Selected video was not fully decoded at the actual frame count")
        path = file(proof["path"])
        if path.stat().st_size != proof.get("bytes") or sha256(path) != proof.get("sha256"):
            raise ValueError("Selected video no longer matches its decode proof")
        if final and (proof.get("condition") != "v2_exact" or proof.get("evaluation_seed") != 10000
                      or proof.get("gap_length") != 20 or len(list((root/"VIDEOS").glob("*.mp4"))) != 1):
            raise ValueError("Final video does not match the single predeclared case")
    return True


def restore_boundary(payload, config, backbone, replay, chain, dropout):
    validate_payload(payload)
    if payload["config"] != config:
        raise ValueError("Minecraft recovery requires the exact configuration")
    boundary = payload["collector"]
    # Validate all software RNGs before loading a model. No world is restored.
    chain._validated_snapshot(boundary["dual_chain"])
    probe = DropoutSchedule(0, config["observation_dropout"])
    probe.load_state_dict(boundary["dropout"])
    python_probe = random.Random()
    python_probe.setstate(payload["rng"]["python"])
    numpy_probe = np.random.RandomState()
    numpy_probe.set_state(payload["rng"]["numpy"])
    backbone.load(payload["backbone"])
    replay.restore(payload["replay"], persist=True)
    restore_chain_boundary(chain, boundary["dual_chain"], boundary["episode_action_count"])
    dropout.load_state_dict(boundary["dropout"])
    random.setstate(payload["rng"]["python"])
    np.random.set_state(payload["rng"]["numpy"])
    return deepcopy(payload["counters"]), deepcopy(payload["scheduler"])


def copy_recovery_replay(checkpoint, payload, run_dir):
    source = Path(checkpoint).resolve().parent.parent/"REPLAY"
    target = Path(run_dir)/"REPLAY"
    target.mkdir(exist_ok=True)
    for record in payload["replay"]["archive"]:
        original, destination = source/record["file"], target/record["file"]
        if destination.exists():
            raise FileExistsError(destination)
        if original.stat().st_size != record["bytes"] or sha256(original) != record["sha256"]:
            raise ValueError("Immutable recovery replay mismatch")
        try:
            os.link(original, destination)
        except OSError:
            shutil.copy2(original, destination)
        atomic_json(destination.with_suffix(".json"), record)


def run(config, run_dir, upstream, *, kind="formal", launch_gate=None, resume=None,
        env_factory=None, backbone_factory=None, evaluator=None, identity=None):
    if kind not in {"formal", "smoke", "test"}:
        raise ValueError("Unknown run kind")
    validate_config(config, formal=kind == "formal")
    if kind == "smoke" and (config["budget_agent_steps"] != 1000 or config["prefill_agent_steps"] != 500
                            or config["checkpoint_every"] != 500 or config["eval_every"] != 1000):
        raise ValueError("Smoke is an explicit separate 1000-action run, not a bypass for formal launch")
    identity = source_identity() if identity is None else deepcopy(identity)
    if kind == "formal" and identity != source_identity():
        raise ValueError("Formal identity must be the actual immutable checkout")
    gates = verify_launch_gate(launch_gate, config, identity) if kind == "formal" else None
    if env_factory is None:
        from .bridge import MinecraftEnv
        env_factory = MinecraftEnv
    if evaluator is None:
        from .evaluation import evaluate
        evaluator = evaluate
    backbone_factory = backbone_factory or MinecraftBackbone
    run_dir = Path(run_dir)
    prepare_fresh_run(run_dir)
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    atomic_json(run_dir/"config.json", config)
    atomic_json(run_dir/"command.json", {"argv": sys.argv, "cwd": str(Path.cwd()), "python": sys.executable})
    (run_dir/"train.pid").write_text(str(os.getpid())+"\n")
    backbone = backbone_factory(config["backbone"], upstream, run_dir)
    meta = {"created_utc": utc(), "identity": identity, "kind": kind, "launch_gates": gates,
            "task_namespace": TASK_NAMESPACE, "initialization": "random_seed_no_transfer",
            "resolved_backbone_config": backbone.resolved_config,
            "units": {"agent_steps": "actual executed training decisions",
                      "env_steps": "acknowledged native synchronous MineRL step calls; not total world ticks",
                      "nominal_step_seconds": 0.05, "world_ticks_including_reset_measured": False},
            "learner_targets": "unmodified real RGB frames including explicitly hidden collector frames",
            "model_updates": "pinned official DreamerV3 losses/optimizer via V2 adapter",
            "policy_sync": "once per episode", "engine_state_serialized": False,
            "recovery": "software RNG/state retained; fresh next episode reset; no engine state equality promise",
            "paper_level_complete": False}
    if kind != "test":
        meta["runtime"] = runtime_metadata()
        (run_dir/"pip-freeze.txt").write_text(subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"], text=True), encoding="utf-8")
    atomic_json(run_dir/"metadata.json", meta)
    replay = Replay(run_dir/"REPLAY", config["replay_capacity_steps"], config["seed"]+101,
                    config.get("replay_cache_episodes", 4))
    chain = DualChain(backbone, seed=config["seed"]+303, **config["dual_chain"])
    dropout = DropoutSchedule(config["seed"]+7001, config["observation_dropout"])
    counters = dict(agent_steps=0, env_steps=0, episodes=0, update_steps=0,
                    masked_observations=0, wall_time_s=0.)
    scheduler = dict(next_checkpoint=config["checkpoint_every"], next_evaluation=config["eval_every"], target_updates=0)
    if resume:
        payload = load_checkpoint(resume)
        if payload["metadata"]["identity"] != identity:
            raise ValueError("Recovery must use the exact frozen source/upstream identity")
        if payload["counters"]["agent_steps"] >= config["budget_agent_steps"]:
            raise ValueError("Completed Minecraft training cannot restart as another run")
        copy_recovery_replay(resume, payload, run_dir)
        counters, scheduler = restore_boundary(payload, config, backbone, replay, chain, dropout)
        meta["parent_checkpoint"] = {"path": str(Path(resume).resolve()), "sha256": sha256(resume)}
        atomic_json(run_dir/"metadata.json", meta)
    started = time.monotonic()-counters["wall_time_s"]
    window_start, window_actions, window_ticks = time.monotonic(), counters["agent_steps"], counters["env_steps"]
    pending = defaultdict(list)
    env, episode, checkpoint, last_eval = None, None, None, None
    actual_counters_complete = True
    if kind != "test":
        print(json.dumps({"event": "minecraft_training_started", "utc": utc(), "kind": kind,
                          "task_namespace": TASK_NAMESPACE, **counters}), flush=True)
    try:
        while counters["agent_steps"] < config["budget_agent_steps"]:
            if env is None:
                env = env_factory(seed=config["seed"], **config["environment"],
                                  diagnostics_path=str(run_dir/"environment_diagnostics.jsonl"),
                                  failure_log_path=str(run_dir/"environment_failures.jsonl"))
            backbone.pin_inference()
            episode_seed = config["seed"]+counters["episodes"]
            image = np.asarray(env.reset(seed=episode_seed)).reshape(64, 64, 3)
            if image.dtype != np.uint8:
                raise ValueError("Minecraft reset must return uint8 RGB64")
            dropout.valid(is_first=True)
            episode = new_episode(image)
            chain.reset(image, episode_id=counters["episodes"], timestamp=0, model_version=backbone.inference_version)
            episode_actions, episode_ticks = 0, 0
            while True:
                action = np.asarray(chain.act()[0], np.float32)
                actual_counters_complete = False  # A worker failure may follow a native call with no reply.
                result = env.step(action)
                actual_counters_complete = True
                ticks = result["env_steps"]
                if type(ticks) is not int or not 1 <= ticks <= 2:
                    raise ValueError("Worker did not report actual one/two acknowledged native calls")
                counters["agent_steps"] += 1
                counters["env_steps"] += ticks
                episode_actions += 1
                episode_ticks += ticks
                image = np.asarray(result["observation"]).reshape(64, 64, 3)
                reward = float(result["reward"])
                if image.dtype != np.uint8 or not np.isfinite(reward):
                    raise ValueError("Invalid Minecraft RGB/reward")
                if result["episode_seed"] != episode_seed or result["continuation"] != (0. if result["terminated"] else 1.):
                    raise ValueError("Worker episode identity or continuation mismatch")
                final = counters["agent_steps"] == config["budget_agent_steps"]
                done = bool(result["done"] or final)
                if episode_ticks > config["environment"]["episode_env_steps"]:
                    raise RuntimeError("Worker exceeded configured native-call horizon")
                if bool(result["done"]) != bool(result["terminated"] or result["truncated"]):
                    raise ValueError("Worker done/termination/truncation flags disagree")
                if ticks != 2 and not result["done"]:
                    raise ValueError("A partial repeat must coincide with an episode ending")
                if episode_ticks == config["environment"]["episode_env_steps"] and not result["done"]:
                    raise ValueError("Worker must end at its native-call horizon")
                valid = dropout.valid(is_last=done)
                # Retain every valid, acknowledged raw transition even if the
                # subsequent policy-state update fails. Inference never receives
                # this raw frame when availability is false.
                episode["action"].append(action.copy())
                episode["image"].append(image.copy())
                episode["reward"].append(reward)
                episode["is_first"].append(False)
                episode["is_last"].append(done)
                episode["is_terminal"].append(bool(result["terminated"]))
                counters["masked_observations"] += int(not valid)
                chain.step(image if valid else None, action, episode_id=counters["episodes"],
                           timestamp=episode_actions, model_version=backbone.inference_version, valid=valid)
                for event in chain.pop_events():
                    append(run_dir/"dual_chain_events.jsonl", {"agent_steps": counters["agent_steps"], **event})
                if done:
                    outcome = episode_outcome(run_dir/"environment_diagnostics.jsonl", episode_seed, episode_ticks,
                                              required=kind != "test")
                    replay.add_episode(episode_arrays(episode), success=outcome["success"])
                scheduler["target_updates"] = target_updates(config, counters["agent_steps"])
                sample_length = config["backbone"]["batch_length"]+config["backbone"]["burnin"]
                while (counters["update_steps"] < scheduler["target_updates"] and replay.can_sample(sample_length)):
                    metrics = backbone.train(batch_for_backbone(replay, config["backbone"]))
                    if not all(np.isfinite(value) for value in metrics.values()):
                        raise FloatingPointError("Nonfinite official learner metric")
                    counters["update_steps"] += 1
                    for name, value in metrics.items():
                        pending[name].append(value)
                counters["wall_time_s"] = time.monotonic()-started
                if counters["agent_steps"] % 100 == 0 or done:
                    append(run_dir/"metrics.jsonl", {"kind": "train", "utc": utc(), **counters,
                           "target_updates": scheduler["target_updates"],
                           "unsatisfied_update_budget": scheduler["target_updates"]-counters["update_steps"],
                           **{name: float(np.mean(values)) for name, values in pending.items()}})
                    pending.clear()
                    atomic_json(run_dir/"status.json", {"state": "running", **counters})
                elapsed = time.monotonic()-window_start
                if elapsed >= 60:
                    append(run_dir/"throughput.jsonl", {"utc": utc(), "measured_seconds": elapsed,
                           "actual_agent_steps": counters["agent_steps"]-window_actions,
                           "actual_env_steps": counters["env_steps"]-window_ticks,
                           "actions_per_second": (counters["agent_steps"]-window_actions)/elapsed,
                           "includes_reset_and_learner": True, "replay_disk_bytes": replay.disk_bytes,
                           "disk_free_bytes": shutil.disk_usage(run_dir).free})
                    if kind != "test":
                        resources(run_dir, counters)
                    window_start, window_actions, window_ticks = time.monotonic(), counters["agent_steps"], counters["env_steps"]
                if done:
                    break
            counters["episodes"] += 1
            append(run_dir/"metrics.jsonl", {"kind": "episode", "utc": utc(), **counters,
                   "episode_return": sum(episode["reward"]), "episode_length": episode_actions,
                   "episode_env_steps": episode_ticks, "episode_seed": episode_seed,
                   "terminated": result["terminated"], "time_limit_truncated": result["truncated"],
                   "budget_truncated": final and not result["done"], "native_outcome": outcome,
                   "replay_sampling": replay.sampling_statistics(sample_length, burnin=config["backbone"]["burnin"])})
            if kind != "test":
                print(json.dumps({"event": "episode_complete", "agent_steps": counters["agent_steps"],
                                  "actual_native_steps": counters["env_steps"], "episode_seed": episode_seed,
                                  "episode_return": sum(episode["reward"])}), flush=True)
            due_eval = counters["agent_steps"] >= scheduler["next_evaluation"] or final
            due_checkpoint = counters["agent_steps"] >= scheduler["next_checkpoint"] or due_eval
            if due_checkpoint:
                scheduled_checkpoint, scheduled_eval = scheduler["next_checkpoint"], scheduler["next_evaluation"]
                scheduler["next_checkpoint"] = next_threshold(scheduler["next_checkpoint"], config["checkpoint_every"], counters["agent_steps"])
                scheduler["next_evaluation"] = next_threshold(scheduler["next_evaluation"], config["eval_every"], counters["agent_steps"])
                finished_chain = chain.state_dict()
                backbone.pin_inference()
                payload = {"format": FORMAT, "task_namespace": TASK_NAMESPACE, "config": config,
                           "metadata": meta, "counters": counters.copy(), "backbone": backbone.snapshot(),
                           "replay": replay.snapshot(), "collector": {
                               "next_operation": "reset_new_episode_and_reanchor", "episode_complete": True,
                               "engine_state_serialized": False, "episode_action_count": episode_actions,
                               "episode_native_steps": episode_ticks, "next_episode_seed": config["seed"]+counters["episodes"],
                               "dual_chain": finished_chain, "dropout": dropout.state_dict(),
                               "budget_truncated": final and not result["done"]},
                           "rng": {"python": random.getstate(), "numpy": np.random.get_state()},
                           "scheduler": scheduler.copy()}
                checkpoint = save_checkpoint(run_dir/"CHECKPOINTS", payload)
                append(run_dir/"events.jsonl", {"event": "checkpoint_saved", "file": checkpoint.name,
                       "agent_steps": counters["agent_steps"], "scheduled_checkpoint": scheduled_checkpoint,
                       "checkpoint_delay_actions": max(0, counters["agent_steps"]-scheduled_checkpoint),
                       "scheduled_evaluation": scheduled_eval, "boundary_only": True})
            if due_eval:
                before = backbone.parameter_hash()
                backbone.load(load_checkpoint(checkpoint, verify_replay=False)["backbone"])
                if before != backbone.parameter_hash():
                    raise RuntimeError("Strict Minecraft checkpoint reload changed model/optimizer")
                env.close()
                env = None
                last_eval = evaluator(backbone, config, run_dir, counters["agent_steps"],
                                      kind="smoke" if kind != "formal" else "final" if final else "periodic",
                                      training_env_steps=counters["env_steps"])
                if before != backbone.parameter_hash():
                    raise RuntimeError("Frozen evaluation changed model/optimizer")
                if kind != "test":
                    verify_evaluation_report(last_eval, run_dir, config,
                                             final=final and kind == "formal", smoke=kind == "smoke")
                window_start, window_actions, window_ticks = time.monotonic(), counters["agent_steps"], counters["env_steps"]
        if counters["agent_steps"] != config["budget_agent_steps"] or checkpoint is None:
            raise RuntimeError("Minecraft action budget incomplete")
        if counters["update_steps"] != scheduler["target_updates"]:
            raise RuntimeError("Insufficient sampleable replay: official update budget was not met")
        completion = {"task_namespace": TASK_NAMESPACE, "identity": identity, "kind": kind,
                      "completed_utc": utc(), "budget_complete": True, **counters,
                      "native_step_bounds_verified": counters["agent_steps"] <= counters["env_steps"] <= 2*counters["agent_steps"],
                      "world_tick_total_measured": False,
                      "checkpoint_full_reload": True, "checkpoint": checkpoint.name, "checkpoint_sha256": sha256(checkpoint),
                      "evaluation_report": last_eval, "single_training_seed": True, "efficacy_demonstrated": False}
        atomic_json(run_dir/"completion.json", completion)
        atomic_json(run_dir/"status.json", {"state": "completed", **counters})
        return completion
    except BaseException as error:
        evidence_error = None
        if episode is not None:
            # Preserve all fully acknowledged observation/reward rows even when
            # the engine or inference fails. These are evidence, not resumable replay.
            try:
                np.savez_compressed(run_dir/"failed_episode_evidence.npz", **episode_arrays(episode))
            except BaseException as write_error:
                evidence_error = repr(write_error)
        atomic_json(run_dir/"failure.json", {"utc": utc(), "error": repr(error),
                    "traceback": traceback.format_exc(), **counters,
                    "actual_interaction_counters_complete": actual_counters_complete,
                    "failed_episode_evidence_write_error": evidence_error})
        raise
    finally:
        if env is not None:
            active_exception = sys.exc_info()[0] is not None
            try:
                env.close()
            except Exception as cleanup_error:
                append(run_dir/"events.jsonl", {"event": "environment_cleanup_failed", "utc": utc(),
                       "error": repr(cleanup_error), "during_failure": active_exception})
                if not active_exception:
                    raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--upstream", required=True)
    parser.add_argument("--launch-gate")
    parser.add_argument("--resume")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    with exclusive_run_lock(args.run_dir):
        run(config, args.run_dir, args.upstream, kind="smoke" if args.smoke else "formal",
            launch_gate=args.launch_gate, resume=args.resume)


if __name__ == "__main__":
    main()
