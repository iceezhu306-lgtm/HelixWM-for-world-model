"""Episode-transactional, frozen-checkpoint Walker continuity study.

No optimizer or training entry point is invoked. Resume admits only verified
finished cases; a process-interrupted episode is repeated from its fixed seed
and the original partial attempt remains available for audit.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import pickle
import random
import shutil
import subprocess
import sys
import time
import traceback
import uuid

import numpy as np

from helixwm_v2.dual_chain import _digest
from helixwm_v3.variants import ARMS, make_chain


REFERENCE_COMMIT = "80cb3555586d01e0e7b0e8d362c27e540032d901"
CHECKPOINT_SHA = "8aba0d3c6a1e9116850f34d24caf2fcdc0c68232d849d5bebbb9ec6239c8816e"
ROOT = Path(__file__).resolve().parents[1]
MIN_FREE = 2 * 1024**3


class RunLock:
    """OS advisory lock is released on process exit, including a crash."""
    def __init__(self, path):
        self.path, self.stream = Path(path), None

    def __enter__(self):
        self.stream = self.path.open("a+b")
        try:
            if os.name == "nt":
                import msvcrt
                self.stream.seek(0); self.stream.write(b"0"); self.stream.flush(); self.stream.seek(0)
                msvcrt.locking(self.stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.stream.close()
            raise RuntimeError("Another process already holds this study-stage output")
        return self

    def __exit__(self, *unused):
        self.stream.close()


def utc():
    return datetime.now(timezone.utc).isoformat()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    with tmp.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(data, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def append_jsonl(path, data):
    with Path(path).open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(data, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def resource_sample(output, case_id):
    sample = {"utc": utc(), "case_id": case_id,
              "disk_free_bytes": shutil.disk_usage(output).free}
    for name in ("memory.current", "memory.max", "memory.events", "memory.peak"):
        path = Path("/sys/fs/cgroup") / name
        if path.is_file():
            raw = path.read_text().strip()
            sample[name.replace(".", "_")] = raw
    try:
        command = ["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu", "--format=csv,noheader,nounits"]
        result = subprocess.run(command, capture_output=True, text=True, timeout=10)
        sample["gpu_csv"] = result.stdout.strip()
        sample["gpu_query_exit"] = result.returncode
    except (OSError, subprocess.TimeoutExpired) as exc:
        sample["gpu_query_error"] = str(exc)
    append_jsonl(Path(output) / "resources.jsonl", sample)


def validate_config(config):
    if not config.get("execution_authorized") or not config.get("runner_implemented"):
        raise ValueError("Use a separately authorized execution manifest; preserve the historical design")
    if config.get("training_updates_allowed") is not False or config.get("training_update_budget") != 0:
        raise ValueError("This runner permits no training updates")
    ident = config["identity"]
    if ident["reference_source_commit"] != REFERENCE_COMMIT or ident["checkpoint_sha256"] != CHECKPOINT_SHA:
        raise ValueError("Wrong frozen reference source/checkpoint")
    env = config["environment"]
    if (env["domain"], env["task"], env["image_size"], env["action_repeat"],
        env["episode_action_decisions"], env["episode_dmc_calls"]) != ("walker", "walk", 64, 2, 500, 1000):
        raise ValueError("Registered Walker observation and interaction units changed")
    phase = config["phase_a"]
    schedule = phase["schedule"]
    if (schedule["missing_observation_start"], schedule["events_per_episode"],
        schedule["missing_lengths_action_decisions"]) != (150, 1, [1, 5, 10, 20, 50]):
        raise ValueError("Registered schedule changed")
    if set(x["id"] for x in phase["arms"]) != set(ARMS) or len(phase["arms"]) != 6:
        raise ValueError("Registered arm matrix changed")
    ref = phase["candidate_reference"]
    if tuple(ref[k] for k in ("max_candidates", "max_missing", "score_temperature", "score_normalization")) != (8, 5, .01, 12288):
        raise ValueError("Frozen V2 inference settings changed")
    expected = {"design_environment_seeds": (20000, 20), "held_out_environment_seeds": (30000, 40),
                "excluded_historical_final_seeds": (10000, 20)}
    for key, values in expected.items():
        spec = phase["splits"][key]
        if (spec["start"], spec["count"]) != values:
            raise ValueError("Registered seed split changed")
    if env["privileged_current_state_to_policy"] or env["physical_dynamics_changed_in_phase_a"]:
        raise ValueError("Phase A may not change policy inputs or physics")


def availability(length, horizon=500, start=150):
    if length not in (0, 1, 5, 10, 20, 50) or not 0 < start < start + max(1, length) < horizon:
        raise ValueError("Invalid fixed missingness schedule")
    return np.asarray([not start <= t < start + length for t in range(horizon + 1)], bool)


def case_plan(config, stage):
    if stage == "smoke":
        seeds = [40000]
        # Full horizons; disjoint seeds. Covers each arm, both return branches,
        # long fallback and exact clean-path parity without tuning held-out data.
        pairs = [("clean", 0)] + [(arm, 5) for arm in ARMS if arm != "clean"]
        pairs += [("v2_exact", 10), ("posterior_return_once", 10), ("locked_ancestor", 50)]
        pairs += [(arm, 0) for arm in ("v2_exact", "locked_ancestor", "posterior_return_once")]
    else:
        key = {"design": "design_environment_seeds", "held_out": "held_out_environment_seeds"}[stage]
        spec = config["phase_a"]["splits"][key]
        seeds = range(spec["start"], spec["start"] + spec["count"])
        pairs = [(arm, length) for arm in ARMS
                 for length in ([0] if arm == "clean" else [1, 5, 10, 20, 50])]
    result = []
    for seed in seeds:
        for arm, length in pairs:
            result.append(dict(case_id=f"{stage}-seed{seed}-{arm}-gap{length}",
                               stage=stage, split=stage, arm=arm, environment_seed=seed,
                               chain_root_seed=2000000 + seed, missing_start=150, missing_length=length))
    return result


def source_identity(config):
    def git(*args):
        return subprocess.run(["git", "-C", str(ROOT), *args], check=True,
                              capture_output=True).stdout
    commit = git("rev-parse", "HEAD").decode().strip()
    if git("status", "--porcelain", "--untracked-files=no").strip():
        raise ValueError("Tracked source must be clean before executing the study")
    if config["identity"].get("future_runner_commit") != commit:
        raise ValueError("Execution manifest must bind the exact committed study source")
    required = ("helixwm_v3/study.py", "helixwm_v3/variants.py", "helixwm_v3/analysis.py")
    for path in required:
        if not git("ls-files", "--", path).strip():
            raise ValueError(f"Uncommitted study implementation: {path}")
    refs = [x for x in git("ls-tree", "-r", "--name-only", REFERENCE_COMMIT, "helixwm_v2").decode().splitlines()
            if x.endswith(".py")]
    refs.append("helixwm/environment.py")
    hashes = {}
    for relative in refs:
        accepted = hashlib.sha256(git("show", f"{REFERENCE_COMMIT}:{relative}")).hexdigest()
        local = file_hash(ROOT / relative)
        # Git's CRLF checkout normalization may differ on Windows; compare
        # canonical Git blob content as well as track actual local bytes.
        local_text = (ROOT / relative).read_bytes().replace(b"\r\n", b"\n")
        if hashlib.sha256(local_text).hexdigest() != accepted:
            raise ValueError(f"Preserved runtime changed: {relative}")
        hashes[relative] = {"accepted_git_blob_sha256": accepted, "local_sha256": local}
    return {"source_commit": commit, "reference_files": hashes}


def frozen_state(backbone):
    agent = backbone.agent
    return {"model_optimizer_sha256": backbone.parameter_hash(),
            "inference_sha256": backbone.parameter_hash(inference=True),
            "inference_version": int(backbone.inference_version),
            "counters": {key: int(getattr(agent, key)) for key in ("n_updates", "n_batches", "n_actions")},
            "host_python_rng_sha256": hashlib.sha256(pickle.dumps(random.getstate(), protocol=5)).hexdigest(),
            "host_numpy_rng_sha256": hashlib.sha256(pickle.dumps(np.random.get_state(), protocol=5)).hexdigest()}


class CountingBackend:
    def __init__(self, backend):
        self.backend = backend
        self.calls = {name: 0 for name in ("observe", "prior", "score", "act")}

    def __getattr__(self, name):
        if name in self.calls:
            def call(*args, **kwargs):
                self.calls[name] += 1
                return getattr(self.backend, name)(*args, **kwargs)
            return call
        return getattr(self.backend, name)


def posture(env):
    # Explicit evaluator-only instrumentation. Never forwarded to the actor,
    # missingness schedule, decoder energy, or observation encoder.
    physics = env._env.physics
    values = float(physics.torso_height()), float(physics.torso_upright())
    if not np.isfinite(values).all():
        raise FloatingPointError("Nonfinite evaluator-only posture")
    return values


def image_array(value):
    value = np.asarray(value)
    if value.dtype != np.uint8 or value.size != 64 * 64 * 3:
        raise ValueError("Environment must return actual uint8 RGB64 without silent conversion")
    return value.reshape(64, 64, 3)


def finite_state(state):
    if not all(np.isfinite(np.asarray(value)).all() for value in state.values()):
        raise FloatingPointError("Nonfinite inferred recurrent/stochastic state")


def _video_frame(env, case, timestamp, available, chain):
    from PIL import Image, ImageDraw
    frame = Image.fromarray(env.render(256))
    draw = ImageDraw.Draw(frame)
    draw.rectangle((0, 0, 256, 38), fill=(0, 0, 0))
    draw.text((4, 3), f"{case['arm']} seed {case['environment_seed']} t={timestamp}", fill="white")
    mode = "observed" if available else ("prior fallback" if chain.abstaining else "candidate repair")
    draw.text((4, 20), f"image: {'available' if available else 'MISSING'} | {mode}", fill="white")
    return np.asarray(frame)


def execute_case(backbone, config, case, directory, env_factory, *, make_video=False, store_images=True):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    mask = availability(case["missing_length"])
    atomic_json(directory / "case.json", {**case, "availability": mask.tolist(),
                "availability_sha256": hashlib.sha256(mask.tobytes()).hexdigest(),
                "rng_semantics": "SeedSequence(chain_root_seed).spawn(4): real=0 actor=1 dream=2 repair=3"})
    (directory / "events.jsonl").touch()
    observed, actions, rewards, counts, heights, uprights, latency, states, frames = ([] for _ in range(9))
    received_frames = []
    executed_actions = executed_dmc = 0
    counters_complete = True
    before = frozen_state(backbone)
    wrapped = CountingBackend(backbone)
    chain = make_chain(wrapped, case["arm"], seed=case["chain_root_seed"],
                       model_version=backbone.inference_version,
                       candidate_config=config["phase_a"]["candidate_reference"])
    env, writer, error = None, None, None
    cleanup_errors = []
    boundary, video_check = None, None
    video_name = "smoke_validation.mp4" if case["stage"] == "smoke" else "formal.mp4"
    started = time.monotonic()
    try:
        env = env_factory(seed=case["environment_seed"], action_repeat=2, image_size=64,
                          episode_env_steps=1000, camera_id=config["environment"]["camera_id"])
        raw = image_array(env.reset())
        if store_images:
            observed.append(raw.copy())
        frames.append(hashlib.sha256(raw.tobytes()).hexdigest())
        chain.reset(raw, episode_id=case["case_id"], model_version=backbone.inference_version)
        finite_state(chain.real)
        states.append(_digest(chain.real))
        received_frames.append(frames[-1])
        if make_video:
            import imageio.v2 as imageio
            writer = imageio.get_writer(str(directory / video_name), fps=20, codec="libx264", quality=8)
            writer.append_data(_video_frame(env, case, 0, True, chain))
        for t in range(500):
            begin = time.perf_counter()
            action = chain.act(deterministic=False)
            actor_s = time.perf_counter() - begin
            try:
                result = env.step(action[0])
            except Exception:
                counters_complete = False  # an environment can fail after partial physics work
                raise
            executed_actions += 1
            count = int(result["env_steps"])
            executed_dmc += count
            raw = image_array(result["observation"])
            reward, count = float(result["reward"]), int(result["env_steps"])
            if not np.isfinite(reward) or count != 2:
                raise FloatingPointError("Unexpected/nonfinite action reward or DMC call count")
            height, upright = posture(env)
            # Commit actual actions/outcomes before assimilating the next image.
            # On an inference exception this executed interaction remains saved.
            actions.append(action[0].copy()); rewards.append(reward); counts.append(count)
            heights.append(height); uprights.append(upright)
            latency.append(actor_s)
            if store_images:
                observed.append(raw.copy())
            frames.append(hashlib.sha256(raw.tobytes()).hexdigest())
            begin = time.perf_counter()
            available = bool(mask[t + 1])
            if case["arm"] == "black_frame":
                received, valid = (raw if available else np.zeros_like(raw)), True
            else:
                received, valid = (raw if available else None), available
            chain.step(received, action, episode_id=case["case_id"], timestamp=t + 1,
                       model_version=backbone.inference_version, valid=valid)
            finite_state(chain.real)
            latency[-1] += time.perf_counter() - begin
            states.append(_digest(chain.real))
            received_frames.append("UNAVAILABLE" if received is None else hashlib.sha256(received.tobytes()).hexdigest())
            for event in chain.pop_events():
                append_jsonl(directory / "events.jsonl", {"case_id": case["case_id"], **event})
            if writer is not None:
                writer.append_data(_video_frame(env, case, t + 1, available, chain))
            if bool(result["done"]) != (t == 499):
                raise RuntimeError("Unexpected episode termination relative to the 500-action time limit")
        boundary = {"format_version": 1, "next_operation": "new_case_from_registered_seed",
                    "case": case, "environment": env.snapshot(), "chain": chain.state_dict(),
                    "checkpoint_sha256": config["identity"]["checkpoint_sha256"],
                    "model_version": int(backbone.inference_version),
                    "host_rng": {"python": random.getstate(), "numpy": np.random.get_state()}}
    except Exception as exc:
        error = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
    finally:
        if writer is not None:
            try:
                writer.close()
            except Exception as exc:
                cleanup_errors.append({"operation": "video_close", "type": type(exc).__name__, "message": str(exc)})
        if env is not None:
            try:
                env.close()
            except Exception as exc:
                cleanup_errors.append({"operation": "environment_close", "type": type(exc).__name__, "message": str(exc)})
    if cleanup_errors:
        if error is None:
            error = {"type": "CleanupFailure", "message": "See cleanup_errors"}
        error["cleanup_errors"] = cleanup_errors
    after = frozen_state(backbone)
    summary = {**case, "status": "technical_failure" if error else "complete", "error": error,
               "actual_eval_agent_steps": executed_actions, "actual_eval_dmc_calls": executed_dmc,
               "actual_interaction_counters_complete": counters_complete,
               "valid_reward_rows": len(rewards),
               "episode_return": float(sum(rewards)), "elapsed_s": time.monotonic() - started,
               "inference_version": int(backbone.inference_version), "backend_calls": wrapped.calls,
               "availability_sha256": hashlib.sha256(mask.tobytes()).hexdigest(),
               "observation_storage": "evaluator_only_raw_RGB_never_available_to_policy_on_masked_steps",
               "raw_observations_saved": store_images,
               "posture_accessors": ["env._env.physics.torso_height()", "env._env.physics.torso_upright()"],
               "hash_gate": {"passed": before == after, "before": before, "after": after}}
    # Missing diagnostic samples after an exception are not fabricated values.
    # Separate lengths let analysis distinguish executed actions and state updates.
    np.savez_compressed(directory / "trace.npz", action_index=np.arange(len(actions), dtype=np.int32),
                        reward=np.asarray(rewards, np.float64), env_steps=np.asarray(counts, np.int16),
                        actions=np.asarray(actions, np.float32).reshape(-1, 6),
                        observations=np.asarray(observed, np.uint8).reshape(-1, 64, 64, 3),
                        availability=mask, torso_height=np.asarray(heights, np.float64),
                        torso_upright=np.asarray(uprights, np.float64), latency_s=np.asarray(latency, np.float64),
                        state_sha256=np.asarray(states, dtype="U64"), frame_sha256=np.asarray(frames, dtype="U64"),
                        policy_received_frame_sha256=np.asarray(received_frames, dtype="U64"),
                        policy_valid_observation=np.ones(501, bool) if case["arm"] == "black_frame" else mask,
                        input_observation_timestamp=np.arange(len(actions), dtype=np.int32),
                        returned_observation_timestamp=np.arange(1, len(actions) + 1, dtype=np.int32),
                        input_available=mask[:len(actions)], returned_available=mask[1:len(actions) + 1])
    if boundary is not None:
        with (directory / "boundary.pkl").open("wb") as stream:
            pickle.dump(boundary, stream, protocol=5)
            stream.flush(); os.fsync(stream.fileno())
    if writer is not None and error is None:
        from helixwm_v2.evaluation import _decode_video
        try:
            video_check = _decode_video(directory / video_name, 501, 20)
            video_check["path"] = video_name
            video_check["path_semantics"] = "relative_to_case_directory"
            video_check["purpose"] = "smoke_validation_excluded_from_formal_package" if case["stage"] == "smoke" else "formal_stage_video"
            summary["video"] = video_check
        except Exception as exc:
            summary["status"] = "technical_failure"
            summary["error"] = {"type": type(exc).__name__, "message": str(exc), "operation": "video_full_decode"}
            summary["video"] = {"path": video_name, "path_semantics": "relative_to_case_directory", "full_decode_pass": False}
    atomic_json(directory / "summary.json", summary)
    if before != after:
        raise RuntimeError("Frozen model/optimizer/counters/host RNG state changed; case not accepted")
    return summary


def finalize_case(directory, final_path, identity_hash):
    directory, final_path = Path(directory), Path(final_path)
    manifest = {"identity_sha256": identity_hash,
                "files": {p.name: {"sha256": file_hash(p), "bytes": p.stat().st_size}
                          for p in sorted(directory.iterdir()) if p.is_file()}}
    atomic_json(directory / "manifest.json", manifest)
    if final_path.exists():
        raise FileExistsError("Refusing to overwrite an accepted case")
    os.replace(directory, final_path)
    return verify_case(final_path, identity_hash)


def verify_case(path, identity_hash):
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest["identity_sha256"] != identity_hash:
        raise ValueError("Case belongs to another source/config/checkpoint identity")
    actual = {p.name for p in path.iterdir() if p.is_file() and p.name != "manifest.json"}
    if actual != set(manifest["files"]):
        raise ValueError("Case file inventory differs from its manifest")
    for name, info in manifest["files"].items():
        file = path / name
        if file.stat().st_size != info["bytes"] or file_hash(file) != info["sha256"]:
            raise ValueError(f"Case file checksum mismatch: {name}")
    summary = json.loads((path / "summary.json").read_text())
    if not summary["hash_gate"]["passed"] or summary["case_id"] != path.name:
        raise ValueError("Case hash gate or identity failed")
    return {"case_id": path.name, "manifest_sha256": file_hash(path / "manifest.json"),
            "status": summary["status"], "bytes": sum(x["bytes"] for x in manifest["files"].values())}


def check_gate(path, identity, required_stage):
    if not path:
        raise ValueError(f"{required_stage} completion gate is required")
    gate = json.loads(Path(path).read_text())
    if gate.get("stage") != required_stage or gate.get("status") != "complete" or \
            gate.get("identity_sha256") != canonical_hash(identity) or not gate.get("zero_updates_verified"):
        raise ValueError(f"Invalid or stale {required_stage} completion gate")
    if required_stage == "smoke" and not gate.get("parity_checks", {}).get("passed"):
        raise ValueError("Smoke parity gates did not pass")
    if gate.get("technical_failures") != 0 or gate.get("completed_cases") != gate.get("planned_cases"):
        raise ValueError("Previous stage has incomplete or technically failed cases")


def smoke_parity(output):
    root = Path(output) / "cases"
    def trace(arm, gap):
        with np.load(root / f"smoke-seed40000-{arm}-gap{gap}" / "trace.npz", allow_pickle=False) as data:
            return {key: data[key] for key in ("actions", "reward", "observations", "state_sha256")}
    reference = trace("clean", 0)
    checks = {}
    for arm in ("v2_exact", "locked_ancestor", "posterior_return_once"):
        other = trace(arm, 0)
        checks[f"clean_{arm}"] = all(np.array_equal(reference[k], other[k]) for k in reference)
    a, b = trace("v2_exact", 10), trace("posterior_return_once", 10)
    checks["posterior_long_gap_exact"] = all(np.array_equal(a[k], b[k]) for k in a)
    video = json.loads((root / "smoke-seed40000-clean-gap0" / "summary.json").read_text()).get("video", {})
    checks["annotated_video_full_decode"] = video.get("full_decode_pass") is True and video.get("frames") == 501 and video.get("fps") == 20
    return {"passed": all(checks.values()), "checks": checks}


def _run_locked(config_path, checkpoint, upstream, output, stage, *, resume=False, smoke_gate=None, design_gate=None):
    config = json.loads(Path(config_path).read_text())
    validate_config(config)
    source = source_identity(config)
    if file_hash(checkpoint) != CHECKPOINT_SHA:
        raise ValueError("Trusted accepted checkpoint SHA256 mismatch")
    identity = {"source": source, "config_sha256": canonical_hash(config),
                "checkpoint_sha256": CHECKPOINT_SHA, "official_upstream_commit": config["identity"]["official_dreamerv3_commit"]}
    identity_hash = canonical_hash(identity)
    if stage != "smoke":
        check_gate(smoke_gate, identity, "smoke")
    if stage == "held_out":
        check_gate(design_gate, identity, "design")
    output = Path(output).resolve()
    if resume:
        if json.loads((output / "identity.json").read_text()) != identity:
            raise ValueError("Resume source/config/checkpoint identity changed")
    else:
        atomic_json(output / "identity.json", identity)
        atomic_json(output / "config.json", config)
    (output / "study.pid").write_text(str(os.getpid()) + "\n")
    # A held-out result must never silently become another stage on resume.
    jobs = case_plan(config, stage)
    atomic_json(output / "plan.json", {"stage": stage, "cases": jobs, "identity_sha256": identity_hash})
    cases_dir = output / "cases"
    partials = output / "partial_attempts"
    cases_dir.mkdir(exist_ok=True); partials.mkdir(exist_ok=True)
    registered = {j["case_id"] for j in jobs}
    if any(p.name not in registered for p in cases_dir.iterdir()):
        raise ValueError("Output contains cases outside this stage's registered plan")
    completed = {p.name: verify_case(p, identity_hash) for p in cases_dir.iterdir()}
    atomic_json(output / "ledger.json", {"identity_sha256": identity_hash, "stage": stage,
                                         "completed": completed, "next_operation": "new_case_from_registered_seed"})
    from helixwm_v2.storage import load_checkpoint, tree_hash
    from helixwm_v2.backbone import OfficialDreamerBackbone
    from helixwm.environment import WalkerEnv
    payload = load_checkpoint(checkpoint, verify_replay=False)
    if payload["counters"]["agent_steps"] != 1000000 or payload["counters"]["env_steps"] != 2000000:
        raise ValueError("Checkpoint has the wrong completed training budget")
    training_cfg = payload["config"]["backbone"]
    backbone = OfficialDreamerBackbone(training_cfg, upstream, output / "backbone_runtime")
    backbone.load(payload["backbone"])
    if tree_hash(backbone.snapshot()["training"]) != tree_hash(payload["backbone"]["training"]):
        raise ValueError("Strict reload failed to preserve official full training state")
    baseline = frozen_state(backbone)
    versions = {}
    for name in ("jax", "jaxlib", "ninjax", "numpy", "dm-control", "mujoco", "imageio", "pillow"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not_installed"
    atomic_json(output / "runtime.json", {"created_utc": utc(), "python": sys.version,
                 "packages": versions, "devices": [str(x) for x in backbone.jax.devices()],
                 "frozen_state": baseline, "checkpoint_path": str(Path(checkpoint).resolve()),
                 "training_updates_allowed": False})
    began = time.monotonic()
    resource_sample(output, None)
    try:
        for case in jobs:
            if case["case_id"] in completed:
                continue
            free = shutil.disk_usage(output).free
            if free < MIN_FREE:
                raise OSError("Less than 2 GiB free; preserving all existing data and stopping before next episode")
            if len(completed) >= 3:
                worst = max(x["bytes"] for x in completed.values())
                if worst * (len(jobs) - len(completed)) + MIN_FREE > free:
                    raise OSError("Measured compressed-case storage forecast exceeds free disk with 2 GiB reserve")
            attempt = partials / (case["case_id"] + "-" + uuid.uuid4().hex)
            video_spec = config["future_preservation"]["video_selection"]
            video = stage == "held_out" and case["arm"] == video_spec["arm"] and \
                    case["environment_seed"] == video_spec["environment_seed"] and case["missing_length"] == video_spec["missing_actions"]
            video = video or (stage == "smoke" and case["arm"] == "clean")
            summary = execute_case(backbone, config, case, attempt, WalkerEnv, make_video=video)
            completed[case["case_id"]] = finalize_case(attempt, cases_dir / case["case_id"], identity_hash)
            atomic_json(output / "ledger.json", {"identity_sha256": identity_hash, "stage": stage,
                         "completed": completed, "next_operation": "new_case_from_registered_seed"})
            append_jsonl(output / "progress.jsonl", {"utc": utc(), "completed_cases": len(completed),
                         "planned_cases": len(jobs), "case_id": case["case_id"], "case_status": summary["status"],
                         "actual_eval_agent_steps": summary["actual_eval_agent_steps"],
                         "episode_return": summary["episode_return"], "free_disk_bytes": shutil.disk_usage(output).free})
            resource_sample(output, case["case_id"])
        final = frozen_state(backbone)
        if final != baseline:
            raise RuntimeError("Study changed the frozen learner or host RNG state")
        parity = smoke_parity(output) if stage == "smoke" else {"passed": True, "evidence": str(smoke_gate)}
        failures = sum(x["status"] != "complete" for x in completed.values())
        if stage == "smoke" and (failures or not parity["passed"]):
            raise RuntimeError("Smoke contains technical failures or failed exact-parity checks")
        from helixwm_v3.analysis import analyze_study
        analysis = analyze_study(output, config, make_plots=stage != "smoke") if stage != "smoke" else None
        final = frozen_state(backbone)
        if final != baseline:
            raise RuntimeError("Post-analysis frozen model/optimizer/counters/host RNG gate failed")
        report = {"status": "complete", "stage": stage, "identity_sha256": identity_hash,
                  "completed_cases": len(completed), "planned_cases": len(jobs), "technical_failures": failures,
                  "zero_updates_verified": True, "frozen_before": baseline, "frozen_after": final,
                  "parity_checks": parity, "elapsed_this_process_s": time.monotonic() - began,
                  "analysis": analysis, "training_updates": 0, "completed_utc": utc()}
        report["storage"] = {"total_case_bytes": sum(x["bytes"] for x in completed.values()),
                             "largest_case_bytes": max(x["bytes"] for x in completed.values()),
                             "free_disk_bytes": shutil.disk_usage(output).free,
                             "reserve_bytes": MIN_FREE}
        atomic_json(output / "completion.json", report)
        return report
    except BaseException as exc:
        atomic_json(output / "interruption.json", {"utc": utc(), "stage": stage, "type": type(exc).__name__,
                     "message": str(exc), "completed_cases": len(completed), "planned_cases": len(jobs),
                     "resume_requires_explicit_flag": True, "partial_attempts_preserved": True})
        raise


def run(config_path, checkpoint, upstream, output, stage, *, resume=False, smoke_gate=None, design_gate=None):
    output = Path(output).resolve()
    if output.exists() and not resume:
        raise FileExistsError("Use a new stage output or explicit --resume; redirect console outside that directory")
    if resume and not output.is_dir():
        raise FileNotFoundError("An explicit resume requires an existing stage directory")
    output.mkdir(parents=True, exist_ok=True)
    with RunLock(output / "run.lock"):
        return _run_locked(config_path, checkpoint, upstream, output, stage, resume=resume,
                           smoke_gate=smoke_gate, design_gate=design_gate)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "checkpoint", "upstream", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--stage", required=True, choices=("smoke", "design", "held_out"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke-gate")
    parser.add_argument("--design-gate")
    args = parser.parse_args()
    report = run(args.config, args.checkpoint, args.upstream, args.output, args.stage,
                 resume=args.resume, smoke_gate=args.smoke_gate, design_gate=args.design_gate)
    print(json.dumps({key: report[key] for key in ("status", "stage", "completed_cases", "technical_failures", "zero_updates_verified")}, indent=2))


if __name__ == "__main__":
    main()
