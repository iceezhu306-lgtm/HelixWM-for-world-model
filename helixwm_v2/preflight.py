"""Actual-size GPU gates for the pinned DreamerV3/dual-chain integration.

The default command is an orchestrator without a JAX runtime. Three sequential
worker processes prepare the checkpoint and reproduce its next operation twice.
This releases GPU allocations between workers and tests process-boundary recovery.
The 500 random actions, 500 policy actions and recovery probes are preflight
data only. The policy episode verifies the formal completed-episode checkpoint
operation separately from recovery in the middle of a missing segment.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np

from helixwm.environment import WalkerEnv
from .backbone import OfficialDreamerBackbone, UPSTREAM_COMMIT, validate_upstream
from .dual_chain import DualChain
from .storage import Replay, atomic_json, load_checkpoint, save_checkpoint, tree_hash


BACKBONE_CONFIG = {"seed": 0, "platform": "cuda", "compute_dtype": "bfloat16",
                   "size": "size12m", "batch_size": 16, "batch_length": 64,
                   "burnin": 32}
ENV_CONFIG = {"seed": 0, "action_repeat": 2, "image_size": 64,
              "episode_env_steps": 1000, "camera_id": 0}
CHAIN_CONFIG = {"mode": "repair", "max_candidates": 8, "max_missing": 5,
                "score_temperature": 0.01, "score_normalization": 12288.0,
                "low_ess_threshold": 2.0, "max_events": 128}
DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs/dual_chain_walker_seed0_1m.json"


def canonical_config_sha256(config):
    """Hash the complete JSON configuration, independent of keys/whitespace."""
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _git(root, *args):
    return subprocess.run(["git", "-C", str(root), *args], check=True,
                          capture_output=True, text=True).stdout.strip()


def clean_git_identity(root):
    """Reject staged and unstaged tracked edits before identifying a release."""
    root = Path(root).resolve()
    if _git(root, "status", "--porcelain", "--untracked-files=no"):
        raise ValueError(f"Tracked source changes invalidate the preflight identity: {root}")
    return {"commit": _git(root, "rev-parse", "HEAD"),
            "tree": _git(root, "rev-parse", "HEAD^{tree}"), "tracked_clean": True}


def read_identity(upstream, config_path=DEFAULT_CONFIG):
    """Recompute the immutable code/config binding used by the formal gate.

    Absolute paths are evidence only. ``identity_sha256`` binds commit/tree,
    pinned upstream and canonical full configuration, and is portable between
    identical frozen releases. Raw config bytes are reported separately.
    """
    root = Path(__file__).resolve().parents[1]
    upstream = validate_upstream(upstream)
    source, official = clean_git_identity(root), clean_git_identity(upstream)
    config_path = Path(config_path).resolve()
    relative = config_path.relative_to(root).as_posix()
    required = {relative, "helixwm/environment.py", "helixwm_v2/preflight.py",
                "helixwm_v2/backbone.py", "helixwm_v2/dual_chain.py", "helixwm_v2/storage.py"}
    required.update(path.relative_to(root).as_posix()
                    for path in (root / "helixwm_v2").rglob("*.py"))
    # A clean tracked status alone must not approve code that is still untracked.
    _git(root, "ls-files", "--error-unmatch", "--", *sorted(required))
    raw = config_path.read_bytes()
    config = json.loads(raw)
    if config.get("backbone") != BACKBONE_CONFIG:
        raise ValueError("Formal backbone configuration differs from the tested size/batch/dtype")
    if {"seed": config.get("seed"), **config.get("environment", {})} != ENV_CONFIG:
        raise ValueError("Formal environment configuration differs from the preflight")
    if {**CHAIN_CONFIG, **config.get("dual_chain", {})} != CHAIN_CONFIG:
        raise ValueError("Formal dual-chain configuration differs from the preflight")
    if config.get("budget_agent_steps") != 1_000_000 or config.get("replay_capacity_steps") != 1_000_000:
        raise ValueError("The bound formal configuration must declare one million actions and replay capacity")
    binding = {"schema_version": 1, "helix_commit": source["commit"],
               "helix_tree": source["tree"], "helix_tracked_clean": source["tracked_clean"],
               "upstream_commit": official["commit"], "upstream_tracked_clean": official["tracked_clean"],
               "config_sha256": canonical_config_sha256(config)}
    return {**binding, "identity_sha256": canonical_config_sha256(binding),
            "config_file_sha256": hashlib.sha256(raw).hexdigest(),
            "config_relative_path": relative, "source_root": str(root), "upstream_root": str(upstream)}


def utc():
    return datetime.now(timezone.utc).isoformat()


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def finite_tree(value, path="root"):
    """Reject numerical NaN/Inf, including nonstandard bfloat16 array leaves."""
    if isinstance(value, dict):
        for name, item in value.items():
            finite_tree(item, f"{path}/{name}")
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            finite_tree(item, f"{path}/{index}")
    elif isinstance(value, (np.ndarray, np.number, float, int, bool)):
        if not np.isfinite(value).all():
            raise FloatingPointError(f"Nonfinite preflight value at {path}")


def training_batch(replay):
    batch = replay.sample(BACKBONE_CONFIG["batch_size"],
                          BACKBONE_CONFIG["batch_length"] + BACKBONE_CONFIG["burnin"])
    shape = batch["reward"].shape
    batch["consec"] = np.zeros(shape, np.int32)
    batch["stepid"] = np.zeros((*shape, 20), np.uint8)
    return batch


def collect_episode(env, replay):
    rng = np.random.default_rng(20261014)
    images = [env.reset().reshape(64, 64, 3)]
    actions, rewards, terminals = [], [0.0], [False]
    dmc_calls = 0
    for step in range(500):
        action = rng.uniform(-1.0, 1.0, (6,)).astype(np.float32)
        result = env.step(action)
        finite_tree(result)
        actions.append(action)
        images.append(result["observation"].reshape(64, 64, 3))
        rewards.append(result["reward"])
        terminals.append(result["continuation"] == 0)
        dmc_calls += result["env_steps"]
        require(result["done"] == (step == 499), "Walker episode length changed")
    require(dmc_calls == 1000, "500 actions must equal 1,000 DMC calls")
    first, last = np.zeros(501, bool), np.zeros(501, bool)
    first[0], last[-1] = True, True
    actions.append(np.zeros(6, np.float32))
    episode = {"image": np.asarray(images, np.uint8),
               "action": np.asarray(actions, np.float32),
               "reward": np.asarray(rewards, np.float32),
               "is_first": first, "is_last": last,
               "is_terminal": np.asarray(terminals, bool)}
    replay.add_episode(episode)
    require(replay.total_steps == 500, "Replay counted resets as actions")
    return {"actions": 500, "dmc_calls": dmc_calls, "frames": len(images),
            "episode_bytes": replay.disk_bytes,
            "uncompressed_episode_bytes": sum(x.nbytes for x in episode.values()),
            "raw_episode_sha256": tree_hash(episode)}, episode


def make_backend(upstream, output):
    backbone = OfficialDreamerBackbone(BACKBONE_CONFIG, upstream, output / "backbone")
    require(backbone.jax.default_backend() == "gpu", "Preflight requires the real GPU backend")
    require(backbone.config.jax.compute_dtype == "bfloat16", "BF16 configuration changed")
    require(backbone.batch_size == 16 and backbone.batch_length == 64 and backbone.burnin == 32,
            "Preflight must exercise the full formal batch and prefix")
    return backbone


def make_chain(backbone):
    return DualChain(backbone, seed=303, model_version=backbone.inference_version, **CHAIN_CONFIG)


def restore(payload, backbone, replay, env):
    backbone.load(payload["backbone"])
    replay.restore(payload["replay"])
    env.restore(payload["environment"])
    chain = make_chain(backbone)
    chain.load_state_dict(payload["dual_chain"])
    require(chain.pending is not None, "Recovery fixture lost its pending missing segment")
    return chain


def continuation(backbone, replay, env, chain):
    """One actual learner update, then the next causal actor/environment step."""
    pin_before = backbone.parameter_hash(inference=True)
    batch = training_batch(replay)
    batch_hash = tree_hash(batch)
    metrics = backbone.train(batch)
    require(bool(metrics), "Official learner returned no metrics")
    finite_tree(metrics, "metrics")
    require(backbone.parameter_hash(inference=True) == pin_before,
            "A training update mutated episode-frozen inference parameters")
    action = np.asarray(chain.act()[0], np.float32)
    finite_tree(action, "next_action")
    result = env.step(action)
    finite_tree(result, "next_environment_result")
    chain.step(result["observation"].reshape(64, 64, 3), action,
               episode_id=1, timestamp=2, valid=True,
               model_version=backbone.inference_version)
    require(chain.pending is None and chain.repaired_segment is not None,
            "The recovered missing segment did not reconnect on real evidence")
    saved = backbone.snapshot()
    finite_tree(saved["training"]["params"], "parameters_and_optimizer")
    return {"batch_sha256": batch_hash, "metrics_sha256": tree_hash(metrics),
            "training_sha256": tree_hash(saved["training"]),
            "frozen_parameters_sha256": backbone.parameter_hash(inference=True),
            "next_action_sha256": tree_hash(action),
            "environment_result_sha256": tree_hash(result),
            "environment_state_sha256": tree_hash(env.snapshot()),
            "dual_chain_sha256": tree_hash(chain.state_dict()),
            "replay_sha256": tree_hash(replay.snapshot()),
            "counters": saved["training"]["counters"],
            "inference_version": backbone.inference_version,
            "metrics": metrics}


def boundary_continuation(backbone, replay, env, chain):
    """The actual next operation after the formal episode-boundary checkpoint."""
    before = backbone.parameter_hash()
    chain.reset(env.reset().reshape(64, 64, 3), episode_id=3, timestamp=0,
                model_version=backbone.inference_version)
    action = np.asarray(chain.act()[0], np.float32)
    result = env.step(action)
    finite_tree(result, "boundary_environment_result")
    chain.step(result["observation"].reshape(64, 64, 3), action, episode_id=3,
               timestamp=1, valid=True, model_version=backbone.inference_version)
    batch = training_batch(replay)
    after = backbone.parameter_hash()
    require(before == after, "Boundary reset/inference changed learner parameters")
    return {"training_parameters_sha256": after,
            "frozen_parameters_sha256": backbone.parameter_hash(inference=True),
            "first_action_sha256": tree_hash(action), "next_batch_sha256": tree_hash(batch),
            "environment_result_sha256": tree_hash(result),
            "environment_state_sha256": tree_hash(env.snapshot()),
            "dual_chain_sha256": tree_hash(chain.state_dict()),
            "replay_sha256": tree_hash(replay.snapshot()),
            "inference_version": backbone.inference_version}


def restore_boundary(payload, backbone, replay, env):
    require(payload["next_operation"] == "reset_new_episode_and_reanchor",
            "Boundary checkpoint declares a different continuation operation")
    backbone.load(payload["backbone"])
    replay.restore(payload["replay"])
    env.restore(payload["environment"])
    chain = make_chain(backbone)
    chain.restore_completed_episode(payload["dual_chain"])
    require(not chain.initialized and chain.pending is None,
            "Completed-episode restoration retained stale latent state")
    return chain


def boundary_fixture(backbone, replay, env, chain, output):
    """Collect a real completed policy episode, then synchronize the new version."""
    images = [env.reset().reshape(64, 64, 3)]
    chain.reset(images[0], episode_id=2, timestamp=0,
                model_version=backbone.inference_version)
    actions, rewards, terminals = [], [0.0], [False]
    for step in range(1, 501):
        action = np.asarray(chain.act()[0], np.float32)
        result = env.step(action)
        finite_tree(result, "boundary_collection")
        image = result["observation"].reshape(64, 64, 3)
        chain.step(image, action, episode_id=2, timestamp=step, valid=True)
        require(result["env_steps"] == 2 and result["done"] == (step == 500),
                "Boundary fixture must complete the real 500-action episode")
        images.append(image)
        actions.append(action)
        rewards.append(result["reward"])
        terminals.append(result["continuation"] == 0)
    actions.append(np.zeros(6, np.float32))
    first, last = np.zeros(501, bool), np.zeros(501, bool)
    first[0], last[-1] = True, True
    replay.add_episode({"image": np.asarray(images, np.uint8),
        "action": np.asarray(actions, np.float32), "reward": np.asarray(rewards, np.float32),
        "is_first": first, "is_last": last, "is_terminal": np.asarray(terminals, bool)})
    completed_chain = chain.state_dict()
    backbone.pin_inference()
    require(completed_chain["state"]["model_version"] != backbone.inference_version,
            "Boundary gate did not exercise a model-version change")
    boundary = {"backbone": backbone.snapshot(), "replay": replay.snapshot(),
                "environment": env.snapshot(), "dual_chain": completed_chain,
                "counters": {"agent_steps": 1000, "env_steps": 2000, "update_steps": 4},
                "configuration": {"backbone": BACKBONE_CONFIG, "environment": ENV_CONFIG,
                                  "dual_chain": CHAIN_CONFIG},
                "next_operation": "reset_new_episode_and_reanchor",
                "scope": "preflight; 1,000 archived actions, excluding throwaway recovery probes"}
    require(boundary["backbone"]["inference_equals_training"], "Boundary model was not deduplicated")
    checkpoint = save_checkpoint(output / "CHECKPOINTS", boundary)
    loaded = load_checkpoint(checkpoint)
    require(tree_hash(boundary) == tree_hash(loaded), "Boundary checkpoint serialization changed state")
    expected = boundary_continuation(backbone, replay, env, chain)
    restored_chain = restore_boundary(loaded, backbone, replay, env)
    repeated = boundary_continuation(backbone, replay, env, restored_chain)
    require(expected == repeated, "Episode-boundary restore differs from uninterrupted reset/first action")
    atomic_json(output / "expected_boundary_continuation.json", expected)
    return checkpoint


def prepare(upstream, output, identity):
    require(not (output / "CHECKPOINTS" / "checkpoint_000000501.pkl").exists(),
            "Preflight output already contains a checkpoint; choose a fresh output directory")
    replay = Replay(output / "REPLAY", capacity_steps=1_000_000, seed=101)
    require(replay.total_steps_seen == 0, "Preflight requires a new replay archive")
    env = WalkerEnv(**ENV_CONFIG)
    try:
        collection, episode = collect_episode(env, replay)
        backbone = make_backend(upstream, output)
        initial_training = backbone.parameter_hash()
        initial_inference = backbone.parameter_hash(inference=True)
        state = backbone.observe(backbone.initial(), np.zeros(6, np.float32),
                                 episode["image"][0], is_first=True, rng=11)
        predicted = backbone.prior(state, episode["action"][0], rng=12)
        candidates = {key: np.repeat(value, 8, axis=0) for key, value in predicted.items()}
        scores = backbone.score(candidates, episode["image"][1])
        action = backbone.act(state, rng=13)
        finite_tree({"observed": state, "predicted": predicted, "scores": scores, "action": action})
        require(scores.shape == (8,) and action.shape == (1, 6), "Inference shape contract failed")
        require(tree_hash(action) == tree_hash(backbone.act(state, rng=13)),
                "Explicit actor RNG was not reproducible")
        require(backbone.parameter_hash() == initial_training and
                backbone.parameter_hash(inference=True) == initial_inference,
                "Pure inference changed model or optimizer state")
        train_metrics, update_seconds = [], []
        for _ in range(3):
            start = time.monotonic()
            metrics = backbone.train(training_batch(replay))
            update_seconds.append(time.monotonic() - start)
            require(bool(metrics), "Official training returned no metrics")
            finite_tree(metrics, "training_metrics")
            train_metrics.append(metrics)
        backbone.flush()
        require(backbone.parameter_hash() != initial_training, "Training did not change parameters")
        require(backbone.parameter_hash(inference=True) == initial_inference,
                "Training mutated the frozen episode parameter copy")
        require(int(backbone.agent.n_updates) == int(backbone.agent.n_batches) == 3,
                "Training update/seed counters diverged")

        # Create a second episode with one genuinely executed, known-missing frame.
        chain = make_chain(backbone)
        chain.reset(env.reset().reshape(64, 64, 3), episode_id=1, timestamp=0)
        issued = np.asarray(chain.act()[0], np.float32)
        result = env.step(issued)
        chain.step(None, issued, episode_id=1, timestamp=1, valid=False)
        require(chain.pending is not None, "The repair test did not create a missing segment")
        payload = {"backbone": backbone.snapshot(), "replay": replay.snapshot(),
                   "environment": env.snapshot(), "dual_chain": chain.state_dict(),
                   "counters": {"agent_steps": 501, "env_steps": 1002, "update_steps": 3},
                   "configuration": {"backbone": BACKBONE_CONFIG, "environment": ENV_CONFIG,
                                     "dual_chain": CHAIN_CONFIG},
                   "scope": "preflight fixture; 500 replay actions plus one pending episode action"}
        require(not payload["backbone"]["inference_equals_training"],
                "Mid-gap gate must preserve a distinct frozen model version")
        checkpoint = save_checkpoint(output / "CHECKPOINTS", payload)
        loaded = load_checkpoint(checkpoint)
        require(tree_hash(loaded) == tree_hash(payload), "Serialized checkpoint changed numerical state")

        chain = restore(loaded, backbone, replay, env)
        expected = continuation(backbone, replay, env, chain)
        chain = restore(loaded, backbone, replay, env)
        repeated = continuation(backbone, replay, env, chain)
        require(expected == repeated, "Same-process snapshot restore did not reproduce the next update/step exactly")

        # A malformed checkpoint must be rejected before deleting or replacing state.
        before_reject = backbone.parameter_hash()
        malformed = dict(loaded["backbone"])
        malformed["training"] = dict(malformed["training"])
        malformed["training"]["params"] = dict(malformed["training"]["params"])
        key = next(iter(malformed["training"]["params"]))
        malformed["training"]["params"].pop(key)
        rejected = False
        try:
            backbone.load(malformed)
        except ValueError:
            rejected = True
        require(rejected and backbone.parameter_hash() == before_reject,
                "Strict loading accepted malformed model parameters or altered live state")

        boundary_checkpoint = boundary_fixture(backbone, replay, env, chain, output)
        boundary_bytes = boundary_checkpoint.stat().st_size

        params = loaded["backbone"]["training"]["params"]
        dtypes = {}
        for value in params.values():
            dtype = str(value.dtype)
            dtypes[dtype] = dtypes.get(dtype, 0) + int(value.size)
        measured_replay = int(np.ceil(collection["episode_bytes"] / 500 * 1_000_000))
        replay_bound = int(np.ceil(collection["uncompressed_episode_bytes"] / 500 * 1_000_000))
        forecast = {"checkpoint_count": 100, "checkpoint_bytes_each": boundary_bytes,
                    "checkpoint_total_bytes": 100 * boundary_bytes,
                    "replay_measured_compression_estimate_bytes": measured_replay,
                    "replay_forecast_with_50_percent_margin_bytes": int(np.ceil(measured_replay * 1.5)),
                    "replay_uncompressed_bound_bytes": replay_bound,
                    "headroom_bytes": 4 * 1024**3,
                    "estimate_is_not_a_guarantee": True}
        forecast["required_estimate_bytes"] = (forecast["checkpoint_total_bytes"] +
            forecast["replay_forecast_with_50_percent_margin_bytes"] + forecast["headroom_bytes"])
        forecast["required_uncompressed_bound_bytes"] = (forecast["checkpoint_total_bytes"] +
            replay_bound + forecast["headroom_bytes"])
        forecast["available_bytes_after_preflight_fixture"] = shutil.disk_usage(output).free
        report = {"passed": True, "utc": utc(), "upstream_commit": UPSTREAM_COMMIT,
                  "identity": identity,
                  "backbone_config": BACKBONE_CONFIG, "resolved_backbone_config": backbone.resolved_config,
                  "collection": collection, "training_batches": 3, "train_metrics": train_metrics,
                  "update_seconds_including_first_compile": update_seconds,
                  "mid_gap_checkpoint": str(checkpoint), "checkpoint_bytes": boundary_bytes,
                  "episode_boundary_checkpoint": str(boundary_checkpoint),
                  "additional_boundary_policy_actions": 500,
                  "mid_gap_checkpoint_bytes": checkpoint.stat().st_size,
                  "total_trainable_model_parameters": int(sum(v.size for k, v in params.items()
                      if k.startswith(("enc/", "dyn/", "dec/", "rew/", "con/", "pol/", "val/")))),
                  "total_model_parameter_and_running_stat_elements": int(sum(v.size for k, v in params.items()
                                                                                 if not k.startswith("opt/"))),
                  "total_parameter_and_optimizer_elements": int(sum(v.size for v in params.values())),
                  "inference_bytes": int(sum(v.nbytes for v in loaded["backbone"]["inference_params"].values())),
                  "parameter_dtype_elements": dtypes, "compute_dtype": "bfloat16",
                  "devices": [{"platform": d.platform, "kind": d.device_kind, "id": d.id}
                              for d in backbone.jax.devices()],
                  "packages": {name: importlib.metadata.version(name) for name in
                               ("jax", "jaxlib", "numpy", "dm-control", "mujoco", "ninjax", "elements")},
                  "disk_forecast": forecast,
                  "gates": {"real_500_action_episode": True, "actual_size_batch_training": True,
                            "finite_metrics_and_parameters": True, "pure_inference": True,
                            "frozen_episode_parameters_survive_training": True,
                            "strict_checkpoint_roundtrip": True, "malformed_checkpoint_rejected": True,
                            "same_process_exact_next_update_and_environment": True,
                            "pending_missing_segment_roundtrip": True,
                            "formal_boundary_next_reset_and_action": True}}
        atomic_json(output / "expected_continuation.json", expected)
        atomic_json(output / "preparation.json", report)
    finally:
        env.close()


def recovery_branch(upstream, output, name, identity):
    prepared = json.loads((output / "preparation.json").read_text())
    require(prepared["identity"] == identity, "Recovery worker code/config differs from preparation")
    checkpoint = output / "CHECKPOINTS" / "checkpoint_000000501.pkl"
    payload = load_checkpoint(checkpoint)
    backbone = make_backend(upstream, output)
    replay = Replay(output / "REPLAY", capacity_steps=1_000_000, seed=101)
    env = WalkerEnv(**ENV_CONFIG)
    try:
        chain = restore(payload, backbone, replay, env)
        result = continuation(backbone, replay, env, chain)
        expected = json.loads((output / "expected_continuation.json").read_text())
        require(result == expected, "Fresh-process next update/action/environment differs from original branch")
        boundary = load_checkpoint(output / "CHECKPOINTS" / "checkpoint_000001000.pkl")
        chain = restore_boundary(boundary, backbone, replay, env)
        boundary_result = boundary_continuation(backbone, replay, env, chain)
        expected_boundary = json.loads((output / "expected_boundary_continuation.json").read_text())
        require(boundary_result == expected_boundary,
                "Fresh-process formal boundary reset/action differs from uninterrupted execution")
        atomic_json(output / f"recovery_{name}.json", {"passed": True, "branch": name, "identity": identity,
                    "utc": utc(), "continuation": result, "boundary_continuation": boundary_result})
    finally:
        env.close()


def orchestrate(upstream, output, config_path):
    summary = {"passed": False, "started_utc": utc(), "upstream_commit": UPSTREAM_COMMIT,
               "backbone_config": BACKBONE_CONFIG, "experiment_class": "preflight_only"}
    try:
        identity = read_identity(upstream, config_path)
        summary["identity"] = identity
        for phase in ("prepare", "a", "b"):
            command = [sys.executable, "-m", "helixwm_v2.preflight", "--upstream", str(upstream),
                       "--output", str(output), "--config", str(config_path), "--worker", phase]
            print(json.dumps({"event": "preflight_worker_start", "phase": phase, "utc": utc()}), flush=True)
            with (output / f"worker_{phase}.log").open("w", encoding="utf-8") as stream:
                result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, timeout=2400)
            require(result.returncode == 0, f"Preflight {phase} failed; inspect worker_{phase}.log")
            require(read_identity(upstream, config_path) == identity,
                    "Source or formal configuration changed during preflight")
        prepared = json.loads((output / "preparation.json").read_text())
        left = json.loads((output / "recovery_a.json").read_text())
        right = json.loads((output / "recovery_b.json").read_text())
        require(left["passed"] and right["passed"] and left["continuation"] == right["continuation"],
                "Independent process recovery branches disagree")
        require(left["boundary_continuation"] == right["boundary_continuation"],
                "Independent process episode-boundary recoveries disagree")
        require(prepared["identity"] == left["identity"] == right["identity"] == identity,
                "Worker code/config identities differ")
        summary.update(prepared)
        summary["gates"]["two_fresh_process_exact_recoveries"] = True
        summary["gates"]["two_fresh_process_formal_boundary_recoveries"] = True
        summary["gates"]["unchanged_pinned_source_and_formal_config"] = True
        forecast = summary["disk_forecast"]
        forecast["available_bytes_after_all_workers"] = shutil.disk_usage(output).free
        sufficient = forecast["available_bytes_after_all_workers"] >= forecast["required_estimate_bytes"]
        summary["gates"]["disk_forecast_with_measured_margin"] = sufficient
        summary["passed"] = bool(sufficient)
        if not sufficient:
            summary["blocking_reason"] = "Insufficient free disk for 100 checkpoints, forecast replay and 4 GiB headroom"
        summary["recovery_scope"] = ("Exact tested next batch/update/action and MuJoCo step on this pinned host; "
                                     "not a proof of cross-hardware or arbitrarily long trajectory equivalence")
    except BaseException as exc:
        summary.update(passed=False, error=str(exc), traceback=traceback.format_exc())
    summary["finished_utc"] = utc()
    atomic_json(output / "preflight.json", summary)
    print(json.dumps({"passed": summary["passed"], "report": str(output / "preflight.json")}), flush=True)
    return 0 if summary["passed"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                        help="Tracked formal JSON configuration bound to this gate")
    parser.add_argument("--worker", choices=("prepare", "a", "b"), help=argparse.SUPPRESS)
    parser.add_argument("--recovery-branch", choices=("a", "b"), help="Run one fresh-process recovery worker")
    args = parser.parse_args()
    output, upstream, config_path = args.output.resolve(), args.upstream.resolve(), args.config.resolve()
    output.mkdir(parents=True, exist_ok=True)
    phase = args.recovery_branch or args.worker
    identity = read_identity(upstream, config_path) if phase else None
    if phase == "prepare":
        prepare(upstream, output, identity)
    elif phase in ("a", "b"):
        recovery_branch(upstream, output, phase, identity)
    else:
        return orchestrate(upstream, output, config_path)
    require(read_identity(upstream, config_path) == identity,
            "Worker source or formal configuration changed while executing")
    return 0


if __name__ == "__main__":
    sys.exit(main())
