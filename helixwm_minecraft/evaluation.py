"""Sequential Minecraft frozen evaluations with actual interaction accounting.

The caller must strictly reload the final checkpoint independently. This module
does not train, start another engine concurrently, or claim engine snapshot
recovery. The saved masks are exogenous full-horizon schedules; early game
termination only truncates their used prefix.
"""
from __future__ import annotations

import copy
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import pickle
import random
import re
import time
import traceback
import uuid

import numpy as np

from helixwm_v2.dual_chain import DualChain, _digest
from helixwm_v2.evaluation import DropoutSchedule, _decode_video
from helixwm_v3.variants import ARMS, make_chain


FINAL_GAPS = (1, 5, 10, 20, 50)


def final_protocol(config):
    """Locked six-arm comparison; these are inference cases, not new training."""
    expected = {"final_arms": list(ARMS), "final_gap_lengths": list(FINAL_GAPS),
                "final_gap_start": 10, "eval_seed_start": 10000,
                "final_video": {"arm": "v2_exact", "seed": 10000, "gap_length": 20}}
    for key, value in expected.items():
        if config.get(key, value) != value:
            raise ValueError("Minecraft final protocol mismatch: " + key)
    return expected


def final_case_specs(config):
    protocol = final_protocol(config)
    return [{"arm": arm, "gap_length": length, "seed": seed,
             "gap_start": protocol["final_gap_start"] if length else None}
            for arm in ARMS for length in ((0,) if arm == "clean" else FINAL_GAPS)
            for seed in range(10000, 10020)]


def _json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _append(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()


def _host_rng_hash():
    return hashlib.sha256(pickle.dumps((random.getstate(), np.random.get_state()), protocol=5)).hexdigest()


def _frozen(backbone):
    return {"model_optimizer_sha256": backbone.parameter_hash(),
            "inference_sha256": backbone.parameter_hash(inference=True),
            "inference_version": int(backbone.inference_version),
            "counters": {key: int(getattr(backbone.agent, key))
                         for key in ("n_updates", "n_batches", "n_actions")},
            "host_python_numpy_rng_sha256": _host_rng_hash()}


def _rgb(value, size):
    array = np.asarray(value)
    if array.dtype != np.uint8 or array.size != size * size * 3:
        raise ValueError("Minecraft must supply actual uint8 RGB observations")
    return array.reshape(size, size, 3)


def _finite_state(chain):
    if not all(np.isfinite(x).all() for x in chain.real.values()):
        raise FloatingPointError("Nonfinite inferred Minecraft state")


def _episode_diagnostics(path, seed, ticks, episode_return):
    """Read privileged outcome evidence only after control has ended."""
    if not path.exists():
        return {"available": False, "success": None, "reason": "diagnostic_file_missing"}
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    resets = [row for row in rows if row["event"] == "reset"]
    native = [row for row in rows if row["event"] == "native_step"]
    if len(resets) != 1 or len(native) != ticks or any(row["episode_seed"] != seed for row in rows) or \
            [row["episode_ticks"] for row in native] != list(range(1, ticks + 1)):
        raise ValueError("Privileged outcome sidecar does not match acknowledged episode ticks/seed")
    for row in resets + native:
        data = row["privileged"]
        if not np.isfinite([*data["position"], data["goal_distance"], data["path_length"]]).all():
            raise ValueError("Nonfinite privileged outcome diagnostics")
    if not np.isclose(sum(row["reward"] for row in native), episode_return, rtol=1e-7, atol=1e-6):
        raise ValueError("Native reward sidecar differs from acknowledged return")
    successful = [row for row in native if row["privileged"]["success"]]
    if len(successful) > 1 or any(row["privileged"]["native_reward"] < 100 for row in successful):
        raise ValueError("Invalid native goal-touch success evidence")
    last = native[-1]["privileged"] if native else resets[0]["privileged"]
    return {"available": True, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "success": bool(successful), "success_native_step": successful[0]["episode_ticks"] if successful else None,
            "initial_goal_distance": resets[0]["privileged"]["goal_distance"],
            "final_goal_distance": last["goal_distance"], "path_length": last["path_length"],
            "native_step_rows": len(native), "policy_access": False,
            "consumed_only_after_episode_end": True}


def _video_frame(env, *, kind, condition, seed, timestamp, available, chain):
    from PIL import Image, ImageDraw
    frame = Image.fromarray(_rgb(env.render(256), 256))
    draw = ImageDraw.Draw(frame)
    draw.rectangle((0, 0, 256, 54), fill=(0, 0, 0))
    draw.text((4, 3), ("SMOKE | " if kind == "smoke" else "") + f"{condition} | seed {seed}", fill="white")
    draw.text((4, 20), f"observation t={timestamp} | " + ("available" if available else "MISSING"), fill="white")
    mode = "prior fallback" if chain.abstaining else "candidate repair" if chain.pending is not None else "observed state"
    draw.text((4, 37), f"raw game view | control: {mode}", fill="white")
    return np.asarray(frame)


def _csv_export(root):
    source = root / "EVAL" / "episodes.jsonl"
    if not source.exists():
        return None
    rows = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    output = source.with_suffix(".csv")
    temporary = output.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=sorted(set().union(*(row.keys() for row in rows))))
        writer.writeheader(); writer.writerows(rows)
    temporary.replace(output)
    return output.relative_to(root).as_posix()


def plot_learning_curve(run_dir, smoke=False):
    root = Path(run_dir)
    reports = [json.loads(p.read_text()) for p in (root / "EVAL" / "reports").glob("*.json")]
    accepted = {r["evaluation_id"] for r in reports
                if r.get("status") == "complete" and r.get("hash_gate", {}).get("passed")}
    path = root / "EVAL" / "episodes.jsonl"
    if not path.exists():
        return {"available": False, "reason": "no_episode_rows"}
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    groups = {}
    for row in rows:
        if row["evaluation_id"] in accepted and row["status"] == "complete" and (row["kind"] == "smoke") == smoke:
            key = (row["condition"], row["agent_steps"], row["evaluation_id"])
            groups.setdefault(key, []).append(row["episode_return"])
    if not groups:
        return {"available": False, "reason": "no_passed_evaluation_of_requested_kind", "smoke": smoke}
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 5.4), constrained_layout=True)
    for name, color in (("clean", "#1766ad"), ("dropout", "#d97706")):
        selected = sorted((step, ident, float(np.mean(values))) for (condition, step, ident), values in groups.items() if condition == name)
        if selected:
            ax.plot([x[0] for x in selected], [x[2] for x in selected], "o-", color=color,
                    label="Clean observations" if name == "clean" else "Known missing observations + V2 repair")
    ax.set(xlabel="Training action decisions", ylabel="Mean undiscounted game episode return",
           title=("SMOKE ONLY | " if smoke else "") + "HelixWM Minecraft | one training seed | frozen evaluation")
    ax.grid(alpha=.25); ax.legend()
    fig.text(.5, -.025, "Unsmoothed means. Actual early endings retained. Evaluation seeds are not training seeds.",
             ha="center", fontsize=8)
    destination = root / ("SMOKE_ARTIFACTS" if smoke else "PLOTS")
    destination.mkdir(parents=True, exist_ok=True)
    files = []
    for suffix in ("png", "pdf"):
        output = destination / ("learning_curve." + suffix)
        fig.savefig(output, dpi=180, bbox_inches="tight")
        files.append(output.relative_to(root).as_posix())
    plt.close(fig)
    return {"available": True, "files": files, "smoothing": "none", "training_seed_count": 1,
            "smoke_only": smoke, "formal_performance_evidence": not smoke}


def plot_final_comparison(root, report):
    """Descriptive fixed-checkpoint returns; no Walker failure criterion reused."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 5.4), constrained_layout=True)
    clean = report["conditions"]["clean"]["mean_return"]
    ax.axhline(clean, color="black", linestyle="--", label="Clean (20 fixed seeds)")
    for arm in ARMS[1:]:
        groups = report["conditions"][arm]["by_gap_length"]
        ax.plot(FINAL_GAPS, [groups[str(g)]["mean_return"] for g in FINAL_GAPS], "o-", label=arm)
    ax.set(xlabel="Scheduled missing observations (action-decision timestamps)",
           ylabel="Mean undiscounted game episode return",
           title="Minecraft | fixed checkpoint | 20 environment seeds per condition")
    ax.grid(alpha=.25); ax.legend(fontsize=8)
    fig.text(.5, -.03, "All early endings retained, including episodes ending before the fault. One training seed.",
             ha="center", fontsize=8)
    files = []
    for suffix in ("png", "pdf"):
        path = root / "PLOTS" / ("final_comparison." + suffix)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=180, bbox_inches="tight"); files.append(path.relative_to(root).as_posix())
    plt.close(fig)
    return {"files": files, "early_ending_selection": "none", "statistical_claim": "descriptive_only"}


def evaluate(backbone, config, run_dir, agent_steps, kind="periodic", episodes=None,
             video=None, env_factory=None, training_env_steps=None):
    """Evaluate periodic clean/dropout or final six arms and restore the pin.

    ``training_env_steps`` counts acknowledged native synchronous calls from the
    trainer, never inferred as twice actions. Final evaluation requires twenty
    environment seeds per condition. ``kind='smoke'`` allows a smaller explicit
    episode count and produces separate, explicitly labeled smoke artifacts.
    """
    if not re.fullmatch(r"[A-Za-z0-9_-]+", kind) or kind not in {"smoke", "periodic", "final"}:
        raise ValueError("Evaluation kind must be smoke, periodic, or final")
    if type(agent_steps) is not int or agent_steps < 0:
        raise ValueError("Nonnegative integer action counter required")
    if training_env_steps is not None and (type(training_env_steps) is not int or training_env_steps < 0):
        raise ValueError("Acknowledged native-step counter must be a nonnegative integer")
    if training_env_steps is not None and not agent_steps <= training_env_steps <= 2 * agent_steps:
        raise ValueError("Acknowledged native steps must lie within one/two per action")
    episodes = (config.get("final_eval_episodes", 20) if kind == "final" else config.get("periodic_eval_episodes", 5)) if episodes is None else episodes
    if type(episodes) is not int or episodes < 1 or (kind == "final" and episodes != 20):
        raise ValueError("Final evaluation needs 20 fixed seeds per condition")
    video = (kind in {"final", "smoke"}) if video is None else bool(video)
    if video and kind not in {"final", "smoke"}:
        raise ValueError("Video is permitted only for final acceptance or isolated smoke validation")
    if kind == "final" and not video:
        raise ValueError("Final acceptance requires the single predeclared clean video")
    if kind == "final" and training_env_steps is None:
        raise ValueError("Final report requires the actual acknowledged native-step counter")
    env_config = copy.deepcopy(config["environment"])
    repeat, size = int(env_config["action_repeat"]), int(env_config["image_size"])
    max_ticks = int(env_config["episode_env_steps"])
    if repeat != 2 or size != 64 or max_ticks < 1:
        raise ValueError("Registered Minecraft interface is RGB64 with action repeat 2")
    horizon = math.ceil(max_ticks / repeat)
    protocol = final_protocol(config) if kind == "final" else None
    if protocol and horizon <= protocol["final_gap_start"] + max(FINAL_GAPS):
        raise ValueError("Final horizon must leave a valid observation after every scheduled gap")
    require_live_diagnostics = env_factory is None
    if env_factory is None:
        from .bridge import MinecraftEnv
        env_factory = MinecraftEnv
    root = Path(run_dir)
    folder = root / "EVAL"
    folder.mkdir(parents=True, exist_ok=True)
    identifier = f"{kind}_{agent_steps:09d}_{uuid.uuid4().hex[:12]}"
    video_path = (root / "SMOKE_ARTIFACTS" / identifier / "smoke_clean_validation.mp4") if kind == "smoke" else \
        root / "VIDEOS" / "v2_exact_seed10000_gap20_final.mp4"
    if video and video_path.exists():
        raise FileExistsError("Do not overwrite a previous formal video or failed attempt")
    report_path = folder / "reports" / (identifier + ".json")
    report = {"evaluation_id": identifier, "kind": kind, "agent_steps": agent_steps,
              "training_env_steps": training_env_steps, "training_env_step_unit": "acknowledged_MineRL_synchronous_step",
              "nominal_seconds_per_native_step": 0.05, "total_engine_world_ticks_measured": False,
              "reset_warmup_steps_in_budget": False,
              "training_seed": int(config.get("seed", 0)), "status": "running",
              "created_utc": datetime.now(timezone.utc).isoformat(), "conditions": {},
              "checkpoint_reload_checked_here": False, "training_updates_allowed": False,
              "evaluation_interactions_in_training_budget": False,
              "conditions_are_same_policy_inference_interventions": True,
              "engine_snapshot_recovery_supported": False,
              "mask_protocol": "fixed_single_gap_at_observation_10" if protocol else "independent_full_horizon_schedule_truncated_at_actual_termination",
              "actor_sampling": "original_stochastic_actor_private_SeedSequence_stream_1",
              "seed_uncertainty": "evaluation_episodes_of_one_training_seed"}
    report["artifact_scope"] = "SMOKE_ONLY_EXCLUDE_FROM_FORMAL_PACKAGE" if kind == "smoke" else "formal_run_evaluation"
    if protocol:
        report["final_protocol"] = protocol
        report["expected_cases"] = 520
        report["early_termination_policy"] = "retain_all_cases_report_exposure_no_selection_or_rerun"
        report["pairing"] = "same_exogenous_environment_and_chain_seeds_not_identical_divergent_trajectories"
        report["interpretation"] = "single_training_seed_development_evaluation_not_untouched_confirmatory_seeds"
    previous_params, previous_version = backbone.inference_params, backbone.inference_version
    previous_inference_hash = backbone.parameter_hash(inference=True)
    start_host_rng = _host_rng_hash()
    before, after, caught = None, None, None
    started = time.monotonic()
    all_rows = []
    try:
        backbone.pin_inference()
        before = _frozen(backbone)
        for condition in (ARMS if protocol else ("clean", "dropout")):
            rows = []
            specs = [case for case in final_case_specs(config) if case["arm"] == condition] if protocol else [
                {"seed": int(config.get("eval_seed_start", 10000)) + index, "gap_length": None, "gap_start": None}
                for index in range(episodes)]
            for index, case in enumerate(specs):
                seed, gap_length, gap_start = case["seed"], case["gap_length"], case["gap_start"]
                case_id = f"{identifier}-{condition}-gap{gap_length}-seed{seed}"
                trace_dir = folder / "traces" / case_id
                trace_dir.mkdir(parents=True, exist_ok=False)
                if protocol:
                    mask = np.ones(horizon + 1, bool)
                    if gap_length:
                        mask[gap_start:gap_start + gap_length] = False
                else:
                    schedule = DropoutSchedule(100000 + seed, config)
                    mask = np.asarray([schedule.valid(is_first=True)] +
                                      [schedule.valid(is_last=(t == horizon)) for t in range(1, horizon + 1)], bool)
                if condition == "clean":
                    mask[:] = True
                chain_args = copy.deepcopy(config.get("dual_chain", {})); chain_args.pop("mode", None)
                chain_args.setdefault("score_normalization", 12288.0)
                chain = make_chain(backbone, condition, seed=200000 + seed, model_version=backbone.inference_version,
                                   candidate_config=chain_args) if protocol else \
                    DualChain(backbone, seed=200000 + seed, model_version=backbone.inference_version,
                              mode="repair", **chain_args)
                env = writer = None
                error = None
                actions, rewards, ticks, valid_used, frame_hashes, state_hashes = [], [], [], [True], [], []
                received_hashes = []
                terminated = truncated = False
                frame_count, actual_steps, actual_ticks = 0, 0, 0
                counters_complete = True
                try:
                    case_env_config = copy.deepcopy(env_config)
                    case_env_config.setdefault("diagnostics_path", str(trace_dir / "engine_diagnostics.jsonl"))
                    case_env_config.setdefault("failure_log_path", str(trace_dir / "engine_failures.jsonl"))
                    env = env_factory(seed=seed, **case_env_config)
                    raw = _rgb(env.reset(seed=seed), size)
                    chain.reset(raw, episode_id=case_id, model_version=backbone.inference_version)
                    _finite_state(chain)
                    frame_hashes.append(hashlib.sha256(raw.tobytes()).hexdigest())
                    received_hashes.append(frame_hashes[-1])
                    state_hashes.append(_digest(chain.real))
                    selected_video = (condition == "v2_exact" and seed == 10000 and gap_length == 20) if protocol else \
                        (condition == "clean" and index == 0)
                    if video and selected_video:
                        import imageio.v2 as imageio
                        video_path.parent.mkdir(parents=True, exist_ok=True)
                        writer = imageio.get_writer(str(video_path), fps=float(config.get("video_fps", 20)),
                                                    codec="libx264", quality=7, macro_block_size=16)
                        writer.append_data(_video_frame(env, kind=kind, condition=condition, seed=seed,
                                                       timestamp=0, available=True, chain=chain)); frame_count = 1
                    for t in range(horizon):
                        action = chain.act(deterministic=False)
                        try:
                            result = env.step(action[0])
                        except Exception:
                            counters_complete = False
                            raise
                        actual_steps += 1
                        step_ticks = int(result["env_steps"])
                        actual_ticks += step_ticks
                        reward = float(result["reward"])
                        if not 1 <= step_ticks <= repeat or not np.isfinite(reward) or actual_ticks > max_ticks:
                            raise ValueError("Invalid acknowledged native-step/reward accounting")
                        done = bool(result["done"])
                        if step_ticks != repeat and not done:
                            raise ValueError("Partial action repetition must coincide with an ending")
                        terminated, truncated = bool(result["terminated"]), bool(result["truncated"])
                        if done != (terminated or truncated):
                            raise ValueError("Minecraft done/terminated/truncated disagree")
                        raw = _rgb(result["observation"], size)
                        actions.append(action[0].copy()); rewards.append(reward); ticks.append(step_ticks)
                        available = bool(mask[t + 1]); valid_used.append(available)
                        frame_hashes.append(hashlib.sha256(raw.tobytes()).hexdigest())
                        received = (raw if available else np.zeros_like(raw)) if condition == "black_frame" else (raw if available else None)
                        policy_valid = True if condition == "black_frame" else available
                        received_hashes.append(hashlib.sha256(received.tobytes()).hexdigest() if received is not None else "")
                        chain.step(received, action, episode_id=case_id,
                                   timestamp=t + 1, model_version=backbone.inference_version, valid=policy_valid)
                        _finite_state(chain)
                        state_hashes.append(_digest(chain.real))
                        for event in chain.pop_events():
                            _append(folder / "events.jsonl", {"evaluation_id": identifier, "case_id": case_id, **event})
                        if writer is not None:
                            writer.append_data(_video_frame(env, kind=kind, condition=condition, seed=seed,
                                                           timestamp=t + 1, available=available, chain=chain)); frame_count += 1
                        if done:
                            break
                    else:
                        raise RuntimeError("Game did not terminate or truncate within the registered tick horizon")
                except Exception as exc:
                    error = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
                finally:
                    for operation, obj in (("video_close", writer), ("environment_close", env)):
                        if obj is not None:
                            try:
                                obj.close()
                                if getattr(obj, "close_error", None):
                                    raise RuntimeError("Owned engine cleanup failed: " + str(obj.close_error))
                            except Exception as exc:
                                if error is None:
                                    error = {"type": type(exc).__name__, "message": str(exc), "operation": operation}
                                else:
                                    error.setdefault("cleanup_errors", []).append({"operation": operation, "message": str(exc)})
                # Raw traces remain available even if the engine or codec fails.
                np.savez_compressed(trace_dir / "trace.npz", actions=np.asarray(actions, np.float32).reshape(-1, 6),
                                    reward=np.asarray(rewards, np.float64), env_steps=np.asarray(ticks, np.int16),
                                    action_index=np.arange(len(rewards), dtype=np.int32),
                                    full_availability=mask, used_availability=np.asarray(valid_used, bool),
                                    input_available=mask[:len(actions)], returned_available=mask[1:len(actions) + 1],
                                    input_observation_timestamp=np.arange(len(actions), dtype=np.int32),
                                    returned_observation_timestamp=np.arange(1, len(actions) + 1, dtype=np.int32),
                                    frame_sha256=np.asarray(frame_hashes, dtype="U64"),
                                    policy_received_frame_sha256=np.asarray(received_hashes, dtype="U64"),
                                    state_sha256=np.asarray(state_hashes, dtype="U64"))
                diagnostics = {"available": False, "success": None, "reason": "technical_case_failure"}
                if error is None:
                    try:
                        diagnostics = _episode_diagnostics(trace_dir / "engine_diagnostics.jsonl", seed, actual_ticks, sum(rewards))
                        if require_live_diagnostics and not diagnostics["available"]:
                            raise ValueError("Real Minecraft evaluation requires its native outcome sidecar")
                    except Exception as exc:
                        error = {"type": type(exc).__name__, "message": str(exc), "operation": "outcome_diagnostics"}
                row = {"evaluation_id": identifier, "case_id": case_id, "kind": kind, "condition": condition,
                       "agent_steps": agent_steps, "training_env_steps": training_env_steps,
                       "training_seed": int(config.get("seed", 0)), "evaluation_seed": seed,
                       "chain_root_seed": 200000 + seed, "dropout_seed": 100000 + seed,
                       "scheduled_gap_length": gap_length, "scheduled_gap_start": gap_start,
                       "missing_input_actions": int(np.count_nonzero(~mask[:len(actions)])),
                       "fault_exposed": bool(np.any(~mask[:len(actions)])),
                       "early_ending_before_fault": bool(protocol and gap_length and actual_steps <= gap_start),
                       "received_valid_return_after_gap": bool(protocol and gap_length and actual_steps >= gap_start + gap_length),
                       "acted_after_valid_return": bool(protocol and gap_length and actual_steps > gap_start + gap_length),
                       "episode_return": float(sum(rewards)), "episode_length": actual_steps,
                       "actual_eval_agent_steps": actual_steps, "actual_eval_env_steps": actual_ticks,
                       "actual_interaction_counters_complete": counters_complete,
                       "valid_reward_rows": len(rewards), "terminated": terminated, "truncated": truncated,
                       "goal_success": diagnostics["success"], "outcome_diagnostics": diagnostics,
                       "missing_observations": sum(not x for x in valid_used),
                       "availability_sha256": hashlib.sha256(mask.tobytes()).hexdigest(),
                       "used_availability_sha256": hashlib.sha256(bytes(valid_used)).hexdigest(),
                       "status": "technical_failure" if error else "complete", "error": error,
                       "trace": (trace_dir / "trace.npz").relative_to(root).as_posix()}
                if writer is not None and error is None:
                    try:
                        proof = _decode_video(video_path, frame_count, float(config.get("video_fps", 20)))
                        proof["path"] = video_path.relative_to(root).as_posix()
                        proof["path_semantics"] = "relative_to_run_directory"
                        proof["frames_expected"] = actual_steps + 1
                        proof["condition"] = condition
                        proof["evaluation_seed"] = seed
                        proof["gap_length"] = gap_length
                        proof["fault_exposed"] = row["fault_exposed"]
                        proof["rendering"] = "raw_game_view_with_policy_availability_and_control_mode_overlay"
                        report["video"] = proof
                    except Exception as exc:
                        error = {"type": type(exc).__name__, "message": str(exc), "operation": "video_full_decode"}
                        row.update(status="technical_failure", error=error)
                _json(trace_dir / "summary.json", row)
                _append(folder / "episodes.jsonl", row)
                rows.append(row); all_rows.append(row)
                if error:
                    raise RuntimeError(f"Minecraft evaluation case failed: {case_id}: {error['message']}")
            returns = np.asarray([row["episode_return"] for row in rows], np.float64)
            report["conditions"][condition] = {"episodes": len(rows), "mean_return": float(returns.mean()),
                "std_return_population": float(returns.std()), "raw_episode_returns": returns.tolist(),
                "evaluation_seeds": [row["evaluation_seed"] for row in rows],
                "actual_eval_agent_steps": sum(row["actual_eval_agent_steps"] for row in rows),
                "actual_eval_env_steps": sum(row["actual_eval_env_steps"] for row in rows),
                "terminated_episodes": sum(row["terminated"] for row in rows),
                "truncated_episodes": sum(row["truncated"] for row in rows),
                "missing_observations": sum(row["missing_observations"] for row in rows)}
            if protocol:
                groups = {}
                for length in ((0,) if condition == "clean" else FINAL_GAPS):
                    subset = [row for row in rows if row["scheduled_gap_length"] == length]
                    groups[str(length)] = {"episodes": len(subset),
                        "mean_return": float(np.mean([row["episode_return"] for row in subset])),
                        "raw_episode_returns": [row["episode_return"] for row in subset],
                        "evaluation_seeds": [row["evaluation_seed"] for row in subset],
                        "fault_exposed_episodes": sum(row["fault_exposed"] for row in subset),
                        "early_ending_before_fault_episodes": sum(row["early_ending_before_fault"] for row in subset),
                        "acted_after_valid_return_episodes": sum(row["acted_after_valid_return"] for row in subset)}
                    success_count = sum(row["goal_success"] is True for row in subset)
                    unknown = sum(row["goal_success"] is None for row in subset)
                    groups[str(length)].update(goal_success_count=success_count, unknown_goal_outcomes=unknown,
                        goal_success_rate=None if unknown else success_count / len(subset))
                report["conditions"][condition]["by_gap_length"] = groups
        report["actual_case_count"] = len(all_rows)
        if protocol and report["actual_case_count"] != 520:
            raise RuntimeError("Final six-arm matrix is incomplete")
        report["status"] = "complete"
    except BaseException as exc:
        caught = exc
        report.update(status="failed", error={"type": type(exc).__name__, "message": str(exc)})
    finally:
        try:
            after = _frozen(backbone)
            passed = before is not None and before == after and start_host_rng == _host_rng_hash()
            report["hash_gate"] = {"passed": passed, "before": before, "after": after,
                                   "scope": "full_model_optimizer_normalizer_target_counters_pinned_inference_host_rng"}
            if not passed:
                report["status"] = "failed"
                caught = caught or RuntimeError("Minecraft evaluation changed frozen state")
        except BaseException as exc:
            report["status"] = "failed"
            report["hash_gate"] = {"passed": False, "error": str(exc)}
            caught = caught or exc
        finally:
            backbone.inference_params, backbone.inference_version = previous_params, previous_version
        restored = backbone.parameter_hash(inference=True) == previous_inference_hash and backbone.inference_version == previous_version
        report["collector_inference_restored"] = restored
        if not restored:
            report["status"] = "failed"
            caught = caught or RuntimeError("Collector's pinned inference state was not restored")
        report["wall_time_s"] = time.monotonic() - started
        report["actual_eval_agent_steps"] = sum(row["actual_eval_agent_steps"] for row in all_rows)
        report["actual_eval_env_steps"] = sum(row["actual_eval_env_steps"] for row in all_rows)
        _json(report_path, report)
        try:
            report["raw_csv"] = _csv_export(root)
            if report["status"] == "complete":
                report["learning_curve"] = plot_learning_curve(root, smoke=(kind == "smoke"))
                if protocol:
                    report["final_comparison"] = plot_final_comparison(root, report)
        except Exception as exc:
            report["status"] = "failed"
            report["artifact_error"] = str(exc)
            caught = caught or exc
        # Artifact generation also must not perturb the next training update.
        post = _frozen(backbone)
        post_ok = before is not None and post["model_optimizer_sha256"] == before["model_optimizer_sha256"] and \
            post["counters"] == before["counters"] and post["host_python_numpy_rng_sha256"] == start_host_rng and \
            post["inference_sha256"] == previous_inference_hash and post["inference_version"] == previous_version
        report["post_artifact_frozen_gate"] = {"passed": post_ok, "after": post}
        if not post_ok:
            report["status"] = "failed"
            caught = caught or RuntimeError("Post-artifact frozen state gate failed")
        _json(report_path, report)
    if caught is not None:
        raise caught
    return report
