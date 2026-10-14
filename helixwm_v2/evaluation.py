"""Frozen, causal paired evaluation of one trained dual-chain model.

Clean, explicit dropout repair, and optional black-frame and prior-only controls
share fixed environment/policy/missingness seeds. They are inference interventions
on one policy, not independently trained baselines or independent training seeds.
"""
from __future__ import annotations

import copy
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import time
import uuid

import numpy as np

from helixwm.environment import WalkerEnv
from .dual_chain import DualChain


class DropoutSchedule:
    """Independent explicit missing-frame bursts; no hidden clean-state input."""

    def __init__(self, seed: int, config: dict):
        values = config.get("observation_dropout", config)
        self.config = {
            "start_probability": float(values.get("start_probability", .02)),
            "min_length": int(values.get("min_length", 1)),
            "max_length": int(values.get("max_length", 5)),
        }
        if not 0 <= self.config["start_probability"] <= 1:
            raise ValueError("start_probability must lie in [0,1]")
        if not 1 <= self.config["min_length"] <= self.config["max_length"] <= 5:
            raise ValueError("missing bursts must have length 1..5")
        self.rng = np.random.default_rng(seed)
        self.remaining = 0
        self.calls = self.missing = self.bursts = 0

    def valid(self, is_first: bool = False, is_last: bool = False) -> bool:
        self.calls += 1
        if is_first or is_last:
            self.remaining = 0
            return True
        if self.remaining:
            self.remaining -= 1
            self.missing += 1
            return False
        if self.rng.random() < self.config["start_probability"]:
            length = int(self.rng.integers(self.config["min_length"], self.config["max_length"] + 1))
            self.remaining = length - 1
            self.bursts += 1
            self.missing += 1
            return False
        return True

    def state_dict(self) -> dict:
        return {"format_version": 1, "config": copy.deepcopy(self.config),
                "rng": copy.deepcopy(self.rng.bit_generator.state),
                "remaining": self.remaining, "calls": self.calls,
                "missing": self.missing, "bursts": self.bursts}

    def load_state_dict(self, saved: dict) -> None:
        if saved.get("format_version") != 1 or saved.get("config") != self.config:
            raise ValueError("dropout schedule snapshot configuration mismatch")
        if not 0 <= saved["remaining"] < self.config["max_length"]:
            raise ValueError("invalid pending missingness length")
        self.rng.bit_generator.state = copy.deepcopy(saved["rng"])
        self.remaining = int(saved["remaining"])
        self.calls, self.missing, self.bursts = (int(saved[key]) for key in ("calls", "missing", "bursts"))


def _append(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")


def _write_json(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _counters(backbone) -> dict:
    agent = getattr(backbone, "agent", backbone)
    return {key: int(getattr(agent, key)) for key in ("n_updates", "n_batches", "n_actions")
            if hasattr(agent, key)}


def _sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def _decode_video(path: Path, expected_frames: int, fps: float) -> dict:
    import imageio.v2 as imageio
    count, shapes = 0, set()
    reader = imageio.get_reader(str(path), format="FFMPEG")
    try:
        metadata = reader.get_meta_data()
        for frame in reader:
            count += 1
            shapes.add(tuple(frame.shape))
            if frame.dtype != np.uint8 or not np.isfinite(frame).all():
                raise ValueError("decoded video frame is invalid")
    finally:
        reader.close()
    actual_fps = float(metadata["fps"])
    if count != expected_frames or shapes != {(256, 256, 3)} or abs(actual_fps - fps) > .01:
        raise ValueError(f"video verification failed: frames={count}, shapes={shapes}, fps={actual_fps}")
    return {"path": str(path), "full_decode_pass": True, "frames": count,
            "fps": actual_fps, "width": 256, "height": 256,
            "sha256": _sha256(path), "bytes": path.stat().st_size}


def export_eval_csv(run_dir: str | Path) -> Path:
    results = Path(run_dir) / "RESULTS"
    rows = [json.loads(line) for line in (results / "eval.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    fields = sorted(set().union(*(row.keys() for row in rows))) if rows else []
    path = results / "eval.csv"
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)
    return path


def plot_learning_curve(run_dir: str | Path) -> dict:
    """Plot un-smoothed evaluation episode means; no fabricated seed interval."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    root = Path(run_dir)
    path = root / "RESULTS" / "eval.jsonl"
    if not path.exists():
        return {"available": False, "reason": "no_evaluation_rows"}
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    reports = root / "RESULTS" / "evaluation_reports"
    if reports.exists():
        passed_reports = set()
        for report_file in reports.glob("*.json"):
            saved = json.loads(report_file.read_text(encoding="utf-8"))
            if saved.get("status") == "complete" and saved.get("hash_gate", {}).get("passed"):
                passed_reports.add(saved["evaluation_id"])
        rows = [row for row in rows if row["evaluation_id"] in passed_reports]
    if not rows:
        return {"available": False, "reason": "no_evaluation_rows"}
    # Each report is a separate measurement. Do not silently average repeated runs.
    groups = {}
    for row in rows:
        key = (row["condition"], int(row["agent_steps"]), row["evaluation_id"])
        groups.setdefault(key, []).append(float(row["episode_return"]))
    fig, axis = plt.subplots(figsize=(9, 5.4), constrained_layout=True)
    styles = {"clean": ("#1766ad", "Clean observations"),
              "dropout": ("#d97706", "Explicit dropout + dual-chain repair"),
              "dropout_disabled": ("#5f6368", "Same dropout, black-frame input (one policy)"),
              "dropout_prior_only": ("#8b5bb3", "Same dropout, prior-only filter (one policy)")}
    for condition in ("clean", "dropout", "dropout_disabled", "dropout_prior_only"):
        selected = sorted((key[1], key[2], float(np.mean(values))) for key, values in groups.items() if key[0] == condition)
        if not selected:
            continue
        color, label = styles[condition]
        axis.plot([item[0] for item in selected], [item[2] for item in selected],
                  marker="o", markersize=3.2, linewidth=1.8, color=color, label=label)
    axis.set(xlabel="Training action decisions (action repeat = 2)",
             ylabel="Mean undiscounted evaluation episode return",
             title="HelixWM dual-chain Walker Walk | training seed 0")
    axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value / 1e6:g}M"))
    axis.set_xlim(left=0)
    axis.grid(alpha=.22)
    axis.legend(loc="best", fontsize=8)
    fig.text(.5, -.025, "Raw checkpoint means, no smoothing. Evaluation seeds are not independent training seeds.",
             ha="center", fontsize=8)
    output = root / "PLOTS"
    output.mkdir(parents=True, exist_ok=True)
    png, pdf = output / "learning_curve.png", output / "learning_curve.pdf"
    fig.savefig(png, dpi=180, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return {"available": True, "png": str(png), "pdf": str(pdf),
            "smoothing": "none", "uncertainty_band": "none_single_training_seed"}


def evaluate(backbone, config: dict, run_dir: str | Path, agent_steps: int,
             kind: str = "periodic", episodes: int = 5, video: bool = False,
             audit_disabled: bool = False) -> dict:
    """Evaluate without updates and restore the collector's pinned snapshot.

    The final caller must independently reload a checkpoint before calling this
    function. This function proves frozen execution, not checkpoint reload.
    A disabled dropout control observes an explicit black frame at missing times;
    a prior-only control skips the encoder and propagates one ordinary prior,
    without candidate selection or ancestral repair. Both use the same missing
    schedule as repair. Neither changes the model or launches another training.
    """
    if not re.fullmatch(r"[A-Za-z0-9_-]+", kind):
        raise ValueError("kind contains unsafe path characters")
    if not isinstance(episodes, int) or episodes < 1 or agent_steps < 0:
        raise ValueError("positive episode count and nonnegative action budget required")
    if video and kind != "final":
        raise ValueError("only final evaluation may save the one formal video")
    root = Path(run_dir)
    results = root / "RESULTS"
    results.mkdir(parents=True, exist_ok=True)
    video_path = root / "VIDEOS" / "clean_training_final.mp4"
    if video and video_path.exists():
        raise FileExistsError("formal video already exists; do not overwrite accepted evidence")
    identifier = f"{kind}_{agent_steps:09d}_{uuid.uuid4().hex[:12]}"
    report_path = results / "evaluation_reports" / f"{identifier}.json"
    report = {"evaluation_id": identifier, "kind": kind, "agent_steps": int(agent_steps),
              "training_env_steps": int(agent_steps) * int(config["environment"]["action_repeat"]),
              "utc": datetime.now(timezone.utc).isoformat(), "status": "running",
              "conditions": {}, "single_training_seed": int(config.get("seed", 0)),
              "evaluation_interactions_in_training_budget": False,
              "conditions_are_one_policy_interventions": True,
              "checkpoint_reload_checked_here": False,
              "actor_sampling": "original_stochastic_policy_with_fixed_private_rng",
              "score_kind": getattr(backbone, "score_kind", "backend_score"),
              "raw_jsonl": str(results / "eval.jsonl"), "raw_csv": str(results / "eval.csv")}
    previous_params, previous_version = backbone.inference_params, backbone.inference_version
    try:
        backbone.pin_inference()
        report["inference_version"] = int(backbone.inference_version)
        before = {"all_parameter_optimizer_state_sha256": backbone.parameter_hash(),
                  "inference_parameter_sha256": backbone.parameter_hash(inference=True),
                  "counters": _counters(backbone)}
    except BaseException:
        backbone.inference_params, backbone.inference_version = previous_params, previous_version
        raise
    conditions = ["clean", "dropout"] + (
        ["dropout_disabled", "dropout_prior_only"] if audit_disabled else [])
    condition_definitions = {
        "clean": "actual valid observations through the ordinary backbone filter",
        "dropout": "known missing frames use bounded candidates and returning-evidence ancestral repair",
        "dropout_disabled": "known missing frames are black images encoded by the ordinary filter",
        "dropout_prior_only": "known missing frames skip encoding; one prior state propagates without candidate repair",
    }
    report["condition_definitions"] = {name: condition_definitions[name] for name in conditions}
    start = time.monotonic()
    exception = None
    try:
        for condition in conditions:
            condition_rows = []
            for index in range(episodes):
                seed = int(config.get("eval_seed_start", 10000)) + index
                env = WalkerEnv(seed=seed, **config["environment"])
                writer = None
                frame_count = 0
                try:
                    raw = env.reset().reshape(config["environment"]["image_size"],
                                              config["environment"]["image_size"], 3)
                    chain_args = copy.deepcopy(config.get("dual_chain", {}))
                    chain_args.pop("mode", None)
                    chain = DualChain(backbone, seed=200000 + seed,
                                      model_version=backbone.inference_version,
                                      mode="disabled" if condition in {"dropout_disabled", "dropout_prior_only"}
                                      else "repair", **chain_args)
                    episode_id = f"{identifier}/{condition}/{seed}"
                    chain.reset(raw, episode_id=episode_id, model_version=backbone.inference_version)
                    schedule = DropoutSchedule(100000 + seed, config)
                    schedule.valid(is_first=True)
                    missing_digest = hashlib.sha256()
                    total, count, env_count, missing_count, repairs, abstentions = 0., 0, 0, 0, 0, 0
                    saturated, coordinates = 0, 0
                    if video and condition == "clean" and index == 0:
                        import imageio.v2 as imageio
                        video_path.parent.mkdir(parents=True, exist_ok=True)
                        writer = imageio.get_writer(str(video_path), format="FFMPEG", mode="I",
                                                    fps=float(config.get("video_fps", 20)),
                                                    codec="libx264", quality=7, macro_block_size=16)
                        writer.append_data(env.render(256))
                        frame_count = 1
                    while True:
                        action = chain.act(deterministic=False)
                        result = env.step(action[0])
                        count += 1
                        env_count += int(result["env_steps"])
                        total += float(result["reward"])
                        saturated += int((np.abs(action) >= .999).sum())
                        coordinates += action.size
                        raw = np.asarray(result["observation"]).reshape(raw.shape)
                        scheduled_valid = schedule.valid(is_last=bool(result["done"]))
                        available = True if condition == "clean" else scheduled_valid
                        missing_digest.update(bytes([int(available)]))
                        missing_count += int(not available)
                        if condition == "dropout_disabled":
                            supplied = raw if available else np.zeros_like(raw)
                            chain.step(supplied, action, episode_id=episode_id, timestamp=count,
                                       model_version=backbone.inference_version, valid=True)
                        else:
                            chain.step(raw if available else None, action, episode_id=episode_id,
                                       timestamp=count, model_version=backbone.inference_version, valid=available)
                        for event in chain.pop_events():
                            repairs += int(event["event"] == "repair_reconnected")
                            abstentions += int(event["event"] == "repair_abstained")
                            _append(results / "evaluation_events.jsonl", {"evaluation_id": identifier,
                                    "condition": condition, "evaluation_seed": seed, **event})
                        if writer is not None:
                            writer.append_data(env.render(256))
                            frame_count += 1
                        if result["done"]:
                            break
                    row = {"evaluation_id": identifier, "kind": kind, "condition": condition,
                           "agent_steps": int(agent_steps), "training_env_steps": report["training_env_steps"],
                           "training_seed": int(config.get("seed", 0)), "evaluation_seed": seed,
                           "policy_seed": 200000 + seed, "dropout_seed": 100000 + seed,
                           "episode": index, "episode_return": total,
                           "episode_length": count, "actual_eval_agent_steps": count,
                           "actual_eval_env_steps": env_count, "action_repeat": int(env.action_repeat),
                           "missing_observations": missing_count,
                           "missing_input_handling": condition_definitions[condition],
                           "availability_sha256": missing_digest.hexdigest(),
                           "repair_count": repairs, "abstention_count": abstentions,
                           "action_coordinate_abs_ge0999_fraction": saturated / max(1, coordinates),
                           "inference_version": int(backbone.inference_version)}
                    if not np.isfinite(total):
                        raise FloatingPointError("nonfinite evaluation return")
                    _append(results / "eval.jsonl", row)
                    condition_rows.append(row)
                finally:
                    if writer is not None:
                        writer.close()
                    env.close()
                if writer is not None:
                    report["video"] = _decode_video(video_path, frame_count, float(config.get("video_fps", 20)))
            returns = np.asarray([row["episode_return"] for row in condition_rows], np.float64)
            report["conditions"][condition] = {
                "episodes": episodes, "mean_return": float(returns.mean()),
                "std_return_population": float(returns.std(ddof=0)),
                "min_return": float(returns.min()), "max_return": float(returns.max()),
                "evaluation_seeds": [row["evaluation_seed"] for row in condition_rows],
                "raw_episode_returns": returns.tolist(),
                "missing_observations": sum(row["missing_observations"] for row in condition_rows),
                "repair_count": sum(row["repair_count"] for row in condition_rows),
                "abstention_count": sum(row["abstention_count"] for row in condition_rows),
            }
        report["status"] = "complete"
    except BaseException as error:
        exception = error
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
    finally:
        try:
            after = {"all_parameter_optimizer_state_sha256": backbone.parameter_hash(),
                     "inference_parameter_sha256": backbone.parameter_hash(inference=True),
                     "counters": _counters(backbone)}
            passed = before == after
            report["hash_gate"] = {"passed": passed, "before": before, "after": after,
                                   "scope": "all official training params including optimizer/EMA plus inference params and counters"}
            if not passed:
                report["status"] = "failed"
                if exception is None:
                    exception = RuntimeError("evaluation changed model/optimizer parameters or counters")
                report["error"] = str(exception)
        except BaseException as hash_error:
            report["status"] = "failed"
            report["hash_gate"] = {"passed": False, "error": str(hash_error)}
            if exception is None:
                exception = hash_error
            report["error"] = str(exception)
        finally:
            backbone.inference_params, backbone.inference_version = previous_params, previous_version
        report["wall_time_s"] = time.monotonic() - start
        report["report_path"] = str(report_path)
        # Persist the zero-update gate before charting; failed or partial
        # evaluations remain inspectable raw data but do not enter the curve.
        _write_json(report_path, report)
        try:
            if (results / "eval.jsonl").exists():
                export_eval_csv(root)
                if report["status"] == "complete":
                    report["learning_curve"] = plot_learning_curve(root)
        except BaseException as artifact_error:
            report["status"] = "failed"
            if exception is None:
                exception = artifact_error
            report["error"] = str(exception)
        _write_json(report_path, report)
    if exception is not None:
        raise exception
    return report
