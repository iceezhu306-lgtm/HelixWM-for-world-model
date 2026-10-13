"""Bounded, read-only HelixWM run inspection using only Python's standard library.

No model is loaded, training resumed, process signalled, or artifact modified.
Example: python examples/inspect_run.py --run-dir /path/to/run --tmux-session helixwm
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time


MAX_DOCUMENT_BYTES = 8 * 1024 * 1024
MAX_LINE_BYTES = 2 * 1024 * 1024
MAX_JSONL_BYTES = 128 * 1024 * 1024
MAX_JSONL_FILES = 1000
COUNTERS = ("agent_steps", "env_steps", "update_steps")


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _safe_json(value):
    if isinstance(value, float) and not math.isfinite(value):
        return repr(value)
    if isinstance(value, dict):
        return {key: _safe_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_json(item) for item in value]
    return value


def _read_document(path: Path, errors: list[str]):
    if not path.is_file():
        return None
    try:
        with path.open("rb") as stream:
            data = stream.read(MAX_DOCUMENT_BYTES + 1)
        if len(data) > MAX_DOCUMENT_BYTES:
            raise ValueError("document exceeds bounded read size")
        return json.loads(data.decode("utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        errors.append(f"{path.name}: {exc}")
        return None


def _update_counters(maxima: dict, row) -> None:
    if isinstance(row, dict):
        for key in COUNTERS:
            value = row.get(key)
            if _number(value) and value >= 0 and int(value) == value:
                maxima[key] = max(maxima[key], int(value))


def _nonfinite_paths(value, prefix="") -> list[str]:
    if isinstance(value, float) and not math.isfinite(value):
        return [prefix]
    if isinstance(value, dict):
        return [found for key, item in value.items() for found in _nonfinite_paths(item, f"{prefix}.{key}" if prefix else key)]
    if isinstance(value, list):
        return [found for index, item in enumerate(value) for found in _nonfinite_paths(item, f"{prefix}[{index}]")]
    return []


def scan_jsonl(root: Path, maxima: dict, max_bytes: int = MAX_JSONL_BYTES) -> dict:
    result = {"files": {}, "bytes_scanned": 0, "readable_rows": 0, "malformed_rows": 0,
              "nonfinite_values": 0, "examples": [], "complete": True, "limit_bytes": max_bytes}
    paths = sorted(root.rglob("*.jsonl"))
    if len(paths) > MAX_JSONL_FILES:
        result["complete"] = False
        result["examples"].append("JSONL file count exceeds bounded scan limit")
    for path in paths[:MAX_JSONL_FILES]:
        relative = path.relative_to(root).as_posix()
        info = {"rows": 0, "malformed_rows": 0, "nonfinite_values": 0, "last_row": None, "mtime_utc": None}
        result["files"][relative] = info
        if path.is_symlink():
            result["complete"] = False
            info["error"] = "symlink skipped"
            continue
        try:
            info["mtime_utc"] = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()
            with path.open("rb") as stream:
                line_number = 0
                while True:
                    remaining = max_bytes - result["bytes_scanned"]
                    if remaining <= 0:
                        result["complete"] = False
                        info["error"] = "total JSONL scan byte limit reached"
                        break
                    line = stream.readline(min(MAX_LINE_BYTES + 1, remaining + 1))
                    if not line:
                        break
                    line_number += 1
                    result["bytes_scanned"] += len(line)
                    if len(line) > MAX_LINE_BYTES or result["bytes_scanned"] > max_bytes:
                        result["complete"] = False
                        info["error"] = "JSONL line or total scan byte limit reached"
                        break
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line.decode("utf-8"))
                        if not isinstance(row, dict):
                            raise ValueError("expected a JSON object")
                    except (UnicodeError, ValueError) as exc:
                        info["malformed_rows"] += 1
                        result["malformed_rows"] += 1
                        if len(result["examples"]) < 20:
                            result["examples"].append(f"{relative}:{line_number}: {exc}")
                        continue
                    info["rows"] += 1
                    result["readable_rows"] += 1
                    info["last_row"] = _safe_json(row)
                    _update_counters(maxima, row)
                    invalid = _nonfinite_paths(row)
                    info["nonfinite_values"] += len(invalid)
                    result["nonfinite_values"] += len(invalid)
                    for key in invalid:
                        if len(result["examples"]) < 20:
                            result["examples"].append(f"{relative}:{line_number}: nonfinite {key}")
        except OSError as exc:
            result["complete"] = False
            info["error"] = str(exc)
        if result["bytes_scanned"] >= max_bytes:
            break
    return result


def inspect_pid(pid, run_dir: Path, proc_root: Path = Path("/proc")) -> dict:
    result = {"pid": pid, "alive": False, "matches_training_module": False,
              "matches_run_dir": False, "is_expected_training_process": False}
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        result["reason"] = "missing or invalid train.pid"
        return result
    directory = proc_root / str(pid)
    result["alive"] = directory.exists()
    try:
        data = (directory / "cmdline").read_bytes()
        if len(data) > 1024 * 1024:
            raise ValueError("process command line exceeds size limit")
        argv = [item.decode("utf-8", errors="replace") for item in data.split(b"\0") if item]
        result["executable"] = Path(argv[0]).name if argv else None
        result["matches_training_module"] = any(argv[index:index + 2] == ["-m", "helixwm.train"] for index in range(len(argv) - 1))
        candidates = []
        for index, item in enumerate(argv):
            if item == "--run-dir" and index + 1 < len(argv):
                candidates.append(argv[index + 1])
            elif item.startswith("--run-dir="):
                candidates.append(item.split("=", 1)[1])
        for value in candidates:
            candidate = Path(value)
            if not candidate.is_absolute():
                candidate = Path(os.readlink(directory / "cwd")) / candidate
            if candidate.resolve() == run_dir.resolve():
                result["matches_run_dir"] = True
        result["is_expected_training_process"] = result["alive"] and result["matches_training_module"] and result["matches_run_dir"]
        if result["is_expected_training_process"]:
            result["argv"] = argv
        else:
            result["argv_omitted"] = "unrelated process arguments are not disclosed"
    except (OSError, ValueError) as exc:
        result["reason"] = str(exc)
    return result


def _command(args: list[str]) -> dict:
    try:
        process = subprocess.run(args, capture_output=True, text=True, timeout=8, check=False)
        return {"returncode": process.returncode, "stdout": process.stdout[-65536:].strip(), "stderr": process.stderr[-4096:].strip()}
    except (OSError, subprocess.SubprocessError) as exc:
        return {"returncode": None, "error": str(exc)}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_checkpoints(root: Path, maxima: dict, errors: list[str]) -> dict:
    directory = root / "CHECKPOINTS"
    result = {"count": 0, "stable_candidates": [], "newest_three": [], "newest_sha256_checked": None}
    paths = sorted(set(directory.glob("checkpoint_*.pt")) | {path.with_suffix(".pt") for path in directory.glob("checkpoint_*.json")})
    for path in paths:
        info = _read_document(path.with_suffix(".json"), errors)
        row = {"file": path.name, "sidecar": path.with_suffix(".json").name,
               "exists": path.is_file(), "sidecar_exists": isinstance(info, dict), "size_matches": False,
               "hash_verified_now": False, "stable_metadata_candidate": False}
        if isinstance(info, dict):
            row.update({"expected_bytes": info.get("bytes"), "expected_sha256": info.get("sha256"), "counters": info.get("counters", {})})
            _update_counters(maxima, info.get("counters"))
            sidecar_stat = path.with_suffix(".json").stat()
            row.update({"sidecar_bytes": sidecar_stat.st_size,
                        "sidecar_mtime_utc": datetime.fromtimestamp(sidecar_stat.st_mtime, timezone.utc).isoformat()})
        if row["exists"]:
            stat = path.stat()
            row.update({"bytes": stat.st_size, "mtime_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(), "age_seconds": max(0, time.time() - stat.st_mtime)})
            row["size_matches"] = row["sidecar_exists"] and row["bytes"] == row.get("expected_bytes")
            digest = info.get("sha256") if isinstance(info, dict) else None
            valid_digest = isinstance(digest, str) and len(digest) == 64 and all(char in "0123456789abcdefABCDEF" for char in digest)
            row["stable_metadata_candidate"] = bool(row["size_matches"] and info.get("file") == path.name and valid_digest and row["age_seconds"] >= 2)
        result["stable_candidates"].append(row)
    result["count"] = len(paths)
    result["newest_three"] = list(reversed(result["stable_candidates"][-3:]))
    if paths:
        newest = result["newest_three"][0]
        path = directory / newest["file"]
        if newest["size_matches"] and not path.is_symlink():
            try:
                before = path.stat()
                digest = _sha256(path)
                after = path.stat()
                stable = (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
                newest.update({"actual_sha256": digest, "unchanged_during_hash": stable,
                               "hash_verified_now": stable and digest == newest.get("expected_sha256")})
                newest["stable_metadata_candidate"] = newest["stable_metadata_candidate"] and newest["hash_verified_now"]
                result["newest_sha256_checked"] = newest["file"]
            except OSError as exc:
                newest["hash_error"] = str(exc)
    # All rows are retained so missing pairs are visible. Only true candidates
    # may be considered for backup; older payload hashes were not reread here.
    result["backup_candidates"] = [row for row in result["stable_candidates"] if row["stable_metadata_candidate"]]
    return result


def inspect_run(run_dir: str | Path, tmux_session: str | None = None, *, proc_root: Path = Path("/proc"),
                cgroup_root: Path = Path("/sys/fs/cgroup"), run_commands: bool = True,
                max_jsonl_bytes: int = MAX_JSONL_BYTES) -> dict:
    root = Path(run_dir).resolve()
    errors, alerts = [], []
    maxima = {key: 0 for key in COUNTERS}
    documents = {name: _read_document(root / f"{name}.json", errors) for name in ("config", "status", "metadata", "completion", "failure")}
    pid = _read_document(root / "train.pid", errors)
    exit_code = _read_document(root / "train.exit", errors)
    for value in documents.values():
        _update_counters(maxima, value)
    scan = scan_jsonl(root, maxima, max_jsonl_bytes)
    checkpoints = inspect_checkpoints(root, maxima, errors)
    process = inspect_pid(pid, root, proc_root)
    config = documents["config"] if isinstance(documents["config"], dict) else {}
    completion = documents["completion"] if isinstance(documents["completion"], dict) else {}
    completed = completion.get("state") == "training_and_frozen_evaluation_complete" and completion.get("budget_complete") is True
    budget = config.get("budget_agent_steps")
    repeat = config.get("environment", {}).get("action_repeat")
    cgroup = {"oom_delta": None, "interpretation": "Current cgroup event counters are cumulative; no new OOM or delta is inferred from this single snapshot."}
    for name in ("memory.current", "memory.max", "memory.events"):
        path = cgroup_root / name
        if path.is_file():
            try:
                value = path.read_text().strip()
                if name == "memory.events":
                    cgroup[name] = {key: int(count) for key, count in (line.split() for line in value.splitlines())}
                else:
                    cgroup[name] = int(value) if value.isdigit() else value
            except (OSError, ValueError) as exc:
                errors.append(f"cgroup {name}: {exc}")
    replay_files = list((root / "REPLAY").glob("episode_*.npz"))
    replay = {"archive_file_count": len(replay_files), "archive_disk_bytes": sum(path.stat().st_size for path in replay_files if path.is_file()), "payload_integrity_checked": False}
    try:
        disk_free = shutil.disk_usage(root).free
    except OSError as exc:
        disk_free = None
        errors.append(f"disk query: {exc}")
    if not root.is_dir():
        alerts.append("run_directory_missing")
    if not completed and not process["is_expected_training_process"]:
        alerts.append("expected_training_process_not_verified; run state is unknown unless an exit/failure marker explains it")
    if process["alive"] and not process["is_expected_training_process"]:
        alerts.append("train_pid_points_to_a_different_process")
    if scan["nonfinite_values"]:
        alerts.append("nonfinite_values_recorded_in_jsonl")
    if scan["malformed_rows"]:
        alerts.append("malformed_jsonl_rows; a currently written last line can be partial")
    if not scan["complete"]:
        alerts.append("jsonl_scan_incomplete_due_to_limits_or_read_errors")
    if exit_code not in (None, 0):
        alerts.append(f"nonzero_train_exit:{exit_code}")
    if documents["failure"] and not completed:
        alerts.append("failure_marker_present")
    if completed and process["is_expected_training_process"]:
        alerts.append("completion_marker_exists_but_training_process_is_still_live; final shutdown may be in progress")
    interval = config.get("checkpoint_every")
    newest = checkpoints["newest_three"][0] if checkpoints["newest_three"] else None
    latest_step = newest.get("counters", {}).get("agent_steps") if newest else None
    if newest and not newest.get("hash_verified_now"):
        alerts.append("newest_checkpoint_not_sha256_verified; it may be incomplete or corrupt")
    if not completed and _number(interval) and maxima["agent_steps"] >= interval:
        if not _number(latest_step) or maxima["agent_steps"] - latest_step >= interval:
            alerts.append("checkpoint_missing_or_stale_relative_to_recorded_action_progress")
    if _number(disk_free) and _number(config.get("minimum_free_disk_bytes")) and disk_free < config["minimum_free_disk_bytes"]:
        alerts.append("disk_free_below_configured_threshold")
    current, maximum = cgroup.get("memory.current"), cgroup.get("memory.max")
    if _number(current) and _number(maximum) and maximum > 0:
        cgroup["memory_fraction"] = current / maximum
        if current / maximum >= 0.9:
            alerts.append("cgroup_memory_at_least_90_percent_of_limit")
    memory_events = cgroup.get("memory.events", {})
    if memory_events.get("oom", 0) > 0 or memory_events.get("oom_kill", 0) > 0:
        alerts.append("nonzero_cumulative_cgroup_oom_events; timing and affected process are unknown")
    if _number(budget) and maxima["agent_steps"] > budget:
        alerts.append("recorded_action_counter_exceeds_configured_budget")
    if _number(repeat) and maxima["agent_steps"] and maxima["env_steps"] != maxima["agent_steps"] * repeat:
        alerts.append("max_recorded_environment_and_action_counters_do_not_match_action_repeat; inspect individual rows")
    tmux = {"requested_session": tmux_session, "queried": False}
    gpu = {"queried": False}
    if run_commands:
        gpu = {"queried": True, **_command(["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu", "--format=csv,noheader,nounits"])}
        if tmux_session:
            session = _command(["tmux", "has-session", "-t", f"={tmux_session}"])
            tmux.update({"queried": True, "has_session": session["returncode"] == 0, "check": session})
            if tmux["has_session"]:
                tmux["pane_tail"] = _command(["tmux", "capture-pane", "-p", "-t", f"={tmux_session}", "-S", "-20"])
                if "stdout" in tmux["pane_tail"]:
                    tmux["pane_tail"]["stdout"] = "\n".join(tmux["pane_tail"]["stdout"].splitlines()[-20:])
    return _safe_json({
        "inspection_utc": utc(), "read_only": True, "run_dir": str(root),
        "documents": documents, "train_exit": exit_code, "process": process,
        "max_recorded_counters": maxima,
        "budget_agent_steps": budget,
        "budget_fraction": maxima["agent_steps"] / budget if _number(budget) and budget > 0 else None,
        "units": {"agent_steps": "action decisions", "env_steps": "actual DMC step calls, not MuJoCo integration substeps", "action_repeat": repeat, "evaluation_interactions_in_training_budget": False},
        "completion_marker_verified": completed, "final_acceptance_verified": False,
        "jsonl_scan": scan, "checkpoints": checkpoints, "replay": replay,
        "cgroup": cgroup, "disk_free_bytes": disk_free, "gpu": gpu, "tmux": tmux,
        "alerts": alerts, "errors": errors,
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--tmux-session")
    args = parser.parse_args()
    print(json.dumps(inspect_run(args.run_dir, args.tmux_session), indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
