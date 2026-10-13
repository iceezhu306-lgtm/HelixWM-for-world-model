"""Single-writer run locks and parent-preserving explicit recovery branches."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import errno
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
from typing import Iterator


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@contextmanager
def exclusive_run_lock(run_dir: str | Path) -> Iterator[Path]:
    """Hold a nonblocking advisory lock for the entire caller's run lifetime.

    All project writers must use this entry point. The persistent lock file
    must not be deleted: replacing its inode could permit duplicate writers.
    """
    run_dir = Path(run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    lock_path = run_dir / ".run.lock"
    if lock_path.is_symlink():
        raise ValueError("run lock must not be a symbolic link")
    with lock_path.open("a+b") as stream:
        if stream.seek(0, os.SEEK_END) == 0:
            stream.write(b"\0")
            stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt
            acquire = lambda: msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            release = lambda: msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            acquire = lambda: fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            release = lambda: fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        try:
            acquire()
        except OSError as error:
            if error.errno in (errno.EACCES, errno.EAGAIN, errno.EDEADLK) or os.name == "nt":
                raise RuntimeError(f"another process already owns the run lock: {run_dir}") from error
            raise
        try:
            yield run_dir
        finally:
            stream.seek(0)
            release()


def prepare_fresh_run(run_dir: str | Path) -> None:
    """Reject existing experiment artifacts while permitting launcher output."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    unexpected = [path.name for path in run_dir.iterdir()
                  if path.name not in {"console.log", ".run.lock"}
                  or not path.is_file() or path.is_symlink()]
    if unexpected:
        raise FileExistsError(f"run directory is not fresh; preserved existing artifacts: {sorted(unexpected)}")


def _verify(path: Path, expected_bytes: int, expected_sha256: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"missing or non-regular immutable source file: {path}")
    if path.stat().st_size != expected_bytes or _sha256(path) != expected_sha256:
        raise ValueError(f"immutable source SHA256 or size mismatch: {path}")


def _link_or_copy(source: Path, destination: Path, size: int, sha256: str) -> str:
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite branch file: {destination}")
    try:
        os.link(source, destination)
        method = "hardlink"
    except FileExistsError:
        raise
    except OSError:
        with source.open("rb") as reader, destination.open("xb") as writer:
            shutil.copyfileobj(reader, writer, length=1024 * 1024)
            writer.flush()
            os.fsync(writer.fileno())
        method = "copy"
    _verify(destination, size, sha256)
    return method


def _write_json(path: Path, content: dict) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(content, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _copy_log_prefix(source: Path, destination: Path, checkpoint_step: int) -> dict:
    result = {"source": str(source), "included_rows": 0, "excluded_rows": 0, "malformed_lines": []}
    if not source.exists():
        result["source_missing"] = True
        return result
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"log source must be a regular file: {source}")
    with source.open("rb") as reader, destination.open("xb") as writer:
        for line_number, raw in enumerate(reader, 1):
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
                step = row.get("agent_steps") if isinstance(row, dict) else None
            except (UnicodeDecodeError, json.JSONDecodeError):
                result["malformed_lines"].append(line_number)
                continue
            numeric = isinstance(step, (int, float)) and not isinstance(step, bool)
            finite = numeric and (isinstance(step, int) or math.isfinite(step))
            if finite and step < checkpoint_step:
                # Preserve the original raw row, adding only a missing final
                # delimiter so a later append cannot concatenate JSON records.
                writer.write(raw)
                if not raw.endswith(b"\n"):
                    writer.write(b"\n")
                result["included_rows"] += 1
            else:
                result["excluded_rows"] += 1
        writer.flush()
        os.fsync(writer.fileno())
    result["sha256"] = _sha256(destination)
    return result


def prepare_resume_branch(run_dir: str | Path, resume_path: str | Path, saved: dict) -> Path:
    """Copy only a verified checkpoint's replay references into a fresh run.

    The source run is never changed. Prefix logs strictly precede the selected
    checkpoint boundary; boundary and abandoned-tail rows remain in the parent.
    The caller reconstructs replay from ``saved['replay']`` after this function.
    Hardlinks are safe only while project checkpoint/replay immutability holds.
    """
    run_dir = Path(run_dir).resolve()
    resume_path = Path(resume_path).resolve()
    if resume_path.parent.name != "CHECKPOINTS" or resume_path.suffix != ".pt":
        raise ValueError("resume checkpoint must be a .pt file inside CHECKPOINTS")
    parent = resume_path.parent.parent
    if run_dir == parent:
        raise ValueError("resume requires a new branch directory; source run must remain unchanged")
    # Nested targets would themselves modify the source directory tree.
    if parent in run_dir.parents or run_dir in parent.parents:
        raise ValueError("resume branch and source run must not contain one another")
    prepare_fresh_run(run_dir)
    counter = saved["counters"]["agent_steps"]
    if not isinstance(counter, int) or isinstance(counter, bool) or counter < 0:
        raise ValueError("checkpoint action-decision counter must be a nonnegative integer")
    if resume_path.name != f"checkpoint_{counter:09d}.pt":
        raise ValueError("checkpoint filename does not match its action-decision counter")
    sidecar_path = resume_path.with_suffix(".json")
    if sidecar_path.is_symlink() or not sidecar_path.is_file():
        raise ValueError("checkpoint sidecar must be a regular file")
    sidecar_raw = sidecar_path.read_bytes()
    info = json.loads(sidecar_raw)
    if info["file"] != resume_path.name or info["counters"] != saved["counters"]:
        raise ValueError("checkpoint sidecar identity or counters do not match loaded state")
    if info["replay"] != saved["replay"]:
        raise ValueError("checkpoint sidecar replay references do not match loaded state")
    _verify(resume_path, info["bytes"], info["sha256"])
    replay_records = saved["replay"]["archive"]
    seen_ids: set[int] = set()
    sources = []
    for record in replay_records:
        episode_id = record["id"]
        if not isinstance(episode_id, int) or isinstance(episode_id, bool) or episode_id < 0 or episode_id in seen_ids:
            raise ValueError("invalid or duplicate replay archive id")
        seen_ids.add(episode_id)
        expected_name = f"episode_{episode_id:08d}.npz"
        if record["file"] != expected_name:
            raise ValueError("invalid replay archive filename; path traversal is forbidden")
        source = parent / "REPLAY" / expected_name
        _verify(source, record["bytes"], record["sha256"])
        sources.append((source, record))
    active_ids = saved["replay"].get("active_ids", [])
    if len(set(active_ids)) != len(active_ids) or not set(active_ids).issubset(seen_ids):
        raise ValueError("checkpoint active replay membership is invalid")

    checkpoint_directory = run_dir / "CHECKPOINTS"
    replay_directory = run_dir / "REPLAY"
    checkpoint_directory.mkdir()
    replay_directory.mkdir()
    methods: dict[str, int] = {"hardlink": 0, "copy": 0}
    for source, record in sources:
        method = _link_or_copy(source, replay_directory / source.name, record["bytes"], record["sha256"])
        methods[method] += 1
    local_checkpoint = checkpoint_directory / resume_path.name
    checkpoint_method = _link_or_copy(resume_path, local_checkpoint, info["bytes"], info["sha256"])
    # Copy sidecar bytes rather than linking mutable JSON metadata.
    with (checkpoint_directory / sidecar_path.name).open("xb") as stream:
        stream.write(sidecar_raw)
        stream.flush()
        os.fsync(stream.fileno())
    _write_json(checkpoint_directory / "latest.json", {
        key: info[key] for key in ("file", "sha256", "bytes", "counters")
    })
    prefixes = {}
    for filename in ("metrics.jsonl", "resources.jsonl", "events.jsonl", "evaluations.jsonl"):
        prefixes[filename] = _copy_log_prefix(parent / filename, run_dir / filename, counter)
    lineage = {
        "format_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "parent_run": str(parent), "parent_checkpoint": str(resume_path),
        "parent_checkpoint_sha256": info["sha256"], "local_checkpoint": str(local_checkpoint),
        "branched_counters": saved["counters"], "parent_preserved": True,
        "copied_checkpoint_count": 1, "copied_replay_archive_count": len(sources),
        "replay_transfer_methods": methods, "checkpoint_transfer_method": checkpoint_method,
        "replay_index_copied": False, "prefix_rule": "numeric agent_steps strictly less than checkpoint agent_steps",
        "log_prefixes": prefixes, "earlier_checkpoint_location": str(parent / "CHECKPOINTS"),
        "retained_history_requires_parent_preservation": True,
        "exact_long_run_recovery_claimed": False,
        "recovery_scope": "explicit same-version state restoration; long-run bitwise equivalence requires separate evidence",
    }
    _write_json(run_dir / "lineage.json", lineage)
    return local_checkpoint
