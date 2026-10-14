"""NumPy episode archives and complete, synchronous JAX-compatible snapshots.

No Torch import is required. Checkpoints use pickle and must only be loaded
from trusted experiment artifacts. Sampled windows preserve raw episode flags;
the training caller must explicitly define its cropped-state/burn-in policy.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, fields, is_dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import pickle
import struct
import tempfile
from typing import Any

import numpy as np


FIELDS = ("image", "action", "reward", "is_first", "is_last", "is_terminal")


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: str | Path, value: Mapping) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="\n", dir=path.parent,
                                         prefix=f".{path.name}-", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _validate_record(record: Mapping) -> str:
    identifier = record["id"]
    if not isinstance(identifier, int) or isinstance(identifier, bool) or identifier < 0:
        raise ValueError("archive ids must be nonnegative integers")
    name = f"episode_{identifier:08d}.npz"
    if record["file"] != name:
        raise ValueError("invalid archive filename or path traversal")
    if not isinstance(record["bytes"], int) or record["bytes"] < 1:
        raise ValueError("invalid archive file size")
    digest = record["sha256"]
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("invalid archive SHA256")
    return name


def verify_archive(directory: str | Path, records: list[dict]) -> None:
    directory = Path(directory)
    ids = set()
    for record in records:
        name = _validate_record(record)
        if record["id"] in ids:
            raise ValueError("duplicate archive id")
        ids.add(record["id"])
        path = directory / name
        if path.is_symlink() or not path.is_file() or path.stat().st_size != record["bytes"] or sha256(path) != record["sha256"]:
            raise ValueError(f"archive file missing or SHA256 mismatch: {name}")


class Replay:
    """Append-only compressed episodes with bounded active sampling membership.

    ``sample(batch, length)`` returns exactly ``length`` aligned observations,
    actions and rewards. Action at index t leads to observation t+1; the last
    full-episode action is a zero placeholder. Sequence starts are uniform
    over all eligible starts, not uniform over episodes. No padding, reset
    insertion, reward rewriting, or episode crossing occurs in this sampler.
    Whole-episode capacity eviction never removes immutable archive files.
    """

    def __init__(self, directory: str | Path, capacity_steps: int = 1_000_000,
                 seed: int = 0, cacheepisodes: int = 32) -> None:
        if not isinstance(capacity_steps, int) or capacity_steps < 1 or not isinstance(cacheepisodes, int) or cacheepisodes < 1:
            raise ValueError("capacity and cacheepisodes must be positive integers")
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.capacity_steps = capacity_steps
        self.cacheepisodes = cacheepisodes
        self.rng = np.random.default_rng(seed)
        self._archive: list[dict] = []
        self._active: list[int] = []
        self._next_id = 0
        self._total_steps_seen = 0
        self._cache: OrderedDict[int, dict[str, np.ndarray]] = OrderedDict()
        if (self.directory / "index.json").exists():
            self.restore(json.loads((self.directory / "index.json").read_text(encoding="utf-8")), persist=False)

    @property
    def episodes(self) -> list[dict]:
        lookup = {record["id"]: record for record in self._archive}
        return [deepcopy(lookup[identifier]) for identifier in self._active]

    @property
    def archive(self) -> list[dict]:
        return deepcopy(self._archive)

    @property
    def archived_episodes(self) -> list[dict]:
        return self.archive

    @property
    def total_steps(self) -> int:
        return sum(record["steps"] for record in self.episodes)

    @property
    def total_steps_seen(self) -> int:
        return self._total_steps_seen

    @property
    def disk_bytes(self) -> int:
        return sum(record["bytes"] for record in self._archive)

    @staticmethod
    def _validate_episode(episode: Mapping) -> dict[str, np.ndarray]:
        if set(episode) != set(FIELDS):
            raise ValueError(f"episode must contain exactly {FIELDS}")
        arrays = {name: np.asarray(episode[name]) for name in FIELDS}
        image = arrays["image"]
        if image.dtype != np.uint8 or image.ndim != 4 or image.shape[1:] != (64, 64, 3) or len(image) < 2:
            raise ValueError("image must be uint8 [N,64,64,3] with N >= 2")
        n = len(image)
        if arrays["action"].shape != (n, 6) or arrays["action"].dtype != np.float32:
            raise ValueError("action must be float32 [N,6]")
        if arrays["reward"].shape != (n,) or arrays["reward"].dtype != np.float32:
            raise ValueError("reward must be float32 [N]")
        for name in ("is_first", "is_last", "is_terminal"):
            if arrays[name].shape != (n,) or arrays[name].dtype != np.bool_:
                raise ValueError(f"{name} must be boolean [N]")
        if not np.isfinite(arrays["action"]).all() or not np.isfinite(arrays["reward"]).all():
            raise ValueError("actions and rewards must be finite")
        if arrays["reward"][0] != 0 or np.any(arrays["action"][-1] != 0):
            raise ValueError("initial reward and final action placeholder must be zero")
        if not arrays["is_first"][0] or arrays["is_first"][1:].any():
            raise ValueError("a complete episode must start exactly once")
        if not arrays["is_last"][-1] or arrays["is_last"][:-1].any():
            raise ValueError("a complete episode must end exactly once")
        if np.any(arrays["is_terminal"] & ~arrays["is_last"]):
            raise ValueError("terminal markers are allowed only at the episode end")
        return {name: np.ascontiguousarray(value) for name, value in arrays.items()}

    def add_episode(self, episode: Mapping) -> dict:
        arrays = self._validate_episode(episode)
        steps = len(arrays["image"]) - 1
        if steps > self.capacity_steps:
            raise ValueError("single episode exceeds active replay capacity")
        identifier = self._next_id
        path = self.directory / f"episode_{identifier:08d}.npz"
        while path.exists() or path.with_suffix(".json").exists():
            identifier += 1
            path = self.directory / f"episode_{identifier:08d}.npz"
        temporary = None
        try:
            with tempfile.NamedTemporaryFile("w+b", dir=self.directory, prefix=".episode-",
                                             suffix=".tmp", delete=False) as stream:
                temporary = Path(stream.name)
                np.savez_compressed(stream, **arrays)
                stream.flush()
                os.fsync(stream.fileno())
            # The replay has one writer. An existing destination is never a
            # legal continuation target, including an interrupted older branch.
            if path.exists():
                raise FileExistsError(path)
            os.replace(temporary, path)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()
        record = {"id": identifier, "file": path.name, "frames": steps + 1, "steps": steps,
                  "bytes": path.stat().st_size, "sha256": sha256(path)}
        atomic_json(path.with_suffix(".json"), record)
        self._archive.append(record)
        self._active.append(identifier)
        self._next_id = identifier + 1
        self._total_steps_seen += steps
        active_steps = self.total_steps
        while active_steps > self.capacity_steps:
            removed = self._active.pop(0)
            active_steps -= next(item["steps"] for item in self._archive if item["id"] == removed)
            self._cache.pop(removed, None)
        atomic_json(self.directory / "index.json", self.snapshot())
        return deepcopy(record)

    def _load(self, record: dict) -> dict[str, np.ndarray]:
        identifier = record["id"]
        if identifier in self._cache:
            self._cache.move_to_end(identifier)
            return self._cache[identifier]
        # Check once when loading into the LRU, not on every cache hit. Archive
        # files are immutable and restore verifies every referenced file.
        verify_archive(self.directory, [record])
        with np.load(self.directory / record["file"], allow_pickle=False) as data:
            arrays = self._validate_episode({name: data[name] for name in data.files})
        if len(arrays["image"]) != record["frames"]:
            raise ValueError("archive frame count mismatch")
        self._cache[identifier] = arrays
        while len(self._cache) > self.cacheepisodes:
            self._cache.popitem(last=False)
        return arrays

    def can_sample(self, length: int) -> bool:
        return length > 0 and any(record["frames"] >= length for record in self.episodes)

    def sample(self, batch: int, length: int) -> dict[str, np.ndarray]:
        if not isinstance(batch, int) or not isinstance(length, int) or min(batch, length) < 1:
            raise ValueError("batch and length must be positive integers")
        records = [record for record in self.episodes if record["frames"] >= length]
        if not records:
            raise ValueError("no active episode has enough frames for this sequence length")
        cumulative = np.cumsum([record["frames"] - length + 1 for record in records], dtype=np.int64)
        draws = self.rng.integers(int(cumulative[-1]), size=batch)
        selected = np.searchsorted(cumulative, draws, side="right")
        result = {name: [] for name in FIELDS}
        for draw, index in zip(draws, selected):
            start = int(draw - (cumulative[index - 1] if index else 0))
            episode = self._load(records[int(index)])
            for name in FIELDS:
                result[name].append(episode[name][start:start + length])
        return {name: np.stack(values) for name, values in result.items()}

    def snapshot(self) -> dict:
        return {"format_version": 2, "capacity_steps": self.capacity_steps,
                "archive": deepcopy(self._archive), "active_ids": list(self._active),
                "next_id": self._next_id, "total_steps_seen": self._total_steps_seen,
                "rng_state": deepcopy(self.rng.bit_generator.state),
                "sampling_contract": "uniform contiguous observation-aligned frames; raw flags preserved"}

    def restore(self, snapshot: Mapping, *, persist: bool = False) -> None:
        """Validate and restore memory state; writing an index is opt-in only."""
        if snapshot.get("format_version") != 2 or snapshot["capacity_steps"] != self.capacity_steps:
            raise ValueError("replay snapshot format or capacity mismatch")
        records = deepcopy(snapshot["archive"])
        verify_archive(self.directory, records)
        lookup = {record["id"]: record for record in records}
        for record in records:
            if record["frames"] != record["steps"] + 1 or record["steps"] < 1:
                raise ValueError("invalid replay archive transition/frame counts")
        active = list(snapshot["active_ids"])
        if len(set(active)) != len(active) or not set(active).issubset(lookup):
            raise ValueError("invalid active replay membership")
        if sum(lookup[identifier]["steps"] for identifier in active) > self.capacity_steps:
            raise ValueError("snapshot active replay exceeds its configured capacity")
        # Build/validate RNG before mutating the live sampler.
        rng = np.random.default_rng()
        rng.bit_generator.state = deepcopy(snapshot["rng_state"])
        self._archive, self._active = records, active
        self._next_id = max(int(snapshot["next_id"]), max(lookup, default=-1) + 1)
        self._total_steps_seen = int(snapshot["total_steps_seen"])
        self.rng = rng
        self._cache.clear()
        if persist:
            atomic_json(self.directory / "index.json", self.snapshot())


@dataclass(frozen=True)
class _TypedJaxKey:
    data: np.ndarray
    implementation: str


def _is_array(value: Any) -> bool:
    return isinstance(value, np.ndarray) or (hasattr(value, "__array__") and hasattr(value, "shape") and hasattr(value, "dtype"))


def _host(value: Any) -> Any:
    if _is_array(value):
        if str(value.dtype).startswith("key<"):
            import jax
            return _TypedJaxKey(np.array(jax.random.key_data(value), copy=True), str(jax.random.key_impl(value)))
        array = np.array(value, copy=True)
        if array.dtype.hasobject:
            raise TypeError("object arrays are not valid numerical state leaves")
        return array
    if isinstance(value, Mapping):
        return {key: _host(item) for key, item in value.items()}
    if isinstance(value, tuple) and hasattr(value, "_fields"):
        return type(value)(*(_host(item) for item in value))
    if isinstance(value, tuple):
        return tuple(_host(item) for item in value)
    if isinstance(value, list):
        return [_host(item) for item in value]
    if is_dataclass(value) and not isinstance(value, type):
        return replace(value, **{field.name: _host(getattr(value, field.name)) for field in fields(value)})
    if isinstance(value, (str, bytes, bool, int, float, complex, type(None), np.generic)):
        return value
    raise TypeError(f"unsupported checkpoint state leaf: {type(value).__module__}.{type(value).__qualname__}")


def _restore_special(value: Any) -> Any:
    if isinstance(value, _TypedJaxKey):
        import jax
        return jax.random.wrap_key_data(value.data, impl=value.implementation)
    if isinstance(value, Mapping):
        return {key: _restore_special(item) for key, item in value.items()}
    if isinstance(value, tuple) and hasattr(value, "_fields"):
        return type(value)(*(_restore_special(item) for item in value))
    if isinstance(value, tuple):
        return tuple(_restore_special(item) for item in value)
    if isinstance(value, list):
        return [_restore_special(item) for item in value]
    if is_dataclass(value) and not isinstance(value, type):
        return replace(value, **{field.name: _restore_special(getattr(value, field.name)) for field in fields(value)})
    return value


def tree_hash(value: Any) -> str:
    """Hash numerical values, dtype identity, shape and container structure.

    NumPy/JAX leaves with equal values hash equally. Including dtype.name as
    well as dtype.str distinguishes bfloat16 from unrelated two-byte voids.
    """
    value = _host(value)
    digest = hashlib.sha256()

    def block(data: bytes) -> None:
        digest.update(struct.pack("!Q", len(data)))
        digest.update(data)

    def visit(item: Any) -> None:
        if isinstance(item, np.ndarray):
            block(b"array")
            block(item.dtype.name.encode())
            block(item.dtype.str.encode())
            block(repr(item.dtype.descr).encode())
            visit(tuple(item.shape))
            block(np.ascontiguousarray(item).tobytes())
        elif isinstance(item, np.generic):
            visit(np.asarray(item))
        elif isinstance(item, Mapping):
            block(b"mapping")
            visit(len(item))
            for key in sorted(item, key=lambda key: (type(key).__module__, type(key).__qualname__, repr(key))):
                visit(key)
                visit(item[key])
        elif isinstance(item, (tuple, list)):
            block(type(item).__module__.encode())
            block(type(item).__qualname__.encode())
            visit(len(item))
            for element in item:
                visit(element)
        elif is_dataclass(item) and not isinstance(item, type):
            block(type(item).__module__.encode())
            block(type(item).__qualname__.encode())
            for field in fields(item):
                visit(field.name)
                visit(getattr(item, field.name))
        elif item is None:
            block(b"none")
        elif isinstance(item, bool):
            block(b"true" if item else b"false")
        elif isinstance(item, int):
            block(b"int")
            block(str(item).encode())
        elif isinstance(item, float):
            block(b"float")
            block(struct.pack("!d", item))
        elif isinstance(item, complex):
            block(b"complex")
            block(struct.pack("!dd", item.real, item.imag))
        elif isinstance(item, str):
            block(b"str")
            block(item.encode("utf-8"))
        elif isinstance(item, bytes):
            block(b"bytes")
            block(item)
        else:
            raise TypeError(f"unsupported hash leaf: {type(item)}")

    visit(value)
    return digest.hexdigest()


def save_checkpoint(directory: str | Path, payload: Mapping, agent_steps: int | None = None) -> Path:
    """Synchronously save a complete caller-supplied state; never overwrite.

    The payload owns backbone, dual-chain, collector, all RNG/optimizer states,
    replay, scheduler, configuration and metadata. This function preserves all
    supplied fields, without silently selecting only model parameters.
    """
    if not isinstance(payload, Mapping) or "counters" not in payload or "replay" not in payload:
        raise ValueError("checkpoint payload requires counters and replay mappings")
    counter = payload["counters"]["agent_steps"]
    if not isinstance(counter, int) or isinstance(counter, bool) or counter < 0:
        raise ValueError("agent_steps must be a nonnegative integer")
    if agent_steps is not None and agent_steps != counter:
        raise ValueError("explicit checkpoint step disagrees with payload counter")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"checkpoint_{counter:09d}.pkl"
    if path.exists() or path.with_suffix(".json").exists():
        raise FileExistsError(f"immutable checkpoint already exists: {path}")
    state = _host(dict(payload))  # Device-to-host transfers finish here.
    records = deepcopy(state["replay"]["archive"])
    for record in records:
        _validate_record(record)
    state_hash = tree_hash(state)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w+b", dir=directory, prefix=".checkpoint-",
                                         suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            pickle.dump({"format_version": 2, "payload": state}, stream, protocol=pickle.HIGHEST_PROTOCOL)
            stream.flush()
            os.fsync(stream.fileno())
        if path.exists():
            raise FileExistsError(path)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    info = {"format_version": 2, "file": path.name, "bytes": path.stat().st_size,
            "sha256": sha256(path), "tree_hash": state_hash, "counters": state["counters"],
            "payload_keys": sorted(state), "replay_archive": records,
            "external_replay_required": True, "exact_long_run_recovery_claimed": False}
    atomic_json(path.with_suffix(".json"), info)
    atomic_json(directory / "latest.json", {key: info[key] for key in ("file", "bytes", "sha256", "tree_hash", "counters")})
    return path


def load_checkpoint(path: str | Path, replay_directory: str | Path | None = None,
                    *, verify_replay: bool = True) -> dict:
    """Load trusted state only after file/tree checks; never write any files.

    Replay verification defaults to sibling ``REPLAY``. Frozen-model-only
    evaluation may explicitly set verify_replay=False; recovery must verify it.
    """
    path = Path(path)
    info = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    if info.get("format_version") != 2 or info["file"] != path.name:
        raise ValueError("checkpoint sidecar format or identity mismatch")
    if path.stat().st_size != info["bytes"] or sha256(path) != info["sha256"]:
        raise ValueError("checkpoint SHA256 or size mismatch")
    with path.open("rb") as stream:
        saved = pickle.load(stream)
        if stream.read(1):
            raise ValueError("unexpected trailing checkpoint data")
    if saved.get("format_version") != 2:
        raise ValueError("unsupported checkpoint format")
    payload = saved["payload"]
    if tree_hash(payload) != info["tree_hash"]:
        raise ValueError("checkpoint numerical tree hash mismatch")
    if payload["counters"] != info["counters"] or payload["replay"]["archive"] != info["replay_archive"]:
        raise ValueError("checkpoint counters or replay references disagree with sidecar")
    if verify_replay:
        replay_directory = Path(replay_directory) if replay_directory is not None else path.parent.parent / "REPLAY"
        verify_archive(replay_directory, info["replay_archive"])
    return _restore_special(payload)
