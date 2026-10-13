"""Append-only compressed episode archive and bounded active replay sampler.

Eviction affects sampling membership, never archived files. This preserves the
replay files referenced by older checkpoints. No observations are converted to
floating point or retained in memory beyond the bounded episode LRU.
"""

from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np


_FIELDS = ("observations", "actions", "rewards", "continuations", "errors", "error_valid", "model_versions")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class DiskReplay:
    """Episode replay with uniform sampling over eligible sequence starts.

    Capacity is counted in action decisions. Complete episodes are evicted from
    the active set; the most recent episode is retained even if it alone exceeds
    capacity. ``total_steps`` reports the actual active size, including that
    explicitly documented possible one-episode overflow.
    """

    def __init__(self, directory: str | Path, capacity_steps: int = 200000, seed: int = 0) -> None:
        if capacity_steps < 1:
            raise ValueError("capacity_steps must be positive")
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.capacity_steps = int(capacity_steps)
        self._rng = np.random.default_rng(seed)
        self._archive: list[dict[str, Any]] = []
        self._active_ids: list[int] = []
        self._next_id = 0
        self._total_steps_seen = 0
        self._dimensions: tuple[int, int] | None = None
        self._cache: OrderedDict[int, dict[str, np.ndarray]] = OrderedDict()
        self.max_cached_episodes = 8
        if self._index_path.exists():
            with self._index_path.open(encoding="utf-8") as stream:
                self.restore(json.load(stream), persist=False)

    @property
    def _index_path(self) -> Path:
        return self.directory / "index.json"

    @property
    def episodes(self) -> list[dict[str, Any]]:
        archive = {record["id"]: record for record in self._archive}
        return [deepcopy(archive[episode_id]) for episode_id in self._active_ids]

    @property
    def archived_episodes(self) -> list[dict[str, Any]]:
        return deepcopy(self._archive)

    @property
    def total_steps(self) -> int:
        return sum(record["steps"] for record in self.episodes)

    @property
    def total_steps_seen(self) -> int:
        return self._total_steps_seen

    @property
    def disk_bytes(self) -> int:
        return sum(record["bytes"] for record in self._archive)

    @property
    def diskbytes(self) -> int:
        return self.disk_bytes

    def _validate_episode(self, episode: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        if set(episode) != set(_FIELDS):
            raise ValueError(f"episode must contain exactly {_FIELDS}")
        arrays = {name: np.asarray(episode[name]) for name in _FIELDS}
        observations, actions = arrays["observations"], arrays["actions"]
        if observations.ndim != 2 or observations.dtype != np.uint8 or observations.shape[0] < 2 or observations.shape[1] < 1:
            raise ValueError("observations must be uint8 [N,O] with N >= 2 and O >= 1")
        n = observations.shape[0]
        if actions.ndim != 2 or actions.shape[0] != n - 1 or actions.shape[1] < 1:
            raise ValueError("actions must have shape [N-1,A]")
        dimensions = (observations.shape[1], actions.shape[1])
        if self._dimensions is not None and dimensions != self._dimensions:
            raise ValueError("episode observation/action dimensions differ from replay")
        for name in ("rewards", "continuations", "error_valid", "model_versions"):
            if arrays[name].shape != (n,):
                raise ValueError(f"{name} must have shape [N]")
        if arrays["errors"].shape != (n, 2):
            raise ValueError("errors must have shape [N,2]")
        if arrays["error_valid"].dtype != np.bool_:
            raise ValueError("error_valid must be boolean")
        if arrays["model_versions"].dtype != np.int64:
            raise ValueError("model_versions must use int64")
        for name in ("actions", "rewards", "continuations", "errors"):
            if not np.issubdtype(arrays[name].dtype, np.number) or not np.isfinite(arrays[name]).all():
                raise ValueError(f"{name} must be finite numeric values")
        if arrays["rewards"][0] != 0 or arrays["error_valid"][0]:
            raise ValueError("initial observation must have zero reward and no error label")
        if np.any(arrays["continuations"] < 0) or np.any(arrays["continuations"] > 1):
            raise ValueError("continuations must be in [0,1]")
        if np.any(arrays["errors"] < 0):
            raise ValueError("errors must be nonnegative")
        arrays = {name: np.ascontiguousarray(value) for name, value in arrays.items()}
        return arrays

    def add_episode(self, episode: dict[str, np.ndarray]) -> dict[str, Any]:
        arrays = self._validate_episode(episode)
        episode_id = self._next_id
        path = self.directory / f"episode_{episode_id:08d}.npz"
        # Unindexed files can survive interruption between atomic episode and
        # index writes. Preserve them and allocate a fresh id on continuation.
        while path.exists():
            episode_id += 1
            path = self.directory / f"episode_{episode_id:08d}.npz"
        temporary: str | None = None
        try:
            with tempfile.NamedTemporaryFile(mode="w+b", prefix=".episode-", suffix=".npz", dir=self.directory, delete=False) as stream:
                temporary = stream.name
                np.savez_compressed(stream, **arrays)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            if temporary is not None and os.path.exists(temporary):
                os.unlink(temporary)
        record = {
            "id": episode_id, "file": path.name, "steps": int(arrays["actions"].shape[0]),
            "bytes": path.stat().st_size, "sha256": _sha256(path),
        }
        self._archive.append(record)
        self._active_ids.append(episode_id)
        self._next_id = episode_id + 1
        self._total_steps_seen += record["steps"]
        self._dimensions = (arrays["observations"].shape[1], arrays["actions"].shape[1])
        while len(self._active_ids) > 1 and self.total_steps > self.capacity_steps:
            removed = self._active_ids.pop(0)
            self._cache.pop(removed, None)
        self._persist()
        return deepcopy(record)

    def _load(self, record: dict[str, Any]) -> dict[str, np.ndarray]:
        episode_id = record["id"]
        if episode_id in self._cache:
            self._cache.move_to_end(episode_id)
            return self._cache[episode_id]
        path = self.directory / record["file"]
        with np.load(path, allow_pickle=False) as stored:
            arrays = {name: stored[name] for name in _FIELDS}
        self._validate_episode(arrays)
        if arrays["actions"].shape[0] != record["steps"]:
            raise ValueError(f"replay episode length mismatch: {path.name}")
        self._cache[episode_id] = arrays
        while len(self._cache) > self.max_cached_episodes:
            self._cache.popitem(last=False)
        return arrays

    def can_sample(self, sequence_length: int) -> bool:
        return sequence_length > 0 and any(record["steps"] >= sequence_length for record in self.episodes)

    def sample(self, batch_size: int, sequence_length: int) -> dict[str, np.ndarray]:
        if batch_size < 1 or sequence_length < 1:
            raise ValueError("batch_size and sequence_length must be positive")
        records = [record for record in self.episodes if record["steps"] >= sequence_length]
        counts = np.asarray([record["steps"] - sequence_length + 1 for record in records], dtype=np.int64)
        if not len(records):
            raise ValueError("no replay episode is long enough for the requested sequence")
        cumulative = np.cumsum(counts)
        draws = self._rng.integers(0, int(cumulative[-1]), size=batch_size)
        selection = np.searchsorted(cumulative, draws, side="right")
        batch: dict[str, list[np.ndarray]] = {name: [] for name in _FIELDS}
        for draw, selected in zip(draws, selection):
            start = int(draw - (cumulative[selected - 1] if selected else 0))
            arrays = self._load(records[int(selected)])
            for name in _FIELDS:
                end = start + sequence_length + (0 if name == "actions" else 1)
                batch[name].append(arrays[name][start:end])
        result = {name: np.stack(values) for name, values in batch.items()}
        result["is_first"] = np.zeros((batch_size, sequence_length + 1), dtype=np.bool_)
        result["is_first"][:, 0] = True
        result["valid"] = np.ones((batch_size, sequence_length + 1), dtype=np.bool_)
        return result

    def snapshot(self) -> dict[str, Any]:
        return {
            "format_version": 1, "capacity_steps": self.capacity_steps,
            "archive": deepcopy(self._archive), "active_ids": list(self._active_ids),
            "next_id": self._next_id, "total_steps_seen": self._total_steps_seen,
            "dimensions": None if self._dimensions is None else list(self._dimensions),
            "rng_state": deepcopy(self._rng.bit_generator.state),
        }

    def restore(self, snapshot: dict[str, Any], *, persist: bool = True) -> None:
        if snapshot.get("format_version") != 1 or snapshot["capacity_steps"] != self.capacity_steps:
            raise ValueError("replay snapshot format or capacity does not match")
        records = deepcopy(snapshot["archive"])
        record_ids: set[int] = set()
        for record in records:
            expected_name = f"episode_{int(record['id']):08d}.npz"
            if record["file"] != expected_name or record["id"] in record_ids:
                raise ValueError("invalid or duplicate replay archive record")
            record_ids.add(record["id"])
            path = self.directory / record["file"]
            if not path.is_file() or path.stat().st_size != record["bytes"] or _sha256(path) != record["sha256"]:
                raise ValueError(f"replay file missing or checksum mismatch: {record['file']}")
        active = list(snapshot["active_ids"])
        if len(set(active)) != len(active) or not set(active).issubset(record_ids):
            raise ValueError("invalid active replay membership")
        # Preserve later archive files when restoring an older training snapshot.
        # Their membership is inactive: only the snapshot's active set is sampled.
        merged = {record["id"]: record for record in self._archive}
        merged.update({record["id"]: record for record in records})
        self._archive = sorted(merged.values(), key=lambda record: record["id"])
        self._active_ids = active
        self._next_id = max(int(snapshot["next_id"]), 1 + max(merged, default=-1))
        self._total_steps_seen = int(snapshot["total_steps_seen"])
        dimensions = snapshot["dimensions"]
        self._dimensions = None if dimensions is None else tuple(dimensions)
        self._rng.bit_generator.state = deepcopy(snapshot["rng_state"])
        self._cache.clear()
        if persist:
            self._persist()

    def _persist(self) -> None:
        temporary: str | None = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", prefix=".index-", suffix=".json", dir=self.directory, delete=False) as stream:
                temporary = stream.name
                json.dump(self.snapshot(), stream, indent=2)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self._index_path)
        finally:
            if temporary is not None and os.path.exists(temporary):
                os.unlink(temporary)
