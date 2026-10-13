"""Atomic, immutable training snapshots with explicit external replay references.

Only load trusted snapshots produced by this project. The full format includes
Python/NumPy RNG states, so loading intentionally uses weights_only=False.
Same-version recovery is tested; cross-hardware bitwise recovery is not claimed.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import random
import tempfile

import numpy as np
import torch


def sha256(path: str | Path) -> str:
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def atomic_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def capture_rng() -> dict:
    return {
        "python": random.getstate(), "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [],
    }


def restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def save_checkpoint(directory: str | Path, learner, config: dict, counters: dict,
                    replay, collector: dict, metadata: dict) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"checkpoint_{counters['agent_steps']:09d}.pt"
    if path.exists():
        raise FileExistsError(f"immutable checkpoint already exists: {path}")
    replay_state = replay.snapshot()
    state = {
        "format_version": 1, "config": config, "counters": counters.copy(),
        "learner": learner.state_dict(),
        "optimizers": {key: value.state_dict() for key, value in learner.optimizers.items()},
        "rng": capture_rng(), "replay": replay_state,
        "collector": collector, "metadata": metadata,
    }
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w+b", prefix=".checkpoint-", suffix=".tmp",
                                         dir=directory, delete=False) as stream:
            temporary = stream.name
            torch.save(state, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)
    info = {"file": path.name, "sha256": sha256(path), "bytes": path.stat().st_size,
            "counters": counters, "metadata": metadata,
            "config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
            "replay": replay_state,
            "recovery_scope": "full learner/optimizers/RNG/replay index/collector/Walker state; external replay files required"}
    atomic_json(path.with_suffix(".json"), info)
    atomic_json(directory / "latest.json", {key: info[key] for key in ("file", "sha256", "bytes", "counters")})
    return path


def load_checkpoint(path: str | Path) -> dict:
    path = Path(path)
    info = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    if path.stat().st_size != info["bytes"] or sha256(path) != info["sha256"]:
        raise ValueError("checkpoint integrity mismatch")
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("format_version") != 1:
        raise ValueError("unsupported checkpoint format")
    if state["counters"] != info["counters"]:
        raise ValueError("checkpoint counters disagree with sidecar")
    return state


def restore_learner(learner, state: dict, *, optimizers: bool = True) -> None:
    learner.load_state_dict(state["learner"], strict=True)
    if optimizers:
        if set(state["optimizers"]) != set(learner.optimizers):
            raise ValueError("optimizer sets differ")
        for key, optimizer in learner.optimizers.items():
            optimizer.load_state_dict(state["optimizers"][key])
