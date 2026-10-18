"""Minecraft-only complete software snapshots at explicit episode boundaries.

Minecraft/Java engine state is NOT serialized. Recovery reuses the next episode
seed and all software RNGs, then resets a fresh episode. It does not promise a
bitwise-identical regenerated world or recovery in the middle of an episode.
"""
from __future__ import annotations

from collections import deque
from copy import deepcopy
import json
import os
from pathlib import Path
import pickle
import tempfile

import numpy as np

from helixwm_v2.storage import (Replay as BaseReplay, FIELDS, atomic_json, sha256, tree_hash,
                                verify_archive, _host, _restore_special)
from .backbone import TASK_NAMESPACE, BACKBONE_FORMAT


FORMAT = "helixwm_minecraft_episode_boundary_v1"
SAMPLING_POLICY = "minecraft_uniform_circular_real_episode_rows_v1"


class Replay(BaseReplay):
    """Sample real rows across complete episodes, including short successes.

    Starts are uniform over every retained observation row. Complete episodes
    form a circular list; the last-to-first join is a genuine episode end
    followed by a genuine reset. A window cannot exceed the total retained
    rows, so no row is repeated within one sequence. No field is padded or
    rewritten. Capacity and environment budgets still count executed actions.

    The pinned DreamerV3 RSSM resets state AND incoming action on ``is_first``;
    its replay value return is cut by ``is_last``. Our outgoing-action rows
    therefore retain the existing zero final-action placeholder and the real
    reset row. The circular start rule also prevents the oldest short episode
    from being permanently confined to burn-in. Success labels are audit-only
    metadata; they never affect sampling probabilities or the learner batch.
    """

    def __init__(self, directory, capacity_steps=1_000_000, seed=0, cacheepisodes=32):
        self._outcomes = {}
        self._sampled_rows = {}
        self._sampled_terminals = {}
        self._sample_calls = 0
        self._sample_sequences = 0
        self._supervised_rows = 0
        self._supervised_resets = 0
        super().__init__(directory, capacity_steps, seed, cacheepisodes)

    @staticmethod
    def _sampling_arguments(length, burnin=0):
        if type(length) is not int or length < 1:
            raise ValueError("sequence length must be a positive integer")
        if type(burnin) is not int or not 0 <= burnin < length:
            raise ValueError("burnin must be a nonnegative integer smaller than sequence length")

    def add_episode(self, episode, *, success=None):
        if success is not None and type(success) is not bool:
            raise ValueError("success must be a native diagnostic boolean or None")
        arrays = self._validate_episode(episode)
        if success is True and not arrays["is_terminal"][-1]:
            raise ValueError("native touch success must be a terminal episode")
        record = super().add_episode(arrays)
        self._outcomes[record["id"]] = success
        atomic_json(self.directory / "index.json", self.snapshot())
        return record

    def can_sample(self, length):
        return (type(length) is int and length > 0 and
                sum(record["frames"] for record in self.episodes) >= length)

    def sample(self, batch, length, *, burnin=0):
        self._sampling_arguments(length, burnin)
        if type(batch) is not int or batch < 1:
            raise ValueError("batch must be a positive integer")
        records = self.episodes
        cumulative = np.cumsum([record["frames"] for record in records], dtype=np.int64)
        total = int(cumulative[-1]) if len(cumulative) else 0
        if total < length:
            raise ValueError("not enough total real observation rows for this sequence length")
        draws = self.rng.integers(total, size=batch)
        prototype = self._load(records[0])
        result = {name: np.empty((batch, length, *prototype[name].shape[1:]),
                                 dtype=prototype[name].dtype) for name in FIELDS}
        for row, draw in enumerate(draws):
            index = int(np.searchsorted(cumulative, draw, side="right"))
            offset = int(draw - (cumulative[index - 1] if index else 0))
            destination = 0
            while destination < length:
                record = records[index]
                episode = self._load(record)
                count = min(record["frames"] - offset, length - destination)
                for name in FIELDS:
                    result[name][row, destination:destination + count] = episode[name][offset:offset + count]
                loss_start = max(destination, burnin)
                trained = max(0, destination + count - loss_start)
                identifier = record["id"]
                self._sampled_rows[identifier] = self._sampled_rows.get(identifier, 0) + trained
                end_position = destination + record["frames"] - 1 - offset
                if loss_start <= end_position < destination + count:
                    self._sampled_terminals[identifier] = self._sampled_terminals.get(identifier, 0) + 1
                destination += count
                index = (index + 1) % len(records)
                offset = 0
        self._sample_calls += 1
        self._sample_sequences += batch
        self._supervised_rows += batch * (length - burnin)
        self._supervised_resets += int(result["is_first"][:, burnin:].sum())
        return result

    def sampling_statistics(self, length, *, burnin=0):
        """Cumulative post-burn-in delivery coverage, not successful updates.

        A learner failure after sampling can prevent an update; update counters
        remain the authoritative count. Coverage denominators use currently
        active episodes, while raw delivery counters include evicted episodes.
        """
        self._sampling_arguments(length, burnin)
        records = self.episodes
        short = [record for record in records if record["frames"] < length]
        successes = [record for record in records if self._outcomes.get(record["id"]) is True]
        short_successes = [record for record in successes if record["frames"] < length]
        covered = lambda items: sum(self._sampled_terminals.get(record["id"], 0) > 0 for record in items)
        ratio = lambda numerator, denominator: numerator / denominator if denominator else None
        return {
            "sampling_policy": SAMPLING_POLICY, "sequence_length": length, "burnin": burnin,
            "active_episodes": len(records), "active_observation_rows": sum(r["frames"] for r in records),
            "short_episodes": len(short), "short_episode_fraction": ratio(len(short), len(records)),
            "known_success_episodes": len(successes), "short_success_episodes": len(short_successes),
            "unknown_outcome_episodes": sum(self._outcomes.get(r["id"]) is None for r in records),
            "success_terminal_episodes_sampled": covered(successes),
            "success_sampling_coverage": ratio(covered(successes), len(successes)),
            "short_success_terminal_episodes_sampled": covered(short_successes),
            "short_success_sampling_coverage": ratio(covered(short_successes), len(short_successes)),
            "episode_end_sampling_coverage": ratio(covered(records), len(records)),
            "sample_calls": self._sample_calls, "sample_sequences": self._sample_sequences,
            "supervised_rows_delivered": self._supervised_rows,
            "supervised_reset_rows_delivered": self._supervised_resets,
            "coverage_semantics": "episode end row delivered after burnin; not an optimizer success claim",
            "success_label_source": "native touch success from privileged worker diagnostics, audit only",
        }

    def snapshot(self):
        snapshot = super().snapshot()
        snapshot["sampling_policy"] = SAMPLING_POLICY
        snapshot["sampling_contract"] = "uniform circular complete-episode real rows; exact fields; no repeated row per window"
        snapshot["episode_success"] = {str(r["id"]): self._outcomes.get(r["id"]) for r in self._archive}
        snapshot["sampling_counters"] = {
            "sample_calls": self._sample_calls, "sample_sequences": self._sample_sequences,
            "supervised_rows": self._supervised_rows, "supervised_resets": self._supervised_resets,
            "rows_by_episode": {str(k): v for k, v in self._sampled_rows.items()},
            "ends_by_episode": {str(k): v for k, v in self._sampled_terminals.items()},
        }
        return snapshot

    def restore(self, snapshot, *, persist=False):
        if snapshot.get("sampling_policy") != SAMPLING_POLICY:
            raise ValueError("Minecraft replay sampling policy mismatch; old/V2 samplers are not interchangeable")
        ids = {str(record["id"]) for record in snapshot["archive"]}
        outcomes = snapshot["episode_success"]
        if set(outcomes) != ids or any(value is not None and type(value) is not bool for value in outcomes.values()):
            raise ValueError("Invalid archived success audit labels")
        counters = snapshot["sampling_counters"]
        scalars = ("sample_calls", "sample_sequences", "supervised_rows", "supervised_resets")
        if set(counters) != set(scalars) | {"rows_by_episode", "ends_by_episode"}:
            raise ValueError("Invalid replay sampling counter schema")
        if any(type(counters[name]) is not int or counters[name] < 0 for name in scalars):
            raise ValueError("Invalid replay sampling counts")
        for name in ("rows_by_episode", "ends_by_episode"):
            values = counters[name]
            if not set(values) <= ids or any(type(v) is not int or v < 0 for v in values.values()):
                raise ValueError("Invalid per-episode sampling counts")
        if sum(counters["rows_by_episode"].values()) != counters["supervised_rows"]:
            raise ValueError("Replay supervised row counts disagree")
        if (counters["supervised_resets"] > counters["supervised_rows"] or
                counters["sample_calls"] > counters["sample_sequences"] or
                any(value > counters["rows_by_episode"].get(key, 0)
                    for key, value in counters["ends_by_episode"].items())):
            raise ValueError("Inconsistent replay coverage counts")
        super().restore(snapshot, persist=False)
        self._outcomes = {int(k): v for k, v in outcomes.items()}
        self._sampled_rows = {int(k): v for k, v in counters["rows_by_episode"].items()}
        self._sampled_terminals = {int(k): v for k, v in counters["ends_by_episode"].items()}
        self._sample_calls, self._sample_sequences = counters["sample_calls"], counters["sample_sequences"]
        self._supervised_rows, self._supervised_resets = counters["supervised_rows"], counters["supervised_resets"]
        if persist:
            atomic_json(self.directory / "index.json", self.snapshot())


def validate_payload(payload):
    if payload.get("format") != FORMAT or payload.get("task_namespace") != TASK_NAMESPACE:
        raise ValueError("Not a Minecraft episode-boundary checkpoint")
    required = {"config", "metadata", "counters", "backbone", "replay", "collector", "rng", "scheduler"}
    if not required <= set(payload):
        raise ValueError("Incomplete Minecraft checkpoint")
    counters, boundary = payload["counters"], payload["collector"]
    for name in ("agent_steps", "env_steps", "episodes", "update_steps"):
        if type(counters[name]) is not int or counters[name] < 0:
            raise ValueError("Invalid checkpoint counters")
    if not counters["agent_steps"] or not counters["episodes"]:
        raise ValueError("A checkpoint requires a completed nonempty episode")
    if not counters["agent_steps"] <= counters["env_steps"] <= 2*counters["agent_steps"]:
        raise ValueError("Acknowledged native-call count is outside action-repeat bounds")
    if payload["replay"]["total_steps_seen"] != counters["agent_steps"]:
        raise ValueError("Replay action count differs from collector")
    if (boundary.get("next_operation") != "reset_new_episode_and_reanchor" or
            boundary.get("engine_state_serialized") is not False or
            boundary.get("episode_complete") is not True):
        raise ValueError("Only explicit completed-episode software recovery is supported")
    length = boundary["episode_action_count"]
    if type(length) is not int or not 1 <= length <= counters["agent_steps"]:
        raise ValueError("Invalid finished episode length")
    state = boundary["dual_chain"]["state"]
    if (state["initialized"] is not True or state["timestamp"] != length or
            state["anchor_time"] != length or state["last_action"] is not None or
            state["pending"] is not None or state["missing_length"] != 0 or state["abstaining"]):
        raise ValueError("Finished chain must have an observed last frame and no pending action/gap")
    expected_seed = payload["config"]["seed"] + counters["episodes"]
    if boundary["next_episode_seed"] != expected_seed:
        raise ValueError("Next explicit Minecraft episode seed does not match schedule")
    backbone = payload["backbone"]
    if backbone.get("format") != BACKBONE_FORMAT or backbone.get("task_namespace") != TASK_NAMESPACE:
        raise ValueError("Checkpoint contains a non-Minecraft backbone")
    official = backbone["backbone"]["training"]["counters"]
    if official["updates"] != counters["update_steps"] or official["batches"] != counters["update_steps"]:
        raise ValueError("Learner counters differ from snapshot")
    target = int(max(0, counters["agent_steps"]-payload["config"]["prefill_agent_steps"])
                 * payload["config"]["train_ratio"] /
                 (payload["config"]["backbone"]["batch_size"] * payload["config"]["backbone"]["batch_length"]))
    if (payload["scheduler"]["target_updates"] != target or counters["update_steps"] > target):
        raise ValueError("Learner scheduler differs from frozen configuration")
    if (boundary["dropout"]["remaining"] != 0 or
            payload["scheduler"]["next_checkpoint"] <= counters["agent_steps"] or
            payload["scheduler"]["next_evaluation"] <= counters["agent_steps"]):
        raise ValueError("Boundary scheduler or pending missingness is invalid")


def save_checkpoint(directory, payload):
    validate_payload(payload)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"minecraft_checkpoint_{payload['counters']['agent_steps']:09d}.pkl"
    if path.exists() or path.with_suffix(".json").exists():
        raise FileExistsError(path)
    state = _host(payload)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("wb", dir=directory, delete=False, prefix=".minecraft-") as file:
            temporary = Path(file.name)
            pickle.dump({"format": FORMAT, "payload": state}, file, protocol=pickle.HIGHEST_PROTOCOL)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    info = {"format": FORMAT, "file": path.name, "sha256": sha256(path),
            "bytes": path.stat().st_size, "tree_hash": tree_hash(state),
            "counters": state["counters"], "replay_archive": state["replay"]["archive"],
            "engine_state_serialized": False, "mid_episode_restore_supported": False}
    atomic_json(path.with_suffix(".json"), info)
    atomic_json(directory / "latest.json", info)
    return path


def load_checkpoint(path, *, verify_replay=True, replay_directory=None):
    path = Path(path)
    info = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    if info.get("format") != FORMAT or info["file"] != path.name:
        raise ValueError("Minecraft checkpoint sidecar mismatch")
    if path.stat().st_size != info["bytes"] or sha256(path) != info["sha256"]:
        raise ValueError("Minecraft checkpoint SHA256/size mismatch")
    with path.open("rb") as file:
        record = pickle.load(file)  # Trusted own experiment files only.
        if file.read(1):
            raise ValueError("Checkpoint has trailing bytes")
    if record.get("format") != FORMAT or tree_hash(record["payload"]) != info["tree_hash"]:
        raise ValueError("Minecraft checkpoint format/tree hash mismatch")
    payload = record["payload"]
    validate_payload(payload)
    if payload["counters"] != info["counters"] or payload["replay"]["archive"] != info["replay_archive"]:
        raise ValueError("Checkpoint sidecar content mismatch")
    if verify_replay:
        verify_archive(Path(replay_directory) if replay_directory else path.parent.parent/"REPLAY",
                       info["replay_archive"])
    return _restore_special(payload)


def restore_chain_boundary(chain, saved, episode_action_count):
    """Restore software streams at any completed episode length, without old latents."""
    state, generators, events = chain._validated_snapshot(saved)
    if (type(episode_action_count) is not int or episode_action_count < 1 or
            state["initialized"] is not True or state["timestamp"] != episode_action_count or
            state["anchor_time"] != episode_action_count or state["real"] is None or
            state["dream"] is None or state["last_action"] is not None or
            state["pending"] is not None or state["missing_length"] or state["abstaining"]):
        raise ValueError("Cannot restore an unfinished Minecraft episode")
    chain.rngs = generators
    chain.events = deque(events, maxlen=chain.config["max_events"])
    chain.branch_counter, chain.event_counter = state["branch_counter"], state["event_counter"]
    chain.model_version = deepcopy(chain.backend.inference_version)
    chain.initialized = False
    for name in ("episode_id", "timestamp", "real", "dream", "anchor", "anchor_time",
                 "last_prediction", "last_action", "pending", "repaired_segment"):
        setattr(chain, name, None)
    chain.abstaining, chain.missing_length = False, 0
