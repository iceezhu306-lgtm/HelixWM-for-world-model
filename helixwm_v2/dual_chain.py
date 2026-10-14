"""Causal, bounded dual-chain state repair around a frozen inference backbone.

The official decoder supplies an energy, not a calibrated observation density.
Prior-proposed candidates are weighted ONLY by newly available observation
energy: multiplying their transition priors again would count the prior twice.
Raw observations/actions are never edited. This module performs no learning.
"""

from __future__ import annotations

from collections import deque
import copy
import hashlib
from typing import Any

import numpy as np


def _copy(value: Any) -> Any:
    return copy.deepcopy(value)


def _digest(value: Any) -> str:
    digest = hashlib.sha256()

    def visit(item: Any) -> None:
        if isinstance(item, dict):
            for key in sorted(item):
                digest.update(str(key).encode())
                visit(item[key])
        elif isinstance(item, (tuple, list)):
            for entry in item:
                visit(entry)
        else:
            array = np.asarray(item)
            digest.update(str(array.dtype).encode())
            digest.update(str(array.shape).encode())
            digest.update(np.ascontiguousarray(array).tobytes())

    visit(value)
    return digest.hexdigest()


def _batch(states: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {key: np.concatenate([state[key] for state in states], axis=0)
            for key in states[0]}


class DualChain:
    """One real chain and bounded prior-proposed diagnostic trajectories.

    Backend states are batch-first dictionaries of NumPy arrays, batch size 1.
    Backend API: initial(batch), observe(state, action, image, is_first, rng),
    prior(state, action, rng), score(batched_states, image), act(state,
    deterministic, rng). Independent RNG streams isolate ordinary filtering
    and actions from extra shadow/repair inference.

    A model version is frozen for an entire episode. A different version is
    accepted only at reset; step never silently combines versioned segments.
    In repair mode an accepted candidate retains its WHOLE ancestral path,
    including recurrent memory. Later evidence revises the recorded inferred
    segment only now; it never rewrites earlier control decisions or raw data.
    """

    FORMAT_VERSION = 1

    def __init__(self, backend: Any, *, seed: int = 0, model_version: Any = 0,
                 mode: str = "repair", max_candidates: int = 8,
                 max_missing: int = 5, score_temperature: float = 0.01,
                 score_normalization: float = 12288.0,
                 low_ess_threshold: float = 2.0, max_events: int = 128):
        if mode not in {"disabled", "shadow", "repair"}:
            raise ValueError("mode must be disabled, shadow, or repair")
        if not 1 <= max_candidates <= 8 or not 1 <= max_missing <= 5:
            raise ValueError("candidate and missing-length budgets must be 1..8 and 1..5")
        if not np.isfinite(score_temperature) or score_temperature <= 0:
            raise ValueError("score_temperature must be finite and positive")
        if not np.isfinite(score_normalization) or score_normalization <= 0:
            raise ValueError("score_normalization must be finite and positive")
        if not np.isfinite(low_ess_threshold) or low_ess_threshold < 1:
            raise ValueError("low_ess_threshold must be finite and at least 1")
        if max_events < 1:
            raise ValueError("max_events must be positive")
        self.backend = backend
        self.config = dict(mode=mode, max_candidates=max_candidates,
                           max_missing=max_missing, score_temperature=score_temperature,
                           score_normalization=score_normalization,
                           low_ess_threshold=low_ess_threshold, max_events=max_events)
        streams = np.random.SeedSequence(seed).spawn(4)
        self.rngs = {name: np.random.default_rng(stream) for name, stream in
                     zip(("real", "actor", "dream", "repair"), streams)}
        self.model_version = _copy(model_version)
        self.events: deque[dict] = deque(maxlen=max_events)
        self.initialized = False
        self.episode_id = None
        self.timestamp = None
        self.real = self.dream = self.anchor = None
        self.anchor_time = None
        self.last_prediction = None
        self.last_action = None
        self.pending = None
        self.abstaining = False
        self.missing_length = 0
        self.repaired_segment = None
        self.branch_counter = 0
        self.event_counter = 0

    @property
    def state(self) -> dict[str, np.ndarray]:
        if not self.initialized:
            raise RuntimeError("reset must precede state access")
        return _copy(self.real)

    def _event(self, kind: str, **fields: Any) -> None:
        self.event_counter += 1
        self.events.append({"event": kind, "event_id": self.event_counter,
                            "episode_id": _copy(self.episode_id),
                            "model_version": _copy(self.model_version), **fields})

    def pop_events(self) -> list[dict]:
        result = _copy(list(self.events))
        self.events.clear()
        return result

    def _check_backend_version(self, version: Any) -> None:
        if hasattr(self.backend, "inference_version") and self.backend.inference_version != version:
            raise ValueError("backend inference snapshot does not match the declared model version")

    def reset(self, observation: np.ndarray, *, episode_id: Any,
              timestamp: int = 0, model_version: Any = None,
              valid: bool = True) -> dict[str, np.ndarray]:
        if not valid:
            raise ValueError("an episode needs an observed initial anchor")
        if not isinstance(timestamp, int) or timestamp < 0:
            raise ValueError("timestamp must be a nonnegative integer")
        self._check_backend_version(self.model_version if model_version is None else model_version)
        if self.pending is not None:
            self._event("repair_abstained", reason="episode_reset",
                        evidence_cutoff=self.timestamp,
                        anchor_time=self.anchor_time)
        if model_version is not None:
            self.model_version = _copy(model_version)
        self.episode_id, self.timestamp = _copy(episode_id), timestamp
        initial = self.backend.initial(batch=1)
        # The first observation is a normal backend reset, never a generated image.
        zeros = np.zeros((1, getattr(self.backend, "action_dim", 6)), np.float32)
        self.real = self.backend.observe(_copy(initial), zeros, _copy(observation),
                                         is_first=True, rng=self.rngs["real"])
        self.dream = _copy(self.real)
        self.anchor = _copy(self.real)
        self.anchor_time = timestamp
        self.last_prediction = self.last_action = self.pending = None
        self.repaired_segment = None
        self.abstaining, self.missing_length = False, 0
        self.initialized = True
        return self.state

    def act(self, deterministic: bool = False) -> np.ndarray:
        if not self.initialized:
            raise RuntimeError("reset must precede act")
        self._check_backend_version(self.model_version)
        if self.last_action is not None:
            raise RuntimeError("previous action has not been paired with an outcome")
        result = np.asarray(self.backend.act(_copy(self.real),
                            deterministic=deterministic, rng=self.rngs["actor"]),
                            dtype=np.float32)
        if result.ndim == 1:
            result = result[None]
        if result.shape != (1, getattr(self.backend, "action_dim", 6)):
            raise ValueError("backend action must have shape [1, action_dim]")
        if not np.isfinite(result).all():
            raise ValueError("backend action must be finite")
        self.last_action = np.clip(result, -1.0, 1.0).copy()
        return self.last_action.copy()

    def _validate_step(self, action: np.ndarray, episode_id: Any, timestamp: int,
                       model_version: Any) -> np.ndarray:
        if not self.initialized:
            raise RuntimeError("reset must precede step")
        if episode_id != self.episode_id:
            raise ValueError("episode mismatch; explicit reset is required")
        if timestamp != self.timestamp + 1:
            raise ValueError("timestamp must follow the previous observation exactly")
        if model_version != self.model_version:
            raise ValueError("model version changed inside an episode; re-encode or reset")
        self._check_backend_version(self.model_version)
        result = np.asarray(action, dtype=np.float32)
        if result.ndim == 1:
            result = result[None]
        if self.last_action is None or result.shape != self.last_action.shape:
            raise ValueError("step requires the action previously issued by act")
        if not np.array_equal(result, self.last_action):
            raise ValueError("executed action differs from the recorded issued action")
        return result.copy()

    def _start_candidates(self) -> None:
        self.branch_counter += 1
        count = self.config["max_candidates"]
        self.pending = {
            "branch_group": self.branch_counter,
            "anchor_time": self.timestamp,
            "anchor_state": _copy(self.real),
            "anchor_state_sha256": _digest(self.real),
            "episode_id": _copy(self.episode_id),
            "model_version": _copy(self.model_version),
            "actions": [], "times": [],
            "states": [_copy(self.real) for _ in range(count)],
            "paths": [[] for _ in range(count)],
            "log_weights": np.full(count, -np.log(count), dtype=np.float64),
        }

    def _advance_candidates(self, action: np.ndarray, timestamp: int) -> None:
        pending = self.pending
        pending["actions"].append(action.copy())
        pending["times"].append(timestamp)
        for index, state in enumerate(pending["states"]):
            candidate = self.backend.prior(_copy(state), action.copy(),
                                           rng=self.rngs["repair"])
            pending["states"][index] = _copy(candidate)
            pending["paths"][index].append(_copy(candidate))

    def _normal_observe(self, previous: dict, action: np.ndarray,
                        observation: np.ndarray) -> dict:
        return self.backend.observe(_copy(previous), action.copy(), _copy(observation),
                                    is_first=False, rng=self.rngs["real"])

    def step(self, observation: np.ndarray | None, executed_action: np.ndarray, *,
             episode_id: Any, timestamp: int, model_version: Any = None,
             valid: bool = True) -> dict[str, np.ndarray]:
        version = self.model_version if model_version is None else model_version
        action = self._validate_step(executed_action, episode_id, timestamp, version)
        previous = _copy(self.real)
        mode = self.config["mode"]
        if valid and observation is None:
            raise ValueError("a valid observation cannot be absent")
        if mode != "disabled":
            self.last_prediction = self.backend.prior(_copy(self.dream), action.copy(),
                                                       rng=self.rngs["dream"])
        else:
            self.last_prediction = None
        if not valid:
            self.missing_length += 1
            if mode != "disabled" and not self.abstaining:
                if self.pending is None:
                    self._start_candidates()
                if self.missing_length > self.config["max_missing"]:
                    self._event("repair_abstained", reason="missing_segment_budget_exceeded",
                                anchor_time=self.pending["anchor_time"],
                                evidence_cutoff=timestamp, missing_length=self.missing_length)
                    self.pending = None
                    self.abstaining = True
                else:
                    self._advance_candidates(action, timestamp)
            # Missing pixels are not inspected, encoded, decoded, or scored.
            if mode == "repair" and self.pending is not None:
                weights = np.exp(self.pending["log_weights"])
                chosen = int(self.rngs["repair"].choice(len(weights), p=weights))
                self.real = _copy(self.pending["states"][chosen])
            else:
                self.real = self.backend.prior(previous, action.copy(), rng=self.rngs["real"])
            self.dream = _copy(self.last_prediction if self.last_prediction is not None else self.real)
        else:
            if self.pending is not None:
                self._advance_candidates(action, timestamp)
                scores = np.asarray(self.backend.score(_batch(self.pending["states"]),
                                                       _copy(observation)), dtype=np.float64).reshape(-1)
                if scores.shape != self.pending["log_weights"].shape:
                    raise ValueError("candidate score count mismatch")
                with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                    log_weights = self.pending["log_weights"] + scores / (
                        self.config["score_normalization"] * self.config["score_temperature"])
                if np.isfinite(log_weights).all():
                    log_weights -= np.max(log_weights)
                    weights = np.exp(log_weights)
                    weights /= weights.sum()
                    ess = float(1.0 / np.square(weights).sum())
                    chosen = int(self.rngs["repair"].choice(len(weights), p=weights))
                    selected_path = _copy(self.pending["paths"][chosen])
                    self.repaired_segment = {
                        "episode_id": _copy(self.episode_id), "model_version": _copy(self.model_version),
                        "anchor_time": self.pending["anchor_time"],
                        "times": _copy(self.pending["times"]), "states": selected_path,
                        "evidence_cutoff": timestamp, "available_at": timestamp,
                        "provenance": "retrospective_candidate_path_with_causal_current_boundary",
                        "raw_observations_modified": False,
                    }
                    if mode == "repair":
                        self.real = _copy(self.pending["states"][chosen])
                    else:
                        self.real = self._normal_observe(previous, action, observation)
                    self._event("repair_reconnected" if mode == "repair" else "shadow_repair_candidate",
                                branch_group=self.pending["branch_group"],
                                anchor_time=self.pending["anchor_time"],
                                anchor_state_sha256=self.pending["anchor_state_sha256"],
                                evidence_cutoff=timestamp, available_at=timestamp,
                                segment_times=_copy(self.pending["times"]),
                                action_sequence_sha256=_digest(self.pending["actions"]),
                                observation_sha256=_digest(observation),
                                candidate_energy_scores=scores.tolist(),
                                candidate_scores_per_pixel=(scores / self.config["score_normalization"]).tolist(),
                                candidate_weights=weights.tolist(), effective_sample_size=ess,
                                low_effective_sample_size=ess < self.config["low_ess_threshold"],
                                selection="one_weighted_candidate_trajectory_not_posterior_certainty",
                                selected_candidate=chosen,
                                candidate_path_sha256=[_digest(path) for path in self.pending["paths"]],
                                selected_path_sha256=_digest(selected_path),
                                score_kind=getattr(self.backend, "score_kind", "backend_score"),
                                score_normalization=self.config["score_normalization"],
                                score_temperature=self.config["score_temperature"],
                                prior_counted_again=False, observed_teacher_qualified=False,
                                control_uses_repair=(mode == "repair"))
                else:
                    self.real = self._normal_observe(previous, action, observation)
                    self._event("repair_abstained", reason="nonfinite_candidate_evidence",
                                anchor_time=self.pending["anchor_time"], evidence_cutoff=timestamp)
            else:
                self.real = self._normal_observe(previous, action, observation)
            # Real-to-Dream re-anchoring uses current available evidence only.
            self.dream = _copy(self.real)
            self.anchor, self.anchor_time = _copy(self.real), timestamp
            self.pending = None
            self.abstaining, self.missing_length = False, 0
        self.timestamp = timestamp
        self.last_action = None
        return self.state

    def state_dict(self) -> dict:
        """Complete same-backend recovery state, suitable for trusted pickle."""
        fields = ("model_version", "initialized", "episode_id", "timestamp", "real", "dream",
                  "anchor", "anchor_time", "last_prediction", "last_action", "pending",
                  "abstaining", "missing_length", "repaired_segment", "branch_counter", "event_counter")
        return {"format_version": self.FORMAT_VERSION, "config": _copy(self.config),
                "state": {name: _copy(getattr(self, name)) for name in fields},
                "rngs": {name: _copy(rng.bit_generator.state) for name, rng in self.rngs.items()},
                "events": _copy(list(self.events))}

    def _validated_snapshot(self, saved: dict) -> tuple[dict, dict, list]:
        """Validate every RNG before replacing any live state or generator."""
        if saved.get("format_version") != self.FORMAT_VERSION or saved.get("config") != self.config:
            raise ValueError("dual-chain snapshot format/configuration mismatch")
        if set(saved.get("rngs", {})) != set(self.rngs):
            raise ValueError("dual-chain RNG snapshot is incomplete")
        expected = set(self.state_dict()["state"])
        if set(saved.get("state", {})) != expected:
            raise ValueError("dual-chain state snapshot is incomplete")
        state = _copy(saved["state"])
        for name in ("branch_counter", "event_counter"):
            if isinstance(state[name], bool) or not isinstance(state[name], int) or state[name] < 0:
                raise ValueError("dual-chain lineage counters must be nonnegative integers")
        events = saved.get("events")
        if not isinstance(events, list) or len(events) > self.config["max_events"]:
            raise ValueError("dual-chain event snapshot must be a bounded list")
        if not all(isinstance(event, dict) for event in events):
            raise ValueError("dual-chain event entries must be dictionaries")
        generators = {}
        for name, value in saved["rngs"].items():
            try:
                generator = np.random.default_rng()
                generator.bit_generator.state = _copy(value)
            except (TypeError, ValueError, KeyError) as error:
                raise ValueError(f"invalid dual-chain {name} RNG snapshot") from error
            generators[name] = generator
        return state, generators, _copy(events)

    def load_state_dict(self, saved: dict) -> None:
        state, generators, events = self._validated_snapshot(saved)
        self._check_backend_version(state["model_version"])
        for name, value in state.items():
            setattr(self, name, value)
        self.rngs = generators
        self.events = deque(events, maxlen=self.config["max_events"])

    def restore_completed_episode(self, saved: dict) -> dict:
        """Restore RNG/lineage continuity across a formal 500-action boundary.

        The checkpoint may pin NEW inference parameters while retaining its
        finished OLD chain only as audit evidence. That old latent state must
        never be loaded as an active state under the new version. Require the
        explicit completed-episode boundary, retain every RNG and lineage/event
        counter, discard active states, and require reset on the next raw initial
        observation. The caller restores the environment/backbone separately and
        must verify its checkpoint's next_operation is reset_new_episode_and_reanchor.

        This method takes no action, samples no model state, and does not reset
        an environment. Existing snapshot schema version 1 remains unchanged.
        """
        state, generators, events = self._validated_snapshot(saved)
        if state["initialized"] is not True or state["timestamp"] != 500:
            raise ValueError("completed-episode recovery requires exactly 500 observed action outcomes")
        if state["anchor_time"] != 500 or state["real"] is None or state["dream"] is None:
            raise ValueError("completed-episode recovery requires a valid final observation anchor")
        if state["last_action"] is not None:
            raise ValueError("completed-episode recovery cannot discard an issued pending action")
        if state["pending"] is not None or state["missing_length"] != 0 or state["abstaining"]:
            raise ValueError("completed-episode recovery cannot discard an unresolved missing segment")
        current_version = _copy(getattr(self.backend, "inference_version", self.model_version))
        audit = {"next_operation": "reset_new_episode_and_reanchor",
                 "completed_episode_id": _copy(state["episode_id"]),
                 "completed_timestamp": 500,
                 "completed_chain_model_version": _copy(state["model_version"]),
                 "next_inference_model_version": _copy(current_version),
                 "completed_real_state_sha256": _digest(state["real"]),
                 "old_latent_state_reused": False,
                 "rng_and_lineage_counters_restored": True}
        # Validation above is complete. No saved latent/anchor/prediction can
        # survive this transition into the differently versioned model.
        self.model_version = current_version
        self.rngs = generators
        self.events = deque(events, maxlen=self.config["max_events"])
        self.branch_counter = state["branch_counter"]
        self.event_counter = state["event_counter"]
        self.initialized = False
        for name in ("episode_id", "timestamp", "real", "dream", "anchor", "anchor_time",
                     "last_prediction", "last_action", "pending", "repaired_segment"):
            setattr(self, name, None)
        self.abstaining, self.missing_length = False, 0
        return audit
