"""Strict experiment contract and policy-independent camera-mask bank.

This module imports neither JAX nor the simulator. Arrays are hashed by content,
not by zip timestamps. A protocol fingerprint includes every registered field.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np


UPSTREAM_COMMIT = "e3f02248693a79dc8b0ebd62c93683888ddaccfe"


def canonical_json(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("utf-8")


def protocol_sha256(config: dict) -> str:
    return hashlib.sha256(canonical_json(config)).hexdigest()


def array_sha256(array) -> str:
    a = np.ascontiguousarray(array)
    if a.dtype.hasobject:
        raise ValueError("Object arrays are not evidence")
    h = hashlib.sha256(canonical_json({"dtype": a.dtype.str, "shape": list(a.shape)}))
    h.update(a.tobytes())
    return h.hexdigest()


def _same(actual, expected, label):
    # bool must not silently satisfy an integer-valued protocol field.
    if type(actual) is not type(expected) or actual != expected:
        raise ValueError(f"Protocol mismatch at {label}: expected {expected!r}")


def validate_protocol(config: dict) -> dict:
    if not isinstance(config, dict):
        raise ValueError("Protocol must be an object")
    canonical_json(config)  # Reject nonfinite JSON, even in unused fields.
    try:
        _same(config["schema_version"], 1, "schema_version")
        _same(config["protocol_id"], "dreamerv3_vs_helixwm_walker_20261026", "protocol_id")
        _same(config["official_core"]["commit"], UPSTREAM_COMMIT, "official_core.commit")
        _same(config["official_core"]["presets"], ["defaults", "dmc_vision", "size12m"], "presets")
        env, tr = config["environment"], config["training"]
        for key, expected in dict(domain="walker", task="walk", observation="RGB_only",
                                  image_size=64, camera_id=0, action_dimensions=6,
                                  action_repeat=2, episode_action_decisions=500,
                                  episode_dmc_calls=1000, dynamics_changed=False,
                                  privileged_current_state_to_policy=False).items():
            _same(env[key], expected, "environment."+key)
        for key, expected in dict(actions_per_arm=1000000, dmc_calls_per_arm=2000000,
                combined_training_actions=2000000, combined_training_dmc_calls=4000000,
                episodes_per_arm=2000, paired_training_seeds=[0], batch_size=16,
                batch_length=64, replay_context=1, custom_burnin=0,
                replay_capacity_transitions=1000000, train_ratio_positions_per_decision=256,
                prefill_actions=5000, nominal_updates_at_budget=248750,
                checkpoint_every_actions=10000, environments_per_arm=1,
                learner_image_source="corrupted_visible_RGB_only_for_both_arms",
                learner_missing_image="uint8_zeros", learner_hidden_clean_targets_allowed=False,
                arm_specific_masked_loss_removal_allowed=False,
                synthetic_candidate_images_to_learner_allowed=False).items():
            _same(tr[key], expected, "training."+key)
        _same(tr["policy_parameter_sync"]["rule"],
              "copy_at_each_episode_boundary_then_hold_for_500_actions", "policy_parameter_sync")
        arms = [(a["id"], a["controller"], a["consumes_availability_flag"]) for a in tr["arms"]]
        if arms != [("D", "black_frame_native", False), ("H", "helix_v2", True)]:
            raise ValueError("Registered D/H arms changed")
        missing = tr["training_missingness"]
        for key, expected in dict(mask_seed=710000,
                start_probability_at_eligible_timestamp=0.02,
                burst_length_uniform_inclusive=[1, 5],
                minimum_valid_observations_between_bursts=1,
                actual_contiguous_gap_max=5,
                burst_must_fit_before_final_observation=True,
                initial_and_final_observations_valid=True,
                same_mask_per_paired_episode=True, masks_apply_during_prefill=True,
                policy_reward_dependent=False).items():
            _same(missing[key], expected, "training_missingness."+key)
        seeds = env["identical_environment_contract"]["training_episode_reset_seeds"]
        _same(seeds, {"start": 100000, "count": 2000, "step": 1}, "reset_seeds")
        final = config["final_evaluation"]
        _same(final["missing_lengths_actions"], [1, 5, 10, 20, 50], "final gaps")
        _same(final["missing_observation_start"], 150, "final gap start")
        _same(final["environment_seeds"], {"start": 60000, "count": 40}, "final seeds")
        _same(final["total_episodes"], 1280, "final episodes")
        _same(final["training_updates"], 0, "evaluation updates")
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Incomplete protocol: {exc}") from exc
    return copy.deepcopy(config)


def load_protocol(path: str | Path) -> dict:
    return validate_protocol(json.loads(Path(path).read_text(encoding="utf-8-sig")))


def training_seed(config: dict, episode_index: int) -> int:
    spec = config["environment"]["identical_environment_contract"]["training_episode_reset_seeds"]
    if type(episode_index) is not int or not 0 <= episode_index < spec["count"]:
        raise ValueError("Training episode index must be in [0, 2000)")
    return spec["start"] + episode_index * spec["step"]


def validate_mask(mask, *, maximum_gap=None) -> np.ndarray:
    a = np.asarray(mask)
    if a.shape != (501,) or a.dtype != np.bool_:
        raise ValueError("An availability mask must be bool[501]")
    if not a[0] or not a[-1]:
        raise ValueError("Initial and final observations must be available")
    a = a.copy()
    if maximum_gap is not None and any(length > maximum_gap for _, length in gap_segments(a)):
        raise ValueError("Mask exceeds the registered contiguous gap limit")
    return a


def gap_segments(mask) -> list[tuple[int, int]]:
    a = np.asarray(mask, dtype=np.bool_)
    missing = np.flatnonzero(~a)
    if not len(missing):
        return []
    chunks = np.split(missing, np.flatnonzero(np.diff(missing) > 1)+1)
    return [(int(c[0]), len(c)) for c in chunks]


def mask_sha256(mask) -> str:
    return array_sha256(validate_mask(mask))


def make_mask_bank(config: dict) -> np.ndarray:
    validate_protocol(config)
    spec = config["training"]["training_missingness"]
    rng = np.random.default_rng(spec["mask_seed"])
    bank = np.ones((2000, 501), np.bool_)
    # Entire bank is determined before any actor, environment, or learner call.
    for row in bank:
        t = 1
        while t < 500:
            if rng.random() < spec["start_probability_at_eligible_timestamp"]:
                length = int(rng.integers(1, 6))
                if t + length <= 500:
                    row[t:t+length] = False
                    t += length + 1  # The following observation is forcibly valid.
                    continue
            t += 1
    return bank


def fixed_gap_mask(length: int, start: int = 150) -> np.ndarray:
    if type(length) is not int or type(start) is not int or length < 0:
        raise ValueError("Gap start/length must be nonnegative integers")
    if not 1 <= start < 500 or start+length > 500:
        raise ValueError("Gap must exclude reset and terminal observations")
    mask = np.ones(501, np.bool_)
    mask[start:start+length] = False
    return mask


def atomic_json(path: str | Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(value, sort_keys=True, indent=2, allow_nan=False).encode()+b"\n"
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("wb", dir=path.parent, prefix=".json-", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def save_mask_bank(config: dict, directory: str | Path) -> dict:
    """Create once, or strictly verify the already saved bank; never replace it."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    bank = make_mask_bank(config)
    path = directory / "mask_bank.npy"
    if path.exists():
        previous = np.load(path, allow_pickle=False)
        if previous.dtype != bank.dtype or not np.array_equal(previous, bank):
            raise ValueError("Existing mask bank does not match this protocol")
    else:
        # Exclusive creation rejects accidental overwrites by a second writer.
        with path.open("xb") as stream:
            np.save(stream, bank, allow_pickle=False)
            stream.flush(); os.fsync(stream.fileno())
    hist = {}
    for row in bank:
        for _, length in gap_segments(row):
            hist[str(length)] = hist.get(str(length), 0)+1
    report = {"format_version": 1, "protocol_sha256": protocol_sha256(config),
              "file": path.name, "shape": list(bank.shape), "dtype": "bool",
              "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
              "content_sha256": array_sha256(bank), "mask_seed": 710000,
              "episode_reset_seeds": [training_seed(config, i) for i in range(2000)],
              "row_sha256": [mask_sha256(row) for row in bank],
              "missing_observations": int((~bank).sum()), "actual_gap_histogram": hist}
    metadata = directory / "mask_bank.json"
    if metadata.exists():
        if json.loads(metadata.read_text()) != report:
            raise ValueError("Existing mask-bank identity differs")
    else:
        atomic_json(metadata, report)
    return report
