"""Validate and enumerate a design; never import a model, connect, or run an episode."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DESIGN = ROOT / "configs" / "continuity_study_v3.json"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def integer(value, label: str, minimum: int = 0) -> int:
    require(type(value) is int and value >= minimum, f"invalid {label}")
    return value


def seed_range(spec: dict, label: str) -> list[int]:
    start = integer(spec["start"], label + " start")
    count = integer(spec["count"], label + " count", 1)
    return list(range(start, start + count))


def canonical_hash(value) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"),
                     ensure_ascii=True, allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def make_availability(horizon: int, start: int, length: int) -> list[bool]:
    """x_0 through x_H; missing x_t affects the following action a_t."""
    require(0 < start < horizon, "gap must start after reset and before final observation")
    require(0 <= length and start + length < horizon,
            "gap must end before the final observation")
    return [not (start <= t < start + length) for t in range(horizon + 1)]


def build_plan(design: dict) -> dict:
    require(design["schema_version"] == 1, "unsupported design schema")
    require(design["status"] == "DESIGN_ONLY", "this utility accepts DESIGN_ONLY inputs")
    for field in ("runner_implemented", "execution_authorized", "training_updates_allowed"):
        require(design[field] is False, f"{field} must remain false")
    require(design["training_update_budget"] == 0, "training budget must be zero")
    require(design["launch_command"] is None, "a design cannot contain a launch command")
    identity = design["identity"]
    require(identity["reference_source_commit"] == "80cb3555586d01e0e7b0e8d362c27e540032d901",
            "reference source must remain the accepted V2 source")
    require(re.fullmatch(r"[0-9a-f]{64}", identity.get("checkpoint_sha256") or "") is not None,
            "bind the accepted checkpoint SHA256 before enumerating a design")
    require(identity["future_runner_commit"] is None, "no new runner is implemented here")
    require(identity["preserve_reference_runtime_unchanged"] is True,
            "preserve the accepted runtime")
    env = design["environment"]
    horizon = integer(env["episode_action_decisions"], "episode horizon", 1)
    repeat = integer(env["action_repeat"], "action repeat", 1)
    require(repeat == 2 and env["episode_dmc_calls"] == horizon * repeat,
            "environment/action units are inconsistent")
    require(env["privileged_current_state_to_policy"] is False,
            "privileged current state cannot enter the policy")
    require(env["physical_dynamics_changed_in_phase_a"] is False,
            "dynamics changes require a different study")
    phase = design["phase_a"]
    schedule = phase["schedule"]
    require(schedule["events_per_episode"] == 1, "phase A specifies one isolated event")
    require(schedule["reward_or_policy_dependent"] is False,
            "the schedule must be exogenous")
    require(schedule["initial_and_final_observations_valid"] is True,
            "endpoint observations must remain valid")
    lengths = schedule["missing_lengths_action_decisions"]
    require(lengths == [1, 5, 10, 20, 50], "do not silently change registered durations")
    start = integer(schedule["missing_observation_start"], "gap start", 1)
    pre = integer(design["metrics"]["pre_fault_actions"], "pre-fault window", 1)
    recovery = integer(design["metrics"]["recovery_window_actions"], "recovery window", 1)
    require(start >= pre and start + max(lengths) + recovery <= horizon,
            "metric windows do not fit the episode")
    ref = phase["candidate_reference"]
    require((ref["max_candidates"], ref["max_missing"], ref["score_temperature"],
             ref["score_normalization"]) == (8, 5, 0.01, 12288),
            "the exact V2 reference inference settings changed")
    require(phase["combined_variant_included"] is False and
            phase["expanded_candidate_horizon_included"] is False,
            "combined changes and expanded horizons are separate studies")
    arms = phase["arms"]
    arm_ids = [a["id"] for a in arms]
    expected = {"clean", "black_frame", "prior_only", "v2_exact", "locked_ancestor",
                "posterior_return_once"}
    require(len(arm_ids) == len(expected) and set(arm_ids) == expected,
            "missing, duplicate, or unregistered arm")
    arm_map = {a["id"]: a for a in arms}
    for arm in ("locked_ancestor", "posterior_return_once"):
        require(arm_map[arm]["new_logic_implemented"] is False,
                "proposal implementation cannot be claimed by this planner")
    require(arm_map["posterior_return_once"]["score_return_frame_before_posterior"] is False,
            "the alternate return rule cannot double-count return evidence")
    split_specs = phase["splits"]
    splits = {
        "design": seed_range(split_specs["design_environment_seeds"], "design seeds"),
        "held_out": seed_range(split_specs["held_out_environment_seeds"], "held-out seeds"),
    }
    excluded = set(seed_range(split_specs["excluded_historical_final_seeds"], "historical seeds"))
    require(not set(splits["design"]) & set(splits["held_out"]), "design and held-out seeds overlap")
    require(not (set(splits["design"]) | set(splits["held_out"])) & excluded,
            "historical final seeds cannot be reused")
    require(split_specs["freeze_source_and_analysis_before_held_out"] is True and
            split_specs["held_out_reuse_for_tuning_allowed"] is False,
            "held-out evaluation must remain untouched during development")
    masks = {}
    for length in [0] + lengths:
        mask = make_availability(horizon, start, length)
        masks[str(length)] = {
            "available": mask,
            "masked_observations": sum(not x for x in mask),
            "sha256": hashlib.sha256(bytes(mask)).hexdigest(),
            "encoding": "one byte per observation x_0..x_H; 1=available, 0=missing",
        }
    jobs = []
    for split, seeds in splits.items():
        for seed in seeds:
            for arm in arm_ids:
                for length in ([0] if arm == "clean" else lengths):
                    jobs.append({
                        "case_id": f"{split}-seed{seed}-{arm}-gap{length}",
                        "split": split, "environment_seed": seed,
                        "actor_seed": 2000000 + seed, "arm": arm,
                        "missing_length": length, "schedule_key": str(length),
                        "availability_sha256": masks[str(length)]["sha256"],
                        "action_upper_bound": horizon,
                        "dmc_call_upper_bound": horizon * repeat,
                        "status": "PLANNED_NOT_EXECUTED",
                    })
    per_seed = 1 + (len(arms) - 1) * len(lengths)
    counts = {split: len(seeds) * per_seed for split, seeds in splits.items()}
    declared = phase["evaluation_cost_upper_bound"]
    observed = {
        "episodes_per_environment_seed": per_seed,
        "clean_episodes_per_environment_seed": 1,
        "design_episodes": counts["design"], "held_out_episodes": counts["held_out"],
        "total_action_decisions": len(jobs) * horizon,
        "total_dmc_calls": len(jobs) * horizon * repeat, "training_updates": 0,
    }
    for key, value in observed.items():
        require(declared[key] == value, f"declared cost mismatch: {key}")
    return {
        "status": "DESIGN_VALIDATED_NOT_EXECUTED", "runner_implemented": False,
        "execution_authorized": False, "model_loaded": False,
        "design_sha256": canonical_hash(design),
        "checkpoint_sha256": identity["checkpoint_sha256"],
        "summary": dict(observed, total_episodes=len(jobs)),
        "schedules": masks, "cases": jobs,
        "limitation": "Plan validation does not verify a runner, live server, model reload, or control performance.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path, default=DEFAULT_DESIGN)
    parser.add_argument("--output", type=Path,
                        help="Optional NEW JSON plan file; never overwritten; no experiment is executed")
    args = parser.parse_args()
    plan = build_plan(json.loads(args.design.read_text(encoding="utf-8-sig")))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(plan, handle, indent=2, ensure_ascii=True, allow_nan=False)
            handle.write("\n")
    print(json.dumps({key: plan[key] for key in (
        "status", "runner_implemented", "execution_authorized", "model_loaded",
        "design_sha256", "checkpoint_sha256", "summary", "limitation")}, indent=2))


if __name__ == "__main__":
    main()
