"""Frozen continuity-study analysis. No model, simulator, or network imports.

Rows refer to action a_t and its reward, not the returned observation x_(t+1).
Only the primary reward estimand imputes the unobserved tail of a technical
failure with zero. Failure flags are observed, false after a complete window,
or unknown; imputed rewards never manufacture or erase observed failures.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np


FAULT_ARMS = ("black_frame", "prior_only", "v2_exact", "locked_ancestor", "posterior_return_once")
TRACE_FIELDS = ("action_index", "reward", "env_steps", "actions", "availability",
                "torso_height", "torso_upright", "latency_s")


def _require(value, message):
    if not value:
        raise ValueError(message)


def _finite(values):
    x = np.asarray(values, dtype=np.float64)
    _require(x.ndim == 1 and x.size > 0 and np.isfinite(x).all(), "finite nonempty vector required")
    return x


def bootstrap_mean_ci(values, *, seed=4000000, draws=10000, level=.95):
    """Each value is one whole seed cluster, already averaged over durations."""
    x = _finite(values)
    _require(draws >= 1 and 0 < level < 1, "invalid bootstrap settings")
    rng = np.random.default_rng(seed)
    means = np.empty(draws, np.float64)
    for begin in range(0, draws, 4096):
        stop = min(draws, begin + 4096)
        means[begin:stop] = x[rng.integers(len(x), size=(stop-begin, len(x)))].mean(axis=1)
    return np.quantile(means, [(1-level)/2, 1-(1-level)/2]).tolist()


def sign_randomization_p(values, *, seed=4000000, draws=100000):
    """One-sided proposal-minus-comparator test; negative is the alternative.

    This relies on the registered within-pair exchangeability assumption. The
    plus-one Monte Carlo correction prevents reporting a sampled p-value of 0.
    """
    x = _finite(values)
    _require(draws >= 1, "positive randomization draw count required")
    observed = float(x.mean())
    rng, count = np.random.default_rng(seed), 0
    tolerance = 1e-12 * max(1., float(np.abs(x).max()))
    for begin in range(0, draws, 4096):
        n = min(4096, draws-begin)
        signs = 2*rng.integers(0, 2, size=(n, len(x)), dtype=np.int8)-1
        count += int(((signs*x).mean(axis=1) <= observed+tolerance).sum())
    return float((count+1)/(draws+1))


def holm_adjust(pvalues):
    """Holm step-down correction, returned in original registered order."""
    p = np.asarray(pvalues, np.float64)
    _require(p.ndim == 1 and len(p) and np.isfinite(p).all() and ((0 <= p) & (p <= 1)).all(),
             "invalid p-values")
    order = np.argsort(p, kind="stable")
    result, previous = np.empty_like(p), 0.
    for rank, index in enumerate(order):
        previous = max(previous, min(1., (len(p)-rank)*float(p[index])))
        result[index] = previous
    return result.tolist()


def clopper_pearson_upper(k, n, confidence=.95):
    """One-sided exact binomial upper bound without requiring SciPy."""
    _require(type(k) is int and type(n) is int and 0 <= k <= n and n > 0 and 0 < confidence < 1,
             "invalid binomial counts/confidence")
    if k == n:
        return 1.
    alpha = 1-confidence
    if k == 0:
        return float(-math.expm1(math.log(alpha)/n))
    lo, hi = 0., 1.
    for _ in range(80):
        p = (lo+hi)/2
        terms = [math.lgamma(n+1)-math.lgamma(i+1)-math.lgamma(n-i+1)
                 + i*math.log(p) + (n-i)*math.log1p(-p) for i in range(k+1)]
        maximum = max(terms)
        cdf = math.exp(maximum)*sum(math.exp(v-maximum) for v in terms)
        if cdf > alpha:
            lo = p
        else:
            hi = p
    return float((lo+hi)/2)


def _run_exists(flags, length):
    current = 0
    for flag in flags:
        current = current+1 if bool(flag) else 0
        if current >= length:
            return True
    return False


def _window_failure(flags, start, end, consecutive):
    observed = flags[start:min(end, len(flags))]
    if _run_exists(observed, consecutive):
        return True
    return False if len(flags) >= end else None


def _validated_trace(summary, trace, design):
    horizon = int(design["environment"]["episode_action_decisions"])
    repeat = int(design["environment"]["action_repeat"])
    _require(summary["status"] in ("complete", "technical_failure"), "unknown case status")
    arrays = {k: np.asarray(trace[k]) for k in TRACE_FIELDS}
    reward = arrays["reward"].astype(np.float64)
    n = len(reward)
    _require(reward.shape == (n,) and 0 <= n <= horizon, "invalid reward shape or episode length")
    for field in ("action_index", "env_steps", "torso_height", "torso_upright", "latency_s"):
        _require(arrays[field].shape == (n,), f"trace length mismatch: {field}")
    _require(np.array_equal(arrays["action_index"], np.arange(n)), "action indices must be a contiguous zero-based prefix")
    _require(arrays["actions"].shape == (n, 6), "actions must have shape [N,6]")
    for field in ("reward", "env_steps", "actions", "torso_height", "torso_upright", "latency_s"):
        _require(np.isfinite(arrays[field]).all(), f"nonfinite recorded trace: {field}")
    calls = arrays["env_steps"]
    _require(((calls == np.floor(calls)) & (calls > 0) & (calls <= repeat)).all(), "invalid actual DMC call counts")
    _require((np.abs(arrays["actions"]) <= 1).all(), "recorded action outside bounds")
    _require((arrays["latency_s"] >= 0).all(), "negative latency")
    _require(((reward >= -1e-8) & (reward <= calls+1e-8)).all(), "Walker reward outside [0,actual DMC calls]")
    actual_actions, actual_calls = summary["actual_eval_agent_steps"], summary["actual_eval_dmc_calls"]
    _require(type(actual_actions) is int and type(actual_calls) is int
             and n <= actual_actions <= horizon and int(calls.sum()) <= actual_calls <= horizon*repeat,
             "summary/trace counters disagree")
    _require(summary.get("valid_reward_rows", n) == n, "valid reward prefix counter mismatch")
    _require(math.isclose(float(summary["episode_return"]), float(reward.sum()), abs_tol=1e-6, rel_tol=1e-10),
             "summary return disagrees with raw rewards")
    if summary["status"] == "complete":
        _require(n == horizon and actual_actions == n and actual_calls == int(calls.sum()) == horizon*repeat
                 and summary.get("actual_interaction_counters_complete", True) is True,
                 "complete case lacks the full observed budget")
    arm = summary["arm"]
    _require(arm in ("clean",)+FAULT_ARMS, "unknown arm")
    gap = int(summary["missing_length"])
    start = int(design["phase_a"]["schedule"]["missing_observation_start"])
    _require(summary.get("missing_start", start) == start, "case gap start differs from design")
    if arm == "clean":
        _require(gap == 0, "clean case must have gap zero")
    else:
        smoke_parity = (summary.get("stage", summary.get("split")) == "smoke" and gap == 0
                        and arm in ("v2_exact", "locked_ancestor", "posterior_return_once"))
        _require(smoke_parity or gap in design["phase_a"]["schedule"]["missing_lengths_action_decisions"],
                 "unregistered gap length")
    _require(start >= design["metrics"]["pre_fault_actions"]
             and start+gap+design["metrics"]["recovery_window_actions"] <= horizon,
             "registered metric windows do not fit episode")
    mask = arrays["availability"]
    _require(mask.shape == (horizon+1,) and mask.dtype == np.bool_, "mask must be boolean x0..xH")
    expected = np.ones(horizon+1, dtype=bool)
    if arm != "clean":
        expected[start:start+gap] = False
    _require(np.array_equal(mask, expected), "availability differs from exact registered schedule")
    digest = hashlib.sha256(mask.astype(np.uint8).tobytes()).hexdigest()
    _require(summary["availability_sha256"] == digest, "availability SHA256 mismatch")
    u = reward / calls.astype(np.float64) if n else np.empty(0, np.float64)
    primary_u = np.zeros(horizon, np.float64)
    primary_u[:n] = u
    return arrays, u, primary_u


def case_metrics(summary, trace, design, clean_trace=None, clean_summary=None):
    """Recompute a case; paired primary uses the same seed's clean trajectory."""
    arrays, u, primary_u = _validated_trace(summary, trace, design)
    if summary["arm"] == "clean":
        clean_summary, clean_trace = summary, trace
    _require(clean_summary is not None and clean_trace is not None, "paired clean reference is required")
    _require(clean_summary["arm"] == "clean" and clean_summary["environment_seed"] == summary["environment_seed"]
             and clean_summary.get("stage", clean_summary.get("split")) == summary.get("stage", summary.get("split")),
             "clean reference identity mismatch")
    for field in ("chain_root_seed", "inference_version"):
        if field in summary or field in clean_summary:
            _require(summary.get(field) == clean_summary.get(field), f"paired {field} mismatch")
    _, _, reference_u = _validated_trace(clean_summary, clean_trace, design)
    metrics = design["metrics"]
    start = int(design["phase_a"]["schedule"]["missing_observation_start"])
    length = int(summary["missing_length"])
    returned = start+length
    end = returned+int(metrics["recovery_window_actions"])
    pre_start = start-int(metrics["pre_fault_actions"])
    n = len(u)
    pre = float(u[pre_start:start].mean()) if n >= start else None
    recovery = metrics["recovery_time"]
    eligible = pre is not None and pre >= recovery["pre_fault_eligibility_min_mean_reward"]
    recovered = None
    seen_after_return = max(0, min(n, end)-returned)
    recovery_status = "ineligible" if not eligible else "right_censored"
    if eligible:
        consecutive = 0
        window = int(recovery["trailing_mean_actions"])
        for elapsed in range(window, seen_after_return+1):
            value = float(u[returned+elapsed-window:returned+elapsed].mean())
            consecutive = consecutive+1 if value >= recovery["threshold_fraction_of_own_pre_fault_mean"]*pre else 0
            if consecutive >= recovery["consecutive_successful_checks"]:
                recovered, recovery_status = elapsed, "observed"
                break
        if recovered is None and n < end:
            recovery_status = "unknown_after_technical_failure"
    failure = metrics["functional_failure"]
    functional = _window_failure(u < failure["normalized_reward_below"], start, end,
                                 int(failure["consecutive_actions"]))
    posture = metrics["posture_failure"]
    height, upright = arrays["torso_height"], arrays["torso_upright"]
    pre_height = float(height[pre_start:start].mean()) if n >= start else None
    pre_upright = float(upright[pre_start:start].mean()) if n >= start else None
    posture_eligible = pre_height is not None and pre_height > 0 and pre_upright >= posture["vertical_axis_alignment_below"]
    posture_failure = posture_full = None
    if posture_eligible:
        flags = ((height < posture["height_below_fraction_of_pre_fault_mean"]*pre_height)
                 | (upright < posture["vertical_axis_alignment_below"]))
        posture_failure = _window_failure(flags, start, end, int(posture["consecutive_actions"]))
        posture_full = _window_failure(flags, 0, len(primary_u), int(posture["consecutive_actions"]))
    technical = summary["status"] == "technical_failure"
    reference_technical = clean_summary["status"] == "technical_failure"
    deficit = float(np.mean(reference_u[start:end]-primary_u[start:end]))
    gate = summary.get("hash_gate", {})
    frozen = gate.get("passed") is True and gate.get("before") == gate.get("after") and "before" in gate
    row = {"case_id": summary["case_id"], "stage": summary.get("stage", summary.get("split")),
           "arm": summary["arm"], "environment_seed": int(summary["environment_seed"]),
           "missing_length": length, "status": summary["status"], "technical_failure": technical,
           "clean_reference_technical_failure": reference_technical,
           "actual_eval_agent_steps": summary["actual_eval_agent_steps"],
           "actual_eval_dmc_calls": summary["actual_eval_dmc_calls"],
           "actual_interaction_counters_complete": summary.get("actual_interaction_counters_complete", True),
           "valid_reward_rows": n, "reward_rows_missing_after_execution": summary["actual_eval_agent_steps"]-n,
           "raw_observed_episode_return": float(arrays["reward"].sum()),
           "primary_zero_filled_action_count": len(primary_u)-n,
           "pre_fault_mean_reward": pre,
           "fault_mean_reward": float(u[start:returned].mean()) if length and n >= returned else None,
           "recovery_mean_reward": float(u[returned:end].mean()) if n >= end else None,
           "paired_integrated_deficit": deficit,
           "complete_case_paired_deficit": deficit if not technical and not reference_technical else None,
           "functional_failure": functional,
           "recovery_time_eligible": bool(eligible), "recovery_time_actions": recovered,
           "recovery_status": recovery_status, "recovery_observed_actions": seen_after_return,
           "recovery_censor_at_actions": None if recovered is not None or not eligible else seen_after_return,
           "pre_fault_height": pre_height, "pre_fault_upright": pre_upright,
           "posture_reference_eligible": bool(posture_eligible),
           "posture_failure": posture_failure, "full_episode_posture_failure": posture_full,
           "no_posture_failure_proxy": False if technical or posture_failure is True else (True if posture_failure is False else None),
           "full_episode_no_posture_failure_proxy": False if technical or posture_full is True else (True if posture_full is False else None),
           "issued_action_fraction": summary["actual_eval_agent_steps"]/len(primary_u),
           "issued_action_fraction_is_lower_bound": not summary.get("actual_interaction_counters_complete", True),
           "frozen_hash_gate_passed": bool(frozen),
           "action_saturation_fraction": float(np.mean(np.abs(arrays["actions"]) >= .999)) if n else None,
           "availability_sha256": summary["availability_sha256"], "error": summary.get("error")}
    for percentile in (50, 95, 99):
        row[f"latency_p{percentile}_s"] = float(np.percentile(arrays["latency_s"], percentile)) if n else None
    return row


def _stats(values):
    x = np.asarray(values, np.float64)
    return {"n": len(x), "mean": float(x.mean()) if len(x) else None,
            "std_population": float(x.std()) if len(x) else None}


def analyze_records(records, design):
    """Aggregate recomputed rows. No model outputs or summary means are trusted."""
    lengths = list(design["phase_a"]["schedule"]["missing_lengths_action_decisions"])
    settings = design["analysis"]
    seed = int(settings["analysis_seed"])
    draws = int(settings["bootstrap_replicates"])
    randomizations = int(settings["paired_sign_randomization_draws"])
    stages = {}
    _require(len({r["case_id"] for r in records}) == len(records), "duplicate case ids")
    for stage in sorted({r["stage"] for r in records}):
        rows = [r for r in records if r["stage"] == stage]
        _require(stage in ("smoke", "design", "held_out"), "unregistered analysis stage")
        indexed = {(r["environment_seed"], r["arm"], r["missing_length"]): r for r in rows}
        _require(len(indexed) == len(rows), "duplicate seed/arm/duration case")
        actual_seeds = sorted({r["environment_seed"] for r in rows})
        if stage in ("design", "held_out"):
            spec = design["phase_a"]["splits"][stage+"_environment_seeds"]
            expected_seeds = list(range(spec["start"], spec["start"]+spec["count"]))
            _require(set(actual_seeds) <= set(expected_seeds), "seed outside registered split")
        else:
            expected_seeds = [40000]
            _require(set(actual_seeds) <= set(expected_seeds), "unregistered smoke seed")
        if stage == "smoke":
            pairs = [("clean", 0)]+[(a, 5) for a in FAULT_ARMS]
            pairs += [("v2_exact", 10), ("posterior_return_once", 10), ("locked_ancestor", 50)]
            pairs += [(a, 0) for a in ("v2_exact", "locked_ancestor", "posterior_return_once")]
            expected = {(s, a, length) for s in expected_seeds for a, length in pairs}
        else:
            expected = {(s, "clean", 0) for s in expected_seeds} | {
                (s, a, length) for s in expected_seeds for a in FAULT_ARMS for length in lengths}
        _require(set(indexed) <= expected, "case outside registered stage matrix")
        missing = sorted(expected-set(indexed))
        complete_matrix = not missing and set(indexed) == expected
        clusters, by_arm, duration_rows = [], {}, []
        for arm in FAULT_ARMS:
            by_arm[arm] = {}
            for s in actual_seeds:
                selected = [indexed.get((s, arm, length)) for length in lengths]
                if all(x is not None for x in selected) and (s, "clean", 0) in indexed:
                    row = {"stage": stage, "environment_seed": s, "arm": arm,
                           "mean_paired_integrated_deficit": float(np.mean([x["paired_integrated_deficit"] for x in selected])),
                           "complete_case_eligible": all(x["complete_case_paired_deficit"] is not None for x in selected),
                           "functional_failures": sum(x["functional_failure"] is True for x in selected),
                           "functional_outcomes_unknown": sum(x["functional_failure"] is None for x in selected),
                           "technical_failures": sum(x["technical_failure"] for x in selected)}
                    row["any_functional_failure"] = (True if row["functional_failures"] else
                                                     (None if row["functional_outcomes_unknown"] else False))
                    clusters.append(row)
                    by_arm[arm][s] = row
            for length in lengths:
                selected = [x for x in rows if x["arm"] == arm and x["missing_length"] == length]
                eligible = [x for x in selected if x["recovery_time_eligible"]]
                known = [x for x in selected if x["functional_failure"] is not None]
                duration_rows.append({"stage": stage, "arm": arm, "missing_length": length,
                    "attempts": len(selected), "mean_raw_return": _stats([x["raw_observed_episode_return"] for x in selected])["mean"],
                    "mean_primary_deficit": _stats([x["paired_integrated_deficit"] for x in selected])["mean"],
                    "functional_failure_count": sum(x["functional_failure"] is True for x in selected),
                    "functional_outcomes_unknown": len(selected)-len(known),
                    "functional_failure_fraction": (sum(x["functional_failure"] is True for x in known)/len(known)
                        if known and len(known) == len(selected) else None),
                    "technical_failures": sum(x["technical_failure"] for x in selected),
                    "recovery_eligible_attempts": len(eligible),
                    "observed_recovery_fraction": sum(x["recovery_time_actions"] is not None for x in eligible)/len(eligible) if eligible else None})
        comparisons = []
        for index, (proposal, comparator) in enumerate(settings["confirmatory_comparisons"]):
            pair_seeds = sorted(set(by_arm[proposal]) & set(by_arm[comparator]))
            values = [by_arm[proposal][s]["mean_paired_integrated_deficit"]-
                      by_arm[comparator][s]["mean_paired_integrated_deficit"] for s in pair_seeds]
            complete_seeds = [s for s in pair_seeds if by_arm[proposal][s]["complete_case_eligible"]
                              and by_arm[comparator][s]["complete_case_eligible"]]
            complete_values = [by_arm[proposal][s]["mean_paired_integrated_deficit"]-
                               by_arm[comparator][s]["mean_paired_integrated_deficit"] for s in complete_seeds]
            related = [x for x in rows if x["arm"] in (proposal, comparator, "clean")]
            guard_unknown = (not complete_matrix or any(x["technical_failure"] or x["functional_failure"] is None
                                                       for x in related if x["arm"] != "clean")
                             or any(x["technical_failure"] for x in related if x["arm"] == "clean"))
            count_p = sum(x["functional_failure"] is True for x in related if x["arm"] == proposal)
            count_c = sum(x["functional_failure"] is True for x in related if x["arm"] == comparator)
            item = {"stage": stage, "proposal": proposal, "comparator": comparator,
                    "direction": "proposal_minus_comparator_deficit; negative favors proposal",
                    "seed_clusters": pair_seeds, **_stats(values),
                    "bootstrap_ci95": bootstrap_mean_ci(values, seed=seed+index, draws=draws) if values else None,
                    "one_sided_p": sign_randomization_p(values, seed=seed+100+index, draws=randomizations) if values else None,
                    "holm_adjusted_p": None,
                    "complete_case_sensitivity": {"seed_clusters": complete_seeds, **_stats(complete_values),
                        "bootstrap_ci95": bootstrap_mean_ci(complete_values, seed=seed+200+index, draws=draws) if complete_values else None},
                    "failure_guard": "INCONCLUSIVE" if guard_unknown else ("PASS_DESCRIPTIVE" if count_p <= count_c else "FAIL_OBSERVED_INCREASE"),
                    "proposal_observed_failure_count": count_p, "comparator_observed_failure_count": count_c}
            comparisons.append(item)
        if all(x["one_sided_p"] is not None for x in comparisons):
            adjusted = holm_adjust([x["one_sided_p"] for x in comparisons])
            for row, p in zip(comparisons, adjusted):
                row["holm_adjusted_p"] = p
        uncertainty = {}
        for arm in FAULT_ARMS:
            arm_clusters = list(by_arm[arm].values())
            known = [x for x in arm_clusters if x["any_functional_failure"] is not None]
            k = sum(x["any_functional_failure"] is True for x in known)
            all_outcomes = len(known) == len(expected_seeds) and all(x["technical_failures"] == 0 for x in arm_clusters)
            uncertainty[arm] = {"event": "any functional failure across all registered durations in one seed cluster",
                "expected_clusters": len(expected_seeds), "known_clusters": len(known), "affected_known_clusters": k,
                "status": "COMPLETE" if all_outcomes else "INCONCLUSIVE_MISSING_OR_TECHNICAL_FAILURE",
                "one_sided_95_upper": clopper_pearson_upper(k, len(known)) if all_outcomes and known else None,
                "certifies_safety_or_paired_risk_difference": False}
        recommendations = {}
        integrity = all(x["frozen_hash_gate_passed"] for x in rows)
        for proposal in ("locked_ancestor", "posterior_return_once"):
            selected = [x for x in comparisons if x["proposal"] == proposal]
            statistical = (len(selected) == 2 and all(x["mean"] is not None and x["mean"] < 0
                and x["holm_adjusted_p"] is not None and x["holm_adjusted_p"] < settings["superiority_adjusted_p_below"]
                and x["failure_guard"] == "PASS_DESCRIPTIVE" for x in selected))
            # An external strict parity/reload gate is required as well. This
            # analysis cannot fabricate it from aggregate returns.
            recommendations[proposal] = {"statistical_and_descriptive_criteria_met": bool(statistical),
                "eligible_for_follow_up_pending_independent_parity_gate": bool(stage == "held_out" and complete_matrix and integrity and statistical),
                "final_recommendation": "REQUIRES_INDEPENDENT_PARITY_AND_RELOAD_GATE" if stage == "held_out" and complete_matrix and integrity and statistical else "NOT_ESTABLISHED",
                "scope": "one frozen trained policy; no safety or multi-training-seed claim"}
        stages[stage] = {"stage": stage, "analysis_scope": "confirmatory_held_out" if stage == "held_out" and complete_matrix else "descriptive_only",
            "matrix_complete": complete_matrix, "expected_cases": len(expected), "observed_cases": len(rows),
            "missing_cases": [list(x) for x in missing], "all_frozen_hash_gates_pass": integrity,
            "technical_failure_cases": sum(x["technical_failure"] for x in rows),
            "seed_cluster_rows": clusters, "duration_rows": duration_rows, "comparisons": comparisons,
            "failure_uncertainty": uncertainty, "recommendations": recommendations,
            "analysis_rng": {"base_seed": seed, "bootstrap_offsets": "comparison index 0..3; complete-case +200..203",
                "sign_randomization_offsets": "+100..103", "bootstrap_draws": draws, "randomization_draws": randomizations},
            "limitations": ["Evaluation seed clusters do not measure retraining variability.",
                "Sign randomization assumes within-pair exchangeability under its null.",
                "Only primary reward tails are zero-filled after technical failure; failure flags are not imputed.",
                "Recovery fractions retain censored/technical attempts in their eligible denominator.",
                "Functional and posture proxies are not certified physical safety."]}
    return {"schema_version": 1, "stages": stages}


def _json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+"\n", encoding="utf-8")
    os.replace(temporary, path)


def _csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
                             for key, value in row.items()})


def plot_stage(stage_report, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    destination = Path(destination)
    rows = stage_report["duration_rows"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.8), constrained_layout=True)
    fields = [("mean_raw_return", "Episode return (technical attempts retained)"),
              ("mean_primary_deficit", "Primary normalized reward deficit (lower is better)"),
              ("functional_failure_fraction", "Observed functional-failure fraction"),
              ("observed_recovery_fraction", "Observed recovery by 100 actions / eligible attempts")]
    for arm in FAULT_ARMS:
        subset = sorted([x for x in rows if x["arm"] == arm], key=lambda x: x["missing_length"])
        for axis, (field, label) in zip(axes.flat, fields):
            axis.plot([x["missing_length"] for x in subset],
                      [np.nan if x[field] is None else x[field] for x in subset], marker="o", label=arm)
            axis.set(xlabel="Exact missing-observation length (action decisions)", ylabel=label)
            axis.grid(alpha=.25)
    axes[0, 1].axhline(0, color="black", linewidth=.8)
    for axis in axes[1]:
        axis.set_ylim(-.03, 1.03)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f"Frozen Walker continuity study: {stage_report['stage']}\n"
                 "One trained policy; unknown failure rates are omitted, never shown as zero", fontsize=11)
    outputs = []
    for extension in ("png", "pdf"):
        path = destination/f"robustness_recovery.{extension}"
        fig.savefig(path, dpi=180)
        outputs.append(str(path))
    plt.close(fig)
    return outputs


def analyze_study(run_dir, design, *, make_plots=True):
    """Read case artifacts and export only into this run's ANALYSIS directory."""
    root = Path(run_dir)
    if isinstance(design, (str, Path)):
        design = json.loads(Path(design).read_text(encoding="utf-8-sig"))
    cases = []
    sources = []
    for path in sorted((root/"cases").glob("*/summary.json")):
        summary = json.loads(path.read_text(encoding="utf-8"))
        _require(path.parent.name == summary["case_id"], "case directory identity mismatch")
        trace_path = path.parent/"trace.npz"
        cases.append((summary, trace_path))
        for item in (path, trace_path):
            digest = hashlib.sha256()
            with item.open("rb") as stream:
                for block in iter(lambda: stream.read(1024*1024), b""):
                    digest.update(block)
            sources.append({"file": str(item.relative_to(root)), "bytes": item.stat().st_size, "sha256": digest.hexdigest()})
    _require(cases, "no case summaries found")
    clean = {}
    for summary, path in cases:
        if summary["arm"] == "clean":
            key = (summary.get("stage", summary.get("split")), summary["environment_seed"])
            _require(key not in clean, "duplicate clean reference")
            with np.load(path, allow_pickle=False) as saved:
                clean[key] = (summary, {key: saved[key] for key in TRACE_FIELDS})
    records = []
    for summary, path in cases:
        key = (summary.get("stage", summary.get("split")), summary["environment_seed"])
        _require(key in clean, "clean reference has not been recorded yet")
        clean_summary, clean_trace = clean[key]
        with np.load(path, allow_pickle=False) as saved:
            trace = {key: saved[key] for key in TRACE_FIELDS}
        records.append(case_metrics(summary, trace, design, clean_trace, clean_summary))
    report = analyze_records(records, design)
    report["design_sha256"] = hashlib.sha256(json.dumps(design, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    report["source_artifacts"] = sources
    report["analysis_source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    destination = root/"ANALYSIS"
    destination.mkdir(parents=True, exist_ok=True)
    for stage, result in report["stages"].items():
        folder = destination/stage
        folder.mkdir(exist_ok=True)
        _csv(folder/"case_metrics.csv", [x for x in records if x["stage"] == stage])
        _csv(folder/"seed_cluster_metrics.csv", result["seed_cluster_rows"])
        _csv(folder/"duration_metrics.csv", result["duration_rows"])
        _csv(folder/"paired_comparisons.csv", result["comparisons"])
        result["plots"] = plot_stage(result, folder) if make_plots else []
        _json(folder/"summary.json", result)
    _json(destination/"analysis.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    result = analyze_study(args.run_dir, args.design, make_plots=not args.no_plots)
    print(json.dumps({"stages": {name: {k: r[k] for k in ("observed_cases", "expected_cases", "matrix_complete", "technical_failure_cases")}
                                 for name, r in result["stages"].items()}}, sort_keys=True))


if __name__ == "__main__":
    main()
