"""Recompute the public second-experiment statistics without SSH or model loading.

Requires Python >=3.10 and NumPy. Paths default to the repository containing this
file. Optional --export-from/--v1-from inputs are retained local package roots;
only explicit projections are exported, never entire directories or checkpoints.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
from collections import Counter, defaultdict

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "results/2026-10-14-dual-chain-walker-seed0"
CONDITIONS = ("clean", "dropout", "dropout_disabled", "dropout_prior_only")
BOOTSTRAP_SEED = 20261017
BOOTSTRAP_SAMPLES = 50000


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows, fields):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fields, lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def project(record, keys):
    return {key: record[key] for key in keys if key in record}


def summarize(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Expected a nonempty finite one-dimensional sample")
    return dict(n=len(values), mean=float(values.mean()), population_sd=float(values.std()),
                sample_sd=float(values.std(ddof=1)) if len(values) > 1 else None,
                median=float(np.median(values)), minimum=float(values.min()), maximum=float(values.max()))


def bootstrap_mean(values, seed=BOOTSTRAP_SEED, samples=BOOTSTRAP_SAMPLES):
    values = np.asarray(values, dtype=np.float64)
    summarize(values)
    if samples < 1:
        raise ValueError("Bootstrap sample count must be positive")
    rng = np.random.Generator(np.random.PCG64(seed))
    means = values[rng.integers(0, len(values), size=(samples, len(values)))].mean(axis=1)
    return [float(x) for x in np.quantile(means, [0.025, 0.975], method="linear")]


def paired_rows(left, right, seed_key="evaluation_seed", check_intervention=True):
    def indexed(rows):
        result = {}
        for row in rows:
            key = int(row[seed_key])
            if key in result:
                raise ValueError("Duplicate evaluation seed in a paired comparison")
            result[key] = row
        return result
    a, b = indexed(left), indexed(right)
    if not a or a.keys() != b.keys():
        raise ValueError("Paired evaluation seed sets differ or are empty")
    differences = []
    for key in sorted(a):
        if check_intervention:
            for field in ("training_seed", "evaluation_id", "policy_seed", "dropout_seed", "agent_steps"):
                if a[key].get(field) != b[key].get(field):
                    raise ValueError("Incompatible pairing metadata: " + field)
            if a[key]["condition"] != "clean" and b[key]["condition"] != "clean":
                if a[key]["availability_sha256"] != b[key]["availability_sha256"]:
                    raise ValueError("Paired dropout schedules differ")
        differences.append(float(a[key]["episode_return"]) - float(b[key]["episode_return"]))
    return differences


def effective_sample_size(weights):
    weights = np.asarray(weights, dtype=np.float64)
    if weights.ndim != 1 or not len(weights) or not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("Invalid candidate weights")
    if not np.isclose(weights.sum(), 1, rtol=0, atol=1e-8):
        raise ValueError("Candidate weights are not normalized")
    return float(1 / np.square(weights).sum())


def parse_gpu_sample(value):
    """Match train.py's query: utilization, used MiB, total MiB, temperature."""
    columns = [int(field.strip()) for field in value.split(",")]
    if len(columns) != 4:
        raise ValueError("Expected four GPU fields")
    utilization, used, total, temperature = columns
    if not 0 <= utilization <= 100 or not 0 <= used <= total:
        raise ValueError("Invalid GPU utilization or memory sample")
    return {"gpu_utilization_percent": utilization, "gpu_used_memory_mib": used}


def audit_dropout_schedule(rows, config):
    """Replay only the recorded RNG schedule and check every saved mask hash.

    This mirrors the fixed source's DropoutSchedule; no simulator is loaded.
    The reset and terminal observations are always valid. Adjacent sampled
    bursts can join, so sampled burst length is not maximum contiguous gap.
    """
    gaps, per_episode = [], []
    settings = config["observation_dropout"]
    for row in rows:
        rng = np.random.default_rng(int(row["dropout_seed"]))
        remaining, availability = 0, []
        for step in range(1, int(row["actual_eval_agent_steps"]) + 1):
            if step == int(row["actual_eval_agent_steps"]):
                valid, remaining = True, 0
            elif remaining:
                valid, remaining = False, remaining - 1
            elif rng.random() < settings["start_probability"]:
                valid = False
                remaining = int(rng.integers(settings["min_length"], settings["max_length"] + 1)) - 1
            else:
                valid = True
            availability.append(int(valid))
        if hashlib.sha256(bytes(availability)).hexdigest() != row["availability_sha256"]:
            raise ValueError("Reconstructed missingness does not match its archived hash")
        episode_gaps, gap = [], 0
        for valid in availability:
            if not valid:
                gap += 1
            elif gap:
                episode_gaps.append(gap)
                gap = 0
        if sum(episode_gaps) != int(row["missing_observations"]):
            raise ValueError("Reconstructed missing count differs from evaluation row")
        gaps.extend(episode_gaps)
        per_episode.append({"evaluation_seed": int(row["evaluation_seed"]), "missing_observations": sum(episode_gaps), "maximum_contiguous_missing": max(episode_gaps, default=0), "contiguous_gaps_over_repair_budget": [value for value in episode_gaps if value > config["dual_chain"]["max_missing"]]})
    return {"all_availability_hashes_match": True, "episodes_checked": len(rows), "sampled_burst_minimum": settings["min_length"], "sampled_burst_maximum": settings["max_length"], "maximum_realized_contiguous_gap": max(gaps, default=0), "contiguous_gap_histogram": dict(sorted(Counter(gaps).items())), "episodes": per_episode, "scope": "Final 20 paired dropout episodes only. Adjacent sampled bursts can join; abstention at missing_length=6 is a trigger, not the realized gap length."}


def verify_provenance(data_dir):
    data_dir = Path(data_dir).resolve()
    manifest = read_json(data_dir / "provenance.json")
    for relative, record in manifest["exports"].items():
        path = (data_dir / relative).resolve()
        if not path.is_relative_to(data_dir):
            raise ValueError("Manifest path escapes the data directory")
        if not path.is_file() or path.stat().st_size != record["bytes"] or sha256(path) != record["sha256"]:
            raise ValueError("Export integrity mismatch: " + relative)
    return manifest


def export_public_bundle(v2, v1, destination):
    """Project small data files; source paths never appear in the public manifest."""
    v2, v1, destination = Path(v2), Path(v1), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    manifest = {"schema_version": 1, "source_labels": {"P2": "Retained second-experiment final package", "P1": "Retained first-experiment final package"},
                "sources": {}, "exports": {}, "transformations": {},
                "omitted": ["model checkpoints", "replay", "Conda environments and caches", "full logs", "SSH credentials and private keys", "physical hosts and absolute machine paths", "periodic evaluation repair-event details"]}

    def source(package, relative):
        path = (v2 if package == "P2" else v1) / relative
        label = package + "/" + relative
        manifest["sources"][label] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
        return path, label

    def record(name, labels, transformation):
        path = destination / name
        manifest["exports"][name] = {"bytes": path.stat().st_size, "sha256": sha256(path), "source_labels": labels}
        manifest["transformations"][name] = transformation

    for package, relative, name in [
        ("P2", "RUN/RESULTS/eval.csv", "eval.csv"), ("P1", "RESULTS/eval_raw.csv", "v1_eval.csv"),
        ("P2", "RUN/config.json", "config.json"), ("P1", "CONFIG/config.json", "v1_config.json"),
        ("P2", "RUN/PLOTS/learning_curve.png", "plots/learning_curve.png"),
        ("P2", "RUN/PLOTS/learning_curve.pdf", "plots/learning_curve.pdf"),
        ("P2", "DIAGNOSTICS/training_diagnostics.png", "plots/training_diagnostics.png"),
        ("P2", "DIAGNOSTICS/training_diagnostics.pdf", "plots/training_diagnostics.pdf"),
        ("P2", "VIDEOS/clean_training_final.mp4", "videos/clean_training_final.mp4")]:
        path, label = source(package, relative)
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        record(name, [label], "Byte-for-byte copy; inspected for public-safe content. No numeric changes.")

    metadata_path, metadata_label = source("P2", "RUN/metadata.json")
    completion_path, completion_label = source("P2", "RUN/completion.json")
    final_path, final_label = source("P2", "RUN/RESULTS/final_summary.json")
    v1_metadata_path, v1_metadata_label = source("P1", "METADATA/metadata.json")
    metadata, completion, final, v1_metadata = map(read_json, (metadata_path, completion_path, final_path, v1_metadata_path))
    public = project(metadata, ["experiment_class", "created_utc", "gpu", "packages", "python", "source_commit", "upstream_commit", "units", "observation_protocol", "baseline_rerun", "paper_level_complete"])
    public["identity"] = project(metadata["identity"], ["helix_commit", "helix_tree", "helix_tracked_clean", "upstream_commit", "upstream_tracked_clean", "config_relative_path", "config_sha256", "config_file_sha256"])
    public["completion"] = project(completion, ["agent_steps", "env_steps", "update_steps", "episodes", "masked_observations", "wall_time_s", "budget_complete", "checkpoint_reload_equal", "checkpoint_sha256", "completed_utc", "single_seed_only", "efficacy_demonstrated"])
    public["frozen_evaluation"] = project(final, ["actor_sampling", "condition_definitions", "conditions_are_one_policy_interventions", "evaluation_interactions_in_training_budget", "evaluation_id", "hash_gate", "score_kind", "checkpoint_reload_checked_here", "inference_version"])
    public["video"] = project(final["video"], ["bytes", "sha256", "frames", "fps", "height", "width", "full_decode_pass"])
    public["v1"] = project(v1_metadata, ["source_commit", "experiment_class", "packages", "units"])
    public["projection_note"] = "Explicit allowed-key projection. Filesystem roots, internal addresses, full commands, and unrelated platform configuration are omitted. Hashes of original private files are provenance identifiers, not hashes of this projection."
    write_json(destination / "metadata.json", public)
    record("metadata.json", [metadata_label, completion_label, final_label, v1_metadata_label], public["projection_note"])
    backbone_path, backbone_label = source("P2", "RUN/resolved_backbone_config.json")
    backbone = read_json(backbone_path)
    projected_backbone = project(backbone, ["agent", "batch_length", "batch_size", "consec_report", "consec_train", "replay", "replay_context", "report_length", "task", "seed"])
    projected_backbone["env"] = {"dmc": backbone["env"]["dmc"]}
    projected_backbone["jax"] = {key: value for key, value in backbone["jax"].items() if key != "coordinator_address"}
    projected_backbone["projection_note"] = "Algorithm and compute fields from resolved backbone config. Unused task presets, logger/network/path settings and upstream run-loop defaults omitted. The actual custom driver's action budget and environment settings are config.json, not the upstream run-loop defaults."
    write_json(destination / "backbone_config.json", projected_backbone)
    record("backbone_config.json", [backbone_label], projected_backbone["projection_note"])

    path, label = source("P2", "RUN/RESULTS/evaluation_events.jsonl")
    keys = ["event", "evaluation_id", "evaluation_seed", "condition", "event_id", "model_version", "anchor_time", "available_at", "evidence_cutoff", "missing_length", "reason", "candidate_scores_per_pixel", "candidate_weights", "effective_sample_size", "selected_candidate", "score_temperature", "score_normalization", "control_uses_repair", "observed_teacher_qualified", "prior_counted_again", "segment_times", "score_kind", "selection"]
    with (destination / "repair_events.jsonl").open("w", encoding="utf-8") as stream:
        for row in jsonl(path):
            if row["evaluation_id"] == final["evaluation_id"]:
                stream.write(json.dumps(project(row, keys), sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
    record("repair_events.jsonl", [label], "All final-evaluation repair and abstention events; explicit field projection. Periodic events and redundant latent/image/path hashes omitted. Candidate scores and weights unchanged.")

    path, label = source("P2", "RUN/dual_chain_events.jsonl")
    fields = ["agent_steps", "episode_id", "event_id", "event", "model_version", "effective_sample_size"] + ["weight_" + str(i) for i in range(8)]
    def training_repairs():
        for row in jsonl(path):
            result = project(row, fields)
            result.update({"weight_" + str(i): weight for i, weight in enumerate(row.get("candidate_weights", []))})
            yield result
    write_csv(destination / "training_repair_weights.csv", training_repairs(), fields)
    record("training_repair_weights.csv", [label], "Every training repair/abstention event; retained action/episode/event/version identifiers, stored ESS and all eight weights. Other fields omitted.")

    path, label = source("P2", "RUN/metrics.jsonl")
    fields = ["agent_steps", "env_steps", "kind", "episodes", "update_steps", "wall_time_s", "masked_observations", "episode_masked_observations", "episode_length", "episode_return", "opt/loss", "opt/grad_norm", "opt/grad_rms"] + ["loss/" + key for key in ["con", "dyn", "image", "policy", "rep", "repval", "rew", "value"]]
    write_csv(destination / "training_metrics.csv", jsonl(path), fields)
    record("training_metrics.csv", [label], "All stored metric rows; selected scalar fields only. Loss/gradient values are original interval means, not individual update traces. Missing fields are empty.")

    path, label = source("P2", "RUN/resources.jsonl")
    fields = ["agent_steps", "env_steps", "wall_time_s", "cgroup_memory_current_bytes", "cgroup_memory_max_bytes", "process_rss_kib", "gpu_utilization_percent", "gpu_used_memory_mib"]
    def resources():
        for row in jsonl(path):
            yield {**project(row, ["agent_steps", "env_steps", "wall_time_s"]), "cgroup_memory_current_bytes": int(row["cgroup_memory_current"]), "cgroup_memory_max_bytes": int(row["cgroup_memory_max"]), "process_rss_kib": int(row["process_memory"]["VmRSS"].split()[0]), **parse_gpu_sample(row["gpu"])}
    write_csv(destination / "resource_samples.csv", resources(), fields)
    record("resource_samples.csv", [label], "All sampled rows; numeric public-safe projection with explicit units. No process IDs, hosts, environment variables, or machine paths.")
    write_json(destination / "provenance.json", manifest)


def analyze(data_dir):
    data_dir = Path(data_dir)
    verify_provenance(data_dir)
    rows = read_csv(data_dir / "eval.csv")
    final_rows = [row for row in rows if row["kind"] == "final"]
    groups = {condition: [row for row in final_rows if row["condition"] == condition] for condition in CONDITIONS}
    if any(len(group) != 20 for group in groups.values()) or len({r["evaluation_id"] for r in final_rows}) != 1:
        raise ValueError("Expected one final evaluation with 20 episodes per condition")
    conditions = {}
    for condition, group in groups.items():
        values = [float(row["episode_return"]) for row in group]
        conditions[condition] = {**summarize(values), "episode_bootstrap_mean_ci95": bootstrap_mean(values)}
        conditions[condition].update({key: sum(int(row[key]) for row in group) for key in ["missing_observations", "repair_count", "abstention_count"]})
    comparisons = {}
    for reference in ["clean", "dropout_disabled", "dropout_prior_only"]:
        values = paired_rows(groups["dropout"], groups[reference])
        comparisons["dropout_minus_" + reference] = {**summarize(values), "paired_episode_bootstrap_mean_ci95": bootstrap_mean(values), "positive_pairs": sum(x > 0 for x in values), "negative_pairs": sum(x < 0 for x in values)}
    v1_rows = [row for row in read_csv(data_dir / "v1_eval.csv") if row["kind"] == "final"]
    v1_final = [{**row, "evaluation_seed": row["eval_seed"]} for row in v1_rows]
    differences = paired_rows(groups["clean"], v1_final, check_intervention=False)
    v1 = summarize([float(row["episode_return"]) for row in v1_final])
    descriptive = {"v1_final": v1, "v2_clean_minus_v1_final": summarize(differences), "ratio_of_means": conditions["clean"]["mean"] / v1["mean"], "percent_change_of_means": 100 * (conditions["clean"]["mean"] / v1["mean"] - 1), "scope": "Descriptive version comparison only; no inferential interval reported. Shared environment seeds do not control changed backbone, optimization, replay, policy sampling, or training missingness. Not an estimate of dual-chain causal benefit."}
    events = list(jsonl(data_dir / "repair_events.jsonl"))
    ess, max_weights = [], []
    for event in events:
        if event["event"] == "repair_reconnected":
            value = effective_sample_size(event["candidate_weights"])
            if not np.isclose(value, event["effective_sample_size"], atol=1e-10, rtol=1e-10):
                raise ValueError("Stored final repair ESS disagrees with weights")
            ess.append(value)
            max_weights.append(max(event["candidate_weights"]))
    if len(ess) != conditions["dropout"]["repair_count"]:
        raise ValueError("Final repair events disagree with episode counts")
    training_ess = []
    training_events = read_csv(data_dir / "training_repair_weights.csv")
    for row in training_events:
        if row["event"] == "repair_reconnected":
            value = effective_sample_size([float(row["weight_" + str(i)]) for i in range(8)])
            if not np.isclose(value, float(row["effective_sample_size"]), atol=1e-10, rtol=1e-10):
                raise ValueError("Stored training repair ESS disagrees with weights")
            training_ess.append(value)
    curve_groups = defaultdict(list)
    for row in rows:
        curve_groups[(row["condition"], int(row["agent_steps"]), row["kind"])].append(float(row["episode_return"]))
    curves = [{"condition": key[0], "agent_steps": key[1], "training_env_steps": key[1] * 2, "kind": key[2], **summarize(values)} for key, values in sorted(curve_groups.items())]
    write_csv(data_dir / "learning_curve.csv", curves, ["condition", "agent_steps", "training_env_steps", "kind", "n", "mean", "population_sd", "sample_sd", "median", "minimum", "maximum"])
    result = {"schema_version": 1, "analysis": {"bootstrap_generator": "NumPy PCG64", "bootstrap_seed": BOOTSTRAP_SEED, "bootstrap_samples": BOOTSTRAP_SAMPLES, "interval": "95% percentile bootstrap over evaluation episodes; conditional on this single trained seed", "training_seeds": [0], "multiple_comparison_adjustment": "none; exploratory contrasts", "rng_reset_policy": "A new PCG64 generator with seed 20261017 is initialized for every interval, matching the prior retained comparison analysis."}, "final_conditions": conditions, "paired_intervention_differences": comparisons, "v1_descriptive_comparison": descriptive, "final_repair_diagnostics": {"event_counts": dict(Counter(event["event"] for event in events)), "ess": summarize(ess), "max_candidate_weight": summarize(max_weights), "ess_above_7_9_count": sum(value > 7.9 for value in ess), "abstention_reasons": dict(Counter(event.get("reason") for event in events if event["event"] == "repair_abstained")), "abstention_trigger_missing_lengths": [event["missing_length"] for event in events if event["event"] == "repair_abstained"], "interpretation": "ESS near 8 means weights near uniform across 8 candidates; not proof of correct uncertainty, calibration, or successful causal repair."}, "dropout_schedule_audit": audit_dropout_schedule(groups["dropout"], read_json(data_dir / "config.json")), "training_repair_diagnostics": {"event_counts": dict(Counter(row["event"] for row in training_events)), "ess": summarize(training_ess)}, "clean_return_retention_under_dropout_percent": 100 * conditions["dropout"]["mean"] / conditions["clean"]["mean"], "learning_curve_rows": len(curves), "limits": ["One training seed; evaluation episodes are not independent training seeds.", "Conditions are inference interventions on one trained policy, not independently trained algorithm ablations.", "No matched clean-trained or corrupted-target Dreamer baseline was run.", "Synthetic missingness is known; this does not test learned anomaly detection or general domain shift.", "ESS is weight diversity, not calibration accuracy.", "No models or replay are published in this compact evidence bundle; full originals remain archived separately."]}
    write_json(data_dir / "summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--export-from", type=Path, help="Optional retained V2 package root")
    parser.add_argument("--v1-from", type=Path, help="Required only with --export-from")
    args = parser.parse_args()
    if bool(args.export_from) != bool(args.v1_from):
        parser.error("--export-from and --v1-from must be supplied together")
    if args.export_from:
        export_public_bundle(args.export_from, args.v1_from, args.data_dir)
    result = analyze(args.data_dir)
    manifest = read_json(args.data_dir / "provenance.json")
    for filename in ("summary.json", "learning_curve.csv", "README.md"):
        path = args.data_dir / filename
        if not path.exists():
            continue
        manifest["exports"][filename] = {"bytes": path.stat().st_size, "sha256": sha256(path), "source_labels": ["P2/RUN/RESULTS/eval.csv", "P1/RESULTS/eval_raw.csv", "P2/RUN/RESULTS/evaluation_events.jsonl", "P2/RUN/dual_chain_events.jsonl"]}
        manifest["transformations"][filename] = ("Authored data dictionary and scope disclosures, reviewed against recomputed statistics." if filename == "README.md" else "Deterministically recomputed by scripts/analyze_second_experiment.py; seed, resampling method and limitations in summary.json.")
    manifest["analysis_generator"] = {"repository_relative_path": "scripts/analyze_second_experiment.py", "sha256": sha256(Path(__file__))}
    write_json(args.data_dir / "provenance.json", manifest)
    print(json.dumps({"status": "PASS", "v2_clean": result["final_conditions"]["clean"]["mean"], "v1_final": result["v1_descriptive_comparison"]["v1_final"]["mean"], "training_seeds": [0]}, sort_keys=True))


if __name__ == "__main__":
    main()
