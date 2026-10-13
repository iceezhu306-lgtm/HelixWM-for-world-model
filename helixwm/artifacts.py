"""Export observed pilot-run data without adding results or training claims."""

import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics


_RAW_TRAIN_FIELDS = ["agent_steps", "env_steps", "update_steps", "wall_time_s", "episode_return", "episode_length"]
_RAW_EVAL_FIELDS = ["agent_steps", "env_steps", "seed", "eval_seed", "episode_return", "episode_length", "kind"]
_GROUP_FIELDS = [
    "agent_steps", "env_steps", "seed", "kind", "checkpoint_sha256", "checkpoint_id", "evaluation_id",
]
_AGGREGATE_FIELDS = _GROUP_FIELDS + [
    "n", "mean", "std", "median", "min", "max", "std_ddof", "std_scope",
]


def _read_jsonl(path: Path, missing: list[str], errors: list[str]) -> list[dict]:
    if not path.is_file():
        missing.append(path.name)
        return []
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("expected a JSON object")
            except (json.JSONDecodeError, ValueError) as exc:
                errors.append(f"{path.name}:{line_number}: {exc}")
                continue
            rows.append(row)
    if not rows:
        missing.append(f"{path.name}: no readable data rows")
    return rows


def _finite_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _csv_value(value):
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=True, sort_keys=True)
    return value


def _write_csv(path: Path, rows: list[dict], preferred: list[str]) -> None:
    extras = sorted({key for row in rows for key in row} - set(preferred))
    fields = preferred + extras
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: _csv_value(value) for key, value in row.items()} for row in rows)


def _aggregate_evaluations(rows: list[dict], errors: list[str]) -> list[dict]:
    groups = {}
    for row_number, row in enumerate(rows, 1):
        required = ("agent_steps", "env_steps", "seed", "episode_return", "episode_length")
        invalid = [key for key in required if not _finite_number(row.get(key))]
        for key in ("agent_steps", "env_steps", "episode_length"):
            if key not in invalid and (row[key] < 0 or int(row[key]) != row[key]):
                invalid.append(key)
        if row.get("kind") not in ("periodic", "final"):
            invalid.append("kind")
        if invalid:
            errors.append(f"evaluations.jsonl readable row {row_number}: excluded from statistics; invalid {', '.join(invalid)}")
            continue
        # Periodic and final protocols remain distinct even at equal steps.
        # Evaluation RNG seeds belong in eval_seed, not the training seed.
        # Repeated evaluation invocations are separate groups, preventing
        # repeated fixed evaluation seeds from inflating one checkpoint's n.
        key = (
            row["agent_steps"], row["env_steps"], row["seed"], row["kind"],
            str(row.get("checkpoint_sha256", "")),
            str(row.get("checkpoint_id", row.get("checkpoint", ""))),
            str(row.get("evaluation_id", "")),
        )
        groups.setdefault(key, []).append(float(row["episode_return"]))
    result = []
    for key, returns in sorted(groups.items()):
        row = dict(zip(_GROUP_FIELDS, key))
        row.update({
            "n": len(returns), "mean": statistics.mean(returns),
            "std": statistics.pstdev(returns), "median": statistics.median(returns),
            "min": min(returns), "max": max(returns), "std_ddof": 0,
            "std_scope": "within_checkpoint_evaluation_episodes",
        })
        result.append(row)
    return result


def _diagnostic_keys(metrics: list[dict]) -> list[str]:
    keys = {key for row in metrics for key, value in row.items() if _finite_number(value)}
    return sorted(key for key in keys if any(
        token in key.lower() for token in ("loss", "risk", "horizon", "disagreement", "grad_norm")
    ) or key.lower() in ("h", "model_observation", "model_reward", "model_continuation", "model_dynamics_kl", "model_representation_kl"))


def _plot_run(plot_dir: Path, metrics: list[dict], aggregates: list[dict], title: str) -> list[Path]:
    # Import lazily: CSV/statistics export works without plotting dependencies.
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(10, 7.5), sharex=True, constrained_layout=True)
    try:
        axes[0].set_title(f"{title} - single training seed")
        curves = {}
        for row in aggregates:
            curves.setdefault((row["seed"], row["kind"]), []).append(row)
        for (seed, kind), curve in sorted(curves.items()):
            curve = sorted(curve, key=lambda row: row["agent_steps"])
            x = [row["agent_steps"] for row in curve]
            mean = [row["mean"] for row in curve]
            std = [row["std"] for row in curve]
            axes[0].errorbar(x, mean, yerr=std, marker="o", markersize=4, capsize=3,
                             linestyle="-" if kind == "periodic" else "none",
                             label=f"{kind}, training seed {seed}: mean +/- episode std (ddof=0)")
        if curves:
            axes[0].legend(fontsize=8, loc="best")
        else:
            axes[0].text(0.5, 0.5, "No valid evaluation episodes recorded", ha="center", va="center", transform=axes[0].transAxes)
        axes[0].set_ylabel("Evaluation episode return")
        episode_rows = [row for row in metrics if _finite_number(row.get("agent_steps")) and _finite_number(row.get("episode_return"))]
        if episode_rows:
            axes[1].scatter([row["agent_steps"] for row in episode_rows], [row["episode_return"] for row in episode_rows], s=11, alpha=0.55, label="Observed training episodes (unsmoothed)")
            axes[1].legend(fontsize=8)
        else:
            axes[1].text(0.5, 0.5, "No training episode returns recorded", ha="center", va="center", transform=axes[1].transAxes)
        axes[1].set_ylabel("Training episode return")
        axes[1].set_xlabel("Action decisions (agent_steps)")
        for axis in axes:
            axis.grid(alpha=0.25)
            axis.ticklabel_format(axis="x", style="plain", useOffset=False)
        outputs = [plot_dir / "learning_curve.png", plot_dir / "learning_curve.pdf"]
        fig.savefig(outputs[0], dpi=180)
        fig.savefig(outputs[1])
    finally:
        plt.close(fig)
    diagnostic_keys = _diagnostic_keys(metrics)
    columns = 2
    rows = max(1, (len(diagnostic_keys) + columns - 1) // columns)
    fig, axes = plt.subplots(rows, columns, figsize=(12, max(3.4, rows * 2.6)), squeeze=False, constrained_layout=True)
    try:
        fig.suptitle(f"{title} - recorded learner diagnostics")
        for axis, key in zip(axes.flat, diagnostic_keys):
            selected = [row for row in metrics if _finite_number(row.get("agent_steps")) and _finite_number(row.get(key))]
            selected.sort(key=lambda row: row["agent_steps"])
            axis.plot([row["agent_steps"] for row in selected], [row[key] for row in selected], linewidth=1)
            axis.set_title(key, fontsize=10)
            axis.set_xlabel("Action decisions (agent_steps)")
            axis.set_ylabel(key)
            axis.ticklabel_format(axis="x", style="plain", useOffset=False)
            axis.grid(alpha=0.25)
        for index, axis in enumerate(axes.flat):
            if index >= len(diagnostic_keys):
                axis.set_visible(False)
        if not diagnostic_keys:
            axes[0, 0].set_visible(True)
            axes[0, 0].text(0.5, 0.5, "No learner loss/risk/horizon metrics recorded", ha="center", va="center", transform=axes[0, 0].transAxes)
            axes[0, 0].set_axis_off()
        diagnostic_path = plot_dir / "diagnostics.png"
        fig.savefig(diagnostic_path, dpi=150)
        outputs.append(diagnostic_path)
    finally:
        plt.close(fig)
    return outputs


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_run(run_dir: str | Path, *, include_manifest: bool = True) -> dict:
    """Export existing records and a SHA256 manifest; never run an evaluation.

    Set include_manifest=False for periodic exports to avoid hashing replay and
    checkpoint archives while training. For final archival, invoke after all run
    writers have stopped. During training,
this is a point-in-time export, and subsequently appended logs invalidate their
manifest hashes. Missing or invalid inputs are explicit in export_report.json.
"""
    root = Path(run_dir).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Run directory does not exist: {root}")
    results_dir = root / "RESULTS"
    plot_dir = root / "PLOTS"
    results_dir.mkdir(exist_ok=True)
    plot_dir.mkdir(exist_ok=True)
    snapshot_started_utc = datetime.now(timezone.utc).isoformat()
    missing, errors = [], []
    config = {}
    config_path = root / "config.json"
    if not config_path.is_file():
        missing.append("config.json")
    else:
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
            if not isinstance(config, dict):
                raise ValueError("expected a JSON object")
        except (json.JSONDecodeError, ValueError) as exc:
            errors.append(f"config.json: {exc}")
            config = {}
    metrics = _read_jsonl(root / "metrics.jsonl", missing, errors)
    evaluations = _read_jsonl(root / "evaluations.jsonl", missing, errors)
    resources = _read_jsonl(root / "resources.jsonl", missing, errors)
    aggregates = _aggregate_evaluations(evaluations, errors)
    if not aggregates:
        missing.append("valid per-checkpoint evaluation statistics")
    if not any(_finite_number(row.get("episode_return")) for row in metrics):
        missing.append("training episode returns")
    if not _diagnostic_keys(metrics):
        missing.append("learner loss/risk/horizon diagnostics")
    outputs = []
    for name, rows, preferred in (
        ("train_metrics.csv", metrics, _RAW_TRAIN_FIELDS),
        ("eval_raw.csv", evaluations, _RAW_EVAL_FIELDS),
        ("aggregate_results.csv", aggregates, _AGGREGATE_FIELDS),
        ("resources.csv", resources, ["agent_steps", "env_steps", "wall_time_s"]),
    ):
        path = results_dir / name
        _write_csv(path, rows, preferred)
        outputs.append(path)
    stats_path = results_dir / "mean_std.json"
    stats_path.write_text(json.dumps({
        "scope": "single_seed_pilot", "stage": "clean_training", "smoothing": "none",
        "grouping": _GROUP_FIELDS,
        "std_definition": "Population standard deviation (ddof=0) across observed evaluation episodes at the same checkpoint and evaluation kind; not uncertainty across training seeds.",
        "groups": aggregates,
    }, indent=2, ensure_ascii=True, allow_nan=False) + "\n", encoding="utf-8")
    outputs.append(stats_path)
    title = f"Walker Walk seed {config.get('seed', 'unknown')}"
    try:
        outputs.extend(_plot_run(plot_dir, metrics, aggregates, title))
    except Exception as exc:
        # Preserve raw data even if optional rendering is unavailable or fails.
        errors.append(f"Plot export failed: {type(exc).__name__}: {exc}")
        missing.append("current learning curve and diagnostic plots")
        # These three paths are exporter-owned derived artifacts. Old plots
        # must not appear to describe the current input snapshot after failure.
        for name in ("learning_curve.png", "learning_curve.pdf", "diagnostics.png"):
            stale_plot = plot_dir / name
            if stale_plot.is_file():
                stale_plot.unlink()
    readme_path = root / "ARTIFACTS_README.md"
    readme_path.write_text(
        "# HelixWM first-version pilot artifacts\n\n"
        f"Experiment: {title}. This directory records one clean-training stage of a single-seed pilot. "
        "It does not establish multi-seed reproducibility, a baseline reproduction, or a performance improvement.\n\n"
        "All curve x axes count action decisions (`agent_steps`). `env_steps` is retained separately in the raw data; "
        "action repetition and the configured budget are recorded in `config.json`. No step count is inferred from another unit.\n\n"
        "`RESULTS/train_metrics.csv` and `RESULTS/eval_raw.csv` preserve all readable input records and fields. "
        "`RESULTS/aggregate_results.csv` and `RESULTS/mean_std.json` group actual evaluations by checkpoint steps, training seed, evaluation kind, "
        "checkpoint identity (`checkpoint_id`), and evaluation invocation (`evaluation_id`) when available. "
        "Separate invocations are never combined into one sample size. `eval_seed` is an episode RNG seed and is not a training seed. "
        "Statistics include n, mean, population standard deviation (ddof=0), median, minimum and maximum. "
        "The standard deviation is episode variability at one checkpoint; it is not a confidence interval or spread across training seeds.\n\n"
        "`PLOTS/learning_curve.png` and `.pdf` show unsmoothed evaluation means with episode standard deviation; "
        "training episode returns are in a separate panel. `PLOTS/diagnostics.png` uses only recorded learner metrics. "
        "Empty panels explicitly identify unavailable data. `RESULTS/resources.csv` retains resource samples.\n\n"
        "This exporter does not create or certify checkpoints, videos, frozen-policy reload evaluations, replay recovery, "
        "training completion, or acceptance. Their separate evidence must be checked before declaring the run complete. "
        "An unfinished run remains unfinished after exporting.\n\n"
        "See `RESULTS/export_report.json` for missing inputs, invalid records, plotting failures, output files and counts. "
        "No values are synthesized for missing records. Invalid JSON lines are reported and remain in the original JSONL file. "
        "Readable evaluation rows excluded from statistics remain in the raw CSV.\n\n"
        "When requested, `MANIFEST.sha256` lists SHA256 hashes sorted by relative path and excludes itself. "
        "It is a point-in-time record. Re-export after all writers stop before final archival; later log writes invalidate earlier hashes. "
        "Periodic exports may explicitly disable the manifest and remove the preceding manifest to prevent stale integrity claims. "
        "The export report records snapshot times and exact manifest coverage; it does not certify final acceptance. "
        "No archive ZIP or software environment is included by this exporter.\n",
        encoding="utf-8",
    )
    outputs.append(readme_path)
    report_path = results_dir / "export_report.json"
    manifest_path = root / "MANIFEST.sha256"
    records = []
    # Avoid even traversing the growing replay/checkpoint archive when the
    # periodic caller requests only current CSVs and plots.
    manifest_candidates = (
        sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix())
        if include_manifest else []
    )
    for path in manifest_candidates:
        if path in (manifest_path, report_path) or not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            errors.append(f"Manifest omitted symlink: {relative}")
            continue
        try:
            before = path.stat()
            digest = _hash_file(path)
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise OSError("file changed while hashing; re-export after writers stop")
            records.append((relative, digest))
        except OSError as exc:
            errors.append(f"Manifest omitted {relative}: {exc}")
    outputs.append(report_path)
    if include_manifest:
        outputs.append(manifest_path)
    elif manifest_path.is_file():
        manifest_path.unlink()
    covered_paths = sorted([relative for relative, digest in records] + [report_path.relative_to(root).as_posix()]) if include_manifest else []
    summary = {
        "run_dir": str(root), "scope": "single_seed_pilot", "stage": "clean_training",
        "files": [path.relative_to(root).as_posix() for path in outputs], "count": len(outputs),
        "missing": missing, "errors": errors,
        "train_rows": len(metrics), "evaluation_rows": len(evaluations),
        "aggregated_evaluation_rows": sum(row["n"] for row in aggregates),
        "evaluation_groups": len(aggregates), "resource_rows": len(resources),
        "snapshot_started_utc": snapshot_started_utc,
        "snapshot_reported_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_included": include_manifest,
        "manifest_file_count": len(covered_paths),
        "manifest_covered_files": covered_paths,
        "manifest_scope": "point-in-time regular run files excluding the manifest itself; not final acceptance" if include_manifest else "disabled for this export; no integrity manifest is claimed",
        "writers_quiescent_verified": False,
        "status": "exported_with_gaps" if missing or errors else "exported",
        "acceptance_status": "not_assessed_by_exporter",
    }
    report_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True, allow_nan=False) + "\n", encoding="utf-8")
    if include_manifest:
        records.append((report_path.relative_to(root).as_posix(), _hash_file(report_path)))
        manifest_path.write_text("".join(f"{digest}  {relative}\n" for relative, digest in sorted(records)), encoding="utf-8")
    return summary
