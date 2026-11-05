"""Explicit provenance for the approved isolated prebuilt Minecraft worker.

This profile is separate from learner dependencies. It identifies the patched
MineRL runtime; it does not certify engine behavior or exact Java-world recovery.
"""
from copy import deepcopy
from functools import lru_cache
import base64
import hashlib
import importlib
import importlib.metadata as metadata
import json
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys


RUNTIME_PROFILE = {
    "format": "helixwm_minecraft_prebuilt_runtime_v1",
    "runtime_variant": "minerl_mirror_0_4_4_cp311",
    "distribution": "minerl-mirror",
    "version": "0.4.4",
    "source_commit": "10f8d481004f05e798d24ace4d0d9ba371996e20",
    "release": "https://github.com/danijar/minerl/releases/tag/v0.4.4-patched",
    "wheel_filename": "minerl_mirror-0.4.4-cp311-cp311-linux_x86_64.whl",
    "wheel_sha256": "b04e2cd21627e835d5b1566db1fe69ed1bff8abed1c804340ebd61c78a1dc631",
    "python": "3.11.17",
    "system": "Linux",
    "machine": "x86_64",
    "gym": "0.23.1",
    "numpy": "1.23.5",
    "java_version": "1.8.0_462",
    "options_policy": "published_patched_defaults_verified_against_wheel_RECORD",
    "engine_version": "Minecraft 1.11.2 / Malmo 0.37.0",
}
PROFILE_SHA256 = hashlib.sha256(json.dumps(
    RUNTIME_PROFILE, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

CRITICAL_FILES = (
    "minerl/__init__.py",
    "minerl/env/_multiagent.py",
    "minerl/Malmo/Minecraft/launchClient.sh",
    "minerl/Malmo/Minecraft/build/libs/MalmoMod-0.37.0-fat.jar",
    "minerl/Malmo/Minecraft/run/options.txt",
    "minerl/data/assets/template_minecraft/options.txt",
)
APPROVED_WORKER_PREFIX = "/root/helixwm_minecraft/env_py311_prebuilt"


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def _hash_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest


def _isolation_metadata():
    """Accept the explicitly selected venv or a standalone Conda prefix."""
    prefix = Path(sys.prefix).resolve()
    executable = Path(sys.executable).resolve()
    _require(prefix == Path(APPROVED_WORKER_PREFIX).resolve(),
             "Minecraft requires the explicitly approved isolated worker prefix")
    _require(executable.is_file() and executable.is_relative_to(prefix)
             and executable == (prefix / "bin" / "python").resolve(),
             "Minecraft worker executable is not owned by the approved prefix")
    evidence = {"prefix": str(prefix), "executable": str(executable)}
    if Path(sys.base_prefix).resolve() != prefix:
        evidence["kind"] = "venv"
        return evidence
    # Conda environments normally have equal prefix/base_prefix; their history
    # and owned executable establish the separate environment instead.
    history = prefix / "conda-meta" / "history"
    _require(history.is_file() and history.stat().st_size > 0,
             "Minecraft requires an isolated worker environment with Conda history")
    _require(not (prefix / "bin" / "conda").exists()
             and not (prefix / "condabin" / "conda").exists(),
             "The Conda base/management environment is not an approved worker")
    evidence.update(kind="conda_prefix", conda_history_path=str(history),
                    conda_history_sha256=_hash_file(history).hexdigest(),
                    conda_history_bytes=history.stat().st_size)
    return evidence


def _record_files(distribution):
    records = {str(entry).replace("\\", "/"): entry for entry in distribution.files or ()}
    checked = {}
    for name in CRITICAL_FILES:
        entry = records.get(name)
        _require(entry is not None and entry.hash is not None and entry.hash.mode == "sha256",
                 "The approved wheel RECORD is missing a critical SHA256 entry: " + name)
        path = Path(distribution.locate_file(entry)).resolve()
        _require(path.is_file() and path.stat().st_size == entry.size,
                 "Installed critical file size differs from wheel RECORD: " + name)
        digest = _hash_file(path)
        recorded = base64.urlsafe_b64encode(digest.digest()).rstrip(b"=").decode()
        _require(recorded == entry.hash.value,
                 "Installed critical file differs from wheel RECORD: " + name)
        checked[name] = {"sha256": digest.hexdigest(), "bytes": path.stat().st_size,
                         "wheel_record_verified": True}
    return checked


def _source_options(distribution):
    result = {}
    for name in CRITICAL_FILES[-2:]:
        values = {}
        path = Path(distribution.locate_file(name))
        for line in path.read_text(encoding="utf-8").splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                if key in {"gamma", "renderDistance", "simulationDistance", "fovEffectScale",
                           "ao", "difficulty", "fov", "bobView"}:
                    values[key] = value
        result[name] = values
    return result


@lru_cache(maxsize=1)
def _inspect_runtime_once():
    """Verify once per worker process, before creating its first Java engine."""
    profile = RUNTIME_PROFILE
    for field, actual in (("python", platform.python_version()),
                          ("system", platform.system()), ("machine", platform.machine())):
        _require(actual == profile[field], "Unapproved Minecraft worker " + field + ": " + actual)
    isolation = _isolation_metadata()
    try:
        metadata.distribution("minerl")
    except metadata.PackageNotFoundError:
        pass
    else:
        raise RuntimeError("Ambiguous runtime: original minerl and patched minerl-mirror must not coexist")
    distribution = metadata.distribution(profile["distribution"])
    name = distribution.metadata["Name"].lower().replace("_", "-")
    _require(name == profile["distribution"] and distribution.version == profile["version"],
             "Unapproved Minecraft distribution identity")
    for name in ("gym", "numpy"):
        _require(metadata.version(name) == profile[name], "Unapproved Minecraft worker dependency: " + name)
    direct = distribution.read_text("direct_url.json")
    _require(direct is not None, "Installed prebuilt wheel must retain pip direct_url provenance")
    archive = json.loads(direct).get("archive_info", {})
    digest = archive.get("hashes", {}).get("sha256")
    if digest is None and archive.get("hash", "").startswith("sha256="):
        digest = archive["hash"].split("=", 1)[1]
    _require(digest == profile["wheel_sha256"], "Installed wheel provenance SHA256 differs from the approved artifact")
    checked = _record_files(distribution)
    module = importlib.import_module("minerl")
    imported = Path(module.__file__).resolve()
    expected = Path(distribution.locate_file("minerl/__init__.py")).resolve()
    _require(imported == expected, "Imported minerl is not owned by the approved wheel distribution")
    _require(importlib.import_module("gym").__version__ == profile["gym"]
             and importlib.import_module("numpy").__version__ == profile["numpy"],
             "Imported Gym/NumPy versions differ from installed distribution metadata")
    java = shutil.which("java")
    _require(java is not None, "The approved Java 8 executable is absent from worker PATH")
    process = subprocess.run([java, "-version"], capture_output=True, text=True, timeout=15, check=True)
    java_text = (process.stderr + "\n" + process.stdout).strip()
    match = re.search(r'\bversion\s+"([^"]+)"', java_text)
    _require(match is not None and match.group(1) == profile["java_version"],
             "The actual Minecraft launcher Java version differs from the approved profile")
    return {
        "runtime_profile": deepcopy(profile), "runtime_profile_sha256": PROFILE_SHA256,
        # Preserve the semantic version field while explicitly distinguishing its distribution.
        "minerl": distribution.version, "distribution": profile["distribution"],
        "runtime_variant": profile["runtime_variant"],
        "distribution_name": profile["distribution"], "distribution_version": distribution.version,
        "runtime_source_commit": profile["source_commit"], "wheel_sha256": profile["wheel_sha256"],
        "python_major_minor": ".".join(profile["python"].split(".")[:2]),
        "python": platform.python_version(), "python_executable": str(Path(sys.executable).resolve()),
        "python_prefix": str(Path(sys.prefix).resolve()), "minerl_module_path": str(imported),
        "worker_isolation": isolation,
        "gym": profile["gym"], "numpy": profile["numpy"],
        "java_executable": str(Path(java).resolve()), "java_version_output": java_text,
        "critical_installed_files": checked, "source_options": _source_options(distribution),
        "source_options_scope": "verified installed templates; copied instance options and rendered output require the real-engine gate",
        "wheel_provenance_verified": True, "import_ownership_verified": True,
        "live_runtime_verified": True,
        "live_runtime_verification_scope": "installed package provenance, imports and Java version; not engine behavior",
        "engine_behavior_validated_here": False, "java_world_bitwise_recovery_claimed": False,
    }


def runtime_metadata():
    """Return a detached copy, so episode logging cannot mutate cached evidence."""
    return deepcopy(_inspect_runtime_once())


def instance_options_metadata(env):
    """Read actual instance options after reset without changing the engine.

    This is evidence, not an acceptance override: an unavailable instance path
    remains explicitly unavailable and must be checked by the real-engine gate.
    """
    result = {"scope": "actual instance run/options.txt after successful reset",
              "status": "unavailable", "instances": [], "options_modified_here": False}
    try:
        underlying = getattr(env, "unwrapped", env)
        instances = getattr(underlying, "instances", None)
        if instances is None:
            result["reason"] = "MineRL instances API not available"
            return result
        for index, instance in enumerate(instances):
            root = getattr(instance, "minecraft_dir", None)
            item = {"index": index, "status": "unavailable"}
            if not isinstance(root, (str, Path)) or not Path(root).is_absolute():
                item["reason"] = "Absolute minecraft_dir is unavailable"
            else:
                path = Path(root) / "run" / "options.txt"
                item["path"] = str(path)
                if not path.is_file():
                    item["reason"] = "Instance run/options.txt not found"
                elif path.stat().st_size > 128 * 1024:
                    item["reason"] = "Unexpectedly large options file"
                else:
                    data = path.read_bytes()
                    values = {}
                    for line in data.decode("utf-8").splitlines():
                        if ":" in line:
                            key, value = line.split(":", 1)
                            if key in {"gamma", "renderDistance", "simulationDistance",
                                       "fovEffectScale", "ao", "difficulty", "fov", "bobView"}:
                                values[key] = value
                    item.update(status="observed", sha256=hashlib.sha256(data).hexdigest(),
                                bytes=len(data), values=values)
            result["instances"].append(item)
        if result["instances"] and all(row["status"] == "observed" for row in result["instances"]):
            result["status"] = "observed"
        else:
            result["reason"] = "Not all actual instance options were observed"
    except Exception as exc:
        result["reason"] = type(exc).__name__ + ": " + str(exc)
    return result
