"""Actual-engine environment gate; two fresh processes, no model or training.

CLI: python -m helixwm_matched.env_probe --config FILE --output NEW_DIRECTORY
The JSON report is evidence only when passed=true. This is not a GPU learner,
training-budget, checkpoint, or full experiment acceptance gate.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import traceback

import numpy as np

from .environment import MatchedWalkerEnv
from .protocol import (array_sha256, atomic_json, canonical_json, load_protocol,
                       make_mask_bank, mask_sha256, protocol_sha256)


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def first_difference(left, right, path="$"):
    """Strict structural equality: no tolerances, coercions, or NaN equality."""
    if type(left) is not type(right):
        return path + ":type"
    if isinstance(left, dict):
        if left.keys() != right.keys():
            return path + ":keys"
        for key in sorted(left):
            diff = first_difference(left[key], right[key], path+"."+str(key))
            if diff:
                return diff
    elif isinstance(left, list):
        if len(left) != len(right):
            return path + ":length"
        for i, (a, b) in enumerate(zip(left, right)):
            diff = first_difference(a, b, path+f"[{i}]")
            if diff:
                return diff
    elif isinstance(left, float) and (not np.isfinite(left) or not np.isfinite(right)):
        return path + ":nonfinite"
    elif left != right:
        return path + ":value"
    return None


def fixed_actions(seed):
    return np.random.default_rng(991000+int(seed)).uniform(-1.25, 1.25, (500, 6)).astype(np.float32)


def runtime_lock(config):
    """Collect the instantiated runtime, not a guessed environment label."""
    import dm_control.suite.walker as walker
    import dm_control.suite.common as common
    source_root = Path(__file__).resolve().parent
    sources = {name: file_sha256(source_root/name) for name in
               ("environment.py", "protocol.py", "env_probe.py")}
    assets = {"walker.py": file_sha256(walker.__file__)}
    xml = Path(walker.__file__).with_suffix(".xml")
    if not xml.is_file():
        raise FileNotFoundError("Pinned Walker XML asset missing")
    assets["walker.xml"] = file_sha256(xml)
    for name, value in sorted(common.ASSETS.items()):
        raw = value.encode() if isinstance(value, str) else bytes(value)
        assets["common/"+name] = hashlib.sha256(raw).hexdigest()
    versions = {}
    for name in ("numpy", "dm-control", "mujoco", "glfw", "PyOpenGL"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    maps = Path("/proc/self/maps")
    libraries = {}
    if maps.is_file():
        for line in maps.read_text().splitlines():
            chunks = line.split(maxsplit=5)
            if len(chunks) != 6 or not chunks[5].startswith("/"):
                continue
            path = Path(chunks[5].removesuffix(" (deleted)"))
            if path.is_file() and any(x in path.name.lower() for x in
                    ("libegl", "libglx", "libopengl", "libgldispatch", "libnvidia-egl", "libnvidia-glcore", "libmujoco")):
                key = str(path.resolve())
                if key not in libraries:
                    libraries[key] = {"bytes": path.stat().st_size, "sha256": file_sha256(path)}
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                         capture_output=True, text=True, timeout=30)
    if gpu.returncode or not gpu.stdout.strip() or not libraries:
        raise RuntimeError("Cannot bind real GPU driver/rendering libraries")
    return {"schema_version": 1, "protocol_sha256": protocol_sha256(config),
            "source_files": sources, "python": {"version": platform.python_version(),
            "implementation": platform.python_implementation(), "executable": str(Path(sys.executable).resolve())},
            "dependencies": versions, "task_assets": assets,
            "rendering_libraries": libraries, "GPU_driver": gpu.stdout.strip().splitlines(),
            "render_environment": {k: os.environ.get(k) for k in ("MUJOCO_GL", "EGL_DEVICE_ID", "CUDA_VISIBLE_DEVICES")},
            "observation": {"keys": ["image"], "shape": [64,64,3], "dtype": "uint8",
                            "preprocessing": "no_resize_no_normalization_native_RGB_render", "camera": 0},
            "actions": {"shape": [6], "bounds": [-1,1], "clip": "clip_then_float32", "repeat": 2},
            "reward": "sum_two_native_rewards", "discount": "product_native_discounts",
            "horizon": {"actions": 500, "DMC_calls": 1000, "native_step_limit": 1000},
            "termination": "native_time_limit_must_be_last_at_call1000;discount0_terminal_else_truncated",
            "reset": "new_suite_environment_with_explicit_seed_each_episode",
            "diagnostics_to_policy": False}


def collect_trace(config, seed, mask, wrapper):
    actions = fixed_actions(seed)
    rows, images, physics, availability = [], [], [], []
    with MatchedWalkerEnv(config, diagnostics=True) as env:
        image = env.reset(seed, mask)
        images.append(array_sha256(image))
        physics.append(env.last_info["probe"]["physics_state_sha256"])
        availability.append(True)
        for index, requested in enumerate(actions):
            image, reward, done, info = env.step(requested)
            if done != (index == 499) or info["env_steps_this_step"] != 2:
                raise RuntimeError("Probe trajectory ended outside the registered budget")
            images.append(array_sha256(image)); availability.append(info["availability"])
            physics.append(info["probe"]["physics_state_sha256"])
            rows.append({"action_index": index, "input_available": bool(mask[index]),
                         "returned_observation_timestamp": index+1,
                         "executed_action": info["executed_action"].tolist(),
                         "reward": reward, "discount": info["discount"],
                         "DMC_calls": info["env_steps_this_step"], "done": done,
                         "terminated": info["terminated"], "truncated": info["truncated"],
                         "subcalls": info["probe"]["subcalls"]})
        lock = runtime_lock(config)  # Bind loaded renderer while an environment exists.
    return {"seed": int(seed), "wrapper": wrapper, "actions": 500, "DMC_calls": 1000,
            "mask_sha256": mask_sha256(mask), "fixed_requested_actions_sha256": array_sha256(actions),
            "visible_RGB_sha256": images, "physics_state_sha256": physics,
            "availability": availability, "rows": rows}, lock


def check_fault_visibility(clean, fault):
    if clean["seed"] != fault["seed"] or clean["fixed_requested_actions_sha256"] != fault["fixed_requested_actions_sha256"]:
        raise ValueError("Clean/fault probe seeds or actions differ")
    if clean["physics_state_sha256"] != fault["physics_state_sha256"]:
        raise ValueError("Rendering/masking changed the fixed-action physics trajectory")
    for a, b in zip(clean["rows"], fault["rows"]):
        # Only the mask-derived input_available may differ.
        a, b = dict(a), dict(b)
        a.pop("input_available"); b.pop("input_available")
        if first_difference(a, b):
            raise ValueError("Masking changed native actions/reward/discount/call/terminal evidence")
    zero = array_sha256(np.zeros((64,64,3), np.uint8))
    for i, valid in enumerate(fault["availability"]):
        expected = clean["visible_RGB_sha256"][i] if valid else zero
        if fault["visible_RGB_sha256"][i] != expected:
            raise ValueError("Visible RGB violates the registered masking contract")


def worker(config_path, output):
    config = load_protocol(config_path)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    bank = make_mask_bank(config)
    traces, lock = {}, None
    for i, seed in enumerate((72000, 72001)):
        for wrapper, mask in (("clean", np.ones(501, np.bool_)), ("registered_faults", bank[i])):
            case = f"seed{seed}-{wrapper}"
            print("environment_probe " + case, flush=True)
            trace, observed_lock = collect_trace(config, seed, mask, wrapper)
            if lock is not None and first_difference(lock, observed_lock):
                raise ValueError("Runtime lock changed between probe episodes")
            lock = observed_lock
            atomic_json(output/(case+".json"), trace)
            traces[case] = trace
        check_fault_visibility(traces[f"seed{seed}-clean"], traces[f"seed{seed}-registered_faults"])
    atomic_json(output/"environment_lock.json", lock)
    result = {"passed": True, "protocol_sha256": protocol_sha256(config),
              "mask_bank_content_sha256": array_sha256(bank),
              "cases": {name: {"file": name+".json", "sha256": file_sha256(output/(name+".json"))}
                        for name in traces}, "environment_lock_sha256": protocol_sha256(lock),
              "actions_executed": 2000, "DMC_calls_executed": 4000}
    atomic_json(output/"worker_result.json", result)
    return result


def run_probe(config_path, output):
    config = load_protocol(config_path)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError("Environment-probe output must be empty; preserve earlier evidence")
    report = {"schema_version": 1, "passed": False,
              "protocol_sha256": protocol_sha256(config), "formal_training_started": False,
              "training_updates": 0, "fresh_processes": 2, "checks": {}}
    try:
        results = []
        for branch in range(2):
            dest = output/f"branch{branch}"
            env = os.environ.copy()
            env.setdefault("MUJOCO_GL", "egl")
            env["PYTHONHASHSEED"] = "0"
            with (output/f"branch{branch}.log").open("wb") as log:
                result = subprocess.run([sys.executable, "-B", "-m", "helixwm_matched.env_probe",
                    "--config", str(Path(config_path).resolve()), "--output", str(dest.resolve()), "--worker"],
                    env=env, stdout=log, stderr=subprocess.STDOUT, timeout=1800)
            if result.returncode:
                raise RuntimeError(f"Fresh branch {branch} exited {result.returncode}; see its retained log")
            results.append(json.loads((dest/"worker_result.json").read_text()))
        difference = first_difference(results[0], results[1])
        if difference:
            raise ValueError("Two-process result differs at " + difference)
        for case in results[0]["cases"]:
            a = json.loads((output/"branch0"/(case+".json")).read_text())
            b = json.loads((output/"branch1"/(case+".json")).read_text())
            difference = first_difference(a, b)
            if difference:
                raise ValueError(case + " differs at " + difference)
        locks = [json.loads((output/f"branch{i}"/"environment_lock.json").read_text()) for i in range(2)]
        if first_difference(*locks):
            raise ValueError("Two-process runtime lock differs")
        atomic_json(output/"environment_lock.json", locks[0])
        report.update(passed=True, environment_lock_sha256=protocol_sha256(locks[0]),
            mask_bank_content_sha256=results[0]["mask_bank_content_sha256"],
            actions_executed=4000, DMC_calls_executed=8000,
            checks={"two_fresh_process_exact_trace": True, "full_500_action_horizons": True,
                    "visible_RGB_no_hidden_frame": True, "same_clipped_actions": True,
                    "all_native_subcalls_rewards_discounts_terminals": True,
                    "physics_unchanged_by_fault_wrapper": True, "runtime_lock_identical": True})
    except BaseException as exc:
        report["error"] = repr(exc)
        report["traceback"] = traceback.format_exc()
        # Partial branch logs/traces remain; never claim their actual actions
        # were zero when the branch did not reach its final result.
        report["partial_interaction_accounting"] = "read_retained_branch_traces_and_logs"
    atomic_json(output/"environment_probe.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        worker(args.config, args.output)
        return 0
    result = run_probe(args.config, args.output)
    print(json.dumps({k: v for k, v in result.items() if k != "traceback"}, indent=2), flush=True)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
