"""Real-engine-only smoke. Does not import a learner or perform optimizer updates."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from helixwm_minecraft.bridge import MinecraftEnv
from helixwm_minecraft.task import task_manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-python", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--timeout-seconds", type=float, default=1200)
    args = parser.parse_args()
    if not 1 <= args.steps <= 100:
        parser.error("Smoke steps must be between 1 and 100")
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.time()
    report = {"kind": "REAL_ENGINE_ENVIRONMENT_SMOKE", "status": "RUNNING", "optimizer_updates": 0,
              "task": task_manifest(args.seed), "start_time_unix": start,
              "actual_engine_runtime": None, "target_visibility_human_inspection": "PENDING"}
    report_path = args.output / "environment_smoke.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    traces = []
    diagnostics = args.output / "privileged_evaluator_only.jsonl"
    env = None
    try:
        env = MinecraftEnv(seed=args.seed, worker_python=args.worker_python, diagnostics_path=diagnostics,
                           timeout_seconds=args.timeout_seconds, failure_log_path=args.output / "worker_failures.jsonl")
        first = env.reset(seed=args.seed).reshape(64, 64, 3)
        if first.dtype != np.uint8 or not np.any(first) or np.unique(first.reshape(-1, 3), axis=0).shape[0] < 2:
            raise RuntimeError("Real reset did not produce a nonempty, nonconstant RGB image")
        Image.fromarray(first).save(args.output / "reset_rgb.png")
        first_hash = hashlib.sha256(first.tobytes()).hexdigest()
        descriptor = env.snapshot()
        for index in range(args.steps):
            action = np.array([0 if index < 2 else 1, 0, 0, 0, -1, -1], np.float32)
            result = env.step(action)
            if not np.isfinite(result["reward"]):
                raise RuntimeError("Nonfinite smoke reward")
            traces.append({"action_index": index, "action": action.tolist(), "reward": result["reward"],
                           "native_steps": result["env_steps"], "terminated": result["terminated"],
                           "truncated": result["truncated"], "rgb_sha256": hashlib.sha256(result["observation"].tobytes()).hexdigest()})
            if result["done"]:
                break
        final = env.last_rgb.copy()
        Image.fromarray(final).save(args.output / "last_rgb.png")
        restored = env.restore(descriptor).reshape(64, 64, 3)
        Image.fromarray(restored).save(args.output / "same_seed_reset_rgb.png")
        contact = Image.new("RGB", (768, 256))
        for panel, array in enumerate((first, final, restored)):
            contact.paste(Image.fromarray(array).resize((256, 256), Image.Resampling.NEAREST), (panel * 256, 0))
        contact.save(args.output / "contact_sheet.png")
        resets = [json.loads(line) for line in diagnostics.read_text(encoding="utf-8").splitlines() if json.loads(line).get("event") == "reset"]
        report.update({"status": "ENGINE_RGB_AND_STEP_PASS_VISUAL_REVIEW_PENDING", "actual_engine_runtime": resets[0]["runtime"],
                       "actions_executed": len(traces), "native_steps_executed": sum(row["native_steps"] for row in traces),
                       "native_counter_unit": "acknowledged_MineRL_synchronous_step_calls",
                       "total_engine_world_ticks_measured": False,
                       "first_rgb_sha256": first_hash, "same_seed_reset_rgb_equal": bool(np.array_equal(first, restored)),
                       "boundary_descriptor_exact_engine_state": False, "completed_time_unix": time.time(),
                       "elapsed_seconds": time.time() - start})
    except Exception as exc:
        report.update({"status": "FAILED", "error_type": type(exc).__name__, "message": str(exc), "elapsed_seconds": time.time() - start})
        raise
    finally:
        if env is not None:
            env.close()
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        (args.output / "native_step_trace.json").write_text(json.dumps(traces, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(json.dumps(report, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
