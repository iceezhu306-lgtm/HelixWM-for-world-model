"""Local subprocess bridge; learner process never imports MineRL or old Gym."""
import json
import os
from pathlib import Path
import queue
import signal
import subprocess
import threading
import time

from .protocol import SCHEMA_VERSION, TASK_ID, ProtocolError, decode_rgb, integer, read_frame, validate_action, validate_result, write_frame


class MinecraftEnv:
    action_dim = 6
    observation_dim = 64 * 64 * 3
    task_id = TASK_ID
    native_step_name = "acknowledged_MineRL_synchronous_step"

    def __init__(self, seed=0, action_repeat=2, image_size=64, episode_env_steps=1000,
                 worker_python=None, diagnostics_path=None, timeout_seconds=120,
                 worker_env=None, failure_log_path=None, worker_command=None):
        if image_size != 64:
            raise ValueError("Minecraft policy observation is fixed to RGB64")
        self.seed = integer(seed, "seed")
        self.action_repeat = integer(action_repeat, "action_repeat", 1, 16)
        self.episode_env_steps = integer(episode_env_steps, "episode_env_steps", 1, 6000)
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.timeout_seconds = float(timeout_seconds)
        self.last_rgb = None
        self.episode_ticks = 0
        self._sequence = 0
        self._closed = False
        self._responses = queue.Queue()
        self.failure_log_path = Path(failure_log_path or (str(diagnostics_path) + ".failures.jsonl" if diagnostics_path else "minecraft_worker_failures.jsonl"))
        self.failure_log_path.parent.mkdir(parents=True, exist_ok=True)
        stderr_path = self.failure_log_path.with_suffix(".stderr.log")
        self._stderr = stderr_path.open("ab", buffering=0)
        if worker_command is None:
            worker_python = worker_python or os.environ.get("HELIXWM_MINECRAFT_PYTHON")
            if not worker_python:
                self._stderr.close()
                raise ValueError("Specify the isolated Python 3.9 worker executable")
            worker_command = [str(worker_python), "-m", "helixwm_minecraft.worker", "--action-repeat", str(action_repeat), "--episode-env-steps", str(episode_env_steps)]
            if diagnostics_path:
                worker_command += ["--diagnostics-path", str(diagnostics_path)]
        environment = os.environ.copy()
        environment.update({str(k): str(v) for k, v in (worker_env or {}).items()})
        source_root = str(Path(__file__).resolve().parents[1])
        environment["PYTHONPATH"] = source_root + os.pathsep + environment.get("PYTHONPATH", "")
        environment.setdefault("MINERL_HEADLESS", "1")
        environment.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
        options = {"start_new_session": True} if os.name == "posix" else {}
        try:
            self.process = subprocess.Popen(worker_command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._stderr, env=environment, **options)
        except Exception:
            self._stderr.close()
            raise
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()

    def _read_loop(self):
        try:
            while True:
                self._responses.put(read_frame(self.process.stdout))
        except Exception as exc:
            self._responses.put(exc)

    def _failure(self, operation, exc):
        with self.failure_log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"time_unix": time.time(), "operation": operation, "episode_seed": self.seed, "episode_ticks": self.episode_ticks, "worker_pid": self.process.pid, "error_type": type(exc).__name__, "message": str(exc)}, allow_nan=False) + "\n")

    def _request(self, operation, **fields):
        if self._closed:
            raise RuntimeError("Minecraft worker is closed")
        self._sequence += 1
        try:
            write_frame(self.process.stdin, {"schema_version": SCHEMA_VERSION, "request_id": self._sequence, "op": operation, **fields})
            try:
                response = self._responses.get(timeout=self.timeout_seconds)
            except queue.Empty as exc:
                raise TimeoutError("Minecraft worker timed out during " + operation) from exc
            if isinstance(response, Exception):
                raise response
            if response.get("schema_version") != SCHEMA_VERSION or response.get("request_id") != self._sequence:
                raise ProtocolError("Mismatched response version or sequence")
            if response.get("status") == "error":
                raise RuntimeError("Minecraft worker error: " + str(response.get("error")))
            if set(response) != {"schema_version", "request_id", "status", "result"} or response["status"] != "ok":
                raise ProtocolError("Unexpected response envelope")
            if operation == "close":
                if response["result"] != {"closed": True}:
                    raise ProtocolError("Invalid close response")
            else:
                validate_result(response["result"], fields.get("seed", self.seed))
            return response["result"]
        except Exception as exc:
            self._failure(operation, exc)
            self._abort()
            raise

    def _array(self, result):
        import numpy as np
        self.last_rgb = np.frombuffer(decode_rgb(result["observation"]), dtype=np.uint8).copy().reshape(64, 64, 3)
        self.episode_ticks = result["episode_ticks"]
        return self.last_rgb.reshape(-1).copy()

    def reset(self, seed=None):
        chosen = self.seed if seed is None else integer(seed, "seed")
        result = self._request("reset", seed=chosen)
        if result["env_steps"] != 0 or result["episode_ticks"] != 0 or result["terminated"] or result["truncated"]:
            error = ProtocolError("Reset returned a non-boundary state")
            self._failure("reset", error)
            self._abort()
            raise error
        self.seed = chosen
        return self._array(result)

    def step(self, action):
        values = validate_action(action.tolist() if hasattr(action, "tolist") else action)
        result = self._request("step", action=values)
        if not 1 <= result["env_steps"] <= self.action_repeat or result["episode_ticks"] != self.episode_ticks + result["env_steps"]:
            error = ProtocolError("Acknowledged native step accounting mismatch")
            self._failure("step", error)
            self._abort()
            raise error
        return {"observation": self._array(result), "reward": result["reward"], "continuation": 0.0 if result["terminated"] else 1.0, "done": result["terminated"] or result["truncated"], "terminated": result["terminated"], "truncated": result["truncated"], "env_steps": result["env_steps"], "episode_seed": result["episode_seed"]}

    def render(self, size=256):
        import numpy as np
        if self.last_rgb is None:
            raise RuntimeError("reset first")
        integer(size, "render size", 1, 2048)
        indices = np.minimum((np.arange(size) * 64 // size), 63)
        return self.last_rgb[indices[:, None], indices[None, :]].copy()

    def snapshot(self):
        if self.last_rgb is None or self.episode_ticks != 0:
            raise RuntimeError("Only a validated reset boundary can be described; no mid-episode Minecraft restore")
        return {"kind": "MinecraftResetBoundaryV1", "task_id": TASK_ID, "episode_seed": self.seed, "action_repeat": self.action_repeat, "episode_env_steps": self.episode_env_steps, "exact_engine_state": False}

    def restore(self, snapshot):
        if snapshot.get("kind") != "MinecraftResetBoundaryV1" or snapshot.get("task_id") != TASK_ID or snapshot.get("exact_engine_state") is not False or snapshot.get("action_repeat") != self.action_repeat or snapshot.get("episode_env_steps") != self.episode_env_steps:
            raise ValueError("Incompatible reset-boundary descriptor")
        return self.reset(seed=snapshot["episode_seed"])

    def _abort(self):
        if self._closed:
            return
        self._closed = True
        # Only descendants of this owned worker are considered. Never pkill Java.
        descendants = []
        try:
            import psutil
            descendants = psutil.Process(self.process.pid).children(recursive=True)
        except (ImportError, OSError):
            pass
        except Exception:
            pass
        for child in reversed(descendants):
            try:
                child.terminate()
            except Exception:
                pass
        if self.process.poll() is None:
            try:
                if os.name == "posix":
                    os.killpg(self.process.pid, signal.SIGTERM)
                else:
                    self.process.terminate()
                self.process.wait(timeout=5)
            except (OSError, subprocess.TimeoutExpired):
                try:
                    if os.name == "posix":
                        os.killpg(self.process.pid, signal.SIGKILL)
                    else:
                        self.process.kill()
                except OSError:
                    pass
                self.process.wait(timeout=5)
        for child in descendants:
            try:
                if child.is_running():
                    child.kill()
            except Exception:
                pass
        for stream in (self.process.stdin, self.process.stdout):
            if stream:
                stream.close()
        self._stderr.close()

    def close(self):
        if not self._closed:
            try:
                self._request("close")
            except Exception as exc:
                # _request persists the failure and aborts this owned worker.
                # Cleanup must not replace the caller's training/eval exception.
                self.close_error = str(exc)
            finally:
                self._abort()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
