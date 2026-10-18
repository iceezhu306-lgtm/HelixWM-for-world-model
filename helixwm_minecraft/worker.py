"""Python 3.9 simulator worker. All policy-visible data use a strict IPC schema."""
import argparse
import json
import math
import os
from pathlib import Path
import signal
import sys
import time

from .protocol import SCHEMA_VERSION, TASK_ID, ProtocolError, encode_rgb, integer, native_action, read_frame, validate_action, write_frame


class EnvironmentService:
    def __init__(self, factory, action_repeat=2, episode_env_steps=1000, diagnostics_path=None):
        self.factory = factory
        self.action_repeat = integer(action_repeat, "action_repeat", 1, 16)
        self.limit = integer(episode_env_steps, "episode_env_steps", 1, 6000)
        self.diagnostics_path = Path(diagnostics_path) if diagnostics_path else None
        self.backend = None
        self.seed = None
        self.ticks = 0  # Legacy field name: acknowledged native step calls, not all engine world ticks.
        self.done = True

    def _record(self, event, **extra):
        if self.diagnostics_path is None:
            return
        self.diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
        row = {"event": event, "time_unix": time.time(), "episode_seed": self.seed, "episode_ticks": self.ticks, "task_id": TASK_ID, **extra}
        with self.diagnostics_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, allow_nan=False, sort_keys=True) + "\n")

    def _result(self, obs, reward=0.0, terminated=False, truncated=False, steps=0):
        # Whitelist, never copy the raw MineRL observation or info dictionary.
        return {"observation": encode_rgb(obs["pov"]), "reward": float(reward), "terminated": bool(terminated), "truncated": bool(truncated), "env_steps": steps, "episode_ticks": self.ticks, "episode_seed": self.seed, "task_id": TASK_ID}

    def reset(self, seed):
        seed = integer(seed, "seed")
        self.close()
        self.seed, self.ticks, self.done = seed, 0, True
        self.backend = self.factory(seed, self.limit)
        observation = self.backend.reset()
        result = self._result(observation)
        self.done = False
        self._record("reset", manifest=getattr(self.backend, "manifest", {}), runtime=getattr(self.backend, "runtime_metadata", {}), privileged=getattr(self.backend, "last_diagnostics", {}))
        return result

    def step(self, action):
        validate_action(action)
        if self.backend is None or self.done:
            raise ProtocolError("reset is required before stepping a new episode")
        native = native_action(action, self.action_repeat)
        reward, steps, terminated, truncated = 0.0, 0, False, False
        for _ in range(self.action_repeat):
            observation, value, native_done, info = self.backend.step(native)
            if not math.isfinite(float(value)):
                raise ProtocolError("Nonfinite simulator reward")
            self.ticks += 1
            steps += 1
            reward += float(value)
            success = bool(info.get("success", False))
            truncated = self.ticks >= self.limit and not success
            terminated = bool(native_done) and not truncated
            if success:
                terminated, truncated = True, False
            self._record("native_step", normalized_action=action, native_action=native, reward=float(value), terminated=terminated, truncated=truncated, privileged=getattr(self.backend, "last_diagnostics", {}))
            if terminated or truncated:
                break
        self.done = terminated or truncated
        return self._result(observation, reward, terminated, truncated, steps)

    def close(self):
        if self.backend is not None:
            backend, self.backend = self.backend, None
            backend.close()
        self.done = True

    def dispatch(self, request):
        if request.get("schema_version") != SCHEMA_VERSION:
            raise ProtocolError("Unsupported schema version")
        integer(request.get("request_id"), "request_id")
        op = request.get("op")
        expected = {"schema_version", "request_id", "op"}
        if op == "reset":
            expected.add("seed")
        elif op == "step":
            expected.add("action")
        elif op != "close":
            raise ProtocolError("Unknown operation")
        if set(request) != expected:
            raise ProtocolError("Unexpected request fields")
        if op == "reset":
            return self.reset(request["seed"])
        if op == "step":
            return self.step(request["action"])
        self.close()
        return {"closed": True}


def serve(input_stream, output_stream, service):
    try:
        while True:
            try:
                request = read_frame(input_stream)
            except EOFError:
                return
            request_id = request.get("request_id")
            try:
                result = service.dispatch(request)
                response = {"schema_version": SCHEMA_VERSION, "request_id": request_id, "status": "ok", "result": result}
            except Exception as exc:
                service._record("worker_failure", error_type=type(exc).__name__, message=str(exc))
                response = {"schema_version": SCHEMA_VERSION, "request_id": request_id, "status": "error", "error": {"type": type(exc).__name__, "message": str(exc)}}
                write_frame(output_stream, response)
                return  # Fail closed after an environment/protocol error.
            write_frame(output_stream, response)
            if request["op"] == "close":
                return
    finally:
        service.close()


def main():
    # Redirect OS fd 1 as well as Python sys.stdout. Native libraries and Java
    # launch diagnostics must not corrupt the framed protocol channel.
    protocol_output = os.fdopen(os.dup(sys.stdout.fileno()), "wb", buffering=0)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    sys.stdout = sys.stderr
    parser = argparse.ArgumentParser()
    parser.add_argument("--action-repeat", type=int, default=2)
    parser.add_argument("--episode-env-steps", type=int, default=1000)
    parser.add_argument("--diagnostics-path")
    args = parser.parse_args()
    from .task import VisibleNavigationBackend
    service = EnvironmentService(VisibleNavigationBackend, args.action_repeat, args.episode_env_steps, args.diagnostics_path)

    def shutdown(signum, frame):
        service.close()
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    serve(sys.stdin.buffer, protocol_output, service)


if __name__ == "__main__":
    main()
