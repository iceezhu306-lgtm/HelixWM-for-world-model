"""No Minecraft/Java download required. Real-engine validation is separate."""
import io
import json
import math
import struct
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from helixwm_minecraft.bridge import MinecraftEnv
from helixwm_minecraft.protocol import ProtocolError, TASK_ID, decode_rgb, encode_rgb, native_action, read_frame, validate_action, validate_result, write_frame
from helixwm_minecraft.task import task_manifest, VisibleNavigationBackend
from helixwm_minecraft.worker import EnvironmentService, serve


class FakeBackend:
    stop_at = None
    succeed = False

    def __init__(self, seed, limit):
        self.seed = seed
        self.count = 0
        self.closed = False
        self.actions = []
        self.manifest = {"seed": seed}
        self.last_diagnostics = {"position": [1, 2, 3], "compass": 987.123}

    def observation(self):
        return {"pov": np.full((64, 64, 3), self.count, np.uint8), "compass": 987.123, "inventory": {"dirt": 55}, "location_stats": {"xpos": 5}}

    def reset(self):
        return self.observation()

    def step(self, action):
        self.actions.append(action)
        self.count += 1
        done = self.stop_at == self.count
        return self.observation(), 0.25, done, {"success": done and self.succeed, "private": "forbidden"}

    def close(self):
        self.closed = True


SUBPROCESS_FAKE = r'''
import os, sys, types
import numpy as np
class Backend:
    def __init__(self, seed, limit):
        self.seed=seed; self.n=0; self.last_diagnostics={}; self.manifest={"fake_test_only": True}
    def obs(self):
        return {"pov":np.full((64,64,3),self.n,np.uint8),"compass":{"angle":123},"inventory":{"dirt":9}}
    def reset(self):
        print("python native log")
        os.write(1,b"raw fd1 native log\n")
        return self.obs()
    def step(self, action):
        self.n+=1
        return self.obs(),0.5,False,{"private":"not_for_policy"}
    def close(self):
        print("closed fake backend")
module=types.ModuleType("helixwm_minecraft.task")
module.VisibleNavigationBackend=Backend
sys.modules["helixwm_minecraft.task"]=module
from helixwm_minecraft.worker import main
main()
'''


class ProtocolTests(unittest.TestCase):
    def test_minerl_random_fallback_and_server_error_fail_closed(self):
        from types import SimpleNamespace
        for info, returned_done, internal_done in [({"error": "timeout"}, True, True), ({}, False, True)]:
            backend = VisibleNavigationBackend.__new__(VisibleNavigationBackend)
            backend.env = SimpleNamespace(action_space=SimpleNamespace(noop=lambda: {}),
                step=lambda action, i=info, d=returned_done: ({"random_fallback": True}, 0., d, i),
                done=internal_done)
            with self.assertRaises(RuntimeError):
                backend.step({})

    def test_framing_and_fragmented_reads(self):
        stream = io.BytesIO()
        write_frame(stream, {"a": 1})
        class Fragmented(io.BytesIO):
            def read(self, count=-1):
                return super().read(min(count, 2))
        self.assertEqual(read_frame(Fragmented(stream.getvalue())), {"a": 1})

    def test_junk_and_oversize_fail_closed(self):
        for payload in (b"LOG!", struct.pack(">I", 2**30), struct.pack(">I", 0)):
            with self.assertRaises(ProtocolError):
                read_frame(io.BytesIO(payload))

    def test_truncated_frame_fails(self):
        with self.assertRaises(EOFError):
            read_frame(io.BytesIO(struct.pack(">I", 5) + b"{}"))

    def test_rgb_strict_schema_and_shape(self):
        value = encode_rgb(np.zeros((64, 64, 3), np.uint8))
        self.assertEqual(len(decode_rgb(value)), 12288)
        value["compass"] = 0
        with self.assertRaises(ProtocolError):
            decode_rgb(value)
        with self.assertRaises(ProtocolError):
            encode_rgb(np.zeros((64, 64, 3), np.float32))

    def test_action_mapping_and_rotation_per_decision(self):
        value = native_action([1, -1, 1, -1, 1, -1], 2)
        self.assertEqual((value["forward"], value["back"], value["left"], value["right"]), (1, 0, 1, 0))
        self.assertEqual(value["camera"], [-5.0, 7.5])
        self.assertEqual((value["jump"], value["sprint"], value["attack"]), (1, 0, 0))

    def test_invalid_actions_rejected(self):
        for action in ([0]*5, [float("nan")]+[0]*5, [float("inf")]+[0]*5, [True]+[0]*5, [1.01]+[0]*5):
            with self.assertRaises(ProtocolError):
                validate_action(action)

    def test_repeat_actual_ticks_and_time_limit(self):
        service = EnvironmentService(FakeBackend, action_repeat=2, episode_env_steps=3)
        service.reset(42)
        first = service.step([0]*6)
        self.assertEqual((first["env_steps"], first["reward"]), (2, 0.5))
        second = service.step([0]*6)
        self.assertEqual((second["env_steps"], second["episode_ticks"], second["reward"]), (1, 3, 0.25))
        self.assertTrue(second["truncated"])
        self.assertFalse(second["terminated"])
        with self.assertRaises(ProtocolError):
            service.step([0]*6)
        service.close()

    def test_success_stops_repeat_and_overrides_limit(self):
        class Success(FakeBackend):
            stop_at = 1
            succeed = True
        service = EnvironmentService(Success, action_repeat=2, episode_env_steps=1)
        service.reset(1)
        result = service.step([0]*6)
        self.assertEqual(result["env_steps"], 1)
        self.assertTrue(result["terminated"])
        self.assertFalse(result["truncated"])

    def test_privileged_data_only_sidecar(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "private.jsonl"
            service = EnvironmentService(FakeBackend, diagnostics_path=path)
            result = service.reset(3)
            validate_result(result, 3)
            self.assertNotIn("compass", json.dumps(result))
            self.assertNotIn("inventory", json.dumps(result))
            self.assertNotIn("location_stats", json.dumps(result))
            self.assertIn("compass", path.read_text())
            contaminated = dict(result, position=[1, 2, 3])
            with self.assertRaises(ProtocolError):
                validate_result(contaminated, 3)
            with self.assertRaises(ProtocolError):
                validate_result(result, 4)

    def test_serve_rejects_extra_request_fields_and_closes(self):
        service = EnvironmentService(FakeBackend)
        request, response = io.BytesIO(), io.BytesIO()
        write_frame(request, {"schema_version": 1, "request_id": 1, "op": "reset", "seed": 1, "hidden": True})
        request.seek(0)
        serve(request, response, service)
        response.seek(0)
        self.assertEqual(read_frame(response)["status"], "error")
        self.assertIsNone(service.backend)

    def test_reset_closes_old_backend_and_exact_seed(self):
        service = EnvironmentService(FakeBackend)
        service.reset(3)
        previous = service.backend
        result = service.reset(43)
        self.assertTrue(previous.closed)
        self.assertEqual((result["episode_seed"], result["episode_ticks"]), (43, 0))

    def test_real_task_manifest_identity_and_stability(self):
        manifest = task_manifest(123)
        self.assertEqual(manifest, task_manifest(123))
        self.assertEqual(manifest["task_id"], TASK_ID)
        self.assertEqual(manifest["inventory"], "empty_no_compass")
        self.assertFalse(manifest["native_benchmark_equivalence"])
        self.assertTrue(-6 <= manifest["goal"]["x"] <= 6)
        self.assertTrue(12 <= manifest["goal"]["z"] <= 20)

    def test_subprocess_raw_stdout_isolated_bridge_boundary_and_render(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "failures.jsonl"
            env = MinecraftEnv(seed=52, worker_command=[sys.executable, "-c", SUBPROCESS_FAKE], failure_log_path=log, timeout_seconds=20)
            try:
                obs = env.reset()
                self.assertEqual((obs.shape, obs.dtype), ((12288,), np.dtype("uint8")))
                boundary = env.snapshot()
                env.restore(boundary)
                result = env.step(np.zeros(6, np.float32))
                self.assertEqual((result["env_steps"], result["reward"], result["continuation"]), (2, 1.0, 1.0))
                with self.assertRaises(RuntimeError):
                    env.snapshot()
                self.assertEqual(env.render(16).shape, (16, 16, 3))
                self.assertEqual(env.episode_ticks, 2)
            finally:
                env.close()
            stderr = log.with_suffix(".stderr.log").read_text()
            self.assertIn("raw fd1 native log", stderr)
            self.assertIn("python native log", stderr)
            self.assertIsNotNone(env.process.poll())

    def test_timeout_persists_failure_and_stops_owned_worker(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "failures.jsonl"
            env = MinecraftEnv(worker_command=[sys.executable, "-c", "import time; time.sleep(10)"], failure_log_path=log, timeout_seconds=0.1)
            with self.assertRaises(TimeoutError):
                env.reset()
            self.assertIn("TimeoutError", log.read_text())
            self.assertIsNotNone(env.process.poll())
            env.close()


if __name__ == "__main__":
    unittest.main()
