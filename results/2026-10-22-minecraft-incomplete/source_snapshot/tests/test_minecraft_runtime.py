"""Offline runtime admission tests; no MineRL engine, GPU, or network."""
from contextlib import ExitStack
import base64
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

SOURCE = Path(__file__).resolve().parents[1] / "helixwm_minecraft" / "runtime.py"
SPEC = importlib.util.spec_from_file_location("minecraft_runtime_candidate", SOURCE)
runtime = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runtime)


class Record(str):
    pass


class Distribution:
    metadata = {"Name": "minerl_mirror"}
    version = "0.4.4"

    def __init__(self, root):
        self.root = root
        self.files = []
        for name in runtime.CRITICAL_FILES:
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            content = b"gamma:5.0\nrenderDistance:6\nao:1\n" if name.endswith("options.txt") else name.encode()
            path.write_bytes(content)
            record = Record(name)
            record.size = len(content)
            record.hash = SimpleNamespace(mode="sha256", value=base64.urlsafe_b64encode(
                hashlib.sha256(content).digest()).rstrip(b"=").decode())
            self.files.append(record)
        self.direct = {"archive_info": {"hashes": {"sha256": runtime.RUNTIME_PROFILE["wheel_sha256"]}}}

    def locate_file(self, name):
        return self.root / str(name)

    def read_text(self, name):
        return json.dumps(self.direct) if self.direct is not None else None


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        runtime._inspect_runtime_once.cache_clear()
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.distribution = Distribution(self.root)
        self.modules = {
            "minerl": SimpleNamespace(__file__=str(self.root / "minerl/__init__.py")),
            "gym": SimpleNamespace(__version__="0.23.1"),
            "numpy": SimpleNamespace(__version__="1.23.5"),
        }
        self.versions = {"gym": "0.23.1", "numpy": "1.23.5"}
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.addCleanup(runtime._inspect_runtime_once.cache_clear)
        for target, value in (
            ("python_version", "3.11.17"), ("system", "Linux"), ("machine", "x86_64"),
        ):
            self.stack.enter_context(mock.patch.object(runtime.platform, target, return_value=value))
        self.stack.enter_context(mock.patch.object(runtime.sys, "prefix", str(self.root / "venv")))
        self.stack.enter_context(mock.patch.object(runtime.sys, "base_prefix", str(self.root / "base")))
        executable = self.root / "venv" / "bin" / "python"
        executable.parent.mkdir(parents=True)
        executable.write_bytes(b"mock Python executable")
        self.stack.enter_context(mock.patch.object(runtime.sys, "executable", str(executable)))
        self.stack.enter_context(mock.patch.object(runtime, "APPROVED_WORKER_PREFIX", str(self.root / "venv")))
        self.lookup = self.stack.enter_context(mock.patch.object(
            runtime.metadata, "distribution", side_effect=self.lookup_distribution))
        self.stack.enter_context(mock.patch.object(runtime.metadata, "version", side_effect=self.versions.__getitem__))
        self.stack.enter_context(mock.patch.object(runtime.importlib, "import_module", side_effect=self.modules.__getitem__))
        self.stack.enter_context(mock.patch.object(runtime.shutil, "which", return_value="/approved/jdk/bin/java"))
        self.java = self.stack.enter_context(mock.patch.object(
            runtime.subprocess, "run", return_value=SimpleNamespace(
                stderr='openjdk version "1.8.0_462"\nOpenJDK 64-Bit Server VM', stdout="")))

    def lookup_distribution(self, name):
        if name == "minerl":
            raise runtime.metadata.PackageNotFoundError(name)
        self.assertEqual(name, "minerl-mirror")
        return self.distribution

    def test_approved_profile_records_explicit_patch_identity(self):
        result = runtime.runtime_metadata()
        self.assertEqual(result["distribution_name"], "minerl-mirror")
        self.assertEqual(result["minerl"], "0.4.4")
        self.assertEqual(result["runtime_variant"], "minerl_mirror_0_4_4_cp311")
        self.assertEqual(result["wheel_sha256"], runtime.RUNTIME_PROFILE["wheel_sha256"])
        self.assertTrue(result["live_runtime_verified"])
        self.assertFalse(result["engine_behavior_validated_here"])
        self.assertFalse(result["java_world_bitwise_recovery_claimed"])
        self.assertEqual(len(result["critical_installed_files"]), 6)

    def test_cached_validation_returns_detached_evidence(self):
        first = runtime.runtime_metadata()
        first["runtime_profile"]["gym"] = "changed"
        second = runtime.runtime_metadata()
        self.assertEqual(second["runtime_profile"]["gym"], "0.23.1")
        self.assertEqual(self.java.call_count, 1)

    def test_original_and_mirror_distributions_cannot_coexist(self):
        self.lookup.side_effect = lambda _: self.distribution
        with self.assertRaisesRegex(RuntimeError, "coexist"):
            runtime.runtime_metadata()

    def test_missing_direct_url_rejected(self):
        self.distribution.direct = None
        with self.assertRaisesRegex(RuntimeError, "direct_url"):
            runtime.runtime_metadata()

    def test_wrong_wheel_hash_rejected(self):
        self.distribution.direct["archive_info"]["hashes"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(RuntimeError, "provenance SHA256"):
            runtime.runtime_metadata()

    def test_legacy_pip_archive_hash_form_supported(self):
        self.distribution.direct = {"archive_info": {"hash": "sha256=" + runtime.RUNTIME_PROFILE["wheel_sha256"]}}
        self.assertTrue(runtime.runtime_metadata()["wheel_provenance_verified"])

    def test_import_shadowing_rejected(self):
        self.modules["minerl"].__file__ = str(self.root / "another/minerl/__init__.py")
        with self.assertRaisesRegex(RuntimeError, "not owned"):
            runtime.runtime_metadata()

    def test_modified_binary_rejected(self):
        path = self.distribution.locate_file(runtime.CRITICAL_FILES[3])
        value = path.read_bytes()
        path.write_bytes(b"X" + value[1:])
        with self.assertRaisesRegex(RuntimeError, "differs from wheel RECORD"):
            runtime.runtime_metadata()

    def test_missing_binary_record_rejected(self):
        self.distribution.files.pop(3)
        with self.assertRaisesRegex(RuntimeError, "missing a critical"):
            runtime.runtime_metadata()

    def test_changed_options_rejected(self):
        self.distribution.locate_file(runtime.CRITICAL_FILES[-1]).write_text("gamma:2.0\n")
        with self.assertRaisesRegex(RuntimeError, "differs from wheel RECORD"):
            runtime.runtime_metadata()

    def test_wrong_java_rejected(self):
        self.java.return_value = SimpleNamespace(stderr='openjdk version "17.0.1"', stdout="")
        with self.assertRaisesRegex(RuntimeError, "Java version"):
            runtime.runtime_metadata()

    def test_wrong_python_patch_rejected(self):
        with mock.patch.object(runtime.platform, "python_version", return_value="3.11.16"):
            with self.assertRaisesRegex(RuntimeError, "worker python"):
                runtime.runtime_metadata()

    def test_wrong_dependency_version_rejected(self):
        self.versions["numpy"] = "2.0.0"
        with self.assertRaisesRegex(RuntimeError, "worker dependency"):
            runtime.runtime_metadata()

    def test_imported_dependency_differs_from_metadata_rejected(self):
        self.modules["gym"].__version__ = "0.26.0"
        with self.assertRaisesRegex(RuntimeError, "Imported Gym"):
            runtime.runtime_metadata()

    def test_system_python_rejected(self):
        with mock.patch.object(runtime.sys, "base_prefix", runtime.sys.prefix):
            with self.assertRaisesRegex(RuntimeError, "isolated worker"):
                runtime.runtime_metadata()

    def test_standalone_conda_equal_prefix_with_history_and_owned_executable(self):
        history = self.root / "venv" / "conda-meta" / "history"
        history.parent.mkdir()
        history.write_text("# cmd: conda create -p approved python=3.11.17\n", encoding="utf-8")
        with mock.patch.object(runtime.sys, "base_prefix", runtime.sys.prefix):
            evidence = runtime.runtime_metadata()["worker_isolation"]
        self.assertEqual(evidence["kind"], "conda_prefix")
        self.assertEqual(evidence["conda_history_sha256"], hashlib.sha256(history.read_bytes()).hexdigest())

    def test_conda_base_management_environment_rejected(self):
        prefix = self.root / "venv"
        (prefix / "conda-meta").mkdir()
        (prefix / "conda-meta" / "history").write_text("installed conda\n", encoding="utf-8")
        (prefix / "bin" / "conda").write_text("manager\n", encoding="utf-8")
        with mock.patch.object(runtime.sys, "base_prefix", runtime.sys.prefix):
            with self.assertRaisesRegex(RuntimeError, "base/management"):
                runtime.runtime_metadata()

    def test_other_environment_prefix_rejected(self):
        with mock.patch.object(runtime, "APPROVED_WORKER_PREFIX", str(self.root / "different")):
            with self.assertRaisesRegex(RuntimeError, "approved isolated worker prefix"):
                runtime.runtime_metadata()

    def test_executable_outside_prefix_rejected(self):
        other = self.root / "external_python"
        other.write_bytes(b"external")
        with mock.patch.object(runtime.sys, "executable", str(other)):
            with self.assertRaisesRegex(RuntimeError, "not owned"):
                runtime.runtime_metadata()

    def test_actual_instance_options_are_separate_from_installed_defaults(self):
        root = self.root / "instance"
        (root / "run").mkdir(parents=True)
        path = root / "run/options.txt"
        data = b"gamma:3.5\nrenderDistance:5\nunrelated:value\n"
        path.write_bytes(data)
        env = SimpleNamespace(instances=[SimpleNamespace(minecraft_dir=str(root))])
        result = runtime.instance_options_metadata(env)
        self.assertEqual(result["status"], "observed")
        self.assertEqual(result["instances"][0]["values"], {"gamma": "3.5", "renderDistance": "5"})
        self.assertEqual(result["instances"][0]["sha256"], hashlib.sha256(data).hexdigest())
        self.assertEqual(path.read_bytes(), data)
        self.assertFalse(result["options_modified_here"])

    def test_actual_instance_options_missing_api_not_claimed_verified(self):
        result = runtime.instance_options_metadata(SimpleNamespace())
        self.assertEqual(result["status"], "unavailable")
        self.assertEqual(result["instances"], [])

    def test_actual_instance_options_missing_file_not_claimed_verified(self):
        env = SimpleNamespace(instances=[SimpleNamespace(minecraft_dir=str(self.root / "missing"))])
        result = runtime.instance_options_metadata(SimpleNamespace(unwrapped=env))
        self.assertEqual(result["status"], "unavailable")
        self.assertEqual(result["instances"][0]["status"], "unavailable")


if __name__ == "__main__":
    unittest.main()
