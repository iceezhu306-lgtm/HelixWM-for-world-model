"""Run-writer exclusion and parent-preserving recovery branch tests."""

import hashlib
import json
import multiprocessing
from pathlib import Path
import tempfile
import unittest

from helixwm.run_storage import exclusive_run_lock, prepare_fresh_run, prepare_resume_branch


def _try_lock(directory, connection):
    try:
        with exclusive_run_lock(directory):
            connection.send("acquired")
    except RuntimeError:
        connection.send("blocked")
    except Exception as error:
        connection.send(repr(error))
    finally:
        connection.close()


def _hash(data):
    return hashlib.sha256(data).hexdigest()


class RunStorageTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name)
        self.parent = self.path / "parent"
        self.branch = self.path / "branch"
        (self.parent / "CHECKPOINTS").mkdir(parents=True)
        (self.parent / "REPLAY").mkdir()
        archive = []
        for episode_id in range(3):
            data = f"fake-immutable-episode-{episode_id}".encode()
            filename = f"episode_{episode_id:08d}.npz"
            (self.parent / "REPLAY" / filename).write_bytes(data)
            archive.append({"id": episode_id, "file": filename, "steps": 25,
                            "bytes": len(data), "sha256": _hash(data)})
        # The latest parent contains a third episode absent from the selected
        # checkpoint; it must never appear in the new branch's replay archive.
        self.saved = {"counters": {"agent_steps": 50, "env_steps": 100, "update_steps": 7},
                      "replay": {"archive": archive[:2], "active_ids": [1]}}
        self.checkpoint = self.parent / "CHECKPOINTS" / "checkpoint_000000050.pt"
        data = b"trusted-loaded-checkpoint-fixture"
        self.checkpoint.write_bytes(data)
        self.info = {"file": self.checkpoint.name, "bytes": len(data), "sha256": _hash(data),
                     "counters": self.saved["counters"], "replay": self.saved["replay"]}
        self.write_sidecar()
        (self.parent / "REPLAY" / "index.json").write_text('{"do_not_copy":"newer replay index"}')
        (self.parent / "CHECKPOINTS" / "checkpoint_000000075.pt").write_bytes(b"later checkpoint")
        for filename in ("metrics.jsonl", "resources.jsonl", "events.jsonl", "evaluations.jsonl"):
            (self.parent / filename).write_bytes(
                b'{ "agent_steps" : 25, "value" : 1 }\r\n'
                b'{"agent_steps":50,"value":2}\n'
                b'{"agent_steps":75,"value":3}\n'
                b'{"event":"without_step"}\n'
                b'{"agent_steps":true}\n'
                b'{"agent_steps":'
            )

    def write_sidecar(self):
        self.checkpoint.with_suffix(".json").write_text(json.dumps(self.info), encoding="utf-8")

    def source_hashes(self):
        return {str(path.relative_to(self.parent)): _hash(path.read_bytes())
                for path in self.parent.rglob("*") if path.is_file()}

    def test_fresh_gate_accepts_only_launcher_log_and_lock(self):
        self.branch.mkdir()
        (self.branch / "console.log").write_text("launcher output")
        with exclusive_run_lock(self.branch):
            prepare_fresh_run(self.branch)
            (self.branch / "metrics.jsonl").write_text("retained data")
            with self.assertRaises(FileExistsError):
                prepare_fresh_run(self.branch)
        self.assertEqual((self.branch / "metrics.jsonl").read_text(), "retained data")

    def test_parallel_process_lock_rejects_second_writer_and_releases(self):
        context = multiprocessing.get_context("spawn")
        with exclusive_run_lock(self.branch):
            reader, writer = context.Pipe(duplex=False)
            process = context.Process(target=_try_lock, args=(str(self.branch), writer))
            process.start()
            writer.close()
            self.assertTrue(reader.poll(10), "child lock attempt did not complete")
            self.assertEqual(reader.recv(), "blocked")
            process.join(10)
            self.assertEqual(process.exitcode, 0)
            reader.close()
        with exclusive_run_lock(self.branch):
            prepare_fresh_run(self.branch)

    def test_lock_releases_when_caller_raises(self):
        with self.assertRaisesRegex(ValueError, "test interruption"):
            with exclusive_run_lock(self.branch):
                raise ValueError("test interruption")
        with exclusive_run_lock(self.branch):
            prepare_fresh_run(self.branch)

    def test_branch_preserves_parent_copies_only_references_and_strict_log_prefix(self):
        before = self.source_hashes()
        with exclusive_run_lock(self.branch):
            copied = prepare_resume_branch(self.branch, self.checkpoint, self.saved)
        self.assertEqual(self.source_hashes(), before)
        self.assertEqual(copied.read_bytes(), self.checkpoint.read_bytes())
        self.assertEqual(sorted(path.name for path in (self.branch / "REPLAY").iterdir()),
                         ["episode_00000000.npz", "episode_00000001.npz"])
        self.assertEqual(len(list((self.branch / "CHECKPOINTS").glob("*.pt"))), 1)
        self.assertFalse((self.branch / "REPLAY" / "index.json").exists())
        self.assertEqual((self.branch / "metrics.jsonl").read_bytes(),
                         b'{ "agent_steps" : 25, "value" : 1 }\r\n')
        lineage = json.loads((self.branch / "lineage.json").read_text())
        self.assertEqual(lineage["parent_checkpoint_sha256"], self.info["sha256"])
        self.assertEqual(lineage["branched_counters"], self.saved["counters"])
        self.assertFalse(lineage["exact_long_run_recovery_claimed"])
        self.assertEqual(lineage["log_prefixes"]["metrics.jsonl"]["malformed_lines"], [6])
        latest = json.loads((self.branch / "CHECKPOINTS" / "latest.json").read_text())
        self.assertEqual(latest["file"], self.checkpoint.name)
        self.assertEqual(latest["sha256"], self.info["sha256"])

    def test_same_directory_nested_target_and_existing_target_rejected(self):
        with self.assertRaisesRegex(ValueError, "new branch"):
            prepare_resume_branch(self.parent, self.checkpoint, self.saved)
        with self.assertRaisesRegex(ValueError, "contain one another"):
            prepare_resume_branch(self.parent / "branch", self.checkpoint, self.saved)
        self.branch.mkdir()
        (self.branch / "config.json").write_text("{}")
        with self.assertRaises(FileExistsError):
            prepare_resume_branch(self.branch, self.checkpoint, self.saved)

    def test_replay_path_traversal_rejected_before_copy(self):
        self.saved["replay"]["archive"][0]["file"] = "../escape.npz"
        self.write_sidecar()
        with self.assertRaisesRegex(ValueError, "path traversal"):
            prepare_resume_branch(self.branch, self.checkpoint, self.saved)
        self.assertFalse((self.branch / "CHECKPOINTS").exists())

    def test_tampered_replay_or_checkpoint_rejected(self):
        replay_file = self.parent / "REPLAY" / "episode_00000000.npz"
        original = replay_file.read_bytes()
        replay_file.write_bytes(b"x" * len(original))
        with self.assertRaisesRegex(ValueError, "SHA256"):
            prepare_resume_branch(self.branch, self.checkpoint, self.saved)
        replay_file.write_bytes(original)
        data = self.checkpoint.read_bytes()
        self.checkpoint.write_bytes(b"x" * len(data))
        with self.assertRaisesRegex(ValueError, "SHA256"):
            prepare_resume_branch(self.branch, self.checkpoint, self.saved)

    def test_sidecar_disagreement_and_missing_active_archive_rejected(self):
        self.info["counters"] = {**self.saved["counters"], "agent_steps": 49}
        self.write_sidecar()
        with self.assertRaisesRegex(ValueError, "counters"):
            prepare_resume_branch(self.branch, self.checkpoint, self.saved)
        self.info["counters"] = self.saved["counters"]
        self.saved["replay"]["active_ids"] = [99]
        self.write_sidecar()
        with self.assertRaisesRegex(ValueError, "membership"):
            prepare_resume_branch(self.branch, self.checkpoint, self.saved)


if __name__ == "__main__":
    unittest.main()
