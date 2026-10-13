"""Snapshot integrity, independent RNG restoration and immutable checkpoint tests."""

import json
from pathlib import Path
import random
import tempfile
import unittest

import numpy as np
import torch

from helixwm.checkpoint import capture_rng, load_checkpoint, restore_learner, restore_rng, save_checkpoint
from helixwm.control import HelixLearner
from helixwm.pixels import PixelRSSM


class ReplayStub:
    def snapshot(self):
        return {"files": [], "test": True}


class CheckpointTests(unittest.TestCase):
    def test_rng_restores_three_independent_streams(self):
        state = capture_rng()
        expected = (random.random(), np.random.rand(), torch.rand(4))
        restore_rng(state)
        self.assertEqual(random.random(), expected[0])
        self.assertEqual(np.random.rand(), expected[1])
        self.assertTrue(torch.equal(torch.rand(4), expected[2]))

    def test_full_atomic_roundtrip_integrity_and_no_overwrite(self):
        torch.set_num_threads(1)
        model = PixelRSSM(hidden_dim=8, groups=2, classes=3, embed_dim=8, cnn_depth=1)
        learner = HelixLearner(model, {"actor_hidden": 8, "critic_hidden": 8, "reliability_hidden": 8})
        with tempfile.TemporaryDirectory() as directory:
            path = save_checkpoint(directory, learner, {"seed": 0}, {"agent_steps": 500}, ReplayStub(), {"episode": {}}, {"source": "test"})
            state = load_checkpoint(path)
            old = {key: value.clone() for key, value in learner.state_dict().items()}
            with torch.no_grad():
                next(learner.parameters()).add_(1)
            restore_learner(learner, state)
            self.assertTrue(all(torch.equal(value, learner.state_dict()[key]) for key, value in old.items()))
            self.assertEqual(json.loads((Path(directory) / "latest.json").read_text())["file"], path.name)
            with self.assertRaises(FileExistsError):
                save_checkpoint(directory, learner, {}, {"agent_steps": 500}, ReplayStub(), {}, {})
            with path.open("ab") as handle:
                handle.write(b"tamper")
            with self.assertRaisesRegex(ValueError, "integrity"):
                load_checkpoint(path)


if __name__ == "__main__":
    unittest.main()
