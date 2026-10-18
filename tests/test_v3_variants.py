"""Inference-only causal, RNG, and recovery tests; no simulator/model downloads."""
import copy
import pickle
import unittest

import numpy as np

from helixwm_v2.dual_chain import DualChain
from helixwm_v3.variants import LockedAncestor, PosteriorReturnOnce, make_chain


class FakeBackend:
    action_dim = 2
    inference_version = 7
    score_kind = "fake_energy"

    def __init__(self):
        self.observed, self.scored, self.prior_calls = [], [], 0

    def initial(self, batch=1):
        return {"deter": np.zeros((batch, 1)), "stoch": np.zeros((batch, 1))}

    def observe(self, state, action, image, is_first=False, rng=None):
        self.observed.append((copy.deepcopy(state), np.asarray(image).copy()))
        return {"deter": state["deter"] + .1,
                "stoch": np.asarray(image).mean() + rng.normal(size=(1, 1)) * .01}

    def prior(self, state, action, rng=None):
        self.prior_calls += 1
        return {"deter": state["deter"] + state["stoch"] + .1,
                "stoch": state["stoch"] + rng.normal(size=state["stoch"].shape) * .1}

    def score(self, states, image):
        self.scored.append(np.asarray(image).copy())
        return -np.square(states["stoch"][:, 0] - np.asarray(image).mean())

    def act(self, state, deterministic=False, rng=None):
        # Same actual actions isolate candidate RNG/lineage, independently of
        # changes to the state consumed by the actor in the real experiment.
        return rng.normal(size=(1, 2)) * .1


def assert_tree(test, a, b):
    if isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    elif isinstance(a, dict):
        test.assertEqual(set(a), set(b))
        for key in a:
            assert_tree(test, a[key], b[key])
    elif isinstance(a, (tuple, list)):
        test.assertEqual(len(a), len(b))
        for x, y in zip(a, b):
            assert_tree(test, x, y)
    else:
        test.assertEqual(a, b)


class VariantTests(unittest.TestCase):
    def make(self, cls=DualChain, seed=29):
        obj = cls(FakeBackend(), seed=seed, model_version=7,
                  score_normalization=1., score_temperature=1.)
        obj.reset(np.array([.2]), episode_id="case", model_version=7)
        return obj

    def advance(self, chain, valid=True):
        action = chain.act()
        state = chain.step(np.array([.4]) if valid else None, action,
                           episode_id="case", timestamp=chain.timestamp + 1,
                           model_version=7, valid=valid)
        return action, state

    def test_clean_exact_parity_all_steps_and_rng(self):
        chains = [self.make(cls) for cls in (DualChain, LockedAncestor, PosteriorReturnOnce)]
        for _ in range(30):
            results = [self.advance(x) for x in chains]
            for result in results[1:]:
                assert_tree(self, results[0], result)
        for obj in chains[1:]:
            assert_tree(self, chains[0].state_dict()["rngs"], obj.state_dict()["rngs"])

    def test_locked_ancestor_keeps_index_and_consumes_unused_draws(self):
        ref, locked = self.make(), self.make(LockedAncestor)
        draws, selected = [], []
        for _ in range(5):
            self.advance(ref, False)
            self.advance(locked, False)
            draws.append(locked.last_candidate_draw)
            selected.append(locked.locked_index)
            assert_tree(self, ref.state_dict()["rngs"], locked.state_dict()["rngs"])
            assert_tree(self, ref.pending, locked.pending)
        self.assertEqual(len(set(selected)), 1)
        self.assertGreater(len(set(draws)), 1)
        self.advance(ref)
        self.advance(locked)
        assert_tree(self, ref.state, locked.state)
        assert_tree(self, ref.state_dict()["rngs"], locked.state_dict()["rngs"])
        self.assertIsNone(locked.locked_index)

    def test_return_posterior_once_no_score_no_candidate_advance(self):
        obj = self.make(PosteriorReturnOnce)
        for _ in range(5):
            self.advance(obj, False)
        previous = obj.state
        nprior = obj.backend.prior_calls
        self.advance(obj)
        self.assertEqual(len(obj.backend.observed), 2)  # initial + return
        self.assertEqual(len(obj.backend.scored), 0)
        self.assertEqual(obj.backend.prior_calls - nprior, 1)  # independent dream only
        assert_tree(self, obj.backend.observed[-1][0], previous)
        self.assertIsNone(obj.pending)
        self.assertIsNone(obj.repaired_segment)
        self.assertEqual(len(obj.unselected_history["times"]), 5)

    def test_long_gaps_preserve_exact_v2_trajectory_and_rng(self):
        for length in (6, 10, 20, 50):
            ref, other = self.make(), self.make(PosteriorReturnOnce)
            for t in range(length + 3):
                assert_tree(self, self.advance(ref, t >= length),
                            self.advance(other, t >= length))
                assert_tree(self, ref.state_dict()["rngs"], other.state_dict()["rngs"])
            self.assertIsNone(other.unselected_history)

    def test_snapshot_midgap_restores_next_state_action_and_rng(self):
        for cls in (LockedAncestor, PosteriorReturnOnce):
            original, restored = self.make(cls), self.make(cls, seed=1)
            for _ in range(3):
                self.advance(original, False)
            restored.load_state_dict(pickle.loads(pickle.dumps(original.state_dict())))
            for valid in (False, False, True, True):
                assert_tree(self, self.advance(original, valid), self.advance(restored, valid))
                assert_tree(self, original.state_dict(), restored.state_dict())

    def test_reset_clears_variant_state(self):
        for cls in (LockedAncestor, PosteriorReturnOnce):
            obj = self.make(cls)
            self.advance(obj, False)
            self.advance(obj)
            obj.reset(np.array([.3]), episode_id="new", model_version=7)
            self.assertIsNone(getattr(obj, "locked_index", None))
            self.assertIsNone(getattr(obj, "unselected_history", None))

    def test_invalid_variant_snapshot_does_not_mutate_live_state(self):
        for cls in (LockedAncestor, PosteriorReturnOnce):
            obj = self.make(cls)
            self.advance(obj, False)
            before = obj.state_dict()
            broken = copy.deepcopy(before)
            broken["variant"]["name"] = "wrong"
            with self.assertRaises(ValueError):
                obj.load_state_dict(broken)
            assert_tree(self, before, obj.state_dict())

    def test_factory_uses_original_root_seed_stream_derivation(self):
        cfg = dict(max_candidates=8, max_missing=5, score_temperature=.01,
                   score_normalization=12288)
        states = []
        for arm in ("clean", "black_frame", "prior_only", "v2_exact", "locked_ancestor", "posterior_return_once"):
            chain = make_chain(FakeBackend(), arm, seed=2020000,
                               model_version=7, candidate_config=cfg)
            states.append(chain.state_dict()["rngs"])
        for state in states[1:]:
            assert_tree(self, states[0], state)


if __name__ == "__main__":
    unittest.main()
