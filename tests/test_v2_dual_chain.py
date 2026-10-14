"""Bounded causal inference tests; no simulator, training, or author dependencies."""
import copy
import pickle
import unittest

import numpy as np

from helixwm_v2.dual_chain import DualChain


class FakeBackend:
    action_dim = 2
    score_kind = "negative_fake_decoder_mse_energy"

    def __init__(self):
        self.observed = []
        self.scored = []

    def initial(self, batch=1):
        return {"deter": np.zeros((batch, 1)), "stoch": np.zeros((batch, 1))}

    def observe(self, state, action, image, is_first=False, rng=None):
        self.observed.append(float(np.asarray(image).mean()))
        noise = rng.normal(0, .01, size=(1, 1))
        return {"deter": np.zeros((1, 1)) if is_first else state["deter"] + state["stoch"] + 1,
                "stoch": np.full((1, 1), float(np.asarray(image).mean())) + noise}

    def prior(self, state, action, rng=None):
        return {"deter": state["deter"] + state["stoch"] + 1,
                "stoch": state["stoch"] + np.asarray(action).sum() + rng.normal(size=state["stoch"].shape)}

    def score(self, states, image):
        self.scored.append(float(np.asarray(image).mean()))
        return -np.square(states["stoch"][:, 0] - float(np.asarray(image).mean()))

    def act(self, state, deterministic=False, rng=None):
        noise = 0. if deterministic else rng.normal(size=(1, 2)) * .05
        return np.tanh(np.repeat(state["stoch"], 2, axis=1)) + noise


def equal_state(test, first, second):
    for key in first:
        np.testing.assert_array_equal(first[key], second[key])


def equal_tree(test, first, second):
    """Compare numerical recovery, not pickle object-alias bookkeeping."""
    if isinstance(first, np.ndarray):
        test.assertEqual(first.dtype, second.dtype)
        np.testing.assert_array_equal(first, second)
    elif isinstance(first, dict):
        test.assertEqual(set(first), set(second))
        for key in first:
            equal_tree(test, first[key], second[key])
    elif isinstance(first, (tuple, list)):
        test.assertEqual(type(first), type(second))
        test.assertEqual(len(first), len(second))
        for left, right in zip(first, second):
            equal_tree(test, left, right)
    else:
        test.assertEqual(first, second)


class DualChainTests(unittest.TestCase):
    def make(self, **kwargs):
        chain = DualChain(FakeBackend(), score_normalization=1., score_temperature=1., **kwargs)
        chain.reset(np.array([.2]), episode_id="episode", model_version="frozen-1")
        return chain

    def advance(self, chain, value, valid=True):
        action = chain.act()
        return chain.step(value, action, episode_id="episode", timestamp=chain.timestamp + 1,
                          model_version="frozen-1", valid=valid)

    def test_clean_shadow_and_disabled_do_not_change_real_rng_or_actions(self):
        chains = [self.make(seed=7, mode=mode) for mode in ("disabled", "shadow", "repair")]
        for time in range(1, 5):
            actions = [chain.act() for chain in chains]
            for action in actions[1:]:
                np.testing.assert_array_equal(actions[0], action)
            states = [chain.step(np.array([time / 10]), action, episode_id="episode",
                                  timestamp=time, model_version="frozen-1")
                      for chain, action in zip(chains, actions)]
            for state in states[1:]:
                equal_state(self, states[0], state)
        self.assertTrue(all(not chain.events for chain in chains))

    def test_missing_pixels_are_ignored_and_weights_do_not_change(self):
        first, second = self.make(seed=5), self.make(seed=5)
        for time in range(1, 4):
            self.advance(first, np.array([np.nan]), False)
            self.advance(second, np.array([1e30]), False)
            equal_state(self, first.state, second.state)
            np.testing.assert_array_equal(first.pending["log_weights"], np.full(8, -np.log(8)))
        self.assertEqual(len(first.backend.observed), 1)
        self.assertEqual(len(first.backend.scored), 0)

    def test_reconnect_preserves_ancestry_and_reanchors_both_chains(self):
        chain = self.make(seed=2)
        self.advance(chain, None, False)
        self.advance(chain, None, False)
        self.advance(chain, np.array([.4]), True)
        event = chain.pop_events()[-1]
        self.assertEqual(event["event"], "repair_reconnected")
        self.assertEqual(event["segment_times"], [1, 2, 3])
        self.assertEqual(event["evidence_cutoff"], 3)
        self.assertEqual(event["available_at"], 3)
        self.assertFalse(event["prior_counted_again"])
        self.assertFalse(event["observed_teacher_qualified"])
        self.assertEqual(len(chain.repaired_segment["states"]), 3)
        equal_state(self, chain.real, chain.dream)
        equal_state(self, chain.real, chain.repaired_segment["states"][-1])
        self.assertIsNone(chain.pending)
        self.assertEqual(chain.anchor_time, 3)

    def test_raw_arrays_are_not_changed(self):
        chain = self.make(seed=9)
        for value, valid in [(np.array([.7]), False), (np.array([.8]), True)]:
            raw = value.copy()
            action = chain.act()
            saved_action = action.copy()
            chain.step(value, action, episode_id="episode", timestamp=chain.timestamp + 1,
                       model_version="frozen-1", valid=valid)
            np.testing.assert_array_equal(raw, value)
            np.testing.assert_array_equal(saved_action, action)

    def test_future_evidence_cannot_change_earlier_online_state(self):
        first, second = self.make(seed=12), self.make(seed=12)
        states = []
        for _ in range(2):
            states.append(self.advance(first, None, False))
            equal_state(self, first.state, self.advance(second, None, False))
        frozen = copy.deepcopy(states)
        self.advance(first, np.array([100.]), True)
        self.advance(second, np.array([-100.]), True)
        for before, after in zip(frozen, states):
            equal_state(self, before, after)

    def test_long_missing_sequence_abstains_then_recovers_on_observation(self):
        chain = self.make()
        for _ in range(7):
            self.advance(chain, None, False)
        self.assertTrue(chain.abstaining)
        self.assertIsNone(chain.pending)
        self.assertEqual(chain.events[-1]["reason"], "missing_segment_budget_exceeded")
        self.advance(chain, np.array([.1]))
        self.assertFalse(chain.abstaining)
        self.assertEqual(chain.anchor_time, 8)

    def test_reset_clears_old_lineage_and_allows_new_version(self):
        chain = self.make()
        self.advance(chain, None, False)
        chain.reset(np.array([.3]), episode_id="second", model_version="frozen-2")
        self.assertIsNone(chain.pending)
        self.assertEqual(chain.timestamp, 0)
        self.assertIsNone(chain.last_action)
        self.assertEqual(chain.model_version, "frozen-2")
        self.assertEqual(chain.events[-1]["reason"], "episode_reset")

    def test_wrong_time_action_episode_and_version_are_rejected_before_mutation(self):
        chain = self.make()
        action = chain.act()
        snapshot = pickle.dumps(chain.state_dict())
        cases = [dict(timestamp=2), dict(model_version="frozen-2"),
                 dict(episode_id="wrong"), dict(executed_action=action + .01)]
        for change in cases:
            arguments = dict(observation=np.array([.1]), executed_action=action,
                             episode_id="episode", timestamp=1, model_version="frozen-1")
            arguments.update(change)
            with self.assertRaises(ValueError):
                chain.step(**arguments)
            self.assertEqual(snapshot, pickle.dumps(chain.state_dict()))

    def test_midsegment_and_pending_action_save_restore_parity(self):
        first = self.make(seed=24)
        self.advance(first, None, False)
        action = first.act()
        saved = pickle.loads(pickle.dumps(first.state_dict()))
        second = self.make(seed=999)
        second.load_state_dict(saved)
        for chain in (first, second):
            chain.step(None, action, episode_id="episode", timestamp=2,
                       model_version="frozen-1", valid=False)
        equal_state(self, first.state, second.state)
        for chain in (first, second):
            self.advance(chain, np.array([.25]), True)
        equal_state(self, first.state, second.state)
        np.testing.assert_array_equal(first.act(), second.act())
        self.assertEqual(first.pop_events(), second.pop_events())

    def test_no_evidence_never_scores_and_event_queue_is_bounded(self):
        chain = self.make(max_events=2)
        for _ in range(3):
            self.advance(chain, None, False)
            self.advance(chain, np.array([.2]), True)
        self.assertEqual(len(chain.events), 2)
        self.assertEqual(len(chain.backend.scored), 3)

    def test_action_is_clipped_and_actual_clipped_value_is_checked(self):
        chain = self.make()
        chain.backend.act = lambda *args, **kwargs: np.array([[2., -2.]])
        action = chain.act()
        np.testing.assert_array_equal(action, [[1., -1.]])
        chain.step(np.array([.2]), action[0], episode_id="episode", timestamp=1,
                   model_version="frozen-1")

    def test_wrong_snapshot_configuration_and_missing_rng_fail(self):
        chain = self.make()
        saved = chain.state_dict()
        saved["rngs"].pop("repair")
        with self.assertRaises(ValueError):
            chain.load_state_dict(saved)
        saved = chain.state_dict()
        saved["config"]["max_candidates"] = 2
        with self.assertRaises(ValueError):
            chain.load_state_dict(saved)

    def test_actual_backend_version_must_match_declared_snapshot(self):
        chain = self.make()
        action = chain.act()
        chain.backend.inference_version = "different"
        with self.assertRaises(ValueError):
            chain.act()
        with self.assertRaises(ValueError):
            chain.step(np.array([.1]), action, episode_id="episode", timestamp=1,
                       model_version="frozen-1")
        with self.assertRaises(ValueError):
            chain.reset(np.array([.1]), episode_id="episode", model_version="frozen-1")

    def test_nonfinite_evidence_abstains_without_generated_pixel_observation(self):
        chain = self.make()
        self.advance(chain, None, False)
        chain.backend.score = lambda states, image: np.full(8, np.nan)
        self.advance(chain, np.array([.3]), True)
        self.assertEqual(chain.events[-1]["reason"], "nonfinite_candidate_evidence")
        self.assertIsNone(chain.pending)
        equal_state(self, chain.real, chain.dream)

    def completed(self, seed=42):
        chain = self.make(seed=seed)
        chain.backend.inference_version = "frozen-1"
        for time in range(1, 501):
            self.advance(chain, None if time == 1 else np.array([time / 1000]), valid=time != 1)
        return chain

    def test_completed_episode_restore_reanchors_new_version_with_exact_rng_continuation(self):
        uninterrupted = self.completed()
        saved = pickle.loads(pickle.dumps(uninterrupted.state_dict()))
        original_saved_bytes = pickle.dumps(saved)
        uninterrupted.backend.inference_version = "frozen-2"
        restored = self.make(seed=999)
        restored.backend.inference_version = "frozen-2"
        audit = restored.restore_completed_episode(saved)
        self.assertFalse(audit["old_latent_state_reused"])
        self.assertEqual(audit["completed_chain_model_version"], "frozen-1")
        self.assertEqual(audit["next_inference_model_version"], "frozen-2")
        self.assertFalse(restored.initialized)
        self.assertIsNone(restored.real)
        self.assertIsNone(restored.anchor)
        self.assertIsNone(restored.repaired_segment)
        self.assertEqual(restored.branch_counter, uninterrupted.branch_counter)
        self.assertEqual(restored.event_counter, uninterrupted.event_counter)
        with self.assertRaises(RuntimeError):
            restored.act()
        for chain in (uninterrupted, restored):
            chain.reset(np.array([.8]), episode_id="next_episode", model_version="frozen-2")
        equal_state(self, uninterrupted.state, restored.state)
        for time, valid in [(1, False), (2, True), (3, True)]:
            actions = [chain.act() for chain in (uninterrupted, restored)]
            np.testing.assert_array_equal(actions[0], actions[1])
            for chain, action in zip((uninterrupted, restored), actions):
                chain.step(np.array([.4]) if valid else None, action, episode_id="next_episode",
                           timestamp=time, model_version="frozen-2", valid=valid)
            equal_state(self, uninterrupted.state, restored.state)
        self.assertEqual(uninterrupted.pop_events(), restored.pop_events())
        equal_tree(self, uninterrupted.state_dict(), restored.state_dict())
        self.assertEqual(original_saved_bytes, pickle.dumps(saved))

    def test_completed_episode_restore_rejects_pending_gap_action_or_wrong_boundary(self):
        saved = self.completed().state_dict()
        target = self.make(seed=900)
        before = pickle.dumps(target.state_dict())
        changes = [dict(timestamp=499), dict(anchor_time=499),
                   dict(last_action=np.zeros((1, 2), np.float32)),
                   dict(pending={"unresolved": True}), dict(missing_length=1),
                   dict(abstaining=True), dict(initialized=False)]
        for change in changes:
            malformed = copy.deepcopy(saved)
            malformed["state"].update(change)
            with self.assertRaises(ValueError):
                target.restore_completed_episode(malformed)
            self.assertEqual(before, pickle.dumps(target.state_dict()))

    def test_completed_episode_restore_validates_all_rng_and_configuration_before_mutation(self):
        saved = self.completed().state_dict()
        target = self.make(seed=900)
        before = pickle.dumps(target.state_dict())
        bad_config = copy.deepcopy(saved)
        bad_config["config"]["max_missing"] = 4
        missing_rng = copy.deepcopy(saved)
        missing_rng["rngs"].pop("dream")
        invalid_rng = copy.deepcopy(saved)
        invalid_rng["rngs"]["repair"] = {"bit_generator": "not_a_generator"}
        for malformed in (bad_config, missing_rng, invalid_rng):
            with self.assertRaises(ValueError):
                target.restore_completed_episode(malformed)
            self.assertEqual(before, pickle.dumps(target.state_dict()))


if __name__ == "__main__":
    unittest.main()
