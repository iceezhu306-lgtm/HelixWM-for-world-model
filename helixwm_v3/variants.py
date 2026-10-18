"""One-component inference changes around the preserved V2 implementation."""
from __future__ import annotations

import copy
import numpy as np

from helixwm_v2.dual_chain import DualChain, _copy, _digest


ARMS = ("clean", "black_frame", "prior_only", "v2_exact",
        "locked_ancestor", "posterior_return_once")


class LockedAncestor(DualChain):
    """Keep the first selected path; super.step still consumes every RNG draw."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.locked_index = None
        self.last_candidate_draw = None
        self._missing_step = False

    def reset(self, *args, **kwargs):
        result = super().reset(*args, **kwargs)
        self.locked_index = self.last_candidate_draw = None
        return result

    def _advance_candidates(self, action, timestamp):
        super()._advance_candidates(action, timestamp)
        if self._missing_step:
            # Predict the next choice without consuming the original stream.
            # The unmodified parent immediately makes this same choice.
            clone = np.random.default_rng()
            clone.bit_generator.state = copy.deepcopy(self.rngs["repair"].bit_generator.state)
            weights = np.exp(self.pending["log_weights"])
            self.last_candidate_draw = int(clone.choice(len(weights), p=weights))

    def step(self, observation, executed_action, **kwargs):
        valid = kwargs.get("valid", True)
        self._missing_step = not valid
        try:
            super().step(observation, executed_action, **kwargs)
        finally:
            self._missing_step = False
        if not valid and self.pending is not None:
            if self.locked_index is None:
                self.locked_index = self.last_candidate_draw
            self.real = _copy(self.pending["states"][self.locked_index])
            self._event("locked_candidate_control", timestamp=self.timestamp,
                        original_selection_draw=self.last_candidate_draw,
                        selected_candidate=self.locked_index,
                        discarded_selection_draw=self.last_candidate_draw != self.locked_index)
        else:
            self.locked_index = self.last_candidate_draw = None
        return self.state

    def state_dict(self):
        saved = super().state_dict()
        saved["variant"] = {"name": "locked_ancestor", "version": 1,
                            "locked_index": self.locked_index,
                            "last_candidate_draw": self.last_candidate_draw}
        return saved

    def load_state_dict(self, saved):
        v = saved.get("variant", {})
        if set(v) != {"name", "version", "locked_index", "last_candidate_draw"} or \
                v["name"] != "locked_ancestor" or v["version"] != 1:
            raise ValueError("Invalid locked-ancestor snapshot")
        for key in ("locked_index", "last_candidate_draw"):
            value = v[key]
            if value is not None and (type(value) is not int or not 0 <= value < self.config["max_candidates"]):
                raise ValueError("Invalid candidate index in snapshot")
        if (v["locked_index"] is None) != (saved["state"]["pending"] is None):
            raise ValueError("Candidate lock disagrees with pending segment")
        super().load_state_dict(saved)
        self.locked_index, self.last_candidate_draw = v["locked_index"], v["last_candidate_draw"]
        self._missing_step = False

    def restore_completed_episode(self, saved):
        raise NotImplementedError("Study recovery starts a separately seeded next case, not a training episode")


class PosteriorReturnOnce(DualChain):
    """Replace bounded-gap return resampling with one native posterior update."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.unselected_history = None

    def reset(self, *args, **kwargs):
        result = super().reset(*args, **kwargs)
        self.unselected_history = None
        return result

    def step(self, observation, executed_action, **kwargs):
        if not kwargs.get("valid", True) or self.pending is None:
            return super().step(observation, executed_action, **kwargs)
        version = kwargs.get("model_version", self.model_version)
        version = self.model_version if version is None else version
        action = self._validate_step(executed_action, kwargs["episode_id"], kwargs["timestamp"], version)
        if observation is None:
            raise ValueError("A valid return observation cannot be absent")
        previous = _copy(self.real)
        timestamp = kwargs["timestamp"]
        # Preserve the parent's independent one-step dream prediction, but do
        # not advance/score candidates or resample with this returned image.
        self.last_prediction = self.backend.prior(_copy(self.dream), action.copy(), rng=self.rngs["dream"])
        self.unselected_history = _copy(self.pending)
        self.real = self._normal_observe(previous, action, observation)
        self.repaired_segment = None
        self._event("posterior_return_once", anchor_time=self.pending["anchor_time"],
                    evidence_cutoff=timestamp, available_at=timestamp,
                    observation_sha256=_digest(observation),
                    unselected_candidate_path_sha256=[_digest(x) for x in self.pending["paths"]],
                    posterior_updates=1, return_energy_evaluations=0,
                    retrospective_history_repaired=False)
        self.dream = _copy(self.real)
        self.anchor, self.anchor_time = _copy(self.real), timestamp
        self.pending = None
        self.abstaining, self.missing_length = False, 0
        self.timestamp, self.last_action = timestamp, None
        return self.state

    def state_dict(self):
        saved = super().state_dict()
        saved["variant"] = {"name": "posterior_return_once", "version": 1,
                            "unselected_history": _copy(self.unselected_history)}
        return saved

    def load_state_dict(self, saved):
        v = saved.get("variant", {})
        if set(v) != {"name", "version", "unselected_history"} or \
                v["name"] != "posterior_return_once" or v["version"] != 1:
            raise ValueError("Invalid posterior-return snapshot")
        if v["unselected_history"] is not None and not isinstance(v["unselected_history"], dict):
            raise ValueError("Invalid diagnostic history")
        super().load_state_dict(saved)
        self.unselected_history = _copy(v["unselected_history"])

    def restore_completed_episode(self, saved):
        raise NotImplementedError("Study recovery starts a separately seeded next case, not a training episode")


def make_chain(backbone, arm, *, seed, model_version, candidate_config):
    if arm not in ARMS:
        raise ValueError(f"Unregistered arm: {arm}")
    cls = {"locked_ancestor": LockedAncestor,
           "posterior_return_once": PosteriorReturnOnce}.get(arm, DualChain)
    config = {key: candidate_config[key] for key in
              ("max_candidates", "max_missing", "score_temperature", "score_normalization")}
    mode = "disabled" if arm in ("black_frame", "prior_only") else "repair"
    return cls(backbone, seed=seed, model_version=model_version, mode=mode, **config)
