"""Read-only Final inference extensions to the unchanged matched Dreamer core.

No learner, loss, optimizer, or upstream file is modified. All auxiliary head
parameters are copied under the same lock and version as the policy parameters.
The three new calls consume only explicit caller-owned inference RNG keys.
"""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import numpy as np

from helixwm_matched.backbone import (
    MatchedDreamerBackbone, UPSTREAM_COMMIT, array_tree_hash, rng_key,
)


FORMAT = "helixwm_final_backbone_v1"
AUX_FORMAT = "helixwm_final_auxiliary_heads_v1"
UPSTREAM_ARCHIVE_SHA256 = "33c508483bb6f04a327459b7f1f8ce4dab72ce5d2cda99eb23dbca050a3ebc43"
# Verified from the archived e3f0224 source. Only CRLF/LF is normalized here;
# the parent adapter additionally requires the exact Git commit and clean tree.
UPSTREAM_API_SHA256 = {
    "dreamerv3/rssm.py": "d6d50166914e94fb8bd17a5d5dbda9d42cdd37b85819bb1e9fff3a64d4ad2eb6",
    "dreamerv3/agent.py": "adce8e4274bc098c218bf9a20fd3327545f0ad7d850b5fe328597382e91b5269",
    "embodied/jax/outs.py": "7e80691f175c71be614f089023cce3a809e0d026c6d5ce89bf566d5f11eb3ed0",
}


def validate_final_upstream_api(path):
    """Bind the private RSSM/distribution APIs to the inspected author source."""
    root = Path(path)
    actual = {}
    for name, expected in UPSTREAM_API_SHA256.items():
        value = (root / name).read_bytes().replace(b"\r\n", b"\n")
        actual[name] = hashlib.sha256(value).hexdigest()
        if actual[name] != expected:
            raise ValueError(f"Final inference upstream API differs: {name}")
    return actual


def auxiliary_keys(params):
    """Only author reward/continuation model parameters, never optimizer slots."""
    keys = tuple(sorted(k for k in params if k.startswith(("rew/", "con/"))))
    if not any(k.startswith("rew/") for k in keys) or not any(k.startswith("con/") for k in keys):
        raise ValueError("Both fixed reward and continuation heads are required")
    return keys


def host_parameter_schema(params):
    return {name: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for name, value in sorted(params.items())}


def inference_bundle_hash(policy, auxiliary, version):
    if type(version) is not int or version < 0:
        raise ValueError("Pinned inference version must be a nonnegative integer")
    return array_tree_hash({"policy": policy, "auxiliary_heads": auxiliary,
                            "model_version": np.asarray(version, np.int64)})


def validate_distribution_info(state, info, expected_version):
    """CPU validation of the explicitly returned single-model distribution."""
    if set(info) != {"probs", "entropy", "model_version"}:
        raise ValueError("Unexpected distribution diagnostic fields")
    if type(info["model_version"]) is not int or info["model_version"] != expected_version:
        raise ValueError("Distribution model version differs from the pinned bundle")
    p, entropy = np.asarray(info["probs"]), np.asarray(info["entropy"])
    if p.dtype != np.float32 or p.shape != state["stoch"].shape or p.ndim != 3:
        raise ValueError("Probabilities must be float32 [B,S,C]")
    if entropy.dtype != np.float32 or entropy.shape != (len(p),):
        raise ValueError("Entropy must be float32 [B]")
    if not np.isfinite(p).all() or not np.isfinite(entropy).all():
        raise FloatingPointError("Nonfinite inference distribution")
    if np.any(p < 0) or np.any(p > 1) or not np.allclose(p.sum(-1), 1, rtol=0, atol=2e-6):
        raise ValueError("Invalid categorical probabilities")
    if np.any(entropy < 0) or np.any(entropy > p.shape[1] * np.log(p.shape[2]) + 1e-4):
        raise ValueError("Invalid categorical entropy")


class FinalBackbone(MatchedDreamerBackbone):
    """Same author learner, with version-pinned diagnostic heads and RSSM APIs.

    ``pin_inference`` retains the parent integer-version return contract. The
    complete bundle identity is available from ``inference_bundle_metadata``.
    ``image_mean`` is the author's decoder prediction in [0,1], not uint8 RGB.
    Reward and continuation are model predictions, not measured outcomes.
    """

    def __init__(self, config, upstream_path, run_dir):
        self._final_api_evidence = validate_final_upstream_api(upstream_path)
        self._auxiliary_params = None
        self._auxiliary_version = -1
        self._auxiliary_hash = None
        super().__init__(config, upstream_path, run_dir)

    def _build_inference(self):
        super()._build_inference()
        jax, jnp, nj, model = self.jax, self.jnp, self.nj, self.model
        from embodied.jax import nets as nn

        def info(logit):
            # RSSM._dist = Agg(OneHot(Categorical(logit, unimix)), 1, sum).
            # Read the distribution AFTER the author's unimix transformation.
            dist = model.dyn._dist(logit)
            probs = jax.nn.softmax(dist.output.dist.logits, -1).astype(jnp.float32)
            return {"probs": probs, "entropy": dist.entropy().astype(jnp.float32)}

        def advance(state, actions):
            state, (feat, _) = model.dyn.imagine(
                nn.cast(state), {"action": nn.cast(actions)}, length=1,
                training=False, single=True)
            return state, info(feat["logit"])

        def posterior(deter, images):
            # Exactly the posterior suffix of e3f0224 RSSM._observe. In
            # particular this does NOT call _core, observe, or imagine.
            reset = jnp.zeros(deter.shape[:1], bool)
            _, _, tokens = model.enc({}, {"image": images}, reset,
                                      training=False, single=True)
            deter, tokens = nn.cast((deter, tokens))
            tokens = tokens.reshape((*deter.shape[:-1], -1))
            dyn = model.dyn
            # Author RSSM._observe is a Module method: Ninjax enters dyn.path
            # before its submodule lookups. This adapter suffix must enter the
            # same absolute scope explicitly; no extra transition/RNG draw.
            with nj.scope(dyn.path, absolute=True):
                x = tokens if dyn.absolute else jnp.concatenate([deter, tokens], -1)
                for i in range(dyn.obslayers):
                    x = dyn.sub(f"obs{i}", nn.Linear, dyn.hidden, **dyn.kw)(x)
                    x = nn.act(dyn.act)(dyn.sub(f"obs{i}norm", nn.Norm, dyn.norm)(x))
                logit = dyn._logit("obslogit", x)
                stoch = nn.cast(dyn._dist(logit).sample(seed=nj.seed()))
                return {"deter": deter, "stoch": stoch}, info(logit)

        def heads(state, with_image):
            state = nn.cast(state)
            features = model.feat2tensor(state)
            result = {"reward": model.rew(features, bdims=1).pred().astype(jnp.float32),
                      "continuation": model.con(features, bdims=1).prob(1).astype(jnp.float32)}
            if with_image:
                reset = jnp.zeros(state["deter"].shape[:1], bool)
                _, _, recon = model.dec({}, state, reset, training=False, single=True)
                result["image_mean"] = recon["image"].pred().astype(jnp.float32)
            return result

        def readonly(fn, static=()):
            pure = nj.pure(fn)
            def call(params, key, *args):
                # Unlike a learner, these functions may not create or modify
                # any model variable. The returned state is an ordinary value.
                _, output = pure(params, *args, seed=key, create=False, modify=False)
                return jax.lax.stop_gradient(output)
            return jax.jit(call, static_argnums=static)

        self._advance_prior_final = readonly(advance)
        self._posterior_from_deter_final = readonly(posterior)
        self._predict_heads_final = readonly(heads, static=(3,))

    def pin_inference(self):
        """Atomically copy policy AND auxiliary heads from the same update."""
        with self.agent.train_lock:
            keys = auxiliary_keys(self.agent.params)
            if set(keys) & set(self.agent.policy_keys):
                raise ValueError("Author policy and auxiliary parameter keys overlap")
            policy = {k: self.agent.params[k].copy() for k in self.agent.policy_keys}
            aux = {k: self.agent.params[k].copy() for k in keys}
            self.jax.block_until_ready((policy, aux))
            version = int(self.agent.n_updates)
            aux_hash = array_tree_hash(self.jax.device_get(aux))
            self.inference_params = policy
            self._auxiliary_params = aux
            self.inference_version = self._auxiliary_version = version
            self._auxiliary_hash = aux_hash
        return version

    def _require_bundle(self):
        if (self._auxiliary_params is None or type(self.inference_version) is not int
                or self.inference_version < 0 or type(self._auxiliary_version) is not int
                or self._auxiliary_version != self.inference_version):
            raise ValueError("Missing or stale fixed auxiliary head bundle")

    def inference_bundle_sha256(self):
        """Hash actual pinned policy + auxiliary arrays + version, never learner."""
        self._require_bundle()
        policy, auxiliary = self.jax.device_get((self.inference_params, self._auxiliary_params))
        return inference_bundle_hash(policy, auxiliary, self.inference_version)

    def inference_bundle_metadata(self):
        self._require_bundle()
        aux_hash = array_tree_hash(self.jax.device_get(self._auxiliary_params))
        if aux_hash != self._auxiliary_hash:
            raise ValueError("Pinned auxiliary parameters changed within their version")
        return {"model_version": self.inference_version,
                "model_sha256": self.inference_bundle_sha256(),
                "policy_sha256": array_tree_hash(self.jax.device_get(self.inference_params)),
                "auxiliary_sha256": aux_hash,
                "upstream_commit": UPSTREAM_COMMIT}

    def _distribution_output(self, result):
        state, diagnostic = self._host(result)
        state = self._state(state)
        diagnostic = {**diagnostic, "model_version": self.inference_version}
        validate_distribution_info(state, diagnostic, self.inference_version)
        return state, diagnostic

    def advance_prior(self, previous, action, rng=None):
        self._require_bundle()
        state = self._state(previous)
        args = (state, self._action(action, len(state["deter"])))
        result = self._advance_prior_final(self.inference_params,
                    self._device(rng_key(rng)), *self._device(args))
        return self._distribution_output(result)

    def posterior_from_deter(self, deter, image, rng=None):
        self._require_bundle()
        deter = np.asarray(deter)
        if deter.dtype != np.float32 or deter.ndim != 2 or len(deter) < 1:
            raise ValueError("Deter must be float32 [B,D_h]")
        if deter.shape != self.initial(len(deter))["deter"].shape or not np.isfinite(deter).all():
            raise ValueError("Deter differs from the pinned latent schema")
        args = (deter, self._image(image, len(deter)))
        result = self._posterior_from_deter_final(self.inference_params,
                    self._device(rng_key(rng)), *self._device(args))
        return self._distribution_output(result)

    def predict_heads(self, state, with_image=False):
        self._require_bundle()
        if type(with_image) is not bool:
            raise ValueError("with_image must be a bool")
        state = self._state(state)
        # No self.agent.params access: an active learner update cannot change
        # this episode's reward/continuation model.
        params = {**self.inference_params, **self._auxiliary_params}
        result = self._host(self._predict_heads_final(params, self._device(rng_key(0)),
                            self._device(state), with_image))
        expected = {"reward": (len(state["deter"]),), "continuation": (len(state["deter"]),)}
        if with_image:
            expected["image_mean"] = (len(state["deter"]), 64, 64, 3)
        if set(result) != set(expected):
            raise ValueError("Unexpected model-head prediction fields")
        for name, shape in expected.items():
            value = np.asarray(result[name])
            if value.dtype != np.float32 or value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f"Invalid fixed-head prediction: {name}")
        for name in ("continuation", "image_mean"):
            if name in result and (np.any(result[name] < 0) or np.any(result[name] > 1)):
                raise ValueError(f"Prediction outside probability/image range: {name}")
        return result

    def snapshot(self):
        self._require_bundle()
        payload = super().snapshot()
        bundle = self.inference_bundle_metadata()
        aux = {k: np.asarray(v).copy() for k, v in self.jax.device_get(self._auxiliary_params).items()}
        payload["format"] = FORMAT
        payload["final_auxiliary_heads"] = {
            "format": AUX_FORMAT, "model_version": self._auxiliary_version,
            "params": aux, "params_sha256": bundle["auxiliary_sha256"],
            "parameter_schema": host_parameter_schema(aux),
        }
        payload["final_inference_bundle"] = bundle
        payload["final_upstream_api_sha256"] = dict(self._final_api_evidence)
        return payload

    def load(self, snapshot):
        if snapshot.get("format") != FORMAT:
            raise ValueError("Final backbone requires its own checkpoint format")
        if snapshot.get("final_upstream_api_sha256") != UPSTREAM_API_SHA256:
            raise ValueError("Final checkpoint upstream API evidence differs")
        aux = snapshot.get("final_auxiliary_heads", {})
        if set(aux) != {"format", "model_version", "params", "params_sha256", "parameter_schema"} or aux.get("format") != AUX_FORMAT:
            raise ValueError("Invalid Final auxiliary checkpoint schema")
        version = aux.get("model_version")
        if type(version) is not int or version != snapshot.get("inference_version"):
            raise ValueError("Auxiliary checkpoint version is not the pinned policy version")
        reference = {k: self.agent.params[k] for k in auxiliary_keys(self.agent.params)}
        self._validate_loaded_parameters(reference, aux["params"])
        aux_hash = array_tree_hash(aux["params"])
        if aux_hash != aux["params_sha256"] or host_parameter_schema(aux["params"]) != aux["parameter_schema"]:
            raise ValueError("Auxiliary checkpoint shape/dtype/SHA256 evidence differs")
        expected_bundle = {"model_version": version,
            "model_sha256": inference_bundle_hash(snapshot["inference_params"], aux["params"], version),
            "policy_sha256": array_tree_hash(snapshot["inference_params"]),
            "auxiliary_sha256": aux_hash, "upstream_commit": UPSTREAM_COMMIT}
        if snapshot.get("final_inference_bundle") != expected_bundle:
            raise ValueError("Final policy/auxiliary bundle hash or version mismatch")
        base = dict(snapshot)
        base["format"] = "helixwm_matched_backbone_v1"
        # Parent validates all training/optimizer/carry counters and schemas
        # before loading. No old matched checkpoint is accepted as Final.
        super().load(base)
        copied_aux = {k: np.asarray(v).copy() for k, v in aux["params"].items()}
        self._auxiliary_params = self._device(copied_aux)
        self._auxiliary_version = version
        self._auxiliary_hash = aux_hash
