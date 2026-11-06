"""CPU contracts for Final; no JAX/model/environment is constructed here.

Set DREAMERV3_ARCHIVE to the verified author tar for source-binding checks.
GPU posterior/prior parity remains a distinct mandatory integration gate.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import inspect
import os
from pathlib import Path
import tarfile
import textwrap
import threading
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from helixwm_final.backbone import (
    AUX_FORMAT, FORMAT, FinalBackbone, UPSTREAM_API_SHA256,
    UPSTREAM_ARCHIVE_SHA256, auxiliary_keys, inference_bundle_hash,
    validate_distribution_info, validate_final_upstream_api,
)
from helixwm_matched.backbone import array_tree_hash


class FakeAgent:
    def __init__(self):
        self.params = {name: np.array([value], np.float32) for name, value in {
            "enc/w": 1, "dyn/w": 2, "dec/w": 3, "pol/w": 4,
            "rew/w": 5, "con/w": .75, "opt/rew/w": 900,
        }.items()}
        self.policy_keys = ("enc/w", "dyn/w", "dec/w", "pol/w")
        self.train_lock = threading.RLock()
        self.n_updates = self.n_batches = 3
        self.pending_outs = self.pending_mets = None
        self.load_calls = 0

    def save(self):
        return {"params": copy.deepcopy(self.params),
                "counters": {"updates": self.n_updates, "batches": self.n_batches, "actions": 0}}

    def load(self, payload):
        self.load_calls += 1
        self.params = copy.deepcopy(payload["params"])
        self.n_updates = payload["counters"]["updates"]
        self.n_batches = payload["counters"]["batches"]


def fake_backbone():
    b = object.__new__(FinalBackbone)
    b.agent = FakeAgent()
    b.jax = SimpleNamespace(device_get=copy.deepcopy, block_until_ready=lambda x: x)
    b._device = copy.deepcopy
    b._host = copy.deepcopy
    b.internal = SimpleNamespace(device_put=lambda x, sharding: copy.deepcopy(x))
    b.agent.train_sharded = "fake"
    b.config = SimpleNamespace(agent=SimpleNamespace(dyn=SimpleNamespace(
        rssm=SimpleNamespace(deter=4, stoch=2, classes=3))))
    b.resolved_config = {"seed": 0, "logdir": "cpu-test-only", "schema": "tiny mock"}
    b.train_carry = {"carry": np.zeros((1, 4), np.float32)}
    b.writeback_updates = 3
    b._final_api_evidence = dict(UPSTREAM_API_SHA256)
    b.pin_inference()
    return b


def fake_distribution(state):
    batch, stochastic, classes = state["stoch"].shape
    return {"probs": np.full((batch, stochastic, classes), 1/classes, np.float32),
            "entropy": np.full(batch, stochastic*np.log(classes), np.float32)}


def fake_heads(params, key, state, with_image):
    batch = len(state["deter"])
    result = {"reward": np.full(batch, params["rew/w"][0], np.float32),
              "continuation": np.full(batch, params["con/w"][0], np.float32)}
    if with_image:
        result["image_mean"] = np.full((batch, 64, 64, 3), .5, np.float32)
    return result


class FinalBackboneUtilityTests(unittest.TestCase):
    def test_auxiliary_keys_exclude_optimizer_and_require_both_heads(self):
        self.assertEqual(auxiliary_keys(FakeAgent().params), ("con/w", "rew/w"))
        for params in ({"rew/w": 1}, {"con/w": 1}, {"opt/rew/w": 1}):
            with self.assertRaises(ValueError): auxiliary_keys(params)

    def test_hash_binds_actual_policy_auxiliary_and_version(self):
        b = fake_backbone()
        original = b.inference_bundle_sha256()
        b.agent.params["rew/w"][:] = 999
        self.assertEqual(original, b.inference_bundle_sha256())
        for field in ("policy", "auxiliary", "version"):
            policy, aux = copy.deepcopy(b.inference_params), copy.deepcopy(b._auxiliary_params)
            version = b.inference_version
            if field == "policy": policy["dyn/w"] += 1
            if field == "auxiliary": aux["rew/w"] += 1
            if field == "version": version += 1
            self.assertNotEqual(original, inference_bundle_hash(policy, aux, version))

    def test_distribution_rejects_invalid_dtype_probability_entropy_and_version(self):
        state = fake_backbone().initial(2)
        valid = {**fake_distribution(state), "model_version": 3}
        validate_distribution_info(state, valid, 3)
        for mutation in ("dtype", "shape", "nan", "sum", "negative", "entropy", "version", "bool_version", "extra"):
            bad = copy.deepcopy(valid)
            if mutation == "dtype": bad["probs"] = bad["probs"].astype(np.float64)
            if mutation == "shape": bad["probs"] = bad["probs"][:1]
            if mutation == "nan": bad["probs"][0, 0, 0] = np.nan
            if mutation == "sum": bad["probs"][0, 0] *= .5
            if mutation == "negative": bad["probs"][0, 0, 0] = -.1
            if mutation == "entropy": bad["entropy"][0] = 100
            if mutation == "version": bad["model_version"] = 4
            if mutation == "bool_version": bad["model_version"] = True
            if mutation == "extra": bad["epistemic"] = True
            with self.subTest(mutation=mutation), self.assertRaises((ValueError, FloatingPointError)):
                validate_distribution_info(state, bad, 3)

    def test_posterior_kernel_has_no_transition_and_one_sampling_call(self):
        tree = ast.parse(inspect.getsource(__import__('helixwm_final.backbone', fromlist=['x'])))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "FinalBackbone")
        build = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_build_inference")
        posterior = next(n for n in build.body if isinstance(n, ast.FunctionDef) and n.name == "posterior")
        calls = [n.func.attr for n in ast.walk(posterior) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]
        self.assertFalse({"_core", "observe", "imagine"} & set(calls))
        self.assertEqual(calls.count("sample"), 1)
        self.assertEqual(calls.count("seed"), 1)

    def test_head_kernel_cannot_access_active_learner_parameters(self):
        source = inspect.getsource(FinalBackbone.predict_heads)
        tree = ast.parse(textwrap.dedent(source))
        accesses = [n for n in ast.walk(tree) if isinstance(n, ast.Attribute)
                    and isinstance(n.value, ast.Attribute) and n.value.attr == "agent"
                    and n.attr == "params"]
        self.assertEqual(accesses, [])

    def test_posterior_submodules_and_sample_use_author_absolute_parent_scope(self):
        tree = ast.parse(inspect.getsource(__import__('helixwm_final.backbone', fromlist=['x'])))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'FinalBackbone')
        build = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_build_inference')
        posterior = next(n for n in build.body if isinstance(n, ast.FunctionDef) and n.name == 'posterior')
        scopes = [n for n in ast.walk(posterior) if isinstance(n, ast.With)]
        self.assertEqual(len(scopes), 1)
        scope = scopes[0]
        call = scope.items[0].context_expr
        self.assertIsInstance(call, ast.Call)
        self.assertEqual(ast.unparse(call.func), 'nj.scope')
        self.assertEqual(ast.unparse(call.args[0]), 'dyn.path')
        self.assertEqual({k.arg: ast.literal_eval(k.value) for k in call.keywords}, {'absolute': True})
        all_calls = [n for n in ast.walk(posterior) if isinstance(n, ast.Call)]
        scoped_calls = list(ast.walk(scope))
        for call in all_calls:
            if isinstance(call.func, ast.Attribute) and call.func.attr in ('sub', '_logit', 'sample', 'seed'):
                self.assertIn(call, scoped_calls)

    @unittest.skipUnless(os.environ.get('DREAMERV3_ARCHIVE'), 'Set DREAMERV3_ARCHIVE for pinned author source audit')
    def test_fixed_author_archive_api_binding_and_modified_source_rejection(self):
        path = Path(os.environ['DREAMERV3_ARCHIVE'])
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), UPSTREAM_ARCHIVE_SHA256)
        source = {}
        with tarfile.open(path) as archive:
            for name in UPSTREAM_API_SHA256:
                member = next(m for m in archive.getmembers() if m.name.endswith('/'+name))
                source[name] = archive.extractfile(member).read()
        def read_file(p):
            return next(value for name, value in source.items() if p.as_posix().endswith(name))
        with mock.patch.object(Path, 'read_bytes', read_file):
            self.assertEqual(validate_final_upstream_api(Path('fake-author-source')), UPSTREAM_API_SHA256)
        source['dreamerv3/rssm.py'] += b'\n# changed\n'
        with mock.patch.object(Path, 'read_bytes', read_file), self.assertRaises(ValueError):
            validate_final_upstream_api(Path('fake-author-source'))


class FinalBackboneMockContractTests(unittest.TestCase):
    def setUp(self):
        self.b = fake_backbone()

    def test_pin_copies_both_heads_same_version_and_preserves_rng(self):
        before_rng = array_tree_hash(np.random.get_state())
        b = self.b
        old = b.inference_bundle_metadata()
        b.agent.params["rew/w"][:] = 7
        b.agent.params["pol/w"][:] = 8
        self.assertEqual(b._auxiliary_params["rew/w"][0], 5)
        self.assertEqual(b.inference_params["pol/w"][0], 4)
        b.agent.n_updates = 4
        self.assertEqual(b.pin_inference(), 4)
        self.assertEqual(b._auxiliary_version, b.inference_version)
        self.assertNotEqual(old["model_sha256"], b.inference_bundle_sha256())
        self.assertEqual(array_tree_hash(np.random.get_state()), before_rng)

    def test_advance_returns_pinned_probs_and_uses_explicit_rng_without_input_mutation(self):
        b = self.b
        inputs = b.initial(2)
        before = copy.deepcopy(inputs)
        def kernel(params, key, state, action):
            self.assertEqual(key.dtype, np.uint32)
            self.assertEqual(action.shape, (2, 6))
            result = {"deter": state["deter"] + 1, "stoch": state["stoch"].copy()}
            return result, fake_distribution(result)
        b._advance_prior_final = kernel
        state, info = b.advance_prior(inputs, np.zeros((2, 6), np.float32), rng=17)
        self.assertTrue(np.array_equal(state["deter"], before["deter"]+1))
        self.assertTrue(np.array_equal(inputs["deter"], before["deter"]))
        self.assertEqual(info["model_version"], 3)
        self.assertEqual(set(state), {"deter", "stoch"})

    def test_posterior_preserves_supplied_current_deter_and_rejects_bad_inputs(self):
        b = self.b
        deter = np.full((2, 4), 17, np.float32)
        def kernel(params, key, h, image):
            state = {"deter": h.copy(), "stoch": b.initial(len(h))["stoch"]}
            return state, fake_distribution(state)
        b._posterior_from_deter_final = kernel
        state, info = b.posterior_from_deter(deter, np.zeros((2, 64, 64, 3), np.uint8), rng=99)
        np.testing.assert_array_equal(state["deter"], deter)
        for invalid in (deter.astype(np.float64), deter[:, :3], np.full((2,4), np.nan, np.float32)):
            with self.assertRaises(ValueError): b.posterior_from_deter(invalid, np.zeros((64,64,3),np.uint8), rng=99)

    def test_heads_stay_fixed_across_active_learner_updates(self):
        b = self.b
        b._predict_heads_final = fake_heads
        state = b.initial(2)
        old = b.predict_heads(state, with_image=True)
        b.agent.params["rew/w"][:] = -999
        b.agent.params["con/w"][:] = .01
        b.agent.n_updates = 4
        current = b.predict_heads(state, with_image=True)
        for key in old: np.testing.assert_array_equal(old[key], current[key])
        self.assertEqual(current["image_mean"].shape, (2, 64, 64, 3))
        b.pin_inference()
        self.assertEqual(b.predict_heads(state)["reward"][0], -999)

    def test_heads_reject_stale_version_and_invalid_predictions(self):
        b = self.b
        b._predict_heads_final = fake_heads
        b._auxiliary_version = 2
        with self.assertRaises(ValueError): b.predict_heads(b.initial())
        b._auxiliary_version = b.inference_version
        for mutation in ("dtype", "shape", "nan", "probability", "extra"):
            def kernel(params, key, state, image, mutation=mutation):
                out = fake_heads(params,key,state,image)
                if mutation == "dtype": out['reward'] = out['reward'].astype(np.float64)
                if mutation == "shape": out['reward'] = out['reward'][:,None]
                if mutation == "nan": out['reward'][0] = np.nan
                if mutation == "probability": out['continuation'][0] = 1.01
                if mutation == "extra": out['truth'] = True
                return out
            b._predict_heads_final = kernel
            with self.subTest(mutation=mutation), self.assertRaises(ValueError): b.predict_heads(b.initial())

    def test_auxiliary_change_within_version_is_detected(self):
        self.b._auxiliary_params["rew/w"] += 1
        with self.assertRaises(ValueError): self.b.inference_bundle_metadata()
        with self.assertRaises(ValueError): self.b.snapshot()

    def test_snapshot_has_no_aux_alias_and_roundtrips(self):
        payload = self.b.snapshot()
        self.assertEqual(payload['format'], FORMAT)
        self.assertEqual(payload['final_auxiliary_heads']['format'], AUX_FORMAT)
        target = fake_backbone()
        target.agent.params['rew/w'][:] = 123
        target.load(payload)
        self.assertEqual(target.inference_bundle_sha256(), self.b.inference_bundle_sha256())
        np.testing.assert_array_equal(target._auxiliary_params['rew/w'], np.array([5.],np.float32))
        payload['final_auxiliary_heads']['params']['rew/w'][:] = 4321
        self.assertEqual(target._auxiliary_params['rew/w'][0], 5)
        self.assertEqual(self.b._auxiliary_params['rew/w'][0], 5)

    def test_restore_rejects_missing_or_corrupt_aux_before_loading_active_training(self):
        source = self.b.snapshot()
        for mutation in ('old_format','api','missing','aux_format','version','bool_version','keys','shape','dtype','nan','hash','schema','bundle_policy','bundle_version'):
            payload = copy.deepcopy(source)
            aux = payload['final_auxiliary_heads']
            if mutation == 'old_format': payload['format'] = 'helixwm_matched_backbone_v1'
            if mutation == 'api': payload['final_upstream_api_sha256']['dreamerv3/rssm.py'] = 'x'
            if mutation == 'missing': del payload['final_auxiliary_heads']
            if mutation == 'aux_format': aux['format'] = 'old'
            if mutation == 'version': aux['model_version'] = 4
            if mutation == 'bool_version': aux['model_version'] = True
            if mutation == 'keys': del aux['params']['con/w']
            if mutation == 'shape': aux['params']['rew/w'] = np.zeros(2,np.float32)
            if mutation == 'dtype': aux['params']['rew/w'] = aux['params']['rew/w'].astype(np.float64)
            if mutation == 'nan': aux['params']['rew/w'][0] = np.nan
            if mutation == 'hash': aux['params']['rew/w'][0] += 1
            if mutation == 'schema': aux['parameter_schema']['rew/w']['shape'] = [2]
            if mutation == 'bundle_policy': payload['final_inference_bundle']['policy_sha256'] = 'bad'
            if mutation == 'bundle_version': payload['final_inference_bundle']['model_version'] = 4
            target = fake_backbone()
            before = array_tree_hash(target.agent.params)
            with self.subTest(mutation=mutation), self.assertRaises((ValueError, FloatingPointError)):
                target.load(payload)
            self.assertEqual(target.agent.load_calls, 0)
            self.assertEqual(array_tree_hash(target.agent.params), before)


if __name__ == '__main__':
    unittest.main()
