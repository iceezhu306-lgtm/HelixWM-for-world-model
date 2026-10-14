"""CPU-only launch and boundary-recovery contracts; no JAX or simulator runs."""
import copy
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest import mock

import numpy as np

from helixwm_v2 import train
from helixwm_v2.dual_chain import DualChain
from helixwm_v2.evaluation import DropoutSchedule
from helixwm_v2.preflight import canonical_config_sha256
from helixwm_v2.storage import sha256, tree_hash


ROOT = Path(__file__).resolve().parents[1]


def configuration():
    return json.loads((ROOT / 'configs/dual_chain_walker_seed0_1m.json').read_text())


class FakeBackbone:
    """Restore all model/optimizer/key/counter fields, not just policy weights."""
    action_dim = 6
    score_kind = 'negative_fake_pixel_mse_energy'

    def __init__(self, version=1125):
        self.inference_version = version
        self.saved = None

    def load(self, saved):
        self.saved = copy.deepcopy(saved)
        self.inference_version = saved['inference_version']

    def initial(self, batch=1):
        return {'deter': np.zeros((batch, 1)), 'stoch': np.zeros((batch, 1))}

    def observe(self, state, action, image, is_first=False, rng=None):
        return {'deter': np.zeros((1, 1)) if is_first else state['deter'] + 1,
                'stoch': np.full((1, 1), float(np.asarray(image).mean()))
                         + rng.normal(0, .01, (1, 1))}

    def prior(self, state, action, rng=None):
        return {'deter': state['deter'] + 1,
                'stoch': state['stoch'] + rng.normal(0, .1, state['stoch'].shape)}

    def score(self, states, image):
        return -np.square(states['stoch'][:, 0] - np.asarray(image).mean())

    def act(self, state, deterministic=False, rng=None):
        return np.tanh(np.repeat(state['stoch'], 6, axis=1) + rng.normal(0, .01, (1, 6)))


class FakeReplay:
    def __init__(self):
        self.rng = np.random.default_rng(900)
        self.saved = None
        self.persist = None

    def restore(self, saved, *, persist=False):
        self.saved = copy.deepcopy(saved)
        self.persist = persist
        self.rng.bit_generator.state = copy.deepcopy(saved['rng_state'])


class FakeEnv:
    def __init__(self):
        self.rng = np.random.RandomState(800)
        self.saved = None
        self.done = True
        self.reset_count = 0

    def restore(self, saved):
        self.saved = copy.deepcopy(saved)
        self.rng.set_state(saved['task_random'])
        self.done = saved['done']

    def reset(self):
        self.reset_count += 1
        self.done = False
        return self.rng.uniform(size=(1,)).astype(np.float32)


def completed_payload():
    cfg = configuration()
    backbone = FakeBackbone()
    chain = DualChain(backbone, seed=303, **cfg['dual_chain'])
    chain.reset(np.array([.2]), episode_id=19, model_version=1125)
    for step in range(1, 501):
        action = chain.act()
        chain.step(None if step == 1 else np.array([.3]), action, episode_id=19,
                   timestamp=step, model_version=1125, valid=step != 1)
    dropout = DropoutSchedule(7001, cfg['observation_dropout'])
    for _ in range(20):
        dropout.valid(is_first=True)
        for step in range(1, 501):
            dropout.valid(is_last=step == 500)
    rng = np.random.default_rng(101)
    rng.random(30)
    return {
        'schema_version': 2, 'config': cfg,
        'counters': {'agent_steps': 10000, 'env_steps': 20000, 'episodes': 20,
                     'update_steps': 1250, 'masked_observations': dropout.missing,
                     'wall_time_s': 100.5},
        'backbone': {'schema_version': 1, 'inference_version': 1250,
                     'inference_equals_training': True, 'inference_params': None,
                     'training': {'params': {'model': np.arange(3, dtype=np.float32),
                                             'optimizer/momentum': np.array([.1, .2], np.float32),
                                             'optimizer/step': np.array(1250, np.int32)},
                                  'counters': {'updates': 1250, 'batches': 1250, 'actions': 0},
                                  'rng': np.array([31, 97], np.uint32)}},
        'replay': {'format_version': 2, 'archive': [], 'active_ids': [],
                   'total_steps_seen': 10000, 'next_id': 20,
                   'rng_state': copy.deepcopy(rng.bit_generator.state)},
        'collector': {'next_operation': 'reset_new_episode_and_reanchor',
                      'environment': {'done': True, 'episode_steps': 1000,
                                      'dmc_step_count': 1000,
                                      'task_random': np.random.RandomState(71).get_state(),
                                      'integration_state': np.array([.5, .7])},
                      'dual_chain': chain.state_dict(), 'dropout': dropout.state_dict()},
        'rng': {'python': random.Random(41).getstate(),
                'numpy': np.random.RandomState(42).get_state()},
        'scheduler': {'target_updates': 1250, 'train_ratio': 256},
    }


class LaunchTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.config = configuration()
        self.identity = {'helix_commit': 'approved-source', 'helix_tree': 'approved-tree',
                         'upstream_commit': train.UPSTREAM_COMMIT,
                         'config_sha256': canonical_config_sha256(self.config)}
        self.identity['identity_sha256'] = canonical_config_sha256(self.identity)
        self.gate = {'passed': True, 'identity': copy.deepcopy(self.identity)}
        self.smoke = {'identity': copy.deepcopy(self.identity), 'budget_complete': True,
                      'agent_steps': 1000, 'env_steps': 2000, 'checkpoint_reload_equal': True,
                      'evaluation_report': {'status': 'complete', 'hash_gate': {'passed': True}}}
        self.gate_path = self.directory / 'gate.json'
        self.smoke_path = self.directory / 'completion.json'
        self.write_evidence()

    def write_evidence(self, exit_status='0'):
        self.gate_path.write_text(json.dumps(self.gate), encoding='utf-8')
        self.smoke_path.write_text(json.dumps(self.smoke), encoding='utf-8')
        (self.directory / 'train.exit').write_text(exit_status, encoding='utf-8')

    def verify(self, config=None):
        return train.verify_launch_gates(self.config if config is None else config, self.identity,
                                         self.gate_path, self.smoke_path)

    def test_formal_config_and_evidence_are_bound_by_actual_file_hashes(self):
        train.validate_config(self.config)
        result = self.verify()
        self.assertEqual(result, {'preflight_sha256': sha256(self.gate_path),
                                  'smoke_completion_sha256': sha256(self.smoke_path)})

    def test_gate_and_smoke_reject_different_source_config_or_upstream(self):
        for record_name in ('gate', 'smoke'):
            for field in ('helix_commit', 'helix_tree', 'upstream_commit', 'config_sha256'):
                with self.subTest(record=record_name, field=field):
                    record = getattr(self, record_name)
                    original = copy.deepcopy(record['identity'])
                    binding = {k: v for k, v in original.items() if k != 'identity_sha256'}
                    binding[field] = 'different'
                    record['identity'] = {**binding, 'identity_sha256': canonical_config_sha256(binding)}
                    self.write_evidence()
                    with self.assertRaisesRegex(ValueError, 'different code/config/upstream'):
                        self.verify()
                    record['identity'] = original
        self.write_evidence()

    def test_in_memory_config_cannot_change_after_preflight(self):
        changed = copy.deepcopy(self.config)
        changed['observation_dropout']['start_probability'] = .03
        with self.assertRaisesRegex(ValueError, 'configuration'):
            self.verify(changed)

    def test_launch_requires_complete_smoke_frozen_eval_and_exit_zero(self):
        original = copy.deepcopy(self.smoke)
        for changes in ({'budget_complete': False}, {'agent_steps': 500}, {'env_steps': 1000},
                        {'checkpoint_reload_equal': False},
                        {'evaluation_report': {'status': 'failed', 'hash_gate': {'passed': True}}},
                        {'evaluation_report': {'status': 'complete', 'hash_gate': {'passed': False}}}):
            with self.subTest(changes=changes):
                self.smoke = {**original, **changes}
                self.write_evidence()
                with self.assertRaisesRegex(ValueError, 'smoke'):
                    self.verify()
        self.smoke = original
        for exit_status in ('1', '', 'running'):
            with self.subTest(exit_status=exit_status):
                self.write_evidence(exit_status)
                with self.assertRaisesRegex(ValueError, 'exit successfully'):
                    self.verify()
        self.write_evidence()
        self.gate['passed'] = False
        self.write_evidence()
        with self.assertRaisesRegex(ValueError, 'preflight'):
            self.verify()
        for gate, smoke in ((None, self.smoke_path), (self.gate_path, None)):
            with self.assertRaisesRegex(ValueError, 'requires both'):
                train.verify_launch_gates(self.config, self.identity, gate, smoke)

    def test_configuration_rejects_wrong_units_or_partial_episode_boundaries(self):
        cases = [('budget_agent_steps', 999999), ('checkpoint_every', 750),
                 ('eval_every', 15500), ('prefill_agent_steps', 1000000),
                 ('prefill_agent_steps', -1), ('train_ratio', 0)]
        for field, value in cases:
            with self.subTest(field=field):
                cfg = copy.deepcopy(self.config)
                cfg[field] = value
                with self.assertRaises(ValueError):
                    train.validate_config(cfg)
        for field, value in (('action_repeat', 1), ('episode_env_steps', 500)):
            cfg = copy.deepcopy(self.config)
            cfg['environment'][field] = value
            with self.assertRaisesRegex(ValueError, 'repeat 2'):
                train.validate_config(cfg)

    def test_formal_budget_cannot_be_reduced_even_with_mocked_passing_gates(self):
        cfg = copy.deepcopy(self.config)
        cfg['budget_agent_steps'] = 10000
        with mock.patch.object(train, 'read_identity', return_value=self.identity), \
             mock.patch.object(train, 'verify_launch_gates', return_value={}), \
             mock.patch.object(train, 'OfficialDreamerBackbone') as backbone:
            with self.assertRaisesRegex(ValueError, 'one million actions'):
                train._run(cfg, self.directory / 'fresh', self.directory)
            backbone.assert_not_called()

    def test_existing_experiment_artifacts_are_preserved_before_any_backend_load(self):
        for artifact in ('config.json', 'train.pid', 'completion.json', 'metrics.jsonl',
                         'REPLAY/index.json', 'CHECKPOINTS/step.pkl'):
            with self.subTest(artifact=artifact), tempfile.TemporaryDirectory() as temporary:
                run_dir = Path(temporary)
                path = run_dir / artifact
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b'preserve this experiment')
                with mock.patch.object(train, 'read_identity') as read_identity, \
                     mock.patch.object(train, 'OfficialDreamerBackbone') as backbone:
                    with self.assertRaises(FileExistsError):
                        train._run(self.config, run_dir, self.directory)
                    read_identity.assert_not_called()
                    backbone.assert_not_called()
                self.assertEqual(path.read_bytes(), b'preserve this experiment')

    def test_lock_refuses_a_second_writer_of_the_same_run(self):
        path = self.directory / 'locked-run'
        with train.exclusive_run_lock(path):
            with self.assertRaisesRegex(RuntimeError, 'another process'):
                with train.exclusive_run_lock(path):
                    self.fail('A second writer acquired the existing run')


class BoundaryRecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original = completed_payload()

    def setUp(self):
        python_state, numpy_state = random.getstate(), np.random.get_state()
        self.addCleanup(random.setstate, python_state)
        self.addCleanup(np.random.set_state, numpy_state)
        self.payload = copy.deepcopy(self.original)

    def components(self):
        backbone, replay, env = FakeBackbone(version=0), FakeReplay(), FakeEnv()
        chain = DualChain(backbone, seed=999, **self.payload['config']['dual_chain'])
        dropout = DropoutSchedule(999, self.payload['config']['observation_dropout'])
        return backbone, replay, chain, dropout, env

    def test_boundary_restores_every_component_and_host_rng_without_reset_or_sampling(self):
        before = tree_hash(self.payload)
        backbone, replay, chain, dropout, env = self.components()
        manager = mock.Mock()
        for name, obj, method in (('backbone', backbone, 'load'), ('replay', replay, 'restore'),
                                  ('env', env, 'restore'), ('chain', chain, 'restore_completed_episode'),
                                  ('dropout', dropout, 'load_state_dict')):
            wrapped = mock.Mock(wraps=getattr(obj, method))
            setattr(obj, method, wrapped)
            manager.attach_mock(wrapped, name)
        result = train.restore_boundary(self.payload, backbone, replay, chain, dropout, env)
        self.assertEqual([call[0] for call in manager.mock_calls],
                         ['backbone', 'replay', 'env', 'chain', 'dropout'])
        self.assertEqual(tree_hash(backbone.saved), tree_hash(self.payload['backbone']))
        self.assertEqual(tree_hash(replay.saved), tree_hash(self.payload['replay']))
        self.assertTrue(replay.persist)
        self.assertEqual(tree_hash(env.saved), tree_hash(self.payload['collector']['environment']))
        self.assertEqual(tree_hash(dropout.state_dict()), tree_hash(self.payload['collector']['dropout']))
        self.assertEqual(tree_hash(random.getstate()), tree_hash(self.payload['rng']['python']))
        self.assertEqual(tree_hash(np.random.get_state()), tree_hash(self.payload['rng']['numpy']))
        self.assertEqual(env.reset_count, 0)
        self.assertFalse(chain.initialized)
        self.assertIsNone(chain.real)
        self.assertEqual(chain.model_version, 1250)
        self.assertEqual(result, self.payload['counters'])
        self.assertIsNot(result, self.payload['counters'])
        self.assertEqual(tree_hash(self.payload), before)
        with self.assertRaisesRegex(RuntimeError, 'reset must precede act'):
            chain.act()

    def test_restored_next_reset_action_dropout_and_replay_draw_match_saved_state(self):
        first = self.components()
        second = self.components()
        train.restore_boundary(self.payload, *first)
        reference = {'python': random.random(), 'numpy': np.random.uniform(size=8)}
        first_image = first[4].reset()
        first[2].reset(first_image, episode_id=20, model_version=1250)
        reference['action'] = first[2].act()
        reference['mask'] = [first[3].valid(is_first=True)] + [first[3].valid() for _ in range(100)]
        reference['replay_draw'] = first[1].rng.integers(0, 10000, size=40)
        random.seed(999)
        np.random.seed(999)
        train.restore_boundary(self.payload, *second)
        self.assertEqual(reference['python'], random.random())
        np.testing.assert_array_equal(reference['numpy'], np.random.uniform(size=8))
        second_image = second[4].reset()
        np.testing.assert_array_equal(first_image, second_image)
        second[2].reset(second_image, episode_id=20, model_version=1250)
        np.testing.assert_array_equal(reference['action'], second[2].act())
        self.assertEqual(reference['mask'],
                         [second[3].valid(is_first=True)] + [second[3].valid() for _ in range(100)])
        np.testing.assert_array_equal(reference['replay_draw'], second[1].rng.integers(0, 10000, size=40))
        self.assertEqual(tree_hash(first[2].state_dict()), tree_hash(second[2].state_dict()))

    def test_unfinished_or_wrong_operation_is_rejected_before_component_mutation(self):
        cases = [('operation', 'continue_current_episode'), ('done', False), ('agent_steps', 10001)]
        for field, value in cases:
            with self.subTest(field=field):
                payload = copy.deepcopy(self.payload)
                if field == 'operation':
                    payload['collector']['next_operation'] = value
                elif field == 'done':
                    payload['collector']['environment']['done'] = value
                else:
                    payload['counters'][field] = value
                objects = [mock.Mock() for _ in range(5)]
                with self.assertRaises(ValueError):
                    train.restore_boundary(payload, *objects)
                for obj in objects:
                    self.assertEqual(obj.mock_calls, [])

    def test_inconsistent_budget_environment_replay_or_learner_counts_fail_before_mutation(self):
        cases = [
            (('counters', 'agent_steps'), 0),
            (('counters', 'agent_steps'), True),
            (('counters', 'agent_steps'), -500),
            (('counters', 'agent_steps'), 10000.0),
            (('counters', 'env_steps'), 10000),
            (('counters', 'episodes'), 19),
            (('collector', 'environment', 'episode_steps'), 999),
            (('replay', 'total_steps_seen'), 9500),
            (('counters', 'update_steps'), 1249),
            (('backbone', 'training', 'counters', 'updates'), 1249),
            (('backbone', 'training', 'counters', 'batches'), 1251),
            (('scheduler', 'target_updates'), 1249),
            (('scheduler', 'train_ratio'), 255),
        ]
        for path, value in cases:
            with self.subTest(path=path, value=value):
                payload = copy.deepcopy(self.payload)
                target = payload
                for key in path[:-1]:
                    target = target[key]
                target[path[-1]] = value
                objects = [mock.Mock() for _ in range(5)]
                with self.assertRaises(ValueError):
                    train.restore_boundary(payload, *objects)
                for obj in objects:
                    self.assertEqual(obj.mock_calls, [])


if __name__ == '__main__':
    unittest.main()
