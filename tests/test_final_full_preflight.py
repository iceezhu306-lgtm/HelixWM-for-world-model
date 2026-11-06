"""Integrated stage orchestration on CPU fakes, not GPU/runtime acceptance."""
import copy
import json
from pathlib import Path
import pickle
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np

from helixwm_final import full_preflight as f,protocol_v3 as p,protocol_v2 as old
from test_final_train import FakeEnv,FakeController,identity


class Replay:
    def __init__(self):self.updates=0;self.rows=0;self.last_rows=0
    def add(self,observation,action,state):
        self.rows+=1;self.last_rows+=int(observation['is_last'])
    def train_once(self):self.updates+=1;return {'loss':.25,'gradient_norm':.5}
    def snapshot(self):return {'synthetic_replay':True}


class Auxiliary:
    def __init__(self):self.count=0;self.bindings=[]
    def update(self,previous_image,action,next_image,reward,continuation,**kw):
        self.count+=1
        if kw['next_available'] is False and np.any(next_image):raise AssertionError('Hidden image leaked')
    def pin(self):return SimpleNamespace(sha256='f'*64,snapshot=lambda:{'count':self.count})
    def snapshot(self):return {'count':self.count}


class Backbone:
    def __init__(self,replay):self.replay=replay;self.version=0;self.pins=[]
    def pin_inference(self):self.version=self.replay.updates;self.pins.append(self.version)
    def snapshot(self):return {'version':self.version}
    def inference_bundle_sha256(self):return 'c'*64


class Controller(FakeController):
    def __init__(self,mode):self.mode=mode
    def snapshot(self):return {'mode':self.mode,'timestamp':self.timestamp}


class FullPreflightTests(unittest.TestCase):
    def setUp(self):self.config=p.registered_protocol()

    def run_loop(self,env=None,gate_failure=False):
        counts=dict(agent_steps=0,env_steps=0,episodes=0,update_steps=0)
        replay=Replay();backbone=Backbone(replay);auxiliary=Auxiliary();events=[]
        def gate(heads):
            events.append(('gate',counts['agent_steps'],backbone.version))
            if gate_failure:raise RuntimeError('Synthetic gate evidence failure')
            return {'active':False},{'model_version':backbone.version},{'path':str(counts['agent_steps'])}
        def final(heads,context,gate):
            events.append(('final',counts['agent_steps'],context['model_version']))
            self.assertFalse(gate['active'])
            return Controller('bank_production_native_fallback')
        result=f.integration_loop(config=self.config,backbone=backbone,auxiliary=auxiliary,native=Controller('native'),
            env=env or FakeEnv(),replay=replay,prefill_rng=np.random.default_rng(1810000),counts=counts,
            measure_gate=gate,make_final=final)
        return result,counts,replay,backbone,auxiliary,events

    def test_first_eleven_masks_exactly_preserve_registered_smoke(self):
        masks=p.full_preflight_masks(self.config)
        self.assertEqual(masks.shape,(12,501));self.assertEqual(masks.dtype,np.bool_)
        np.testing.assert_array_equal(masks[:11],old.make_mask_bank(old.registered_protocol(),partition='smoke'))
        self.assertTrue(masks[:,0].all());self.assertTrue(masks[:,-1].all())
        self.assertTrue((~masks[-1]).any())

    def test_actual_episode_engine_counts_native_then_full_controller_without_extra_steps(self):
        result,counts,replay,backbone,auxiliary,events=self.run_loop()
        self.assertEqual(counts,dict(agent_steps=6000,env_steps=12000,episodes=12,update_steps=250))
        self.assertEqual(events,[('gate',5500,125),('final',5500,125),('gate',6000,250)])
        self.assertEqual(replay.rows,6012);self.assertEqual(replay.last_rows,12)
        self.assertEqual(auxiliary.count,6000);self.assertEqual(replay.updates,250)
        self.assertEqual(len(backbone.pins),13)
        self.assertEqual(result['previous_gate_reference'],{'path':'5500'})
        self.assertEqual(result['gate_reference'],{'path':'6000'})
        self.assertEqual(result['completed_controller']['mode'],'bank_production_native_fallback')
        self.assertFalse(any(x['active'] for x in result['bank_gate_ledger']))

    def test_gate_error_stops_at_5500_without_starting_final_episode(self):
        with self.assertRaisesRegex(RuntimeError,'gate evidence'):
            self.run_loop(gate_failure=True)

    def test_render_failure_preserves_real_acknowledged_interactions(self):
        counts=dict(agent_steps=0,env_steps=0,episodes=0,update_steps=0)
        replay=Replay();aux=Auxiliary();backbone=Backbone(replay)
        with self.assertRaisesRegex(RuntimeError,'rendering'):
            f.integration_loop(config=self.config,backbone=backbone,auxiliary=aux,native=Controller('native'),
                env=FakeEnv(raise_after=7),replay=replay,prefill_rng=np.random.default_rng(1810000),counts=counts,
                measure_gate=lambda h:self.fail('No gate after failure'),make_final=lambda *x:self.fail('No Final after failure'))
        self.assertEqual(counts,dict(agent_steps=7,env_steps=14,episodes=0,update_steps=0))

    def test_resume_counts_and_formal_identity_cannot_impersonate_preflight(self):
        bad=identity();bad['arm']='F'
        with self.assertRaisesRegex(ValueError,'preflight identity'):f.validate_probe_identity(bad,self.config)
        bad['run_id']='preflight_test';f.validate_probe_identity(bad,self.config)
        with self.assertRaisesRegex(ValueError,'6000'):f.validate_probe_identity(bad,self.config,dict(agent_steps=10000))
        with self.assertRaisesRegex(ValueError,'Fresh zero'):
            f.integration_loop(config=self.config,backbone=None,auxiliary=None,native=None,env=None,replay=None,
                prefill_rng=None,counts=dict(agent_steps=5500,env_steps=11000,episodes=11,update_steps=125),
                measure_gate=None,make_final=None)

    def test_registration_does_not_change_the_formal_million_action_budget(self):
        registration=self.config['full_integration_preflight']
        self.assertEqual(registration['counts']['agent_steps'],6000)
        self.assertEqual(self.config['training']['actions_per_arm'],1000000)
        self.assertEqual(registration['formal_training_actions'],0)
        self.assertTrue(registration['fresh_initialization']);self.assertFalse(registration['old_checkpoint_loaded'])
        self.assertEqual(registration['bank_replay_environment_actions'],0)

    def test_full_checkpoint_metadata_explicitly_preserves_preflight_scope(self):
        ident=identity();ident.update(arm='F',run_id='preflight_checkpoint')
        counts=copy.deepcopy(self.config['full_integration_preflight']['counts'])
        replay=Replay();backbone=Backbone(replay);aux=Auxiliary();heads=aux.pin()
        def dependencies(root,path,*args):
            path.write_bytes(b'{"synthetic":true}\n');return f.train.descriptor(path,root)
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            with patch.object(f.train,'_checkpoint_dependencies',side_effect=dependencies),patch('helixwm_matched.backbone.NativeReplay.referenced_files',return_value=[]):
                record=f.save_checkpoint(root,config=self.config,identity=ident,counts=counts,backbone=backbone,
                    replay=replay,auxiliary=aux,heads=heads,completed_controller={'synthetic':True},
                    prefill_rng=np.random.default_rng(1810000),gate={'artifact_sha256':'e'*64},trajectory=None,
                    bank_reference={'path':'BANK_SHARED'},gate_reference={'path':'6000'},previous_gate_reference={'path':'5500'})
            with (root/record['checkpoint']['path']).open('rb') as stream:payload=pickle.load(stream)
            for value in (record,payload):
                self.assertEqual(value['format'],f.FORMAT);self.assertEqual(value['formal_training_actions'],0)
                self.assertTrue(value['fresh_initialization']);self.assertFalse(value['old_checkpoint_loaded'])
                self.assertEqual(value['probe_registration'],self.config['full_integration_preflight'])
                self.assertEqual(value['counts'],counts)
            self.assertEqual(record['completed_previous_gate_reference'],{'path':'5500'})
            self.assertFalse(payload['auto_resume_allowed'])


if __name__=='__main__':unittest.main()
