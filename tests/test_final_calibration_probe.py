"""CPU boundary checks only; none of these fixtures is GPU acceptance."""
import copy
import json
from pathlib import Path
import random
import tempfile
import unittest

import numpy as np

from helixwm_final import calibration_probe as p


class Environment:
    def __init__(self):
        self.action_decisions=0;self.episode_env_steps=0
        self.rendered=False;self.closed=False
    def reset(self,*,seed,mask):
        assert seed==72099 and mask.shape==(501,) and mask.all()
        self.rendered=True
        return np.zeros((64,64,3),np.uint8)
    def close(self):self.closed=True


class Backbone:
    def __init__(self):
        self.state={'training':{'params':{'weight':np.array([1.,2.]),'optimizer':np.array([3.])},
                               'counters':{'actions':5500,'updates':125}},
                    'train_carry':{'x':np.zeros(2)},'extra_snapshot_state':np.ones(2)}
    def snapshot(self):return copy.deepcopy(self.state)
    def inference_bundle_sha256(self):return 'a'*64


class Heads:
    def __init__(self):self.state={'weights':np.array([1.])}
    def snapshot(self):return copy.deepcopy(self.state)


class CalibrationProbeTests(unittest.TestCase):
    def test_runtime_gate_renders_before_lock_and_closes_without_actions(self):
        env=Environment()
        def lock(config):
            self.assertTrue(env.rendered)
            return {'engine':'native'}
        result=p.verify_environment_runtime({}, {'engine':'native'},env_factory=lambda:env,lock_fn=lock)
        self.assertTrue(result['passed']);self.assertTrue(env.closed)
        self.assertEqual((result['agent_steps'],result['env_steps']),(0,0))

    def test_mismatched_renderer_closes_and_rejects(self):
        env=Environment()
        with self.assertRaisesRegex(RuntimeError,'lock differs'):
            p.verify_environment_runtime({}, {'engine':'native'},env_factory=lambda:env,
                                         lock_fn=lambda config:{'engine':'other'})
        self.assertTrue(env.closed)

    def test_full_snapshot_and_real_head_rehash_cover_unlabelled_fields(self):
        backbone,heads=Backbone(),Heads()
        before,whole=p.capture_frozen_state(backbone,heads)
        after,whole_after=p.capture_frozen_state(backbone,heads)
        self.assertEqual(before,after);self.assertEqual(whole,whole_after)
        backbone.state['extra_snapshot_state'][0]=8
        after,whole_after=p.capture_frozen_state(backbone,heads)
        self.assertEqual(before,after);self.assertNotEqual(whole,whole_after)
        heads.state['weights'][0]=9
        self.assertNotEqual(before['epistemic_heads'],p.capture_frozen_state(backbone,heads)[0]['epistemic_heads'])
        backbone.state['training']['params']['optimizer'][0]=4
        self.assertNotEqual(before['optimizer'],p.capture_frozen_state(backbone,heads)[0]['optimizer'])

    def test_host_rng_is_covered_without_leaving_test_side_effects(self):
        state=random.getstate()
        try:
            first=p.capture_frozen_state(Backbone(),Heads())[0]
            random.random()
            second=p.capture_frozen_state(Backbone(),Heads())[0]
            self.assertNotEqual(first['training_rng'],second['training_rng'])
        finally:random.setstate(state)

    def test_parent_requires_measured_bootstrap_counts_and_head_lineage(self):
        parent=dict(stage='isolated_shared_native_bootstrap_preflight',fresh_initialization=True,
                    old_checkpoint_loaded=False,formal_training_actions=0,
                    source_sha256='a'*64,protocol_sha256='b'*64)
        counts=dict(agent_steps=5500,env_steps=11000,episodes=11,update_steps=125)
        binding={k:parent[k] for k in ('source_sha256','protocol_sha256')}
        payload=dict(format='helixwm_final_bootstrap_probe_v1',identity=parent,counts=counts,
                     heads={'binding':binding},auxiliary={'binding':binding})
        p.validate_parent(payload,parent,{'counts':counts})
        wrong=copy.deepcopy(payload);wrong['counts']['update_steps']=0
        with self.assertRaisesRegex(ValueError,'incomplete'):p.validate_parent(wrong,parent,{'counts':counts})
        wrong=copy.deepcopy(payload);wrong['heads']['binding']={'source_sha256':'f'*64,'protocol_sha256':'b'*64}
        with self.assertRaisesRegex(ValueError,'lineage'):p.validate_parent(wrong,parent,{'counts':counts})

    def test_failure_counts_include_finished_fit_and_cumulative_failed_validation(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            (root/'fit_seed55000_gap1.counts.json').write_text(json.dumps({'actions':25,'dmc_calls':50}))
            (root/'validation_seed55010_gap1.counts.json').write_text(json.dumps({'actions':25,'dmc_calls':50}))
            (root/'validation.failure.json').write_text(json.dumps({'acknowledged_actions':35,'acknowledged_dmc_calls':69}))
            self.assertEqual(p.acknowledged_counts(root,ValueError(),{}),
                             {'agent_steps':60,'env_steps':119,'update_steps':0})


if __name__=='__main__':unittest.main()
