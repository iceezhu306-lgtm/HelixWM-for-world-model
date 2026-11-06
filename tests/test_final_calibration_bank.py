"""CPU-only bank/factory contracts with a deterministic fake simulator."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np

from helixwm_final.calibration_bank import collect_bank, load_bank, fixed_actions, sha256
from helixwm_final.environment import FinalWalkerEnv, original_environment_contract
from helixwm_final.protocol import load_protocol, protocol_sha256
from helixwm_matched.environment import MatchedWalkerEnv
from test_matched_environment import FakeNative

BINDING={'environment_lock_sha256':'a'*64,'source_sha256':'b'*64}
CONFIG=Path(__file__).resolve().parents[1]/'configs/helixwm_final_matched_20261105.json'


class BankTests(unittest.TestCase):
    def setUp(self):self.config=load_protocol(CONFIG)

    def fake(self,config,**kwargs):return FinalWalkerEnv(config,env_factory=FakeNative,**kwargs)

    def test_same_factory_same_actions_exact_same_visible_results(self):
        new=self.fake(self.config)
        old=MatchedWalkerEnv(original_environment_contract(),env_factory=FakeNative)
        mask=np.ones(501,bool);mask[17:117]=False
        try:
            np.testing.assert_array_equal(new.reset(55000,mask),old.reset(55000,mask))
            for action in fixed_actions(55000):
                image,r,d,info=new.step(action);other,r2,d2,info2=old.step(action)
                np.testing.assert_array_equal(image,other)
                self.assertEqual((r,d),(r2,d2))
                for key in ('availability','observation_timestamp','env_steps_this_step','episode_env_steps'):
                    self.assertEqual(info[key],info2[key])
        finally:new.close();old.close()

    def test_bank_budget_splits_sha_shapes_and_global_rng(self):
        before=np.random.get_state()
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'bank'
            manifest=collect_bank(self.config,root,env_class=self.fake,**BINDING)
            bank,_=load_bank(root,sha256(root/'manifest.json'),protocol_sha256(self.config),**BINDING)
            self.assertEqual((manifest['actions'],manifest['dmc_calls']),(2480,4960))
            self.assertEqual(len(bank),20)
            self.assertTrue(set(bank).isdisjoint(range(60000,60040)))
            self.assertEqual(bank[55000]['image'].shape,(125,64,64,3))
            with self.assertRaises(FileExistsError):collect_bank(self.config,root,env_class=self.fake,**BINDING)
        after=np.random.get_state()
        self.assertEqual(before[0],after[0]);np.testing.assert_array_equal(before[1],after[1]);self.assertEqual(before[2:],after[2:])

    def test_replaced_array_or_seed_manifest_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'bank';collect_bank(self.config,root,env_class=self.fake,**BINDING)
            digest=sha256(root/'manifest.json')
            path=root/'seed55000.npz';raw=path.read_bytes();path.write_bytes(raw+b'changed')
            with self.assertRaises(ValueError):load_bank(root,digest,protocol_sha256(self.config),**BINDING)
            path.write_bytes(raw)
            manifest=json.loads((root/'manifest.json').read_text());manifest['fit_seeds'][0]=60000
            (root/'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):load_bank(root,digest,protocol_sha256(self.config),**BINDING)
            with self.assertRaises(ValueError):load_bank(root,sha256(root/'manifest.json'),protocol_sha256(self.config),**BINDING)

    def test_partial_bank_is_explicit_failure(self):
        class Broken(FakeNative):
            def step(self,a):
                if self.calls==4:raise RuntimeError('test injected simulator failure')
                return super().step(a)
        def broken(config,**kwargs):return FinalWalkerEnv(config,env_factory=Broken,**kwargs)
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'bank'
            with self.assertRaisesRegex(RuntimeError,'injected'):collect_bank(self.config,root,env_class=broken,**BINDING)
            failure=json.loads((root/'failure.json').read_text())
            self.assertTrue(failure['partial_bank_not_usable'])
            self.assertEqual((failure['acknowledged_actions'],failure['acknowledged_dmc_calls']),(2,4))
            self.assertFalse((root/'manifest.json').exists())

    def test_shared_environment_and_bank_seed_gates(self):
        for seed in (True,55000.0,60000):
            with self.assertRaises(ValueError):fixed_actions(seed)
        changed=copy.deepcopy(self.config);changed['environment']['camera_id']=1
        with self.assertRaises(ValueError):self.fake(changed)

    def test_no_silent_image_conversion(self):
        class WrongImage(FinalWalkerEnv):
            def __init__(self,cfg,**kw):super().__init__(cfg,env_factory=FakeNative,**kw)
            def reset(self,*args,**kwargs):return super().reset(*args,**kwargs).astype(np.float32)
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'bad'
            with self.assertRaisesRegex(ValueError,'raw image'):
                collect_bank(self.config,root,env_class=WrongImage,**BINDING)
            self.assertTrue((root/'failure.json').exists())
            self.assertFalse((root/'manifest.json').exists())

    def test_environment_binding_and_counter_types(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'bank';collect_bank(self.config,root,env_class=self.fake,**BINDING)
            with self.assertRaises(ValueError):
                load_bank(root,sha256(root/'manifest.json'),protocol_sha256(self.config),
                          **{**BINDING,'environment_lock_sha256':'c'*64})
            manifest=json.loads((root/'manifest.json').read_text());manifest['actions']=2480.0
            (root/'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                load_bank(root,sha256(root/'manifest.json'),protocol_sha256(self.config),**BINDING)


if __name__=='__main__':unittest.main()
