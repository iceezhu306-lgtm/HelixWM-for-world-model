"""CPU boundary/evidence tests only; no fabricated successful GPU recovery."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import pickle
import random
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from helixwm_final import recovery_probe as r
from helixwm_final.epistemic import BootstrapDynamics


def identity(source='a'*64):
    return dict(source_sha256=source,protocol_sha256='b'*64,environment_lock_sha256='e'*64,
        stage='isolated_shared_native_bootstrap_preflight',fresh_initialization=True,
        formal_training_actions=0,old_checkpoint_loaded=False)


def fixture():
    ident=identity();aux=BootstrapDynamics(14,{k:ident[k] for k in ('source_sha256','protocol_sha256')})
    frame=np.zeros((64,64,3),np.uint8)
    for i in range(1,41):aux.update(frame,np.zeros(6),frame,0.,1.,transition_id=i,previous_available=True,next_available=True)
    for i in range(41,5501):aux.update(None,None,None,None,None,transition_id=i,previous_available=False,next_available=False)
    heads=aux.pin();rng=np.random.default_rng(34).bit_generator.state
    descriptor={'path':'c'*64+'.npz','bytes':4,'file_sha256':'d'*64}
    replay=dict(format='helixwm_native_replay_exact_v1',rows_added=5511,length=65,
        online=False,capacity=1000000,chunksize=1024,
        chunks=[{'length':n,'arrays':{'fixture':descriptor}} for n in [1024]*5+[391]])
    core=dict(format='helixwm_final_backbone_v1',training={'counters':{'updates':125,'batches':125}},
        writeback_updates=125,inference_version=125,final_inference_bundle={'model_sha256':'e'*64})
    completed=dict(controller='black_frame_native',chain={'state':dict(episode_id=10,timestamp=500,
        anchor_time=500,last_action=None,pending=None,missing_length=0,abstaining=False),
        'rngs':{k:copy.deepcopy(rng) for k in ('real','actor','dream','repair')}})
    payload=dict(format=r.FORMAT,identity=ident,counts=copy.deepcopy(r.EXPECTED_COUNTS),backbone=core,
        auxiliary=aux.snapshot(),heads=heads.snapshot(),replay=replay,completed_native_controller=completed,
        prefill_rng=copy.deepcopy(rng),numpy_rng=np.random.RandomState(2).get_state(),python_rng=random.Random(2).getstate())
    sidecar=dict(counts=copy.deepcopy(payload['counts']),model_sha256='e'*64,
        epistemic_heads_sha256=heads.sha256,replay_dependencies=r._inventory(replay))
    return payload,sidecar,ident


class RecoveryEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.payload,cls.sidecar,cls.identity=fixture()

    def setUp(self):
        self.temp=Path(tempfile.mkdtemp(prefix='final_recovery_'))

    def tearDown(self):shutil.rmtree(self.temp)

    def test_exact_bootstrap_boundary_and_rng_validation_are_side_effect_free(self):
        before=r.tree_hash((np.random.get_state(),random.getstate()))
        counts=r.validate_payload(self.payload,self.sidecar,self.identity)
        self.assertEqual(counts,r.EXPECTED_COUNTS)
        self.assertEqual(before,r.tree_hash((np.random.get_state(),random.getstate())))

    def test_old_formal_checkpoint_never_accepted(self):
        for format_name in ('helixwm_matched_episode_boundary_v1','helixwm_final_formal_v1'):
            p=copy.deepcopy(self.payload);p['format']=format_name
            with self.assertRaises(ValueError):r.validate_payload(p,self.sidecar,self.identity)

    def test_wrong_budget_replay_or_unresolved_action_rejected(self):
        for field in ('agent_steps','env_steps','episodes','update_steps'):
            p=copy.deepcopy(self.payload);p['counts'][field]+=1
            with self.assertRaises(ValueError):r.validate_payload(p,self.sidecar,self.identity)
        p=copy.deepcopy(self.payload);p['replay']['rows_added']-=1
        with self.assertRaises(ValueError):r.validate_payload(p,self.sidecar,self.identity)
        p=copy.deepcopy(self.payload);p['completed_native_controller']['chain']['state']['last_action']=np.zeros(6)
        with self.assertRaises(ValueError):r.validate_payload(p,self.sidecar,self.identity)

    def test_missing_auxiliary_update_or_foreign_pin_rejected(self):
        p=copy.deepcopy(self.payload);p['heads']['accepted_pairs']+=1
        with self.assertRaises(ValueError):r.validate_payload(p,self.sidecar,self.identity)
        s=copy.deepcopy(self.sidecar);s['epistemic_heads_sha256']='f'*64
        with self.assertRaises(ValueError):r.validate_payload(self.payload,s,self.identity)

    def test_missing_replay_dependencies_and_corrupt_rng_rejected(self):
        s=copy.deepcopy(self.sidecar);s['replay_dependencies']=[]
        with self.assertRaises(ValueError):r.validate_payload(self.payload,s,self.identity)
        p=copy.deepcopy(self.payload);p['prefill_rng']={'bad':'rng'}
        with self.assertRaises(ValueError):r.validate_payload(p,self.sidecar,self.identity)

    def test_tree_hash_distinguishes_nested_structure_numpy_scalar_and_lists(self):
        self.assertNotEqual(r.tree_hash({'a':{'b':1},'c':2}),r.tree_hash({'a':{},'b':1,'c':2}))
        self.assertNotEqual(r.tree_hash([1,2]),r.tree_hash((1,2)))
        self.assertNotEqual(r.tree_hash(np.float32(1)),r.tree_hash(1.))
        with self.assertRaises(ValueError):r.tree_hash(np.asarray([np.nan]))

    def test_identity_rejects_changed_source_protocol_or_claimed_formal_actions(self):
        r.verify_identity(self.identity,runtime_source_sha256='a'*64,protocol_sha256='b'*64,environment_lock_sha256='e'*64)
        for key,value in [('source_sha256','c'*64),('environment_lock_sha256','c'*64),('formal_training_actions',1),('old_checkpoint_loaded',True)]:
            x=copy.deepcopy(self.identity);x[key]=value
            with self.assertRaises(ValueError):r.verify_identity(x,runtime_source_sha256='a'*64,protocol_sha256='b'*64,environment_lock_sha256='e'*64)

    def test_output_cannot_overlap_original_or_overwrite_evidence(self):
        source=self.temp/'source';source.mkdir();checkpoint=source/'bootstrap_000005500.pkl'
        for output in (source,source/'nested',self.temp):
            with self.assertRaises((ValueError,FileExistsError)):r.validate_output_path(checkpoint,output)
        self.assertEqual(r.validate_output_path(checkpoint,self.temp/'new'),self.temp/'new')

    def test_replay_copy_verified_and_cannot_mutate_source(self):
        src=self.temp/'src';src.mkdir();name='c'*64+'.npz';(src/name).write_bytes(b'data')
        records=[{'path':name,'bytes':4,'file_sha256':hashlib.sha256(b'data').hexdigest()}]
        result=r.materialize_replay(src,self.temp/'dst',records)
        self.assertEqual(result['hardlinks'],0)
        (self.temp/'dst'/name).write_bytes(b'else')
        self.assertEqual((src/name).read_bytes(),b'data')
        with self.assertRaises(FileExistsError):r.materialize_replay(src,self.temp/'dst',records)
        with self.assertRaises(ValueError):r.materialize_replay(src,self.temp/'unsafe',[dict(records[0],path='../x.npz')])
        with self.assertRaises(ValueError):r.materialize_replay(src,self.temp/'duplicate',records*2)

    def test_replay_corruption_stops_before_output_creation(self):
        src=self.temp/'src';src.mkdir();name='c'*64+'.npz';(src/name).write_bytes(b'data')
        with self.assertRaises(ValueError):r.materialize_replay(src,self.temp/'dst',
            [{'path':name,'bytes':4,'file_sha256':'f'*64}])
        self.assertFalse((self.temp/'dst').exists())

    def manifest_fixture(self):
        runtime=self.temp/'runtime';runtime.mkdir()
        names=['helixwm_final/backbone.py','helixwm_final/epistemic.py',
               'helixwm_final/protocol_v2.py','helixwm_matched/backbone.py']
        for name in names:
            path=runtime/name;path.parent.mkdir(exist_ok=True);path.write_bytes(b'# preserved original bytes\n')
        (runtime/'helixwm_v2').mkdir()
        manifest={n:r.sha_file(runtime/n) for n in names}
        return runtime,manifest,hashlib.sha256(r._json(manifest)).hexdigest()

    def test_original_source_manifest_does_not_include_independent_tool(self):
        runtime,manifest,digest=self.manifest_fixture()
        tool=self.temp/'independent_recovery.py';tool.write_text('# outside original runtime')
        self.assertEqual(r.verify_runtime_manifest(runtime,manifest,digest),digest)
        (runtime/'helixwm_final'/'extra.py').write_text('# forbidden source mutation')
        with self.assertRaises(ValueError):r.verify_runtime_manifest(runtime,manifest,digest)

    def test_runtime_manifest_tampering_and_path_escape_rejected(self):
        runtime,manifest,digest=self.manifest_fixture()
        (runtime/'helixwm_final/backbone.py').write_text('tampered')
        with self.assertRaises(ValueError):r.verify_runtime_manifest(runtime,manifest,digest)
        bad={'../outside.py':'a'*64};badsha=hashlib.sha256(r._json(bad)).hexdigest()
        with self.assertRaises(ValueError):r.verify_runtime_manifest(runtime,bad,badsha)

    def test_failed_identity_or_checkpoint_hash_never_unpickles(self):
        runtime,manifest,source=self.manifest_fixture()
        run=self.temp/'run';run.mkdir();ident=identity(source)
        (run/'identity.json').write_bytes(r._json(ident));(run/'source_manifest.json').write_bytes(r._json(manifest))
        checkpoint=run/'bootstrap_000005500.pkl';checkpoint.write_bytes(b'not even a pickle')
        lock=run/'lock.json';lock.write_text('{}')
        ident['environment_lock_sha256']=r.sha_file(lock);(run/'identity.json').write_bytes(r._json(ident))
        args=argparse.Namespace(source_identity=run/'identity.json',source_identity_sha256=r.sha_file(run/'identity.json'),
            runtime_source_root=runtime,tool_sha256=r.sha_file(Path(r.__file__)),protocol=self.temp/'not-used.json',
            checkpoint=checkpoint,checkpoint_sha256='f'*64,environment_lock=lock,environment_lock_sha256=r.sha_file(lock))
        with patch('helixwm_final.protocol_v2.load_protocol',return_value={}),patch('helixwm_final.protocol_v2.protocol_sha256',return_value='b'*64),patch.object(r.pickle,'loads') as unpickle:
            with self.assertRaises(ValueError):r.verify_and_load(args)
            unpickle.assert_not_called()
        args.source_identity_sha256='c'*64
        with patch.object(r.pickle,'loads') as unpickle:
            with self.assertRaises(ValueError):r.verify_and_load(args)
            unpickle.assert_not_called()

    def test_sample_descriptor_requires_actual_native_dimensions(self):
        batch={'image':np.zeros((16,65,2),np.uint8),'consec':np.zeros((16,65),np.int32)}
        report=r.sample_descriptor(batch);self.assertEqual(report['learning_positions'],1024)
        bad=dict(batch,consec=np.ones((16,65),np.int32))
        with self.assertRaises(ValueError):r.sample_descriptor(bad)
        with self.assertRaises(ValueError):r.sample_descriptor({'consec':np.zeros((16,64),np.int32)})

    def test_failed_or_incomplete_workers_never_pass_recovery(self):
        result=r.compare_results([{'passed':False},{'passed':False}],[1,None])
        self.assertFalse(result['passed']);self.assertFalse(result['formal_training_resumed'])
        self.assertFalse(result['uninterrupted_live_process_comparison_performed'])
        self.assertFalse(result['cross_hardware_or_long_horizon_identity_proven'])
        fake={'passed':True,'all_devices_gpu':True,'original_files_unchanged':True,'actual_counts_complete':True,
            'probe_actions':4,'probe_dmc_calls':8,'probe_updates':1,'auxiliary_update_attempts':4,
            'auxiliary_eligible_updates':4,'restored_comparisons':{k:True for k in r.RESTORED_FIELDS},'comparison':{}}
        self.assertFalse(r.compare_results([fake,fake],[0,0])['passed'])


if __name__=='__main__':unittest.main()
