"""CPU contract fixtures only. No successful real recovery artifact is generated."""
import argparse
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from helixwm_final import final_checkpoint as c
from helixwm_final import final_recovery_probe as r


def synthetic_result(scope=c.FORMAT):
    counts={'agent_steps':10000,'env_steps':20000,'episodes':20,'update_steps':1250}
    if scope==c.PREFLIGHT_FORMAT:counts={'agent_steps':6000,'env_steps':12000,'episodes':12,'update_steps':250}
    identity={'run_id':'preflight_fixture' if scope==c.PREFLIGHT_FORMAT else 'formal_fixture','arm':'F',
        'source_sha256':'a'*64,'protocol_sha256':'b'*64,'environment_lock_sha256':'c'*64,
        'bank_manifest_sha256':'d'*64,'upstream_commit':c.UPSTREAM}
    compared={key:'CPU_FIXTURE_NOT_PRODUCTION_EVIDENCE' for key in r.COMPARISON_FIELDS}
    compared.update(identity=identity,checkpoint_format=scope,source_counts=counts,
        isolated_counts=dict(counts,agent_steps=counts['agent_steps']+4,env_steps=counts['env_steps']+8,
                             update_steps=counts['update_steps']+1),registration=copy.deepcopy(r.REGISTRATION),
        steps=[{'action':[0.]*6,'reward':0.,'info':{'observation_timestamp':i,'availability':i!=3,'env_steps_this_step':2}}
               for i in range(1,5)],gpu_devices=[{'platform':'gpu','kind':'CPU_FIXTURE_ONLY','id':0}])
    return {'schema':'helixwm_final_bank_recovery_worker_v3','passed':True,'all_devices_gpu':True,
        'original_files_unchanged':True,'actual_counts_complete':True,'formal_training_actions':0,
        'probe_actions':4,'probe_dmc_calls':8,'probe_updates':1,'auxiliary_update_attempts':4,'auxiliary_eligible_updates':2,
        'restored_comparisons':{k:True for k in r.RESTORED_FIELDS},'comparison':compared}


class FormalRecoveryContracts(unittest.TestCase):
    def test_synthetic_comparator_distinguishes_formal_and_preflight_scope(self):
        for scope in (c.FORMAT,c.PREFLIGHT_FORMAT):
            result=synthetic_result(scope);report=r.compare_results([result,copy.deepcopy(result)],[0,0])
            self.assertTrue(report['passed'])
            self.assertEqual(report['formal_checkpoint_recovery_proven'],scope==c.FORMAT)
            self.assertEqual(report['integrated_preflight_checkpoint_recovery_proven'],scope==c.PREFLIGHT_FORMAT)
            self.assertFalse(report['uninterrupted_process_reference'])
            self.assertFalse(report['automatic_formal_resume'])
            self.assertEqual(report['formal_training_actions'],0)

    def test_bootstrap_result_or_two_same_wrong_scopes_never_accepted(self):
        for scope in ('helixwm_final_bootstrap_probe_v1','helixwm_final_full_episode_boundary_v1'):
            result=synthetic_result(scope)
            self.assertFalse(r.compare_results([result,copy.deepcopy(result)],[0,0])['passed'])

    def test_actual_counts_and_registered_mask_cannot_be_faked_by_matching_hashes(self):
        for field in ('actions','updates','mask','step_count','gpu'):
            result=synthetic_result()
            if field=='actions':result['comparison']['isolated_counts']['agent_steps']+=1
            if field=='updates':result['probe_updates']=0
            if field=='mask':result['comparison']['steps'][2]['info']['availability']=True
            if field=='step_count':result['comparison']['steps'].pop()
            if field=='gpu':result['comparison']['gpu_devices'][0]['platform']='cpu'
            self.assertFalse(r.compare_results([result,copy.deepcopy(result)],[0,0])['passed'])

    def test_each_state_rng_sampler_update_and_action_difference_fails_exactly(self):
        for field in ('restored_private_rng','sample','sampler_rng_after_sample','after_backbone','after_auxiliary','after_collector'):
            left=synthetic_result();right=copy.deepcopy(left);right['comparison'][field]='DIFFERENT'
            report=r.compare_results([left,right],[0,0]);self.assertFalse(report['passed']);self.assertIn(field,report['mismatched_fields'])

    def test_missing_fields_failed_process_or_nonfinite_evidence_fail_closed(self):
        original=synthetic_result()
        for case in ('schema','exit','nonfinite','incomplete'):
            left=copy.deepcopy(original);right=copy.deepcopy(original);codes=[0,0]
            if case=='schema':right['comparison'].pop('after_host_rng')
            if case=='exit':codes[1]=1
            if case=='nonfinite':right['comparison']['metrics']={'loss':float('nan')}
            if case=='incomplete':right['actual_counts_complete']=False
            self.assertFalse(r.compare_results([left,right],codes)['passed'])
        with self.assertRaises(ValueError):r.compare_results([original],[0])

    def test_tool_digest_and_identity_errors_precede_loader_or_any_gpu_call(self):
        args=argparse.Namespace(tool_bundle_sha256='a'*64)
        with patch.object(c,'load_checkpoint') as loader:
            with self.assertRaises(ValueError):r.load_inputs(args)
            loader.assert_not_called()

    def test_new_output_cannot_overwrite_or_overlap_input(self):
        root=Path(__file__).resolve().parents[2]/'preflight'/'formal_recovery_cpu';root.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as directory:
            base=Path(directory);source=base/'source';source.mkdir()
            for output in (source,source/'nested',base):
                with self.assertRaises((ValueError,FileExistsError)):r.isolated_output(source,output)
            self.assertEqual(r.isolated_output(source,base/'new'),base/'new')

    def test_preflight_opt_in_does_not_relax_normal_or_final_budget(self):
        pre={'agent_steps':6000,'env_steps':12000,'episodes':12,'update_steps':250}
        with self.assertRaises(ValueError):c.validate_counts(pre)
        self.assertEqual(c.validate_counts(pre,preflight=True),pre)
        with self.assertRaises(ValueError):c.validate_counts(pre,preflight=True,require_final=True)
        for bad in ({**pre,'agent_steps':6500},{**pre,'episodes':11},{**pre,'update_steps':125}):
            with self.assertRaises(ValueError):c.validate_counts(bad,preflight=True)


if __name__=='__main__':unittest.main()
