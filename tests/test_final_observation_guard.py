"""Synthetic mechanics/evidence tests; no GPU or deployment validation claimed."""
import copy
import hashlib
import inspect
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np

from helixwm_final import observation_guard as g


def fixture_records(seeds):
    rows=[]
    base=(64 + ((np.indices((64,64)).sum(0)%2)*64)).astype(np.uint8)
    base=np.repeat(base[:,:,None],3,axis=2)
    for seed in seeds:
        clean=np.stack([base + (t%4) for t in range(24)])
        visible=clean.copy()
        visible[4:8]=0
        visible[8:12]=clean[7]
        visible[12:16]=np.random.default_rng(seed).integers(0,256,(4,64,64,3),dtype=np.uint8)
        rows.append(dict(seed=seed,images=visible,prior_images=clean.astype(np.float32)/255,
            labels=['clean']*4+['black']*4+['freeze']*4+['noise']*4+['clean']*8))
    return rows


def binding(fit,heldout):
    return dict(source_sha256='1'*64,model_sha256='2'*64,model_version=7,protocol_sha256='3'*64,
        fit_data_sha256=g.records_sha256(fit),heldout_data_sha256=g.records_sha256(heldout))


class ObservationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fit=fixture_records(g.FIT_SEEDS);cls.heldout=fixture_records(g.HELDOUT_SEEDS)
        cls.binding=binding(cls.fit,cls.heldout)
        cls.calibration=g.fit_observation_calibration(cls.fit,cls.heldout,binding=cls.binding,
            measurement_kind='synthetic_test_fixture')
        cls.temp=Path(tempfile.mkdtemp(prefix='observation_guard_'))
        cls.bundle=cls.temp/'sealed'
        cls.receipt=g.seal_observation_calibration(cls.bundle,cls.fit,cls.heldout,binding=cls.binding,
            measurement_kind='synthetic_test_fixture')

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.temp)

    def guard(self,calibration=None,**kwargs):
        calibration=self.calibration if calibration is None else calibration
        return g.ObservationGuard(calibration,expected_binding=self.binding,
            expected_calibration_sha256=g.calibration_sha256(calibration),allow_synthetic_for_tests=True,**kwargs)

    def prior(self,frame,t):
        return dict(image_mean=frame.astype(np.float32)/255,target_timestamp=t,conditioning_timestamp=t-1,
            model_sha256=self.binding['model_sha256'],model_version=self.binding['model_version'])

    def copy_bundle(self,name):
        path=self.temp/name
        shutil.copytree(self.bundle,path)
        return path

    def test_runtime_has_no_privileged_input(self):
        names=set(inspect.signature(g.ObservationGuard.step).parameters)
        self.assertEqual(names,{'self','image','timestamp','prior_prediction'})
        with self.assertRaises(TypeError):
            self.guard().step(self.fit[0]['images'][0],timestamp=0,availability=True)

    def test_missing_calibration_blocks_without_mutation(self):
        guard=g.ObservationGuard(expected_binding=self.binding)
        before=guard.snapshot()['sha256']
        with self.assertRaises(RuntimeError):guard.step(self.fit[0]['images'][0],timestamp=0)
        self.assertEqual(before,guard.snapshot()['sha256'])

    def test_synthetic_does_not_activate_by_default(self):
        guard=g.ObservationGuard(self.calibration,expected_binding=self.binding,
            expected_calibration_sha256=g.calibration_sha256(self.calibration))
        with self.assertRaises(RuntimeError):guard.step(self.fit[0]['images'][0],timestamp=0)
        self.assertIn('NOT_VALIDATED',guard.calibration_status)

    def test_plain_measured_self_signed_dictionary_rejected(self):
        candidate=copy.deepcopy(self.calibration);candidate['measurement_kind']='measured_visible_model'
        with self.assertRaisesRegex(ValueError,'self-hashed'):
            g.ObservationGuard(candidate,expected_binding=self.binding,
                expected_calibration_sha256=g.calibration_sha256(candidate))
        with self.assertRaises(ValueError):g.VerifiedObservationCalibration(candidate)

    def test_quarantine_precedes_confirmation_and_posterior(self):
        guard=self.guard();row=self.heldout[0];outputs=[]
        for t in range(6):
            prediction=dict(image_mean=row['prior_images'][t],target_timestamp=t,conditioning_timestamp=t-1,
                model_sha256=self.binding['model_sha256'],model_version=7)
            outputs.append(guard.step(row['images'][t],timestamp=t,prior_prediction=prediction))
        self.assertEqual(outputs[4]['status'],'WARNING')
        self.assertFalse(outputs[4]['allow_posterior']);self.assertIsNone(outputs[4]['accepted_image'])
        self.assertIsNotNone(outputs[4]['quarantine_sha256'])
        self.assertEqual(outputs[5]['status'],'FAULT')
        self.assertFalse(outputs[5]['allow_posterior'])

    def test_static_clean_frame_is_not_freeze(self):
        guard=self.guard();frame=self.fit[0]['images'][0]
        for t in range(30):
            output=guard.step(frame,timestamp=t,prior_prediction=self.prior(frame,t))
            self.assertTrue(output['allow_posterior'])
            self.assertNotIn('freeze_with_innovation',output['reasons'])
        self.assertGreater(output['features']['repeat_run'],3)

    def test_recovery_requires_two_clear_frames(self):
        guard=self.guard();good=self.fit[0]['images'][0];bad=np.zeros_like(good)
        sequence=[good,bad,bad,good,good]
        outputs=[guard.step(frame,timestamp=t,prior_prediction=self.prior(good,t)) for t,frame in enumerate(sequence)]
        self.assertEqual(outputs[3]['status'],'RECOVERING');self.assertIsNone(outputs[3]['accepted_image'])
        self.assertEqual(outputs[4]['status'],'CLEAR');self.assertTrue(outputs[4]['recovered'])
        self.assertEqual(outputs[4]['counters']['recoveries'],1)

    def test_known_mask_passthrough_preserves_every_image(self):
        guard=g.ObservationGuard(expected_binding=self.binding,mode='known_mask_passthrough')
        for t,frame in enumerate(self.fit[0]['images']):
            result=guard.step(frame,timestamp=t)
            self.assertTrue(result['allow_posterior']);np.testing.assert_array_equal(result['accepted_image'],frame)
            self.assertEqual(result['status'],'KNOWN_MASK_PASSTHROUGH')
            result['accepted_image'][:]=255
            np.testing.assert_array_equal(guard.snapshot()['state']['previous'],frame)

    def test_prediction_order_and_model_binding_atomic(self):
        guard=self.guard();frame=self.fit[0]['images'][0];before=guard.snapshot()['sha256']
        for field,value in [('conditioning_timestamp',0),('target_timestamp',1),('model_sha256','f'*64),('model_version',8)]:
            p=self.prior(frame,0);p[field]=value
            with self.assertRaises(ValueError):guard.step(frame,timestamp=0,prior_prediction=p)
            self.assertEqual(before,guard.snapshot()['sha256'])
        with self.assertRaises(ValueError):guard.step(frame,timestamp=0)

    def test_nonfinite_prediction_and_wrong_image_rejected(self):
        guard=self.guard();frame=self.fit[0]['images'][0]
        p=self.prior(frame,0);p['image_mean'][0,0,0]=np.nan
        with self.assertRaises(ValueError):guard.step(frame,timestamp=0,prior_prediction=p)
        with self.assertRaises(ValueError):guard.step(frame.astype(np.float32),timestamp=0)
        with self.assertRaises(ValueError):guard.step(frame,timestamp=True)

    def test_timestamp_contiguity(self):
        guard=self.guard();frame=self.fit[0]['images'][0]
        guard.step(frame,timestamp=0,prior_prediction=self.prior(frame,0))
        for t in (0,2):
            with self.assertRaises(ValueError):guard.step(frame,timestamp=t,prior_prediction=self.prior(frame,t))

    def test_snapshot_restores_next_decision_exactly(self):
        guard=self.guard();good=self.fit[0]['images'][0];bad=np.zeros_like(good)
        guard.step(good,timestamp=0,prior_prediction=self.prior(good,0))
        guard.step(bad,timestamp=1,prior_prediction=self.prior(good,1))
        snapshot=guard.snapshot();restored=self.guard()
        restored.load(snapshot,expected_snapshot_sha256=snapshot['sha256'])
        for t,frame in enumerate([bad,good,good,good],start=2):
            a=guard.step(frame,timestamp=t,prior_prediction=self.prior(good,t))
            b=restored.step(frame,timestamp=t,prior_prediction=self.prior(good,t))
            self.assertEqual(g._digest(a),g._digest(b))
        self.assertEqual(guard.snapshot()['sha256'],restored.snapshot()['sha256'])

    def test_snapshot_corruption_and_state_contradictions_are_atomic(self):
        guard=self.guard();frame=self.fit[0]['images'][0]
        guard.step(frame,timestamp=0,prior_prediction=self.prior(frame,0));before=guard.snapshot()
        bad=copy.deepcopy(before);bad['state']['previous'][0,0,0]^=1
        with self.assertRaises(ValueError):guard.load(bad,expected_snapshot_sha256=before['sha256'])
        bad=copy.deepcopy(before);bad['state']['repeat_run']=999;bad['sha256']=g._digest(bad['state'])
        with self.assertRaises(ValueError):guard.load(bad,expected_snapshot_sha256=bad['sha256'])
        self.assertEqual(before['sha256'],guard.snapshot()['sha256'])

    def test_snapshot_binding_rejects_different_model(self):
        guard=self.guard();snapshot=guard.snapshot();bad=copy.deepcopy(snapshot)
        bad['state']['binding']['model_version']=8;bad['sha256']=g._digest(bad['state'])
        with self.assertRaises(ValueError):guard.load(bad,expected_snapshot_sha256=bad['sha256'])

    def test_detector_uses_no_global_rng(self):
        before=g._digest(np.random.get_state());guard=self.guard();frame=self.fit[0]['images'][0]
        guard.step(frame,timestamp=0,prior_prediction=self.prior(frame,0))
        self.assertEqual(before,g._digest(np.random.get_state()))

    def test_state_memory_is_bounded(self):
        guard=self.guard();frame=self.fit[0]['images'][0]
        for t in range(101):guard.step(frame,timestamp=t,prior_prediction=self.prior(frame,t))
        arrays=[v for v in guard.snapshot()['state'].values() if isinstance(v,np.ndarray)]
        self.assertLessEqual(sum(x.nbytes for x in arrays),2*64*64*3)

    def test_fit_does_not_tune_on_heldout(self):
        changed=copy.deepcopy(self.heldout)
        for row in changed:row['images'][12:16]=255
        b=binding(self.fit,changed)
        other=g.fit_observation_calibration(self.fit,changed,binding=b,measurement_kind='synthetic_test_fixture')
        self.assertEqual(self.calibration['thresholds'],other['thresholds'])
        self.assertEqual(self.calibration['fit_statistics'],other['fit_statistics'])
        self.assertFalse(other['heldout_report']['thresholds_changed_using_heldout'])

    def test_partition_leakage_wrong_digest_and_missing_fault_coverage(self):
        changed=copy.deepcopy(self.fit);changed[0]['seed']=60000
        with self.assertRaises(ValueError):g.fit_observation_calibration(changed,self.heldout,binding=self.binding,measurement_kind='synthetic_test_fixture')
        b=copy.deepcopy(self.binding);b['fit_data_sha256']='f'*64
        with self.assertRaises(ValueError):g.fit_observation_calibration(self.fit,self.heldout,binding=b,measurement_kind='synthetic_test_fixture')
        changed=copy.deepcopy(self.heldout);changed[0]['labels']=['clean']*24
        with self.assertRaises(ValueError):g.fit_observation_calibration(self.fit,changed,binding=binding(self.fit,changed),measurement_kind='synthetic_test_fixture')

    def test_heldout_report_counts_false_positives_negatives_latency(self):
        report=self.calibration['heldout_report'];counts=report['counts']
        self.assertEqual(counts['clean_frames']+counts['fault_frames'],240)
        self.assertEqual(set(report['by_fault_type']),{'black','freeze','noise'})
        for episode in report['episodes']:
            segment=episode['segments'][0]
            self.assertEqual(segment['quarantine_delay_frames'],0)
            self.assertEqual(segment['confirmation_delay_frames'],1)
            self.assertEqual(segment['recovery_delay_frames'],1)
            self.assertFalse(segment['recovery_right_censored'])
        self.assertEqual(report['false_negative_acceptance_rate'],0.)
        self.assertGreater(report['false_positive_rejection_rate'],0.)  # Deliberate recovery quarantine.

    def test_file_backed_synthetic_bundle_requires_explicit_test_override(self):
        with self.assertRaises(ValueError):g.load_observation_calibration(self.bundle,
            expected_manifest_sha256=self.receipt['manifest_sha256'],expected_binding=self.binding)
        loaded=g.load_observation_calibration(self.bundle,expected_manifest_sha256=self.receipt['manifest_sha256'],
            expected_binding=self.binding,allow_synthetic_for_tests=True)
        self.assertEqual(dict(loaded),self.calibration)
        guard=self.guard(loaded)
        self.assertEqual(guard.calibration_status,'SYNTHETIC_TEST_ONLY_NOT_VALIDATED')

    def test_external_manifest_binding_and_member_corruption(self):
        with self.assertRaises(ValueError):g.load_observation_calibration(self.bundle,
            expected_manifest_sha256='a'*64,expected_binding=self.binding,allow_synthetic_for_tests=True)
        path=self.copy_bundle('corrupt_member');member=path/'fit_55000.npz'
        raw=bytearray(member.read_bytes());raw[-1]^=1;member.write_bytes(raw)
        with self.assertRaisesRegex(ValueError,'SHA/length'):g.load_observation_calibration(path,
            expected_manifest_sha256=self.receipt['manifest_sha256'],expected_binding=self.binding,allow_synthetic_for_tests=True)

    def test_manifest_extra_file_and_no_overwrite(self):
        path=self.copy_bundle('extra_member');(path/'extra.txt').write_text('not registered')
        with self.assertRaises(ValueError):g.load_observation_calibration(path,
            expected_manifest_sha256=self.receipt['manifest_sha256'],expected_binding=self.binding,allow_synthetic_for_tests=True)
        with self.assertRaises(FileExistsError):g.seal_observation_calibration(self.bundle,self.fit,self.heldout,
            binding=self.binding,measurement_kind='synthetic_test_fixture')

    def test_self_signed_report_cannot_replace_recomputed_evidence(self):
        path=self.copy_bundle('false_report');cp=path/'calibration.json';record=json.loads(cp.read_bytes())
        record['heldout_report']['false_negative_acceptance_rate']=.9
        raw=g._canonical(record);cp.write_bytes(raw)
        manifest=json.loads((path/'manifest.json').read_bytes())
        manifest['calibration_sha256']=g.calibration_sha256(record)
        for e in manifest['entries']:
            if e['path']=='calibration.json':e.update(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
        raw=g._canonical(manifest);(path/'manifest.json').write_bytes(raw)
        with self.assertRaisesRegex(ValueError,'reproduce'):g.load_observation_calibration(path,
            expected_manifest_sha256=hashlib.sha256(raw).hexdigest(),expected_binding=self.binding,allow_synthetic_for_tests=True)

    def test_verified_handle_mutation_rejected(self):
        loaded=g.load_observation_calibration(self.bundle,expected_manifest_sha256=self.receipt['manifest_sha256'],
            expected_binding=self.binding,allow_synthetic_for_tests=True)
        loaded['thresholds']['innovation_mse_min']=.9
        with self.assertRaisesRegex(ValueError,'modified'):self.guard(loaded)


if __name__=='__main__':unittest.main()
