"""Visible-only observation screening, with fit-only thresholds and quarantine.

This is a bounded engineering detector, not semantic corruption recognition.
Dark/flat images, excessive local variation, exact repeats AND inconsistent
prior predictions, and model innovation are only anomaly evidence. Real dark
or highly textured scenes and model errors can produce false alarms. Fault
labels are permitted only in independent offline calibration/evaluation data.
They never enter ObservationGuard.step. Known-mask experiments use the explicit
passthrough mode; this module cannot silently alter their model inputs.

Features are image mean/std, mean adjacent-pixel absolute difference, temporal
L1, exact-repeat run, and MSE to a MODEL-BOUND PRE-OBSERVATION prediction.
The latter must be generated before seeing the current image. Metadata checks
cannot prove that ordering by themselves: the integrating caller must audit it.
Thresholds use only clean fit frames: lower Q.01 and upper Q.99, method=higher.
Two anomalous frames confirm; two clear frames recover. These integer durations
are fixed engineering rules, not heldout-tuned thresholds or safety guarantees.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np

SCHEMA = 'helixwm_visible_observation_guard_v1'
FIT_SEEDS = tuple(range(55000, 55010))
HELDOUT_SEEDS = tuple(range(55010, 55020))
BINDING_FIELDS = {'source_sha256', 'model_sha256', 'model_version', 'protocol_sha256',
                  'fit_data_sha256', 'heldout_data_sha256'}
STATS = ('mean', 'std', 'high_frequency', 'temporal_l1', 'innovation_mse', 'repeat_run')
RULES = {'lower_quantile': .01, 'upper_quantile': .99, 'quantile_method': 'higher',
         'confirm_frames': 2, 'recovery_frames': 2, 'minimum_repeat_run': 3}
THRESHOLD_KEYS = {'dark_mean_max', 'flat_std_max', 'noise_high_frequency_min',
                  'innovation_mse_min', 'freeze_repeat_min'}
LABELS = ('clean', 'black', 'freeze', 'noise')
BUNDLE_SCHEMA = 'helixwm_observation_guard_bundle_v1'
_FILE_VERIFIED_TOKEN = object()


class VerifiedObservationCalibration(dict):
    """File-verified evidence handle, not serializable activation authority.

    The external manifest digest must come from the trusted collection job.
    Hashes authenticate bytes, not the physical origin of those bytes. Neither
    this loader nor a ``measurement_kind`` string proves that a GPU produced
    predictions; collection must separately audit prediction-before-observation.
    """
    def __init__(self, value, *, _token=None):
        if _token is not _FILE_VERIFIED_TOKEN:
            raise ValueError('Only the file-backed loader can issue observation calibration')
        super().__init__(value)
        self._issued_digest = _digest(value)

    def __deepcopy__(self, memo):
        if _digest(self) != self._issued_digest:
            raise ValueError('Observation calibration handle was modified')
        return VerifiedObservationCalibration(copy.deepcopy(dict(self), memo),
                                               _token=_FILE_VERIFIED_TOKEN)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def _digest(value):
    """Deterministic typed array content hash, not a ZIP timestamp hash."""
    h = hashlib.sha256()
    def visit(x):
        if isinstance(x, np.ndarray):
            if x.dtype.hasobject:
                raise ValueError('Object arrays cannot be trusted evidence')
            h.update(b'A'+_canonical({'dtype': x.dtype.str, 'shape': list(x.shape)}))
            h.update(np.ascontiguousarray(x).tobytes())
        elif isinstance(x, dict):
            h.update(b'D')
            for key in sorted(x):
                h.update(_canonical(key));visit(x[key])
        elif isinstance(x, (list, tuple)):
            h.update(b'L'+str(len(x)).encode())
            for item in x:visit(item)
        else:
            h.update(b'J'+_canonical(x))
    visit(value)
    return h.hexdigest()


def _is_sha(x):
    return type(x) is str and len(x) == 64 and all(c in '0123456789abcdef' for c in x)


def _binding(value):
    if type(value) is not dict or set(value) != BINDING_FIELDS:
        raise ValueError('Exact source/model/protocol/fit/heldout binding required')
    for key in BINDING_FIELDS-{'model_version'}:
        if not _is_sha(value[key]):raise ValueError('Invalid external digest: '+key)
    if type(value['model_version']) is not int or value['model_version'] < 0:
        raise ValueError('Model version must be nonnegative integer')
    return copy.deepcopy(value)


def _image(image):
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.shape != (64,64,3):
        raise ValueError('Visible image must be uint8[64,64,3]')
    return image.copy()


def _prediction(prediction, timestamp, binding):
    if prediction is None:return None
    fields = {'image_mean', 'target_timestamp', 'conditioning_timestamp', 'model_sha256', 'model_version'}
    if type(prediction) is not dict or set(prediction) != fields:
        raise ValueError('Only a bound pre-observation prediction is accepted')
    if (type(prediction['target_timestamp']) is not int or prediction['target_timestamp'] != timestamp
            or type(prediction['conditioning_timestamp']) is not int
            or not -1 <= prediction['conditioning_timestamp'] < timestamp):
        raise ValueError('Prediction must precede the current observation timestamp')
    for key in ('model_sha256', 'model_version'):
        if type(prediction[key]) is not type(binding[key]) or prediction[key] != binding[key]:
            raise ValueError('Prediction is bound to a different fixed model')
    image = prediction['image_mean']
    if (not isinstance(image, np.ndarray) or image.shape != (64,64,3)
            or image.dtype not in (np.dtype('float32'), np.dtype('float64'))
            or not np.isfinite(image).all() or np.any(image < 0) or np.any(image > 1)):
        raise ValueError('Prior image_mean must be finite float[64,64,3] in [0,1]')
    return image.astype(np.float64, copy=True)


def visible_features(image, previous_image=None, prior_image=None, repeat_run=0):
    """Pure image-derived features. No availability, physics or hidden image API."""
    image = _image(image)
    x = image.astype(np.float64)/255.0
    if previous_image is not None:
        previous_image = _image(previous_image)
    repeated = previous_image is not None and np.array_equal(image, previous_image)
    if type(repeat_run) is not int or repeat_run < 0:
        raise ValueError('Repeat run must be a nonnegative integer')
    run = repeat_run+1 if repeated else 0
    high_frequency = float((np.abs(np.diff(x, axis=0)).mean()+np.abs(np.diff(x, axis=1)).mean())/2)
    innovation = None
    if prior_image is not None:
        p = np.asarray(prior_image)
        if p.shape != x.shape or p.dtype.kind != 'f' or not np.isfinite(p).all() or np.any(p < 0) or np.any(p > 1):
            raise ValueError('Invalid prior prediction feature input')
        innovation = float(np.square(x-p.astype(np.float64)).mean())
    return {'mean': float(x.mean()), 'std': float(x.std()), 'high_frequency': high_frequency,
            'temporal_l1': None if previous_image is None else float(np.abs(x-previous_image.astype(np.float64)/255).mean()),
            'innovation_mse': innovation, 'repeat_run': run}


def records_sha256(records):
    return _digest(records)


def calibration_sha256(record):
    return _digest(record)


def _validate_records(records, seeds):
    if not isinstance(records, (list, tuple)) or len(records) != len(seeds):
        raise ValueError('One independent trajectory for each registered partition seed is required')
    seen=set()
    for row in records:
        if type(row) is not dict or set(row) != {'seed','images','prior_images','labels'}:
            raise ValueError('Unexpected calibration fields; no privileged runtime state allowed')
        seed=row['seed']
        if type(seed) is not int or seed not in seeds or seed in seen:
            raise ValueError('Repeated/wrong partition seed; final-test seeds are forbidden')
        seen.add(seed)
        images=row['images'];predictions=row['prior_images'];labels=row['labels']
        if (not isinstance(images,np.ndarray) or images.dtype!=np.uint8 or images.ndim!=4
                or images.shape[1:]!=(64,64,3) or len(images)<4):
            raise ValueError('Calibration requires actual visible uint8[T,64,64,3] images')
        if (not isinstance(predictions,np.ndarray) or predictions.dtype not in (np.dtype('float32'),np.dtype('float64'))
                or predictions.shape!=images.shape or not np.isfinite(predictions).all()
                or np.any(predictions<0) or np.any(predictions>1)):
            raise ValueError('Measured pre-observation prior images are required for every calibration frame')
        if type(labels) is not list or len(labels)!=len(images) or any(type(x) is not str or x not in LABELS for x in labels):
            raise ValueError('Offline scoring labels must match the independent injected-fault trace')
        if not all(name in labels for name in ('clean',)):
            raise ValueError('A trajectory needs clean reference frames')
        if tuple(seeds)==HELDOUT_SEEDS and (set(labels)!=set(LABELS) or labels[0]!='clean' or labels[-1]!='clean'):
            raise ValueError('Heldout traces need each registered fault type with clean boundary frames')


def _thresholds(value):
    if type(value) is not dict or set(value)!=THRESHOLD_KEYS:
        raise ValueError('Threshold schema mismatch')
    for key in THRESHOLD_KEYS-{'freeze_repeat_min'}:
        if type(value[key]) is not float or not math.isfinite(value[key]) or not 0 <= value[key] <= 1:
            raise ValueError('Threshold must be a finite measured scalar')
    if type(value['freeze_repeat_min']) is not int or value['freeze_repeat_min']<3:
        raise ValueError('Freeze confirmation needs at least three repeated transitions')
    return copy.deepcopy(value)


def _reasons(features, thresholds):
    reasons=[]
    if features['std'] < thresholds['flat_std_max']:
        reasons.append('flat')
        if features['mean'] < thresholds['dark_mean_max']:reasons.append('dark_flat')
    if features['high_frequency'] > thresholds['noise_high_frequency_min']:reasons.append('high_frequency')
    high_innovation=(features['innovation_mse'] is not None
                     and features['innovation_mse'] > thresholds['innovation_mse_min'])
    if high_innovation:reasons.append('model_innovation')
    # Repetition ALONE, even a long genuinely static scene, is not evidence of freeze.
    if features['repeat_run']>=thresholds['freeze_repeat_min'] and high_innovation:
        reasons.append('freeze_with_innovation')
    return reasons


class ObservationGuard:
    """Frozen detector with a one-frame copy quarantine and explicit passthrough."""
    def __init__(self, calibration=None, *, expected_binding,
                 expected_calibration_sha256=None, mode='unknown_observation',
                 allow_synthetic_for_tests=False):
        if mode not in ('unknown_observation','known_mask_passthrough'):
            raise ValueError('Mode must explicitly separate unknown faults from known-mask comparison')
        if type(allow_synthetic_for_tests) is not bool:raise ValueError('Test override must be explicit bool')
        self._binding=_binding(expected_binding)
        self.mode=mode
        self._calibration=copy.deepcopy(calibration)
        self._calibration_sha=None
        self._thresholds=None
        self._synthetic_test_override=allow_synthetic_for_tests
        self.calibration_status='NOT_VALIDATED_NO_CALIBRATION'
        if calibration is not None:
            if not isinstance(calibration, dict):
                raise ValueError('Observation calibration must be a structured record')
            if not _is_sha(expected_calibration_sha256) or calibration_sha256(calibration)!=expected_calibration_sha256:
                raise ValueError('Calibration does not match external trusted digest')
            if (set(calibration)!={'schema','binding','rules','thresholds','fit_statistics','heldout_report','measurement_kind'}
                    or calibration['schema']!=SCHEMA or calibration['binding']!=self._binding or calibration['rules']!=RULES):
                raise ValueError('Calibration schema/source/model/protocol/data binding mismatch')
            if calibration['measurement_kind'] not in ('measured_visible_model','synthetic_test_fixture'):
                raise ValueError('Missing explicit measured/synthetic provenance')
            self._calibration_sha=expected_calibration_sha256
            if calibration['measurement_kind']=='measured_visible_model':
                if (not isinstance(calibration, VerifiedObservationCalibration)
                        or calibration._issued_digest != expected_calibration_sha256):
                    raise ValueError('A self-hashed dictionary is not file-verified observation evidence')
                self._thresholds=_thresholds(calibration['thresholds'])
                self.calibration_status='CALIBRATED_HELDOUT_REPORTED'
            elif allow_synthetic_for_tests:
                self._thresholds=_thresholds(calibration['thresholds'])
                self.calibration_status='SYNTHETIC_TEST_ONLY_NOT_VALIDATED'
            else:self.calibration_status='NOT_VALIDATED_SYNTHETIC_CALIBRATION'
        self.reset()

    @property
    def calibration(self):return copy.deepcopy(self._calibration)

    def reset(self):
        self._last_timestamp=None
        self._previous=None
        self._quarantine=None
        self._repeat_run=0
        self._anomaly_streak=0
        self._clear_streak=0
        self._phase='CLEAR'
        self._counts={'frames':0,'warnings':0,'confirmed_fault_frames':0,'rejected_frames':0,'accepted_frames':0,'recoveries':0}

    def step(self, image, *, timestamp, prior_prediction=None):
        # This signature deliberately has no known availability or labels parameter.
        if self.mode == 'unknown_observation' and self._thresholds is None:
            raise RuntimeError('Unknown-observation routing is BLOCKED until calibration is verified')
        frame=_image(image)
        if type(timestamp) is not int or timestamp<0 or (self._last_timestamp is not None and timestamp!=self._last_timestamp+1):
            raise ValueError('Guard needs consecutive nonnegative observation timestamps')
        prediction=_prediction(prior_prediction,timestamp,self._binding)
        if self.mode == 'unknown_observation' and prediction is None:
            raise ValueError('The calibrated unknown-fault mode requires its pre-observation model prediction')
        features=visible_features(frame,self._previous,prediction,self._repeat_run)
        reasons=[]
        recovered=False
        if self.mode=='known_mask_passthrough':
            status='KNOWN_MASK_PASSTHROUGH';accept=True
        else:
            reasons=_reasons(features,self._thresholds)
            if reasons:
                self._anomaly_streak+=1;self._clear_streak=0
                if self._phase in ('FAULT','RECOVERING') or self._anomaly_streak>=RULES['confirm_frames']:
                    self._phase='FAULT'
                else:self._phase='WARNING'
                accept=False
            elif self._phase in ('WARNING','FAULT','RECOVERING'):
                self._clear_streak+=1;self._anomaly_streak=0
                if self._clear_streak>=RULES['recovery_frames']:
                    self._phase='CLEAR';self._clear_streak=0;recovered=True;accept=True
                else:self._phase='RECOVERING';accept=False
            else:
                self._anomaly_streak=0;self._clear_streak=0;accept=True
            status=self._phase
        self._previous=frame.copy()
        self._repeat_run=features['repeat_run']
        self._last_timestamp=timestamp
        self._quarantine=None if accept else frame.copy()
        self._counts['frames']+=1
        self._counts['warnings']+=int(status=='WARNING')
        self._counts['confirmed_fault_frames']+=int(status=='FAULT')
        self._counts['rejected_frames']+=int(not accept)
        self._counts['accepted_frames']+=int(accept)
        self._counts['recoveries']+=int(recovered)
        return {'status':status,'warning':status=='WARNING','confirmed_fault':status=='FAULT',
                'allow_posterior':accept,'accepted_image':frame.copy() if accept else None,
                'quarantine_sha256':None if self._quarantine is None else _digest(self._quarantine),
                'reasons':reasons,'features':features,'recovered':recovered,'timestamp':timestamp,
                'calibration_status':self.calibration_status,'innovation_available':prediction is not None,
                'mode':self.mode,'counters':copy.deepcopy(self._counts)}

    def snapshot(self):
        state={'schema':SCHEMA,'binding':copy.deepcopy(self._binding),'calibration_sha256':self._calibration_sha,
               'mode':self.mode,'calibration_status':self.calibration_status,
               'synthetic_test_override':self._synthetic_test_override,'last_timestamp':self._last_timestamp,
               'previous':None if self._previous is None else self._previous.copy(),
               'quarantine':None if self._quarantine is None else self._quarantine.copy(),
               'repeat_run':self._repeat_run,'anomaly_streak':self._anomaly_streak,
               'clear_streak':self._clear_streak,'phase':self._phase,'counts':copy.deepcopy(self._counts)}
        return {'state':state,'sha256':_digest(state)}

    def load(self, record, *, expected_snapshot_sha256):
        if (type(record) is not dict or set(record)!={'state','sha256'} or not _is_sha(expected_snapshot_sha256)
                or record['sha256']!=expected_snapshot_sha256 or _digest(record['state'])!=expected_snapshot_sha256):
            raise ValueError('Snapshot differs from trusted external digest')
        state=record['state'];prototype=self.snapshot()['state']
        if set(state)!=set(prototype):raise ValueError('Snapshot fields changed')
        for key in ('schema','binding','calibration_sha256','mode','calibration_status','synthetic_test_override'):
            if type(state[key]) is not type(prototype[key]) or _canonical(state[key])!=_canonical(prototype[key]):
                raise ValueError('Snapshot identity changed: '+key)
        for key in ('repeat_run','anomaly_streak','clear_streak'):
            if type(state[key]) is not int or state[key]<0:raise ValueError('Invalid state counter')
        last=state['last_timestamp']
        if last is not None and (type(last) is not int or last<0):raise ValueError('Invalid last timestamp')
        for key in ('previous','quarantine'):
            if state[key] is not None:_image(state[key])
        if state['phase'] not in ('CLEAR','WARNING','FAULT','RECOVERING'):raise ValueError('Invalid state phase')
        counts=state['counts']
        if set(counts)!=set(self._counts) or any(type(v) is not int or v<0 for v in counts.values()):
            raise ValueError('Invalid cumulative counters')
        if counts['frames']!=counts['accepted_frames']+counts['rejected_frames']:
            raise ValueError('Frame acceptance accounting contradicts snapshot')
        if any(value > counts['frames'] for value in counts.values()):
            raise ValueError('A cumulative event counter exceeds the frame count')
        if (last is None)!=(state['previous'] is None) or (last is None)!=(counts['frames']==0):
            raise ValueError('Initial state identity contradicts image/counter state')
        if state['repeat_run'] > max(0, counts['frames']-1):
            raise ValueError('Repeat run exceeds observed transitions')
        phase = state['phase']
        if (phase == 'CLEAR') != (state['quarantine'] is None):
            raise ValueError('Quarantine and phase contradict each other')
        if state['quarantine'] is not None and not np.array_equal(state['quarantine'], state['previous']):
            raise ValueError('Quarantine must contain exactly the most recent rejected observation')
        if phase == 'CLEAR' and (state['anomaly_streak'] or state['clear_streak']):
            raise ValueError('Clear state has residual anomaly/recovery streaks')
        if phase == 'WARNING' and (state['anomaly_streak'] != 1 or state['clear_streak']):
            raise ValueError('Warning state is not a first unconfirmed anomaly')
        if phase == 'RECOVERING' and (state['clear_streak'] != 1 or state['anomaly_streak']):
            raise ValueError('Recovery state must wait for its second clean frame')
        if phase == 'FAULT' and (state['anomaly_streak'] < 1 or state['clear_streak']):
            raise ValueError('Confirmed fault state has inconsistent streaks')
        self._last_timestamp=last;self._previous=copy.deepcopy(state['previous']);self._quarantine=copy.deepcopy(state['quarantine'])
        self._repeat_run=state['repeat_run'];self._anomaly_streak=state['anomaly_streak'];self._clear_streak=state['clear_streak']
        self._phase=state['phase'];self._counts=copy.deepcopy(counts)


def _fit_thresholds(records):
    values={key:[] for key in STATS}
    for row in records:
        previous=None;repeat=0
        for t,image in enumerate(row['images']):
            f=visible_features(image,previous,row['prior_images'][t],repeat)
            repeat=f['repeat_run'];previous=image
            if row['labels'][t]=='clean':
                for key in STATS:
                    if f[key] is not None:values[key].append(f[key])
    if any(len(v)<10 for v in values.values()):raise ValueError('Insufficient actual clean fit features')
    stats={key:{'count':len(v),'q01':float(np.quantile(v,.01,method='higher')),
                'q99':float(np.quantile(v,.99,method='higher'))} for key,v in values.items()}
    thresholds={'dark_mean_max':stats['mean']['q01'],'flat_std_max':stats['std']['q01'],
                'noise_high_frequency_min':stats['high_frequency']['q99'],
                'innovation_mse_min':stats['innovation_mse']['q99'],
                'freeze_repeat_min':max(3,int(stats['repeat_run']['q99'])+1)}
    return _thresholds(thresholds),stats


def evaluate_heldout(calibration, records, *, expected_binding):
    """Independent labels are used only here, NEVER as detector inputs."""
    expected_binding = _binding(expected_binding)
    if calibration.get('binding') != expected_binding or calibration.get('rules') != RULES:
        raise ValueError('Heldout evaluation must use the frozen bound calibration')
    _thresholds(calibration['thresholds'])
    _validate_records(records,HELDOUT_SEEDS)
    if records_sha256(records)!=expected_binding['heldout_data_sha256']:
        raise ValueError('Heldout records differ from external digest')
    totals={'clean_frames':0,'fault_frames':0,'false_positive_rejections':0,'false_positive_confirmations':0,
            'false_negative_accepted_faults':0,'false_negative_unconfirmed_faults':0,'rejected_frames':0}
    episodes=[]
    by_fault={label:{'frames':0,'accepted':0,'unconfirmed':0} for label in LABELS if label!='clean'}
    # Offline evaluation may exercise an unissued candidate, but this private
    # synthetic mechanism route never authorizes a production guard instance.
    mechanism = copy.deepcopy(dict(calibration))
    mechanism['measurement_kind'] = 'synthetic_test_fixture'
    for row in records:
        guard=ObservationGuard(mechanism,expected_binding=expected_binding,
             expected_calibration_sha256=calibration_sha256(mechanism),allow_synthetic_for_tests=True)
        outputs=[]
        for t,image in enumerate(row['images']):
            prior={'image_mean':row['prior_images'][t],'target_timestamp':t,'conditioning_timestamp':t-1,
                   'model_sha256':expected_binding['model_sha256'],'model_version':expected_binding['model_version']}
            outputs.append(guard.step(image,timestamp=t,prior_prediction=prior))
        clean=np.asarray([v=='clean' for v in row['labels']])
        accepted=np.asarray([v['allow_posterior'] for v in outputs]);confirmed=np.asarray([v['confirmed_fault'] for v in outputs])
        count={'clean_frames':int(clean.sum()),'fault_frames':int((~clean).sum()),
               'false_positive_rejections':int((clean&~accepted).sum()),
               'false_positive_confirmations':int((clean&confirmed).sum()),
               'false_negative_accepted_faults':int((~clean&accepted).sum()),
               'false_negative_unconfirmed_faults':int((~clean&~confirmed).sum()),
               'rejected_frames':int((~accepted).sum())}
        for key in totals:totals[key]+=count[key]
        for label, result in by_fault.items():
            mask=np.asarray([v==label for v in row['labels']])
            result['frames']+=int(mask.sum())
            result['accepted']+=int((mask&accepted).sum())
            result['unconfirmed']+=int((mask&~confirmed).sum())
        segments=[];t=0
        while t<len(clean):
            if clean[t]:t+=1;continue
            start=t
            while t<len(clean) and not clean[t]:t+=1
            end=t
            confirmed_times=[k for k in range(start,end) if confirmed[k]]
            rejected_times=[k for k in range(start,end) if not accepted[k]]
            next_fault=next((k for k in range(end,len(clean)) if not clean[k]),len(clean))
            recovered_times=[k for k in range(end,next_fault) if accepted[k]]
            segments.append({'start':start,'end_exclusive':end,'fault_types':sorted(set(row['labels'][start:end])),
                             'confirmation_delay_frames':None if not confirmed_times else confirmed_times[0]-start,
                             'quarantine_delay_frames':None if not rejected_times else rejected_times[0]-start,
                             'recovery_delay_frames':None if not recovered_times else recovered_times[0]-end,
                             'recovery_right_censored':not bool(recovered_times)})
        episodes.append({'seed':row['seed'],'counts':count,'segments':segments,
                         'statuses':[v['status'] for v in outputs]})
    def rate(n,d):return None if not totals[d] else totals[n]/totals[d]
    return {'status':'EMPIRICAL_HELDOUT_REPORT_NOT_UNIVERSAL_VALIDATION','counts':totals,
            'false_positive_rejection_rate':rate('false_positive_rejections','clean_frames'),
            'false_positive_confirmation_rate':rate('false_positive_confirmations','clean_frames'),
            'false_negative_acceptance_rate':rate('false_negative_accepted_faults','fault_frames'),
            'false_negative_confirmation_rate':rate('false_negative_unconfirmed_faults','fault_frames'),
            'by_fault_type':by_fault,'episodes':episodes,'thresholds_changed_using_heldout':False}


def fit_observation_calibration(fit_records, heldout_records, *, binding, measurement_kind):
    """Freeze fit-derived thresholds, then score untouched heldout trajectories."""
    binding=_binding(binding)
    if measurement_kind not in ('measured_visible_model','synthetic_test_fixture'):
        raise ValueError('Measured provenance or explicitly synthetic fixture required')
    _validate_records(fit_records,FIT_SEEDS);_validate_records(heldout_records,HELDOUT_SEEDS)
    if records_sha256(fit_records)!=binding['fit_data_sha256'] or records_sha256(heldout_records)!=binding['heldout_data_sha256']:
        raise ValueError('Calibration data do not match externally trusted digests')
    thresholds,statistics=_fit_thresholds(fit_records)
    calibration={'schema':SCHEMA,'binding':binding,'rules':copy.deepcopy(RULES),'thresholds':thresholds,
                 'fit_statistics':statistics,'heldout_report':None,'measurement_kind':measurement_kind}
    calibration['heldout_report']=evaluate_heldout(calibration,heldout_records,expected_binding=binding)
    return calibration


def seal_observation_calibration(directory, fit_records, heldout_records, *, binding,
                                 measurement_kind):
    """Create a new immutable evidence bundle; never overwrite an existing path.

    The return value is a manifest digest, not deployment permission. Retain it
    in the trusted collection record together with actual model/source hashes,
    frozen model/RNG evidence and the pre-observation invocation trace. Inputs
    are the visible corrupted images plus pre-observation model predictions;
    hidden clean target images are neither required nor accepted in the schema.
    """
    calibration=fit_observation_calibration(fit_records,heldout_records,
                      binding=binding,measurement_kind=measurement_kind)
    directory=Path(directory)
    directory.mkdir(parents=True,exist_ok=False)
    entries=[]
    def record_file(path):
        raw=path.read_bytes()
        entries.append({'path':path.name,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()})
    for phase,records in (('fit',fit_records),('heldout',heldout_records)):
        for row in records:
            path=directory/(phase+'_'+str(row['seed'])+'.npz')
            with path.open('xb') as out:
                np.savez_compressed(out,images=row['images'],prior_images=row['prior_images'],
                    labels=np.asarray(row['labels'],dtype='<U6'))
            record_file(path)
    path=directory/'calibration.json'
    with path.open('xb') as out:out.write(_canonical(calibration))
    record_file(path)
    manifest={'schema':BUNDLE_SCHEMA,'binding':copy.deepcopy(binding),
              'measurement_kind':measurement_kind,
              'calibration_sha256':calibration_sha256(calibration),
              'entries':sorted(entries,key=lambda x:x['path'])}
    raw=_canonical(manifest)
    with (directory/'manifest.json').open('xb') as out:out.write(raw)
    return {'manifest_sha256':hashlib.sha256(raw).hexdigest(),
            'calibration_sha256':calibration_sha256(calibration),'file_count':len(entries)+1,
            'status':'SEALED_NOT_DEPLOYMENT_AUTHORITY'}


def load_observation_calibration(directory, *, expected_manifest_sha256, expected_binding,
                                allow_synthetic_for_tests=False):
    """Verify trusted bundle bytes and recompute fit and independent evaluation.

    No ``passed`` flag or caller-supplied calibration dictionary is accepted.
    Synthetic bundles cannot issue production activation; an explicit test flag
    only permits a handle that remains marked synthetic and needs the separate
    constructor test flag. A trusted manifest authenticates a collection job,
    not a mathematical claim of reliable semantic fault detection.
    """
    if type(allow_synthetic_for_tests) is not bool:
        raise ValueError('Synthetic test override must be an explicit bool')
    binding=_binding(expected_binding)
    if not _is_sha(expected_manifest_sha256):raise ValueError('External manifest SHA is required')
    directory=Path(directory)
    if directory.is_symlink() or not directory.is_dir():raise ValueError('A regular evidence directory is required')
    manifest_path=directory/'manifest.json'
    if manifest_path.is_symlink() or not manifest_path.is_file():raise ValueError('Manifest must be a regular file')
    manifest_bytes=manifest_path.read_bytes()
    if hashlib.sha256(manifest_bytes).hexdigest()!=expected_manifest_sha256:
        raise ValueError('Observation manifest differs from external trusted digest')
    manifest=json.loads(manifest_bytes)
    if (type(manifest) is not dict
            or set(manifest)!={'schema','binding','measurement_kind','calibration_sha256','entries'}
            or manifest['schema']!=BUNDLE_SCHEMA or manifest['binding']!=binding
            or not _is_sha(manifest['calibration_sha256'])):
        raise ValueError('Observation manifest schema/binding mismatch')
    kind=manifest['measurement_kind']
    if kind not in ('measured_visible_model','synthetic_test_fixture'):
        raise ValueError('Unknown observation measurement provenance')
    if kind=='synthetic_test_fixture' and not allow_synthetic_for_tests:
        raise ValueError('Synthetic observation data cannot authorize production')
    expected_files={'calibration.json'}|{'fit_'+str(s)+'.npz' for s in FIT_SEEDS}|{'heldout_'+str(s)+'.npz' for s in HELDOUT_SEEDS}
    entries=manifest['entries']
    if type(entries) is not list or len(entries)!=len(expected_files):raise ValueError('Incomplete observation evidence manifest')
    verified={}
    for entry in entries:
        if (type(entry) is not dict or set(entry)!={'path','bytes','sha256'}
                or type(entry['path']) is not str or entry['path'] not in expected_files
                or entry['path'] in verified or not _is_sha(entry['sha256'])
                or type(entry['bytes']) is not int or entry['bytes']<0):
            raise ValueError('Invalid, duplicate or escaping evidence member')
        path=directory/entry['path']
        if path.is_symlink() or not path.is_file():raise ValueError('Evidence member is not a regular file')
        raw=path.read_bytes()
        if len(raw)!=entry['bytes'] or hashlib.sha256(raw).hexdigest()!=entry['sha256']:
            raise ValueError('Observation evidence member SHA/length mismatch: '+entry['path'])
        verified[entry['path']]=raw
    if set(verified)!=expected_files or {p.name for p in directory.iterdir()}!=expected_files|{'manifest.json'}:
        raise ValueError('Observation bundle member set differs from its frozen manifest')
    import io
    partitions=[]
    for phase,seeds in (('fit',FIT_SEEDS),('heldout',HELDOUT_SEEDS)):
        rows=[]
        for seed in seeds:
            with np.load(io.BytesIO(verified[phase+'_'+str(seed)+'.npz']),allow_pickle=False) as data:
                if set(data.files)!={'images','prior_images','labels'}:raise ValueError('Unexpected observation NPZ fields')
                labels=data['labels']
                if labels.ndim!=1 or labels.dtype.kind!='U':raise ValueError('Invalid offline label encoding')
                rows.append({'seed':seed,'images':data['images'].copy(),
                             'prior_images':data['prior_images'].copy(),'labels':labels.tolist()})
        partitions.append(rows)
    reproduced=fit_observation_calibration(*partitions,binding=binding,measurement_kind=kind)
    recorded=json.loads(verified['calibration.json'])
    if (_canonical(recorded)!=_canonical(reproduced)
            or calibration_sha256(reproduced)!=manifest['calibration_sha256']):
        raise ValueError('Calibration thresholds/heldout results do not reproduce from sealed raw data')
    if manifest_path.read_bytes()!=manifest_bytes:
        raise ValueError('Observation manifest changed during verification')
    for name,raw in verified.items():
        if (directory/name).is_symlink() or (directory/name).read_bytes()!=raw:
            raise ValueError('Observation evidence member changed during verification: '+name)
    return VerifiedObservationCalibration(reproduced,_token=_FILE_VERIFIED_TOKEN)
