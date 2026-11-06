"""Model-only replay of ONE shared independent action bank; zero env calls.

The controller receives forced historical actions, never its proposed action's
imaginary ground truth. Virtual hidden bank frames cannot enter its posterior.
Raw model planning outputs are counterfactual diagnostics, not real outcomes.
This module imports no environment and never updates learner or auxiliary heads.
"""
from __future__ import annotations
import copy
import hashlib
import os
from pathlib import Path
import time
import numpy as np

from . import bank_gate as g
from . import calibration_bank as original
from .belief import canonical_json,accumulate_risk
from .epistemic import tree_digest
from .full_controller import FullController


def collector_sha256():return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def load_original_bank(directory,manifest_sha256,*,environment_lock_sha256,origin_source_sha256,origin_protocol_sha256):
    """Explicit immutable C1-bank lineage bridge; never relabel its manifest."""
    bank,manifest=original.load_bank(directory,manifest_sha256,origin_protocol_sha256,
        environment_lock_sha256=environment_lock_sha256,source_sha256=origin_source_sha256)
    origin={'directory':str(Path(directory).resolve()),'manifest_sha256':manifest_sha256,
        'environment_lock_sha256':environment_lock_sha256,'source_sha256':origin_source_sha256,
        'protocol_sha256':origin_protocol_sha256,'actions':2480,'dmc_calls':4960,
        'shared_identically_between_arms':True,'learner_or_heads_updates':0,
        'files':copy.deepcopy(manifest['files'])}
    return bank,origin


def _check_bank(bank):
    if type(bank) is not dict or any(type(seed) is not int for seed in bank):
        raise ValueError('Independent bank requires exact integer seed keys; aliases are forbidden')
    if set(bank)!=set(g.FIT_SEEDS+g.VALIDATION_SEEDS):raise ValueError('Independent FIT/HELDOUT seed sets differ')
    for seed,record in bank.items():
        schema={'image':((125,64,64,3),np.uint8),'action':((124,6),np.float32),
                'reward':((124,),np.float64),'dmc_calls':((124,),np.int32)}
        if set(record)!=set(schema):raise ValueError('Unexpected independent bank fields')
        for key,(shape,dtype) in schema.items():
            g.numeric._array(record[key],dtype,shape,'bank '+key)
        if not np.array_equal(record['action'],original.fixed_actions(seed)) or np.any(record['dmc_calls']!=2):
            raise ValueError('Bank must preserve original exogenous action tape and repeat2')
        if np.any(record['reward']<0) or np.any(record['reward']>2):raise ValueError('Invalid actual bank rewards')


def _bank_content_hash(bank):
    """Canonicalize the seed-indexed bank without weakening generic tree hashes.

    Float/NumPy/string aliases must be rejected before Python key equality can
    make them look like registered seeds. The exact integer seed is retained in
    a sorted list whose record dictionaries have only string field names.
    """
    _check_bank(bank)
    return tree_digest([{'seed':seed,'record':bank[seed]} for seed in sorted(bank)])


def measure_window(backbone,heads,context,bank_record,*,seed,gap,phase,candidate=None):
    g.context_valid(context)
    if context['collector_sha256']!=collector_sha256():raise ValueError('Executed bank collector source mismatch')
    expected=g.FIT_SEEDS if phase=='fit' else g.VALIDATION_SEEDS
    if type(seed) is not int or seed not in expected or type(gap) is not int or gap not in g.GAPS:raise ValueError('Invalid bank partition/window')
    if phase=='validation':g.validate_fitted_candidate(candidate,context)
    elif candidate is not None:raise ValueError('FIT cannot use a fitted candidate')
    controller=FullController(backbone,heads,context,seed=int(np.random.SeedSequence([760000,seed,gap]).generate_state(1)[0]),
        execution='bank_'+phase,fitted_bank_candidate=candidate)
    diagnostic_rng=np.random.default_rng(np.random.SeedSequence([760001,seed,gap]))
    controller.reset(bank_record['image'][0].copy(),episode_id=seed)
    arrays={k:[] for k in g.ARRAY_FIELDS if k!='previous_action'};plans=[];previous=None
    for t in range(1,17+gap+8):
        # State features refer to BEFORE the exact incoming recorded action.
        action=bank_record['action'][t-1].copy()
        query=g.support_query(controller.auxiliary_state,action.reshape(1,6))[0]
        prior=np.zeros(6,np.float64) if controller.last_executed_action is None else controller.last_executed_action.astype(np.float64).copy()
        executed=controller.act(external_action=action,deterministic=False)
        if not np.array_equal(action,executed):raise ValueError('Forced historical action was modified')
        visible=not 17<=t<17+gap
        image=bank_record['image'][t].copy() if visible else np.zeros((64,64,3),np.uint8)
        old_risk=controller.cumulative_risk
        info={'availability':visible,'observation_timestamp':t,'env_steps_this_step':2,'executed_action':action}
        controller.step(image,action,info,episode_id=seed)
        if t<17:continue
        if t==17:previous=prior
        pred=controller.last_prediction
        if pred is None or pred['timestamp']!=t:raise ValueError('Missing pre-observation predictions')
        metrics=pred['metrics'];reset=controller.last_metrics['risk_reset']
        risk=old_risk if visible else accumulate_risk(old_risk,metrics['uncertainty'],100)
        if not np.isclose(controller.cumulative_risk,0. if reset else risk,rtol=0,atol=2e-12):raise ValueError('Bank risk event differs')
        values={'timestamp':t,'availability':visible,'actions':action.astype(np.float64),
            'actual_rewards':float(bank_record['reward'][t-1]),'predictive_entropy':metrics['predictive_entropy'],
            'expected_entropy':metrics['expected_entropy'],'candidate_weights':controller.weights.copy(),
            'cumulative_risk':risk,'risk_reset':reset,'fallback':False,
            'fusion_lambda':controller.last_metrics['fusion_lambda'],'fusion_kl':controller.last_metrics['fusion_kl'],
            'epistemic_disagreement':controller.current_epistemic,'support_query':query}
        for k,v in values.items():arrays[k].append(copy.deepcopy(v))
        if visible:
            if pred['prediction_error'] is None or controller.current_temporal_error is None:raise ValueError('Observed errors missing')
            arrays['energy'].append(np.asarray(pred['prediction_error'],np.float64).copy())
            arrays['temporal_prediction_error'].append(float(controller.current_temporal_error))
        elif pred['prediction_error'] is not None or controller.current_temporal_error is not None:
            raise ValueError('Virtual hidden bank target leaked into controller diagnostics')
        before=tree_digest(controller.snapshot())
        proposed=np.asarray(backbone.act(controller.state,deterministic=False,rng=diagnostic_rng),np.float32)
        measured,_=controller.scratch_plans(proposed[0],rng=diagnostic_rng,archive=False)
        if tree_digest(controller.snapshot())!=before:raise ValueError('Scratch planning changed actual recorded history')
        plans.extend(copy.deepcopy(p['measurement']) for p in measured)
    row={k:np.asarray(v,np.bool_ if k in ('availability','risk_reset','fallback') else np.int32 if k=='timestamp' else np.float64)
         for k,v in arrays.items()}
    row.update(seed=seed,gap=gap,phase=phase,context=copy.deepcopy(context),measurement_protocol=g.MEASUREMENT_PROTOCOL,
        candidate_sha256=None if candidate is None else candidate['sha256'],execution_mode='forced_recorded_actions',
        model_updates=0,hidden_target_to_controller=False,hidden_target_to_learner=False,
        planning_actions_are_counterfactual=True,counterfactual_actual_outcomes_available=False,
        plans=plans,previous_action=previous)
    return row


def _write(path,value):
    with Path(path).open('xb') as f:f.write(canonical_json(value))


def measure_and_write(backbone,heads,context,bank,*,directory,bank_origin,frozen_state_hashes,device,gpu):
    """Synchronously measure one pin from the shared bank and seal its gate.

    No new model anchor, process, environment sample or model update is needed
    per pin. A second real fresh-process execution is reserved for preflight and
    final recovery audits. Caller must pass actual live freeze measurements.
    """
    g.context_valid(context);_check_bank(bank)
    if bank_origin['manifest_sha256']!=context['bank_manifest_sha256']:raise ValueError('Wrong shared bank identity')
    if context['measurement_kind']=='actual_gpu_independent_bank':
        from .provenance import source_digest
        source_before=source_digest()
        if source_before[0]!=context['source_sha256']:raise ValueError('Actual current runtime source differs from external context')
        if device!='cuda' or not isinstance(gpu,str) or not gpu.strip():raise ValueError('Actual GPU identity required')
        actual,origin=load_original_bank(bank_origin['directory'],bank_origin['manifest_sha256'],
            environment_lock_sha256=bank_origin['environment_lock_sha256'],origin_source_sha256=bank_origin['source_sha256'],
            origin_protocol_sha256=bank_origin['protocol_sha256'])
        if origin!=bank_origin or _bank_content_hash(actual)!=_bank_content_hash(bank):raise ValueError('Supplied arrays differ from immutable independent bank')
    root=Path(directory).resolve();root.mkdir(parents=True,exist_ok=False)
    before=copy.deepcopy(frozen_state_hashes());rows=[];windows=[];fit_candidate=None
    try:
        for phase,seeds in (('fit',g.FIT_SEEDS),('validation',g.VALIDATION_SEEDS)):
            if phase=='validation':
                fit_candidate=g.fit_candidate(rows,context);_write(root/'candidate.json',fit_candidate)
                sealed=time.perf_counter_ns()
                if frozen_state_hashes()!=before:raise ValueError('Model changed during FIT; HELDOUT not started')
                heldout_started=time.perf_counter_ns()
            for seed in seeds:
                for gap in g.GAPS:
                    row=measure_window(backbone,heads,context,bank[seed],seed=seed,gap=gap,phase=phase,candidate=fit_candidate)
                    rows.append(row);windows.append(g.write_window(row,root))
        after=copy.deepcopy(frozen_state_hashes())
        if before!=after:raise ValueError('Model/optimizer/normalizer/counters/host-RNG/heads changed during bank replay')
        if context['measurement_kind']=='actual_gpu_independent_bank' and source_digest()!=source_before:
            raise ValueError('Source changed during actual bank replay')
        record={'schema':g.SCHEMA,'context':context,'bank_origin':copy.deepcopy(bank_origin),
            'process_id':os.getpid(),'device':device,'gpu':gpu,'freeze_before':before,'freeze_after':after,
            'environment_actions':0,'author_updates':0,'auxiliary_updates':0,
            'offline_action_prefixes':6600,'windows':windows,'rows_sha256':g.rows_digest(rows),
            'candidate':g.numeric.file_descriptor(root/'candidate.json',root),
            'candidate_sealed_ns':sealed,'heldout_started_ns':heldout_started,'completed_ns':time.perf_counter_ns()}
        _write(root/'bank_measurement.json',record)
        measurement_sha=g.numeric.file_descriptor(root/'bank_measurement.json',root)['sha256']
        sealed_manifest=g.seal_measurement(root,measurement_sha,context)
        handle=g.load_bank_gate(root,sealed_manifest['sha256'],context,require_active=False)
        result={'manifest_sha256':sealed_manifest['sha256'],'measurement_sha256':measurement_sha,
                'active':handle['active'],'validation':handle['validation'],'environment_actions':0,
                'scope':g.SCOPE,'fresh_process_verified':False}
        _write(root/'result.json',result);return result
    except BaseException as exc:
        _write(root/'failure.json',{'error':repr(exc),'completed_windows':len(rows),
            'environment_actions':0,'partial_results_retained':True,'active':False})
        raise
