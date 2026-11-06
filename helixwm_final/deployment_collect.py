"""Actual closed-loop fit/heldout collection, separate from learner data.

The caller supplies a pinned author backbone, pinned independent heads, a fresh
environment factory and a retained checkpoint/RNG anchor. This module never
trains, receives hidden clean observations, or issues production authority.
One fit phase and one frozen-candidate heldout phase are written exclusively.
Independent fresh-process replay is a separate gate, not a copied success flag.
"""
from __future__ import annotations

import copy
import hashlib
import os
from pathlib import Path
import time

import numpy as np

from . import deployment_calibration as d
from .belief import accumulate_risk, canonical_json
from .epistemic import tree_digest
from .full_controller import FullController

CONTROLLER_SEED_ROOT = 760000
DIAGNOSTIC_SEED_ROOT = 760001
WARMUP_ACTIONS, RECOVERY_ACTIONS = 16, 8


def collector_sha256():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def scenario_bank():
    """Registered reset/mask cases only; this is not an open-loop image bank."""
    return {'schema': d.SCHEMA, 'measurement_protocol': d.MEASUREMENT_PROTOCOL,
            'fit_seeds': list(d.FIT_SEEDS), 'validation_seeds': list(d.VALIDATION_SEEDS),
            'gaps': list(d.GAPS), 'warmup_actions': WARMUP_ACTIONS,
            'recovery_actions': RECOVERY_ACTIONS, 'gap_first_observation': 17,
            'controller_seed_root': CONTROLLER_SEED_ROOT,
            'diagnostic_seed_root': DIAGNOSTIC_SEED_ROOT,
            'policy_deterministic': False, 'training_updates': 0,
            'included_in_training_budget': False,
            'total_actions': 6600, 'total_DMC_calls': 13200}


def scenario_bank_sha256():
    return hashlib.sha256(canonical_json(scenario_bank())).hexdigest()


def _write_json(path, value):
    path = Path(path)
    with path.open('xb') as stream:
        stream.write(canonical_json(value))
    return path


def _context(context):
    d._context(context)
    if context['collector_sha256'] != collector_sha256():
        raise ValueError('External collector SHA does not match executed collector bytes')
    if context['bank_manifest_sha256'] != scenario_bank_sha256():
        raise ValueError('Closed-loop scenario schedule differs from externally bound bank')


def _frame(image):
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.shape != (64,64,3):
        raise ValueError('Only visible RGB64 uint8 frames may enter calibration control')
    return image.copy()


def _seed(seed, gap, root):
    return int(np.random.SeedSequence([root, seed, gap]).generate_state(1)[0])


class WindowCollectionError(RuntimeError):
    def __init__(self, message, *, actions, dmc_calls):
        super().__init__(message)
        self.acknowledged_actions = actions
        self.acknowledged_dmc_calls = dmc_calls


def collect_window(backbone, heads, context, *, env_factory, seed, gap, phase,
                   anchor_sha256, candidate=None, trajectory=None):
    """Return (raw row, replay trace, count audit) for one actual closed loop.

    ``row.actions[t]`` is the incoming executed action producing observation
    ``row.timestamp[t]``. Scratch plans are from that CURRENT state and execute
    no environment interaction. Their previous action equals row.actions[t].
    Risk is recorded before any qualified observation reset at that outcome.
    Temporal errors exist only for the eight observed recovery frames.
    """
    _context(context)
    seeds = d.FIT_SEEDS if phase == 'fit' else d.VALIDATION_SEEDS if phase == 'validation' else ()
    if type(seed) is not int or seed not in seeds or type(gap) is not int or gap not in d.GAPS:
        raise ValueError('Unregistered calibration phase/seed/gap')
    if not d._sha(anchor_sha256):
        raise ValueError('Retained checkpoint/head/RNG anchor identity required')
    if phase == 'validation':
        d.validate_candidate_for_calibration(candidate, context)
    elif candidate is not None:
        raise ValueError('Fit pilot cannot receive a fitted candidate')
    controller = FullController(backbone, heads, context,
        seed=_seed(seed,gap,CONTROLLER_SEED_ROOT), execution='calibration_'+phase,
        fitted_candidate=candidate, trajectory=trajectory)
    probe_rng = np.random.default_rng(np.random.SeedSequence([DIAGNOSTIC_SEED_ROOT, seed, gap]))
    mask = np.ones(501,np.bool_)
    mask[17:17+gap] = False
    env = env_factory()
    actions_count = calls = 0
    arrays = {name: [] for name in d.ARRAY_FIELDS if name != 'previous_action'}
    plans, observation_hashes, controller_hashes, rng_hashes = [], [], [], []
    previous_action = None
    try:
        image = _frame(env.reset(seed=seed, mask=mask))
        controller.reset(image,episode_id=seed)
        for timestamp in range(1,17+gap+8):
            prior_action = (np.zeros(6,np.float64) if controller.last_executed_action is None
                            else controller.last_executed_action.astype(np.float64).copy())
            action = controller.act(deterministic=False)
            before_a, before_c = env.action_decisions, env.episode_env_steps
            try:
                image,reward,done,info = env.step(action)
            finally:
                actions_count += env.action_decisions-before_a
                calls += env.episode_env_steps-before_c
            if (done or type(info.get('availability')) is not bool
                    or info['availability'] != bool(mask[timestamp])
                    or type(info.get('observation_timestamp')) is not int
                    or info['observation_timestamp'] != timestamp
                    or info.get('env_steps_this_step') != 2
                    or not np.array_equal(info.get('executed_action'), action)):
                raise ValueError('Calibration outcome changed mask/time/action/native-call contract')
            if (isinstance(reward,(bool,np.bool_)) or not isinstance(reward,(float,np.float64))
                    or not np.isfinite(reward) or not 0. <= reward <= 2.):
                raise ValueError('Invalid actual DMC reward')
            image = _frame(image)
            old_risk = controller.cumulative_risk
            controller.step(image,action,info,episode_id=seed)
            if timestamp < 17:
                continue
            if timestamp == 17:
                previous_action = prior_action
            prediction = controller.last_prediction
            if prediction is None or prediction['timestamp'] != timestamp or len(prediction['weights']) != 8:
                raise ValueError('Actual bounded candidate prediction diagnostics missing')
            metrics = prediction['metrics']
            risk_before_reset = (old_risk if info['availability'] else
                                 accumulate_risk(old_risk, metrics['uncertainty'], 100))
            reset = controller.last_metrics['risk_reset']
            if type(reset) is not bool:
                raise ValueError('Qualified reset must be a bool event')
            expected_after = 0. if reset else risk_before_reset
            if not np.isclose(controller.cumulative_risk,expected_after,rtol=0,atol=2e-12):
                raise ValueError('Controller reset event omitted or differs from risk-state update')
            values = {'timestamp': timestamp, 'availability': info['availability'],
                'actions': action.astype(np.float64), 'actual_rewards': float(reward),
                'predictive_entropy': metrics['predictive_entropy'],
                'expected_entropy': metrics['expected_entropy'],
                'candidate_weights': controller.weights.copy(),
                'cumulative_risk': float(risk_before_reset), 'risk_reset': reset,
                'fallback': controller.last_action_decision['kind']=='prior_only_fallback',
                'fusion_lambda': controller.last_metrics['fusion_lambda'],
                'fusion_kl': controller.last_metrics['fusion_kl'],
                'epistemic_disagreement': controller.current_epistemic}
            for name,value in values.items():
                arrays[name].append(copy.deepcopy(value))
            if info['availability']:
                error = prediction['prediction_error']
                if error is None or np.asarray(error).shape != (8,) or controller.current_temporal_error is None:
                    raise ValueError('Observed recovery requires actual pre-observation errors')
                arrays['energy'].append(np.asarray(error,np.float64).copy())
                arrays['temporal_prediction_error'].append(float(controller.current_temporal_error))
            elif prediction['prediction_error'] is not None or controller.current_temporal_error is not None:
                raise ValueError('Missing observation cannot produce a measured prediction error')
            # The diagnostic stream cannot consume actor/fusion/planning RNG.
            before_snapshot = controller.snapshot()
            if trajectory is not None:
                before_snapshot['integration']['trajectory'] = None
            before = tree_digest(before_snapshot)
            proposal = np.asarray(backbone.act(controller.state,deterministic=False,rng=probe_rng),np.float32)
            if proposal.shape != (1,6):
                raise ValueError('Expected one diagnostic proposal action')
            measured,_ = controller.scratch_plans(proposal[0],rng=probe_rng,archive=trajectory is not None)
            after_snapshot = controller.snapshot()
            if trajectory is not None:
                after_snapshot['integration']['trajectory'] = None
            if tree_digest(after_snapshot) != before:
                raise ValueError('Scratch measurement mutated collector control state')
            plans.extend(copy.deepcopy(item['measurement']) for item in measured)
            observation_hashes.append(tree_digest(image))
            controller_hashes.append(tree_digest(controller.snapshot()))
            rng_hashes.append(tree_digest({'controller':{k:v.bit_generator.state for k,v in controller.rngs.items()},
                                          'screen':controller.screen_rng.bit_generator.state,
                                          'diagnostic':probe_rng.bit_generator.state}))
        expected_actions = 16+gap+8
        if actions_count != expected_actions or calls != 2*expected_actions:
            raise ValueError('Calibration window actual action/DMC count differs')
    except BaseException as exc:
        if isinstance(exc,(KeyboardInterrupt,SystemExit)):
            exc.acknowledged_actions=actions_count
            exc.acknowledged_dmc_calls=calls
            raise
        raise WindowCollectionError(str(exc), actions=actions_count, dmc_calls=calls) from exc
    finally:
        env.close()
    booleans = {'availability','risk_reset','fallback'}
    row = {name:np.asarray(value, np.bool_ if name in booleans else
                           np.int32 if name=='timestamp' else np.float64)
           for name,value in arrays.items()}
    row.update(seed=seed,gap=gap,phase=phase,context=copy.deepcopy(context),
        measurement_protocol=d.MEASUREMENT_PROTOCOL,
        candidate_sha256=None if candidate is None else candidate['sha256'],
        model_updates=0,hidden_target_to_controller=False,hidden_target_to_learner=False,
        replay_anchor_sha256=anchor_sha256,trace_sha256='0'*64,
        plans=plans,previous_action=previous_action)
    trace = {'schema':d.SCHEMA,'context':copy.deepcopy(context),'phase':phase,'seed':seed,'gap':gap,
        'execution_mode':'closed_loop','diagnostics_sha256':d.diagnostic_content_sha256(row),
        'replay_anchor_sha256':anchor_sha256,'observation_sha256':observation_hashes,
        'controller_state_sha256':controller_hashes,'rng_sha256':rng_hashes}
    row['trace_sha256'] = hashlib.sha256(canonical_json(trace)).hexdigest()
    audit = {'phase':phase,'seed':seed,'gap':gap,'actions':actions_count,'dmc_calls':calls,
        'process_id':os.getpid(),'model_updates':0,'included_in_training_budget':False,
        'controller_seed':_seed(seed,gap,CONTROLLER_SEED_ROOT)}
    return row,trace,audit


def collect_phase(backbone, heads, context, *, env_factory, phase, directory,
                  anchor_sha256, candidate=None):
    """Write 60 non-overwriting windows and traces; preserve any failed attempt."""
    root=Path(directory).resolve()
    root.mkdir(parents=True,exist_ok=True)
    claim=d.claim_phase(root,context,phase,candidate)
    started_ns=time.perf_counter_ns()
    seeds=d.FIT_SEEDS if phase=='fit' else d.VALIDATION_SEEDS
    rows,windows,traces,audits=[],[],[],[]
    try:
        for seed in seeds:
            for gap in d.GAPS:
                row,trace,audit=collect_window(backbone,heads,context,env_factory=env_factory,
                    seed=seed,gap=gap,phase=phase,anchor_sha256=anchor_sha256,candidate=candidate)
                audits.append(audit)
                path=_write_json(root/f'{phase}_seed{seed}_gap{gap}.trace.json',trace)
                descriptor=d.file_descriptor(path,root)
                if descriptor['sha256']!=row['trace_sha256']:
                    raise ValueError('Written trace bytes do not match row commitment')
                windows.append(d.write_measurement_window(row,root));traces.append(descriptor)
                _write_json(root/f'{phase}_seed{seed}_gap{gap}.counts.json',audit)
                rows.append(row)
    except BaseException as exc:
        failure={'phase':phase,'error_type':type(exc).__name__,'error':str(exc),
            'completed_windows':len(rows),'acknowledged_actions':sum(a['actions'] for a in audits)+getattr(exc,'acknowledged_actions',0),
            'acknowledged_dmc_calls':sum(a['dmc_calls'] for a in audits)+getattr(exc,'acknowledged_dmc_calls',0),
            'partial_results_retained':True,'usable_for_deployment':False}
        _write_json(root/f'{phase}.failure.json',failure)
        raise
    return {'rows':rows,'windows':windows,'raw_traces':traces,'audits':audits,
            'claim':d.file_descriptor(claim,root),'started_ns':started_ns,'completed_ns':time.perf_counter_ns()}


def collect_sequence(backbone,heads,context,*,env_factory,directory,anchor_sha256,
                     anchor_descriptor,frozen_state_hashes,device,gpu):
    """Fit once then validate once; leave deployment blocked pending replay.

    ``frozen_state_hashes`` is a caller function measuring all FREEZE_COMPONENTS
    from actual live state, not a constant passed:true object. The retained
    anchor must already exist under directory; the independent replay job later
    verifies every window and supplies the final manifest. No manifest is issued
    here, even when heldout empirical coverage happens to pass.
    """
    _context(context)
    if not callable(frozen_state_hashes):
        raise ValueError('Live frozen-state measurement callback is mandatory')
    root=Path(directory).resolve();root.mkdir(parents=True,exist_ok=True)
    d._verified_file(root,anchor_descriptor)
    if anchor_descriptor['sha256']!=anchor_sha256:
        raise ValueError('Retained anchor file differs from the externally supplied identity')
    def measure():
        value=copy.deepcopy(frozen_state_hashes())
        if (type(value) is not dict or set(value)!=set(d.FREEZE_COMPONENTS)
                or any(not d._sha(v) for v in value.values())
                or value['model']!=context['model_sha256']
                or value['epistemic_heads']!=context['epistemic_heads_sha256']):
            raise ValueError('Live frozen-state identities are incomplete or mismatched')
        return value
    before=measure()
    fit=collect_phase(backbone,heads,context,env_factory=env_factory,phase='fit',directory=root,
                      anchor_sha256=anchor_sha256)
    candidate=d.fit_candidate(fit['rows'],context)
    candidate_path=_write_json(root/'candidate.json',candidate)
    sealed_ns=time.perf_counter_ns()
    if measure()!=before:
        _write_json(root/'freeze.failure.json',{'status':'MODEL_CHANGED_DURING_FIT','before':before,'after':measure()})
        raise ValueError('Model/optimizer/heads/RNG changed during fit; heldout collection not started')
    validation=collect_phase(backbone,heads,context,env_factory=env_factory,phase='validation',directory=root,
                             anchor_sha256=anchor_sha256,candidate=candidate)
    audit=d.evaluate_candidate(candidate,validation['rows'],context)
    freeze={'schema':d.SCHEMA,'context':copy.deepcopy(context),'device':device,'gpu':gpu,'model_updates':0,
        'before':before,'after':measure(),
        'fit_records_sha256':candidate['fit_records_sha256'],
        'validation_records_sha256':audit['validation_records_sha256']}
    freeze_path=_write_json(root/'freeze.json',freeze)
    d._frozen_evidence(freeze,context,candidate['fit_records_sha256'],audit['validation_records_sha256'])
    ledger={'schema':d.SCHEMA,'context':copy.deepcopy(context),'event_clock':d.EVENT_CLOCK,
        'fit_attempts':1,'validation_attempts':1,
        'holdout_threshold_revisions':0,'candidate_sha256':candidate['sha256'],
        'fit_records_sha256':candidate['fit_records_sha256'],
        'validation_records_sha256':audit['validation_records_sha256'],
        'fit_completed_ns':fit['completed_ns'],'candidate_sealed_ns':sealed_ns,
        'validation_started_ns':validation['started_ns'],'validation_completed_ns':validation['completed_ns']}
    ledger_path=_write_json(root/'ledger.json',ledger)
    result={'schema':d.SCHEMA,'context':copy.deepcopy(context),
        'status':'INDEPENDENT_REPLAY_REQUIRED' if audit['validation']['passed'] else 'CLOSED_LOOP_VALIDATION_FAILED',
        'active':False,'candidate':d.file_descriptor(candidate_path,root),
        'windows':fit['windows']+validation['windows'],'raw_traces':fit['raw_traces']+validation['raw_traces'],
        'freeze_evidence':d.file_descriptor(freeze_path,root),'phase_ledger':d.file_descriptor(ledger_path,root),
        'phase_claims':{'fit':fit['claim'],'validation':validation['claim']},
        'replay_anchors':[anchor_descriptor],
        'counts':{'actions':sum(r['actions'] for r in fit['audits']+validation['audits']),
                  'dmc_calls':sum(r['dmc_calls'] for r in fit['audits']+validation['audits'])},
        'validation_audit':audit,'independent_replay_completed':False}
    _write_json(root/'collection_pending_replay.json',result)
    return result
