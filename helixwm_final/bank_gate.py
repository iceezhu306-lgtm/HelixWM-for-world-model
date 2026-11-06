"""Independent-bank empirical gates. These are NOT closed-loop certificates.

Only a file-backed, numerically recomputed bundle can create VerifiedBankGate.
Thresholds/support use FIT seeds only. HELDOUT never changes them. Bank samples
must never train the author model or auxiliary heads. Novel plans have no bank
counterfactual outcomes; support merely rejects empirically out-of-domain inputs.
"""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
import shutil
import os
import numpy as np

from . import deployment_calibration as numeric
from .belief import canonical_json, accumulate_risk
from .planning import HORIZONS, rollout_statistics, uncertainty_from_sufficient

SCHEMA='helixwm_independent_bank_gate_v3'
MEASUREMENT_PROTOCOL='independent_action_aligned_bank_v3'
SCOPE='off_policy_empirical_not_closed_loop_certificate'
FIT_SEEDS=numeric.FIT_SEEDS
VALIDATION_SEEDS=numeric.VALIDATION_SEEDS
GAPS=numeric.GAPS
ARRAY_FIELDS=numeric.ARRAY_FIELDS+('support_query',)
PLAN_ARRAY_FIELDS=numeric.PLAN_ARRAY_FIELDS+('support_queries',)
SUPPORT_RULE={'features':'mean_of_three_auxiliary_pre_action_features16_plus_executed_action6',
    'standardization':'FIT_mean_std_floor_1e-6','radius':'FIT_Q95_higher_third_distinct_other_seed_nearest_distance',
    'minimum_distinct_fit_seeds':3,'all_plan_steps_and_particles_required':True,
    'claims_dynamics_error_bound':False}
_TOKEN=object()


def digest(value):return hashlib.sha256(canonical_json(value)).hexdigest()


def context_valid(context):
    if not isinstance(context,dict) or set(context)!=set(numeric.CONTEXT_FIELDS):raise ValueError('Bank context fields differ')
    if any(not numeric._sha(context[k]) for k in numeric.SHA_FIELDS):raise ValueError('Current byte identities required')
    if type(context['model_version']) is not int or context['model_version']<0:raise ValueError('Invalid current model version')
    if context['measurement_kind'] not in ('synthetic_test_fixture','actual_gpu_independent_bank'):
        raise ValueError('Independent-bank provenance required; closed-loop labels are not interchangeable')
    return context


class _FrozenDict(dict):
    def _immutable(self,*args,**kwargs):raise TypeError('Verified bank content is immutable')
    __setitem__=__delitem__=clear=pop=popitem=setdefault=update=__ior__=_immutable
    def __deepcopy__(self,memo):return self
    def __reduce__(self):return (_FrozenDict,(dict(self),))


def _freeze(value):
    if isinstance(value,dict):return _FrozenDict({k:_freeze(v) for k,v in value.items()})
    if isinstance(value,(tuple,list)):return tuple(_freeze(v) for v in value)
    if value is None or type(value) in (str,bool,int,float):return value
    raise TypeError('Unexpected mutable content inside verified bank evidence')


class VerifiedBankGate(dict):
    def __init__(self,value,*,_token=None):
        if _token is not _TOKEN:raise ValueError('Only verified file loading issues a bank gate')
        super().__init__({k:_freeze(v) for k,v in value.items()})
        self._seal=digest(dict(self));self._bound_items=tuple(self.items())
    def __deepcopy__(self,memo):
        verify_bank_gate(self,self['context'],require_active=False)
        return VerifiedBankGate(copy.deepcopy(dict(self),memo),_token=_TOKEN)
    def __reduce_ex__(self,protocol):
        # A checkpoint is evidence, never an executable capability. Restoration
        # must reload and reverify the external raw-file manifest.
        return (dict,(dict(self),))


def verify_bank_gate(handle,context,*,require_active=True):
    context_valid(context)
    # Nested evidence is immutable. Identity comparisons detect replacement at
    # the root without serializing the entire support library on every action.
    if (not isinstance(handle,VerifiedBankGate) or len(handle)!=len(handle._bound_items)
            or any(k not in handle or handle[k] is not v for k,v in handle._bound_items)):
        raise ValueError('Bank handle is forged or modified')
    if handle['context']!=context:raise ValueError('Bank source/model/heads/version/protocol/data binding differs')
    if require_active and not handle['active']:raise ValueError('Current empirical bank gate is inactive')
    return True


def support_query(auxiliary_state,actions):
    state=np.asarray(auxiliary_state,np.float64);a=np.asarray(actions,np.float64)
    if state.ndim!=3 or state.shape[0]!=3 or state.shape[2]!=16 or a.shape!=(state.shape[1],6):
        raise ValueError('Support needs pre-action independent-head state[3,K,16] and action[K,6]')
    if not np.isfinite(state).all() or not np.isfinite(a).all() or np.any(np.abs(a)>1):raise ValueError('Invalid support query')
    return np.concatenate((state.mean(0),a),axis=1)


def _seed_distances(query,table,exclude_seed=None):
    points=np.asarray(table['points'],np.float64);labels=np.asarray(table['seeds'])
    scale=np.asarray(table['scale']);mean=np.asarray(table['mean'])
    q=(np.asarray(query,np.float64)-mean)/scale;p=(points-mean)/scale
    distances=[]
    for seed in FIT_SEEDS:
        if seed==exclude_seed:continue
        subset=p[labels==seed]
        if not len(subset):raise ValueError('Support is missing a FIT seed')
        # Bounded memory; callers use at most 8 candidates or one FIT seed.
        distances.append(np.sqrt(np.square(q[:,None,:]-subset[None,:,:]).sum(-1)).min(1))
    return np.stack(distances,axis=1)


def fit_support(rows):
    points=np.concatenate([r['support_query'] for r in rows]);labels=np.concatenate([
        np.full(len(r['support_query']),r['seed'],np.int32) for r in rows])
    table={'rule':copy.deepcopy(SUPPORT_RULE),'points':points.tolist(),'seeds':labels.tolist(),
           'mean':points.mean(0).tolist(),'scale':np.maximum(points.std(0),1e-6).tolist()}
    distances=[]
    for seed in FIT_SEEDS:
        ds=_seed_distances(points[labels==seed],table,exclude_seed=seed)
        distances.extend(np.sort(ds,axis=1)[:,2].tolist())
    table['radius']=numeric._q(distances,.95)
    return table


def plan_support(gate,gap,queries):
    if type(gap) is not int or gap not in GAPS:raise ValueError('Unsupported gap')
    q=np.asarray(queries,np.float64)
    if q.ndim!=3 or q.shape[0] not in HORIZONS or q.shape[2]!=22 or not np.isfinite(q).all():
        raise ValueError('Every horizon/candidate support query must be supplied')
    table=gate['support_by_gap'][str(gap)]
    ds=_seed_distances(q.reshape(-1,22),table)
    count=(ds<=table['radius']).sum(1).reshape(q.shape[:2])
    return {'supported':bool(np.all(count>=3)),'distinct_fit_seed_count':count.tolist(),
            'third_nearest_seed_distance':np.sort(ds,axis=1)[:,2].reshape(q.shape[:2]).tolist(),
            'radius':table['radius'],'scope':SCOPE,'closed_loop_certified':False}


def rows_digest(rows):
    records=[]
    for row in sorted(rows,key=lambda r:(r['seed'],r['gap'])):
        item={k:v for k,v in row.items() if k not in ARRAY_FIELDS and k!='plans'}
        item['arrays']={k:numeric._array_hash(row[k]) for k in ARRAY_FIELDS}
        item['plans']=[]
        for plan in row['plans']:
            p={k:v for k,v in plan.items() if k not in PLAN_ARRAY_FIELDS}
            p['arrays']={k:numeric._array_hash(plan[k]) for k in PLAN_ARRAY_FIELDS};item['plans'].append(p)
        records.append(digest(item))
    return digest(records)


def validate_rows(rows,context,phase,candidate_sha256=None):
    context_valid(context);seeds=FIT_SEEDS if phase=='fit' else VALIDATION_SEEDS
    if len(rows)!=60:raise ValueError('Exactly ten independent seeds by six virtual gaps required')
    seen=set();metrics={}
    for r in rows:
        s,g=r['seed'],r['gap'];n=g+8
        if type(s) is not int or s not in seeds or type(g) is not int or g not in GAPS or (s,g) in seen:
            raise ValueError('Wrong/duplicate seed or gap; threshold partitions must be disjoint')
        seen.add((s,g))
        if (r['context']!=context or r['phase']!=phase or r['measurement_protocol']!=MEASUREMENT_PROTOCOL
                or r['candidate_sha256']!=candidate_sha256 or r['execution_mode']!='forced_recorded_actions'
                or type(r['model_updates']) is not int or r['model_updates']!=0
                or r['hidden_target_to_controller'] is not False or r['hidden_target_to_learner'] is not False):
            raise ValueError('Mislabelled off-policy evidence or hidden/learner data leakage')
        if r.get('planning_actions_are_counterfactual') is not True or r.get('counterfactual_actual_outcomes_available') is not False:
            raise ValueError('Novel plan predictions must not claim recorded ground truth')
        a=numeric._array
        if not np.array_equal(a(r['timestamp'],np.int32,(n,),'timestamp'),np.arange(17,17+n,dtype=np.int32)):
            raise ValueError('Wrong bank timeline')
        if not np.array_equal(a(r['availability'],np.bool_,(n,),'availability'),np.arange(n)>=g):raise ValueError('Wrong virtual mask')
        a(r['actions'],np.float64,(n,6),'recorded actions',-1,1);a(r['previous_action'],np.float64,(6,),'previous',-1,1)
        a(r['actual_rewards'],np.float64,(n,),'actual bank rewards',0,2)
        a(r['support_query'],np.float64,(n,22),'support query')
        if not np.array_equal(r['support_query'][:,-6:],r['actions']):raise ValueError('Support action is not recorded action')
        for k in ('predictive_entropy','expected_entropy','fusion_lambda'):a(r[k],np.float64,(n,),k,0,1)
        for k in ('fusion_kl','cumulative_risk','epistemic_disagreement'):a(r[k],np.float64,(n,),k,0)
        for k in ('fallback','risk_reset'):a(r[k],np.bool_,(n,),k)
        a(r['energy'],np.float64,(8,8),'visible recovery error',0)
        a(r['temporal_prediction_error'],np.float64,(8,),'visible temporal error',0)
        w=a(r['candidate_weights'],np.float64,(n,8),'weights',0,1)
        if not np.allclose(w.sum(1),1,rtol=0,atol=2e-12):raise ValueError('Unnormalized weights')
        u,j=zip(*(uncertainty_from_sufficient(x,y) for x,y in zip(r['predictive_entropy'],r['expected_entropy'])))
        risk=0.
        for t in range(n):
            if t<g:risk=accumulate_risk(risk,u[t])
            if not np.isclose(risk,r['cumulative_risk'][t],rtol=0,atol=2e-12):raise ValueError('Bank risk operator differs')
            if r['risk_reset'][t]:
                if t<g:raise ValueError('Missing observation cannot reset risk')
                risk=0.
        stats=[];ids=set()
        for plan in r['plans']:
            h=plan['horizon']
            a(plan['support_queries'],np.float64,(h,8,22),'plan support queries')
            if not np.array_equal(plan['support_queries'][...,-6:],plan['actions']):raise ValueError('Plan support must bind its own proposed actions')
            numeric._validate_plan({k:v for k,v in plan.items() if k!='support_queries'},r,ids)
            pu,pj=zip(*(uncertainty_from_sufficient(x,y) for x,y in zip(plan['predictive_entropy'],plan['expected_entropy'])))
            stats.append(rollout_statistics(plan['rewards'],plan['continuations'],np.array(pu),np.array(pj),
                plan['actions'],plan['previous_action'],plan['weights'],plan['epistemic_disagreement']))
        if {p['horizon'] for p in stats}!=set(HORIZONS):raise ValueError('All planning horizons required')
        metrics[(s,g)]={'uncertainty':np.array(u),'disagreement':np.array(j),'ess':1./np.square(w).sum(1)/8,'plans':stats}
    return metrics


def fit_candidate(rows,context):
    m=validate_rows(rows,context,'fit');thresholds={};planning={};support={}
    for g in GAPS:
        chosen=[r for r in rows if r['gap']==g];ms=[m[(r['seed'],g)] for r in chosen]
        energy=np.concatenate([r['energy'] for r in chosen]);max_e=numeric._q(energy.min(1))
        temp=max(1e-6,numeric._q((energy-energy.min(1,keepdims=True)).mean(1)))
        ess,_=numeric.fit_reweighted_ess(energy,max_e,temp)
        thresholds[str(g)]={'max_energy':max_e,'temperature':temp,'min_ess_ratio':numeric._q(ess,.05),
            'max_uncertainty':numeric._q(np.concatenate([x['uncertainty'] for x in ms])),
            'max_disagreement':numeric._q(np.concatenate([x['disagreement'] for x in ms])),
            'max_epistemic_disagreement':numeric._q(np.concatenate([r['epistemic_disagreement'] for r in chosen])),
            'max_temporal_prediction_error':numeric._q(np.concatenate([r['temporal_prediction_error'] for r in chosen])),
            'max_cumulative_risk':numeric._q(np.concatenate([r['cumulative_risk'] for r in chosen])),
            **numeric.STRUCTURAL_CAPS,'planning_horizon':5,'reward_weight':1.,'uncertainty_weight':.25,
            'termination_weight':1.,'slew_weight':.1,'disagreement_weight':.25}
        planning[str(g)]={str(h):{key:numeric._q([p[key] for x in ms for p in x['plans'] if p['horizon']==h])
            for key in ('uncertainty','disagreement','termination_risk','epistemic_disagreement')} for h in HORIZONS}
        thresholds[str(g)]['max_termination_risk']=planning[str(g)]['5']['termination_risk']
        support[str(g)]=fit_support(chosen)
    candidate={'schema':SCHEMA,'context':copy.deepcopy(context),'thresholds_by_gap':thresholds,
        'planning_limits_by_gap_horizon':planning,'support_by_gap':support,'fit_records_sha256':rows_digest(rows),
        'active':False,'frozen':True,'evidence_scope':SCOPE,'physical_safety_proven':False,
        'fit_seeds':list(FIT_SEEDS),'heldout_seeds':list(VALIDATION_SEEDS),
        'coverage_revision':copy.deepcopy(numeric.COVERAGE_REVISION),'fit_attempt':1}
    candidate['sha256']=digest(candidate);return candidate


def validate_fitted_candidate(candidate,context):
    context_valid(context)
    if (not isinstance(candidate,dict) or candidate.get('schema')!=SCHEMA or candidate.get('context')!=context
            or candidate.get('sha256')!=digest({k:v for k,v in candidate.items() if k!='sha256'})
            or candidate.get('active') is not False or candidate.get('fit_seeds')!=list(FIT_SEEDS)
            or candidate.get('heldout_seeds')!=list(VALIDATION_SEEDS)):
        raise ValueError('Not a bound inactive bank candidate')
    from .belief import validate_thresholds
    validate_thresholds(candidate['thresholds_by_gap'])
    for g in GAPS:
        table=candidate['support_by_gap'][str(g)]
        if table['rule']!=SUPPORT_RULE or set(table['seeds'])!=set(FIT_SEEDS):raise ValueError('Support library leaked outside FIT')
        if set(candidate['planning_limits_by_gap_horizon'][str(g)])!={'1','3','5'}:raise ValueError('Incomplete horizon set')
    return True


def evaluate_candidate(candidate,rows,context):
    validate_fitted_candidate(candidate,context);m=validate_rows(rows,context,'validation',candidate['sha256'])
    cells={};groups={}
    for r in rows:
        s,g=r['seed'],r['gap'];v=m[(s,g)];t=candidate['thresholds_by_gap'][str(g)]
        values={'energy':r['energy'].min(1),'uncertainty':v['uncertainty'],'disagreement':v['disagreement'],
            'epistemic_disagreement':r['epistemic_disagreement'],'temporal_prediction_error':r['temporal_prediction_error'],
            'cumulative_risk':r['cumulative_risk']}
        coverage={k:float(np.mean(x<=t['max_'+k])) for k,x in values.items()}
        coverage['ess']=float(np.mean(v['ess']>=t['min_ess_ratio']))
        for h in HORIZONS:
            for key,limit in candidate['planning_limits_by_gap_horizon'][str(g)][str(h)].items():
                coverage[f'plan{h}_{key}']=float(np.mean([p[key]<=limit for p in v['plans'] if p['horizon']==h]))
        invariants={'fusion_caps':bool(np.all(r['fusion_lambda']<=.25) and np.all(r['fusion_kl']<=.02))}
        for key,x in values.items():
            # Cumulative-risk reset is conditioned on observed innovation, not
            # on previous risk, so it is not added as a new reset prerequisite.
            if key=='cumulative_risk':continue
            observed=x if len(x)==8 else x[g:]
            invariants['reset_qualified_'+key]=bool(np.all(~r['risk_reset'][g:] | (observed<=t['max_'+key])))
        invariants['reset_qualified_ess']=bool(np.all(~r['risk_reset'][g:] | (v['ess'][g:]>=t['min_ess_ratio'])))
        support=plan_support(candidate,g,r['support_query'][None,:,:])
        cells[f'{s}:{g}']={'coverage':coverage,'invariants':invariants,
            'recorded_path_support_fraction':float(np.mean(np.array(support['distinct_fit_seed_count'])>=3))}
    for g in GAPS:
        selected=[cells[f'{s}:{g}'] for s in VALIDATION_SEEDS]
        means={k:float(np.mean([x['coverage'][k] for x in selected])) for k in selected[0]['coverage']}
        failures=[k for k,v in means.items() if v<(.90 if k=='ess' else .95)]
        structural=all(all(x['invariants'].values()) for x in selected)
        groups[str(g)]={'coverage':means,'failed_metrics':failures,'structural_invariants_passed':structural,
                        'passed':not failures and structural}
    return {'passed':all(x['passed'] for x in groups.values()),'by_gap':groups,'per_cell':cells,
            'heldout_records_sha256':rows_digest(rows),'thresholds_changed_using_heldout':False,
            'scope':SCOPE,'novel_actions_have_observed_counterfactual_outcomes':False}


def write_window(row,root):
    root=Path(root);name=f"{row['phase']}_seed{row['seed']}_gap{row['gap']}"
    arrays={k:row[k] for k in ARRAY_FIELDS};meta={k:v for k,v in row.items() if k not in ARRAY_FIELDS and k!='plans'}
    meta['plans']=[]
    for i,p in enumerate(row['plans']):
        meta['plans'].append({k:v for k,v in p.items() if k not in PLAN_ARRAY_FIELDS})
        arrays.update({f'plan_{i}_{k}':p[k] for k in PLAN_ARRAY_FIELDS})
    with (root/(name+'.npz')).open('xb') as f:np.savez_compressed(f,**arrays)
    with (root/(name+'.json')).open('xb') as f:f.write(canonical_json(meta))
    return {k:numeric.file_descriptor(root/(name+suffix),root) for k,suffix in (('metadata','.json'),('arrays','.npz'))}


def load_window(root,descriptor):
    meta=numeric._read_json(numeric._verified_file(root,descriptor['metadata']))
    with np.load(numeric._verified_file(root,descriptor['arrays']),allow_pickle=False) as data:
        expected=set(ARRAY_FIELDS)|{f'plan_{i}_{k}' for i in range(len(meta['plans'])) for k in PLAN_ARRAY_FIELDS}
        if set(data.files)!=expected:raise ValueError('Raw bank arrays differ')
        for k in ARRAY_FIELDS:meta[k]=data[k].copy()
        for i,p in enumerate(meta['plans']):
            for k in PLAN_ARRAY_FIELDS:p[k]=data[f'plan_{i}_{k}'].copy()
    return meta


def _load_measurement(root,expected_sha,context,*,bank_directory=None):
    root=Path(root);path=root/'bank_measurement.json'
    if numeric.file_descriptor(path,root)['sha256']!=expected_sha:raise ValueError('External measurement SHA differs')
    record=numeric._read_json(path)
    if record['context']!=context or record['schema']!=SCHEMA:raise ValueError('Measurement context differs')
    if record['bank_origin']['manifest_sha256']!=context['bank_manifest_sha256']:raise ValueError('Shared bank origin differs')
    original_bank=None
    if context['measurement_kind']=='actual_gpu_independent_bank':
        from .calibration_bank import load_bank
        origin=record['bank_origin']
        original_bank,manifest=load_bank(bank_directory or origin['directory'],origin['manifest_sha256'],origin['protocol_sha256'],
            environment_lock_sha256=origin['environment_lock_sha256'],source_sha256=origin['source_sha256'])
        if origin['files']!=manifest['files']:raise ValueError('Bank origin input file list differs')
    if record['environment_actions']!=0 or record['author_updates']!=0 or record['auxiliary_updates']!=0:raise ValueError('Offline calibration acquired extra data or trained')
    before,after=record['freeze_before'],record['freeze_after']
    if set(before)!=set(numeric.FREEZE_COMPONENTS) or before!=after or any(not numeric._sha(v) for v in before.values()):raise ValueError('Frozen state changed')
    if before['model']!=context['model_sha256'] or before['epistemic_heads']!=context['epistemic_heads_sha256']:raise ValueError('Measured model/heads differ')
    if type(record['process_id']) is not int or record['process_id']<=0:raise ValueError('Measurement process missing')
    rows=[load_window(root,x) for x in record['windows']]
    if original_bank is not None:
        for row in rows:
            source=original_bank[row['seed']];end=16+row['gap']+8
            if (not np.array_equal(row['actions'],source['action'][16:end].astype(np.float64))
                    or not np.array_equal(row['actual_rewards'],source['reward'][16:end])
                    or not np.array_equal(row['previous_action'],source['action'][15].astype(np.float64))):
                raise ValueError('Recorded-action/reward evidence does not align with immutable bank')
    candidate=fit_candidate([r for r in rows if r['phase']=='fit'],context)
    retained=numeric._read_json(numeric._verified_file(root,record['candidate']))
    if candidate!=retained:raise ValueError('FIT recomputation differs from sealed candidate')
    audit=evaluate_candidate(candidate,[r for r in rows if r['phase']=='validation'],context)
    if record['rows_sha256']!=rows_digest(rows):raise ValueError('Raw bank data digest differs')
    if not 0<=record['candidate_sealed_ns']<record['heldout_started_ns']<=record['completed_ns']:raise ValueError('FIT must be sealed before HELDOUT')
    return record,candidate,audit


def seal_measurement(directory,measurement_sha256,context):
    """Per-pin synchronous numeric verification; no model-anchor duplication.

    Independence refers to the bank dataset, not a new process at every pin.
    Actual fresh-process replay remains a preflight/final audit, not an invented
    recurring 2,000-process production requirement.
    """
    root=Path(directory);_load_measurement(root,measurement_sha256,context)
    manifest={'schema':SCHEMA,'context':context,'reference_sha256':measurement_sha256,
              'verification':'raw_array_fit_heldout_recomputation','fresh_process_verified':False}
    with (root/'bank_gate_manifest.json').open('xb') as f:f.write(canonical_json(manifest))
    return numeric.file_descriptor(root/'bank_gate_manifest.json',root)


def finalize_independent(reference,replayed,reference_sha256,replayed_sha256,context):
    """Join two actually executed independent computations; never overwrite."""
    root=Path(reference);other=Path(replayed)
    a,_,_= _load_measurement(root,reference_sha256,context)
    b,_,_= _load_measurement(other,replayed_sha256,context)
    if a['process_id']==b['process_id']:raise ValueError('Independent fresh process required')
    if a['rows_sha256']!=b['rows_sha256'] or a['freeze_before']!=b['freeze_before']:raise ValueError('Independent raw replay or frozen identities differ')
    target=root/'INDEPENDENT_RECOMPUTATION';target.mkdir(exist_ok=False)
    needed={'bank_measurement.json',b['candidate']['path']}
    for item in b['windows']:needed.update(x['path'] for x in item.values())
    for name in needed:
        source=other/name;dest=target/name;dest.parent.mkdir(parents=True,exist_ok=True)
        try:os.link(source,dest)
        except OSError:shutil.copyfile(source,dest)
    if numeric.file_descriptor(target/'bank_measurement.json',target)['sha256']!=replayed_sha256:raise ValueError('Independent evidence copy changed')
    manifest={'schema':SCHEMA,'context':context,'reference_sha256':reference_sha256,
              'independent_directory':target.name,'independent_sha256':replayed_sha256}
    with (root/'independent_recomputation.json').open('xb') as f:f.write(canonical_json(manifest))
    return numeric.file_descriptor(root/'independent_recomputation.json',root)


def load_bank_gate(directory,expected_manifest_sha256,context,*,require_active=True,bank_directory=None):
    context_valid(context);root=Path(directory);path=root/'bank_gate_manifest.json'
    if numeric.file_descriptor(path,root)['sha256']!=expected_manifest_sha256:raise ValueError('External gate manifest SHA differs')
    manifest=numeric._read_json(path)
    if (manifest['schema']!=SCHEMA or manifest['context']!=context
            or manifest['verification']!='raw_array_fit_heldout_recomputation'
            or manifest['fresh_process_verified'] is not False):raise ValueError('Gate identity differs')
    a,candidate,audit=_load_measurement(root,manifest['reference_sha256'],context,bank_directory=bank_directory)
    actual=context['measurement_kind']=='actual_gpu_independent_bank'
    if actual and (a.get('device')!='cuda' or not a.get('gpu')):raise ValueError('Actual GPU measurement missing')
    result={**candidate,'active':bool(actual and audit['passed']),'validation':audit,
            'artifact_sha256':expected_manifest_sha256,'independent_recomputation_verified':False,
            'raw_array_numeric_recomputation_verified':True,
            'bank_origin':a['bank_origin'],'same_hardware_numeric_reproducibility_only':True}
    handle=VerifiedBankGate(result,_token=_TOKEN)
    verify_bank_gate(handle,context,require_active=require_active);return handle
