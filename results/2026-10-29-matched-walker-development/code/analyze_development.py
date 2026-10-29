"""Reproducible descriptive analysis of fixed, archived development evaluations.

No model, checkpoint, replay, simulator, network, or training imports/access.
Only the frozen 20k..1M snapshot below is read. These five development
environment seeds do not replace the registered 40-seed final evaluation.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parent
WORK_ROOT = ROOT.parents[1] if ROOT.name == 'code' else ROOT
os.environ.setdefault('MPLCONFIGDIR', str(WORK_ROOT / 'cache' / 'matplotlib'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DEFAULT_SOURCE = WORK_ROOT / 'evidence_20261029T120145Z'
SNAPSHOTS = {'D': 'runs/D', 'H': 'runs/H'}
STEPS = list(range(20000, 1000001, 20000))
SEEDS = list(range(50000, 50005))
NAMES = {'D': 'DreamerV3 (D)', 'H': 'HelixWM V2 (H)'}
COLORS = {'D': '#2878B5', 'H': '#D95319'}
CONTROLLER = {'D': 'black_frame_native', 'H': 'helix_v2'}
LATEST = 1000000


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def mask_sha(mask):
    header = json.dumps({'dtype': mask.dtype.str, 'shape': list(mask.shape)},
                        sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf8')
    return sha(header + np.ascontiguousarray(mask).tobytes())


def write_bytes(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        require(path.read_bytes() == data, f'Refusing to replace different existing file: {path}')
    else:
        with path.open('xb') as stream:
            stream.write(data)


def write_json(path, value):
    write_bytes(path, (json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n').encode('utf8'))


def write_csv(path, rows):
    buf = io.StringIO(newline='')
    fields = list(dict.fromkeys(k for row in rows for k in row))
    writer = csv.DictWriter(buf, fieldnames=fields, lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    write_bytes(path, buf.getvalue().encode('utf8'))


def stats(values):
    a = np.asarray(values, dtype=np.float64)
    return {'n': len(a), 'mean': float(a.mean()), 'sample_sd': float(a.std(ddof=1)),
            'minimum': float(a.min()), 'maximum': float(a.max())}


def exists_run(flags, length):
    run = 0
    for v in flags:
        run = run+1 if v else 0
        if run >= length:
            return True
    return False


def recovery(u):
    pre = float(u[100:150].mean())
    if pre < .75:
        return {'eligible': False, 'time_actions': None, 'status': 'ineligible'}
    consecutive = 0
    for elapsed in range(10, 101):
        value = float(u[170+elapsed-10:170+elapsed].mean())
        consecutive = consecutive+1 if value >= .9*pre else 0
        if consecutive >= 10:
            return {'eligible': True, 'time_actions': elapsed, 'status': 'observed'}
    return {'eligible': True, 'time_actions': None, 'status': 'right_censored_at_100'}


def save_figure(fig, destination, stem):
    outputs = []
    for ext in ('png', 'pdf'):
        buffer = io.BytesIO()
        metadata = {'Creator': 'HelixWM development comparison analysis'}
        if ext == 'pdf':
            metadata.update(CreationDate=None, ModDate=None)
        fig.savefig(buffer, format=ext, dpi=180, facecolor='white', metadata=metadata)
        write_bytes(destination / f'{stem}.{ext}', buffer.getvalue())
        outputs.append(f'figures/{stem}.{ext}')
    plt.close(fig)
    return outputs


def plots(rounds, paired, traces, target):
    plt.rcParams.update({'font.size': 10, 'axes.titlesize': 12, 'axes.labelsize': 10,
                         'legend.fontsize': 9, 'axes.spines.top': False,
                         'axes.spines.right': False, 'pdf.fonttype': 42})
    outputs = []
    fig, axes = plt.subplots(1, 2, figsize=(12.3, 5.3))
    for axis, condition, title in zip(axes, ('clean', 'gap20'),
             ('Clean observations', '20 missing observations: D native vs H Helix')):
        for arm in ('D', 'H'):
            rows = [r for r in rounds if r['arm'] == arm and r['condition'] == condition]
            x = np.array([r['training_actions'] for r in rows])/1000
            m = np.array([r['mean_episode_return'] for r in rows])
            sd = np.array([r['sample_sd_episode_return'] for r in rows])
            axis.plot(x, m, color=COLORS[arm], linewidth=2, label=NAMES[arm])
            axis.fill_between(x, m-sd, m+sd, color=COLORS[arm], alpha=.14)
        axis.set(xlabel='Training action decisions (thousands)', ylabel='Episode return (maximum 1,000)',
                 title=title, xlim=(20, 1000), ylim=(0, 1050))
        axis.grid(alpha=.2)
        axis.legend(loc='lower right')
    fig.suptitle('Matched Walker development learning curves', fontsize=15, y=.98)
    fig.text(.5, .035, '50 common rounds; five reused environment seeds (50000–50004). Shading: mean ± 1 sample SD, not confidence intervals.\nOne training seed per model. Fixed final training checkpoint: 1,000,000 actions; held-out final evaluation is excluded.',
             ha='center', va='bottom', fontsize=9, color='#444444')
    fig.subplots_adjust(bottom=.22, top=.85, left=.065, right=.98, wspace=.24)
    outputs += save_figure(fig, target, 'development_learning_curves')

    fig, axes = plt.subplots(1, 3, figsize=(13.3, 5.4))
    for axis, field, title, label in zip(axes,
            ('clean_return', 'gap20_return', 'clean_minus_gap20'),
            ('Clean: both clean_native', 'Gap 20: paired environment seeds', 'Clean-to-gap return loss'),
            ('Episode return', 'Episode return', 'Clean return minus gap-20 return')):
        for row in paired:
            values = [row[f'D_{field}'], row[f'H_{field}']]
            axis.plot([0, 1], values, color='#B6B6B6', alpha=.7, linewidth=1)
            for x, arm in enumerate(('D', 'H')):
                axis.scatter(x, values[x], color=COLORS[arm], s=36, zorder=3)
        means = [np.mean([r[f'{a}_{field}'] for r in paired]) for a in ('D', 'H')]
        axis.scatter([0, 1], means, marker='_', color='black', s=600, linewidth=2.5, zorder=4)
        axis.set_xticks([0, 1], ['DreamerV3 D', 'HelixWM H'])
        axis.set(xlim=(-.3, 1.65), title=title, ylabel=label)
        axis.grid(axis='y', alpha=.2)
        if field == 'clean_minus_gap20':
            axis.axhline(0, color='#888888', linewidth=.8)
    fig.suptitle('1,000,000-action development snapshot: paired outcomes', fontsize=15, y=.98)
    fig.text(.5, .035, 'Each line links one environment seed; black ticks show five-seed means. Single training seed per model.\nThe loss panel compares each model with its own clean trajectory. Descriptive results; no confirmatory significance claim.',
             ha='center', va='bottom', fontsize=9, color='#444444')
    fig.subplots_adjust(bottom=.22, top=.84, left=.055, right=.985, wspace=.31)
    outputs += save_figure(fig, target, 'latest_paired_returns')

    fig, axes = plt.subplots(1, 2, figsize=(12.3, 5.4))
    x = np.arange(100, 270)
    for axis, arm in zip(axes, ('D', 'H')):
        clean = np.stack([traces[(arm, s, 'clean')][100:270] for s in SEEDS])
        fault = np.stack([traces[(arm, s, 'gap20')][100:270] for s in SEEDS])
        for row in fault:
            axis.plot(x, row, color=COLORS[arm], linewidth=.6, alpha=.2)
        axis.plot(x, clean.mean(axis=0), color='#626262', linestyle='--', linewidth=1.5,
                  label='Own clean mean')
        axis.plot(x, fault.mean(axis=0), color=COLORS[arm], linewidth=2, label='Gap-20 mean')
        axis.axvspan(150, 170, color='#BA75C6', alpha=.16, label='Observation unavailable')
        axis.axvline(170, color='#777777', linestyle=':', linewidth=1)
        axis.set(xlabel='Action decision index t (zero based)', ylabel='Reward per DMC call (0–1)',
                 title=NAMES[arm], xlim=(100, 269), ylim=(-.02, 1.04))
        axis.grid(alpha=.18)
        axis.legend(loc='lower left')
    fig.suptitle('1,000,000 actions: reward through a 20-observation gap', fontsize=15, y=.98)
    fig.text(.5, .035, 'Unsmoothed reward / actual DMC calls. Thin lines: five gap-condition episodes. Missing x150…x169 affects a150…a169.\nSeparate model trajectories; Helix candidate repair is capped at 5 missing observations, with prior fallback for longer gaps.',
             ha='center', va='bottom', fontsize=9, color='#444444')
    fig.subplots_adjust(bottom=.22, top=.85, left=.065, right=.98, wspace=.24)
    outputs += save_figure(fig, target, 'latest_fault_recovery_trajectories')
    return outputs


def run(source, destination):
    data = destination/'data'
    figures = destination/'figures'
    data.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    source_files, episodes, rounds, round_checks, traces = {}, [], [], [], {}
    raw_latest = {}

    def read(path):
        contents = path.read_bytes()
        rel = path.relative_to(source).as_posix()
        source_files[rel] = {'path': rel, 'bytes': len(contents), 'sha256': sha(contents)}
        return contents

    protocol_path = source/'runs/D/protocol.json'
    protocol = json.loads(read(protocol_path))
    write_bytes(data/'registered_protocol.json', protocol_path.read_bytes())
    require(protocol['environment']['action_repeat'] == 2, 'Unexpected repeat')
    for arm, snapshot in SNAPSHOTS.items():
        evalroot = source/snapshot/'EVAL'
        require(evalroot.is_dir(), f'Fixed snapshot missing: {snapshot}')
        require(read(source/snapshot/'train.exit').decode().strip() == '0', 'Training did not exit zero')
        completion = json.loads(read(source/snapshot/'completion.json'))
        for name, count in [('agent_steps', 1000000), ('env_steps', 2000000), ('episodes', 2000), ('update_steps', 248750)]:
            require(completion[name] == count, 'Training completion budget differs')
        jsonl = read(source/snapshot/'eval.jsonl')
        jsonl_rows = [json.loads(line) for line in jsonl.splitlines() if line.strip()]
        require(len(jsonl_rows) == 500, 'Expected 500 development episodes per model')
        write_bytes(data/f'{arm}_development_eval.jsonl', jsonl)
        for step in STEPS:
            folder = evalroot/f'{step:09d}'
            report_bytes = read(folder/'frozen_report.json')
            ledger_bytes = read(folder/'interaction_ledger.json')
            frozen, ledger = json.loads(report_bytes), json.loads(ledger_bytes)
            require(frozen['passed'] is True and frozen['before_hash'] == frozen['after_hash'], 'Freeze failed')
            require(frozen['training_actions'] == step and frozen['episodes'] == 10 and frozen['training_updates'] == 0,
                    'Wrong frozen counts')
            require(frozen['actual_evaluation_actions'] == 5000 and frozen['actual_evaluation_dmc_calls'] == 10000,
                    'Frozen evaluation budget differs')
            require(ledger == {'acknowledged_actions': 5000, 'acknowledged_dmc_calls': 10000,
                               'passed': True, 'stage': 'complete'}, 'Ledger not complete')
            host = json.loads(read(folder/'host_zero_update.json'))
            require(host['passed'] and host['before_sha256'] == host['after_sha256'], 'Host RNG zero-update failure')
            expected_ids = {(c, g, s) for c, g in [('clean_native', 0), (CONTROLLER[arm], 20)] for s in SEEDS}
            actual_ids = {(r['controller'], r['gap'], r['seed']) for r in frozen['rows']}
            require(len(frozen['rows']) == 10 and actual_ids == expected_ids, 'Seed/condition matrix differs')
            frozen_index = {(r['controller'], r['gap'], r['seed']): r for r in frozen['rows']}
            round_checks.append({'arm': arm, 'training_actions': step, 'passed': True,
                                 'freeze_hash_equal': True, 'ledger_complete': True,
                                 'host_zero_update_verified': True,
                                 'episodes': 10, 'actions': 5000, 'dmc_calls': 10000})
            for condition, controller, gap in [('clean', 'clean_native', 0), ('gap20', CONTROLLER[arm], 20)]:
                selected = []
                for seed in SEEDS:
                    name = f'{controller}_seed{seed}_gap{gap}'
                    row_bytes, trace_bytes = read(folder/(name+'.json')), read(folder/(name+'.npz'))
                    row = json.loads(row_bytes)
                    require((row['controller'], row['gap'], row['seed']) == (controller, gap, seed), 'Case identity differs')
                    require(row['actions'] == 500 and row['dmc_calls'] == 1000, 'Case budget differs')
                    for key in ('controller', 'gap', 'seed', 'actions', 'dmc_calls', 'episode_return', 'availability_sha256'):
                        require(row[key] == frozen_index[(controller, gap, seed)][key], 'Frozen/case record differs')
                    with np.load(io.BytesIO(trace_bytes), allow_pickle=False) as saved:
                        trace = {k: saved[k] for k in saved.files}
                    require(set(trace) == {'action', 'reward', 'dmc_calls', 'availability_outcome', 'availability_before_action'},
                            'Unexpected trace schema')
                    require(trace['action'].shape == (500, 6) and np.isfinite(trace['action']).all() and
                            (np.abs(trace['action']) <= 1).all(), 'Invalid actions')
                    require(trace['reward'].shape == (500,) and np.isfinite(trace['reward']).all() and
                            ((trace['reward'] >= 0) & (trace['reward'] <= 2)).all(), 'Invalid rewards')
                    require(trace['dmc_calls'].shape == (500,) and (trace['dmc_calls'] == 2).all(), 'Invalid calls')
                    require(abs(float(trace['reward'].sum()) - row['episode_return']) < 1e-8, 'Reward sum differs')
                    mask = np.ones(501, bool)
                    if gap:
                        mask[150:150+gap] = False
                    require(row['availability_sha256'] == mask_sha(mask), 'Mask hash differs')
                    require(np.array_equal(trace['availability_before_action'], mask[:-1]) and
                            np.array_equal(trace['availability_outcome'], mask[1:]), 'Mask/action alignment differs')
                    u = trace['reward']/trace['dmc_calls']
                    out = {'arm': arm, 'condition': condition, 'controller': controller,
                           'training_seed': 0, 'training_actions': step, 'environment_seed': seed,
                           'gap': gap, 'episode_return': row['episode_return'],
                           'actions': 500, 'dmc_calls': 1000,
                           'availability_sha256': row['availability_sha256'], 'frozen_verified': True,
                           'trace_verified': True}
                    episodes.append(out)
                    selected.append(row['episode_return'])
                    if step == LATEST:
                        traces[(arm, seed, condition)] = u
                        raw_latest[f'{arm}/{name}.json'] = row_bytes
                        raw_latest[f'{arm}/{name}.npz'] = trace_bytes
                s = stats(selected)
                rounds.append({'arm': arm, 'condition': condition, 'controller': controller,
                               'training_actions': step, 'n_environment_seeds': s['n'],
                               'mean_episode_return': s['mean'], 'sample_sd_episode_return': s['sample_sd'],
                               'minimum_episode_return': s['minimum'], 'maximum_episode_return': s['maximum']})

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(raw_latest):
            info = zipfile.ZipInfo(name, date_time=(2026, 10, 29, 12, 1, 45))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, raw_latest[name])
    write_bytes(data/'latest_traces.zip', buffer.getvalue())
    write_json(data/'latest_traces_manifest.json', {'archive_sha256': sha(buffer.getvalue()),
        'training_actions': LATEST, 'cases': 20,
        'files': {name: {'bytes': len(contents), 'sha256': sha(contents)} for name, contents in sorted(raw_latest.items())}})

    paired = []
    for seed in SEEDS:
        row = {'seed': seed, 'training_actions': LATEST}
        for arm in ('D', 'H'):
            clean, fault = traces[(arm, seed, 'clean')], traces[(arm, seed, 'gap20')]
            row[f'{arm}_clean_return'] = float(clean.sum()*2)
            row[f'{arm}_gap20_return'] = float(fault.sum()*2)
            row[f'{arm}_clean_minus_gap20'] = row[f'{arm}_clean_return']-row[f'{arm}_gap20_return']
            row[f'{arm}_gap_window_score'] = float(fault[150:270].mean())
            row[f'{arm}_gap_window_own_clean_deficit'] = float((clean[150:270]-fault[150:270]).mean())
            row[f'{arm}_functional_failure'] = exists_run(fault[150:270] < .2, 20)
            rec = recovery(fault)
            for k, value in rec.items():
                row[f'{arm}_recovery_{k}'] = value
        for metric in ('clean_return', 'gap20_return', 'clean_minus_gap20', 'gap_window_score', 'gap_window_own_clean_deficit'):
            row[f'H_minus_D_{metric}'] = row[f'H_{metric}']-row[f'D_{metric}']
        paired.append(row)
    latest = {}
    for arm in ('D', 'H'):
        latest[arm] = {m: stats([r[f'{arm}_{m}'] for r in paired]) for m in
                      ('clean_return', 'gap20_return', 'clean_minus_gap20', 'gap_window_score', 'gap_window_own_clean_deficit')}
        latest[arm]['functional_failures_gap20_window'] = sum(r[f'{arm}_functional_failure'] for r in paired)
        latest[arm]['recovery_eligible'] = sum(r[f'{arm}_recovery_eligible'] for r in paired)
        latest[arm]['recovered_by_100_actions'] = sum(r[f'{arm}_recovery_status'] == 'observed' for r in paired)
    differences = {m: stats([r[f'H_minus_D_{m}'] for r in paired]) for m in
                   ('clean_return', 'gap20_return', 'clean_minus_gap20', 'gap_window_score', 'gap_window_own_clean_deficit')}
    latest['H_minus_D'] = differences
    latest['gap20_relative_return_change_percent'] = 100*differences['gap20_return']['mean']/latest['D']['gap20_return']['mean']
    latest['H_gap20_wins_out_of_5'] = sum(r['H_minus_D_gap20_return'] > 0 for r in paired)
    latest['clean_return_H_over_D'] = latest['H']['clean_return']['mean']/latest['D']['clean_return']['mean']
    write_csv(data/'development_episodes.csv', episodes)
    write_csv(data/'development_round_summary.csv', rounds)
    write_csv(data/'latest_paired_outcomes.csv', paired)
    trajectory_rows = [{'arm': arm, 'seed': seed, 'condition': condition, 'action_index': t,
                        'normalized_reward': float(u[t]),
                        'availability_before_action': not(condition == 'gap20' and 150 <= t < 170)}
                       for (arm, seed, condition), u in sorted(traces.items()) for t in range(500)]
    write_csv(data/'latest_action_reward_traces.csv', trajectory_rows)
    plot_outputs = plots(rounds, paired, traces, figures)
    summary = {'analysis_type': 'descriptive_development_snapshot', 'analysis_cutoff_actions': LATEST,
        'training_seed_per_arm': 1, 'training_seeds': [0], 'environment_seeds': SEEDS,
        'evaluation_rounds_per_arm': len(STEPS), 'evaluation_episodes_total': len(episodes),
        'evaluation_actions_total': len(episodes)*500, 'evaluation_dmc_calls_total': len(episodes)*1000,
        'fixed_source_snapshots': SNAPSHOTS, 'latest': latest, 'source_round_verification': round_checks,
        'all_1000_traces_checked_for_finite_rewards_budget_mask_alignment_and_reward_sum': True,
        'all_100_round_freeze_host_rng_and_ledger_checks_passed': True,
        'sample_sd_definition': 'ddof=1 across five environment seeds; not a confidence interval',
        'confirmatory_final_evaluation_included': False,
        'confirmatory_significance_or_superiority_claim': False,
        'final_matrix_expected_cases': 1280, 'final_matrix_cases_used': 0,
        'selection_rule': 'All 50 development rounds 20,000..1,000,000, step 20,000; final training checkpoint, no best-checkpoint selection',
        'reward_window': {'start': 150, 'gap': 20, 'recovery': 100, 'stop_exclusive': 270},
        'important_limits': [
            'Five reused development environment seeds, one independently trained model per arm.',
            'No D/prior_only development evaluation; stronger-control claim unavailable.',
            'The final 40-seed matrix and its two Holm-corrected primary tests are not analyzed here.',
            'Matched environment/masks do not imply identical model trajectories.',
            'Helix uses known availability; this is not automatic anomaly detection.',
            'Candidate repair is capped at five missing observations; long gaps use prior fallback.',
            'Functional failure is a reward-based proxy, not certified falling or safety.',
            'Both training budgets are complete; the held-out confirmatory final matrix is not included.'
        ], 'figure_outputs': plot_outputs}
    write_json(data/'development_summary.json', summary)
    write_json(data/'source_manifest_sha256.json', {'path_base': 'experiment_data_root',
        'fixed_source_snapshots': SNAPSHOTS, 'files': [source_files[k] for k in sorted(source_files)],
        'verification': {'all_rounds': round_checks, 'source_files_read': len(source_files),
                         'trace_records_verified': len(episodes), 'relative_paths_only': True}})
    write_json(data/'analysis_provenance.json', {'analysis_script_sha256': sha(Path(__file__).read_bytes()),
        'python_packages': {'numpy': np.__version__, 'matplotlib': matplotlib.__version__},
        'fixed_source_snapshots': SNAPSHOTS, 'cutoff_actions': LATEST, 'analysis_rng_used': False,
        'training_or_network_access': False})
    print(json.dumps({'episodes': len(episodes), 'rounds_per_arm': len(STEPS), 'latest': latest,
                      'figures': plot_outputs}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence-root', '--source-root', dest='source_root', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--output-dir', type=Path, default=ROOT/'publication')
    args = parser.parse_args()
    run(args.source_root, args.output_dir)
