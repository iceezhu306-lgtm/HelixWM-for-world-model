"""Recompute published development statistics and plots without a server/model.

Run: python code/recompute.py --output-dir ../recomputed
Requires Python, numpy and matplotlib. Inputs are solely the publication data
folder. No checkpoint/replay, private filesystem, JAX or network is required.
"""
from __future__ import annotations
import argparse
import csv
import io
import json
from pathlib import Path
import zipfile
import numpy as np
import analyze_development as a


def close(actual, expected, label):
    if isinstance(expected, dict):
        a.require(set(actual) == set(expected), f'{label}: keys differ')
        for k in expected:
            close(actual[k], expected[k], f'{label}/{k}')
    elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
        a.require(np.isclose(actual, expected, atol=1e-9, rtol=1e-12), f'{label}: number differs')
    else:
        a.require(actual == expected, f'{label}: value differs')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--publication-root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    data = args.publication_root/'data'
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    summary = json.loads((data/'development_summary.json').read_text('utf8'))
    rows = []
    for arm in ('D', 'H'):
        raw = [json.loads(x) for x in (data/f'{arm}_development_eval.jsonl').read_text('utf8').splitlines() if x]
        a.require(len(raw) == 500, 'Each model requires 500 development rows')
        a.require(all(x['arm'] == arm for x in raw), 'Unexpected arm')
        rows.extend(raw)
    rounds = []
    for arm in ('D', 'H'):
        for step in a.STEPS:
            for condition, controller, gap in [('clean', 'clean_native', 0), ('gap20', a.CONTROLLER[arm], 20)]:
                selected = [r for r in rows if r['arm'] == arm and r['training_actions'] == step and r['controller'] == controller]
                a.require(len(selected) == 5 and sorted(r['seed'] for r in selected) == a.SEEDS, 'Seed grid differs')
                a.require(all(r['actions'] == 500 and r['dmc_calls'] == 1000 and r['gap'] == gap for r in selected), 'Budget differs')
                stat = a.stats([r['episode_return'] for r in selected])
                rounds.append({'arm': arm, 'condition': condition, 'controller': controller,
                    'training_actions': step, 'n_environment_seeds': 5, 'mean_episode_return': stat['mean'],
                    'sample_sd_episode_return': stat['sample_sd'], 'minimum_episode_return': stat['minimum'],
                    'maximum_episode_return': stat['maximum']})
    with (data/'development_round_summary.csv').open(newline='', encoding='utf8') as stream:
        published_rounds = list(csv.DictReader(stream))
    a.require(len(published_rounds) == 200, 'Published round count differs')
    for actual, expected in zip(rounds, published_rounds):
        for key, val in actual.items():
            if isinstance(val, (float, int)):
                close(val, float(expected[key]), f'round/{key}')
            else:
                close(val, expected[key], f'round/{key}')
    zipped = (data/'latest_traces.zip').read_bytes()
    manifest = json.loads((data/'latest_traces_manifest.json').read_text('utf8'))
    a.require(a.sha(zipped) == manifest['archive_sha256'], 'Archive SHA differs')
    traces = {}
    with zipfile.ZipFile(io.BytesIO(zipped)) as archive:
        a.require(archive.testzip() is None, 'Archive CRC failed')
        a.require(set(archive.namelist()) == set(manifest['files']), 'Archive member set differs')
        for name, record in manifest['files'].items():
            contents = archive.read(name)
            a.require(a.sha(contents) == record['sha256'] and len(contents) == record['bytes'], 'Member SHA differs')
        for arm in ('D', 'H'):
            for condition, controller, gap in [('clean', 'clean_native', 0), ('gap20', a.CONTROLLER[arm], 20)]:
                for seed in a.SEEDS:
                    name = f'{arm}/{controller}_seed{seed}_gap{gap}'
                    row = json.loads(archive.read(name+'.json'))
                    with np.load(io.BytesIO(archive.read(name+'.npz')), allow_pickle=False) as saved:
                        trace = {k: saved[k] for k in saved.files}
                    mask = np.ones(501, bool)
                    if gap:
                        mask[150:170] = False
                    a.require(row['availability_sha256'] == a.mask_sha(mask), 'Mask SHA differs')
                    a.require(np.array_equal(trace['availability_before_action'], mask[:-1]) and
                              np.array_equal(trace['availability_outcome'], mask[1:]), 'Mask timing differs')
                    a.require(trace['reward'].shape == (500,) and np.isfinite(trace['reward']).all() and
                              (trace['dmc_calls'] == 2).all(), 'Invalid reward trace')
                    close(float(trace['reward'].sum()), row['episode_return'], 'Trace return')
                    traces[(arm, seed, condition)] = trace['reward']/trace['dmc_calls']
    paired = []
    for seed in a.SEEDS:
        row = {'seed': seed, 'training_actions': a.LATEST}
        for arm in ('D', 'H'):
            clean, fault = traces[(arm, seed, 'clean')], traces[(arm, seed, 'gap20')]
            row[f'{arm}_clean_return'] = float(clean.sum()*2)
            row[f'{arm}_gap20_return'] = float(fault.sum()*2)
            row[f'{arm}_clean_minus_gap20'] = row[f'{arm}_clean_return']-row[f'{arm}_gap20_return']
            row[f'{arm}_gap_window_score'] = float(fault[150:270].mean())
            row[f'{arm}_gap_window_own_clean_deficit'] = float((clean[150:270]-fault[150:270]).mean())
            row[f'{arm}_functional_failure'] = a.exists_run(fault[150:270] < .2, 20)
            for k, value in a.recovery(fault).items():
                row[f'{arm}_recovery_{k}'] = value
        for metric in ('clean_return', 'gap20_return', 'clean_minus_gap20', 'gap_window_score', 'gap_window_own_clean_deficit'):
            row[f'H_minus_D_{metric}'] = row[f'H_{metric}']-row[f'D_{metric}']
        paired.append(row)
    latest = {}
    metrics = ('clean_return', 'gap20_return', 'clean_minus_gap20', 'gap_window_score', 'gap_window_own_clean_deficit')
    for arm in ('D', 'H'):
        latest[arm] = {m: a.stats([r[f'{arm}_{m}'] for r in paired]) for m in metrics}
        latest[arm]['functional_failures_gap20_window'] = sum(r[f'{arm}_functional_failure'] for r in paired)
        latest[arm]['recovery_eligible'] = sum(r[f'{arm}_recovery_eligible'] for r in paired)
        latest[arm]['recovered_by_100_actions'] = sum(r[f'{arm}_recovery_status'] == 'observed' for r in paired)
    latest['H_minus_D'] = {m: a.stats([r[f'H_minus_D_{m}'] for r in paired]) for m in metrics}
    latest['gap20_relative_return_change_percent'] = 100*latest['H_minus_D']['gap20_return']['mean']/latest['D']['gap20_return']['mean']
    latest['H_gap20_wins_out_of_5'] = sum(r['H_minus_D_gap20_return'] > 0 for r in paired)
    latest['clean_return_H_over_D'] = latest['H']['clean_return']['mean']/latest['D']['clean_return']['mean']
    close(latest, summary['latest'], 'Latest summary')
    a.write_csv(output/'recomputed_round_summary.csv', rounds)
    a.write_csv(output/'recomputed_latest_paired.csv', paired)
    figures = a.plots(rounds, paired, traces, output/'figures')
    a.write_json(output/'recompute_verification.json', {'passed': True, 'raw_jsonl_rows': len(rows),
        'latest_trace_cases': len(traces), 'trace_zip_crc_and_sha_passed': True,
        'all_summary_numbers_match': True, 'training_or_network_access': False, 'figures': figures})
    print(json.dumps({'passed': True, 'episodes': len(rows), 'latest_cases': len(traces)}))


if __name__ == '__main__':
    main()
