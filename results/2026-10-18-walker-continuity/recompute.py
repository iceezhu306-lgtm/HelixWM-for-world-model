"""Verify this publication and recompute basic tables from its archived CSVs.

Python 3.10+ standard library only. This does not load a model, run a simulator,
reconstruct metrics from omitted per-step NPZ files, or rerun hypothesis tests.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

ARMS = ('clean', 'black_frame', 'prior_only', 'v2_exact',
        'locked_ancestor', 'posterior_return_once')
GAPS = (1, 5, 10, 20, 50)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_csv(path):
    with path.open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def verify_manifest(root):
    manifest = json.loads((root / 'publication_manifest.json').read_text(encoding='utf-8'))
    for name, record in manifest['files'].items():
        path = (root / name).resolve()
        require(path.is_relative_to(root.resolve()), 'Unsafe manifest path')
        require(path.is_file() and path.stat().st_size == record['bytes'] and
                digest(path) == record['sha256'], 'File verification failed: ' + name)
    return len(manifest['files'])


def number(row, field):
    value = float(row[field])
    require(math.isfinite(value), 'Missing/nonfinite ' + field)
    return value


def analyze(root, stage):
    data = root / 'data' / stage
    rows = read_csv(data / 'case_metrics.csv')
    seeds = range(20000, 20020) if stage == 'design' else range(30000, 30040)
    expected = {(s, a, g) for s in seeds for a in ARMS for g in ((0,) if a == 'clean' else GAPS)}
    keys = [(int(r['environment_seed']), r['arm'], int(r['missing_length'])) for r in rows]
    require(len(keys) == len(set(keys)) and set(keys) == expected, stage + ': incomplete/duplicate matrix')
    require(all(r['stage'] == stage and r['status'] == 'complete' and
                r['technical_failure'] == 'False' and r['clean_reference_technical_failure'] == 'False' and
                r['frozen_hash_gate_passed'] == 'True' and r['actual_interaction_counters_complete'] == 'True'
                and int(r['actual_eval_agent_steps']) == int(r['valid_reward_rows']) == 500 and
                int(r['actual_eval_dmc_calls']) == 1000 and
                int(r['primary_zero_filled_action_count']) == 0 and
                r['functional_failure'] in ('True', 'False') for r in rows), stage + ': case contract changed')
    groups = defaultdict(list)
    clusters = defaultdict(list)
    for row in rows:
        groups[(row['arm'], int(row['missing_length']))].append(row)
        if row['arm'] != 'clean':
            clusters[(int(row['environment_seed']), row['arm'])].append(number(row, 'paired_integrated_deficit'))
    duration = []
    for (arm, gap), members in sorted(groups.items()):
        returns = [number(r, 'raw_observed_episode_return') for r in members]
        deficit = [number(r, 'paired_integrated_deficit') for r in members]
        duration.append(dict(stage=stage, arm=arm, missing_length=gap, episodes=len(members),
            mean_raw_return=statistics.fmean(returns), std_raw_return_population=statistics.pstdev(returns),
            mean_primary_deficit=statistics.fmean(deficit),
            functional_failure_count=sum(r['functional_failure'] == 'True' for r in members)))
    archived = {(r['arm'], int(r['missing_length'])): r for r in read_csv(data / 'duration_metrics.csv')}
    for row in duration:
        if row['arm'] == 'clean':
            continue
        other = archived[(row['arm'], row['missing_length'])]
        require(int(other['attempts']) == row['episodes'] and
                int(other['functional_failure_count']) == row['functional_failure_count'], 'Duration count mismatch')
        for field in ('mean_raw_return', 'mean_primary_deficit'):
            require(math.isclose(row[field], number(other, field), abs_tol=1e-10), 'Duration metric mismatch')
    cluster_means = {key: statistics.fmean(values) for key, values in clusters.items()}
    for row in read_csv(data / 'seed_cluster_metrics.csv'):
        key = (int(row['environment_seed']), row['arm'])
        require(math.isclose(cluster_means[key], number(row, 'mean_paired_integrated_deficit'),
                             abs_tol=1e-12), 'Seed-cluster mean mismatch')
    comparisons = []
    for row in read_csv(data / 'paired_comparisons.csv'):
        differences = [cluster_means[(s, row['proposal'])] - cluster_means[(s, row['comparator'])] for s in seeds]
        mean = statistics.fmean(differences)
        require(math.isclose(mean, number(row, 'mean'), abs_tol=1e-12), 'Paired contrast mismatch')
        comparisons.append(dict(stage=stage, proposal=row['proposal'], comparator=row['comparator'],
            evaluation_seed_clusters=len(differences), mean_deficit_difference=mean,
            archived_bootstrap_ci95=json.loads(row['bootstrap_ci95']),
            archived_holm_adjusted_p=number(row, 'holm_adjusted_p'),
            archived_failure_guard=row['failure_guard']))
    return dict(stage=stage, cases=len(rows), technical_failures=0,
                action_decisions=sum(int(r['actual_eval_agent_steps']) for r in rows),
                dmc_calls=sum(int(r['actual_eval_dmc_calls']) for r in rows),
                duration_table=duration, contrasts=comparisons)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--output', type=Path, help='New output directory for recomputed CSV/JSON; never overwritten')
    args = parser.parse_args(argv)
    root = args.root.resolve()
    verified = verify_manifest(root)
    stages = [analyze(root, stage) for stage in ('design', 'held_out')]
    result = dict(passed=True, verified_publication_files=verified, training_seeds=1,
        new_training_updates=0, total_cases=sum(s['cases'] for s in stages),
        stages=stages, scope='Arithmetic recheck of published case-level archived analysis CSVs.',
        limitations=['Per-step traces and model weights are deliberately omitted.',
                     'Bootstrap intervals, randomization p-values and decision labels are archived results, not rerun here.',
                     'Raw reward/posture metric extraction requires the separately retained full archive.'])
    require(result['total_cases'] == 1560, 'Wrong total case count')
    if args.output:
        output = args.output.resolve()
        output.mkdir(parents=True, exist_ok=False)
        (output / 'recomputed_summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        table = [row for stage in stages for row in stage['duration_table']]
        with (output / 'recomputed_duration_metrics.csv').open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(table[0]))
            writer.writeheader(); writer.writerows(table)
    print(json.dumps({k: v for k, v in result.items() if k != 'stages'}, indent=2))


if __name__ == '__main__':
    main()
