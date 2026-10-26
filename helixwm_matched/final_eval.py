"""Fixed 1280-episode final frozen evaluation of two independently trained arms.

No training, replay loading, checkpoint selection, retry, or resume is performed.
The sole formal video combines three predeclared cases already in this matrix;
diagnostic frames never enter a policy. Statistical analysis is a separate step.
"""
from __future__ import annotations

import argparse
import copy
import csv
import datetime
import hashlib
import json
import os
from pathlib import Path
import pickle
import random
import time
import traceback

import numpy as np

from .protocol import (atomic_json, fixed_gap_mask, load_protocol, mask_sha256,
                       protocol_sha256, validate_protocol)
from .recovery_probe import jsonable, tree_hash

VIDEO_PANELS = (('D', 'black_frame_native'), ('H', 'helix_v2'), ('D', 'prior_only'))
VIDEO_SEED, VIDEO_GAP, VIDEO_FPS = 60000, 20, 20


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''): h.update(block)
    return h.hexdigest()


def case_id(arm, controller, seed, gap):
    return f'{arm}_{controller}_seed{seed}_gap{gap:02d}'


def build_plan(config):
    validate_protocol(config)
    final = config['final_evaluation']
    if final['controllers'] != ['clean_native', 'black_frame_native', 'prior_only', 'helix_v2']:
        raise ValueError('Unregistered final controller matrix')
    selection = config['operations']['final_video_selection']
    if (selection['environment_seed'], selection['gap_actions'], selection['panels']) != (
            VIDEO_SEED, VIDEO_GAP,
            ['D_final_1000000/black_frame_native', 'H_final_1000000/helix_v2', 'D_final_1000000/prior_only']):
        raise ValueError('Formal video selection differs from the registered protocol')
    plan = []
    for arm in ('D', 'H'):
        for controller in final['controllers']:
            gaps = [0] if controller == 'clean_native' else [1, 5, 10, 20, 50]
            for gap in gaps:
                for seed in range(60000, 60040):
                    mask = np.ones(501, bool) if gap == 0 else fixed_gap_mask(gap)
                    plan.append({'case_id': case_id(arm, controller, seed, gap),
                        'arm': arm, 'controller': controller, 'seed': seed, 'gap': gap,
                        'missing_start': 150 if gap else None, 'actions': 500, 'dmc_calls': 1000,
                        'availability_sha256': mask_sha256(mask),
                        'record_video_frames': (arm, controller) in VIDEO_PANELS
                            and seed == VIDEO_SEED and gap == VIDEO_GAP})
    if len(plan) != 1280 or len({r['case_id'] for r in plan}) != 1280 or sum(r['record_video_frames'] for r in plan) != 3:
        raise RuntimeError('Final matrix cardinality differs')
    return plan


def validate_checkpoint_payload(payload, record, config, expected_identity, arm):
    if payload.get('format') != 'helixwm_matched_episode_boundary_v1':
        raise ValueError('Final evaluation requires the matched boundary format')
    if payload.get('identity') != expected_identity or record.get('identity') != expected_identity:
        raise ValueError('Final checkpoint source/protocol/environment identity mismatch')
    counters = {'agent_steps': 1000000, 'env_steps': 2000000, 'episodes': 2000, 'update_steps': 248750}
    if payload.get('counters') != counters or any(record.get(k) != v for k, v in counters.items()):
        raise ValueError('Final checkpoint did not complete the registered full budget')
    if payload.get('next_operation') != 'reset_new_episode_and_reanchor':
        raise ValueError('Final checkpoint is not an episode boundary')
    core = payload['backbone']; native = core['training']['counters']
    if native.get('updates') != 248750 or native.get('batches') != 248750 or core['writeback_updates'] != 248750:
        raise ValueError('Final native learner/writeback budget differs')
    if core['inference_version'] != 248750:
        raise ValueError('Final inference copy is not pinned to the final learner')
    name = 'black_frame_native' if arm == 'D' else 'helix_v2'
    controller = payload['completed_controller']
    state = controller['chain']['state']
    if controller['controller'] != name or state['episode_id'] != 1999 or state['timestamp'] != 500 or state['anchor_time'] != 500:
        raise ValueError('Final checkpoint arm or completed controller differs')
    if state['last_action'] is not None or state['pending'] is not None or state['missing_length'] != 0 or state['abstaining']:
        raise ValueError('Final controller contains an unresolved action or gap')
    if payload['replay']['rows_added'] != 1002000:
        raise ValueError('Replay row count is not actions plus reset/final accounting')
    # Private training RNG is preserved as evidence, never consumed by eval.
    generator = np.random.default_rng(); generator.bit_generator.state = copy.deepcopy(payload['prefill_rng'])
    local_np = np.random.RandomState(); local_np.set_state(payload['numpy_global_rng'])
    local_py = random.Random(); local_py.setstate(payload['python_rng'])
    return counters


def load_final_checkpoint(path, config, identity, arm):
    from .backbone import NativeReplay
    path = Path(path).resolve()
    record_path = path.with_suffix('.json')
    record = json.loads(record_path.read_text())
    digest = sha_file(path)
    if record.get('sha256') != digest or record.get('bytes') != path.stat().st_size:
        raise ValueError('Final checkpoint file SHA256/size mismatch')
    if path.parent.name != 'CHECKPOINTS' or record.get('file', '').replace('\\', '/') != 'CHECKPOINTS/' + path.name:
        raise ValueError('Final checkpoint location differs from its metadata')
    # Trusted local training pickle only. File hashes are integrity checks, not
    # a sandbox for untrusted Python pickle content.
    with path.open('rb') as stream: payload = pickle.load(stream)
    validate_checkpoint_payload(payload, record, config, identity, arm)
    if record['replay_files'] != NativeReplay.referenced_files(payload['replay']):
        raise ValueError('Final checkpoint replay-reference metadata differs')
    run = path.parent.parent
    if (run/'train.exit').read_text().strip() != '0': raise ValueError('Training did not exit zero')
    completion = json.loads((run/'completion.json').read_text())
    if (completion.get('training_complete') is not True or completion.get('stage') != 'formal'
            or completion.get('arm') != arm or completion.get('identity') != identity
            or completion.get('final_checkpoint', {}).get('sha256') != digest
            or any(completion.get(k) != v for k, v in payload['counters'].items())):
        raise ValueError('Formal completion evidence does not match the final checkpoint')
    return payload, record


def frozen_fingerprint(backbone):
    state = backbone.snapshot()
    return {'full_backbone_sha256': tree_hash(state),
            'model_optimizer_normalizer_target_sha256': tree_hash(state['training']['params']),
            'native_counters': copy.deepcopy(state['training']['counters']),
            'native_training_seed': state['resolved_config']['seed'],
            'native_train_carry_sha256': tree_hash(state['train_carry']),
            'inference_sha256': tree_hash(state['inference_params']),
            'inference_version': state['inference_version'],
            'numpy_host_rng_sha256': tree_hash(np.random.get_state()),
            'python_host_rng_sha256': tree_hash(random.getstate())}


def check_episode(row, planned):
    if any(row.get(k) != planned[k] for k in ('controller', 'seed', 'gap', 'actions', 'dmc_calls', 'availability_sha256')):
        raise ValueError('Actual episode identity/count differs from its registered cell')
    trace = row['trace']
    schema = {'action': ((500, 6), np.float32), 'reward': ((500,), np.float64),
              'dmc_calls': ((500,), np.int32), 'availability_outcome': ((500,), np.bool_),
              'availability_before_action': ((500,), np.bool_)}
    if set(trace) != set(schema): raise ValueError('Unexpected final trace schema')
    for key, (shape, dtype) in schema.items():
        value = np.asarray(trace[key])
        if value.shape != shape or value.dtype != dtype or not np.isfinite(value).all():
            raise ValueError(f'Invalid trace field {key}')
    if np.any(np.abs(trace['action']) > 1) or np.any(trace['dmc_calls'] != 2):
        raise ValueError('Actual action bounds/native-call counts differ')
    if np.any(trace['reward'] < 0) or np.any(trace['reward'] > 2):
        raise ValueError('Walker reward falls outside registered bounds')
    mask = np.ones(501, bool) if planned['gap'] == 0 else fixed_gap_mask(planned['gap'])
    if not np.array_equal(trace['availability_before_action'], mask[:-1]) or not np.array_equal(trace['availability_outcome'], mask[1:]):
        raise ValueError('Actual camera schedule differs from registered mask')
    if row['episode_return'] != float(sum(trace['reward'])):
        raise ValueError('Episode return does not equal retained raw rewards')


class FrameCollector:
    def __init__(self): self.frames, self.availability = [], []
    def __call__(self, frame, timestamp, info):
        frame = np.asarray(frame)
        if type(timestamp) is not int or timestamp != len(self.frames) or timestamp > 500:
            raise ValueError('Video callback timestamp is not consecutive 0..500')
        if frame.shape != (256, 256, 3) or frame.dtype != np.uint8:
            raise ValueError('Video callback must return real RGB256')
        self.frames.append(frame.copy()); self.availability.append(bool(info['availability']))

    def save(self, directory, *, complete):
        if complete and len(self.frames) != 501: raise ValueError('Formal source video requires 501 frames')
        path = Path(directory)/('video_frames.npz' if complete else 'partial_video_frames.npz')
        with path.open('xb') as stream:
            np.savez_compressed(stream, frames=np.asarray(self.frames, np.uint8),
                                availability=np.asarray(self.availability, bool),
                                timestamp=np.arange(len(self.frames), dtype=np.int32))
        return {'file': path.name, 'sha256': sha_file(path), 'bytes': path.stat().st_size,
                'frames': len(self.frames), 'complete': complete,
                'image_semantics': 'evaluator_only_real_scene_not_policy_input'}


def make_video(root, panel_records):
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw, ImageFont
    root = Path(root); out = root/'VIDEOS'; out.mkdir(exist_ok=False)
    arrays, sources = [], []
    for arm, controller in VIDEO_PANELS:
        cid = case_id(arm, controller, VIDEO_SEED, VIDEO_GAP)
        record = panel_records[cid]
        path = root/'CASES'/cid/record['file']
        if sha_file(path) != record['sha256']: raise ValueError('Video source frames changed')
        with np.load(path, allow_pickle=False) as data:
            frames, available, times = data['frames'], data['availability'], data['timestamp']
        if frames.shape != (501, 256, 256, 3) or frames.dtype != np.uint8 or not np.array_equal(times, np.arange(501)):
            raise ValueError('Formal panel frame contract differs')
        if not np.array_equal(available, fixed_gap_mask(20)):
            raise ValueError('Formal panel availability differs')
        arrays.append((frames, available))
        sources.append({'case_id': cid, 'source_frames_sha256': record['sha256']})
    labels = ['D: black-frame', 'H: Helix V2', 'D: prior-only']
    try: font = ImageFont.truetype('DejaVuSans.ttf', 14)
    except OSError: font = ImageFont.load_default()
    partial = out/'matched_walker_comparison.partial.mp4'
    writer = imageio.get_writer(str(partial), fps=VIDEO_FPS, codec='libx264',
                                macro_block_size=16, ffmpeg_log_level='error')
    try:
        for t in range(501):
            canvas = Image.new('RGB', (768, 304), (18, 22, 29))
            draw = ImageDraw.Draw(canvas)
            for p, (frames, available) in enumerate(arrays):
                x = p*256; canvas.paste(Image.fromarray(frames[t]), (x, 48))
                draw.text((x+6, 4), labels[p]+' | seed 60000', fill='white', font=font)
                draw.text((x+6, 24), f't={t:03d}/500 | '+('OBSERVED' if available[t] else 'MISSING'),
                          fill=(140, 240, 170) if available[t] else (255, 170, 100), font=font)
            writer.append_data(np.asarray(canvas))
    finally: writer.close()
    reader = imageio.get_reader(str(partial)); decoded = 0
    try:
        metadata = reader.get_meta_data()
        for frame in reader:
            if frame.shape != (304, 768, 3) or frame.dtype != np.uint8:
                raise ValueError('Decoded formal MP4 shape/dtype mismatch')
            decoded += 1
    finally: reader.close()
    if decoded != 501 or abs(float(metadata['fps'])-VIDEO_FPS) > 1e-8:
        raise ValueError('Formal MP4 did not fully decode at its declared playback rate')
    target = out/'matched_walker_comparison.mp4'; partial.rename(target)
    if len(list(root.rglob('*.mp4'))) != 1: raise ValueError('Expected exactly one formal MP4')
    return {'file': target.relative_to(root).as_posix(), 'sha256': sha_file(target),
            'bytes': target.stat().st_size, 'full_decode_pass': True, 'frames': decoded,
            'dimensions': [768, 304], 'fps': VIDEO_FPS, 'fps_semantics': 'presentation_playback_rate',
            'sources': sources, 'extra_evaluation_episodes': 0,
            'synchronized_by': 'action_timestamp_not_identical_trajectory',
            'real_scene_not_policy_input': True, 'availability_labels': ['OBSERVED', 'MISSING']}


def source_manifest():
    from .train import source_digest
    _, runtime = source_digest()
    root = Path(__file__).resolve().parent.parent
    records = dict(runtime)
    for relative in ('helixwm_matched/final_eval.py', 'helixwm_matched/recovery_probe.py'):
        records[relative] = sha_file(root/relative)
    return {'files': records, 'manifest_sha256': protocol_sha256(records),
            'evaluation_module_excluded_from_training_runtime_hash': True}


def run(args):
    os.environ.setdefault('MUJOCO_GL', 'egl'); os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
    os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
    from .backbone import MatchedDreamerBackbone
    from .evaluation import evaluate_episode
    from .env_probe import first_difference, runtime_lock
    from .train import source_digest
    cfg = load_protocol(args.protocol); plan = build_plan(cfg)
    root = Path(args.output_dir).resolve()
    inputs = {'D': Path(args.checkpoint_d).resolve(), 'H': Path(args.checkpoint_h).resolve()}
    for cp in inputs.values():
        source = cp.parent.parent
        if root == source or root.is_relative_to(source) or source.is_relative_to(root):
            raise ValueError('Final evaluation must be isolated from both original training runs')
    if inputs['D'] == inputs['H']: raise ValueError('D and H require distinct final checkpoints')
    root.mkdir(parents=True, exist_ok=False)
    (root/'CASES').mkdir()
    identity = {'protocol_sha256': protocol_sha256(cfg), 'source_sha256': source_digest()[0],
                'environment_lock_sha256': sha_file(args.environment_lock),
                'upstream_commit': cfg['official_core']['commit']}
    frozen_sources = source_manifest()
    atomic_json(root/'source_manifest.json', frozen_sources)
    atomic_json(root/'protocol.json', cfg)
    atomic_json(root/'identity.json', identity)
    atomic_json(root/'plan.json', {'cases': plan, 'count': len(plan),
                'actions': 640000, 'dmc_calls': 1280000, 'training_updates': 0})
    expected_lock = json.loads(Path(args.environment_lock).read_text())
    atomic_json(root/'environment_lock.json', expected_lock)
    counts = {'completed_episodes': 0, 'acknowledged_actions': 0, 'acknowledged_dmc_calls': 0}
    reports, panel_records, checkpoints = {}, {}, {}
    rows = []; active = None; returned = False; frames = None; started = time.monotonic()
    try:
        # Verify both final inputs before spending GPU evaluation work.
        for arm, cp in inputs.items():
            payload, record = load_final_checkpoint(cp, cfg, identity, arm)
            checkpoints[arm] = {'sha256': record['sha256'], 'bytes': record['bytes'],
                               'metadata_sha256': sha_file(cp.with_suffix('.json')),
                               'counters': payload['counters'], 'controller': payload['completed_controller']['controller']}
            del payload
        atomic_json(root/'checkpoint_inputs.json', checkpoints)
        backbone = MatchedDreamerBackbone({'seed': 0, 'platform': 'cuda', 'size': 'size12m',
                     'compute_dtype': 'bfloat16', 'batch_size': 16, 'batch_length': 64}, args.upstream, root/'MODEL')
        if not all(d.platform == 'gpu' for d in backbone.jax.devices()): raise RuntimeError('Final evaluation requires the verified GPU runtime')
        for arm in ('D', 'H'):
            payload, record = load_final_checkpoint(inputs[arm], cfg, identity, arm)
            backbone.load(payload['backbone'])
            loaded = backbone.snapshot()
            reload_checks = {k: tree_hash(loaded[k]) == tree_hash(payload['backbone'][k])
                             for k in ('training', 'train_carry', 'inference_params', 'inference_version', 'writeback_updates')}
            if not all(reload_checks.values()): raise RuntimeError('Strict final checkpoint reload differs')
            preserved_training_rng = tree_hash({'collector': payload['completed_controller']['chain']['rngs'],
                'prefill': payload['prefill_rng'], 'sampler': payload['replay']['sampler_rng']})
            np.random.set_state(payload['numpy_global_rng']); random.setstate(payload['python_rng'])
            before = frozen_fingerprint(backbone)
            for planned in [p for p in plan if p['arm'] == arm]:
                active = root/'CASES'/planned['case_id']; active.mkdir(exist_ok=False)
                atomic_json(active/'planned.json', planned)
                atomic_json(root/'status.json', {'phase': 'evaluating', 'utc': now(), 'current': planned,
                            **counts, 'elapsed_s': time.monotonic()-started})
                returned = False; frames = FrameCollector() if planned['record_video_frames'] else None
                row = evaluate_episode(backbone, cfg, planned['controller'], planned['seed'], planned['gap'],
                    partial_report=active/'partial_failure.json', render_callback=frames)
                returned = True
                counts['acknowledged_actions'] += row['actions']; counts['acknowledged_dmc_calls'] += row['dmc_calls']
                # Preserve real outputs even if a later identity/freeze check
                # fails; no unknown reward prefix is silently replaced by zero.
                trace_path = active/'trace.npz'
                with trace_path.open('xb') as stream: np.savez_compressed(stream, **row['trace'])
                atomic_json(active/'observed.json', jsonable({k: v for k, v in row.items() if k != 'trace'}))
                check_episode(row, planned)
                after = frozen_fingerprint(backbone)
                if after != before: raise RuntimeError('Frozen evaluation changed model/optimizer/counter/host RNG')
                live_lock = runtime_lock(cfg)
                mismatch = first_difference(expected_lock, live_lock)
                if mismatch: raise RuntimeError(f'Final evaluation environment lock mismatch: {mismatch}')
                row.pop('trace'); events = row.pop('repair_events')
                video_record = frames.save(active, complete=True) if frames is not None else None
                if video_record is not None: panel_records[planned['case_id']] = video_record
                summary = {**row, 'case_id': planned['case_id'], 'arm': arm, 'status': 'complete',
                    'training_seed': 0, 'training_actions': 1000000, 'training_dmc_calls': 2000000,
                    'controller_rng_root_seed': planned['seed'],
                    'controller_rng_streams': ['real', 'actor', 'dream', 'repair'],
                    'checkpoint_sha256': record['sha256'], 'trace_sha256': sha_file(trace_path),
                    'repair_events': events, 'video_frames': video_record,
                    'frozen': {'passed': True, 'before': before, 'after': after},
                    'actual_environment_lock_sha256': protocol_sha256(live_lock)}
                atomic_json(active/'summary.json', jsonable(summary))
                compact = {k: summary[k] for k in ('case_id', 'arm', 'controller', 'seed', 'gap',
                           'status', 'episode_return', 'actions', 'dmc_calls', 'checkpoint_sha256', 'trace_sha256')}
                rows.append(compact); counts['completed_episodes'] += 1
                with (root/'episodes.jsonl').open('a', encoding='utf8') as stream:
                    stream.write(json.dumps(compact, allow_nan=False)+'\n')
                atomic_json(root/'interaction_ledger.json', {'phase': 'in_progress', **counts})
                active = None; frames = None
            after = frozen_fingerprint(backbone)
            if before != after or sha_file(inputs[arm]) != record['sha256']:
                raise RuntimeError('Final arm freeze or original checkpoint integrity failed')
            if preserved_training_rng != tree_hash({'collector': payload['completed_controller']['chain']['rngs'],
                    'prefill': payload['prefill_rng'], 'sampler': payload['replay']['sampler_rng']}):
                raise RuntimeError('Preserved training RNG payload changed')
            reports[arm] = {'passed': True, 'episodes': 640, 'reload_checks': reload_checks,
                           'before': before, 'after': after, 'training_updates': 0,
                           'preserved_training_private_rng_sha256': preserved_training_rng,
                           'training_private_rng_consumed': False,
                           'prefill_rng_clone_used_for_schema_validation_only': True}
            atomic_json(root/f'frozen_{arm}.json', reports[arm])
            del payload, loaded
        expected_counts = {'completed_episodes': 1280, 'acknowledged_actions': 640000, 'acknowledged_dmc_calls': 1280000}
        if counts != expected_counts: raise RuntimeError('Final matrix interaction budget differs')
        artifact_before = frozen_fingerprint(backbone)
        video = make_video(root, panel_records)
        artifact_after = frozen_fingerprint(backbone)
        if artifact_before != artifact_after: raise RuntimeError('Media generation changed frozen state or host RNG')
        with (root/'episodes.csv').open('x', newline='', encoding='utf8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        if source_manifest() != frozen_sources: raise RuntimeError('Evaluation source changed during execution')
        report = {'format': 'helixwm_matched_final_evaluation_v1', 'passed': True, 'utc': now(),
            'identity': identity, **counts, 'training_updates': 0, 'checkpoint_inputs': checkpoints,
            'arms': reports, 'video': video, 'artifact_freeze_passed': True,
            'all_environments_and_video_handles_closed': True,
            'statistical_analysis_complete': False, 'scientific_superiority_established': False,
            'one_training_seed_per_arm': True, 'new_independent_training_performed_here': False,
            'resume_or_retry_performed': False, 'wall_seconds': time.monotonic()-started}
        atomic_json(root/'completion.json', report)
        atomic_json(root/'status.json', {'phase': 'complete', 'utc': now(), **counts})
        atomic_json(root/'interaction_ledger.json', {'phase': 'complete', 'passed': True, **counts})
        inventory = {p.relative_to(root).as_posix(): {'sha256': sha_file(p), 'bytes': p.stat().st_size}
                     for p in sorted(root.rglob('*')) if p.is_file()}
        atomic_json(root/'file_manifest.json', {'files': inventory,
                    'excludes_self_and_final_exit_marker': True})
        for name, item in inventory.items():
            if sha_file(root/name) != item['sha256']: raise RuntimeError('Final artifact hash changed after inventory')
        (root/'evaluation.exit').write_text('0\n')
        return report
    except BaseException as exc:
        counts_complete = True
        preservation_error = None
        if active is not None:
            partial = active/'partial_failure.json'
            if not returned and partial.exists():
                prefix = json.loads(partial.read_text())
                counts['acknowledged_actions'] += int(prefix.get('acknowledged_actions', 0))
                counts['acknowledged_dmc_calls'] += int(prefix.get('acknowledged_dmc_calls', 0))
                counts_complete = all(k in prefix for k in ('acknowledged_actions', 'acknowledged_dmc_calls'))
            elif not returned:
                counts_complete = False
            if frames is not None and frames.frames and not (active/'partial_video_frames.npz').exists():
                try: frames.save(active, complete=False)
                except BaseException as video_error: preservation_error = repr(video_error)
        failure = {'passed': False, 'phase': 'failed', 'utc': now(), 'error': repr(exc),
                   'traceback': traceback.format_exc(), **counts,
                   'failed_case': active.name if active is not None else None,
                   'actual_interaction_counts_complete': counts_complete,
                   'partial_video_preservation_error': preservation_error,
                   'automatic_retry_or_resume': False,
                   'remaining_cases_uncompleted': 1280-counts['completed_episodes'],
                   'unstarted_cases': 1280-counts['completed_episodes']-(1 if active is not None else 0)}
        atomic_json(root/'failure.json', failure)
        atomic_json(root/'status.json', failure)
        atomic_json(root/'interaction_ledger.json', failure)
        (root/'evaluation.exit').write_text('1\n')
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol', type=Path, required=True)
    p.add_argument('--upstream', type=Path, required=True)
    p.add_argument('--checkpoint-d', type=Path, required=True)
    p.add_argument('--checkpoint-h', type=Path, required=True)
    p.add_argument('--environment-lock', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    existed = args.output_dir.exists()
    try:
        result = run(args)
    except BaseException as exc:
        # Setup failures after this invocation created its output also get a
        # failure marker. Never touch an already existing evaluation directory.
        if not existed and args.output_dir.is_dir():
            if not (args.output_dir/'failure.json').exists():
                atomic_json(args.output_dir/'failure.json', {'passed': False, 'phase': 'setup_failed',
                            'error': repr(exc), 'traceback': traceback.format_exc()})
            (args.output_dir/'evaluation.exit').write_text('1\n')
        raise
    print(json.dumps({'passed': result['passed'], 'episodes': result['completed_episodes'],
                      'output': str(args.output_dir.resolve())}), flush=True)


if __name__ == '__main__': main()
