"""Fresh-process software recovery gate for a completed Minecraft smoke run.

One real next-seed reset supplies a recorded RGB frame. Three fresh learner
processes restore the same complete checkpoint and use THAT SAME frame to test
policy/RNG/replay/optimizer continuity. This does not compare regenerated Java
worlds, serialize an engine, or claim bitwise Minecraft environment recovery.
Run only after the independent real-engine and smoke checks have succeeded.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import traceback

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import numpy as np

from helixwm_v2.dual_chain import DualChain
from helixwm_v2.evaluation import DropoutSchedule
from .backbone import MinecraftBackbone, TASK_NAMESPACE
from .storage import Replay, atomic_json, load_checkpoint, restore_chain_boundary, sha256, tree_hash
from .train import batch_for_backbone, source_identity, utc


def comparable_backbone(snapshot):
    snapshot = deepcopy(snapshot)
    snapshot["backbone"].get("resolved_config", {}).pop("logdir", None)
    return snapshot


def software_probe(payload, backbone, replay, initial_rgb):
    """No Minecraft commands here; an identical recorded first frame is given.

    All state needed for the next replay sample and one original learner update
    is restored before this probe. It mutates only this temporary probe model.
    """
    config = payload["config"]
    initial_rgb = np.asarray(initial_rgb)
    if initial_rgb.shape != (64, 64, 3) or initial_rgb.dtype != np.uint8:
        raise ValueError("Recovery probe needs an unmodified uint8 RGB64 first frame")
    backbone.load(payload["backbone"])
    if tree_hash(comparable_backbone(backbone.snapshot())) != tree_hash(comparable_backbone(payload["backbone"])):
        raise ValueError("Full model/optimizer/counter/inference snapshot did not reload exactly")
    restored_model_hash = backbone.parameter_hash()
    replay.restore(payload["replay"], persist=False)
    chain = DualChain(backbone, seed=0, **config["dual_chain"])
    boundary = payload["collector"]
    restore_chain_boundary(chain, boundary["dual_chain"], boundary["episode_action_count"])
    dropout = DropoutSchedule(0, config["observation_dropout"])
    dropout.load_state_dict(boundary["dropout"])
    random.setstate(payload["rng"]["python"])
    np.random.set_state(payload["rng"]["numpy"])
    host_draws = {"python": random.random(), "numpy": np.random.random(4).tolist()}
    chain.reset(initial_rgb, episode_id=payload["counters"]["episodes"],
                timestamp=0, model_version=backbone.inference_version)
    action = chain.act()[0]
    mask = [dropout.valid(is_first=True)]+[dropout.valid() for _ in range(10)]
    if backbone.parameter_hash() != restored_model_hash:
        raise ValueError("Recovery policy inference changed model/optimizer")
    batch = batch_for_backbone(replay, config["backbone"])
    batch_hash = tree_hash(batch)
    updates_before = payload["counters"]["update_steps"]
    metrics = backbone.train(batch)
    if not metrics or not all(np.isfinite(value) for value in metrics.values()):
        raise FloatingPointError("Probe official update returned no finite learner evidence")
    updated = backbone.snapshot()
    official = updated["backbone"]["training"]["counters"]
    if official["updates"] != updates_before+1 or official["batches"] != updates_before+1:
        raise ValueError("Probe must execute exactly one original learner update")
    return {"full_checkpoint_reload_equal": True, "model_optimizer_before_sha256": restored_model_hash,
            "next_initial_frame_sha256": tree_hash(initial_rgb), "next_action": action.tolist(),
            "next_action_sha256": tree_hash(action), "next_chain_sha256": tree_hash(chain.state_dict()),
            "next_mask": mask, "next_mask_state_sha256": tree_hash(dropout.state_dict()),
            "host_rng_draws": host_draws, "next_replay_batch_sha256": batch_hash,
            "next_replay_state_sha256": tree_hash(replay.snapshot()),
            "next_model_optimizer_sha256": backbone.parameter_hash(),
            "next_full_backbone_sha256": tree_hash(comparable_backbone(updated)),
            "next_update_counters": official, "finite_official_update": True}


def worker(checkpoint, upstream, output, phase):
    output = Path(output)
    work = output/phase
    work.mkdir(exist_ok=False)
    payload = load_checkpoint(checkpoint)
    if payload["metadata"]["identity"] != source_identity():
        raise ValueError("Recovery gate source differs from immutable smoke source")
    config = payload["config"]
    frame_path = output/"recorded_next_reset_rgb.npy"
    if phase == "reference":
        from .bridge import MinecraftEnv
        next_seed = payload["collector"]["next_episode_seed"]
        env = MinecraftEnv(seed=next_seed, **config["environment"],
                           diagnostics_path=str(work/"environment_diagnostics.jsonl"),
                           failure_log_path=str(work/"environment_failures.jsonl"))
        try:
            image = env.reset(seed=next_seed).reshape(64, 64, 3)
            with frame_path.open("xb") as file:
                np.save(file, image, allow_pickle=False)
        finally:
            env.close()
    else:
        image = np.load(frame_path, allow_pickle=False)
    backbone = MinecraftBackbone(config["backbone"], upstream, work)
    replay = Replay(Path(checkpoint).parent.parent/"REPLAY", config["replay_capacity_steps"],
                    seed=0, cacheepisodes=config.get("replay_cache_episodes", 4))
    result = software_probe(payload, backbone, replay, image)
    report = {"phase": phase, "utc": utc(), "task_namespace": TASK_NAMESPACE,
              "checkpoint_sha256": sha256(checkpoint), "identity": payload["metadata"]["identity"],
              "recorded_reset_frame_file_sha256": sha256(frame_path),
              "next_episode_seed": payload["collector"]["next_episode_seed"], "result": result,
              "engine_mid_episode_state_restored": False, "engine_reset_equality_tested": False}
    atomic_json(work/"report.json", report)
    return report


def orchestrate(checkpoint, upstream, output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    reports = []
    try:
        for phase in ("reference", "recovered_a", "recovered_b"):
            with (output/f"{phase}.log").open("x", encoding="utf-8") as log:
                subprocess.run([sys.executable, "-m", "helixwm_minecraft.preflight", "--checkpoint", str(checkpoint),
                                "--upstream", str(upstream), "--output-dir", str(output), "--phase", phase],
                               stdout=log, stderr=subprocess.STDOUT, check=True)
            reports.append(json.loads((output/phase/"report.json").read_text()))
        same = all(report["result"] == reports[0]["result"] for report in reports[1:])
        checkpoint_same = all(report["checkpoint_sha256"] == reports[0]["checkpoint_sha256"] for report in reports[1:])
        frame_same = all(report["recorded_reset_frame_file_sha256"] == reports[0]["recorded_reset_frame_file_sha256"]
                         for report in reports[1:])
        passed = same and checkpoint_same and frame_same
        result = {"passed": passed, "utc": utc(), "task_namespace": TASK_NAMESPACE,
                  "identity": reports[0]["identity"], "checkpoint_sha256": reports[0]["checkpoint_sha256"],
                  "full_checkpoint_reload": all(r["result"]["full_checkpoint_reload_equal"] for r in reports),
                  "fresh_process_boundary_restore": passed, "processes": 3,
                  "one_native_official_update_per_probe": True, "same_recorded_first_frame": frame_same,
                  "java_world_bitwise_equality_claimed": False, "engine_mid_episode_state_restored": False,
                  "scope": "full software state, action, replay crop, host RNG, missingness and next optimizer update",
                  "reports": [{"path": f"{r['phase']}/report.json", "sha256": sha256(output/r["phase"]/"report.json")}
                              for r in reports]}
        atomic_json(output/"software_boundary_restore.json", result)
        if not passed:
            raise RuntimeError("Fresh-process Minecraft software continuation differs")
        return result
    except BaseException as error:
        atomic_json(output/"failure.json", {"passed": False, "utc": utc(), "error": repr(error),
                                           "traceback": traceback.format_exc()})
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--upstream", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--phase", choices=("reference", "recovered_a", "recovered_b"))
    args = parser.parse_args()
    if args.phase:
        worker(Path(args.checkpoint).resolve(), Path(args.upstream).resolve(), args.output_dir, args.phase)
    else:
        orchestrate(Path(args.checkpoint).resolve(), Path(args.upstream).resolve(), args.output_dir)


if __name__ == "__main__":
    main()
