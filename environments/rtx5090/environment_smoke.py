"""Verify the HelixWM CUDA and headless Walker environment without training.

Run inside the selected environment with the HelixWM package installed:
    python environment_smoke.py --output /path/to/new/smoke_directory

This makes zero optimizer updates. The saved checkpoint contains initialized
model weights only, not replay, RNG, environment, or resumable training state.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import traceback
from datetime import datetime, timezone


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def check_cuda_and_model(output: Path) -> dict:
    import torch

    from helixwm.losses import cross_correction_loss, world_model_loss
    from helixwm.rssm import RSSM

    require(torch.cuda.is_available(), "CUDA is not available")
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.set_num_threads(1)
    device = torch.device("cuda:0")
    # Peak-memory counters require an initialized CUDA allocator in PyTorch 2.7.
    torch.cuda.init()
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    left = torch.randn(128, 128, device=device, requires_grad=True)
    right = torch.randn(128, 128, device=device, requires_grad=True)
    product = left @ right
    matmul_loss = product.square().mean()
    matmul_loss.backward()
    torch.cuda.synchronize(device)
    require(bool(torch.isfinite(product).all()), "CUDA matmul produced nonfinite values")
    require(bool(torch.isfinite(matmul_loss)), "CUDA matmul loss is nonfinite")
    require(left.grad is not None and bool(torch.isfinite(left.grad).all()),
            "CUDA left-input gradient is missing or nonfinite")
    require(right.grad is not None and bool(torch.isfinite(right.grad).all()),
            "CUDA right-input gradient is missing or nonfinite")

    # Exercise the convolution kernels that a future RGB encoder would need.
    # This isolated operator check does not change the reference RSSM design.
    convolution = torch.nn.Conv2d(3, 8, kernel_size=3, padding=1).to(device)
    convolution_parameters = {name: value.detach().clone()
                              for name, value in convolution.state_dict().items()}
    rgb_input = torch.randn(2, 3, 64, 64, device=device, requires_grad=True)
    rgb_features = convolution(rgb_input)
    convolution_loss = rgb_features.square().mean()
    convolution_loss.backward()
    torch.cuda.synchronize(device)
    require(bool(torch.isfinite(rgb_features).all()) and bool(torch.isfinite(convolution_loss)),
            "CUDA convolution output or loss is nonfinite")
    require(rgb_input.grad is not None and bool(torch.isfinite(rgb_input.grad).all()),
            "CUDA convolution input gradient is missing or nonfinite")
    require(all(parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
                for parameter in convolution.parameters()),
            "CUDA convolution parameter gradient is missing or nonfinite")
    require(all(torch.equal(convolution_parameters[name], value)
                for name, value in convolution.state_dict().items()),
            "Convolution parameters changed despite the zero-update requirement")

    dimensions = dict(obs_dim=5, action_dim=2, hidden_dim=32,
                      groups=4, classes=4, embed_dim=32)
    model = RSSM(**dimensions).to(device)
    initial_parameters = {name: tensor.detach().cpu().clone()
                          for name, tensor in model.state_dict().items()}
    batch, steps = 2, 6
    observations = torch.randn(batch, steps + 1, dimensions["obs_dim"], device=device)
    actions = torch.tanh(torch.randn(batch, steps, dimensions["action_dim"], device=device))
    rewards = torch.randn(batch, steps + 1, device=device)
    continuations = torch.ones_like(rewards)
    valid = torch.ones_like(rewards, dtype=torch.bool)
    is_first = torch.zeros_like(valid)
    is_first[:, 0] = True
    real = model.observe(observations, actions, is_first, sample=False)
    base = world_model_loss(model, real, observations, rewards, continuations, valid, is_first)
    correction = cross_correction_loss(model, real, actions, is_first, valid,
                                       max_horizon=3, sample=False)
    total = base["total"] + 0.1 * correction
    require(bool(torch.isfinite(total)), "RSSM loss is nonfinite")
    total.backward()
    gradients = {name: parameter.grad for name, parameter in model.named_parameters()}
    require(bool(gradients), "RSSM has no parameters")
    require(all(gradient is not None for gradient in gradients.values()),
            "At least one RSSM parameter did not receive a gradient")
    require(all(bool(torch.isfinite(gradient).all()) for gradient in gradients.values()),
            "RSSM produced nonfinite gradients")
    require(all(torch.equal(initial_parameters[name], tensor.detach().cpu())
                for name, tensor in model.state_dict().items()),
            "A model parameter changed despite the zero-update requirement")

    checkpoint = output / "initialized_model_state_dict.pt"
    torch.save({"model_state_dict": model.state_dict(), "model_dimensions": dimensions,
                "optimizer_steps": 0, "purpose": "environment_smoke_only"}, checkpoint)
    checkpoint_hash = sha256(checkpoint)
    restored_payload = torch.load(checkpoint, map_location=device, weights_only=True)
    restored = RSSM(**restored_payload["model_dimensions"]).to(device)
    restored.load_state_dict(restored_payload["model_state_dict"], strict=True)
    model.eval()
    restored.eval()

    def deterministic_outputs(net):
        observed = net.observe(observations, actions, is_first, sample=False)
        imagined = net.imagine(observed.states[2].detach(), actions[:, 2:5], sample=False)
        tensors = [observed.prior_logits, observed.posterior_logits, imagined.prior_logits]
        for state in (*observed.states, *imagined.states):
            tensors.extend((state.h, state.z, *net.decode(state).values()))
        return tensors

    with torch.no_grad():
        before, after = deterministic_outputs(model), deterministic_outputs(restored)
    require(len(before) == len(after) and all(torch.equal(a, b) for a, b in zip(before, after)),
            "Reloaded model does not produce bitwise-identical deterministic outputs")
    require(checkpoint_hash == sha256(checkpoint), "Checkpoint checksum changed during reload")
    require(all(torch.equal(initial_parameters[name], tensor.detach().cpu())
                for name, tensor in restored.state_dict().items()),
            "Reloaded model differs from initialized model parameters")
    torch.cuda.synchronize(device)
    return {
        "status": "PASS", "device": str(device),
        "device_name": torch.cuda.get_device_name(device),
        "device_capability": list(torch.cuda.get_device_capability(device)),
        "compiled_architectures": torch.cuda.get_arch_list(),
        "torch_version": torch.__version__, "torch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "gpu_total_memory_bytes": torch.cuda.get_device_properties(device).total_memory,
        "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "matmul": {"shape": list(product.shape), "loss": matmul_loss.detach().item(),
                   "outputs_and_gradients_finite": True},
        "convolution": {
            "scope": "isolated synthetic RGB operator check; not an RSSM architecture change",
            "input_shape": list(rgb_input.shape), "output_shape": list(rgb_features.shape),
            "kernel_size": list(convolution.kernel_size),
            "loss": convolution_loss.detach().item(),
            "outputs_input_and_parameter_gradients_finite": True,
            "parameters_unchanged": True, "optimizer_steps": 0,
        },
        "rssm": {
            "data_source": "synthetic vector observations; not Walker observations",
            "observation_shape": list(observations.shape), "action_shape": list(actions.shape),
            "prior_logits_shape": list(real.prior_logits.shape),
            "losses": {name: loss.detach().item() for name, loss in base.items()},
            "cross_correction_loss": correction.detach().item(),
            "combined_loss": total.detach().item(),
            "all_parameter_gradients_present_and_finite": True,
            "parameters_unchanged": True, "optimizer_steps": 0,
        },
        "checkpoint": {
            "file": checkpoint.name, "bytes": checkpoint.stat().st_size,
            "sha256": checkpoint_hash, "strict_state_dict_reload": "PASS",
            "deterministic_sample_false_output_match": "bitwise_exact",
            "compared_output_tensors": len(before), "optimizer_steps": 0,
            "contents": "initialized model state_dict and configuration only",
            "exact_training_resume_supported": False,
            "missing_training_state": ["optimizer", "replay", "RNG", "environment", "training_progress"],
        },
    }


def check_walker(output: Path, control_steps: int) -> dict:
    # These variables apply only to this verification process and its imports.
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    import numpy as np
    from dm_control import suite
    from PIL import Image

    environment = suite.load("walker", "walk", task_kwargs={"random": 0})
    try:
        timestep = environment.reset()
        specification = environment.action_spec()
        action = np.zeros(specification.shape, dtype=specification.dtype)
        require(bool(np.all(action >= specification.minimum) and np.all(action <= specification.maximum)),
                "Zero action is outside Walker action bounds")
        initial_time = float(environment.physics.data.time)
        rewards = []
        for _ in range(control_steps):
            require(not timestep.last(), "Walker episode ended before requested smoke steps")
            timestep = environment.step(action)
            require(timestep.reward is not None and bool(np.isfinite(timestep.reward)),
                    "Walker reward is missing or nonfinite")
            require(all(bool(np.isfinite(value).all()) for value in timestep.observation.values()),
                    "Walker observation contains nonfinite values")
            rewards.append(float(timestep.reward))
        pixels = environment.physics.render(height=64, width=64, camera_id=0)
        require(pixels.shape == (64, 64, 3) and pixels.dtype == np.uint8,
                "Unexpected Walker render shape or dtype")
        require(float(pixels.std()) > 0, "Walker rendering produced a constant image")
        image_file = output / "walker_walk_egl_smoke.png"
        Image.fromarray(pixels).save(image_file)
        with Image.open(image_file) as decoded:
            decoded.load()
            require(decoded.size == (64, 64) and decoded.mode == "RGB",
                    "Saved render did not decode as a 64x64 RGB PNG")
        return {
            "status": "PASS", "domain": "walker", "task": "walk", "seed": 0,
            "actions": "zero actions only", "control_step_calls": control_steps,
            "counting_scope": "environment smoke only; not training frames or an experiment budget",
            "simulation_time_start_seconds": initial_time,
            "simulation_time_end_seconds": float(environment.physics.data.time),
            "control_timestep_seconds": float(environment.control_timestep()),
            "physics_timestep_seconds": float(environment.physics.timestep()),
            "action_shape": list(specification.shape),
            "observation_shapes": {key: list(value.shape) for key, value in timestep.observation.items()},
            "rewards": rewards, "episode_ended": bool(timestep.last()),
            "render": {"requested_backend": "egl", "file": image_file.name,
                       "shape": list(pixels.shape), "dtype": str(pixels.dtype),
                       "pixel_standard_deviation": float(pixels.std()),
                       "saved_png_decode": "PASS", "sha256": sha256(image_file)},
            "trained_policy_evaluation": False,
        }
    finally:
        # dm_control 1.0.28 inherits a no-op Environment.close(). Physics.free()
        # explicitly releases its MuJoCo and EGL rendering contexts before
        # interpreter shutdown; the physics object is never used afterward.
        environment.physics.free()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True,
                        help="New output directory; existing nonempty directories are rejected")
    parser.add_argument("--control-steps", type=int, default=4)
    arguments = parser.parse_args()
    if not 1 <= arguments.control_steps <= 20:
        parser.error("--control-steps must be between 1 and 20")
    output = arguments.output.expanduser().resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error("--output must be absent or an empty directory")
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": 1,
        "scope": "environment validation only; no formal training or policy evaluation",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version, "executable": sys.executable,
        "platform": platform.platform(), "output_directory": str(output),
        "script_sha256": sha256(Path(__file__)), "optimizer_steps": 0,
        "versions": {name: package_version(name) for name in
                     ("torch", "dm-control", "mujoco", "numpy", "Pillow", "helixwm-reference")},
        "checks": {},
    }
    for name, operation in (
        ("cuda_model_checkpoint", lambda: check_cuda_and_model(output)),
        ("walker_headless", lambda: check_walker(output, arguments.control_steps)),
    ):
        try:
            report["checks"][name] = operation()
        except Exception as error:
            report["checks"][name] = {
                "status": "FAIL", "error_type": type(error).__name__,
                "error": str(error), "traceback": traceback.format_exc(),
            }
    report["status"] = "PASS" if all(check["status"] == "PASS" for check in report["checks"].values()) else "FAIL"
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    serialized = json.dumps(report, indent=2, ensure_ascii=True)
    (output / "environment_smoke_report.json").write_text(serialized + "\n", encoding="utf-8")
    print(serialized, flush=True)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
