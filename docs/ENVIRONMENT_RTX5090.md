# HelixWM RTX 5090 Environment Runbook

Verified: 2026-10-13. Scope: isolated environment preparation and bounded smoke verification. No formal training was started.

## Current verification status

The isolated environment is installed and the final verification **passed**. The final smoke ran from `2026-10-13T10:28:24.747696Z` to `2026-10-13T10:28:27.422880Z`; its report is in `smoke/final_validation/environment_smoke_report.json`, its log is `logs/environment-smoke-final.log`, and `smoke-final.exit` contains `0`. This establishes environment readiness for development, not a completed training system or experiment.

| Check | Status | Evidence to retain |
|---|---|---|
| Isolated environment installation | PASS: exit 0 and `ENVIRONMENT_INSTALL_PASS` | `logs/setup.log`, `logs/setup.exit` |
| Dependency consistency | PASS | `logs/pip-check.log` |
| Reference unit tests | PASS: 30 tests in 0.269 s on server CPU | `logs/unit-tests.log` |
| Synthetic reference demonstration | PASS: zero optimizer steps | `logs/synthetic-trace.log` |
| CUDA matrix multiplication and backward pass | PASS: 128 x 128 output; finite gradients | Final smoke JSON report |
| Isolated CUDA convolution and backward pass | PASS: input 2 x 3 x 64 x 64, output 2 x 8 x 64 x 64; finite gradients | Final smoke JSON report |
| RSSM loss and gradient checks on CUDA | PASS: finite gradients, unchanged weights | Final smoke JSON report |
| Initialized model save/reload with zero updates | PASS: 58 output tensors bitwise equal | Final smoke JSON report and SHA256 |
| Walker reset/step and EGL RGB rendering | PASS: 4 zero-action control steps, decoded 64 x 64 RGB PNG | Final smoke JSON report and PNG |

The combined RSSM smoke loss was `4.9467597007751465`. This loss is from initialized weights and synthetic inputs; it is not a learning-performance result. The initialized state-dictionary artifact has SHA256 `f5c9fdfb8b7aeb2dfa4accc2e49d25576ed68fdaf889cbe579bbf2ee1bb84428c`.

The first smoke attempt failed before CUDA computation because the instrumentation called `torch.cuda.reset_peak_memory_stats` before lazy CUDA initialization; its independent Walker check passed. The initial report is preserved under `smoke/initial_validation`, and the original script is preserved as `metadata/environment_smoke_first_attempt.py`. The repair explicitly calls `torch.cuda.init()` and `torch.cuda.set_device(device)` before resetting memory statistics. Only verification instrumentation changed; HelixWM model source was unchanged. The separate final smoke above passed after this repair and includes an isolated convolution check.

## Compatibility decision

The supplied base environment has Python 3.10.8 and PyTorch 2.1.2+cu121. A real CUDA operation failed with `CUDA error: no kernel image is available for execution on the device`. The RTX 5090 reports compute capability 12.0 (`sm_120`), while that PyTorch build lists older GPU architectures. A successful `torch.cuda.is_available()` call alone is therefore insufficient.

A separate environment was created on the data disk, preserving the existing base environment and NVIDIA driver. The observed driver is 580.105.08; no driver upgrade was performed. PyTorch 2.7 introduced Blackwell support with CUDA 12.8 wheels. The installed wheel supplies its CUDA runtime; the base system's CUDA 12.1 compiler was not replaced for these checks, which build no custom CUDA extensions.

The instance allocation was observed as one RTX 5090 with about 32 GB VRAM, 25 CPU cores, a 90 GiB cgroup memory limit, and a 50 GB data disk. These are an environment snapshot, not a demonstrated requirement or capacity estimate for future HelixWM training. Use the cgroup limit when checking instance memory; host-wide `free` output can be misleading.

## Pinned environment and source

| Component | Installed version |
|---|---|
| Python | 3.12.12 |
| PyTorch | 2.7.1+cu128, official CUDA 12.8 wheel index |
| PyTorch CUDA runtime / cuDNN | 12.8 / reported version integer 90701 |
| NumPy | 2.2.6 |
| SciPy | 1.15.3 |
| dm_control | 1.0.28 |
| MuJoCo | 3.3.0 |
| Pillow | 11.2.1 |
| imageio | 2.37.0 |
| imageio-ffmpeg | 0.6.0 |
| pip / setuptools / wheel | 25.2 / 80.9.0 / 0.45.1 |

The final GPU check reported compute capability 12.0 (`sm_120`). The dm_control 1.0.28 release explicitly upgrades to MuJoCo 3.3.0; both are pinned to preserve that pairing. dm_control was installed as its released package, not as an editable checkout. HelixWM itself is installed as an editable package at a fixed source commit, without modifying its source.

- Repository: <https://github.com/iceezhu306-lgtm/HelixWM-for-world-model>
- Source commit: `745231d9a2049c7945a78aa041a2ca845580a6ca`
- Checkout mode: detached HEAD.

The [observed package freeze](../environments/rtx5090/pip-freeze-observed.txt) records all resolved Python dependencies. The [final smoke report](../environments/rtx5090/validation-report.json) preserves the measured checks. When rebuilding, install the CUDA wheel from the official `cu128` index before using the package freeze; the default PyPI index alone does not identify this GPU wheel source.

Do not install the repository's `requirements-test.txt` over this GPU environment: it describes the earlier local CPU validation and pins PyTorch 2.6.0. This runbook deliberately records a separate GPU compatibility environment.

## Data disk layout

```text
/root/autodl-tmp/helixwm/
  env/        # Independent Conda environment
  source/     # Fixed HelixWM checkout
  scripts/    # Setup, activation, and environment smoke scripts
  metadata/   # Package locks, source identity, GPU and system package records
  logs/       # Setup and verification logs, exit markers
  smoke/      # Bounded verification outputs; not experiment results
  cache/      # Remote Conda, pip, and temporary installation files
```

The setup script redirects package caches and temporary installation files to this root. Do not copy the Conda environment or package caches into experiment archives. Retain small manifests and reproduction commands instead.

## Installation and tmux

The setup requires `tmux`, `ffmpeg`, `libegl1`, `libgl1`, and `libglew2.2` on this Ubuntu 22.04 host. Install only missing required system packages; do not perform a general system upgrade. No display server is required for the intended EGL check.

The versioned scripts are [setup_environment.sh](../environments/rtx5090/setup_environment.sh), [activate_helixwm.sh](../environments/rtx5090/activate_helixwm.sh), and [environment_smoke.py](../environments/rtx5090/environment_smoke.py). For a fresh installation, place them in the remote `scripts` directory before use. Inspect existing processes, directories, and tmux sessions before launching setup to avoid duplicate work. The recorded installation is already complete; the following setup command is for reproduction on a fresh environment.

```bash
tmux new-session -d -s helixwm_setup \
  'bash /root/autodl-tmp/helixwm/scripts/setup_environment.sh'
tmux list-sessions
tmux capture-pane -p -t helixwm_setup -S -80
tail -n 80 /root/autodl-tmp/helixwm/logs/setup.log
```

The script writes `logs/setup.exit` and `logs/setup.finished_utc` when it exits. A zero exit marker and the final `ENVIRONMENT_INSTALL_PASS` line indicate that the installation script completed; they do not replace the GPU and simulator checks below. A tmux session may disappear normally after its command exits. tmux protects a process from SSH disconnection, not from instance shutdown or restart.

The setup script intentionally refuses to run if `env` already exists. If setup stops partway through, inspect its log and repair the specific failed stage. Do not rerun blindly, delete the environment automatically, or start a second installer.

## Activation

```bash
source /root/autodl-tmp/helixwm/scripts/activate_helixwm.sh
python --version
python -c 'import torch; print(torch.__version__, torch.version.cuda)'
```

Activation selects `/root/autodl-tmp/helixwm/env`, enters the source checkout, and sets `MUJOCO_GL=egl`, `PYOPENGL_PLATFORM=egl`, and `MUJOCO_EGL_DEVICE_ID=0` before simulator imports. It starts no training. These variables apply to the current shell and its child processes; they do not modify the base environment globally.

## Bounded verification commands

These commands reproduce the bounded checks after installation. Preserve both stdout/stderr and command exit status. Use fresh log names or copy existing evidence first when repeating a check.

```bash
source /root/autodl-tmp/helixwm/scripts/activate_helixwm.sh
set -o pipefail
python -m pip check 2>&1 | tee /root/autodl-tmp/helixwm/logs/pip-check.log
python -m unittest discover -s tests -v 2>&1 | tee /root/autodl-tmp/helixwm/logs/unit-tests.log
python -m examples.synthetic_trace 2>&1 | tee /root/autodl-tmp/helixwm/logs/synthetic-trace.log
SMOKE_OUTPUT="/root/autodl-tmp/helixwm/smoke/$(date -u +%Y%m%dT%H%M%SZ)"
python /root/autodl-tmp/helixwm/scripts/environment_smoke.py \
  --output "$SMOKE_OUTPUT" --control-steps 4 \
  2>&1 | tee /root/autodl-tmp/helixwm/logs/environment-smoke.log
```

The pinned source contains 30 unit tests; all passed in 0.269 seconds on the server CPU. This recorded duration is not a benchmark or runtime guarantee. The earlier local CPU validation in the repository remains separate historical evidence. The remote synthetic demonstration also passed, using synthetic data and zero optimizer steps.

The smoke script performs real CUDA matrix multiplication and backpropagation, an isolated convolution forward/backward check, and an RSSM forward/backward check with synthetic vector observations. The convolution exercises GPU kernels useful to a future RGB encoder; it does not implement or validate a complete pixel encoder. The script requires finite losses and gradients and unchanged model parameters. It then saves an initialized model state dictionary, strictly reloads it, checks SHA256, and compares deterministic `sample=False` outputs exactly. All of these checks passed in the final run.

The checkpoint is **an initialized-model smoke artifact**, not a trained model or a resumable training checkpoint. No optimizer step is taken. It contains no optimizer state, replay buffer, RNG state, environment state, or training progress. A successful round trip cannot establish exact training recovery or frozen trained-policy evaluation.

The separate Walker check loads `walker/walk` with seed 0, resets, applies four zero-action control steps, validates finite observations/rewards, renders a 64 x 64 RGB image with EGL, and decodes the saved PNG. Its control-step count is not a training-frame budget. It is not a policy evaluation or evidence of learning. The simulator is explicitly closed after this check.

The output directory must be absent or empty. The script writes `environment_smoke_report.json`, `initialized_model_state_dict.pt`, and `walker_walk_egl_smoke.png`. A failed component remains visible in the JSON even if another component succeeds. Record the actual overall status; do not infer success from file existence alone.

## Evidence and limits

Retain the setup/verification scripts, complete command logs and exit codes, `pip-freeze.txt`, `conda-explicit.txt`, `conda-environment.yml`, `source-commit.txt`, `source-status.txt`, `nvidia-smi.txt`, `system-packages.txt`, and smoke outputs with SHA256 values. Package manifests record the resolved installation in addition to the intended pins. Keep connection credentials and private keys out of these records and out of GitHub.

The current repository implements mathematical reference components. It does not yet provide a complete pixel encoder, environment/replay training loop, full actor/critic optimization pipeline, or formal checkpoint/replay/RNG recovery. Successful environment verification does not make it a ready-to-run Walker training system. Formal training configuration, budgets, checkpoint policy, evaluation, learning curves, and multi-seed claims require a later implementation and experiment decision.

No baseline instance, existing experiment archive, training process, seed schedule, or recurring monitor is changed by this runbook.

## Official compatibility references

- [PyTorch 2.7: Blackwell support and CUDA 12.8 wheels](https://pytorch.org/blog/pytorch-2-7/)
- [Official versioned PyTorch installation commands](https://pytorch.org/get-started/previous-versions/)
- [CUDA 12.8 toolkit release notes and driver compatibility](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-toolkit-release-notes/index.html)
- [dm_control 1.0.28 release: MuJoCo 3.3.0 pairing](https://github.com/google-deepmind/dm_control/releases/tag/1.0.28)
- [dm_control 1.0.28 README: rendering and installation](https://github.com/google-deepmind/dm_control/blob/1.0.28/README.md)

