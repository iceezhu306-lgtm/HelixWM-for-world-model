# Real Minecraft visual-navigation transfer protocol

Date: 2026-10-18. **The user selected real Minecraft.** The superseded driving-scene drafts have been removed. No CARLA/CarRacing choice or installation is needed for this stage.

Status: **SCENE_SELECTED_ENVIRONMENT_SMOKE_PENDING**. The local worker/bridge and task specification exist and have passed 14 protocol tests using an explicitly synthetic test backend. MineRL installation, real Java/OpenGL rendering, target visibility, learner integration, training, and formal evaluation are not established by those tests. No result or completed training is claimed in this protocol.

The design trains one new seed-0 policy for **1,000,000 action decisions**, then evaluates six inference interventions using the identical final checkpoint. It transfers the method to a different task, not Walker model weights. Same RGB resolution and six-dimensional actor output do not make a Walker checkpoint a Minecraft policy. This is one training seed; it does not establish multi-seed efficacy or paper-level reproduction.

## 1. Engine and task identity

Use **MineRL 0.4.4** with its actual Minecraft/Malmo engine. Its pinned source is `ccc8c659d2b8e4737eda418593dee94c101c9ba3`. The packaged build specifies Minecraft **1.11.2**, Malmo **0.37.0**, Java 8 and Gradle **4.10.2**. Do not replace it with Crafter, a prerecorded video, a fake MineRL entrypoint or a new renderer and call that Minecraft. [Official tag](https://github.com/minerllabs/minerl/tree/v0.4.4), [Minecraft build](https://github.com/minerllabs/minerl/blob/v0.4.4/minerl/Malmo/Minecraft/build.gradle), [Gradle wrapper](https://github.com/minerllabs/minerl/blob/v0.4.4/minerl/Malmo/Minecraft/gradle/wrapper/gradle-wrapper.properties).

The native `MineRLNavigateDense-v0` is a useful engine smoke task: RGB64 navigation toward a diamond block about 64 meters away, with a compass observation, native touch reward and distance shaping. Its target can be obscured or slightly below the surface. Removing the compass does not guarantee a visually observable goal. It is therefore not silently reused as a purported visible-goal benchmark. [Native task source](https://github.com/minerllabs/minerl/blob/v0.4.4/minerl/herobraine/env_specs/navigate_specs.py), [native tutorial](https://minerl.readthedocs.io/en/v0.4.4/tutorials/first_agent.html).

The implemented custom specification is **`HelixWMVisibleNavigation-v0`** in [task.py](../../helixwm_minecraft/task.py), built through MineRL's documented custom-environment handlers and `make(fake=False)`:

- Real Minecraft flat terrain, clear fixed daylight, no spawned mobs, and empty inventory with **no compass item**.
- Spawn `(0.5, 4.0, 0.5)`, yaw/pitch zero. A three-block-high diamond pillar at integer x in [-6,6], z in [12,20], y=4..6. The episode seed deterministically selects the target coordinates using SHA256; the complete manifest is hashed and recorded.
- The geometry is intended to put the target in the initial forward view. **A rendered RGB check must verify this**; geometry calculations alone do not establish pixel visibility or correct camera convention.
- Native reward +100 and termination for touching diamond. Additional dense reward is the previous minus current horizontal distance to the target center, computed inside the worker. The sum of this shaping term telescopes, rather than rewarding repeated forward/back motion independently.
- Maximum 1,000 acknowledged native MineRL steps. No attack, block placement or crafting action is exposed in this initial navigation wrapper. This is a custom visual-navigation development task, not native MineRL benchmark equivalence, full Minecraft competence or a survival benchmark.

The initial layout has a finite set of target positions. Different reset seeds may generate identical target geometry. Held-out seed evaluation alone therefore does **not** establish unseen-map/geometry generalization. Record geometry as well as full manifest hashes. Later obstacle, terrain or map families require a separate predeclared task/split; do not change the task after inspecting favorable outcomes. [Official custom-environment tutorial](https://minerl.readthedocs.io/en/v0.4.4/tutorials/custom_environments.html).

## 2. Isolated installation proposal

Keep the existing Python 3.12/JAX learner environment unchanged. Run Minecraft in an independent **Python 3.9.23** worker, with proposed compatibility pins `numpy==1.23.5`, `gym==0.19.0`, `pillow==9.5.0`, `opencv-python==4.8.1.78`, `minerl==0.4.4`. NumPy 1.23.5 retains legacy aliases used in MineRL. Old Gym packaging needs an isolated older packaging toolchain; proposed pins are pip 23.0.1, setuptools 65.5.0 and wheel 0.38.4. These are proposed deployment pins, not a claim of a tested installation. Save the final complete lock and every compatibility deviation. [MineRL requirements](https://github.com/minerllabs/minerl/blob/v0.4.4/requirements.txt).

Verified official distribution metadata:

| Distribution | Bytes | SHA256 |
|---|---:|---|
| `minerl-0.4.4.tar.gz` | 70,117,418 | `8cdaefade9eac20177044cb87c009fe5b539ef7dd74c59f4103a8b161ae19580` |
| Temurin `OpenJDK8U-jdk_x64_linux_hotspot_8u462b08.tar.gz` | 103,087,414 | `5d64ae542b59a962b3caadadd346f4b1c3010879a28bb02d928326993de16e79` |

Sources: [MineRL PyPI](https://pypi.org/project/minerl/0.4.4/), [Temurin release](https://github.com/adoptium/temurin8-binaries/releases/tag/jdk8u462-b08), [official JDK checksum](https://github.com/adoptium/temurin8-binaries/releases/download/jdk8u462-b08/OpenJDK8U-jdk_x64_linux_hotspot_8u462b08.tar.gz.sha256.txt). Use the full JDK, not only a JRE. This old simulator needs Java 8; do not change the system-wide Java default or the verified learner environment.

The following commands are an explicit **installation recipe, not an execution log**. Resolve existing packages and free space before executing; keep all task-specific temporary files under the new work area. Conda uses only conda-forge here, without enabling defaults or accepting unrelated terms.

```bash
export MC_ROOT=/root/helixwm_minecraft
mkdir -p "$MC_ROOT"/{tmp,downloads,java8,conda_pkgs,logs}
export TMPDIR="$MC_ROOT/tmp"
export CONDA_PKGS_DIRS="$MC_ROOT/conda_pkgs"
conda create --override-channels -c conda-forge -p "$MC_ROOT/env_py39" python=3.9.23 -y
curl --fail --location --retry 2 -o "$MC_ROOT/downloads/jdk8.tar.gz" \
  https://github.com/adoptium/temurin8-binaries/releases/download/jdk8u462-b08/OpenJDK8U-jdk_x64_linux_hotspot_8u462b08.tar.gz
printf '%s  %s\n' 5d64ae542b59a962b3caadadd346f4b1c3010879a28bb02d928326993de16e79 \
  "$MC_ROOT/downloads/jdk8.tar.gz" | sha256sum --check -
tar -xzf "$MC_ROOT/downloads/jdk8.tar.gz" -C "$MC_ROOT/java8" --strip-components=1
export JAVA_HOME="$MC_ROOT/java8"
export PATH="$JAVA_HOME/bin:$PATH"
java -version
javac -version
"$MC_ROOT/env_py39/bin/python" -m pip install pip==23.0.1 setuptools==65.5.0 wheel==0.38.4
"$MC_ROOT/env_py39/bin/python" -m pip install --no-build-isolation \
  numpy==1.23.5 gym==0.19.0 pillow==9.5.0 opencv-python==4.8.1.78 minerl==0.4.4
"$MC_ROOT/env_py39/bin/python" -m pip check
```

Where absent, system dependencies for the isolated headless worker include `xvfb`, `xauth`, `libgl1-mesa-dri`, `libglx-mesa0`, `libglu1-mesa`, `mesa-utils`, `libxrender1`, `libxtst6`, `libxi6`, and compiler/build tools. Record exact installed versions. Use `MINERL_HEADLESS=1` and `LIBGL_ALWAYS_SOFTWARE=1` initially. Confirm the actual OpenGL renderer with `xvfb-run -a glxinfo -B`, then an actual MineRL reset/step image. This path is **Xvfb/Mesa OpenGL, not Vulkan**; the RTX 5090 remains available to JAX. Software rendering is slower than GPU rendering, so measure end-to-end throughput. Do not modify host drivers/Xorg or install a VirtualGL stack automatically. [Official headless instructions](https://minerl.readthedocs.io/en/v0.4.4/tutorials/first_agent.html), [official performance limits](https://minerl.readthedocs.io/en/v0.4.4/notes/performance-tips.html).

Installation builds a Java fat JAR through Gradle and may fetch Minecraft/Forge/Maven assets. Old external repositories can be unavailable; preserve the exact error and pin any necessary build-only repair. Do not skip the build, substitute the fake backend, or claim success from `pip install` alone. A full real reset, native steps and nonempty RGB are required. No MineRL human demonstration dataset is needed or requested. [Build entrypoint](https://github.com/minerllabs/minerl/blob/v0.4.4/setup.py), [launcher](https://github.com/minerllabs/minerl/blob/v0.4.4/minerl/Malmo/Minecraft/launchClient.sh).

The real-engine-only [environment smoke script](../../scripts/minecraft_environment_smoke.py) accepts the isolated worker interpreter and creates a new output directory (it refuses to overwrite one). After exporting the task's verified `JAVA_HOME`, `PATH` and `TMPDIR`, run from the uploaded repository:

```bash
MINERL_HEADLESS=1 LIBGL_ALWAYS_SOFTWARE=1 \
  /root/helixwm_minecraft/env_py39/bin/python scripts/minecraft_environment_smoke.py \
  --worker-python /root/helixwm_minecraft/env_py39/bin/python \
  --output /root/helixwm_minecraft/smoke/visible_nav_001 \
  --seed 123 --steps 8 --timeout-seconds 1200
```

It performs zero optimizer updates and produces initial/final/repeated-reset PNGs, a contact sheet, actual installed runtime metadata, action/native-step traces and the evaluator-only sidecar. A passing reset/step report still says `VISUAL_REVIEW_PENDING`; inspect the contact sheet to establish that the real target is visible. The same-seed image equality field is measured and must not be assumed. Repeat with the existing learner interpreter as the outer process to verify cross-Python IPC without modifying that environment.

## 3. Capacity gate

The coordinating agent's deployment snapshot is approximately 30 GiB free on the system volume and 9.5 GiB on the data volume, with a proposed new-task system-storage budget of approximately 25 GiB, a 90 GiB RAM allowance and one 32 GB RTX 5090. These are snapshots, not continuing guarantees. CARLA's former package calculations do not apply to Minecraft.

The 70 MB MineRL distribution and 103 MB JDK archive are **not installed footprints**. Gradle/assets, build copies, the separate Python environment and runtime temporary Minecraft copies must be measured. MineRL's instance manager copies its Minecraft directory when starting an instance; account for that peak and ensure `TMPDIR` is on the intended volume. [Instance manager source](https://github.com/minerllabs/minerl/blob/v0.4.4/minerl/env/malmo.py).

One million single-copy RGB64 uint8 arrays alone contain 12,288,000,000 bytes, or **11.444092 GiB uncompressed**. The full run also needs actions/rewards, replay indices, checkpoint/recovery state, logs, evaluation traces and packaging headroom. Measure bytes per representative transition and checkpoint, then project the total across the system/data volumes before formal launch. Keep a declared reserve and low-space response. Do not assume compression will make the budget fit, delete previous experiments, expand storage, reduce the training budget or checkpoint cadence silently. A short environment smoke is not evidence of sufficient storage for the full experiment.

## 4. Worker and learner contracts

[worker.py](../../helixwm_minecraft/worker.py) owns the real MineRL environment. [bridge.py](../../helixwm_minecraft/bridge.py) runs in the current learner interpreter without importing MineRL, Gym or Java. Communication uses local stdin/stdout pipes, a 4-byte big-endian length prefix and versioned JSON. RGB64 bytes are base64-encoded; no Python pickle or network service is used for this bridge. MineRL's own local engine connection remains an internal simulator detail.

The worker duplicates the protocol output descriptor, then redirects **both OS stdout fd 1 and Python stdout** to stderr before importing simulator modules. Console/native-library chatter therefore cannot be interpreted as protocol data. Frames are bounded to 1 MiB, schema fields are whitelisted, request IDs/seeds are checked, nonfinite/out-of-bounds actions are rejected, and protocol/environment failures close the service. Timeouts and failure records are persisted. Cleanup targets the owned worker/process group and its descendants only; never kill unrelated Java processes by name. Cleanup errors must not hide an existing training/evaluation failure.

`MinecraftEnv(seed=..., action_repeat=2, image_size=64, episode_env_steps=1000, worker_python=..., diagnostics_path=...)` exposes:

- `reset(seed=exact_episode_seed)`: flat uint8 RGB with 12,288 elements. An omitted seed repeats the current seed; the trainer must explicitly assign/save each episode seed.
- `step(action)`: flat RGB, summed reward, `terminated`, `truncated`, `done`, actual native `env_steps`, episode seed and continuation. True task termination gives continuation 0; time-limit truncation gives continuation 1. Repeat stops immediately on termination or the native limit.
- `render(size=256)`: deterministic nearest-neighbor enlargement of the **last returned image**, with no additional simulator observation. It cannot accidentally expose a newer hidden frame.
- `snapshot()` only at a reset boundary: a task/seed/reset descriptor explicitly labeled `exact_engine_state=False`. Mid-episode snapshot is rejected. `restore()` recreates that reset; bitwise engine reproducibility still requires live validation.

The six normalized continuous actor outputs have **new Minecraft meanings**:

| Component | Mapping |
|---|---|
| 0 | Forward if >0.2, backward if <-0.2, otherwise neither |
| 1 | Strafe right if >0.2, left if <-0.2, otherwise neither |
| 2 | Yaw change up to 15 degrees per action decision |
| 3 | Pitch change up to 10 degrees per action decision |
| 4 | Jump if >0 |
| 5 | Sprint if >0 |

MineRL camera order is `[pitch, yaw]`. Divide the requested rotation by repeat2 and apply it once per native step; hold the keyboard controls. Attack/sneak remain zero and inventory/place actions are not exposed. Retaining a six-dimensional head is architectural reuse, not action-semantic equivalence to Walker or full native MineRL action coverage.

Only RGB, reward, terminal/truncation flags, task identity, seeds and native-step counters cross the observation bridge. Raw `compass`, `inventory`, `location_stats`, goal coordinates and other `info` fields do not. The task uses privileged coordinates internally for dense reward; per-native-step position, distance, native/progress rewards, path length and success are written to a **separate evaluator/audit-only JSONL**. Learner policy inference and repair must not read that file or receive these fields. Synthetic visual faults are applied outside the worker; raw simulator images remain immutable evidence and cannot leak into online decisions during a gap.

`env_steps` and the legacy field name `episode_ticks` count **acknowledged synchronous MineRL step calls**, not an instrumented count of all Minecraft world ticks. The official Python implementation issues one client and one server step with no hidden frame-skip. Java requests one synchronous client tick and normally one server tick; it can skip the server tick after termination. Reset/peek also executes warm-up ticks outside this counter. Thus 0.05 seconds per native step and 10 Hz at repeat2 are **nominal simulated time**, not measured wall-clock or total engine elapsed time. Preserve this distinction in plots, metadata and comparisons. [Python step protocol](https://github.com/minerllabs/minerl/blob/v0.4.4/minerl/env/_multiagent.py), [Java synchronous stepping and reset warm-up](https://github.com/minerllabs/minerl/blob/v0.4.4/minerl/Malmo/Minecraft/src/main/java/com/microsoft/Malmo/Client/MalmoEnvServer.java), [nominal timing and synchronization](https://github.com/minerllabs/minerl/blob/v0.4.4/minerl/Malmo/Minecraft/src/main/java/com/microsoft/Malmo/Utils/TimeHelper.java).

MineRL can catch a failed client connection and return a random fallback observation with `info['error']`. A server-send failure can instead leave internal and returned termination inconsistent. The custom backend rejects both cases before interpreting the observation or reward; a failure remains a technical failure with potentially incomplete step accounting, never a success or fabricated transition.

## 5. Training and same-checkpoint evaluation

Create a fresh task-specific policy and checkpoint namespace. Proposed reuse settings are DreamerV3 size12m, batch16/length64, burn-in32, train-ratio256 and 5,000-action prefill. Resolve these against the new collector/replay interface and freeze actual settings before launch; never retain DMC metadata. The 1M action budget includes prefill. Repeat2 means at most 2M **acknowledged synchronous Minecraft/MineRL step calls**, not DMC calls or video frames. Their nominal duration is 0.05 simulated seconds, giving nominal 10 Hz decisions. Record native-step and optimization counts separately.

Train exactly one seed in this phase, with explicit per-episode reset seeds, an immutable task/configuration manifest and periodic checkpoint/reload checks. Any simulated training missingness and the real-target/predicted-target distinction must be stated in the actual collector configuration. Do not silently mix a clean-trained policy with a claim about a differently trained dual-chain collector. The source/configuration hash must resolve that choice before formal launch.

Minecraft uses a task-specific replay sampler because a successful episode can contain fewer than the 96 observations required by the 32-position burn-in plus 64-position update. Keeping only individually long episodes would silently exclude short successes. The sampler draws starts uniformly from all real rows in the retained complete episodes, arranged circularly. It requires at least 96 total real rows and never repeats a row within one sequence. Every joined boundary retains the original terminal/truncation observation, outgoing zero action placeholder, and next episode's real reset observation with `is_first=True`; no images, rewards or flags are padded or rewritten. Circular starts ensure even the oldest short episode can appear in the supervised suffix rather than only in burn-in. This changes the sampler for Minecraft only, without success-based prioritization.

This contract is supported by the pinned official DreamerV3 implementation: [`RSSM._observe`](https://github.com/danijar/dreamerv3/blob/e3f02248693a79dc8b0ebd62c93683888ddaccfe/dreamerv3/rssm.py#L75-L79) masks latent state and incoming action at every reset; [`Agent._apply_replay_context`](https://github.com/danijar/dreamerv3/blob/e3f02248693a79dc8b0ebd62c93683888ddaccfe/dreamerv3/agent.py#L312-L320) aligns incoming actions; replay value targets stop at `is_last` through [`lambda_return`](https://github.com/danijar/dreamerv3/blob/e3f02248693a79dc8b0ebd62c93683888ddaccfe/dreamerv3/agent.py#L449-L490). The existing adapter's burn-in respects these same reset flags. Native success labels come only from the separate privileged diagnostic ledger after an episode, not from shaped-reward thresholds or terminal flags. They are used solely to audit coverage, never as actor inputs or sampling weights.

Persist the replay sampling policy, success/unknown labels, RNG and coverage counters in checkpoints. Report the fraction of active episodes shorter than 96 rows, and the fraction of known successes whose real terminal reward row was delivered after burn-in. A sampled row is not evidence that a subsequent optimizer step succeeded; keep the actual update counters. Reject an incompatible older sampler snapshot. Nine local regression tests cover short-success/long-failure mixing, all-short replay, exact outgoing-action/reset alignment, absence of padding or repeated rows, restored sampling and counters, eviction denominators, unknown outcomes and unchanged V2 behavior. A finite official GPU update on a cross-episode batch remains a separate live gate.

After training, compare `clean`, `black_frame`, `prior_only`, `v2_exact`, `locked_ancestor`, and `posterior_return_once` on the **identical final checkpoint**. Keep the registered V2 eight-candidate budget, five-observation repair horizon and long-gap fallback. Do not combine the two proposals or increase the horizon without a separate intervention. All six arms are frozen: model, optimizer, target/normalizer, pinned inference parameters, training RNG and training counters are hashed before/after; update counters stay unchanged. Evaluation RNG advances separately. Require clean-path parity, single-use return evidence and no hidden-image/privileged-state leakage.

The short navigation task may finish well before Walker's fault index 150. The proposed Minecraft onset is **action-observation index 10**, with one exogenous gap of length 1, 5, 10, 20 or 50 actions (nominally 0.1, 0.5, 1, 2 or 5 simulated seconds). This is a scene-specific proposal to lock before confirmation. Retain cases that finish before/during the gap; do not resample until a favorable event appears or fabricate a full recovery window after successful termination. Long gaps exercise fallback, not a secretly enlarged repair mechanism. Start with known missing observations; blur, stale frames, occlusion and learned fault detection require separately registered parameters/labels and are not automatically implemented by this worker.

The approved initial final matrix uses environment seeds **10000-10019**: 20 clean episodes plus five faulty arms at five lengths and 20 seeds, for **520 cases**. Periodic checks use five episodes each for clean and the registered dropout condition. These IDs are reused in the training-period evaluation surface, so this is a **single-seed development final evaluation**, not an untouched confirmatory split. A sequential training reset-seed schedule may also reach these IDs; record actual overlap and do not claim seed exclusion unless implemented. Save exact task manifests and note geometry overlap. Use a fixed actor RNG derivation and full mask hashes. Same initial seed does not imply identical trajectories once actions diverge. Freeze the final evaluation implementation, checkpoint, task, schedules, analysis and metric definitions before running the final matrix. A later confirmatory study needs a newly registered, untouched split and is not auto-launched.

Task outcomes are target-touch success, raw return, final goal distance/progress, time to goal and technical failures. Record fault-window and post-return behavior only where defined; successful completion, time-limit truncation and failure must remain distinguishable. Dense reward is not a substitute for true target contact. Report all seeds and lengths, early termination, unreached gaps and censoring. Use paired environment-seed-cluster summaries/intervals for one trained policy; do not describe them as training-seed confidence. Negative or inconclusive repair advantages must be retained. Exact thresholds and confirmatory failure guards must be locked by the scene evaluation implementation, not borrowed from Walker reward/posture thresholds.

## 6. Recovery, runtime gates and acceptance

Training recovery initially promises only a **completed or explicitly budget-truncated episode boundary**: persist model/optimizer/normalizer/target state, replay state, interaction/update counters, software RNG, next exact episode seed, task identity and a finished-episode ledger. Do not claim exact restoration of Minecraft's internal mid-episode physics/world state. Audit any interactions consumed after the last recoverable boundary; a crash must not silently erase budget or duplicate replay. Evaluation resumes only at validated case boundaries using an immutable completed-case ledger.

Required gates before a formal run:

1. Isolated dependency install and `pip check`; JDK and OpenGL renderer identity recorded; no change to the old learner/V2 archive.
2. Real native MineRL reset/step smoke, followed by custom-task reset/step and rendered target-visibility inspection. Verify actual engine identity and nonempty RGB; a fake protocol test cannot satisfy this gate.
3. Exact action/native-step/seed/terminal accounting, deterministic task manifest, privilege stripping, reset isolation, stdout quarantine, timeout and owned-process cleanup.
4. Measured simulator/learner throughput, RAM/VRAM, full storage projection and reserved headroom.
5. Finite learner smoke, strict checkpoint reload, boundary recovery evidence and SHA256 transfer; source/config/task binding prevents a Walker checkpoint from being accepted.
6. Six-arm frozen-state/clean-parity tests, predeclared masks, raw metric definitions, failure policy and final budget accounting.

The local 15-test protocol suite passed before real installation: framing/fragmentation, malformed/oversized data, RGB schema, action validation/mapping, repeat/termination accounting, sidecar-only privilege, reset identity, raw-fd stdout isolation, boundary restrictions, cached rendering, timeout persistence, owned-worker shutdown and rejection of MineRL's synthetic error fallback. Its mock engine exists only in tests and must not appear in formal deployment.

Run the authorized stage under tmux in an isolated release directory with PID/start identity, console and exit markers. Inspect once every **six hours**, then end the check. Record progress, finite metrics, process state, GPU/RAM, OOM, disk headroom, checkpoint age and backup hashes; notify on milestones, completion, failures or required decisions. Do not continuously poll, blindly restart, change seeds/configurations, auto-launch another task, or shut down/release infrastructure.

Apply the user's fully read World Model/MBRL preservation original (SHA256 `FCFAF37D880DDEF449007C5B7B99B8904EC9249FC36D0B7495F241FA27AE0067`). Save a separate directory under `D:\World_Model_Baselines\Experiment_Notes\HelixWM\minecraft_visible_navigation_seed0_<execution_date>`, preserving old experiments, with `FINAL_PACKAGE`, `CHECKPOINTS`, `RESULTS`, `PLOTS`, `VIDEOS`, `LOGS`, `CONFIG`, `COMMANDS`, `METADATA` and README/summary.

Retain raw episode CSV and per-action/privileged-sidecar JSONL, fault masks and hashes, task/source/config/environment metadata, actual commands, full formal logs separate from smoke, checkpoint/recovery state, original learning data and analysis code. Save the training learning curve and frozen robustness/recovery figures in **PNG and PDF**, with units, seed, task and checkpoint labels. Save **one formal MP4 for this initial study**: `v2_exact`, seed10000, 20-action gap, with received-observation/availability labels. Keep the predeclared case even if it fails or ends early; do not select a prettier episode. Fully decode the video and verify plotted values against raw data.

After writes stop, build the full archive, archive SHA256 and per-file hash manifest; verify archive integrity and extracted hashes. Verify strict reload/zero-update evidence, raw summaries, figure readability and MP4 decode before marking this single seed accepted. Large checkpoints/replay/archives belong on D drive or artifact storage; publish only compact sanitized code/configuration/results/figures without credentials, private infrastructure, entire environments or caches. Missing evidence is reported, not relabeled as acceptance. Pause monitoring after the authorized stage is accepted/backed up; no automatic extra seeds or scenes. At least three independent training seeds would be needed for a later paper-level training-robustness claim.
