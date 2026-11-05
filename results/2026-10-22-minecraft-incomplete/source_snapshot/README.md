# Exact historical Minecraft source snapshot

This directory preserves selected **original Git blob bytes** from commit `d570bee97717838c8c90f3b4754ad534e906e7b8`, the frozen source used for the stopped October 2026 Minecraft formal run. It is an independent historical snapshot, not a patch to the current matched-Walker source tree.

## Scope

Included: all tracked `helixwm_minecraft/` and `helixwm_v2/` files; `tests/test_minecraft*`; the Minecraft 1M-action and smoke configurations; `requirements-v2-5090.lock.txt`; and `pyproject.toml`. `source_manifest.json` maps every original relative path to its Git blob identifier, SHA256 and byte count.

This is a scoped source archive, not a complete standalone installable repository: other package directories referenced by project metadata or imports are not duplicated here. For a complete historical checkout, use the fixed commit above together with its pinned upstream dependencies. No datasets, model weights, replay, environment directories, downloaded wheels, videos or SSH operations are included.

## Version boundary

The repository root currently contains matched-Walker code and a different historical Minecraft version. Do **not** mix this directory's `runtime.py` with the root directory's older `task.py`, evaluator or configuration. The snapshot includes the runtime admission module and its tests because they were tracked source, not generated deployment glue.

The fixed `/root/helixwm_minecraft/env_py311_prebuilt` prefix is the historical worker admission rule and configured installation location. It is not a credential or public SSH endpoint. Source bytes are preserved without redaction so the manifest remains comparable to the original commit; portability changes would require a separately identified version and validation.

## Validation scope

Every exported file was checked against its original Git blob identifier and SHA256. A static scan checked for private keys, common GitHub token formats, credential assignments, Windows personal-directory paths and the experiment provider's SSH endpoints; findings are recorded in `source_manifest.json`. This is not an exhaustive security certification.

**Tests were not rerun, dependencies were not installed, and the simulator was not started for this publication.** The snapshot does not change the recorded training failure or establish final scientific acceptance.
