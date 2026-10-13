#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=/root/autodl-tmp/helixwm
PREFIX="$ROOT/env"
export CONDA_PKGS_DIRS="$ROOT/cache/conda"
export PIP_CACHE_DIR="$ROOT/cache/pip"
export TMPDIR="$ROOT/cache/tmp"
mkdir -p "$ROOT"/{metadata,logs,scripts,smoke,cache/tmp}
exec > >(tee -a "$ROOT/logs/setup.log") 2>&1
trap 'result=$?; printf "%s\n" "$result" > "$ROOT/logs/setup.exit"; date -u +%FT%TZ > "$ROOT/logs/setup.finished_utc"' EXIT
date -u +%FT%TZ
printf 'Creating an isolated environment; no training will be started.\n'
test ! -e "$PREFIX"
/root/miniconda3/bin/conda create -y --prefix "$PREFIX" --override-channels -c https://repo.anaconda.com/pkgs/main python=3.12.12 pip
"$PREFIX/bin/python" -m pip install --disable-pip-version-check --timeout 120 --retries 5 pip==25.2 setuptools==80.9.0 wheel==0.45.1
"$PREFIX/bin/python" -m pip install --disable-pip-version-check --timeout 120 --retries 5 torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
"$PREFIX/bin/python" -m pip install --disable-pip-version-check --timeout 120 --retries 5 numpy==2.2.6 scipy==1.15.3 dm_control==1.0.28 mujoco==3.3.0 Pillow==11.2.1 imageio==2.37.0 imageio-ffmpeg==0.6.0
if [ ! -d "$ROOT/source/.git" ]; then
  timeout 150 git clone https://github.com/iceezhu306-lgtm/HelixWM-for-world-model.git "$ROOT/source"
fi
git -C "$ROOT/source" checkout --detach 745231d9a2049c7945a78aa041a2ca845580a6ca
"$PREFIX/bin/python" -m pip install --no-deps --no-build-isolation -e "$ROOT/source"
"$PREFIX/bin/python" -m pip check
"$PREFIX/bin/python" -m pip freeze --all > "$ROOT/metadata/pip-freeze.txt"
/root/miniconda3/bin/conda list --prefix "$PREFIX" --explicit > "$ROOT/metadata/conda-explicit.txt"
/root/miniconda3/bin/conda env export --prefix "$PREFIX" > "$ROOT/metadata/conda-environment.yml"
git -C "$ROOT/source" rev-parse HEAD > "$ROOT/metadata/source-commit.txt"
git -C "$ROOT/source" status --short > "$ROOT/metadata/source-status.txt"
nvidia-smi -q > "$ROOT/metadata/nvidia-smi.txt"
dpkg-query -W tmux ffmpeg libegl1 libgl1 libglew2.2 > "$ROOT/metadata/system-packages.txt"
printf 'ENVIRONMENT_INSTALL_PASS\n'
