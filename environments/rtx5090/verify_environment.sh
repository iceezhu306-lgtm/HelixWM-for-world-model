#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=/root/autodl-tmp/helixwm
source "$ROOT/scripts/activate_helixwm.sh"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
exec > >(tee "$ROOT/logs/verification.log") 2>&1
trap 'result=$?; printf "%s\n" "$result" > "$ROOT/logs/verification.exit"; date -u +%FT%TZ > "$ROOT/logs/verification.finished_utc"' EXIT
date -u +%FT%TZ
python -m pip check 2>&1 | tee "$ROOT/logs/pip-check.log"
python -m unittest discover -s tests -v 2>&1 | tee "$ROOT/logs/unit-tests.log"
python -m examples.synthetic_trace 2>&1 | tee "$ROOT/logs/synthetic-trace.log"
python "$ROOT/scripts/environment_smoke.py" --output "$ROOT/smoke/initial_validation" --control-steps 4 2>&1 | tee "$ROOT/logs/environment-smoke.log"
git rev-parse HEAD
git diff --exit-code
printf 'ENVIRONMENT_VERIFICATION_PASS\n'
