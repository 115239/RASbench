#!/usr/bin/env bash
set -euo pipefail
task_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
task_isaac_python="${PAPER200_NATIVE_ISAAC_PYTHON:-${task_script_dir}/../mc/runner_paper_200/safety_embodiment/competition_runner/native/isaac_python.sh}"
export BASELINE_PYTHON_KIND=isaac
exec "$task_isaac_python" -m paper200.resident_pool "$@"
