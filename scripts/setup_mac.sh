#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

if [[ -z "${STAR_PYTHON:-}" ]]; then
  if command -v python3.11 >/dev/null 2>&1; then
    STAR_PYTHON=python3.11
  elif command -v python3.12 >/dev/null 2>&1; then
    STAR_PYTHON=python3.12
  else
    echo 'Install native Python 3.11 first: brew install python@3.11' >&2
    exit 1
  fi
fi

"$STAR_PYTHON" - <<'PY'
import platform
import sys
if platform.system() != 'Darwin' or platform.machine() != 'arm64':
    raise SystemExit('Use native arm64 Python on an Apple Silicon Mac. For CPU/Linux, install with pip directly.')
if not (3, 11) <= sys.version_info[:2] <= (3, 12):
    raise SystemExit('Python 3.11 or 3.12 is required.')
if tuple(map(int, platform.mac_ver()[0].split('.')[:2])) < (14, 0):
    raise SystemExit('The pinned native dependency wheels require macOS 14 or newer.')
PY

if [[ ! -x .venv/bin/python ]]; then
  "$STAR_PYTHON" -m venv .venv
fi
.venv/bin/python -m pip install -c requirements.lock -e '.[dev]'
.venv/bin/python -m pip check
.venv/bin/python -m star doctor
echo 'Ready. Run: source .venv/bin/activate'
