#!/usr/bin/env bash
# Run Digital SAT RW generator using the project virtualenv.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
VENV_PY="$ROOT/.venv/bin/python3"

if [[ ! -x "$VENV_PY" ]]; then
  echo "Missing .venv. Create it with:" >&2
  echo "  python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt" >&2
  exit 1
fi

exec "$VENV_PY" "$ROOT/generate_digital_sat_rw.py" "$@"
