#!/usr/bin/env bash
# Everything a change should pass before it is pushed: the layer rules, the Python tests,
# and the web console's type check, tests and build.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== layer rules =="
python tests/architecture/layers.py

echo "== python tests =="
python -m pytest -q -W error::DeprecationWarning "$@"

if command -v npm >/dev/null 2>&1; then
  echo "== web console =="
  cd web
  [ -d node_modules ] || npm ci --no-audit --no-fund
  npm test
  npm run build
else
  echo "== web console: skipped (npm not found) =="
fi
