#!/usr/bin/env bash
set -e
# fonts sometimes help PIL and matplotlib look correct
python - <<'PY'
print("Environment OK")
PY

