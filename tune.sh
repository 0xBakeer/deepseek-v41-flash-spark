#!/usr/bin/env bash
# Choose what this box should be good at, see what it costs, run it.
# Reads the same .env as ./start.sh and writes the settings it manages back
# into it. --list / --print / --write do not need a terminal.
set -euo pipefail
cd "$(dirname "$0")"
[[ -f .env ]] && { set -a; . ./.env; set +a; }
exec "${PYTHON:-python3}" tools/tune.py "$@"
