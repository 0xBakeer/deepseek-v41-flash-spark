#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run.sh -- the container path, one entry point.
#
#   ./run.sh setup     pull the image (BUILD=1 builds it here) and fetch the 510 GB checkpoint
#   ./run.sh serve     start the server on http://127.0.0.1:$PORT/v1 (detached) and wait for /health
#   ./run.sh logs      follow it
#   ./run.sh stop      stop and remove the container, then wait for the memory to come back
#   ./run.sh shell     a shell inside the running container
#   ./run.sh bench ..  run bench/bench.py against it (arguments pass straight through)
#   ./run.sh config    show the resolved compose configuration
#
# For the native path -- a venv on the box, no container -- use ./start.sh and ./stop.sh
# instead; both read the same .env. docs/install.md compares the two.
#
# Reads ./.env (see env.example). Environment beats .env, as everywhere in this recipe:
#   PORT           published port on 127.0.0.1                    (8000)
#   MODEL_DIR      host path to the checkpoint                    (./models/DeepSeek-V4.1-Flash)
#   MODELS_DIR     host directory bind-mounted at /models         (dirname of MODEL_DIR)
#   MODEL_NAME     checkpoint directory name under it             (basename of MODEL_DIR)
#   RECIPE_VERSION image tag                                      (VERSION)
#   GHCR_OWNER     registry namespace                             (0xbakeer)
#   BUILD=1        build the image locally instead of pulling it
#   MIN_FREE_GIB   memory the server demands before it will load  (90)
# ---------------------------------------------------------------------------
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

log() { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
die() { printf '\033[1;31mERROR:\033[0m %s\n' "$*" >&2; exit 1; }

# Environment wins over .env (`PORT=8001 ./run.sh serve`), same rule as start.sh.
declare -A _CLI=()
for v in PORT MODEL_DIR MODELS_DIR MODEL_NAME RECIPE_VERSION GHCR_OWNER MIN_FREE_GIB \
         MAX_SEQ ARENA_GB TRACE_STATS DEFAULT_THINKING DEFAULT_EFFORT SPEC \
         SERVED_MODEL_NAME EXTRA_FLAGS HF_TOKEN; do
    [[ -n "${!v:-}" ]] && _CLI[$v]="${!v}"
done
# shellcheck disable=SC1091
[[ -f .env ]] && { set -a; . ./.env; set +a; }
for v in "${!_CLI[@]}"; do printf -v "$v" '%s' "${_CLI[$v]}"; done

PORT="${PORT:-8000}"
RECIPE_VERSION="${RECIPE_VERSION:-$(cat VERSION 2>/dev/null || echo 0.1.0-wip)}"
MODEL_DIR="${MODEL_DIR:-./models/DeepSeek-V4.1-Flash}"
# The container always finds the checkpoint under its /models mount, so what compose
# needs is the host directory to bind and the name of the checkpoint inside it.
MODELS_DIR="${MODELS_DIR:-$(dirname "$MODEL_DIR")}"
MODEL_NAME="${MODEL_NAME:-$(basename "$MODEL_DIR")}"
export PORT RECIPE_VERSION MODELS_DIR MODEL_NAME
export GHCR_OWNER="${GHCR_OWNER:-0xbakeer}"
unset MODEL_DIR   # compose composes its own; a host path inside the container is a bug

DC=(docker compose)
SERVICE=v41

need() {
    command -v docker >/dev/null || die "docker not found"
    docker compose version >/dev/null 2>&1 || die "docker compose (v2) not found"
    [[ "$(uname -m)" == "aarch64" || "$(uname -m)" == "arm64" ]] || \
        die "this recipe is arm64 (GB10 / DGX Spark); this machine is $(uname -m)"
    docker info 2>/dev/null | grep -qi nvidia || \
        log "warning: no NVIDIA runtime visible to docker -- install nvidia-container-toolkit if 'serve' fails"
}

case "${1:-}" in
  setup)
    need
    mkdir -p "$MODELS_DIR" results
    if [[ "${BUILD:-0}" == "1" ]]; then
        log "building the image here (no engine compile in this recipe; it is a pip install)"
        "${DC[@]}" build
    else
        log "pulling ghcr.io/$GHCR_OWNER/deepseek-v41-flash-spark:$RECIPE_VERSION"
        "${DC[@]}" pull "$SERVICE"
    fi
    log "fetching the checkpoint into $MODELS_DIR/$MODEL_NAME (510 GB, resumable -- re-run if it stops)"
    "${DC[@]}" run --rm --no-deps "$SERVICE" download
    log "done. next: ./run.sh serve"
    ;;

  serve)
    need
    log "starting deepseek-v41-flash ($RECIPE_VERSION)"
    log "the warm start streams tens of GB of FP4 experts off NVMe before the socket binds"
    "${DC[@]}" up -d "$SERVICE"
    log "waiting for http://127.0.0.1:$PORT/health (up to 45 min)"
    for _ in $(seq 1 540); do
        if ! "${DC[@]}" ps --status running --format '{{.Service}}' 2>/dev/null | grep -qx "$SERVICE"; then
            echo; "${DC[@]}" logs --tail 40 "$SERVICE" >&2
            die "the container stopped during startup (./run.sh logs for the rest)"
        fi
        if curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
            echo
            curl -fsS "http://127.0.0.1:$PORT/health" && echo
            cat <<MSG

  endpoint  http://127.0.0.1:$PORT/v1
  logs      ./run.sh logs
  bench     ./run.sh bench --workload prose --runs 3 --out results/prose.json
  stop      ./run.sh stop
MSG
            exit 0
        fi
        printf '.'
        sleep 5
    done
    echo
    die "no /health after 45 minutes -- ./run.sh logs"
    ;;

  logs)  exec "${DC[@]}" logs -f --tail 200 "$SERVICE" ;;

  stop)
    "${DC[@]}" down --timeout 180
    # The arena is tens of GB of pinned and page-cache-backed memory and the kernel
    # reclaims it lazily. Starting the next server before that lands is how this box
    # gets wedged, so wait for the pool the way ./stop.sh does.
    if [[ -r /proc/meminfo ]]; then
        target="${MIN_FREE_GIB:-90}"
        printf 'waiting for memory to be released (target >= %s GiB): ' "$target"
        for _ in $(seq 1 60); do
            a=$(awk '/^MemAvailable:/ {printf "%d", $2/1048576}' /proc/meminfo)
            if (( a >= target )); then echo "OK, ${a} GiB available"; exit 0; fi
            sleep 3
        done
        echo "WARNING: still below ${target} GiB after 3 min -- do not start another server yet" >&2
        exit 1
    fi
    ;;

  shell) exec "${DC[@]}" exec "$SERVICE" bash ;;

  bench)
    shift
    # On the host, against the published port: bench.py is stdlib-only, and running it
    # outside the container keeps its requests off the server's own memory pool.
    exec python3 bench/bench.py --base "http://127.0.0.1:$PORT" "$@"
    ;;

  config) exec "${DC[@]}" config ;;

  ""|-h|--help|help) sed -n '3,27p' "$0" | sed 's/^# \{0,1\}//' ;;
  *) die "unknown command: $1 (try ./run.sh help)" ;;
esac
