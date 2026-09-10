#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# start.sh -- serve DeepSeek-V4.1-Flash on one DGX Spark.
#
#   ./start.sh                 # start, wait for /health, print the endpoint
#   ./start.sh --no-wait       # start and return immediately (tail logs yourself)
#   PORT=8001 ./start.sh       # environment beats .env
#   ARENA_GB=60 ./start.sh     # pin the resident expert arena instead of auto
#
# This is not SGLang and not a container: it launches `server/app.py --engine v41`
# (engine/v41_engine.py) with nohup, writes logs/server.pid and logs to
# logs/server.log. Stop it with ./stop.sh.
#
# Why the health wait is 20 minutes: the warm start fills the resident FP4
# expert arena from the checkpoint, which is ~80 GB of NVMe reads before the
# HTTP socket is even bound. A server that is "not up yet" at minute 6 is normal.
# ---------------------------------------------------------------------------
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

err()  { echo "ERROR: $*" >&2; exit 1; }
info() { echo "--- $*"; }

WAIT=true
for arg in "$@"; do
    case "$arg" in
        --no-wait) WAIT=false ;;
        -h|--help) sed -n '3,16p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) err "unknown argument '$arg' (only --no-wait)" ;;
    esac
done

# Environment wins over .env, so `PORT=8001 ./start.sh` works.
declare -A _CLI=()
for v in MODEL_DIR PYTHON SERVED_MODEL_NAME HOST PORT MAX_SEQ ARENA_GB \
         TRACE_STATS DEFAULT_THINKING DEFAULT_EFFORT SPEC EXTRA_FLAGS; do
    [[ -n "${!v:-}" ]] && _CLI[$v]="${!v}"
done
# shellcheck disable=SC1091
[[ -f .env ]] && { set -a; . ./.env; set +a; }
for v in "${!_CLI[@]}"; do printf -v "$v" '%s' "${_CLI[$v]}"; done

MODEL_DIR="${MODEL_DIR:-$HOME/models/DeepSeek-V4.1-Flash}"
PYTHON="${PYTHON:-$HOME/recipes/ling3-flash-dgx-spark/.venv/bin/python}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-deepseek-v4.1-flash}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
MAX_SEQ="${MAX_SEQ:-32768}"
ARENA_GB="${ARENA_GB:-}"                 # empty = size from free GPU memory
TRACE_STATS="${TRACE_STATS:-results/trace-full/stats/coverage.json}"
TRACE_STATS_FALLBACK="${TRACE_STATS_FALLBACK:-results/trace-partial/stats/coverage.json}"
DEFAULT_THINKING="${DEFAULT_THINKING:-off}"
DEFAULT_EFFORT="${DEFAULT_EFFORT:-75}"
SPEC="${SPEC:-1}"
# Whitespace-separated extra flags for server/app.py (A/B runs). Empty by default.
EXTRA_FLAGS="${EXTRA_FLAGS:-}"
MIN_FREE_GIB="${MIN_FREE_GIB:-90}"
HEALTH_TIMEOUT_S="${HEALTH_TIMEOUT_S:-1200}"

LOG_DIR="$SCRIPT_DIR/logs"
LOG_FILE="$LOG_DIR/server.log"
PID_FILE="$LOG_DIR/server.pid"

# --- sanity ---------------------------------------------------------------
[[ -d "$MODEL_DIR" ]] || err "model dir not found: $MODEL_DIR (set MODEL_DIR in .env)"
[[ -f "$MODEL_DIR/tokenizer.json" ]] || err "$MODEL_DIR has no tokenizer.json"
[[ -f "$MODEL_DIR/encoding/encoding.py" ]] || err "$MODEL_DIR has no encoding/encoding.py"
[[ -x "$PYTHON" ]] || err "interpreter not executable: $PYTHON (set PYTHON in .env)"
[[ -f server/app.py ]] || err "server/app.py missing -- run this from a full checkout"
case "$DEFAULT_THINKING" in on|off) ;; *) err "DEFAULT_THINKING must be on|off (got '$DEFAULT_THINKING')" ;; esac
case "$SPEC" in 0|1) ;; *) err "SPEC must be 0|1 (got '$SPEC')" ;; esac

# --- guard: is the port already taken? ------------------------------------
port_busy() {
    if command -v ss >/dev/null 2>&1; then
        [[ -n "$(ss -ltnH "sport = :$PORT" 2>/dev/null)" ]]
    elif command -v lsof >/dev/null 2>&1; then
        lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1
    else
        (exec 3<>"/dev/tcp/127.0.0.1/$PORT") 2>/dev/null
    fi
}
if port_busy; then
    holder=""
    command -v ss >/dev/null 2>&1 && holder="$(ss -ltnpH "sport = :$PORT" 2>/dev/null | tr -s ' ')"
    err "port $PORT is already in use${holder:+ by: $holder}.
     If it is our own server, ./stop.sh. Otherwise pick another PORT."
fi

# --- guard: does something else own the box's memory? ---------------------
# MemAvailable is the "available" column of `free -g`. This model needs the
# unified pool essentially to itself: the resident expert arena is sized from
# what is free at load time, so starting next to another server does not OOM,
# it silently gives us a tiny arena and a NVMe-bound 1 tok/s server -- or wedges
# the driver with no OOM and no logs. Refuse instead.
if [[ -r /proc/meminfo ]]; then
    avail_gib=$(awk '/^MemAvailable:/ {printf "%d", $2/1048576}' /proc/meminfo)
    if (( avail_gib < MIN_FREE_GIB )); then
        echo "ERROR: only ${avail_gib} GiB available (need >= ${MIN_FREE_GIB} GiB)." >&2
        echo "     Biggest resident processes:" >&2
        ps -eo pid,rss,comm --sort=-rss 2>/dev/null | head -6 |
            awk 'NR==1{print "       PID      RSS_GB  COMMAND"; next} {printf "       %-8s %-7.1f %s\n", $1, $2/1048576, $3}' >&2
        if command -v docker >/dev/null 2>&1 && docker ps --format '{{.Names}}' 2>/dev/null | grep -qx 'vllm-fn-tp1'; then
            echo "     The Qwen vLLM container vllm-fn-tp1 is running. Stop it with:" >&2
            echo "         docker stop vllm-fn-tp1" >&2
        else
            echo "     Stop whatever holds the pool (the usual occupant is the Qwen vLLM" >&2
            echo "     container: docker stop vllm-fn-tp1) and wait for MemAvailable to recover." >&2
        fi
        exit 1
    fi
else
    echo "WARNING: no /proc/meminfo -- skipping the memory guard (not a Linux box?)" >&2
fi

# --- guard: are we already running? ---------------------------------------
if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    err "server already running (pid $(cat "$PID_FILE")). ./stop.sh first."
fi

# --- assemble the command -------------------------------------------------
FLAGS=(
    --model-dir "$MODEL_DIR"
    --host "$HOST" --port "$PORT"
    --served-model-name "$SERVED_MODEL_NAME"
    --default-thinking "$DEFAULT_THINKING"
    --default-effort "$DEFAULT_EFFORT"
    --engine v41
    --max-seq "$MAX_SEQ"
)
[[ -n "$ARENA_GB" ]] && FLAGS+=(--arena-gb "$ARENA_GB")
[[ "$SPEC" == "0" ]] && FLAGS+=(--no-spec)

# The full trace is 40 layers and takes days of shard downloads; until it lands,
# rank the warm start with the partial (layers 0-3) coverage rather than with
# nothing, which would warm experts in index order.
TRACE_USED=""
if [[ -n "$TRACE_STATS" && -f "$TRACE_STATS" ]]; then
    TRACE_USED="$TRACE_STATS"
elif [[ -n "$TRACE_STATS_FALLBACK" && -f "$TRACE_STATS_FALLBACK" ]]; then
    TRACE_USED="$TRACE_STATS_FALLBACK"
    info "trace stats $TRACE_STATS missing; falling back to $TRACE_USED (layers 0-3 only)"
else
    info "no trace stats found ($TRACE_STATS, $TRACE_STATS_FALLBACK) -- warm start will use index order"
fi
[[ -n "$TRACE_USED" ]] && FLAGS+=(--trace-stats "$TRACE_USED")
# shellcheck disable=SC2206
[[ -n "$EXTRA_FLAGS" ]] && FLAGS+=($EXTRA_FLAGS)

mkdir -p "$LOG_DIR"
info "model=$MODEL_DIR  max_seq=$MAX_SEQ  arena=${ARENA_GB:-auto}  spec=$SPEC  thinking=$DEFAULT_THINKING/$DEFAULT_EFFORT  trace=${TRACE_USED:-none}"
info "log: $LOG_FILE"

: > "$LOG_FILE"
nohup "$PYTHON" server/app.py "${FLAGS[@]}" >>"$LOG_FILE" 2>&1 &
SERVER_PID=$!
echo "$SERVER_PID" > "$PID_FILE"
info "started pid $SERVER_PID"

if [[ "$WAIT" == false ]]; then
    echo "not waiting (--no-wait). Watch it come up with: tail -f $LOG_FILE"
    exit 0
fi

# --- wait for /health -----------------------------------------------------
# The socket is bound only after the engine is constructed, so this loop is
# mostly watching an 80 GB warm start, not an HTTP handshake.
echo -n "waiting for http://$HOST:$PORT/health (up to $((HEALTH_TIMEOUT_S / 60)) min, warm start reads ~80 GB from NVMe): "
deadline=$(( $(date +%s) + HEALTH_TIMEOUT_S ))
while :; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo
        echo "--- last 30 log lines ---" >&2
        tail -n 30 "$LOG_FILE" >&2
        rm -f "$PID_FILE"
        err "server exited during startup (see $LOG_FILE)"
    fi
    if curl -sf -o /dev/null "http://$HOST:$PORT/health"; then
        echo " up"
        break
    fi
    if (( $(date +%s) >= deadline )); then
        echo
        echo "--- last 30 log lines ---" >&2
        tail -n 30 "$LOG_FILE" >&2
        err "no /health after ${HEALTH_TIMEOUT_S}s. It may still be warming up: watch $LOG_FILE, or ./stop.sh."
    fi
    echo -n "."
    sleep 5
done

curl -sf "http://$HOST:$PORT/health" && echo
cat <<MSG

  endpoint  http://$HOST:$PORT/v1
  model     $SERVED_MODEL_NAME
  logs      $LOG_FILE
  bench     python3 bench/bench.py --workload prose --runs 3 --out results/prose.json
  stop      ./stop.sh
MSG
