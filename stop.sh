#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# stop.sh -- stop the DeepSeek-V4.1-Flash server started by ./start.sh.
#
#   ./stop.sh              # SIGTERM, then SIGKILL, then wait for memory
#   ./stop.sh --force      # straight to SIGKILL
#   MIN_FREE_GIB=95 ./stop.sh
#
# One process, not a process tree (server/app.py is threads, not forks), so the
# pidfile is enough in the normal case. The pkill fallback is for a server that
# was started by hand or whose pidfile was lost -- the bracketed pattern keeps
# pkill from matching this script's own command line.
#
# The wait at the end matters: the resident FP4 expert arena is tens of GB of
# pinned/page-cache-backed memory and the kernel reclaims it lazily. Starting
# the next server before that lands is how a 121 GiB box gets wedged.
# ---------------------------------------------------------------------------
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

FORCE=false
[[ "${1:-}" == "-f" || "${1:-}" == "--force" ]] && FORCE=true
MIN_FREE_GIB="${MIN_FREE_GIB:-90}"
PID_FILE="$SCRIPT_DIR/logs/server.pid"
SELF=$$

avail_gib() {
    [[ -r /proc/meminfo ]] || { echo "?"; return; }
    awk '/^MemAvailable:/ {printf "%d", $2/1048576}' /proc/meminfo
}

server_pids() {
    {
        # The pidfile is the normal case.
        if [[ -f "$PID_FILE" ]]; then
            local p
            p="$(cat "$PID_FILE" 2>/dev/null)"
            [[ -n "$p" ]] && kill -0 "$p" 2>/dev/null && echo "$p"
        fi
        # Fallback for a server started by hand or with a lost pidfile.
        # Bracketed pattern: 'app.p[y]' never matches this script or the pgrep
        # command line itself, so a stale match cannot make us kill our own shell.
        pgrep -f 'python.*server/app\.p[y]' 2>/dev/null || true
    } | grep -E '^[0-9]+$' | sort -un | grep -vx "$SELF" | grep -vx "$PPID" || true
}

pids=$(server_pids)
if [[ -z "$pids" ]]; then
    echo "no DeepSeek-V4.1-Flash server running (MemAvailable $(avail_gib) GiB)"
    rm -f "$PID_FILE"
    exit 0
fi

echo "stopping pids: $(echo "$pids" | tr '\n' ' ')"

if [[ "$FORCE" == false ]]; then
    for p in $pids; do kill -TERM "$p" 2>/dev/null || true; done
    for _ in $(seq 1 30); do
        [[ -z "$(server_pids)" ]] && break
        sleep 2
    done
fi

remaining=$(server_pids)
if [[ -n "$remaining" ]]; then
    echo "escalating to SIGKILL: $(echo "$remaining" | tr '\n' ' ')"
    for p in $remaining; do kill -KILL "$p" 2>/dev/null || true; done
    sleep 5
fi

if [[ -n "$(server_pids)" ]]; then
    echo "ERROR: server processes survived SIGKILL: $(server_pids | tr '\n' ' ')" >&2
    exit 1
fi
rm -f "$PID_FILE"

if [[ ! -r /proc/meminfo ]]; then
    echo "stopped."
    exit 0
fi

echo -n "waiting for memory to be released (target >= ${MIN_FREE_GIB} GiB): "
for _ in $(seq 1 60); do
    a=$(avail_gib)
    if (( a >= MIN_FREE_GIB )); then
        echo "OK, ${a} GiB available"
        exit 0
    fi
    sleep 3
done
echo "WARNING: only $(avail_gib) GiB available after 3 min -- do not start another server yet" >&2
exit 1
