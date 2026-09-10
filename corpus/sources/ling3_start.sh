#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# start.sh -- serve Ling-3.0-flash (MXFP4) on one DGX Spark with SGLang.
#
#   ./start.sh                      # PROFILE from .env (default humming-dspark)
#   PROFILE=cookbook ./start.sh     # reproduce the SGLang cookbook cell
#   PROFILE=humming-nextn ./start.sh
#
# Profiles
#   cookbook        --moe-runner-backend flashinfer_mxfp4 + DSPARK.
#                   The SGLang cookbook's verified dgx-spark|mxfp4 cell.
#   humming-nextn   humming MoE + online FP8 LM head + NEXTN 3-step.
#                   The Ant/NVIDIA DGX Spark notebook.
#   humming-dspark  humming MoE + online FP8 LM head + DSPARK.  <-- this recipe
#                   The combination neither public source publishes.
# ---------------------------------------------------------------------------
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

err()  { echo "ERROR: $*" >&2; exit 1; }
info() { echo "--- $*"; }

# Environment wins over .env, so `PROFILE=cookbook ./start.sh` works.
declare -A _CLI=()
for v in PROFILE MEM_FRACTION_STATIC PORT MAX_RUNNING_REQUESTS YARN_OVERRIDE \
         MAX_MAMBA_CACHE_SIZE CHUNKED_PREFILL_SIZE PAGE_SIZE; do
    [[ -n "${!v:-}" ]] && _CLI[$v]="${!v}"
done
[[ -f .env ]] && { set -a; . ./.env; set +a; }
for v in "${!_CLI[@]}"; do printf -v "$v" '%s' "${_CLI[$v]}"; done

PROFILE="${PROFILE:-humming-dspark}"
MODEL_DIR="${MODEL_DIR:-$HOME/models/Ling-3.0-flash-fp4}"
DRAFT_DIR="${DRAFT_DIR:-$HOME/models/Ling-3.0-flash-dspark}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-ling-3.0-flash}"
HOST="${HOST:-0.0.0.0}"; PORT="${PORT:-30000}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.75}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-1}"
MAX_MAMBA_CACHE_SIZE="${MAX_MAMBA_CACHE_SIZE:-64}"
CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-8192}"
MAX_PREFILL_TOKENS="${MAX_PREFILL_TOKENS:-16384}"
PAGE_SIZE="${PAGE_SIZE:-64}"
RANDOM_SEED="${RANDOM_SEED:-308534008}"
REASONING_PARSER="${REASONING_PARSER:-ling3}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-ling3}"
YARN_OVERRIDE="${YARN_OVERRIDE:-0}"
# Tuning passthrough (tune.sh): whitespace-separated extra server flags and
# KEY=VALUE env entries. Empty by default, so plain ./start.sh is unchanged.
EXTRA_FLAGS="${EXTRA_FLAGS:-}"
EXTRA_ENV="${EXTRA_ENV:-}"
# JSON-valued flags cannot ride in EXTRA_FLAGS: that is word-split on spaces,
# which shreds an object like {"enable_thinking": true} into three argv items.
# Give the chat-template default its own variable, appended as one element.
# THIS IS THE THINKING SWITCH, and thinking on is the recommended setting:
#   DEFAULT_CHAT_TEMPLATE_KWARGS='{"enable_thinking": true}' ./start.sh
DEFAULT_CHAT_TEMPLATE_KWARGS="${DEFAULT_CHAT_TEMPLATE_KWARGS:-}"

[[ -d "$MODEL_DIR" ]] || err "target weights not found: $MODEL_DIR (run ./download.py)"
[[ -f .venv/bin/activate ]] || err ".venv missing -- run ./setup.sh first"

case "$PROFILE" in
    cookbook|cookbook-exact|humming-nextn|humming-dspark) ;;
    *) err "unknown PROFILE '$PROFILE' (cookbook|cookbook-exact|humming-nextn|humming-dspark)" ;;
esac

# --- per-profile flags ----------------------------------------------------
FLAGS=()
ENV_EXTRA=()
case "$PROFILE" in
  cookbook)
    MOE_BACKEND=flashinfer_mxfp4
    ENV_EXTRA+=("SGLANG_ENABLE_FP8_LM_HEAD=0")
    SPEC=dspark
    ;;
  cookbook-exact)
    # The SGLang cookbook's dgx-spark|mxfp4|low-latency|dspark cell, verbatim:
    # nothing but the model, tp 1, the MoE backend, the DSPARK flags and 0.85.
    # Exists purely as the calibration point against the published 9.48 ms TPOT.
    MOE_BACKEND=flashinfer_mxfp4
    ENV_EXTRA+=("SGLANG_ENABLE_FP8_LM_HEAD=0")
    SPEC=dspark
    ;;
  humming-nextn)
    MOE_BACKEND=humming
    ENV_EXTRA+=("SGLANG_ENABLE_FP8_LM_HEAD=1")
    SPEC=nextn
    ;;
  humming-dspark)
    MOE_BACKEND=humming
    ENV_EXTRA+=("SGLANG_ENABLE_FP8_LM_HEAD=1")
    SPEC=dspark
    ;;
esac

case "$SPEC" in
  dspark)
    [[ -d "$DRAFT_DIR" ]] || err "DSpark drafter not found: $DRAFT_DIR (run ./download.py)"
    # The drafter's block_size is 8, so the verify window is 9 tokens. The KDA
    # ReplaySSM ring must be a power of two >= 2x the window: 16 fails startup
    # validation, 32 is the smallest that passes.
    FLAGS+=(--speculative-algorithm DSPARK
            --speculative-draft-model-path "$DRAFT_DIR"
            --enable-linear-replayssm-spec
            --linear-replayssm-cache-len 32)
    ;;
  nextn)
    FLAGS+=(--speculative-algorithm NEXTN
            --speculative-draft-model-path "$MODEL_DIR"
            --speculative-num-steps 3
            --speculative-eagle-topk 1
            --speculative-num-draft-tokens 4)
    ;;
esac

# shellcheck disable=SC2206
[[ -n "$EXTRA_FLAGS" ]] && FLAGS+=($EXTRA_FLAGS)
[[ -n "$DEFAULT_CHAT_TEMPLATE_KWARGS" ]] && FLAGS+=(--default-chat-template-kwargs "$DEFAULT_CHAT_TEMPLATE_KWARGS")
# shellcheck disable=SC2206
[[ -n "$EXTRA_ENV" ]] && ENV_EXTRA+=($EXTRA_ENV)

if [[ "$YARN_OVERRIDE" == "1" ]]; then
    FLAGS+=(--json-model-override-args '{"max_position_embeddings":262144,"rope_scaling":{"rope_type":"yarn","factor":2.0,"rope_theta":6000000,"partial_rotary_factor":0.5,"original_max_position_embeddings":131072}}')
fi

# --- guard: the box holds one ~60 GB model at a time ------------------------
# Loading these weights on top of another resident model is how a single Spark
# wedges (no OOM, no logs, driver stops responding). Refuse unless the pool is
# genuinely free. Override with MIN_FREE_GIB for a deliberately tight layout.
MIN_FREE_GIB="${MIN_FREE_GIB:-85}"
_avail=$(awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo)
if (( _avail < MIN_FREE_GIB )); then
    err "only ${_avail} GiB available (need >= ${MIN_FREE_GIB}). Another model server is
     probably resident. Stop it and wait for MemAvailable to recover before starting Ling."
fi

# FlashInfer autotune is off by default (Ant's guidance: JIT during weight load
# is a wedge path). DROP_AUTOTUNE=1 lets tune.sh A/B it now that kernels are
# precompiled. Expands to nothing when dropped -- never to a stray positional.
AUTOTUNE_FLAG=--disable-flashinfer-autotune
[[ -n "${DROP_AUTOTUNE:-}" ]] && AUTOTUNE_FLAG=""

# shellcheck disable=SC1091
source .venv/bin/activate

info "profile=$PROFILE  moe=$MOE_BACKEND  spec=$SPEC  mem_fraction=$MEM_FRACTION_STATIC  port=$PORT  extra_flags=[$EXTRA_FLAGS]  chat_kwargs=[$DEFAULT_CHAT_TEMPLATE_KWARGS]  extra_env=[$EXTRA_ENV]"

mkdir -p logs "$HOME/.humming/cache"

if [[ "$PROFILE" == "cookbook-exact" ]]; then
    exec env SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 "${ENV_EXTRA[@]}" \
      python3 -m sglang.launch_server \
        --model-path "$MODEL_DIR" \
        --served-model-name "$SERVED_MODEL_NAME" \
        --trust-remote-code \
        --tp 1 \
        --moe-runner-backend flashinfer_mxfp4 \
        "${FLAGS[@]}" \
        --mem-fraction-static 0.85 \
        --host "$HOST" --port "$PORT"
fi

env \
  SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 \
  SGLANG_JIT_DEEPGEMM_PRECOMPILE=0 \
  SGLANG_ENABLE_JIT_DEEPGEMM=0 \
  SGLANG_DSV4_FP4_DEQUANT=0 \
  SGLANG_FP8_IGNORED_LAYERS="" \
  HUMMING_COMPILER=nvrtc \
  HUMMING_CACHE_DIR="$HOME/.humming/cache" \
  "${ENV_EXTRA[@]}" \
  python3 -m sglang.launch_server \
    --model-path "$MODEL_DIR" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --trust-remote-code \
    --dtype bfloat16 \
    --tp-size 1 --ep-size 1 \
    --host "$HOST" --port "$PORT" \
    --max-running-requests "$MAX_RUNNING_REQUESTS" \
    --max-mamba-cache-size "$MAX_MAMBA_CACHE_SIZE" \
    --chunked-prefill-size "$CHUNKED_PREFILL_SIZE" \
    --max-prefill-tokens "$MAX_PREFILL_TOKENS" \
    --page-size "$PAGE_SIZE" \
    --cuda-graph-backend-decode full \
    --cuda-graph-max-bs-decode 1 \
    --cuda-graph-bs-decode 1 \
    --cuda-graph-backend-prefill disabled \
    --random-seed "$RANDOM_SEED" \
    --reasoning-parser "$REASONING_PARSER" \
    --tool-call-parser "$TOOL_CALL_PARSER" \
    --attention-backend flashinfer \
    $AUTOTUNE_FLAG \
    --mem-fraction-static "$MEM_FRACTION_STATIC" \
    --fp8-gemm-backend cutlass \
    --moe-runner-backend "$MOE_BACKEND" \
    --flashinfer-mxfp4-moe-precision default \
    --disable-shared-experts-fusion \
    "${FLAGS[@]}"
