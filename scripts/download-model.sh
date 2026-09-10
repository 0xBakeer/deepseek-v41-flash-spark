#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# download-model.sh -- fetch DeepSeek-V4.1-Flash into $MODEL_DIR.
#
# 510 GB in 48 safetensors shards. Two of them (47 and 48) are the Engram
# n-gram tables at 101.5 GB each; the serving engine reads rows out of them at
# run time, so unlike the tracer it cannot do without them.
#
# Safe to re-run: huggingface_hub skips complete files and resumes partial ones,
# so an interrupted download continues where it stopped. Expect hours.
#
#   MODEL_REPO       deepseek-ai/DeepSeek-V4.1-Flash
#   MODEL_REVISION   optional; pin a commit for a reproducible checkpoint
#   MODEL_DIR        where it lands (/models/DeepSeek-V4.1-Flash)
#   HF_TOKEN         optional -- the repo is public; a token only raises rate limits
#   ALLOW_PATTERNS   optional, whitespace-separated glob list. The engram-less
#                    subset (307 GB, tracing only, will NOT serve) is:
#                      ALLOW_PATTERNS="*.json *.py *.md *.txt tokenizer* \
#                        model-0000[1-9]-of-00048.safetensors \
#                        model-000[1-3]?-of-00048.safetensors \
#                        model-0004[0-6]-of-00048.safetensors"
#   WORKERS          parallel file workers (4)
# ---------------------------------------------------------------------------
set -euo pipefail

: "${MODEL_REPO:=deepseek-ai/DeepSeek-V4.1-Flash}"
: "${MODEL_REVISION:=}"
: "${MODEL_DIR:=/models/DeepSeek-V4.1-Flash}"
: "${ALLOW_PATTERNS:=}"
: "${WORKERS:=4}"

mkdir -p "$MODEL_DIR"

complete() {
    [[ -f "$MODEL_DIR/model.safetensors.index.json" ]] || return 1
    python3 - "$MODEL_DIR" <<'PY'
import json, os, sys
d = sys.argv[1]
idx = json.load(open(os.path.join(d, "model.safetensors.index.json")))
shards = sorted(set(idx["weight_map"].values()))
missing = [s for s in shards if not os.path.isfile(os.path.join(d, s))]
if missing:
    print(f"{len(missing)} of {len(shards)} shards still missing, first: {missing[0]}")
    sys.exit(1)
sys.exit(0)
PY
}

if [[ -z "$ALLOW_PATTERNS" ]] && complete >/dev/null 2>&1; then
    echo "checkpoint already complete in $MODEL_DIR"
    exit 0
fi

# 510 GB of weights plus the filesystem's own slack. The engine streams experts
# from these files with O_DIRECT on every miss, so this has to be local NVMe --
# a network filesystem turns every expert miss into a network round trip.
avail_gb=$(df -BG --output=avail "$MODEL_DIR" 2>/dev/null | tail -1 | tr -dc '0-9' || echo 0)
echo "WARNING: DeepSeek-V4.1-Flash is 510 GB (288.8 GB of FP4 experts, 203 GB of Engram tables)."
if [[ "${avail_gb:-0}" -lt 600 ]]; then
    echo "WARNING: only ${avail_gb} GB free under $MODEL_DIR; 600 GB is the comfortable figure." >&2
fi
[[ -n "${HF_TOKEN:-}" ]] && echo "using HF_TOKEN" || echo "no HF_TOKEN (fine -- the repo is public)"

echo "fetching ${MODEL_REPO}${MODEL_REVISION:+@$MODEL_REVISION} into $MODEL_DIR"
MODEL_REPO="$MODEL_REPO" MODEL_REVISION="$MODEL_REVISION" MODEL_DIR="$MODEL_DIR" \
ALLOW_PATTERNS="$ALLOW_PATTERNS" WORKERS="$WORKERS" python3 - <<'PY'
import os, sys
from huggingface_hub import snapshot_download

patterns = os.environ.get("ALLOW_PATTERNS", "").split() or None
kw = dict(
    repo_id=os.environ["MODEL_REPO"],
    local_dir=os.environ["MODEL_DIR"],
    max_workers=int(os.environ.get("WORKERS", "4")),
    token=os.environ.get("HF_TOKEN") or None,
)
if os.environ.get("MODEL_REVISION"):
    kw["revision"] = os.environ["MODEL_REVISION"]
if patterns:
    kw["allow_patterns"] = patterns
    print("allow_patterns:", patterns)
path = snapshot_download(**kw)
print("snapshot at", path)
PY

if [[ -n "$ALLOW_PATTERNS" ]]; then
    echo "partial download finished (ALLOW_PATTERNS was set) -- not checking for a complete index"
    exit 0
fi
complete || { echo "download finished but shards are missing -- re-run to resume" >&2; exit 1; }
echo "checkpoint complete"
