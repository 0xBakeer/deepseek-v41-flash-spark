#!/usr/bin/env python3
"""
bench.py -- measure the DeepSeek-V4.1-Flash server (server/app.py --engine v41)
on this DGX Spark.

Adapted from the ling3-flash-spark harness, so a row here lines up with a row
there, plus the two things that are specific to this recipe:

  * the model is NOT resident -- most of the 288.8 GB of routed experts live on
    NVMe and are streamed in on a miss. So a speed number is meaningless without
    the expert hit rate and the GB read that produced it. Every run records the
    server's ``x_engine_stats`` (acceptance length, expert hit rate, NVMe GB,
    engram rows) next to its tok/s.
  * two one-shot workloads (``angry-birds``, ``mario``) whose generated HTML is
    written to ``results/oneshots/`` so the games can actually be opened. Those
    are the same prompts the routing trace corpus uses
    (``tools/make_corpus.py::ONE_SHOTS``), which is what makes their expert
    hit rate comparable to the trace's coverage numbers.

Two gotchas inherited from the ling3 harness:
  1. Speculative decoding packs SEVERAL tokens into one SSE chunk, so counting
     chunks under-reports badly. We always take usage.completion_tokens.
  2. Repeated filler text re-tokenises at ~6.8 chars/token instead of ~4, which
     silently shortens the prompt. Every ``random`` run builds a fresh prompt of
     *verified* token length, measured with the server's own tokenizer via
     /v1/debug/prompt (which renders the prompt without running the engine).

Usage:
    python3 bench/bench.py --workload prose --runs 3 --out results/prose.json
    python3 bench/bench.py --workload random --isl 8192 --osl 1024
    python3 bench/bench.py --workload angry-birds --label hot --thinking
"""
import argparse, hashlib, json, os, random, re, statistics, string, sys, time
from urllib import request as urlrequest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
ONESHOT_DIR = os.path.join(REPO_ROOT, "results", "oneshots")


def post(base, path, payload, api_key=None, timeout=3600):
    data = json.dumps(payload).encode()
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urlrequest.Request(base.rstrip("/") + path, data=data, headers=headers)
    return urlrequest.urlopen(req, timeout=timeout)


def get_json(base, path, api_key=None, timeout=30):
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    req = urlrequest.Request(base.rstrip("/") + path, headers=headers)
    with urlrequest.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def token_len(base, prompt, model, api_key, thinking):
    """Exact rendered-prompt length from the server's own tokenizer.

    /v1/debug/prompt runs the chat template and the tokenizer and returns the
    ids -- but never touches the engine. The ling3 harness used a 1-token
    completion for this; here a prefill is minutes of NVMe streaming, so a
    fitting loop built on it would cost more than the benchmark.
    """
    try:
        out = post(base, "/v1/debug/prompt",
                   {"model": model, "messages": [{"role": "user", "content": prompt}],
                    "chat_template_kwargs": {"thinking": thinking}}, api_key, timeout=120)
        return json.loads(out.read())["prompt_tokens"]
    except Exception:
        return None


def make_prompt(target_tokens, seed):
    """Unique, non-repetitive text so nothing is prefix-cached or run-merged."""
    rnd = random.Random(seed)
    words = []
    for i in range(target_tokens * 3 + 64):
        w = "".join(rnd.choice(string.ascii_lowercase) for _ in range(rnd.randint(3, 9)))
        words.append(f"{i}:{w}")
    return " ".join(words)


def fit_prompt(base, model, isl, seed, api_key, thinking):
    """Scale a unique word list until the server's own tokenizer reports ~isl.

    Ratio iteration, not a bounded bisect: each filler item is several tokens,
    so a word-count bisect with naive bounds saturates at its ceiling and
    silently hands back a prompt twice the intended length.
    """
    pool = make_prompt(isl, seed).split()
    words = min(len(pool), max(16, isl // 2))
    best_txt, best_n = None, None
    for _ in range(8):
        txt = " ".join(pool[:words])
        n = token_len(base, txt, model, api_key, thinking)
        if n is None:
            return txt, None
        if best_n is None or abs(n - isl) < abs(best_n - isl):
            best_txt, best_n = txt, n
        if abs(n - isl) <= max(8, isl // 200):      # within 0.5%
            return txt, n
        scaled = int(words * (isl / max(n, 1)))
        words = max(16, min(len(pool), scaled))
    return best_txt, best_n


# The one-shot prompts are copied verbatim from tools/make_corpus.py::ONE_SHOTS.
# Do not reword them: the routing trace in results/ was taken on these exact
# strings, so changing them makes the hit rates incomparable.
ONE_SHOTS = {
    "angry-birds": "Write a complete Angry Birds style game as a single self-contained HTML file: canvas rendering, "
                   "a slingshot with drag-to-aim, projectile physics with gravity, destructible block structures, "
                   "pigs as targets, a score counter, and a restart button. No external libraries or assets.",
    "mario": "Write a complete side-scrolling Mario style platformer as a single HTML file with inline JavaScript: "
             "keyboard controls, jumping with gravity, moving enemies you can stomp, coins, a scrolling level, "
             "and a win condition. No external libraries.",
}

WORKLOADS = {
    # Short prompt, long natural generation -- the number that matches how the
    # box is actually used day to day, and what a speculative drafter is good at.
    "prose": "Write a detailed, flowing essay of about 900 words on how tidal "
             "forces shaped the evolution of coastal ecosystems. Use full "
             "paragraphs and continuous prose, no bullet points or headings.",
    "code":  "Write a complete, production-quality Python module implementing an "
             "LRU cache with a TTL per entry, thread safety, and an eviction "
             "callback. Include full docstrings, type hints, and a pytest suite "
             "covering expiry, eviction order, and concurrent access.",
}
WORKLOADS.update(ONE_SHOTS)
ALL_WORKLOADS = ["random"] + sorted(WORKLOADS)

# Stats worth putting in the summary line; the full x_engine_stats is kept raw.
HEADLINE_STATS = ["accept_len_mean", "expert_hit_rate", "nvme_gb", "engram_rows"]


def run_once(base, model, prompt, osl, api_key, thinking, effort, temperature, seed=None):
    """One streamed chat completion. Returns timings + the engine's own stats."""
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": osl,
        "temperature": temperature,
        "top_p": 0.95,
        "stream": True,
        "stream_options": {"include_usage": True},
        # server/README.md precedence: chat_template_kwargs.thinking is checked
        # first and is the only unambiguous on/off switch. reasoning_effort is
        # sent separately so it sets the budget without flipping thinking back on.
        "chat_template_kwargs": {"thinking": bool(thinking)},
    }
    if effort is not None:
        body["reasoning_effort"] = effort
    if seed is not None:
        body["seed"] = seed
    t0 = time.perf_counter()
    ttft = None
    completion_tokens = None
    reasoning_tokens = None
    finish_reason = None
    engine_stats = {}
    content, reasoning = [], []
    resp = post(base, "/v1/chat/completions", body, api_key)
    for raw in resp:
        line = raw.decode("utf-8", "replace").strip()
        if not line.startswith("data:"):
            continue
        chunk = line[5:].strip()
        if chunk == "[DONE]":
            break
        try:
            obj = json.loads(chunk)
        except json.JSONDecodeError:
            continue
        if obj.get("error"):
            raise RuntimeError(f"server error mid-stream: {obj['error']}")
        # x_engine_stats and usage ride the final chunk (the one with
        # finish_reason); include_usage adds a second, empty-choices one.
        if obj.get("x_engine_stats"):
            engine_stats = obj["x_engine_stats"]
        if obj.get("usage"):
            completion_tokens = obj["usage"].get("completion_tokens")
            details = obj["usage"].get("completion_tokens_details") or {}
            reasoning_tokens = details.get("reasoning_tokens")
        ch = obj.get("choices") or []
        if ch:
            if ch[0].get("finish_reason"):
                finish_reason = ch[0]["finish_reason"]
            d = ch[0].get("delta") or {}
            if d.get("content"):
                content.append(d["content"])
            if d.get("reasoning_content"):
                reasoning.append(d["reasoning_content"])
            if ttft is None and (d.get("content") or d.get("reasoning_content")):
                ttft = time.perf_counter() - t0
    total = time.perf_counter() - t0
    if not completion_tokens:
        raise RuntimeError("server returned no usage.completion_tokens -- cannot measure honestly")
    if ttft is None:
        ttft = total
    decode_s = max(total - ttft, 1e-9)
    tpot_ms = (decode_s / max(completion_tokens - 1, 1)) * 1000.0
    return {
        "ttft_ms": ttft * 1000.0,
        "tpot_ms": tpot_ms,
        "decode_tok_s": (completion_tokens - 1) / decode_s,
        "completion_tokens": completion_tokens,
        "reasoning_tokens": reasoning_tokens,
        "finish_reason": finish_reason,
        "total_s": total,
        "x_engine_stats": engine_stats,
        "_content": "".join(content),
        "_reasoning": "".join(reasoning),
    }


# ---------------------------------------------------------------------------
# One-shot artefacts
# ---------------------------------------------------------------------------

_FENCE = re.compile(r"```(?:html|HTML)?\s*\n(.*?)```", re.S)


def extract_artifact(text):
    """Return (body, extension) -- a playable HTML file when one is in there."""
    for block in _FENCE.findall(text):
        if re.search(r"<!doctype html|<html|<canvas", block, re.I):
            return block.strip() + "\n", "html"
    m = re.search(r"(<!doctype html.*|<html.*)", text, re.I | re.S)
    if m:
        return m.group(1).strip() + "\n", "html"
    return text, "txt"


def save_oneshot(label, workload, run_tag, text):
    os.makedirs(ONESHOT_DIR, exist_ok=True)
    body, ext = extract_artifact(text)
    stem = f"{label}-{workload}" if run_tag in ("", "run1") else f"{label}-{workload}-{run_tag}"
    path = os.path.join(ONESHOT_DIR, f"{stem}.{ext}")
    with open(path, "w") as f:
        f.write(body)
    return path


def main():
    ap = argparse.ArgumentParser(
        description="Benchmark the DeepSeek-V4.1-Flash server on this box.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Default follows $PORT (and $BENCH_BASE) so a bench can never silently
    # measure a closed port while the server runs somewhere else.
    ap.add_argument("--base", default=os.environ.get("BENCH_BASE")
                    or f"http://127.0.0.1:{os.environ.get('PORT', '8000')}")
    ap.add_argument("--model", default="deepseek-v4.1-flash")
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--workload", default="prose", choices=ALL_WORKLOADS)
    ap.add_argument("--isl", type=int, default=8192, help="target prompt tokens (random workload only)")
    ap.add_argument("--osl", type=int, default=None,
                    help="max output tokens (default 1024, or 16384 for a one-shot game)")
    ap.add_argument("--runs", type=int, default=None, help="measured runs (default 3, or 1 for a one-shot)")
    ap.add_argument("--warmup", type=int, default=None, help="unmeasured runs first (default 1, or 0 for a one-shot)")
    ap.add_argument("--thinking", action="store_true", help="ask for thinking mode (default: server default, off)")
    ap.add_argument("--effort", type=int, default=None, help="reasoning effort 1-100 (only meaningful with --thinking)")
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=None, help="sampling seed passed to the server")
    ap.add_argument("--seed-salt", default=None,
                    help="Fixes the prompt seed. Runs that share a salt get IDENTICAL prompts, "
                         "which is required to compare two server configs on the random "
                         "workload -- there the generated text decides both acceptance and which "
                         "experts are touched, so different prompts are different experiments. "
                         "Default: derived from --label.")
    ap.add_argument("--label", default="run")
    ap.add_argument("--out", default=None, help="write the summary JSON here")
    ap.add_argument("--no-save-oneshot", action="store_true",
                    help="do not write results/oneshots/<label>-<workload>.html")
    a = ap.parse_args()

    oneshot = a.workload in ONE_SHOTS
    if a.osl is None:
        a.osl = 16384 if oneshot else 1024
    if a.runs is None:
        a.runs = 1 if oneshot else 3
    if a.warmup is None:
        a.warmup = 0 if oneshot else 1

    served = get_json(a.base, "/v1/models", a.api_key)["data"][0]["id"]
    if served != a.model:
        print(f"note: server serves '{served}', using that")
        a.model = served
    health = get_json(a.base, "/health", a.api_key)

    print(f"== {a.label}: workload={a.workload} isl={a.isl if a.workload == 'random' else 'n/a'} "
          f"osl={a.osl} runs={a.runs} warmup={a.warmup} thinking={a.thinking} engine={health.get('engine')}")
    results = []
    # Seed from the label too: two benches in one server session (e.g. the
    # thinking and non-thinking rows) must not share prompts, or the second is
    # served warm -- both from any prefix cache and from an expert arena the
    # first run already filled, and its numbers are fiction.
    salt_src = a.seed_salt if a.seed_salt is not None else a.label
    label_salt = int(hashlib.sha1(salt_src.encode()).hexdigest()[:6], 16) % 100000
    for i in range(a.warmup + a.runs):
        seed = 1000 + i + label_salt          # fresh prompt every run
        if a.workload == "random":
            prompt, n = fit_prompt(a.base, a.model, a.isl, seed, a.api_key, a.thinking)
        elif oneshot:
            # No uniqueness tag: these prompts must stay byte-identical to the
            # ones the routing trace was taken on.
            prompt = WORKLOADS[a.workload]
            n = token_len(a.base, prompt, a.model, a.api_key, a.thinking)
        else:
            # Real prompt; a unique tag keeps a prefix cache from serving run N
            # from run N-1's blocks, which would fake a near-zero TTFT.
            prompt = f"[req {seed}] " + WORKLOADS[a.workload]
            n = token_len(a.base, prompt, a.model, a.api_key, a.thinking)
        r = run_once(a.base, a.model, prompt, a.osl, a.api_key, a.thinking, a.effort, a.temperature, a.seed)
        r["prompt_tokens_actual"] = n
        measured = i >= a.warmup
        tag = "warmup" if not measured else f"run{i - a.warmup + 1}"
        st = r["x_engine_stats"] or {}
        if oneshot and not a.no_save_oneshot:
            r["artifact"] = save_oneshot(a.label, a.workload, tag, r["_content"] or r["_reasoning"])
        text = r.pop("_content")
        r.pop("_reasoning")
        r["completion_chars"] = len(text)
        print(f"  {tag:7s} isl={n} ttft={r['ttft_ms']:9.1f} ms  tpot={r['tpot_ms']:7.2f} ms  "
              f"decode={r['decode_tok_s']:6.2f} tok/s  out={r['completion_tokens']} ({r['finish_reason']})")
        print(f"          accept_len={st.get('accept_len_mean')}  expert_hit={st.get('expert_hit_rate')}  "
              f"nvme={st.get('nvme_gb')} GB  engram_rows={st.get('engram_rows')}")
        if "artifact" in r:
            print(f"          wrote {os.path.relpath(r['artifact'], REPO_ROOT)}")
        if measured:
            results.append(r)

    def med(key):
        vals = [x[key] for x in results if x.get(key) is not None]
        return round(statistics.median(vals), 4) if vals else None

    engine_meds = {}
    for k in HEADLINE_STATS:
        vals = [x["x_engine_stats"].get(k) for x in results
                if isinstance(x.get("x_engine_stats"), dict) and x["x_engine_stats"].get(k) is not None]
        engine_meds[k] = round(statistics.median(vals), 4) if vals else None

    summary = {
        "label": a.label, "workload": a.workload, "seed_salt": a.seed_salt,
        "isl": a.isl if a.workload == "random" else None, "osl": a.osl,
        "runs": a.runs, "warmup": a.warmup,
        "thinking": a.thinking, "effort": a.effort, "temperature": a.temperature,
        "base": a.base, "model": a.model, "health": health,
        "ttft_ms_median": med("ttft_ms"),
        "tpot_ms_median": med("tpot_ms"),
        "decode_tok_s_median": med("decode_tok_s"),
        "engine_median": engine_meds,
        "raw": results,
    }
    print(f"  -> MEDIAN  ttft={summary['ttft_ms_median']} ms  tpot={summary['tpot_ms_median']} ms  "
          f"decode={summary['decode_tok_s_median']} tok/s")
    print(f"  -> ENGINE  accept_len={engine_meds['accept_len_mean']}  "
          f"expert_hit={engine_meds['expert_hit_rate']}  nvme={engine_meds['nvme_gb']} GB  "
          f"engram_rows={engine_meds['engram_rows']}")
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"  wrote {a.out}")


if __name__ == "__main__":
    sys.exit(main())
