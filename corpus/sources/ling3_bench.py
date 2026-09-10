#!/usr/bin/env python3
"""
bench.py -- measure Ling-3.0-flash on this DGX Spark on the SAME workload the
SGLang cookbook reports, so the numbers are directly comparable.

Cookbook cell (hw=dgx-spark, quant=mxfp4, strategy=low-latency, spec=dspark):
    dataset=random  isl=8192  osl=1024  max_concurrency=1
    -> ttft_ms 2172.03, tpot_ms 9.48   (== 105.5 output tok/s)

Two gotchas this harness exists to avoid:
  1. Speculative decoding packs SEVERAL tokens into one SSE chunk, so counting
     chunks under-reports badly. We always take usage.completion_tokens.
  2. Repeated filler text gets prefix-cached and re-tokenised at ~6.8 chars/token
     instead of ~4, which silently shortens the prompt. Every run builds a fresh
     random prompt of *verified* token length.

Usage:
    python3 bench.py --label humming-dspark --isl 8192 --osl 1024 --runs 3
"""
import argparse, hashlib, json, os, random, statistics, string, sys, time
from urllib import request as urlrequest

def post(base, path, payload, api_key=None, timeout=1800, stream=False):
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

def token_len(base, text, model, api_key=None):
    """Exact prompt length from the server's own tokenizer."""
    try:
        out = post(base, "/v1/completions",
                   {"model": model, "prompt": text, "max_tokens": 1,
                    "temperature": 0, "stream": False}, api_key)
        return json.loads(out.read())["usage"]["prompt_tokens"]
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

def fit_prompt(base, model, isl, seed, api_key):
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
        n = token_len(base, txt, model, api_key)
        if n is None:
            return txt, None
        if best_n is None or abs(n - isl) < abs(best_n - isl):
            best_txt, best_n = txt, n
        if abs(n - isl) <= max(8, isl // 200):      # within 0.5%
            return txt, n
        scaled = int(words * (isl / max(n, 1)))
        words = max(16, min(len(pool), scaled))
    return best_txt, best_n


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


def run_once(base, model, prompt, osl, api_key, thinking, temperature):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": osl,
        "ignore_eos": True,              # force the full osl so TPOT is a clean decode number
        "temperature": temperature,
        "top_p": 0.95,
        "top_k": 20,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": thinking},
    }
    t0 = time.perf_counter()
    ttft = None
    completion_tokens = None
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
        if obj.get("usage"):
            completion_tokens = obj["usage"].get("completion_tokens")
        ch = obj.get("choices") or []
        if ch and ttft is None:
            d = ch[0].get("delta") or {}
            if d.get("content") or d.get("reasoning_content"):
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
        "total_s": total,
    }

def main():
    ap = argparse.ArgumentParser()
    # Default follows $PORT (and $BENCH_BASE) so a bench can never silently
    # measure a closed port while the server runs somewhere else.
    ap.add_argument("--base", default=os.environ.get("BENCH_BASE")
                    or f"http://127.0.0.1:{os.environ.get('PORT', '30000')}")
    ap.add_argument("--model", default="ling-3.0-flash")
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--isl", type=int, default=8192)
    ap.add_argument("--osl", type=int, default=1024)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--thinking", action="store_true")
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--workload", default="random", choices=["random", "prose", "code"])
    ap.add_argument("--seed-salt", default=None,
                    help="Fixes the prompt seed. Runs that share a salt get IDENTICAL prompts, "
                         "which is required to compare two server configs on the random "
                         "workload -- there the generated text decides acceptance, so different "
                         "prompts are different experiments. Default: derived from --label, "
                         "which keeps two rows on the SAME server off each other's prefix cache.")
    ap.add_argument("--label", default="run")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    served = get_json(a.base, "/v1/models", a.api_key)["data"][0]["id"]
    if served != a.model:
        print(f"note: server serves '{served}', using that")
        a.model = served

    print(f"== {a.label}: workload={a.workload} isl={a.isl} osl={a.osl} runs={a.runs} thinking={a.thinking}")
    results = []
    # Seed from the label too: two benches in one server session (e.g. the
    # temp-0.6 and greedy rows) must not share prompts, or the second one is
    # served from the prefix cache and its TTFT is fiction.
    salt_src = a.seed_salt if a.seed_salt is not None else a.label
    label_salt = int(hashlib.sha1(salt_src.encode()).hexdigest()[:6], 16) % 100000
    for i in range(a.warmup + a.runs):
        seed = 1000 + i + label_salt          # fresh prompt every run: no cache hits
        if a.workload == "random":
            prompt, n = fit_prompt(a.base, a.model, a.isl, seed, a.api_key)
        else:
            # Real prompt; a unique tag keeps the prefix cache from serving run N
            # from run N-1's blocks, which would fake a near-zero TTFT.
            prompt = f"[req {seed}] " + WORKLOADS[a.workload]
            n = token_len(a.base, prompt, a.model, a.api_key)
        r = run_once(a.base, a.model, prompt, a.osl, a.api_key, a.thinking, a.temperature)
        r["prompt_tokens_actual"] = n
        tag = "warmup" if i < a.warmup else f"run{i - a.warmup + 1}"
        print(f"  {tag:7s} isl={n} ttft={r['ttft_ms']:8.1f} ms  tpot={r['tpot_ms']:6.2f} ms  "
              f"decode={r['decode_tok_s']:6.2f} tok/s  out={r['completion_tokens']}")
        if i >= a.warmup:
            results.append(r)

    med = lambda k: statistics.median(x[k] for x in results)
    summary = {
        "label": a.label, "workload": a.workload, "seed_salt": a.seed_salt, "isl": a.isl, "osl": a.osl, "runs": a.runs,
        "thinking": a.thinking,
        "ttft_ms_median": round(med("ttft_ms"), 2),
        "tpot_ms_median": round(med("tpot_ms"), 3),
        "decode_tok_s_median": round(med("decode_tok_s"), 2),
        "raw": results,
    }
    print(f"  -> MEDIAN  ttft={summary['ttft_ms_median']} ms  "
          f"tpot={summary['tpot_ms_median']} ms  decode={summary['decode_tok_s_median']} tok/s")
    if a.out:
        with open(a.out, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"  wrote {a.out}")

if __name__ == "__main__":
    sys.exit(main())
