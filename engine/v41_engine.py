"""
v41_engine.py -- the generation loop: chunked prefill, DSpark block drafting + verification with
rejection sampling, cache rollback, and the Engine API the server uses.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "tools"))

from engine import experts as EX  # noqa: E402
from engine.engram import EngramTable, make_hash_state  # noqa: E402
from engine.model import Caches, Model, Weights  # noqa: E402
import v41_ref as R  # noqa: E402


def log(*a):
    print(time.strftime("%H:%M:%S"), "[engine]", *a, flush=True)


def host_available_bytes():
    """MemAvailable from /proc/meminfo, or None off Linux."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


class FixedStore:
    """DSpark experts: 3 x 128, all resident, slot = k * 128 + e."""

    def __init__(self, arena):
        self.arena = arena
        self.stats = {"hits": 0, "misses": 0}

    def resolve(self, layer, experts, prefill):
        k = layer - 40
        return (experts.to(torch.int32) + k * 128)


def sample_probs(logits: torch.Tensor, temperature: float, top_p: float) -> torch.Tensor:
    """[V] fp32 logits -> probability vector (temperature + nucleus)."""
    if temperature <= 0:
        p = torch.zeros_like(logits)
        p[logits.argmax()] = 1.0
        return p
    p = torch.softmax(logits / temperature, dim=-1)
    if top_p < 1.0:
        sp, si = p.sort(descending=True)
        cum = sp.cumsum(0)
        keep = cum - sp < top_p
        sp = torch.where(keep, sp, torch.zeros_like(sp))
        p = torch.zeros_like(p).scatter_(0, si, sp)
        p = p / p.sum()
    return p


class V41Engine:
    def __init__(self, model_dir: str, max_seq: int = 32768, arena_gb: float | None = None, device: str = "cuda",
                 trace_stats: str | None = None, act_quant: bool = False, spec: bool = True, io_threads: int = 12,
                 transient_slots: int = 400, keep_free_gb: float = 20.0):
        self.model_dir = model_dir
        self.device = device
        self.spec = spec
        self.lock = threading.Lock()
        index = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))
        self.args = R.Args.from_json(os.path.join(model_dir, "inference", "config.json"))
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.eos_token_id = 1
        self.max_context = max_seq

        self.act_quant = act_quant
        self.trace_stats = trace_stats
        try:
            import fp4_moe as K
            self.moe_fn, arena_cls = K.moe_forward, K.ExpertArena
            self.kernel = "triton-fp4"
            log("using Triton FP4 MoE kernel")
        except Exception as e:  # noqa: BLE001
            from engine import moe_fallback as K
            self.moe_fn, arena_cls = K.moe_forward, K.ExpertArena
            self.kernel = "dequant-fallback"
            log(f"Triton kernel unavailable ({e!r}); using the slow dequant fallback")

        self.W = Weights(model_dir, index, self.args, device, log=log, act_quant=act_quant)
        self.caches = Caches(self.args, max_seq, device)
        # DSpark experts: all resident
        self.W.dspark_arena = arena_cls(384, device)
        self.W.dspark_store = FixedStore(self.W.dspark_arena)
        # main expert arena: size from what is left.
        # On GB10 the GPU and the host share one pool, and `torch.cuda.mem_get_info()` counts the
        # *page cache* as used -- after a 470 GB download it reports 30 GB free on a box with 118 GiB
        # actually available, which silently gives a 23 GB arena. /proc/meminfo's MemAvailable is the
        # honest number (it counts reclaimable page cache), so take the larger of the two and keep a
        # hard floor of `keep_free_gb` under it: this box hard-resets if MemAvailable goes negative.
        free, total = torch.cuda.mem_get_info()
        host_avail = host_available_bytes()
        budget = float(max(free, host_avail or 0))
        reserve = 8e9 + 0.25e9 * (max_seq / 8192)  # activations, indexer slices, page cache headroom
        auto = arena_gb is None
        if auto:
            arena_gb = max(10.0, (budget - reserve) / 1e9 * 0.82)
        if host_avail is not None:
            cap = (host_avail - keep_free_gb * 1e9) / 1e9
            if arena_gb > cap:
                log(f"arena {arena_gb:.1f} GB capped to {cap:.1f} GB (MemAvailable {host_avail / 1e9:.1f} GB, "
                    f"keep_free {keep_free_gb} GB)")
                arena_gb = max(10.0, cap)
        slots = int(arena_gb * 1e9 / EX.EXPERT_BYTES)
        self.arena_gb, self.slots = round(arena_gb, 1), slots
        log(f"CUDA free {free / 1e9:.1f} GB of {total / 1e9:.1f}; host MemAvailable "
            f"{(host_avail or 0) / 1e9:.1f} GB; arena {arena_gb:.1f} GB = {slots} expert slots "
            f"({slots / 15360 * 100:.0f}% of all routed experts, {'auto' if auto else 'pinned'})")
        self.arena = arena_cls(slots, device)
        self.store = EX.ExpertStore(model_dir, index, self.arena, self.args.n_layers, transient_slots=transient_slots,
                                    io_threads=io_threads)
        # load the DSpark experts (all 3 x 128 resident in their own arena)
        for k in range(3):
            for e in range(128):
                w1, s1, w2, s2, w3, s3 = self.store.read_expert(40 + k, e, prefix=f"mtp.{k}.ffn.experts.{e}.")
                self.W.dspark_arena.load_slot(k * 128 + e, w1.view(*EX.W13_SHAPE), s1.view(*EX.S13_SHAPE),
                                              w2.view(*EX.W2_SHAPE), s2.view(*EX.S2_SHAPE), w3.view(*EX.W13_SHAPE),
                                              s3.view(*EX.S13_SHAPE))
        log("DSpark experts resident")

        self.model = Model(self.W, self.store, self.caches, self.moe_fn, act_quant=act_quant)
        self.model.hash_state = make_hash_state(model_dir, self.tokenizer, max_seq, device)
        self.tables = {L: EngramTable(model_dir, index, L, device) for L in self.args.engram_layer_ids}
        self.model.engram_rows = lambda L, h: self.tables[L].rows(h)
        ranked = EX.rank_from_trace(trace_stats) if trace_stats else [(L, e) for e in range(384) for L in range(40)]
        self.store.warm_start(ranked, log=log)
        torch.cuda.synchronize()
        self.last_stats = {}
        log("ready")

    # ------------------------------------------------------------------ generation
    def _reset(self):
        c = self.caches
        c.len = 0
        c._chunk_inputs.clear()
        for L in c.pending:
            c.pending[L] = None
        self.model.stats = {"attn_s": 0.0, "moe_s": 0.0, "engram_s": 0.0, "tokens": 0}
        for t in self.tables.values():
            t.stats = {"rows": 0, "seconds": 0.0, "calls": 0}
        self.store.stats.update({"hits": 0, "misses": 0, "prefill_misses": 0, "bytes_read": 0, "read_s": 0.0})

    def generate(self, prompt_ids, *, max_tokens=4096, temperature=1.0, top_p=0.95, stop_token_ids=None, seed=None,
                 ignore_eos=False):
        """Yield bursts of new token ids.

        ``ignore_eos``: keep decoding until ``max_tokens`` even if EOS/stop ids come up. The
        benchmark needs runs of a fixed output length -- otherwise a decode-rate comparison is
        really a comparison of how early each config decided to stop.
        """
        with self.lock:
            yield from self._generate(list(prompt_ids), max_tokens, temperature, top_p, set(stop_token_ids or ()),
                                      seed, ignore_eos)

    def _generate(self, prompt, max_tokens, temperature, top_p, stop_ids, seed, ignore_eos=False):
        if seed is not None:
            torch.manual_seed(seed)
        stop_ids = set() if ignore_eos else (set(stop_ids) | {self.eos_token_id})
        self._reset()
        m = self.model
        P = len(prompt)
        assert P + max_tokens + 8 <= self.max_context, f"prompt {P} + max_tokens {max_tokens} > context {self.max_context}"
        ids = torch.tensor(prompt, dtype=torch.long, device=self.device)
        t_start = time.perf_counter()
        # Every counter the stats epilogue reads is initialised here, because the epilogue runs in a
        # `finally`: the server closes the generator on a stop string or a client disconnect, which
        # raises GeneratorExit at the pending `yield`, and an aborted request must still report the
        # work it did instead of the previous request's numbers.
        n_out = 0
        steps = 0
        pos = P
        accepted_hist = []
        t_prefill = 0.0
        t_decode0 = t_start
        try:
            yield from self._decode_loop(ids, P, max_tokens, temperature, top_p, stop_ids,
                                         _st := {})
        finally:
            n_out = _st.get("n_out", n_out)
            steps = _st.get("steps", steps)
            accepted_hist = _st.get("accepted", accepted_hist)
            t_prefill = _st.get("t_prefill", t_prefill)
            t_dec = max(time.perf_counter() - _st.get("t_decode0", t_start), 1e-9)
            st = self.store.stats
            m = self.model
            self.last_stats = {
                "prompt_tokens": P, "completion_tokens": n_out, "prefill_s": round(t_prefill, 3),
                "prefill_tok_s": round(P / t_prefill, 2) if t_prefill > 0 else None,
                "decode_s": round(t_dec, 3), "decode_tok_s": round(max(n_out - 1, 0) / t_dec, 2),
                "steps": steps,
                "accept_len_mean": round(float(np.mean(accepted_hist)) + 1, 2) if accepted_hist else None,
                "expert_hit_rate": round(self.store.hit_rate(), 4), "expert_misses": st["misses"],
                "prefill_expert_misses": st["prefill_misses"], "nvme_gb": round(st["bytes_read"] / 1e9, 2),
                "nvme_read_s": round(st["read_s"], 2),
                "engram_rows": sum(t.stats["rows"] for t in self.tables.values()),
                "engram_s": round(sum(t.stats["seconds"] for t in self.tables.values()), 3),
                "attn_s": round(m.stats["attn_s"], 2), "moe_s": round(m.stats["moe_s"], 2),
            }

    def _decode_loop(self, ids, P, max_tokens, temperature, top_p, stop_ids, out_st):
        m = self.model
        t_start = time.perf_counter()
        out_st["t_decode0"] = t_start
        # prefill in chunks
        logits = None
        main_tail = None
        for s in range(0, P, 512):
            chunk = ids[s:s + 512]
            last = s + len(chunk) >= P
            logits, mh = m.forward(chunk, s, prefill=True, need_logits=last)
            main_tail = mh if main_tail is None else torch.cat([main_tail, mh])[-256:]
            if self.spec:
                m.dspark_seed(mh, s)
        t_prefill = time.perf_counter() - t_start
        out_st["t_prefill"] = t_prefill
        p = sample_probs(logits[-1], temperature, top_p)
        tok = int(torch.multinomial(p, 1)) if temperature > 0 else int(p.argmax())
        out = [tok]
        n_out = 1
        pos = P  # position of `tok` (not yet forwarded)
        accepted_hist = []
        out_st.update(n_out=1, steps=0, accepted=accepted_hist, t_decode0=time.perf_counter())
        steps = 0
        yield [tok]
        while n_out < max_tokens and tok not in stop_ids:
            if self.spec:
                drafts, q, conf = m.dspark_draft(tok, pos - 1, temperature)
                block = torch.cat([torch.tensor([tok], device=self.device), drafts])  # 6 tokens at pos..pos+5
                logits, mh = m.forward(block, pos, prefill=False)
                # verify drafts[i] (position pos+1+i) against logits[i]
                a = 0
                new = []
                bonus = None
                for i in range(5):
                    pt = sample_probs(logits[i], temperature, top_p)
                    d = int(drafts[i])
                    if temperature <= 0:
                        ok = int(pt.argmax()) == d
                    else:
                        r = torch.rand((), device=self.device)
                        ok = bool(r < (pt[d] / q[i][d].clamp_min(1e-20)).clamp(max=1.0))
                    if ok:
                        a += 1
                        new.append(d)
                        if d in stop_ids:
                            break
                    else:
                        if temperature <= 0:
                            bonus = int(pt.argmax())
                        else:
                            resid = (pt - q[i]).clamp_min(0)
                            if float(resid.sum()) <= 0:
                                resid = pt
                            bonus = int(torch.multinomial(resid / resid.sum(), 1))
                        break
                if bonus is None and not (new and new[-1] in stop_ids):
                    pt = sample_probs(logits[a] if a < 5 else logits[5], temperature, top_p)
                    bonus = int(torch.multinomial(pt, 1)) if temperature > 0 else int(pt.argmax())
                # caches valid for positions < pos + a + 1 (tok + accepted drafts)
                m.c.rollback(pos + a + 1)
                m.dspark_seed(mh[:a + 1], pos)
                accepted_hist.append(a)
                emitted = list(new)
                if bonus is not None:
                    emitted.append(bonus)
                pos = pos + a + 1
                tok = emitted[-1] if emitted else tok
                if emitted:
                    out += emitted
                    n_out += len(emitted)
                    out_st["n_out"] = n_out
                    yield emitted
                    if any(t in stop_ids for t in emitted):
                        break
                steps += 1
                out_st["steps"] = steps
            else:
                logits, mh = m.forward(torch.tensor([tok], device=self.device), pos, prefill=False)
                pt = sample_probs(logits[0], temperature, top_p)
                tok = int(torch.multinomial(pt, 1)) if temperature > 0 else int(pt.argmax())
                pos += 1
                out.append(tok)
                n_out += 1
                steps += 1
                out_st.update(n_out=n_out, steps=steps)
                yield [tok]
        out_st.update(n_out=n_out, steps=steps)

    # ------------------------------------------------------------------ introspection
    def config(self):
        """Static engine configuration -- everything a measured number has to be quoted with."""
        return {
            "engine": "v41",
            "arena_gb": self.arena_gb,
            "arena_slots": self.slots,
            "lru_slots": self.store.lru_slots,
            "transient_slots": self.store.transient_slots,
            "resident_expert_pct": round(self.slots / (self.args.n_layers * self.args.n_routed_experts) * 100, 1),
            "max_seq": self.max_context,
            "spec": self.spec,
            "trace_stats": self.trace_stats,
            "kernel": self.kernel,
            "act_quant": self.act_quant,
        }

    def stats(self):
        return {**self.config(), **self.last_stats}

    def close(self):
        for pool in [self.store.pool] + [t.pool for t in self.tables.values()]:
            try:
                pool.shutdown(wait=False)
            except Exception:  # noqa: BLE001
                pass

    # ------------------------------------------------------------------ teacher forcing
    @torch.inference_mode()
    def teacher_forced(self, corpus_path: str, max_len: int = 512):
        """Run every corpus sequence through `Model.forward` in ONE chunk and report mean NLL and
        top-1 next-token accuracy per category -- the same measurement
        `tools/expert_trace.py` makes with the pure-torch tracer, so the two are directly
        comparable and any drift between `engine/model.py` and `tools/v41_ref.py` shows up here."""
        res = {}
        per_seq = []
        for line in open(corpus_path):
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            ids = self.tokenizer.encode(d["text"], add_special_tokens=False)
            assert len(ids) <= max_len, (d["id"], len(ids))
            self._reset()
            t0 = time.perf_counter()
            logits, _ = self.model.forward(torch.tensor(ids, dtype=torch.long, device=self.device), 0,
                                           prefill=True, need_logits=True)
            tgt = torch.tensor(ids[1:], device=self.device)
            lp = torch.log_softmax(logits[:-1].float(), dim=-1)
            nll = -lp.gather(1, tgt[:, None]).squeeze(1)
            top1 = (logits[:-1].argmax(-1) == tgt).float()
            c = res.setdefault(d["category"], {"nll": [], "top1": []})
            c["nll"].append(nll.cpu()); c["top1"].append(top1.cpu())
            per_seq.append({"id": d["id"], "category": d["category"], "n": len(ids),
                            "mean_nll": round(float(nll.mean()), 4), "top1_acc": round(float(top1.mean()), 4),
                            "seconds": round(time.perf_counter() - t0, 2)})
            log(f"{d['id']}: n={len(ids)} nll={float(nll.mean()):.4f} top1={float(top1.mean()):.4f} "
                f"({time.perf_counter() - t0:.1f}s)")
        summary = {k: {"mean_nll": float(torch.cat(v["nll"]).mean()), "top1_acc": float(torch.cat(v["top1"]).mean()),
                       "n": int(torch.cat(v["nll"]).numel())} for k, v in res.items()}
        return {"summary": summary, "config": self.config(), "per_seq": per_seq}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=os.environ.get("MODEL_DIR") or "./models/DeepSeek-V4.1-Flash")
    ap.add_argument("--max-seq", type=int, default=32768)
    ap.add_argument("--arena-gb", type=float, default=None)
    ap.add_argument("--trace-stats", default=None)
    ap.add_argument("--prompt", default="Write a Python function that returns the n-th Fibonacci number.")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--no-spec", action="store_true")
    ap.add_argument("--thinking", action="store_true")
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--ignore-eos", action="store_true")
    ap.add_argument("--act-quant", action="store_true",
                    help="fake-quantize activations to fp8 like the reference (and like the tracer)")
    ap.add_argument("--teacher-forced", default=None,
                    help="corpus jsonl: run each sequence through Model.forward in one chunk and report NLL/top-1")
    ap.add_argument("--tf-out", default=None, help="write the teacher-forced result JSON here")
    ap.add_argument("--spec-ab", action="store_true",
                    help="one load, three runs: greedy without spec, greedy with spec (must match) and "
                         "one sampled spec run at --temperature/--top-p")
    ap.add_argument("--ab-out", default=None, help="write the --spec-ab result JSON here")
    a = ap.parse_args()
    eng = V41Engine(a.model_dir, max_seq=a.max_seq, arena_gb=a.arena_gb, trace_stats=a.trace_stats,
                    spec=not a.no_spec, act_quant=a.act_quant)
    if a.teacher_forced:
        res = eng.teacher_forced(a.teacher_forced, max_len=min(512, a.max_seq))
        print(json.dumps(res["summary"], indent=1))
        if a.tf_out:
            os.makedirs(os.path.dirname(os.path.abspath(a.tf_out)), exist_ok=True)
            json.dump(res, open(a.tf_out, "w"), indent=1)
            print("wrote", a.tf_out)
        raise SystemExit(0)
    sys.path.insert(0, os.path.join(a.model_dir, "encoding"))
    from encoding import encode_messages
    prompt = encode_messages([{"role": "user", "content": a.prompt}], thinking_mode="thinking" if a.thinking else "chat")
    if isinstance(prompt, tuple):
        prompt = prompt[0]
    ids = eng.tokenizer.encode(prompt, add_special_tokens=False)
    def run(tag, spec, temperature, top_p):
        eng.spec = spec
        print(f"\n===== {tag}: spec={spec} temperature={temperature} top_p={top_p} =====", flush=True)
        toks, text = [], []
        for burst in eng.generate(ids, max_tokens=a.max_tokens, temperature=temperature, top_p=top_p,
                                  ignore_eos=a.ignore_eos):
            toks += list(burst)
            piece = eng.tokenizer.decode(burst)
            text.append(piece)
            print(piece, end="", flush=True)
        print()
        st = eng.stats()
        print(json.dumps(st, indent=1), flush=True)
        return {"tag": tag, "spec": spec, "temperature": temperature, "top_p": top_p,
                "tokens": toks, "text": "".join(text), "stats": st}

    if a.spec_ab:
        # One process, one load: `spec` is just a flag on the engine, so the two runs share the
        # arena and the LRU state. Greedy speculative decoding must reproduce greedy autoregressive
        # decoding token for token (the target verifies every draft), so the first divergence index
        # is the whole test.
        runs = [run("greedy-nospec", False, 0.0, 1.0), run("greedy-spec", True, 0.0, 1.0)]
        A, B = runs[0]["tokens"], runs[1]["tokens"]
        first = next((i for i in range(min(len(A), len(B))) if A[i] != B[i]), None)
        n_eq = len(A) if first is None else first
        print(f"\n== greedy spec vs non-spec: {n_eq}/{min(len(A), len(B))} identical leading tokens; "
              f"first divergence at {first}")
        if first is not None:
            print("  nospec:", repr(eng.tokenizer.decode(A[max(0, first - 8):first + 8])))
            print("  spec  :", repr(eng.tokenizer.decode(B[max(0, first - 8):first + 8])))
        if a.temperature > 0:
            runs.append(run(f"sampled-spec-t{a.temperature}", True, a.temperature, a.top_p))
        if a.ab_out:
            os.makedirs(os.path.dirname(os.path.abspath(a.ab_out)), exist_ok=True)
            json.dump({"prompt": a.prompt, "max_tokens": a.max_tokens, "config": eng.config(),
                       "identical_leading_tokens": n_eq, "first_divergence": first, "runs": runs},
                      open(a.ab_out, "w"), indent=1)
            print("wrote", a.ab_out)
    else:
        run("run", not a.no_spec, a.temperature, a.top_p)
