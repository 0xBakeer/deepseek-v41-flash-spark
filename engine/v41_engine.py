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
                 transient_slots: int = 400):
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

        try:
            import fp4_moe as K
            self.moe_fn, arena_cls = K.moe_forward, K.ExpertArena
            log("using Triton FP4 MoE kernel")
        except Exception as e:  # noqa: BLE001
            from engine import moe_fallback as K
            self.moe_fn, arena_cls = K.moe_forward, K.ExpertArena
            log(f"Triton kernel unavailable ({e!r}); using the slow dequant fallback")

        self.W = Weights(model_dir, index, self.args, device, log=log, act_quant=act_quant)
        self.caches = Caches(self.args, max_seq, device)
        # DSpark experts: all resident
        self.W.dspark_arena = arena_cls(384, device)
        self.W.dspark_store = FixedStore(self.W.dspark_arena)
        # main expert arena: size from what is left
        free, total = torch.cuda.mem_get_info()
        reserve = 6e9 + 0.25e9 * (max_seq / 8192)  # activations, indexer slices, page cache headroom
        if arena_gb is None:
            arena_gb = max(10.0, (free - reserve) / 1e9 * 0.90)
        slots = int(arena_gb * 1e9 / EX.EXPERT_BYTES)
        log(f"GPU free {free / 1e9:.1f} GB of {total / 1e9:.1f}; arena {arena_gb:.1f} GB = {slots} expert slots "
            f"({slots / 15360 * 100:.0f}% of all routed experts)")
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

    def generate(self, prompt_ids, *, max_tokens=4096, temperature=1.0, top_p=0.95, stop_token_ids=None, seed=None):
        with self.lock:
            yield from self._generate(list(prompt_ids), max_tokens, temperature, top_p, set(stop_token_ids or ()), seed)

    def _generate(self, prompt, max_tokens, temperature, top_p, stop_ids, seed):
        if seed is not None:
            torch.manual_seed(seed)
        stop_ids = set(stop_ids) | {self.eos_token_id}
        self._reset()
        m = self.model
        P = len(prompt)
        assert P + max_tokens + 8 <= self.max_context, f"prompt {P} + max_tokens {max_tokens} > context {self.max_context}"
        ids = torch.tensor(prompt, dtype=torch.long, device=self.device)
        t_start = time.perf_counter()
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
        p = sample_probs(logits[-1], temperature, top_p)
        tok = int(torch.multinomial(p, 1)) if temperature > 0 else int(p.argmax())
        out = [tok]
        yield [tok]
        n_out = 1
        pos = P  # position of `tok` (not yet forwarded)
        accepted_hist = []
        t_decode0 = time.perf_counter()
        steps = 0
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
                    yield emitted
                    if any(t in stop_ids for t in emitted):
                        break
                steps += 1
            else:
                logits, mh = m.forward(torch.tensor([tok], device=self.device), pos, prefill=False)
                pt = sample_probs(logits[0], temperature, top_p)
                tok = int(torch.multinomial(pt, 1)) if temperature > 0 else int(pt.argmax())
                pos += 1
                out.append(tok)
                n_out += 1
                steps += 1
                yield [tok]
        t_dec = time.perf_counter() - t_decode0
        st = self.store.stats
        self.last_stats = {
            "prompt_tokens": P, "completion_tokens": n_out, "prefill_s": round(t_prefill, 3),
            "decode_s": round(t_dec, 3), "decode_tok_s": round((n_out - 1) / max(t_dec, 1e-6), 2),
            "steps": steps, "accept_len_mean": round(float(np.mean(accepted_hist)) + 1, 2) if accepted_hist else None,
            "expert_hit_rate": round(self.store.hit_rate(), 4), "expert_misses": st["misses"],
            "prefill_expert_misses": st["prefill_misses"], "nvme_gb": round(st["bytes_read"] / 1e9, 2),
            "nvme_read_s": round(st["read_s"], 2), "engram_rows": sum(t.stats["rows"] for t in self.tables.values()),
            "engram_s": round(sum(t.stats["seconds"] for t in self.tables.values()), 3),
            "attn_s": round(m.stats["attn_s"], 2), "moe_s": round(m.stats["moe_s"], 2),
        }

    def stats(self):
        return dict(self.last_stats)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=os.path.expanduser("~/models/DeepSeek-V4.1-Flash"))
    ap.add_argument("--max-seq", type=int, default=32768)
    ap.add_argument("--arena-gb", type=float, default=None)
    ap.add_argument("--trace-stats", default=None)
    ap.add_argument("--prompt", default="Write a Python function that returns the n-th Fibonacci number.")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--no-spec", action="store_true")
    ap.add_argument("--thinking", action="store_true")
    a = ap.parse_args()
    eng = V41Engine(a.model_dir, max_seq=a.max_seq, arena_gb=a.arena_gb, trace_stats=a.trace_stats, spec=not a.no_spec)
    sys.path.insert(0, os.path.join(a.model_dir, "encoding"))
    from encoding import encode_messages
    prompt = encode_messages([{"role": "user", "content": a.prompt}], thinking_mode="thinking" if a.thinking else "chat")
    if isinstance(prompt, tuple):
        prompt = prompt[0]
    ids = eng.tokenizer.encode(prompt, add_special_tokens=False)
    text = []
    for burst in eng.generate(ids, max_tokens=a.max_tokens, temperature=a.temperature):
        s = eng.tokenizer.decode(burst)
        text.append(s)
        print(s, end="", flush=True)
    print()
    print(json.dumps(eng.stats(), indent=1))
