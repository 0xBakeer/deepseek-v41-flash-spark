"""
fastdecode.py -- CUDA-graph decode path for the 6-token DSpark verify step (and the 5-token draft).

Why: profiled with every expert resident, one verify step of `Model.forward` costs ~436 ms, of which
~360 ms is CPU launch overhead (~10k tiny ops) and the GPU side is slowed by fp32 GEMMs (the fp32 LM
head alone: 37 ms at 72 GB/s). This module runs the same math with static buffers so it can be
captured into CUDA graphs, uses bf16 GEMMs (fp32 accumulation) everywhere the checkpoint stores bf16
or fp8, one fused Triton kernel for the Hyper-Connection Sinkhorn coefficients, and fixed-length
(masked) indexer scoring instead of context-length-dependent slices.

Semantics are those of engine/model.py (window ring, shared compressed KV, Hierarchical Sparse
Indexer with the layer-20 candidate pool, engram rows, mHC single-pass shift, DSpark draft with the
Markov head). Per backbone layer there are two graphs: A = attention + HC + router (ends with the
expert ids), then the host resolves expert slots (LRU / NVMe), then B = MoE + shared expert + HC
residual. With every routed expert resident (`--prune-keep`) the host step is a LUT lookup and the
whole layer could be one graph; that is left for later.

Numerics vs `Model.forward`: identical math, but GEMMs are not issued in fixed 16-row tiles and the
head is bf16, so the two paths can differ in the last bits (the chunk-invariance guarantee of
model.py does not extend across the two paths). `engine/test_fastdecode.py` measures the gap.
"""

from __future__ import annotations

import time

import torch
import torch.nn.functional as F

from engine import model as M
from engine.hc_sinkhorn import hc_split_sinkhorn
import v41_ref as R

T_VERIFY = 6   # tok + 5 drafts
T_DRAFT = 5


def _lin(x, w):  # bf16 tensor -> cuBLAS; FP8Weight -> Triton fp8 kernel (stored format, half the bytes)
    return R.dense(x, w)


class FastDecoder:
    def __init__(self, model: M.Model, engine, use_graphs: bool = True):
        self.m = model
        self.eng = engine
        self.a = model.args
        self.dev = model.dev
        self.W = model.W
        self.c = model.c
        self.use_graphs = use_graphs
        a = self.a
        dev = self.dev
        T = T_VERIFY
        # ---- static buffers (inputs of the graphs)
        self.ids = torch.zeros(T, dtype=torch.long, device=dev)
        self.pos = torch.zeros(T, dtype=torch.long, device=dev)
        self.eg_rows = {L: torch.zeros(T, a.engram_n_heads * (a.engram_max_ngram_size - 1), a.engram_head_dim,
                                       dtype=torch.float32, device=dev) for L in a.engram_layer_ids}
        self.slots = torch.zeros(T, a.n_activated_experts, dtype=torch.int32, device=dev)
        # ---- static state carried between graphs of one step
        self.h = torch.zeros(T, a.hc_mult, a.dim, dtype=torch.bfloat16, device=dev)
        self.pre_mix = torch.zeros(T, a.hc_mult, dtype=torch.float32, device=dev)
        self.attn_pre = torch.zeros(T, a.hc_mult, dtype=torch.float32, device=dev)
        self.ffn_post = torch.zeros(T, a.hc_mult, dtype=torch.float32, device=dev)
        self.ffn_comb = torch.zeros(T, a.hc_mult, a.hc_mult, dtype=torch.float32, device=dev)
        self.ffn_pre = torch.zeros(T, a.hc_mult, dtype=torch.float32, device=dev)
        self.y = torch.zeros(T, a.dim, dtype=torch.bfloat16, device=dev)
        self.route_idx = torch.zeros(T, a.n_activated_experts, dtype=torch.long, device=dev)
        self.route_w = torch.zeros(T, a.n_activated_experts, dtype=torch.float32, device=dev)
        self.topk = torch.full((T, a.index_topk), -1, dtype=torch.long, device=dev)
        n_cand = self._n_cache(1)
        self.candidates = torch.zeros(T, n_cand, dtype=torch.bool, device=dev)
        self.main_hidden = torch.zeros(T, a.dim * len(a.dspark_target_layer_ids), dtype=torch.float32, device=dev)
        self.logits = torch.zeros(T, a.vocab_size, dtype=torch.float32, device=dev)
        # draft
        self.d_tok = torch.zeros(1, dtype=torch.long, device=dev)
        self.d_last = torch.zeros(1, dtype=torch.long, device=dev)   # last main position
        self.d_noise = torch.zeros(T_DRAFT, a.vocab_size, dtype=torch.float32, device=dev)  # gumbel noise
        self.d_temp = torch.zeros(1, dtype=torch.float32, device=dev)
        self.d_out = torch.zeros(T_DRAFT, dtype=torch.long, device=dev)
        self.d_probs = torch.zeros(T_DRAFT, a.vocab_size, dtype=torch.float32, device=dev)
        # weights in decode-friendly dtypes (views/copies; small)
        self.head_bf16 = self.W.head.to(torch.bfloat16)
        self.gate_bf16 = [w.gate_w.to(torch.bfloat16) for w in self.W.layers]
        self.mtp_gate_bf16 = [w.gate_w.to(torch.bfloat16) for w in self.W.mtp]
        self.markov_embed_bf16 = self.W.mtp[2].markov_embed.to(torch.bfloat16)
        self.markov_head_bf16 = self.W.mtp[2].markov_head.to(torch.bfloat16)
        self.win_off = torch.arange(a.window_size - 1, -1, -1, device=dev)
        self.graphs = {}
        self.pool = None
        # resident mode: (layer, expert) -> arena slot as a device table, so the router's expert ids can be
        # turned into slots inside the graph and the whole layer is ONE graph (no host round-trip per layer)
        self.lut = None
        self.lut_version = -1
        self.pend_buf = {L: torch.zeros(2, a.head_dim, dtype=torch.float32, device=dev)
                         for L in a.kv_source_layers if a.compress_ratios[L] > 1}
        self.kvl_buf = {L: torch.zeros(T, a.head_dim, dtype=torch.float32, device=dev) for L in self.pend_buf}
        self.sc_buf = {L: torch.zeros(T, a.head_dim, dtype=torch.float32, device=dev) for L in self.pend_buf}
        R.MM_TILE = 0  # plain GEMMs in this path (and from now on in prefill too); the 16-row tiling was a test aid
        self.stats = {"steps": 0, "graph_s": 0.0, "resolve_s": 0.0, "engram_s": 0.0, "draft_s": 0.0}

    # ------------------------------------------------------------------ helpers
    def _n_cache(self, r):
        # rows of the shared compressed cache for ratio r (Caches allocates max_seq // r + 1)
        return self.c.max_seq // r + 1

    def _rope(self, x, fq, inverse=False):
        rd = self.a.rope_head_dim
        return torch.cat([x[..., :-rd], R.apply_rotary(x[..., -rd:], fq, inverse=inverse)], dim=-1)

    def _hc_mixes(self, x, hc_fn, hc_scale, hc_base):
        xf = x.flatten(1).float()
        rsqrt = torch.rsqrt(xf.square().mean(-1, keepdim=True) + self.a.norm_eps)
        mixes = F.linear(xf, hc_fn) * rsqrt
        return hc_split_sinkhorn(mixes, hc_scale, hc_base, self.a.hc_mult, self.a.hc_sinkhorn_iters, self.a.hc_eps)

    # ------------------------------------------------------------------ attention (decode, T tokens)
    def _attention(self, x, w, L, ring, freqs, pos, sh_state, mtp_last=None):
        a = self.a
        T = x.size(0)
        fq = freqs[pos]
        qr = R.rmsnorm(_lin(x, w.wq_a), w.q_norm, a.norm_eps)
        q = self._rope(_lin(qr, w.wq_b).view(T, a.n_heads, a.head_dim), fq)
        kv = self._rope(R.rmsnorm(_lin(x, w.wkv), w.kv_norm, a.norm_eps), fq)
        if mtp_last is None:
            wpos = pos[:, None] - self.win_off[None, :]
            ring[pos % M.RING] = kv
            wkv = ring[wpos.clamp_min(0) % M.RING]
            wmask = wpos >= 0
            kv_all, mask = wkv, wmask
            if w.ratio:
                rows, cmask = self._compressed(x, qr, w, L, pos, sh_state)
                kv_all = torch.cat([wkv, rows], dim=1)
                mask = torch.cat([wmask, cmask], dim=1)
        else:
            wpos = mtp_last - self.win_off  # [128]
            wkv = ring[wpos.clamp_min(0) % M.RING][None].expand(T, -1, -1)
            kv_all = torch.cat([wkv, kv[None].expand(T, -1, -1)], dim=1)
            mask = torch.cat([(wpos >= 0)[None].expand(T, -1),
                              torch.ones(T, T, dtype=torch.bool, device=self.dev)], dim=1)
        scale = a.head_dim ** -0.5
        scores = torch.einsum("thd,tnd->thn", q.float(), kv_all.float()) * scale
        scores = scores.masked_fill(~mask[:, None, :], float("-inf"))
        mx = scores.amax(dim=-1, keepdim=True).clamp_min(-1e30)
        p = torch.exp(scores - mx)
        denom = p.sum(-1, keepdim=True) + torch.exp(w.attn_sink[None, :, None] - mx)
        o = torch.einsum("thn,tnd->thd", p / denom, kv_all.float()).to(torch.bfloat16)
        o = self._rope(o, fq, inverse=True).reshape(T, a.o_groups, -1)
        o = torch.einsum("sgd,grd->sgr", o, w.wo_a)
        return _lin(o.flatten(1), w.wo_b)

    def _compressed(self, x, qr, w, L, pos, st):
        """st: dict with 'parity' (python int, S % 2, fixed per graph), 'ckv', 'ik', 'ratio'."""
        a = self.a
        r = w.ratio
        c = self.c
        T = x.size(0)
        if w.is_kv_source:
            if r > 1:
                xf = x.float()
                kvl, sc = F.linear(xf, w.comp_wkv), F.linear(xf, w.comp_wgate)
                # static per-layer copies for Caches.rollback (host patches the tuple after the step)
                self.kvl_buf[L].copy_(kvl); self.sc_buf[L].copy_(sc)
                buf = self.pend_buf[L]  # [2, 512]: kv, score of the pending token (host-maintained)
                if st["parity"] == 1:  # S odd: pending + t0, (t1,t2), (t3,t4); t5 -> new pending
                    kvl2 = torch.cat([buf[0][None], kvl]); sc2 = torch.cat([buf[1][None], sc])
                    g_kv = kvl2[:6].unflatten(0, (-1, r)); g_sc = sc2[:6].unflatten(0, (-1, r))
                    buf[0].copy_(kvl[5]); buf[1].copy_(sc[5])
                else:  # S even: (t0,t1),(t2,t3),(t4,t5), no pending
                    g_kv = kvl.unflatten(0, (-1, r)); g_sc = sc.unflatten(0, (-1, r))
                latent = (g_kv * g_sc.softmax(dim=1)).sum(dim=1)
                latent = R.rmsnorm(latent.to(torch.bfloat16), w.comp_norm, a.norm_eps)
                j0 = (pos[0] - st["parity"]) // r
            else:
                latent = R.rmsnorm(_lin(x, w.comp_wkv), w.comp_norm, a.norm_eps)
                j0 = pos[0]
            nj = latent.size(0)
            jidx = j0 + torch.arange(nj, device=self.dev)
            fj = self.m.freqs_c[jidx * r]
            if L in self.W.indexers:
                iw = self.W.indexers[L]
                k = R.rmsnorm(_lin(latent, iw.wk), iw.k_norm, a.norm_eps)
                c.ik[L][jidx] = self._rope(k, fj)
            c.ckv[L][jidx] = self._rope(latent, fj)
            st["ckv"], st["ik"], st["ratio"] = c.ckv[L], c.ik[L], r
        compress_lens = (pos + 1) // r
        if L in self.W.indexers:
            self.topk.copy_(self._indexer(x, qr, L, pos, compress_lens, st))
        idx = self.topk
        rows = st["ckv"][idx.clamp_min(0)]
        return rows, idx >= 0

    def _indexer(self, x, qr, L, pos, compress_lens, st):
        a = self.a
        iw = self.W.indexers[L]
        T = x.size(0)
        q = self._rope(_lin(qr, iw.wq_b).view(T, a.index_n_heads, a.index_head_dim), self.m.freqs_c[pos])
        wts = _lin(x, iw.weights_proj).float() * (a.index_head_dim ** -0.5 * a.index_n_heads ** -0.5)
        k = st["ik"]  # full cache [N, 128]; positions >= compress_lens are masked below
        sc = torch.einsum("thd,nd->thn", q, k)
        score = (sc.float().relu_() * wts[:, :, None]).sum(dim=1)  # [T, N]
        cpos = torch.arange(score.size(1), device=self.dev)
        score.masked_fill_(cpos[None, :] >= compress_lens[:, None], float("-inf"))
        if L == a.candidate_source_layer:
            self.candidates.copy_(M.Model._select_candidates(score, compress_lens, a.candidate_topk_blocks,
                                                             a.candidate_block_size)[:, :self.candidates.size(1)])
        elif 0 <= a.candidate_source_layer < L:
            score = score.masked_fill(~self.candidates[:, :score.size(1)], float("-inf"))
        idx = score.topk(a.index_topk, dim=-1, sorted=False).indices.sort(dim=-1).values
        return torch.where(idx < compress_lens[:, None], idx, torch.full_like(idx, -1))

    # ------------------------------------------------------------------ layer graphs
    def _tap(self, name, L, t):
        f = getattr(self, 'tap', None)
        if f is not None:
            f(name, L, t)

    def _layer_a(self, L, sh_state):
        """attention + HC + router for backbone layer L, reading self.h/self.pre_mix/self.pos."""
        a = self.a
        w = self.W.layers[L]
        h = self.h
        self._tap('h_in', L, h)
        if L in self.W.engram:
            h = R.engram_forward(h, self.eg_rows[L], self.W.engram[L], a)
        if L in a.dspark_target_layer_ids:
            i = list(a.dspark_target_layer_ids).index(L)
            self.main_hidden[:, i * a.dim:(i + 1) * a.dim] = h.float().mean(dim=1)
        residual = h
        attn_pre, attn_post, attn_comb = self._hc_mixes(h, w.hc_attn_fn, w.hc_attn_scale, w.hc_attn_base)
        y = R.rmsnorm(R.hc_pre(h, self.pre_mix), w.attn_norm, a.norm_eps)
        self._tap('attn_x', L, y)
        y = self._attention(y, w, L, self.c.win[L], self.m.freqs_c if w.ratio else self.m.freqs_w, self.pos, sh_state)
        self._tap('attn_out', L, y)
        h = R.hc_post(y, residual, attn_post, attn_comb)
        self.h.copy_(h)
        ffn_pre, ffn_post, ffn_comb = self._hc_mixes(h, w.hc_ffn_fn, w.hc_ffn_scale, w.hc_ffn_base)
        self.ffn_pre.copy_(ffn_pre); self.ffn_post.copy_(ffn_post); self.ffn_comb.copy_(ffn_comb)
        y = R.rmsnorm(R.hc_pre(h, attn_pre), w.ffn_norm, a.norm_eps)
        self.y.copy_(y)
        scores = F.softplus(F.linear(y, self.gate_bf16[L]).float()).sqrt()  # bf16 weights (as stored) x bf16 act, fp32 accumulate
        logits = scores + w.gate_bias
        pm = getattr(self.m, "prune_mask", None)
        if pm is not None and L in pm:
            logits = logits.masked_fill(~pm[L], float("-inf"))
        idx = logits.topk(a.n_activated_experts, dim=-1)[1]
        wts = scores.gather(1, idx)
        wts = wts / (wts.sum(dim=-1, keepdim=True) + 1e-20) * a.route_scale
        self.route_idx.copy_(idx); self.route_w.copy_(wts)
        self._tap('moe_in', L, y); self._tap('route_idx', L, idx); self._tap('topk', L, self.topk)

    def _layer_b(self, L):
        a = self.a
        w = self.W.layers[L]
        out = self.m.moe_fn(self.y, self.slots, self.route_w, self.m.store.arena, a.swiglu_limit).float()
        out += R.expert_ffn(self.y, w.sh_w1, w.sh_w2, w.sh_w3, a.swiglu_limit).float()
        h = R.hc_post(out.to(torch.bfloat16), self.h, self.ffn_post, self.ffn_comb)
        self.h.copy_(h); self.pre_mix.copy_(self.ffn_pre)

    def _final(self):
        a = self.a
        x = R.rmsnorm(R.hc_pre(self.h, self.pre_mix), self.W.norm, a.norm_eps)
        self.logits.copy_(_lin(x, self.head_bf16).float())
        # seed the drafter rings for all 6 positions (harmless beyond the accepted ones)
        m0 = self.W.mtp[0]
        main_x = R.rmsnorm(_lin(self.main_hidden.to(torch.bfloat16), m0.main_proj), m0.main_norm, a.norm_eps)
        fq = self.m.freqs_w[self.pos]
        for k, w in enumerate(self.W.mtp):
            kv = self._rope(R.rmsnorm(_lin(main_x, w.wkv), w.kv_norm, a.norm_eps), fq)
            self.c.mtp_win[k][self.pos % M.RING] = kv

    def _draft(self):
        """DSpark draft: 5 positions after d_last; gumbel-max sampling with self.d_noise (temperature in d_temp,
        0 => argmax)."""
        a = self.a
        ids = torch.full((T_DRAFT,), a.dspark_noise_token_id, dtype=torch.long, device=self.dev)
        ids[0] = self.d_tok[0]
        h = self.W.embed[ids].unsqueeze(1).repeat(1, a.hc_mult, 1)
        pre_mix = torch.zeros(T_DRAFT, a.hc_mult, device=self.dev); pre_mix[:, 0] = 1.0
        pos = self.d_last[0] + 1 + torch.arange(T_DRAFT, device=self.dev)
        for k, w in enumerate(self.W.mtp):
            residual = h
            attn_pre, attn_post, attn_comb = self._hc_mixes(h, w.hc_attn_fn, w.hc_attn_scale, w.hc_attn_base)
            y = R.rmsnorm(R.hc_pre(h, pre_mix), w.attn_norm, a.norm_eps)
            y = self._attention(y, w, 40 + k, self.c.mtp_win[k], self.m.freqs_w, pos, None, mtp_last=self.d_last[0])
            h = R.hc_post(y, residual, attn_post, attn_comb)
            residual = h
            ffn_pre, ffn_post, ffn_comb = self._hc_mixes(h, w.hc_ffn_fn, w.hc_ffn_scale, w.hc_ffn_base)
            y = R.rmsnorm(R.hc_pre(h, attn_pre), w.ffn_norm, a.norm_eps)
            scores = F.softplus(F.linear(y, self.mtp_gate_bf16[k]).float()).sqrt()
            idx = (scores + w.gate_bias).topk(3, dim=-1)[1]
            wts = scores.gather(1, idx); wts = wts / (wts.sum(dim=-1, keepdim=True) + 1e-20) * a.route_scale
            slots = (idx.to(torch.int32) + k * 128)
            out = self.m.moe_fn(y, slots, wts, self.W.dspark_arena, a.swiglu_limit).float()
            out += R.expert_ffn(y, w.sh_w1, w.sh_w2, w.sh_w3, a.swiglu_limit).float()
            h = R.hc_post(out.to(torch.bfloat16), residual, ffn_post, ffn_comb)
            pre_mix = ffn_pre
        w = self.W.mtp[2]
        x = R.rmsnorm(R.hc_pre(h, pre_mix), w.norm, a.norm_eps)
        logits = _lin(x, self.head_bf16).float()  # [5, V]
        prev = self.d_tok[0]
        temp = self.d_temp[0]
        for i in range(T_DRAFT):
            bias = _lin(self.markov_embed_bf16[prev.view(1)], self.markov_head_bf16).float()[0]  # 1-d index: a 0-d tensor index syncs
            lg = logits[i] + bias
            greedy = lg.argmax()
            p = torch.softmax(lg / temp.clamp_min(1e-5), dim=-1)
            sampled = (torch.log(p.clamp_min(1e-30)) + self.d_noise[i]).argmax()
            nxt = torch.where(temp > 0, sampled, greedy)
            onehot = torch.zeros_like(p).scatter_(0, greedy.view(1), 1.0)
            self.d_probs[i].copy_(torch.where(temp > 0, p, onehot))
            self.d_out[i] = nxt
            prev = nxt

    def build_lut(self):
        """Device slot table from the store's LRU. Only valid while no expert is evicted/loaded; the
        engine rebuilds it whenever the store reports a miss."""
        st = self.m.store
        lut = torch.full((self.a.n_layers, self.a.n_routed_experts), -1, dtype=torch.int32)
        for (L, e), slot in st.lru.items():
            lut[L, e] = slot
        self.lut = lut.to(self.dev)
        self.lut_version = st.stats.get("misses", 0)

    def _layer_ab(self, L, sh_state):
        self._layer_a(L, sh_state)
        self.slots.copy_(self.lut[L][self.route_idx])  # -1 never occurs while the LUT is valid
        self._layer_b(L)

    # ------------------------------------------------------------------ capture
    def capture(self, S_parity: int):
        """Capture the per-layer graphs for a step whose start position has the given parity (the
        ratio-2 compressor grouping depends on it). Two captures per process at most."""
        if not self.use_graphs:
            return
        key = S_parity
        if key in self.graphs:
            return
        if self.pool is None:
            self.pool = torch.cuda.graph_pool_handle()
        st = {"parity": S_parity, "ckv": None, "ik": None, "ratio": 0}
        gA, gB = [], []
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            # warm-up run (allocations, triton compiles) on a scratch copy of the state
            saved = [self.h.clone(), self.pre_mix.clone()]
            for L in range(self.a.n_layers):
                self._layer_a(L, st); self._layer_b(L)
            self._final(); self._draft()
            self.h.copy_(saved[0]); self.pre_mix.copy_(saved[1])
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        st = {"parity": S_parity, "ckv": None, "ik": None, "ratio": 0}
        for L in range(self.a.n_layers):
            if self.lut is not None:
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g, pool=self.pool):
                    self._layer_ab(L, st)
                gA.append(g); gB.append(None)
                continue
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=self.pool):
                self._layer_a(L, st)
            gA.append(g)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=self.pool):
                self._layer_b(L)
            gB.append(g)
        gF = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gF, pool=self.pool):
            self._final()
        gD = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gD, pool=self.pool):
            self._draft()
        self.graphs[key] = (gA, gB, gF, gD)
        torch.cuda.synchronize()

    # ------------------------------------------------------------------ run
    def _resolve(self, L):
        # host: expert ids -> arena slots (loads misses from NVMe)
        idx = self.route_idx
        slots = self.m.store.resolve(L, idx, False)
        self.slots.copy_(slots)

    def step(self, block_ids: torch.Tensor, S: int, engram_rows: dict):
        """Run the 6-token verify block at positions S..S+5. engram_rows: {L: [6,24,256] fp32}.
        Returns (logits [6,V] fp32, main_hidden [6, 15360] fp32) as views of static buffers."""
        a = self.a
        assert block_ids.numel() == T_VERIFY
        assert self.c.len == S, (self.c.len, S)
        self.ids.copy_(block_ids)
        self.pos.copy_(S + torch.arange(T_VERIFY, device=self.dev))
        for L, rows in engram_rows.items():
            self.eg_rows[L].copy_(rows)
        self.h.copy_(self.W.embed[self.ids].unsqueeze(1).repeat(1, a.hc_mult, 1))
        self.pre_mix.zero_(); self.pre_mix[:, 0] = 1.0
        parity = S % 2
        self.prepare_pending_buffers()
        self.capture(parity)
        self.prepare_pending_buffers()  # capture's warm-up/capture runs overwrite the buffers
        t0 = time.perf_counter()
        if self.use_graphs:
            gA, gB, gF, gD = self.graphs[parity]
            for L in range(a.n_layers):
                gA[L].replay()
                if gB[L] is not None:
                    self._resolve(L)
                    gB[L].replay()
            gF.replay()
            if self.lut is not None:
                # bookkeeping the host resolve would have done: LRU touch is irrelevant while resident
                self.m.store.stats["hits"] += int(a.n_layers * self.route_idx.numel())
        else:
            st = {"parity": parity, "ckv": None, "ik": None, "ratio": 0}
            for L in range(a.n_layers):
                self._layer_a(L, st); self._resolve(L); self._layer_b(L)
            self._final()
        # host-side bookkeeping for Caches.rollback (same tuple layout as model.py)
        for L in self.kvl_buf:
            before = self.c.pending.get(L)
            self.c._chunk_inputs[L] = (S, self.kvl_buf[L], self.sc_buf[L], before)
            if parity == 1:
                self.c.pending[L] = (self.kvl_buf[L][5].clone(), self.sc_buf[L][5].clone())  # t5 unpaired
            else:
                self.c.pending[L] = None
        self.c.len = S + T_VERIFY
        self.stats["steps"] += 1
        self.stats["graph_s"] += time.perf_counter() - t0
        return self.logits, self.main_hidden

    def draft(self, tok: int, last_main_pos: int, temperature: float):
        self.d_tok.fill_(tok); self.d_last.fill_(last_main_pos); self.d_temp.fill_(temperature)
        if temperature > 0:
            u = torch.rand_like(self.d_noise).clamp_min(1e-30)
            self.d_noise.copy_(-torch.log(-torch.log(u)))
        t0 = time.perf_counter()
        if self.use_graphs and self.graphs:
            self.graphs[next(iter(self.graphs))][3].replay()
        else:
            self._draft()
        self.stats["draft_s"] += time.perf_counter() - t0
        return self.d_out, self.d_probs

    def prepare_pending_buffers(self):
        """Copy Caches.pending (host truth) into the static buffers the graphs read."""
        for L, buf in self.pend_buf.items():
            p = self.c.pending.get(L)
            if p is not None:
                buf[0].copy_(p[0]); buf[1].copy_(p[1])
