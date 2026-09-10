"""
v41_ref.py -- a pure-PyTorch re-implementation of the DeepSeek-V4.1-Flash text
forward pass for SHORT sequences (<= 512 tokens), used by tools/expert_trace.py.

Why this exists: the reference `inference/model.py` needs tilelang CUDA kernels,
a TP=8 checkpoint conversion, and the whole 510 GB model resident. For an expert
routing trace we only need to run the model layer by layer (streaming one 7.4 GB
layer shard at a time) over teacher-forced prompts, and record the router's
top-6 choices. Everything here mirrors `inference/model.py` and `kernel.py`
(read them side by side); deviations are listed at the bottom of this docstring.

Exactness argument for the attention path (see NOTES.md, Phase 0):
  * index_topk = 512 and candidate pool = 2048 blocks x 8 positions. For a
    sequence of T <= 512 tokens the number of compressed positions is
    T // ratio <= 512, so the indexer's top-k keeps EVERY causally visible
    position and the candidate filter never prunes. The indexer therefore has
    no effect on the output and is skipped here. This is exact, not an
    approximation, as long as T <= 512.
  * Sliding window (128) + all completed compressed groups, both concatenated
    into one softmax with the learned attention sink -- same as `sparse_attn`.

Deviations from the reference (all documented, none affect routing materially):
  * GEMMs run in bf16 with fp32 accumulation instead of fp8 x fp8 / fp8 x fp4
    tensor-core GEMMs. Activations ARE fake-quantized to fp8 (per-32 groups,
    power-of-two E8M0 scales) before every quantized linear, exactly like
    `act_quant(..., scale_fmt="ue8m0")`, and weights are dequantized with their
    stored block scales, so the numerics differ only by accumulation order.
  * The softmax probabilities are kept in fp32 for the PV product (the kernel
    casts them to bf16 first).
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from functools import lru_cache

import torch
import torch.nn.functional as F

FP4_TABLE = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)
FP4_GRID = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)


# ----------------------------------------------------------------------------- dtype helpers
def e8m0_to_float(s: torch.Tensor) -> torch.Tensor:
    """UE8M0 scale bytes -> float32 (2^(b-127)); avoids relying on torch's e8m0 cast support."""
    if s.dtype == torch.uint8:
        b = s
    else:
        b = s.view(torch.uint8)
    return torch.exp2(b.to(torch.float32) - 127.0)


def fp8_to_float(w: torch.Tensor) -> torch.Tensor:
    return w.to(torch.float32)


def dequant_fp8_block(weight: torch.Tensor, scale: torch.Tensor, block: int = 32) -> torch.Tensor:
    """fp8 [N,K] with UE8M0 scale [ceil(N/b), ceil(K/b)] -> bf16 [N,K]."""
    n, k = weight.shape
    s = e8m0_to_float(scale)
    s = s.repeat_interleave(block, 0)[:n].repeat_interleave(block, 1)[:, :k]
    return (weight.to(torch.float32) * s).to(torch.bfloat16)


def dequant_fp4_packed(weight_i8: torch.Tensor, scale: torch.Tensor, group: int = 32) -> torch.Tensor:
    """Packed FP4 [N, K/2] (int8/uint8, low nibble = even element) with UE8M0 scale [N, K/32] -> bf16 [N,K].
    Packing order matches `convert.py:cast_e2m1fn_to_e4m3fn`."""
    x = weight_i8.view(torch.uint8)
    low = x & 0x0F
    high = (x >> 4) & 0x0F
    tab = FP4_TABLE.to(x.device)
    vals = torch.stack([tab[low.long()], tab[high.long()]], dim=-1).flatten(1)  # [N, K]
    s = e8m0_to_float(scale).repeat_interleave(group, 1)  # [N, K]
    return (vals * s).to(torch.bfloat16)


def act_qdq_fp8(x: torch.Tensor, block: int = 32) -> torch.Tensor:
    """Emulates kernel.act_quant(x, 32, scale_fmt='ue8m0') followed by dequantization:
    per-row per-32-group amax, scale rounded UP to a power of two, values clamped to +-448 and
    rounded to e4m3. Returns bf16 of the same shape."""
    shape = x.shape
    xf = x.float().reshape(-1, block)
    amax = xf.abs().amax(dim=1, keepdim=True).clamp_min(1e-4)
    s = torch.exp2(torch.ceil(torch.log2(amax / 448.0)))
    y = (xf / s).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).float() * s
    return y.reshape(shape).to(torch.bfloat16)


def fp4_qdq(x: torch.Tensor, group: int, scale_dtype: str) -> torch.Tensor:
    """Emulates kernel.fp4_act_quant(x, group, inplace=True): FP4 e2m1 values with either
    UE8M0 (power-of-two) or E4M3 group scales, dequantized back. Used for the compressed
    (global) KV cache which the model stores in FP4 (E4M3 scales, groups of 16)."""
    shape = x.shape
    xf = x.float().reshape(-1, group)
    amax = xf.abs().amax(dim=1, keepdim=True)
    if scale_dtype == "e4m3":
        amax = amax.clamp_min(6 * 2 ** -9)
        s = (amax / 6.0).to(torch.float8_e4m3fn).float()
    else:
        amax = amax.clamp_min(6 * 2 ** -126)
        s = torch.exp2(torch.ceil(torch.log2(amax / 6.0)))
    v = (xf / s).clamp(-6.0, 6.0)
    grid = FP4_GRID.to(x.device)
    mid = (grid[1:] + grid[:-1]) / 2
    idx = torch.bucketize(v.abs(), mid)  # nearest grid point (ties go up; measure-zero on real data)
    q = grid[idx] * torch.sign(v)
    return (q * s).reshape(shape).to(x.dtype)


# ----------------------------------------------------------------------------- rope
@lru_cache(8)
def precompute_freqs_cis(dim, seqlen, original_seq_len, base, factor, beta_fast, beta_slow, device="cuda"):
    """Verbatim port of model.precompute_freqs_cis (YaRN when original_seq_len > 0)."""
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if original_seq_len > 0:
        def corrected_dim(rotations):
            return dim * math.log(original_seq_len / (rotations * 2 * math.pi)) / (2 * math.log(base))

        low = max(math.floor(corrected_dim(beta_fast)), 0)
        high = min(math.ceil(corrected_dim(beta_slow)), dim - 1)
        ramp = ((torch.arange(dim // 2, dtype=torch.float32) - low) / max(high - low, 1e-3)).clamp(0, 1)
        smooth = 1 - ramp
        freqs = freqs / factor * (1 - smooth) + freqs * smooth
    freqs = torch.outer(torch.arange(seqlen), freqs)
    return torch.polar(torch.ones_like(freqs), freqs).to(device)


def apply_rotary(x: torch.Tensor, freqs_cis: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    """Rotate the last dim of x (adjacent pairs as complex). x: [s, d] or [s, h, d]; freqs_cis: [s, d/2]."""
    xc = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
    f = freqs_cis.conj() if inverse else freqs_cis
    if xc.ndim == 3:
        f = f.view(xc.size(0), 1, xc.size(-1))
    y = torch.view_as_real(xc * f).flatten(-2)
    return y.to(x.dtype)


# ----------------------------------------------------------------------------- args
@dataclass
class Args:
    vocab_size: int = 129280
    dim: int = 5120
    moe_inter_dim: int = 2304
    n_layers: int = 40
    n_heads: int = 64
    n_routed_experts: int = 384
    n_activated_experts: int = 6
    route_scale: float = 1.5
    swiglu_limit: float = 10.0
    q_lora_rank: int = 1280
    head_dim: int = 512
    rope_head_dim: int = 64
    norm_eps: float = 1e-20
    o_groups: int = 8
    o_lora_rank: int = 1024
    window_size: int = 128
    kv_source_layers: tuple = (2, 8, 14, 20)
    index_source_layers: tuple = (2, 8, 14, 20, 24, 28, 32, 36)
    original_seq_len: int = 65536
    rope_theta: float = 10000.0
    rope_factor: float = 16
    beta_fast: int = 32
    beta_slow: int = 1
    index_topk: int = 512
    index_n_heads: int = 32
    index_head_dim: int = 128
    candidate_source_layer: int = 20
    dspark_target_layer_ids: tuple = (37, 38, 39)
    dspark_block_size: int = 5
    dspark_noise_token_id: int = 128799
    candidate_topk_blocks: int = 2048
    candidate_block_size: int = 8
    compress_rope_theta: float = 160000.0
    compress_ratios: tuple = tuple([0, 0] + [2] * 18 + [1] * 20 + [0, 0, 0])
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-6
    engram_layer_ids: tuple = (1, 14)
    engram_n_heads: int = 8
    engram_head_dim: int = 256
    engram_max_ngram_size: int = 4

    @classmethod
    def from_json(cls, path: str) -> "Args":
        cfg = json.load(open(path))
        keep = {k: v for k, v in cfg.items() if k in cls.__dataclass_fields__}
        for k in ("kv_source_layers", "index_source_layers", "compress_ratios", "engram_layer_ids", "dspark_target_layer_ids"):
            if k in keep:
                keep[k] = tuple(keep[k])
        return cls(**keep)


# ----------------------------------------------------------------------------- weights
class LayerWeights:
    """Everything of one backbone layer except the routed experts, dequantized to bf16 on `device`."""

    def __init__(self, get, layer: int, args: Args, device: str):
        p = f"layers.{layer}."
        self.layer = layer
        self.args = args
        dev = device

        def bf(name):
            return get(p + name).to(dev).to(torch.bfloat16)

        def f32(name):
            return get(p + name).to(dev).to(torch.float32)

        def fp8lin(name):
            return dequant_fp8_block(get(p + name + ".weight").to(dev), get(p + name + ".scale").to(dev))

        self.attn_norm = bf("attn_norm.weight")
        self.ffn_norm = bf("ffn_norm.weight")
        self.attn_sink = f32("attn.attn_sink")
        self.q_norm = bf("attn.q_norm.weight")
        self.kv_norm = bf("attn.kv_norm.weight")
        self.wq_a = fp8lin("attn.wq_a")
        self.wq_b = fp8lin("attn.wq_b")
        self.wkv = fp8lin("attn.wkv")
        self.wo_a = fp8lin("attn.wo_a").view(args.o_groups, args.o_lora_rank, -1)  # convert.py dequantizes it
        self.wo_b = fp8lin("attn.wo_b")
        self.hc_attn_fn = f32("hc_attn_fn")
        self.hc_ffn_fn = f32("hc_ffn_fn")
        self.hc_attn_base = f32("hc_attn_base")
        self.hc_ffn_base = f32("hc_ffn_base")
        self.hc_attn_scale = f32("hc_attn_scale")
        self.hc_ffn_scale = f32("hc_ffn_scale")
        self.gate_w = f32("ffn.gate.weight")
        self.gate_bias = f32("ffn.gate.bias")
        self.sh_w1 = fp8lin("ffn.shared_experts.w1")
        self.sh_w2 = fp8lin("ffn.shared_experts.w2")
        self.sh_w3 = fp8lin("ffn.shared_experts.w3")
        self.ratio = args.compress_ratios[layer]
        self.is_kv_source = layer in args.kv_source_layers
        if self.is_kv_source:
            self.comp_norm = bf("attn.compressor.norm.weight")
            if self.ratio > 1:
                self.comp_wkv = f32("attn.compressor.wkv.weight")
                self.comp_wgate = f32("attn.compressor.wgate.weight")
            else:
                self.comp_wkv = bf("attn.compressor.wkv.weight")


class EngramWeights:
    def __init__(self, get, layer: int, device: str):
        p = f"layers.{layer}.engram."
        self.wkv = dequant_fp8_block(get(p + "wkv.weight").to(device), get(p + "wkv.scale").to(device))
        self.q_weight = get(p + "q_weight").to(device).float()
        self.k_weight = get(p + "k_weight").to(device).float()


# ----------------------------------------------------------------------------- ops
MM_TILE = 0  # 0 = plain GEMMs. >0 = run every activation GEMM in fixed-size row tiles; see mm().


def mm(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """F.linear(x, w), but with a result that does not depend on how many rows are in the call.

    cuBLAS picks its tile shape AND its split-K count from M, so F.linear(x[:m], w) is generally
    NOT the first m rows of F.linear(x, w) -- for the small-N projections here (wkv N=512, the
    router gate N=384, hc_fn N=24) the two differ by ~1e-4 relative, which is enough to round a
    bf16 activation to a different ulp and to flip a borderline router top-k. A GEMM row never
    depends on the other rows in the call, so issuing every call with exactly MM_TILE rows makes
    the result identical for any chunk length. Set MM_TILE from engine/model.py; 0 keeps the
    plain behaviour for tools/expert_trace.py and the stored reference trace.
    """
    B = MM_TILE
    if B <= 0 or x.ndim != 2 or x.size(0) == B:
        return F.linear(x, w)
    m = x.size(0)
    n = (m + B - 1) // B * B
    if n != m:
        x = torch.cat([x, x.new_zeros(n - m, x.size(1))])
    out = torch.empty(n, w.size(0), dtype=torch.promote_types(x.dtype, w.dtype), device=x.device)
    for i in range(0, n, B):
        out[i:i + B] = F.linear(x[i:i + B], w)
    return out[:m]


def tiled_rows(fn, *xs: torch.Tensor):
    """Apply `fn` to its argument(s) in fixed-size row tiles when MM_TILE > 0 (see mm()).

    Torch picks the block/vector configuration of a last-dim reduction from the number of rows, so
    x.square().mean(-1) is not row-count-invariant either (it differs for M <= ~14 against a large
    M). Same cure as for the GEMMs: always work on exactly MM_TILE rows at a time. `fn` may return
    a tensor or a tuple of tensors; the tail rows of the last tile are zero padding and dropped.
    """
    B = MM_TILE
    m = xs[0].size(0)
    if B <= 0 or m == B:
        return fn(*xs)
    n = (m + B - 1) // B * B
    if n != m:
        xs = tuple(torch.cat([x, x.new_zeros(n - m, *x.shape[1:])]) for x in xs)
    outs = [fn(*[x[i:i + B] for x in xs]) for i in range(0, n, B)]
    if isinstance(outs[0], tuple):
        return tuple(torch.cat([o[k] for o in outs])[:m] for k in range(len(outs[0])))
    return torch.cat(outs)[:m]


def rms_rsqrt(x: torch.Tensor, eps: float) -> torch.Tensor:
    """rsqrt(mean(x^2, -1, keepdim=True) + eps), row-count-invariant."""
    return tiled_rows(lambda t: torch.rsqrt(t.square().mean(-1, keepdim=True) + eps), x)


def rmsnorm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    dtype = x.dtype
    xf = x.float()
    xf = xf * rms_rsqrt(xf, eps)
    return (w.float() * xf).to(dtype)


def qlinear(x: torch.Tensor, w_bf16: torch.Tensor) -> torch.Tensor:
    """Quantized-weight linear: fake-quantize the activation to fp8 (as the kernels do), bf16 GEMM."""
    return mm(act_qdq_fp8(x), w_bf16)


def hc_split_sinkhorn(mixes, hc_scale, hc_base, hc: int, iters: int, eps: float):
    """Port of kernel.hc_split_sinkhorn_kernel. mixes: [n, (2+hc)*hc] fp32."""
    pre = torch.sigmoid(mixes[:, :hc] * hc_scale[0] + hc_base[:hc]) + eps
    post = 2 * torch.sigmoid(mixes[:, hc : 2 * hc] * hc_scale[1] + hc_base[hc : 2 * hc])
    comb = (mixes[:, 2 * hc :] * hc_scale[2] + hc_base[2 * hc :]).view(-1, hc, hc)
    comb = comb.softmax(-1) + eps
    comb = comb / (comb.sum(-2, keepdim=True) + eps)
    for _ in range(iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
    return pre, post, comb


def hc_mixes(x: torch.Tensor, hc_fn, hc_scale, hc_base, args: Args):
    """x: [s, hc, d] -> (pre [s,hc], post [s,hc], comb [s,hc,hc]); normalized over the flattened stream."""
    xf = x.flatten(1).float()
    rsqrt = rms_rsqrt(xf, args.norm_eps)
    mixes = mm(xf, hc_fn) * rsqrt
    return tiled_rows(lambda t: hc_split_sinkhorn(t, hc_scale, hc_base, args.hc_mult,
                                                  args.hc_sinkhorn_iters, args.hc_eps), mixes)


def hc_pre(x: torch.Tensor, pre_mix: torch.Tensor) -> torch.Tensor:
    return torch.sum(pre_mix.unsqueeze(-1) * x.float(), dim=1).to(x.dtype)


def hc_post(x: torch.Tensor, residual: torch.Tensor, post: torch.Tensor, comb: torch.Tensor) -> torch.Tensor:
    y = post.unsqueeze(-1) * x.unsqueeze(1) + torch.sum(comb.unsqueeze(-1) * residual.unsqueeze(1), dim=2)
    return y.type_as(x)


def router(x: torch.Tensor, w: LayerWeights, args: Args):
    """Gate.forward for text tokens: sqrt(softplus(scores)); bias only steers selection."""
    scores = mm(x.float(), w.gate_w)
    scores = F.softplus(scores).sqrt()
    indices = (scores + w.gate_bias).topk(args.n_activated_experts, dim=-1)[1]
    weights = scores.gather(1, indices)
    weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
    weights = weights * args.route_scale
    return weights, indices, scores


def expert_ffn(x: torch.Tensor, w1, w2, w3, limit: float, weights: torch.Tensor | None = None) -> torch.Tensor:
    dtype = x.dtype
    gate = qlinear(x, w1).float()
    up = qlinear(x, w3).float()
    if limit > 0:
        up = torch.clamp(up, min=-limit, max=limit)
        gate = torch.clamp(gate, max=limit)
    h = F.silu(gate) * up
    if weights is not None:
        h = weights * h
    return qlinear(h.to(dtype), w2)


class ExpertLoader:
    """Loads one routed expert's three FP4 matrices from an open safetensors shard and dequantizes."""

    def __init__(self, get, layer: int, device: str):
        self.get, self.layer, self.device = get, layer, device

    def __call__(self, e: int):
        p = f"layers.{self.layer}.ffn.experts.{e}."
        d = self.device
        w1 = dequant_fp4_packed(self.get(p + "w1.weight").to(d), self.get(p + "w1.scale").to(d))
        w2 = dequant_fp4_packed(self.get(p + "w2.weight").to(d), self.get(p + "w2.scale").to(d))
        w3 = dequant_fp4_packed(self.get(p + "w3.weight").to(d), self.get(p + "w3.scale").to(d))
        return w1, w2, w3


# ----------------------------------------------------------------------------- attention (prefill, T <= 512)
class SeqState:
    """Per-sequence state carried across layers: hc residual stream, shifted pre_mix, shared compressed KV."""

    def __init__(self, h: torch.Tensor, pre_mix: torch.Tensor):
        self.h = h  # [T, hc, d] bf16
        self.pre_mix = pre_mix  # [T, hc] fp32
        self.compress_kv: torch.Tensor | None = None  # [T//ratio, head_dim] bf16, RoPE'd + FP4-qdq'd
        self.compress_ratio: int = 0


def attention(x: torch.Tensor, w: LayerWeights, st: SeqState, args: Args) -> torch.Tensor:
    """x: [T, d] (post attn_norm). Exact for T <= index_topk (see module docstring)."""
    T = x.size(0)
    dev = x.device
    rd = args.rope_head_dim
    ratio = w.ratio
    if ratio:
        freqs = precompute_freqs_cis(rd, T, args.original_seq_len, args.compress_rope_theta,
                                     args.rope_factor, args.beta_fast, args.beta_slow, str(dev))
    else:
        freqs = precompute_freqs_cis(rd, T, 0, args.rope_theta, args.rope_factor, args.beta_fast,
                                     args.beta_slow, str(dev))

    qr = rmsnorm(qlinear(x, w.wq_a), w.q_norm, args.norm_eps)
    q = qlinear(qr, w.wq_b).view(T, args.n_heads, args.head_dim)
    q = torch.cat([q[..., :-rd], apply_rotary(q[..., -rd:], freqs)], dim=-1)

    # sliding-window KV (fp8-quantized after RoPE, over the whole vector incl. the RoPE tail)
    kv = rmsnorm(qlinear(x, w.wkv), w.kv_norm, args.norm_eps)
    kv = torch.cat([kv[:, :-rd], apply_rotary(kv[:, -rd:], freqs)], dim=-1)
    kv = act_qdq_fp8(kv)
    pos = torch.arange(T, device=dev)
    win_mask = (pos[None, :] <= pos[:, None]) & (pos[None, :] > pos[:, None] - args.window_size)  # [T, T]

    kv_all, mask = kv, win_mask
    if ratio:
        if w.is_kv_source:
            if ratio > 1:
                xf = x.float()
                ckv, score = F.linear(xf, w.comp_wkv), F.linear(xf, w.comp_wgate)
                cutoff = T - T % ratio
                ckv, score = ckv[:cutoff].unflatten(0, (-1, ratio)), score[:cutoff].unflatten(0, (-1, ratio))
                latent = (ckv * score.softmax(dim=1)).sum(dim=1)
                latent = rmsnorm(latent.to(torch.bfloat16), w.comp_norm, args.norm_eps)
            else:
                latent = rmsnorm(F.linear(x, w.comp_wkv), w.comp_norm, args.norm_eps)
            n_c = latent.size(0)
            cf = freqs[: T - T % ratio : ratio]
            latent = torch.cat([latent[:, :-rd], apply_rotary(latent[:, -rd:], cf)], dim=-1)
            latent = fp4_qdq(latent, 16, "e4m3")
            st.compress_kv, st.compress_ratio = latent, ratio
        assert st.compress_kv is not None and st.compress_ratio == ratio, (w.layer, st.compress_ratio, ratio)
        ckv = st.compress_kv
        n_c = ckv.size(0)
        compress_lens = (pos + 1) // ratio  # groups completed before/at the query
        c_mask = torch.arange(n_c, device=dev)[None, :] < compress_lens[:, None]
        kv_all = torch.cat([kv, ckv], dim=0)
        mask = torch.cat([win_mask, c_mask], dim=1)

    scores = torch.einsum("thd,nd->thn", q.float(), kv_all.float()) * (args.head_dim ** -0.5)
    scores = scores.masked_fill(~mask[:, None, :], float("-inf"))
    m = scores.amax(dim=-1, keepdim=True).clamp_min(-1e30)
    p = torch.exp(scores - m)
    denom = p.sum(-1, keepdim=True) + torch.exp(w.attn_sink[None, :, None] - m)
    o = torch.einsum("thn,nd->thd", p / denom, kv_all.float()).to(torch.bfloat16)
    o = torch.cat([o[..., :-rd], apply_rotary(o[..., -rd:], freqs, inverse=True)], dim=-1)

    o = o.reshape(T, args.o_groups, -1)
    o = torch.einsum("sgd,grd->sgr", o, w.wo_a)
    return qlinear(o.flatten(1), w.wo_b)


# ----------------------------------------------------------------------------- engram
def engram_forward(h: torch.Tensor, rows: torch.Tensor, ew: EngramWeights, args: Args) -> torch.Tensor:
    """h: [T, hc, d]; rows: [T, 24, 256] float32 dequantized table rows (already scaled)."""
    T = h.size(0)
    kv = qlinear(rows.reshape(T, -1).to(torch.bfloat16), ew.wkv)
    key, value = kv.split([args.hc_mult * args.dim, args.dim], dim=-1)
    key = key.float().view(T, args.hc_mult, args.dim)
    weight = ew.q_weight * ew.k_weight
    hf, eps = h.float(), args.norm_eps

    def _gate(hh, kk):
        rstd = rms_rsqrt(hh, eps) * rms_rsqrt(kk, eps)
        dot = (hh * weight * kk).sum(-1, keepdim=True) * rstd * args.dim ** -0.5
        return torch.sigmoid(torch.copysign(dot.abs().clamp_min(1e-6).sqrt(), dot))

    gate = tiled_rows(_gate, hf, key)  # [T, hc, 1]
    return (hf + gate * value.float().unsqueeze(1)).to(h.dtype)


# ----------------------------------------------------------------------------- block
def block_forward(st: SeqState, w: LayerWeights, experts: ExpertLoader, args: Args, expert_cache: dict,
                  record=None):
    """One backbone block over one sequence. `record(indices, weights, scores)` receives the router output.
    `expert_cache` maps expert id -> (w1,w2,w3) for experts already dequantized in this layer."""
    x = st.h
    residual = x
    attn_pre, attn_post, attn_comb = hc_mixes(x, w.hc_attn_fn, w.hc_attn_scale, w.hc_attn_base, args)
    y = hc_pre(x, st.pre_mix)
    y = rmsnorm(y, w.attn_norm, args.norm_eps)
    y = attention(y, w, st, args)
    x = hc_post(y, residual, attn_post, attn_comb)

    residual = x
    ffn_pre, ffn_post, ffn_comb = hc_mixes(x, w.hc_ffn_fn, w.hc_ffn_scale, w.hc_ffn_base, args)
    y = hc_pre(x, attn_pre)
    y = rmsnorm(y, w.ffn_norm, args.norm_eps)

    weights, indices, scores = router(y, w, args)
    if record is not None:
        record(indices, weights, scores)
    out = torch.zeros_like(y, dtype=torch.float32)
    for e in torch.unique(indices).tolist():
        if e not in expert_cache:
            expert_cache[e] = experts(e)
        w1, w2, w3 = expert_cache[e]
        idx, top = torch.where(indices == e)
        out[idx] += expert_ffn(y[idx], w1, w2, w3, args.swiglu_limit, weights[idx, top, None]).float()
    out += expert_ffn(y, w.sh_w1, w.sh_w2, w.sh_w3, args.swiglu_limit).float()
    y = out.to(y.dtype)

    st.h = hc_post(y, residual, ffn_post, ffn_comb)
    st.pre_mix = ffn_pre
