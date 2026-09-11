"""
decode_attn.py -- one Triton kernel for the sinked softmax attention of the decode path.

Shape of the problem (engine/fastdecode.py `_attention`): T query tokens (6 in a verify block,
5 in a DSpark draft), h = 64 query heads, and a per-token key set that every head of that token
shares. That key set comes in two pieces which the caller used to `torch.cat` into one [T, n, d]
tensor: the 128 sliding-window rows gathered from the layer's ring, and either the 512 rows the
CSA2 indexer selected from the compressed cache or (in the draft) the T draft keys themselves.
d = head_dim = 512 (the last 64 dims carry RoPE). The torch version ran this as two fp32 SIMT
batched GEMMs plus the masked_fill / exp / sum elementwise passes around them.

The kernel takes the two pieces as two base pointers plus the boundary n1 and walks them as one key
axis, so the `torch.cat` disappears (it wrote and re-read ~3.9 MB per layer, ~1.3 ms per step). The
second piece may also be a stride-0 broadcast view, which is what the draft's window is, so that
expand is not materialised either. The mask stays a single [T, n1+n2] tensor -- catting the two
masks costs 3.8 kB per layer and keeps the indexing simple.

The math is flash-decoding of exactly what the torch path did:
  scores = q . k * head_dim**-0.5, masked to -inf,
  m      = max_n scores, clamped to >= -1e30 (an all-masked row keeps a finite m),
  denom  = sum_n exp(scores - m) + exp(sink_h - m),
  o      = sum_n exp(scores - m) / denom * k.
Q and K are read in bf16 and every dot accumulates in fp32; the scores and the softmax never touch
bf16. The PV product would, since `tl.dot` needs a tensor-core dtype for both operands, so the
probability matrix is split into a bf16 high part and a bf16 remainder and the two are accumulated
separately (PV_SPLIT=1) -- that keeps ~16 mantissa bits of the probabilities instead of 8, which
matters because the probabilities are dense (~640 keys, so a single bf16 rounding of p costs about
as much accuracy as the final bf16 rounding of o).

d is handled as DA + DB, two powers of two, so that a head dim which is not itself a power of two
(e.g. 576) does not have to be padded up to 1024: the accumulator is the register-hungry part. For
this checkpoint d = 512 and DB = 0, so the second block is compiled away.

With T = 6 and 64 heads there are only T * 64/BLOCK_H = 24 programs, half the 48 SMs of a GB10, so
the key axis is also split SPLIT ways; the partial (acc, m, l) triples are combined by a second,
cheap kernel. Both kernels have static shapes, allocate nothing beyond their outputs and do no host
synchronisation, so the pair captures into a CUDA graph.
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

NEG = -1e30
_NEG = tl.constexpr(-1e30)  # module-level constexpr so the jitted kernels may read it


@triton.jit
def _seg(KV, MB, lo, hi, base, q1, q2, m_i, l_i, acc1, acc2, stride_kn, scale,
         DA: tl.constexpr, DB: tl.constexpr, DBP: tl.constexpr,
         BLOCK_H: tl.constexpr, BLOCK_N: tl.constexpr, PV_SPLIT: tl.constexpr):
    """Online-softmax pass over the key rows [lo, hi) of one segment. `base` is subtracted from the
    global key index to get the row inside this segment; the mask is always indexed globally."""
    da = tl.arange(0, DA)
    db = tl.arange(0, DBP)
    for n0 in range(lo, hi, BLOCK_N):
        offs_n = n0 + tl.arange(0, BLOCK_N)
        nm = offs_n < hi
        kp = KV + (offs_n - base)[:, None] * stride_kn
        k1 = tl.load(kp + da[None, :], mask=nm[:, None], other=0.0)
        s = tl.dot(q1, tl.trans(k1), out_dtype=tl.float32)
        if DB > 0:
            k2 = tl.load(kp + (DA + db)[None, :], mask=nm[:, None], other=0.0)
            s += tl.dot(q2, tl.trans(k2), out_dtype=tl.float32)
        else:
            k2 = tl.zeros((BLOCK_N, DBP), tl.bfloat16)
        s = s * scale
        keep = tl.load(MB + offs_n, mask=nm, other=0) != 0
        s = tl.where(keep[None, :] & nm[None, :], s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        m_new = tl.maximum(m_new, _NEG)  # all-masked rows keep a finite max, as the torch path does
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        acc1 = acc1 * alpha[:, None]
        acc2 = acc2 * alpha[:, None]
        ph = p.to(tl.bfloat16)
        acc1 += tl.dot(ph, k1, out_dtype=tl.float32)
        if PV_SPLIT:
            pl = (p - ph.to(tl.float32)).to(tl.bfloat16)
            acc1 += tl.dot(pl, k1, out_dtype=tl.float32)
        if DB > 0:
            acc2 += tl.dot(ph, k2, out_dtype=tl.float32)
            if PV_SPLIT:
                acc2 += tl.dot(pl, k2, out_dtype=tl.float32)
        m_i = m_new
    return m_i, l_i, acc1, acc2


@triton.jit
def _dattn_kernel(Q, KV1, KV2, MSK, SINK, OUT, MP, LP,
                  T, H, N, N1,
                  stride_qt, stride_qh, stride_k1t, stride_k1n, stride_k2t, stride_k2n, stride_mt,
                  stride_ot, stride_oh, stride_os,
                  stride_pt, stride_ph,
                  scale,
                  DA: tl.constexpr, DB: tl.constexpr, DBP: tl.constexpr, BLOCK_H: tl.constexpr,
                  BLOCK_N: tl.constexpr, SPLIT: tl.constexpr, FINAL: tl.constexpr,
                  PV_SPLIT: tl.constexpr, TWO: tl.constexpr):
    pid_h = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_s = tl.program_id(2)
    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    h_mask = offs_h < H
    da = tl.arange(0, DA)
    db = tl.arange(0, DBP)

    qb = Q + pid_t * stride_qt + offs_h[:, None] * stride_qh
    q1 = tl.load(qb + da[None, :], mask=h_mask[:, None], other=0.0)
    if DB > 0:  # head dims that are not a power of two (D = DA + DB); compile-time branch
        q2 = tl.load(qb + (DA + db)[None, :], mask=h_mask[:, None], other=0.0)
    else:
        q2 = tl.zeros((BLOCK_H, DBP), tl.bfloat16)

    per = tl.cdiv(N, SPLIT)
    n_lo = pid_s * per
    n_hi = tl.minimum(n_lo + per, N)

    m_i = tl.full((BLOCK_H,), _NEG, tl.float32)
    l_i = tl.zeros((BLOCK_H,), tl.float32)
    acc1 = tl.zeros((BLOCK_H, DA), tl.float32)
    acc2 = tl.zeros((BLOCK_H, DBP), tl.float32)
    mb = MSK + pid_t * stride_mt

    m_i, l_i, acc1, acc2 = _seg(KV1 + pid_t * stride_k1t, mb, n_lo, tl.minimum(n_hi, N1), 0,
                                q1, q2, m_i, l_i, acc1, acc2, stride_k1n, scale,
                                DA, DB, DBP, BLOCK_H, BLOCK_N, PV_SPLIT)
    if TWO:
        m_i, l_i, acc1, acc2 = _seg(KV2 + pid_t * stride_k2t, mb, tl.maximum(n_lo, N1), n_hi, N1,
                                    q1, q2, m_i, l_i, acc1, acc2, stride_k2n, scale,
                                    DA, DB, DBP, BLOCK_H, BLOCK_N, PV_SPLIT)

    if FINAL:
        sink = tl.load(SINK + offs_h, mask=h_mask, other=0.0).to(tl.float32)
        denom = l_i + tl.exp(sink - m_i)  # all-masked row: exp(sink + 1e30) = inf -> o = 0
        ob = OUT + pid_t * stride_ot + offs_h[:, None] * stride_oh
        tl.store(ob + da[None, :], (acc1 / denom[:, None]).to(tl.bfloat16), mask=h_mask[:, None])
        if DB > 0:
            tl.store(ob + (DA + db)[None, :], (acc2 / denom[:, None]).to(tl.bfloat16), mask=h_mask[:, None])
    else:
        ob = OUT + pid_t * stride_ot + offs_h[:, None] * stride_oh + pid_s * stride_os
        tl.store(ob + da[None, :], acc1, mask=h_mask[:, None])
        if DB > 0:
            tl.store(ob + (DA + db)[None, :], acc2, mask=h_mask[:, None])
        pp = pid_t * stride_pt + offs_h * stride_ph + pid_s
        tl.store(MP + pp, m_i, mask=h_mask)
        tl.store(LP + pp, l_i, mask=h_mask)


@triton.jit
def _dattn_combine(ACC, MP, LP, SINK, OUT, H, D,
                   stride_at, stride_ah, stride_as, stride_ot, stride_oh,
                   stride_pt, stride_ph,
                   SPLIT: tl.constexpr, BD: tl.constexpr):
    pid = tl.program_id(0)
    t = pid // H
    h = pid % H
    si = tl.arange(0, SPLIT)
    d = tl.arange(0, BD)
    dm = d < D
    pp = t * stride_pt + h * stride_ph + si
    m_s = tl.load(MP + pp)
    l_s = tl.load(LP + pp)
    m = tl.maximum(tl.max(m_s), _NEG)
    wgt = tl.exp(m_s - m)
    sink = tl.load(SINK + h).to(tl.float32)
    denom = tl.sum(l_s * wgt) + tl.exp(sink - m)
    acc = tl.load(ACC + t * stride_at + h * stride_ah + si[:, None] * stride_as + d[None, :],
                  mask=dm[None, :], other=0.0)
    o = tl.sum(acc * wgt[:, None], 0) / denom
    tl.store(OUT + t * stride_ot + h * stride_oh + d, o.to(tl.bfloat16), mask=dm)


def _split_d(d: int):
    da = 1 << (d.bit_length() - 1)
    if da == d:
        return d, 0
    db = d - da
    assert db & (db - 1) == 0, f"head dim {d} is not a sum of two powers of two"
    return da, db


BLOCK_H = int(os.environ.get("DSV41_ATTN_BLOCK_H", 16))
BLOCK_N = int(os.environ.get("DSV41_ATTN_BLOCK_N", 32))
N_SPLIT = int(os.environ.get("DSV41_ATTN_SPLIT", 2))
PV_SPLIT = int(os.environ.get("DSV41_ATTN_PV_SPLIT", 1))
NUM_WARPS = int(os.environ.get("DSV41_ATTN_WARPS", 4))
NUM_STAGES = int(os.environ.get("DSV41_ATTN_STAGES", 2))


def decode_attention(q: torch.Tensor, kv1: torch.Tensor, kv2: torch.Tensor | None,
                     mask: torch.Tensor, sink: torch.Tensor, scale: float,
                     split: int | None = None, pv_split: int | None = None,
                     block_n: int | None = None) -> torch.Tensor:
    """q bf16 [T, H, D]; kv1 bf16 [T, N1, D] and optional kv2 bf16 [T, N2, D] (either may be a
    stride-0 broadcast along T); mask bool [T, N1+N2]; sink fp32 [H] -> o bf16 [T, H, D]."""
    T, H, D = q.shape
    N1 = kv1.shape[1]
    N2 = 0 if kv2 is None else kv2.shape[1]
    N = N1 + N2
    assert kv1.shape[2] == D and mask.shape == (T, N)
    assert q.stride(-1) == 1 and kv1.stride(-1) == 1 and mask.stride(-1) == 1
    assert q.dtype == torch.bfloat16 and kv1.dtype == torch.bfloat16
    if kv2 is not None:
        assert kv2.shape[2] == D and kv2.stride(-1) == 1 and kv2.dtype == torch.bfloat16
    DA, DB = _split_d(D)   # head_dim 512 is a power of two -> DB = 0 and the second block vanishes
    DBP = max(DB, 16)
    bn = BLOCK_N if block_n is None else block_n
    pv = PV_SPLIT if pv_split is None else pv_split
    sp = N_SPLIT if split is None else split
    sp = max(1, min(sp, triton.cdiv(N, bn)))
    if sp & (sp - 1):  # the combine kernel indexes the splits with a power-of-two arange
        sp = 1 << (sp.bit_length() - 1)
    msk = mask if mask.dtype == torch.int8 else mask.view(torch.int8)  # metadata-only, graph-safe
    k2 = kv1 if kv2 is None else kv2
    s2t, s2n = k2.stride(0), k2.stride(1)
    o = torch.empty(T, H, D, dtype=torch.bfloat16, device=q.device)
    grid = (triton.cdiv(H, BLOCK_H), T, sp)
    args = (q, kv1, k2, msk, sink)
    common = dict(DA=DA, DB=DB, DBP=DBP, BLOCK_H=BLOCK_H, BLOCK_N=bn, PV_SPLIT=pv,
                  TWO=1 if kv2 is not None else 0, num_warps=NUM_WARPS, num_stages=NUM_STAGES)
    if sp == 1:
        _dattn_kernel[grid](*args, o, o, o, T, H, N, N1,
                            q.stride(0), q.stride(1), kv1.stride(0), kv1.stride(1), s2t, s2n,
                            msk.stride(0), o.stride(0), o.stride(1), 0, 0, 0, scale,
                            SPLIT=1, FINAL=1, **common)
        return o
    acc = torch.empty(T, H, sp, D, dtype=torch.float32, device=q.device)
    mp = torch.empty(T, H, sp, dtype=torch.float32, device=q.device)
    lp = torch.empty(T, H, sp, dtype=torch.float32, device=q.device)
    _dattn_kernel[grid](*args, acc, mp, lp, T, H, N, N1,
                        q.stride(0), q.stride(1), kv1.stride(0), kv1.stride(1), s2t, s2n,
                        msk.stride(0), acc.stride(0), acc.stride(1), acc.stride(2),
                        mp.stride(0), mp.stride(1), scale, SPLIT=sp, FINAL=0, **common)
    BD = 1 << (D - 1).bit_length()
    _dattn_combine[(T * H,)](acc, mp, lp, sink, o, H, D,
                             acc.stride(0), acc.stride(1), acc.stride(2), o.stride(0), o.stride(1),
                             mp.stride(0), mp.stride(1),
                             SPLIT=sp, BD=BD, num_warps=4, num_stages=1)
    return o


def decode_attention_ref(q, kv, mask, sink, scale):
    """The torch path this replaces (engine/fastdecode.py `_attention`), for tests."""
    scores = torch.einsum("thd,tnd->thn", q.float(), kv.float()) * scale
    scores = scores.masked_fill(~mask[:, None, :], float("-inf"))
    mx = scores.amax(dim=-1, keepdim=True).clamp_min(NEG)
    p = torch.exp(scores - mx)
    denom = p.sum(-1, keepdim=True) + torch.exp(sink[None, :, None] - mx)
    return torch.einsum("thn,tnd->thd", p / denom, kv.float()).to(torch.bfloat16)
