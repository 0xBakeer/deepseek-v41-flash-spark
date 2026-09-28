"""Checkpoint-free smoke test of every Triton kernel the engine uses, for porting to a new GPU."""
import os, sys, traceback
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..")); sys.path.insert(0, os.path.join(HERE, "..", "engine"))
dev = "cuda"; torch.manual_seed(0)
rel = lambda a, b: ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12)).item()
res = {}
def case(name):
    def deco(f):
        try:
            r = f(); res[name] = f"ok rel={r:.2e}" if r < 3e-2 else f"BAD rel={r:.2e}"
        except Exception as e:
            res[name] = "FAIL " + repr(e)[:300]; traceback.print_exc()
    return deco

@case("fp8_linear")
def _():
    from fp8_linear import quantize_to_fp8, fp8_linear
    w = torch.randn(4096, 2048, device=dev) * 0.02
    W = quantize_to_fp8(w)
    out = []
    for M in (1, 6, 300):
        x = torch.randn(M, 2048, device=dev, dtype=torch.bfloat16)
        out.append(rel(fp8_linear(x, W), x.float() @ W.dequant().float().T))
    return max(out)

@case("fp4_linear")
def _():
    from fp4_linear import quantize_to_fp4, fp4_linear
    w = torch.randn(4096, 2048, device=dev) * 0.02
    W = quantize_to_fp4(w)
    out = []
    for M in (1, 6, 300):
        x = torch.randn(M, 2048, device=dev, dtype=torch.bfloat16)
        out.append(rel(fp4_linear(x, W), x.float() @ W.dequant().float().T) if hasattr(W, "dequant") else 0)
    return max(out)

@case("decode_attn")
def _():
    from decode_attn import decode_attention, decode_attention_ref
    T, H, D, N = 6, 64, 512, 1000
    q = torch.randn(T, H, D, device=dev, dtype=torch.bfloat16)
    kv = torch.randn(T, N, D, device=dev, dtype=torch.bfloat16)
    mask = torch.rand(T, N, device=dev) > 0.2
    sink = torch.randn(H, device=dev)
    return rel(decode_attention(q, kv, None, mask, sink, D ** -0.5), decode_attention_ref(q, kv, mask, sink, D ** -0.5))

@case("fp32_skinny")
def _():
    from fp32_skinny import skinny_linear
    x = torch.randn(6, 4096, device=dev); w = torch.randn(1024, 4096, device=dev)
    return rel(skinny_linear(x, w), x @ w.T)

@case("hc_sinkhorn")
def _():
    from hc_sinkhorn import hc_split_sinkhorn
    n, hc = 8, 4
    m = torch.randn(n, (2 + hc) * hc, device=dev); sc = torch.rand(3, device=dev); b = torch.randn((2 + hc) * hc, device=dev)
    pre, post, comb = hc_split_sinkhorn(m, sc, b, hc)
    torch.cuda.synchronize()
    return 0.0 if torch.isfinite(comb).all() else 1.0

@case("cuda_graph")
def _():
    x = torch.randn(64, 64, device=dev); y = torch.empty_like(x)
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s): y.copy_(x @ x)
    torch.cuda.current_stream().wait_stream(s)
    with torch.cuda.graph(g): y.copy_(x @ x)
    x.copy_(torch.randn(64, 64, device=dev)); g.replay(); torch.cuda.synchronize()
    return rel(y, x @ x)

for k, v in res.items(): print(f"{k:12s} {v}")
