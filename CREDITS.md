# Credits

This recipe is a thin layer over other people's work plus a lot of arithmetic. What is ours
is the single-box design — the expert arena with its LRU and transient ring, the `O_DIRECT`
streaming path, the Triton FP4 grouped-MoE kernel, the chunk-invariant port, the stdlib
server, the routing trace and every measurement. Everything below is somebody else's.

## The model

**[DeepSeek](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)** — DeepSeek-V4.1-Flash:
the architecture (Engram n-gram memory, hyper-connections, CSA2/CED sparse attention, the
FP4 MoE, the DSpark drafter), the weights, and the reference implementation under
`inference/` that this engine's math is ported from — `model.py`, `engram.py`, `kernel.py`
and the chat encoder in `encoding/`, which the server calls directly rather than
reimplementing. The tech report is where the numbers this repo quotes for global KV size
and Decoder SWA Bounded Replay come from. The weights carry DeepSeek's licence; read it
before deploying commercially.

Also DeepSeek's, and read while designing this: **`deepseek-recipe`** (the protocol and
chat-template layer, whose `reasoning_effort` string mapping the server's own mapping
follows), **`DeepSelect`** (the indexer top-k kernel — `sm_100a`/`sm_103a` only, which is
part of why this box needs its own path) and **`DeepJIT`**.

## The prior art this design is built on

**[tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark](https://github.com/tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark)**
— the four-Spark TP4 vLLM build, and the most useful single source for this repo. The
**engram-on-disk** patch (`DSV41_ENGRAM_DISK=1`: table tensors skipped at load, rows read
with `preadv` by a thread pool before the forward so CUDA graphs still work, dequantised on
CPU into a pinned staging buffer) is the idea `engine/engram.py` follows on one box, and
their measurement of it — 24 serial `pread`s ≈ 17 ms/step against 3.1 ms parallel — is why
it is a thread pool here too. Their SM12x notes are the map of this hardware's sharp edges:
block size 64/128 for the sparse SWA and indexer caches, and `top_k_per_row_decode` instead
of `persistent_topk`, which wants 128 KB of shared memory per block where GB10 has 99 KB.
Their honest "TP2 does not fit either way" is what made a streaming design the only option
left for a single box.

**[ssd-moe / deepseek-v4-flash-mlx](https://github.com/ssd-moe)** — the observation that a
bounded expert cache captures most expert accesses on the previous model ("a 32 GB cache
captures 80%+"). This repo's coverage curve is the same question asked with a measured
routing trace instead of a rule of thumb, and the answer on V4.1 turned out to be less
generous.

**vLLM's PLE mmap work** — the prior art for treating a per-layer table as something you
map and read rather than something you load, which is the shape both the Engram reader and
the expert store take here.

**[0xSero/deepseek-v4.1-flash-4x-rtx-pro-6000](https://github.com/0xSero)** — a bounded
DDR5 engram row cache with exact NVMe reads on misses, on very different hardware; a useful
second data point for the row-cache idea.

## The platform and the tools

**NVIDIA** — the DGX Spark / GB10, the CUDA 13 container images, and PyTorch's cu130
aarch64 wheels, without which none of this runs on arm64.

**[OpenAI Triton](https://github.com/triton-lang/triton)** — the kernel language the FP4
grouped-MoE forward is written in, and the JIT that compiles it for `sm_121a` on the box.

**[vLLM](https://github.com/vllm-project/vllm)** — `moe_align_block_size`, whose idea of
padding each expert's run of `(token, k)` pairs to a block multiple is what the routing
kernel here does.

## The house templates

**[ling3-flash-spark](https://github.com/0xBakeer/ling3-flash-spark)** — the recipe shape
this one copies: `start.sh`/`stop.sh` with memory guards, the benchmark harness and its
four rules (usage-based token counts, fresh verified prompts, label-salted seeds,
`ignore_eos`), and the `results/` layout. `bench/bench.py` here is an adaptation of that
harness, so rows from the two are directly comparable.

**[deepseek-v4-flash-spark](https://github.com/0xBakeer/deepseek-v4-flash-spark)** — the
container and release shape: the `run.sh` dispatcher, the GHCR workflow on `v*` tags, the
"a version is a measurement epoch" changelog, and the entrypoint/download split.

Assembled, measured and documented by 0xBakeer.
