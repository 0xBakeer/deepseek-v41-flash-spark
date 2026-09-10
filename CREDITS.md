# Credits

This recipe is a thin layer over other people's work plus a lot of arithmetic. What is ours
is the single-box design — the expert arena with its LRU and transient ring, the `O_DIRECT`
streaming path, the Triton FP4 grouped-MoE kernel, the chunk-invariant port, the stdlib
server, the routing trace and every measurement.

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

## Where the techniques come from

Both of the load-bearing ideas here were built and measured in earlier recipes of mine on
the same class of box, and this repo is the third iteration of them.

**[qwen38-flash-next-spark](https://github.com/0xBakeer/qwen38-flash-next-spark)** — the
**NVMe-resident table** pattern: keep a table that does not fit in unified memory on the SSD,
map it, and let the page cache serve the rows a step actually touches instead of loading the
whole thing. That is exactly the shape `engine/engram.py` takes for V4.1's 203 GB of Engram
n-gram tables — 48 random 264-byte rows per token, read with buffered `preadv` in a thread
pool under `POSIX_FADV_RANDOM`, with a small process-local row cache in front. It is also
where the idea generalises from: an expert is the same problem at 18.8 MB instead of 264
bytes, which is what `engine/experts.py` does with `O_DIRECT` and an explicit arena.

**[ling3-flash-spark](https://github.com/0xBakeer/ling3-flash-spark)** — the recipe shape
this one copies: `start.sh`/`stop.sh` with the memory and port guards, the `run.sh`
dispatcher and container split, the GHCR workflow on `v*` tags, the "a version is a
measurement epoch" changelog, the `results/` layout, and the benchmark harness with its four
rules (usage-based token counts, fresh verified prompts, label-salted seeds, `ignore_eos`).
`bench/bench.py` here is an adaptation of that harness, so rows from the two are directly
comparable.

## The platform and the tools

**NVIDIA** — the DGX Spark / GB10, the CUDA 13 container images, and PyTorch's cu130
aarch64 wheels, without which none of this runs on arm64.

**[OpenAI Triton](https://github.com/triton-lang/triton)** — the kernel language the FP4
grouped-MoE forward is written in, and the JIT that compiles it for `sm_121a` on the box.

Assembled, measured and documented by Khaled Bakeer (0xBakeer).
