# deepseek-v41-flash-spark

DeepSeek-V4.1-Flash on a single NVIDIA DGX Spark class box (GB10, 128 GB unified memory). Work in
progress; the status of each phase is in NOTES.md and what does not work is in LIMITATIONS.md.

**Status (2026-09-10): Phase 0 done, no serving recipe yet.** The model is 510 GB on disk
(288.8 GB of FP4 routed experts, 203 GB of Engram n-gram tables, ~19 GB of everything else). The
question this repo answers is whether one box can serve it without dropping below Q4-class expert
quality, and if so how fast.

## What is here

| path | what |
|---|---|
| `NOTES.md` | running log: checkpoint layout, architecture facts that matter for serving (Engram lookup path, router, mHC, CED/CSA2 attention, DSpark), the public landscape, size arithmetic, measured trace results |
| `LIMITATIONS.md` | what is not done and why, with exact reasons |
| `tools/v41_ref.py` | pure-PyTorch port of the V4.1 text forward pass, exact for sequences <= 512 tokens, no tilelang needed |
| `tools/expert_trace.py` | layer-streaming router trace: runs a corpus through one 7.4 GB layer shard at a time, records the top-6 experts per token per layer, resumable as shards arrive |
| `tools/engram_rows.py` | fetches only the Engram rows a corpus needs (multipart HTTP range requests against the 101 GB shards; nothing else is downloaded) |
| `tools/make_corpus.py` | builds the teacher-forced trace corpus in the V4.1 chat format |
| `tools/expert_stats.py` | coverage curves, LRU hit-rate simulation per token and per DSpark block, memory projection |
| `results/trace-partial/` | layers 0-3 routing trace over 10,760 tokens + coverage tables/plots |
| `corpus/` | the trace corpus and its (public, MIT) sources |

## Phase 0 headline

Measured on this box, layers 0-3 of 40, 10,760 tokens (coding + general):

| layer | experts used / 384 | top-25% of experts cover | top-30% cover | top-50% cover |
|---|---|---|---|---|
| 0 | 381 | 59.4% | 65.4% | 84.2% |
| 1 | 380 | 61.4% | 67.2% | 84.6% |
| 2 | 370 | 69.4% | 75.2% | 90.2% |
| 3 | 374 | 76.1% | 80.8% | 93.0% |

30% is roughly the share of experts that fits in memory next to everything else. Coverage rises
with depth; the decoder layers (20-39) have not been traced yet.

Size arithmetic that needs no trace: keeping every expert resident would require ~1.3 bits per
weight on average, so an all-resident recipe cannot meet the quality floor. The only quality-preserving
single-box design is a resident hot set at FP4 plus NVMe streaming for the rest (NOTES.md 0.8).

## Reproduce the trace

```bash
# on the box, with a venv that has torch (CUDA), transformers, safetensors, numpy, sympy
python3 tools/make_corpus.py --tokenizer ~/models/DeepSeek-V4.1-Flash --code ... --prose ... --out corpus/trace_corpus.jsonl
python3 tools/engram_rows.py --model-dir ~/models/DeepSeek-V4.1-Flash --corpus corpus/trace_corpus.jsonl --out engram_rows
python3 tools/expert_trace.py --model-dir ~/models/DeepSeek-V4.1-Flash --corpus corpus/trace_corpus.jsonl \
    --engram-dir engram_rows --out results/trace-YYYYMMDD --layers 0-39 --resume
python3 tools/expert_stats.py --trace results/trace-YYYYMMDD --out results/trace-YYYYMMDD/stats
```

`--model-dir` needs the layer shards you want to trace (`model-0000{3..42}-of-00048.safetensors`),
`model-00002` (embed), the `inference/` folder and the tokenizer. The engram shards are not needed.

## License

MIT (this repo). DeepSeek-V4.1-Flash weights and reference code are MIT (deepseek-ai).
