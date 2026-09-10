# LIMITATIONS

Honest list of what this repo does not (yet) do, with the exact reason. Updated per phase.

## Status: Phase 0 only (2026-09-10)

* **No serving recipe exists yet.** Nothing in this repo runs DeepSeek-V4.1-Flash as a server. Phase 0
  produced the architecture notes, the expert-routing tracer and a partial (4 of 40 layers) routing
  histogram. See NOTES.md.
* **The routing histogram covers layers 0-3 only.** The other 36 layer shards (266 GB) were not
  downloaded because of the 50 GB ask-first rule. Coverage numbers in `results/trace-partial/` must
  not be extrapolated to the decoder layers.
* **The tracer is exact only for sequences of <= 512 tokens** (the indexer never prunes below that
  length). Long-context routing (which experts fire at 100k tokens of context) is not measured.
* **The tracer's numerics are not the reference kernels'**: bf16 GEMMs with fp32 accumulation and
  fake-quantized fp8 activations instead of fp8 x fp4 tensor-core GEMMs. Routing decisions are
  argmax-of-6 over 384 and are robust to this, but the teacher-forced accuracy check that would
  prove the port end-to-end needs all 40 layers plus the head and has not run yet.
* **Corpus size is small** (10,760 tokens, 50 sequences, two categories). Enough for a coverage
  shape, not for per-expert frequencies in the tail.
* **Engram tables are not on the box.** The trace fetched only the ~157k rows it needed per engram
  layer over HTTP. Any serving recipe needs the two 101 GB shards on local NVMe.

## Known blockers for a single-box recipe (from the size arithmetic, NOTES.md 0.8)

* 288.8 GB of FP4 routed experts against ~85-90 GB of expert budget = ~1.3 bits per weight average
  if everything must be resident. **No all-resident scheme meets the Q4-class quality floor.** The
  only quality-preserving path is a resident hot set plus NVMe streaming for the rest, and whether
  that reaches 10 tok/s depends on the full-model miss rate (unmeasured) and the NVMe random-read
  throughput for 18.8 MB objects (unmeasured).
* No engine has a single-GPU expert-streaming path for `deepseek_v41` today. vLLM (`dsv41-feat`) and
  SGLang both assume all experts resident across TP ranks; the tonyd2wild 4x Spark build is the
  closest working code (engram-on-disk + SM12x fixes) but states "TP2 does not fit either way".
