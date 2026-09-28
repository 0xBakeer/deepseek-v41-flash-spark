# AMD Strix Halo (ROCm, gfx1151) — experimental

The engine also serves on an AMD Ryzen AI Max+ 395 / Radeon 8060S box. Nothing about the GB10
path changes: every change below is gated on `torch.version.hip` (or an explicit environment
switch), and the GB10 kernel tests give the same results and the same bandwidth before and after.

Measured 2026-09-28 on one box, from this checkout, with the numbers taken the same way as the
GB10 rows beside them. Quality was spot-checked by hand, not gated with `tools/gate_profile.py`.

## The box

| | GB10 (reference) | Strix Halo |
|---|---|---|
| GPU | NVIDIA GB10, `sm_121a` | Radeon 8060S, `gfx1151` (RDNA 3.5, wave32, 64 KB LDS) |
| memory the GPU can use | ~121 GiB unified with the host | 96 GiB carved out in the BIOS; ~31 GB left to the host |
| measured copy bandwidth | ~210 GB/s | ~210 GB/s |
| measured bf16 matmul | ~97 TFLOPS | ~33 TFLOPS |
| NVMe, `O_DIRECT` | 5.4–5.6 GB/s | 5.5 GB/s (1 stream), 6.7 GB/s (4 streams), PCIe 4.0 x4 |
| software | torch 2.13.0+cu130, triton 3.7.1 | torch 2.11.0+rocm10.0.0, triton 3.8.0 |

Decode is bandwidth-bound and the two boxes have the same bandwidth, so the same step can run at
a similar speed. Prefill is compute-bound, and there the 8060S has about a third of the GB10.

## Running it

There is no native ROCm venv recipe here; the engine runs inside the
[`kyuz0/vllm-therock-gfx1151`](https://hub.docker.com/r/kyuz0/vllm-therock-gfx1151) image
(tag `rocm10.0.0-torch2.11.0-vllm0.28.0`), which ships a torch and triton built for gfx1151.
Only torch, triton, transformers and safetensors are used from it.

```bash
docker run -d --init --name dsv41 --network host --ipc host --ulimit memlock=-1 \
  --device /dev/kfd --device /dev/dri --group-add video --security-opt seccomp=unconfined \
  -v "$PWD":/app -v /path/to/models:/models -w /app \
  kyuz0/vllm-therock-gfx1151:rocm10.0.0-torch2.11.0-vllm0.28.0 sleep infinity
docker exec dsv41 bash -c "cd /app && ./start.sh"
```

`--init` is not optional: with `sleep` as PID 1 a server that exits stays a zombie, `kill -0`
still succeeds on it, and `./stop.sh` reports it as surviving `SIGKILL`.

The `.env` the numbers below were taken with (the keep-set is `results/keepsets/topics` with the
topics listed; add or swap topics for your own traffic, see [`tune.md`](tune.md)):

```bash
MODEL_DIR=/models/DeepSeek-V4.1-Flash
PYTHON=python
EXPERT_FORMAT=fp4             # CB3 runs but is ~2x slower here, see below
PRUNE_KEEP=0.31
ARENA_GB=78                   # the arena is in device memory; the host is not involved
TRANSIENT_SLOTS=160           # the keep-set is larger than the arena, so misses stream
KEEP_FREE_GB=1
MIN_FREE_GIB=20               # start.sh checks host MemAvailable, and this host has ~31 GB
DSV41_PREFILL_CHUNK=1024      # halves the prefill transient, which buys arena
DSV41_MEM_FLOOR_GB=0.5        # the watchdog guards host RAM, which the arena does not use
DSV41_DENSE_FP4=              # dense layers stay FP8: the FP8 kernels are the fast ones here
DSV41_HEAD_FMT=fp8
DSV41_PRUNE_RANK=maxmin
DSV41_PRUNE_SOURCE=saliency
TRACE_STATS=results/keepsets/topics/coverage.json
EXPERT_TOPICS=academic,chinese,english,reasoning,reasoning_code,technical,translation
MAX_SEQ=32768
```

The engine notices that device memory is not the host's (`device total` far from `MemTotal`) and
sizes the arena from device free memory instead of `MemAvailable`; `DSV41_UNIFIED_MEM=0/1`
overrides the guess. Unlike the GB10, running the device out of memory is an ordinary HIP OOM,
not a hard reset, so the arena can take nearly all of it.

## What had to change

To make it run:

| problem on gfx1151 | change |
|---|---|
| the FP4 MoE kernel decodes with PTX `cvt.rn.f16x2.e2m1x2` | a portable decode in `tools/fp4_moe.py` (below) |
| the 3-stage FP8 pipeline needs 65 KB of LDS; RDNA has 64 KB | 2 stages on ROCm |
| the arena was sized from host `MemAvailable` (~30 GB here) | `device_memory_is_unified()` in `engine/v41_engine.py` |
| `safe_open` maps the whole 101.5 GB Engram shard; the host's RAM + swap is smaller, so heuristic overcommit refuses it (`ENOMEM`) | `open_safetensors()` in `engine/model.py` falls back to reading the few tensors it needs with `pread` |
| CB3 uses PTX `prmt`/`lop3` | the v3 entry point runs the existing v2 integer kernels on ROCm |

To make it fast. `rocprofv3` over five graphed verify steps, same configuration on both boxes
(FP4, keep 0.21 so everything is resident, dense FP8):

| kernel | gfx1151 before | gfx1151 after | GB10 |
|---|---|---|---|
| `_moe_up_kernel` | 215 ms/step | 54 ms/step | 51 ms/step |
| `_moe_down_kernel` | 45 | 25 | 24 |
| dense FP8 (all shapes) | 115 | ~45 | 37 |
| **wall clock per step** | **440 ms** | **179 ms** | **144 ms** |

Three causes, in order of size:

1. **Decode was ALU-bound.** gfx11 has no FP4 or FP8 conversion instruction. The portable FP4
   decode shifts the three magnitude bits of an E2M1 nibble to bit 9 of an fp16, which is exactly
   2^-14 times the E2M1 value for normals and the subnormal alike, and folds the 2^14 into the
   UE8M0 scale. FP8 does the same with e4m3 → bf16 (2^-120, applied as two exact 2^60 factors so
   no fp32 intermediate overflows) and one bf16 dot per 32-wide K group, scaled in fp32.
2. **Power-of-two row pitch.** FP8 weights with 4096- or 8192-byte rows put the rows a program
   walks in lockstep on the same memory channel: a [5120, 8192] weight streamed at 80 GB/s, and
   at 195–212 GB/s with 128 bytes of padding per row. `_pad_rows()` gives any K that is a multiple
   of 256 a pitch of K + 128 (~3 % memory).
3. **Tuning.** The MoE block tables were swept on gfx1151 (`tools/sweep_fp4_moe.py`).

One regression was found and fixed on the way: an fp16 variant of the FP8 kernel overflowed on
the DSpark drafter's un-normalised hidden state. Acceptance on a counting prompt fell from 5.9
to 1.0 while the target model's text looked fine. The kernel keeps activations in bf16.

## The numbers

Both boxes, same method: `bench/bench.py --runs 2 --warmup 1 --osl 256 --ignore-eos`, median;
prefill from one request each at 1,335 and 6,453 prompt tokens. Each box at its own working
point: GB10 CB3 keep 0.40 (everything resident), Strix Halo FP4 keep 0.31 (arena 78 GB).

| | GB10 | Strix Halo | ratio |
|---|---|---|---|
| decode, `code` | 20.4 tok/s (acceptance 2.95) | 12.2 tok/s (acceptance 2.65) | 0.60 |
| decode, `prose` | 13.8 tok/s (acceptance 1.99) | 10.3 tok/s (acceptance 2.20) ¹ | 0.75 |
| verify step (tpot × acceptance) | 144–145 ms | 214–225 ms | ~0.66 |
| TTFT, 45–62-token prompt | 1.4–1.6 s | 2.0–2.1 s | |
| prefill, 1,335 tokens | 134 tok/s | 105 tok/s | 0.78 |
| prefill, 6,453 tokens | 245 tok/s | 157 tok/s | 0.64 |
| expert hit rate | 1.0 | 0.997 | |
| NVMe read per request | 0 | 4–6 GB short, 14–20 GB for the long prompt | |

¹ The first of the two runs. The second repeated itself (acceptance 4.51), which inflates tok/s
without saying anything about the box; the step time of that run (225 ms) is still in line.

Compare step times, not tok/s: tok/s moves with how predictable the text is, the step does not.

## Why the keep is lower than on the GB10

| | GB10 | Strix Halo |
|---|---|---|
| expert format | CB3, 14.45 MB | FP4, 18.80 MB |
| expert arena | 90 GB | 78 GB |
| experts that fit | 6,226 | 4,148 (160 of them transient) |
| keep that is fully resident | 0.40 | 0.268 |

Most of the gap is the format. CB3 on gfx1151 is correct but decode-bound: the v2 kernels spend
~10 integer ops per weight where the PTX v3 path spends ~3, and at T=6 over 36 experts it runs
8.98 ms against 4.40 for FP4. End to end, CB3 keep 0.343 fully resident served 6.5–6.8 tok/s
against 8.2–10.3 for FP4 keep 0.31 (one 128-token run each). An RDNA byte-permute
(`v_perm_b32`) version of the v3 lookup might close that; it has not been written.

Measured alternatives at arena 78 GB (one 128-token run each, `code` / `prose`): FP4 keep 0.268
fully resident 11.8 / 10.3 tok/s; FP4 keep 0.31 10.3 / 8.2; FP4 keep 0.40 (54 % resident, arena
63 GB) 6.5 / 4.8. Keep 0.21 is faster still and degenerates.

## Tried, and not kept

- Smaller `O_DIRECT` read chunks for streamed misses (4 → 2 → 1 MB): no change. The disk does
  6.7 GB/s; a miss is slow because the step waits for it layer by layer.
- 128-byte-wide MoE tiles split in registers: 2x slower (RDNA does the split through LDS).
- Retiling w1/w3 in the arena, and padding their 2560-byte rows: 1–4 %, for 3.4 % of the arena.
- Splitting gate and up into separate programs: no change.

## Limitations

- Quality at keep 0.31 has not been gated; it is one step below the GB10's 0.40 on the pruning
  ladder in [`RESULTS.md`](../RESULTS.md) §2.4.
- The CB2 (2-bit) tier's kernel is still PTX-only; no shipped configuration uses it, and
  `tools/test_cb3_moe.py` reports its two CB2 checks as failures on ROCm.
- Dense FP4 (`DSV41_DENSE_FP4`) runs but is slower than FP8 on gfx1151.
- `./stop.sh` waits for host `MemAvailable` to reach 90 GiB before returning, which a 31 GB host
  never does: it gives up after three minutes. The server itself is gone long before that.
- Only the decode loop was profiled. Prefill, long contexts and long generations were not.

## Profiling on ROCm

The torch profiler mis-attributes kernel times inside HIP graph replays (the same step reported
78 and then 38 calls of `_moe_up_kernel`, 3.2 ms and then 0.5 ms each). Use rocprofv3:

```bash
rocprofv3 --kernel-trace --marker-trace --output-format csv -d /tmp/rp -- python engine/profile_rocprof.py
python tools/rocprof_summary.py /tmp/rp
```

For microbenchmarks, rotate through more than 256 MB of weights: the 8060S has a 32 MB MALL, and
a 32 MB weight run in a loop reports ~200 GB/s from cache that the engine never sees.

Checkpoint-free kernel checks: `tools/test_kernels_synth.py` (every Triton kernel against a torch
reference) and `tools/test_fp4_portable.py` (the portable FP4 decode). With the checkpoint,
`tools/test_fp4_moe.py`, `engine/test_kernels.py` and `tools/test_cb3_moe.py` pass on gfx1151
except for the two CB2 checks above.
