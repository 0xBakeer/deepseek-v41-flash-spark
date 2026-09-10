# Install

Target: one DGX Spark class box — NVIDIA GB10, `sm_121a`, 128 GB unified memory
(≈121 GiB visible), 20 cores, local NVMe — on DGX OS / Ubuntu 24.04 with a CUDA 13
driver. Nothing here is multi-GPU and nothing here is x86.

## Host prerequisites

| what | why |
|---|---|
| GB10 / `sm_121a`, CUDA 13 driver (580.x or newer) | the Triton MoE kernel emits `cvt.rn.f16x2.e2m1x2` and is compiled for `sm_121a`; nothing older names that target |
| ≥ 600 GB free on **local NVMe** | the checkpoint is 510 GB and the engine reads experts out of it with `O_DIRECT` on every miss. A network filesystem turns each miss into a network round trip; overlayfs will not do `O_DIRECT` at all |
| the box essentially to itself | GPU and host draw on one pool. The resident expert arena is sized from what is free at load time, so a neighbour does not cause an OOM — it causes a small arena and a permanently NVMe-bound server |
| Python 3.11 or 3.12 (native path) or docker + nvidia-container-toolkit (container path) | both are described below |

Check the two that are easy to get wrong:

```bash
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv
df -h /path/to/where/the/weights/will/live
awk '/MemAvailable/ {printf "%.1f GiB available\n", $2/1048576}' /proc/meminfo
```

## The checkpoint

510 GB in 48 safetensors shards: 288.8 GB of FP4 routed experts (one 7.39 GB shard per
layer), 203 GB of Engram n-gram tables (shards 47 and 48, 101.5 GB each), ~19 GB of
everything else. Serving needs all of it — the Engram tables are read row by row at run
time, so unlike the routing tracer the server cannot skip them.

```bash
# container path
./run.sh setup                     # pulls the image, then downloads into ./models

# native path
MODEL_DIR=./models/DeepSeek-V4.1-Flash ./scripts/download-model.sh
```

Both call [`scripts/download-model.sh`](../scripts/download-model.sh), which is a
`snapshot_download` of `deepseek-ai/DeepSeek-V4.1-Flash`: resumable, safe to re-run,
skips complete shards. `HF_TOKEN` is optional (the repo is public; a token only raises
rate limits). Expect hours, and re-run it until it says `checkpoint complete`.

## Two ways to run it

They are the same server. The container exists so the box does not have to host a
torch/CUDA 13 environment; the native path exists because on a machine with one memory
pool, one fewer layer between the engine and the NVMe is worth something.

### Native (venv)

There is no `setup.sh` here: the engine is plain PyTorch, so any interpreter with the
right wheels will run it. On aarch64 that means the cu130 index:

```bash
python3.12 -m venv .venv && . .venv/bin/activate
pip install torch==2.13.0+cu130 --index-url https://download.pytorch.org/whl/cu130
pip install "transformers>=4.57" "tokenizers>=0.21" "safetensors>=0.5" numpy sympy
```

`triton` arrives as a dependency of `torch` and must not be replaced with a PyPI build —
the kernel in `tools/fp4_moe.py` is JIT-compiled against whichever Triton is installed,
and the one on the cu130 index is the one this torch was tested with.

```bash
cp env.example .env      # then set MODEL_DIR and PYTHON
./start.sh               # nohup server/app.py --engine v41; waits for /health
./stop.sh
```

`PYTHON` in [`env.example`](../env.example) points at that interpreter. `start.sh`
refuses to start if the port is taken or if less than `MIN_FREE_GIB` (90) is available,
and names the processes holding the pool.

### Container

```bash
cp env.example .env      # PORT, MAX_SEQ, ARENA_GB, SPEC, thinking defaults
./run.sh setup           # pull + download
./run.sh serve           # detached, waits for /health
./run.sh logs
./run.sh stop
```

[`compose.yaml`](../compose.yaml) publishes the API on `127.0.0.1:8000` only, mounts
`./models` at `/models` and `./results` at `/app/results`, and reads `./.env`. Note that
`MODEL_DIR` in `.env` is a **host** path: inside the container the checkpoint is always
under the `/models` mount, and `./run.sh` derives the bind mount and the directory name
from it (override with `MODELS_DIR` / `MODEL_NAME` if you want to be explicit).

The equivalent by hand, if you would rather not use compose:

```bash
docker run -d --name deepseek-v41-flash \
  --gpus all --ipc=host --ulimit memlock=-1 \
  -p 127.0.0.1:8000:8000 \
  -v /path/to/models:/models -v "$PWD/results:/app/results" \
  -e MAX_SEQ=32768 -e SPEC=1 \
  ghcr.io/<owner>/deepseek-v41-flash-spark:<version>
```

Three flags are not optional and one is a trap:

* `--gpus all` — the container needs the GB10 and the driver's `libcuda`.
* `--ipc=host` — the engine keeps pinned staging buffers for the `O_DIRECT` reads; the
  default 64 MB `/dev/shm` and the default IPC namespace get in the way of that.
* `--ulimit memlock=-1` — pinned memory is what makes the NVMe path fast; without it the
  staging buffers fall back to pageable memory.
* **Do not set `--memory`.** On unified memory that is a limit on GPU allocations too,
  and the arena auto-sizer will quietly shrink to fit it. There is no cgroup number that
  means "host RAM but not GPU" on this hardware.

The image is built by [GitHub Actions on an arm64 runner](../.github/workflows/image.yml)
and published to GHCR on `v*` tags. Building it on the box works (`BUILD=1 ./run.sh
setup`) and takes minutes, not the twenty of an engine compile — there is no C++
extension in this recipe.

## First start

The socket is bound only *after* the warm start has filled the resident FP4 expert arena
from NVMe, ranked by the measured routing trace. That is tens of GB of reads: on the
measured run, 3,891 experts / 73.2 GB in 16 s at 4.6 GB/s once the non-expert weights
(~19 GB, ~63 s) were up. A server that is not answering at minute 5 is normal; the health
waits are 20 minutes native and 45 in the container for that reason.

```bash
curl -s localhost:8000/health | python3 -m json.tool
curl -s localhost:8000/v1/chat/completions -H 'content-type: application/json' \
  -d '{"messages":[{"role":"user","content":"What is 2+2?"}],"max_tokens":64}'
```

`/health` carries `engine_config` — arena GB, arena slots, resident expert percentage,
kernel, spec on/off — so you never have to remember what the server was started with.

Next: [architecture](architecture.md) · [OpenAI API](openai-api.md) ·
[benchmarking](benchmarking.md) · [gotchas](gotchas.md)
