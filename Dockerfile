# DeepSeek-V4.1-Flash on one DGX Spark (GB10 / sm_121a) -- serving image.
#
# There is nothing to compile here. The engine is plain PyTorch plus one Triton kernel
# (tools/fp4_moe.py) that is JIT-compiled on the first MoE call, so the image is a Python
# environment and a copy of this repo -- no C++ extension build, no engine fork, minutes
# rather than the twenty of the V4 recipe.
#
# The weights are NOT in the image and never will be: 510 GB. Mount them at /models
# (scripts/download-model.sh fetches them) and give the container the whole box.
#
# Build it on an arm64 machine -- normally GitHub's ubuntu-24.04-arm runner via
# .github/workflows/image.yml, or the Spark itself:
#
#   docker build -t deepseek-v41-flash-spark .
#
# ---------------------------------------------------------------------------------------
# Why the *devel* base and not -runtime
# ---------------------------------------------------------------------------------------
# Triton compiles tools/fp4_moe.py at run time, which means ptxas has to run inside this
# container, and it has to be a ptxas that knows `sm_121a` and the FP4 decode instruction
# the kernel emits as inline PTX (`cvt.rn.f16x2.e2m1x2`). Triton wheels do bundle a ptxas
# under triton/backends/nvidia/bin/, but that bundled copy has shipped behind the driver
# before -- the CUDA-12.8 ptxas in the triton 3.5 wheels could not name sm_121 at all
# (pytorch/pytorch#163801) -- and a recipe that fails at the first token because of a
# vendored compiler is not worth the ~2 GB the devel layer costs. So: the CUDA 13 devel
# image, and TRITON_PTXAS_PATH pinned at its ptxas. If you would rather trust the wheel,
# swap the base for nvidia/cuda:13.0.2-runtime-ubuntu24.04 and drop TRITON_PTXAS_PATH;
# everything else in this file is unchanged. Nothing else here needs nvcc.
FROM nvidia/cuda:13.0.2-devel-ubuntu24.04

ARG TORCH_VERSION=2.13.0
ARG RECIPE_VERSION=0.1.0-wip

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    CUDA_HOME=/usr/local/cuda \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:/usr/local/cuda/bin:$PATH

# python3.12 is Ubuntu 24.04's system interpreter and has a torch cu130 aarch64 wheel
# (cp312); curl is the healthcheck, procps is for looking at the box from inside.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv python3-dev ca-certificates curl procps \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv \
    && pip install --upgrade pip wheel setuptools

# PyTorch for CUDA 13.0. The cu130 index is the only place with an aarch64 build, and it
# is where `triton` comes from too -- torch pulls it as a dependency, so the Triton the
# kernel JITs against is always the one this torch was tested with. Verified present:
# torch-2.13.0+cu130-cp312-cp312-manylinux_2_28_aarch64.whl.
RUN pip install "torch==${TORCH_VERSION}+cu130" --index-url https://download.pytorch.org/whl/cu130

# Everything else from PyPI. safetensors reads the checkpoint headers (the engine parses
# the shard headers itself and then reads spans with O_DIRECT); tokenizers/transformers
# load tokenizer.json; sympy and numpy are used by the reference math in tools/v41_ref.py;
# huggingface_hub is only for scripts/download-model.sh.
RUN pip install \
        "transformers>=4.57" \
        "tokenizers>=0.21" \
        "safetensors>=0.5" \
        "numpy>=1.26" \
        "sympy>=1.13" \
        "huggingface_hub>=0.35"

WORKDIR /app
COPY . /app
RUN chmod +x /app/scripts/*.sh /app/run.sh 2>/dev/null || true \
    && echo "${RECIPE_VERSION}" > /app/VERSION \
    && mkdir -p /models /app/logs /app/results /app/.triton

# Build-time sanity: the imports the engine does at start-up must work, and the Triton
# kernel module must at least parse. It cannot be *run* here -- there is no GPU on a
# build runner -- so the first real compile still happens on the box.
RUN python -c "import torch, triton, transformers, tokenizers, safetensors, numpy, sympy; \
print('torch', torch.__version__, 'triton', triton.__version__)" \
    && python -c "import sys; sys.path.insert(0, '/app/tools'); import fp4_moe; print('fp4_moe ok')"

LABEL org.opencontainers.image.title="DeepSeek-V4.1-Flash on one DGX Spark" \
      org.opencontainers.image.description="Plain-PyTorch engine with a Triton FP4 MoE kernel and NVMe expert streaming" \
      org.opencontainers.image.version="${RECIPE_VERSION}" \
      org.opencontainers.image.licenses="MIT"

ENV MODEL_DIR=/models/DeepSeek-V4.1-Flash \
    MODEL_REPO=deepseek-ai/DeepSeek-V4.1-Flash \
    SERVED_MODEL_NAME=deepseek-v4.1-flash \
    HOST=0.0.0.0 \
    PORT=8000 \
    MAX_SEQ=32768 \
    DEFAULT_THINKING=off \
    DEFAULT_EFFORT=75 \
    SPEC=1 \
    TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas \
    TRITON_CACHE_DIR=/app/.triton \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    TOKENIZERS_PARALLELISM=false \
    HF_HOME=/models/.hf \
    PYTHONUNBUFFERED=1

VOLUME ["/models"]
EXPOSE 8000

# start-period is 45 minutes because it has to be: the socket is bound only after the warm
# start has pulled tens of GB of FP4 experts off NVMe. A container that is not answering at
# minute 10 is normal, not sick.
HEALTHCHECK --interval=30s --timeout=5s --start-period=45m --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/health || exit 1

ENTRYPOINT ["/app/scripts/entrypoint.sh"]
