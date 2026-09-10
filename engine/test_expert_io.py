"""
test_expert_io.py -- the O_DIRECT expert reader must return exactly the checkpoint's bytes.

`ExpertStore` reads an expert as two contiguous file runs cut into `DSV41_READ_CHUNK_MB` aligned
pieces that several threads issue in parallel (the arithmetic around alignment, the run tail that
may reach past EOF, and the offsets of the six tensors inside the staging buffer are all easy to
get subtly wrong, and a wrong expert shows up only as slightly worse text). This compares the
reader's bytes against `safetensors.safe_open` for a few experts at several chunk sizes.

Run on the box (it needs the real checkpoint; ~250 MB of pinned buffers, safe next to the server):

    python engine/test_expert_io.py --model-dir ~/models/DeepSeek-V4.1-Flash
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch
from safetensors import safe_open

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from engine import experts as EX  # noqa: E402


class _StubArena:
    """`read_expert` never touches the arena; only `.slots` is read at construction."""

    slots = 1024


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=os.environ.get("MODEL_DIR") or "./models/DeepSeek-V4.1-Flash")
    ap.add_argument("--chunks-mb", default="0,1,4,20", help="DSV41_READ_CHUNK_MB values to test (0 = no split)")
    a = ap.parse_args()
    index = json.load(open(os.path.join(a.model_dir, "model.safetensors.index.json")))
    cases = [(0, 0), (0, 383), (7, 123), (39, 1)]
    bad = 0
    for chunk in [float(c) for c in a.chunks_mb.split(",")]:
        store = EX.ExpertStore(a.model_dir, index, _StubArena(), 40, read_chunk_mb=chunk)
        t0 = time.perf_counter()
        for layer, e in cases:
            got = store.read_expert(layer, e)
            p = f"layers.{layer}.ffn.experts.{e}."
            f = safe_open(os.path.join(a.model_dir, index["weight_map"][p + "w1.weight"]), "pt", device="cpu")
            for i, name in enumerate(EX.NAMES):
                want = f.get_tensor(p + name).view(torch.uint8).reshape(-1)
                if not torch.equal(got[i].reshape(-1), want):
                    print(f"MISMATCH chunk={chunk} layer={layer} expert={e} {name}")
                    bad += 1
        dt = time.perf_counter() - t0
        gb = store.stats["bytes_read"] / 1e9
        print(f"chunk={chunk:>5} MB  {len(cases)} experts byte-exact  "
              f"({gb:.2f} GB in {dt:.2f}s = {gb / max(dt, 1e-9):.2f} GB/s, one expert at a time)")
        store.pool.shutdown(wait=True)
        store.read_pool.shutdown(wait=True)
    print("FAILED" if bad else "ok")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
