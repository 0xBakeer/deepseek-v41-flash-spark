"""dense_inventory.py -- which dense (non-routed-expert) weights the decode step reads, how many
bytes of them, and what FP4 would save. Reads only the safetensors headers; nothing is loaded.

The verify step graph (engine/fastdecode.py `step`) runs, once per step:
  * 40 backbone layers: attn.wq_a, attn.wq_b, attn.wkv, attn.wo_b (the `_fp8_linear_kernel` group),
    attn.wo_a (the grouped fp8 kernel) and ffn.shared_experts.w1/w2/w3;
  * one indexer wq_b on each of the 8 index_source_layers;
  * one engram wkv on each of the 2 engram layers;
  * `_final`: mtp.0.main_proj once and mtp.k.attn.wkv for k = 0,1,2.
That is 294 `_fp8_linear_kernel` calls + 40 `_fp8_grouped_kernel` calls per step, which is what the
profile shows (882 / 3 iterations).

The draft graph (`_draft`, a separate replay) runs the three DSpark blocks whole.

    python tools/dense_inventory.py
"""
import json
import os
import struct
import sys

MD = os.path.expanduser(os.environ.get("MODEL_DIR", "~/models/DeepSeek-V4.1-Flash"))
CFG = json.load(open(f"{MD}/config.json"))["text_config"]
NL = CFG["num_hidden_layers"]
INDEX_LAYERS = CFG["index_source_layer_ids"]
ENGRAM_LAYERS = CFG["engram_layer_ids"]
NMTP = CFG["num_nextn_predict_layers"]

_hdr = {}


def header(f):
    if f not in _hdr:
        with open(os.path.join(MD, f), "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            _hdr[f] = json.loads(fh.read(n))
    return _hdr[f]


def main():
    wm = json.load(open(f"{MD}/model.safetensors.index.json"))["weight_map"]

    def nbytes(name):
        h = header(wm[name])[name]
        return h["data_offsets"][1] - h["data_offsets"][0], h["shape"]

    def pair(p):  # weight + its scale table, as stored
        wb, shape = nbytes(p + ".weight")
        sb, _ = nbytes(p + ".scale")
        return wb, sb, shape

    # (group, prefix, per-step multiplicity in the verify step, per-step multiplicity in the draft)
    ATTN = ("attn.wq_a", "attn.wq_b", "attn.wkv", "attn.wo_b")
    SHARED = ("ffn.shared_experts.w1", "ffn.shared_experts.w2", "ffn.shared_experts.w3")
    items = []  # (group, name, weight bytes, scale bytes, step count, draft count, resident count)
    for L in range(NL):
        for s in ATTN:
            items.append(("attention", f"layers.{L}.{s}", *pair(f"layers.{L}.{s}")[:2], 1, 0, 1))
        for s in SHARED:
            items.append(("shared experts", f"layers.{L}.{s}", *pair(f"layers.{L}.{s}")[:2], 1, 0, 1))
        items.append(("wo_a (grouped)", f"layers.{L}.attn.wo_a", *pair(f"layers.{L}.attn.wo_a")[:2], 1, 0, 1))
    for L in INDEX_LAYERS:
        items.append(("other", f"layers.{L}.attn.indexer.wq_b", *pair(f"layers.{L}.attn.indexer.wq_b")[:2], 1, 0, 1))
    for L in ENGRAM_LAYERS:
        items.append(("other", f"layers.{L}.engram.wkv", *pair(f"layers.{L}.engram.wkv")[:2], 1, 0, 1))
    for k in range(NMTP):
        for s in ATTN:
            items.append(("attention", f"mtp.{k}.{s}", *pair(f"mtp.{k}.{s}")[:2], 1 if s == "attn.wkv" else 0, 1, 1))
        for s in SHARED:
            items.append(("shared experts", f"mtp.{k}.{s}", *pair(f"mtp.{k}.{s}")[:2], 0, 1, 1))
        items.append(("wo_a (grouped)", f"mtp.{k}.attn.wo_a", *pair(f"mtp.{k}.attn.wo_a")[:2], 0, 1, 1))
    items.append(("other", "mtp.0.main_proj", *pair("mtp.0.main_proj")[:2], 1, 0, 1))

    groups = ("attention", "shared experts", "wo_a (grouped)", "other")
    MB = 1e6
    print(f"model: {MD}")
    print(f"{NL} backbone layers, index sources {INDEX_LAYERS}, engram layers {ENGRAM_LAYERS}, "
          f"{NMTP} DSpark blocks\n")
    print("Bytes read per VERIFY STEP (one 6-token graphed step), by group")
    print(f"{'group':18s} {'calls':>6s} {'weight MB':>11s} {'scale MB':>9s} {'total MB':>9s} "
          f"{'fp4 MB':>8s} {'saved MB':>9s}")
    tot = [0, 0, 0, 0.0]
    for g in groups:
        n = sum(it[4] for it in items if it[0] == g)
        w = sum(it[2] * it[4] for it in items if it[0] == g)
        s = sum(it[3] * it[4] for it in items if it[0] == g)
        # fp4: codes = weight/2, scales = weight/16 (one uint8 per 32 weights vs 1 fp8 byte each)
        f4 = w / 2 + w / 32
        print(f"{g:18s} {n:6d} {w/MB:11.1f} {s/MB:9.2f} {(w+s)/MB:9.1f} {f4/MB:8.1f} {(w+s-f4)/MB:9.1f}")
        tot[0] += n; tot[1] += w; tot[2] += s; tot[3] += f4
    print(f"{'TOTAL':18s} {tot[0]:6d} {tot[1]/MB:11.1f} {tot[2]/MB:9.2f} {(tot[1]+tot[2])/MB:9.1f} "
          f"{tot[3]/MB:8.1f} {(tot[1]+tot[2]-tot[3])/MB:9.1f}")

    print("\nPer-weight detail (one instance of each distinct shape)")
    seen = set()
    for g, name, w, s, *_ in items:
        key = (g, name.split(".", 2)[-1])
        if key in seen:
            continue
        seen.add(key)
        sh = header(wm[name + ".weight"])[name + ".weight"]["shape"]
        print(f"  {g:16s} {key[1]:26s} {str(sh):18s} {(w+s)/MB:8.2f} MB -> {(w/2+w/32)/MB:7.2f} MB")

    print("\nBytes read per DRAFT graph (the three DSpark blocks, one replay)")
    for g in groups:
        n = sum(it[5] for it in items if it[0] == g)
        w = sum(it[2] * it[5] for it in items if it[0] == g)
        s = sum(it[3] * it[5] for it in items if it[0] == g)
        if n:
            print(f"  {g:18s} {n:4d} calls {(w+s)/MB:9.1f} MB -> fp4 {(w/2+w/32)/MB:7.1f} MB")

    print("\nRESIDENT bytes (every instance held on the GPU), by group")
    for g in groups:
        w = sum(it[2] * it[6] for it in items if it[0] == g)
        s = sum(it[3] * it[6] for it in items if it[0] == g)
        f4 = w / 2 + w / 32
        print(f"  {g:18s} {(w+s)/2**30:7.3f} GiB -> fp4 {f4/2**30:7.3f} GiB  (frees {(w+s-f4)/2**30:6.3f} GiB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
