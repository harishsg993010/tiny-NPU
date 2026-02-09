#!/usr/bin/env python3
"""Quantize FP32 GPT-2 weights to INT8 and pack into weights.bin for NPU demo."""
import argparse, os, sys, json
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from ddr_map import *


def quantize_and_pack(fp32_path, outdir):
    os.makedirs(outdir, exist_ok=True)
    d = np.load(fp32_path)

    q = {}   # quantized int8 tensors
    meta = {} # metadata (scales, etc.)

    # Embeddings: quantize to full int8 range (target_max=32 for better resolution)
    q["wte"] = quantize_tensor(d["wte"], target_max=32)
    q["wpe"] = quantize_tensor(d["wpe"], target_max=32)

    for i in range(N_LAYERS):
        # Weight matrices: quantize conservatively (target_max=4)
        for name in ["wq", "wk", "wv", "wo", "w1", "w2"]:
            key = f"block{i}.{name}"
            q[key] = quantize_tensor(d[key], target_max=QUANT_WEIGHT_MAX)

        # LN betas: direct round (small values near 0)
        q[f"block{i}.ln1.beta"] = quantize_beta(d[f"block{i}.ln1.beta"])
        q[f"block{i}.ln2.beta"] = quantize_beta(d[f"block{i}.ln2.beta"])

    q["ln_f.beta"] = quantize_beta(d["ln_f.beta"])

    # lm_head: same quant as regular weight
    q["lm_head"] = quantize_tensor(d["lm_head"], target_max=QUANT_WEIGHT_MAX)

    # ── Pack into weights.bin ──────────────────────────────────────────
    buf = bytearray(WEIGHTS_TOTAL)

    def put(offset, arr):
        flat = arr.flatten().astype(np.int8)
        buf[offset:offset + len(flat)] = flat.tobytes()

    put(WTE_OFFSET, q["wte"])       # [256][16]
    put(WPE_OFFSET, q["wpe"])       # [32][16]

    for i in range(N_LAYERS):
        base = BLOCKS_OFFSET + i * BLOCK_SIZE
        put(base + BLK_LN1_BETA, q[f"block{i}.ln1.beta"])
        put(base + BLK_WQ,       q[f"block{i}.wq"])
        put(base + BLK_WK,       q[f"block{i}.wk"])
        put(base + BLK_WV,       q[f"block{i}.wv"])
        put(base + BLK_WO,       q[f"block{i}.wo"])
        put(base + BLK_LN2_BETA, q[f"block{i}.ln2.beta"])
        put(base + BLK_W1,       q[f"block{i}.w1"])
        put(base + BLK_W2,       q[f"block{i}.w2"])

    put(LN_F_OFFSET,  q["ln_f.beta"])
    put(LM_HEAD_OFFSET, q["lm_head"])

    weights_path = os.path.join(outdir, "weights.bin")
    with open(weights_path, "wb") as f:
        f.write(buf)
    print(f"Packed {len(buf)} bytes -> {weights_path}")

    # Also save quantized npz for the golden model
    npz_path = os.path.join(outdir, "gpt2_tiny_int8.npz")
    np.savez(npz_path, **q)
    print(f"Quantized tensors -> {npz_path}")

    # Save metadata
    meta = {
        "hidden": HIDDEN, "ffn_dim": FFN_DIM, "n_heads": N_HEADS,
        "n_layers": N_LAYERS, "vocab_size": VOCAB_SIZE, "max_seq": MAX_SEQ,
        "gemm_scale": GEMM_SCALE, "gemm_shift": GEMM_SHIFT,
        "weights_total": WEIGHTS_TOTAL,
    }
    meta_path = os.path.join(outdir, "config.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Config -> {meta_path}")

    return npz_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp32", default=None, help="Path to gpt2_tiny_fp32.npz")
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()
    outdir = args.outdir or os.environ.get("DEMO_OUTDIR", ".")
    fp32_path = args.fp32 or os.path.join(outdir, "gpt2_tiny_fp32.npz")
    quantize_and_pack(fp32_path, outdir)
