#!/usr/bin/env python3
"""Export GPT-2 weights from Hugging Face, sliced to NPU tiny configuration."""
import argparse, os, sys
import numpy as np

# Add parent so we can import ddr_map
sys.path.insert(0, os.path.dirname(__file__))
from ddr_map import HIDDEN, FFN_DIM, N_LAYERS, VOCAB_SIZE, MAX_SEQ


def export(model_name="gpt2", outfile="gpt2_tiny_fp32.npz"):
    from transformers import GPT2LMHeadModel
    print(f"Loading {model_name} from Hugging Face...")
    model = GPT2LMHeadModel.from_pretrained(model_name)
    sd = model.state_dict()

    weights = {}
    H = HIDDEN   # 16
    F = FFN_DIM  # 64
    V = VOCAB_SIZE  # 256
    S = MAX_SEQ  # 32
    H_full = model.config.n_embd  # 768 for gpt2

    # Token embeddings: (50257, 768) -> (256, 16)
    weights["wte"] = sd["transformer.wte.weight"].numpy()[:V, :H].copy()
    # Position embeddings: (1024, 768) -> (32, 16)
    weights["wpe"] = sd["transformer.wpe.weight"].numpy()[:S, :H].copy()

    for i in range(N_LAYERS):
        pfx = f"transformer.h.{i}."
        # LN1 gamma/beta  (768,) -> (16,)
        weights[f"block{i}.ln1.gamma"] = sd[pfx + "ln_1.weight"].numpy()[:H].copy()
        weights[f"block{i}.ln1.beta"]  = sd[pfx + "ln_1.bias"].numpy()[:H].copy()

        # c_attn weight (768, 2304) -> Q,K,V each (16,16)
        ca_w = sd[pfx + "attn.c_attn.weight"].numpy()
        weights[f"block{i}.wq"] = ca_w[:H, :H].copy()
        weights[f"block{i}.wk"] = ca_w[:H, H_full:H_full + H].copy()
        weights[f"block{i}.wv"] = ca_w[:H, 2*H_full:2*H_full + H].copy()

        # c_proj weight (768, 768) -> (16, 16)
        weights[f"block{i}.wo"] = sd[pfx + "attn.c_proj.weight"].numpy()[:H, :H].copy()

        # LN2 gamma/beta
        weights[f"block{i}.ln2.gamma"] = sd[pfx + "ln_2.weight"].numpy()[:H].copy()
        weights[f"block{i}.ln2.beta"]  = sd[pfx + "ln_2.bias"].numpy()[:H].copy()

        # FFN: c_fc (768, 3072) -> (16, 64), c_proj (3072, 768) -> (64, 16)
        weights[f"block{i}.w1"] = sd[pfx + "mlp.c_fc.weight"].numpy()[:H, :F].copy()
        weights[f"block{i}.w2"] = sd[pfx + "mlp.c_proj.weight"].numpy()[:F, :H].copy()

    # Final LayerNorm
    weights["ln_f.gamma"] = sd["transformer.ln_f.weight"].numpy()[:H].copy()
    weights["ln_f.beta"]  = sd["transformer.ln_f.bias"].numpy()[:H].copy()

    # lm_head: tied to wte in GPT-2 -> (256, 16)
    weights["lm_head"] = weights["wte"].copy()

    np.savez(outfile, **weights)
    print(f"Exported {len(weights)} tensors -> {outfile}")
    return outfile


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--outfile", default=None)
    args = ap.parse_args()
    outdir = os.environ.get("DEMO_OUTDIR", ".")
    outfile = args.outfile or os.path.join(outdir, "gpt2_tiny_fp32.npz")
    export(args.model, outfile)
