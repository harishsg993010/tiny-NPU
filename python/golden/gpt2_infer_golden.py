#!/usr/bin/env python3
"""
Golden INT8 GPT-2 inference matching NPU hardware exactly.

Uses the SAME quantized weights and the SAME fixed-point arithmetic
(GEMM, LayerNorm, Softmax, GELU) as the NPU.
"""
import argparse, os, sys, json
import numpy as np

# Import existing golden modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
from ddr_map import (HIDDEN, FFN_DIM, N_HEADS, HEAD_DIM, N_LAYERS,
                     VOCAB_SIZE, MAX_SEQ, GEMM_SCALE, GEMM_SHIFT,
                     BLOCKS_OFFSET, BLOCK_SIZE, BLK_LN1_BETA, BLK_WQ,
                     BLK_WK, BLK_WV, BLK_WO, BLK_LN2_BETA, BLK_W1, BLK_W2,
                     WTE_OFFSET, WPE_OFFSET, LN_F_OFFSET, LM_HEAD_OFFSET,
                     WTE_SIZE, WPE_SIZE, LN_F_SIZE, LM_HEAD_SIZE, WEIGHTS_TOTAL)
from golden.gemm_ref import gemm_int8
from golden.softmax_ref import softmax_fixed
from golden.layernorm_ref import layernorm_fixed
from golden.gelu_ref import gelu_fixed
from golden.quant import clamp_i8


class TinyGPT2Golden:
    """INT8 GPT-2 inference golden model."""

    def __init__(self, weights_bin_path):
        """Load quantized weights from weights.bin."""
        with open(weights_bin_path, "rb") as f:
            raw = f.read()
        assert len(raw) == WEIGHTS_TOTAL, f"Expected {WEIGHTS_TOTAL} bytes, got {len(raw)}"
        buf = np.frombuffer(raw, dtype=np.int8)

        self.wte = buf[WTE_OFFSET:WTE_OFFSET + WTE_SIZE].reshape(VOCAB_SIZE, HIDDEN).copy()
        self.wpe = buf[WPE_OFFSET:WPE_OFFSET + WPE_SIZE].reshape(MAX_SEQ, HIDDEN).copy()

        self.blocks = []
        for i in range(N_LAYERS):
            base = BLOCKS_OFFSET + i * BLOCK_SIZE
            blk = {
                "ln1_beta": buf[base + BLK_LN1_BETA : base + BLK_LN1_BETA + HIDDEN].copy(),
                "wq": buf[base + BLK_WQ : base + BLK_WQ + HIDDEN*HIDDEN].reshape(HIDDEN, HIDDEN).copy(),
                "wk": buf[base + BLK_WK : base + BLK_WK + HIDDEN*HIDDEN].reshape(HIDDEN, HIDDEN).copy(),
                "wv": buf[base + BLK_WV : base + BLK_WV + HIDDEN*HIDDEN].reshape(HIDDEN, HIDDEN).copy(),
                "wo": buf[base + BLK_WO : base + BLK_WO + HIDDEN*HIDDEN].reshape(HIDDEN, HIDDEN).copy(),
                "ln2_beta": buf[base + BLK_LN2_BETA : base + BLK_LN2_BETA + HIDDEN].copy(),
                "w1": buf[base + BLK_W1 : base + BLK_W1 + HIDDEN*FFN_DIM].reshape(HIDDEN, FFN_DIM).copy(),
                "w2": buf[base + BLK_W2 : base + BLK_W2 + FFN_DIM*HIDDEN].reshape(FFN_DIM, HIDDEN).copy(),
            }
            self.blocks.append(blk)

        self.ln_f_beta = buf[LN_F_OFFSET:LN_F_OFFSET + LN_F_SIZE].copy()
        self.lm_head = buf[LM_HEAD_OFFSET:LM_HEAD_OFFSET + LM_HEAD_SIZE].reshape(VOCAB_SIZE, HIDDEN).copy()

    def embed(self, token_ids):
        """Compute embeddings: WTE[token] + WPE[pos], clamped to int8."""
        S = len(token_ids)
        emb = np.zeros((S, HIDDEN), dtype=np.int8)
        for p, tok in enumerate(token_ids):
            for h in range(HIDDEN):
                val = int(self.wte[tok, h]) + int(self.wpe[p, h])
                emb[p, h] = np.clip(val, -128, 127)
        return emb

    def run_block(self, x, blk_idx):
        """Run one transformer block (matching NPU microcode exactly)."""
        blk = self.blocks[blk_idx]
        S = x.shape[0]
        scale, shift = GEMM_SCALE, GEMM_SHIFT

        # 1. LayerNorm 1
        ln1_out = np.zeros_like(x)
        for s in range(S):
            ln1_out[s] = layernorm_fixed(x[s], beta=blk["ln1_beta"])

        # 2. Q, K, V GEMMs
        Q = gemm_int8(ln1_out, blk["wq"], scale=scale, shift=shift)
        K = gemm_int8(ln1_out, blk["wk"], scale=scale, shift=shift)
        V = gemm_int8(ln1_out, blk["wv"], scale=scale, shift=shift)

        # 3. Attention scores: S_mat = Q * K^T
        S_mat = gemm_int8(Q, K, transpose_b=True, scale=scale, shift=shift)

        # 4. Softmax (per row, causal mask)
        P = np.zeros((S, S), dtype=np.int8)
        for s in range(S):
            causal = np.arange(S) <= s
            P[s] = softmax_fixed(S_mat[s], causal_mask=causal)

        # 5. Attention output: ATTN = P * V
        ATTN = gemm_int8(P, V, scale=scale, shift=shift)

        # 6. Output projection: WO_out = ATTN * Wo
        WO_out = gemm_int8(ATTN, blk["wo"], scale=scale, shift=shift)

        # 7. Residual 1: X2 = WO_out + X
        X2 = clamp_i8(WO_out.astype(np.int16) + x.astype(np.int16))

        # 8. LayerNorm 2
        ln2_out = np.zeros_like(X2)
        for s in range(S):
            ln2_out[s] = layernorm_fixed(X2[s], beta=blk["ln2_beta"])

        # 9. FFN1 = LN2_out * W1
        ffn1 = gemm_int8(ln2_out, blk["w1"], scale=scale, shift=shift)

        # 10. GELU
        gelu_out = gelu_fixed(ffn1)

        # 11. FFN2 = GELU_out * W2
        ffn2 = gemm_int8(gelu_out, blk["w2"], scale=scale, shift=shift)

        # 12. Residual 2: X_out = FFN2 + X2
        X_out = clamp_i8(ffn2.astype(np.int16) + X2.astype(np.int16))

        return X_out

    def forward(self, token_ids):
        """Full forward pass: embed -> blocks -> ln_f -> lm_head -> logits."""
        # Embeddings
        x = self.embed(token_ids)

        # Transformer blocks
        for b in range(N_LAYERS):
            x = self.run_block(x, b)

        # Final LayerNorm (last token only for efficiency, but compute all for simplicity)
        ln_f_out = np.zeros_like(x)
        for s in range(x.shape[0]):
            ln_f_out[s] = layernorm_fixed(x[s], beta=self.ln_f_beta)

        # LM Head: logits = ln_f_out[-1] @ lm_head.T (last token only)
        last_hidden = ln_f_out[-1:, :]  # [1, 16]
        logits = gemm_int8(last_hidden, self.lm_head, transpose_b=True,
                           scale=GEMM_SCALE, shift=GEMM_SHIFT)  # [1, 256]
        return logits[0]  # [256]

    def generate(self, prompt_tokens, max_new_tokens=20,
                 temperature=0.0, rng=None):
        """Autoregressive decode loop with optional temperature sampling."""
        tokens = list(prompt_tokens)
        generated_logits = []
        generated_tokens = []

        for step in range(max_new_tokens):
            seq_len = len(tokens)
            if seq_len > MAX_SEQ:
                print(f"Warning: seq_len {seq_len} > MAX_SEQ {MAX_SEQ}, truncating")
                break

            logits = self.forward(tokens)

            if temperature > 0 and rng is not None:
                # Temperature-based sampling
                logits_f = logits.astype(np.float64)
                logits_f = (logits_f - logits_f.max()) / temperature
                probs = np.exp(logits_f)
                probs = probs / probs.sum()
                next_tok = int(rng.choice(len(probs), p=probs))
            else:
                next_tok = int(np.argmax(logits))

            generated_logits.append(logits.copy())
            generated_tokens.append(next_tok)
            tokens.append(next_tok)

            # Check for common stop conditions (no EOS in reduced vocab)
            print(f"  Step {step}: token={next_tok} logits_range=[{logits.min()}, {logits.max()}]")

        return tokens, generated_tokens, generated_logits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=None, help="Path to weights.bin")
    ap.add_argument("--prompt", default="Hello", help="Input prompt")
    ap.add_argument("--max-tokens", type=int, default=20)
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="Sampling temperature (0 = greedy)")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed for sampling")
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    outdir = args.outdir or os.environ.get("DEMO_OUTDIR", ".")
    weights_path = args.weights or os.path.join(outdir, "weights.bin")

    # Byte-level tokenization using GPT-2's byte vocabulary.
    # Our model uses the first 256 entries of GPT-2's embedding matrix,
    # which correspond to the 256 byte-level tokens.  We encode the prompt
    # by mapping each UTF-8 byte to its GPT-2 byte-token ID, so every
    # token ID is guaranteed to be in [0, 255].
    from transformers import GPT2Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # Reproduce GPT-2's bytes_to_unicode() mapping
    def bytes_to_unicode():
        bs = (list(range(ord("!"), ord("~")+1))
              + list(range(ord("\xa1"), ord("\xac")+1))
              + list(range(ord("\xae"), ord("\xff")+1)))
        cs = list(bs)
        n = 0
        for b in range(256):
            if b not in bs:
                bs.append(b)
                cs.append(256 + n)
                n += 1
        return dict(zip(bs, [chr(c) for c in cs]))

    byte_encoder = bytes_to_unicode()  # byte_value -> unicode_char

    # Build byte-value -> token-id mapping via the tokenizer vocab
    byte_to_tid = {}
    vocab = tokenizer.get_vocab()
    for byte_val, uchar in byte_encoder.items():
        tid = vocab.get(uchar)
        if tid is not None and tid < VOCAB_SIZE:
            byte_to_tid[byte_val] = tid

    # Build token-id -> byte-value mapping (for decoding)
    tid_to_byte = {v: k for k, v in byte_to_tid.items()}

    # Encode prompt as UTF-8 bytes -> token IDs
    prompt_bytes = args.prompt.encode("utf-8")
    prompt_ids = [byte_to_tid.get(b, 0) for b in prompt_bytes]

    print(f"Prompt: '{args.prompt}' -> bytes {list(prompt_bytes)} -> tokens: {prompt_ids}")

    # Save prompt tokens
    with open(os.path.join(outdir, "prompt_tokens.txt"), "w") as f:
        f.write(" ".join(str(t) for t in prompt_ids) + "\n")

    # Run golden inference
    model = TinyGPT2Golden(weights_path)
    rng = np.random.RandomState(args.seed) if args.temperature > 0 else None
    all_tokens, gen_tokens, gen_logits = model.generate(
        prompt_ids, args.max_tokens,
        temperature=args.temperature, rng=rng)

    # Decode generated tokens back to text via byte mapping
    gen_bytes = bytes([tid_to_byte.get(t, ord("?")) for t in all_tokens])
    decoded = gen_bytes.decode("utf-8", errors="replace")
    print(f"\nGolden generated text: '{decoded}'")
    print(f"Golden tokens: {all_tokens}")

    # Save golden results
    with open(os.path.join(outdir, "golden_tokens.txt"), "w") as f:
        for t in gen_tokens:
            f.write(f"{t}\n")

    with open(os.path.join(outdir, "golden_text.txt"), "w") as f:
        f.write(decoded + "\n")

    # Save logits as binary (for C++ comparison)
    logits_arr = np.array(gen_logits, dtype=np.int8)  # [max_tokens, VOCAB_SIZE]
    logits_arr.tofile(os.path.join(outdir, "golden_logits.bin"))

    # Save metadata (temperature, seed)
    meta = {
        "temperature": args.temperature,
        "seed": args.seed,
        "max_tokens": args.max_tokens,
        "prompt": args.prompt,
        "hidden": HIDDEN,
        "n_layers": N_LAYERS,
        "ffn_dim": FFN_DIM,
    }
    with open(os.path.join(outdir, "golden_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nGolden results saved to {outdir}/")
    print("  golden_tokens.txt, golden_text.txt, golden_logits.bin, golden_meta.json")


if __name__ == "__main__":
    main()
