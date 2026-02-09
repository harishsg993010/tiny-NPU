"""Reference GPT-2 transformer block using fixed-point arithmetic."""
import numpy as np
from .quant import clamp_i8, requantize
from .gemm_ref import gemm_int8
from .softmax_ref import softmax_fixed
from .layernorm_ref import layernorm_fixed
from .gelu_ref import gelu_fixed

class GPT2BlockRef:
    """Reference implementation of one GPT-2 transformer block."""

    def __init__(self, hidden=64, heads=4, head_dim=16, ffn_dim=256, seq_len=8,
                 scale=1, shift=7):
        self.hidden = hidden
        self.heads = heads
        self.head_dim = head_dim
        self.ffn_dim = ffn_dim
        self.seq_len = seq_len
        self.scale = scale
        self.shift = shift

        # Initialize random int8 weights
        rng = np.random.RandomState(42)
        self.ln1_gamma = rng.randint(-64, 64, hidden).astype(np.int8)
        self.ln1_beta = rng.randint(-64, 64, hidden).astype(np.int8)
        self.Wqkv = rng.randint(-32, 32, (hidden, 3 * hidden)).astype(np.int8)
        self.Wo = rng.randint(-32, 32, (hidden, hidden)).astype(np.int8)
        self.ln2_gamma = rng.randint(-64, 64, hidden).astype(np.int8)
        self.ln2_beta = rng.randint(-64, 64, hidden).astype(np.int8)
        self.W1 = rng.randint(-32, 32, (hidden, ffn_dim)).astype(np.int8)
        self.W2 = rng.randint(-32, 32, (ffn_dim, hidden)).astype(np.int8)

    def forward(self, x, kv_cache_k=None, kv_cache_v=None):
        """
        Full transformer block forward pass.
        x: int8 [seq_len, hidden]
        Returns: int8 [seq_len, hidden], updated kv_cache
        """
        S, H = x.shape
        assert H == self.hidden

        # 1. LayerNorm 1
        x1 = np.zeros_like(x)
        for s in range(S):
            x1[s] = layernorm_fixed(x[s], self.ln1_gamma, self.ln1_beta)

        # 2. QKV projection
        qkv = gemm_int8(x1, self.Wqkv, scale=self.scale, shift=self.shift)

        # 3. Split into Q, K, V [seq, heads, head_dim]
        q = qkv[:, :self.hidden].reshape(S, self.heads, self.head_dim)
        k = qkv[:, self.hidden:2*self.hidden].reshape(S, self.heads, self.head_dim)
        v = qkv[:, 2*self.hidden:].reshape(S, self.heads, self.head_dim)

        # 4. Update KV cache
        if kv_cache_k is not None:
            k = np.concatenate([kv_cache_k, k], axis=0)
            v = np.concatenate([kv_cache_v, v], axis=0)
        T = k.shape[0]  # total sequence length including cache

        # 5-7. Multi-head attention
        o = np.zeros((S, self.heads, self.head_dim), dtype=np.int8)
        for h in range(self.heads):
            q_h = q[:, h, :]  # [S, head_dim]
            k_h = k[:, h, :]  # [T, head_dim]
            v_h = v[:, h, :]  # [T, head_dim]

            # Scores: Q @ K^T [S, T]
            scores = gemm_int8(q_h, k_h, transpose_b=True, scale=self.scale, shift=self.shift)

            # Softmax per row with causal mask
            probs = np.zeros((S, T), dtype=np.int8)
            for s in range(S):
                causal = np.arange(T) <= (s + T - S)  # causal mask
                probs[s] = softmax_fixed(scores[s], causal_mask=causal)

            # Context: P @ V [S, head_dim]
            o[:, h, :] = gemm_int8(probs, v_h, scale=self.scale, shift=self.shift)

        # Reshape back to [S, hidden]
        o_flat = o.reshape(S, self.hidden)

        # 8. Output projection
        y = gemm_int8(o_flat, self.Wo, scale=self.scale, shift=self.shift)

        # 9. Residual add
        x2 = clamp_i8(x.astype(np.int16) + y.astype(np.int16))

        # 10. LayerNorm 2
        x3 = np.zeros_like(x2)
        for s in range(S):
            x3[s] = layernorm_fixed(x2[s], self.ln2_gamma, self.ln2_beta)

        # 11. FFN W1
        h = gemm_int8(x3, self.W1, scale=self.scale, shift=self.shift)

        # 12. GELU
        h2 = gelu_fixed(h)

        # 13. FFN W2
        z = gemm_int8(h2, self.W2, scale=self.scale, shift=self.shift)

        # 14. Residual add
        x_out = clamp_i8(x2.astype(np.int16) + z.astype(np.int16))

        return x_out, k, v

def run_reference_block(hidden=64, heads=4, head_dim=16, ffn_dim=256, seq_len=8, seed=42):
    """Run a reference transformer block and return inputs/outputs for verification."""
    rng = np.random.RandomState(seed)
    x = rng.randint(-64, 64, (seq_len, hidden)).astype(np.int8)

    block = GPT2BlockRef(hidden=hidden, heads=heads, head_dim=head_dim,
                         ffn_dim=ffn_dim, seq_len=seq_len)
    x_out, k_cache, v_cache = block.forward(x)

    return {
        'input': x,
        'output': x_out,
        'k_cache': k_cache,
        'v_cache': v_cache,
        'weights': {
            'Wqkv': block.Wqkv,
            'Wo': block.Wo,
            'W1': block.W1,
            'W2': block.W2,
            'ln1_gamma': block.ln1_gamma,
            'ln1_beta': block.ln1_beta,
            'ln2_gamma': block.ln2_gamma,
            'ln2_beta': block.ln2_beta,
        }
    }

if __name__ == '__main__':
    result = run_reference_block()
    print(f"Input shape: {result['input'].shape}")
    print(f"Output shape: {result['output'].shape}")
    print(f"Output sample: {result['output'][0, :8]}")
