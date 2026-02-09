"""
Reference single attention head pipeline in fixed-point.
Uses gemm_ref and softmax_ref to compute:
  Q = X * Wq, K = X * Wk, V = X * Wv
  S = Q * K^T
  P = softmax(S, causal mask)
  O = P * V
All operations match RTL exactly.
"""
import numpy as np
from .gemm_ref import gemm_int8
from .softmax_ref import softmax_fixed


def attention_head_fixed(X, Wq, Wk, Wv, scale=1, shift=7, causal=True):
    """
    Compute single attention head in fixed-point matching RTL.

    Args:
        X:  int8 [seq_len, head_dim] - input activations
        Wq: int8 [head_dim, head_dim] - query weight matrix
        Wk: int8 [head_dim, head_dim] - key weight matrix
        Wv: int8 [head_dim, head_dim] - value weight matrix
        scale: int - requantization scale (default 1)
        shift: int - requantization shift (default 7)
        causal: bool - apply causal masking to softmax

    Returns:
        dict with keys: Q, K, V, S, P, O (all int8 numpy arrays)
    """
    seq_len = X.shape[0]

    # Q = X * Wq
    Q = gemm_int8(X, Wq, scale=scale, shift=shift)

    # K = X * Wk
    K = gemm_int8(X, Wk, scale=scale, shift=shift)

    # V = X * Wv
    V = gemm_int8(X, Wv, scale=scale, shift=shift)

    # S = Q * K^T
    S = gemm_int8(Q, K, transpose_b=True, scale=scale, shift=shift)

    # P = softmax(S) row by row with causal mask
    P = np.zeros_like(S)
    for row in range(seq_len):
        if causal:
            # Causal mask: keep positions 0..row, mask positions row+1..seq_len-1
            mask = np.array([i <= row for i in range(seq_len)])
        else:
            mask = None
        # No attention scaling beyond what's in the scores
        P[row] = softmax_fixed(S[row], causal_mask=mask,
                               attn_scale_num=1, attn_scale_shift=0)

    # O = P * V
    O = gemm_int8(P, V, scale=scale, shift=shift)

    return {'Q': Q, 'K': K, 'V': V, 'S': S, 'P': P, 'O': O}


def main():
    """Generate and print golden values for the integration test."""
    np.random.seed(42)

    seq_len = 8
    head_dim = 16

    # Generate data matching tb_integration.cpp (srand(42), range [-10,10] and [-5,5])
    # Use same RNG sequence as C: rand()%21-10 and rand()%11-5
    # We replicate the C rand() sequence approximately using numpy
    rng = np.random.RandomState(42)

    X = np.zeros((seq_len, head_dim), dtype=np.int8)
    for i in range(seq_len):
        for j in range(head_dim):
            X[i, j] = rng.randint(-10, 11)

    Wq = np.zeros((head_dim, head_dim), dtype=np.int8)
    Wk = np.zeros((head_dim, head_dim), dtype=np.int8)
    Wv = np.zeros((head_dim, head_dim), dtype=np.int8)
    for i in range(head_dim):
        for j in range(head_dim):
            Wq[i, j] = rng.randint(-5, 6)
            Wk[i, j] = rng.randint(-5, 6)
            Wv[i, j] = rng.randint(-5, 6)

    results = attention_head_fixed(X, Wq, Wk, Wv, scale=1, shift=7, causal=True)

    print("=== Attention Head Reference ===")
    print(f"seq_len={seq_len}, head_dim={head_dim}")
    print()

    for name in ['Q', 'K', 'V', 'S', 'P', 'O']:
        arr = results[name]
        print(f"{name} shape={arr.shape}:")
        for row in range(min(arr.shape[0], 4)):
            vals = ' '.join(f'{int(v):4d}' for v in arr[row])
            print(f"  [{row}] {vals}")
        if arr.shape[0] > 4:
            print(f"  ... ({arr.shape[0] - 4} more rows)")
        print()

    # Print P row sums (should be ~127 for full rows, less for masked)
    print("P row sums:", [int(np.sum(results['P'][i])) for i in range(seq_len)])
    print()

    return results


if __name__ == '__main__':
    main()
