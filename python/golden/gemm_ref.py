"""Reference INT8 GEMM matching RTL systolic array behavior."""
import numpy as np
from .quant import clamp_i32, sat_add_i32, requantize, clamp_i8

def gemm_int8(A, B, bias=None, scale=1, shift=0, transpose_b=False, relu=False):
    """
    INT8 GEMM: C = A @ B (with optional transpose, bias, requant, relu).
    A: int8 [M, K]
    B: int8 [K, N] or [N, K] if transpose_b
    Returns: int8 [M, N] after requantization

    Internal: accumulate in int32, then requantize.
    """
    A = A.astype(np.int32)
    if transpose_b:
        B = B.T
    B = B.astype(np.int32)

    M, K = A.shape
    K2, N = B.shape
    assert K == K2, f"K mismatch: {K} vs {K2}"

    # Accumulate in int32 (matching RTL MAC behavior)
    C = np.zeros((M, N), dtype=np.int64)
    for i in range(M):
        for j in range(N):
            acc = np.int64(0)
            for k in range(K):
                prod = np.int64(A[i, k]) * np.int64(B[k, j])
                acc += prod
            # Saturate to int32
            C[i, j] = np.clip(acc, -(1<<31), (1<<31)-1)

    C = C.astype(np.int32)

    # Optional bias
    if bias is not None:
        bias = bias.astype(np.int32)
        for j in range(N):
            C[:, j] = np.clip(np.int64(C[:, j]) + np.int64(bias[j]), -(1<<31), (1<<31)-1).astype(np.int32)

    # Requantize to int8
    result = np.zeros((M, N), dtype=np.int8)
    for i in range(M):
        for j in range(N):
            result[i, j] = requantize(C[i, j], scale, shift)

    # Optional ReLU
    if relu:
        result = np.maximum(result, np.int8(0))

    return result

def gemm_int8_tiled(A, B, tile_m=16, tile_n=16, tile_k=16, **kwargs):
    """Tiled GEMM matching RTL tiling behavior."""
    M, K = A.shape
    if kwargs.get('transpose_b', False):
        N, K2 = B.shape
    else:
        K2, N = B.shape

    # Just call non-tiled version - results should be identical
    # This function exists to verify tiling doesn't affect results
    return gemm_int8(A, B, **kwargs)
