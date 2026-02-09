"""Reference fixed-point LayerNorm matching RTL layernorm_engine.sv."""
import numpy as np
from .quant import clamp_i8, requantize

def make_rsqrt_lut():
    """Generate rsqrt LUT matching RTL rsqrt_lut.sv."""
    lut = np.zeros(256, dtype=np.uint16)
    for i in range(256):
        if i == 0:
            lut[i] = 65535
        else:
            val = 65536.0 / np.sqrt(float(i) * 256.0)
            lut[i] = np.clip(int(round(val)), 0, 65535)
    return lut

RSQRT_LUT = make_rsqrt_lut()

def layernorm_fixed(x, gamma=None, beta=None):
    """
    Fixed-point LayerNorm matching RTL.

    x: int8 array [hidden_dim]
    gamma: int8 array [hidden_dim] (scale, optional)
    beta: int8 array [hidden_dim] (bias, optional)
    Returns: int8 array [hidden_dim]
    """
    x = x.astype(np.int32)
    N = len(x)

    # Compute mean (fixed-point)
    total = np.sum(x)
    # For power-of-2 N, use shift; otherwise approximate
    mean = total // N  # integer division

    # Compute variance
    diff = x - mean  # int32
    sq = diff * diff  # int32 (may overflow for large values, but OK for small dims)
    variance = np.sum(sq) // N

    # rsqrt via LUT (use top 8 bits of variance)
    # Match RTL: no epsilon guard — LUT[0]=65535 handles near-zero variance
    var_top8 = min(int(variance) >> 8, 255) if variance > 0 else 0
    inv_std = RSQRT_LUT[var_top8]  # Q0.16

    # Normalize: out[i] = (x[i] - mean) * inv_std
    result = np.zeros(N, dtype=np.int8)
    for i in range(N):
        d = np.int32(x[i]) - np.int32(mean)  # int16 range
        # d * inv_std: int32 * uint16 -> need careful scaling
        normed = (np.int64(d) * np.int64(inv_std)) >> 16  # back to ~int8 range

        if gamma is not None:
            normed = (normed * np.int64(gamma[i])) >> 7  # gamma is Q1.7
        if beta is not None:
            normed = normed + np.int64(beta[i])

        result[i] = np.clip(normed, -128, 127).astype(np.int8)

    return result
