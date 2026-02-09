"""Reference fixed-point softmax matching RTL LUT-based implementation."""
import numpy as np
from .quant import clamp_i8, clamp_i16

# Match RTL exp LUT: exp(x/32) * 256, for signed int8 input
def make_exp_lut(scale=32.0, output_scale=256):
    """Generate exp LUT matching RTL exp_lut.sv."""
    lut = np.zeros(256, dtype=np.uint16)
    for i in range(256):
        signed_i = np.array(i, dtype=np.uint8).view(np.int8)  # interpret as signed
        x = float(signed_i) / scale
        val = np.exp(x) * output_scale
        lut[i] = np.clip(int(round(val)), 0, 65535)
    return lut

# Match RTL recip LUT: 65536 / x for uint16 input (use top 8 bits)
def make_recip_lut():
    """Generate reciprocal LUT matching RTL recip_lut.sv."""
    lut = np.zeros(256, dtype=np.uint16)
    for i in range(256):
        if i == 0:
            lut[i] = 65535  # max value for 1/0
        else:
            val = 65536.0 / (i * 256.0 / 256.0)  # Simplified: 65536/i but scaled
            # Actually: input is Q8.8 sum. Top 8 bits = integer part.
            # recip = 65536 / (i + 0.5) to center the bin
            val = 65536.0 / max(i, 1)
            lut[i] = np.clip(int(round(val)), 0, 65535)
    return lut

EXP_LUT = make_exp_lut()
RECIP_LUT = make_recip_lut()

def softmax_fixed(scores, causal_mask=None, attn_scale_num=1, attn_scale_shift=3):
    """
    Fixed-point softmax matching RTL softmax_engine.sv.

    scores: int8 array [T] (one row of attention scores)
    causal_mask: bool array [T], True = keep, False = mask to -128
    attn_scale_num, attn_scale_shift: attention scale = num / (2^shift)
    Returns: int8 array [T] (probabilities, sum ~= 127)
    """
    scores = scores.astype(np.int8)
    T = len(scores)

    # Apply causal mask
    if causal_mask is not None:
        scores = np.where(causal_mask, scores, np.int8(-128))

    # Pass 1: find max
    max_val = np.max(scores)

    # Pass 2: subtract max, scale, exp, sum
    exp_vals = np.zeros(T, dtype=np.uint16)
    exp_sum = np.int32(0)
    for i in range(T):
        # Subtract max (result in [-255, 0] range but clamped to int8)
        diff = np.int16(scores[i]) - np.int16(max_val)
        diff = np.clip(diff, -128, 127)
        diff_i8 = np.int8(diff)

        # Apply attention scale (multiply and shift)
        scaled = (np.int16(diff_i8) * np.int16(attn_scale_num)) >> attn_scale_shift
        scaled = np.clip(scaled, -128, 127)

        # Exp LUT lookup
        idx = np.uint8(np.int8(scaled))
        exp_vals[i] = EXP_LUT[idx]
        exp_sum += np.int32(exp_vals[i])

    # Pass 3: normalize (multiply by 1/sum)
    # Get reciprocal from LUT using top 8 bits of sum
    if exp_sum == 0:
        return np.zeros(T, dtype=np.int8)

    sum_top8 = min(int(exp_sum) >> 8, 255)
    if sum_top8 == 0:
        sum_top8 = 1
    inv_sum = RECIP_LUT[sum_top8]

    result = np.zeros(T, dtype=np.int8)
    for i in range(T):
        # prob = (exp_val * inv_sum) >> 16, then scale to int8 range [0..127]
        # exp_val is Q8.8 (0..65535), inv_sum is Q0.16 (0..65535)
        # product is Q8.24, shift by 17 to get Q0.7 (~0..127)
        prob32 = (np.int64(exp_vals[i]) * np.int64(inv_sum)) >> 17
        result[i] = np.clip(prob32, 0, 127).astype(np.int8)

    return result
