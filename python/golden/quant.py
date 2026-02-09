"""Quantization utilities matching RTL fixed-point arithmetic exactly."""
import numpy as np

def clamp_i8(x):
    """Clamp to signed 8-bit range [-128, 127]."""
    return np.clip(x, -128, 127).astype(np.int8)

def clamp_i16(x):
    """Clamp to signed 16-bit range."""
    return np.clip(x, -32768, 32767).astype(np.int16)

def clamp_i32(x):
    """Clamp to signed 32-bit range."""
    return np.clip(x, -(1<<31), (1<<31)-1).astype(np.int32)

def mul_i8(a, b):
    """Multiply two int8 values -> int16, matching RTL."""
    return np.int16(np.int16(a) * np.int16(b))

def sat_add_i32(a, b):
    """Saturating 32-bit add."""
    result = np.int64(a) + np.int64(b)
    return clamp_i32(result)

def round_nearest_even(x):
    """Round to nearest even (banker's rounding), matching RTL.
    x is a float or fixed-point value.
    """
    return np.rint(x).astype(np.int32)

def requantize(acc, scale, shift):
    """Requantize int32 accumulator to int8.
    result = clamp_i8(round((acc * scale) >> shift))
    Scale is int8, shift is uint8.
    Matches RTL requantize function exactly.
    """
    acc = np.int64(acc)
    scale = np.int64(scale)
    # Multiply
    product = acc * scale
    # Add rounding bias before shift: (1 << (shift-1)) for round-to-nearest
    if shift > 0:
        rounding = np.int64(1) << (int(shift) - 1)
        product = product + rounding
    # Arithmetic right shift
    result = product >> int(shift)
    return clamp_i8(result)

# Vectorized versions
def requantize_vec(acc_vec, scale, shift):
    """Requantize a vector of int32 accumulators to int8."""
    return np.array([requantize(a, scale, shift) for a in acc_vec.flatten()]).reshape(acc_vec.shape).astype(np.int8)
