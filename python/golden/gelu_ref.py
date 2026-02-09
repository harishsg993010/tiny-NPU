"""Reference GELU activation matching RTL gelu_lut.sv."""
import numpy as np
from .quant import clamp_i8

def make_gelu_lut(input_scale=32.0):
    """Generate GELU LUT matching RTL gelu_lut.sv.
    Maps int8 -> int8 with GELU activation.
    input_scale: maps int8 to float (x_float = x_int8 / input_scale)
    """
    lut = np.zeros(256, dtype=np.int8)
    for i in range(256):
        signed_i = np.array(i, dtype=np.uint8).view(np.int8)
        x = float(signed_i) / input_scale
        # GELU(x) = x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3)))
        from scipy.special import erf
        gelu_val = x * 0.5 * (1.0 + erf(x / np.sqrt(2.0)))
        # Scale back to int8
        result = gelu_val * input_scale
        lut[i] = np.clip(int(round(result)), -128, 127)
    return lut

GELU_LUT = None  # Lazy init to avoid scipy import at module load

def _get_gelu_lut():
    global GELU_LUT
    if GELU_LUT is None:
        GELU_LUT = make_gelu_lut()
    return GELU_LUT

def gelu_fixed(x):
    """
    Fixed-point GELU matching RTL.
    x: int8 array
    Returns: int8 array
    """
    lut = _get_gelu_lut()
    x = x.astype(np.int8)
    result = np.zeros_like(x, dtype=np.int8)
    for idx in np.ndindex(x.shape):
        unsigned_idx = np.uint8(x[idx])
        result[idx] = lut[unsigned_idx]
    return result
