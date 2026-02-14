#!/usr/bin/env python3
"""Generate LUT hex values for exp, log, sqrt, rsqrt."""
import math

def signed_i(i):
    return i if i < 128 else i - 256

def to_twos_comp_hex(val):
    val = int(val)
    if val < 0:
        val = val & 0xFF
    return f"{val:02X}"

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

# EXP: LUT[i] = clamp(round(exp(signed_i/32)*32), -128, 127)
print("EXP:")
for i in range(256):
    si = signed_i(i)
    raw = math.exp(si / 32.0) * 32.0
    result = clamp(round(raw), -128, 127)
    print(f"  rom[{i:3d}] = 8'h{to_twos_comp_hex(result)};")

# LOG: LUT[i] = clamp(round(log(max(signed_i/32, 0.001))*32), -128, 127)
print("\nLOG:")
for i in range(256):
    si = signed_i(i)
    x = max(si / 32.0, 0.001)
    raw = math.log(x) * 32.0
    result = clamp(round(raw), -128, 127)
    print(f"  rom[{i:3d}] = 8'h{to_twos_comp_hex(result)};")

# SQRT: LUT[i] = clamp(round(sqrt(max(signed_i/32, 0))*32), -128, 127)
print("\nSQRT:")
for i in range(256):
    si = signed_i(i)
    x = max(si / 32.0, 0.0)
    raw = math.sqrt(x) * 32.0
    result = clamp(round(raw), -128, 127)
    print(f"  rom[{i:3d}] = 8'h{to_twos_comp_hex(result)};")

# RSQRT: LUT[i] = clamp(round(1/sqrt(max(signed_i/32, 0.001))*32), -128, 127)
print("\nRSQRT:")
for i in range(256):
    si = signed_i(i)
    x = max(si / 32.0, 0.001)
    raw = (1.0 / math.sqrt(x)) * 32.0
    result = clamp(round(raw), -128, 127)
    print(f"  rom[{i:3d}] = 8'h{to_twos_comp_hex(result)};")
