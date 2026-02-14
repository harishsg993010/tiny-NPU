#!/usr/bin/env python3
"""
Generate LUT ROM files for NPU.
Outputs SystemVerilog case statements or $readmemh files for:
- exp_lut (256 entries, uint16)
- recip_lut (256 entries, uint16)
- rsqrt_lut (256 entries, uint16)
- gelu_lut (256 entries, int8)
"""
import numpy as np
import os
import argparse

def gen_exp_lut(scale=32.0, output_scale=256):
    """Generate exp LUT: exp(x_signed / scale) * output_scale."""
    lut = np.zeros(256, dtype=np.uint16)
    for i in range(256):
        signed_i = float(np.array(i, dtype=np.uint8).view(np.int8))
        x = signed_i / scale
        val = np.exp(x) * output_scale
        lut[i] = np.clip(int(round(val)), 0, 65535)
    return lut

def gen_recip_lut():
    """Generate reciprocal LUT: 65536 / x for 8-bit index."""
    lut = np.zeros(256, dtype=np.uint16)
    for i in range(256):
        if i == 0:
            lut[i] = 65535
        else:
            val = 65536.0 / float(i)
            lut[i] = np.clip(int(round(val)), 0, 65535)
    return lut

def gen_rsqrt_lut():
    """Generate inverse sqrt LUT: 256 / sqrt(x) for 8-bit index."""
    lut = np.zeros(256, dtype=np.uint16)
    for i in range(256):
        if i == 0:
            lut[i] = 65535
        else:
            val = 65536.0 / np.sqrt(float(i) * 256.0)
            lut[i] = np.clip(int(round(val)), 0, 65535)
    return lut

def gen_gelu_lut(input_scale=32.0):
    """Generate GELU LUT: int8 -> int8."""
    from scipy.special import erf
    lut = np.zeros(256, dtype=np.int8)
    for i in range(256):
        signed_i = float(np.array(i, dtype=np.uint8).view(np.int8))
        x = signed_i / input_scale
        gelu_val = x * 0.5 * (1.0 + erf(x / np.sqrt(2.0)))
        result = gelu_val * input_scale
        lut[i] = np.clip(int(round(result)), -128, 127)
    return lut

def gen_silu_lut(input_scale=32.0):
    """Generate SiLU LUT: int8 -> int8.
    SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))"""
    lut = np.zeros(256, dtype=np.int8)
    for i in range(256):
        signed_i = float(np.array(i, dtype=np.uint8).view(np.int8))
        x = signed_i / input_scale
        sigmoid_x = 1.0 / (1.0 + np.exp(-x))
        silu_val = x * sigmoid_x
        result = silu_val * input_scale
        lut[i] = np.clip(int(round(result)), -128, 127)
    return lut

def gen_graph_exp_lut(input_scale=32.0):
    """Generate graph-mode exp LUT: int8 -> int8.
    LUT[i] = clamp(round(exp(signed_i / scale) * scale), -128, 127)"""
    lut = np.zeros(256, dtype=np.int8)
    for i in range(256):
        signed_i = float(np.array(i, dtype=np.uint8).view(np.int8))
        x = signed_i / input_scale
        val = np.exp(x) * input_scale
        lut[i] = np.clip(int(round(val)), -128, 127)
    return lut

def gen_graph_log_lut(input_scale=32.0):
    """Generate graph-mode log LUT: int8 -> int8.
    LUT[i] = clamp(round(log(max(signed_i / scale, eps)) * scale), -128, 127)"""
    lut = np.zeros(256, dtype=np.int8)
    eps = 1e-6
    for i in range(256):
        signed_i = float(np.array(i, dtype=np.uint8).view(np.int8))
        x = signed_i / input_scale
        val = np.log(max(x, eps)) * input_scale
        lut[i] = np.clip(int(round(val)), -128, 127)
    return lut

def gen_graph_sqrt_lut(input_scale=32.0):
    """Generate graph-mode sqrt LUT: int8 -> int8.
    LUT[i] = clamp(round(sqrt(max(signed_i / scale, 0)) * scale), -128, 127)"""
    lut = np.zeros(256, dtype=np.int8)
    for i in range(256):
        signed_i = float(np.array(i, dtype=np.uint8).view(np.int8))
        x = signed_i / input_scale
        val = np.sqrt(max(x, 0)) * input_scale
        lut[i] = np.clip(int(round(val)), -128, 127)
    return lut

def gen_graph_rsqrt_lut(input_scale=32.0):
    """Generate graph-mode rsqrt LUT: int8 -> int8.
    LUT[i] = clamp(round(rsqrt(max(signed_i / scale, eps)) * scale), -128, 127)"""
    lut = np.zeros(256, dtype=np.int8)
    eps = 1e-6
    for i in range(256):
        signed_i = float(np.array(i, dtype=np.uint8).view(np.int8))
        x = signed_i / input_scale
        if x > 0:
            val = (1.0 / np.sqrt(x)) * input_scale
        else:
            val = (1.0 / np.sqrt(eps)) * input_scale
        lut[i] = np.clip(int(round(val)), -128, 127)
    return lut

def gen_rope_tables(max_seq=16, head_dim=16, base=10000.0):
    """Generate RoPE sin/cos tables: int8 Q1.7 format.
    Returns (sin_table, cos_table), each [max_seq, head_dim//2]."""
    half_dim = head_dim // 2
    sin_table = np.zeros((max_seq, half_dim), dtype=np.int8)
    cos_table = np.zeros((max_seq, half_dim), dtype=np.int8)
    for pos in range(max_seq):
        for i in range(half_dim):
            freq = 1.0 / (base ** (2.0 * i / head_dim))
            angle = pos * freq
            sin_table[pos, i] = np.clip(int(round(np.sin(angle) * 128.0)), -128, 127)
            cos_table[pos, i] = np.clip(int(round(np.cos(angle) * 128.0)), -128, 127)
    return sin_table, cos_table

def write_sv_case(name, lut, data_width, output_dir):
    """Write a SystemVerilog case-statement LUT."""
    filepath = os.path.join(output_dir, f'{name}_init.sv')
    addr_bits = int(np.ceil(np.log2(len(lut))))

    with open(filepath, 'w') as f:
        f.write(f'// Auto-generated by make_lut.py - DO NOT EDIT\n')
        f.write(f'// LUT: {name}, {len(lut)} entries, {data_width}-bit\n\n')
        f.write(f'function automatic logic [{data_width-1}:0] {name}_lookup;\n')
        f.write(f'  input logic [{addr_bits-1}:0] addr;\n')
        f.write(f'  case (addr)\n')
        for i, val in enumerate(lut):
            if data_width == 8:
                # Signed int8
                uval = int(val) & 0xFF
                f.write(f"    {addr_bits}'d{i}: {name}_lookup = {data_width}'h{uval:02x};\n")
            else:
                uval = int(val) & 0xFFFF
                f.write(f"    {addr_bits}'d{i}: {name}_lookup = {data_width}'h{uval:04x};\n")
        f.write(f'    default: {name}_lookup = {data_width}\'d0;\n')
        f.write(f'  endcase\n')
        f.write(f'endfunction\n')

    print(f"Generated {filepath}")

def write_memh(name, lut, data_width, output_dir):
    """Write a $readmemh file."""
    filepath = os.path.join(output_dir, f'{name}.mem')
    fmt_w = data_width // 4

    with open(filepath, 'w') as f:
        f.write(f'// Auto-generated by make_lut.py\n')
        for val in lut:
            if data_width == 8:
                uval = int(val) & 0xFF
                f.write(f'{uval:02x}\n')
            else:
                uval = int(val) & 0xFFFF
                f.write(f'{uval:04x}\n')

    print(f"Generated {filepath}")

def main():
    parser = argparse.ArgumentParser(description='Generate NPU LUT files')
    parser.add_argument('-o', '--output-dir', default='.', help='Output directory')
    parser.add_argument('--format', choices=['case', 'memh', 'both'], default='both',
                       help='Output format')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    luts = [
        ('exp_lut', gen_exp_lut(), 16),
        ('recip_lut', gen_recip_lut(), 16),
        ('rsqrt_lut', gen_rsqrt_lut(), 16),
        ('gelu_lut', gen_gelu_lut(), 8),
        ('silu_lut', gen_silu_lut(), 8),
        ('graph_exp_lut', gen_graph_exp_lut(), 8),
        ('graph_log_lut', gen_graph_log_lut(), 8),
        ('graph_sqrt_lut', gen_graph_sqrt_lut(), 8),
        ('graph_rsqrt_lut', gen_graph_rsqrt_lut(), 8),
    ]

    for name, data, width in luts:
        if args.format in ('case', 'both'):
            write_sv_case(name, data, width, args.output_dir)
        if args.format in ('memh', 'both'):
            write_memh(name, data, width, args.output_dir)

    # RoPE tables (flat sin/cos as memh files)
    sin_tbl, cos_tbl = gen_rope_tables()
    if args.format in ('memh', 'both'):
        write_memh('rope_sin', sin_tbl.flatten(), 8, args.output_dir)
        write_memh('rope_cos', cos_tbl.flatten(), 8, args.output_dir)

if __name__ == '__main__':
    main()
