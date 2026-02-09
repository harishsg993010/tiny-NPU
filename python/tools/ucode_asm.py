#!/usr/bin/env python3
"""
NPU Microcode Assembler.
Converts assembly text to 128-bit binary microcode words.

Instruction format (128 bits):
  [7:0]     opcode
  [15:8]    flags
  [31:16]   dst_base    (SRAM address units)
  [47:32]   src0_base
  [63:48]   src1_base
  [79:64]   M
  [95:80]   N
  [111:96]  K
  [127:112] imm         (scale/shift/mask/etc)

Opcodes:
  NOP=0, DMA_LOAD=1, DMA_STORE=2, GEMM=3, VEC=4, SOFTMAX=5,
  LAYERNORM=6, GELU=7, KV_APPEND=8, KV_READ=9, BARRIER=10, END=255

Flags:
  bit0: TRANSPOSE_B
  bit1: BIAS_EN
  bit2: REQUANT
  bit3: RELU
  bit4: CAUSAL_MASK
  bit5: ACCUMULATE
  bit6: PINGPONG

Example assembly:
  DMA_LOAD  flags=0x00 dst=0x0000 src0=0x0000 src1=0x0000 M=64  N=64 K=0  imm=0
  BARRIER
  GEMM      flags=0x04 dst=0x1000 src0=0x0000 src1=0x0800 M=8   N=64 K=64 imm=0x0107  # scale=1,shift=7
  BARRIER
  END
"""
import sys
import struct
import re
import argparse

OPCODES = {
    'NOP': 0, 'DMA_LOAD': 1, 'DMA_STORE': 2, 'GEMM': 3, 'VEC': 4,
    'SOFTMAX': 5, 'LAYERNORM': 6, 'GELU': 7, 'KV_APPEND': 8,
    'KV_READ': 9, 'BARRIER': 10, 'END': 255
}

FLAGS = {
    'TRANSPOSE_B': 1 << 0,
    'BIAS_EN': 1 << 1,
    'REQUANT': 1 << 2,
    'RELU': 1 << 3,
    'CAUSAL_MASK': 1 << 4,
    'ACCUMULATE': 1 << 5,
    'PINGPONG': 1 << 6,
}

def parse_int(s):
    """Parse integer from string (supports hex 0x prefix)."""
    s = s.strip()
    if s.startswith('0x') or s.startswith('0X'):
        return int(s, 16)
    return int(s)

def encode_instruction(opcode, flags=0, dst=0, src0=0, src1=0, M=0, N=0, K=0, imm=0):
    """Encode a single 128-bit instruction as 16 bytes (little-endian)."""
    word = 0
    word |= (opcode & 0xFF)
    word |= (flags & 0xFF) << 8
    word |= (dst & 0xFFFF) << 16
    word |= (src0 & 0xFFFF) << 32
    word |= (src1 & 0xFFFF) << 48
    word |= (M & 0xFFFF) << 64
    word |= (N & 0xFFFF) << 80
    word |= (K & 0xFFFF) << 96
    word |= (imm & 0xFFFF) << 112
    # Pack as two 64-bit little-endian values
    lo = word & 0xFFFFFFFFFFFFFFFF
    hi = (word >> 64) & 0xFFFFFFFFFFFFFFFF
    return struct.pack('<QQ', lo, hi)

def parse_line(line):
    """Parse one assembly line into an instruction dict."""
    # Remove comments
    line = line.split('#')[0].strip()
    if not line:
        return None

    parts = line.split()
    mnemonic = parts[0].upper()

    if mnemonic not in OPCODES:
        raise ValueError(f"Unknown opcode: {mnemonic}")

    opcode = OPCODES[mnemonic]
    fields = {'opcode': opcode, 'flags': 0, 'dst': 0, 'src0': 0, 'src1': 0,
              'M': 0, 'N': 0, 'K': 0, 'imm': 0}

    # Parse key=value pairs
    for part in parts[1:]:
        if '=' not in part:
            continue
        key, val = part.split('=', 1)
        key = key.strip().lower()
        val = parse_int(val)
        if key in fields:
            fields[key] = val

    return fields

def assemble(text):
    """Assemble text into binary microcode."""
    binary = b''
    for i, line in enumerate(text.strip().split('\n')):
        fields = parse_line(line)
        if fields is None:
            continue
        binary += encode_instruction(**fields)
    return binary

def assemble_file(input_path, output_path, hex_output=False):
    """Assemble a file."""
    with open(input_path, 'r') as f:
        text = f.read()

    binary = assemble(text)

    if hex_output:
        with open(output_path, 'w') as f:
            for i in range(0, len(binary), 16):
                chunk = binary[i:i+16]
                # Write as 4 x 32-bit hex words (for $readmemh)
                for j in range(0, 16, 4):
                    word = struct.unpack_from('<I', chunk, j)[0]
                    f.write(f'{word:08x}\n')
    else:
        with open(output_path, 'wb') as f:
            f.write(binary)

    print(f"Assembled {len(binary)//16} instructions -> {output_path}")

def gen_tiny_transformer_ucode(hidden=64, heads=4, head_dim=16, ffn_dim=256, seq_len=8):
    """Generate microcode for a tiny transformer block verification.

    Memory layout (SRAM addresses in bytes):
    0x0000 - 0x01FF: ACT input (seq_len * hidden = 8*64 = 512 bytes)
    0x0200 - 0x03FF: ACT scratch 1
    0x0400 - 0x05FF: ACT scratch 2
    0x0600 - 0x07FF: QKV output (seq_len * 3*hidden = 8*192 = 1536, but tiled)
    0x1000 - 0x1FFF: Weight tiles (loaded as needed)
    0x2000 - 0x2FFF: ACC scratch (int32)
    0x3000 - 0x3FFF: Softmax scratch

    DDR layout (offsets from DDR_BASE_WGT):
    0x000000: Wqkv [hidden, 3*hidden] = 64*192 = 12288 bytes
    0x003000: Wo [hidden, hidden] = 4096
    0x004000: W1 [hidden, ffn_dim] = 16384
    0x008000: W2 [ffn_dim, hidden] = 16384
    0x00C000: LN1 gamma/beta [hidden]*2 = 128
    0x00C080: LN2 gamma/beta [hidden]*2 = 128
    """
    lines = []

    # Scale/shift packed in imm: scale in low byte, shift in high byte
    imm_requant = (7 << 8) | 1  # shift=7, scale=1

    # Step 0: Load input activations from DDR
    lines.append(f"DMA_LOAD  flags=0x00 dst=0x0000 src0=0x0000 src1=0x0000 M={seq_len} N={hidden} K=0 imm=0")
    lines.append("BARRIER")

    # Step 1: LayerNorm 1 (in-place or to scratch)
    lines.append(f"DMA_LOAD  flags=0x00 dst=0x1000 src0=0x0000 src1=0x0000 M={hidden} N=2 K=0 imm=4")  # Load LN1 gamma/beta, imm=4 -> DDR offset 0xC000
    lines.append("BARRIER")
    lines.append(f"LAYERNORM flags=0x00 dst=0x0200 src0=0x0000 src1=0x1000 M={seq_len} N={hidden} K=0 imm=0")
    lines.append("BARRIER")

    # Step 2: QKV GEMM: x1 * Wqkv
    # Load Wqkv tile by tile and compute
    lines.append(f"DMA_LOAD  flags=0x01 dst=0x1000 src0=0x0000 src1=0x0000 M={hidden} N={3*hidden} K=0 imm=0")  # Load Wqkv
    lines.append("BARRIER")
    lines.append(f"GEMM      flags=0x04 dst=0x0400 src0=0x0200 src1=0x1000 M={seq_len} N={3*hidden} K={hidden} imm={imm_requant}")
    lines.append("BARRIER")

    # Steps 3-7: Attention per head (simplified - process all heads via address offsets)
    for h in range(heads):
        q_off = 0x0400 + h * head_dim
        k_off = 0x0400 + hidden + h * head_dim
        v_off = 0x0400 + 2*hidden + h * head_dim
        score_off = 0x3000 + h * seq_len * seq_len
        prob_off = 0x3000 + heads * seq_len * seq_len + h * seq_len * seq_len
        out_off = 0x0600 + h * head_dim

        # Q*K^T -> scores
        lines.append(f"GEMM      flags=0x05 dst={score_off:#06x} src0={q_off:#06x} src1={k_off:#06x} M={seq_len} N={seq_len} K={head_dim} imm={imm_requant}")
        lines.append("BARRIER")

        # Softmax
        lines.append(f"SOFTMAX   flags=0x10 dst={prob_off:#06x} src0={score_off:#06x} src1=0x0000 M={seq_len} N={seq_len} K=0 imm=0x0003")
        lines.append("BARRIER")

        # P*V -> output
        lines.append(f"GEMM      flags=0x04 dst={out_off:#06x} src0={prob_off:#06x} src1={v_off:#06x} M={seq_len} N={head_dim} K={seq_len} imm={imm_requant}")
        lines.append("BARRIER")

    # Step 8: Output projection
    lines.append(f"DMA_LOAD  flags=0x01 dst=0x1000 src0=0x0000 src1=0x0000 M={hidden} N={hidden} K=0 imm=1")  # Load Wo
    lines.append("BARRIER")
    lines.append(f"GEMM      flags=0x04 dst=0x0200 src0=0x0600 src1=0x1000 M={seq_len} N={hidden} K={hidden} imm={imm_requant}")
    lines.append("BARRIER")

    # Step 9: Residual add: x + y -> x2
    lines.append(f"VEC       flags=0x00 dst=0x0400 src0=0x0000 src1=0x0200 M={seq_len} N={hidden} K=0 imm=0")
    lines.append("BARRIER")

    # Step 10: LayerNorm 2
    lines.append(f"DMA_LOAD  flags=0x00 dst=0x1000 src0=0x0000 src1=0x0000 M={hidden} N=2 K=0 imm=5")
    lines.append("BARRIER")
    lines.append(f"LAYERNORM flags=0x00 dst=0x0200 src0=0x0400 src1=0x1000 M={seq_len} N={hidden} K=0 imm=0")
    lines.append("BARRIER")

    # Step 11: FFN W1
    lines.append(f"DMA_LOAD  flags=0x01 dst=0x1000 src0=0x0000 src1=0x0000 M={hidden} N={ffn_dim} K=0 imm=2")
    lines.append("BARRIER")
    lines.append(f"GEMM      flags=0x04 dst=0x0600 src0=0x0200 src1=0x1000 M={seq_len} N={ffn_dim} K={hidden} imm={imm_requant}")
    lines.append("BARRIER")

    # Step 12: GELU
    lines.append(f"GELU      flags=0x00 dst=0x0600 src0=0x0600 src1=0x0000 M={seq_len} N={ffn_dim} K=0 imm=0")
    lines.append("BARRIER")

    # Step 13: FFN W2
    lines.append(f"DMA_LOAD  flags=0x01 dst=0x1000 src0=0x0000 src1=0x0000 M={ffn_dim} N={hidden} K=0 imm=3")
    lines.append("BARRIER")
    lines.append(f"GEMM      flags=0x04 dst=0x0200 src0=0x0600 src1=0x1000 M={seq_len} N={hidden} K={ffn_dim} imm={imm_requant}")
    lines.append("BARRIER")

    # Step 14: Residual add: x2 + z -> x_out
    lines.append(f"VEC       flags=0x00 dst=0x0000 src0=0x0400 src1=0x0200 M={seq_len} N={hidden} K=0 imm=0")
    lines.append("BARRIER")

    # Store result back to DDR
    lines.append(f"DMA_STORE flags=0x00 dst=0x0000 src0=0x0000 src1=0x0000 M={seq_len} N={hidden} K=0 imm=0")
    lines.append("BARRIER")

    lines.append("END")

    return '\n'.join(lines)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NPU Microcode Assembler')
    parser.add_argument('input', nargs='?', help='Input assembly file')
    parser.add_argument('-o', '--output', default='ucode.bin', help='Output binary file')
    parser.add_argument('--hex', action='store_true', help='Output as hex (for $readmemh)')
    parser.add_argument('--gen-tiny', action='store_true', help='Generate tiny transformer block ucode')
    args = parser.parse_args()

    if args.gen_tiny:
        text = gen_tiny_transformer_ucode()
        print(text)
        binary = assemble(text)
        output = args.output
        if args.hex:
            with open(output, 'w') as f:
                for i in range(0, len(binary), 16):
                    chunk = binary[i:i+16]
                    for j in range(0, 16, 4):
                        word = struct.unpack_from('<I', chunk, j)[0]
                        f.write(f'{word:08x}\n')
        else:
            with open(output, 'wb') as f:
                f.write(binary)
        print(f"\nGenerated {len(binary)//16} instructions -> {output}")
    elif args.input:
        assemble_file(args.input, args.output, args.hex)
    else:
        parser.print_help()
