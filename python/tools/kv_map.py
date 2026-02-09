"""KV cache layout constants and decode-mode SRAM0 addresses.

Single source of truth for KV cache dimensions, instruction encoding,
and the SRAM0 memory map used during single-token decode steps.

All constants must match the C++ counterparts in tb_demo_infer.cpp.
"""
from ddr_map import (HIDDEN, N_HEADS, HEAD_DIM, FFN_DIM, N_LAYERS,
                     VOCAB_SIZE, MAX_SEQ,
                     ADDR_WQ, ADDR_WK, ADDR_WV, ADDR_WO, ADDR_W1, ADDR_W2)

# =============================================================================
# KV Cache Dimensions
# =============================================================================
KV_MAX_SEQ   = MAX_SEQ     # 16  (max cached positions)
KV_NUM_HEADS = N_HEADS     # 1
KV_HEAD_DIM  = HEAD_DIM    # 64  (= HIDDEN for single-head)

# Per-layer KV cache size:  2 * MAX_SEQ * HIDDEN bytes (K + V)
KV_LAYER_SIZE = 2 * KV_MAX_SEQ * HIDDEN   # 2048
KV_TOTAL_SIZE = N_LAYERS * KV_LAYER_SIZE   # 8192

# =============================================================================
# Decode-mode SRAM0 addresses  (S = 1, weights unchanged at 0x0000-0xBFFF)
# =============================================================================
# Weights occupy the same addresses as prefill mode:
#   ADDR_WQ = 0x0000   [64][64] = 4096B
#   ADDR_WK = 0x1000   [64][64] = 4096B
#   ADDR_WV = 0x2000   [64][64] = 4096B
#   ADDR_WO = 0x3000   [64][64] = 4096B
#   ADDR_W1 = 0x4000   [64][256]= 16384B
#   ADDR_W2 = 0x8000   [256][64]= 16384B

# Activations for single-token decode (all [1][dim]):
ADDR_DEC_X        = 0xC000   # [1][64]  = 64B    input hidden state
ADDR_DEC_LN1_OUT  = 0xC040   # [1][64]  = 64B    LayerNorm1 output
ADDR_DEC_Q        = 0xC080   # [1][64]  = 64B    query (new token)
ADDR_DEC_K_NEW    = 0xC0C0   # [1][64]  = 64B    key (new token)
ADDR_DEC_V_NEW    = 0xC100   # [1][64]  = 64B    value (new token)
ADDR_DEC_K_CACHE  = 0xC140   # [16][64] = 1024B  all cached keys
ADDR_DEC_V_CACHE  = 0xC540   # [16][64] = 1024B  all cached values
ADDR_DEC_S        = 0xC940   # [1][16]  = 16B    attention scores
ADDR_DEC_P        = 0xC950   # [1][16]  = 16B    attention probs
ADDR_DEC_ATTN     = 0xC960   # [1][64]  = 64B    attention output
ADDR_DEC_WO_OUT   = 0xC9A0   # [1][64]  = 64B    output projection
ADDR_DEC_X2       = 0xC9E0   # [1][64]  = 64B    residual 1 result
ADDR_DEC_LN2_OUT  = 0xCA20   # [1][64]  = 64B    LayerNorm2 output
ADDR_DEC_FFN1     = 0xCA60   # [1][256] = 256B   FFN up-projection
ADDR_DEC_FFN2     = 0xCB60   # [1][64]  = 64B    FFN down-projection
ADDR_DEC_X_OUT    = 0xCBA0   # [1][64]  = 64B    block output
# End: 0xCBE0 (52192B) -- fits easily in 64KB SRAM0

# SRAM1 layout is identical for prefill and decode:
#   S1_LN1_BETA  = 0x0000 [64]
#   S1_LN2_BETA  = 0x0040 [64]
#   S1_LN_F_BETA = 0x0080 [64]
#   S1_RESID     = 0x0100 [1][64] = 64B  (decode: only 1 row)

# =============================================================================
# KV_APPEND instruction encoding  (opcode = 8)
# =============================================================================
# src0  = SRAM0 source address (K or V row data)
# M     = layer_id
# K     = time_index (position in sequence)
# N     = vector_length (HIDDEN = 64)
# flags = bit 0: is_v (0 = K, 1 = V)
# imm   = head_id (0 for single-head)

# =============================================================================
# KV_READ instruction encoding  (opcode = 9)
# =============================================================================
# dst   = SRAM0 destination address
# M     = layer_id
# K     = time_len (number of vectors to read, starting at t=0)
# N     = vector_length (HIDDEN = 64)
# flags = bit 0: is_v (0 = K, 1 = V)
# imm   = head_id (0 for single-head)

# =============================================================================
# Opcodes (must match isa_pkg.sv)
# =============================================================================
OP_KV_APPEND = 8
OP_KV_READ   = 9
