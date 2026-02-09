// =============================================================================
// KV Cache Package - Instruction field mapping and constants
// =============================================================================
// Documents how OP_KV_APPEND and OP_KV_READ instructions encode their fields
// in the 128-bit microcode word.  The actual KV storage may be in hardware
// (kv_cache_bank.sv) or software (C++ testbench shim) depending on the
// integration target.
//
// KV cache layout: K[layer][head][time][head_dim]  (V identical)
//   - head_dim contiguous in memory for efficient QK^T streaming
//   - Address formula: {is_v, layer, head, time}
// =============================================================================
`ifndef KV_PKG_SV
`define KV_PKG_SV

package kv_pkg;

    // =========================================================================
    // Instruction field mapping for OP_KV_APPEND (opcode 8)
    // =========================================================================
    // Writes one vector (K or V) for a single time-step into the KV cache.
    //
    //   src0_base [47:32] = SRAM0 source address of the K or V row
    //   M         [79:64] = layer_id
    //   K        [111:96] = time_index (position in sequence)
    //   N         [95:80] = vector_length (HIDDEN)
    //   flags      [15:8] = bit 0: is_v (0 = K, 1 = V)
    //   imm      [127:112]= head_id

    // =========================================================================
    // Instruction field mapping for OP_KV_READ (opcode 9)
    // =========================================================================
    // Reads a contiguous range of vectors from the KV cache into SRAM0.
    //
    //   dst_base   [31:16] = SRAM0 destination address
    //   M          [79:64] = layer_id
    //   K         [111:96] = time_len (number of vectors to read, starting at t=0)
    //   N          [95:80] = vector_length (HIDDEN)
    //   flags       [15:8] = bit 0: is_v (0 = K, 1 = V)
    //   imm       [127:112]= head_id

    // =========================================================================
    // Flag bit positions
    // =========================================================================
    parameter int KV_FLAG_IS_V = 0;   // flags[0]: 0 = K cache, 1 = V cache

endpackage

`endif
