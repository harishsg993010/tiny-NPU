// =============================================================================
// tb_kv_cache_sim.cpp - KV Cache Correctness Test
// Verifies that the KV-cache decode path produces bit-exact results vs
// full-recompute. Reuses the same gpt2_block_top DUT as tb_demo_infer.cpp.
// =============================================================================

#include <verilated.h>
#include <verilated_vcd_c.h>
#include "Vgpt2_block_top.h"

#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <cassert>
#include <cstdlib>
#include <algorithm>
#include <string>

// =============================================================================
// Global simulation state
// =============================================================================
vluint64_t main_time = 0;
double sc_time_stamp() { return main_time; }

static Vgpt2_block_top* dut;
static VerilatedVcdC*    tfp;
static int               tc = 0;

// =============================================================================
// Model configuration (must match ddr_map.py)
// =============================================================================
static const int HIDDEN     = 64;
static const int HEAD_DIM   = 16;
static const int HEADS      = 4;
static const int FFN_DIM    = 256;
static const int N_LAYERS   = 4;
static const int VOCAB_SIZE = 256;
static const int MAX_SEQ    = 16;

// Quantization
static const int GEMM_SCALE = 1;
static const int GEMM_SHIFT = 9;
static const uint16_t GEMM_IMM     = 0x0901;
static const uint16_t GEMM_IMM_K64 = 0x0901;  // shift=9, scale=1 for K=64
static const uint16_t GEMM_IMM_K16 = 0x0701;  // shift=7, scale=1 for K=16
static const uint16_t GEMM_IMM_KS  = 0x0701;  // shift=7, scale=1 for K=S

// =============================================================================
// SRAM0 Memory layout for block execution (byte addresses)
// Must match ddr_map.py ADDR_* constants
// =============================================================================
static const uint16_t ADDR_WQ       = 0x0000;  // [64][64]  = 4096B
static const uint16_t ADDR_WK       = 0x1000;  // [64][64]  = 4096B
static const uint16_t ADDR_WV       = 0x2000;  // [64][64]  = 4096B
static const uint16_t ADDR_WO       = 0x3000;  // [64][64]  = 4096B
static const uint16_t ADDR_W1       = 0x4000;  // [64][256] = 16384B
static const uint16_t ADDR_W2       = 0x8000;  // [256][64] = 16384B
static const uint16_t ADDR_X        = 0xC000;  // [16][64]  = 1024B
static const uint16_t ADDR_LN1_OUT  = 0xC400;  // [16][64]  = 1024B
static const uint16_t ADDR_Q_H      = 0xC800;  // [S][16] per-head Q
static const uint16_t ADDR_K_H      = 0xC900;  // [S][16] per-head K
static const uint16_t ADDR_V_H      = 0xCA00;  // [S][16] per-head V
static const uint16_t ADDR_S        = 0xCB00;  // [S][S]  score
static const uint16_t ADDR_P        = 0xCC00;  // [S][S]  softmax
static const uint16_t ADDR_ATTN_H   = 0xCD00;  // [S][16] per-head context
static const uint16_t ADDR_ATTN     = 0xCE00;  // [S][64] concat all heads
static const uint16_t ADDR_WO_OUT   = 0xD200;  // [S][64]
static const uint16_t ADDR_X2       = 0xD600;  // [S][64]
static const uint16_t ADDR_LN2_OUT  = 0xDA00;  // [S][64]
static const uint16_t ADDR_FFN1     = 0xDE00;  // [S][256]
static const uint16_t ADDR_FFN2     = 0xEE00;  // [S][64]
static const uint16_t ADDR_X_OUT    = 0xF200;  // [S][64]

// Keep old names for backward compat where needed
static const uint16_t ADDR_Q        = ADDR_Q_H;
static const uint16_t ADDR_K        = ADDR_K_H;
static const uint16_t ADDR_V        = ADDR_V_H;

// SRAM0 addresses for lm_head phase (reuses SRAM0 from 0x0000)
static const uint16_t ADDR_LM_INPUT  = 0x0000;  // [1][64]  = 64B
static const uint16_t ADDR_LM_WEIGHT = 0x0100;  // [256][64] = 16384B
static const uint16_t ADDR_LM_OUTPUT = 0x4100;  // [1][256] = 256B

// SRAM1 addresses
static const uint16_t S1_LN1_BETA  = 0x0000;  // [64]
static const uint16_t S1_LN2_BETA  = 0x0040;  // [64]
static const uint16_t S1_LN_F_BETA = 0x0080;  // [64]
static const uint16_t S1_RESID     = 0x0100;  // [16][64] = 1024B

// weights.bin layout offsets (must match ddr_map.py)
static const int WTE_OFFSET    = 0;
static const int WTE_SIZE      = VOCAB_SIZE * HIDDEN;       // 16384
static const int WPE_OFFSET    = WTE_OFFSET + WTE_SIZE;     // 16384
static const int WPE_SIZE      = MAX_SEQ * HIDDEN;           // 1024
static const int BLOCKS_OFFSET = WPE_OFFSET + WPE_SIZE;     // 17408
static const int BLOCK_SIZE_B  = 64 + 4096 + 4096 + 4096 + 4096 + 64 + 16384 + 16384; // 49280
static const int BLK_LN1_BETA  = 0;
static const int BLK_WQ        = 64;
static const int BLK_WK        = 4160;
static const int BLK_WV        = 8256;
static const int BLK_WO        = 12352;
static const int BLK_LN2_BETA  = 16448;
static const int BLK_W1        = 16512;
static const int BLK_W2        = 32896;
static const int LN_F_OFFSET   = BLOCKS_OFFSET + N_LAYERS * BLOCK_SIZE_B; // 214528
static const int LN_F_SIZE     = 64;
static const int LM_HEAD_OFFSET = LN_F_OFFSET + LN_F_SIZE;  // 214592
static const int LM_HEAD_SIZE  = VOCAB_SIZE * HIDDEN;        // 16384
static const int WEIGHTS_TOTAL = LM_HEAD_OFFSET + LM_HEAD_SIZE; // 230976

// =============================================================================
// Decode-mode SRAM0 addresses (S=1, weights same region) - per-head layout
// Must match kv_map.py ADDR_DEC_* constants
// =============================================================================
static const uint16_t ADDR_DEC_X        = 0xC000;  // [1][64]  = 64B
static const uint16_t ADDR_DEC_LN1_OUT  = 0xC040;  // [1][64]  = 64B
static const uint16_t ADDR_DEC_Q_H      = 0xC080;  // [1][16]  = 16B per-head
static const uint16_t ADDR_DEC_K_NEW    = 0xC090;  // [1][16]  = 16B per-head
static const uint16_t ADDR_DEC_V_NEW    = 0xC0A0;  // [1][16]  = 16B per-head
static const uint16_t ADDR_DEC_K_CACHE  = 0xC0B0;  // [T][16]  = 256B per-head
static const uint16_t ADDR_DEC_V_CACHE  = 0xC1B0;  // [T][16]  = 256B per-head
static const uint16_t ADDR_DEC_S        = 0xC2B0;  // [1][T]
static const uint16_t ADDR_DEC_P        = 0xC2C0;  // [1][T]
static const uint16_t ADDR_DEC_ATTN_H   = 0xC2D0;  // [1][16] per-head
static const uint16_t ADDR_DEC_ATTN     = 0xC2E0;  // [1][64] concat
static const uint16_t ADDR_DEC_WO_OUT   = 0xC320;  // [1][64]
static const uint16_t ADDR_DEC_X2       = 0xC360;  // [1][64]
static const uint16_t ADDR_DEC_LN2_OUT  = 0xC3A0;  // [1][64]
static const uint16_t ADDR_DEC_FFN1     = 0xC3E0;  // [1][256]
static const uint16_t ADDR_DEC_FFN2     = 0xC4E0;  // [1][64]
static const uint16_t ADDR_DEC_X_OUT    = 0xC520;  // [1][64]

// Keep old name for backward compat
static const uint16_t ADDR_DEC_Q        = ADDR_DEC_Q_H;

// =============================================================================
// Opcodes and Flags
// =============================================================================
static const uint8_t OP_DMA_LOAD   = 1;
static const uint8_t OP_GEMM       = 3;
static const uint8_t OP_VEC        = 4;
static const uint8_t OP_SOFTMAX    = 5;
static const uint8_t OP_LAYERNORM  = 6;
static const uint8_t OP_GELU       = 7;
static const uint8_t OP_KV_APPEND  = 8;
static const uint8_t OP_KV_READ    = 9;
static const uint8_t OP_BARRIER    = 10;
static const uint8_t OP_END        = 255;

static const uint8_t FLAG_TRANSPOSE_B = 0x01;
static const uint8_t FLAG_COPY2D      = 0x04;
static const uint8_t FLAG_REQUANT     = 0x04;
static const uint8_t FLAG_CAUSAL_MASK = 0x10;

// =============================================================================
// Debug dump flag
// =============================================================================
static bool NPU_DUMP = false;

// =============================================================================
// Clock / Reset Helpers
// =============================================================================
void tick() {
    dut->clk = 0;
    dut->eval();
    if (tfp) tfp->dump(tc * 10);
    tc++;
    dut->clk = 1;
    dut->eval();
    if (tfp) tfp->dump(tc * 10 + 5);
    tc++;
    main_time = tc;
}

void reset_dut(int cycles = 10) {
    dut->rst_n = 0;
    for (int i = 0; i < cycles; i++) tick();
    dut->rst_n = 1;
    tick();
}

// =============================================================================
// SRAM Helpers
// =============================================================================
void sram0_write(uint16_t addr, uint8_t data) {
    dut->tb_sram0_wr_en   = 1;
    dut->tb_sram0_wr_addr = addr;
    dut->tb_sram0_wr_data = data;
    tick();
    dut->tb_sram0_wr_en = 0;
}

uint8_t sram0_read(uint16_t addr) {
    dut->tb_sram0_rd_en   = 1;
    dut->tb_sram0_rd_addr = addr;
    tick();
    dut->tb_sram0_rd_en = 0;
    return dut->tb_sram0_rd_data;
}

void sram1_write(uint16_t addr, uint8_t data) {
    dut->tb_sram1_wr_en   = 1;
    dut->tb_sram1_wr_addr = addr;
    dut->tb_sram1_wr_data = data;
    tick();
    dut->tb_sram1_wr_en = 0;
}

uint8_t sram1_read(uint16_t addr) {
    dut->tb_sram1_rd_en   = 1;
    dut->tb_sram1_rd_addr = addr;
    tick();
    dut->tb_sram1_rd_en = 0;
    return dut->tb_sram1_rd_data;
}

void ucode_write(uint16_t addr, uint64_t hi, uint64_t lo) {
    dut->uc_wr_en   = 1;
    dut->uc_wr_addr = addr;
    dut->uc_wr_data[0] = (uint32_t)(lo & 0xFFFFFFFF);
    dut->uc_wr_data[1] = (uint32_t)(lo >> 32);
    dut->uc_wr_data[2] = (uint32_t)(hi & 0xFFFFFFFF);
    dut->uc_wr_data[3] = (uint32_t)(hi >> 32);
    tick();
    dut->uc_wr_en = 0;
}

void encode_instr(uint8_t opcode, uint8_t flags, uint16_t dst, uint16_t src0,
                  uint16_t src1, uint16_t M, uint16_t N, uint16_t K,
                  uint16_t imm, uint64_t& hi, uint64_t& lo) {
    lo = (uint64_t)opcode
       | ((uint64_t)flags  << 8)
       | ((uint64_t)dst    << 16)
       | ((uint64_t)src0   << 32)
       | ((uint64_t)src1   << 48);
    hi = (uint64_t)M
       | ((uint64_t)N  << 16)
       | ((uint64_t)K  << 32)
       | ((uint64_t)imm << 48);
}

// =============================================================================
// Weight storage
// =============================================================================
static std::vector<int8_t> weights_buf;

// Per-block weight pointers (into weights_buf)
struct BlockWeights {
    const int8_t* ln1_beta;  // [HIDDEN]
    const int8_t* wq;        // [HIDDEN][HIDDEN]
    const int8_t* wk;        // [HIDDEN][HIDDEN]
    const int8_t* wv;        // [HIDDEN][HIDDEN]
    const int8_t* wo;        // [HIDDEN][HIDDEN]
    const int8_t* ln2_beta;  // [HIDDEN]
    const int8_t* w1;        // [HIDDEN][FFN_DIM]
    const int8_t* w2;        // [FFN_DIM][HIDDEN]
};

static BlockWeights block_weights[N_LAYERS];
static const int8_t* wte;       // [VOCAB_SIZE][HIDDEN]
static const int8_t* wpe;       // [MAX_SEQ][HIDDEN]
static const int8_t* ln_f_beta; // [HIDDEN]
static const int8_t* lm_head;   // [VOCAB_SIZE][HIDDEN]

bool load_weights(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) {
        std::cerr << "ERROR: Cannot open weights file: " << path << std::endl;
        return false;
    }
    f.seekg(0, std::ios::end);
    size_t sz = f.tellg();
    f.seekg(0, std::ios::beg);
    if ((int)sz != WEIGHTS_TOTAL) {
        std::cerr << "ERROR: weights.bin size " << sz << " != expected " << WEIGHTS_TOTAL << std::endl;
        return false;
    }
    weights_buf.resize(sz);
    f.read(reinterpret_cast<char*>(weights_buf.data()), sz);
    f.close();

    wte = weights_buf.data() + WTE_OFFSET;
    wpe = weights_buf.data() + WPE_OFFSET;

    for (int i = 0; i < N_LAYERS; i++) {
        int base = BLOCKS_OFFSET + i * BLOCK_SIZE_B;
        block_weights[i].ln1_beta = weights_buf.data() + base + BLK_LN1_BETA;
        block_weights[i].wq       = weights_buf.data() + base + BLK_WQ;
        block_weights[i].wk       = weights_buf.data() + base + BLK_WK;
        block_weights[i].wv       = weights_buf.data() + base + BLK_WV;
        block_weights[i].wo       = weights_buf.data() + base + BLK_WO;
        block_weights[i].ln2_beta = weights_buf.data() + base + BLK_LN2_BETA;
        block_weights[i].w1       = weights_buf.data() + base + BLK_W1;
        block_weights[i].w2       = weights_buf.data() + base + BLK_W2;
    }

    ln_f_beta = weights_buf.data() + LN_F_OFFSET;
    lm_head   = weights_buf.data() + LM_HEAD_OFFSET;

    std::cout << "Loaded weights: " << sz << " bytes from " << path << std::endl;
    return true;
}

// =============================================================================
// Embedding (host-side, not on NPU)
// =============================================================================
void compute_embeddings(const std::vector<int>& tokens, int8_t* emb) {
    int S = (int)tokens.size();
    for (int p = 0; p < S; p++) {
        int tok = tokens[p];
        for (int h = 0; h < HIDDEN; h++) {
            int val = (int)wte[tok * HIDDEN + h] + (int)wpe[p * HIDDEN + h];
            if (val > 127) val = 127;
            if (val < -128) val = -128;
            emb[p * HIDDEN + h] = (int8_t)val;
        }
    }
}

// =============================================================================
// DMA handler
// =============================================================================
void handle_dma_command() {
    uint16_t src = dut->dma_src;
    uint16_t dst = dut->dma_dst;
    uint16_t len = dut->dma_len;

    if (NPU_DUMP) {
        std::cout << "  DMA: src=0x" << std::hex << src
                  << " dst=0x" << dst << " len=" << std::dec << len << std::endl;
    }

    for (int i = 0; i < len; i++) {
        uint8_t val = sram0_read(src + i);
        sram1_write(dst + i, val);
    }
}

// =============================================================================
// Generate microcode for one transformer block (full-recompute, no KV cache)
// Multi-head attention: 4 heads, HEAD_DIM=16, HIDDEN=64
// Returns number of instructions written
// =============================================================================
int gen_block_microcode(int S, int ucode_addr) {
    uint64_t hi, lo;
    int addr = ucode_addr;

    // LayerNorm 1 (S rows)
    for (int i = 0; i < S; i++) {
        uint16_t src = ADDR_X + i * HIDDEN;
        uint16_t dst = ADDR_LN1_OUT + i * HIDDEN;
        encode_instr(OP_LAYERNORM, 0, dst, src, S1_LN1_BETA,
                     0, HIDDEN, 0, 0, hi, lo);
        ucode_write(addr++, hi, lo);
    }

    // BARRIER
    encode_instr(OP_BARRIER, 0, 0, 0, 0, 0, 0, 0, 0, hi, lo);
    ucode_write(addr++, hi, lo);

    // GEMM Q_full = LN1_out * Wq  [S,64]*[64,64] -> [S,64]
    // Output goes to ADDR_Q_H temporarily (will be sliced per-head)
    // Actually: compute full [S,64] Q, then per-head loop uses slices
    // We compute Q = LN1*WQ into a temp area, then per-head attention
    // But we only have ADDR_Q_H for [S,16]. We need the full [S,64] Q.
    // Strategy: compute full QKV [S,64] into ADDR_ATTN area (temp), then
    // per-head loop extracts slices. Actually, simpler: do QKV GEMMs
    // producing full [S,64], then for each head, slice out columns.
    // The hardware GEMM with head-blocked weights already produces per-head:
    //   WQ is stored as [HEADS][HIDDEN][HEAD_DIM] = head-blocked
    //   So GEMM LN1 * WQ_h gives [S][HEAD_DIM] directly.

    // Per-head attention loop
    for (int h = 0; h < HEADS; h++) {
        uint16_t wq_h = ADDR_WQ + h * HIDDEN * HEAD_DIM;
        uint16_t wk_h = ADDR_WK + h * HIDDEN * HEAD_DIM;
        uint16_t wv_h = ADDR_WV + h * HIDDEN * HEAD_DIM;

        // GEMM Q_h = LN1_out * WQ_h  [S,64]*[64,16] -> [S,16]
        encode_instr(OP_GEMM, FLAG_REQUANT, ADDR_Q_H, ADDR_LN1_OUT, wq_h,
                     S, HEAD_DIM, HIDDEN, GEMM_IMM_K64, hi, lo);
        ucode_write(addr++, hi, lo);
        encode_instr(OP_BARRIER, 0, 0, 0, 0, 0, 0, 0, 0, hi, lo);
        ucode_write(addr++, hi, lo);

        // GEMM K_h = LN1_out * WK_h  [S,64]*[64,16] -> [S,16]
        encode_instr(OP_GEMM, FLAG_REQUANT, ADDR_K_H, ADDR_LN1_OUT, wk_h,
                     S, HEAD_DIM, HIDDEN, GEMM_IMM_K64, hi, lo);
        ucode_write(addr++, hi, lo);
        encode_instr(OP_BARRIER, 0, 0, 0, 0, 0, 0, 0, 0, hi, lo);
        ucode_write(addr++, hi, lo);

        // GEMM V_h = LN1_out * WV_h  [S,64]*[64,16] -> [S,16]
        encode_instr(OP_GEMM, FLAG_REQUANT, ADDR_V_H, ADDR_LN1_OUT, wv_h,
                     S, HEAD_DIM, HIDDEN, GEMM_IMM_K64, hi, lo);
        ucode_write(addr++, hi, lo);
        encode_instr(OP_BARRIER, 0, 0, 0, 0, 0, 0, 0, 0, hi, lo);
        ucode_write(addr++, hi, lo);

        // GEMM S = Q_h * K_h^T  [S,16]*[S,16]^T -> [S,S]
        encode_instr(OP_GEMM, FLAG_TRANSPOSE_B | FLAG_REQUANT, ADDR_S, ADDR_Q_H, ADDR_K_H,
                     S, S, HEAD_DIM, GEMM_IMM_K16, hi, lo);
        ucode_write(addr++, hi, lo);
        encode_instr(OP_BARRIER, 0, 0, 0, 0, 0, 0, 0, 0, hi, lo);
        ucode_write(addr++, hi, lo);

        // Softmax rows (S rows with causal mask)
        for (int i = 0; i < S; i++) {
            uint16_t src = ADDR_S + i * S;
            uint16_t dst = ADDR_P + i * S;
            uint16_t sm_imm = ((uint16_t)S) << 8;
            encode_instr(OP_SOFTMAX, FLAG_CAUSAL_MASK, dst, src, 0,
                         0, S, i, sm_imm, hi, lo);
            ucode_write(addr++, hi, lo);
        }
        encode_instr(OP_BARRIER, 0, 0, 0, 0, 0, 0, 0, 0, hi, lo);
        ucode_write(addr++, hi, lo);

        // GEMM ATTN_h = P * V_h  [S,S]*[S,16] -> [S,16]
        encode_instr(OP_GEMM, FLAG_REQUANT, ADDR_ATTN_H, ADDR_P, ADDR_V_H,
                     S, HEAD_DIM, S, GEMM_IMM_KS, hi, lo);
        ucode_write(addr++, hi, lo);
        encode_instr(OP_BARRIER, 0, 0, 0, 0, 0, 0, 0, 0, hi, lo);
        ucode_write(addr++, hi, lo);

        // COPY2D: scatter ATTN_h [S,16] into ATTN [S,64] at column h*HEAD_DIM
        // VEC with FLAG_COPY2D: src0=ADDR_ATTN_H, dst=ADDR_ATTN+h*HEAD_DIM
        // length=HEAD_DIM (cols), M=S (rows), K=HEAD_DIM (src stride), imm=HIDDEN (dst stride)
        encode_instr(OP_VEC, FLAG_COPY2D,
                     ADDR_ATTN + h * HEAD_DIM, ADDR_ATTN_H, 0,
                     S, HEAD_DIM, HEAD_DIM, HIDDEN, hi, lo);
        ucode_write(addr++, hi, lo);
        encode_instr(OP_BARRIER, 0, 0, 0, 0, 0, 0, 0, 0, hi, lo);
        ucode_write(addr++, hi, lo);
    }

    // GEMM WO_out = ATTN * Wo  [S,64]*[64,64] -> [S,64]
    encode_instr(OP_GEMM, FLAG_REQUANT, ADDR_WO_OUT, ADDR_ATTN, ADDR_WO,
                 S, HIDDEN, HIDDEN, GEMM_IMM_K64, hi, lo);
    ucode_write(addr++, hi, lo);

    // BARRIER
    encode_instr(OP_BARRIER, 0, 0, 0, 0, 0, 0, 0, 0, hi, lo);
    ucode_write(addr++, hi, lo);

    // VEC ADD: X2 = WO_out + X (src1 from SRAM1 residual)
    encode_instr(OP_VEC, 0x00, ADDR_X2, ADDR_WO_OUT, S1_RESID,
                 0, S * HIDDEN, 0, 0, hi, lo);
    ucode_write(addr++, hi, lo);

    // BARRIER
    encode_instr(OP_BARRIER, 0, 0, 0, 0, 0, 0, 0, 0, hi, lo);
    ucode_write(addr++, hi, lo);

    // DMA_LOAD: copy X2 from SRAM0 to SRAM1 for second residual
    encode_instr(OP_DMA_LOAD, 0, S1_RESID, ADDR_X2, 0,
                 0, S * HIDDEN, 0, 0, hi, lo);
    ucode_write(addr++, hi, lo);

    // BARRIER
    encode_instr(OP_BARRIER, 0, 0, 0, 0, 0, 0, 0, 0, hi, lo);
    ucode_write(addr++, hi, lo);

    // LayerNorm 2 (S rows)
    for (int i = 0; i < S; i++) {
        uint16_t src = ADDR_X2 + i * HIDDEN;
        uint16_t dst = ADDR_LN2_OUT + i * HIDDEN;
        encode_instr(OP_LAYERNORM, 0, dst, src, S1_LN2_BETA,
                     0, HIDDEN, 0, 0, hi, lo);
        ucode_write(addr++, hi, lo);
    }

    // BARRIER
    encode_instr(OP_BARRIER, 0, 0, 0, 0, 0, 0, 0, 0, hi, lo);
    ucode_write(addr++, hi, lo);

    // GEMM FFN1 = LN2_out * W1
    encode_instr(OP_GEMM, FLAG_REQUANT, ADDR_FFN1, ADDR_LN2_OUT, ADDR_W1,
                 S, FFN_DIM, HIDDEN, GEMM_IMM_K64, hi, lo);
    ucode_write(addr++, hi, lo);

    // BARRIER
    encode_instr(OP_BARRIER, 0, 0, 0, 0, 0, 0, 0, 0, hi, lo);
    ucode_write(addr++, hi, lo);

    // GELU in-place on FFN1
    encode_instr(OP_GELU, 0, ADDR_FFN1, ADDR_FFN1, 0,
                 0, S * FFN_DIM, 0, 0, hi, lo);
    ucode_write(addr++, hi, lo);

    // BARRIER
    encode_instr(OP_BARRIER, 0, 0, 0, 0, 0, 0, 0, 0, hi, lo);
    ucode_write(addr++, hi, lo);

    // GEMM FFN2 = GELU_out * W2
    encode_instr(OP_GEMM, FLAG_REQUANT, ADDR_FFN2, ADDR_FFN1, ADDR_W2,
                 S, HIDDEN, FFN_DIM, GEMM_IMM, hi, lo);
    ucode_write(addr++, hi, lo);

    // BARRIER
    encode_instr(OP_BARRIER, 0, 0, 0, 0, 0, 0, 0, 0, hi, lo);
    ucode_write(addr++, hi, lo);

    // VEC ADD: X_OUT = FFN2 + X2 (src1 from SRAM1 residual)
    encode_instr(OP_VEC, 0x00, ADDR_X_OUT, ADDR_FFN2, S1_RESID,
                 0, S * HIDDEN, 0, 0, hi, lo);
    ucode_write(addr++, hi, lo);

    // BARRIER
    encode_instr(OP_BARRIER, 0, 0, 0, 0, 0, 0, 0, 0, hi, lo);
    ucode_write(addr++, hi, lo);

    // END
    encode_instr(OP_END, 0, 0, 0, 0, 0, 0, 0, 0, hi, lo);
    ucode_write(addr++, hi, lo);

    return addr - ucode_addr;
}

// =============================================================================
// Generate microcode for final LayerNorm on last token only
// =============================================================================
int gen_ln_f_microcode(int ucode_addr) {
    uint64_t hi, lo;
    int addr = ucode_addr;

    // LN_F on last-token hidden state at ADDR_X (we'll place it there)
    // output to same location
    encode_instr(OP_LAYERNORM, 0, ADDR_LM_INPUT, ADDR_LM_INPUT, S1_LN_F_BETA,
                 0, HIDDEN, 0, 0, hi, lo);
    ucode_write(addr++, hi, lo);

    // BARRIER
    encode_instr(OP_BARRIER, 0, 0, 0, 0, 0, 0, 0, 0, hi, lo);
    ucode_write(addr++, hi, lo);

    // END
    encode_instr(OP_END, 0, 0, 0, 0, 0, 0, 0, 0, hi, lo);
    ucode_write(addr++, hi, lo);

    return addr - ucode_addr;
}

// =============================================================================
// Generate microcode for lm_head: logits = hidden * lm_head^T
// M=1, N=256, K=64
// =============================================================================
int gen_lm_head_microcode(int ucode_addr) {
    uint64_t hi, lo;
    int addr = ucode_addr;

    // GEMM: logits = input * lm_head^T
    // input at ADDR_LM_INPUT [1][64], weight at ADDR_LM_WEIGHT [256][64]
    // output at ADDR_LM_OUTPUT [1][256]
    encode_instr(OP_GEMM, FLAG_TRANSPOSE_B | FLAG_REQUANT,
                 ADDR_LM_OUTPUT, ADDR_LM_INPUT, ADDR_LM_WEIGHT,
                 1, VOCAB_SIZE, HIDDEN, GEMM_IMM_K64, hi, lo);
    ucode_write(addr++, hi, lo);

    // BARRIER
    encode_instr(OP_BARRIER, 0, 0, 0, 0, 0, 0, 0, 0, hi, lo);
    ucode_write(addr++, hi, lo);

    // END
    encode_instr(OP_END, 0, 0, 0, 0, 0, 0, 0, 0, hi, lo);
    ucode_write(addr++, hi, lo);

    return addr - ucode_addr;
}

// =============================================================================
// Generate PREFILL block microcode: same as gen_block_microcode but adds
// KV_APPEND instructions after K and V GEMMs to populate the KV cache.
// Multi-head: per-head QKV GEMMs, score, softmax, context, COPY2D scatter,
// and KV_APPEND per head.
// =============================================================================
int gen_prefill_block_microcode(int S, int blk_idx, int ucode_addr) {
    uint64_t hi, lo;
    int addr = ucode_addr;

    // LayerNorm 1 (S rows)
    for (int i = 0; i < S; i++) {
        uint16_t src = ADDR_X + i * HIDDEN;
        uint16_t dst = ADDR_LN1_OUT + i * HIDDEN;
        encode_instr(OP_LAYERNORM, 0, dst, src, S1_LN1_BETA,
                     0, HIDDEN, 0, 0, hi, lo);
        ucode_write(addr++, hi, lo);
    }
    encode_instr(OP_BARRIER, 0, 0, 0, 0, 0, 0, 0, 0, hi, lo);
    ucode_write(addr++, hi, lo);

    // Per-head attention loop with KV_APPEND
    for (int h = 0; h < HEADS; h++) {
        uint16_t wq_h = ADDR_WQ + h * HIDDEN * HEAD_DIM;
        uint16_t wk_h = ADDR_WK + h * HIDDEN * HEAD_DIM;
        uint16_t wv_h = ADDR_WV + h * HIDDEN * HEAD_DIM;

        // GEMM Q_h = LN1_out * WQ_h  [S,64]*[64,16] -> [S,16]
        encode_instr(OP_GEMM, FLAG_REQUANT, ADDR_Q_H, ADDR_LN1_OUT, wq_h,
                     S, HEAD_DIM, HIDDEN, GEMM_IMM_K64, hi, lo);
        ucode_write(addr++, hi, lo);
        encode_instr(OP_BARRIER, 0, 0, 0, 0, 0, 0, 0, 0, hi, lo);
        ucode_write(addr++, hi, lo);

        // GEMM K_h = LN1_out * WK_h  [S,64]*[64,16] -> [S,16]
        encode_instr(OP_GEMM, FLAG_REQUANT, ADDR_K_H, ADDR_LN1_OUT, wk_h,
                     S, HEAD_DIM, HIDDEN, GEMM_IMM_K64, hi, lo);
        ucode_write(addr++, hi, lo);
        encode_instr(OP_BARRIER, 0, 0, 0, 0, 0, 0, 0, 0, hi, lo);
        ucode_write(addr++, hi, lo);

        // GEMM V_h = LN1_out * WV_h  [S,64]*[64,16] -> [S,16]
        encode_instr(OP_GEMM, FLAG_REQUANT, ADDR_V_H, ADDR_LN1_OUT, wv_h,
                     S, HEAD_DIM, HIDDEN, GEMM_IMM_K64, hi, lo);
        ucode_write(addr++, hi, lo);
        encode_instr(OP_BARRIER, 0, 0, 0, 0, 0, 0, 0, 0, hi, lo);
        ucode_write(addr++, hi, lo);

        // KV_APPEND: store K_h[0..S-1] into KV cache for this head
        for (int i = 0; i < S; i++) {
            // KV_APPEND K: src0=ADDR_K_H+i*HEAD_DIM, M=layer, K=time_i, N=HEAD_DIM, flags=0(K), imm=head
            encode_instr(OP_KV_APPEND, 0x00, 0, ADDR_K_H + i * HEAD_DIM, 0,
                         blk_idx, HEAD_DIM, i, h, hi, lo);
            ucode_write(addr++, hi, lo);
        }
        encode_instr(OP_BARRIER, 0, 0, 0, 0, 0, 0, 0, 0, hi, lo);
        ucode_write(addr++, hi, lo);

        // KV_APPEND: store V_h[0..S-1] into KV cache for this head
        for (int i = 0; i < S; i++) {
            // KV_APPEND V: flags=0x01(V), imm=head
            encode_instr(OP_KV_APPEND, 0x01, 0, ADDR_V_H + i * HEAD_DIM, 0,
                         blk_idx, HEAD_DIM, i, h, hi, lo);
            ucode_write(addr++, hi, lo);
        }
        encode_instr(OP_BARRIER, 0, 0, 0, 0, 0, 0, 0, 0, hi, lo);
        ucode_write(addr++, hi, lo);

        // GEMM S = Q_h * K_h^T  [S,16]*[S,16]^T -> [S,S]
        encode_instr(OP_GEMM, FLAG_TRANSPOSE_B | FLAG_REQUANT, ADDR_S, ADDR_Q_H, ADDR_K_H,
                     S, S, HEAD_DIM, GEMM_IMM_K16, hi, lo);
        ucode_write(addr++, hi, lo);
        encode_instr(OP_BARRIER, 0, 0, 0, 0, 0, 0, 0, 0, hi, lo);
        ucode_write(addr++, hi, lo);

        // Softmax rows (S rows with causal mask)
        for (int i = 0; i < S; i++) {
            uint16_t src = ADDR_S + i * S;
            uint16_t dst = ADDR_P + i * S;
            uint16_t sm_imm = ((uint16_t)S) << 8;
            encode_instr(OP_SOFTMAX, FLAG_CAUSAL_MASK, dst, src, 0,
                         0, S, i, sm_imm, hi, lo);
            ucode_write(addr++, hi, lo);
        }
        encode_instr(OP_BARRIER, 0, 0, 0, 0, 0, 0, 0, 0, hi, lo);
        ucode_write(addr++, hi, lo);

        // GEMM ATTN_h = P * V_h  [S,S]*[S,16] -> [S,16]
        encode_instr(OP_GEMM, FLAG_REQUANT, ADDR_ATTN_H, ADDR_P, ADDR_V_H,
                     S, HEAD_DIM, S, GEMM_IMM_KS, hi, lo);
        ucode_write(addr++, hi, lo);
        encode_instr(OP_BARRIER, 0, 0, 0, 0, 0, 0, 0, 0, hi, lo);
        ucode_write(addr++, hi, lo);

        // COPY2D: scatter ATTN_h [S,16] into ATTN [S,64] at column h*HEAD_DIM
        encode_instr(OP_VEC, FLAG_COPY2D,
                     ADDR_ATTN + h * HEAD_DIM, ADDR_ATTN_H, 0,
                     S, HEAD_DIM, HEAD_DIM, HIDDEN, hi, lo);
        ucode_write(addr++, hi, lo);
        encode_instr(OP_BARRIER, 0, 0, 0, 0, 0, 0, 0, 0, hi, lo);
        ucode_write(addr++, hi, lo);
    }

    // GEMM WO_out = ATTN * Wo  [S,64]*[64,64] -> [S,64]
    encode_instr(OP_GEMM, FLAG_REQUANT, ADDR_WO_OUT, ADDR_ATTN, ADDR_WO,
                 S, HIDDEN, HIDDEN, GEMM_IMM_K64, hi, lo);
    ucode_write(addr++, hi, lo);
    encode_instr(OP_BARRIER, 0, 0, 0, 0, 0, 0, 0, 0, hi, lo);
    ucode_write(addr++, hi, lo);

    // VEC ADD: X2 = WO_out + X (SRAM1 residual)
    encode_instr(OP_VEC, 0x00, ADDR_X2, ADDR_WO_OUT, S1_RESID,
                 0, S * HIDDEN, 0, 0, hi, lo);
    ucode_write(addr++, hi, lo);
    encode_instr(OP_BARRIER, 0, 0, 0, 0, 0, 0, 0, 0, hi, lo);
    ucode_write(addr++, hi, lo);

    // DMA_LOAD: copy X2 to SRAM1 residual
    encode_instr(OP_DMA_LOAD, 0, S1_RESID, ADDR_X2, 0,
                 0, S * HIDDEN, 0, 0, hi, lo);
    ucode_write(addr++, hi, lo);
    encode_instr(OP_BARRIER, 0, 0, 0, 0, 0, 0, 0, 0, hi, lo);
    ucode_write(addr++, hi, lo);

    // LayerNorm 2 (S rows)
    for (int i = 0; i < S; i++) {
        uint16_t src = ADDR_X2 + i * HIDDEN;
        uint16_t dst = ADDR_LN2_OUT + i * HIDDEN;
        encode_instr(OP_LAYERNORM, 0, dst, src, S1_LN2_BETA,
                     0, HIDDEN, 0, 0, hi, lo);
        ucode_write(addr++, hi, lo);
    }
    encode_instr(OP_BARRIER, 0, 0, 0, 0, 0, 0, 0, 0, hi, lo);
    ucode_write(addr++, hi, lo);

    // GEMM FFN1 = LN2_out * W1
    encode_instr(OP_GEMM, FLAG_REQUANT, ADDR_FFN1, ADDR_LN2_OUT, ADDR_W1,
                 S, FFN_DIM, HIDDEN, GEMM_IMM_K64, hi, lo);
    ucode_write(addr++, hi, lo);
    encode_instr(OP_BARRIER, 0, 0, 0, 0, 0, 0, 0, 0, hi, lo);
    ucode_write(addr++, hi, lo);

    // GELU
    encode_instr(OP_GELU, 0, ADDR_FFN1, ADDR_FFN1, 0,
                 0, S * FFN_DIM, 0, 0, hi, lo);
    ucode_write(addr++, hi, lo);
    encode_instr(OP_BARRIER, 0, 0, 0, 0, 0, 0, 0, 0, hi, lo);
    ucode_write(addr++, hi, lo);

    // GEMM FFN2 = GELU_out * W2
    encode_instr(OP_GEMM, FLAG_REQUANT, ADDR_FFN2, ADDR_FFN1, ADDR_W2,
                 S, HIDDEN, FFN_DIM, GEMM_IMM, hi, lo);
    ucode_write(addr++, hi, lo);
    encode_instr(OP_BARRIER, 0, 0, 0, 0, 0, 0, 0, 0, hi, lo);
    ucode_write(addr++, hi, lo);

    // VEC ADD: X_OUT = FFN2 + X2 (SRAM1 residual)
    encode_instr(OP_VEC, 0x00, ADDR_X_OUT, ADDR_FFN2, S1_RESID,
                 0, S * HIDDEN, 0, 0, hi, lo);
    ucode_write(addr++, hi, lo);
    encode_instr(OP_BARRIER, 0, 0, 0, 0, 0, 0, 0, 0, hi, lo);
    ucode_write(addr++, hi, lo);

    // END
    encode_instr(OP_END, 0, 0, 0, 0, 0, 0, 0, 0, hi, lo);
    ucode_write(addr++, hi, lo);

    return addr - ucode_addr;
}

// =============================================================================
// Generate DECODE block microcode: single-token attention using KV cache
// Multi-head: per-head decode with KV cache
// T = total sequence positions including new token (e.g. prompt_len + step)
// blk_idx = transformer block index (layer)
// =============================================================================
int gen_decode_block_microcode(int T, int blk_idx, int ucode_addr) {
    uint64_t hi, lo;
    int addr = ucode_addr;
    int T_len = T + 1;  // total positions including new token at index T

    // LayerNorm 1 (single row)
    encode_instr(OP_LAYERNORM, 0, ADDR_DEC_LN1_OUT, ADDR_DEC_X, S1_LN1_BETA,
                 0, HIDDEN, 0, 0, hi, lo);
    ucode_write(addr++, hi, lo);
    encode_instr(OP_BARRIER, 0, 0, 0, 0, 0, 0, 0, 0, hi, lo);
    ucode_write(addr++, hi, lo);

    // Per-head attention loop for decode
    for (int h = 0; h < HEADS; h++) {
        uint16_t wq_h = ADDR_WQ + h * HIDDEN * HEAD_DIM;
        uint16_t wk_h = ADDR_WK + h * HIDDEN * HEAD_DIM;
        uint16_t wv_h = ADDR_WV + h * HIDDEN * HEAD_DIM;

        // GEMM Q_h = LN1_out * WQ_h  [1,64]*[64,16] -> [1,16]
        encode_instr(OP_GEMM, FLAG_REQUANT, ADDR_DEC_Q_H, ADDR_DEC_LN1_OUT, wq_h,
                     1, HEAD_DIM, HIDDEN, GEMM_IMM_K64, hi, lo);
        ucode_write(addr++, hi, lo);
        encode_instr(OP_BARRIER, 0, 0, 0, 0, 0, 0, 0, 0, hi, lo);
        ucode_write(addr++, hi, lo);

        // GEMM K_new_h = LN1_out * WK_h  [1,64]*[64,16] -> [1,16]
        encode_instr(OP_GEMM, FLAG_REQUANT, ADDR_DEC_K_NEW, ADDR_DEC_LN1_OUT, wk_h,
                     1, HEAD_DIM, HIDDEN, GEMM_IMM_K64, hi, lo);
        ucode_write(addr++, hi, lo);
        encode_instr(OP_BARRIER, 0, 0, 0, 0, 0, 0, 0, 0, hi, lo);
        ucode_write(addr++, hi, lo);

        // GEMM V_new_h = LN1_out * WV_h  [1,64]*[64,16] -> [1,16]
        encode_instr(OP_GEMM, FLAG_REQUANT, ADDR_DEC_V_NEW, ADDR_DEC_LN1_OUT, wv_h,
                     1, HEAD_DIM, HIDDEN, GEMM_IMM_K64, hi, lo);
        ucode_write(addr++, hi, lo);
        encode_instr(OP_BARRIER, 0, 0, 0, 0, 0, 0, 0, 0, hi, lo);
        ucode_write(addr++, hi, lo);

        // KV_APPEND K_new_h at position T, head h
        encode_instr(OP_KV_APPEND, 0x00, 0, ADDR_DEC_K_NEW, 0,
                     blk_idx, HEAD_DIM, T, h, hi, lo);
        ucode_write(addr++, hi, lo);

        // KV_APPEND V_new_h at position T, head h
        encode_instr(OP_KV_APPEND, 0x01, 0, ADDR_DEC_V_NEW, 0,
                     blk_idx, HEAD_DIM, T, h, hi, lo);
        ucode_write(addr++, hi, lo);
        encode_instr(OP_BARRIER, 0, 0, 0, 0, 0, 0, 0, 0, hi, lo);
        ucode_write(addr++, hi, lo);

        // KV_READ: load K_cache_h[0..T] into SRAM0
        encode_instr(OP_KV_READ, 0x00, ADDR_DEC_K_CACHE, 0, 0,
                     blk_idx, HEAD_DIM, T_len, h, hi, lo);
        ucode_write(addr++, hi, lo);
        encode_instr(OP_BARRIER, 0, 0, 0, 0, 0, 0, 0, 0, hi, lo);
        ucode_write(addr++, hi, lo);

        // KV_READ: load V_cache_h[0..T] into SRAM0
        encode_instr(OP_KV_READ, 0x01, ADDR_DEC_V_CACHE, 0, 0,
                     blk_idx, HEAD_DIM, T_len, h, hi, lo);
        ucode_write(addr++, hi, lo);
        encode_instr(OP_BARRIER, 0, 0, 0, 0, 0, 0, 0, 0, hi, lo);
        ucode_write(addr++, hi, lo);

        // GEMM S = Q_h * K_cache_h^T  [1,16]*[T_len,16]^T -> [1,T_len]
        encode_instr(OP_GEMM, FLAG_TRANSPOSE_B | FLAG_REQUANT,
                     ADDR_DEC_S, ADDR_DEC_Q_H, ADDR_DEC_K_CACHE,
                     1, T_len, HEAD_DIM, GEMM_IMM_K16, hi, lo);
        ucode_write(addr++, hi, lo);
        encode_instr(OP_BARRIER, 0, 0, 0, 0, 0, 0, 0, 0, hi, lo);
        ucode_write(addr++, hi, lo);

        // Softmax (single row, length T_len, causal_limit=T)
        uint16_t sm_imm = ((uint16_t)T_len) << 8;
        encode_instr(OP_SOFTMAX, FLAG_CAUSAL_MASK, ADDR_DEC_P, ADDR_DEC_S, 0,
                     0, T_len, T, sm_imm, hi, lo);
        ucode_write(addr++, hi, lo);
        encode_instr(OP_BARRIER, 0, 0, 0, 0, 0, 0, 0, 0, hi, lo);
        ucode_write(addr++, hi, lo);

        // GEMM ATTN_h = P * V_cache_h  [1,T_len]*[T_len,16] -> [1,16]
        encode_instr(OP_GEMM, FLAG_REQUANT, ADDR_DEC_ATTN_H, ADDR_DEC_P, ADDR_DEC_V_CACHE,
                     1, HEAD_DIM, T_len, GEMM_IMM_KS, hi, lo);
        ucode_write(addr++, hi, lo);
        encode_instr(OP_BARRIER, 0, 0, 0, 0, 0, 0, 0, 0, hi, lo);
        ucode_write(addr++, hi, lo);

        // COPY2D: scatter ATTN_h [1,16] into ATTN [1,64] at column h*HEAD_DIM
        encode_instr(OP_VEC, FLAG_COPY2D,
                     ADDR_DEC_ATTN + h * HEAD_DIM, ADDR_DEC_ATTN_H, 0,
                     1, HEAD_DIM, HEAD_DIM, HIDDEN, hi, lo);
        ucode_write(addr++, hi, lo);
        encode_instr(OP_BARRIER, 0, 0, 0, 0, 0, 0, 0, 0, hi, lo);
        ucode_write(addr++, hi, lo);
    }

    // GEMM WO_out = ATTN * Wo  [1,64]*[64,64] -> [1,64]
    encode_instr(OP_GEMM, FLAG_REQUANT, ADDR_DEC_WO_OUT, ADDR_DEC_ATTN, ADDR_WO,
                 1, HIDDEN, HIDDEN, GEMM_IMM_K64, hi, lo);
    ucode_write(addr++, hi, lo);
    encode_instr(OP_BARRIER, 0, 0, 0, 0, 0, 0, 0, 0, hi, lo);
    ucode_write(addr++, hi, lo);

    // VEC ADD: X2 = WO_out + X (SRAM1 residual)
    encode_instr(OP_VEC, 0x00, ADDR_DEC_X2, ADDR_DEC_WO_OUT, S1_RESID,
                 0, HIDDEN, 0, 0, hi, lo);
    ucode_write(addr++, hi, lo);
    encode_instr(OP_BARRIER, 0, 0, 0, 0, 0, 0, 0, 0, hi, lo);
    ucode_write(addr++, hi, lo);

    // DMA_LOAD: copy X2 to SRAM1 residual
    encode_instr(OP_DMA_LOAD, 0, S1_RESID, ADDR_DEC_X2, 0,
                 0, HIDDEN, 0, 0, hi, lo);
    ucode_write(addr++, hi, lo);
    encode_instr(OP_BARRIER, 0, 0, 0, 0, 0, 0, 0, 0, hi, lo);
    ucode_write(addr++, hi, lo);

    // LayerNorm 2 (single row)
    encode_instr(OP_LAYERNORM, 0, ADDR_DEC_LN2_OUT, ADDR_DEC_X2, S1_LN2_BETA,
                 0, HIDDEN, 0, 0, hi, lo);
    ucode_write(addr++, hi, lo);
    encode_instr(OP_BARRIER, 0, 0, 0, 0, 0, 0, 0, 0, hi, lo);
    ucode_write(addr++, hi, lo);

    // GEMM FFN1 = LN2_out * W1  (M=1, N=FFN_DIM, K=HIDDEN)
    encode_instr(OP_GEMM, FLAG_REQUANT, ADDR_DEC_FFN1, ADDR_DEC_LN2_OUT, ADDR_W1,
                 1, FFN_DIM, HIDDEN, GEMM_IMM_K64, hi, lo);
    ucode_write(addr++, hi, lo);
    encode_instr(OP_BARRIER, 0, 0, 0, 0, 0, 0, 0, 0, hi, lo);
    ucode_write(addr++, hi, lo);

    // GELU
    encode_instr(OP_GELU, 0, ADDR_DEC_FFN1, ADDR_DEC_FFN1, 0,
                 0, FFN_DIM, 0, 0, hi, lo);
    ucode_write(addr++, hi, lo);
    encode_instr(OP_BARRIER, 0, 0, 0, 0, 0, 0, 0, 0, hi, lo);
    ucode_write(addr++, hi, lo);

    // GEMM FFN2 = GELU_out * W2  (M=1, N=HIDDEN, K=FFN_DIM)
    encode_instr(OP_GEMM, FLAG_REQUANT, ADDR_DEC_FFN2, ADDR_DEC_FFN1, ADDR_W2,
                 1, HIDDEN, FFN_DIM, GEMM_IMM, hi, lo);
    ucode_write(addr++, hi, lo);
    encode_instr(OP_BARRIER, 0, 0, 0, 0, 0, 0, 0, 0, hi, lo);
    ucode_write(addr++, hi, lo);

    // VEC ADD: X_OUT = FFN2 + X2 (SRAM1 residual)
    encode_instr(OP_VEC, 0x00, ADDR_DEC_X_OUT, ADDR_DEC_FFN2, S1_RESID,
                 0, HIDDEN, 0, 0, hi, lo);
    ucode_write(addr++, hi, lo);
    encode_instr(OP_BARRIER, 0, 0, 0, 0, 0, 0, 0, 0, hi, lo);
    ucode_write(addr++, hi, lo);

    // END
    encode_instr(OP_END, 0, 0, 0, 0, 0, 0, 0, 0, hi, lo);
    ucode_write(addr++, hi, lo);

    return addr - ucode_addr;
}

// =============================================================================
// Run NPU program until program_end, handling DMA + KV shims
// Returns cycle count, or -1 on timeout
// =============================================================================
int run_until_done(int max_cycles = 2000000) {
    int cycle = 0;
    while (cycle < max_cycles) {
        tick();
        cycle++;

        // DMA interception
        if (dut->dma_cmd_captured) {
            handle_dma_command();
            dut->dma_done_pulse = 1;
            tick(); cycle++;
            dut->dma_done_pulse = 0;
        }

        if (dut->program_end) {
            for (int i = 0; i < 10; i++) tick();
            return cycle;
        }
    }
    return -1;
}

// =============================================================================
// Load block weights into SRAM0 + SRAM1
// WQ/WK/WV are loaded as flat bytes (already head-blocked in weights.bin)
// =============================================================================
void load_block_to_srams(int blk_idx, const int8_t* x_data, int S) {
    const BlockWeights& bw = block_weights[blk_idx];

    // SRAM0: weights - load wq/wk/wv as flat bytes (head-blocked layout)
    for (int i = 0; i < HIDDEN * HIDDEN; i++) {
        sram0_write(ADDR_WQ + i, (uint8_t)bw.wq[i]);
        sram0_write(ADDR_WK + i, (uint8_t)bw.wk[i]);
        sram0_write(ADDR_WV + i, (uint8_t)bw.wv[i]);
    }

    for (int i = 0; i < HIDDEN; i++)
        for (int j = 0; j < HIDDEN; j++)
            sram0_write(ADDR_WO + i * HIDDEN + j, (uint8_t)bw.wo[i * HIDDEN + j]);

    for (int i = 0; i < HIDDEN; i++)
        for (int j = 0; j < FFN_DIM; j++)
            sram0_write(ADDR_W1 + i * FFN_DIM + j, (uint8_t)bw.w1[i * FFN_DIM + j]);

    for (int i = 0; i < FFN_DIM; i++)
        for (int j = 0; j < HIDDEN; j++)
            sram0_write(ADDR_W2 + i * HIDDEN + j, (uint8_t)bw.w2[i * HIDDEN + j]);

    // SRAM0: input activations
    for (int i = 0; i < S; i++)
        for (int j = 0; j < HIDDEN; j++)
            sram0_write(ADDR_X + i * HIDDEN + j, (uint8_t)x_data[i * HIDDEN + j]);

    // SRAM1: LN betas
    for (int j = 0; j < HIDDEN; j++) {
        sram1_write(S1_LN1_BETA + j, (uint8_t)bw.ln1_beta[j]);
        sram1_write(S1_LN2_BETA + j, (uint8_t)bw.ln2_beta[j]);
    }

    // SRAM1: copy X for first residual add
    for (int i = 0; i < S; i++)
        for (int j = 0; j < HIDDEN; j++)
            sram1_write(S1_RESID + i * HIDDEN + j, (uint8_t)x_data[i * HIDDEN + j]);
}

// =============================================================================
// Read output from SRAM0 after block execution
// =============================================================================
void read_block_output(int S, int8_t* out) {
    for (int i = 0; i < S; i++)
        for (int j = 0; j < HIDDEN; j++)
            out[i * HIDDEN + j] = (int8_t)sram0_read(ADDR_X_OUT + i * HIDDEN + j);
}

// =============================================================================
// Load block weights + single-token activation for decode mode
// WQ/WK/WV are loaded as flat bytes (already head-blocked in weights.bin)
// =============================================================================
void load_decode_block_to_srams(int blk_idx, const int8_t* x_data) {
    const BlockWeights& bw = block_weights[blk_idx];

    // SRAM0: weights - load wq/wk/wv as flat bytes (head-blocked layout)
    for (int i = 0; i < HIDDEN * HIDDEN; i++) {
        sram0_write(ADDR_WQ + i, (uint8_t)bw.wq[i]);
        sram0_write(ADDR_WK + i, (uint8_t)bw.wk[i]);
        sram0_write(ADDR_WV + i, (uint8_t)bw.wv[i]);
    }

    for (int i = 0; i < HIDDEN; i++)
        for (int j = 0; j < HIDDEN; j++)
            sram0_write(ADDR_WO + i * HIDDEN + j, (uint8_t)bw.wo[i * HIDDEN + j]);

    for (int i = 0; i < HIDDEN; i++)
        for (int j = 0; j < FFN_DIM; j++)
            sram0_write(ADDR_W1 + i * FFN_DIM + j, (uint8_t)bw.w1[i * FFN_DIM + j]);

    for (int i = 0; i < FFN_DIM; i++)
        for (int j = 0; j < HIDDEN; j++)
            sram0_write(ADDR_W2 + i * HIDDEN + j, (uint8_t)bw.w2[i * HIDDEN + j]);

    // SRAM0: single-token input at decode address
    for (int j = 0; j < HIDDEN; j++)
        sram0_write(ADDR_DEC_X + j, (uint8_t)x_data[j]);

    // SRAM1: LN betas
    for (int j = 0; j < HIDDEN; j++) {
        sram1_write(S1_LN1_BETA + j, (uint8_t)bw.ln1_beta[j]);
        sram1_write(S1_LN2_BETA + j, (uint8_t)bw.ln2_beta[j]);
    }

    // SRAM1: copy X for first residual add (single row)
    for (int j = 0; j < HIDDEN; j++)
        sram1_write(S1_RESID + j, (uint8_t)x_data[j]);
}

// =============================================================================
// Read output from SRAM0 after decode block execution (single token)
// =============================================================================
void read_decode_block_output(int8_t* out) {
    for (int j = 0; j < HIDDEN; j++)
        out[j] = (int8_t)sram0_read(ADDR_DEC_X_OUT + j);
}

// =============================================================================
// Read prompt tokens from file
// =============================================================================
std::vector<int> read_prompt_tokens(const std::string& path) {
    std::vector<int> tokens;
    std::ifstream f(path);
    if (!f.is_open()) {
        std::cerr << "ERROR: Cannot open prompt tokens file: " << path << std::endl;
        return tokens;
    }
    int t;
    while (f >> t) {
        tokens.push_back(t);
    }
    return tokens;
}

// =============================================================================
// Main - KV Cache Correctness Test
// =============================================================================
int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    Verilated::traceEverOn(true);

    // Parse --datadir
    std::string datadir = ".";
    if (getenv("DEMO_OUTDIR")) datadir = getenv("DEMO_OUTDIR");
    for (int i = 1; i < argc - 1; i++) {
        if (std::string(argv[i]) == "--datadir") datadir = argv[i + 1];
    }

    std::cout << "============================================" << std::endl;
    std::cout << "  KV Cache Correctness Test" << std::endl;
    std::cout << "  datadir=" << datadir << std::endl;
    std::cout << "============================================" << std::endl;

    // Load weights
    std::string weights_path = datadir + "/weights.bin";
    if (!load_weights(weights_path)) return 1;

    // Load prompt tokens
    std::string prompt_path = datadir + "/prompt_tokens.txt";
    std::vector<int> tokens = read_prompt_tokens(prompt_path);
    if (tokens.empty()) {
        std::cerr << "ERROR: No prompt tokens loaded" << std::endl;
        return 1;
    }
    // If prompt has more than MAX_SEQ-2 tokens, truncate to MAX_SEQ-2
    // (need room for at least 2 decode steps)
    if ((int)tokens.size() > MAX_SEQ - 2) {
        tokens.resize(MAX_SEQ - 2);
        std::cout << "Truncated prompt to " << tokens.size() << " tokens" << std::endl;
    }

    // Create DUT (no VCD trace for test)
    dut = new Vgpt2_block_top;
    tfp = nullptr;

    // Initialize inputs
    dut->clk = 0; dut->rst_n = 0; dut->start_pulse = 0; dut->ucode_len = 0;
    dut->uc_wr_en = 0; dut->uc_wr_addr = 0;
    memset(dut->uc_wr_data, 0, sizeof(dut->uc_wr_data));
    dut->tb_sram0_wr_en = 0; dut->tb_sram0_wr_addr = 0; dut->tb_sram0_wr_data = 0;
    dut->tb_sram0_rd_en = 0; dut->tb_sram0_rd_addr = 0;
    dut->tb_sram1_wr_en = 0; dut->tb_sram1_wr_addr = 0; dut->tb_sram1_wr_data = 0;
    dut->tb_sram1_rd_en = 0; dut->tb_sram1_rd_addr = 0;
    dut->dma_done_pulse = 0;

    reset_dut();

    int prompt_len = (int)tokens.size();
    int num_decode_steps = std::min(2, MAX_SEQ - prompt_len);
    bool all_pass = true;

    // ======================================================================
    // TEST 1: Prefill == Full-recompute (same S tokens)
    // ======================================================================
    std::cout << "\n--- TEST 1: Prefill vs Full-recompute (S=" << prompt_len << ") ---" << std::endl;
    {
        // Embed prompt
        std::vector<int8_t> emb(prompt_len * HIDDEN);
        compute_embeddings(tokens, emb.data());

        // --- Full-recompute path ---
        std::vector<int8_t> x_full(emb.begin(), emb.end());
        std::vector<int8_t> x_full_next(prompt_len * HIDDEN);

        for (int blk = 0; blk < N_LAYERS; blk++) {
            reset_dut();
            load_block_to_srams(blk, x_full.data(), prompt_len);
            int n = gen_block_microcode(prompt_len, 0);
            dut->ucode_len = n;
            dut->start_pulse = 1; tick(); dut->start_pulse = 0;
            int cy = run_until_done();
            if (cy < 0) { std::cerr << "TIMEOUT full-recompute blk " << blk << std::endl; return 1; }
            read_block_output(prompt_len, x_full_next.data());
            x_full = x_full_next;
        }

        // --- KV-cached prefill path ---
        // KV cache is now in hardware (kv_cache_bank), cleared by reset
        std::vector<int8_t> x_kv(emb.begin(), emb.end());
        std::vector<int8_t> x_kv_next(prompt_len * HIDDEN);

        for (int blk = 0; blk < N_LAYERS; blk++) {
            reset_dut();
            load_block_to_srams(blk, x_kv.data(), prompt_len);
            int n = gen_prefill_block_microcode(prompt_len, blk, 0);
            dut->ucode_len = n;
            dut->start_pulse = 1; tick(); dut->start_pulse = 0;
            int cy = run_until_done();
            if (cy < 0) { std::cerr << "TIMEOUT prefill blk " << blk << std::endl; return 1; }
            read_block_output(prompt_len, x_kv_next.data());
            x_kv = x_kv_next;
        }

        // Compare block outputs
        int max_err = 0;
        int mismatches = 0;
        for (int i = 0; i < prompt_len * HIDDEN; i++) {
            int err = abs((int)x_full[i] - (int)x_kv[i]);
            if (err > max_err) max_err = err;
            if (err > 0) mismatches++;
        }

        bool t1_pass = (max_err == 0);
        std::cout << "  Block output comparison: max_err=" << max_err
                  << " mismatches=" << mismatches << "/" << prompt_len * HIDDEN
                  << (t1_pass ? " PASS" : " FAIL") << std::endl;
        if (!t1_pass) all_pass = false;

        // --- Also compare LN_F + lm_head logits ---
        // Full-recompute LN_F + lm_head
        reset_dut();
        for (int j = 0; j < HIDDEN; j++)
            sram1_write(S1_LN_F_BETA + j, (uint8_t)ln_f_beta[j]);
        for (int j = 0; j < HIDDEN; j++)
            sram0_write(ADDR_LM_INPUT + j, (uint8_t)x_full[(prompt_len-1)*HIDDEN + j]);
        { int n = gen_ln_f_microcode(0); dut->ucode_len = n;
          dut->start_pulse = 1; tick(); dut->start_pulse = 0;
          run_until_done(); }

        int8_t ln_full[HIDDEN];
        for (int j = 0; j < HIDDEN; j++)
            ln_full[j] = (int8_t)sram0_read(ADDR_LM_INPUT + j);

        reset_dut();
        for (int j = 0; j < HIDDEN; j++)
            sram0_write(ADDR_LM_INPUT + j, (uint8_t)ln_full[j]);
        for (int i = 0; i < VOCAB_SIZE; i++)
            for (int j = 0; j < HIDDEN; j++)
                sram0_write(ADDR_LM_WEIGHT + i*HIDDEN + j, (uint8_t)lm_head[i*HIDDEN + j]);
        { int n = gen_lm_head_microcode(0); dut->ucode_len = n;
          dut->start_pulse = 1; tick(); dut->start_pulse = 0;
          run_until_done(); }

        int8_t logits_full[VOCAB_SIZE];
        for (int i = 0; i < VOCAB_SIZE; i++)
            logits_full[i] = (int8_t)sram0_read(ADDR_LM_OUTPUT + i);

        // KV-cached LN_F + lm_head (same, using x_kv)
        reset_dut();
        for (int j = 0; j < HIDDEN; j++)
            sram1_write(S1_LN_F_BETA + j, (uint8_t)ln_f_beta[j]);
        for (int j = 0; j < HIDDEN; j++)
            sram0_write(ADDR_LM_INPUT + j, (uint8_t)x_kv[(prompt_len-1)*HIDDEN + j]);
        { int n = gen_ln_f_microcode(0); dut->ucode_len = n;
          dut->start_pulse = 1; tick(); dut->start_pulse = 0;
          run_until_done(); }

        int8_t ln_kv[HIDDEN];
        for (int j = 0; j < HIDDEN; j++)
            ln_kv[j] = (int8_t)sram0_read(ADDR_LM_INPUT + j);

        reset_dut();
        for (int j = 0; j < HIDDEN; j++)
            sram0_write(ADDR_LM_INPUT + j, (uint8_t)ln_kv[j]);
        for (int i = 0; i < VOCAB_SIZE; i++)
            for (int j = 0; j < HIDDEN; j++)
                sram0_write(ADDR_LM_WEIGHT + i*HIDDEN + j, (uint8_t)lm_head[i*HIDDEN + j]);
        { int n = gen_lm_head_microcode(0); dut->ucode_len = n;
          dut->start_pulse = 1; tick(); dut->start_pulse = 0;
          run_until_done(); }

        int8_t logits_kv[VOCAB_SIZE];
        for (int i = 0; i < VOCAB_SIZE; i++)
            logits_kv[i] = (int8_t)sram0_read(ADDR_LM_OUTPUT + i);

        int logit_err = 0;
        for (int i = 0; i < VOCAB_SIZE; i++) {
            int err = abs((int)logits_full[i] - (int)logits_kv[i]);
            if (err > logit_err) logit_err = err;
        }

        int tok_full = std::distance(logits_full, std::max_element(logits_full, logits_full + VOCAB_SIZE));
        int tok_kv = std::distance(logits_kv, std::max_element(logits_kv, logits_kv + VOCAB_SIZE));

        bool t1_logit_pass = (logit_err == 0);
        std::cout << "  Logit comparison: max_err=" << logit_err
                  << " tok_full=" << tok_full << " tok_kv=" << tok_kv
                  << (t1_logit_pass ? " PASS" : " FAIL") << std::endl;
        if (!t1_logit_pass) all_pass = false;

        // Pick next token (greedy from full path)
        tokens.push_back(tok_full);
    }

    // ======================================================================
    // TEST 2: Decode vs Full-recompute (prompt_len + 1 tokens)
    // ======================================================================
    if (num_decode_steps >= 1) {
    std::cout << "\n--- TEST 2: Decode vs Full-recompute (S=" << tokens.size() << ") ---" << std::endl;
    {
        int S = (int)tokens.size();  // prompt_len + 1

        // --- Full-recompute with S tokens ---
        std::vector<int8_t> emb_full(S * HIDDEN);
        compute_embeddings(tokens, emb_full.data());

        std::vector<int8_t> x_full(emb_full.begin(), emb_full.end());
        std::vector<int8_t> x_full_next(S * HIDDEN);

        for (int blk = 0; blk < N_LAYERS; blk++) {
            reset_dut();
            load_block_to_srams(blk, x_full.data(), S);
            int n = gen_block_microcode(S, 0);
            dut->ucode_len = n;
            dut->start_pulse = 1; tick(); dut->start_pulse = 0;
            int cy = run_until_done();
            if (cy < 0) { std::cerr << "TIMEOUT full-recompute blk " << blk << std::endl; return 1; }
            read_block_output(S, x_full_next.data());
            x_full = x_full_next;
        }

        // Full-recompute LN_F + lm_head on last token
        reset_dut();
        for (int j = 0; j < HIDDEN; j++)
            sram1_write(S1_LN_F_BETA + j, (uint8_t)ln_f_beta[j]);
        for (int j = 0; j < HIDDEN; j++)
            sram0_write(ADDR_LM_INPUT + j, (uint8_t)x_full[(S-1)*HIDDEN + j]);
        { int n = gen_ln_f_microcode(0); dut->ucode_len = n;
          dut->start_pulse = 1; tick(); dut->start_pulse = 0;
          run_until_done(); }

        int8_t ln_full[HIDDEN];
        for (int j = 0; j < HIDDEN; j++)
            ln_full[j] = (int8_t)sram0_read(ADDR_LM_INPUT + j);

        reset_dut();
        for (int j = 0; j < HIDDEN; j++)
            sram0_write(ADDR_LM_INPUT + j, (uint8_t)ln_full[j]);
        for (int i = 0; i < VOCAB_SIZE; i++)
            for (int j = 0; j < HIDDEN; j++)
                sram0_write(ADDR_LM_WEIGHT + i*HIDDEN + j, (uint8_t)lm_head[i*HIDDEN + j]);
        { int n = gen_lm_head_microcode(0); dut->ucode_len = n;
          dut->start_pulse = 1; tick(); dut->start_pulse = 0;
          run_until_done(); }

        int8_t logits_full[VOCAB_SIZE];
        for (int i = 0; i < VOCAB_SIZE; i++)
            logits_full[i] = (int8_t)sram0_read(ADDR_LM_OUTPUT + i);

        // --- KV-cached decode path ---
        // KV cache was already populated by prefill in Test 1.
        // Now decode the new token.
        int tok = tokens.back();
        int pos = S - 1;  // position of new token
        int T = pos;

        // Embed single token
        int8_t x_single[HIDDEN];
        for (int h = 0; h < HIDDEN; h++) {
            int val = (int)wte[tok*HIDDEN + h] + (int)wpe[pos*HIDDEN + h];
            if (val > 127) val = 127;
            if (val < -128) val = -128;
            x_single[h] = (int8_t)val;
        }

        int8_t x_kv[HIDDEN], x_kv_next[HIDDEN];
        memcpy(x_kv, x_single, HIDDEN);

        for (int blk = 0; blk < N_LAYERS; blk++) {
            reset_dut();
            load_decode_block_to_srams(blk, x_kv);
            int n = gen_decode_block_microcode(T, blk, 0);
            dut->ucode_len = n;
            dut->start_pulse = 1; tick(); dut->start_pulse = 0;
            int cy = run_until_done();
            if (cy < 0) { std::cerr << "TIMEOUT decode blk " << blk << std::endl; return 1; }
            read_decode_block_output(x_kv_next);
            memcpy(x_kv, x_kv_next, HIDDEN);
        }

        // KV-cached LN_F + lm_head
        reset_dut();
        for (int j = 0; j < HIDDEN; j++)
            sram1_write(S1_LN_F_BETA + j, (uint8_t)ln_f_beta[j]);
        for (int j = 0; j < HIDDEN; j++)
            sram0_write(ADDR_LM_INPUT + j, (uint8_t)x_kv[j]);
        { int n = gen_ln_f_microcode(0); dut->ucode_len = n;
          dut->start_pulse = 1; tick(); dut->start_pulse = 0;
          run_until_done(); }

        int8_t ln_kv[HIDDEN];
        for (int j = 0; j < HIDDEN; j++)
            ln_kv[j] = (int8_t)sram0_read(ADDR_LM_INPUT + j);

        reset_dut();
        for (int j = 0; j < HIDDEN; j++)
            sram0_write(ADDR_LM_INPUT + j, (uint8_t)ln_kv[j]);
        for (int i = 0; i < VOCAB_SIZE; i++)
            for (int j = 0; j < HIDDEN; j++)
                sram0_write(ADDR_LM_WEIGHT + i*HIDDEN + j, (uint8_t)lm_head[i*HIDDEN + j]);
        { int n = gen_lm_head_microcode(0); dut->ucode_len = n;
          dut->start_pulse = 1; tick(); dut->start_pulse = 0;
          run_until_done(); }

        int8_t logits_kv[VOCAB_SIZE];
        for (int i = 0; i < VOCAB_SIZE; i++)
            logits_kv[i] = (int8_t)sram0_read(ADDR_LM_OUTPUT + i);

        // Compare
        int logit_err = 0;
        for (int i = 0; i < VOCAB_SIZE; i++) {
            int err = abs((int)logits_full[i] - (int)logits_kv[i]);
            if (err > logit_err) logit_err = err;
        }

        int tok_full = std::distance(logits_full, std::max_element(logits_full, logits_full + VOCAB_SIZE));
        int tok_kv = std::distance(logits_kv, std::max_element(logits_kv, logits_kv + VOCAB_SIZE));

        bool t2_pass = (logit_err == 0);
        std::cout << "  Logit comparison: max_err=" << logit_err
                  << " tok_full=" << tok_full << " tok_kv=" << tok_kv
                  << (t2_pass ? " PASS" : " FAIL") << std::endl;
        if (!t2_pass) all_pass = false;
    }
    }

    // Summary
    std::cout << "\n============================================" << std::endl;
    std::cout << "  KV CACHE TEST: " << (all_pass ? "PASS" : "FAIL") << std::endl;
    std::cout << "============================================" << std::endl;

    delete dut;
    return all_pass ? 0 : 1;
}
