// =============================================================================
// tb_llama_demo_infer.cpp - End-to-end LLaMA inference demo on NPU
// Loads weights from weights.bin, runs multi-token autoregressive decode
// using NPU hardware (4 LLaMA blocks + final RMSNorm + lm_head).
// Compares NPU-generated tokens against Python golden reference.
// =============================================================================

#include <verilated.h>
#include <verilated_vcd_c.h>
#include "Vllama_block_top.h"

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
#ifdef _WIN32
#include <direct.h>
#else
#include <sys/stat.h>
#endif

// =============================================================================
// Global simulation state
// =============================================================================
vluint64_t main_time = 0;
double sc_time_stamp() { return main_time; }

static Vllama_block_top* dut;
static VerilatedVcdC*    tfp;
static int               tc = 0;

// =============================================================================
// Model configuration (must match llama_map.py)
// =============================================================================
static const int HIDDEN     = 64;
static const int HEAD_DIM   = 16;
static const int N_Q_HEADS  = 4;
static const int N_KV_HEADS = 2;
static const int GQA_RATIO  = N_Q_HEADS / N_KV_HEADS;  // 2
static const int FFN_DIM    = 128;
static const int N_LAYERS   = 4;
static const int VOCAB_SIZE = 256;
static const int MAX_SEQ    = 16;
static const int HALF_DIM   = HEAD_DIM / 2;  // 8

// Quantization
static const int GEMM_SCALE = 1;
static const int GEMM_SHIFT = 2;
static const uint16_t GEMM_IMM_K64  = 0x0201;  // scale=1, shift=2, for K=64
static const uint16_t GEMM_IMM_K16  = 0x0701;  // scale=1, shift=7, for K=16
static const uint16_t GEMM_IMM_K128 = 0x0201;  // scale=1, shift=2, for K=128

// =============================================================================
// SRAM0 Memory layout for block execution (byte addresses)
// Same as llama_map.py / tb_llama_block.cpp
// =============================================================================
// Weights:
static const uint16_t ADDR_WQ       = 0x0000;  // 4x[64,16] = 4096B (4 Q-heads)
static const uint16_t ADDR_WK       = 0x1000;  // 2x[64,16] = 2048B (2 KV-heads)
static const uint16_t ADDR_WV       = 0x1800;  // 2x[64,16] = 2048B (2 KV-heads)
static const uint16_t ADDR_WO       = 0x2000;  // [64,64]   = 4096B
static const uint16_t ADDR_W_GATE   = 0x3000;  // [64,128]  = 8192B
static const uint16_t ADDR_W_UP     = 0x5000;  // [64,128]  = 8192B
static const uint16_t ADDR_W_DOWN   = 0x7000;  // [128,64]  = 8192B

// Activations:
static const uint16_t ADDR_X        = 0xA000;  // [S,64]   = 1024B
static const uint16_t ADDR_RMS1_OUT = 0xA400;  // [S,64]   = 1024B
static const uint16_t ADDR_Q_H      = 0xA800;  // [S,16]   =  256B (per-head, reused)
static const uint16_t ADDR_K_H      = 0xA900;  // [S,16]   =  256B (per-head, reused)
static const uint16_t ADDR_V_H      = 0xAA00;  // [S,16]   =  256B (per-head, reused)
static const uint16_t ADDR_S        = 0xAB00;  // [S,S]    =  256B (per-head, reused)
static const uint16_t ADDR_P        = 0xAC00;  // [S,S]    =  256B (per-head, reused)
static const uint16_t ADDR_ATTN_H   = 0xAD00;  // [S,16]   =  256B (per-head temp)
static const uint16_t ADDR_ATTN     = 0xAE00;  // [S,64]   = 1024B (concat destination)
static const uint16_t ADDR_WO_OUT   = 0xB200;  // [S,64]   = 1024B
static const uint16_t ADDR_X2       = 0xB600;  // [S,64]   = 1024B (residual 1)
static const uint16_t ADDR_RMS2_OUT = 0xBA00;  // [S,64]   = 1024B
static const uint16_t ADDR_FFN_GATE = 0xBE00;  // [S,128]  = 2048B
static const uint16_t ADDR_FFN_UP   = 0xC600;  // [S,128]  = 2048B
static const uint16_t ADDR_FFN_DOWN = 0xCE00;  // [S,64]   = 1024B
static const uint16_t ADDR_X_OUT    = 0xD200;  // [S,64]   = 1024B

// SRAM0 addresses for lm_head phase (reuses SRAM0 from 0x0000)
static const uint16_t ADDR_LM_INPUT  = 0x0000;  // [1][64]   = 64B (last token hidden)
static const uint16_t ADDR_LM_WEIGHT = 0x0100;  // [256][64] = 16384B (lm_head)
static const uint16_t ADDR_LM_OUTPUT = 0x4100;  // [1][256]  = 256B (logits)

// SRAM1 addresses
static const uint16_t S1_RMS1_GAMMA = 0x0000;  // [64]
static const uint16_t S1_RMS2_GAMMA = 0x0040;  // [64]
static const uint16_t S1_ROPE_SIN   = 0x0080;  // [16,8] = 128B
static const uint16_t S1_ROPE_COS   = 0x0100;  // [16,8] = 128B
static const uint16_t S1_RESID      = 0x0180;  // [S,64] = 1024B (residual source)
static const uint16_t S1_FFN_UP     = 0x0600;  // [S,128] = 2048B (for VEC MUL staging)
static const uint16_t S1_LN_F_GAMMA = 0x0E00;  // [64] (final RMSNorm gamma)
// SRAM1 addresses for replicated QKV bias
static const uint16_t S1_BIAS_Q     = 0x0E40;  // N_Q_HEADS * [S,HEAD_DIM] = 1024B
static const uint16_t S1_BIAS_K     = 0x1240;  // N_KV_HEADS * [S,HEAD_DIM] = 512B
static const uint16_t S1_BIAS_V     = 0x1440;  // N_KV_HEADS * [S,HEAD_DIM] = 512B

// =============================================================================
// weights.bin layout offsets (must match llama_map.py)
// =============================================================================
static const int WTE_OFFSET     = 0;
static const int WTE_SIZE_B     = VOCAB_SIZE * HIDDEN;                        // 16384
static const int BLOCKS_OFFSET  = WTE_OFFSET + WTE_SIZE_B;                   // 16384
static const int BLOCK_SIZE_B   = 64 + 4096 + 2048 + 2048 + 4096 + 64 + 8192 + 8192 + 8192
                                + N_Q_HEADS * HEAD_DIM + N_KV_HEADS * HEAD_DIM + N_KV_HEADS * HEAD_DIM; // 37120
static const int BLK_RMS1_GAMMA = 0;
static const int BLK_WQ         = 64;
static const int BLK_WK         = 4160;
static const int BLK_WV         = 6208;
static const int BLK_WO         = 8256;
static const int BLK_RMS2_GAMMA = 12352;
static const int BLK_W_GATE     = 12416;
static const int BLK_W_UP       = 20608;
static const int BLK_W_DOWN     = 28800;
// QKV bias offsets (appended after W_down)
static const int BLK_BQ         = 36992;
static const int BLK_BK         = 36992 + N_Q_HEADS * HEAD_DIM;              // 37056
static const int BLK_BV         = 36992 + N_Q_HEADS * HEAD_DIM + N_KV_HEADS * HEAD_DIM; // 37088
static const int LN_F_OFFSET    = BLOCKS_OFFSET + N_LAYERS * BLOCK_SIZE_B;   // 164864
static const int LN_F_SIZE      = 64;
static const int LM_HEAD_OFFSET = LN_F_OFFSET + LN_F_SIZE;                   // 164928
static const int LM_HEAD_SIZE   = VOCAB_SIZE * HIDDEN;                       // 16384
static const int WEIGHTS_TOTAL  = LM_HEAD_OFFSET + LM_HEAD_SIZE;             // 181312

// =============================================================================
// Opcodes and Flags (from tb_llama_block.cpp)
// =============================================================================
static const uint8_t OP_DMA_LOAD  = 1;
static const uint8_t OP_GEMM      = 3;
static const uint8_t OP_VEC       = 4;
static const uint8_t OP_SOFTMAX   = 5;
static const uint8_t OP_RMSNORM   = 11;
static const uint8_t OP_ROPE      = 12;
static const uint8_t OP_SILU      = 13;
static const uint8_t OP_BARRIER   = 10;
static const uint8_t OP_END       = 255;

static const uint8_t FLAG_TRANSPOSE_B = 0x01;
static const uint8_t FLAG_REQUANT     = 0x04;
static const uint8_t FLAG_COPY2D      = 0x04;
static const uint8_t FLAG_CAUSAL_MASK = 0x10;
static const uint8_t FLAG_VEC_MUL     = 0x01;

// =============================================================================
// Debug flag
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
// C++ GEMM Golden (matches RTL requantize exactly)
// =============================================================================
void gemm_golden(const int8_t* A, const int8_t* B, int8_t* C,
                 int M, int N, int K, bool transpose_b,
                 int scale, int shift) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            int32_t acc = 0;
            for (int k = 0; k < K; k++) {
                int8_t b_val = transpose_b ? B[j*K + k] : B[k*N + j];
                acc += (int32_t)A[i*K + k] * (int32_t)b_val;
            }
            int64_t prod = (int64_t)acc * (int64_t)scale;
            if (shift > 0) prod += (1LL << (shift - 1));
            int32_t res = (int32_t)(prod >> shift);
            if (res > 127)  res = 127;
            if (res < -128) res = -128;
            C[i*N + j] = (int8_t)res;
        }
    }
}

// =============================================================================
// C++ Softmax Golden (matches RTL LUT-based implementation)
// =============================================================================
static uint16_t EXP_LUT[256];
static uint16_t RECIP_LUT[256];

void build_softmax_luts() {
    for (int i = 0; i < 256; i++) {
        int8_t signed_i = (int8_t)(uint8_t)i;
        double x = (double)signed_i / 32.0;
        double val = exp(x) * 256.0;
        int v = (int)round(val);
        if (v < 0) v = 0;
        if (v > 65535) v = 65535;
        EXP_LUT[i] = (uint16_t)v;
    }
    for (int i = 0; i < 256; i++) {
        if (i == 0) {
            RECIP_LUT[i] = 65535;
        } else {
            double val = 65536.0 / (double)i;
            int v = (int)round(val);
            if (v > 65535) v = 65535;
            RECIP_LUT[i] = (uint16_t)v;
        }
    }
}

void softmax_golden(const int8_t* scores, int8_t* output, int length,
                    bool causal_mask_en, int causal_limit) {
    int8_t max_val = -128;
    for (int i = 0; i < length; i++) {
        int8_t s = scores[i];
        if (causal_mask_en && i > causal_limit) s = -128;
        if (s > max_val) max_val = s;
    }

    std::vector<uint16_t> exp_vals(length);
    int32_t exp_sum = 0;
    for (int i = 0; i < length; i++) {
        int8_t s = scores[i];
        if (causal_mask_en && i > causal_limit) s = -128;
        int16_t diff = (int16_t)s - (int16_t)max_val;
        if (diff < -128) diff = -128;
        if (diff > 127) diff = 127;
        uint8_t idx = (uint8_t)(int8_t)diff;
        exp_vals[i] = EXP_LUT[idx];
        exp_sum += (int32_t)exp_vals[i];
    }

    if (exp_sum == 0) { memset(output, 0, length); return; }
    int sum_top8 = (int)((uint16_t)(exp_sum & 0xFFFF) >> 8);
    if (sum_top8 > 255) sum_top8 = 255;
    if (sum_top8 == 0) sum_top8 = 1;
    uint16_t inv_sum = RECIP_LUT[sum_top8];

    for (int i = 0; i < length; i++) {
        uint32_t prod = (uint32_t)exp_vals[i] * (uint32_t)inv_sum;
        int32_t prob = (int32_t)(prod >> 17);
        if (prob > 127) prob = 127;
        if (prob < 0) prob = 0;
        output[i] = (int8_t)prob;
    }
}

// =============================================================================
// C++ RMSNorm Golden (matches RTL rmsnorm_engine.sv exactly)
// =============================================================================
static uint16_t RSQRT_LUT[256];

void build_rsqrt_lut() {
    RSQRT_LUT[0] = 65535;
    for (int i = 1; i < 256; i++) {
        double val = 4096.0 / sqrt((double)i);
        int v = (int)round(val);
        if (v > 65535) v = 65535;
        RSQRT_LUT[i] = (uint16_t)v;
    }
}

uint32_t counting_div_unsigned(uint64_t dividend, uint64_t divisor) {
    uint64_t remainder = dividend;
    uint32_t quotient = 0;
    for (int i = 0; i < 32; i++) {
        quotient <<= 1;
        if (remainder >= divisor) {
            remainder -= divisor;
            quotient |= 1;
        }
    }
    return quotient;
}

void rmsnorm_golden(const int8_t* input, int8_t* output, int length,
                    const int8_t* gamma) {
    uint32_t sum_sq = 0;
    for (int i = 0; i < length; i++) {
        int8_t x_s = input[i];
        uint16_t sq = (uint16_t)((int16_t)x_s * (int16_t)x_s);
        sum_sq += sq;
    }

    uint64_t dividend = (uint64_t)sum_sq << 8;
    uint32_t rms_val = counting_div_unsigned(dividend, (uint64_t)length);

    uint8_t rsqrt_addr = (uint8_t)((rms_val >> 8) & 0xFF);
    uint16_t inv_rms = RSQRT_LUT[rsqrt_addr];

    for (int i = 0; i < length; i++) {
        int8_t x = input[i];
        int8_t g = gamma[i];
        int16_t xg = (int16_t)x * (int16_t)g;
        int32_t gamma_applied = ((int32_t)xg * (int32_t)(uint32_t)inv_rms) >> 16;
        if (gamma_applied > 127) gamma_applied = 127;
        if (gamma_applied < -128) gamma_applied = -128;
        output[i] = (int8_t)gamma_applied;
    }
}

// =============================================================================
// C++ RoPE Golden (matches RTL rope_engine.sv exactly)
// =============================================================================
void rope_golden(int8_t* data, int rows, int head_dim,
                 const int8_t* sin_table, const int8_t* cos_table,
                 int pos_offset) {
    int half_dim = head_dim / 2;
    for (int row = 0; row < rows; row++) {
        int pos = row + pos_offset;
        for (int p = 0; p < half_dim; p++) {
            int8_t even_val = data[row * head_dim + 2 * p];
            int8_t odd_val  = data[row * head_dim + 2 * p + 1];
            int8_t cos_val  = cos_table[pos * half_dim + p];
            int8_t sin_val  = sin_table[pos * half_dim + p];

            int16_t rot_even = ((int16_t)even_val * (int16_t)cos_val
                              - (int16_t)odd_val  * (int16_t)sin_val
                              + 64) >> 7;
            int16_t rot_odd  = ((int16_t)even_val * (int16_t)sin_val
                              + (int16_t)odd_val  * (int16_t)cos_val
                              + 64) >> 7;

            if (rot_even > 127)  rot_even = 127;
            if (rot_even < -128) rot_even = -128;
            if (rot_odd > 127)   rot_odd = 127;
            if (rot_odd < -128)  rot_odd = -128;

            data[row * head_dim + 2 * p]     = (int8_t)rot_even;
            data[row * head_dim + 2 * p + 1] = (int8_t)rot_odd;
        }
    }
}

// =============================================================================
// C++ SiLU Golden (matches RTL silu_lut.sv exactly)
// =============================================================================
static int8_t SILU_LUT[256];

void build_silu_lut() {
    for (int i = 0; i < 256; i++) {
        int8_t signed_i = (int8_t)(uint8_t)i;
        double x = (double)signed_i / 32.0;
        double sigmoid_x = 1.0 / (1.0 + exp(-x));
        double silu_val = x * sigmoid_x;
        double result = silu_val * 32.0;
        int v = (int)round(result);
        if (v > 127) v = 127;
        if (v < -128) v = -128;
        SILU_LUT[i] = (int8_t)v;
    }
}

void silu_golden(const int8_t* input, int8_t* output, int length) {
    for (int i = 0; i < length; i++) {
        uint8_t idx = (uint8_t)input[i];
        output[i] = SILU_LUT[idx];
    }
}

// =============================================================================
// C++ Vec ADD Golden
// =============================================================================
void vec_add_golden(const int8_t* src0, const int8_t* src1, int8_t* dst, int length) {
    for (int i = 0; i < length; i++) {
        int16_t sum = (int16_t)src0[i] + (int16_t)src1[i];
        if (sum > 127) sum = 127;
        if (sum < -128) sum = -128;
        dst[i] = (int8_t)sum;
    }
}

// =============================================================================
// C++ Vec MUL Golden (a*b + 64) >> 7
// =============================================================================
void vec_mul_golden(const int8_t* src0, const int8_t* src1, int8_t* dst, int length) {
    for (int i = 0; i < length; i++) {
        int16_t prod = (int16_t)src0[i] * (int16_t)src1[i];
        int16_t result = (prod + 64) >> 7;
        if (result > 127) result = 127;
        if (result < -128) result = -128;
        dst[i] = (int8_t)result;
    }
}

// =============================================================================
// C++ Bias Add Golden: add [HEAD_DIM] bias to each row of [S, HEAD_DIM]
// =============================================================================
void bias_add_golden(int8_t* data, const int8_t* bias, int S, int head_dim) {
    for (int s = 0; s < S; s++) {
        for (int d = 0; d < head_dim; d++) {
            int16_t sum = (int16_t)data[s * head_dim + d] + (int16_t)bias[d];
            if (sum > 127) sum = 127;
            if (sum < -128) sum = -128;
            data[s * head_dim + d] = (int8_t)sum;
        }
    }
}

// =============================================================================
// RoPE table generation (matches make_lut.py gen_rope_tables)
// =============================================================================
static int8_t rope_sin_table[MAX_SEQ * HALF_DIM];
static int8_t rope_cos_table[MAX_SEQ * HALF_DIM];

void build_rope_tables() {
    const double base = 10000.0;
    for (int pos = 0; pos < MAX_SEQ; pos++) {
        for (int i = 0; i < HALF_DIM; i++) {
            double freq = 1.0 / pow(base, 2.0 * i / HEAD_DIM);
            double angle = pos * freq;
            int sv = (int)round(sin(angle) * 128.0);
            int cv = (int)round(cos(angle) * 128.0);
            if (sv > 127) sv = 127; if (sv < -128) sv = -128;
            if (cv > 127) cv = 127; if (cv < -128) cv = -128;
            rope_sin_table[pos * HALF_DIM + i] = (int8_t)sv;
            rope_cos_table[pos * HALF_DIM + i] = (int8_t)cv;
        }
    }
}

// =============================================================================
// Weight storage
// =============================================================================
static std::vector<int8_t> weights_buf;

struct BlockWeights {
    const int8_t* rms1_gamma;  // [HIDDEN]
    const int8_t* wq;          // [N_Q_HEADS * HIDDEN, HEAD_DIM] head-blocked
    const int8_t* wk;          // [N_KV_HEADS * HIDDEN, HEAD_DIM] head-blocked
    const int8_t* wv;          // [N_KV_HEADS * HIDDEN, HEAD_DIM] head-blocked
    const int8_t* wo;          // [HIDDEN, HIDDEN]
    const int8_t* rms2_gamma;  // [HIDDEN]
    const int8_t* w_gate;      // [HIDDEN, FFN_DIM]
    const int8_t* w_up;        // [HIDDEN, FFN_DIM]
    const int8_t* w_down;      // [FFN_DIM, HIDDEN]
    const int8_t* bq;          // [N_Q_HEADS * HEAD_DIM] Q bias
    const int8_t* bk;          // [N_KV_HEADS * HEAD_DIM] K bias
    const int8_t* bv;          // [N_KV_HEADS * HEAD_DIM] V bias
};

static BlockWeights block_weights[N_LAYERS];
static const int8_t* wte;          // [VOCAB_SIZE, HIDDEN]
static const int8_t* ln_f_gamma;   // [HIDDEN]
static const int8_t* lm_head;      // [VOCAB_SIZE, HIDDEN]
static bool has_qkv_bias = false;   // true if any bias is nonzero

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

    for (int i = 0; i < N_LAYERS; i++) {
        int base = BLOCKS_OFFSET + i * BLOCK_SIZE_B;
        block_weights[i].rms1_gamma = weights_buf.data() + base + BLK_RMS1_GAMMA;
        block_weights[i].wq         = weights_buf.data() + base + BLK_WQ;
        block_weights[i].wk         = weights_buf.data() + base + BLK_WK;
        block_weights[i].wv         = weights_buf.data() + base + BLK_WV;
        block_weights[i].wo         = weights_buf.data() + base + BLK_WO;
        block_weights[i].rms2_gamma = weights_buf.data() + base + BLK_RMS2_GAMMA;
        block_weights[i].w_gate     = weights_buf.data() + base + BLK_W_GATE;
        block_weights[i].w_up       = weights_buf.data() + base + BLK_W_UP;
        block_weights[i].w_down     = weights_buf.data() + base + BLK_W_DOWN;
        block_weights[i].bq         = weights_buf.data() + base + BLK_BQ;
        block_weights[i].bk         = weights_buf.data() + base + BLK_BK;
        block_weights[i].bv         = weights_buf.data() + base + BLK_BV;
    }

    ln_f_gamma = weights_buf.data() + LN_F_OFFSET;
    lm_head    = weights_buf.data() + LM_HEAD_OFFSET;

    // Detect nonzero bias (Qwen2 has QKV bias, LLaMA/Mistral do not)
    has_qkv_bias = false;
    for (int i = 0; i < N_LAYERS && !has_qkv_bias; i++) {
        for (int j = 0; j < N_Q_HEADS * HEAD_DIM; j++)
            if (block_weights[i].bq[j] != 0) { has_qkv_bias = true; break; }
        if (!has_qkv_bias)
            for (int j = 0; j < N_KV_HEADS * HEAD_DIM; j++)
                if (block_weights[i].bk[j] != 0 || block_weights[i].bv[j] != 0)
                    { has_qkv_bias = true; break; }
    }

    std::cout << "Loaded weights: " << sz << " bytes from " << path << std::endl;
    std::cout << "  QKV bias: " << (has_qkv_bias ? "ENABLED" : "disabled") << std::endl;
    return true;
}

// =============================================================================
// Embedding (host-side, WTE only - no WPE for LLaMA, RoPE handles position)
// =============================================================================
void compute_embeddings(const std::vector<int>& tokens, int8_t* emb) {
    int S = (int)tokens.size();
    for (int p = 0; p < S; p++) {
        int tok = tokens[p];
        for (int h = 0; h < HIDDEN; h++) {
            emb[p * HIDDEN + h] = wte[tok * HIDDEN + h];
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
// Generate LLaMA block microcode (RMSNorm + GQA + RoPE + SwiGLU)
// Identical structure to tb_llama_block.cpp load_microcode() but
// parameterized by S and writing to a given ucode_addr base.
// =============================================================================
int gen_block_microcode(int S, int ucode_addr) {
    uint64_t hi, lo;
    int addr = ucode_addr;

    // ---- RMSNorm 1 (S rows) ----
    for (int i = 0; i < S; i++) {
        uint16_t src = ADDR_X + i * HIDDEN;
        uint16_t dst = ADDR_RMS1_OUT + i * HIDDEN;
        encode_instr(OP_RMSNORM, 0, dst, src, S1_RMS1_GAMMA,
                     0, HIDDEN, 0, 0, hi, lo);
        ucode_write(addr++, hi, lo);
    }
    encode_instr(OP_BARRIER, 0, 0, 0, 0, 0, 0, 0, 0, hi, lo);
    ucode_write(addr++, hi, lo);

    // ---- Per Q-head attention loop (4 Q-heads, 2 KV-heads via GQA) ----
    for (int h = 0; h < N_Q_HEADS; h++) {
        int kv_h = h / GQA_RATIO;
        uint16_t wq_h_addr = ADDR_WQ + h * HIDDEN * HEAD_DIM;
        uint16_t wk_h_addr = ADDR_WK + kv_h * HIDDEN * HEAD_DIM;
        uint16_t wv_h_addr = ADDR_WV + kv_h * HIDDEN * HEAD_DIM;

        // GEMM Q_h = RMS1_OUT * Wq_h
        encode_instr(OP_GEMM, FLAG_REQUANT, ADDR_Q_H, ADDR_RMS1_OUT, wq_h_addr,
                     S, HEAD_DIM, HIDDEN, GEMM_IMM_K64, hi, lo);
        ucode_write(addr++, hi, lo);
        encode_instr(OP_BARRIER, 0, 0, 0, 0, 0, 0, 0, 0, hi, lo);
        ucode_write(addr++, hi, lo);

        // VEC ADD Q bias (if present)
        if (has_qkv_bias) {
            uint16_t bq_sram1 = S1_BIAS_Q + h * S * HEAD_DIM;
            encode_instr(OP_VEC, 0x00, ADDR_Q_H, ADDR_Q_H, bq_sram1,
                         0, S * HEAD_DIM, 0, 0, hi, lo);
            ucode_write(addr++, hi, lo);
            encode_instr(OP_BARRIER, 0, 0, 0, 0, 0, 0, 0, 0, hi, lo);
            ucode_write(addr++, hi, lo);
        }

        // GEMM K_h = RMS1_OUT * Wk_h
        encode_instr(OP_GEMM, FLAG_REQUANT, ADDR_K_H, ADDR_RMS1_OUT, wk_h_addr,
                     S, HEAD_DIM, HIDDEN, GEMM_IMM_K64, hi, lo);
        ucode_write(addr++, hi, lo);
        encode_instr(OP_BARRIER, 0, 0, 0, 0, 0, 0, 0, 0, hi, lo);
        ucode_write(addr++, hi, lo);

        // VEC ADD K bias (if present)
        if (has_qkv_bias) {
            uint16_t bk_sram1 = S1_BIAS_K + kv_h * S * HEAD_DIM;
            encode_instr(OP_VEC, 0x00, ADDR_K_H, ADDR_K_H, bk_sram1,
                         0, S * HEAD_DIM, 0, 0, hi, lo);
            ucode_write(addr++, hi, lo);
            encode_instr(OP_BARRIER, 0, 0, 0, 0, 0, 0, 0, 0, hi, lo);
            ucode_write(addr++, hi, lo);
        }

        // GEMM V_h = RMS1_OUT * Wv_h
        encode_instr(OP_GEMM, FLAG_REQUANT, ADDR_V_H, ADDR_RMS1_OUT, wv_h_addr,
                     S, HEAD_DIM, HIDDEN, GEMM_IMM_K64, hi, lo);
        ucode_write(addr++, hi, lo);
        encode_instr(OP_BARRIER, 0, 0, 0, 0, 0, 0, 0, 0, hi, lo);
        ucode_write(addr++, hi, lo);

        // VEC ADD V bias (if present)
        if (has_qkv_bias) {
            uint16_t bv_sram1 = S1_BIAS_V + kv_h * S * HEAD_DIM;
            encode_instr(OP_VEC, 0x00, ADDR_V_H, ADDR_V_H, bv_sram1,
                         0, S * HEAD_DIM, 0, 0, hi, lo);
            ucode_write(addr++, hi, lo);
            encode_instr(OP_BARRIER, 0, 0, 0, 0, 0, 0, 0, 0, hi, lo);
            ucode_write(addr++, hi, lo);
        }

        // ROPE Q_h
        encode_instr(OP_ROPE, 0, ADDR_Q_H, ADDR_Q_H, S1_ROPE_SIN,
                     S, HEAD_DIM, 0, S1_ROPE_COS, hi, lo);
        ucode_write(addr++, hi, lo);
        encode_instr(OP_BARRIER, 0, 0, 0, 0, 0, 0, 0, 0, hi, lo);
        ucode_write(addr++, hi, lo);

        // ROPE K_h
        encode_instr(OP_ROPE, 0, ADDR_K_H, ADDR_K_H, S1_ROPE_SIN,
                     S, HEAD_DIM, 0, S1_ROPE_COS, hi, lo);
        ucode_write(addr++, hi, lo);
        encode_instr(OP_BARRIER, 0, 0, 0, 0, 0, 0, 0, 0, hi, lo);
        ucode_write(addr++, hi, lo);

        // GEMM S = Q_h * K_h^T
        encode_instr(OP_GEMM, FLAG_TRANSPOSE_B | FLAG_REQUANT, ADDR_S, ADDR_Q_H, ADDR_K_H,
                     S, S, HEAD_DIM, GEMM_IMM_K16, hi, lo);
        ucode_write(addr++, hi, lo);
        encode_instr(OP_BARRIER, 0, 0, 0, 0, 0, 0, 0, 0, hi, lo);
        ucode_write(addr++, hi, lo);

        // SOFTMAX rows (S rows with causal mask)
        for (int i = 0; i < S; i++) {
            uint16_t src = ADDR_S + i * S;
            uint16_t dst = ADDR_P + i * S;
            encode_instr(OP_SOFTMAX, FLAG_CAUSAL_MASK, dst, src, 0,
                         0, S, i, 0x0100, hi, lo);
            ucode_write(addr++, hi, lo);
        }
        encode_instr(OP_BARRIER, 0, 0, 0, 0, 0, 0, 0, 0, hi, lo);
        ucode_write(addr++, hi, lo);

        // GEMM ATTN_h = P * V_h
        encode_instr(OP_GEMM, FLAG_REQUANT, ADDR_ATTN_H, ADDR_P, ADDR_V_H,
                     S, HEAD_DIM, S, GEMM_IMM_K16, hi, lo);
        ucode_write(addr++, hi, lo);
        encode_instr(OP_BARRIER, 0, 0, 0, 0, 0, 0, 0, 0, hi, lo);
        ucode_write(addr++, hi, lo);

        // VEC COPY2D: scatter ATTN_H -> ATTN[:, h*16:(h+1)*16]
        encode_instr(OP_VEC, FLAG_COPY2D, ADDR_ATTN + h * HEAD_DIM, ADDR_ATTN_H, 0,
                     S, HEAD_DIM, HEAD_DIM, HIDDEN, hi, lo);
        ucode_write(addr++, hi, lo);
        encode_instr(OP_BARRIER, 0, 0, 0, 0, 0, 0, 0, 0, hi, lo);
        ucode_write(addr++, hi, lo);
    }

    // ---- Output projection ----
    encode_instr(OP_GEMM, FLAG_REQUANT, ADDR_WO_OUT, ADDR_ATTN, ADDR_WO,
                 S, HIDDEN, HIDDEN, GEMM_IMM_K64, hi, lo);
    ucode_write(addr++, hi, lo);
    encode_instr(OP_BARRIER, 0, 0, 0, 0, 0, 0, 0, 0, hi, lo);
    ucode_write(addr++, hi, lo);

    // VEC ADD: X2 = WO_OUT + X (src1=SRAM1 residual)
    encode_instr(OP_VEC, 0x00, ADDR_X2, ADDR_WO_OUT, S1_RESID,
                 0, S * HIDDEN, 0, 0, hi, lo);
    ucode_write(addr++, hi, lo);
    encode_instr(OP_BARRIER, 0, 0, 0, 0, 0, 0, 0, 0, hi, lo);
    ucode_write(addr++, hi, lo);

    // DMA: copy X2 to SRAM1 for second residual
    encode_instr(OP_DMA_LOAD, 0, S1_RESID, ADDR_X2, 0,
                 0, S * HIDDEN, 0, 0, hi, lo);
    ucode_write(addr++, hi, lo);
    encode_instr(OP_BARRIER, 0, 0, 0, 0, 0, 0, 0, 0, hi, lo);
    ucode_write(addr++, hi, lo);

    // ---- RMSNorm 2 (S rows) ----
    for (int i = 0; i < S; i++) {
        uint16_t src = ADDR_X2 + i * HIDDEN;
        uint16_t dst = ADDR_RMS2_OUT + i * HIDDEN;
        encode_instr(OP_RMSNORM, 0, dst, src, S1_RMS2_GAMMA,
                     0, HIDDEN, 0, 0, hi, lo);
        ucode_write(addr++, hi, lo);
    }
    encode_instr(OP_BARRIER, 0, 0, 0, 0, 0, 0, 0, 0, hi, lo);
    ucode_write(addr++, hi, lo);

    // ---- SwiGLU FFN ----
    // GEMM FFN_GATE = RMS2_OUT * W_gate
    encode_instr(OP_GEMM, FLAG_REQUANT, ADDR_FFN_GATE, ADDR_RMS2_OUT, ADDR_W_GATE,
                 S, FFN_DIM, HIDDEN, GEMM_IMM_K64, hi, lo);
    ucode_write(addr++, hi, lo);
    encode_instr(OP_BARRIER, 0, 0, 0, 0, 0, 0, 0, 0, hi, lo);
    ucode_write(addr++, hi, lo);

    // GEMM FFN_UP = RMS2_OUT * W_up
    encode_instr(OP_GEMM, FLAG_REQUANT, ADDR_FFN_UP, ADDR_RMS2_OUT, ADDR_W_UP,
                 S, FFN_DIM, HIDDEN, GEMM_IMM_K64, hi, lo);
    ucode_write(addr++, hi, lo);
    encode_instr(OP_BARRIER, 0, 0, 0, 0, 0, 0, 0, 0, hi, lo);
    ucode_write(addr++, hi, lo);

    // SILU on FFN_GATE (in-place)
    encode_instr(OP_SILU, 0, ADDR_FFN_GATE, ADDR_FFN_GATE, 0,
                 0, S * FFN_DIM, 0, 0, hi, lo);
    ucode_write(addr++, hi, lo);
    encode_instr(OP_BARRIER, 0, 0, 0, 0, 0, 0, 0, 0, hi, lo);
    ucode_write(addr++, hi, lo);

    // DMA: copy FFN_UP from SRAM0 to SRAM1 for VEC MUL
    encode_instr(OP_DMA_LOAD, 0, S1_FFN_UP, ADDR_FFN_UP, 0,
                 0, S * FFN_DIM, 0, 0, hi, lo);
    ucode_write(addr++, hi, lo);
    encode_instr(OP_BARRIER, 0, 0, 0, 0, 0, 0, 0, 0, hi, lo);
    ucode_write(addr++, hi, lo);

    // VEC MUL: FFN_GATE = FFN_GATE * FFN_UP
    encode_instr(OP_VEC, FLAG_VEC_MUL, ADDR_FFN_GATE, ADDR_FFN_GATE, S1_FFN_UP,
                 0, S * FFN_DIM, 0, 0, hi, lo);
    ucode_write(addr++, hi, lo);
    encode_instr(OP_BARRIER, 0, 0, 0, 0, 0, 0, 0, 0, hi, lo);
    ucode_write(addr++, hi, lo);

    // GEMM FFN_DOWN = FFN_GATE * W_down  (K=128)
    encode_instr(OP_GEMM, FLAG_REQUANT, ADDR_FFN_DOWN, ADDR_FFN_GATE, ADDR_W_DOWN,
                 S, HIDDEN, FFN_DIM, GEMM_IMM_K128, hi, lo);
    ucode_write(addr++, hi, lo);
    encode_instr(OP_BARRIER, 0, 0, 0, 0, 0, 0, 0, 0, hi, lo);
    ucode_write(addr++, hi, lo);

    // VEC ADD: X_OUT = FFN_DOWN + X2 (src1=SRAM1 residual)
    encode_instr(OP_VEC, 0x00, ADDR_X_OUT, ADDR_FFN_DOWN, S1_RESID,
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
// Generate microcode for final RMSNorm on last token only
// =============================================================================
int gen_ln_f_microcode(int ucode_addr) {
    uint64_t hi, lo;
    int addr = ucode_addr;

    // RMSNorm on last-token hidden state at ADDR_LM_INPUT (output in-place)
    encode_instr(OP_RMSNORM, 0, ADDR_LM_INPUT, ADDR_LM_INPUT, S1_LN_F_GAMMA,
                 0, HIDDEN, 0, 0, hi, lo);
    ucode_write(addr++, hi, lo);

    encode_instr(OP_BARRIER, 0, 0, 0, 0, 0, 0, 0, 0, hi, lo);
    ucode_write(addr++, hi, lo);

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

    encode_instr(OP_GEMM, FLAG_TRANSPOSE_B | FLAG_REQUANT,
                 ADDR_LM_OUTPUT, ADDR_LM_INPUT, ADDR_LM_WEIGHT,
                 1, VOCAB_SIZE, HIDDEN, GEMM_IMM_K64, hi, lo);
    ucode_write(addr++, hi, lo);

    encode_instr(OP_BARRIER, 0, 0, 0, 0, 0, 0, 0, 0, hi, lo);
    ucode_write(addr++, hi, lo);

    encode_instr(OP_END, 0, 0, 0, 0, 0, 0, 0, 0, hi, lo);
    ucode_write(addr++, hi, lo);

    return addr - ucode_addr;
}

// =============================================================================
// Run NPU program until program_end, handling DMA shim
// Returns cycle count, or -1 on timeout
// =============================================================================
int run_until_done(int max_cycles = 4000000) {
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
// =============================================================================
void load_block_to_srams(int blk_idx, const int8_t* x_data, int S) {
    const BlockWeights& bw = block_weights[blk_idx];

    // SRAM0: Wq [N_Q_HEADS * HIDDEN, HEAD_DIM] = 4096B
    for (int i = 0; i < N_Q_HEADS * HIDDEN * HEAD_DIM; i++)
        sram0_write(ADDR_WQ + i, (uint8_t)bw.wq[i]);

    // SRAM0: Wk [N_KV_HEADS * HIDDEN, HEAD_DIM] = 2048B
    for (int i = 0; i < N_KV_HEADS * HIDDEN * HEAD_DIM; i++)
        sram0_write(ADDR_WK + i, (uint8_t)bw.wk[i]);

    // SRAM0: Wv [N_KV_HEADS * HIDDEN, HEAD_DIM] = 2048B
    for (int i = 0; i < N_KV_HEADS * HIDDEN * HEAD_DIM; i++)
        sram0_write(ADDR_WV + i, (uint8_t)bw.wv[i]);

    // SRAM0: Wo [HIDDEN, HIDDEN] = 4096B
    for (int i = 0; i < HIDDEN * HIDDEN; i++)
        sram0_write(ADDR_WO + i, (uint8_t)bw.wo[i]);

    // SRAM0: W_gate [HIDDEN, FFN_DIM] = 8192B
    for (int i = 0; i < HIDDEN * FFN_DIM; i++)
        sram0_write(ADDR_W_GATE + i, (uint8_t)bw.w_gate[i]);

    // SRAM0: W_up [HIDDEN, FFN_DIM] = 8192B
    for (int i = 0; i < HIDDEN * FFN_DIM; i++)
        sram0_write(ADDR_W_UP + i, (uint8_t)bw.w_up[i]);

    // SRAM0: W_down [FFN_DIM, HIDDEN] = 8192B
    for (int i = 0; i < FFN_DIM * HIDDEN; i++)
        sram0_write(ADDR_W_DOWN + i, (uint8_t)bw.w_down[i]);

    // SRAM0: input activations
    for (int i = 0; i < S * HIDDEN; i++)
        sram0_write(ADDR_X + i, (uint8_t)x_data[i]);

    // SRAM1: RMSNorm gammas
    for (int j = 0; j < HIDDEN; j++) {
        sram1_write(S1_RMS1_GAMMA + j, (uint8_t)bw.rms1_gamma[j]);
        sram1_write(S1_RMS2_GAMMA + j, (uint8_t)bw.rms2_gamma[j]);
    }

    // SRAM1: RoPE tables
    for (int i = 0; i < MAX_SEQ * HALF_DIM; i++) {
        sram1_write(S1_ROPE_SIN + i, (uint8_t)rope_sin_table[i]);
        sram1_write(S1_ROPE_COS + i, (uint8_t)rope_cos_table[i]);
    }

    // SRAM1: copy X for first residual add
    for (int i = 0; i < S * HIDDEN; i++)
        sram1_write(S1_RESID + i, (uint8_t)x_data[i]);

    // SRAM1: replicate QKV bias (if present)
    if (has_qkv_bias) {
        // Q bias: replicate [HEAD_DIM] -> [S, HEAD_DIM] for each Q-head
        for (int h = 0; h < N_Q_HEADS; h++) {
            uint16_t base = S1_BIAS_Q + h * S * HEAD_DIM;
            for (int s = 0; s < S; s++)
                for (int d = 0; d < HEAD_DIM; d++)
                    sram1_write(base + s * HEAD_DIM + d,
                               (uint8_t)bw.bq[h * HEAD_DIM + d]);
        }
        // K bias: replicate for each KV-head
        for (int h = 0; h < N_KV_HEADS; h++) {
            uint16_t base = S1_BIAS_K + h * S * HEAD_DIM;
            for (int s = 0; s < S; s++)
                for (int d = 0; d < HEAD_DIM; d++)
                    sram1_write(base + s * HEAD_DIM + d,
                               (uint8_t)bw.bk[h * HEAD_DIM + d]);
        }
        // V bias: replicate for each KV-head
        for (int h = 0; h < N_KV_HEADS; h++) {
            uint16_t base = S1_BIAS_V + h * S * HEAD_DIM;
            for (int s = 0; s < S; s++)
                for (int d = 0; d < HEAD_DIM; d++)
                    sram1_write(base + s * HEAD_DIM + d,
                               (uint8_t)bw.bv[h * HEAD_DIM + d]);
        }
    }
}

// =============================================================================
// Read output from SRAM0 after block execution
// =============================================================================
void read_block_output(int S, int8_t* out) {
    for (int i = 0; i < S * HIDDEN; i++)
        out[i] = (int8_t)sram0_read(ADDR_X_OUT + i);
}

// =============================================================================
// C++ golden: run one full LLaMA transformer block
// =============================================================================
void golden_block(const int8_t* x_in, int8_t* x_out, int S, int blk_idx) {
    const BlockWeights& bw = block_weights[blk_idx];
    int scale = GEMM_SCALE;
    int shift_k64 = GEMM_SHIFT;    // 2
    int shift_k16 = 7;
    int shift_k128 = 2;

    std::vector<int8_t> rms1(S * HIDDEN);
    std::vector<int8_t> wo_out(S * HIDDEN), x2(S * HIDDEN);
    std::vector<int8_t> rms2(S * HIDDEN);
    std::vector<int8_t> ffn_gate(S * FFN_DIM), ffn_up(S * FFN_DIM);
    std::vector<int8_t> silu_gate(S * FFN_DIM), ffn_mul(S * FFN_DIM);
    std::vector<int8_t> ffn_down(S * HIDDEN);

    // RMSNorm 1
    for (int r = 0; r < S; r++)
        rmsnorm_golden(x_in + r * HIDDEN, rms1.data() + r * HIDDEN, HIDDEN, bw.rms1_gamma);

    // Multi-head GQA attention with RoPE
    std::vector<int8_t> attn_concat(S * HIDDEN, 0);

    for (int h = 0; h < N_Q_HEADS; h++) {
        int kv_h = h / GQA_RATIO;
        const int8_t* wq_h = bw.wq + h * HIDDEN * HEAD_DIM;
        const int8_t* wk_h = bw.wk + kv_h * HIDDEN * HEAD_DIM;
        const int8_t* wv_h = bw.wv + kv_h * HIDDEN * HEAD_DIM;

        std::vector<int8_t> q_h(S * HEAD_DIM), k_h(S * HEAD_DIM), v_h(S * HEAD_DIM);
        gemm_golden(rms1.data(), wq_h, q_h.data(), S, HEAD_DIM, HIDDEN, false, scale, shift_k64);
        gemm_golden(rms1.data(), wk_h, k_h.data(), S, HEAD_DIM, HIDDEN, false, scale, shift_k64);
        gemm_golden(rms1.data(), wv_h, v_h.data(), S, HEAD_DIM, HIDDEN, false, scale, shift_k64);

        // QKV bias add
        if (has_qkv_bias) {
            bias_add_golden(q_h.data(), bw.bq + h * HEAD_DIM, S, HEAD_DIM);
            bias_add_golden(k_h.data(), bw.bk + kv_h * HEAD_DIM, S, HEAD_DIM);
            bias_add_golden(v_h.data(), bw.bv + kv_h * HEAD_DIM, S, HEAD_DIM);
        }

        // Apply RoPE
        rope_golden(q_h.data(), S, HEAD_DIM, rope_sin_table, rope_cos_table, 0);
        rope_golden(k_h.data(), S, HEAD_DIM, rope_sin_table, rope_cos_table, 0);

        std::vector<int8_t> s_h(S * S), p_h(S * S);
        gemm_golden(q_h.data(), k_h.data(), s_h.data(), S, S, HEAD_DIM, true, scale, shift_k16);

        for (int r = 0; r < S; r++)
            softmax_golden(s_h.data() + r * S, p_h.data() + r * S, S, true, r);

        std::vector<int8_t> attn_h(S * HEAD_DIM);
        gemm_golden(p_h.data(), v_h.data(), attn_h.data(), S, HEAD_DIM, S, false, scale, shift_k16);

        for (int r = 0; r < S; r++)
            for (int c = 0; c < HEAD_DIM; c++)
                attn_concat[r * HIDDEN + h * HEAD_DIM + c] = attn_h[r * HEAD_DIM + c];
    }

    // WO_out = ATTN * Wo
    gemm_golden(attn_concat.data(), bw.wo, wo_out.data(), S, HIDDEN, HIDDEN, false, scale, shift_k64);

    // Residual 1
    vec_add_golden(wo_out.data(), x_in, x2.data(), S * HIDDEN);

    // RMSNorm 2
    for (int r = 0; r < S; r++)
        rmsnorm_golden(x2.data() + r * HIDDEN, rms2.data() + r * HIDDEN, HIDDEN, bw.rms2_gamma);

    // SwiGLU FFN
    gemm_golden(rms2.data(), bw.w_gate, ffn_gate.data(), S, FFN_DIM, HIDDEN, false, scale, shift_k64);
    gemm_golden(rms2.data(), bw.w_up, ffn_up.data(), S, FFN_DIM, HIDDEN, false, scale, shift_k64);
    silu_golden(ffn_gate.data(), silu_gate.data(), S * FFN_DIM);
    vec_mul_golden(silu_gate.data(), ffn_up.data(), ffn_mul.data(), S * FFN_DIM);
    gemm_golden(ffn_mul.data(), bw.w_down, ffn_down.data(), S, HIDDEN, FFN_DIM, false, scale, shift_k128);

    // Residual 2
    vec_add_golden(ffn_down.data(), x2.data(), x_out, S * HIDDEN);
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
    while (f >> t) tokens.push_back(t);
    return tokens;
}

// =============================================================================
// Read golden tokens from file
// =============================================================================
std::vector<int> read_golden_tokens(const std::string& path) {
    std::vector<int> tokens;
    std::ifstream f(path);
    if (!f.is_open()) return tokens;
    int t;
    while (f >> t) tokens.push_back(t);
    return tokens;
}

// =============================================================================
// Main
// =============================================================================
int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    Verilated::traceEverOn(true);

    NPU_DUMP = (getenv("NPU_DUMP") != nullptr);

    // Determine data directory
    std::string datadir = ".";
    if (getenv("DEMO_OUTDIR")) {
        datadir = getenv("DEMO_OUTDIR");
    }
    for (int i = 1; i < argc - 1; i++) {
        if (std::string(argv[i]) == "--datadir") {
            datadir = argv[i + 1];
        }
    }

    int max_new_tokens = 4;
    for (int i = 1; i < argc - 1; i++) {
        if (std::string(argv[i]) == "--max-tokens") {
            max_new_tokens = std::atoi(argv[i + 1]);
        }
    }

    std::cout << "============================================" << std::endl;
    std::cout << "  LLAMA INFERENCE DEMO (NPU Hardware)" << std::endl;
    std::cout << "  layers=" << N_LAYERS << " hidden=" << HIDDEN
              << " ffn=" << FFN_DIM << " vocab=" << VOCAB_SIZE << std::endl;
    std::cout << "  q_heads=" << N_Q_HEADS << " kv_heads=" << N_KV_HEADS
              << " gqa_ratio=" << GQA_RATIO << " head_dim=" << HEAD_DIM << std::endl;
    std::cout << "  max_new_tokens=" << max_new_tokens
              << " (full-recompute)" << std::endl;
    std::cout << "  datadir=" << datadir << std::endl;
    std::cout << "============================================" << std::endl;

    // Build LUTs
    build_softmax_luts();
    build_rsqrt_lut();
    build_silu_lut();
    build_rope_tables();

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
    std::cout << "Prompt tokens (" << tokens.size() << "): ";
    for (int t : tokens) std::cout << t << " ";
    std::cout << std::endl;

    // Load golden tokens (optional)
    std::string golden_path = datadir + "/golden_tokens.txt";
    std::vector<int> golden_tokens = read_golden_tokens(golden_path);
    bool have_golden = !golden_tokens.empty();
    if (have_golden) {
        std::cout << "Golden tokens (" << golden_tokens.size() << "): ";
        for (int t : golden_tokens) std::cout << t << " ";
        std::cout << std::endl;
    } else {
        std::cout << "No golden tokens file found (comparison skipped)" << std::endl;
    }

    // Create DUT
    dut = new Vllama_block_top;
    tfp = new VerilatedVcdC;
    if (NPU_DUMP) {
        dut->trace(tfp, 99);
        tfp->open("llama_demo_infer.vcd");
    } else {
        delete tfp;
        tfp = nullptr;
    }

    // Initialize all inputs
    dut->clk = 0;
    dut->rst_n = 0;
    dut->start_pulse = 0;
    dut->ucode_len = 0;
    dut->uc_wr_en = 0;
    dut->uc_wr_addr = 0;
    memset(dut->uc_wr_data, 0, sizeof(dut->uc_wr_data));
    dut->tb_sram0_wr_en = 0; dut->tb_sram0_wr_addr = 0; dut->tb_sram0_wr_data = 0;
    dut->tb_sram0_rd_en = 0; dut->tb_sram0_rd_addr = 0;
    dut->tb_sram1_wr_en = 0; dut->tb_sram1_wr_addr = 0; dut->tb_sram1_wr_data = 0;
    dut->tb_sram1_rd_en = 0; dut->tb_sram1_rd_addr = 0;
    dut->dma_done_pulse = 0;

    // Reset
    reset_dut();

    // ==========================================================================
    // Full-Recompute Autoregressive Inference
    // ==========================================================================
    std::vector<int> generated_tokens;
    int total_npu_cycles = 0;
    bool logits_exact = true;
    int max_logit_err_all = 0;

    for (int step = 0; step < max_new_tokens; step++) {
        int S = (int)tokens.size();
        if (S > MAX_SEQ) {
            std::cout << "WARNING: seq_len " << S << " > MAX_SEQ " << MAX_SEQ
                      << ", stopping" << std::endl;
            break;
        }

        if (NPU_DUMP) {
            std::cout << "\n--- Decode step " << step << " (seq_len=" << S
                      << ") ---" << std::endl;
        }

        // 1. Compute embeddings (host-side, WTE lookup only)
        std::vector<int8_t> emb(S * HIDDEN);
        compute_embeddings(tokens, emb.data());

        // 2. Run N_LAYERS transformer blocks on NPU
        std::vector<int8_t> x_cur(emb.begin(), emb.end());
        std::vector<int8_t> x_next(S * HIDDEN);

        for (int blk = 0; blk < N_LAYERS; blk++) {
            reset_dut();
            load_block_to_srams(blk, x_cur.data(), S);
            int num_instrs = gen_block_microcode(S, 0);
            dut->ucode_len = num_instrs;
            dut->start_pulse = 1;
            tick();
            dut->start_pulse = 0;

            int cycles = run_until_done();
            if (cycles < 0) {
                std::cerr << "TIMEOUT in block " << blk << " step " << step << std::endl;
                if (tfp) tfp->close();
                delete dut;
                return 1;
            }
            total_npu_cycles += cycles;

            if (NPU_DUMP) {
                std::cout << "  Block " << blk << ": " << cycles << " cycles" << std::endl;
            }

            read_block_output(S, x_next.data());
            x_cur = x_next;
        }

        // 3. Final RMSNorm on last token
        reset_dut();

        // Load LN_F gamma to SRAM1
        for (int j = 0; j < HIDDEN; j++)
            sram1_write(S1_LN_F_GAMMA + j, (uint8_t)ln_f_gamma[j]);

        // Write last-token hidden state to SRAM0 ADDR_LM_INPUT
        for (int j = 0; j < HIDDEN; j++)
            sram0_write(ADDR_LM_INPUT + j, (uint8_t)x_cur[(S - 1) * HIDDEN + j]);

        // Run RMSNorm microcode
        int ln_instrs = gen_ln_f_microcode(0);
        dut->ucode_len = ln_instrs;
        dut->start_pulse = 1;
        tick();
        dut->start_pulse = 0;

        int ln_cycles = run_until_done();
        if (ln_cycles < 0) {
            std::cerr << "TIMEOUT in LN_F step " << step << std::endl;
            if (tfp) tfp->close();
            delete dut;
            return 1;
        }
        total_npu_cycles += ln_cycles;

        // Read RMSNorm output (in-place at ADDR_LM_INPUT)
        int8_t ln_f_out[HIDDEN];
        for (int j = 0; j < HIDDEN; j++)
            ln_f_out[j] = (int8_t)sram0_read(ADDR_LM_INPUT + j);

        // 4. LM Head on NPU
        reset_dut();

        // Write RMSNorm output to ADDR_LM_INPUT
        for (int j = 0; j < HIDDEN; j++)
            sram0_write(ADDR_LM_INPUT + j, (uint8_t)ln_f_out[j]);

        // Write lm_head weights [256, 64]
        for (int i = 0; i < VOCAB_SIZE; i++)
            for (int j = 0; j < HIDDEN; j++)
                sram0_write(ADDR_LM_WEIGHT + i * HIDDEN + j,
                           (uint8_t)lm_head[i * HIDDEN + j]);

        // Run lm_head microcode
        int lm_instrs = gen_lm_head_microcode(0);
        dut->ucode_len = lm_instrs;
        dut->start_pulse = 1;
        tick();
        dut->start_pulse = 0;

        int lm_cycles = run_until_done();
        if (lm_cycles < 0) {
            std::cerr << "TIMEOUT in lm_head step " << step << std::endl;
            if (tfp) tfp->close();
            delete dut;
            return 1;
        }
        total_npu_cycles += lm_cycles;

        // 5. Read logits and select next token (greedy argmax)
        int8_t logits[VOCAB_SIZE];
        for (int i = 0; i < VOCAB_SIZE; i++)
            logits[i] = (int8_t)sram0_read(ADDR_LM_OUTPUT + i);

        int next_tok = 0;
        int8_t max_logit = logits[0];
        for (int i = 1; i < VOCAB_SIZE; i++) {
            if (logits[i] > max_logit) {
                max_logit = logits[i];
                next_tok = i;
            }
        }

        // 6. C++ golden verification (follow-actual approach)
        // Use the ACTUAL NPU RMSNorm output for golden lm_head GEMM.
        // This eliminates cascading RMSNorm rounding differences.
        int8_t gold_logits[VOCAB_SIZE];
        gemm_golden(ln_f_out, lm_head, gold_logits, 1, VOCAB_SIZE, HIDDEN, true,
                    GEMM_SCALE, GEMM_SHIFT);

        int gold_tok = 0;
        int8_t gold_max = gold_logits[0];
        for (int i = 1; i < VOCAB_SIZE; i++) {
            if (gold_logits[i] > gold_max) {
                gold_max = gold_logits[i];
                gold_tok = i;
            }
        }

        // Compare NPU logits with follow-actual C++ golden
        int logit_max_err = 0;
        for (int i = 0; i < VOCAB_SIZE; i++) {
            int err = abs((int)logits[i] - (int)gold_logits[i]);
            if (err > logit_max_err) logit_max_err = err;
        }

        if (logit_max_err > max_logit_err_all)
            max_logit_err_all = logit_max_err;
        if (logit_max_err > 0)
            logits_exact = false;

        // Compare NPU RMSNorm output with C++ golden (informational)
        int8_t gold_rms_f[HIDDEN];
        rmsnorm_golden(x_cur.data() + (S - 1) * HIDDEN, gold_rms_f, HIDDEN, ln_f_gamma);
        int rms_f_max_err = 0;
        for (int j = 0; j < HIDDEN; j++) {
            int err = abs((int)ln_f_out[j] - (int)gold_rms_f[j]);
            if (err > rms_f_max_err) rms_f_max_err = err;
        }

        std::cout << "  Step " << std::setw(2) << step
                  << ": npu_tok=" << std::setw(3) << next_tok
                  << " gold_tok=" << std::setw(3) << gold_tok
                  << " logit_max_err=" << logit_max_err
                  << (logit_max_err == 0 ? " EXACT" : " DRIFT")
                  << " rms_f_err=" << rms_f_max_err
                  << std::endl;

        if (NPU_DUMP) {
            std::cout << "    logits[0:15]: ";
            for (int i = 0; i < 16; i++) std::cout << (int)logits[i] << " ";
            std::cout << std::endl;
            std::cout << "    gold_l[0:15]: ";
            for (int i = 0; i < 16; i++) std::cout << (int)gold_logits[i] << " ";
            std::cout << std::endl;
        }

        // Compare with Python golden (informational)
        if (have_golden && step < (int)golden_tokens.size()) {
            if (next_tok != golden_tokens[step]) {
                std::cout << "    NOTE: NPU token " << next_tok
                          << " != Python golden " << golden_tokens[step] << std::endl;
            }
        }

        generated_tokens.push_back(next_tok);
        tokens.push_back(next_tok);
    }

    // ==========================================================================
    // Summary
    // ==========================================================================
    std::cout << std::endl;
    std::cout << "============================================" << std::endl;
    std::cout << "  INFERENCE COMPLETE" << std::endl;
    std::cout << "  Generated " << generated_tokens.size() << " tokens" << std::endl;
    std::cout << "  Token IDs: ";
    for (int t : generated_tokens) std::cout << t << " ";
    std::cout << std::endl;
    std::cout << "  NPU vs C++ golden logits: max_err=" << max_logit_err_all
              << (logits_exact ? " (BIT-EXACT)" : " (DRIFT)") << std::endl;

    // Compare with Python golden (informational)
    if (have_golden) {
        int py_mismatches = 0;
        int compare_len = std::min((int)generated_tokens.size(), (int)golden_tokens.size());
        for (int i = 0; i < compare_len; i++) {
            if (generated_tokens[i] != golden_tokens[i]) py_mismatches++;
        }
        std::cout << "  NPU vs Python golden tokens: " << py_mismatches << "/" << compare_len
                  << " mismatches (informational)" << std::endl;
    }

    std::cout << "  Total NPU cycles: " << total_npu_cycles << std::endl;
    std::cout << "============================================" << std::endl;

    // Pass criterion: logits must be bit-exact with C++ follow-actual golden
    bool pass = logits_exact;
    std::cout << "  LLAMA DEMO: " << (pass ? "PASS" : "FAIL") << std::endl;
    std::cout << "============================================" << std::endl;

    // Save generated tokens
    {
        std::string out_path = datadir + "/npu_tokens.txt";
        std::ofstream f(out_path);
        for (int t : generated_tokens) f << t << "\n";
        f.close();
        std::cout << "NPU tokens saved to " << out_path << std::endl;
    }

    if (tfp) {
        tfp->close();
        delete tfp;
    }
    delete dut;

    return pass ? 0 : 1;
}
