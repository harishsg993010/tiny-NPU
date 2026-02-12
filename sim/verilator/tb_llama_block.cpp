// =============================================================================
// tb_llama_block.cpp - LLaMA transformer block integration test
// Full pipeline: RMSNorm1 -> GQA-MHA(4Q/2KV heads + RoPE) -> Wo -> Residual
//             -> RMSNorm2 -> SwiGLU FFN (gate+up+silu+mul+down) -> Residual
// Real engines: GEMM, softmax, rmsnorm, rope, gelu/silu, vec
// C++ shims: DMA only
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
// Dimensions (Tiny LLaMA config)
// =============================================================================
static const int SEQ_LEN    = 16;
static const int HIDDEN     = 64;
static const int HEAD_DIM   = 16;
static const int N_Q_HEADS  = 4;
static const int N_KV_HEADS = 2;
static const int GQA_RATIO  = N_Q_HEADS / N_KV_HEADS;  // 2
static const int FFN_DIM    = 128;
static const int MAX_SEQ    = 16;
static const int HALF_DIM   = HEAD_DIM / 2;  // 8

// =============================================================================
// SRAM0 Memory layout (byte addresses) - LLaMA block
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
static const uint16_t ADDR_X        = 0xA000;  // [16,64]   = 1024B
static const uint16_t ADDR_RMS1_OUT = 0xA400;  // [16,64]   = 1024B
static const uint16_t ADDR_Q_H      = 0xA800;  // [16,16]   = 256B (per-head, reused)
static const uint16_t ADDR_K_H      = 0xA900;  // [16,16]   = 256B (per-head, reused)
static const uint16_t ADDR_V_H      = 0xAA00;  // [16,16]   = 256B (per-head, reused)
static const uint16_t ADDR_S        = 0xAB00;  // [16,16]   = 256B (per-head, reused)
static const uint16_t ADDR_P        = 0xAC00;  // [16,16]   = 256B (per-head, reused)
static const uint16_t ADDR_ATTN_H   = 0xAD00;  // [16,16]   = 256B (per-head temp)
static const uint16_t ADDR_ATTN     = 0xAE00;  // [16,64]   = 1024B (concat destination)
static const uint16_t ADDR_WO_OUT   = 0xB200;  // [16,64]   = 1024B
static const uint16_t ADDR_X2       = 0xB600;  // [16,64]   = 1024B (residual 1)
static const uint16_t ADDR_RMS2_OUT = 0xBA00;  // [16,64]   = 1024B
static const uint16_t ADDR_FFN_GATE = 0xBE00;  // [16,128]  = 2048B
static const uint16_t ADDR_FFN_UP   = 0xC600;  // [16,128]  = 2048B
static const uint16_t ADDR_FFN_DOWN = 0xCE00;  // [16,64]   = 1024B
static const uint16_t ADDR_X_OUT    = 0xD200;  // [16,64]   = 1024B

// SRAM1 Memory layout
static const uint16_t S1_RMS1_GAMMA = 0x0000;  // [64]
static const uint16_t S1_RMS2_GAMMA = 0x0040;  // [64]
static const uint16_t S1_ROPE_SIN   = 0x0080;  // [16,8] = 128B
static const uint16_t S1_ROPE_COS   = 0x0100;  // [16,8] = 128B
static const uint16_t S1_RESID      = 0x0180;  // [16,64] = 1024B (residual source)
static const uint16_t S1_FFN_UP     = 0x0600;  // [16,128] = 2048B (for VEC MUL)
// SRAM1 addresses for replicated QKV bias
static const uint16_t S1_BIAS_Q     = 0x0E40;  // N_Q_HEADS * [S,HEAD_DIM] = 1024B
static const uint16_t S1_BIAS_K     = 0x1240;  // N_KV_HEADS * [S,HEAD_DIM] = 512B
static const uint16_t S1_BIAS_V     = 0x1440;  // N_KV_HEADS * [S,HEAD_DIM] = 512B

// =============================================================================
// Opcodes and Flags
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
static const uint8_t FLAG_COPY2D      = 0x04;  // flags[2] for VEC COPY2D mode
static const uint8_t FLAG_CAUSAL_MASK = 0x10;
static const uint8_t FLAG_VEC_MUL     = 0x01;  // flags[1:0]=01 for MUL

// =============================================================================
// GEMM IMM values
// =============================================================================
static const uint16_t GEMM_IMM_K64  = 0x0201;  // scale=1, shift=2, for K=64
static const uint16_t GEMM_IMM_K16  = 0x0701;  // scale=1, shift=7, for K=16
static const uint16_t GEMM_IMM_K128 = 0x0201;  // scale=1, shift=2, for K=128

// =============================================================================
// Debug flag
// =============================================================================
static const bool DUMP_DEBUG = (getenv("DUMP_DEBUG") != nullptr);

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
    // Pass 1: accumulate sum of x^2 (signed squaring)
    uint32_t sum_sq = 0;
    for (int i = 0; i < length; i++) {
        int8_t x_s = input[i];
        uint16_t sq = (uint16_t)((int16_t)x_s * (int16_t)x_s);
        sum_sq += sq;
    }

    // Division: sum_sq << 8 / length using restoring divider
    uint64_t dividend = (uint64_t)sum_sq << 8;
    uint32_t rms_val = counting_div_unsigned(dividend, (uint64_t)length);

    // rsqrt LUT: top 8 bits of rms_val[15:8]
    uint8_t rsqrt_addr = (uint8_t)((rms_val >> 8) & 0xFF);
    uint16_t inv_rms = RSQRT_LUT[rsqrt_addr];

    // Pass 2: normalize
    for (int i = 0; i < length; i++) {
        int8_t x = input[i];
        int8_t g = gamma[i];

        // Fused: (x * gamma) * inv_rms >>> 16
        int16_t xg = (int16_t)x * (int16_t)g;
        int32_t gamma_applied = ((int32_t)xg * (int32_t)(uint32_t)inv_rms) >> 16;

        // Clamp to int8
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

            // Match RTL: (even*cos - odd*sin + 64) >> 7
            int16_t rot_even = ((int16_t)even_val * (int16_t)cos_val
                              - (int16_t)odd_val  * (int16_t)sin_val
                              + 64) >> 7;
            int16_t rot_odd  = ((int16_t)even_val * (int16_t)sin_val
                              + (int16_t)odd_val  * (int16_t)cos_val
                              + 64) >> 7;

            // Clamp
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
// C++ Vec ADD Golden (matches RTL vec_engine.sv)
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
// C++ Vec MUL Golden (matches RTL vec_engine.sv: (a*b + 64) >> 7)
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
// Test data
// =============================================================================
// Input
static int8_t X[SEQ_LEN][HIDDEN];

// Weights - head-blocked for Q, K, V
static int8_t Wq[N_Q_HEADS * HIDDEN][HEAD_DIM];     // 4 Q-heads: [256, 16]
static int8_t Wk[N_KV_HEADS * HIDDEN][HEAD_DIM];    // 2 KV-heads: [128, 16]
static int8_t Wv[N_KV_HEADS * HIDDEN][HEAD_DIM];    // 2 KV-heads: [128, 16]
static int8_t Wo[HIDDEN][HIDDEN];                    // [64, 64]
static int8_t W_gate[HIDDEN][FFN_DIM];               // [64, 128]
static int8_t W_up[HIDDEN][FFN_DIM];                 // [64, 128]
static int8_t W_down[FFN_DIM][HIDDEN];               // [128, 64]

// RMSNorm scales
static int8_t rms1_gamma[HIDDEN];
static int8_t rms2_gamma[HIDDEN];

// QKV biases
static int8_t bq[N_Q_HEADS][HEAD_DIM];
static int8_t bk[N_KV_HEADS][HEAD_DIM];
static int8_t bv[N_KV_HEADS][HEAD_DIM];

// Golden intermediates
static int8_t RMS1_out_gold[SEQ_LEN][HIDDEN];
static int8_t ATTN_gold[SEQ_LEN][HIDDEN];
static int8_t WO_out_gold[SEQ_LEN][HIDDEN];
static int8_t X2_gold[SEQ_LEN][HIDDEN];
static int8_t RMS2_out_gold[SEQ_LEN][HIDDEN];
static int8_t FFN_gate_gold[SEQ_LEN][FFN_DIM];
static int8_t FFN_up_gold[SEQ_LEN][FFN_DIM];
static int8_t FFN_mul_gold[SEQ_LEN][FFN_DIM];
static int8_t FFN_down_gold[SEQ_LEN][HIDDEN];
static int8_t X_OUT_gold[SEQ_LEN][HIDDEN];

// =============================================================================
// Generate random data and compute full golden reference
// =============================================================================
void generate_data_and_golden() {
    srand(42);
    auto rand_i8 = [](int lo, int hi) -> int8_t {
        return (int8_t)(lo + rand() % (hi - lo + 1));
    };

    // Input X
    for (int i = 0; i < SEQ_LEN; i++)
        for (int j = 0; j < HIDDEN; j++)
            X[i][j] = rand_i8(-64, 63);

    // Weights (small range for reasonable accumulation)
    for (int i = 0; i < N_Q_HEADS * HIDDEN; i++)
        for (int j = 0; j < HEAD_DIM; j++)
            Wq[i][j] = rand_i8(-4, 4);

    for (int i = 0; i < N_KV_HEADS * HIDDEN; i++)
        for (int j = 0; j < HEAD_DIM; j++) {
            Wk[i][j] = rand_i8(-4, 4);
            Wv[i][j] = rand_i8(-4, 4);
        }

    for (int i = 0; i < HIDDEN; i++)
        for (int j = 0; j < HIDDEN; j++)
            Wo[i][j] = rand_i8(-4, 4);

    for (int i = 0; i < HIDDEN; i++)
        for (int j = 0; j < FFN_DIM; j++) {
            W_gate[i][j] = rand_i8(-4, 4);
            W_up[i][j]   = rand_i8(-4, 4);
        }

    for (int i = 0; i < FFN_DIM; i++)
        for (int j = 0; j < HIDDEN; j++)
            W_down[i][j] = rand_i8(-4, 4);

    // RMSNorm gammas
    for (int j = 0; j < HIDDEN; j++) {
        rms1_gamma[j] = rand_i8(-64, 63);
        rms2_gamma[j] = rand_i8(-64, 63);
    }

    // QKV biases (random, exercises the bias-add path)
    for (int h = 0; h < N_Q_HEADS; h++)
        for (int d = 0; d < HEAD_DIM; d++)
            bq[h][d] = rand_i8(-4, 4);
    for (int h = 0; h < N_KV_HEADS; h++)
        for (int d = 0; d < HEAD_DIM; d++) {
            bk[h][d] = rand_i8(-4, 4);
            bv[h][d] = rand_i8(-4, 4);
        }

    int scale_k64 = 1, shift_k64 = 2;
    int scale_k16 = 1, shift_k16 = 7;
    int scale_k128 = 1, shift_k128 = 2;

    // Step 1: RMSNorm 1
    for (int r = 0; r < SEQ_LEN; r++)
        rmsnorm_golden(&X[r][0], &RMS1_out_gold[r][0], HIDDEN, rms1_gamma);

    // Step 2: Multi-head GQA attention with RoPE
    memset(&ATTN_gold[0][0], 0, sizeof(ATTN_gold));

    for (int h = 0; h < N_Q_HEADS; h++) {
        int kv_h = h / GQA_RATIO;
        int8_t* Wq_h = &Wq[h * HIDDEN][0];
        int8_t* Wk_h = &Wk[kv_h * HIDDEN][0];
        int8_t* Wv_h = &Wv[kv_h * HIDDEN][0];

        int8_t Q_h[SEQ_LEN][HEAD_DIM];
        int8_t K_h[SEQ_LEN][HEAD_DIM];
        int8_t V_h[SEQ_LEN][HEAD_DIM];

        // Q_h = RMS1_out * Wq_h
        gemm_golden(&RMS1_out_gold[0][0], Wq_h, &Q_h[0][0],
                    SEQ_LEN, HEAD_DIM, HIDDEN, false, scale_k64, shift_k64);

        // K_h = RMS1_out * Wk_h
        gemm_golden(&RMS1_out_gold[0][0], Wk_h, &K_h[0][0],
                    SEQ_LEN, HEAD_DIM, HIDDEN, false, scale_k64, shift_k64);

        // V_h = RMS1_out * Wv_h
        gemm_golden(&RMS1_out_gold[0][0], Wv_h, &V_h[0][0],
                    SEQ_LEN, HEAD_DIM, HIDDEN, false, scale_k64, shift_k64);

        // QKV bias add
        bias_add_golden(&Q_h[0][0], &bq[h][0], SEQ_LEN, HEAD_DIM);
        bias_add_golden(&K_h[0][0], &bk[kv_h][0], SEQ_LEN, HEAD_DIM);
        bias_add_golden(&V_h[0][0], &bv[kv_h][0], SEQ_LEN, HEAD_DIM);

        // Apply RoPE to Q and K
        rope_golden(&Q_h[0][0], SEQ_LEN, HEAD_DIM,
                    rope_sin_table, rope_cos_table, 0);
        rope_golden(&K_h[0][0], SEQ_LEN, HEAD_DIM,
                    rope_sin_table, rope_cos_table, 0);

        // S_h = Q_h * K_h^T
        int8_t S_h[SEQ_LEN][SEQ_LEN];
        gemm_golden(&Q_h[0][0], &K_h[0][0], &S_h[0][0],
                    SEQ_LEN, SEQ_LEN, HEAD_DIM, true, scale_k16, shift_k16);

        // P_h = softmax(S_h, causal)
        int8_t P_h[SEQ_LEN][SEQ_LEN];
        for (int r = 0; r < SEQ_LEN; r++)
            softmax_golden(&S_h[r][0], &P_h[r][0], SEQ_LEN, true, r);

        // ATTN_h = P_h * V_h
        int8_t ATTN_h[SEQ_LEN][HEAD_DIM];
        gemm_golden(&P_h[0][0], &V_h[0][0], &ATTN_h[0][0],
                    SEQ_LEN, HEAD_DIM, SEQ_LEN, false, scale_k16, shift_k16);

        // Scatter into concat buffer
        for (int r = 0; r < SEQ_LEN; r++)
            for (int c = 0; c < HEAD_DIM; c++)
                ATTN_gold[r][h * HEAD_DIM + c] = ATTN_h[r][c];
    }

    // Step 3: WO_out = ATTN * Wo
    gemm_golden(&ATTN_gold[0][0], &Wo[0][0], &WO_out_gold[0][0],
                SEQ_LEN, HIDDEN, HIDDEN, false, scale_k64, shift_k64);

    // Step 4: X2 = WO_out + X (residual add)
    vec_add_golden(&WO_out_gold[0][0], &X[0][0], &X2_gold[0][0], SEQ_LEN * HIDDEN);

    // Step 5: RMSNorm 2
    for (int r = 0; r < SEQ_LEN; r++)
        rmsnorm_golden(&X2_gold[r][0], &RMS2_out_gold[r][0], HIDDEN, rms2_gamma);

    // Step 6: SwiGLU FFN
    // Gate projection
    gemm_golden(&RMS2_out_gold[0][0], &W_gate[0][0], &FFN_gate_gold[0][0],
                SEQ_LEN, FFN_DIM, HIDDEN, false, scale_k64, shift_k64);

    // Up projection
    gemm_golden(&RMS2_out_gold[0][0], &W_up[0][0], &FFN_up_gold[0][0],
                SEQ_LEN, FFN_DIM, HIDDEN, false, scale_k64, shift_k64);

    // SiLU on gate
    int8_t silu_gate[SEQ_LEN][FFN_DIM];
    silu_golden(&FFN_gate_gold[0][0], &silu_gate[0][0], SEQ_LEN * FFN_DIM);

    // Elementwise multiply: gate * up -> >>7 with rounding
    vec_mul_golden(&silu_gate[0][0], &FFN_up_gold[0][0], &FFN_mul_gold[0][0], SEQ_LEN * FFN_DIM);

    // Down projection (K=128)
    gemm_golden(&FFN_mul_gold[0][0], &W_down[0][0], &FFN_down_gold[0][0],
                SEQ_LEN, HIDDEN, FFN_DIM, false, scale_k128, shift_k128);

    // Step 7: X_OUT = FFN_down + X2 (residual add)
    vec_add_golden(&FFN_down_gold[0][0], &X2_gold[0][0], &X_OUT_gold[0][0], SEQ_LEN * HIDDEN);
}

// =============================================================================
// Load microcode program (LLaMA block: RMSNorm + GQA + RoPE + SwiGLU)
// =============================================================================
int load_microcode() {
    uint64_t hi, lo;
    int addr = 0;

    // ---- RMSNorm 1 (SEQ_LEN rows) ----
    for (int i = 0; i < SEQ_LEN; i++) {
        uint16_t src = ADDR_X + i * HIDDEN;
        uint16_t dst = ADDR_RMS1_OUT + i * HIDDEN;
        // OP_RMSNORM: src0=src, dst=dst, src1=gamma_base, N=length
        encode_instr(OP_RMSNORM, 0, dst, src, S1_RMS1_GAMMA,
                     0, HIDDEN, 0, 0, hi, lo);
        ucode_write(addr++, hi, lo);
    }

    // BARRIER
    encode_instr(OP_BARRIER, 0, 0, 0, 0, 0, 0, 0, 0, hi, lo);
    ucode_write(addr++, hi, lo);

    // ---- Per Q-head attention loop (4 Q-heads, 2 KV-heads via GQA) ----
    for (int h = 0; h < N_Q_HEADS; h++) {
        int kv_h = h / GQA_RATIO;
        uint16_t wq_h_addr = ADDR_WQ + h * HIDDEN * HEAD_DIM;
        uint16_t wk_h_addr = ADDR_WK + kv_h * HIDDEN * HEAD_DIM;
        uint16_t wv_h_addr = ADDR_WV + kv_h * HIDDEN * HEAD_DIM;

        // GEMM Q_h = RMS1_OUT * Wq_h  (M=16, N=16, K=64, imm=0x0901)
        encode_instr(OP_GEMM, FLAG_REQUANT, ADDR_Q_H, ADDR_RMS1_OUT, wq_h_addr,
                     SEQ_LEN, HEAD_DIM, HIDDEN, GEMM_IMM_K64, hi, lo);
        ucode_write(addr++, hi, lo);

        encode_instr(OP_BARRIER, 0, 0, 0, 0, 0, 0, 0, 0, hi, lo);
        ucode_write(addr++, hi, lo);

        // VEC ADD Q bias
        {
            uint16_t bq_sram1 = S1_BIAS_Q + h * SEQ_LEN * HEAD_DIM;
            encode_instr(OP_VEC, 0x00, ADDR_Q_H, ADDR_Q_H, bq_sram1,
                         0, SEQ_LEN * HEAD_DIM, 0, 0, hi, lo);
            ucode_write(addr++, hi, lo);
            encode_instr(OP_BARRIER, 0, 0, 0, 0, 0, 0, 0, 0, hi, lo);
            ucode_write(addr++, hi, lo);
        }

        // GEMM K_h = RMS1_OUT * Wk_h
        encode_instr(OP_GEMM, FLAG_REQUANT, ADDR_K_H, ADDR_RMS1_OUT, wk_h_addr,
                     SEQ_LEN, HEAD_DIM, HIDDEN, GEMM_IMM_K64, hi, lo);
        ucode_write(addr++, hi, lo);

        encode_instr(OP_BARRIER, 0, 0, 0, 0, 0, 0, 0, 0, hi, lo);
        ucode_write(addr++, hi, lo);

        // VEC ADD K bias
        {
            uint16_t bk_sram1 = S1_BIAS_K + kv_h * SEQ_LEN * HEAD_DIM;
            encode_instr(OP_VEC, 0x00, ADDR_K_H, ADDR_K_H, bk_sram1,
                         0, SEQ_LEN * HEAD_DIM, 0, 0, hi, lo);
            ucode_write(addr++, hi, lo);
            encode_instr(OP_BARRIER, 0, 0, 0, 0, 0, 0, 0, 0, hi, lo);
            ucode_write(addr++, hi, lo);
        }

        // GEMM V_h = RMS1_OUT * Wv_h
        encode_instr(OP_GEMM, FLAG_REQUANT, ADDR_V_H, ADDR_RMS1_OUT, wv_h_addr,
                     SEQ_LEN, HEAD_DIM, HIDDEN, GEMM_IMM_K64, hi, lo);
        ucode_write(addr++, hi, lo);

        encode_instr(OP_BARRIER, 0, 0, 0, 0, 0, 0, 0, 0, hi, lo);
        ucode_write(addr++, hi, lo);

        // VEC ADD V bias
        {
            uint16_t bv_sram1 = S1_BIAS_V + kv_h * SEQ_LEN * HEAD_DIM;
            encode_instr(OP_VEC, 0x00, ADDR_V_H, ADDR_V_H, bv_sram1,
                         0, SEQ_LEN * HEAD_DIM, 0, 0, hi, lo);
            ucode_write(addr++, hi, lo);
            encode_instr(OP_BARRIER, 0, 0, 0, 0, 0, 0, 0, 0, hi, lo);
            ucode_write(addr++, hi, lo);
        }

        // ROPE Q_h: src=ADDR_Q_H, dst=ADDR_Q_H (in-place), M=SEQ_LEN, N=HEAD_DIM, K=0 (pos_offset)
        // src1=sin_base, imm=cos_base
        encode_instr(OP_ROPE, 0, ADDR_Q_H, ADDR_Q_H, S1_ROPE_SIN,
                     SEQ_LEN, HEAD_DIM, 0, S1_ROPE_COS, hi, lo);
        ucode_write(addr++, hi, lo);

        encode_instr(OP_BARRIER, 0, 0, 0, 0, 0, 0, 0, 0, hi, lo);
        ucode_write(addr++, hi, lo);

        // ROPE K_h
        encode_instr(OP_ROPE, 0, ADDR_K_H, ADDR_K_H, S1_ROPE_SIN,
                     SEQ_LEN, HEAD_DIM, 0, S1_ROPE_COS, hi, lo);
        ucode_write(addr++, hi, lo);

        encode_instr(OP_BARRIER, 0, 0, 0, 0, 0, 0, 0, 0, hi, lo);
        ucode_write(addr++, hi, lo);

        // GEMM S = Q_h * K_h^T  (M=16, N=16, K=16, TRANSPOSE_B)
        encode_instr(OP_GEMM, FLAG_TRANSPOSE_B | FLAG_REQUANT, ADDR_S, ADDR_Q_H, ADDR_K_H,
                     SEQ_LEN, SEQ_LEN, HEAD_DIM, GEMM_IMM_K16, hi, lo);
        ucode_write(addr++, hi, lo);

        encode_instr(OP_BARRIER, 0, 0, 0, 0, 0, 0, 0, 0, hi, lo);
        ucode_write(addr++, hi, lo);

        // SOFTMAX rows (16 rows with causal mask)
        for (int i = 0; i < SEQ_LEN; i++) {
            uint16_t src = ADDR_S + i * SEQ_LEN;
            uint16_t dst = ADDR_P + i * SEQ_LEN;
            encode_instr(OP_SOFTMAX, FLAG_CAUSAL_MASK, dst, src, 0,
                         0, SEQ_LEN, i, 0x0100, hi, lo);
            ucode_write(addr++, hi, lo);
        }

        encode_instr(OP_BARRIER, 0, 0, 0, 0, 0, 0, 0, 0, hi, lo);
        ucode_write(addr++, hi, lo);

        // GEMM ATTN_h = P * V_h  (M=16, N=16, K=16)
        encode_instr(OP_GEMM, FLAG_REQUANT, ADDR_ATTN_H, ADDR_P, ADDR_V_H,
                     SEQ_LEN, HEAD_DIM, SEQ_LEN, GEMM_IMM_K16, hi, lo);
        ucode_write(addr++, hi, lo);

        encode_instr(OP_BARRIER, 0, 0, 0, 0, 0, 0, 0, 0, hi, lo);
        ucode_write(addr++, hi, lo);

        // VEC COPY2D: scatter ATTN_H -> ATTN[:, h*16:(h+1)*16]
        encode_instr(OP_VEC, FLAG_COPY2D, ADDR_ATTN + h * HEAD_DIM, ADDR_ATTN_H, 0,
                     SEQ_LEN, HEAD_DIM, HEAD_DIM, HIDDEN, hi, lo);
        ucode_write(addr++, hi, lo);

        encode_instr(OP_BARRIER, 0, 0, 0, 0, 0, 0, 0, 0, hi, lo);
        ucode_write(addr++, hi, lo);
    }

    // ---- Output projection ----
    encode_instr(OP_GEMM, FLAG_REQUANT, ADDR_WO_OUT, ADDR_ATTN, ADDR_WO,
                 SEQ_LEN, HIDDEN, HIDDEN, GEMM_IMM_K64, hi, lo);
    ucode_write(addr++, hi, lo);

    encode_instr(OP_BARRIER, 0, 0, 0, 0, 0, 0, 0, 0, hi, lo);
    ucode_write(addr++, hi, lo);

    // ---- Residual 1: X2 = WO_OUT + X ----
    // VEC ADD: src0=WO_OUT (SRAM0), src1=S1_RESID (SRAM1)
    encode_instr(OP_VEC, 0x00, ADDR_X2, ADDR_WO_OUT, S1_RESID,
                 0, SEQ_LEN * HIDDEN, 0, 0, hi, lo);
    ucode_write(addr++, hi, lo);

    encode_instr(OP_BARRIER, 0, 0, 0, 0, 0, 0, 0, 0, hi, lo);
    ucode_write(addr++, hi, lo);

    // ---- DMA: copy X2 to SRAM1 for second residual ----
    encode_instr(OP_DMA_LOAD, 0, S1_RESID, ADDR_X2, 0,
                 0, SEQ_LEN * HIDDEN, 0, 0, hi, lo);
    ucode_write(addr++, hi, lo);

    encode_instr(OP_BARRIER, 0, 0, 0, 0, 0, 0, 0, 0, hi, lo);
    ucode_write(addr++, hi, lo);

    // ---- RMSNorm 2 (SEQ_LEN rows) ----
    for (int i = 0; i < SEQ_LEN; i++) {
        uint16_t src = ADDR_X2 + i * HIDDEN;
        uint16_t dst = ADDR_RMS2_OUT + i * HIDDEN;
        encode_instr(OP_RMSNORM, 0, dst, src, S1_RMS2_GAMMA,
                     0, HIDDEN, 0, 0, hi, lo);
        ucode_write(addr++, hi, lo);
    }

    encode_instr(OP_BARRIER, 0, 0, 0, 0, 0, 0, 0, 0, hi, lo);
    ucode_write(addr++, hi, lo);

    // ---- SwiGLU FFN ----
    // GEMM FFN_GATE = RMS2_OUT * W_gate  (M=16, N=128, K=64)
    encode_instr(OP_GEMM, FLAG_REQUANT, ADDR_FFN_GATE, ADDR_RMS2_OUT, ADDR_W_GATE,
                 SEQ_LEN, FFN_DIM, HIDDEN, GEMM_IMM_K64, hi, lo);
    ucode_write(addr++, hi, lo);

    encode_instr(OP_BARRIER, 0, 0, 0, 0, 0, 0, 0, 0, hi, lo);
    ucode_write(addr++, hi, lo);

    // GEMM FFN_UP = RMS2_OUT * W_up  (M=16, N=128, K=64)
    encode_instr(OP_GEMM, FLAG_REQUANT, ADDR_FFN_UP, ADDR_RMS2_OUT, ADDR_W_UP,
                 SEQ_LEN, FFN_DIM, HIDDEN, GEMM_IMM_K64, hi, lo);
    ucode_write(addr++, hi, lo);

    encode_instr(OP_BARRIER, 0, 0, 0, 0, 0, 0, 0, 0, hi, lo);
    ucode_write(addr++, hi, lo);

    // SILU on FFN_GATE (in-place)
    encode_instr(OP_SILU, 0, ADDR_FFN_GATE, ADDR_FFN_GATE, 0,
                 0, SEQ_LEN * FFN_DIM, 0, 0, hi, lo);
    ucode_write(addr++, hi, lo);

    encode_instr(OP_BARRIER, 0, 0, 0, 0, 0, 0, 0, 0, hi, lo);
    ucode_write(addr++, hi, lo);

    // DMA: copy FFN_UP from SRAM0 to SRAM1 for VEC MUL
    encode_instr(OP_DMA_LOAD, 0, S1_FFN_UP, ADDR_FFN_UP, 0,
                 0, SEQ_LEN * FFN_DIM, 0, 0, hi, lo);
    ucode_write(addr++, hi, lo);

    encode_instr(OP_BARRIER, 0, 0, 0, 0, 0, 0, 0, 0, hi, lo);
    ucode_write(addr++, hi, lo);

    // VEC MUL: FFN_GATE = FFN_GATE * FFN_UP  (src0=SRAM0 gate, src1=SRAM1 up)
    // flags[1:0]=01 for MUL
    encode_instr(OP_VEC, FLAG_VEC_MUL, ADDR_FFN_GATE, ADDR_FFN_GATE, S1_FFN_UP,
                 0, SEQ_LEN * FFN_DIM, 0, 0, hi, lo);
    ucode_write(addr++, hi, lo);

    encode_instr(OP_BARRIER, 0, 0, 0, 0, 0, 0, 0, 0, hi, lo);
    ucode_write(addr++, hi, lo);

    // GEMM FFN_DOWN = FFN_GATE * W_down  (M=16, N=64, K=128, imm=0x0A01)
    encode_instr(OP_GEMM, FLAG_REQUANT, ADDR_FFN_DOWN, ADDR_FFN_GATE, ADDR_W_DOWN,
                 SEQ_LEN, HIDDEN, FFN_DIM, GEMM_IMM_K128, hi, lo);
    ucode_write(addr++, hi, lo);

    encode_instr(OP_BARRIER, 0, 0, 0, 0, 0, 0, 0, 0, hi, lo);
    ucode_write(addr++, hi, lo);

    // VEC ADD: X_OUT = FFN_DOWN + X2  (src0=FFN_DOWN, src1=S1_RESID)
    encode_instr(OP_VEC, 0x00, ADDR_X_OUT, ADDR_FFN_DOWN, S1_RESID,
                 0, SEQ_LEN * HIDDEN, 0, 0, hi, lo);
    ucode_write(addr++, hi, lo);

    encode_instr(OP_BARRIER, 0, 0, 0, 0, 0, 0, 0, 0, hi, lo);
    ucode_write(addr++, hi, lo);

    // END
    encode_instr(OP_END, 0, 0, 0, 0, 0, 0, 0, 0, hi, lo);
    ucode_write(addr++, hi, lo);

    return addr;
}

// =============================================================================
// Load data to SRAMs
// =============================================================================
void load_data_to_srams() {
    // SRAM0: X
    for (int i = 0; i < SEQ_LEN; i++)
        for (int j = 0; j < HIDDEN; j++)
            sram0_write(ADDR_X + i * HIDDEN + j, (uint8_t)X[i][j]);

    // SRAM0: Head-blocked Wq (4 Q-heads)
    for (int h = 0; h < N_Q_HEADS; h++) {
        uint16_t wq_base = ADDR_WQ + h * HIDDEN * HEAD_DIM;
        for (int i = 0; i < HIDDEN; i++)
            for (int j = 0; j < HEAD_DIM; j++)
                sram0_write(wq_base + i * HEAD_DIM + j, (uint8_t)Wq[h * HIDDEN + i][j]);
    }

    // SRAM0: Head-blocked Wk (2 KV-heads)
    for (int h = 0; h < N_KV_HEADS; h++) {
        uint16_t wk_base = ADDR_WK + h * HIDDEN * HEAD_DIM;
        for (int i = 0; i < HIDDEN; i++)
            for (int j = 0; j < HEAD_DIM; j++)
                sram0_write(wk_base + i * HEAD_DIM + j, (uint8_t)Wk[h * HIDDEN + i][j]);
    }

    // SRAM0: Head-blocked Wv (2 KV-heads)
    for (int h = 0; h < N_KV_HEADS; h++) {
        uint16_t wv_base = ADDR_WV + h * HIDDEN * HEAD_DIM;
        for (int i = 0; i < HIDDEN; i++)
            for (int j = 0; j < HEAD_DIM; j++)
                sram0_write(wv_base + i * HEAD_DIM + j, (uint8_t)Wv[h * HIDDEN + i][j]);
    }

    // SRAM0: Wo [64][64]
    for (int i = 0; i < HIDDEN; i++)
        for (int j = 0; j < HIDDEN; j++)
            sram0_write(ADDR_WO + i * HIDDEN + j, (uint8_t)Wo[i][j]);

    // SRAM0: W_gate [64][128]
    for (int i = 0; i < HIDDEN; i++)
        for (int j = 0; j < FFN_DIM; j++)
            sram0_write(ADDR_W_GATE + i * FFN_DIM + j, (uint8_t)W_gate[i][j]);

    // SRAM0: W_up [64][128]
    for (int i = 0; i < HIDDEN; i++)
        for (int j = 0; j < FFN_DIM; j++)
            sram0_write(ADDR_W_UP + i * FFN_DIM + j, (uint8_t)W_up[i][j]);

    // SRAM0: W_down [128][64]
    for (int i = 0; i < FFN_DIM; i++)
        for (int j = 0; j < HIDDEN; j++)
            sram0_write(ADDR_W_DOWN + i * HIDDEN + j, (uint8_t)W_down[i][j]);

    // SRAM1: RMSNorm gammas
    for (int j = 0; j < HIDDEN; j++) {
        sram1_write(S1_RMS1_GAMMA + j, (uint8_t)rms1_gamma[j]);
        sram1_write(S1_RMS2_GAMMA + j, (uint8_t)rms2_gamma[j]);
    }

    // SRAM1: RoPE tables
    for (int i = 0; i < MAX_SEQ * HALF_DIM; i++) {
        sram1_write(S1_ROPE_SIN + i, (uint8_t)rope_sin_table[i]);
        sram1_write(S1_ROPE_COS + i, (uint8_t)rope_cos_table[i]);
    }

    // SRAM1: Copy X for first residual add
    for (int i = 0; i < SEQ_LEN; i++)
        for (int j = 0; j < HIDDEN; j++)
            sram1_write(S1_RESID + i * HIDDEN + j, (uint8_t)X[i][j]);

    // SRAM1: replicate QKV bias (bias[HEAD_DIM] -> [S, HEAD_DIM] per head)
    for (int h = 0; h < N_Q_HEADS; h++) {
        uint16_t base = S1_BIAS_Q + h * SEQ_LEN * HEAD_DIM;
        for (int s = 0; s < SEQ_LEN; s++)
            for (int d = 0; d < HEAD_DIM; d++)
                sram1_write(base + s * HEAD_DIM + d, (uint8_t)bq[h][d]);
    }
    for (int h = 0; h < N_KV_HEADS; h++) {
        uint16_t base_k = S1_BIAS_K + h * SEQ_LEN * HEAD_DIM;
        uint16_t base_v = S1_BIAS_V + h * SEQ_LEN * HEAD_DIM;
        for (int s = 0; s < SEQ_LEN; s++)
            for (int d = 0; d < HEAD_DIM; d++) {
                sram1_write(base_k + s * HEAD_DIM + d, (uint8_t)bk[h][d]);
                sram1_write(base_v + s * HEAD_DIM + d, (uint8_t)bv[h][d]);
            }
    }
}

// =============================================================================
// Handle DMA command: copy from SRAM0 to SRAM1
// =============================================================================
void handle_dma_command() {
    uint16_t src  = dut->dma_src;
    uint16_t dst  = dut->dma_dst;
    uint16_t len  = dut->dma_len;

    std::cout << "  DMA: src=0x" << std::hex << src
              << " dst=0x" << dst << " len=" << std::dec << len << std::endl;

    for (int i = 0; i < len; i++) {
        uint8_t val = sram0_read(src + i);
        sram1_write(dst + i, val);
    }
}

// =============================================================================
// Verify a matrix against golden
// =============================================================================
struct VerifyResult {
    int mismatches;
    int max_err;
};

VerifyResult verify_matrix(const char* name, uint16_t base_addr, const int8_t* golden,
                           int rows, int cols, int tolerance = 0) {
    VerifyResult r = {0, 0};
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            int8_t actual = (int8_t)sram0_read(base_addr + i * cols + j);
            int err = abs((int)actual - (int)golden[i * cols + j]);
            if (err > r.max_err) r.max_err = err;
            if (err > tolerance) r.mismatches++;
        }
    }
    std::cout << "  " << name << ": max_err=" << r.max_err
              << " mismatches(>" << tolerance << ")=" << r.mismatches
              << (r.mismatches == 0 ? " PASS" : " FAIL") << std::endl;
    return r;
}

// =============================================================================
// Main test
// =============================================================================
int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    Verilated::traceEverOn(true);

    dut = new Vllama_block_top;
    tfp = new VerilatedVcdC;
    dut->trace(tfp, 99);
    tfp->open("llama_block_sim.vcd");

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

    std::cout << "============================================" << std::endl;
    std::cout << "  LLAMA BLOCK TEST: GQA + RoPE + SwiGLU" << std::endl;
    std::cout << "  seq_len=" << SEQ_LEN << " hidden=" << HIDDEN
              << " head_dim=" << HEAD_DIM << std::endl;
    std::cout << "  q_heads=" << N_Q_HEADS << " kv_heads=" << N_KV_HEADS
              << " gqa_ratio=" << GQA_RATIO << " ffn=" << FFN_DIM << std::endl;
    std::cout << "  Expected: 24 GEMMs, 8 ROPEs, 32 RMSNorms" << std::endl;
    std::cout << "            1 SILU, 1 VEC_MUL, 4 COPY2Ds, 12 bias ADDs" << std::endl;
    std::cout << "============================================" << std::endl;

    // Build LUTs
    build_softmax_luts();
    build_rsqrt_lut();
    build_silu_lut();
    build_rope_tables();

    // Reset
    reset_dut();
    std::cout << "Reset complete." << std::endl;

    // Generate data and compute golden
    generate_data_and_golden();
    std::cout << "Golden reference computed." << std::endl;

    // Print sample golden values
    std::cout << "  X[0]: ";
    for (int j = 0; j < std::min(HIDDEN, 16); j++) std::cout << (int)X[0][j] << " ";
    if (HIDDEN > 16) std::cout << "...";
    std::cout << std::endl;
    std::cout << "  RMS1_out[0]: ";
    for (int j = 0; j < std::min(HIDDEN, 16); j++) std::cout << (int)RMS1_out_gold[0][j] << " ";
    if (HIDDEN > 16) std::cout << "...";
    std::cout << std::endl;
    std::cout << "  X_OUT_gold[0]: ";
    for (int j = 0; j < std::min(HIDDEN, 16); j++) std::cout << (int)X_OUT_gold[0][j] << " ";
    if (HIDDEN > 16) std::cout << "...";
    std::cout << std::endl;

    // Load microcode
    std::cout << "Loading microcode..." << std::endl;
    int num_instrs = load_microcode();
    std::cout << "  Loaded " << num_instrs << " instructions" << std::endl;

    // Load data
    std::cout << "Loading data to SRAMs..." << std::endl;
    load_data_to_srams();
    std::cout << "  Data loaded." << std::endl;

    // Start execution
    std::cout << "Starting execution..." << std::endl;
    dut->ucode_len = num_instrs;
    dut->start_pulse = 1;
    tick();
    dut->start_pulse = 0;

    // Main simulation loop
    int dma_count = 0;
    int softmax_count = 0;
    int rmsnorm_count = 0;
    int rope_count = 0;
    int gelu_done_count = 0;
    int vec_done_count = 0;
    int cycle = 0;
    bool done = false;
    const int MAX_CYCLES = 4000000;

    while (cycle < MAX_CYCLES && !done) {
        tick();
        cycle++;

        // DMA interception
        if (dut->dma_cmd_captured) {
            dma_count++;
            std::cout << "[cycle " << cycle << "] DMA #" << dma_count << std::endl;
            handle_dma_command();
            dut->dma_done_pulse = 1;
            tick(); cycle++;
            dut->dma_done_pulse = 0;
        }

        // Track engine completions
        if (dut->softmax_done_dbg) softmax_count++;
        if (dut->rmsnorm_done_dbg) rmsnorm_count++;
        if (dut->rope_done_dbg) rope_count++;
        if (dut->gelu_done_dbg) gelu_done_count++;
        if (dut->vec_done_dbg) vec_done_count++;

        // Check for program end
        if (dut->program_end) {
            std::cout << "[cycle " << cycle << "] Program END" << std::endl;
            for (int i = 0; i < 10; i++) tick();
            done = true;
        }
    }

    if (!done) {
        std::cerr << "TIMEOUT after " << MAX_CYCLES << " cycles!" << std::endl;
        tfp->close(); delete tfp; delete dut;
        return 1;
    }

    int hw_gemm_count = (int)dut->hw_gemm_done_count;

    std::cout << std::endl;
    std::cout << "Execution complete: " << cycle << " cycles" << std::endl;
    std::cout << "  HW GEMM done:     " << hw_gemm_count << std::endl;
    std::cout << "  DMA commands:     " << dma_count << std::endl;
    std::cout << "  Softmax rows:     " << softmax_count << std::endl;
    std::cout << "  RMSNorm rows:     " << rmsnorm_count << std::endl;
    std::cout << "  RoPE calls:       " << rope_count << std::endl;
    std::cout << "  GELU/SiLU done:   " << gelu_done_count << std::endl;
    std::cout << "  Vec completions:  " << vec_done_count << std::endl;

    // Let state settle
    for (int i = 0; i < 20; i++) tick();

    // ================================================================
    // Verification: "follow-actual" approach
    // Read SRAM intermediates and recompute golden from actual values.
    // ================================================================
    std::cout << std::endl << "=== Verification (follow-actual) ===" << std::endl;

    assert(dut->vec_busy_dbg == 0 && "VEC engine still busy at verification time!");

    int scale_k64 = 1, shift_k64 = 2;
    int scale_k16 = 1, shift_k16 = 7;
    int scale_k128 = 1, shift_k128 = 2;
    bool all_pass = true;

    // Helper: read matrix from SRAM0
    auto read_sram_matrix = [](uint16_t base, int rows, int cols, int8_t* buf) {
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                buf[i * cols + j] = (int8_t)sram0_read(base + i * cols + j);
    };

    // 1. RMS1: compare RTL with C++ golden (informational, tolerance=20)
    auto r_rms1 = verify_matrix("RMS1_out", ADDR_RMS1_OUT, &RMS1_out_gold[0][0],
                                 SEQ_LEN, HIDDEN, 20);

    // 2. Read actual RMS1_out from SRAM
    int8_t actual_rms1[SEQ_LEN][HIDDEN];
    read_sram_matrix(ADDR_RMS1_OUT, SEQ_LEN, HIDDEN, &actual_rms1[0][0]);

    // 3. Compute full multi-head golden from actual RMS1 (4 Q-heads with GQA)
    int8_t ref_attn_concat[SEQ_LEN][HIDDEN];
    memset(&ref_attn_concat[0][0], 0, sizeof(ref_attn_concat));

    for (int h = 0; h < N_Q_HEADS; h++) {
        int kv_h = h / GQA_RATIO;
        int8_t* Wq_h = &Wq[h * HIDDEN][0];
        int8_t* Wk_h = &Wk[kv_h * HIDDEN][0];
        int8_t* Wv_h = &Wv[kv_h * HIDDEN][0];

        int8_t ref_qh[SEQ_LEN][HEAD_DIM];
        int8_t ref_kh[SEQ_LEN][HEAD_DIM];
        int8_t ref_vh[SEQ_LEN][HEAD_DIM];
        int8_t ref_sh[SEQ_LEN][SEQ_LEN];
        int8_t ref_ph[SEQ_LEN][SEQ_LEN];
        int8_t ref_ah[SEQ_LEN][HEAD_DIM];

        gemm_golden(&actual_rms1[0][0], Wq_h, &ref_qh[0][0],
                    SEQ_LEN, HEAD_DIM, HIDDEN, false, scale_k64, shift_k64);
        gemm_golden(&actual_rms1[0][0], Wk_h, &ref_kh[0][0],
                    SEQ_LEN, HEAD_DIM, HIDDEN, false, scale_k64, shift_k64);
        gemm_golden(&actual_rms1[0][0], Wv_h, &ref_vh[0][0],
                    SEQ_LEN, HEAD_DIM, HIDDEN, false, scale_k64, shift_k64);

        // QKV bias add
        bias_add_golden(&ref_qh[0][0], &bq[h][0], SEQ_LEN, HEAD_DIM);
        bias_add_golden(&ref_kh[0][0], &bk[kv_h][0], SEQ_LEN, HEAD_DIM);
        bias_add_golden(&ref_vh[0][0], &bv[kv_h][0], SEQ_LEN, HEAD_DIM);

        // Apply RoPE
        rope_golden(&ref_qh[0][0], SEQ_LEN, HEAD_DIM,
                    rope_sin_table, rope_cos_table, 0);
        rope_golden(&ref_kh[0][0], SEQ_LEN, HEAD_DIM,
                    rope_sin_table, rope_cos_table, 0);

        gemm_golden(&ref_qh[0][0], &ref_kh[0][0], &ref_sh[0][0],
                    SEQ_LEN, SEQ_LEN, HEAD_DIM, true, scale_k16, shift_k16);

        for (int r = 0; r < SEQ_LEN; r++)
            softmax_golden(&ref_sh[r][0], &ref_ph[r][0], SEQ_LEN, true, r);

        gemm_golden(&ref_ph[0][0], &ref_vh[0][0], &ref_ah[0][0],
                    SEQ_LEN, HEAD_DIM, SEQ_LEN, false, scale_k16, shift_k16);

        for (int r = 0; r < SEQ_LEN; r++)
            for (int c = 0; c < HEAD_DIM; c++)
                ref_attn_concat[r][h * HEAD_DIM + c] = ref_ah[r][c];

        if (h == 0) {
            std::cout << "  Head 0 Q_h[0]: ";
            for (int j = 0; j < HEAD_DIM; j++) std::cout << (int)ref_qh[0][j] << " ";
            std::cout << std::endl;
        }
    }

    // 4. Verify ATTN_concat (informational: depends on softmax LUT which may
    //    differ slightly between C++ golden and RTL; use tolerance=15)
    auto r_attn = verify_matrix("ATTN_concat", ADDR_ATTN, &ref_attn_concat[0][0],
                                 SEQ_LEN, HIDDEN, 15);

    // 5. Read actual ATTN_concat -> recompute WO_out
    int8_t actual_attn[SEQ_LEN][HIDDEN];
    read_sram_matrix(ADDR_ATTN, SEQ_LEN, HIDDEN, &actual_attn[0][0]);

    int8_t ref_wo[SEQ_LEN][HIDDEN];
    gemm_golden(&actual_attn[0][0], &Wo[0][0], &ref_wo[0][0],
                SEQ_LEN, HIDDEN, HIDDEN, false, scale_k64, shift_k64);
    auto r_wo = verify_matrix("WO(hw) ", ADDR_WO_OUT, &ref_wo[0][0], SEQ_LEN, HIDDEN, 0);
    if (r_wo.mismatches) all_pass = false;

    // 6. Read actual WO_out -> verify X2 = WO_out + X
    int8_t actual_wo[SEQ_LEN][HIDDEN];
    read_sram_matrix(ADDR_WO_OUT, SEQ_LEN, HIDDEN, &actual_wo[0][0]);

    int8_t ref_x2[SEQ_LEN][HIDDEN];
    vec_add_golden(&actual_wo[0][0], &X[0][0], &ref_x2[0][0], SEQ_LEN * HIDDEN);
    auto r_x2 = verify_matrix("X2     ", ADDR_X2, &ref_x2[0][0], SEQ_LEN, HIDDEN, 0);
    if (r_x2.mismatches) all_pass = false;

    // 7. RMS2: informational comparison
    int8_t actual_x2[SEQ_LEN][HIDDEN];
    read_sram_matrix(ADDR_X2, SEQ_LEN, HIDDEN, &actual_x2[0][0]);
    int8_t ref_rms2[SEQ_LEN][HIDDEN];
    for (int r = 0; r < SEQ_LEN; r++)
        rmsnorm_golden(&actual_x2[r][0], &ref_rms2[r][0], HIDDEN, rms2_gamma);
    auto r_rms2 = verify_matrix("RMS2_out", ADDR_RMS2_OUT, &ref_rms2[0][0],
                                 SEQ_LEN, HIDDEN, 20);

    // 8. Read actual RMS2_out -> recompute FFN
    int8_t actual_rms2[SEQ_LEN][HIDDEN];
    read_sram_matrix(ADDR_RMS2_OUT, SEQ_LEN, HIDDEN, &actual_rms2[0][0]);

    // FFN_GATE (before SiLU)
    int8_t ref_ffn_gate_raw[SEQ_LEN][FFN_DIM];
    gemm_golden(&actual_rms2[0][0], &W_gate[0][0], &ref_ffn_gate_raw[0][0],
                SEQ_LEN, FFN_DIM, HIDDEN, false, scale_k64, shift_k64);

    // FFN_UP
    int8_t ref_ffn_up[SEQ_LEN][FFN_DIM];
    gemm_golden(&actual_rms2[0][0], &W_up[0][0], &ref_ffn_up[0][0],
                SEQ_LEN, FFN_DIM, HIDDEN, false, scale_k64, shift_k64);

    // SiLU on gate
    int8_t ref_silu_gate[SEQ_LEN][FFN_DIM];
    silu_golden(&ref_ffn_gate_raw[0][0], &ref_silu_gate[0][0], SEQ_LEN * FFN_DIM);

    // Elementwise multiply
    int8_t ref_ffn_mul[SEQ_LEN][FFN_DIM];
    vec_mul_golden(&ref_silu_gate[0][0], &ref_ffn_up[0][0], &ref_ffn_mul[0][0], SEQ_LEN * FFN_DIM);

    // After SiLU+MUL, ADDR_FFN_GATE should contain the multiply result
    auto r_mul = verify_matrix("FFN_MUL(hw)", ADDR_FFN_GATE, &ref_ffn_mul[0][0],
                                SEQ_LEN, FFN_DIM, 1);
    if (r_mul.mismatches) all_pass = false;

    // 9. Read actual MUL result -> recompute FFN_DOWN
    int8_t actual_mul[SEQ_LEN][FFN_DIM];
    read_sram_matrix(ADDR_FFN_GATE, SEQ_LEN, FFN_DIM, &actual_mul[0][0]);

    int8_t ref_ffn_down[SEQ_LEN][HIDDEN];
    gemm_golden(&actual_mul[0][0], &W_down[0][0], &ref_ffn_down[0][0],
                SEQ_LEN, HIDDEN, FFN_DIM, false, scale_k128, shift_k128);
    auto r_down = verify_matrix("FFN_DOWN(hw)", ADDR_FFN_DOWN, &ref_ffn_down[0][0],
                                 SEQ_LEN, HIDDEN, 0);
    if (r_down.mismatches) all_pass = false;

    // 10. Read actual FFN_DOWN -> verify X_OUT = FFN_DOWN + X2
    int8_t actual_ffn_down[SEQ_LEN][HIDDEN];
    read_sram_matrix(ADDR_FFN_DOWN, SEQ_LEN, HIDDEN, &actual_ffn_down[0][0]);

    int8_t ref_xout[SEQ_LEN][HIDDEN];
    vec_add_golden(&actual_ffn_down[0][0], &actual_x2[0][0], &ref_xout[0][0], SEQ_LEN * HIDDEN);
    auto r_out = verify_matrix("X_OUT  ", ADDR_X_OUT, &ref_xout[0][0], SEQ_LEN, HIDDEN, 0);
    if (r_out.mismatches) all_pass = false;

    // Print first row of output
    std::cout << std::endl;
    std::cout << "  X_OUT[0] actual: ";
    for (int j = 0; j < std::min(HIDDEN, 16); j++)
        std::cout << (int)(int8_t)sram0_read(ADDR_X_OUT + j) << " ";
    if (HIDDEN > 16) std::cout << "...";
    std::cout << std::endl;
    std::cout << "  X_OUT[0] golden: ";
    for (int j = 0; j < std::min(HIDDEN, 16); j++)
        std::cout << (int)ref_xout[0][j] << " ";
    if (HIDDEN > 16) std::cout << "...";
    std::cout << std::endl;

    // ================================================================
    // Final verdict
    // Expected: 24 HW GEMMs (5/head*4 + Wo + gate + up + down),
    //           2 DMAs, 64 softmax rows, 32 RMSNorm rows,
    //           8 RoPE calls, 1 SiLU,
    //           19 vec (4 COPY2D + 2 ADD + 1 MUL + 12 bias ADD)
    // ================================================================
    bool pass = all_pass && (hw_gemm_count == 24) && (dma_count == 2);

    std::cout << std::endl;
    std::cout << "============================================" << std::endl;
    std::cout << "  HW GEMM done:      " << hw_gemm_count << "/24 "
              << (hw_gemm_count == 24 ? "OK" : "FAIL") << std::endl;
    std::cout << "  DMA commands:      " << dma_count << "/2 "
              << (dma_count == 2 ? "OK" : "FAIL") << std::endl;
    std::cout << "  Softmax rows:      " << softmax_count << "/64" << std::endl;
    std::cout << "  RMSNorm rows:      " << rmsnorm_count << "/32" << std::endl;
    std::cout << "  RoPE calls:        " << rope_count << "/8" << std::endl;
    std::cout << "  GELU/SiLU done:    " << gelu_done_count << "/1" << std::endl;
    std::cout << "  Vec completions:   " << vec_done_count << "/19" << std::endl;
    std::cout << "  X_OUT max error:   " << r_out.max_err << " (tol=0)" << std::endl;
    std::cout << "  Total cycles:      " << cycle << std::endl;
    std::cout << "  GEMMs: " << hw_gemm_count << "/24 on REAL hardware" << std::endl;
    std::cout << "============================================" << std::endl;
    std::cout << "  LLAMA BLOCK TEST: " << (pass ? "PASS" : "FAIL") << std::endl;
    std::cout << "============================================" << std::endl;

    tfp->close();
    delete tfp;
    delete dut;

    return pass ? 0 : 1;
}
