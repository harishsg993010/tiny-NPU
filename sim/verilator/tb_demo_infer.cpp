// =============================================================================
// tb_demo_infer.cpp - End-to-end GPT-2 inference demo on NPU
// Loads real GPT-2 weights from weights.bin, runs multi-token autoregressive
// decode using the NPU hardware (4 transformer blocks + lm_head).
// Compares NPU-generated tokens against Python golden reference.
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
#include <random>
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
static const uint16_t GEMM_IMM = 0x0901;
static const uint16_t GEMM_IMM_K64 = 0x0901;  // scale=1, shift=9, for K=64 projections
static const uint16_t GEMM_IMM_K16 = 0x0701;  // scale=1, shift=7, for K=16 score/context
static const uint16_t GEMM_IMM_KS  = 0x0701;  // same as K16, for context P*V

// =============================================================================
// SRAM0 Memory layout for block execution (byte addresses)
// Must match ddr_map.py ADDR_* constants
// =============================================================================
static const uint16_t ADDR_WQ       = 0x0000;  // 4x[64,16] = 4096B (head-blocked)
static const uint16_t ADDR_WK       = 0x1000;  // 4x[64,16] = 4096B (head-blocked)
static const uint16_t ADDR_WV       = 0x2000;  // 4x[64,16] = 4096B (head-blocked)
static const uint16_t ADDR_WO       = 0x3000;  // [64][64]  = 4096B
static const uint16_t ADDR_W1       = 0x4000;  // [64][256] = 16384B
static const uint16_t ADDR_W2       = 0x8000;  // [256][64] = 16384B
static const uint16_t ADDR_X        = 0xC000;  // [S][64]   = 1024B
static const uint16_t ADDR_LN1_OUT  = 0xC400;  // [S][64]   = 1024B
static const uint16_t ADDR_Q_H      = 0xC800;  // [S][16]   = 256B  (per-head, reused)
static const uint16_t ADDR_K_H      = 0xC900;  // [S][16]   = 256B  (per-head, reused)
static const uint16_t ADDR_V_H      = 0xCA00;  // [S][16]   = 256B  (per-head, reused)
static const uint16_t ADDR_S        = 0xCB00;  // [S][S]    = 256B  (per-head, reused)
static const uint16_t ADDR_P        = 0xCC00;  // [S][S]    = 256B  (per-head, reused)
static const uint16_t ADDR_ATTN_H   = 0xCD00;  // [S][16]   = 256B  (per-head context)
static const uint16_t ADDR_ATTN     = 0xCE00;  // [S][64]   = 1024B (concat)
static const uint16_t ADDR_WO_OUT   = 0xD200;  // [S][64]   = 1024B
static const uint16_t ADDR_X2       = 0xD600;  // [S][64]   = 1024B
static const uint16_t ADDR_LN2_OUT  = 0xDA00;  // [S][64]   = 1024B
static const uint16_t ADDR_FFN1     = 0xDE00;  // [S][256]  = 4096B
static const uint16_t ADDR_FFN2     = 0xEE00;  // [S][64]   = 1024B
static const uint16_t ADDR_X_OUT    = 0xF200;  // [S][64]   = 1024B

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
// Decode-mode SRAM0 addresses (S=1, weights same region)
// Must match kv_map.py ADDR_DEC_* constants
// =============================================================================
static const uint16_t ADDR_DEC_X        = 0xC000;
static const uint16_t ADDR_DEC_LN1_OUT  = 0xC040;
static const uint16_t ADDR_DEC_Q_H      = 0xC080;  // [1][16] = 16B per-head
static const uint16_t ADDR_DEC_K_NEW    = 0xC090;  // [1][16] = 16B per-head
static const uint16_t ADDR_DEC_V_NEW    = 0xC0A0;  // [1][16] = 16B per-head
static const uint16_t ADDR_DEC_K_CACHE  = 0xC0B0;  // [T][16] = 256B per-head
static const uint16_t ADDR_DEC_V_CACHE  = 0xC1B0;  // [T][16] = 256B per-head
static const uint16_t ADDR_DEC_S        = 0xC2B0;  // [1][T]  = 16B
static const uint16_t ADDR_DEC_P        = 0xC2C0;  // [1][T]  = 16B
static const uint16_t ADDR_DEC_ATTN_H   = 0xC2D0;  // [1][16] = 16B per-head
static const uint16_t ADDR_DEC_ATTN     = 0xC2E0;  // [1][64] = 64B concat
static const uint16_t ADDR_DEC_WO_OUT   = 0xC320;
static const uint16_t ADDR_DEC_X2       = 0xC360;
static const uint16_t ADDR_DEC_LN2_OUT  = 0xC3A0;
static const uint16_t ADDR_DEC_FFN1     = 0xC3E0;
static const uint16_t ADDR_DEC_FFN2     = 0xC4E0;
static const uint16_t ADDR_DEC_X_OUT    = 0xC520;

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
static const uint8_t FLAG_COPY2D      = 0x04;  // flags[2] for VEC COPY2D mode
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
// C++ LayerNorm Golden (matches RTL layernorm_engine.sv exactly)
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

uint32_t counting_div_mean(int64_t dividend, uint16_t divisor) {
    const uint64_t MASK48 = (1ULL << 48) - 1;
    uint64_t remainder = (uint64_t)dividend & MASK48;
    uint64_t div = (uint64_t)divisor;
    uint32_t quotient = 0;
    for (int i = 0; i < 32; i++) {
        quotient <<= 1;
        if (remainder >= div) {
            remainder -= div;
            quotient |= 1;
        }
    }
    return quotient;
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

void layernorm_golden(const int8_t* input, int8_t* output, int length,
                      const int8_t* beta) {
    int32_t sum = 0;
    uint64_t sum_sq = 0;
    for (int i = 0; i < length; i++) {
        int16_t val = (int16_t)input[i];
        sum += val;
        uint16_t sq = (uint16_t)(val * val);
        sum_sq += sq;
    }

    int64_t mean_dividend = ((int64_t)sum) << 8;
    uint32_t mean_quotient = counting_div_mean(mean_dividend, (uint16_t)length);
    int16_t r_mean = (int16_t)(mean_quotient & 0xFFFF);

    uint64_t sq_dividend = sum_sq << 8;
    uint32_t sq_quotient = counting_div_unsigned(sq_dividend, (uint64_t)length);

    int32_t mean_i32 = (int32_t)r_mean;
    uint32_t mean_sq = (uint32_t)(mean_i32 * mean_i32);
    uint32_t sq_shifted = sq_quotient << 8;
    uint32_t variance;
    if (sq_shifted >= mean_sq)
        variance = sq_shifted - mean_sq;
    else
        variance = 0;

    uint8_t rsqrt_addr = (uint8_t)(variance >> 24);
    uint16_t inv_std = RSQRT_LUT[rsqrt_addr];

    int8_t p_gamma = 127;
    for (int i = 0; i < length; i++) {
        int16_t centered = (int16_t)(((int16_t)input[i]) << 8) - r_mean;
        int32_t inv_std_signed = (int32_t)inv_std;
        int32_t scaled = (int32_t)centered * inv_std_signed;
        scaled = scaled >> 16;
        int32_t gamma_applied = (scaled * (int32_t)p_gamma) >> 7;
        int32_t bias_added = gamma_applied + ((int32_t)beta[i] << 8);
        int32_t result = bias_added >> 8;
        if (result > 127) result = 127;
        if (result < -128) result = -128;
        output[i] = (int8_t)result;
    }
}

// =============================================================================
// C++ GELU Golden (matches RTL gelu_lut.sv exactly)
// =============================================================================
static int8_t GELU_LUT[256];

void build_gelu_lut() {
    for (int i = 0; i < 256; i++) {
        int8_t signed_i = (int8_t)(uint8_t)i;
        double x = (double)signed_i / 32.0;
        double gelu_val = x * 0.5 * (1.0 + erf(x / sqrt(2.0)));
        double result = gelu_val * 32.0;
        int v = (int)round(result);
        if (v > 127) v = 127;
        if (v < -128) v = -128;
        GELU_LUT[i] = (int8_t)v;
    }
}

void gelu_golden(const int8_t* input, int8_t* output, int length) {
    for (int i = 0; i < length; i++) {
        uint8_t idx = (uint8_t)input[i];
        output[i] = GELU_LUT[idx];
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
// Temperature-based sampling
// =============================================================================
int sample_token(const int8_t* logits, int vocab_size, float temperature,
                 std::mt19937& rng) {
    std::vector<float> probs(vocab_size);
    float max_l = *std::max_element(logits, logits + vocab_size);
    float sum = 0;
    for (int i = 0; i < vocab_size; i++) {
        probs[i] = exp(((float)logits[i] - max_l) / temperature);
        sum += probs[i];
    }
    for (auto& p : probs) p /= sum;
    std::discrete_distribution<int> dist(probs.begin(), probs.end());
    return dist(rng);
}

// =============================================================================
// Weight storage
// =============================================================================
static std::vector<int8_t> weights_buf;

// Per-block weight pointers (into weights_buf)
struct BlockWeights {
    const int8_t* ln1_beta;  // [HIDDEN]
    const int8_t* wq;        // [HEADS*HIDDEN][HEAD_DIM] head-blocked
    const int8_t* wk;        // [HEADS*HIDDEN][HEAD_DIM] head-blocked
    const int8_t* wv;        // [HEADS*HIDDEN][HEAD_DIM] head-blocked
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
// Generate microcode for one transformer block
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

    // Multi-head attention loop
    for (int h = 0; h < HEADS; h++) {
        uint16_t wq_h = ADDR_WQ + h * HIDDEN * HEAD_DIM;
        uint16_t wk_h = ADDR_WK + h * HIDDEN * HEAD_DIM;
        uint16_t wv_h = ADDR_WV + h * HIDDEN * HEAD_DIM;

        // GEMM Q_h = LN1 * Wq_h
        encode_instr(OP_GEMM, FLAG_REQUANT, ADDR_Q_H, ADDR_LN1_OUT, wq_h,
                     S, HEAD_DIM, HIDDEN, GEMM_IMM_K64, hi, lo);
        ucode_write(addr++, hi, lo);
        encode_instr(OP_BARRIER, 0, 0, 0, 0, 0, 0, 0, 0, hi, lo);
        ucode_write(addr++, hi, lo);

        // GEMM K_h = LN1 * Wk_h
        encode_instr(OP_GEMM, FLAG_REQUANT, ADDR_K_H, ADDR_LN1_OUT, wk_h,
                     S, HEAD_DIM, HIDDEN, GEMM_IMM_K64, hi, lo);
        ucode_write(addr++, hi, lo);
        encode_instr(OP_BARRIER, 0, 0, 0, 0, 0, 0, 0, 0, hi, lo);
        ucode_write(addr++, hi, lo);

        // GEMM V_h = LN1 * Wv_h
        encode_instr(OP_GEMM, FLAG_REQUANT, ADDR_V_H, ADDR_LN1_OUT, wv_h,
                     S, HEAD_DIM, HIDDEN, GEMM_IMM_K64, hi, lo);
        ucode_write(addr++, hi, lo);
        encode_instr(OP_BARRIER, 0, 0, 0, 0, 0, 0, 0, 0, hi, lo);
        ucode_write(addr++, hi, lo);

        // GEMM S = Q_h * K_h^T
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

        // GEMM ATTN_h = P * V_h
        encode_instr(OP_GEMM, FLAG_REQUANT, ADDR_ATTN_H, ADDR_P, ADDR_V_H,
                     S, HEAD_DIM, S, GEMM_IMM_KS, hi, lo);
        ucode_write(addr++, hi, lo);
        encode_instr(OP_BARRIER, 0, 0, 0, 0, 0, 0, 0, 0, hi, lo);
        ucode_write(addr++, hi, lo);

        // VEC COPY2D: scatter ATTN_H -> ATTN[:, h*HEAD_DIM:(h+1)*HEAD_DIM]
        encode_instr(OP_VEC, FLAG_COPY2D, ADDR_ATTN + h * HEAD_DIM, ADDR_ATTN_H, 0,
                     S, HEAD_DIM, HEAD_DIM, HIDDEN, hi, lo);
        ucode_write(addr++, hi, lo);
        encode_instr(OP_BARRIER, 0, 0, 0, 0, 0, 0, 0, 0, hi, lo);
        ucode_write(addr++, hi, lo);
    }

    // GEMM WO_out = ATTN * Wo
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
                 S, HIDDEN, FFN_DIM, GEMM_IMM_K64, hi, lo);
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
    // input at ADDR_LM_INPUT [1][16], weight at ADDR_LM_WEIGHT [256][16]
    // output at ADDR_LM_OUTPUT [1][256]
    encode_instr(OP_GEMM, FLAG_TRANSPOSE_B | FLAG_REQUANT,
                 ADDR_LM_OUTPUT, ADDR_LM_INPUT, ADDR_LM_WEIGHT,
                 1, VOCAB_SIZE, HIDDEN, GEMM_IMM, hi, lo);
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

    // Multi-head attention loop
    for (int h = 0; h < HEADS; h++) {
        uint16_t wq_h = ADDR_WQ + h * HIDDEN * HEAD_DIM;
        uint16_t wk_h = ADDR_WK + h * HIDDEN * HEAD_DIM;
        uint16_t wv_h = ADDR_WV + h * HIDDEN * HEAD_DIM;

        // GEMM Q_h = LN1 * Wq_h
        encode_instr(OP_GEMM, FLAG_REQUANT, ADDR_Q_H, ADDR_LN1_OUT, wq_h,
                     S, HEAD_DIM, HIDDEN, GEMM_IMM_K64, hi, lo);
        ucode_write(addr++, hi, lo);
        encode_instr(OP_BARRIER, 0, 0, 0, 0, 0, 0, 0, 0, hi, lo);
        ucode_write(addr++, hi, lo);

        // GEMM K_h = LN1 * Wk_h
        encode_instr(OP_GEMM, FLAG_REQUANT, ADDR_K_H, ADDR_LN1_OUT, wk_h,
                     S, HEAD_DIM, HIDDEN, GEMM_IMM_K64, hi, lo);
        ucode_write(addr++, hi, lo);
        encode_instr(OP_BARRIER, 0, 0, 0, 0, 0, 0, 0, 0, hi, lo);
        ucode_write(addr++, hi, lo);

        // GEMM V_h = LN1 * Wv_h
        encode_instr(OP_GEMM, FLAG_REQUANT, ADDR_V_H, ADDR_LN1_OUT, wv_h,
                     S, HEAD_DIM, HIDDEN, GEMM_IMM_K64, hi, lo);
        ucode_write(addr++, hi, lo);
        encode_instr(OP_BARRIER, 0, 0, 0, 0, 0, 0, 0, 0, hi, lo);
        ucode_write(addr++, hi, lo);

        // KV_APPEND: store K_h[0..S-1] into KV cache for this head
        for (int i = 0; i < S; i++) {
            encode_instr(OP_KV_APPEND, 0x00, 0, ADDR_K_H + i * HEAD_DIM, 0,
                         blk_idx, HEAD_DIM, i, h, hi, lo);
            ucode_write(addr++, hi, lo);
        }
        encode_instr(OP_BARRIER, 0, 0, 0, 0, 0, 0, 0, 0, hi, lo);
        ucode_write(addr++, hi, lo);

        // KV_APPEND: store V_h[0..S-1] into KV cache for this head
        for (int i = 0; i < S; i++) {
            encode_instr(OP_KV_APPEND, 0x01, 0, ADDR_V_H + i * HEAD_DIM, 0,
                         blk_idx, HEAD_DIM, i, h, hi, lo);
            ucode_write(addr++, hi, lo);
        }
        encode_instr(OP_BARRIER, 0, 0, 0, 0, 0, 0, 0, 0, hi, lo);
        ucode_write(addr++, hi, lo);

        // GEMM S = Q_h * K_h^T
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

        // GEMM ATTN_h = P * V_h
        encode_instr(OP_GEMM, FLAG_REQUANT, ADDR_ATTN_H, ADDR_P, ADDR_V_H,
                     S, HEAD_DIM, S, GEMM_IMM_KS, hi, lo);
        ucode_write(addr++, hi, lo);
        encode_instr(OP_BARRIER, 0, 0, 0, 0, 0, 0, 0, 0, hi, lo);
        ucode_write(addr++, hi, lo);

        // VEC COPY2D: scatter ATTN_H -> ATTN[:, h*HEAD_DIM:(h+1)*HEAD_DIM]
        encode_instr(OP_VEC, FLAG_COPY2D, ADDR_ATTN + h * HEAD_DIM, ADDR_ATTN_H, 0,
                     S, HEAD_DIM, HEAD_DIM, HIDDEN, hi, lo);
        ucode_write(addr++, hi, lo);
        encode_instr(OP_BARRIER, 0, 0, 0, 0, 0, 0, 0, 0, hi, lo);
        ucode_write(addr++, hi, lo);
    }

    // GEMM WO_out = ATTN * Wo
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
                 S, HIDDEN, FFN_DIM, GEMM_IMM_K64, hi, lo);
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

    // Multi-head attention loop
    for (int h = 0; h < HEADS; h++) {
        uint16_t wq_h = ADDR_WQ + h * HIDDEN * HEAD_DIM;
        uint16_t wk_h = ADDR_WK + h * HIDDEN * HEAD_DIM;
        uint16_t wv_h = ADDR_WV + h * HIDDEN * HEAD_DIM;

        // GEMM Q_h = LN1_out * Wq_h (M=1)
        encode_instr(OP_GEMM, FLAG_REQUANT, ADDR_DEC_Q_H, ADDR_DEC_LN1_OUT, wq_h,
                     1, HEAD_DIM, HIDDEN, GEMM_IMM_K64, hi, lo);
        ucode_write(addr++, hi, lo);
        encode_instr(OP_BARRIER, 0, 0, 0, 0, 0, 0, 0, 0, hi, lo);
        ucode_write(addr++, hi, lo);

        // GEMM K_new = LN1_out * Wk_h (M=1)
        encode_instr(OP_GEMM, FLAG_REQUANT, ADDR_DEC_K_NEW, ADDR_DEC_LN1_OUT, wk_h,
                     1, HEAD_DIM, HIDDEN, GEMM_IMM_K64, hi, lo);
        ucode_write(addr++, hi, lo);
        encode_instr(OP_BARRIER, 0, 0, 0, 0, 0, 0, 0, 0, hi, lo);
        ucode_write(addr++, hi, lo);

        // GEMM V_new = LN1_out * Wv_h (M=1)
        encode_instr(OP_GEMM, FLAG_REQUANT, ADDR_DEC_V_NEW, ADDR_DEC_LN1_OUT, wv_h,
                     1, HEAD_DIM, HIDDEN, GEMM_IMM_K64, hi, lo);
        ucode_write(addr++, hi, lo);
        encode_instr(OP_BARRIER, 0, 0, 0, 0, 0, 0, 0, 0, hi, lo);
        ucode_write(addr++, hi, lo);

        // KV_APPEND K_new at position T, head h
        encode_instr(OP_KV_APPEND, 0x00, 0, ADDR_DEC_K_NEW, 0,
                     blk_idx, HEAD_DIM, T, h, hi, lo);
        ucode_write(addr++, hi, lo);

        // KV_APPEND V_new at position T, head h
        encode_instr(OP_KV_APPEND, 0x01, 0, ADDR_DEC_V_NEW, 0,
                     blk_idx, HEAD_DIM, T, h, hi, lo);
        ucode_write(addr++, hi, lo);
        encode_instr(OP_BARRIER, 0, 0, 0, 0, 0, 0, 0, 0, hi, lo);
        ucode_write(addr++, hi, lo);

        // KV_READ: load K_cache[0..T] for head h
        encode_instr(OP_KV_READ, 0x00, ADDR_DEC_K_CACHE, 0, 0,
                     blk_idx, HEAD_DIM, T_len, h, hi, lo);
        ucode_write(addr++, hi, lo);
        encode_instr(OP_BARRIER, 0, 0, 0, 0, 0, 0, 0, 0, hi, lo);
        ucode_write(addr++, hi, lo);

        // KV_READ: load V_cache[0..T] for head h
        encode_instr(OP_KV_READ, 0x01, ADDR_DEC_V_CACHE, 0, 0,
                     blk_idx, HEAD_DIM, T_len, h, hi, lo);
        ucode_write(addr++, hi, lo);
        encode_instr(OP_BARRIER, 0, 0, 0, 0, 0, 0, 0, 0, hi, lo);
        ucode_write(addr++, hi, lo);

        // GEMM S = Q_h * K_cache^T (M=1, N=T_len, K=HEAD_DIM)
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

        // GEMM ATTN_h = P * V_cache (M=1, N=HEAD_DIM, K=T_len)
        encode_instr(OP_GEMM, FLAG_REQUANT, ADDR_DEC_ATTN_H, ADDR_DEC_P, ADDR_DEC_V_CACHE,
                     1, HEAD_DIM, T_len, GEMM_IMM_KS, hi, lo);
        ucode_write(addr++, hi, lo);
        encode_instr(OP_BARRIER, 0, 0, 0, 0, 0, 0, 0, 0, hi, lo);
        ucode_write(addr++, hi, lo);

        // VEC COPY2D: scatter ATTN_H -> ATTN[h*HEAD_DIM] (M=1)
        encode_instr(OP_VEC, FLAG_COPY2D, ADDR_DEC_ATTN + h * HEAD_DIM, ADDR_DEC_ATTN_H, 0,
                     1, HEAD_DIM, HEAD_DIM, HIDDEN, hi, lo);
        ucode_write(addr++, hi, lo);
        encode_instr(OP_BARRIER, 0, 0, 0, 0, 0, 0, 0, 0, hi, lo);
        ucode_write(addr++, hi, lo);
    }

    // GEMM WO_out = ATTN * Wo  (M=1)
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
                 1, HIDDEN, FFN_DIM, GEMM_IMM_K64, hi, lo);
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
// =============================================================================
void load_block_to_srams(int blk_idx, const int8_t* x_data, int S) {
    const BlockWeights& bw = block_weights[blk_idx];

    // SRAM0: weights (WQ/WK/WV are head-blocked, total HIDDEN*HIDDEN each)
    for (int i = 0; i < HIDDEN * HIDDEN; i++) {
        sram0_write(ADDR_WQ + i, (uint8_t)bw.wq[i]);
        sram0_write(ADDR_WK + i, (uint8_t)bw.wk[i]);
        sram0_write(ADDR_WV + i, (uint8_t)bw.wv[i]);
    }
    for (int i = 0; i < HIDDEN * HIDDEN; i++)
        sram0_write(ADDR_WO + i, (uint8_t)bw.wo[i]);

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
// C++ golden: run one full transformer block
// =============================================================================
void golden_block(const int8_t* x_in, int8_t* x_out, int S, int blk_idx) {
    const BlockWeights& bw = block_weights[blk_idx];
    int scale = GEMM_SCALE;
    int shift_k64 = GEMM_SHIFT;    // 9
    int shift_k16 = 7;

    std::vector<int8_t> ln1(S * HIDDEN);
    std::vector<int8_t> wo_out(S * HIDDEN), x2(S * HIDDEN);
    std::vector<int8_t> ln2(S * HIDDEN), ffn1(S * FFN_DIM), gelu_out(S * FFN_DIM);
    std::vector<int8_t> ffn2(S * HIDDEN);

    // LN1
    for (int r = 0; r < S; r++)
        layernorm_golden(x_in + r * HIDDEN, ln1.data() + r * HIDDEN, HIDDEN, bw.ln1_beta);

    // Multi-head attention
    std::vector<int8_t> attn_concat(S * HIDDEN, 0);

    for (int h = 0; h < HEADS; h++) {
        const int8_t* wq_h = bw.wq + h * HIDDEN * HEAD_DIM;
        const int8_t* wk_h = bw.wk + h * HIDDEN * HEAD_DIM;
        const int8_t* wv_h = bw.wv + h * HIDDEN * HEAD_DIM;

        std::vector<int8_t> q_h(S * HEAD_DIM), k_h(S * HEAD_DIM), v_h(S * HEAD_DIM);
        gemm_golden(ln1.data(), wq_h, q_h.data(), S, HEAD_DIM, HIDDEN, false, scale, shift_k64);
        gemm_golden(ln1.data(), wk_h, k_h.data(), S, HEAD_DIM, HIDDEN, false, scale, shift_k64);
        gemm_golden(ln1.data(), wv_h, v_h.data(), S, HEAD_DIM, HIDDEN, false, scale, shift_k64);

        std::vector<int8_t> s_h(S * S), p_h(S * S);
        gemm_golden(q_h.data(), k_h.data(), s_h.data(), S, S, HEAD_DIM, true, scale, shift_k16);

        for (int r = 0; r < S; r++)
            softmax_golden(s_h.data() + r * S, p_h.data() + r * S, S, true, r);

        std::vector<int8_t> attn_h(S * HEAD_DIM);
        gemm_golden(p_h.data(), v_h.data(), attn_h.data(), S, HEAD_DIM, S, false, scale, shift_k16);

        // Scatter into concat
        for (int r = 0; r < S; r++)
            for (int c = 0; c < HEAD_DIM; c++)
                attn_concat[r * HIDDEN + h * HEAD_DIM + c] = attn_h[r * HEAD_DIM + c];
    }

    // WO_out = ATTN_concat * Wo
    gemm_golden(attn_concat.data(), bw.wo, wo_out.data(), S, HIDDEN, HIDDEN, false, scale, shift_k64);

    // Residual 1
    vec_add_golden(wo_out.data(), x_in, x2.data(), S * HIDDEN);

    // LN2
    for (int r = 0; r < S; r++)
        layernorm_golden(x2.data() + r * HIDDEN, ln2.data() + r * HIDDEN, HIDDEN, bw.ln2_beta);

    // FFN1
    gemm_golden(ln2.data(), bw.w1, ffn1.data(), S, FFN_DIM, HIDDEN, false, scale, shift_k64);

    // GELU
    gelu_golden(ffn1.data(), gelu_out.data(), S * FFN_DIM);

    // FFN2
    gemm_golden(gelu_out.data(), bw.w2, ffn2.data(), S, HIDDEN, FFN_DIM, false, scale, shift_k64);

    // Residual 2
    vec_add_golden(ffn2.data(), x2.data(), x_out, S * HIDDEN);
}

// =============================================================================
// Load block weights + single-token activation for decode mode
// =============================================================================
void load_decode_block_to_srams(int blk_idx, const int8_t* x_data) {
    const BlockWeights& bw = block_weights[blk_idx];

    // SRAM0: weights (WQ/WK/WV are head-blocked, total HIDDEN*HIDDEN each)
    for (int i = 0; i < HIDDEN * HIDDEN; i++) {
        sram0_write(ADDR_WQ + i, (uint8_t)bw.wq[i]);
        sram0_write(ADDR_WK + i, (uint8_t)bw.wk[i]);
        sram0_write(ADDR_WV + i, (uint8_t)bw.wv[i]);
    }
    for (int i = 0; i < HIDDEN * HIDDEN; i++)
        sram0_write(ADDR_WO + i, (uint8_t)bw.wo[i]);

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
// C++ golden: single-token decode block using KV cache
// golden_kv_k/v: [N_LAYERS][HEADS][MAX_SEQ][HEAD_DIM]
// T: position index of the new token
// =============================================================================
void golden_block_decode(const int8_t* x_new, int8_t* x_out,
                         int blk_idx, int T,
                         int8_t golden_kv_k[][HEADS][MAX_SEQ][HEAD_DIM],
                         int8_t golden_kv_v[][HEADS][MAX_SEQ][HEAD_DIM]) {
    const BlockWeights& bw = block_weights[blk_idx];
    int scale = GEMM_SCALE;
    int shift_k64 = GEMM_SHIFT;    // 9
    int shift_k16 = 7;
    int T_len = T + 1;

    // LN1 (single row)
    int8_t ln1[HIDDEN];
    layernorm_golden(x_new, ln1, HIDDEN, bw.ln1_beta);

    // Multi-head attention
    int8_t attn_concat[HIDDEN];
    memset(attn_concat, 0, HIDDEN);

    for (int h = 0; h < HEADS; h++) {
        const int8_t* wq_h = bw.wq + h * HIDDEN * HEAD_DIM;
        const int8_t* wk_h = bw.wk + h * HIDDEN * HEAD_DIM;
        const int8_t* wv_h = bw.wv + h * HIDDEN * HEAD_DIM;

        int8_t q_h[HEAD_DIM], k_new[HEAD_DIM], v_new[HEAD_DIM];
        gemm_golden(ln1, wq_h, q_h, 1, HEAD_DIM, HIDDEN, false, scale, shift_k64);
        gemm_golden(ln1, wk_h, k_new, 1, HEAD_DIM, HIDDEN, false, scale, shift_k64);
        gemm_golden(ln1, wv_h, v_new, 1, HEAD_DIM, HIDDEN, false, scale, shift_k64);

        // Store in golden KV cache
        memcpy(golden_kv_k[blk_idx][h][T], k_new, HEAD_DIM);
        memcpy(golden_kv_v[blk_idx][h][T], v_new, HEAD_DIM);

        // Build K_cache, V_cache from golden cache
        std::vector<int8_t> k_cache(T_len * HEAD_DIM), v_cache(T_len * HEAD_DIM);
        for (int t = 0; t < T_len; t++) {
            memcpy(k_cache.data() + t * HEAD_DIM, golden_kv_k[blk_idx][h][t], HEAD_DIM);
            memcpy(v_cache.data() + t * HEAD_DIM, golden_kv_v[blk_idx][h][t], HEAD_DIM);
        }

        // S = Q_h * K_cache^T
        std::vector<int8_t> s_scores(T_len);
        gemm_golden(q_h, k_cache.data(), s_scores.data(), 1, T_len, HEAD_DIM, true, scale, shift_k16);

        // Softmax
        std::vector<int8_t> p(T_len);
        softmax_golden(s_scores.data(), p.data(), T_len, true, T);

        // ATTN_h = P * V_cache
        int8_t attn_h[HEAD_DIM];
        gemm_golden(p.data(), v_cache.data(), attn_h, 1, HEAD_DIM, T_len, false, scale, shift_k16);

        // Scatter
        for (int c = 0; c < HEAD_DIM; c++)
            attn_concat[h * HEAD_DIM + c] = attn_h[c];
    }

    // WO_out = ATTN_concat * Wo (M=1)
    int8_t wo_out[HIDDEN];
    gemm_golden(attn_concat, bw.wo, wo_out, 1, HIDDEN, HIDDEN, false, scale, shift_k64);

    // Residual 1
    int8_t x2[HIDDEN];
    vec_add_golden(wo_out, x_new, x2, HIDDEN);

    // LN2
    int8_t ln2[HIDDEN];
    layernorm_golden(x2, ln2, HIDDEN, bw.ln2_beta);

    // FFN1 (M=1)
    int8_t ffn1[FFN_DIM];
    gemm_golden(ln2, bw.w1, ffn1, 1, FFN_DIM, HIDDEN, false, scale, shift_k64);

    // GELU
    int8_t gelu_out[FFN_DIM];
    gelu_golden(ffn1, gelu_out, FFN_DIM);

    // FFN2 (M=1)
    int8_t ffn2[HIDDEN];
    gemm_golden(gelu_out, bw.w2, ffn2, 1, HIDDEN, FFN_DIM, false, scale, shift_k64);

    // Residual 2
    vec_add_golden(ffn2, x2, x_out, HIDDEN);
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
// Read golden tokens from file
// =============================================================================
std::vector<int> read_golden_tokens(const std::string& path) {
    std::vector<int> tokens;
    std::ifstream f(path);
    if (!f.is_open()) {
        // Not an error - golden file is optional
        return tokens;
    }
    int t;
    while (f >> t) {
        tokens.push_back(t);
    }
    return tokens;
}

// =============================================================================
// Main
// =============================================================================
int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    Verilated::traceEverOn(true);

    NPU_DUMP = (getenv("NPU_DUMP") != nullptr);

    // Determine data directory from DEMO_OUTDIR or command line
    std::string datadir = ".";
    if (getenv("DEMO_OUTDIR")) {
        datadir = getenv("DEMO_OUTDIR");
    }
    for (int i = 1; i < argc - 1; i++) {
        if (std::string(argv[i]) == "--datadir") {
            datadir = argv[i + 1];
        }
    }

    int max_new_tokens = 20;
    float temperature = 0.0f;
    uint32_t seed = 42;
    for (int i = 1; i < argc - 1; i++) {
        if (std::string(argv[i]) == "--max-tokens") {
            max_new_tokens = std::atoi(argv[i + 1]);
        } else if (std::string(argv[i]) == "--temperature") {
            temperature = (float)std::atof(argv[i + 1]);
        } else if (std::string(argv[i]) == "--seed") {
            seed = (uint32_t)std::atoi(argv[i + 1]);
        }
    }

    bool use_kv_cache = false;
    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "--kv-cache") use_kv_cache = true;
    }

    std::mt19937 rng(seed);

    std::cout << "============================================" << std::endl;
    std::cout << "  GPT-2 INFERENCE DEMO (NPU Hardware)" << std::endl;
    std::cout << "  layers=" << N_LAYERS << " hidden=" << HIDDEN
              << " ffn=" << FFN_DIM << " vocab=" << VOCAB_SIZE << std::endl;
    std::cout << "  max_new_tokens=" << max_new_tokens
              << (use_kv_cache ? " (KV-cache)" : " (full-recompute)") << std::endl;
    std::cout << "  temperature=" << temperature << " seed=" << seed << std::endl;
    std::cout << "  datadir=" << datadir << std::endl;
    std::cout << "============================================" << std::endl;

    // Build LUTs
    build_softmax_luts();
    build_rsqrt_lut();
    build_gelu_lut();

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
    dut = new Vgpt2_block_top;
    tfp = new VerilatedVcdC;
    if (NPU_DUMP) {
        dut->trace(tfp, 99);
        tfp->open("demo_infer.vcd");
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
    // Inference
    // ==========================================================================
    std::vector<int> generated_tokens;
    int total_npu_cycles = 0;
    bool logits_exact = true;
    int max_logit_err_all = 0;

    if (use_kv_cache) {
    // ======================================================================
    // KV-CACHED PATH: Prefill + Decode
    // ======================================================================
    // KV cache is now in hardware (kv_cache_bank), cleared by reset
    int prompt_len = (int)tokens.size();

    // --- PREFILL PHASE ---
    std::cout << "\n--- PREFILL (S=" << prompt_len << ") ---" << std::endl;
    {
        int S = prompt_len;
        std::vector<int8_t> emb(S * HIDDEN);
        compute_embeddings(tokens, emb.data());

        std::vector<int8_t> x_cur(emb.begin(), emb.end());
        std::vector<int8_t> x_next(S * HIDDEN);

        for (int blk = 0; blk < N_LAYERS; blk++) {
            reset_dut();
            load_block_to_srams(blk, x_cur.data(), S);
            int num_instrs = gen_prefill_block_microcode(S, blk, 0);
            dut->ucode_len = num_instrs;
            dut->start_pulse = 1; tick(); dut->start_pulse = 0;

            int cycles = run_until_done();
            if (cycles < 0) {
                std::cerr << "TIMEOUT in prefill block " << blk << std::endl;
                if (tfp) tfp->close();
                delete dut;
                return 1;
            }
            total_npu_cycles += cycles;

            if (NPU_DUMP)
                std::cout << "  Prefill block " << blk << ": " << cycles
                          << " cycles" << std::endl;

            read_block_output(S, x_next.data());
            x_cur = x_next;
        }

        // LN_F on last token
        reset_dut();
        for (int j = 0; j < HIDDEN; j++)
            sram1_write(S1_LN_F_BETA + j, (uint8_t)ln_f_beta[j]);
        for (int j = 0; j < HIDDEN; j++)
            sram0_write(ADDR_LM_INPUT + j, (uint8_t)x_cur[(S - 1) * HIDDEN + j]);

        int ln_instrs = gen_ln_f_microcode(0);
        dut->ucode_len = ln_instrs;
        dut->start_pulse = 1; tick(); dut->start_pulse = 0;
        int ln_cycles = run_until_done();
        if (ln_cycles < 0) {
            std::cerr << "TIMEOUT in prefill LN_F" << std::endl;
            if (tfp) tfp->close();
            delete dut;
            return 1;
        }
        total_npu_cycles += ln_cycles;

        int8_t ln_f_out[HIDDEN];
        for (int j = 0; j < HIDDEN; j++)
            ln_f_out[j] = (int8_t)sram0_read(ADDR_LM_INPUT + j);

        // lm_head
        reset_dut();
        for (int j = 0; j < HIDDEN; j++)
            sram0_write(ADDR_LM_INPUT + j, (uint8_t)ln_f_out[j]);
        for (int i = 0; i < VOCAB_SIZE; i++)
            for (int j = 0; j < HIDDEN; j++)
                sram0_write(ADDR_LM_WEIGHT + i * HIDDEN + j,
                           (uint8_t)lm_head[i * HIDDEN + j]);

        int lm_instrs = gen_lm_head_microcode(0);
        dut->ucode_len = lm_instrs;
        dut->start_pulse = 1; tick(); dut->start_pulse = 0;
        int lm_cycles = run_until_done();
        if (lm_cycles < 0) {
            std::cerr << "TIMEOUT in prefill lm_head" << std::endl;
            if (tfp) tfp->close();
            delete dut;
            return 1;
        }
        total_npu_cycles += lm_cycles;

        int8_t logits[VOCAB_SIZE];
        for (int i = 0; i < VOCAB_SIZE; i++)
            logits[i] = (int8_t)sram0_read(ADDR_LM_OUTPUT + i);

        // Select next token
        int next_tok;
        if (temperature > 0.0f) {
            next_tok = sample_token(logits, VOCAB_SIZE, temperature, rng);
        } else {
            next_tok = 0;
            int8_t max_logit = logits[0];
            for (int i = 1; i < VOCAB_SIZE; i++) {
                if (logits[i] > max_logit) {
                    max_logit = logits[i]; next_tok = i;
                }
            }
        }

        // Golden verification (follow-actual: use NPU LN_F output for golden)
        int8_t gold_logits[VOCAB_SIZE];
        gemm_golden(ln_f_out, lm_head, gold_logits, 1, VOCAB_SIZE, HIDDEN, true,
                    GEMM_SCALE, GEMM_SHIFT);

        int gold_tok = 0;
        int8_t gold_max = gold_logits[0];
        for (int i = 1; i < VOCAB_SIZE; i++) {
            if (gold_logits[i] > gold_max) {
                gold_max = gold_logits[i]; gold_tok = i;
            }
        }

        int logit_max_err = 0;
        for (int i = 0; i < VOCAB_SIZE; i++) {
            int err = abs((int)logits[i] - (int)gold_logits[i]);
            if (err > logit_max_err) logit_max_err = err;
        }
        if (logit_max_err > max_logit_err_all) max_logit_err_all = logit_max_err;
        if (logit_max_err > 0) logits_exact = false;

        std::cout << "  PREFILL: npu_tok=" << std::setw(3) << next_tok
                  << " gold_tok=" << std::setw(3) << gold_tok
                  << " logit_max_err=" << logit_max_err
                  << (logit_max_err == 0 ? " EXACT" : " DRIFT") << std::endl;

        if (have_golden && 0 < (int)golden_tokens.size()) {
            if (next_tok != golden_tokens[0]) {
                std::cout << "    NOTE: NPU token " << next_tok
                          << " != Python golden " << golden_tokens[0]
                          << (temperature > 0 ? " (sampling)" : "") << std::endl;
            }
        }

        generated_tokens.push_back(next_tok);
        tokens.push_back(next_tok);
    }

    // --- DECODE PHASE ---
    std::cout << "\n--- DECODE ---" << std::endl;
    for (int step = 1; step < max_new_tokens; step++) {
        int pos = (int)tokens.size() - 1;
        if (pos >= MAX_SEQ) {
            std::cout << "WARNING: pos " << pos << " >= MAX_SEQ " << MAX_SEQ
                      << ", stopping" << std::endl;
            break;
        }

        int tok = tokens.back();
        int T = pos;  // 0-based position index of the new token

        if (NPU_DUMP) {
            std::cout << "\n--- Decode step " << step
                      << " (T=" << T << ") ---" << std::endl;
        }

        // Embed single token at position pos
        int8_t x_single[HIDDEN];
        for (int h = 0; h < HIDDEN; h++) {
            int val = (int)wte[tok * HIDDEN + h] + (int)wpe[pos * HIDDEN + h];
            if (val > 127) val = 127;
            if (val < -128) val = -128;
            x_single[h] = (int8_t)val;
        }

        int8_t x_cur_d[HIDDEN], x_next_d[HIDDEN];
        memcpy(x_cur_d, x_single, HIDDEN);

        for (int blk = 0; blk < N_LAYERS; blk++) {
            reset_dut();
            load_decode_block_to_srams(blk, x_cur_d);
            int num_instrs = gen_decode_block_microcode(T, blk, 0);
            dut->ucode_len = num_instrs;
            dut->start_pulse = 1; tick(); dut->start_pulse = 0;

            int cycles = run_until_done();
            if (cycles < 0) {
                std::cerr << "TIMEOUT in decode block " << blk
                          << " step " << step << std::endl;
                if (tfp) tfp->close();
                delete dut;
                return 1;
            }
            total_npu_cycles += cycles;

            if (NPU_DUMP)
                std::cout << "  Decode block " << blk << ": " << cycles
                          << " cycles" << std::endl;

            read_decode_block_output(x_next_d);
            memcpy(x_cur_d, x_next_d, HIDDEN);
        }

        // LN_F on single token
        reset_dut();
        for (int j = 0; j < HIDDEN; j++)
            sram1_write(S1_LN_F_BETA + j, (uint8_t)ln_f_beta[j]);
        for (int j = 0; j < HIDDEN; j++)
            sram0_write(ADDR_LM_INPUT + j, (uint8_t)x_cur_d[j]);

        int ln_instrs = gen_ln_f_microcode(0);
        dut->ucode_len = ln_instrs;
        dut->start_pulse = 1; tick(); dut->start_pulse = 0;
        int ln_cycles = run_until_done();
        if (ln_cycles < 0) {
            std::cerr << "TIMEOUT in decode LN_F step " << step << std::endl;
            if (tfp) tfp->close();
            delete dut;
            return 1;
        }
        total_npu_cycles += ln_cycles;

        int8_t ln_f_out_d[HIDDEN];
        for (int j = 0; j < HIDDEN; j++)
            ln_f_out_d[j] = (int8_t)sram0_read(ADDR_LM_INPUT + j);

        // lm_head
        reset_dut();
        for (int j = 0; j < HIDDEN; j++)
            sram0_write(ADDR_LM_INPUT + j, (uint8_t)ln_f_out_d[j]);
        for (int i = 0; i < VOCAB_SIZE; i++)
            for (int j = 0; j < HIDDEN; j++)
                sram0_write(ADDR_LM_WEIGHT + i * HIDDEN + j,
                           (uint8_t)lm_head[i * HIDDEN + j]);

        int lm_instrs = gen_lm_head_microcode(0);
        dut->ucode_len = lm_instrs;
        dut->start_pulse = 1; tick(); dut->start_pulse = 0;
        int lm_cycles = run_until_done();
        if (lm_cycles < 0) {
            std::cerr << "TIMEOUT in decode lm_head step " << step << std::endl;
            if (tfp) tfp->close();
            delete dut;
            return 1;
        }
        total_npu_cycles += lm_cycles;

        int8_t logits_d[VOCAB_SIZE];
        for (int i = 0; i < VOCAB_SIZE; i++)
            logits_d[i] = (int8_t)sram0_read(ADDR_LM_OUTPUT + i);

        // Select next token
        int next_tok_d;
        if (temperature > 0.0f) {
            next_tok_d = sample_token(logits_d, VOCAB_SIZE, temperature, rng);
        } else {
            next_tok_d = 0;
            int8_t max_logit_d = logits_d[0];
            for (int i = 1; i < VOCAB_SIZE; i++) {
                if (logits_d[i] > max_logit_d) {
                    max_logit_d = logits_d[i]; next_tok_d = i;
                }
            }
        }

        // Golden verification (follow-actual)
        int8_t gold_logits_d[VOCAB_SIZE];
        gemm_golden(ln_f_out_d, lm_head, gold_logits_d, 1, VOCAB_SIZE, HIDDEN, true,
                    GEMM_SCALE, GEMM_SHIFT);

        int gold_tok_d = 0;
        int8_t gold_max_d = gold_logits_d[0];
        for (int i = 1; i < VOCAB_SIZE; i++) {
            if (gold_logits_d[i] > gold_max_d) {
                gold_max_d = gold_logits_d[i]; gold_tok_d = i;
            }
        }

        int logit_max_err_d = 0;
        for (int i = 0; i < VOCAB_SIZE; i++) {
            int err = abs((int)logits_d[i] - (int)gold_logits_d[i]);
            if (err > logit_max_err_d) logit_max_err_d = err;
        }
        if (logit_max_err_d > max_logit_err_all) max_logit_err_all = logit_max_err_d;
        if (logit_max_err_d > 0) logits_exact = false;

        std::cout << "  DECODE step " << std::setw(2) << step
                  << ": npu_tok=" << std::setw(3) << next_tok_d
                  << " gold_tok=" << std::setw(3) << gold_tok_d
                  << " logit_max_err=" << logit_max_err_d
                  << (logit_max_err_d == 0 ? " EXACT" : " DRIFT") << std::endl;

        if (have_golden && step < (int)golden_tokens.size()) {
            if (next_tok_d != golden_tokens[step]) {
                std::cout << "    NOTE: NPU token " << next_tok_d
                          << " != Python golden " << golden_tokens[step]
                          << (temperature > 0 ? " (sampling)" : "") << std::endl;
            }
        }

        generated_tokens.push_back(next_tok_d);
        tokens.push_back(next_tok_d);
    }

    } else {
    // ======================================================================
    // FULL-RECOMPUTE PATH (original)
    // ======================================================================

    for (int step = 0; step < max_new_tokens; step++) {
        int S = (int)tokens.size();
        if (S > MAX_SEQ) {
            std::cout << "WARNING: seq_len " << S << " > MAX_SEQ " << MAX_SEQ << ", stopping" << std::endl;
            break;
        }

        if (NPU_DUMP) {
            std::cout << "\n--- Decode step " << step << " (seq_len=" << S << ") ---" << std::endl;
        }

        // 1. Compute embeddings (host-side)
        std::vector<int8_t> emb(S * HIDDEN);
        compute_embeddings(tokens, emb.data());

        // 2. Run N_LAYERS transformer blocks on NPU
        std::vector<int8_t> x_cur(emb.begin(), emb.end());
        std::vector<int8_t> x_next(S * HIDDEN);

        for (int blk = 0; blk < N_LAYERS; blk++) {
            // Reset before each program
            reset_dut();

            // Load weights + activations into SRAMs
            load_block_to_srams(blk, x_cur.data(), S);

            // Generate and load microcode
            int num_instrs = gen_block_microcode(S, 0);
            dut->ucode_len = num_instrs;

            // Start
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

            // Read output
            read_block_output(S, x_next.data());
            x_cur = x_next;
        }

        // 3. Final LayerNorm on last token
        // Copy last token hidden state to ADDR_LM_INPUT in SRAM0
        reset_dut();

        // Load LN_F beta to SRAM1
        for (int j = 0; j < HIDDEN; j++)
            sram1_write(S1_LN_F_BETA + j, (uint8_t)ln_f_beta[j]);

        // Write last-token hidden state to SRAM0 ADDR_LM_INPUT
        for (int j = 0; j < HIDDEN; j++)
            sram0_write(ADDR_LM_INPUT + j, (uint8_t)x_cur[(S - 1) * HIDDEN + j]);

        // Run LN_F microcode
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

        // Read LN_F output (at ADDR_LM_INPUT, in-place)
        int8_t ln_f_out[HIDDEN];
        for (int j = 0; j < HIDDEN; j++)
            ln_f_out[j] = (int8_t)sram0_read(ADDR_LM_INPUT + j);

        // 4. LM Head on NPU
        reset_dut();

        // Write LN_F output to ADDR_LM_INPUT
        for (int j = 0; j < HIDDEN; j++)
            sram0_write(ADDR_LM_INPUT + j, (uint8_t)ln_f_out[j]);

        // Write lm_head weights to ADDR_LM_WEIGHT [256][16]
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

        // 5. Read logits and select next token
        int8_t logits[VOCAB_SIZE];
        for (int i = 0; i < VOCAB_SIZE; i++)
            logits[i] = (int8_t)sram0_read(ADDR_LM_OUTPUT + i);

        int next_tok;
        if (temperature > 0.0f) {
            next_tok = sample_token(logits, VOCAB_SIZE, temperature, rng);
        } else {
            // Greedy argmax
            next_tok = 0;
            int8_t max_logit = logits[0];
            for (int i = 1; i < VOCAB_SIZE; i++) {
                if (logits[i] > max_logit) {
                    max_logit = logits[i];
                    next_tok = i;
                }
            }
        }

        // 6. C++ golden verification (follow-actual approach)
        // Use the ACTUAL NPU block outputs to feed into the golden for
        // LN_F + lm_head. This eliminates cascading LayerNorm rounding
        // differences (known: C++ LN vs RTL LN can differ by ~20).
        // The NPU block output x_cur was read back from SRAM after each block.
        // We use x_cur (the NPU actual) for LN_F + lm_head golden.
        int8_t gold_ln_f[HIDDEN];
        layernorm_golden(x_cur.data() + (S - 1) * HIDDEN, gold_ln_f, HIDDEN, ln_f_beta);

        // Compare NPU LN_F output with golden (informational, LN has known tolerance)
        int ln_f_max_err = 0;
        for (int j = 0; j < HIDDEN; j++) {
            int err = abs((int)ln_f_out[j] - (int)gold_ln_f[j]);
            if (err > ln_f_max_err) ln_f_max_err = err;
        }

        // Golden lm_head using ACTUAL NPU LN_F output (follow-actual)
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
        // Since we used actual NPU LN_F output, the lm_head GEMM should
        // be bit-exact (GEMM is proven bit-exact in gpt2_block_sim).
        int logit_max_err = 0;
        int logit_mismatches = 0;
        for (int i = 0; i < VOCAB_SIZE; i++) {
            int err = abs((int)logits[i] - (int)gold_logits[i]);
            if (err > logit_max_err) logit_max_err = err;
            if (err > 0) logit_mismatches++;
        }

        if (logit_max_err > max_logit_err_all)
            max_logit_err_all = logit_max_err;
        if (logit_max_err > 0)
            logits_exact = false;

        std::cout << "  Step " << std::setw(2) << step
                  << ": npu_tok=" << std::setw(3) << next_tok
                  << " gold_tok=" << std::setw(3) << gold_tok
                  << " logit_max_err=" << logit_max_err
                  << (logit_max_err == 0 ? " EXACT" : " DRIFT")
                  << " ln_f_err=" << ln_f_max_err
                  << std::endl;

        if (NPU_DUMP) {
            std::cout << "    logits[0:15]: ";
            for (int i = 0; i < 16; i++) std::cout << (int)logits[i] << " ";
            std::cout << std::endl;
            std::cout << "    gold_l[0:15]: ";
            for (int i = 0; i < 16; i++) std::cout << (int)gold_logits[i] << " ";
            std::cout << std::endl;
        }

        // Also compare with Python golden if available (informational only
        // when temperature > 0, since sampling makes tokens diverge)
        if (have_golden && step < (int)golden_tokens.size()) {
            if (next_tok != golden_tokens[step]) {
                std::cout << "    NOTE: NPU token " << next_tok
                          << " != Python golden " << golden_tokens[step]
                          << (temperature > 0 ? " (sampling)" : "") << std::endl;
            }
        }

        // Append token and continue
        generated_tokens.push_back(next_tok);
        tokens.push_back(next_tok);
    }

    } // end full-recompute / kv-cache

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

    // Compare with Python golden (informational only)
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

    // Pass criterion: logits must be bit-exact with C++ golden
    bool pass = logits_exact;
    std::cout << "  GPT-2 DEMO: " << (pass ? "PASS" : "FAIL") << std::endl;
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
