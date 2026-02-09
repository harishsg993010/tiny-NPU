// =============================================================================
// tb_integration.cpp - Integration test: single attention head pipeline
// Runs microcode through control plane, intercepts GEMM in C++, runs real
// softmax in hardware, verifies output against golden reference.
// =============================================================================

#include <verilated.h>
#include <verilated_vcd_c.h>
#include "Vintegration_top.h"

#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <iostream>
#include <iomanip>
#include <cassert>
#include <cstdlib>
#include <algorithm>
#include <numeric>

// =============================================================================
// Global simulation state
// =============================================================================
vluint64_t main_time = 0;
double sc_time_stamp() { return main_time; }

static Vintegration_top* dut;
static VerilatedVcdC*    tfp;
static int               tc = 0;

// =============================================================================
// Attention head dimensions
// =============================================================================
static const int SEQ_LEN  = 8;
static const int HEAD_DIM = 16;

// Memory layout (byte addresses in DATA SRAM)
static const uint16_t ADDR_X   = 0x0000;  // Input [8][16] = 128B
static const uint16_t ADDR_WQ  = 0x0100;  // Query weights [16][16] = 256B
static const uint16_t ADDR_WK  = 0x0200;  // Key weights [16][16] = 256B
static const uint16_t ADDR_WV  = 0x0300;  // Value weights [16][16] = 256B
static const uint16_t ADDR_Q   = 0x0400;  // Query output [8][16] = 128B
static const uint16_t ADDR_K   = 0x0480;  // Key output [8][16] = 128B
static const uint16_t ADDR_V   = 0x0500;  // Value output [8][16] = 128B
static const uint16_t ADDR_S   = 0x0580;  // Score matrix [8][8] = 64B
static const uint16_t ADDR_P   = 0x05C0;  // Softmax output [8][8] = 64B
static const uint16_t ADDR_O   = 0x0600;  // Final output [8][16] = 128B

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
void sram_write(uint16_t addr, uint8_t data) {
    dut->tb_sram_wr_en   = 1;
    dut->tb_sram_wr_addr = addr;
    dut->tb_sram_wr_data = data;
    tick();
    dut->tb_sram_wr_en = 0;
}

uint8_t sram_read(uint16_t addr) {
    dut->tb_sram_rd_en   = 1;
    dut->tb_sram_rd_addr = addr;
    tick();  // SRAM captures address
    dut->tb_sram_rd_en = 0;
    return dut->tb_sram_rd_data;
}

void ucode_write(uint16_t addr, uint64_t hi, uint64_t lo) {
    dut->uc_wr_en   = 1;
    dut->uc_wr_addr = addr;
    // Verilator 128-bit: uc_wr_data[0]=bits[31:0], [1]=bits[63:32],
    //                     [2]=bits[95:64], [3]=bits[127:96]
    dut->uc_wr_data[0] = (uint32_t)(lo & 0xFFFFFFFF);
    dut->uc_wr_data[1] = (uint32_t)(lo >> 32);
    dut->uc_wr_data[2] = (uint32_t)(hi & 0xFFFFFFFF);
    dut->uc_wr_data[3] = (uint32_t)(hi >> 32);
    tick();
    dut->uc_wr_en = 0;
}

// =============================================================================
// Instruction encoder
// Encoding: [127:112]=imm [111:96]=K [95:80]=N [79:64]=M
//           [63:48]=src1  [47:32]=src0 [31:16]=dst [15:8]=flags [7:0]=opcode
// =============================================================================
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

// Opcodes (matching isa_pkg.sv)
static const uint8_t OP_NOP     = 0;
static const uint8_t OP_GEMM    = 3;
static const uint8_t OP_SOFTMAX = 5;
static const uint8_t OP_BARRIER = 10;
static const uint8_t OP_END     = 255;

// Flags
static const uint8_t FLAG_TRANSPOSE_B = 0x01;
static const uint8_t FLAG_REQUANT     = 0x04;
static const uint8_t FLAG_CAUSAL_MASK = 0x10;

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
            // Requantize: (acc * scale + round) >> shift, clamp to int8
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
// C++ Softmax Golden (matches RTL LUT-based implementation with >>17 fix)
// =============================================================================

// Build exp LUT matching RTL exp_lut.sv exactly
static uint16_t EXP_LUT[256];
static uint16_t RECIP_LUT[256];

void build_luts() {
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
    // Pass 1: find max
    int8_t max_val = -128;
    for (int i = 0; i < length; i++) {
        int8_t s = scores[i];
        if (causal_mask_en && i > causal_limit) s = -128;
        if (s > max_val) max_val = s;
    }

    // Pass 2: exp + sum
    std::vector<uint16_t> exp_vals(length);
    int32_t exp_sum = 0;
    for (int i = 0; i < length; i++) {
        int8_t s = scores[i];
        if (causal_mask_en && i > causal_limit) s = -128;

        int16_t diff = (int16_t)s - (int16_t)max_val;
        if (diff < -128) diff = -128;
        if (diff > 127) diff = 127;
        int8_t diff_i8 = (int8_t)diff;

        uint8_t idx = (uint8_t)diff_i8;
        exp_vals[i] = EXP_LUT[idx];
        exp_sum += (int32_t)exp_vals[i];
    }

    // Pass 3: normalize
    if (exp_sum == 0) {
        memset(output, 0, length);
        return;
    }
    int sum_top8 = (int)(exp_sum >> 8);
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
// Data Arrays
// =============================================================================
static int8_t X[SEQ_LEN][HEAD_DIM];
static int8_t Wq[HEAD_DIM][HEAD_DIM];
static int8_t Wk[HEAD_DIM][HEAD_DIM];
static int8_t Wv[HEAD_DIM][HEAD_DIM];

// Golden intermediate/output arrays
static int8_t Q_gold[SEQ_LEN][HEAD_DIM];
static int8_t K_gold[SEQ_LEN][HEAD_DIM];
static int8_t V_gold[SEQ_LEN][HEAD_DIM];
static int8_t S_gold[SEQ_LEN][SEQ_LEN];
static int8_t P_gold[SEQ_LEN][SEQ_LEN];
static int8_t O_gold[SEQ_LEN][HEAD_DIM];

// =============================================================================
// Generate test data and compute golden
// =============================================================================
void generate_data_and_golden() {
    srand(42);

    // Generate inputs
    for (int i = 0; i < SEQ_LEN; i++)
        for (int j = 0; j < HEAD_DIM; j++)
            X[i][j] = (int8_t)((rand() % 21) - 10);  // [-10, 10]

    for (int i = 0; i < HEAD_DIM; i++)
        for (int j = 0; j < HEAD_DIM; j++) {
            Wq[i][j] = (int8_t)((rand() % 11) - 5);  // [-5, 5]
            Wk[i][j] = (int8_t)((rand() % 11) - 5);
            Wv[i][j] = (int8_t)((rand() % 11) - 5);
        }

    // scale=1, shift=7 for requantization (imm = 0x0701)
    int scale = 1, shift = 7;

    // Q = X * Wq (no transpose)
    gemm_golden(&X[0][0], &Wq[0][0], &Q_gold[0][0],
                SEQ_LEN, HEAD_DIM, HEAD_DIM, false, scale, shift);

    // K = X * Wk
    gemm_golden(&X[0][0], &Wk[0][0], &K_gold[0][0],
                SEQ_LEN, HEAD_DIM, HEAD_DIM, false, scale, shift);

    // V = X * Wv
    gemm_golden(&X[0][0], &Wv[0][0], &V_gold[0][0],
                SEQ_LEN, HEAD_DIM, HEAD_DIM, false, scale, shift);

    // S = Q * K^T (transpose_b)
    gemm_golden(&Q_gold[0][0], &K_gold[0][0], &S_gold[0][0],
                SEQ_LEN, SEQ_LEN, HEAD_DIM, true, scale, shift);

    // P = softmax(S) row-by-row with causal mask
    for (int row = 0; row < SEQ_LEN; row++) {
        softmax_golden(&S_gold[row][0], &P_gold[row][0], SEQ_LEN,
                       true, row);  // causal_limit = row
    }

    // O = P * V (no transpose)
    gemm_golden(&P_gold[0][0], &V_gold[0][0], &O_gold[0][0],
                SEQ_LEN, HEAD_DIM, SEQ_LEN, false, scale, shift);
}

// =============================================================================
// Load microcode program (20 instructions)
// =============================================================================
void load_microcode() {
    uint64_t hi, lo;
    int addr = 0;

    // 0: GEMM Q = X * Wq  (M=8, N=16, K=16, flags=REQUANT, imm=shift=7|scale=1)
    encode_instr(OP_GEMM, FLAG_REQUANT, ADDR_Q, ADDR_X, ADDR_WQ,
                 SEQ_LEN, HEAD_DIM, HEAD_DIM, 0x0701, hi, lo);
    ucode_write(addr++, hi, lo);

    // 1: BARRIER
    encode_instr(OP_BARRIER, 0, 0, 0, 0, 0, 0, 0, 0, hi, lo);
    ucode_write(addr++, hi, lo);

    // 2: GEMM K = X * Wk
    encode_instr(OP_GEMM, FLAG_REQUANT, ADDR_K, ADDR_X, ADDR_WK,
                 SEQ_LEN, HEAD_DIM, HEAD_DIM, 0x0701, hi, lo);
    ucode_write(addr++, hi, lo);

    // 3: BARRIER
    encode_instr(OP_BARRIER, 0, 0, 0, 0, 0, 0, 0, 0, hi, lo);
    ucode_write(addr++, hi, lo);

    // 4: GEMM V = X * Wv
    encode_instr(OP_GEMM, FLAG_REQUANT, ADDR_V, ADDR_X, ADDR_WV,
                 SEQ_LEN, HEAD_DIM, HEAD_DIM, 0x0701, hi, lo);
    ucode_write(addr++, hi, lo);

    // 5: BARRIER
    encode_instr(OP_BARRIER, 0, 0, 0, 0, 0, 0, 0, 0, hi, lo);
    ucode_write(addr++, hi, lo);

    // 6: GEMM S = Q * K^T  (TRANSPOSE_B | REQUANT)
    encode_instr(OP_GEMM, FLAG_TRANSPOSE_B | FLAG_REQUANT, ADDR_S, ADDR_Q, ADDR_K,
                 SEQ_LEN, SEQ_LEN, HEAD_DIM, 0x0701, hi, lo);
    ucode_write(addr++, hi, lo);

    // 7: BARRIER
    encode_instr(OP_BARRIER, 0, 0, 0, 0, 0, 0, 0, 0, hi, lo);
    ucode_write(addr++, hi, lo);

    // 8-15: SOFTMAX rows  (N=8, K=causal_limit=i, flags=CAUSAL_MASK, imm=0x0100)
    for (int i = 0; i < SEQ_LEN; i++) {
        uint16_t src = ADDR_S + i * SEQ_LEN;
        uint16_t dst = ADDR_P + i * SEQ_LEN;
        encode_instr(OP_SOFTMAX, FLAG_CAUSAL_MASK, dst, src, 0,
                     0, SEQ_LEN, i, 0x0100, hi, lo);
        ucode_write(addr++, hi, lo);
    }

    // 16: BARRIER
    encode_instr(OP_BARRIER, 0, 0, 0, 0, 0, 0, 0, 0, hi, lo);
    ucode_write(addr++, hi, lo);

    // 17: GEMM O = P * V
    encode_instr(OP_GEMM, FLAG_REQUANT, ADDR_O, ADDR_P, ADDR_V,
                 SEQ_LEN, HEAD_DIM, SEQ_LEN, 0x0701, hi, lo);
    ucode_write(addr++, hi, lo);

    // 18: BARRIER
    encode_instr(OP_BARRIER, 0, 0, 0, 0, 0, 0, 0, 0, hi, lo);
    ucode_write(addr++, hi, lo);

    // 19: END
    encode_instr(OP_END, 0, 0, 0, 0, 0, 0, 0, 0, hi, lo);
    ucode_write(addr++, hi, lo);

    std::cout << "  Loaded " << addr << " microcode instructions" << std::endl;
}

// =============================================================================
// Load data arrays into DATA SRAM
// =============================================================================
void load_data_to_sram() {
    // X at ADDR_X
    for (int i = 0; i < SEQ_LEN; i++)
        for (int j = 0; j < HEAD_DIM; j++)
            sram_write(ADDR_X + i * HEAD_DIM + j, (uint8_t)X[i][j]);

    // Wq at ADDR_WQ
    for (int i = 0; i < HEAD_DIM; i++)
        for (int j = 0; j < HEAD_DIM; j++)
            sram_write(ADDR_WQ + i * HEAD_DIM + j, (uint8_t)Wq[i][j]);

    // Wk at ADDR_WK
    for (int i = 0; i < HEAD_DIM; i++)
        for (int j = 0; j < HEAD_DIM; j++)
            sram_write(ADDR_WK + i * HEAD_DIM + j, (uint8_t)Wk[i][j]);

    // Wv at ADDR_WV
    for (int i = 0; i < HEAD_DIM; i++)
        for (int j = 0; j < HEAD_DIM; j++)
            sram_write(ADDR_WV + i * HEAD_DIM + j, (uint8_t)Wv[i][j]);

    std::cout << "  Loaded " << (128 + 256*3) << " bytes to DATA SRAM" << std::endl;
}

// =============================================================================
// Handle GEMM command: read inputs from SRAM, compute, write result to SRAM
// =============================================================================
void handle_gemm_command() {
    uint16_t src0  = dut->gemm_src0;
    uint16_t src1  = dut->gemm_src1;
    uint16_t dst   = dut->gemm_dst;
    uint16_t m     = dut->gemm_M;
    uint16_t n     = dut->gemm_N;
    uint16_t k     = dut->gemm_K;
    uint8_t  flags = dut->gemm_flags;
    uint16_t imm   = dut->gemm_imm;

    bool transpose_b = (flags & FLAG_TRANSPOSE_B) != 0;
    int scale = imm & 0xFF;
    int shift = (imm >> 8) & 0xFF;

    std::cout << "  GEMM: src0=0x" << std::hex << src0
              << " src1=0x" << src1 << " dst=0x" << dst
              << " M=" << std::dec << m << " N=" << n << " K=" << k
              << " trans=" << transpose_b
              << " scale=" << scale << " shift=" << shift << std::endl;

    // Read A[M][K] from SRAM
    std::vector<int8_t> A(m * k);
    for (int i = 0; i < m; i++)
        for (int j = 0; j < k; j++)
            A[i * k + j] = (int8_t)sram_read(src0 + i * k + j);

    // Read B from SRAM
    // If transpose_b: B is stored as [N][K], read N*K elements
    // If not: B is stored as [K][N], read K*N elements
    int b_rows = transpose_b ? n : k;
    int b_cols = transpose_b ? k : n;
    std::vector<int8_t> B(b_rows * b_cols);
    for (int i = 0; i < b_rows; i++)
        for (int j = 0; j < b_cols; j++)
            B[i * b_cols + j] = (int8_t)sram_read(src1 + i * b_cols + j);

    // Compute GEMM
    std::vector<int8_t> C(m * n);
    gemm_golden(A.data(), B.data(), C.data(), m, n, k, transpose_b, scale, shift);

    // Write result C[M][N] to SRAM
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            sram_write(dst + i * n + j, (uint8_t)C[i * n + j]);
}

// =============================================================================
// Main test
// =============================================================================
int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    Verilated::traceEverOn(true);

    dut = new Vintegration_top;
    tfp = new VerilatedVcdC;
    dut->trace(tfp, 99);
    tfp->open("integration_sim.vcd");

    // Initialize all inputs
    dut->clk = 0;
    dut->rst_n = 0;
    dut->start_pulse = 0;
    dut->ucode_len = 0;
    dut->uc_wr_en = 0;
    dut->uc_wr_addr = 0;
    dut->uc_wr_data[0] = 0; dut->uc_wr_data[1] = 0;
    dut->uc_wr_data[2] = 0; dut->uc_wr_data[3] = 0;
    dut->tb_sram_wr_en = 0;
    dut->tb_sram_wr_addr = 0;
    dut->tb_sram_wr_data = 0;
    dut->tb_sram_rd_en = 0;
    dut->tb_sram_rd_addr = 0;
    dut->gemm_done_pulse = 0;

    std::cout << "============================================" << std::endl;
    std::cout << "  INTEGRATION TEST: Single Attention Head" << std::endl;
    std::cout << "  seq_len=" << SEQ_LEN << " head_dim=" << HEAD_DIM << std::endl;
    std::cout << "============================================" << std::endl;

    // Build LUTs
    build_luts();

    // Reset
    reset_dut();
    std::cout << "Reset complete." << std::endl;

    // Generate data and compute golden
    generate_data_and_golden();
    std::cout << "Golden reference computed." << std::endl;

    // Print some golden values for debug
    std::cout << "  X[0]: ";
    for (int j = 0; j < HEAD_DIM; j++) std::cout << (int)X[0][j] << " ";
    std::cout << std::endl;
    std::cout << "  Q_gold[0]: ";
    for (int j = 0; j < HEAD_DIM; j++) std::cout << (int)Q_gold[0][j] << " ";
    std::cout << std::endl;
    std::cout << "  S_gold[0]: ";
    for (int j = 0; j < SEQ_LEN; j++) std::cout << (int)S_gold[0][j] << " ";
    std::cout << std::endl;
    std::cout << "  P_gold[0]: ";
    for (int j = 0; j < SEQ_LEN; j++) std::cout << (int)P_gold[0][j] << " ";
    std::cout << std::endl;
    std::cout << "  O_gold[0]: ";
    for (int j = 0; j < HEAD_DIM; j++) std::cout << (int)O_gold[0][j] << " ";
    std::cout << std::endl;

    // Load microcode
    std::cout << "Loading microcode..." << std::endl;
    load_microcode();

    // Load data to SRAM
    std::cout << "Loading data to SRAM..." << std::endl;
    load_data_to_sram();

    // Start execution
    std::cout << "Starting execution..." << std::endl;
    dut->ucode_len = 20;
    dut->start_pulse = 1;
    tick();
    dut->start_pulse = 0;

    // Main simulation loop
    int gemm_count = 0;
    int softmax_count = 0;
    int cycle = 0;
    bool done = false;
    const int MAX_CYCLES = 200000;

    while (cycle < MAX_CYCLES && !done) {
        tick();
        cycle++;

        // Check for GEMM command capture
        if (dut->gemm_cmd_captured) {
            gemm_count++;
            std::cout << "[cycle " << cycle << "] GEMM command #" << gemm_count << std::endl;

            // Handle the GEMM in C++
            handle_gemm_command();

            // Signal done to hardware
            dut->gemm_done_pulse = 1;
            tick();
            cycle++;
            dut->gemm_done_pulse = 0;
        }

        // Track softmax activity
        if (dut->softmax_done_dbg) {
            softmax_count++;
            if (softmax_count <= SEQ_LEN)
                std::cout << "[cycle " << cycle << "] Softmax row #" << softmax_count << " done" << std::endl;
        }

        // Check for program end
        if (dut->program_end) {
            std::cout << "[cycle " << cycle << "] Program END" << std::endl;
            // Let pipeline drain
            for (int i = 0; i < 10; i++) tick();
            done = true;
        }
    }

    if (!done) {
        std::cerr << "TIMEOUT after " << MAX_CYCLES << " cycles!" << std::endl;
        tfp->close();
        delete tfp;
        delete dut;
        return 1;
    }

    std::cout << std::endl;
    std::cout << "Execution complete: " << cycle << " cycles" << std::endl;
    std::cout << "  GEMM commands: " << gemm_count << std::endl;
    std::cout << "  Softmax rows:  " << softmax_count << std::endl;

    // Wait for engines to be idle, then read results
    for (int i = 0; i < 20; i++) tick();

    // ================================================================
    // Verify output
    // ================================================================
    std::cout << std::endl << "=== Verification ===" << std::endl;

    // Read Q from SRAM and compare
    int q_mismatches = 0;
    for (int i = 0; i < SEQ_LEN; i++)
        for (int j = 0; j < HEAD_DIM; j++) {
            int8_t actual = (int8_t)sram_read(ADDR_Q + i * HEAD_DIM + j);
            if (actual != Q_gold[i][j]) q_mismatches++;
        }
    std::cout << "  Q: " << (q_mismatches == 0 ? "MATCH" : "MISMATCH")
              << " (" << q_mismatches << " errors)" << std::endl;

    // Read K
    int k_mismatches = 0;
    for (int i = 0; i < SEQ_LEN; i++)
        for (int j = 0; j < HEAD_DIM; j++) {
            int8_t actual = (int8_t)sram_read(ADDR_K + i * HEAD_DIM + j);
            if (actual != K_gold[i][j]) k_mismatches++;
        }
    std::cout << "  K: " << (k_mismatches == 0 ? "MATCH" : "MISMATCH")
              << " (" << k_mismatches << " errors)" << std::endl;

    // Read V
    int v_mismatches = 0;
    for (int i = 0; i < SEQ_LEN; i++)
        for (int j = 0; j < HEAD_DIM; j++) {
            int8_t actual = (int8_t)sram_read(ADDR_V + i * HEAD_DIM + j);
            if (actual != V_gold[i][j]) v_mismatches++;
        }
    std::cout << "  V: " << (v_mismatches == 0 ? "MATCH" : "MISMATCH")
              << " (" << v_mismatches << " errors)" << std::endl;

    // Read S (score matrix)
    int s_mismatches = 0;
    for (int i = 0; i < SEQ_LEN; i++)
        for (int j = 0; j < SEQ_LEN; j++) {
            int8_t actual = (int8_t)sram_read(ADDR_S + i * SEQ_LEN + j);
            if (actual != S_gold[i][j]) s_mismatches++;
        }
    std::cout << "  S: " << (s_mismatches == 0 ? "MATCH" : "MISMATCH")
              << " (" << s_mismatches << " errors)" << std::endl;

    // Read P (softmax output) - this ran in hardware
    int p_mismatches = 0;
    int p_max_err = 0;
    std::cout << "  P (softmax output, hardware):" << std::endl;
    for (int i = 0; i < SEQ_LEN; i++) {
        std::cout << "    row " << i << ": ";
        for (int j = 0; j < SEQ_LEN; j++) {
            int8_t actual = (int8_t)sram_read(ADDR_P + i * SEQ_LEN + j);
            int err = abs((int)actual - (int)P_gold[i][j]);
            if (err > p_max_err) p_max_err = err;
            if (err > 2) p_mismatches++;
            std::cout << (int)actual << "(" << (int)P_gold[i][j] << ") ";
        }
        std::cout << std::endl;
    }
    std::cout << "  P: max_abs_error=" << p_max_err
              << " mismatches(>2)=" << p_mismatches << std::endl;

    // Read O (final output)
    int o_mismatches = 0;
    int o_max_err = 0;
    for (int i = 0; i < SEQ_LEN; i++)
        for (int j = 0; j < HEAD_DIM; j++) {
            int8_t actual = (int8_t)sram_read(ADDR_O + i * HEAD_DIM + j);
            int err = abs((int)actual - (int)O_gold[i][j]);
            if (err > o_max_err) o_max_err = err;
            if (err > 2) o_mismatches++;
        }

    std::cout << "  O: max_abs_error=" << o_max_err
              << " mismatches(>2)=" << o_mismatches << std::endl;

    // Print first row of O for inspection
    std::cout << "  O[0] actual: ";
    for (int j = 0; j < HEAD_DIM; j++)
        std::cout << (int)(int8_t)sram_read(ADDR_O + j) << " ";
    std::cout << std::endl;
    std::cout << "  O[0] golden: ";
    for (int j = 0; j < HEAD_DIM; j++)
        std::cout << (int)O_gold[0][j] << " ";
    std::cout << std::endl;

    // ================================================================
    // Final verdict
    // ================================================================
    std::cout << std::endl;
    std::cout << "============================================" << std::endl;
    bool pass = (gemm_count == 5) &&
                (q_mismatches == 0) &&
                (k_mismatches == 0) &&
                (v_mismatches == 0) &&
                (s_mismatches == 0) &&
                (o_max_err <= 2);

    std::cout << "  GEMM commands intercepted: " << gemm_count << "/5 "
              << (gemm_count == 5 ? "OK" : "FAIL") << std::endl;
    std::cout << "  Softmax rows completed:    " << softmax_count << "/8" << std::endl;
    std::cout << "  Q exact match:             " << (q_mismatches == 0 ? "PASS" : "FAIL") << std::endl;
    std::cout << "  K exact match:             " << (k_mismatches == 0 ? "PASS" : "FAIL") << std::endl;
    std::cout << "  V exact match:             " << (v_mismatches == 0 ? "PASS" : "FAIL") << std::endl;
    std::cout << "  S exact match:             " << (s_mismatches == 0 ? "PASS" : "FAIL") << std::endl;
    std::cout << "  P max error:               " << p_max_err << " (tol=2)" << std::endl;
    std::cout << "  O max error:               " << o_max_err << " (tol=2)" << std::endl;
    std::cout << "  Total cycles:              " << cycle << std::endl;
    std::cout << "============================================" << std::endl;
    std::cout << "  INTEGRATION TEST: " << (pass ? "PASS" : "FAIL") << std::endl;
    std::cout << "============================================" << std::endl;

    tfp->close();
    delete tfp;
    delete dut;

    return pass ? 0 : 1;
}
