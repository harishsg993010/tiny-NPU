// =============================================================================
// tb_onnx_fuzz.cpp - Fuzz test for Graph Mode (50 cases)
// Loops over 50 test cases, each in its own subdirectory. For each case:
// reset DUT, load program/tdesc/ddr_image, run, compare golden.
// VCD tracing is DISABLED for speed.
// =============================================================================
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <vector>
#include <string>
#include <algorithm>

#include "Vonnx_sim_top.h"
#include "verilated.h"
#include "verilated_vcd_c.h"

#include "../../include/graph_isa.h"
#include "../../include/ddr_graph.h"

// =============================================================================
// DDR Memory Model (byte-addressable)
// =============================================================================
static constexpr size_t DDR_SIZE = 16 * 1024 * 1024; // 16 MB
static uint8_t ddr_mem[DDR_SIZE];

// =============================================================================
// Helpers
// =============================================================================
static std::string g_datadir = "build/graph_fuzz";

static bool load_file(const std::string& path, uint8_t* dst, size_t max_size, size_t& actual_size) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) {
        fprintf(stderr, "ERROR: Cannot open %s\n", path.c_str());
        return false;
    }
    fseek(f, 0, SEEK_END);
    actual_size = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (actual_size > max_size) {
        fprintf(stderr, "ERROR: File %s too large (%zu > %zu)\n", path.c_str(), actual_size, max_size);
        fclose(f);
        return false;
    }
    size_t rd = fread(dst, 1, actual_size, f);
    fclose(f);
    return rd == actual_size;
}

static bool load_file_to_ddr(const std::string& path, uint32_t ddr_offset, size_t& actual_size) {
    return load_file(path, ddr_mem + ddr_offset, DDR_SIZE - ddr_offset, actual_size);
}

// =============================================================================
// Clock and tick
// =============================================================================
static vluint64_t sim_time = 0;
static Vonnx_sim_top* dut = nullptr;
static VerilatedVcdC* tfp = nullptr; // Always nullptr for fuzz (no VCD)

static void tick() {
    dut->clk = 0;
    dut->eval();
    if (tfp) tfp->dump(sim_time++);
    dut->clk = 1;
    dut->eval();
    if (tfp) tfp->dump(sim_time++);
}

// =============================================================================
// SRAM0 access via TB ports (only when engines are idle)
// =============================================================================
static void sram0_write(uint16_t addr, uint8_t data) {
    dut->tb_sram0_wr_en   = 1;
    dut->tb_sram0_wr_addr = addr;
    dut->tb_sram0_wr_data = data;
    tick();
    dut->tb_sram0_wr_en = 0;
}

static uint8_t sram0_read(uint16_t addr) {
    dut->tb_sram0_rd_en   = 1;
    dut->tb_sram0_rd_addr = addr;
    tick(); // issue read
    dut->tb_sram0_rd_en = 0;
    tick(); // get data
    return dut->tb_sram0_rd_data;
}

// =============================================================================
// Load program into program SRAM
// =============================================================================
static int load_program(const std::string& path) {
    size_t sz;
    std::vector<uint8_t> buf(64 * 1024);
    if (!load_file(path, buf.data(), buf.size(), sz)) return -1;

    int num_instrs = sz / 16;
    printf("  Loading %d instructions from %s\n", num_instrs, path.c_str());

    for (int i = 0; i < num_instrs; i++) {
        uint32_t w0, w1, w2, w3;
        memcpy(&w0, buf.data() + i * 16 + 0, 4);
        memcpy(&w1, buf.data() + i * 16 + 4, 4);
        memcpy(&w2, buf.data() + i * 16 + 8, 4);
        memcpy(&w3, buf.data() + i * 16 + 12, 4);

        // Pack as 128-bit: word3 in [127:96], word0 in [31:0]
        dut->prog_wr_en   = 1;
        dut->prog_wr_addr = i;
        // Verilator represents 128-bit as array of uint32_t [4]
        dut->prog_wr_data[0] = w0;
        dut->prog_wr_data[1] = w1;
        dut->prog_wr_data[2] = w2;
        dut->prog_wr_data[3] = w3;
        tick();
    }
    dut->prog_wr_en = 0;
    return num_instrs;
}

// =============================================================================
// Load tensor descriptors into tensor table
// =============================================================================
static int load_tdesc(const std::string& path) {
    size_t sz;
    std::vector<uint8_t> buf(256 * 32); // max 256 entries x 32 bytes
    if (!load_file(path, buf.data(), buf.size(), sz)) return -1;

    int num_descs = sz / 32;
    printf("  Loading %d tensor descriptors from %s\n", num_descs, path.c_str());

    for (int i = 0; i < num_descs; i++) {
        dut->tdesc_wr_en   = 1;
        dut->tdesc_wr_addr = i;
        // 256-bit = 8 x uint32_t
        for (int w = 0; w < 8; w++) {
            uint32_t val;
            memcpy(&val, buf.data() + i * 32 + w * 4, 4);
            dut->tdesc_wr_data[w] = val;
        }
        tick();
    }
    dut->tdesc_wr_en = 0;
    return num_descs;
}

// =============================================================================
// DMA handler: called each cycle when DMA command is captured
// =============================================================================
static void handle_dma() {
    if (!dut->dma_cmd_captured) return;

    uint32_t ddr_addr  = dut->dma_ddr_addr;
    uint16_t sram_addr = dut->dma_sram_addr;
    uint16_t length    = dut->dma_length;
    bool     store     = dut->dma_direction; // 1=SRAM->DDR
    bool     strided   = dut->dma_strided;

    if (strided) {
        // Strided DMA: multiple blocks
        uint32_t stride    = dut->dma_stride;
        uint16_t count     = dut->dma_count;
        uint16_t block_len = dut->dma_block_len;

        printf("  DMA STRIDED: ddr=0x%08x sram=0x%04x dir=%s stride=%u count=%u block=%u\n",
               ddr_addr, sram_addr, store ? "STORE" : "LOAD", stride, count, block_len);

        uint32_t ddr_off = ddr_addr;
        uint16_t sram_off = sram_addr;
        for (int b = 0; b < count; b++) {
            for (int j = 0; j < block_len; j++) {
                if (store) {
                    ddr_mem[ddr_off + j] = sram0_read(sram_off + j);
                } else {
                    sram0_write(sram_off + j, ddr_mem[ddr_off + j]);
                }
            }
            ddr_off  += stride;
            sram_off += block_len;
        }
    } else {
        // Simple DMA
        printf("  DMA %s: ddr=0x%08x sram=0x%04x len=%u\n",
               store ? "STORE" : "LOAD", ddr_addr, sram_addr, length);

        for (int i = 0; i < length; i++) {
            if (store) {
                ddr_mem[ddr_addr + i] = sram0_read(sram_addr + i);
            } else {
                sram0_write(sram_addr + i, ddr_mem[ddr_addr + i]);
            }
        }
    }

    // Signal DMA complete
    dut->dma_done_pulse = 1;
    tick();
    dut->dma_done_pulse = 0;
}

// =============================================================================
// Run a single fuzz case. Returns true if PASS, false if FAIL.
// =============================================================================
static bool run_case(const std::string& casedir) {
    printf("  --- Case directory: %s ---\n", casedir.c_str());

    // Reset DUT
    dut->rst_n = 0;
    dut->start_pulse = 0;
    dut->prog_len = 0;
    dut->prog_wr_en = 0;
    dut->tdesc_wr_en = 0;
    dut->tb_sram0_wr_en = 0;
    dut->tb_sram0_rd_en = 0;
    dut->dma_done_pulse = 0;
    for (int i = 0; i < 10; i++) tick();
    dut->rst_n = 1;
    for (int i = 0; i < 5; i++) tick();

    // Clear DDR
    memset(ddr_mem, 0, DDR_SIZE);

    // Load DDR image
    size_t ddr_img_sz;
    std::string ddr_img_path = casedir + "/ddr_image.bin";
    FILE* f = fopen(ddr_img_path.c_str(), "rb");
    if (f) {
        fseek(f, 0, SEEK_END);
        ddr_img_sz = ftell(f);
        fseek(f, 0, SEEK_SET);
        fread(ddr_mem, 1, std::min(ddr_img_sz, DDR_SIZE), f);
        fclose(f);
        printf("  Loaded DDR image: %zu bytes\n", ddr_img_sz);
    } else {
        printf("  No DDR image found (OK for END-only test)\n");
    }

    // Load program
    int num_instrs = load_program(casedir + "/program.bin");
    if (num_instrs < 0) {
        printf("  FAIL: Could not load program\n");
        return false;
    }

    // Load tensor descriptors
    {
        std::string tdesc_path = casedir + "/tdesc.bin";
        FILE* tf = fopen(tdesc_path.c_str(), "rb");
        if (tf) {
            fclose(tf);
            int nd = load_tdesc(tdesc_path);
            if (nd < 0) {
                printf("  FAIL: Could not load tensor descriptors\n");
                return false;
            }
        } else {
            printf("  No tensor descriptors (OK for END-only test)\n");
        }
    }

    // Start execution
    printf("  Starting graph execution (%d instructions)...\n", num_instrs);
    dut->prog_len = num_instrs + 1; // +1 for END instruction
    dut->start_pulse = 1;
    tick();
    dut->start_pulse = 0;

    // Run until done or timeout
    {
        int timeout = 1000000;
        bool done = false;
        while (timeout-- > 0 && !done) {
            tick();

            // Handle DMA requests
            handle_dma();

            // Check done
            if (dut->graph_done) {
                done = true;
                printf("  Graph execution DONE at cycle %lu\n", (unsigned long)(sim_time / 2));
                printf("  Status: 0x%08x  PC: %d  LastOp: 0x%02x\n",
                       dut->graph_status, dut->graph_pc, dut->graph_last_op);
                printf("  perf_total_cycles: %u\n", dut->perf_total_cycles);
            }

            // Check error
            if (dut->graph_status & 0x4) {
                printf("  ERROR detected! Status: 0x%08x  PC: %d  LastOp: 0x%02x\n",
                       dut->graph_status, dut->graph_pc, dut->graph_last_op);
                return false;
            }
        }

        if (!done) {
            printf("  FAIL: Timeout waiting for graph_done\n");
            return false;
        }
    }

    // Compare golden
    {
        std::string golden_path = casedir + "/golden.bin";
        size_t golden_sz;
        std::vector<uint8_t> golden(65536);
        FILE* gf = fopen(golden_path.c_str(), "rb");
        if (gf) {
            fseek(gf, 0, SEEK_END);
            golden_sz = ftell(gf);
            fseek(gf, 0, SEEK_SET);
            fread(golden.data(), 1, golden_sz, gf);
            fclose(gf);

            printf("  Comparing %zu bytes of output against golden...\n", golden_sz);

            int mismatches = 0;
            for (size_t i = 0; i < golden_sz; i++) {
                uint8_t actual = ddr_mem[DDR_GRAPH_IO_BASE + i];
                uint8_t expected = golden[i];
                if (actual != expected) {
                    if (mismatches < 20) {
                        printf("    MISMATCH at [%zu]: got 0x%02x (%d), expected 0x%02x (%d)\n",
                               i, actual, (int8_t)actual, expected, (int8_t)expected);
                    }
                    mismatches++;
                }
            }

            if (mismatches == 0) {
                printf("  PASS: All %zu bytes match golden!\n", golden_sz);
                return true;
            } else {
                printf("  FAIL: %d mismatches out of %zu bytes\n", mismatches, golden_sz);
                return false;
            }
        } else {
            printf("  No golden file found - skipping comparison (OK for structural test)\n");
            printf("  PASS: Graph execution completed successfully\n");
            return true;
        }
    }
}

// =============================================================================
// Main
// =============================================================================
int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);

    // Parse --datadir argument
    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "--datadir" && i + 1 < argc) {
            g_datadir = argv[i + 1];
            i++;
        }
    }

    printf("=== ONNX Fuzz Test (50 cases) ===\n");
    printf("Data directory: %s\n", g_datadir.c_str());

    dut = new Vonnx_sim_top;

    // VCD trace DISABLED for fuzz (too slow for 50 cases)
    Verilated::traceEverOn(false);
    tfp = nullptr;

    // Run all 50 cases
    int num_cases = 50;
    int pass_count = 0;

    for (int c = 0; c < num_cases; c++) {
        char case_name[64];
        snprintf(case_name, sizeof(case_name), "/case_%d", c);
        std::string casedir = g_datadir + case_name;

        printf("\n--- Fuzz Case %d/50 ---\n", c);

        bool passed = run_case(casedir);
        if (passed) {
            pass_count++;
        } else {
            printf("  Case %d FAILED\n", c);
        }
    }

    // Summary
    printf("\n========================================\n");
    printf("Fuzz: %d/%d passed\n", pass_count, num_cases);
    printf("========================================\n");

    if (pass_count == num_cases) {
        printf("PASS: All fuzz cases passed!\n");
    } else {
        printf("FAIL: %d fuzz cases failed\n", num_cases - pass_count);
    }

    // Run a few more cycles
    for (int i = 0; i < 20; i++) tick();

    delete dut;
    return (pass_count == num_cases) ? 0 : 1;
}
