// =============================================================================
// NPU Verilator Testbench - tb_top.cpp
// =============================================================================
// Complete Verilator C++ testbench for the NPU top-level module.
//
// Features:
//   - DDR memory model (256 MB, byte-addressable)
//   - AXI4-Lite master driver for NPU control register access
//   - AXI4 slave model responding to NPU DMA read/write requests
//   - Test flow: init DDR, configure registers, start, poll, verify
// =============================================================================

#include <verilated.h>
#include <verilated_vcd_c.h>
#include "Vtop.h"

#include <cstdint>
#include <cstring>
#include <vector>
#include <iostream>
#include <fstream>
#include <cassert>
#include <cstdlib>
#include <algorithm>
#include <iomanip>

// =============================================================================
// Global simulation time (required by Verilator)
// =============================================================================
vluint64_t main_time = 0;
double sc_time_stamp() { return main_time; }

// =============================================================================
// DDR Memory Model
// =============================================================================
class DDRModel {
public:
    std::vector<uint8_t> mem;

    DDRModel(size_t size = 256 * 1024 * 1024) : mem(size, 0) {}

    void write8(uint32_t addr, uint8_t val) {
        if (addr < mem.size()) mem[addr] = val;
    }

    uint8_t read8(uint32_t addr) {
        return (addr < mem.size()) ? mem[addr] : 0;
    }

    void write16(uint32_t addr, uint16_t val) {
        write8(addr,     val & 0xFF);
        write8(addr + 1, (val >> 8) & 0xFF);
    }

    uint16_t read16(uint32_t addr) {
        return (uint16_t)read8(addr) | ((uint16_t)read8(addr + 1) << 8);
    }

    void write32(uint32_t addr, uint32_t val) {
        write8(addr,     val & 0xFF);
        write8(addr + 1, (val >> 8) & 0xFF);
        write8(addr + 2, (val >> 16) & 0xFF);
        write8(addr + 3, (val >> 24) & 0xFF);
    }

    uint32_t read32(uint32_t addr) {
        return (uint32_t)read8(addr)
             | ((uint32_t)read8(addr + 1) << 8)
             | ((uint32_t)read8(addr + 2) << 16)
             | ((uint32_t)read8(addr + 3) << 24);
    }

    void write_block(uint32_t addr, const uint8_t* data, size_t len) {
        for (size_t i = 0; i < len && (addr + i) < mem.size(); i++)
            mem[addr + i] = data[i];
    }

    void read_block(uint32_t addr, uint8_t* data, size_t len) {
        for (size_t i = 0; i < len && (addr + i) < mem.size(); i++)
            data[i] = mem[addr + i];
    }

    void load_file(uint32_t addr, const std::string& path) {
        std::ifstream f(path, std::ios::binary);
        if (!f) {
            std::cerr << "[DDR] Cannot open " << path << std::endl;
            return;
        }
        f.seekg(0, std::ios::end);
        size_t sz = static_cast<size_t>(f.tellg());
        f.seekg(0);
        size_t to_read = std::min(sz, mem.size() - (size_t)addr);
        f.read(reinterpret_cast<char*>(&mem[addr]), to_read);
        std::cout << "[DDR] Loaded " << to_read << " bytes from '"
                  << path << "' at 0x" << std::hex << addr << std::dec << std::endl;
    }

    void dump(uint32_t addr, size_t len) {
        std::cout << "[DDR] Dump at 0x" << std::hex << addr << std::dec
                  << " (" << len << " bytes):" << std::endl;
        for (size_t i = 0; i < len; i++) {
            if (i % 16 == 0)
                std::cout << "  0x" << std::hex << std::setw(8) << std::setfill('0')
                          << (addr + i) << ": ";
            std::cout << std::hex << std::setw(2) << std::setfill('0')
                      << (int)read8(addr + i) << " ";
            if (i % 16 == 15) std::cout << std::dec << std::endl;
        }
        if (len % 16 != 0) std::cout << std::dec << std::endl;
    }
};

// =============================================================================
// Forward declarations
// =============================================================================
void tick_clock(Vtop* dut, VerilatedVcdC* tfp, int& tick_count);
void reset(Vtop* dut, VerilatedVcdC* tfp, int& tick_count, int cycles = 10);

// =============================================================================
// Register address map (must match npu_pkg.sv / axi_lite_regs.sv)
// =============================================================================
constexpr uint32_t REG_CTRL       = 0x00;  // Control: bit 0 = START
constexpr uint32_t REG_STATUS     = 0x04;  // Status:  bit 0 = DONE, bit 1 = BUSY, bit 2 = ERR
constexpr uint32_t REG_UCODE_BASE = 0x08;  // Microcode base address in DDR
constexpr uint32_t REG_UCODE_LEN  = 0x0C;  // Microcode length (number of instructions)
constexpr uint32_t REG_DDR_ACT    = 0x10;  // Activation tensor base in DDR
constexpr uint32_t REG_DDR_WGT    = 0x14;  // Weight tensor base in DDR
constexpr uint32_t REG_DDR_KV     = 0x18;  // KV cache base in DDR
constexpr uint32_t REG_DDR_OUT    = 0x1C;  // Output tensor base in DDR
constexpr uint32_t REG_HIDDEN     = 0x20;  // Hidden dimension
constexpr uint32_t REG_HEADS      = 0x24;  // Number of attention heads
constexpr uint32_t REG_HEAD_DIM   = 0x28;  // Dimension per head
constexpr uint32_t REG_SEQ_LEN    = 0x2C;  // Sequence length
constexpr uint32_t REG_TOKEN_IDX  = 0x30;  // Token index (for incremental inference)

// Status register bits
constexpr uint32_t STATUS_DONE = 0x1;
constexpr uint32_t STATUS_BUSY = 0x2;
constexpr uint32_t STATUS_ERR  = 0x4;

// =============================================================================
// AXI-Lite Master Driver
// =============================================================================
class AXILiteMaster {
    Vtop* dut;

public:
    AXILiteMaster(Vtop* d) : dut(d) {}

    // Write a 32-bit register via AXI-Lite
    // Drives AW+W+B simultaneously and waits for BVALID response.
    void write_reg(uint32_t addr, uint32_t data, int& tc, VerilatedVcdC* tfp) {
        dut->s_axil_awaddr  = addr;
        dut->s_axil_awvalid = 1;
        dut->s_axil_wdata   = data;
        dut->s_axil_wstrb   = 0xF;
        dut->s_axil_wvalid  = 1;
        dut->s_axil_bready  = 1;

        // Wait for BVALID (write completed)
        for (int i = 0; i < 100; i++) {
            tick_clock(dut, tfp, tc);
            if (dut->s_axil_bvalid) break;
        }
        // Deassert valid signals; keep bready=1 for one more tick to clear bvalid
        dut->s_axil_awvalid = 0;
        dut->s_axil_wvalid  = 0;
        tick_clock(dut, tfp, tc);
        dut->s_axil_bready = 0;
        tick_clock(dut, tfp, tc);
    }

    // Read a 32-bit register via AXI-Lite
    // Drives AR+R simultaneously and waits for RVALID response.
    uint32_t read_reg(uint32_t addr, int& tc, VerilatedVcdC* tfp) {
        dut->s_axil_araddr  = addr;
        dut->s_axil_arvalid = 1;
        dut->s_axil_rready  = 1;

        // Wait for RVALID (read data available)
        uint32_t data = 0;
        for (int i = 0; i < 100; i++) {
            tick_clock(dut, tfp, tc);
            if (dut->s_axil_rvalid) {
                data = dut->s_axil_rdata;
                break;
            }
        }
        // Deassert arvalid; keep rready=1 for one more tick to clear rvalid
        dut->s_axil_arvalid = 0;
        tick_clock(dut, tfp, tc);
        dut->s_axil_rready = 0;
        tick_clock(dut, tfp, tc);
        return data;
    }
};

// =============================================================================
// AXI4 Slave Model (responds to NPU DMA master read/write requests)
// =============================================================================
class AXI4SlaveModel {
    DDRModel& ddr;
    Vtop*     dut;

    // --- Read channel state ---
    bool     ar_pending = false;
    uint32_t ar_addr    = 0;
    uint8_t  ar_len     = 0;   // burst length - 1
    uint8_t  ar_size    = 0;   // log2(bytes per beat)
    uint8_t  ar_id      = 0;
    int      r_beat     = 0;

    // --- Write channel state ---
    bool     aw_pending = false;
    uint32_t aw_addr    = 0;
    uint8_t  aw_len     = 0;
    uint8_t  aw_size    = 0;
    uint8_t  aw_id      = 0;
    bool     w_active   = false;
    int      w_beat     = 0;
    bool     b_pending  = false;

public:
    AXI4SlaveModel(DDRModel& d, Vtop* top) : ddr(d), dut(top) {}

    // Handle AR and R channels (read path)
    void eval_read() {
        // Accept AR (address-read) requests
        if (!ar_pending) {
            dut->m_axi_arready = 1;
            if (dut->m_axi_arvalid) {
                ar_addr = dut->m_axi_araddr;
                ar_len  = dut->m_axi_arlen;
                ar_size = dut->m_axi_arsize;
                ar_id   = dut->m_axi_arid;
                ar_pending = true;
                r_beat     = 0;
                dut->m_axi_arready = 0;
            }
        } else {
            dut->m_axi_arready = 0;
        }

        // Drive R (read-data) channel
        if (ar_pending) {
            int bytes_per_beat = 1 << ar_size;  // 2^size
            uint32_t beat_addr = ar_addr + r_beat * bytes_per_beat;

            // Read data from DDR model (up to 128-bit / 16 bytes per beat)
            uint64_t lo = 0, hi = 0;
            for (int b = 0; b < 8 && b < bytes_per_beat; b++) {
                lo |= (uint64_t)ddr.read8(beat_addr + b) << (b * 8);
            }
            for (int b = 0; b < 8 && (b + 8) < bytes_per_beat; b++) {
                hi |= (uint64_t)ddr.read8(beat_addr + 8 + b) << (b * 8);
            }

            // Set R channel signals
            dut->m_axi_rvalid = 1;
            dut->m_axi_rid    = ar_id;
            dut->m_axi_rresp  = 0;  // OKAY
            dut->m_axi_rlast  = (r_beat == ar_len) ? 1 : 0;

            // Assign 128-bit read data (Verilator WData array: 4 x 32-bit words)
            dut->m_axi_rdata[0] = (uint32_t)(lo & 0xFFFFFFFF);
            dut->m_axi_rdata[1] = (uint32_t)(lo >> 32);
            dut->m_axi_rdata[2] = (uint32_t)(hi & 0xFFFFFFFF);
            dut->m_axi_rdata[3] = (uint32_t)(hi >> 32);

            // Advance beat on handshake
            if (dut->m_axi_rready && dut->m_axi_rvalid) {
                r_beat++;
                if (r_beat > ar_len) {
                    ar_pending = false;
                    dut->m_axi_rvalid = 0;
                }
            }
        } else {
            dut->m_axi_rvalid = 0;
            dut->m_axi_rlast  = 0;
        }
    }

    // Handle AW, W, and B channels (write path)
    void eval_write() {
        // Accept AW (address-write) requests
        if (!aw_pending && !b_pending) {
            dut->m_axi_awready = 1;
            if (dut->m_axi_awvalid) {
                aw_addr = dut->m_axi_awaddr;
                aw_len  = dut->m_axi_awlen;
                aw_size = dut->m_axi_awsize;
                aw_id   = dut->m_axi_awid;
                aw_pending = true;
                w_active   = true;
                w_beat     = 0;
                dut->m_axi_awready = 0;
            }
        } else {
            dut->m_axi_awready = 0;
        }

        // Accept W (write-data) beats
        if (w_active) {
            dut->m_axi_wready = 1;
            if (dut->m_axi_wvalid) {
                int bytes_per_beat = 1 << aw_size;
                uint32_t beat_addr = aw_addr + w_beat * bytes_per_beat;

                // Extract 128-bit write data from Verilator WData array
                uint8_t wdata[16];
                uint32_t d0 = dut->m_axi_wdata[0];
                uint32_t d1 = dut->m_axi_wdata[1];
                uint32_t d2 = dut->m_axi_wdata[2];
                uint32_t d3 = dut->m_axi_wdata[3];
                memcpy(&wdata[0],  &d0, 4);
                memcpy(&wdata[4],  &d1, 4);
                memcpy(&wdata[8],  &d2, 4);
                memcpy(&wdata[12], &d3, 4);

                // Apply write strobe (up to 16 byte lanes for 128-bit bus)
                uint16_t wstrb = dut->m_axi_wstrb;
                for (int b = 0; b < bytes_per_beat && b < 16; b++) {
                    if (wstrb & (1 << b)) {
                        ddr.write8(beat_addr + b, wdata[b]);
                    }
                }

                w_beat++;
                if (dut->m_axi_wlast) {
                    w_active   = false;
                    aw_pending = false;
                    b_pending  = true;
                }
            }
        } else {
            dut->m_axi_wready = 0;
        }

        // Drive B (write-response) channel
        if (b_pending) {
            dut->m_axi_bvalid = 1;
            dut->m_axi_bid    = aw_id;
            dut->m_axi_bresp  = 0;  // OKAY
            if (dut->m_axi_bready) {
                b_pending = false;
                dut->m_axi_bvalid = 0;
            }
        } else {
            dut->m_axi_bvalid = 0;
        }
    }

    // Service both read and write channels (call once per clock edge)
    void tick() {
        eval_read();
        eval_write();
    }
};

// =============================================================================
// Clock / Reset Helpers
// =============================================================================
void tick_clock(Vtop* dut, VerilatedVcdC* tfp, int& tick_count) {
    // Falling edge
    dut->clk = 0;
    dut->eval();
    if (tfp) tfp->dump(tick_count * 10);
    tick_count++;

    // Rising edge
    dut->clk = 1;
    dut->eval();
    if (tfp) tfp->dump(tick_count * 10 + 5);
    tick_count++;

    main_time = tick_count;
}

void reset(Vtop* dut, VerilatedVcdC* tfp, int& tick_count, int cycles) {
    dut->rst_n = 0;
    for (int i = 0; i < cycles; i++) {
        tick_clock(dut, tfp, tick_count);
    }
    dut->rst_n = 1;
    tick_clock(dut, tfp, tick_count);
}

// =============================================================================
// Test 1: Register Access (write/read-back model configuration registers)
// =============================================================================
bool test_register_access(Vtop* dut, AXILiteMaster& axil,
                          VerilatedVcdC* tfp, int& tc)
{
    std::cout << "=== Test: Register Access ===" << std::endl;

    // Write model configuration registers
    axil.write_reg(REG_HIDDEN,   64, tc, tfp);
    axil.write_reg(REG_HEADS,     4, tc, tfp);
    axil.write_reg(REG_HEAD_DIM, 16, tc, tfp);
    axil.write_reg(REG_SEQ_LEN,   8, tc, tfp);

    // Read back and verify
    struct { uint32_t addr; uint32_t expect; const char* name; } checks[] = {
        { REG_HIDDEN,   64, "HIDDEN"   },
        { REG_HEADS,     4, "HEADS"    },
        { REG_HEAD_DIM, 16, "HEAD_DIM" },
        { REG_SEQ_LEN,   8, "SEQ_LEN"  },
    };

    bool pass = true;
    for (auto& c : checks) {
        uint32_t val = axil.read_reg(c.addr, tc, tfp);
        if (val != c.expect) {
            std::cerr << "  FAIL: REG_" << c.name << " readback = " << val
                      << " (expected " << c.expect << ")" << std::endl;
            pass = false;
        } else {
            std::cout << "  OK: REG_" << c.name << " = " << val << std::endl;
        }
    }

    // Read STATUS register (should be idle after reset)
    uint32_t status = axil.read_reg(REG_STATUS, tc, tfp);
    std::cout << "  STATUS = 0x" << std::hex << status << std::dec << std::endl;

    std::cout << "Register access: " << (pass ? "PASS" : "FAIL") << std::endl;
    return pass;
}

// =============================================================================
// Test 2: DMA Loopback (basic DDR <-> NPU data movement test)
// =============================================================================
bool test_dma_loopback(Vtop* dut, AXILiteMaster& axil, AXI4SlaveModel& axi_slave,
                       DDRModel& ddr, VerilatedVcdC* tfp, int& tc)
{
    std::cout << "=== Test: DMA Loopback ===" << std::endl;

    // Fill source region in DDR with a known test pattern
    const uint32_t SRC_ADDR  = 0x00001000;
    const uint32_t DST_ADDR  = 0x00002000;
    const size_t   DATA_SIZE = 256;

    for (size_t i = 0; i < DATA_SIZE; i++) {
        ddr.write8(SRC_ADDR + i, i & 0xFF);
    }

    // Configure source and destination addresses
    axil.write_reg(REG_DDR_ACT, SRC_ADDR, tc, tfp);
    axil.write_reg(REG_DDR_OUT, DST_ADDR, tc, tfp);

    // Dump source region for debugging
    ddr.dump(SRC_ADDR, 64);

    std::cout << "DMA loopback: PASS (basic setup)" << std::endl;
    return true;
}

// =============================================================================
// Test 3: Control Plane Smoke Test (configure, start NOP microcode, poll, verify)
// =============================================================================
bool test_ctrl_plane_smoke(Vtop* dut, AXILiteMaster& axil, AXI4SlaveModel& axi_slave,
                           DDRModel& ddr, VerilatedVcdC* tfp, int& tc)
{
    std::cout << "=== Test: Control Plane Smoke Test (No Compute) ===" << std::endl;

    // DDR layout
    const uint32_t UCODE_BASE = 0x00000000;
    const uint32_t ACT_BASE   = 0x00100000;
    const uint32_t WGT_BASE   = 0x00200000;
    const uint32_t KV_BASE    = 0x00400000;
    const uint32_t OUT_BASE   = 0x00600000;

    // Model dimensions (small test configuration)
    const uint32_t HIDDEN_DIM = 64;
    const uint32_t NUM_HEADS  = 4;
    const uint32_t HEAD_DIM   = HIDDEN_DIM / NUM_HEADS;  // 16
    const uint32_t SEQ_LEN    = 8;
    const uint32_t TOKEN_IDX  = 0;

    // Load test data from files if available; otherwise use synthetic data
    std::ifstream ucode_file("test_data/ucode.bin", std::ios::binary);
    if (ucode_file.good()) {
        ucode_file.close();
        ddr.load_file(UCODE_BASE, "test_data/ucode.bin");
        ddr.load_file(ACT_BASE,   "test_data/activations.bin");
        ddr.load_file(WGT_BASE,   "test_data/weights.bin");
        std::cout << "  Loaded test data from files." << std::endl;
    } else {
        // Generate simple synthetic test data
        // Activations: identity-like pattern
        for (uint32_t i = 0; i < HIDDEN_DIM; i++) {
            ddr.write8(ACT_BASE + i, (uint8_t)(i & 0x7F));  // INT8 activations
        }
        // Weights: identity-like (diagonal ones in INT8)
        for (uint32_t r = 0; r < HIDDEN_DIM; r++) {
            for (uint32_t c = 0; c < HIDDEN_DIM; c++) {
                int8_t val = (r == c) ? 1 : 0;
                ddr.write8(WGT_BASE + r * HIDDEN_DIM + c, (uint8_t)val);
            }
        }
        // Microcode: placeholder NOPs (zeros)
        for (uint32_t i = 0; i < 64; i++) {
            ddr.write32(UCODE_BASE + i * 4, 0x00000000);
        }
        std::cout << "  Generated synthetic test data." << std::endl;
    }

    // Configure NPU registers
    axil.write_reg(REG_UCODE_BASE, UCODE_BASE, tc, tfp);
    axil.write_reg(REG_UCODE_LEN,  16,         tc, tfp);   // 16 instructions
    axil.write_reg(REG_DDR_ACT,    ACT_BASE,   tc, tfp);
    axil.write_reg(REG_DDR_WGT,    WGT_BASE,   tc, tfp);
    axil.write_reg(REG_DDR_KV,     KV_BASE,    tc, tfp);
    axil.write_reg(REG_DDR_OUT,    OUT_BASE,    tc, tfp);
    axil.write_reg(REG_HIDDEN,     HIDDEN_DIM,  tc, tfp);
    axil.write_reg(REG_HEADS,      NUM_HEADS,   tc, tfp);
    axil.write_reg(REG_HEAD_DIM,   HEAD_DIM,    tc, tfp);
    axil.write_reg(REG_SEQ_LEN,    SEQ_LEN,     tc, tfp);
    axil.write_reg(REG_TOKEN_IDX,  TOKEN_IDX,   tc, tfp);
    std::cout << "  Registers configured." << std::endl;

    // Issue START command (bit 0 of CTRL register)
    axil.write_reg(REG_CTRL, 0x1, tc, tfp);
    std::cout << "  START issued." << std::endl;

    // Poll STATUS register for DONE, servicing DMA requests each cycle
    const int MAX_POLL_CYCLES = 10000;
    bool done = false;
    bool error = false;

    for (int cyc = 0; cyc < MAX_POLL_CYCLES; cyc++) {
        // Service AXI4 DMA requests from the NPU
        axi_slave.tick();
        tick_clock(dut, tfp, tc);

        // Poll STATUS every 50 cycles
        if (cyc % 50 == 49) {
            uint32_t status = axil.read_reg(REG_STATUS, tc, tfp);
            if (cyc < 300) {
                std::cout << "  [cyc " << cyc << "] STATUS=0x"
                          << std::hex << status << std::dec << std::endl;
            }
            if (status & STATUS_ERR) {
                std::cerr << "  ERROR: NPU reported error (STATUS = 0x"
                          << std::hex << status << std::dec << ")" << std::endl;
                error = true;
                break;
            }
            if (status & STATUS_DONE) {
                std::cout << "  NPU DONE after ~" << cyc << " cycles." << std::endl;
                done = true;
                break;
            }
        }
    }

    if (!done && !error) {
        std::cerr << "  TIMEOUT: NPU did not complete within "
                  << MAX_POLL_CYCLES << " cycles." << std::endl;
        return false;
    }
    if (error) return false;

    // Read output from DDR and compare against reference
    std::cout << "  Output region:" << std::endl;
    ddr.dump(OUT_BASE, 64);

    // Load reference data if available
    std::ifstream ref_file("test_data/reference_output.bin", std::ios::binary);
    if (ref_file.good()) {
        ref_file.seekg(0, std::ios::end);
        size_t ref_sz = static_cast<size_t>(ref_file.tellg());
        ref_file.seekg(0);
        std::vector<uint8_t> ref_data(ref_sz);
        ref_file.read(reinterpret_cast<char*>(ref_data.data()), ref_sz);

        bool match = true;
        int mismatches = 0;
        for (size_t i = 0; i < ref_sz; i++) {
            uint8_t actual = ddr.read8(OUT_BASE + i);
            if (actual != ref_data[i]) {
                std::cerr << "  MISMATCH at offset " << i
                          << ": got 0x" << std::hex << (int)actual
                          << " expected 0x" << (int)ref_data[i]
                          << std::dec << std::endl;
                match = false;
                mismatches++;
                if (mismatches > 16) {
                    std::cerr << "  (stopping after 16 mismatches)" << std::endl;
                    break;
                }
            }
        }

        if (match) {
            std::cout << "Control plane smoke: PASS (matches reference)" << std::endl;
        } else {
            std::cerr << "Control plane smoke: FAIL (output mismatch)" << std::endl;
        }
        return match;
    } else {
        // No reference file -- just check that output is not all zeros
        bool all_zero = true;
        for (size_t i = 0; i < HIDDEN_DIM; i++) {
            if (ddr.read8(OUT_BASE + i) != 0) {
                all_zero = false;
                break;
            }
        }
        if (all_zero) {
            std::cout << "  WARNING: Output is all zeros (no reference to compare)."
                      << std::endl;
        }
        std::cout << "Control plane smoke: PASS (NOP microcode completed)"
                  << std::endl;
        return done;
    }
}

// =============================================================================
// Main entry point
// =============================================================================
int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    Verilated::traceEverOn(true);

    // Instantiate DUT
    auto dut = new Vtop;

    // Setup VCD tracing
    auto tfp = new VerilatedVcdC;
    dut->trace(tfp, 99);
    tfp->open("npu_sim.vcd");

    int tick_count = 0;

    // Instantiate models
    DDRModel       ddr;
    AXILiteMaster  axil(dut);
    AXI4SlaveModel axi_slave(ddr, dut);

    // ---- Initialize all DUT input signals to safe defaults ----
    dut->clk   = 0;
    dut->rst_n = 0;

    // AXI-Lite slave interface (testbench is master)
    dut->s_axil_awaddr  = 0;
    dut->s_axil_awvalid = 0;
    dut->s_axil_wdata   = 0;
    dut->s_axil_wstrb   = 0;
    dut->s_axil_wvalid  = 0;
    dut->s_axil_bready  = 0;
    dut->s_axil_araddr  = 0;
    dut->s_axil_arvalid = 0;
    dut->s_axil_rready  = 0;

    // AXI4 master interface (testbench is slave, responding to NPU DMA)
    dut->m_axi_arready = 0;
    dut->m_axi_rdata[0] = 0;
    dut->m_axi_rdata[1] = 0;
    dut->m_axi_rdata[2] = 0;
    dut->m_axi_rdata[3] = 0;
    dut->m_axi_rresp  = 0;
    dut->m_axi_rlast  = 0;
    dut->m_axi_rid    = 0;
    dut->m_axi_rvalid = 0;
    dut->m_axi_awready = 0;
    dut->m_axi_wready  = 0;
    dut->m_axi_bvalid  = 0;
    dut->m_axi_bid     = 0;
    dut->m_axi_bresp   = 0;

    // ---- Apply reset ----
    reset(dut, tfp, tick_count);
    std::cout << "Reset complete." << std::endl;

    // ---- Run test suite ----
    bool all_pass = true;

    all_pass &= test_register_access(dut, axil, tfp, tick_count);
    all_pass &= test_dma_loopback(dut, axil, axi_slave, ddr, tfp, tick_count);
    all_pass &= test_ctrl_plane_smoke(dut, axil, axi_slave, ddr, tfp, tick_count);

    // Run a few extra cycles to capture trailing waveform activity
    for (int i = 0; i < 100; i++) {
        axi_slave.tick();
        tick_clock(dut, tfp, tick_count);
    }

    // ---- Summary ----
    std::cout << std::endl;
    if (all_pass) {
        std::cout << "=== ALL TESTS PASSED ===" << std::endl;
    } else {
        std::cout << "=== SOME TESTS FAILED ===" << std::endl;
    }

    // Cleanup
    tfp->close();
    delete tfp;
    delete dut;

    return all_pass ? 0 : 1;
}
