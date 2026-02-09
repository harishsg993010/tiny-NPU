# =============================================================================
# NPU - Transformer Inference Accelerator
# Top-level Makefile
# =============================================================================

# Directories
RTL_DIR     := rtl
SIM_DIR     := sim/verilator
PY_DIR      := python
BUILD_DIR   := build
WAVE_DIR    := waves
LUT_DIR     := rtl/ops

# Verilator
VERILATOR   ?= verilator
VERILATOR_FLAGS := --cc --trace --trace-structs -Wall \
    -Wno-UNUSED -Wno-UNDRIVEN -Wno-PINCONNECTEMPTY \
    --x-assign unique --x-initial unique \
    -DSIMULATION \
    -I$(RTL_DIR)/pkg -I$(RTL_DIR)/bus -I$(RTL_DIR)/mem \
    -I$(RTL_DIR)/ctrl -I$(RTL_DIR)/gemm -I$(RTL_DIR)/ops

# SystemVerilog sources (order matters for packages)
SV_PKG := \
    $(RTL_DIR)/pkg/npu_pkg.sv \
    $(RTL_DIR)/pkg/isa_pkg.sv \
    $(RTL_DIR)/pkg/fixed_pkg.sv \
    $(RTL_DIR)/bus/axi_types.sv

SV_SRC := \
    $(RTL_DIR)/bus/axi_lite_regs.sv \
    $(RTL_DIR)/bus/axi_dma_rd.sv \
    $(RTL_DIR)/bus/axi_dma_wr.sv \
    $(RTL_DIR)/mem/sram_dp.sv \
    $(RTL_DIR)/mem/banked_sram.sv \
    $(RTL_DIR)/mem/kv_cache_bank.sv \
    $(RTL_DIR)/ctrl/addr_gen.sv \
    $(RTL_DIR)/ctrl/scoreboard.sv \
    $(RTL_DIR)/ctrl/barrier.sv \
    $(RTL_DIR)/ctrl/ucode_fetch.sv \
    $(RTL_DIR)/ctrl/ucode_decode.sv \
    $(RTL_DIR)/gemm/mac_int8.sv \
    $(RTL_DIR)/gemm/pe.sv \
    $(RTL_DIR)/gemm/systolic_array.sv \
    $(RTL_DIR)/gemm/gemm_ctrl.sv \
    $(RTL_DIR)/gemm/gemm_post.sv \
    $(RTL_DIR)/ops/vec_engine.sv \
    $(RTL_DIR)/ops/reduce_max.sv \
    $(RTL_DIR)/ops/reduce_sum.sv \
    $(RTL_DIR)/ops/exp_lut.sv \
    $(RTL_DIR)/ops/recip_lut.sv \
    $(RTL_DIR)/ops/softmax_engine.sv \
    $(RTL_DIR)/ops/mean_var_engine.sv \
    $(RTL_DIR)/ops/rsqrt_lut.sv \
    $(RTL_DIR)/ops/layernorm_engine.sv \
    $(RTL_DIR)/ops/gelu_lut.sv \
    $(RTL_DIR)/ops/gelu_engine.sv \
    $(RTL_DIR)/top.sv

ALL_SV := $(SV_PKG) $(SV_SRC)

TB_CPP := $(SIM_DIR)/tb_top.cpp

TOP_MODULE := top

# Python
PYTHON ?= python3

# Targets
.PHONY: all sim test luts wave clean lint help cmake_sim

all: sim

# ---- Verilator Simulation (direct Makefile) ----
sim: $(BUILD_DIR)/Vtop
	@echo "=== Running NPU Simulation ==="
	cd $(BUILD_DIR) && ./Vtop +trace
	@echo "=== Simulation Complete ==="

$(BUILD_DIR)/Vtop: $(ALL_SV) $(TB_CPP)
	@mkdir -p $(BUILD_DIR)
	$(VERILATOR) $(VERILATOR_FLAGS) \
		--top-module $(TOP_MODULE) \
		--prefix Vtop \
		--Mdir $(BUILD_DIR)/obj_dir \
		--exe $(abspath $(TB_CPP)) \
		$(ALL_SV)
	$(MAKE) -C $(BUILD_DIR)/obj_dir -f Vtop.mk Vtop
	cp $(BUILD_DIR)/obj_dir/Vtop $(BUILD_DIR)/Vtop

# ---- CMake-based Verilator build ----
cmake_sim:
	@mkdir -p $(BUILD_DIR)/cmake
	cd $(BUILD_DIR)/cmake && cmake ../../$(SIM_DIR) && make -j$$(nproc)
	@echo "Built via CMake: $(BUILD_DIR)/cmake/npu_sim"

# ---- Python Golden Model Tests ----
test:
	@echo "=== Running Python Golden Model Tests ==="
	cd $(PY_DIR) && $(PYTHON) -m tests.test_end2end
	@echo "=== Tests Complete ==="

# ---- Generate LUT files ----
luts:
	@echo "=== Generating LUT files ==="
	$(PYTHON) $(PY_DIR)/tools/make_lut.py -o $(LUT_DIR) --format both
	@echo "=== LUTs Generated ==="

# ---- Waveform viewing ----
wave: sim
	@mkdir -p $(WAVE_DIR)
	@if [ -f $(BUILD_DIR)/npu_sim.vcd ]; then \
		cp $(BUILD_DIR)/npu_sim.vcd $(WAVE_DIR)/; \
		echo "VCD file: $(WAVE_DIR)/npu_sim.vcd"; \
		echo "To view: gtkwave $(WAVE_DIR)/npu_sim.vcd &"; \
	else \
		echo "No VCD found. Run 'make sim' first."; \
	fi

# ---- Linting ----
lint:
	@echo "=== Running Verilator Lint ==="
	$(VERILATOR) --lint-only $(VERILATOR_FLAGS) \
		--top-module $(TOP_MODULE) \
		$(ALL_SV)
	@echo "=== Lint Clean ==="

# ---- Generate microcode for tiny test ----
ucode:
	@echo "=== Generating Microcode ==="
	$(PYTHON) $(PY_DIR)/tools/ucode_asm.py --gen-tiny --hex -o $(BUILD_DIR)/ucode.hex
	@echo "=== Microcode Generated ==="

# ---- Clean ----
clean:
	rm -rf $(BUILD_DIR)
	rm -rf $(WAVE_DIR)
	rm -f $(LUT_DIR)/*.mem
	rm -f $(LUT_DIR)/*_init.sv
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	@echo "=== Cleaned ==="

# ---- Help ----
help:
	@echo "NPU Transformer Accelerator - Build Targets"
	@echo "============================================"
	@echo "  make sim       - Build and run Verilator simulation"
	@echo "  make cmake_sim - Build via CMake (alternative)"
	@echo "  make test      - Run Python golden model tests"
	@echo "  make luts      - Generate LUT ROM files"
	@echo "  make wave      - Generate VCD and show gtkwave instructions"
	@echo "  make lint      - Run Verilator lint checks"
	@echo "  make ucode     - Generate microcode for tiny test"
	@echo "  make clean     - Remove build artifacts"
	@echo "  make help      - Show this help"
