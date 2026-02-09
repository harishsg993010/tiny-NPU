# tiny-npu

A minimal transformer inference accelerator in SystemVerilog, optimized for learning how NPUs (Neural Processing Units) work from the ground up.

Built with fully documented SystemVerilog RTL, a complete 128-bit microcode ISA, working GPT-2 inference running on real HuggingFace weights, KV-cache optimization, and full Verilator simulation with cycle-accurate verification.

### Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
  - [NPU](#npu)
  - [Memory](#memory)
  - [Engines](#engines)
- [ISA](#isa)
- [Execution](#execution)
  - [Microcode Controller](#microcode-controller)
  - [Transformer Block](#transformer-block)
  - [KV-Cache Decode](#kv-cache-decode)
- [Inference Demo](#inference-demo)
- [Simulation](#simulation)
- [Advanced Functionality](#advanced-functionality)
- [Next Steps](#next-steps)

# Overview

If you want to learn how a CPU or GPU works at a hardware level, there are excellent resources available - from textbooks to open-source implementations like [tiny-gpu](https://github.com/adam-maj/tiny-gpu).

NPUs are a different story.

The AI hardware market is one of the most competitive in the industry. The internal architectures of production NPUs from NVIDIA (Tensor Cores), Google (TPU), Apple (Neural Engine), and others remain proprietary. While there are plenty of resources on *using* these accelerators through frameworks like PyTorch and TensorFlow, there's almost nothing to help you understand how they actually work at the RTL level.

The best option is to go through research papers and try to reverse-engineer what's happening, or dig through complex open-source implementations that prioritize completeness over readability.

This is why I built `tiny-npu`.

## What is tiny-npu?

> [!IMPORTANT]
>
> **tiny-npu** is a minimal, fully synthesizable transformer inference accelerator in SystemVerilog, optimized for learning about how NPUs work from the ground up.
>
> Rather than building a general-purpose matrix accelerator, tiny-npu focuses on implementing every operation needed to run a real transformer model end-to-end - from GEMM to Softmax to LayerNorm - so you can see exactly how silicon turns weights into words.

With this motivation in mind, we can strip away the complexity of production-grade accelerators (multi-chip interconnects, HBM controllers, sparsity engines) and focus on the core elements that make transformer inference work in hardware:

1. **Systolic Array** - How does a 16x16 grid of multiply-accumulate units perform matrix multiplication?
2. **Fixed-Point Arithmetic** - How do you compute softmax, layer normalization, and GELU without floating point?
3. **Microcode Sequencing** - How does a controller orchestrate dozens of operations to execute a full transformer block?
4. **Memory Management** - How do you fit weights, activations, and intermediate results in limited on-chip SRAM?
5. **KV-Cache Optimization** - How does caching key/value vectors make autoregressive decoding faster?

The result: a chip that loads real GPT-2 weights from HuggingFace, runs INT8 quantized inference through 4 transformer layers, and generates text - all verified cycle-accurate against a Python golden model.

# Architecture

```
                           AXI4-Lite Host Interface
                          (Control Registers + Start)
                                     |
                    +----------------v-----------------+
                    |       Microcode Controller        |
                    |   Fetch --> Decode --> Dispatch    |
                    |   + Scoreboard (6 engine slots)   |
                    |   + Barrier synchronization       |
                    +--+----+----+----+----+----+------+
                       |    |    |    |    |    |
            +----------+    |    |    |    |    +----------+
            |               |    |    |    |               |
    +-------v------+ +------v--+ | +--v------+  +---------v--------+
    |  DMA Engine  | |  GEMM   | | | Softmax |  |   Vec Engine     |
    |  AXI4 Master | | 16x16   | | | 3-pass  |  |  (Add/Mul/Scale) |
    |  Read + Write| | Systolic| | | LUT-    |  +------------------+
    +--------------+ | Array   | | | based   |
                     | 256 MACs| | +---------+
                     +---------+ |
                         +-------v-------+  +----------+
                         |  LayerNorm    |  |   GELU   |
                         |  Engine       |  |  Engine  |
                         |  (mean+var+   |  | (LUT-    |
                         |   normalize)  |  |  based)  |
                         +---------------+  +----------+
                                  |
                    +-------------v--------------+
                    |     On-Chip SRAM Banks      |
                    | SRAM0 (64KB) | SRAM1 (8KB) |
                    |  Weights +   |  LN Betas + |
                    |  Activations |  Residuals  |
                    +-----------------------------+
```

## NPU

tiny-npu is built to execute one transformer block at a time, layer by layer.

The host CPU orchestrates inference by:

1. Loading quantized INT8 weights into external DDR memory
2. Writing the microcode program for one block into SRAM
3. Loading per-layer weights from DDR into on-chip SRAM via DMA
4. Starting the microcode controller which runs the block autonomously
5. Reading back the output activations and repeating for the next layer

The NPU itself consists of the following units:

### Control Registers (AXI4-Lite)

The AXI4-Lite slave interface exposes configuration registers that the host uses to set up execution. These include:

| Register | Address | Description |
|----------|---------|-------------|
| `CTRL` | 0x00 | Start bit + soft reset |
| `STATUS` | 0x04 | Done / busy / error flags |
| `UCODE_BASE` | 0x08 | DDR address of microcode program |
| `UCODE_LEN` | 0x0C | Number of microcode instructions |
| `DDR_BASE_WGT` | 0x14 | DDR base address for weights |
| `MODEL_HIDDEN` | 0x20 | Hidden dimension |
| `SEQ_LEN` | 0x2C | Current sequence length |

### Microcode Controller

The sequencing brain of the NPU. It fetches 128-bit microcode instructions from SRAM, decodes them into engine-specific commands, and dispatches them to the appropriate hardware engine.

A **scoreboard** tracks which of the 6 engines are currently busy, ensuring instructions only dispatch when their target engine is free. A **barrier** instruction forces the controller to stall until all engines complete, which is needed between dependent operations (e.g., wait for GEMM before starting Softmax).

### Dispatcher

Unlike a GPU where a dispatcher distributes threads across cores, the NPU dispatcher routes decoded instructions to the correct engine. Each instruction's opcode determines which engine receives the work. The scoreboard prevents dispatch conflicts, and the barrier instruction provides explicit synchronization.

## Memory

### On-Chip SRAM

tiny-npu uses two on-chip SRAM banks for all computation:

**SRAM0 (64KB)** - The main workspace. Holds all weights for one transformer block (48KB) plus all intermediate activations (15KB). The memory map is carefully designed so that 6 weight matrices (Wq, Wk, Wv, Wo, W1, W2) and all intermediate tensors fit simultaneously:

```
SRAM0 Memory Map (one transformer block)
=========================================
0x0000 +-----------+
       |  Wq       |  4096B  [64][64]   Query weights
0x1000 +-----------+
       |  Wk       |  4096B  [64][64]   Key weights
0x2000 +-----------+
       |  Wv       |  4096B  [64][64]   Value weights
0x3000 +-----------+
       |  Wo       |  4096B  [64][64]   Output projection
0x4000 +-----------+
       |  W1       | 16384B  [64][256]  FFN up-project
0x8000 +-----------+
       |  W2       | 16384B  [256][64]  FFN down-project
0xC000 +-----------+
       |  X        |  1024B  [16][64]   Input activations
0xC400 |  LN1_OUT  |  1024B             LayerNorm output
0xC800 |  Q        |  1024B             Query
0xCC00 |  K        |  1024B             Key
0xD000 |  V        |  1024B             Value
0xD400 |  S        |   256B  [16][16]   Attention scores
0xD500 |  P        |   256B  [16][16]   Softmax output
0xD600 |  ATTN     |  1024B             Attention context
0xDA00 |  WO_OUT   |  1024B             Output projection result
0xDE00 |  X2       |  1024B             First residual
0xE200 |  LN2_OUT  |  1024B             Second LayerNorm
0xE600 |  FFN1     |  4096B  [16][256]  FFN up-project output
0xF600 |  FFN2     |  1024B             FFN down-project output
0xFA00 |  X_OUT    |  1024B             Block output
0xFE00 +-----------+
       Total: ~63.5KB of 64KB used
```

Every byte has a purpose. The entire 64KB address space is utilized, with only ~512 bytes to spare. This tight packing is a direct consequence of the ISA using 16-bit address fields (max 64KB addressable).

**SRAM1 (8KB)** - Auxiliary storage for LayerNorm beta parameters and residual connections. Separated from SRAM0 because the vec engine needs to read from both SRAM0 and SRAM1 simultaneously for residual adds.

### External Memory (DDR)

Weights for all layers are stored in external DDR memory and loaded one block at a time via DMA. The weight file (`weights.bin`) is ~226KB and contains:

| Section | Offset | Size | Description |
|---------|--------|------|-------------|
| WTE | 0 | 16KB | Token embedding table [256][64] |
| WPE | 16384 | 1KB | Position embedding table [16][64] |
| Block 0 | 17408 | 48KB | Layer 0 weights (LN betas + 6 matrices) |
| Block 1 | 66688 | 48KB | Layer 1 weights |
| Block 2 | 115968 | 48KB | Layer 2 weights |
| Block 3 | 165248 | 48KB | Layer 3 weights |
| LN_F beta | 214528 | 64B | Final LayerNorm beta |
| LM head | 214592 | 16KB | Language model head [256][64] |

## Engines

Each engine is a specialized hardware unit optimized for one class of operation. All engines share access to SRAM0/SRAM1 and report completion back to the scoreboard.

### GEMM Engine (Engine 0)

The core compute unit. A **16x16 weight-stationary systolic array** that performs INT8 matrix multiplication with INT32 accumulation.

```
Weight-Stationary Systolic Array (16x16)
=========================================

  Activations flow DOWN (one row per cycle)
         |     |     |     |           |
         v     v     v     v           v
       +-----+-----+-----+-----   +-----+
  -->  | MAC | MAC | MAC | MAC ... | MAC |  --> partial sums flow RIGHT
       +-----+-----+-----+-----   +-----+
  -->  | MAC | MAC | MAC | MAC ... | MAC |  -->
       +-----+-----+-----+-----   +-----+
  -->  | MAC | MAC | MAC | MAC ... | MAC |  -->
       +-----+-----+-----+-----   +-----+
         :     :     :     :           :
       +-----+-----+-----+-----   +-----+
  -->  | MAC | MAC | MAC | MAC ... | MAC |  -->  Output
       +-----+-----+-----+-----   +-----+

  256 MACs = 256 INT8 multiplies per cycle
  Each MAC: accumulator += activation * weight (INT32)
```

For matrices larger than 16x16, the **GEMM controller** tiles the computation automatically. A `[16][64] * [64][256]` GEMM is broken into `(1)(4)(16) = 64` tiles of 16x16, with partial sums accumulated across the K dimension.

After accumulation, the **post-processing unit** applies requantization: `result_i8 = clamp(round((acc * scale) >> shift))`. This keeps all intermediate activations in INT8, minimizing SRAM bandwidth requirements.

### Softmax Engine (Engine 1)

A 3-pass fixed-point softmax:

1. **Pass 1 - Max**: Find the maximum value across the row (for numerical stability)
2. **Pass 2 - Exp + Sum**: Compute `exp(x - max)` using a 256-entry LUT, accumulate the sum
3. **Pass 3 - Normalize**: Multiply each exp value by `1/sum` using a reciprocal LUT

Supports an optional **causal mask** flag that masks out future positions for autoregressive attention.

### LayerNorm Engine (Engine 2)

Two-pass layer normalization:

1. **Pass 1**: Compute mean and variance across the hidden dimension using a dedicated `mean_var_engine`
2. **Pass 2**: Normalize each element: `(x - mean) * rsqrt(var + eps) + beta`, using an inverse-sqrt LUT

The beta (bias) parameter is read from SRAM1 while the input is read from SRAM0, enabling simultaneous access.

### GELU Engine (Engine 3)

The GELU activation function approximated via a 256-entry lookup table mapping INT8 inputs to INT8 outputs. Applied elementwise to the FFN intermediate activations.

### Vec Engine (Engine 4)

A general-purpose vector unit for elementwise operations:

- **VEC_ADD** - Saturating INT8 addition (used for residual connections: `x + sublayer_output`)
- **VEC_MUL** - Elementwise multiply
- **VEC_SCALE_SHIFT** - Scale and shift
- **VEC_CLAMP** - Clamp to range

### DMA Engine (Engine 5)

Transfers data between external DDR memory and on-chip SRAM via AXI4 burst transactions. The DMA engine handles both read (DDR -> SRAM) and write (SRAM -> DDR) directions with configurable burst lengths.

In simulation, DMA transfers are intercepted by a C++ software shim that models the DDR memory, eliminating the need for an actual AXI interconnect.

# ISA

tiny-npu implements a 128-bit fixed-width microcode ISA. Every instruction is the same size and format, which simplifies the hardware decoder at the cost of some encoding efficiency.

```
128-bit Microcode Instruction Format
======================================

  127        112 111         96 95          80 79          64
  +-----------+--------------+--------------+--------------+
  |    imm    |      K       |      N       |      M       |
  | (16 bits) |  (16 bits)   |  (16 bits)   |  (16 bits)   |
  +-----------+--------------+--------------+--------------+

  63          48 47          32 31          16 15    8 7    0
  +-----------+--------------+--------------+-------+------+
  |   src1    |    src0      |     dst      | flags |opcode|
  | (16 bits) |  (16 bits)   |  (16 bits)   |(8 bit)|(8bit)|
  +-----------+--------------+--------------+-------+------+
```

The fields serve different purposes depending on the opcode:

| Field | GEMM | Softmax/LN/GELU | VEC | DMA | KV_APPEND/READ |
|-------|------|-----------------|-----|-----|----------------|
| `dst` | Output address | Output address | Output address | SRAM address | SRAM address |
| `src0` | Activation address | Input address | Input A address | DDR address | Cache address |
| `src1` | Weight address | Beta (SRAM1) | Input B address | - | - |
| `M` | Rows | Rows | Rows | Byte count | Layer ID |
| `N` | Columns | Hidden dim | Length | - | Vector length |
| `K` | Inner dim | - | - | - | Time position |
| `imm` | scale\|shift | - | - | - | - |
| `flags` | transpose, requant | causal mask | sub-op | direction | - |

### Opcodes

| Code | Mnemonic | Engine | Description |
|------|----------|--------|-------------|
| 0 | `NOP` | - | No operation |
| 1 | `DMA_LOAD` | DMA | Copy data from DDR to SRAM |
| 2 | `DMA_STORE` | DMA | Store data from SRAM to DDR |
| 3 | `GEMM` | GEMM | Matrix multiply via systolic array |
| 4 | `VEC` | Vec | Vector elementwise operation |
| 5 | `SOFTMAX` | Softmax | Row-wise softmax with optional causal mask |
| 6 | `LAYERNORM` | LayerNorm | Layer normalization with learned beta |
| 7 | `GELU` | GELU | GELU activation via LUT |
| 8 | `KV_APPEND` | DMA* | Write K or V vector to KV cache |
| 9 | `KV_READ` | DMA* | Read cached K/V vectors into SRAM |
| 10 | `BARRIER` | All | Wait for all engines to complete |
| 255 | `END` | - | End of program |

*KV instructions share the DMA scoreboard slot with mutual exclusion.

### GEMM Flags

| Bit | Name | Description |
|-----|------|-------------|
| 0 | `TRANSPOSE_B` | Transpose the weight matrix before multiply |
| 1 | `BIAS_EN` | Add bias vector after multiply |
| 2 | `REQUANT` | Apply requantization: `clamp((acc * scale + round) >> shift)` |
| 3 | `RELU` | Apply ReLU after requantization |
| 4 | `CAUSAL_MASK` | Apply causal (lower-triangular) mask |
| 5 | `ACCUMULATE` | Accumulate with existing values in output buffer |

# Execution

### Microcode Controller

The microcode controller follows a simple 3-stage pipeline:

```
  FETCH -----------> DECODE -----------> DISPATCH
  Read 128-bit       Extract opcode,     Check scoreboard,
  instruction from   flags, addresses,   dispatch to target
  SRAM at PC         dimensions          engine if free
       |                                      |
       +------<--- PC++ (or stall) ---<-------+
```

The controller stalls (holds PC) when:
- The target engine's scoreboard slot is busy
- A `BARRIER` instruction is encountered and any engine is still running

When an engine completes its operation, it pulses a `done` signal, clearing its scoreboard slot and allowing the next instruction targeting that engine to dispatch.

### Transformer Block

Here is the complete microcode program that executes one GPT-2 transformer block. This is the actual sequence of instructions that tiny-npu runs:

```asm
; =============================================
; GPT-2 Transformer Block Microcode
; Input:  X[S, 64] in SRAM0 @ 0xC000
; Output: X_OUT[S, 64] in SRAM0 @ 0xFA00
; =============================================

; --- Pre-Attention LayerNorm ---
LAYERNORM  dst=0xC400  src0=0xC000  src1=0x0000   M=S  N=64       ; LN1(X) -> LN1_OUT

BARRIER                                                             ; wait for LN

; --- QKV Projections (3 GEMMs) ---
GEMM       dst=0xC800  src0=0xC400  src1=0x0000   M=S  N=64  K=64  ; LN1_OUT * Wq -> Q
GEMM       dst=0xCC00  src0=0xC400  src1=0x1000   M=S  N=64  K=64  ; LN1_OUT * Wk -> K
GEMM       dst=0xD000  src0=0xC400  src1=0x2000   M=S  N=64  K=64  ; LN1_OUT * Wv -> V

BARRIER                                                             ; wait for QKV

; --- Attention Scores ---
GEMM       dst=0xD400  src0=0xC800  src1=0xCC00   M=S  N=S   K=64  ; Q * K^T -> S
                                                   flags=TRANSPOSE_B

BARRIER

; --- Causal Softmax ---
SOFTMAX    dst=0xD500  src0=0xD400                 M=S  N=S         ; softmax(S) -> P
                                                   flags=CAUSAL_MASK

BARRIER

; --- Attention Context ---
GEMM       dst=0xD600  src0=0xD500  src1=0xD000   M=S  N=64  K=S   ; P * V -> ATTN

BARRIER

; --- Output Projection ---
GEMM       dst=0xDA00  src0=0xD600  src1=0x3000   M=S  N=64  K=64  ; ATTN * Wo -> WO_OUT

BARRIER

; --- First Residual Add ---
VEC        dst=0xDE00  src0=0xDA00  src1=0x0100   M=S  N=64        ; WO_OUT + X -> X2
           flags=VEC_ADD                                             ; (X from SRAM1 residual)

BARRIER

; --- Pre-FFN LayerNorm ---
LAYERNORM  dst=0xE200  src0=0xDE00  src1=0x0040   M=S  N=64       ; LN2(X2) -> LN2_OUT

BARRIER

; --- FFN Up-Project ---
GEMM       dst=0xE600  src0=0xE200  src1=0x4000   M=S  N=256 K=64  ; LN2_OUT * W1 -> FFN1

BARRIER

; --- GELU Activation ---
GELU       dst=0xE600  src0=0xE600                 M=S  N=256       ; gelu(FFN1) -> FFN1
                                                                     ; (in-place)

BARRIER

; --- FFN Down-Project ---
GEMM       dst=0xF600  src0=0xE600  src1=0x8000   M=S  N=64  K=256 ; FFN1 * W2 -> FFN2

BARRIER

; --- Second Residual Add ---
VEC        dst=0xFA00  src0=0xF600  src1=0x0100   M=S  N=64        ; FFN2 + X2 -> X_OUT
           flags=VEC_ADD

END
```

This sequence executes 8 GEMM operations, 2 LayerNorms, 1 Softmax, 1 GELU, and 2 residual adds. The C++ testbench generates this microcode programmatically based on the current sequence length and model dimensions, and feeds it to the RTL.

### KV-Cache Decode

In standard autoregressive generation, each new token requires recomputing attention over the entire sequence. For a sequence of length S, this means O(S^2) compute per step.

KV-caching splits inference into two phases:

**Prefill Phase** (runs once for the initial prompt):
```
For each layer:
  1. Run full transformer block (same as above)
  2. After computing K and V, execute KV_APPEND to store them:
     KV_APPEND  src=K_addr  M=layer  K=0  N=64    ; cache K[0..S-1]
     KV_APPEND  src=V_addr  M=layer  K=0  N=64    ; cache V[0..S-1]
```

**Decode Phase** (runs for each new token):
```
For each layer:
  1. Compute Q, K, V for ONLY the new token (M=1 instead of M=S)
  2. Append new K, V to cache:
     KV_APPEND  src=K_addr  M=layer  K=T  N=64    ; cache K[T]
     KV_APPEND  src=V_addr  M=layer  K=T  N=64    ; cache V[T]
  3. Read ALL cached K, V vectors back:
     KV_READ    dst=K_addr  M=layer  K=T+1  N=64  ; read K[0..T]
     KV_READ    dst=V_addr  M=layer  K=T+1  N=64  ; read V[0..T]
  4. Compute attention: Q[1,64] * K[T+1,64]^T -> S[1,T+1]
  5. Softmax, context multiply, output projection, FFN (all with M=1)
```

The decode phase computes O(S) per step instead of O(S^2), yielding significant speedup:

| Mode | NPU Cycles | Speedup |
|------|-----------|---------|
| Full-recompute | 9,811,450 | 1.0x |
| KV-cache | 5,453,250 | **1.8x** |

*(Measured with 5 prompt tokens + 10 generated tokens)*

Both modes produce **bit-exact identical logits** at every step - the KV cache is a pure performance optimization with no accuracy impact.

In tiny-npu, the KV cache is implemented as a C++ software shim that intercepts `KV_APPEND` and `KV_READ` instructions from the RTL. The hardware issues these instructions through the same scoreboard slot as DMA (Engine 5), with mutual exclusion guards. This approach lets us verify the full KV-cache dataflow without requiring large on-chip SRAM for the cache itself.

# Inference Demo

tiny-npu runs real GPT-2 inference, not a toy example. Here's what happens end-to-end:

### Model Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| Hidden dimension | 64 | First 64 of GPT-2's 768 |
| Layers | 4 | First 4 of 12 |
| Attention heads | 1 | Single-head (head_dim = hidden) |
| FFN dimension | 256 | 4x hidden |
| Vocabulary | 256 | Byte-level (first 256 GPT-2 tokens) |
| Max sequence | 16 | Fits in 64KB SRAM with all weights |
| Quantization | INT8 | Per-tensor symmetric, shift=9 |

### Pipeline

```
  HuggingFace GPT-2         Python Quantizer         NPU Hardware
  ================         ================         =============
  FP32 weights     ---->   INT8 weights.bin  ---->   Load via DMA
  (124M params)            (226KB subset)            into SRAM0
                                |
                           Golden inference   ---->   Compare logits
                           (Python reference)         (must be bit-exact)
```

1. **Export** - Download real GPT-2 weights from HuggingFace, slice to our model dimensions
2. **Quantize** - Per-tensor symmetric INT8 quantization, pack into `weights.bin`
3. **Golden** - Run Python reference inference to produce expected token sequence
4. **NPU Sim** - Run Verilator RTL simulation with the same weights and prompt
5. **Verify** - Compare NPU logits against golden (bit-exact, max_err=0)
6. **Decode** - Convert generated tokens back to text

### Verification Strategy

All verification uses **follow-actual** golden comparison. Rather than comparing NPU outputs against a pre-computed golden at every layer (which would accumulate rounding differences), we:

1. Let the NPU compute all transformer blocks autonomously
2. Read the actual NPU output from SRAM
3. Feed that actual output into a C++ golden model for the final LN + lm_head GEMM
4. Compare the NPU's logits against this golden - must be **bit-exact** (max_err=0)

This approach is tolerant of small LayerNorm rounding differences (~1-2 values between C++ and RTL) while still proving that every GEMM, Softmax, GELU, and Vec operation is cycle-accurate.

# Simulation

### Prerequisites

```bash
# WSL2 Ubuntu 22.04 (or any Linux)
sudo apt update
sudo apt install -y build-essential cmake python3 python3-pip verilator

# Python dependencies (only needed for inference demo)
pip3 install numpy transformers
```

### Build

```bash
cd npu/sim/verilator
mkdir -p build && cd build
cmake ..
cmake --build . -j$(nproc)
```

This builds 6 simulation targets:

| # | Target | Description |
|---|--------|-------------|
| 1 | `npu_sim` | Control plane smoke test (AXI registers, reset) |
| 2 | `engine_sim` | Individual engine compute tests (6 tests) |
| 3 | `integration_sim` | Single attention head integration |
| 4 | `gpt2_block_sim` | Full transformer block (8 HW GEMMs verified) |
| 5 | `demo_infer` | End-to-end GPT-2 inference demo |
| 6 | `kv_cache_sim` | KV cache correctness (bit-exact vs full-recompute) |

### Run Unit Tests

Targets 1-4 use built-in test vectors and require no external data:

```bash
cd npu/sim/verilator/build

./npu_sim             # Control plane smoke test
./engine_sim          # Engine compute tests (GEMM, Softmax, LN, GELU, Vec, DMA)
./integration_sim     # Attention head integration
./gpt2_block_sim      # Full transformer block (8 HW GEMMs)
```

### Run GPT-2 Inference Demo

The full demo pipeline exports real GPT-2 weights, quantizes them, and runs NPU inference.

**Option A: Automated** (recommended)

```bash
cd npu/sim/verilator

# Greedy decoding
./run_demo.sh --prompt "Hello" --max-tokens 10

# Temperature sampling
./run_demo.sh --prompt "Hello" --max-tokens 10 --temperature 0.8 --seed 42

# KV-cache mode (1.8x faster)
./run_demo.sh --prompt "Hello" --max-tokens 10 --temperature 0.8 --kv-cache

# Skip Python steps (reuse existing weights.bin)
./run_demo.sh --prompt "Hello" --max-tokens 10 --skip-python
```

**Option B: Step-by-step**

```bash
cd npu

# 1. Export FP32 weights from HuggingFace
DEMO_OUTDIR=sim/verilator/build/demo_data python3 python/tools/export_gpt2_weights.py

# 2. Quantize to INT8 and pack weights.bin
DEMO_OUTDIR=sim/verilator/build/demo_data python3 python/tools/quantize_pack.py

# 3. Run Python golden inference
DEMO_OUTDIR=sim/verilator/build/demo_data python3 python/golden/gpt2_infer_golden.py \
    --prompt "Hello" --max-tokens 10 --temperature 0.8 --seed 42 \
    --outdir sim/verilator/build/demo_data

# 4. Build and run NPU simulation
cd sim/verilator/build
cmake --build . --target demo_infer -j$(nproc)

# Full-recompute mode
./demo_infer --datadir demo_data --max-tokens 10 --temperature 0.8 --seed 42

# KV-cache mode
./demo_infer --datadir demo_data --max-tokens 10 --temperature 0.8 --seed 42 --kv-cache
```

### Run KV Cache Correctness Test

Verifies that KV-cached decode produces bit-exact results vs full-recompute:

```bash
cd npu/sim/verilator/build
./kv_cache_sim --datadir demo_data
```

Expected output:
```
--- TEST 1: Prefill vs Full-recompute (S=5) ---
  Block output comparison: max_err=0 mismatches=0/320 PASS
  Logit comparison: max_err=0 tok_full=0 tok_kv=0 PASS

--- TEST 2: Decode vs Full-recompute (S=6) ---
  Logit comparison: max_err=0 tok_full=0 tok_kv=0 PASS

  KV CACHE TEST: PASS
```

### Run Full Regression

```bash
cd npu/sim/verilator/build

# All unit tests
./npu_sim && ./engine_sim && ./integration_sim && ./gpt2_block_sim

# KV cache test + inference demo (needs weights.bin from Python pipeline)
./kv_cache_sim --datadir demo_data
./demo_infer --datadir demo_data --max-tokens 10
./demo_infer --datadir demo_data --max-tokens 10 --kv-cache
```

### Debug with Waveforms

```bash
# Enable VCD tracing
NPU_DUMP=1 ./demo_infer --datadir demo_data --max-tokens 1

# View waveform
gtkwave demo_infer.vcd &
```

# Advanced Functionality

For the sake of clarity and learnability, tiny-npu omits many optimizations found in production NPUs. Here are some of the most important ones:

### Multi-Head Attention

tiny-npu uses single-head attention (head_dim = hidden_dim = 64). Production transformers split the hidden dimension across multiple attention heads (e.g., GPT-2 uses 12 heads of 64 dims each). This requires either:
- Sequential per-head computation with SRAM swapping
- Parallel head computation with wider datapaths
- Fused multi-head attention hardware

### Floating Point / Mixed Precision

tiny-npu uses pure INT8 computation with INT32 accumulators. Production NPUs typically use:
- **FP16/BF16** for activations and weights (better dynamic range)
- **FP32** accumulators (prevents overflow in large GEMMs)
- **INT8/INT4** with dynamic quantization (for inference optimization)

The fixed-point approach in tiny-npu works well for small models but would need wider data types for production-scale models.

### Hardware KV Cache

tiny-npu's KV cache is implemented in C++ software. A production design would store KV vectors in dedicated on-chip SRAM or HBM, with hardware address generation and prefetching. The `kv_cache_bank.sv` module in the RTL is a start toward this but is not yet integrated into the datapath.

### Pipelining and Double Buffering

tiny-npu executes operations sequentially with explicit barriers between dependent operations. Production NPUs use:
- **Double buffering** - Load next layer's weights while current layer computes
- **Instruction pipelining** - Overlap fetch/decode/execute stages
- **Engine pipelining** - Start LayerNorm while GEMM is still finishing its last tile

### Memory Coalescing and Tiling

tiny-npu loads entire weight matrices into SRAM before computation. More sophisticated approaches:
- **Weight streaming** - Tile weights and stream them through the systolic array
- **Activation reuse** - Reorder computation to maximize data locality
- **Fusion** - Combine GEMM + bias + activation into a single tiled pass

### Sparsity

Production NPUs increasingly exploit sparsity in weights and activations:
- **Structured pruning** (2:4 sparsity) - Skip zero multiplications in the systolic array
- **Dynamic sparsity** - Skip attention heads or FFN neurons with zero activations

# Next Steps

Improvements I want to make to the design:

- [ ] Integrate hardware KV cache (`kv_cache_bank.sv`) into the datapath
- [ ] Add double buffering for weight loading (DMA + compute overlap)
- [ ] Multi-head attention support (4 heads x 16 dims)
- [ ] Scale to full GPT-2 dimensions (hidden=768) with weight streaming
- [ ] Add FP16 accumulation mode for better dynamic range
- [ ] FPGA synthesis and on-board demo (Xilinx Zynq / Artix-7)
- [ ] Support for more model architectures (LLaMA-style RoPE, GQA)

# Repository Structure

```
npu/
  rtl/
    pkg/                     SystemVerilog packages
      npu_pkg.sv               Engine IDs, data widths, utility functions
      isa_pkg.sv               128-bit instruction format, opcodes
      fixed_pkg.sv             Fixed-point arithmetic helpers
      kv_pkg.sv                KV cache instruction field documentation
    top.sv                   Full top-level (AXI bus + DMA + control)
    bus/                     AXI4 / AXI4-Lite interfaces
      axi_types.sv             Type definitions
      axi_lite_regs.sv         Host control register file
      axi_dma_rd.sv            AXI DMA read engine
      axi_dma_wr.sv            AXI DMA write engine
    ctrl/                    Microcode controller
      ucode_fetch.sv           Instruction fetch unit
      ucode_decode.sv          Decode + dispatch to engines
      scoreboard.sv            6-engine busy tracking
      barrier.sv               Barrier synchronization
      addr_gen.sv              Address generation
    mem/                     Memory subsystem
      sram_dp.sv               Dual-port SRAM primitive
      banked_sram.sv           Multi-bank SRAM wrapper
      kv_cache_bank.sv         Hardware KV cache (SRAM-backed)
    gemm/                    GEMM engine
      mac_int8.sv              INT8 multiply-accumulate unit
      pe.sv                    Processing element (MAC + register)
      systolic_array.sv        16x16 weight-stationary systolic array
      gemm_ctrl.sv             GEMM tiling controller
      gemm_post.sv             Post-processing (requantize + clamp)
    ops/                     Compute engines
      softmax_engine.sv        3-pass softmax (max, exp+sum, normalize)
      layernorm_engine.sv      LayerNorm (mean, variance, normalize)
      gelu_engine.sv           GELU activation via LUT
      vec_engine.sv            Vector elementwise operations
      exp_lut.sv               Exponential lookup table
      recip_lut.sv             Reciprocal lookup table
      rsqrt_lut.sv             Inverse square root lookup table
      gelu_lut.sv              GELU lookup table
      reduce_max.sv            Tree-based max reduction
      reduce_sum.sv            Tree-based sum reduction
      mean_var_engine.sv       Mean and variance computation
  sim/
    verilator/               Verilator simulation environment
      CMakeLists.txt           Build system (6 targets)
      gpt2_block_top.sv        Testbench top (SRAMs + ucode + engines)
      engine_tb_top.sv         Engine-level testbench wrapper
      integration_top.sv       Integration testbench wrapper
      tb_top.cpp               Target 1: Control plane smoke test
      tb_engines.cpp           Target 2: Engine compute tests
      tb_integration.cpp       Target 3: Attention head integration
      tb_gpt2_block.cpp        Target 4: Full transformer block test
      tb_demo_infer.cpp        Target 5: GPT-2 inference demo
      tb_kv_cache_sim.cpp      Target 6: KV cache correctness test
      run_demo.sh              Automated demo runner script
  python/
    golden/                  Python golden reference models
      gemm_ref.py              INT8 GEMM (bit-exact match to RTL)
      softmax_ref.py           Fixed-point softmax with LUT
      layernorm_ref.py         Fixed-point LayerNorm
      gelu_ref.py              GELU via LUT
      quant.py                 Quantization utilities
      attention_head_ref.py    Full attention head golden
      gpt2_block_ref.py        Full transformer block golden
      gpt2_infer_golden.py     Multi-step inference golden
    tools/                   Build and debug tools
      ddr_map.py               Model config, SRAM layout, weight offsets
      kv_map.py                KV cache constants, decode-mode addresses
      export_gpt2_weights.py   Export weights from HuggingFace GPT-2
      quantize_pack.py         INT8 quantization + weights.bin packing
      make_lut.py              LUT initialization file generator
      ucode_asm.py             Microcode assembler
    tests/
      test_end2end.py          Python-level end-to-end tests
    requirements.txt           Python dependencies
  README.md                  This file
```

## Synthesize for FPGA (Vivado)

1. Create a new Vivado project targeting your FPGA (e.g., Artix-7, Kintex-7, Zynq)

2. Add all RTL sources:
   ```tcl
   add_files -fileset sources_1 [glob rtl/pkg/*.sv]
   add_files -fileset sources_1 [glob rtl/bus/*.sv]
   add_files -fileset sources_1 [glob rtl/mem/*.sv]
   add_files -fileset sources_1 [glob rtl/ctrl/*.sv]
   add_files -fileset sources_1 [glob rtl/gemm/*.sv]
   add_files -fileset sources_1 [glob rtl/ops/*.sv]
   add_files -fileset sources_1 rtl/top.sv
   ```

3. Set `top` as the top module

4. Run synthesis:
   ```tcl
   launch_runs synth_1 -jobs 8
   wait_on_run synth_1
   launch_runs impl_1 -to_step write_bitstream -jobs 8
   wait_on_run impl_1
   ```

### Estimated FPGA Resources (16x16 systolic array)

| Resource | Estimate |
|----------|----------|
| LUTs | ~30K-50K |
| FFs | ~20K-40K |
| BRAMs | ~80-120 (36Kb) |
| DSPs | 256 (one per MAC) |

## License

This project is provided for educational and research purposes.
