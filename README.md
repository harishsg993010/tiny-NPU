# tiny-npu

A minimal neural processing unit in SystemVerilog, optimized for learning how NPUs work from the ground up.

Built with fully documented SystemVerilog RTL, two execution modes (LLM Mode for transformer inference and Graph Mode for ONNX models), a complete 128-bit microcode ISA, working GPT-2, LLaMA, Mistral and Qwen2 inference running on real HuggingFace weights, KV-cache optimization, an ONNX compiler supporting Gemm, Conv, Reduce, Exp/Log/Sqrt, Gather, Slice, Concat, BatchNorm, AvgPool and more, 50-model fuzz testing, and full Verilator simulation with cycle-accurate verification.

### Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
  - [NPU](#npu)
  - [Memory](#memory)
  - [Engines](#engines)
- [Execution Modes](#execution-modes)
  - [LLM Mode](#llm-mode)
  - [Graph Mode (ONNX)](#graph-mode-onnx)
- [LLM Mode ISA](#llm-mode-isa)
- [LLM Mode Execution](#llm-mode-execution)
  - [Microcode Controller](#microcode-controller)
  - [Transformer Block](#transformer-block)
  - [KV-Cache Decode](#kv-cache-decode)
- [Graph Mode ISA](#graph-mode-isa)
- [Graph Mode Execution](#graph-mode-execution)
  - [Graph Pipeline](#graph-pipeline)
  - [Tensor Descriptor Table](#tensor-descriptor-table)
  - [ONNX Compiler](#onnx-compiler)
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
> **tiny-npu** is a minimal, fully synthesizable neural processing unit in SystemVerilog, optimized for learning about how NPUs work from the ground up.
>
> It supports two execution modes: **LLM Mode** for running real transformer models (GPT-2, LLaMA, Mistral, Qwen2) with a 128-bit microcode ISA, and **Graph Mode** for running ONNX models with a dedicated graph ISA, tensor descriptor table, and 9 hardware engines (GEMM, Softmax, Reduce, Math LUT, Gather, Slice, Concat, AvgPool, plus inline element-wise). Both modes share the same compute core and on-chip SRAM.

With this motivation in mind, we can strip away the complexity of production-grade accelerators (multi-chip interconnects, HBM controllers, sparsity engines) and focus on the core elements that make neural network inference work in hardware:

1. **Systolic Array** - How does a 16x16 grid of multiply-accumulate units perform matrix multiplication?
2. **Fixed-Point Arithmetic** - How do you compute softmax, layer normalization, and GELU without floating point?
3. **Microcode Sequencing** - How does a controller orchestrate dozens of operations to execute a full transformer block?
4. **Memory Management** - How do you fit weights, activations, and intermediate results in limited on-chip SRAM?
5. **KV-Cache Optimization** - How does caching key/value vectors make autoregressive decoding faster?
6. **Graph Execution** - How does a hardware FSM walk an ONNX computation graph, issuing DMA, GEMM, and element-wise ops automatically?
7. **Dedicated Engines** - How do purpose-built hardware engines for reduce, gather, slice, concat, and pooling accelerate graph execution?

The result: a chip that runs real GPT-2, LLaMA, Mistral and Qwen2 weights from HuggingFace through INT8 quantized inference with multi-head attention, and also compiles and executes arbitrary ONNX models end-to-end (verified across 50 fuzz-generated random graphs) - all cycle-accurate against Python and C++ golden models.

# Architecture

```
                           AXI4-Lite Host Interface
                          (Control Registers + Start)
                                     |
                              REG_EXEC_MODE
                             /             \
                    LLM Mode               Graph Mode
                    ========               ==========
    +----------------v-----------------+   +-----------v-----------+
    |       Microcode Controller        |   |     Graph Pipeline     |
    |   Fetch --> Decode --> Dispatch    |   | Fetch->Decode->Dispatch|
    |   + Scoreboard (6 engine slots)   |   | + Tensor Desc Table    |
    |   + Barrier synchronization       |   | + DMA Shim + EW loops  |
    +--+----+----+----+----+----+------+   +---+----+----+---------+
       |    |    |    |    |    |               |    |    |
       +----+----+----+----+----+-------+------+----+----+
            |    |    |    |    |        |
    +-------v------+ +------v--+ | +--v------+  +---------v--------+
    |  DMA Engine  | |  GEMM   | | | Softmax |  |   Vec Engine     |
    |  AXI4 Master | | 16x16   | | | 3-pass  |  |  (Add/Mul/Scale) |
    |  Read + Write| | Systolic| | | LUT-    |  +------------------+
    +--------------+ | Array   | | | based   |
                     | 256 MACs| | +---------+
                     +---------+ |
                         +-------v-------+  +----------+  +----------+
                         |  LayerNorm /  |  |   GELU / |  |   RoPE   |
                         |  RMSNorm     |  |   SiLU   |  |  Engine  |
                         |  Engines      |  |  Engines |  | (rotate) |
                         +---------------+  +----------+  +----------+

                    Graph Mode Phase 3 Engines (dedicated hardware)
                    ================================================
    +-----------+ +-----------+ +-----------+ +-----------+ +-----------+
    |  Reduce   | |   Math    | |  Gather   | |Slice/Concat| | AvgPool2D|
    | Sum/Max/  | | Exp/Log/  | | Axis-0    | | Last-dim  | | Sliding  |
    |  Mean     | | Sqrt/Rsqrt| | row copy  | | operations| | window   |
    | INT32 acc | | 256-entry | |           | |           | | INT32 acc|
    +-----------+ |  LUTs     | +-----------+ +-----------+ +----------+
                  +-----------+
                                  |
                    +-------------v--------------+
                    |     On-Chip SRAM Banks      |
                    | SRAM0 (64KB) | SRAM1 (8KB) |
                    |  Weights +   |  LN Betas + |
                    |  Activations |  Residuals  |
                    +-----------------------------+
```

The NPU supports two execution modes selected via the `REG_EXEC_MODE` register:

- **LLM Mode** (`EXEC_MODE=0`, default) - The microcode controller fetches 128-bit instructions from SRAM, decodes them, and dispatches to 6 independent engines via a scoreboard. Used for transformer inference (GPT-2, LLaMA, Mistral, Qwen2).

- **Graph Mode** (`EXEC_MODE=1`) - The graph pipeline fetches 128-bit graph ISA instructions from a dedicated program SRAM, looks up tensor descriptors from a 256-entry table, and dispatches to 9 hardware engines serially (one at a time). Supports GEMM, Softmax, Reduce (Sum/Max/Mean), Math (Exp/Log/Sqrt/Rsqrt via LUT), Gather, Slice, Concat, AvgPool2D, and inline element-wise ops. Used for ONNX model inference.

Both modes share all compute engines (GEMM, Softmax, etc.) and SRAM. They are fully isolated - switching modes requires no reset, just writing the `REG_EXEC_MODE` register.

## NPU

tiny-npu is built to execute neural network inference layer by layer.

In **LLM Mode**, the host CPU orchestrates inference by:

1. Loading quantized INT8 weights into external DDR memory
2. Writing the microcode program for one block into SRAM
3. Loading per-layer weights from DDR into on-chip SRAM via DMA
4. Starting the microcode controller which runs the block autonomously
5. Reading back the output activations and repeating for the next layer

In **Graph Mode**, the ONNX compiler handles all orchestration:

1. Compile an ONNX model to graph ISA instructions, tensor descriptors, and a DDR image
2. Load the program, descriptors, and DDR image into the NPU
3. Start the graph pipeline which executes the entire model autonomously
4. Read back the output from DDR

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
| `EXEC_MODE` | 0x38 | Execution mode: 0=LLM, 1=Graph |
| `GRAPH_STATUS` | 0x3C | Graph mode status (read-only) |
| `GRAPH_PC` | 0x40 | Graph mode program counter (read-only) |
| `GRAPH_LAST_OP` | 0x44 | Graph mode last opcode (read-only) |

### Microcode Controller (LLM Mode)

The sequencing brain of the NPU in LLM Mode. It fetches 128-bit microcode instructions from SRAM, decodes them into engine-specific commands, and dispatches them to the appropriate hardware engine.

A **scoreboard** tracks which of the 6 engines are currently busy, ensuring instructions only dispatch when their target engine is free. A **barrier** instruction forces the controller to stall until all engines complete, which is needed between dependent operations (e.g., wait for GEMM before starting Softmax).

### Graph Pipeline (Graph Mode)

A 3-stage FSM that executes graph ISA instructions serially:

1. **Fetch** - Read 128-bit instructions sequentially from program SRAM, detect `OP_G_END`
2. **Decode** - Combinational decode into `graph_instr_t` struct (opcode, tensor IDs, flags, immediates)
3. **Dispatch** - Look up tensor descriptors, program engines (GEMM, DMA) or execute internal loops (element-wise add, ReLU)

The dispatch FSM handles each operation type:
- **DMA_LOAD/STORE** - Drive the DMA shim with DDR address, SRAM address, and length from the tensor descriptor
- **GEMM** - Program `gemm_ctrl` with SRAM addresses and shapes from descriptors, support `TRANSPOSE_B`
- **EW_ADD/MUL/SUB** - Internal 3-cycle/element loop: read A from SRAM, read B from SRAM, compute + write
- **RELU** - Internal 2-cycle/element loop: read from SRAM, write `max(0, val)`
- **SOFTMAX** - Program the softmax engine with source/destination from descriptors
- **REDUCE_SUM/MAX/MEAN** - Program the reduce engine with axis dimension and outer count
- **EXP/LOG/SQRT/RSQRT** - Program the math engine (element-wise LUT, 5 cycles/element)
- **GATHER** - Program the gather engine with row size and index tensor
- **SLICE/CONCAT** - Program slice or concat engine with row geometry
- **AVGPOOL2D** - Program the pooling engine with kernel/stride and channel dimensions

### Dispatcher

Unlike a GPU where a dispatcher distributes threads across cores, the NPU dispatcher routes decoded instructions to the correct engine. Each instruction's opcode determines which engine receives the work. In LLM Mode, the scoreboard prevents dispatch conflicts and barrier instructions provide explicit synchronization. In Graph Mode, execution is fully serialized (one operation at a time).

## Memory

### On-Chip SRAM

tiny-npu uses two on-chip SRAM banks for all computation:

**SRAM0 (64KB)** - The main workspace. In LLM Mode, it holds all weights for one transformer block (48KB) plus all intermediate activations (~14KB). In Graph Mode, the ONNX compiler allocates SRAM0 space for all tensors using a bump allocator. The memory map is carefully designed so that weights and activations fit simultaneously.

In LLM Mode, QKV weights use a **head-blocked layout**: each of Wq, Wk, Wv stores 4 contiguous `[64,16]` per-head slices rather than a single `[64,64]` matrix. This lets the microcode address each head's weights directly without reshaping.

Per-head attention buffers (Q_H, K_H, V_H, S, P, ATTN_H) are only 256 bytes each and are **reused across heads**. After each head computes its attention output, a `COPY2D` instruction scatters the result into the correct columns of the full-width ATTN concat buffer.

```
SRAM0 Memory Map (LLM Mode - one transformer block, 4-head attention)
======================================================================
0x0000 +-----------+
       |  Wq       |  4096B  4x[64,16]  Query weights (head-blocked)
0x1000 +-----------+
       |  Wk       |  4096B  4x[64,16]  Key weights (head-blocked)
0x2000 +-----------+
       |  Wv       |  4096B  4x[64,16]  Value weights (head-blocked)
0x3000 +-----------+
       |  Wo       |  4096B  [64][64]   Output projection
0x4000 +-----------+
       |  W1       | 16384B  [64][256]  FFN up-project
0x8000 +-----------+
       |  W2       | 16384B  [256][64]  FFN down-project
0xC000 +-----------+
       |  X        |  1024B  [16][64]   Input activations
0xC400 |  LN1_OUT  |  1024B             LayerNorm output
0xC800 |  Q_H      |   256B  [16][16]   Per-head query (reused)
0xC900 |  K_H      |   256B  [16][16]   Per-head key (reused)
0xCA00 |  V_H      |   256B  [16][16]   Per-head value (reused)
0xCB00 |  S        |   256B  [16][16]   Attention scores (reused)
0xCC00 |  P        |   256B  [16][16]   Softmax output (reused)
0xCD00 |  ATTN_H   |   256B  [16][16]   Per-head context (reused)
0xCE00 |  ATTN     |  1024B  [16][64]   Concat attention (scatter dest)
0xD200 |  WO_OUT   |  1024B             Output projection result
0xD600 |  X2       |  1024B             First residual
0xDA00 |  LN2_OUT  |  1024B             Second LayerNorm
0xDE00 |  FFN1     |  4096B  [16][256]  FFN up-project output
0xEE00 |  FFN2     |  1024B             FFN down-project output
0xF200 |  X_OUT    |  1024B             Block output
0xF600 +-----------+
       Total: ~61.5KB of 64KB used
```

The head-blocked QKV layout keeps total weight size unchanged (3 x 4096B = 12KB for QKV) while enabling direct per-head addressing: head h's query weights are at `Wq + h * 1024`.

In **Graph Mode**, SRAM0 is allocated dynamically by the ONNX compiler with a first-fit memory planner that reuses freed buffers. The SRAM0 mux priority ensures correct arbitration: `ew > gemm > softmax > reduce > math > gather > slice > concat > avgpool > testbench`.

**SRAM1 (8KB)** - Auxiliary storage for LayerNorm beta parameters and residual connections (LLM Mode only). Separated from SRAM0 because the vec engine needs to read from both SRAM0 and SRAM1 simultaneously for residual adds.

### External Memory (DDR)

**LLM Mode**: Weights for all layers are stored in external DDR memory and loaded one block at a time via DMA. The weight file (`weights.bin`) is ~226KB.

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

**Graph Mode**: The ONNX compiler packs all data into a single `ddr_image.bin`:

| Region | Base Address | Max Size | Description |
|--------|-------------|----------|-------------|
| Program | 0x00100000 | 64KB | Graph ISA instructions |
| Descriptors | 0x00200000 | 8KB | Tensor descriptor table (256 x 32B) |
| Data | 0x00300000 | 1MB | Quantized weights, biases, im2col |
| IO | 0x00400000 | 64KB | Model input/output tensors |
| Scratch | 0x00500000 | - | Scratch space |

## Engines

Each engine is a specialized hardware unit optimized for one class of operation. All engines share access to SRAM0/SRAM1 and report completion back to the scoreboard (LLM Mode) or the graph dispatch FSM (Graph Mode).

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

In Graph Mode, the graph dispatch FSM programs `gemm_ctrl` directly with SRAM addresses and shapes from tensor descriptors. The `TRANSPOSE_B` flag is supported for weight matrices stored in `[N, K]` layout.

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

A general-purpose vector unit for elementwise and data movement operations:

- **VEC_ADD** - Saturating INT8 addition (used for residual connections: `x + sublayer_output`)
- **VEC_MUL** - Elementwise multiply
- **VEC_SCALE_SHIFT** - Scale and shift
- **VEC_CLAMP** - Clamp to range
- **VEC_COPY2D** - 2D strided scatter/gather within SRAM0. Used to scatter per-head attention outputs (`[S,16]` with stride 16) into the concatenated attention buffer (`[S,64]` with stride 64). The M field sets the row count, K sets the source stride, and imm sets the destination stride.

### DMA Engine (Engine 5)

Transfers data between external DDR memory and on-chip SRAM via AXI4 burst transactions. The DMA engine handles both read (DDR -> SRAM) and write (SRAM -> DDR) directions with configurable burst lengths.

In simulation, DMA transfers are intercepted by a C++ software shim that models the DDR memory, eliminating the need for an actual AXI interconnect.

# Execution Modes

## LLM Mode

LLM Mode is the original execution path, designed for transformer model inference. The host CPU generates microcode programs that orchestrate multi-head attention, FFN, LayerNorm, and residual connections across 4 transformer layers. Supported models:

| Model | Architecture | Special Features |
|-------|-------------|-----------------|
| GPT-2 | LayerNorm, MHA, GELU FFN | Standard transformer |
| LLaMA (MicroLlama) | RMSNorm, GQA, RoPE, SwiGLU | Grouped-query attention, rotary embeddings |
| Mistral-300M | Same as LLaMA | Identical tensor naming |
| Qwen2-0.5B | Same as LLaMA + QKV bias | Tied embeddings, bias via VEC_ADD |

All models use INT8 quantization and produce **bit-exact** logits compared to the C++ golden model.

## Graph Mode (ONNX)

Graph Mode is a new execution path that runs compiled ONNX models. Instead of hand-crafted microcode, an **ONNX compiler** automatically lowers model graphs into a sequence of graph ISA instructions with tensor descriptors.

The graph pipeline:
1. Fetches 128-bit graph ISA instructions from a dedicated program SRAM
2. Decodes them into opcode + tensor IDs
3. Looks up tensor descriptors (DDR address, SRAM address, shape, size) from a 256-entry table
4. Dispatches to one of 9 hardware engines or executes inline element-wise loops

Supported ONNX operations:

| ONNX Op | Graph ISA Lowering |
|---------|-------------------|
| `Gemm` | DMA_LOAD weights + input, GEMM, EW_ADD bias |
| `Relu` | RELU (in-place 2-cycle/element loop) |
| `Conv` | im2col (pre-materialized in DDR) + GEMM + EW_ADD bias |
| `Reshape` / `Flatten` | SRAM alias (no hardware op) |
| `Softmax` | SOFTMAX engine |
| `ReduceSum` / `ReduceMax` / `ReduceMean` | REDUCE engine (INT32 accumulator) |
| `Exp` / `Log` / `Sqrt` | MATH engine (256-entry LUT, 5 cycles/element) |
| `Gather` | GATHER engine (axis-0 row copy with bounds check) |
| `Slice` | SLICE engine (materialized N-D slice) |
| `Concat` | CONCAT engine (last-dim interleave) |
| `Add` / `Sub` / `Mul` | EW_ADD / EW_SUB / EW_MUL (inline 3-cycle/element loop) |
| `BatchNormalization` | Compiler lowering to EW_MUL + EW_ADD (pre-computed scale/offset) |
| `AveragePool` | AVGPOOL2D engine (sliding window, INT32 accumulator) |

Verified models:
- **MLP**: `[1,32] -> Gemm -> ReLU -> Gemm -> [1,8]` (exact byte match)
- **CNN**: `[1,1,8,8] -> Conv(3x3,4 filters) -> ReLU -> Flatten -> Gemm -> [1,4]` (exact byte match)
- **Reduce**: `[1,8,4] -> ReduceSum(axis=2) -> ReduceMax(axis=1) -> [1]` (exact byte match)
- **Math**: `[1,16] -> Exp -> Log -> Sqrt -> [1,16]` (exact byte match)
- **Gather**: `[4,8] + indices -> Gather(axis=0) -> [2,8]` (exact byte match)
- **Slice+Concat**: `[1,8] -> Slice -> [1,4]`, then Concat -> `[1,12]` (exact byte match)
- **BatchNorm+Pool**: `[1,4,6,6] -> BN -> ReLU -> AvgPool(2x2) -> [1,4,3,3]` (exact byte match)
- **Fuzz**: 50 randomly generated graphs (5-15 nodes, random shapes) - all pass
- **Stress**: 56-node deep graph exercising all engine types (exact byte match)

# LLM Mode ISA

tiny-npu's LLM Mode implements a 128-bit fixed-width microcode ISA. Every instruction is the same size and format, which simplifies the hardware decoder at the cost of some encoding efficiency.

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
| `M` | Rows | Rows | Rows / COPY2D rows | Byte count | Layer ID |
| `N` | Columns | Hidden dim | Length / COPY2D cols | - | Vector length |
| `K` | Inner dim | - | COPY2D src stride | - | Time position |
| `imm` | scale\|shift | - | COPY2D dst stride | - | Head ID |
| `flags` | transpose, requant | causal mask | sub-op (bit2=COPY2D) | direction | is_v (bit0) |

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

# LLM Mode Execution

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

Here is the complete microcode program that executes one GPT-2 transformer block with 4-head attention. This is the actual sequence of instructions that tiny-npu runs:

```asm
; =============================================
; GPT-2 Transformer Block Microcode (4-head)
; Input:  X[S, 64] in SRAM0 @ 0xC000
; Output: X_OUT[S, 64] in SRAM0 @ 0xF200
; =============================================

; --- Pre-Attention LayerNorm ---
LAYERNORM  dst=0xC400  src0=0xC000  src1=0x0000   M=S  N=64       ; LN1(X) -> LN1_OUT

BARRIER

; --- Multi-Head Attention (4 heads, sequential) ---
; Each head: QKV projection -> score -> softmax -> context -> scatter
; Per-head buffers (Q_H, K_H, V_H, S, P, ATTN_H) are reused each iteration

; Head 0: Wq_h0 @ 0x0000, Wk_h0 @ 0x1000, Wv_h0 @ 0x2000
GEMM       dst=0xC800  src0=0xC400  src1=0x0000   M=S  N=16 K=64  ; LN1 * Wq_h0 -> Q_H
BARRIER
GEMM       dst=0xC900  src0=0xC400  src1=0x1000   M=S  N=16 K=64  ; LN1 * Wk_h0 -> K_H
BARRIER
GEMM       dst=0xCA00  src0=0xC400  src1=0x2000   M=S  N=16 K=64  ; LN1 * Wv_h0 -> V_H
BARRIER
GEMM       dst=0xCB00  src0=0xC800  src1=0xC900   M=S  N=S  K=16  ; Q_H * K_H^T -> S
           flags=TRANSPOSE_B  imm=0x0701                            ; shift=7 for K=16
BARRIER
SOFTMAX    dst=0xCC00  src0=0xCB00                 M=S  N=S         ; softmax(S) -> P
           flags=CAUSAL_MASK
BARRIER
GEMM       dst=0xCD00  src0=0xCC00  src1=0xCA00   M=S  N=16 K=S   ; P * V_H -> ATTN_H
           imm=0x0701                                                ; shift=7 for K=S
BARRIER
VEC        dst=0xCE00  src0=0xCD00  M=S  K=16  imm=64              ; COPY2D: scatter ATTN_H
           flags=VEC_COPY2D  N=16                                    ; -> ATTN[:, 0:16]

BARRIER

; Head 1: Wq_h1 @ 0x0400, Wk_h1 @ 0x1400, Wv_h1 @ 0x2400
;   (same pattern, dst_base=ATTN+16 for COPY2D scatter)
; Head 2: Wq_h2 @ 0x0800, Wk_h2 @ 0x1800, Wv_h2 @ 0x2800
; Head 3: Wq_h3 @ 0x0C00, Wk_h3 @ 0x1C00, Wv_h3 @ 0x2C00

; ... (heads 1-3 follow identical pattern with offset weight/scatter addresses) ...

; --- Output Projection ---
GEMM       dst=0xD200  src0=0xCE00  src1=0x3000   M=S  N=64 K=64  ; ATTN * Wo -> WO_OUT

BARRIER

; --- First Residual Add ---
VEC        dst=0xD600  src0=0xD200  src1=0x0100   M=S  N=64        ; WO_OUT + X -> X2
           flags=VEC_ADD                                             ; (X from SRAM1 residual)

BARRIER

; --- Pre-FFN LayerNorm ---
LAYERNORM  dst=0xDA00  src0=0xD600  src1=0x0040   M=S  N=64       ; LN2(X2) -> LN2_OUT

BARRIER

; --- FFN Up-Project ---
GEMM       dst=0xDE00  src0=0xDA00  src1=0x4000   M=S  N=256 K=64  ; LN2_OUT * W1 -> FFN1

BARRIER

; --- GELU Activation ---
GELU       dst=0xDE00  src0=0xDE00                 M=S  N=256       ; gelu(FFN1) -> FFN1
                                                                     ; (in-place)

BARRIER

; --- FFN Down-Project ---
GEMM       dst=0xEE00  src0=0xDE00  src1=0x8000   M=S  N=64  K=256 ; FFN1 * W2 -> FFN2

BARRIER

; --- Second Residual Add ---
VEC        dst=0xF200  src0=0xEE00  src1=0x0100   M=S  N=64        ; FFN2 + X2 -> X_OUT
           flags=VEC_ADD

END
```

This sequence executes 23 GEMM operations (5 per head x 4 + Wo + FFN1 + FFN2), 2 LayerNorms, 4 Softmax, 1 GELU, 4 COPY2D scatters, and 2 residual adds (~113 instructions total). The C++ testbench generates this microcode programmatically based on the current sequence length and model dimensions, and feeds it to the RTL.

### KV-Cache Decode

In standard autoregressive generation, each new token requires recomputing attention over the entire sequence. For a sequence of length S, this means O(S^2) compute per step.

KV-caching splits inference into two phases:

**Prefill Phase** (runs once for the initial prompt):
```
For each layer:
  1. Run full transformer block with 4-head attention (same as above)
  2. After computing K_H and V_H for each head, execute KV_APPEND:
     For h = 0..3:
       KV_APPEND  src=K_H  M=layer  K=0  N=16  imm=h  ; cache K_h[0..S-1]
       KV_APPEND  src=V_H  M=layer  K=0  N=16  imm=h  ; cache V_h[0..S-1]
```

**Decode Phase** (runs for each new token):
```
For each layer:
  For h = 0..3:
    1. Compute Q_H, K_H, V_H for ONLY the new token (M=1, N=16)
    2. Append new K_H, V_H to per-head cache:
       KV_APPEND  src=K_H  M=layer  K=T  N=16  imm=h  ; cache K_h[T]
       KV_APPEND  src=V_H  M=layer  K=T  N=16  imm=h  ; cache V_h[T]
    3. Read ALL cached K_H, V_H vectors for this head:
       KV_READ    dst=K_cache  M=layer  K=T+1  N=16  imm=h  ; read K_h[0..T]
       KV_READ    dst=V_cache  M=layer  K=T+1  N=16  imm=h  ; read V_h[0..T]
    4. Compute attention: Q_H[1,16] * K_cache[T+1,16]^T -> S[1,T+1]
    5. Softmax, context, COPY2D scatter into ATTN concat
  Output projection, residual, LN2, FFN (all with M=1)
```

The decode phase computes O(S) per step instead of O(S^2), yielding significant speedup:

| Mode | NPU Cycles | Speedup |
|------|-----------|---------|
| Full-recompute | 9,811,450 | 1.0x |
| KV-cache | 5,453,250 | **1.8x** |

*(Measured with 5 prompt tokens + 10 generated tokens)*

Both modes produce **bit-exact identical logits** at every step - the KV cache is a pure performance optimization with no accuracy impact.

In tiny-npu, the KV cache is implemented entirely in hardware. A dedicated **KV cache controller** (`kv_ctrl.sv`) bridges the 8-bit SRAM0 interface to the 128-bit `kv_cache_bank.sv` storage module. When the microcode decoder dispatches a `KV_APPEND` or `KV_READ` instruction, the controller FSM autonomously transfers data between SRAM0 and the cache bank — packing bytes into 128-bit vectors for appends, and unpacking vectors back to bytes for reads. The controller shares the DMA scoreboard slot (Engine 5) with mutual exclusion guards, and uses single-vector reads to avoid backpressure on the narrow SRAM0 port.

# Graph Mode ISA

Graph Mode uses a separate 128-bit instruction format optimized for tensor operations with 8-bit tensor IDs (referencing entries in the tensor descriptor table):

```
128-bit Graph ISA Instruction Format
======================================

  127             96 95              64 63     48 47   40 39   32
  +----------------+------------------+---------+------+------+
  |     imm2       |       imm1       |  imm0   | src2 | src1 |
  |   (32 bits)    |    (32 bits)     |(16 bits)|(8bit)|(8bit)|
  +----------------+------------------+---------+------+------+

  31   24 23   16 15    8 7       0
  +------+------+-------+---------+
  | src0 |  dst | flags | opcode  |
  |(8bit)|(8bit)|(8 bit)| (8 bit) |
  +------+------+-------+---------+
```

### Graph Opcodes

| Code | Mnemonic | Engine | Description |
|------|----------|--------|-------------|
| 0x00 | `OP_G_END` | - | End of program (zeros = END for safety) |
| 0x01 | `OP_G_BARRIER` | - | No-op in serialized mode |
| 0x10 | `OP_G_DMA_LOAD` | DMA | Load tensor from DDR to SRAM (src0 = tensor ID) |
| 0x11 | `OP_G_DMA_STORE` | DMA | Store tensor from SRAM to DDR (src0 = tensor ID) |
| 0x12 | `OP_G_DMA_STRIDED` | DMA | Strided DMA (for im2col patterns) |
| 0x20 | `OP_G_GEMM` | GEMM | Matrix multiply (src0=A, src1=B, dst=C tensor IDs) |
| 0x30 | `OP_G_EW_ADD` | Inline | Element-wise add with saturation |
| 0x31 | `OP_G_EW_MUL` | Inline | Element-wise multiply (Q7 fixed-point) |
| 0x32 | `OP_G_EW_SUB` | Inline | Element-wise subtract with saturation |
| 0x38 | `OP_G_RELU` | Inline | ReLU activation (in-place capable) |
| 0x40 | `OP_G_SOFTMAX` | Softmax | Softmax via the softmax engine |
| 0x50 | `OP_G_REDUCE_SUM` | Reduce | Sum reduction along axis |
| 0x51 | `OP_G_REDUCE_MAX` | Reduce | Max reduction along axis |
| 0x52 | `OP_G_REDUCE_MEAN` | Reduce | Mean reduction along axis |
| 0x58 | `OP_G_EXP` | Math | Element-wise exp via 256-entry LUT |
| 0x59 | `OP_G_LOG` | Math | Element-wise log via 256-entry LUT |
| 0x5A | `OP_G_SQRT` | Math | Element-wise sqrt via 256-entry LUT |
| 0x5B | `OP_G_RSQRT` | Math | Element-wise rsqrt via 256-entry LUT |
| 0x60 | `OP_G_GATHER` | Gather | Axis-0 gather (row copy by index) |
| 0x68 | `OP_G_SLICE` | Slice | Materialized N-D slice |
| 0x69 | `OP_G_CONCAT` | Concat | Last-dimension concatenation |
| 0x70 | `OP_G_AVGPOOL2D` | AvgPool | 2D average pooling with sliding window |

### GEMM Flags (Graph Mode)

| Bit | Name | Description |
|-----|------|-------------|
| 0 | `TRANSPOSE_B` | Weight stored as [N,K], transpose during multiply |
| 2 | `REQUANT` | Apply requantization: imm0 encodes scale (low byte) and shift (high byte) |

# Graph Mode Execution

## Graph Pipeline

The graph pipeline is a 3-module hierarchy:

```
  graph_top
  +-- graph_fetch     Sequential instruction fetch from program SRAM
  +-- graph_decode    Combinational 128-bit -> graph_instr_t decode
  +-- graph_dispatch  Main FSM: descriptor lookup -> engine control -> EW loops
```

The dispatch FSM follows this flow for each instruction:

```
GD_IDLE -> GD_FETCH_WAIT -> GD_DECODE -> GD_TDESC0 -> GD_TDESC1 -> GD_TDESC2
  -> GD_EXEC_xxx -> (GD_WAIT_DONE or EW loop) -> GD_NEXT -> GD_FETCH_WAIT ...
  -> GD_DONE (on OP_G_END)
```

Three cycles are spent looking up tensor descriptors (src0, src1, dst) from the combinational tensor table. Then the FSM branches based on opcode:

- **DMA**: Drives the DMA shim with addresses/length from the descriptor, waits for `dma_done`
- **GEMM**: Programs `gemm_ctrl` with SRAM addresses and M/N/K from descriptors, waits for `gm_done`
- **EW_ADD/MUL/SUB**: Runs an internal 3-state loop (RD_A -> RD_B -> COMPUTE) for each element
- **RELU**: Runs an internal 2-state loop (RD -> WR) for each element
- **SOFTMAX**: Programs the softmax engine, waits for `sm_done`
- **REDUCE**: Programs the reduce engine with `reduce_dim` (from `imm0`) and `outer_count` (from `imm1`), waits for `re_done`
- **MATH (EXP/LOG/SQRT/RSQRT)**: Programs the math engine with opcode and length, waits for `me_done`
- **GATHER**: Programs the gather engine with source data/index/destination addresses and row geometry, waits for `ga_done`
- **SLICE**: Programs the slice engine with source/destination row lengths and start offset, waits for `sl_done`
- **CONCAT**: Programs the concat engine with two source bases and row lengths, waits for `ct_done`
- **AVGPOOL2D**: Programs the pooling engine with C/H/W and kernel/stride parameters, waits for `ap_done`

Hardware performance counters track cycles spent in each engine type. A configurable timeout (default 1M cycles per op) triggers `GERR_TIMEOUT` if any engine hangs.

Error detection halts the FSM on bad opcodes, GEMM shape mismatches, or engine timeouts.

## Tensor Descriptor Table

A 256-entry register array where each entry is 256 bits (32 bytes):

```
Tensor Descriptor (256-bit = 32 bytes)
=======================================
Bits [31:0]     ddr_addr     DDR byte address
Bits [47:32]    sram_addr    SRAM0 byte address
Bits [63:48]    size_bytes   Total tensor size in bytes
Bits [79:64]    shape0       Dimension 0 (e.g., M or batch)
Bits [95:80]    shape1       Dimension 1 (e.g., N or channels)
Bits [111:96]   shape2       Dimension 2
Bits [127:112]  shape3       Dimension 3
Bits [135:128]  rank         Number of dimensions
Bits [143:136]  dtype        Data type (0 = INT8)
Bits [151:144]  flags        Tensor flags
Bits [255:152]  reserved     Padding
```

The descriptor table supports 3 simultaneous combinational reads (for src0, src1, dst) and 1 clocked write (for loading descriptors from the testbench/host).

## ONNX Compiler

The ONNX compiler (`python/onnx_compiler/compile.py`) transforms an ONNX model into NPU-ready artifacts:

```
  ONNX Model (.onnx)
        |
   Shape Inference
        |
   INT8 Quantization (per-tensor symmetric)
        |
   DDR Allocation (64B-aligned, weights + biases + im2col)
        |
   SRAM0 Allocation (memory planner with first-fit reuse)
        |
   Tensor Descriptor Table (256 entries x 32B)
        |
   Op Lowering (Gemm, Conv, Reduce, Math, Gather, Slice, Concat, BN, Pool, ...)
        |
   Golden Computation (INT8 with INT32 acc, matching RTL exactly)
        |
   Output: program.bin, tdesc.bin, ddr_image.bin, golden.bin, manifest.json
```

Key compiler decisions:
- **Requantization**: `shift = ceil(log2(K))`, `scale = 1`. Rounding: `(acc + (1 << (shift-1))) >> shift`
- **Bias handling**: Bias is added AFTER requantization via a separate `EW_ADD` instruction (matches RTL pipeline order)
- **Conv lowering**: im2col is pre-materialized by the compiler in DDR, then loaded to SRAM and multiplied via GEMM
- **BatchNorm lowering**: Pre-computes `scale/sqrt(var+eps)` and `bias - mean*multiplier`, then emits EW_MUL + EW_ADD (no new RTL needed)
- **Memory planner**: First-fit allocator with liveness analysis. Tensors are freed after their last use and memory is reclaimed for subsequent allocations. `--no-reuse` flag disables reuse for debugging.
- **SRAM tracking**: The compiler tracks which tensors are "live" in SRAM to avoid redundant DMA loads

# Inference Demo

tiny-npu runs real GPT-2 inference, not a toy example. Here's what happens end-to-end:

### Model Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| Hidden dimension | 64 | First 64 of GPT-2's 768 |
| Layers | 4 | First 4 of 12 |
| Attention heads | 4 | 4 heads x 16 head_dim = 64 hidden |
| Head dimension | 16 | Heads processed sequentially, buffers reused |
| FFN dimension | 256 | 4x hidden |
| Vocabulary | 256 | Byte-level (first 256 GPT-2 tokens) |
| Max sequence | 16 | Fits in 64KB SRAM with all weights |
| Quantization | INT8 | Per-tensor symmetric, shift=9 (K=64), shift=7 (K=16) |

### LLaMA Model Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| Hidden dimension | 64 | First 64 of MicroLlama's 1024 |
| Layers | 4 | First 4 of MicroLlama's 22 |
| Q attention heads | 4 | 4 heads x 16 head_dim = 64 hidden |
| KV attention heads | 2 | Grouped-Query Attention (GQA ratio = 2) |
| Head dimension | 16 | With RoPE positional encoding |
| FFN dimension | 128 | SwiGLU (gate + up + down projections) |
| Vocabulary | 256 | Byte-level (first 256 LLaMA tokens) |
| Max sequence | 16 | Fits in 64KB SRAM with all weights |
| Quantization | INT8 | Per-tensor symmetric, shift=2 (K=64), shift=7 (K=16), shift=2 (K=128) |

### Qwen2 Model Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| Hidden dimension | 64 | First 64 of Qwen2-0.5B's 896 |
| Layers | 4 | First 4 of Qwen2-0.5B's 24 |
| Q attention heads | 4 | 4 heads x 16 head_dim = 64 hidden |
| KV attention heads | 2 | Grouped-Query Attention (GQA ratio = 2) |
| Head dimension | 16 | With RoPE positional encoding |
| FFN dimension | 128 | SwiGLU (gate + up + down projections) |
| Vocabulary | 256 | Byte-level (first 256 Qwen2 tokens) |
| QKV bias | Yes | Applied via VEC_ADD after each Q/K/V GEMM |
| Tied embeddings | Yes | lm_head reuses embed_tokens weights |
| Quantization | INT8 | Same as LLaMA |

Qwen2 uses the same LLaMA pipeline (RMSNorm, RoPE, GQA, SwiGLU) with two additions: **QKV bias** vectors added after each Q/K/V projection via `VEC_ADD` instructions, and **tied embeddings** where the language model head reuses the token embedding weights. The bias support requires no RTL changes — the existing `VEC_ADD` engine handles it by reading bias vectors replicated in SRAM1.

#### LLaMA-Specific Hardware

The LLaMA pipeline adds three new engines beyond GPT-2:

- **RMSNorm Engine** - Two-pass RMSNorm: Pass 1 accumulates sum(x^2) with signed squaring and computes inv_rms via rsqrt LUT. Pass 2 applies fused `(x * gamma) * inv_rms >> 16` to preserve full dynamic range.
- **RoPE Engine** - Rotary positional encoding applied to Q and K projections. Uses precomputed sin/cos tables in Q1.7 format. Operates on pairs of elements within each head dimension.
- **SiLU Engine** - SiLU (Swish) activation via 256-entry LUT, used in the SwiGLU FFN gate path.

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

# Python dependencies (only needed for inference demos)
pip3 install numpy transformers huggingface_hub safetensors

# Additional dependency for ONNX Graph Mode
pip3 install onnx
```

### Build

```bash
cd npu/sim/verilator
mkdir -p build && cd build
cmake ..
cmake --build . -j$(nproc)
```

This builds 17 simulation targets:

| # | Target | Mode | Description |
|---|--------|------|-------------|
| 1 | `npu_sim` | LLM | Control plane smoke test (AXI registers, reset) |
| 2 | `engine_sim` | LLM | Individual engine compute tests (6 tests) |
| 3 | `integration_sim` | LLM | Attention head integration |
| 4 | `gpt2_block_sim` | LLM | Full transformer block (23 HW GEMMs, 4-head attention) |
| 5 | `demo_infer` | LLM | End-to-end GPT-2 inference demo |
| 6 | `kv_cache_sim` | LLM | KV cache correctness (bit-exact vs full-recompute) |
| 7 | `llama_block_sim` | LLM | Full LLaMA transformer block (GQA, RoPE, SwiGLU) |
| 8 | `llama_demo_infer` | LLM | End-to-end LLaMA inference demo |
| 9 | `onnx_smoke_sim` | Graph | MLP smoke test (Gemm -> ReLU -> Gemm) |
| 10 | `onnx_cnn_smoke_sim` | Graph | CNN smoke test (Conv -> ReLU -> Flatten -> Gemm) |
| 11 | `onnx_reduce_sim` | Graph | Reduce engine test (ReduceSum + ReduceMax) |
| 12 | `onnx_math_sim` | Graph | Math engine test (Exp -> Log -> Sqrt) |
| 13 | `onnx_gather_sim` | Graph | Gather engine test (axis-0 row gather) |
| 14 | `onnx_slice_concat_sim` | Graph | Slice + Concat engine test |
| 15 | `onnx_batchnorm_pool_sim` | Graph | BatchNorm lowering + AvgPool2D engine test |
| 16 | `onnx_fuzz_sim` | Graph | 50 random fuzz models (all ops, random shapes) |
| 17 | `onnx_stress_sim` | Graph | 56-node stress test with perf counters |

### Run Unit Tests (LLM Mode)

Targets 1-4 use built-in test vectors and require no external data:

```bash
cd npu/sim/verilator/build

./npu_sim             # Control plane smoke test
./engine_sim          # Engine compute tests (GEMM, Softmax, LN, GELU, Vec, DMA)
./integration_sim     # Attention head integration
./gpt2_block_sim      # Full transformer block (23 HW GEMMs, 4-head)
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

### Run LLaMA Inference Demo

The LLaMA demo supports both random weights and real MicroLlama weights from HuggingFace.

**Random weights** (no dependencies beyond numpy):

```bash
cd npu/python/tools
python3 llama_gen_weights.py --outdir ../../sim/verilator/build/llama_demo_data

cd ../../sim/verilator/build
./llama_demo_infer --datadir llama_demo_data
```

**HuggingFace MicroLlama weights** (requires `huggingface_hub`, `safetensors`):

```bash
cd npu/python/tools
python3 llama_gen_weights_hf.py --outdir ../../sim/verilator/build/llama_demo_data_hf

cd ../../sim/verilator/build
./llama_demo_infer --datadir llama_demo_data_hf
```

**HuggingFace Mistral-300M weights** (requires `huggingface_hub`, `safetensors`):

Mistral uses an identical architecture to LLaMA (RMSNorm, RoPE, GQA, SwiGLU) with identical tensor naming, so the same `llama_demo_infer` testbench is reused with no changes.

```bash
cd npu/python/tools
python3 mistral_gen_weights_hf.py --outdir ../../sim/verilator/build/mistral_demo_data

cd ../../sim/verilator/build
./llama_demo_infer --datadir mistral_demo_data
```

**HuggingFace Qwen2-0.5B weights** (requires `huggingface_hub`, `safetensors`):

Qwen2 uses the same RMSNorm + RoPE + GQA + SwiGLU architecture as LLaMA, but adds **QKV bias** (`q_proj.bias`, `k_proj.bias`, `v_proj.bias`) and uses **tied embeddings** (no separate `lm_head.weight`). The bias is applied via `VEC_ADD` instructions after each Q/K/V GEMM, requiring no RTL changes.

```bash
cd npu/python/tools
python3 qwen_gen_weights_hf.py --outdir ../../sim/verilator/build/qwen_demo_data

cd ../../sim/verilator/build
./llama_demo_infer --datadir qwen_demo_data
```

Expected output:
```
  Step  0: npu_tok=160 gold_tok=160 logit_max_err=0 EXACT
  Step  1: npu_tok=172 gold_tok=172 logit_max_err=0 EXACT
  Step  2: npu_tok=  2 gold_tok=  2 logit_max_err=0 EXACT
  Step  3: npu_tok=141 gold_tok=141 logit_max_err=0 EXACT

  LLAMA DEMO: PASS
```

The NPU hardware produces **bit-exact** logits compared to the C++ golden model (`logit_max_err=0`).

### Run ONNX Graph Mode Tests

Graph Mode tests require generating an ONNX model and compiling it first:

**MLP Smoke Test** (Gemm -> ReLU -> Gemm):

```bash
cd npu

# Generate the MLP ONNX model
python3 python/onnx_compiler/gen_mlp_onnx.py

# Compile to graph artifacts
python3 python/onnx_compiler/compile.py \
    --model models/mlp_32_16_8.onnx \
    --outdir sim/verilator/build/graph

# Run the smoke test
cd sim/verilator/build
./onnx_smoke_sim --datadir graph
```

**CNN Smoke Test** (Conv -> ReLU -> Flatten -> Gemm):

```bash
cd npu
python3 python/onnx_compiler/gen_cnn_onnx.py
python3 python/onnx_compiler/compile.py \
    --model models/conv1x_mini.onnx \
    --outdir sim/verilator/build/graph_cnn
cd sim/verilator/build
./onnx_cnn_smoke_sim --datadir graph_cnn
```

**Phase 3 Engine Tests** (Reduce, Math, Gather, Slice/Concat, BatchNorm+Pool):

```bash
cd npu

# Generate and compile all Phase 3 test models
python3 python/onnx_compiler/gen_reduce_onnx.py
python3 python/onnx_compiler/compile.py --model models/reduce_test.onnx --outdir sim/verilator/build/graph_reduce

python3 python/onnx_compiler/gen_math_onnx.py
python3 python/onnx_compiler/compile.py --model models/math_test.onnx --outdir sim/verilator/build/graph_math

python3 python/onnx_compiler/gen_gather_onnx.py
python3 python/onnx_compiler/compile.py --model models/gather_test.onnx --outdir sim/verilator/build/graph_gather

python3 python/onnx_compiler/gen_slice_concat_onnx.py
python3 python/onnx_compiler/compile.py --model models/slice_concat_test.onnx --outdir sim/verilator/build/graph_slice_concat

python3 python/onnx_compiler/gen_batchnorm_pool_onnx.py
python3 python/onnx_compiler/compile.py --model models/batchnorm_pool_test.onnx --outdir sim/verilator/build/graph_bn_pool

# Run all Phase 3 tests
cd sim/verilator/build
./onnx_reduce_sim --datadir graph_reduce
./onnx_math_sim --datadir graph_math
./onnx_gather_sim --datadir graph_gather
./onnx_slice_concat_sim --datadir graph_slice_concat
./onnx_batchnorm_pool_sim --datadir graph_bn_pool
```

**Fuzz Test** (50 random graphs):

```bash
cd npu

# Generate 50 random ONNX models
python3 python/onnx_compiler/gen_fuzz_onnx.py

# Compile all 50 cases
for i in $(seq 0 49); do
    python3 python/onnx_compiler/compile.py \
        --model models/fuzz/case_${i}.onnx \
        --outdir sim/verilator/build/graph_fuzz/case_${i}
done

# Run fuzz test (loops over all 50 cases)
cd sim/verilator/build
./onnx_fuzz_sim --datadir graph_fuzz
```

**Stress Test** (56-node deep graph with perf counters):

```bash
cd npu
python3 python/onnx_compiler/gen_stress_onnx.py
python3 python/onnx_compiler/compile.py --model models/stress_test.onnx --outdir sim/verilator/build/graph_stress
cd sim/verilator/build
./onnx_stress_sim --datadir graph_stress
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

# LLM Mode unit tests
./npu_sim && ./engine_sim && ./integration_sim && ./gpt2_block_sim

# KV cache + inference demos (needs weights.bin from Python pipeline)
./kv_cache_sim --datadir demo_data
./demo_infer --datadir demo_data --max-tokens 10
./demo_infer --datadir demo_data --max-tokens 10 --kv-cache

# Graph Mode Phase 2 (needs ONNX model compilation)
./onnx_smoke_sim --datadir graph
./onnx_cnn_smoke_sim --datadir graph_cnn

# Graph Mode Phase 3 (needs Phase 3 model compilation)
./onnx_reduce_sim --datadir graph_reduce
./onnx_math_sim --datadir graph_math
./onnx_gather_sim --datadir graph_gather
./onnx_slice_concat_sim --datadir graph_slice_concat
./onnx_batchnorm_pool_sim --datadir graph_bn_pool
./onnx_fuzz_sim --datadir graph_fuzz
./onnx_stress_sim --datadir graph_stress
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

### Floating Point / Mixed Precision

tiny-npu uses pure INT8 computation with INT32 accumulators. Production NPUs typically use:
- **FP16/BF16** for activations and weights (better dynamic range)
- **FP32** accumulators (prevents overflow in large GEMMs)
- **INT8/INT4** with dynamic quantization (for inference optimization)

The fixed-point approach in tiny-npu works well for small models but would need wider data types for production-scale models.

### Hardware KV Cache

tiny-npu includes a fully integrated hardware KV cache. The `kv_cache_bank.sv` module provides dedicated SRAM-backed storage for key/value vectors (4 layers x 4 heads x 512 sequence positions x 16-element vectors), and the `kv_ctrl.sv` FSM controller handles all SRAM0-to-cache data transfers autonomously. A production design would additionally include HBM-backed storage for longer sequences and hardware prefetching for streaming reads.

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

- [x] Integrate hardware KV cache (`kv_cache_bank.sv` + `kv_ctrl.sv`) into the datapath
- [ ] Add double buffering for weight loading (DMA + compute overlap)
- [ ] Scale to full GPT-2 dimensions (hidden=768) with weight streaming
- [ ] Add FP16 accumulation mode for better dynamic range
- [ ] FPGA synthesis and on-board demo (Xilinx Zynq / Artix-7)
- [ ] Graph Mode: parallel engine execution (pipeline DMA with compute)
- [ ] Graph Mode: automatic SRAM tiling for models exceeding 64KB
- [x] Support for more model architectures (LLaMA, Mistral, Qwen2 with GQA, RoPE, QKV bias)
- [x] Graph Mode v1: ONNX compilation and execution (MLP, CNN with im2col)
- [x] Graph Mode v2: 9 dedicated engines (Reduce, Math LUT, Gather, Slice, Concat, AvgPool2D), BatchNorm lowering, memory planner with reuse, perf counters, 50-model fuzz testing

# Repository Structure

```
npu/
  rtl/
    pkg/                     SystemVerilog packages
      npu_pkg.sv               Engine IDs, data widths, exec mode constants
      isa_pkg.sv               128-bit instruction format, opcodes (LLM Mode)
      fixed_pkg.sv             Fixed-point arithmetic helpers
      kv_pkg.sv                KV cache instruction field documentation
    top.sv                   Full top-level (AXI bus + DMA + control + mode mux)
    bus/                     AXI4 / AXI4-Lite interfaces
      axi_types.sv             Type definitions
      axi_lite_regs.sv         Host control register file (incl. EXEC_MODE, graph status)
      axi_dma_rd.sv            AXI DMA read engine
      axi_dma_wr.sv            AXI DMA write engine
    ctrl/                    Microcode controller (LLM Mode)
      ucode_fetch.sv           Instruction fetch unit
      ucode_decode.sv          Decode + dispatch to engines
      scoreboard.sv            6-engine busy tracking
      barrier.sv               Barrier synchronization
      addr_gen.sv              Address generation
      kv_ctrl.sv               KV cache controller FSM (SRAM0 <-> kv_cache_bank bridge)
    graph/                   Graph pipeline (Graph Mode)
      graph_isa_pkg.sv         Graph ISA opcodes (23 opcodes), instruction/descriptor structs
      graph_fetch.sv           Sequential instruction fetch from program SRAM
      graph_decode.sv          Combinational 128-bit -> graph_instr_t decode
      graph_dispatch.sv        Main FSM: descriptor lookup, engine control, EW loops, perf counters
      graph_top.sv             Wrapper: fetch + decode + dispatch + ew_busy
      tensor_table.sv          256-entry tensor descriptor register array
      reduce_engine.sv         Reduce engine: Sum/Max/Mean with INT32 accumulator
      math_engine.sv           Math engine: Exp/Log/Sqrt/Rsqrt via 256-entry LUTs
      graph_exp_lut.sv         INT8 -> INT8 exponential lookup table
      graph_log_lut.sv         INT8 -> INT8 logarithm lookup table
      graph_sqrt_lut.sv        INT8 -> INT8 square root lookup table
      graph_rsqrt_lut.sv       INT8 -> INT8 inverse square root lookup table
      gather_engine.sv         Gather engine: axis-0 row copy with bounds check
      slice_engine.sv          Slice engine: materialized N-D slice
      concat_engine.sv         Concat engine: last-dimension interleave
      avgpool2d_engine.sv      AvgPool2D engine: sliding window with INT32 accumulator
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
      rmsnorm_engine.sv        RMSNorm (2-pass, fused x*gamma*inv_rms)
      rope_engine.sv           Rotary positional encoding
      silu_lut.sv              SiLU activation via LUT
      vec_engine.sv            Vector elementwise operations
      exp_lut.sv               Exponential lookup table
      recip_lut.sv             Reciprocal lookup table
      rsqrt_lut.sv             Inverse square root lookup table
      gelu_lut.sv              GELU lookup table
      reduce_max.sv            Tree-based max reduction
      reduce_sum.sv            Tree-based sum reduction
      mean_var_engine.sv       Mean and variance computation
  include/                   Shared C++ headers
    graph_isa.h                C++ mirror of graph_isa_pkg (encode, decode, structs)
    ddr_graph.h                DDR memory map constants for Graph Mode
  sim/
    verilator/               Verilator simulation environment
      CMakeLists.txt           Build system (17 targets)
      gpt2_block_top.sv        GPT-2 testbench top (SRAMs + ucode + engines)
      llama_block_top.sv       LLaMA testbench top (SRAMs + ucode + engines)
      onnx_sim_top.sv          Graph Mode testbench top (SRAMs + graph pipeline + 9 engines)
      engine_tb_top.sv         Engine-level testbench wrapper
      integration_top.sv       Integration testbench wrapper
      tb_top.cpp               Target 1: Control plane smoke test
      tb_engines.cpp           Target 2: Engine compute tests
      tb_integration.cpp       Target 3: Attention head integration
      tb_gpt2_block.cpp        Target 4: Full GPT-2 transformer block test
      tb_demo_infer.cpp        Target 5: GPT-2 inference demo
      tb_kv_cache_sim.cpp      Target 6: KV cache correctness test
      tb_llama_block.cpp       Target 7: Full LLaMA transformer block test
      tb_llama_demo_infer.cpp  Target 8: LLaMA inference demo
      tb_onnx_smoke.cpp        Target 9: MLP smoke test (Graph Mode)
      tb_onnx_cnn_smoke.cpp    Target 10: CNN smoke test (Graph Mode)
      tb_onnx_reduce.cpp       Target 11: Reduce engine test
      tb_onnx_math.cpp         Target 12: Math engine test
      tb_onnx_gather.cpp       Target 13: Gather engine test
      tb_onnx_slice_concat.cpp Target 14: Slice + Concat engine test
      tb_onnx_batchnorm_pool.cpp Target 15: BatchNorm + AvgPool2D test
      tb_onnx_fuzz.cpp         Target 16: 50-model fuzz test
      tb_onnx_stress.cpp       Target 17: Stress test with perf counters
      run_demo.sh              Automated demo runner script
  python/
    golden/                  Python golden reference models
      gemm_ref.py              INT8 GEMM (bit-exact match to RTL)
      softmax_ref.py           Fixed-point softmax with LUT
      layernorm_ref.py         Fixed-point LayerNorm
      gelu_ref.py              GELU via LUT
      rmsnorm_ref.py           RMSNorm (fused fixed-point)
      rope_ref.py              Rotary positional encoding
      silu_ref.py              SiLU activation via LUT
      llama_infer_golden.py    LLaMA multi-step inference golden
      quant.py                 Quantization utilities
      attention_head_ref.py    Full attention head golden
      gpt2_block_ref.py        Full transformer block golden
      gpt2_infer_golden.py     Multi-step inference golden
    tools/                   Build and debug tools
      ddr_map.py               GPT-2 model config, SRAM layout, weight offsets
      llama_map.py             LLaMA model config, SRAM layout, quant params
      kv_map.py                KV cache constants, decode-mode addresses
      export_gpt2_weights.py   Export weights from HuggingFace GPT-2
      quantize_pack.py         INT8 quantization + weights.bin packing
      llama_gen_weights.py     Generate random LLaMA weights + golden
      llama_gen_weights_hf.py  Generate LLaMA weights from HuggingFace MicroLlama
      mistral_gen_weights_hf.py Generate LLaMA weights from HuggingFace Mistral-300M
      qwen_gen_weights_hf.py   Generate LLaMA weights from HuggingFace Qwen2-0.5B (QKV bias)
      make_lut.py              LUT initialization file generator
      ucode_asm.py             Microcode assembler
    onnx_compiler/           ONNX -> Graph Mode compiler
      __init__.py              Package init
      compile.py               Main compiler: ONNX -> program.bin + tdesc.bin + ddr_image.bin
      gen_mlp_onnx.py          Generate MLP test model (32 -> 16 -> 8)
      gen_cnn_onnx.py          Generate CNN test model (Conv3x3 -> ReLU -> FC)
      gen_reduce_onnx.py       Generate Reduce test model (ReduceSum + ReduceMax)
      gen_math_onnx.py         Generate Math test model (Exp -> Log -> Sqrt)
      gen_gather_onnx.py       Generate Gather test model (axis-0 row gather)
      gen_slice_concat_onnx.py Generate Slice + Concat test model
      gen_batchnorm_pool_onnx.py Generate BatchNorm + AvgPool test model
      gen_fuzz_onnx.py         Generate 50 random fuzz models
      gen_stress_onnx.py       Generate 56-node stress model
    tests/
      test_end2end.py          Python-level end-to-end tests
    requirements.txt           Python dependencies
  models/                    Generated ONNX models (by gen_*.py scripts)
    mlp_32_16_8.onnx           MLP test model
    conv1x_mini.onnx           CNN test model
    reduce_test.onnx           Reduce test model
    math_test.onnx             Math test model
    gather_test.onnx           Gather test model
    slice_concat_test.onnx     Slice + Concat test model
    batchnorm_pool_test.onnx   BatchNorm + AvgPool test model
    stress_test.onnx           Stress test model
    fuzz/                      50 random fuzz models (case_0.onnx .. case_49.onnx)
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
   add_files -fileset sources_1 [glob rtl/graph/*.sv]
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
