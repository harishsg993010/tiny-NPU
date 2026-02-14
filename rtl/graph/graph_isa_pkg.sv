// =============================================================================
// Graph ISA Package - Opcodes, instruction struct, tensor descriptor, helpers
// Used by Graph Mode execution path (ONNX MLP/CNN models)
// =============================================================================
`ifndef GRAPH_ISA_PKG_SV
`define GRAPH_ISA_PKG_SV

package graph_isa_pkg;

    // =========================================================================
    // Graph ISA Opcodes (8-bit)
    // =========================================================================
    parameter logic [7:0] OP_G_END          = 8'h00;
    parameter logic [7:0] OP_G_BARRIER      = 8'h01;
    parameter logic [7:0] OP_G_DMA_LOAD     = 8'h10;
    parameter logic [7:0] OP_G_DMA_STORE    = 8'h11;
    parameter logic [7:0] OP_G_DMA_STRIDED  = 8'h12;
    parameter logic [7:0] OP_G_GEMM         = 8'h20;
    parameter logic [7:0] OP_G_EW_ADD       = 8'h30;
    parameter logic [7:0] OP_G_EW_MUL       = 8'h31;
    parameter logic [7:0] OP_G_EW_SUB       = 8'h32;
    parameter logic [7:0] OP_G_RELU         = 8'h38;
    parameter logic [7:0] OP_G_SOFTMAX      = 8'h40;

    // Phase 3 opcodes
    parameter logic [7:0] OP_G_REDUCE_SUM   = 8'h50;
    parameter logic [7:0] OP_G_REDUCE_MAX   = 8'h51;
    parameter logic [7:0] OP_G_REDUCE_MEAN  = 8'h52;
    parameter logic [7:0] OP_G_EXP          = 8'h58;
    parameter logic [7:0] OP_G_LOG          = 8'h59;
    parameter logic [7:0] OP_G_SQRT         = 8'h5A;
    parameter logic [7:0] OP_G_RSQRT        = 8'h5B;
    parameter logic [7:0] OP_G_GATHER       = 8'h60;
    parameter logic [7:0] OP_G_SLICE        = 8'h68;
    parameter logic [7:0] OP_G_CONCAT       = 8'h69;
    parameter logic [7:0] OP_G_PAD          = 8'h6A;
    parameter logic [7:0] OP_G_AVGPOOL2D    = 8'h70;

    // =========================================================================
    // Graph ISA Instruction Format (128-bit)
    //   word0[31:0]:  { src0[7:0], dst[7:0], flags[7:0], opcode[7:0] }
    //   word1[31:0]:  { imm0[15:0], src2[7:0], src1[7:0] }
    //   word2[31:0]:  imm1
    //   word3[31:0]:  imm2
    // =========================================================================
    typedef struct packed {
        logic [31:0] imm2;        // [127:96] word3
        logic [31:0] imm1;        // [95:64]  word2
        logic [15:0] imm0;        // [63:48]  word1[31:16]
        logic [7:0]  src2;        // [47:40]  word1[15:8]
        logic [7:0]  src1;        // [39:32]  word1[7:0]
        logic [7:0]  src0;        // [31:24]  word0[31:24]
        logic [7:0]  dst;         // [23:16]  word0[23:16]
        logic [7:0]  flags;       // [15:8]   word0[15:8]
        logic [7:0]  opcode;      // [7:0]    word0[7:0]
    } graph_instr_t;

    // =========================================================================
    // Tensor Descriptor (256-bit = 32 bytes)
    //   [31:0]    ddr_addr
    //   [47:32]   sram_addr
    //   [63:48]   size_bytes
    //   [79:64]   shape0
    //   [95:80]   shape1
    //   [111:96]  shape2
    //   [127:112] shape3
    //   [135:128] rank
    //   [143:136] dtype
    //   [151:144] flags
    //   [255:152] reserved
    // =========================================================================
    typedef struct packed {
        logic [103:0] reserved;   // [255:152]
        logic [7:0]   flags;      // [151:144]
        logic [7:0]   dtype;      // [143:136]
        logic [7:0]   rank;       // [135:128]
        logic [15:0]  shape3;     // [127:112]
        logic [15:0]  shape2;     // [111:96]
        logic [15:0]  shape1;     // [95:80]
        logic [15:0]  shape0;     // [79:64]
        logic [15:0]  size_bytes; // [63:48]
        logic [15:0]  sram_addr;  // [47:32]
        logic [31:0]  ddr_addr;   // [31:0]
    } tensor_desc_t;

    // =========================================================================
    // Error codes for graph_dispatch
    // =========================================================================
    parameter logic [7:0] GERR_NONE           = 8'h00;
    parameter logic [7:0] GERR_BAD_OPCODE     = 8'h01;
    parameter logic [7:0] GERR_SHAPE_MISMATCH = 8'h02;
    parameter logic [7:0] GERR_TIMEOUT        = 8'h03;

    // =========================================================================
    // Graph dispatch status bits
    // =========================================================================
    parameter int GSTAT_DONE  = 0;
    parameter int GSTAT_BUSY  = 1;
    parameter int GSTAT_ERROR = 2;

    // =========================================================================
    // Decode helper: extract graph_instr_t from raw 128-bit
    // =========================================================================
    function automatic graph_instr_t decode_graph_instr(input logic [127:0] raw);
        return graph_instr_t'(raw);
    endfunction

endpackage

`endif
