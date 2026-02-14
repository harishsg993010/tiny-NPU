// =============================================================================
// Graph ISA - C++ mirror of graph_isa_pkg.sv
// Opcodes, instruction encoding, tensor descriptor struct
// =============================================================================
#ifndef GRAPH_ISA_H
#define GRAPH_ISA_H

#include <cstdint>
#include <cstring>
#include <vector>
#include <array>

// =========================================================================
// Graph ISA Opcodes
// =========================================================================
static constexpr uint8_t OP_G_END          = 0x00;
static constexpr uint8_t OP_G_BARRIER      = 0x01;
static constexpr uint8_t OP_G_DMA_LOAD     = 0x10;
static constexpr uint8_t OP_G_DMA_STORE    = 0x11;
static constexpr uint8_t OP_G_DMA_STRIDED  = 0x12;
static constexpr uint8_t OP_G_GEMM         = 0x20;
static constexpr uint8_t OP_G_EW_ADD       = 0x30;
static constexpr uint8_t OP_G_EW_MUL       = 0x31;
static constexpr uint8_t OP_G_EW_SUB       = 0x32;
static constexpr uint8_t OP_G_RELU         = 0x38;
static constexpr uint8_t OP_G_SOFTMAX      = 0x40;

// Phase 3 opcodes
static constexpr uint8_t OP_G_REDUCE_SUM   = 0x50;
static constexpr uint8_t OP_G_REDUCE_MAX   = 0x51;
static constexpr uint8_t OP_G_REDUCE_MEAN  = 0x52;
static constexpr uint8_t OP_G_EXP          = 0x58;
static constexpr uint8_t OP_G_LOG          = 0x59;
static constexpr uint8_t OP_G_SQRT         = 0x5A;
static constexpr uint8_t OP_G_RSQRT        = 0x5B;
static constexpr uint8_t OP_G_GATHER       = 0x60;
static constexpr uint8_t OP_G_SLICE        = 0x68;
static constexpr uint8_t OP_G_CONCAT       = 0x69;
static constexpr uint8_t OP_G_PAD          = 0x6A;
static constexpr uint8_t OP_G_AVGPOOL2D    = 0x70;

// =========================================================================
// GEMM flags (matches isa_pkg.sv FLAG_* constants)
// =========================================================================
static constexpr uint8_t GFLAG_TRANSPOSE_B = 0x01;
static constexpr uint8_t GFLAG_BIAS_EN     = 0x02;
static constexpr uint8_t GFLAG_REQUANT     = 0x04;
static constexpr uint8_t GFLAG_RELU        = 0x08;

// =========================================================================
// Graph Instruction (128-bit = 16 bytes)
// Layout:
//   word0[31:0]:  { src0[31:24], dst[23:16], flags[15:8], opcode[7:0] }
//   word1[31:0]:  { imm0[31:16], src2[15:8], src1[7:0] }
//   word2[31:0]:  imm1
//   word3[31:0]:  imm2
// =========================================================================
struct GraphInstr {
    uint32_t word0;
    uint32_t word1;
    uint32_t word2;
    uint32_t word3;
};

inline GraphInstr encode_graph_instr(
    uint8_t opcode, uint8_t flags, uint8_t dst, uint8_t src0,
    uint8_t src1, uint8_t src2, uint16_t imm0,
    uint32_t imm1 = 0, uint32_t imm2 = 0)
{
    GraphInstr instr;
    instr.word0 = (uint32_t(src0) << 24) | (uint32_t(dst) << 16) |
                  (uint32_t(flags) << 8)  | uint32_t(opcode);
    instr.word1 = (uint32_t(imm0) << 16) | (uint32_t(src2) << 8) |
                  uint32_t(src1);
    instr.word2 = imm1;
    instr.word3 = imm2;
    return instr;
}

// =========================================================================
// Tensor Descriptor (256-bit = 32 bytes)
// Must match graph_isa_pkg.sv tensor_desc_t layout exactly
// =========================================================================
struct TensorDesc {
    uint32_t ddr_addr;     // [31:0]
    uint16_t sram_addr;    // [47:32]
    uint16_t size_bytes;   // [63:48]
    uint16_t shape0;       // [79:64]
    uint16_t shape1;       // [95:80]
    uint16_t shape2;       // [111:96]
    uint16_t shape3;       // [127:112]
    uint8_t  rank;         // [135:128]
    uint8_t  dtype;        // [143:136]
    uint8_t  flags;        // [151:144]
    uint8_t  reserved[13]; // [255:152] padding to 32 bytes
};
static_assert(sizeof(TensorDesc) == 32, "TensorDesc must be 32 bytes");

// Serialize tensor descriptor to 32 bytes (little-endian)
inline std::vector<uint8_t> serialize_tensor_desc(const TensorDesc& td) {
    std::vector<uint8_t> buf(32, 0);
    std::memcpy(buf.data(), &td, 32);
    return buf;
}

// =========================================================================
// Convenience: encode a full program as bytes
// =========================================================================
inline std::vector<uint8_t> encode_program(const std::vector<GraphInstr>& instrs) {
    std::vector<uint8_t> prog;
    prog.reserve(instrs.size() * 16);
    for (const auto& instr : instrs) {
        // Little-endian word0..word3
        for (int w = 0; w < 4; w++) {
            uint32_t word;
            switch (w) {
                case 0: word = instr.word0; break;
                case 1: word = instr.word1; break;
                case 2: word = instr.word2; break;
                case 3: word = instr.word3; break;
            }
            prog.push_back(word & 0xFF);
            prog.push_back((word >> 8) & 0xFF);
            prog.push_back((word >> 16) & 0xFF);
            prog.push_back((word >> 24) & 0xFF);
        }
    }
    return prog;
}

#endif // GRAPH_ISA_H
