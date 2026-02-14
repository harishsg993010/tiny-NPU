#!/usr/bin/env python3
"""
ONNX → NPU Graph Mode Compiler

Compiles an ONNX model (MLP or CNN) into:
  - program.bin   : Graph ISA instructions
  - tdesc.bin     : Tensor descriptor table
  - ddr_image.bin : DDR memory image (weights, inputs)
  - golden.bin    : Expected output (int8)
  - manifest.json : Metadata

Supports: Gemm, Relu, Conv (via im2col), Reshape/Flatten

Usage:
  python compile.py --model models/mlp_32_16_8.onnx --outdir build/graph
"""

import argparse
import json
import math
import os
import struct
import sys
from collections import OrderedDict

import numpy as np

try:
    import onnx
    from onnx import numpy_helper, shape_inference
except ImportError:
    print("ERROR: onnx package required. Run: pip install onnx")
    sys.exit(1)

# =========================================================================
# Constants matching include/graph_isa.h and include/ddr_graph.h
# =========================================================================
OP_G_END         = 0x00
OP_G_DMA_LOAD    = 0x10
OP_G_DMA_STORE   = 0x11
OP_G_DMA_STRIDED = 0x12
OP_G_GEMM        = 0x20
OP_G_EW_ADD      = 0x30
OP_G_EW_MUL      = 0x31
OP_G_EW_SUB      = 0x32
OP_G_RELU        = 0x38
OP_G_SOFTMAX     = 0x40

# Phase 3 opcodes
OP_G_REDUCE_SUM  = 0x50
OP_G_REDUCE_MAX  = 0x51
OP_G_REDUCE_MEAN = 0x52
OP_G_EXP         = 0x58
OP_G_LOG         = 0x59
OP_G_SQRT        = 0x5A
OP_G_RSQRT       = 0x5B
OP_G_GATHER      = 0x60
OP_G_SLICE       = 0x68
OP_G_CONCAT      = 0x69
OP_G_PAD         = 0x6A
OP_G_AVGPOOL2D   = 0x70

GFLAG_TRANSPOSE_B = 0x01
GFLAG_BIAS_EN     = 0x02
GFLAG_REQUANT     = 0x04
GFLAG_RELU        = 0x08

DDR_GRAPH_DATA_BASE = 0x00300000
DDR_GRAPH_IO_BASE   = 0x00400000

# =========================================================================
# Instruction encoding
# =========================================================================
def encode_instr(opcode, flags=0, dst=0, src0=0, src1=0, src2=0,
                 imm0=0, imm1=0, imm2=0):
    """Encode a 128-bit (16-byte) graph instruction."""
    word0 = (src0 << 24) | (dst << 16) | (flags << 8) | opcode
    word1 = (imm0 << 16) | (src2 << 8) | src1
    word2 = imm1 & 0xFFFFFFFF
    word3 = imm2 & 0xFFFFFFFF
    return struct.pack('<IIII', word0, word1, word2, word3)


# =========================================================================
# Tensor descriptor encoding (32 bytes)
# =========================================================================
def encode_tdesc(ddr_addr, sram_addr, size_bytes, shape, rank=2, dtype=0, flags=0):
    """Encode a 256-bit (32-byte) tensor descriptor."""
    s = [0, 0, 0, 0]
    for i, v in enumerate(shape[:4]):
        s[i] = v
    buf = struct.pack('<I', ddr_addr)            # [31:0]
    buf += struct.pack('<H', sram_addr & 0xFFFF) # [47:32]
    buf += struct.pack('<H', size_bytes & 0xFFFF) # [63:48]
    buf += struct.pack('<HHHH', s[0], s[1], s[2], s[3]) # shapes
    buf += struct.pack('<BBB', rank, dtype, flags) # rank, dtype, flags
    buf += b'\x00' * 13  # reserved, pad to 32 bytes
    assert len(buf) == 32
    return buf


# =========================================================================
# Quantization helpers
# =========================================================================
def quantize_int8(arr):
    """Symmetric int8 quantization: scale = max(|arr|) / 127"""
    amax = np.max(np.abs(arr))
    if amax < 1e-10:
        return np.zeros_like(arr, dtype=np.int8), 1.0
    scale = amax / 127.0
    q = np.clip(np.round(arr / scale), -128, 127).astype(np.int8)
    return q, scale


def int8_gemm_golden(A_q, B_q, bias_q, scale, shift):
    """INT8 GEMM with INT32 accumulate, then requantize to INT8.
    C = requant(A_q @ B_q + bias_q)
    """
    # A_q: [M, K] int8, B_q: [K, N] int8
    M, K = A_q.shape
    _, N = B_q.shape

    # INT32 accumulate
    acc = np.zeros((M, N), dtype=np.int32)
    for m in range(M):
        for n in range(N):
            s = np.int32(0)
            for k in range(K):
                s += np.int32(A_q[m, k]) * np.int32(B_q[k, n])
            acc[m, n] = s

    # Add bias (broadcast)
    if bias_q is not None:
        for m in range(M):
            for n in range(N):
                acc[m, n] += np.int32(bias_q[n])

    # Requantize: (acc * scale_int) >> shift, round-to-nearest
    result = np.zeros((M, N), dtype=np.int8)
    for m in range(M):
        for n in range(N):
            product = int(acc[m, n]) * int(scale)
            if shift > 0:
                product += (1 << (shift - 1))  # rounding
            shifted = product >> shift
            result[m, n] = np.clip(shifted, -128, 127).astype(np.int8)

    return result


# =========================================================================
# im2col for Conv lowering
# =========================================================================
def im2col(input_data, kh, kw, stride=1, pad=0):
    """Convert [C, H, W] input to [out_h*out_w, C*kh*kw] matrix."""
    C, H, W = input_data.shape
    out_h = (H + 2 * pad - kh) // stride + 1
    out_w = (W + 2 * pad - kw) // stride + 1

    if pad > 0:
        input_data = np.pad(input_data,
                           ((0, 0), (pad, pad), (pad, pad)),
                           mode='constant')

    col = np.zeros((out_h * out_w, C * kh * kw), dtype=input_data.dtype)
    idx = 0
    for i in range(out_h):
        for j in range(out_w):
            patch = input_data[:, i*stride:i*stride+kh, j*stride:j*stride+kw]
            col[idx] = patch.flatten()
            idx += 1
    return col


# =========================================================================
# Memory Planner v2 - First-fit with free list and merge
# =========================================================================
class MemoryPlanner:
    """SRAM allocator with tensor reuse via liveness analysis."""
    def __init__(self, capacity=65536):
        self.capacity = capacity
        self.free_list = []  # list of (offset, size) sorted by offset
        self.bump = 0
        self.allocs = {}     # name -> (offset, size)
        self.peak = 0

    def alloc(self, name, size):
        """Allocate size bytes, 16-byte aligned. Returns offset."""
        size = (size + 15) & ~15  # align to 16 bytes
        # First-fit from free list
        for i, (off, sz) in enumerate(self.free_list):
            if sz >= size:
                self.free_list.pop(i)
                if sz > size:
                    # Return remainder to free list
                    self.free_list.append((off + size, sz - size))
                    self.free_list.sort()
                self.allocs[name] = (off, size)
                return off
        # Bump allocate
        if self.bump + size > self.capacity:
            raise RuntimeError(f"SRAM overflow: need {self.bump + size}, have {self.capacity}")
        off = self.bump
        self.bump += size
        self.peak = max(self.peak, self.bump)
        self.allocs[name] = (off, size)
        return off

    def free(self, name):
        """Return allocation to free list, merge adjacent blocks."""
        if name not in self.allocs:
            return
        off, size = self.allocs.pop(name)
        self.free_list.append((off, size))
        self.free_list.sort()
        # Merge adjacent free blocks
        merged = []
        for foff, fsz in self.free_list:
            if merged and merged[-1][0] + merged[-1][1] == foff:
                merged[-1] = (merged[-1][0], merged[-1][1] + fsz)
            else:
                merged.append((foff, fsz))
        self.free_list = merged


# =========================================================================
# Compiler
# =========================================================================
class GraphCompiler:
    def __init__(self, model_path, use_reuse=True):
        self.model = onnx.load(model_path)
        self.model = shape_inference.infer_shapes(self.model)
        self.graph = self.model.graph

        # Initializer lookup
        self.initializers = {}
        for init in self.graph.initializer:
            self.initializers[init.name] = numpy_helper.to_array(init)

        # Shape lookup from value_info + graph inputs/outputs
        self.shapes = {}
        for vi in list(self.graph.value_info) + list(self.graph.input) + list(self.graph.output):
            dims = [d.dim_value for d in vi.type.tensor_type.shape.dim]
            if all(d > 0 for d in dims):
                self.shapes[vi.name] = tuple(dims)

        # DDR allocation
        self.ddr_offset = DDR_GRAPH_DATA_BASE
        self.ddr_allocs = {}  # name -> (offset, size)
        self.ddr_image = bytearray()

        # SRAM allocation
        self.sram_offset = 0
        self.sram_allocs = {}  # name -> (offset, size)

        # Memory planner
        self.use_reuse = use_reuse
        self.mem_planner = MemoryPlanner() if use_reuse else None
        self.sram_no_reuse_offset = 0  # track usage without reuse for stats

        # Tensor descriptors
        self.tdesc_list = []  # list of (name, bytes)
        self.tdesc_id = {}    # name -> index

        # Instructions
        self.instructions = []

        # Track which tensors have valid data in SRAM (skip DMA_LOAD for these)
        self.sram_live = set()

        # Quantization scales
        self.quant_scales = {}

    def _alloc_ddr(self, name, data_bytes):
        """Allocate DDR space, 64-byte aligned."""
        offset = self.ddr_offset
        # Align to 64 bytes
        offset = (offset + 63) & ~63
        size = len(data_bytes)
        self.ddr_allocs[name] = (offset, size)
        # Extend ddr_image
        needed = (offset - DDR_GRAPH_DATA_BASE) + size
        while len(self.ddr_image) < needed:
            self.ddr_image.extend(b'\x00' * 1024)
        self.ddr_image[offset - DDR_GRAPH_DATA_BASE:
                       offset - DDR_GRAPH_DATA_BASE + size] = data_bytes
        self.ddr_offset = offset + size
        return offset

    def _alloc_sram(self, name, size):
        """Allocate SRAM0 space."""
        # Track non-reuse usage for stats
        self.sram_no_reuse_offset += (size + 15) & ~15

        if self.use_reuse and self.mem_planner is not None:
            offset = self.mem_planner.alloc(name, size)
            self.sram_allocs[name] = (offset, size)
            self.sram_offset = max(self.sram_offset, self.mem_planner.peak)
            return offset
        else:
            if self.sram_offset + size > 65536:
                raise RuntimeError(f"SRAM0 overflow: need {self.sram_offset + size}, have 65536")
            offset = self.sram_offset
            self.sram_allocs[name] = (offset, size)
            self.sram_offset = offset + size
            # Align next alloc to 16 bytes
            self.sram_offset = (self.sram_offset + 15) & ~15
            return offset

    def _free_sram(self, name):
        """Free SRAM allocation (only effective with memory planner)."""
        if self.use_reuse and self.mem_planner is not None:
            self.mem_planner.free(name)

    def _add_tdesc(self, name, ddr_addr, sram_addr, size_bytes, shape, rank=2):
        """Register a tensor descriptor."""
        idx = len(self.tdesc_list)
        self.tdesc_id[name] = idx
        self.tdesc_list.append((name, encode_tdesc(
            ddr_addr, sram_addr, size_bytes, shape, rank)))
        return idx

    def _emit(self, instr_bytes):
        """Emit an instruction."""
        self.instructions.append(instr_bytes)

    def compile(self):
        """Main compilation pipeline."""
        # 1. Topological sort (ONNX graph is already sorted)
        ops = list(self.graph.node)

        # 2. Quantize initializers
        quant_data = {}
        for name, arr in self.initializers.items():
            if arr.dtype in (np.float32, np.float64):
                q, scale = quantize_int8(arr.flatten().astype(np.float32))
                quant_data[name] = q.reshape(arr.shape)
                self.quant_scales[name] = scale
            elif arr.dtype == np.int64:
                # Shape tensors (for Reshape) - keep as-is
                quant_data[name] = arr
                self.quant_scales[name] = 1.0
            else:
                quant_data[name] = arr
                self.quant_scales[name] = 1.0

        # 3. Allocate DDR for weights/biases
        for name, arr in quant_data.items():
            if arr.dtype == np.int64:
                continue  # skip shape constants
            data = arr.astype(np.int8).tobytes()
            self._alloc_ddr(name, data)

        # 4. Allocate input in DDR
        input_name = self.graph.input[0].name
        input_shape = self.shapes.get(input_name, (1,))
        input_size = int(np.prod(input_shape))
        # Quantize input (use small random for testing)
        np.random.seed(99)
        input_fp = np.random.randn(*input_shape).astype(np.float32) * 0.5
        input_q, input_scale = quantize_int8(input_fp)
        self.quant_scales[input_name] = input_scale
        quant_data[input_name] = input_q

        input_ddr_offset = self._alloc_ddr(input_name, input_q.tobytes())

        # 5. Allocate SRAM and create tensor descriptors
        # Process ops to determine what needs to be in SRAM
        tensor_shapes = dict(self.shapes)

        # Allocate SRAM for input
        input_sram = self._alloc_sram(input_name, input_size)
        self._add_tdesc(input_name, input_ddr_offset, input_sram,
                       input_size, list(input_shape), len(input_shape))

        # 6. Liveness analysis for memory reuse
        # Compute last_use[tensor_name] = last op index that reads it
        last_use = {}
        output_name_graph = self.graph.output[0].name
        for op_idx, op in enumerate(ops):
            for inp_name in op.input:
                if inp_name not in self.initializers:
                    last_use[inp_name] = op_idx

        # 7. Lower ops
        current_data = {input_name: quant_data[input_name]}

        for op_idx, op in enumerate(ops):
            op_type = op.op_type

            if op_type == 'Gemm':
                self._lower_gemm(op, quant_data, current_data, tensor_shapes)
            elif op_type == 'Relu':
                self._lower_relu(op, quant_data, current_data, tensor_shapes)
            elif op_type == 'Conv':
                self._lower_conv(op, quant_data, current_data, tensor_shapes)
            elif op_type in ('Reshape', 'Flatten'):
                self._lower_reshape(op, quant_data, current_data, tensor_shapes)
            elif op_type in ('ReduceSum', 'ReduceMax', 'ReduceMean'):
                self._lower_reduce(op, quant_data, current_data, tensor_shapes)
            elif op_type == 'Exp':
                self._lower_math(op, quant_data, current_data, tensor_shapes, OP_G_EXP)
            elif op_type == 'Log':
                self._lower_math(op, quant_data, current_data, tensor_shapes, OP_G_LOG)
            elif op_type == 'Sqrt':
                self._lower_math(op, quant_data, current_data, tensor_shapes, OP_G_SQRT)
            elif op_type == 'Reciprocal':
                self._lower_math(op, quant_data, current_data, tensor_shapes, OP_G_RSQRT)
            elif op_type == 'Gather':
                self._lower_gather(op, quant_data, current_data, tensor_shapes)
            elif op_type == 'Slice':
                self._lower_slice(op, quant_data, current_data, tensor_shapes)
            elif op_type == 'Concat':
                self._lower_concat(op, quant_data, current_data, tensor_shapes)
            elif op_type == 'BatchNormalization':
                self._lower_batchnorm(op, quant_data, current_data, tensor_shapes)
            elif op_type == 'AveragePool':
                self._lower_avgpool2d(op, quant_data, current_data, tensor_shapes)
            else:
                print(f"WARNING: Unsupported op '{op_type}', skipping")

            # Free tensors whose last use is this op (skip graph output)
            if self.use_reuse:
                for inp_name in op.input:
                    if (inp_name in last_use and
                        last_use[inp_name] == op_idx and
                        inp_name != output_name_graph and
                        inp_name in self.sram_allocs):
                        self._free_sram(inp_name)

        # 8. DMA_STORE output
        output_name = self.graph.output[0].name
        if output_name in self.tdesc_id:
            out_id = self.tdesc_id[output_name]
            # Update the output tensor descriptor DDR addr to IO base
            out_sram, out_size = self.sram_allocs[output_name]
            out_shape = tensor_shapes.get(output_name, (1,))
            # Re-create descriptor with IO base
            self.tdesc_list[out_id] = (output_name, encode_tdesc(
                DDR_GRAPH_IO_BASE, out_sram, out_size,
                list(out_shape), len(out_shape)))

            self._emit(encode_instr(OP_G_DMA_STORE, src0=out_id))

        # 8. END instruction
        self._emit(encode_instr(OP_G_END))

        # 9. Compute golden output
        golden = self._compute_golden(ops, quant_data, current_data, tensor_shapes)

        return golden

    def _lower_gemm(self, op, quant_data, current_data, tensor_shapes):
        """Lower Gemm op: DMA_LOAD weight, DMA_LOAD bias, GEMM, EW_ADD bias."""
        input_name = op.input[0]
        weight_name = op.input[1]
        bias_name = op.input[2] if len(op.input) > 2 else None
        output_name = op.output[0]

        # Get transB attribute
        transB = 0
        for attr in op.attribute:
            if attr.name == 'transB':
                transB = attr.i

        # Weight shape
        W = quant_data[weight_name]
        if transB:
            # W is [N, K], GEMM computes input @ W^T = [M, K] @ [K, N]
            N, K = W.shape
        else:
            K, N = W.shape

        # Input shape: [M, K]
        in_sram, in_size = self.sram_allocs[input_name]
        in_shape = tensor_shapes.get(input_name, (1, K))
        M = in_shape[0] if len(in_shape) > 1 else 1

        # Allocate SRAM for weight
        w_size = W.size
        w_sram = self._alloc_sram(weight_name, w_size)
        w_ddr = self.ddr_allocs[weight_name][0]

        # Weight needs to be stored as [K, N] for non-transposed GEMM
        # If transB=1, weight is [N, K] and we use TRANSPOSE_B flag
        if transB:
            # Store weight as [N, K] in row-major, use transpose flag
            w_shape = [N, K]
        else:
            w_shape = [K, N]

        w_id = self._add_tdesc(weight_name, w_ddr, w_sram, w_size, w_shape)

        # DMA_LOAD weight
        self._emit(encode_instr(OP_G_DMA_LOAD, src0=w_id))
        self.sram_live.add(weight_name)

        # DMA_LOAD input (only if not already in SRAM from a previous op)
        in_id = self.tdesc_id[input_name]
        if input_name not in self.sram_live:
            self._emit(encode_instr(OP_G_DMA_LOAD, src0=in_id))
            self.sram_live.add(input_name)

        # Allocate output SRAM
        out_size = M * N
        out_sram = self._alloc_sram(output_name, out_size)
        out_shape = [M, N]
        tensor_shapes[output_name] = tuple(out_shape)
        out_ddr = DDR_GRAPH_IO_BASE  # placeholder
        out_id = self._add_tdesc(output_name, out_ddr, out_sram, out_size, out_shape)

        # Compute requant params
        shift = max(0, int(math.ceil(math.log2(max(K, 1)))))
        scale_int = 1
        # Encode: imm0 = scale[7:0] | (shift[7:0] << 8)
        imm0 = (scale_int & 0xFF) | ((shift & 0xFF) << 8)

        # GEMM flags
        flags = GFLAG_REQUANT
        if transB:
            flags |= GFLAG_TRANSPOSE_B

        # GEMM instruction: src0=input, src1=weight, dst=output
        self._emit(encode_instr(OP_G_GEMM, flags=flags,
                                src0=in_id, src1=w_id, dst=out_id,
                                imm0=imm0))

        # Handle bias
        if bias_name and bias_name in quant_data:
            bias_q = quant_data[bias_name]
            b_size = bias_q.size
            b_sram = self._alloc_sram(bias_name, b_size)
            b_ddr = self.ddr_allocs[bias_name][0]
            # Bias descriptor: shape [1, N] for broadcast add
            b_id = self._add_tdesc(bias_name, b_ddr, b_sram, b_size, [1, N])

            # DMA_LOAD bias
            self._emit(encode_instr(OP_G_DMA_LOAD, src0=b_id))

            # We need a "broadcasted bias" tensor in SRAM matching output shape
            # For simplicity with EW_ADD, we'll create a temporary with bias
            # replicated for each row
            bcast_name = bias_name + '_bcast'
            bcast_size = M * N
            bcast_sram = self._alloc_sram(bcast_name, bcast_size)
            bcast_id = self._add_tdesc(bcast_name, 0, bcast_sram, bcast_size, [M, N])

            # The bias needs to be broadcast - we handle this by writing
            # the bias data M times in the compiler (pre-materialized in DDR)
            bias_bcast = np.tile(bias_q.flatten(), M)
            bcast_ddr = self._alloc_ddr(bcast_name, bias_bcast.astype(np.int8).tobytes())
            # Update the descriptor with correct DDR addr
            self.tdesc_list[bcast_id] = (bcast_name, encode_tdesc(
                bcast_ddr, bcast_sram, bcast_size, [M, N]))

            # DMA_LOAD broadcast bias
            self._emit(encode_instr(OP_G_DMA_LOAD, src0=bcast_id))

            # EW_ADD: output = output + bias_broadcast
            # src0=GEMM output, src1=bias_broadcast, dst=output (in-place)
            self._emit(encode_instr(OP_G_EW_ADD, src0=out_id, src1=bcast_id, dst=out_id))

        # Mark output as live in SRAM
        self.sram_live.add(output_name)
        current_data[output_name] = None  # computed at runtime

    def _lower_relu(self, op, quant_data, current_data, tensor_shapes):
        """Lower Relu op."""
        input_name = op.input[0]
        output_name = op.output[0]

        in_id = self.tdesc_id[input_name]
        in_sram, in_size = self.sram_allocs[input_name]
        in_shape = tensor_shapes.get(input_name, (1,))

        # Ensure input is in SRAM
        if input_name not in self.sram_live:
            self._emit(encode_instr(OP_G_DMA_LOAD, src0=in_id))
            self.sram_live.add(input_name)

        # For in-place RELU, output uses same SRAM as input
        out_sram = in_sram
        out_size = in_size
        tensor_shapes[output_name] = in_shape
        self.sram_allocs[output_name] = (out_sram, out_size)

        out_id = self._add_tdesc(output_name, 0, out_sram, out_size,
                                list(in_shape), len(in_shape))

        # RELU: src0=input, dst=output (can be same SRAM location)
        self._emit(encode_instr(OP_G_RELU, src0=in_id, dst=out_id))

        self.sram_live.add(output_name)
        current_data[output_name] = None

    def _lower_conv(self, op, quant_data, current_data, tensor_shapes):
        """Lower Conv via im2col: pre-materialize im2col in DDR, then GEMM."""
        input_name = op.input[0]
        weight_name = op.input[1]
        bias_name = op.input[2] if len(op.input) > 2 else None
        output_name = op.output[0]

        # Get attributes
        kernel_shape = [3, 3]
        pads = [0, 0, 0, 0]
        strides = [1, 1]
        for attr in op.attribute:
            if attr.name == 'kernel_shape':
                kernel_shape = list(attr.ints)
            elif attr.name == 'pads':
                pads = list(attr.ints)
            elif attr.name == 'strides':
                strides = list(attr.ints)

        kh, kw = kernel_shape
        pad = pads[0]  # assume symmetric
        stride = strides[0]

        # Input shape: [N, C, H, W]
        in_shape = tensor_shapes.get(input_name)
        if in_shape is None or len(in_shape) != 4:
            raise RuntimeError(f"Conv input {input_name} shape unknown or not 4D")
        _, C_in, H, W_dim = in_shape

        # Weight shape: [C_out, C_in, kh, kw]
        W_fp = self.initializers[weight_name]
        C_out = W_fp.shape[0]

        out_h = (H + 2 * pad - kh) // stride + 1
        out_w = (W_dim + 2 * pad - kw) // stride + 1

        # im2col: [out_h*out_w, C_in*kh*kw]
        # Use quantized input for im2col
        input_q = quant_data.get(input_name)
        if input_q is None:
            # Input was computed by previous op - use zeros as placeholder
            # (golden will be computed separately)
            input_q = np.zeros(in_shape, dtype=np.int8)

        im2col_data = im2col(input_q[0], kh, kw, stride, pad)  # [out_h*out_w, C_in*kh*kw]
        im2col_name = input_name + '_im2col'
        im2col_M = out_h * out_w
        im2col_K = C_in * kh * kw

        # Allocate im2col in DDR and SRAM
        im2col_ddr = self._alloc_ddr(im2col_name, im2col_data.astype(np.int8).tobytes())
        im2col_sram = self._alloc_sram(im2col_name, im2col_M * im2col_K)
        im2col_id = self._add_tdesc(im2col_name, im2col_ddr, im2col_sram,
                                    im2col_M * im2col_K, [im2col_M, im2col_K])

        # Reshape weight: [C_out, C_in*kh*kw] -> this is already transposed form
        W_q = quant_data[weight_name].reshape(C_out, -1)  # [C_out, im2col_K]
        w_size = W_q.size
        w_sram = self._alloc_sram(weight_name + '_reshaped', w_size)
        w_ddr = self._alloc_ddr(weight_name + '_reshaped', W_q.astype(np.int8).tobytes())
        w_id = self._add_tdesc(weight_name + '_reshaped', w_ddr, w_sram,
                               w_size, [C_out, im2col_K])

        # DMA_LOAD im2col data and weights
        self._emit(encode_instr(OP_G_DMA_LOAD, src0=im2col_id))
        self._emit(encode_instr(OP_G_DMA_LOAD, src0=w_id))

        # Output: [im2col_M, C_out] = [out_h*out_w, C_out]
        out_size = im2col_M * C_out
        out_sram = self._alloc_sram(output_name, out_size)
        tensor_shapes[output_name] = (1, C_out, out_h, out_w)
        out_id = self._add_tdesc(output_name, DDR_GRAPH_IO_BASE, out_sram,
                                out_size, [im2col_M, C_out])

        # GEMM: im2col @ W^T = [M, K] @ [K, N] where W stored as [N, K]
        shift = max(0, int(math.ceil(math.log2(max(im2col_K, 1)))))
        scale_int = 1
        imm0 = (scale_int & 0xFF) | ((shift & 0xFF) << 8)
        flags = GFLAG_REQUANT | GFLAG_TRANSPOSE_B

        self._emit(encode_instr(OP_G_GEMM, flags=flags,
                                src0=im2col_id, src1=w_id, dst=out_id,
                                imm0=imm0))

        # Handle bias
        if bias_name and bias_name in quant_data:
            bias_q = quant_data[bias_name]
            bcast_name = bias_name + '_conv_bcast'
            bias_bcast = np.tile(bias_q.flatten(), im2col_M)
            bcast_size = bias_bcast.size
            bcast_ddr = self._alloc_ddr(bcast_name, bias_bcast.astype(np.int8).tobytes())
            bcast_sram = self._alloc_sram(bcast_name, bcast_size)
            bcast_id = self._add_tdesc(bcast_name, bcast_ddr, bcast_sram,
                                       bcast_size, [im2col_M, C_out])

            self._emit(encode_instr(OP_G_DMA_LOAD, src0=bcast_id))
            self._emit(encode_instr(OP_G_EW_ADD, src0=out_id, src1=bcast_id, dst=out_id))

        self.sram_live.add(output_name)
        current_data[output_name] = None
        quant_data[im2col_name] = im2col_data

    def _lower_reshape(self, op, quant_data, current_data, tensor_shapes):
        """Lower Reshape/Flatten: just alias the SRAM location."""
        input_name = op.input[0]
        output_name = op.output[0]

        if input_name not in self.sram_allocs:
            print(f"WARNING: Reshape input {input_name} not in SRAM")
            return

        in_sram, in_size = self.sram_allocs[input_name]
        out_shape = self.shapes.get(output_name)
        if out_shape is None:
            # Compute from reshape target
            if len(op.input) > 1 and op.input[1] in self.initializers:
                out_shape = tuple(self.initializers[op.input[1]].astype(int).tolist())
            else:
                out_shape = (1, in_size)

        tensor_shapes[output_name] = out_shape
        self.sram_allocs[output_name] = (in_sram, in_size)

        # Create descriptor with new shape (same SRAM location)
        in_id = self.tdesc_id.get(input_name)
        out_id = self._add_tdesc(output_name, 0, in_sram, in_size,
                                list(out_shape), len(out_shape))
        self.tdesc_id[output_name] = out_id

        # Input is already live in SRAM; the reshape is just an alias
        if input_name in self.sram_live:
            self.sram_live.add(output_name)
        current_data[output_name] = None

    def _lower_reduce(self, op, quant_data, current_data, tensor_shapes):
        """Lower ReduceSum/ReduceMax/ReduceMean via reduce_engine."""
        input_name = op.input[0]
        output_name = op.output[0]

        # Determine opcode
        opcode_map = {
            'ReduceSum': OP_G_REDUCE_SUM,
            'ReduceMax': OP_G_REDUCE_MAX,
            'ReduceMean': OP_G_REDUCE_MEAN,
        }
        opcode = opcode_map[op.op_type]

        # Get axis and keepdims attributes
        axes = None
        keepdims = 1
        for attr in op.attribute:
            if attr.name == 'axes':
                axes = list(attr.ints)
            elif attr.name == 'keepdims':
                keepdims = attr.i

        # Also check for axes as second input (ONNX opset 18+)
        if axes is None and len(op.input) > 1 and op.input[1] in self.initializers:
            axes = self.initializers[op.input[1]].flatten().tolist()

        in_shape = tensor_shapes.get(input_name, (1,))
        if axes is None:
            axes = list(range(len(in_shape)))

        # Normalize negative axes
        axes = [a if a >= 0 else a + len(in_shape) for a in axes]
        axis = axes[0]  # reduce_engine handles one axis

        # Compute reduce_dim and outer_count
        reduce_dim = in_shape[axis]
        outer_count = 1
        for i, s in enumerate(in_shape):
            if i != axis:
                outer_count *= s

        # Input must be in SRAM
        in_id = self.tdesc_id[input_name]
        if input_name not in self.sram_live:
            self._emit(encode_instr(OP_G_DMA_LOAD, src0=in_id))
            self.sram_live.add(input_name)

        # Allocate output
        out_shape = list(in_shape)
        if keepdims:
            out_shape[axis] = 1
        else:
            out_shape.pop(axis)
        if not out_shape:
            out_shape = [1]
        out_size = int(np.prod(out_shape))
        out_sram = self._alloc_sram(output_name, out_size)
        tensor_shapes[output_name] = tuple(out_shape)
        out_id = self._add_tdesc(output_name, 0, out_sram, out_size,
                                list(out_shape), len(out_shape))

        # Create a dummy src2 descriptor for dst (reduce engine uses src0=src, dst via src2)
        # Actually, the dispatch reads: src0 → src_base, src2 (=dst) → dst_base
        # imm0 = reduce_dim, imm1 = outer_count
        self._emit(encode_instr(opcode, src0=in_id, dst=out_id,
                                imm0=reduce_dim, imm1=outer_count))

        self.sram_live.add(output_name)
        current_data[output_name] = None

    def _lower_math(self, op, quant_data, current_data, tensor_shapes, opcode):
        """Lower element-wise math op (Exp/Log/Sqrt/Rsqrt) via math_engine."""
        input_name = op.input[0]
        output_name = op.output[0]

        in_shape = tensor_shapes.get(input_name, (1,))
        in_id = self.tdesc_id[input_name]
        if input_name not in self.sram_live:
            self._emit(encode_instr(OP_G_DMA_LOAD, src0=in_id))
            self.sram_live.add(input_name)

        in_sram, in_size = self.sram_allocs[input_name]

        # Allocate output (same shape as input)
        out_size = in_size
        out_sram = self._alloc_sram(output_name, out_size)
        tensor_shapes[output_name] = in_shape
        out_id = self._add_tdesc(output_name, 0, out_sram, out_size,
                                list(in_shape), len(in_shape))

        # Math engine: src0=input, src2=output, imm0=length
        self._emit(encode_instr(opcode, src0=in_id, dst=out_id,
                                imm0=in_size))

        self.sram_live.add(output_name)
        current_data[output_name] = None

    def _lower_gather(self, op, quant_data, current_data, tensor_shapes):
        """Lower Gather (axis=0) via gather_engine."""
        data_name = op.input[0]
        indices_name = op.input[1]
        output_name = op.output[0]

        axis = 0
        for attr in op.attribute:
            if attr.name == 'axis':
                axis = attr.i
        if axis != 0:
            print(f"WARNING: Gather axis={axis} not supported, only axis=0")

        data_shape = tensor_shapes.get(data_name, (1,))
        num_rows = data_shape[0] if len(data_shape) > 1 else 1
        row_size = int(np.prod(data_shape[1:])) if len(data_shape) > 1 else int(np.prod(data_shape))

        # Ensure data is in SRAM
        data_id = self.tdesc_id.get(data_name)
        if data_id is None:
            # Data might be an initializer that needs allocation
            if data_name in quant_data:
                arr = quant_data[data_name]
                d_size = arr.size
                d_ddr = self.ddr_allocs.get(data_name, (0, 0))[0]
                d_sram = self._alloc_sram(data_name, d_size)
                data_id = self._add_tdesc(data_name, d_ddr, d_sram, d_size,
                                         list(data_shape), len(data_shape))
        if data_name not in self.sram_live:
            self._emit(encode_instr(OP_G_DMA_LOAD, src0=data_id))
            self.sram_live.add(data_name)

        # Indices
        if indices_name in self.initializers:
            idx_arr = self.initializers[indices_name].flatten().astype(np.int8)
            idx_size = idx_arr.size
            idx_ddr = self._alloc_ddr(indices_name, idx_arr.tobytes())
            idx_sram = self._alloc_sram(indices_name, idx_size)
            idx_id = self._add_tdesc(indices_name, idx_ddr, idx_sram, idx_size,
                                    [idx_size], 1)
            self._emit(encode_instr(OP_G_DMA_LOAD, src0=idx_id))
            self.sram_live.add(indices_name)
            quant_data[indices_name] = idx_arr
            num_indices = idx_size
        else:
            idx_id = self.tdesc_id[indices_name]
            if indices_name not in self.sram_live:
                self._emit(encode_instr(OP_G_DMA_LOAD, src0=idx_id))
                self.sram_live.add(indices_name)
            idx_sram, idx_size = self.sram_allocs[indices_name]
            num_indices = idx_size

        # Output: [num_indices, row_size]
        out_size = num_indices * row_size
        out_sram = self._alloc_sram(output_name, out_size)
        out_shape = [num_indices, row_size] if row_size > 1 else [num_indices]
        tensor_shapes[output_name] = tuple(out_shape)
        out_id = self._add_tdesc(output_name, 0, out_sram, out_size,
                                list(out_shape), len(out_shape))

        # Gather: src0=data, src1=indices, dst=output
        # imm0=row_size, imm1=num_rows, imm2=num_indices
        self._emit(encode_instr(OP_G_GATHER, src0=data_id, src1=idx_id, dst=out_id,
                                imm0=row_size, imm1=num_rows, imm2=num_indices))

        self.sram_live.add(output_name)
        current_data[output_name] = None

    def _lower_slice(self, op, quant_data, current_data, tensor_shapes):
        """Lower Slice via slice_engine."""
        input_name = op.input[0]
        output_name = op.output[0]

        # ONNX Slice: inputs are data, starts, ends, axes, steps
        starts = self.initializers.get(op.input[1], np.array([0])).flatten().tolist() if len(op.input) > 1 else [0]
        ends = self.initializers.get(op.input[2], np.array([0])).flatten().tolist() if len(op.input) > 2 else [0]
        axes_vals = self.initializers.get(op.input[3], np.array([0])).flatten().tolist() if len(op.input) > 3 else [0]

        in_shape = tensor_shapes.get(input_name, (1,))
        axis = int(axes_vals[0])
        if axis < 0:
            axis += len(in_shape)
        start = int(starts[0])
        end = int(ends[0])
        if end > in_shape[axis]:
            end = in_shape[axis]
        if start < 0:
            start += in_shape[axis]
        slice_len = end - start

        in_id = self.tdesc_id[input_name]
        if input_name not in self.sram_live:
            self._emit(encode_instr(OP_G_DMA_LOAD, src0=in_id))
            self.sram_live.add(input_name)

        in_sram, in_size = self.sram_allocs[input_name]

        # Compute row parameters for the last-dim slice
        # For axis=last_dim: src_row_len = in_shape[-1], dst_row_len = slice_len,
        # start_offset = start, num_rows = product of all other dims
        if axis == len(in_shape) - 1:
            src_row_len = in_shape[-1]
            dst_row_len = slice_len
            start_offset = start
            num_rows = int(np.prod(in_shape[:-1])) if len(in_shape) > 1 else 1
        else:
            # For non-last axis: flatten and treat as 2D
            inner = int(np.prod(in_shape[axis+1:])) if axis + 1 < len(in_shape) else 1
            src_row_len = in_shape[axis] * inner
            dst_row_len = slice_len * inner
            start_offset = start * inner
            num_rows = int(np.prod(in_shape[:axis])) if axis > 0 else 1

        # Output shape
        out_shape = list(in_shape)
        out_shape[axis] = slice_len
        out_size = int(np.prod(out_shape))
        out_sram = self._alloc_sram(output_name, out_size)
        tensor_shapes[output_name] = tuple(out_shape)
        out_id = self._add_tdesc(output_name, 0, out_sram, out_size,
                                list(out_shape), len(out_shape))

        # Slice: src0=input, src2=output, imm0=start_offset, imm1=slice_length(dst_row_len)
        # Additional params encoded: src_row_len in the shape fields
        # The dispatch uses: td0.sram_addr→src, td2.sram_addr→dst,
        # td0.shape1→src_row_len, imm0→start_offset, imm1[15:0]→dst_row_len,
        # td0.shape0→num_rows
        # So we need to ensure td0 shape fields have the right values
        # Update src descriptor shape to encode row layout
        self._emit(encode_instr(OP_G_SLICE, src0=in_id, dst=out_id,
                                imm0=start_offset,
                                imm1=((num_rows & 0xFFFF) << 16) | (dst_row_len & 0xFFFF),
                                imm2=src_row_len))

        self.sram_live.add(output_name)
        current_data[output_name] = None

    def _lower_concat(self, op, quant_data, current_data, tensor_shapes):
        """Lower Concat (2 inputs, last-dim) via concat_engine."""
        if len(op.input) < 2:
            print(f"WARNING: Concat needs at least 2 inputs")
            return

        src0_name = op.input[0]
        src1_name = op.input[1]
        output_name = op.output[0]

        axis = 0
        for attr in op.attribute:
            if attr.name == 'axis':
                axis = attr.i

        src0_shape = tensor_shapes.get(src0_name, (1,))
        src1_shape = tensor_shapes.get(src1_name, (1,))
        if axis < 0:
            axis += len(src0_shape)

        src0_id = self.tdesc_id[src0_name]
        src1_id = self.tdesc_id[src1_name]
        if src0_name not in self.sram_live:
            self._emit(encode_instr(OP_G_DMA_LOAD, src0=src0_id))
            self.sram_live.add(src0_name)
        if src1_name not in self.sram_live:
            self._emit(encode_instr(OP_G_DMA_LOAD, src0=src1_id))
            self.sram_live.add(src1_name)

        # Last-dim concat
        if axis == len(src0_shape) - 1:
            src0_row_len = src0_shape[-1]
            src1_row_len = src1_shape[-1]
            num_rows = int(np.prod(src0_shape[:-1])) if len(src0_shape) > 1 else 1
        else:
            inner = int(np.prod(src0_shape[axis+1:])) if axis + 1 < len(src0_shape) else 1
            src0_row_len = src0_shape[axis] * inner
            src1_row_len = src1_shape[axis] * inner
            num_rows = int(np.prod(src0_shape[:axis])) if axis > 0 else 1

        out_shape = list(src0_shape)
        out_shape[axis] = src0_shape[axis] + src1_shape[axis]
        out_size = int(np.prod(out_shape))
        out_sram = self._alloc_sram(output_name, out_size)
        tensor_shapes[output_name] = tuple(out_shape)
        out_id = self._add_tdesc(output_name, 0, out_sram, out_size,
                                list(out_shape), len(out_shape))

        # Concat: src0=first, src1=second, dst=output
        # imm0=src0_row_len, imm1=src1_row_len, imm2=num_rows
        self._emit(encode_instr(OP_G_CONCAT, src0=src0_id, src1=src1_id, dst=out_id,
                                imm0=src0_row_len, imm1=src1_row_len, imm2=num_rows))

        self.sram_live.add(output_name)
        current_data[output_name] = None

    def _lower_batchnorm(self, op, quant_data, current_data, tensor_shapes):
        """Lower BatchNormalization to EW_MUL + EW_ADD (compiler lowering)."""
        input_name = op.input[0]
        scale_name = op.input[1]
        bias_name = op.input[2]
        mean_name = op.input[3]
        var_name = op.input[4]
        output_name = op.output[0]

        epsilon = 1e-5
        for attr in op.attribute:
            if attr.name == 'epsilon':
                epsilon = attr.f

        in_shape = tensor_shapes.get(input_name, (1,))
        in_id = self.tdesc_id[input_name]
        if input_name not in self.sram_live:
            self._emit(encode_instr(OP_G_DMA_LOAD, src0=in_id))
            self.sram_live.add(input_name)

        in_sram, in_size = self.sram_allocs[input_name]

        # Compute multiplier = scale / sqrt(var + eps) and offset = bias - mean * multiplier
        # These are per-channel, need broadcast to match input shape
        scale_fp = self.initializers[scale_name].astype(np.float32)
        bias_fp = self.initializers[bias_name].astype(np.float32)
        mean_fp = self.initializers[mean_name].astype(np.float32)
        var_fp = self.initializers[var_name].astype(np.float32)

        multiplier_fp = scale_fp / np.sqrt(var_fp + epsilon)
        offset_fp = bias_fp - mean_fp * multiplier_fp

        # Quantize multiplier and offset
        mult_q, _ = quantize_int8(multiplier_fp)
        off_q, _ = quantize_int8(offset_fp)

        # Broadcast to match input size (NCHW: replicate per channel across H*W)
        C = in_shape[1] if len(in_shape) > 1 else 1
        spatial = int(np.prod(in_shape[2:])) if len(in_shape) > 2 else 1
        mult_bcast = np.repeat(mult_q, spatial)
        mult_bcast = np.tile(mult_bcast, in_shape[0] if len(in_shape) > 0 else 1)
        off_bcast = np.repeat(off_q, spatial)
        off_bcast = np.tile(off_bcast, in_shape[0] if len(in_shape) > 0 else 1)

        # Allocate multiplier broadcast in DDR+SRAM
        mult_name = output_name + '_bn_mult'
        mult_ddr = self._alloc_ddr(mult_name, mult_bcast.astype(np.int8).tobytes())
        mult_sram = self._alloc_sram(mult_name, mult_bcast.size)
        mult_id = self._add_tdesc(mult_name, mult_ddr, mult_sram,
                                 mult_bcast.size, list(in_shape), len(in_shape))
        self._emit(encode_instr(OP_G_DMA_LOAD, src0=mult_id))

        # EW_MUL: input * multiplier → temp
        temp_name = output_name + '_bn_temp'
        temp_sram = self._alloc_sram(temp_name, in_size)
        temp_id = self._add_tdesc(temp_name, 0, temp_sram, in_size,
                                 list(in_shape), len(in_shape))
        self.sram_allocs[temp_name] = (temp_sram, in_size)
        self._emit(encode_instr(OP_G_EW_MUL, src0=in_id, src1=mult_id, dst=temp_id))

        # Allocate offset broadcast in DDR+SRAM
        off_name = output_name + '_bn_off'
        off_ddr = self._alloc_ddr(off_name, off_bcast.astype(np.int8).tobytes())
        off_sram = self._alloc_sram(off_name, off_bcast.size)
        off_id = self._add_tdesc(off_name, off_ddr, off_sram,
                                off_bcast.size, list(in_shape), len(in_shape))
        self._emit(encode_instr(OP_G_DMA_LOAD, src0=off_id))

        # EW_ADD: temp + offset → output
        out_sram = self._alloc_sram(output_name, in_size)
        tensor_shapes[output_name] = in_shape
        out_id = self._add_tdesc(output_name, 0, out_sram, in_size,
                                list(in_shape), len(in_shape))
        self.sram_allocs[output_name] = (out_sram, in_size)
        self._emit(encode_instr(OP_G_EW_ADD, src0=temp_id, src1=off_id, dst=out_id))

        self.sram_live.add(output_name)
        current_data[output_name] = None

    def _lower_avgpool2d(self, op, quant_data, current_data, tensor_shapes):
        """Lower AveragePool via avgpool2d_engine."""
        input_name = op.input[0]
        output_name = op.output[0]

        # Get attributes
        kernel_shape = [2, 2]
        strides = [2, 2]
        for attr in op.attribute:
            if attr.name == 'kernel_shape':
                kernel_shape = list(attr.ints)
            elif attr.name == 'strides':
                strides = list(attr.ints)

        kh, kw = kernel_shape
        sh, sw = strides

        # Input shape: [N, C, H, W]
        in_shape = tensor_shapes.get(input_name, (1, 1, 1, 1))
        if len(in_shape) != 4:
            print(f"WARNING: AvgPool input shape is not 4D: {in_shape}")
            return
        N, C, H, W_dim = in_shape

        # Ensure input is in SRAM
        in_id = self.tdesc_id[input_name]
        if input_name not in self.sram_live:
            self._emit(encode_instr(OP_G_DMA_LOAD, src0=in_id))
            self.sram_live.add(input_name)

        # Compute output shape
        out_h = (H - kh) // sh + 1
        out_w = (W_dim - kw) // sw + 1
        out_shape = [N, C, out_h, out_w]
        out_size = int(np.prod(out_shape))

        # Allocate output SRAM
        out_sram = self._alloc_sram(output_name, out_size)
        tensor_shapes[output_name] = tuple(out_shape)
        out_id = self._add_tdesc(output_name, 0, out_sram, out_size,
                                list(out_shape), len(out_shape))

        # Encode: imm0 = (kh << 8) | kw, imm1 = (sh << 8) | sw
        imm0 = ((kh & 0xFF) << 8) | (kw & 0xFF)
        imm1 = ((sh & 0xFF) << 8) | (sw & 0xFF)

        # AvgPool2D: src0=input, dst=output
        self._emit(encode_instr(OP_G_AVGPOOL2D, src0=in_id, dst=out_id,
                                imm0=imm0, imm1=imm1))

        self.sram_live.add(output_name)
        current_data[output_name] = None

    def _compute_golden(self, ops, quant_data, current_data, tensor_shapes):
        """Compute golden output using pure int8 arithmetic."""
        input_name = self.graph.input[0].name
        tensors = {input_name: quant_data[input_name].flatten().astype(np.int8)}

        for op in ops:
            if op.op_type == 'Gemm':
                inp = tensors[op.input[0]]
                W = quant_data[op.input[1]]
                bias = quant_data[op.input[2]] if len(op.input) > 2 else None

                transB = 0
                for attr in op.attribute:
                    if attr.name == 'transB':
                        transB = attr.i

                in_shape = tensor_shapes.get(op.input[0], (1, inp.size))
                M = in_shape[0] if len(in_shape) > 1 else 1
                if transB:
                    N, K = W.shape
                else:
                    K, N = W.shape

                A = inp.reshape(M, K).astype(np.int8)
                if transB:
                    B = W.reshape(N, K).T.astype(np.int8)  # [K, N]
                else:
                    B = W.reshape(K, N).astype(np.int8)

                shift = max(0, int(math.ceil(math.log2(max(K, 1)))))
                scale_int = 1

                # GEMM without bias (RTL adds bias via separate EW_ADD after requant)
                result = int8_gemm_golden(A, B, None, scale_int, shift)

                # Add bias after requant (matches RTL: saturating INT8 add)
                if bias is not None:
                    bias_flat = bias.flatten().astype(np.int8)
                    for m in range(M):
                        for n in range(N):
                            val = int(result[m, n]) + int(bias_flat[n])
                            result[m, n] = np.clip(val, -128, 127).astype(np.int8)

                tensors[op.output[0]] = result.flatten()
                tensor_shapes[op.output[0]] = (M, N)

            elif op.op_type == 'Relu':
                inp = tensors[op.input[0]]
                tensors[op.output[0]] = np.maximum(inp, 0).astype(np.int8)
                tensor_shapes[op.output[0]] = tensor_shapes.get(op.input[0])

            elif op.op_type == 'Conv':
                inp = tensors[op.input[0]]
                W_fp = self.initializers[op.input[1]]
                W_q = quant_data[op.input[1]]
                bias = quant_data[op.input[2]] if len(op.input) > 2 else None

                kernel_shape = [3, 3]
                pads = [0, 0, 0, 0]
                strides = [1, 1]
                for attr in op.attribute:
                    if attr.name == 'kernel_shape': kernel_shape = list(attr.ints)
                    elif attr.name == 'pads': pads = list(attr.ints)
                    elif attr.name == 'strides': strides = list(attr.ints)

                kh, kw = kernel_shape
                pad = pads[0]
                stride = strides[0]

                in_shape = tensor_shapes.get(op.input[0])
                C_in, H, W_dim = in_shape[1], in_shape[2], in_shape[3]
                C_out = W_fp.shape[0]
                out_h = (H + 2 * pad - kh) // stride + 1
                out_w = (W_dim + 2 * pad - kw) // stride + 1

                # im2col
                input_4d = inp.reshape(1, C_in, H, W_dim).astype(np.int8)
                col = im2col(input_4d[0], kh, kw, stride, pad)  # [M, K]

                # Weight as [C_out, K] -> transpose to [K, C_out]
                W_2d = W_q.reshape(C_out, -1)  # [C_out, K]
                # GEMM: col @ W_2d^T = [M, K] @ [K, C_out]
                B_mat = W_2d.T.astype(np.int8)  # [K, C_out]

                K_dim = col.shape[1]
                shift = max(0, int(math.ceil(math.log2(max(K_dim, 1)))))
                M_dim = col.shape[0]
                # GEMM without bias (RTL adds bias via separate EW_ADD after requant)
                result = int8_gemm_golden(col.astype(np.int8), B_mat, None, 1, shift)

                # Add bias after requant (matches RTL)
                if bias is not None:
                    bias_flat = bias.flatten().astype(np.int8)
                    for m in range(M_dim):
                        for n in range(C_out):
                            val = int(result[m, n]) + int(bias_flat[n])
                            result[m, n] = np.clip(val, -128, 127).astype(np.int8)

                tensors[op.output[0]] = result.flatten()
                tensor_shapes[op.output[0]] = (1, C_out, out_h, out_w)

            elif op.op_type in ('Reshape', 'Flatten'):
                tensors[op.output[0]] = tensors[op.input[0]]
                out_shape = self.shapes.get(op.output[0])
                if out_shape is None:
                    if len(op.input) > 1 and op.input[1] in self.initializers:
                        out_shape = tuple(self.initializers[op.input[1]].astype(int).tolist())
                    else:
                        out_shape = (1, tensors[op.input[0]].size)
                tensor_shapes[op.output[0]] = out_shape

            elif op.op_type in ('ReduceSum', 'ReduceMax', 'ReduceMean'):
                inp = tensors[op.input[0]]
                in_shape = tensor_shapes.get(op.input[0], (inp.size,))
                inp_nd = inp.reshape(in_shape)

                axes = None
                keepdims = 1
                for attr in op.attribute:
                    if attr.name == 'axes':
                        axes = list(attr.ints)
                    elif attr.name == 'keepdims':
                        keepdims = attr.i
                if axes is None and len(op.input) > 1 and op.input[1] in self.initializers:
                    axes = self.initializers[op.input[1]].flatten().tolist()
                if axes is None:
                    axes = list(range(len(in_shape)))
                axes = [a if a >= 0 else a + len(in_shape) for a in axes]
                axis = axes[0]

                reduce_dim = in_shape[axis]
                # INT32 accumulation matching RTL
                # INT32 accumulation matching RTL
                acc = np.sum(inp_nd.astype(np.int32), axis=axis, keepdims=bool(keepdims))
                if op.op_type == 'ReduceSum':
                    result = np.clip(acc, -128, 127).astype(np.int8)
                elif op.op_type == 'ReduceMax':
                    result = np.max(inp_nd.astype(np.int8), axis=axis, keepdims=bool(keepdims))
                else:  # ReduceMean
                    # Match RTL: (acc + reduce_dim/2) / reduce_dim
                    acc_rounded = acc + int(reduce_dim // 2)
                    result_32 = acc_rounded // int(reduce_dim)
                    result = np.clip(result_32, -128, 127).astype(np.int8)

                tensors[op.output[0]] = result.flatten().astype(np.int8)
                out_shape = tuple(result.shape) if result.ndim > 0 else (1,)
                tensor_shapes[op.output[0]] = out_shape

            elif op.op_type in ('Exp', 'Log', 'Sqrt', 'Reciprocal'):
                inp = tensors[op.input[0]]
                in_shape = tensor_shapes.get(op.input[0], (inp.size,))
                # Match RTL LUT: input_scale=32, output_scale=32
                input_scale = 32.0
                result = np.zeros_like(inp, dtype=np.int8)
                for i in range(inp.size):
                    signed_val = int(np.int8(inp[i]))
                    x = signed_val / input_scale
                    if op.op_type == 'Exp':
                        y = np.exp(x)
                    elif op.op_type == 'Log':
                        y = np.log(max(x, 1e-6)) if x > 0 else np.log(1e-6)
                    elif op.op_type == 'Sqrt':
                        y = np.sqrt(max(x, 0))
                    else:  # Reciprocal (rsqrt)
                        y = 1.0 / np.sqrt(max(x, 1e-6)) if x > 0 else 1.0 / np.sqrt(1e-6)
                    result[i] = np.clip(int(round(y * input_scale)), -128, 127)
                tensors[op.output[0]] = result.astype(np.int8)
                tensor_shapes[op.output[0]] = in_shape

            elif op.op_type == 'Gather':
                data = tensors[op.input[0]]
                data_shape = tensor_shapes.get(op.input[0], (data.size,))
                data_nd = data.reshape(data_shape)

                if op.input[1] in self.initializers:
                    indices = self.initializers[op.input[1]].flatten().astype(int)
                elif op.input[1] in tensors:
                    indices = tensors[op.input[1]].astype(int)
                else:
                    indices = np.array([0])

                axis = 0
                for attr in op.attribute:
                    if attr.name == 'axis':
                        axis = attr.i

                result = np.take(data_nd, indices, axis=axis)
                tensors[op.output[0]] = result.flatten().astype(np.int8)
                tensor_shapes[op.output[0]] = tuple(result.shape)

            elif op.op_type == 'Slice':
                inp = tensors[op.input[0]]
                in_shape = tensor_shapes.get(op.input[0], (inp.size,))
                inp_nd = inp.reshape(in_shape)

                starts = self.initializers.get(op.input[1], np.array([0])).flatten().tolist() if len(op.input) > 1 else [0]
                ends = self.initializers.get(op.input[2], np.array([0])).flatten().tolist() if len(op.input) > 2 else [0]
                axes_vals = self.initializers.get(op.input[3], np.array([0])).flatten().tolist() if len(op.input) > 3 else [0]

                axis = int(axes_vals[0])
                if axis < 0:
                    axis += len(in_shape)
                start = int(starts[0])
                end = int(ends[0])
                if end > in_shape[axis]:
                    end = in_shape[axis]
                if start < 0:
                    start += in_shape[axis]

                slices = [slice(None)] * len(in_shape)
                slices[axis] = slice(start, end)
                result = inp_nd[tuple(slices)]
                tensors[op.output[0]] = result.flatten().astype(np.int8)
                tensor_shapes[op.output[0]] = tuple(result.shape)

            elif op.op_type == 'Concat':
                arrays = []
                for name in op.input:
                    arr = tensors[name]
                    shape = tensor_shapes.get(name, (arr.size,))
                    arrays.append(arr.reshape(shape))

                axis = 0
                for attr in op.attribute:
                    if attr.name == 'axis':
                        axis = attr.i

                result = np.concatenate(arrays, axis=axis)
                tensors[op.output[0]] = result.flatten().astype(np.int8)
                tensor_shapes[op.output[0]] = tuple(result.shape)

            elif op.op_type == 'BatchNormalization':
                inp = tensors[op.input[0]]
                in_shape = tensor_shapes.get(op.input[0], (1,))

                scale_fp = self.initializers[op.input[1]].astype(np.float32)
                bias_fp = self.initializers[op.input[2]].astype(np.float32)
                mean_fp = self.initializers[op.input[3]].astype(np.float32)
                var_fp = self.initializers[op.input[4]].astype(np.float32)

                epsilon = 1e-5
                for attr in op.attribute:
                    if attr.name == 'epsilon':
                        epsilon = attr.f

                multiplier_fp = scale_fp / np.sqrt(var_fp + epsilon)
                offset_fp = bias_fp - mean_fp * multiplier_fp
                mult_q, _ = quantize_int8(multiplier_fp)
                off_q, _ = quantize_int8(offset_fp)

                C = in_shape[1] if len(in_shape) > 1 else 1
                spatial = int(np.prod(in_shape[2:])) if len(in_shape) > 2 else 1
                mult_bcast = np.repeat(mult_q, spatial)
                mult_bcast = np.tile(mult_bcast, in_shape[0] if len(in_shape) > 0 else 1)
                off_bcast = np.repeat(off_q, spatial)
                off_bcast = np.tile(off_bcast, in_shape[0] if len(in_shape) > 0 else 1)

                # EW_MUL: saturating int8 multiply (product >> 7)
                temp = np.zeros(inp.size, dtype=np.int8)
                for i in range(inp.size):
                    product = int(np.int8(inp[i])) * int(np.int8(mult_bcast[i]))
                    temp[i] = np.clip(product >> 7, -128, 127)

                # EW_ADD: saturating int8 add
                result = np.zeros(inp.size, dtype=np.int8)
                for i in range(inp.size):
                    val = int(temp[i]) + int(np.int8(off_bcast[i]))
                    result[i] = np.clip(val, -128, 127)

                tensors[op.output[0]] = result.astype(np.int8)
                tensor_shapes[op.output[0]] = in_shape

            elif op.op_type == 'AveragePool':
                inp = tensors[op.input[0]]
                in_shape = tensor_shapes.get(op.input[0], (1, 1, 1, 1))
                N, C, H, W_dim = in_shape

                kernel_shape = [2, 2]
                strides_p = [2, 2]
                for attr in op.attribute:
                    if attr.name == 'kernel_shape':
                        kernel_shape = list(attr.ints)
                    elif attr.name == 'strides':
                        strides_p = list(attr.ints)

                kh, kw = kernel_shape
                sh, sw = strides_p
                out_h = (H - kh) // sh + 1
                out_w = (W_dim - kw) // sw + 1
                inp_nd = inp.reshape(N, C, H, W_dim).astype(np.int32)

                result = np.zeros((N, C, out_h, out_w), dtype=np.int8)
                pool_size = kh * kw
                for n in range(N):
                    for c in range(C):
                        for oh in range(out_h):
                            for ow in range(out_w):
                                acc = np.int32(0)
                                for ph in range(kh):
                                    for pw in range(kw):
                                        acc += inp_nd[n, c, oh*sh+ph, ow*sw+pw]
                                # Match RTL reduce_mean: (acc + pool_size/2) / pool_size
                                acc_rounded = int(acc) + pool_size // 2
                                val = acc_rounded // pool_size
                                result[n, c, oh, ow] = np.clip(val, -128, 127)

                tensors[op.output[0]] = result.flatten().astype(np.int8)
                tensor_shapes[op.output[0]] = (N, C, out_h, out_w)

        output_name = self.graph.output[0].name
        return tensors.get(output_name, np.array([], dtype=np.int8))

    def write_outputs(self, outdir, golden):
        """Write all artifacts to output directory."""
        os.makedirs(outdir, exist_ok=True)

        # program.bin
        prog_path = os.path.join(outdir, 'program.bin')
        with open(prog_path, 'wb') as f:
            for instr in self.instructions:
                f.write(instr)
        print(f"  Written {len(self.instructions)} instructions to {prog_path}")

        # tdesc.bin
        tdesc_path = os.path.join(outdir, 'tdesc.bin')
        with open(tdesc_path, 'wb') as f:
            for name, data in self.tdesc_list:
                f.write(data)
        print(f"  Written {len(self.tdesc_list)} descriptors to {tdesc_path}")

        # ddr_image.bin
        # Build full DDR image: data region starts at DDR_GRAPH_DATA_BASE
        ddr_path = os.path.join(outdir, 'ddr_image.bin')
        full_ddr = bytearray(DDR_GRAPH_DATA_BASE) + self.ddr_image
        with open(ddr_path, 'wb') as f:
            f.write(full_ddr)
        print(f"  Written DDR image ({len(full_ddr)} bytes) to {ddr_path}")

        # golden.bin
        golden_path = os.path.join(outdir, 'golden.bin')
        golden_bytes = golden.astype(np.int8).tobytes()
        with open(golden_path, 'wb') as f:
            f.write(golden_bytes)
        print(f"  Written golden ({len(golden_bytes)} bytes) to {golden_path}")

        # golden.npy
        np.save(os.path.join(outdir, 'golden.npy'), golden.astype(np.int8))

        # manifest.json
        manifest = {
            'num_instructions': len(self.instructions),
            'num_descriptors': len(self.tdesc_list),
            'ddr_image_size': len(full_ddr),
            'golden_size': len(golden_bytes),
            'sram_used': self.sram_offset,
            'descriptors': [(name, idx) for idx, (name, _) in enumerate(self.tdesc_list)],
            'sram_allocs': {k: {'offset': v[0], 'size': v[1]}
                           for k, v in self.sram_allocs.items()},
        }
        manifest_path = os.path.join(outdir, 'manifest.json')
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        print(f"  Written manifest to {manifest_path}")


def main():
    parser = argparse.ArgumentParser(description='ONNX → NPU Graph ISA Compiler')
    parser.add_argument('--model', required=True, help='Path to ONNX model')
    parser.add_argument('--outdir', default='build/graph', help='Output directory')
    parser.add_argument('--no-reuse', action='store_true',
                       help='Disable memory planner (use bump allocator)')
    args = parser.parse_args()

    print(f"=== ONNX Compiler ===")
    print(f"Model: {args.model}")
    print(f"Output: {args.outdir}")
    print(f"Memory reuse: {'disabled' if args.no_reuse else 'enabled'}")

    compiler = GraphCompiler(args.model, use_reuse=not args.no_reuse)
    golden = compiler.compile()

    print(f"\nCompilation results:")
    print(f"  Instructions: {len(compiler.instructions)}")
    print(f"  Descriptors:  {len(compiler.tdesc_list)}")
    print(f"  SRAM used:    {compiler.sram_offset} / 65536 bytes")
    if compiler.use_reuse:
        no_reuse = compiler.sram_no_reuse_offset
        with_reuse = compiler.sram_offset
        if no_reuse > 0:
            reduction = 100.0 * (1.0 - with_reuse / no_reuse)
            print(f"  Peak SRAM: {with_reuse}b with reuse vs {no_reuse}b without ({reduction:.0f}% reduction)")
    print(f"  Golden shape: {golden.shape}")

    compiler.write_outputs(args.outdir, golden)
    print(f"\nDone!")


if __name__ == '__main__':
    main()
