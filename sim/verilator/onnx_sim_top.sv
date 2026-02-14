// =============================================================================
// onnx_sim_top.sv - Graph Mode simulation wrapper
// Contains: program SRAM, tensor table, SRAM0 (65536), ACC SRAM, scratch SRAM,
//           gemm_ctrl, systolic_array, softmax_engine, graph pipeline, DMA shim,
//           Phase 3 engines: reduce, math, gather, slice, concat, avgpool2d
//
// SRAM0 mux priority: gp_ew > gm > sm > re > me > ga > sl > ct > ap > tb
// =============================================================================
`default_nettype none

module onnx_sim_top
    import npu_pkg::*;
    import graph_isa_pkg::*;
#(
    parameter int SRAM0_DEPTH   = 65536,
    parameter int SCRATCH_DEPTH = 4096,
    parameter int PROG_DEPTH    = 1024,
    parameter int SRAM0_AW      = $clog2(SRAM0_DEPTH),
    parameter int SCR_AW        = $clog2(SCRATCH_DEPTH),
    parameter int PROG_AW       = $clog2(PROG_DEPTH)
)(
    input  wire                clk,
    input  wire                rst_n,

    // --- Control ---
    input  wire                start_pulse,
    input  wire  [15:0]        prog_len,
    output wire                graph_done,

    // --- Program SRAM write (TB loads program via port B) ---
    input  wire                prog_wr_en,
    input  wire  [PROG_AW-1:0] prog_wr_addr,
    input  wire  [127:0]       prog_wr_data,

    // --- Tensor Table write (TB loads descriptors) ---
    input  wire                tdesc_wr_en,
    input  wire  [7:0]         tdesc_wr_addr,
    input  wire  [255:0]       tdesc_wr_data,

    // --- DATA SRAM0 TB access ---
    input  wire                tb_sram0_wr_en,
    input  wire  [SRAM0_AW-1:0] tb_sram0_wr_addr,
    input  wire  [7:0]         tb_sram0_wr_data,
    input  wire                tb_sram0_rd_en,
    input  wire  [SRAM0_AW-1:0] tb_sram0_rd_addr,
    output wire  [7:0]         tb_sram0_rd_data,

    // --- DMA command capture (exposed to C++) ---
    output wire                dma_cmd_captured,
    output wire  [31:0]        dma_ddr_addr,
    output wire  [15:0]        dma_sram_addr,
    output wire  [15:0]        dma_length,
    output wire                dma_direction,
    output wire                dma_strided,
    output wire  [31:0]        dma_stride,
    output wire  [15:0]        dma_count,
    output wire  [15:0]        dma_block_len,
    input  wire                dma_done_pulse,

    // --- Performance counters (exposed to C++) ---
    output wire  [31:0]        perf_total_cycles,
    output wire  [31:0]        perf_gemm_cycles,
    output wire  [31:0]        perf_softmax_cycles,
    output wire  [31:0]        perf_dma_cycles,
    output wire  [31:0]        perf_reduce_cycles,
    output wire  [31:0]        perf_math_cycles,
    output wire  [31:0]        perf_gather_cycles,
    output wire  [31:0]        perf_slice_cycles,
    output wire  [31:0]        perf_concat_cycles,
    output wire  [31:0]        perf_avgpool_cycles,
    output wire  [31:0]        perf_ew_cycles,

    // --- Debug ---
    output wire  [31:0]        graph_status,
    output wire  [15:0]        graph_pc,
    output wire  [7:0]         graph_last_op,
    output wire                graph_busy
);

    // ================================================================
    // Program SRAM (128-bit x PROG_DEPTH)
    // ================================================================
    logic                  prog_rd_en;
    logic [PROG_AW-1:0]   prog_rd_addr;
    logic [127:0]          prog_rd_data;

    sram_dp #(.DEPTH(PROG_DEPTH), .WIDTH(128)) u_prog_sram (
        .clk    (clk),
        .en_a   (prog_rd_en),
        .we_a   (1'b0),
        .addr_a (prog_rd_addr),
        .din_a  (128'd0),
        .dout_a (prog_rd_data),
        .en_b   (prog_wr_en),
        .we_b   (prog_wr_en),
        .addr_b (prog_wr_addr),
        .din_b  (prog_wr_data),
        .dout_b ()
    );

    // ================================================================
    // Tensor Descriptor Table
    // ================================================================
    logic [7:0]   td_rd0_addr, td_rd1_addr, td_rd2_addr;
    logic [255:0] td_rd0_data, td_rd1_data, td_rd2_data;

    tensor_table u_tensor_table (
        .clk      (clk),
        .rst_n    (rst_n),
        .wr_en    (tdesc_wr_en),
        .wr_addr  (tdesc_wr_addr),
        .wr_data  (tdesc_wr_data),
        .rd0_addr (td_rd0_addr),
        .rd0_data (td_rd0_data),
        .rd1_addr (td_rd1_addr),
        .rd1_data (td_rd1_data),
        .rd2_addr (td_rd2_addr),
        .rd2_data (td_rd2_data)
    );

    // ================================================================
    // GEMM Engine: gemm_ctrl + systolic_array + ACC SRAM
    // ================================================================
    logic        gm_busy, gm_done;
    logic        gm_rd_en, gm_wr_en;
    logic [15:0] gm_rd_addr, gm_wr_addr;
    logic [7:0]  gm_wr_data;
    logic        gm_sa_clear, gm_sa_en;
    logic signed [7:0]  gm_sa_a_col [16];
    logic signed [7:0]  gm_sa_b_row [16];
    logic signed [31:0] gm_sa_acc   [16][16];

    logic        gm_acc_rd_en, gm_acc_wr_en;
    logic [7:0]  gm_acc_rd_addr, gm_acc_wr_addr;
    logic signed [31:0] gm_acc_rd_data, gm_acc_wr_data;

    // Graph pipeline signals
    logic        gp_gm_cmd_valid;
    logic [15:0] gp_gm_cmd_src0, gp_gm_cmd_src1, gp_gm_cmd_dst;
    logic [15:0] gp_gm_cmd_M, gp_gm_cmd_N, gp_gm_cmd_K;
    logic [7:0]  gp_gm_cmd_flags;
    logic [15:0] gp_gm_cmd_imm;

    gemm_ctrl #(
        .ARRAY_M     (16),
        .ARRAY_N     (16),
        .DATA_W      (8),
        .ACC_W       (32),
        .SRAM_ADDR_W (16)
    ) u_gemm_ctrl (
        .clk          (clk),
        .rst_n        (rst_n),
        .cmd_valid    (gp_gm_cmd_valid),
        .cmd_src0     (gp_gm_cmd_src0),
        .cmd_src1     (gp_gm_cmd_src1),
        .cmd_dst      (gp_gm_cmd_dst),
        .cmd_M        (gp_gm_cmd_M),
        .cmd_N        (gp_gm_cmd_N),
        .cmd_K        (gp_gm_cmd_K),
        .cmd_flags    (gp_gm_cmd_flags),
        .cmd_imm      (gp_gm_cmd_imm),
        .sram_rd_en   (gm_rd_en),
        .sram_rd_addr (gm_rd_addr),
        .sram_rd_data (s0_a_dout),
        .sram_wr_en   (gm_wr_en),
        .sram_wr_addr (gm_wr_addr),
        .sram_wr_data (gm_wr_data),
        .acc_rd_en    (gm_acc_rd_en),
        .acc_rd_addr  (gm_acc_rd_addr),
        .acc_rd_data  (gm_acc_rd_data),
        .acc_wr_en    (gm_acc_wr_en),
        .acc_wr_addr  (gm_acc_wr_addr),
        .acc_wr_data  (gm_acc_wr_data),
        .sa_clear     (gm_sa_clear),
        .sa_en        (gm_sa_en),
        .sa_a_col     (gm_sa_a_col),
        .sa_b_row     (gm_sa_b_row),
        .sa_acc       (gm_sa_acc),
        .busy         (gm_busy),
        .done         (gm_done)
    );

    systolic_array #(
        .M      (16),
        .N      (16),
        .DATA_W (8),
        .ACC_W  (32)
    ) u_systolic (
        .clk       (clk),
        .rst_n     (rst_n),
        .clear_acc (gm_sa_clear),
        .en        (gm_sa_en),
        .a_col     (gm_sa_a_col),
        .b_row     (gm_sa_b_row),
        .acc_out   (gm_sa_acc),
        .acc_valid ()
    );

    // ACC SRAM (32-bit x 256 entries)
    sram_dp #(.DEPTH(256), .WIDTH(32)) u_acc_sram (
        .clk    (clk),
        .en_a   (gm_acc_rd_en),
        .we_a   (1'b0),
        .addr_a (gm_acc_rd_addr),
        .din_a  (32'd0),
        .dout_a (gm_acc_rd_data),
        .en_b   (gm_acc_wr_en),
        .we_b   (gm_acc_wr_en),
        .addr_b (gm_acc_wr_addr),
        .din_b  (gm_acc_wr_data),
        .dout_b ()
    );

    // ================================================================
    // Softmax Engine
    // ================================================================
    logic        sm_busy, sm_done;
    logic        sm_rd_en;
    logic [15:0] sm_rd_addr;
    logic [7:0]  sm_rd_data;
    logic        sm_wr_en;
    logic [15:0] sm_wr_addr;
    logic [7:0]  sm_wr_data;
    logic        sm_scr_wr_en;
    logic [15:0] sm_scr_wr_addr, sm_scr_wr_data;
    logic        sm_scr_rd_en;
    logic [15:0] sm_scr_rd_addr, sm_scr_rd_data;

    logic        gp_sm_cmd_valid;
    logic [15:0] gp_sm_src_base, gp_sm_dst_base, gp_sm_length;

    softmax_engine u_softmax (
        .clk             (clk),
        .rst_n           (rst_n),
        .cmd_valid       (gp_sm_cmd_valid),
        .cmd_ready       (),
        .length          (gp_sm_length),
        .src_base        (gp_sm_src_base),
        .dst_base        (gp_sm_dst_base),
        .scale_factor    (16'd256),
        .causal_mask_en  (1'b0),
        .causal_limit    (16'd0),
        .sram_rd_en      (sm_rd_en),
        .sram_rd_addr    (sm_rd_addr),
        .sram_rd_data    (sm_rd_data),
        .sram_wr_en      (sm_wr_en),
        .sram_wr_addr    (sm_wr_addr),
        .sram_wr_data    (sm_wr_data),
        .scratch_wr_en   (sm_scr_wr_en),
        .scratch_wr_addr (sm_scr_wr_addr),
        .scratch_wr_data (sm_scr_wr_data),
        .scratch_rd_en   (sm_scr_rd_en),
        .scratch_rd_addr (sm_scr_rd_addr),
        .scratch_rd_data (sm_scr_rd_data),
        .busy            (sm_busy),
        .done            (sm_done)
    );

    // SCRATCH SRAM (16-bit x SCRATCH_DEPTH)
    sram_dp #(.DEPTH(SCRATCH_DEPTH), .WIDTH(16)) u_scratch_sram (
        .clk    (clk),
        .en_a   (sm_scr_rd_en),
        .we_a   (1'b0),
        .addr_a (sm_scr_rd_addr[SCR_AW-1:0]),
        .din_a  (16'd0),
        .dout_a (sm_scr_rd_data),
        .en_b   (sm_scr_wr_en),
        .we_b   (sm_scr_wr_en),
        .addr_b (sm_scr_wr_addr[SCR_AW-1:0]),
        .din_b  (sm_scr_wr_data),
        .dout_b ()
    );

    // ================================================================
    // Phase 3 Engines
    // ================================================================

    // --- Reduce Engine ---
    logic        re_busy, re_done;
    logic        re_rd_en, re_wr_en;
    logic [SRAM0_AW-1:0] re_rd_addr, re_wr_addr;
    logic [7:0]  re_wr_data;

    logic        gp_re_cmd_valid;
    logic [7:0]  gp_re_cmd_opcode;
    logic [15:0] gp_re_cmd_src_base, gp_re_cmd_dst_base;
    logic [15:0] gp_re_cmd_reduce_dim, gp_re_cmd_outer_count;

    reduce_engine #(.SRAM0_AW(SRAM0_AW)) u_reduce (
        .clk            (clk),
        .rst_n          (rst_n),
        .cmd_valid      (gp_re_cmd_valid),
        .cmd_opcode     (gp_re_cmd_opcode),
        .cmd_src_base   (gp_re_cmd_src_base),
        .cmd_dst_base   (gp_re_cmd_dst_base),
        .cmd_reduce_dim (gp_re_cmd_reduce_dim),
        .cmd_outer_count(gp_re_cmd_outer_count),
        .sram_rd_en     (re_rd_en),
        .sram_rd_addr   (re_rd_addr),
        .sram_rd_data   (s0_a_dout),
        .sram_wr_en     (re_wr_en),
        .sram_wr_addr   (re_wr_addr),
        .sram_wr_data   (re_wr_data),
        .busy           (re_busy),
        .done           (re_done)
    );

    // --- Math Engine ---
    logic        me_busy, me_done;
    logic        me_rd_en, me_wr_en;
    logic [SRAM0_AW-1:0] me_rd_addr, me_wr_addr;
    logic [7:0]  me_wr_data;

    logic        gp_me_cmd_valid;
    logic [7:0]  gp_me_cmd_opcode;
    logic [15:0] gp_me_cmd_src_base, gp_me_cmd_dst_base, gp_me_cmd_length;

    math_engine #(.SRAM0_AW(SRAM0_AW)) u_math (
        .clk          (clk),
        .rst_n        (rst_n),
        .cmd_valid    (gp_me_cmd_valid),
        .cmd_opcode   (gp_me_cmd_opcode),
        .cmd_src_base (gp_me_cmd_src_base),
        .cmd_dst_base (gp_me_cmd_dst_base),
        .cmd_length   (gp_me_cmd_length),
        .sram_rd_en   (me_rd_en),
        .sram_rd_addr (me_rd_addr),
        .sram_rd_data (s0_a_dout),
        .sram_wr_en   (me_wr_en),
        .sram_wr_addr (me_wr_addr),
        .sram_wr_data (me_wr_data),
        .busy         (me_busy),
        .done         (me_done)
    );

    // --- Gather Engine ---
    logic        ga_busy, ga_done;
    logic        ga_rd_en, ga_wr_en;
    logic [SRAM0_AW-1:0] ga_rd_addr, ga_wr_addr;
    logic [7:0]  ga_wr_data;

    logic        gp_ga_cmd_valid;
    logic [15:0] gp_ga_cmd_src_base, gp_ga_cmd_idx_base, gp_ga_cmd_dst_base;
    logic [15:0] gp_ga_cmd_num_indices, gp_ga_cmd_row_size, gp_ga_cmd_num_rows;

    gather_engine #(.SRAM0_AW(SRAM0_AW)) u_gather (
        .clk            (clk),
        .rst_n          (rst_n),
        .cmd_valid      (gp_ga_cmd_valid),
        .cmd_src_base   (gp_ga_cmd_src_base),
        .cmd_idx_base   (gp_ga_cmd_idx_base),
        .cmd_dst_base   (gp_ga_cmd_dst_base),
        .cmd_num_indices(gp_ga_cmd_num_indices),
        .cmd_row_size   (gp_ga_cmd_row_size),
        .cmd_num_rows   (gp_ga_cmd_num_rows),
        .sram_rd_en     (ga_rd_en),
        .sram_rd_addr   (ga_rd_addr),
        .sram_rd_data   (s0_a_dout),
        .sram_wr_en     (ga_wr_en),
        .sram_wr_addr   (ga_wr_addr),
        .sram_wr_data   (ga_wr_data),
        .busy           (ga_busy),
        .done           (ga_done)
    );

    // --- Slice Engine ---
    logic        sl_busy, sl_done;
    logic        sl_rd_en, sl_wr_en;
    logic [SRAM0_AW-1:0] sl_rd_addr, sl_wr_addr;
    logic [7:0]  sl_wr_data;

    logic        gp_sl_cmd_valid;
    logic [15:0] gp_sl_cmd_src_base, gp_sl_cmd_dst_base;
    logic [15:0] gp_sl_cmd_src_row_len, gp_sl_cmd_dst_row_len;
    logic [15:0] gp_sl_cmd_start_offset, gp_sl_cmd_num_rows;

    slice_engine #(.SRAM0_AW(SRAM0_AW)) u_slice (
        .clk             (clk),
        .rst_n           (rst_n),
        .cmd_valid       (gp_sl_cmd_valid),
        .cmd_src_base    (gp_sl_cmd_src_base),
        .cmd_dst_base    (gp_sl_cmd_dst_base),
        .cmd_src_row_len (gp_sl_cmd_src_row_len),
        .cmd_dst_row_len (gp_sl_cmd_dst_row_len),
        .cmd_start_offset(gp_sl_cmd_start_offset),
        .cmd_num_rows    (gp_sl_cmd_num_rows),
        .sram_rd_en      (sl_rd_en),
        .sram_rd_addr    (sl_rd_addr),
        .sram_rd_data    (s0_a_dout),
        .sram_wr_en      (sl_wr_en),
        .sram_wr_addr    (sl_wr_addr),
        .sram_wr_data    (sl_wr_data),
        .busy            (sl_busy),
        .done            (sl_done)
    );

    // --- Concat Engine ---
    logic        ct_busy, ct_done;
    logic        ct_rd_en, ct_wr_en;
    logic [SRAM0_AW-1:0] ct_rd_addr, ct_wr_addr;
    logic [7:0]  ct_wr_data;

    logic        gp_ct_cmd_valid;
    logic [15:0] gp_ct_cmd_src0_base, gp_ct_cmd_src1_base, gp_ct_cmd_dst_base;
    logic [15:0] gp_ct_cmd_src0_row_len, gp_ct_cmd_src1_row_len, gp_ct_cmd_num_rows;

    concat_engine #(.SRAM0_AW(SRAM0_AW)) u_concat (
        .clk              (clk),
        .rst_n            (rst_n),
        .cmd_valid        (gp_ct_cmd_valid),
        .cmd_src0_base    (gp_ct_cmd_src0_base),
        .cmd_src1_base    (gp_ct_cmd_src1_base),
        .cmd_dst_base     (gp_ct_cmd_dst_base),
        .cmd_src0_row_len (gp_ct_cmd_src0_row_len),
        .cmd_src1_row_len (gp_ct_cmd_src1_row_len),
        .cmd_num_rows     (gp_ct_cmd_num_rows),
        .sram_rd_en       (ct_rd_en),
        .sram_rd_addr     (ct_rd_addr),
        .sram_rd_data     (s0_a_dout),
        .sram_wr_en       (ct_wr_en),
        .sram_wr_addr     (ct_wr_addr),
        .sram_wr_data     (ct_wr_data),
        .busy             (ct_busy),
        .done             (ct_done)
    );

    // --- AvgPool2D Engine ---
    logic        ap_rd_en;
    logic [SRAM0_AW-1:0] ap_rd_addr;
    logic        ap_wr_en;
    logic [SRAM0_AW-1:0] ap_wr_addr;
    logic [7:0]  ap_wr_data;
    logic        ap_done;

    logic        ap_cmd_valid;
    logic [15:0] ap_cmd_src_base, ap_cmd_dst_base;
    logic [15:0] ap_cmd_C, ap_cmd_H, ap_cmd_W;
    logic [7:0]  ap_cmd_kh, ap_cmd_kw, ap_cmd_sh, ap_cmd_sw;

    avgpool2d_engine #(.SRAM0_AW(SRAM0_AW)) u_avgpool2d (
        .clk           (clk),
        .rst_n         (rst_n),
        .cmd_valid     (ap_cmd_valid),
        .cmd_src_base  (ap_cmd_src_base),
        .cmd_dst_base  (ap_cmd_dst_base),
        .cmd_C         (ap_cmd_C),
        .cmd_H         (ap_cmd_H),
        .cmd_W         (ap_cmd_W),
        .cmd_kh        (ap_cmd_kh),
        .cmd_kw        (ap_cmd_kw),
        .cmd_sh        (ap_cmd_sh),
        .cmd_sw        (ap_cmd_sw),
        .sram_rd_en    (ap_rd_en),
        .sram_rd_addr  (ap_rd_addr),
        .sram_rd_data  (s0_a_dout),
        .sram_wr_en    (ap_wr_en),
        .sram_wr_addr  (ap_wr_addr),
        .sram_wr_data  (ap_wr_data),
        .busy          (),
        .done          (ap_done)
    );

    // ================================================================
    // Graph Pipeline (graph_top)
    // ================================================================
    logic        gp_ew_rd_en, gp_ew_wr_en;
    logic [SRAM0_AW-1:0] gp_ew_rd_addr, gp_ew_wr_addr;
    logic [7:0]  gp_ew_wr_data;
    logic        gp_ew_busy;

    logic        gp_dma_cmd_valid;
    logic [31:0] gp_dma_ddr_addr;
    logic [15:0] gp_dma_sram_addr, gp_dma_length;
    logic        gp_dma_direction, gp_dma_strided;
    logic [31:0] gp_dma_stride;
    logic [15:0] gp_dma_count, gp_dma_block_len;

    logic        gp_done, gp_busy;
    logic [31:0] gp_status;
    logic [15:0] gp_pc;
    logic [7:0]  gp_last_op;

    // Perf counter wires
    logic [31:0] gp_perf_total, gp_perf_gemm, gp_perf_softmax, gp_perf_dma;
    logic [31:0] gp_perf_reduce, gp_perf_math, gp_perf_gather;
    logic [31:0] gp_perf_slice, gp_perf_concat, gp_perf_avgpool, gp_perf_ew;

    graph_top #(
        .PROG_SRAM_AW (PROG_AW),
        .SRAM0_AW     (SRAM0_AW)
    ) u_graph_top (
        .clk           (clk),
        .rst_n         (rst_n),
        .start         (start_pulse),
        .prog_len      (prog_len),
        .prog_rd_en    (prog_rd_en),
        .prog_rd_addr  (prog_rd_addr),
        .prog_rd_data  (prog_rd_data),
        .td_rd0_addr   (td_rd0_addr),
        .td_rd0_data   (td_rd0_data),
        .td_rd1_addr   (td_rd1_addr),
        .td_rd1_data   (td_rd1_data),
        .td_rd2_addr   (td_rd2_addr),
        .td_rd2_data   (td_rd2_data),
        // GEMM
        .gm_cmd_valid  (gp_gm_cmd_valid),
        .gm_cmd_src0   (gp_gm_cmd_src0),
        .gm_cmd_src1   (gp_gm_cmd_src1),
        .gm_cmd_dst    (gp_gm_cmd_dst),
        .gm_cmd_M      (gp_gm_cmd_M),
        .gm_cmd_N      (gp_gm_cmd_N),
        .gm_cmd_K      (gp_gm_cmd_K),
        .gm_cmd_flags  (gp_gm_cmd_flags),
        .gm_cmd_imm    (gp_gm_cmd_imm),
        .gm_done       (gm_done),
        // Softmax
        .sm_cmd_valid  (gp_sm_cmd_valid),
        .sm_src_base   (gp_sm_src_base),
        .sm_dst_base   (gp_sm_dst_base),
        .sm_length     (gp_sm_length),
        .sm_done       (sm_done),
        // Reduce
        .re_cmd_valid      (gp_re_cmd_valid),
        .re_cmd_opcode     (gp_re_cmd_opcode),
        .re_cmd_src_base   (gp_re_cmd_src_base),
        .re_cmd_dst_base   (gp_re_cmd_dst_base),
        .re_cmd_reduce_dim (gp_re_cmd_reduce_dim),
        .re_cmd_outer_count(gp_re_cmd_outer_count),
        .re_done           (re_done),
        // Math
        .me_cmd_valid  (gp_me_cmd_valid),
        .me_cmd_opcode (gp_me_cmd_opcode),
        .me_cmd_src_base(gp_me_cmd_src_base),
        .me_cmd_dst_base(gp_me_cmd_dst_base),
        .me_cmd_length (gp_me_cmd_length),
        .me_done       (me_done),
        // Gather
        .ga_cmd_valid      (gp_ga_cmd_valid),
        .ga_cmd_src_base   (gp_ga_cmd_src_base),
        .ga_cmd_idx_base   (gp_ga_cmd_idx_base),
        .ga_cmd_dst_base   (gp_ga_cmd_dst_base),
        .ga_cmd_num_indices(gp_ga_cmd_num_indices),
        .ga_cmd_row_size   (gp_ga_cmd_row_size),
        .ga_cmd_num_rows   (gp_ga_cmd_num_rows),
        .ga_done           (ga_done),
        // Slice
        .sl_cmd_valid      (gp_sl_cmd_valid),
        .sl_cmd_src_base   (gp_sl_cmd_src_base),
        .sl_cmd_dst_base   (gp_sl_cmd_dst_base),
        .sl_cmd_src_row_len(gp_sl_cmd_src_row_len),
        .sl_cmd_dst_row_len(gp_sl_cmd_dst_row_len),
        .sl_cmd_start_offset(gp_sl_cmd_start_offset),
        .sl_cmd_num_rows   (gp_sl_cmd_num_rows),
        .sl_done           (sl_done),
        // Concat
        .ct_cmd_valid      (gp_ct_cmd_valid),
        .ct_cmd_src0_base  (gp_ct_cmd_src0_base),
        .ct_cmd_src1_base  (gp_ct_cmd_src1_base),
        .ct_cmd_dst_base   (gp_ct_cmd_dst_base),
        .ct_cmd_src0_row_len(gp_ct_cmd_src0_row_len),
        .ct_cmd_src1_row_len(gp_ct_cmd_src1_row_len),
        .ct_cmd_num_rows   (gp_ct_cmd_num_rows),
        .ct_done           (ct_done),
        // AvgPool2D
        .ap_cmd_valid     (ap_cmd_valid),
        .ap_cmd_src_base  (ap_cmd_src_base),
        .ap_cmd_dst_base  (ap_cmd_dst_base),
        .ap_cmd_C         (ap_cmd_C),
        .ap_cmd_H         (ap_cmd_H),
        .ap_cmd_W         (ap_cmd_W),
        .ap_cmd_kh        (ap_cmd_kh),
        .ap_cmd_kw        (ap_cmd_kw),
        .ap_cmd_sh        (ap_cmd_sh),
        .ap_cmd_sw        (ap_cmd_sw),
        .ap_done          (ap_done),
        .perf_avgpool_cycles (gp_perf_avgpool),
        // DMA
        .dma_cmd_valid (gp_dma_cmd_valid),
        .dma_ddr_addr  (gp_dma_ddr_addr),
        .dma_sram_addr (gp_dma_sram_addr),
        .dma_length    (gp_dma_length),
        .dma_direction (gp_dma_direction),
        .dma_strided   (gp_dma_strided),
        .dma_stride    (gp_dma_stride),
        .dma_count     (gp_dma_count),
        .dma_block_len (gp_dma_block_len),
        .dma_done      (dma_done_internal),
        // EW SRAM
        .ew_rd_en      (gp_ew_rd_en),
        .ew_rd_addr    (gp_ew_rd_addr),
        .ew_rd_data    (s0_a_dout),
        .ew_wr_en      (gp_ew_wr_en),
        .ew_wr_addr    (gp_ew_wr_addr),
        .ew_wr_data    (gp_ew_wr_data),
        .ew_busy       (gp_ew_busy),
        // Perf counters
        .perf_total_cycles  (gp_perf_total),
        .perf_gemm_cycles   (gp_perf_gemm),
        .perf_softmax_cycles(gp_perf_softmax),
        .perf_dma_cycles    (gp_perf_dma),
        .perf_reduce_cycles (gp_perf_reduce),
        .perf_math_cycles   (gp_perf_math),
        .perf_gather_cycles (gp_perf_gather),
        .perf_slice_cycles  (gp_perf_slice),
        .perf_concat_cycles (gp_perf_concat),
        .perf_ew_cycles     (gp_perf_ew),
        // Status
        .graph_done    (gp_done),
        .graph_busy    (gp_busy),
        .graph_status  (gp_status),
        .graph_pc      (gp_pc),
        .graph_last_op (gp_last_op)
    );

    // ================================================================
    // DMA Shim - capture DMA commands for C++ TB handling
    // ================================================================
    logic        dma_active;
    logic        dma_captured_r;
    logic [31:0] dma_ddr_addr_r;
    logic [15:0] dma_sram_addr_r, dma_length_r;
    logic        dma_direction_r, dma_strided_r;
    logic [31:0] dma_stride_r;
    logic [15:0] dma_count_r, dma_block_len_r;
    logic        dma_done_internal;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            dma_active      <= 1'b0;
            dma_captured_r  <= 1'b0;
            dma_ddr_addr_r  <= '0;
            dma_sram_addr_r <= '0;
            dma_length_r    <= '0;
            dma_direction_r <= 1'b0;
            dma_strided_r   <= 1'b0;
            dma_stride_r    <= '0;
            dma_count_r     <= '0;
            dma_block_len_r <= '0;
        end else begin
            dma_captured_r <= 1'b0;

            if (gp_dma_cmd_valid && !dma_active) begin
                dma_active      <= 1'b1;
                dma_captured_r  <= 1'b1;
                dma_ddr_addr_r  <= gp_dma_ddr_addr;
                dma_sram_addr_r <= gp_dma_sram_addr;
                dma_length_r    <= gp_dma_length;
                dma_direction_r <= gp_dma_direction;
                dma_strided_r   <= gp_dma_strided;
                dma_stride_r    <= gp_dma_stride;
                dma_count_r     <= gp_dma_count;
                dma_block_len_r <= gp_dma_block_len;
            end else if (dma_done_pulse && dma_active) begin
                dma_active <= 1'b0;
            end
        end
    end

    assign dma_done_internal = dma_done_pulse && dma_active;
    assign dma_cmd_captured  = dma_captured_r;
    assign dma_ddr_addr      = dma_ddr_addr_r;
    assign dma_sram_addr     = dma_sram_addr_r;
    assign dma_length        = dma_length_r;
    assign dma_direction     = dma_direction_r;
    assign dma_strided        = dma_strided_r;
    assign dma_stride        = dma_stride_r;
    assign dma_count         = dma_count_r;
    assign dma_block_len     = dma_block_len_r;

    // ================================================================
    // DATA SRAM0 (8-bit x SRAM0_DEPTH)
    // Mux priority: gp_ew > gm > sm > re > me > ga > sl > ct > ap > tb
    // ================================================================
    logic                 s0_a_en;
    logic [SRAM0_AW-1:0] s0_a_addr;
    logic [7:0]           s0_a_dout;
    logic                 s0_b_en, s0_b_we;
    logic [SRAM0_AW-1:0] s0_b_addr;
    logic [7:0]           s0_b_din;

    // SRAM0 read mux (port A)
    always_comb begin
        if (gp_ew_busy && gp_ew_rd_en) begin
            s0_a_en   = 1'b1;
            s0_a_addr = gp_ew_rd_addr;
        end else if (gm_busy) begin
            s0_a_en   = gm_rd_en;
            s0_a_addr = gm_rd_addr[SRAM0_AW-1:0];
        end else if (sm_busy) begin
            s0_a_en   = sm_rd_en;
            s0_a_addr = sm_rd_addr[SRAM0_AW-1:0];
        end else if (re_busy) begin
            s0_a_en   = re_rd_en;
            s0_a_addr = re_rd_addr;
        end else if (me_busy) begin
            s0_a_en   = me_rd_en;
            s0_a_addr = me_rd_addr;
        end else if (ga_busy) begin
            s0_a_en   = ga_rd_en;
            s0_a_addr = ga_rd_addr;
        end else if (sl_busy) begin
            s0_a_en   = sl_rd_en;
            s0_a_addr = sl_rd_addr;
        end else if (ct_busy) begin
            s0_a_en   = ct_rd_en;
            s0_a_addr = ct_rd_addr;
        end else if (ap_rd_en) begin
            s0_a_en   = 1'b1;
            s0_a_addr = ap_rd_addr;
        end else begin
            s0_a_en   = tb_sram0_rd_en;
            s0_a_addr = tb_sram0_rd_addr;
        end
    end

    // SRAM0 write mux (port B)
    always_comb begin
        if (gp_ew_busy && gp_ew_wr_en) begin
            s0_b_en   = 1'b1;
            s0_b_we   = 1'b1;
            s0_b_addr = gp_ew_wr_addr;
            s0_b_din  = gp_ew_wr_data;
        end else if (gm_busy) begin
            s0_b_en   = gm_wr_en;
            s0_b_we   = gm_wr_en;
            s0_b_addr = gm_wr_addr[SRAM0_AW-1:0];
            s0_b_din  = gm_wr_data;
        end else if (sm_busy) begin
            s0_b_en   = sm_wr_en;
            s0_b_we   = sm_wr_en;
            s0_b_addr = sm_wr_addr[SRAM0_AW-1:0];
            s0_b_din  = sm_wr_data;
        end else if (re_busy) begin
            s0_b_en   = re_wr_en;
            s0_b_we   = re_wr_en;
            s0_b_addr = re_wr_addr;
            s0_b_din  = re_wr_data;
        end else if (me_busy) begin
            s0_b_en   = me_wr_en;
            s0_b_we   = me_wr_en;
            s0_b_addr = me_wr_addr;
            s0_b_din  = me_wr_data;
        end else if (ga_busy) begin
            s0_b_en   = ga_wr_en;
            s0_b_we   = ga_wr_en;
            s0_b_addr = ga_wr_addr;
            s0_b_din  = ga_wr_data;
        end else if (sl_busy) begin
            s0_b_en   = sl_wr_en;
            s0_b_we   = sl_wr_en;
            s0_b_addr = sl_wr_addr;
            s0_b_din  = sl_wr_data;
        end else if (ct_busy) begin
            s0_b_en   = ct_wr_en;
            s0_b_we   = ct_wr_en;
            s0_b_addr = ct_wr_addr;
            s0_b_din  = ct_wr_data;
        end else if (ap_wr_en) begin
            s0_b_en   = 1'b1;
            s0_b_we   = 1'b1;
            s0_b_addr = ap_wr_addr;
            s0_b_din  = ap_wr_data;
        end else begin
            s0_b_en   = tb_sram0_wr_en;
            s0_b_we   = tb_sram0_wr_en;
            s0_b_addr = tb_sram0_wr_addr;
            s0_b_din  = tb_sram0_wr_data;
        end
    end

    sram_dp #(.DEPTH(SRAM0_DEPTH), .WIDTH(8)) u_sram0 (
        .clk    (clk),
        .en_a   (s0_a_en),
        .we_a   (1'b0),
        .addr_a (s0_a_addr),
        .din_a  (8'd0),
        .dout_a (s0_a_dout),
        .en_b   (s0_b_en),
        .we_b   (s0_b_we),
        .addr_b (s0_b_addr),
        .din_b  (s0_b_din),
        .dout_b ()
    );

    // Broadcast SRAM0 read data
    assign sm_rd_data       = s0_a_dout;
    assign tb_sram0_rd_data = s0_a_dout;

    // ================================================================
    // Output assignments
    // ================================================================
    assign graph_done    = gp_done;
    assign graph_busy    = gp_busy;
    assign graph_status  = gp_status;
    assign graph_pc      = gp_pc;
    assign graph_last_op = gp_last_op;

    // Performance counters
    assign perf_total_cycles   = gp_perf_total;
    assign perf_gemm_cycles    = gp_perf_gemm;
    assign perf_softmax_cycles = gp_perf_softmax;
    assign perf_dma_cycles     = gp_perf_dma;
    assign perf_reduce_cycles  = gp_perf_reduce;
    assign perf_math_cycles    = gp_perf_math;
    assign perf_gather_cycles  = gp_perf_gather;
    assign perf_slice_cycles   = gp_perf_slice;
    assign perf_concat_cycles  = gp_perf_concat;
    assign perf_avgpool_cycles = gp_perf_avgpool;
    assign perf_ew_cycles      = gp_perf_ew;

endmodule

`default_nettype wire
