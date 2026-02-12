// =============================================================================
// llama_block_top.sv - Integration wrapper for LLaMA transformer block
// Wires control plane (fetch/decode/scoreboard/barrier) to 8 engines:
//   GEMM, softmax, layernorm, gelu+silu, vec, DMA shim, rmsnorm, rope
// Supports GQA attention with RoPE and SwiGLU FFN.
// =============================================================================
`default_nettype none

module llama_block_top
    import npu_pkg::*;
    import isa_pkg::*;
#(
    parameter int SRAM0_DEPTH   = 65536,
    parameter int SRAM1_DEPTH   = 4096,
    parameter int SCRATCH_DEPTH = 4096,
    parameter int UCODE_DEPTH_P = 1024,
    parameter int SRAM0_AW      = $clog2(SRAM0_DEPTH),
    parameter int SRAM1_AW      = $clog2(SRAM1_DEPTH),
    parameter int SCR_AW        = $clog2(SCRATCH_DEPTH),
    parameter int UC_ADDR_W     = $clog2(UCODE_DEPTH_P)
)(
    input  wire                clk,
    input  wire                rst_n,

    // --- Control ---
    input  wire                start_pulse,
    input  wire  [15:0]        ucode_len,
    output wire                program_end,

    // --- UCODE SRAM write (TB loads microcode via port B) ---
    input  wire                uc_wr_en,
    input  wire  [UC_ADDR_W-1:0] uc_wr_addr,
    input  wire  [127:0]       uc_wr_data,

    // --- DATA SRAM0 TB access ---
    input  wire                tb_sram0_wr_en,
    input  wire  [SRAM0_AW-1:0] tb_sram0_wr_addr,
    input  wire  [7:0]         tb_sram0_wr_data,
    input  wire                tb_sram0_rd_en,
    input  wire  [SRAM0_AW-1:0] tb_sram0_rd_addr,
    output wire  [7:0]         tb_sram0_rd_data,

    // --- DATA SRAM1 TB access ---
    input  wire                tb_sram1_wr_en,
    input  wire  [SRAM1_AW-1:0] tb_sram1_wr_addr,
    input  wire  [7:0]         tb_sram1_wr_data,
    input  wire                tb_sram1_rd_en,
    input  wire  [SRAM1_AW-1:0] tb_sram1_rd_addr,
    output wire  [7:0]         tb_sram1_rd_data,

    // --- DMA command capture (exposed to C++) ---
    output wire                dma_cmd_captured,
    output wire  [15:0]        dma_src,
    output wire  [15:0]        dma_dst,
    output wire  [15:0]        dma_len,
    output wire  [7:0]         dma_flags,
    input  wire                dma_done_pulse,

    // --- Debug ---
    output wire  [7:0]         engine_busy_dbg,
    output wire                all_idle_dbg,
    output wire  [4:0]         hw_gemm_done_count,
    output wire                softmax_done_dbg,
    output wire                layernorm_done_dbg,
    output wire                gelu_done_dbg,
    output wire                vec_busy_dbg,
    output wire                vec_done_dbg,
    output wire                rmsnorm_done_dbg,
    output wire                rope_done_dbg
);

    // ================================================================
    // UCODE SRAM (128-bit x UCODE_DEPTH_P)
    // ================================================================
    logic                   uc_rd_en;
    logic [UC_ADDR_W-1:0]  uc_rd_addr;
    logic [127:0]           uc_rd_data;

    sram_dp #(.DEPTH(UCODE_DEPTH_P), .WIDTH(128)) u_ucode_sram (
        .clk    (clk),
        .en_a   (uc_rd_en),
        .we_a   (1'b0),
        .addr_a (uc_rd_addr),
        .din_a  (128'd0),
        .dout_a (uc_rd_data),
        .en_b   (uc_wr_en),
        .we_b   (uc_wr_en),
        .addr_b (uc_wr_addr),
        .din_b  (uc_wr_data),
        .dout_b ()
    );

    // ================================================================
    // Microcode Fetch
    // ================================================================
    logic              fetch_instr_valid;
    logic              fetch_instr_ready;
    logic [127:0]      fetch_instr_data;
    logic              fetch_done;
    logic              fetch_busy;
    logic [UC_ADDR_W-1:0] fetch_pc;
    logic [UC_ADDR_W-1:0] fetch_rd_addr;
    assign uc_rd_addr = fetch_rd_addr;

    logic rd_valid_q;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) rd_valid_q <= 1'b0;
        else        rd_valid_q <= uc_rd_en;
    end

    ucode_fetch #(
        .INSTR_W     (128),
        .SRAM_ADDR_W (UC_ADDR_W)
    ) u_fetch (
        .clk             (clk),
        .rst_n           (rst_n),
        .start           (start_pulse),
        .stop            (1'b0),
        .ucode_base_addr ({UC_ADDR_W{1'b0}}),
        .ucode_len       (ucode_len[UC_ADDR_W-1:0]),
        .rd_en           (uc_rd_en),
        .rd_addr         (fetch_rd_addr),
        .rd_data         (uc_rd_data),
        .rd_valid        (rd_valid_q),
        .instr_valid     (fetch_instr_valid),
        .instr_ready     (fetch_instr_ready),
        .instr_data      (fetch_instr_data),
        .pc              (fetch_pc),
        .done            (fetch_done),
        .busy            (fetch_busy)
    );

    // ================================================================
    // Scoreboard (8 engines)
    // ================================================================
    logic [7:0]  engine_done_vec;
    logic [7:0]  engine_busy_vec;
    logic [7:0]  can_issue_vec;
    logic        all_idle;
    logic        issue_valid;
    logic [2:0]  issue_engine_id;

    scoreboard #(.NUM_ENGINES(8)) u_scoreboard (
        .clk             (clk),
        .rst_n           (rst_n),
        .issue_valid     (issue_valid),
        .issue_engine_id (issue_engine_id),
        .engine_done     (engine_done_vec),
        .engine_busy     (engine_busy_vec),
        .can_issue       (can_issue_vec),
        .all_idle        (all_idle)
    );

    // ================================================================
    // Barrier
    // ================================================================
    logic barrier_trigger;
    logic barrier_stall;
    logic barrier_done;

    barrier u_barrier (
        .clk      (clk),
        .rst_n    (rst_n),
        .trigger  (barrier_trigger),
        .all_idle (all_idle),
        .stall    (barrier_stall),
        .done     (barrier_done)
    );

    // ================================================================
    // Microcode Decode
    // ================================================================
    logic        gemm_cmd_valid_dec;
    logic [15:0] gemm_cmd_src0_dec, gemm_cmd_src1_dec, gemm_cmd_dst_dec;
    logic [15:0] gemm_cmd_m_dec, gemm_cmd_n_dec, gemm_cmd_k_dec;
    logic [7:0]  gemm_cmd_flags_dec;

    logic        softmax_cmd_valid_dec;
    logic [15:0] softmax_cmd_src0_dec, softmax_cmd_dst_dec, softmax_cmd_len_dec;
    logic [7:0]  softmax_cmd_flags_dec;

    logic        layernorm_cmd_valid_dec;
    logic [15:0] layernorm_cmd_src0_dec, layernorm_cmd_dst_dec, layernorm_cmd_len_dec;
    logic [7:0]  layernorm_cmd_flags_dec;

    logic        gelu_cmd_valid_dec;
    logic [15:0] gelu_cmd_src0_dec, gelu_cmd_dst_dec, gelu_cmd_len_dec;
    logic [7:0]  gelu_cmd_flags_dec;

    logic        vec_cmd_valid_dec;
    logic [15:0] vec_cmd_src0_dec, vec_cmd_src1_dec, vec_cmd_dst_dec;
    logic [15:0] vec_cmd_len_dec, vec_cmd_imm_dec;
    logic [7:0]  vec_cmd_flags_dec;

    logic        dma_rd_cmd_valid_dec;
    logic [15:0] dma_rd_cmd_src_dec, dma_rd_cmd_dst_dec, dma_rd_cmd_len_dec;
    logic [7:0]  dma_rd_cmd_flags_dec;

    logic        dma_wr_cmd_valid_dec;
    logic [15:0] dma_wr_cmd_src_dec, dma_wr_cmd_dst_dec, dma_wr_cmd_len_dec;
    logic [7:0]  dma_wr_cmd_flags_dec;

    logic        kv_cmd_valid_dec;
    logic [7:0]  kv_cmd_opcode_dec;
    logic [15:0] kv_cmd_src0_dec, kv_cmd_dst_dec, kv_cmd_len_dec, kv_cmd_imm_dec;
    logic [7:0]  kv_cmd_flags_dec;

    // RMSNorm decode outputs
    logic        rmsnorm_cmd_valid_dec;
    logic [15:0] rmsnorm_cmd_src0_dec, rmsnorm_cmd_dst_dec, rmsnorm_cmd_len_dec;
    logic [15:0] rmsnorm_cmd_gamma_dec;

    // RoPE decode outputs
    logic        rope_cmd_valid_dec;
    logic [15:0] rope_cmd_src0_dec, rope_cmd_dst_dec;
    logic [15:0] rope_cmd_num_rows_dec, rope_cmd_head_dim_dec;
    logic [15:0] rope_cmd_pos_offset_dec, rope_cmd_sin_base_dec, rope_cmd_cos_base_dec;

    // SiLU mode
    logic        silu_mode_dec;

    logic        program_end_dec;
    ucode_instr_t decoded_instr;

    ucode_decode u_decode (
        .clk                (clk),
        .rst_n              (rst_n),
        .instr_valid        (fetch_instr_valid),
        .instr_data         (fetch_instr_data),
        .instr_ready        (fetch_instr_ready),
        .can_issue          (can_issue_vec),
        .all_idle           (all_idle),
        .gemm_cmd_valid     (gemm_cmd_valid_dec),
        .gemm_cmd_src0      (gemm_cmd_src0_dec),
        .gemm_cmd_src1      (gemm_cmd_src1_dec),
        .gemm_cmd_dst       (gemm_cmd_dst_dec),
        .gemm_cmd_m         (gemm_cmd_m_dec),
        .gemm_cmd_n         (gemm_cmd_n_dec),
        .gemm_cmd_k         (gemm_cmd_k_dec),
        .gemm_cmd_flags     (gemm_cmd_flags_dec),
        .softmax_cmd_valid  (softmax_cmd_valid_dec),
        .softmax_cmd_src0   (softmax_cmd_src0_dec),
        .softmax_cmd_dst    (softmax_cmd_dst_dec),
        .softmax_cmd_len    (softmax_cmd_len_dec),
        .softmax_cmd_flags  (softmax_cmd_flags_dec),
        .layernorm_cmd_valid(layernorm_cmd_valid_dec),
        .layernorm_cmd_src0 (layernorm_cmd_src0_dec),
        .layernorm_cmd_dst  (layernorm_cmd_dst_dec),
        .layernorm_cmd_len  (layernorm_cmd_len_dec),
        .layernorm_cmd_flags(layernorm_cmd_flags_dec),
        .gelu_cmd_valid     (gelu_cmd_valid_dec),
        .gelu_cmd_src0      (gelu_cmd_src0_dec),
        .gelu_cmd_dst       (gelu_cmd_dst_dec),
        .gelu_cmd_len       (gelu_cmd_len_dec),
        .gelu_cmd_flags     (gelu_cmd_flags_dec),
        .vec_cmd_valid      (vec_cmd_valid_dec),
        .vec_cmd_src0       (vec_cmd_src0_dec),
        .vec_cmd_src1       (vec_cmd_src1_dec),
        .vec_cmd_dst        (vec_cmd_dst_dec),
        .vec_cmd_len        (vec_cmd_len_dec),
        .vec_cmd_flags      (vec_cmd_flags_dec),
        .vec_cmd_imm        (vec_cmd_imm_dec),
        .dma_rd_cmd_valid   (dma_rd_cmd_valid_dec),
        .dma_rd_cmd_src     (dma_rd_cmd_src_dec),
        .dma_rd_cmd_dst     (dma_rd_cmd_dst_dec),
        .dma_rd_cmd_len     (dma_rd_cmd_len_dec),
        .dma_rd_cmd_flags   (dma_rd_cmd_flags_dec),
        .dma_wr_cmd_valid   (dma_wr_cmd_valid_dec),
        .dma_wr_cmd_src     (dma_wr_cmd_src_dec),
        .dma_wr_cmd_dst     (dma_wr_cmd_dst_dec),
        .dma_wr_cmd_len     (dma_wr_cmd_len_dec),
        .dma_wr_cmd_flags   (dma_wr_cmd_flags_dec),
        .kv_cmd_valid       (kv_cmd_valid_dec),
        .kv_cmd_opcode      (kv_cmd_opcode_dec),
        .kv_cmd_src0        (kv_cmd_src0_dec),
        .kv_cmd_dst         (kv_cmd_dst_dec),
        .kv_cmd_len         (kv_cmd_len_dec),
        .kv_cmd_flags       (kv_cmd_flags_dec),
        .kv_cmd_imm         (kv_cmd_imm_dec),
        .rmsnorm_cmd_valid  (rmsnorm_cmd_valid_dec),
        .rmsnorm_cmd_src0   (rmsnorm_cmd_src0_dec),
        .rmsnorm_cmd_dst    (rmsnorm_cmd_dst_dec),
        .rmsnorm_cmd_len    (rmsnorm_cmd_len_dec),
        .rmsnorm_cmd_gamma  (rmsnorm_cmd_gamma_dec),
        .rope_cmd_valid     (rope_cmd_valid_dec),
        .rope_cmd_src0      (rope_cmd_src0_dec),
        .rope_cmd_dst       (rope_cmd_dst_dec),
        .rope_cmd_num_rows  (rope_cmd_num_rows_dec),
        .rope_cmd_head_dim  (rope_cmd_head_dim_dec),
        .rope_cmd_pos_offset(rope_cmd_pos_offset_dec),
        .rope_cmd_sin_base  (rope_cmd_sin_base_dec),
        .rope_cmd_cos_base  (rope_cmd_cos_base_dec),
        .silu_mode          (silu_mode_dec),
        .barrier_trigger    (barrier_trigger),
        .issue_valid        (issue_valid),
        .issue_engine_id    (issue_engine_id),
        .decoded_instr      (decoded_instr),
        .program_end        (program_end_dec)
    );

    // ================================================================
    // Real GEMM Engine: gemm_ctrl + systolic_array
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

    gemm_ctrl #(
        .ARRAY_M     (16),
        .ARRAY_N     (16),
        .DATA_W      (8),
        .ACC_W       (32),
        .SRAM_ADDR_W (16)
    ) u_gemm_ctrl (
        .clk          (clk),
        .rst_n        (rst_n),
        .cmd_valid    (gemm_cmd_valid_dec),
        .cmd_src0     (gemm_cmd_src0_dec),
        .cmd_src1     (gemm_cmd_src1_dec),
        .cmd_dst      (gemm_cmd_dst_dec),
        .cmd_M        (gemm_cmd_m_dec),
        .cmd_N        (gemm_cmd_n_dec),
        .cmd_K        (gemm_cmd_k_dec),
        .cmd_flags    (gemm_cmd_flags_dec),
        .cmd_imm      (decoded_instr.imm),
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

    // ================================================================
    // ACC SRAM (32-bit x 256 entries)
    // ================================================================
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

    // HW GEMM done counter
    logic [4:0] hw_gemm_cnt;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            hw_gemm_cnt <= 5'd0;
        else if (start_pulse)
            hw_gemm_cnt <= 5'd0;
        else if (gm_done)
            hw_gemm_cnt <= hw_gemm_cnt + 5'd1;
    end
    assign hw_gemm_done_count = hw_gemm_cnt;

    // ================================================================
    // DMA Shim
    // ================================================================
    logic        dma_active;
    logic [15:0] dma_src_r, dma_dst_r, dma_len_r;
    logic [7:0]  dma_flags_r;
    logic        dma_captured_r;
    logic        dma_done_internal;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            dma_active     <= 1'b0;
            dma_captured_r <= 1'b0;
            dma_src_r      <= '0;
            dma_dst_r      <= '0;
            dma_len_r      <= '0;
            dma_flags_r    <= '0;
        end else begin
            dma_captured_r <= 1'b0;

            if ((dma_rd_cmd_valid_dec || dma_wr_cmd_valid_dec) && !dma_active && !kv_busy) begin
                dma_active     <= 1'b1;
                dma_captured_r <= 1'b1;
                dma_src_r      <= dma_rd_cmd_valid_dec ? dma_rd_cmd_src_dec : dma_wr_cmd_src_dec;
                dma_dst_r      <= dma_rd_cmd_valid_dec ? dma_rd_cmd_dst_dec : dma_wr_cmd_dst_dec;
                dma_len_r      <= dma_rd_cmd_valid_dec ? dma_rd_cmd_len_dec : dma_wr_cmd_len_dec;
                dma_flags_r    <= dma_rd_cmd_valid_dec ? dma_rd_cmd_flags_dec : dma_wr_cmd_flags_dec;
            end else if (dma_done_pulse && dma_active) begin
                dma_active <= 1'b0;
            end
        end
    end

    assign dma_done_internal = dma_done_pulse && dma_active;
    assign dma_cmd_captured  = dma_captured_r;
    assign dma_src           = dma_src_r;
    assign dma_dst           = dma_dst_r;
    assign dma_len           = dma_len_r;
    assign dma_flags         = dma_flags_r;

    // ================================================================
    // Engine wires
    // ================================================================
    // Softmax
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

    // LayerNorm
    logic        ln_busy, ln_done;
    logic        ln_rd0_en;
    logic [15:0] ln_rd0_addr;
    logic [7:0]  ln_rd0_data;
    logic        ln_rd1_en;
    logic [15:0] ln_rd1_addr;
    logic [7:0]  ln_rd1_data;
    logic        ln_wr_en;
    logic [15:0] ln_wr_addr;
    logic [7:0]  ln_wr_data;

    // GELU/SiLU
    logic        ge_busy, ge_done;
    logic        ge_rd_en;
    logic [15:0] ge_rd_addr;
    logic [7:0]  ge_rd_data;
    logic        ge_wr_en;
    logic [15:0] ge_wr_addr;
    logic [7:0]  ge_wr_data;

    // Vec
    logic        ve_busy, ve_done;
    logic        ve_rd0_en;
    logic [15:0] ve_rd0_addr;
    logic [7:0]  ve_rd0_data;
    logic        ve_rd1_en;
    logic [15:0] ve_rd1_addr;
    logic [7:0]  ve_rd1_data;
    logic        ve_wr_en;
    logic [15:0] ve_wr_addr;
    logic [7:0]  ve_wr_data;

    // RMSNorm
    logic        rn_busy, rn_done;
    logic        rn_rd0_en;
    logic [15:0] rn_rd0_addr;
    logic [7:0]  rn_rd0_data;
    logic        rn_rd1_en;
    logic [15:0] rn_rd1_addr;
    logic [7:0]  rn_rd1_data;
    logic        rn_wr_en;
    logic [15:0] rn_wr_addr;
    logic [7:0]  rn_wr_data;

    // RoPE
    logic        rp_busy, rp_done;
    logic        rp_rd0_en;
    logic [15:0] rp_rd0_addr;
    logic [7:0]  rp_rd0_data;
    logic        rp_rd1_en;
    logic [15:0] rp_rd1_addr;
    logic [7:0]  rp_rd1_data;
    logic        rp_wr_en;
    logic [15:0] rp_wr_addr;
    logic [7:0]  rp_wr_data;

    // ================================================================
    // KV Cache Controller + Bank
    // ================================================================
    localparam int KV_MAX_LAYERS = 4;
    localparam int KV_MAX_HEADS  = 4;
    localparam int KV_MAX_SEQ    = 512;
    localparam int KV_HEAD_DIM   = 16;
    localparam int KV_VEC_W      = KV_HEAD_DIM * 8;

    logic        kv_busy, kv_done;
    logic        kv_rd_en;
    logic [15:0] kv_rd_addr;
    logic        kv_wr_en;
    logic [15:0] kv_wr_addr;
    logic [7:0]  kv_wr_data;

    // KV cache bank connections
    logic                                  kv_append_valid, kv_append_ready, kv_append_done;
    logic [$clog2(KV_MAX_LAYERS)-1:0]      kv_append_layer;
    logic [$clog2(KV_MAX_HEADS)-1:0]       kv_append_head;
    logic [$clog2(KV_MAX_SEQ)-1:0]         kv_append_time;
    logic                                  kv_append_is_v;
    logic [KV_VEC_W-1:0]                   kv_append_data;

    logic                                  kv_read_req_valid, kv_read_req_ready;
    logic [$clog2(KV_MAX_LAYERS)-1:0]      kv_read_layer;
    logic [$clog2(KV_MAX_HEADS)-1:0]       kv_read_head;
    logic [$clog2(KV_MAX_SEQ)-1:0]         kv_read_time_start, kv_read_time_len;
    logic                                  kv_read_is_v;
    logic                                  kv_read_data_valid;
    logic [KV_VEC_W-1:0]                   kv_read_data;
    logic                                  kv_read_data_last;

    kv_cache_bank #(
        .MAX_LAYERS (KV_MAX_LAYERS),
        .MAX_HEADS  (KV_MAX_HEADS),
        .MAX_SEQ    (KV_MAX_SEQ),
        .HEAD_DIM   (KV_HEAD_DIM)
    ) u_kv_bank (
        .clk             (clk),
        .rst_n           (rst_n),
        .append_valid    (kv_append_valid),
        .append_ready    (kv_append_ready),
        .append_layer    (kv_append_layer),
        .append_head     (kv_append_head),
        .append_time     (kv_append_time),
        .append_is_v     (kv_append_is_v),
        .append_data     (kv_append_data),
        .append_done     (kv_append_done),
        .read_req_valid  (kv_read_req_valid),
        .read_req_ready  (kv_read_req_ready),
        .read_layer      (kv_read_layer),
        .read_head       (kv_read_head),
        .read_time_start (kv_read_time_start),
        .read_time_len   (kv_read_time_len),
        .read_is_v       (kv_read_is_v),
        .read_data_valid (kv_read_data_valid),
        .read_data       (kv_read_data),
        .read_data_last  (kv_read_data_last)
    );

    kv_ctrl #(
        .MAX_LAYERS (KV_MAX_LAYERS),
        .MAX_HEADS  (KV_MAX_HEADS),
        .MAX_SEQ    (KV_MAX_SEQ),
        .HEAD_DIM   (KV_HEAD_DIM)
    ) u_kv_ctrl (
        .clk             (clk),
        .rst_n           (rst_n),
        .cmd_valid       (kv_cmd_valid_dec),
        .cmd_opcode      (kv_cmd_opcode_dec),
        .cmd_src0        (kv_cmd_src0_dec),
        .cmd_dst         (kv_cmd_dst_dec),
        .cmd_m           (decoded_instr.M),
        .cmd_n           (decoded_instr.N),
        .cmd_k           (decoded_instr.K),
        .cmd_flags       (kv_cmd_flags_dec),
        .cmd_imm         (kv_cmd_imm_dec),
        .sram_rd_en      (kv_rd_en),
        .sram_rd_addr    (kv_rd_addr),
        .sram_rd_data    (s0_a_dout),
        .sram_wr_en      (kv_wr_en),
        .sram_wr_addr    (kv_wr_addr),
        .sram_wr_data    (kv_wr_data),
        .append_valid    (kv_append_valid),
        .append_ready    (kv_append_ready),
        .append_layer    (kv_append_layer),
        .append_head     (kv_append_head),
        .append_time     (kv_append_time),
        .append_is_v     (kv_append_is_v),
        .append_data     (kv_append_data),
        .append_done     (kv_append_done),
        .read_req_valid  (kv_read_req_valid),
        .read_req_ready  (kv_read_req_ready),
        .read_layer      (kv_read_layer),
        .read_head       (kv_read_head),
        .read_time_start (kv_read_time_start),
        .read_time_len   (kv_read_time_len),
        .read_is_v       (kv_read_is_v),
        .read_data_valid (kv_read_data_valid),
        .read_data       (kv_read_data),
        .read_data_last  (kv_read_data_last),
        .busy            (kv_busy),
        .done            (kv_done)
    );

    // ================================================================
    // DATA SRAM0 (8-bit x SRAM0_DEPTH)
    // ================================================================
    logic                    s0_a_en;
    logic [SRAM0_AW-1:0]    s0_a_addr;
    logic [7:0]              s0_a_dout;
    logic                    s0_b_en, s0_b_we;
    logic [SRAM0_AW-1:0]    s0_b_addr;
    logic [7:0]              s0_b_din;

    // SRAM0 read mux (port A) - GEMM HW has highest priority
    always_comb begin
        if (gm_busy) begin
            s0_a_en   = gm_rd_en;
            s0_a_addr = gm_rd_addr[SRAM0_AW-1:0];
        end else if (sm_busy) begin
            s0_a_en   = sm_rd_en;
            s0_a_addr = sm_rd_addr[SRAM0_AW-1:0];
        end else if (ln_busy) begin
            s0_a_en   = ln_rd0_en;
            s0_a_addr = ln_rd0_addr[SRAM0_AW-1:0];
        end else if (rn_busy) begin
            s0_a_en   = rn_rd0_en;
            s0_a_addr = rn_rd0_addr[SRAM0_AW-1:0];
        end else if (rp_busy) begin
            s0_a_en   = rp_rd0_en;
            s0_a_addr = rp_rd0_addr[SRAM0_AW-1:0];
        end else if (ge_busy) begin
            s0_a_en   = ge_rd_en;
            s0_a_addr = ge_rd_addr[SRAM0_AW-1:0];
        end else if (ve_busy) begin
            s0_a_en   = ve_rd0_en;
            s0_a_addr = ve_rd0_addr[SRAM0_AW-1:0];
        end else if (kv_busy) begin
            s0_a_en   = kv_rd_en;
            s0_a_addr = kv_rd_addr[SRAM0_AW-1:0];
        end else begin
            s0_a_en   = tb_sram0_rd_en;
            s0_a_addr = tb_sram0_rd_addr;
        end
    end

    // SRAM0 write mux (port B) - GEMM HW has highest priority
    always_comb begin
        if (gm_busy) begin
            s0_b_en   = gm_wr_en;
            s0_b_we   = gm_wr_en;
            s0_b_addr = gm_wr_addr[SRAM0_AW-1:0];
            s0_b_din  = gm_wr_data;
        end else if (sm_busy) begin
            s0_b_en   = sm_wr_en;
            s0_b_we   = sm_wr_en;
            s0_b_addr = sm_wr_addr[SRAM0_AW-1:0];
            s0_b_din  = sm_wr_data;
        end else if (ln_busy) begin
            s0_b_en   = ln_wr_en;
            s0_b_we   = ln_wr_en;
            s0_b_addr = ln_wr_addr[SRAM0_AW-1:0];
            s0_b_din  = ln_wr_data;
        end else if (rn_busy) begin
            s0_b_en   = rn_wr_en;
            s0_b_we   = rn_wr_en;
            s0_b_addr = rn_wr_addr[SRAM0_AW-1:0];
            s0_b_din  = rn_wr_data;
        end else if (rp_busy) begin
            s0_b_en   = rp_wr_en;
            s0_b_we   = rp_wr_en;
            s0_b_addr = rp_wr_addr[SRAM0_AW-1:0];
            s0_b_din  = rp_wr_data;
        end else if (ge_busy) begin
            s0_b_en   = ge_wr_en;
            s0_b_we   = ge_wr_en;
            s0_b_addr = ge_wr_addr[SRAM0_AW-1:0];
            s0_b_din  = ge_wr_data;
        end else if (ve_busy) begin
            s0_b_en   = ve_wr_en;
            s0_b_we   = ve_wr_en;
            s0_b_addr = ve_wr_addr[SRAM0_AW-1:0];
            s0_b_din  = ve_wr_data;
        end else if (kv_busy) begin
            s0_b_en   = kv_wr_en;
            s0_b_we   = kv_wr_en;
            s0_b_addr = kv_wr_addr[SRAM0_AW-1:0];
            s0_b_din  = kv_wr_data;
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
    assign ln_rd0_data      = s0_a_dout;
    assign rn_rd0_data      = s0_a_dout;
    assign rp_rd0_data      = s0_a_dout;
    assign ge_rd_data       = s0_a_dout;
    assign ve_rd0_data      = s0_a_dout;
    assign tb_sram0_rd_data = s0_a_dout;

    // ================================================================
    // DATA SRAM1 (8-bit x SRAM1_DEPTH)
    // ================================================================
    logic                    s1_a_en;
    logic [SRAM1_AW-1:0]    s1_a_addr;
    logic [7:0]              s1_a_dout;

    // SRAM1 read mux (port A) - add rmsnorm and rope
    always_comb begin
        if (ln_busy) begin
            s1_a_en   = ln_rd1_en;
            s1_a_addr = ln_rd1_addr[SRAM1_AW-1:0];
        end else if (rn_busy) begin
            s1_a_en   = rn_rd1_en;
            s1_a_addr = rn_rd1_addr[SRAM1_AW-1:0];
        end else if (rp_busy) begin
            s1_a_en   = rp_rd1_en;
            s1_a_addr = rp_rd1_addr[SRAM1_AW-1:0];
        end else if (ve_busy) begin
            s1_a_en   = ve_rd1_en;
            s1_a_addr = ve_rd1_addr[SRAM1_AW-1:0];
        end else begin
            s1_a_en   = tb_sram1_rd_en;
            s1_a_addr = tb_sram1_rd_addr;
        end
    end

    sram_dp #(.DEPTH(SRAM1_DEPTH), .WIDTH(8)) u_sram1 (
        .clk    (clk),
        .en_a   (s1_a_en),
        .we_a   (1'b0),
        .addr_a (s1_a_addr),
        .din_a  (8'd0),
        .dout_a (s1_a_dout),
        .en_b   (tb_sram1_wr_en),
        .we_b   (tb_sram1_wr_en),
        .addr_b (tb_sram1_wr_addr),
        .din_b  (tb_sram1_wr_data),
        .dout_b ()
    );

    assign ln_rd1_data      = s1_a_dout;
    assign rn_rd1_data      = s1_a_dout;
    assign rp_rd1_data      = s1_a_dout;
    assign ve_rd1_data      = s1_a_dout;
    assign tb_sram1_rd_data = s1_a_dout;

    // ================================================================
    // SCRATCH SRAM (16-bit x SCRATCH_DEPTH) - softmax intermediates
    // ================================================================
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
    // Softmax Engine
    // ================================================================
    softmax_engine u_softmax (
        .clk             (clk),
        .rst_n           (rst_n),
        .cmd_valid       (softmax_cmd_valid_dec),
        .cmd_ready       (),
        .length          (softmax_cmd_len_dec),
        .src_base        (softmax_cmd_src0_dec),
        .dst_base        (softmax_cmd_dst_dec),
        .scale_factor    (decoded_instr.imm),
        .causal_mask_en  (softmax_cmd_flags_dec[FLAG_CAUSAL_MASK]),
        .causal_limit    (decoded_instr.K),
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

    // ================================================================
    // LayerNorm Engine
    // ================================================================
    layernorm_engine u_layernorm (
        .clk             (clk),
        .rst_n           (rst_n),
        .cmd_valid       (layernorm_cmd_valid_dec),
        .cmd_ready       (),
        .length          (layernorm_cmd_len_dec),
        .src_base        (layernorm_cmd_src0_dec),
        .dst_base        (layernorm_cmd_dst_dec),
        .gamma_base      (16'd0),
        .beta_base       (decoded_instr.src1_base),
        .sram_rd0_en     (ln_rd0_en),
        .sram_rd0_addr   (ln_rd0_addr),
        .sram_rd0_data   (ln_rd0_data),
        .sram_rd1_en     (ln_rd1_en),
        .sram_rd1_addr   (ln_rd1_addr),
        .sram_rd1_data   (ln_rd1_data),
        .sram_wr_en      (ln_wr_en),
        .sram_wr_addr    (ln_wr_addr),
        .sram_wr_data    (ln_wr_data),
        .busy            (ln_busy),
        .done            (ln_done)
    );

    // ================================================================
    // GELU/SiLU Engine
    // ================================================================
    gelu_engine u_gelu (
        .clk             (clk),
        .rst_n           (rst_n),
        .cmd_valid       (gelu_cmd_valid_dec),
        .cmd_ready       (),
        .length          (gelu_cmd_len_dec),
        .src_base        (gelu_cmd_src0_dec),
        .dst_base        (gelu_cmd_dst_dec),
        .silu_mode       (silu_mode_dec),
        .sram_rd_en      (ge_rd_en),
        .sram_rd_addr    (ge_rd_addr),
        .sram_rd_data    (ge_rd_data),
        .sram_wr_en      (ge_wr_en),
        .sram_wr_addr    (ge_wr_addr),
        .sram_wr_data    (ge_wr_data),
        .busy            (ge_busy),
        .done            (ge_done)
    );

    // ================================================================
    // Vec Engine
    // ================================================================
    vec_engine u_vec (
        .clk             (clk),
        .rst_n           (rst_n),
        .cmd_valid       (vec_cmd_valid_dec),
        .cmd_ready       (),
        .opcode          (vec_cmd_flags_dec[1:0]),
        .length          (vec_cmd_len_dec),
        .src0_base       (vec_cmd_src0_dec),
        .src1_base       (vec_cmd_src1_dec),
        .dst_base        (vec_cmd_dst_dec),
        .scale           (vec_cmd_imm_dec[7:0]),
        .shift           (vec_cmd_imm_dec[15:8]),
        .copy2d_mode     (vec_cmd_flags_dec[2]),
        .cmd_M           (decoded_instr.M),
        .cmd_K           (decoded_instr.K),
        .cmd_imm         (decoded_instr.imm),
        .sram_rd0_en     (ve_rd0_en),
        .sram_rd0_addr   (ve_rd0_addr),
        .sram_rd0_data   (ve_rd0_data),
        .sram_rd1_en     (ve_rd1_en),
        .sram_rd1_addr   (ve_rd1_addr),
        .sram_rd1_data   (ve_rd1_data),
        .sram_wr_en      (ve_wr_en),
        .sram_wr_addr    (ve_wr_addr),
        .sram_wr_data    (ve_wr_data),
        .busy            (ve_busy),
        .done            (ve_done)
    );

    // ================================================================
    // RMSNorm Engine
    // ================================================================
    rmsnorm_engine u_rmsnorm (
        .clk             (clk),
        .rst_n           (rst_n),
        .cmd_valid       (rmsnorm_cmd_valid_dec),
        .cmd_ready       (),
        .length          (rmsnorm_cmd_len_dec),
        .src_base        (rmsnorm_cmd_src0_dec),
        .dst_base        (rmsnorm_cmd_dst_dec),
        .gamma_base      (rmsnorm_cmd_gamma_dec),
        .sram_rd0_en     (rn_rd0_en),
        .sram_rd0_addr   (rn_rd0_addr),
        .sram_rd0_data   (rn_rd0_data),
        .sram_rd1_en     (rn_rd1_en),
        .sram_rd1_addr   (rn_rd1_addr),
        .sram_rd1_data   (rn_rd1_data),
        .sram_wr_en      (rn_wr_en),
        .sram_wr_addr    (rn_wr_addr),
        .sram_wr_data    (rn_wr_data),
        .busy            (rn_busy),
        .done            (rn_done)
    );

    // ================================================================
    // RoPE Engine
    // ================================================================
    rope_engine u_rope (
        .clk             (clk),
        .rst_n           (rst_n),
        .cmd_valid       (rope_cmd_valid_dec),
        .cmd_ready       (),
        .src_base        (rope_cmd_src0_dec),
        .dst_base        (rope_cmd_dst_dec),
        .num_rows        (rope_cmd_num_rows_dec),
        .head_dim        (rope_cmd_head_dim_dec),
        .pos_offset      (rope_cmd_pos_offset_dec),
        .sin_base        (rope_cmd_sin_base_dec),
        .cos_base        (rope_cmd_cos_base_dec),
        .sram_rd0_en     (rp_rd0_en),
        .sram_rd0_addr   (rp_rd0_addr),
        .sram_rd0_data   (rp_rd0_data),
        .sram_rd1_en     (rp_rd1_en),
        .sram_rd1_addr   (rp_rd1_addr),
        .sram_rd1_data   (rp_rd1_data),
        .sram_wr_en      (rp_wr_en),
        .sram_wr_addr    (rp_wr_addr),
        .sram_wr_data    (rp_wr_data),
        .busy            (rp_busy),
        .done            (rp_done)
    );

    // ================================================================
    // Engine done vector wiring
    // ================================================================
    assign engine_done_vec[0] = gm_done;                  // ENG_GEMM
    assign engine_done_vec[1] = sm_done;                  // ENG_SOFTMAX
    assign engine_done_vec[2] = ln_done;                  // ENG_LAYERNORM
    assign engine_done_vec[3] = ge_done;                  // ENG_GELU
    assign engine_done_vec[4] = ve_done;                  // ENG_VEC
    assign engine_done_vec[5] = dma_done_internal | kv_done; // ENG_DMA
    assign engine_done_vec[6] = rn_done;                  // ENG_RMSNORM
    assign engine_done_vec[7] = rp_done;                  // ENG_ROPE

    // ================================================================
    // program_end latch
    // ================================================================
    logic fetch_done_latch;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            fetch_done_latch <= 1'b0;
        else if (start_pulse)
            fetch_done_latch <= 1'b0;
        else if (fetch_done || program_end_dec)
            fetch_done_latch <= 1'b1;
    end

    logic program_end_latch;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            program_end_latch <= 1'b0;
        else if (start_pulse)
            program_end_latch <= 1'b0;
        else if (fetch_done_latch && all_idle)
            program_end_latch <= 1'b1;
    end
    assign program_end = program_end_latch;

    // ================================================================
    // Debug outputs
    // ================================================================
    assign engine_busy_dbg = engine_busy_vec;
    assign all_idle_dbg    = all_idle;
    assign softmax_done_dbg  = sm_done;
    assign layernorm_done_dbg = ln_done;
    assign gelu_done_dbg     = ge_done;
    assign vec_busy_dbg      = ve_busy;
    assign vec_done_dbg      = ve_done;
    assign rmsnorm_done_dbg  = rn_done;
    assign rope_done_dbg     = rp_done;

endmodule

`default_nettype wire
