// =============================================================================
// integration_top.sv - Integration wrapper for single attention head pipeline
// Wires control plane (fetch/decode/scoreboard/barrier) to real softmax engine
// and a GEMM shim that exposes commands to C++ for golden computation.
// =============================================================================
`default_nettype none

module integration_top
    import npu_pkg::*;
    import isa_pkg::*;
#(
    parameter int SRAM_DEPTH    = 4096,
    parameter int SRAM_ADDR_W   = $clog2(SRAM_DEPTH),
    parameter int UCODE_DEPTH_P = 1024,
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

    // --- DATA SRAM0 TB access (read/write when engines idle) ---
    input  wire                tb_sram_wr_en,
    input  wire  [SRAM_ADDR_W-1:0] tb_sram_wr_addr,
    input  wire  [7:0]         tb_sram_wr_data,
    input  wire                tb_sram_rd_en,
    input  wire  [SRAM_ADDR_W-1:0] tb_sram_rd_addr,
    output wire  [7:0]         tb_sram_rd_data,

    // --- GEMM command capture (exposed to C++) ---
    output wire                gemm_cmd_captured,
    output wire  [15:0]        gemm_src0,
    output wire  [15:0]        gemm_src1,
    output wire  [15:0]        gemm_dst,
    output wire  [15:0]        gemm_M,
    output wire  [15:0]        gemm_N,
    output wire  [15:0]        gemm_K,
    output wire  [7:0]         gemm_flags,
    output wire  [15:0]        gemm_imm,
    input  wire                gemm_done_pulse,

    // --- Debug ---
    output wire  [5:0]         engine_busy_dbg,
    output wire                all_idle_dbg,
    output wire                softmax_busy_dbg,
    output wire                softmax_done_dbg
);

    // ================================================================
    // UCODE SRAM (128-bit x UCODE_DEPTH_P)
    // Port A: fetch reads
    // Port B: TB writes
    // ================================================================
    logic                   uc_rd_en;
    logic [UC_ADDR_W-1:0]  uc_rd_addr;
    logic [127:0]           uc_rd_data;

    sram_dp #(.DEPTH(UCODE_DEPTH_P), .WIDTH(128)) u_ucode_sram (
        .clk    (clk),
        // Port A: fetch read
        .en_a   (uc_rd_en),
        .we_a   (1'b0),
        .addr_a (uc_rd_addr),
        .din_a  (128'd0),
        .dout_a (uc_rd_data),
        // Port B: TB write
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

    // Fetch unit uses UC_ADDR_W for its SRAM port
    logic [UC_ADDR_W-1:0] fetch_rd_addr;
    assign uc_rd_addr = fetch_rd_addr;

    // rd_valid: SRAM has 1-cycle read latency, pipeline it
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
    // Scoreboard
    // ================================================================
    logic [5:0]  engine_done_vec;
    logic [5:0]  engine_busy_vec;
    logic [5:0]  can_issue_vec;
    logic        all_idle;
    logic        issue_valid;
    logic [2:0]  issue_engine_id;

    scoreboard #(.NUM_ENGINES(6)) u_scoreboard (
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
    // Decode outputs
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

    logic        program_end_dec;
    ucode_instr_t decoded_instr;

    ucode_decode u_decode (
        .clk                (clk),
        .rst_n              (rst_n),
        // From fetch
        .instr_valid        (fetch_instr_valid),
        .instr_data         (fetch_instr_data),
        .instr_ready        (fetch_instr_ready),
        // From scoreboard
        .can_issue          (can_issue_vec),
        .all_idle           (all_idle),
        // GEMM
        .gemm_cmd_valid     (gemm_cmd_valid_dec),
        .gemm_cmd_src0      (gemm_cmd_src0_dec),
        .gemm_cmd_src1      (gemm_cmd_src1_dec),
        .gemm_cmd_dst       (gemm_cmd_dst_dec),
        .gemm_cmd_m         (gemm_cmd_m_dec),
        .gemm_cmd_n         (gemm_cmd_n_dec),
        .gemm_cmd_k         (gemm_cmd_k_dec),
        .gemm_cmd_flags     (gemm_cmd_flags_dec),
        // Softmax
        .softmax_cmd_valid  (softmax_cmd_valid_dec),
        .softmax_cmd_src0   (softmax_cmd_src0_dec),
        .softmax_cmd_dst    (softmax_cmd_dst_dec),
        .softmax_cmd_len    (softmax_cmd_len_dec),
        .softmax_cmd_flags  (softmax_cmd_flags_dec),
        // LayerNorm
        .layernorm_cmd_valid(layernorm_cmd_valid_dec),
        .layernorm_cmd_src0 (layernorm_cmd_src0_dec),
        .layernorm_cmd_dst  (layernorm_cmd_dst_dec),
        .layernorm_cmd_len  (layernorm_cmd_len_dec),
        .layernorm_cmd_flags(layernorm_cmd_flags_dec),
        // GELU
        .gelu_cmd_valid     (gelu_cmd_valid_dec),
        .gelu_cmd_src0      (gelu_cmd_src0_dec),
        .gelu_cmd_dst       (gelu_cmd_dst_dec),
        .gelu_cmd_len       (gelu_cmd_len_dec),
        .gelu_cmd_flags     (gelu_cmd_flags_dec),
        // Vec
        .vec_cmd_valid      (vec_cmd_valid_dec),
        .vec_cmd_src0       (vec_cmd_src0_dec),
        .vec_cmd_src1       (vec_cmd_src1_dec),
        .vec_cmd_dst        (vec_cmd_dst_dec),
        .vec_cmd_len        (vec_cmd_len_dec),
        .vec_cmd_flags      (vec_cmd_flags_dec),
        .vec_cmd_imm        (vec_cmd_imm_dec),
        // DMA
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
        // KV
        .kv_cmd_valid       (kv_cmd_valid_dec),
        .kv_cmd_opcode      (kv_cmd_opcode_dec),
        .kv_cmd_src0        (kv_cmd_src0_dec),
        .kv_cmd_dst         (kv_cmd_dst_dec),
        .kv_cmd_len         (kv_cmd_len_dec),
        .kv_cmd_flags       (kv_cmd_flags_dec),
        .kv_cmd_imm         (kv_cmd_imm_dec),
        // Barrier
        .barrier_trigger    (barrier_trigger),
        // Scoreboard
        .issue_valid        (issue_valid),
        .issue_engine_id    (issue_engine_id),
        // Debug
        .decoded_instr      (decoded_instr),
        .program_end        (program_end_dec)
    );

    // ================================================================
    // GEMM Shim: capture command, hold busy, let C++ compute and pulse done
    // ================================================================
    logic        gemm_active;
    logic [15:0] gemm_src0_r, gemm_src1_r, gemm_dst_r;
    logic [15:0] gemm_m_r, gemm_n_r, gemm_k_r;
    logic [7:0]  gemm_flags_r;
    logic [15:0] gemm_imm_r;
    logic        gemm_captured_r;
    logic        gemm_done_internal;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            gemm_active    <= 1'b0;
            gemm_captured_r <= 1'b0;
            gemm_src0_r    <= '0;
            gemm_src1_r    <= '0;
            gemm_dst_r     <= '0;
            gemm_m_r       <= '0;
            gemm_n_r       <= '0;
            gemm_k_r       <= '0;
            gemm_flags_r   <= '0;
            gemm_imm_r     <= '0;
        end else begin
            gemm_captured_r <= 1'b0;  // pulse
            if (gemm_cmd_valid_dec && !gemm_active) begin
                gemm_active    <= 1'b1;
                gemm_captured_r <= 1'b1;
                gemm_src0_r    <= gemm_cmd_src0_dec;
                gemm_src1_r    <= gemm_cmd_src1_dec;
                gemm_dst_r     <= gemm_cmd_dst_dec;
                gemm_m_r       <= gemm_cmd_m_dec;
                gemm_n_r       <= gemm_cmd_n_dec;
                gemm_k_r       <= gemm_cmd_k_dec;
                gemm_flags_r   <= gemm_cmd_flags_dec;
                gemm_imm_r     <= decoded_instr.imm;
            end else if (gemm_done_pulse && gemm_active) begin
                gemm_active <= 1'b0;
            end
        end
    end

    assign gemm_done_internal = gemm_done_pulse && gemm_active;
    assign gemm_cmd_captured  = gemm_captured_r;
    assign gemm_src0          = gemm_src0_r;
    assign gemm_src1          = gemm_src1_r;
    assign gemm_dst           = gemm_dst_r;
    assign gemm_M             = gemm_m_r;
    assign gemm_N             = gemm_n_r;
    assign gemm_K             = gemm_k_r;
    assign gemm_flags         = gemm_flags_r;
    assign gemm_imm           = gemm_imm_r;

    // ================================================================
    // DATA SRAM (8-bit x SRAM_DEPTH)
    // Port A (read): softmax or TB
    // Port B (write): softmax or TB
    // ================================================================
    logic                    softmax_busy_i, softmax_done_i;
    logic                    sm_rd_en;
    logic [15:0]             sm_rd_addr;
    logic [7:0]              sm_rd_data;
    logic                    sm_wr_en;
    logic [15:0]             sm_wr_addr;
    logic [7:0]              sm_wr_data;

    logic                    ds_a_en;
    logic [SRAM_ADDR_W-1:0]  ds_a_addr;
    logic [7:0]              ds_a_dout;
    logic                    ds_b_en, ds_b_we;
    logic [SRAM_ADDR_W-1:0]  ds_b_addr;
    logic [7:0]              ds_b_din;

    always_comb begin
        if (softmax_busy_i) begin
            ds_a_en   = sm_rd_en;
            ds_a_addr = sm_rd_addr[SRAM_ADDR_W-1:0];
            ds_b_en   = sm_wr_en;
            ds_b_we   = sm_wr_en;
            ds_b_addr = sm_wr_addr[SRAM_ADDR_W-1:0];
            ds_b_din  = sm_wr_data;
        end else begin
            // TB access
            ds_a_en   = tb_sram_rd_en;
            ds_a_addr = tb_sram_rd_addr;
            ds_b_en   = tb_sram_wr_en;
            ds_b_we   = tb_sram_wr_en;
            ds_b_addr = tb_sram_wr_addr;
            ds_b_din  = tb_sram_wr_data;
        end
    end

    sram_dp #(.DEPTH(SRAM_DEPTH), .WIDTH(8)) u_data_sram (
        .clk    (clk),
        .en_a   (ds_a_en),
        .we_a   (1'b0),
        .addr_a (ds_a_addr),
        .din_a  (8'd0),
        .dout_a (ds_a_dout),
        .en_b   (ds_b_en),
        .we_b   (ds_b_we),
        .addr_b (ds_b_addr),
        .din_b  (ds_b_din),
        .dout_b ()
    );

    assign sm_rd_data       = ds_a_dout;
    assign tb_sram_rd_data  = ds_a_dout;

    // ================================================================
    // SCRATCH SRAM (16-bit x SRAM_DEPTH, softmax intermediates)
    // Port A (read): softmax or TB (TB not needed, but wired for debug)
    // Port B (write): softmax
    // ================================================================
    logic        sm_scr_wr_en;
    logic [15:0] sm_scr_wr_addr;
    logic [15:0] sm_scr_wr_data;
    logic        sm_scr_rd_en;
    logic [15:0] sm_scr_rd_addr;
    logic [15:0] sm_scr_rd_data;

    sram_dp #(.DEPTH(SRAM_DEPTH), .WIDTH(16)) u_scratch_sram (
        .clk    (clk),
        // Port A: softmax read
        .en_a   (sm_scr_rd_en),
        .we_a   (1'b0),
        .addr_a (sm_scr_rd_addr[SRAM_ADDR_W-1:0]),
        .din_a  (16'd0),
        .dout_a (sm_scr_rd_data),
        // Port B: softmax write
        .en_b   (sm_scr_wr_en),
        .we_b   (sm_scr_wr_en),
        .addr_b (sm_scr_wr_addr[SRAM_ADDR_W-1:0]),
        .din_b  (sm_scr_wr_data),
        .dout_b ()
    );

    // ================================================================
    // Softmax Engine (real hardware)
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
        .busy            (softmax_busy_i),
        .done            (softmax_done_i)
    );

    // ================================================================
    // Engine done vector wiring
    // ================================================================
    assign engine_done_vec[0] = gemm_done_internal;   // GEMM
    assign engine_done_vec[1] = softmax_done_i;        // Softmax (real)
    assign engine_done_vec[2] = 1'b0;                  // LayerNorm (not used)
    assign engine_done_vec[3] = 1'b0;                  // GELU (not used)
    assign engine_done_vec[4] = 1'b0;                  // Vec (not used)
    assign engine_done_vec[5] = 1'b0;                  // DMA (not used)

    // ================================================================
    // Output assignments
    // ================================================================
    // Latch program_end: fetch_done fires when OP_END is detected (fetch
    // absorbs it without dispatching to decode). We latch it because the
    // pulse may occur while C++ is busy handling a GEMM command.
    logic program_end_latch;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            program_end_latch <= 1'b0;
        else if (start_pulse)
            program_end_latch <= 1'b0;
        else if (fetch_done || program_end_dec)
            program_end_latch <= 1'b1;
    end
    assign program_end = program_end_latch;
    assign engine_busy_dbg  = engine_busy_vec;
    assign all_idle_dbg     = all_idle;
    assign softmax_busy_dbg = softmax_busy_i;
    assign softmax_done_dbg = softmax_done_i;

endmodule

`default_nettype wire
