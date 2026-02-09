// =============================================================================
// NPU Top-Level Module
// Transformer Inference Accelerator (INT8)
// Synthesizable SystemVerilog - Xilinx FPGA Target
// =============================================================================
`default_nettype none

module top
    import npu_pkg::*;
    import isa_pkg::*;
    import axi_types_pkg::*;
#(
    parameter int ARRAY_M       = 16,
    parameter int ARRAY_N       = 16,
    parameter int P_DATA_W      = 8,
    parameter int P_ACC_W       = 32,
    parameter int P_AXI_DATA_W  = 128,
    parameter int P_AXI_ADDR_W  = 32,
    parameter int P_AXI_STRB_W  = P_AXI_DATA_W / 8,
    parameter int P_AXI_ID_W    = 4,
    parameter int P_SRAM_ADDR_W = 16,
    parameter int P_UCODE_DEPTH = 1024
)(
    input  wire                         clk,
    input  wire                         rst_n,

    // ---- AXI4-Lite Slave (Control Plane) ----
    input  wire  [31:0]                 s_axil_awaddr,
    input  wire                         s_axil_awvalid,
    output wire                         s_axil_awready,
    input  wire  [31:0]                 s_axil_wdata,
    input  wire  [3:0]                  s_axil_wstrb,
    input  wire                         s_axil_wvalid,
    output wire                         s_axil_wready,
    output wire  [1:0]                  s_axil_bresp,
    output wire                         s_axil_bvalid,
    input  wire                         s_axil_bready,
    input  wire  [31:0]                 s_axil_araddr,
    input  wire                         s_axil_arvalid,
    output wire                         s_axil_arready,
    output wire  [31:0]                 s_axil_rdata,
    output wire  [1:0]                  s_axil_rresp,
    output wire                         s_axil_rvalid,
    input  wire                         s_axil_rready,

    // ---- AXI4 Master (Data Plane - DMA) ----
    output wire  [P_AXI_ID_W-1:0]      m_axi_arid,
    output wire  [P_AXI_ADDR_W-1:0]    m_axi_araddr,
    output wire  [7:0]                  m_axi_arlen,
    output wire  [2:0]                  m_axi_arsize,
    output wire  [1:0]                  m_axi_arburst,
    output wire                         m_axi_arvalid,
    input  wire                         m_axi_arready,
    input  wire  [P_AXI_ID_W-1:0]      m_axi_rid,
    input  wire  [P_AXI_DATA_W-1:0]    m_axi_rdata,
    input  wire  [1:0]                  m_axi_rresp,
    input  wire                         m_axi_rlast,
    input  wire                         m_axi_rvalid,
    output wire                         m_axi_rready,
    output wire  [P_AXI_ID_W-1:0]      m_axi_awid,
    output wire  [P_AXI_ADDR_W-1:0]    m_axi_awaddr,
    output wire  [7:0]                  m_axi_awlen,
    output wire  [2:0]                  m_axi_awsize,
    output wire  [1:0]                  m_axi_awburst,
    output wire                         m_axi_awvalid,
    input  wire                         m_axi_awready,
    output wire  [P_AXI_DATA_W-1:0]    m_axi_wdata,
    output wire  [P_AXI_STRB_W-1:0]    m_axi_wstrb,
    output wire                         m_axi_wlast,
    output wire                         m_axi_wvalid,
    input  wire                         m_axi_wready,
    input  wire  [P_AXI_ID_W-1:0]      m_axi_bid,
    input  wire  [1:0]                  m_axi_bresp,
    input  wire                         m_axi_bvalid,
    output wire                         m_axi_bready
);

    // =========================================================================
    // Internal signals
    // =========================================================================
    // Control registers
    wire        start_pulse;
    wire        soft_reset;
    wire [31:0] reg_ctrl;
    wire [31:0] reg_status;
    wire [31:0] reg_ucode_base;
    wire [31:0] reg_ucode_len;
    wire [31:0] reg_ddr_base_act;
    wire [31:0] reg_ddr_base_wgt;
    wire [31:0] reg_ddr_base_kv;
    wire [31:0] reg_ddr_base_out;
    wire [31:0] reg_model_hidden;
    wire [31:0] reg_model_heads;
    wire [31:0] reg_model_head_dim;
    wire [31:0] reg_seq_len;
    wire [31:0] reg_token_idx;
    wire [31:0] reg_debug_ctrl;

    logic npu_done;
    logic npu_busy;
    logic npu_error;

    wire rst_int_n;
    assign rst_int_n = rst_n & ~soft_reset;

    // Scoreboard
    logic                        issue_valid;
    logic [2:0]                  issue_engine_id;
    logic [NUM_ENGINES-1:0]      engine_busy;
    logic [NUM_ENGINES-1:0]      can_issue;
    logic [NUM_ENGINES-1:0]      engine_done_vec;
    logic                        all_idle;

    // Barrier
    logic barrier_trigger;
    logic barrier_stall;

    // Fetch -> Decode
    logic              fetch_instr_valid;
    logic              fetch_instr_ready;
    logic [127:0]      fetch_instr_data;
    logic [P_SRAM_ADDR_W-1:0] fetch_pc;
    logic              fetch_done;

    // UCODE SRAM
    logic                       uc_rd_en;
    logic [P_SRAM_ADDR_W-1:0]  uc_rd_addr;
    logic [127:0]               uc_rd_data;
    logic                       uc_rd_valid;

    // Engine command signals from decode
    logic       gemm_cmd_valid, softmax_cmd_valid, layernorm_cmd_valid;
    logic       gelu_cmd_valid, vec_cmd_valid, dma_rd_cmd_valid, dma_wr_cmd_valid;
    logic       kv_cmd_valid;

    // Engine status
    logic gemm_done, softmax_done, layernorm_done, gelu_done, vec_done;
    logic dma_rd_done, dma_wr_done;
    logic dma_rd_busy, dma_wr_busy;

    // DMA command
    logic        dma_rd_int_cmd_valid, dma_rd_int_cmd_ready;
    logic [31:0] dma_rd_int_cmd_addr;
    logic [23:0] dma_rd_int_cmd_len;
    logic [3:0]  dma_rd_int_cmd_tag;
    logic        dma_rd_data_valid, dma_rd_data_ready;
    logic [P_AXI_DATA_W-1:0] dma_rd_data;
    logic        dma_rd_data_last;

    logic        dma_wr_int_cmd_valid, dma_wr_int_cmd_ready;
    logic [31:0] dma_wr_int_cmd_addr;
    logic [23:0] dma_wr_int_cmd_len;
    logic [3:0]  dma_wr_int_cmd_tag;

    // Decoded fields from decoder
    logic [15:0] dec_gemm_src0, dec_gemm_src1, dec_gemm_dst;
    logic [15:0] dec_gemm_m, dec_gemm_n, dec_gemm_k;
    logic [7:0]  dec_gemm_flags;
    logic [15:0] dec_dma_rd_src, dec_dma_rd_dst, dec_dma_rd_len;
    logic [7:0]  dec_dma_rd_flags;
    logic [15:0] dec_dma_wr_src, dec_dma_wr_dst, dec_dma_wr_len;
    logic [7:0]  dec_dma_wr_flags;
    ucode_instr_t decoded_instr;
    logic program_end;

    // Compose engine_done vector
    assign engine_done_vec[0] = gemm_done;
    assign engine_done_vec[1] = softmax_done;
    assign engine_done_vec[2] = layernorm_done;
    assign engine_done_vec[3] = gelu_done;
    assign engine_done_vec[4] = vec_done;
    assign engine_done_vec[5] = dma_rd_done | dma_wr_done;

    assign npu_busy = |engine_busy;
    assign npu_error = 1'b0; // TODO: aggregate errors

    // Latch done signal so it persists until next START
    logic npu_done_latch;
    always_ff @(posedge clk or negedge rst_int_n) begin
        if (!rst_int_n)
            npu_done_latch <= 1'b0;
        else if (start_pulse)
            npu_done_latch <= 1'b0;
        else if (program_end || fetch_done)
            npu_done_latch <= 1'b1;
    end
    assign npu_done = npu_done_latch;

    // =========================================================================
    // AXI4-Lite Register Bank
    // =========================================================================
    axi_lite_regs u_regs (
        .clk              (clk),
        .rst_n            (rst_n),
        .s_axil_awaddr    (s_axil_awaddr),
        .s_axil_awvalid   (s_axil_awvalid),
        .s_axil_awready   (s_axil_awready),
        .s_axil_wdata     (s_axil_wdata),
        .s_axil_wstrb     (s_axil_wstrb),
        .s_axil_wvalid    (s_axil_wvalid),
        .s_axil_wready    (s_axil_wready),
        .s_axil_bresp     (s_axil_bresp),
        .s_axil_bvalid    (s_axil_bvalid),
        .s_axil_bready    (s_axil_bready),
        .s_axil_araddr    (s_axil_araddr),
        .s_axil_arvalid   (s_axil_arvalid),
        .s_axil_arready   (s_axil_arready),
        .s_axil_rdata     (s_axil_rdata),
        .s_axil_rresp     (s_axil_rresp),
        .s_axil_rvalid    (s_axil_rvalid),
        .s_axil_rready    (s_axil_rready),
        .done_i           (npu_done),
        .busy_i           (npu_busy),
        .error_i          (npu_error),
        .ctrl_o           (reg_ctrl),
        .status_o         (reg_status),
        .ucode_base_o     (reg_ucode_base),
        .ucode_len_o      (reg_ucode_len),
        .ddr_base_act_o   (reg_ddr_base_act),
        .ddr_base_wgt_o   (reg_ddr_base_wgt),
        .ddr_base_kv_o    (reg_ddr_base_kv),
        .ddr_base_out_o   (reg_ddr_base_out),
        .model_hidden_o   (reg_model_hidden),
        .model_heads_o    (reg_model_heads),
        .model_head_dim_o (reg_model_head_dim),
        .seq_len_o        (reg_seq_len),
        .token_idx_o      (reg_token_idx),
        .debug_ctrl_o     (reg_debug_ctrl),
        .start_pulse_o    (start_pulse),
        .soft_reset_o     (soft_reset)
    );

    // =========================================================================
    // AXI4 DMA Read Master
    // =========================================================================
    axi_dma_rd #(
        .AXI_DATA_W    (P_AXI_DATA_W),
        .AXI_ADDR_W    (P_AXI_ADDR_W)
    ) u_dma_rd (
        .clk            (clk),
        .rst_n          (rst_int_n),
        .cmd_valid      (dma_rd_int_cmd_valid),
        .cmd_ready      (dma_rd_int_cmd_ready),
        .cmd_addr       (dma_rd_int_cmd_addr),
        .cmd_len        (dma_rd_int_cmd_len),
        .cmd_tag        (dma_rd_int_cmd_tag),
        .m_axi_arid     (m_axi_arid),
        .m_axi_araddr   (m_axi_araddr),
        .m_axi_arlen    (m_axi_arlen),
        .m_axi_arsize   (m_axi_arsize),
        .m_axi_arburst  (m_axi_arburst),
        .m_axi_arvalid  (m_axi_arvalid),
        .m_axi_arready  (m_axi_arready),
        .m_axi_rid      (m_axi_rid),
        .m_axi_rdata    (m_axi_rdata),
        .m_axi_rresp    (m_axi_rresp),
        .m_axi_rlast    (m_axi_rlast),
        .m_axi_rvalid   (m_axi_rvalid),
        .m_axi_rready   (m_axi_rready),
        .data_valid     (dma_rd_data_valid),
        .data_ready     (dma_rd_data_ready),
        .data_out       (dma_rd_data),
        .data_last      (dma_rd_data_last),
        .data_tag       (),
        .busy           (dma_rd_busy),
        .done           (dma_rd_done),
        .error          ()
    );

    // =========================================================================
    // AXI4 DMA Write Master
    // =========================================================================
    axi_dma_wr #(
        .AXI_DATA_W    (P_AXI_DATA_W),
        .AXI_ADDR_W    (P_AXI_ADDR_W)
    ) u_dma_wr (
        .clk            (clk),
        .rst_n          (rst_int_n),
        .cmd_valid      (dma_wr_int_cmd_valid),
        .cmd_ready      (dma_wr_int_cmd_ready),
        .cmd_addr       (dma_wr_int_cmd_addr),
        .cmd_len        (dma_wr_int_cmd_len),
        .cmd_tag        (dma_wr_int_cmd_tag),
        .m_axi_awid     (m_axi_awid),
        .m_axi_awaddr   (m_axi_awaddr),
        .m_axi_awlen    (m_axi_awlen),
        .m_axi_awsize   (m_axi_awsize),
        .m_axi_awburst  (m_axi_awburst),
        .m_axi_awvalid  (m_axi_awvalid),
        .m_axi_awready  (m_axi_awready),
        .m_axi_wdata    (m_axi_wdata),
        .m_axi_wstrb    (m_axi_wstrb),
        .m_axi_wlast    (m_axi_wlast),
        .m_axi_wvalid   (m_axi_wvalid),
        .m_axi_wready   (m_axi_wready),
        .m_axi_bid      (m_axi_bid),
        .m_axi_bresp    (m_axi_bresp),
        .m_axi_bvalid   (m_axi_bvalid),
        .m_axi_bready   (m_axi_bready),
        .data_valid     (1'b0),       // Placeholder: connect when STORE implemented
        .data_ready     (),
        .data_in        ({P_AXI_DATA_W{1'b0}}),
        .data_last      (1'b0),
        .busy           (dma_wr_busy),
        .done           (dma_wr_done),
        .error          ()
    );

    // =========================================================================
    // UCODE SRAM (128-bit wide, single bank)
    // =========================================================================
    sram_dp #(
        .DEPTH (P_UCODE_DEPTH),
        .WIDTH (128)
    ) u_ucode_sram (
        .clk    (clk),
        .en_a   (uc_rd_en),
        .we_a   (1'b0),
        .addr_a (uc_rd_addr[$clog2(P_UCODE_DEPTH)-1:0]),
        .din_a  (128'b0),
        .dout_a (uc_rd_data),
        .en_b   (1'b0),
        .we_b   (1'b0),
        .addr_b ({$clog2(P_UCODE_DEPTH){1'b0}}),
        .din_b  (128'b0),
        .dout_b ()
    );

    // SRAM read valid (1-cycle latency)
    logic uc_rd_en_d;
    always_ff @(posedge clk or negedge rst_int_n) begin
        if (!rst_int_n) uc_rd_en_d <= 1'b0;
        else            uc_rd_en_d <= uc_rd_en;
    end
    assign uc_rd_valid = uc_rd_en_d;

    // =========================================================================
    // Microcode Fetch
    // =========================================================================
    ucode_fetch #(
        .INSTR_W     (128),
        .SRAM_ADDR_W (P_SRAM_ADDR_W)
    ) u_fetch (
        .clk            (clk),
        .rst_n          (rst_int_n),
        .start          (start_pulse),
        .stop           (1'b0),
        .ucode_base_addr({P_SRAM_ADDR_W{1'b0}}),
        .ucode_len      (reg_ucode_len[P_SRAM_ADDR_W-1:0]),
        .rd_en          (uc_rd_en),
        .rd_addr        (uc_rd_addr),
        .rd_data        (uc_rd_data),
        .rd_valid       (uc_rd_valid),
        .instr_valid    (fetch_instr_valid),
        .instr_ready    (fetch_instr_ready),
        .instr_data     (fetch_instr_data),
        .pc             (fetch_pc),
        .done           (fetch_done),
        .busy           ()
    );

    // =========================================================================
    // Microcode Decode & Dispatch
    // =========================================================================
    ucode_decode u_decode (
        .clk                (clk),
        .rst_n              (rst_int_n),
        .instr_valid        (fetch_instr_valid),
        .instr_data         (fetch_instr_data),
        .instr_ready        (fetch_instr_ready),
        // Scoreboard
        .can_issue          (can_issue),
        .all_idle           (all_idle),
        // GEMM
        .gemm_cmd_valid     (gemm_cmd_valid),
        .gemm_cmd_src0      (dec_gemm_src0),
        .gemm_cmd_src1      (dec_gemm_src1),
        .gemm_cmd_dst       (dec_gemm_dst),
        .gemm_cmd_m         (dec_gemm_m),
        .gemm_cmd_n         (dec_gemm_n),
        .gemm_cmd_k         (dec_gemm_k),
        .gemm_cmd_flags     (dec_gemm_flags),
        // Softmax
        .softmax_cmd_valid  (softmax_cmd_valid),
        .softmax_cmd_src0   (),
        .softmax_cmd_dst    (),
        .softmax_cmd_len    (),
        .softmax_cmd_flags  (),
        // LayerNorm
        .layernorm_cmd_valid(layernorm_cmd_valid),
        .layernorm_cmd_src0 (),
        .layernorm_cmd_dst  (),
        .layernorm_cmd_len  (),
        .layernorm_cmd_flags(),
        // GELU
        .gelu_cmd_valid     (gelu_cmd_valid),
        .gelu_cmd_src0      (),
        .gelu_cmd_dst       (),
        .gelu_cmd_len       (),
        .gelu_cmd_flags     (),
        // Vec
        .vec_cmd_valid      (vec_cmd_valid),
        .vec_cmd_src0       (),
        .vec_cmd_src1       (),
        .vec_cmd_dst        (),
        .vec_cmd_len        (),
        .vec_cmd_flags      (),
        .vec_cmd_imm        (),
        // DMA Read
        .dma_rd_cmd_valid   (dma_rd_cmd_valid),
        .dma_rd_cmd_src     (dec_dma_rd_src),
        .dma_rd_cmd_dst     (dec_dma_rd_dst),
        .dma_rd_cmd_len     (dec_dma_rd_len),
        .dma_rd_cmd_flags   (dec_dma_rd_flags),
        // DMA Write
        .dma_wr_cmd_valid   (dma_wr_cmd_valid),
        .dma_wr_cmd_src     (dec_dma_wr_src),
        .dma_wr_cmd_dst     (dec_dma_wr_dst),
        .dma_wr_cmd_len     (dec_dma_wr_len),
        .dma_wr_cmd_flags   (dec_dma_wr_flags),
        // KV
        .kv_cmd_valid       (kv_cmd_valid),
        .kv_cmd_opcode      (),
        .kv_cmd_src0        (),
        .kv_cmd_dst         (),
        .kv_cmd_len         (),
        .kv_cmd_flags       (),
        .kv_cmd_imm         (),
        // Barrier
        .barrier_trigger    (barrier_trigger),
        // Scoreboard issue
        .issue_valid        (issue_valid),
        .issue_engine_id    (issue_engine_id),
        // Debug
        .decoded_instr      (decoded_instr),
        .program_end        (program_end)
    );

    // =========================================================================
    // Scoreboard
    // =========================================================================
    scoreboard #(
        .NUM_ENGINES (NUM_ENGINES)
    ) u_scoreboard (
        .clk              (clk),
        .rst_n            (rst_int_n),
        .issue_valid      (issue_valid),
        .issue_engine_id  (issue_engine_id),
        .engine_done      (engine_done_vec),
        .engine_busy      (engine_busy),
        .can_issue        (can_issue),
        .all_idle         (all_idle)
    );

    // =========================================================================
    // Barrier
    // =========================================================================
    barrier u_barrier (
        .clk      (clk),
        .rst_n    (rst_int_n),
        .trigger  (barrier_trigger),
        .all_idle (all_idle),
        .stall    (barrier_stall),
        .done     ()
    );

    // =========================================================================
    // DMA Command Translation (decode -> DMA engine)
    // =========================================================================
    always_comb begin
        dma_rd_int_cmd_valid = dma_rd_cmd_valid;
        case (dec_dma_rd_flags[2:0])
            3'd0:    dma_rd_int_cmd_addr = reg_ddr_base_act + {16'b0, dec_dma_rd_src};
            3'd1:    dma_rd_int_cmd_addr = reg_ddr_base_wgt + {16'b0, dec_dma_rd_src};
            3'd2:    dma_rd_int_cmd_addr = reg_ddr_base_kv  + {16'b0, dec_dma_rd_src};
            3'd4:    dma_rd_int_cmd_addr = reg_ucode_base   + {16'b0, dec_dma_rd_src};
            default: dma_rd_int_cmd_addr = reg_ddr_base_act + {16'b0, dec_dma_rd_src};
        endcase
        dma_rd_int_cmd_len = {8'b0, dec_dma_rd_len};
        dma_rd_int_cmd_tag = dec_dma_rd_flags[6:3];
    end

    always_comb begin
        dma_wr_int_cmd_valid = dma_wr_cmd_valid;
        case (dec_dma_wr_flags[2:0])
            3'd0:    dma_wr_int_cmd_addr = reg_ddr_base_act + {16'b0, dec_dma_wr_dst};
            3'd3:    dma_wr_int_cmd_addr = reg_ddr_base_out + {16'b0, dec_dma_wr_dst};
            default: dma_wr_int_cmd_addr = reg_ddr_base_out + {16'b0, dec_dma_wr_dst};
        endcase
        dma_wr_int_cmd_len = {8'b0, dec_dma_wr_len};
        dma_wr_int_cmd_tag = dec_dma_wr_flags[6:3];
    end

    assign dma_rd_data_ready = 1'b1; // Always accept DMA data (placeholder)

    // =========================================================================
    // Engine stubs: placeholder done signals for engines not yet fully wired
    // Full wiring requires SRAM arbitration (future enhancement)
    // =========================================================================

    // GEMM: generate done pulse one cycle after cmd_valid
    // In full implementation, gemm_ctrl drives the systolic array
    logic gemm_busy_r;
    logic [15:0] gemm_timer;
    always_ff @(posedge clk or negedge rst_int_n) begin
        if (!rst_int_n) begin
            gemm_busy_r <= 1'b0;
            gemm_timer  <= '0;
            gemm_done   <= 1'b0;
        end else begin
            gemm_done <= 1'b0;
            if (gemm_cmd_valid && !gemm_busy_r) begin
                gemm_busy_r <= 1'b1;
                gemm_timer  <= 16'd10; // Simulated latency
            end else if (gemm_busy_r) begin
                if (gemm_timer == 0) begin
                    gemm_busy_r <= 1'b0;
                    gemm_done   <= 1'b1;
                end else begin
                    gemm_timer <= gemm_timer - 1;
                end
            end
        end
    end

    // Softmax engine done stub
    logic softmax_busy_r;
    logic [15:0] softmax_timer;
    always_ff @(posedge clk or negedge rst_int_n) begin
        if (!rst_int_n) begin
            softmax_busy_r <= 1'b0;
            softmax_timer  <= '0;
            softmax_done   <= 1'b0;
        end else begin
            softmax_done <= 1'b0;
            if (softmax_cmd_valid && !softmax_busy_r) begin
                softmax_busy_r <= 1'b1;
                softmax_timer  <= 16'd10;
            end else if (softmax_busy_r) begin
                if (softmax_timer == 0) begin
                    softmax_busy_r <= 1'b0;
                    softmax_done   <= 1'b1;
                end else begin
                    softmax_timer <= softmax_timer - 1;
                end
            end
        end
    end

    // LayerNorm engine done stub
    logic layernorm_busy_r;
    logic [15:0] layernorm_timer;
    always_ff @(posedge clk or negedge rst_int_n) begin
        if (!rst_int_n) begin
            layernorm_busy_r <= 1'b0;
            layernorm_timer  <= '0;
            layernorm_done   <= 1'b0;
        end else begin
            layernorm_done <= 1'b0;
            if (layernorm_cmd_valid && !layernorm_busy_r) begin
                layernorm_busy_r <= 1'b1;
                layernorm_timer  <= 16'd10;
            end else if (layernorm_busy_r) begin
                if (layernorm_timer == 0) begin
                    layernorm_busy_r <= 1'b0;
                    layernorm_done   <= 1'b1;
                end else begin
                    layernorm_timer <= layernorm_timer - 1;
                end
            end
        end
    end

    // GELU engine done stub
    logic gelu_busy_r;
    logic [15:0] gelu_timer;
    always_ff @(posedge clk or negedge rst_int_n) begin
        if (!rst_int_n) begin
            gelu_busy_r <= 1'b0;
            gelu_timer  <= '0;
            gelu_done   <= 1'b0;
        end else begin
            gelu_done <= 1'b0;
            if (gelu_cmd_valid && !gelu_busy_r) begin
                gelu_busy_r <= 1'b1;
                gelu_timer  <= 16'd5;
            end else if (gelu_busy_r) begin
                if (gelu_timer == 0) begin
                    gelu_busy_r <= 1'b0;
                    gelu_done   <= 1'b1;
                end else begin
                    gelu_timer <= gelu_timer - 1;
                end
            end
        end
    end

    // Vec engine done stub
    logic vec_busy_r;
    logic [15:0] vec_timer;
    always_ff @(posedge clk or negedge rst_int_n) begin
        if (!rst_int_n) begin
            vec_busy_r <= 1'b0;
            vec_timer  <= '0;
            vec_done   <= 1'b0;
        end else begin
            vec_done <= 1'b0;
            if (vec_cmd_valid && !vec_busy_r) begin
                vec_busy_r <= 1'b1;
                vec_timer  <= 16'd5;
            end else if (vec_busy_r) begin
                if (vec_timer == 0) begin
                    vec_busy_r <= 1'b0;
                    vec_done   <= 1'b1;
                end else begin
                    vec_timer <= vec_timer - 1;
                end
            end
        end
    end

    // =========================================================================
    // Simulation-only assertions
    // =========================================================================
`ifndef SYNTHESIS
    always @(posedge clk) begin
        if (rst_int_n && issue_valid) begin
            assert (issue_engine_id < NUM_ENGINES[2:0])
                else $error("Invalid engine ID: %0d", issue_engine_id);
        end
    end
`endif

endmodule

`default_nettype wire
