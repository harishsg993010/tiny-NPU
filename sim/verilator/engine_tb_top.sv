// =============================================================================
// engine_tb_top.sv - Testbench wrapper for engine-level compute testing
// Instantiates SRAMs, compute engines, and systolic array with TB-accessible ports
// =============================================================================
`default_nettype none

module engine_tb_top
    import npu_pkg::*;
    import fixed_pkg::*;
#(
    parameter int SRAM_DEPTH  = 4096,
    parameter int SRAM_ADDR_W = $clog2(SRAM_DEPTH)
)(
    input  wire                        clk,
    input  wire                        rst_n,

    // ---- TB SRAM0 access (data SRAM 0, 8-bit) ----
    input  wire                        tb_sram0_wr_en,
    input  wire  [SRAM_ADDR_W-1:0]    tb_sram0_wr_addr,
    input  wire  [7:0]                tb_sram0_wr_data,
    input  wire                        tb_sram0_rd_en,
    input  wire  [SRAM_ADDR_W-1:0]    tb_sram0_rd_addr,
    output wire  [7:0]                tb_sram0_rd_data,

    // ---- TB SRAM1 access (data SRAM 1, 8-bit) ----
    input  wire                        tb_sram1_wr_en,
    input  wire  [SRAM_ADDR_W-1:0]    tb_sram1_wr_addr,
    input  wire  [7:0]                tb_sram1_wr_data,
    input  wire                        tb_sram1_rd_en,
    input  wire  [SRAM_ADDR_W-1:0]    tb_sram1_rd_addr,
    output wire  [7:0]                tb_sram1_rd_data,

    // ---- TB Scratch SRAM access (16-bit, for softmax) ----
    input  wire                        tb_scratch_wr_en,
    input  wire  [SRAM_ADDR_W-1:0]    tb_scratch_wr_addr,
    input  wire  [15:0]               tb_scratch_wr_data,
    input  wire                        tb_scratch_rd_en,
    input  wire  [SRAM_ADDR_W-1:0]    tb_scratch_rd_addr,
    output wire  [15:0]               tb_scratch_rd_data,

    // ---- GELU engine command ----
    input  wire                        gelu_cmd_valid,
    input  wire  [15:0]               gelu_length,
    input  wire  [15:0]               gelu_src_base,
    input  wire  [15:0]               gelu_dst_base,
    output wire                        gelu_busy,
    output wire                        gelu_done,

    // ---- Softmax engine command ----
    input  wire                        softmax_cmd_valid,
    input  wire  [15:0]               softmax_length,
    input  wire  [15:0]               softmax_src_base,
    input  wire  [15:0]               softmax_dst_base,
    input  wire  [15:0]               softmax_scale,
    input  wire                        softmax_causal_en,
    input  wire  [15:0]               softmax_causal_limit,
    output wire                        softmax_busy,
    output wire                        softmax_done,

    // ---- LayerNorm engine command ----
    input  wire                        layernorm_cmd_valid,
    input  wire  [15:0]               layernorm_length,
    input  wire  [15:0]               layernorm_src_base,
    input  wire  [15:0]               layernorm_dst_base,
    input  wire  [15:0]               layernorm_gamma_base,
    input  wire  [15:0]               layernorm_beta_base,
    output wire                        layernorm_busy,
    output wire                        layernorm_done,

    // ---- Vec engine command ----
    input  wire                        vec_cmd_valid,
    input  wire  [1:0]                vec_opcode,
    input  wire  [15:0]               vec_length,
    input  wire  [15:0]               vec_src0_base,
    input  wire  [15:0]               vec_src1_base,
    input  wire  [15:0]               vec_dst_base,
    input  wire  [7:0]                vec_scale,
    input  wire  [7:0]                vec_shift,
    output wire                        vec_busy,
    output wire                        vec_done,

    // ---- Systolic array direct drive ----
    input  wire                        sa_en,
    input  wire                        sa_clear_acc,
    input  wire  [127:0]              sa_a_col_flat,  // packed 16 x int8
    input  wire  [127:0]              sa_b_row_flat,  // packed 16 x int8
    input  wire  [3:0]                sa_acc_rd_row,
    input  wire  [3:0]                sa_acc_rd_col,
    output wire signed [31:0]         sa_acc_rd_data,
    output wire                        sa_acc_valid
);

    // ================================================================
    // Engine busy / done signals
    // ================================================================
    logic gelu_busy_i, softmax_busy_i, layernorm_busy_i, vec_busy_i;
    logic gelu_done_i, softmax_done_i, layernorm_done_i, vec_done_i;
    assign gelu_busy      = gelu_busy_i;
    assign softmax_busy   = softmax_busy_i;
    assign layernorm_busy = layernorm_busy_i;
    assign vec_busy       = vec_busy_i;
    assign gelu_done      = gelu_done_i;
    assign softmax_done   = softmax_done_i;
    assign layernorm_done = layernorm_done_i;
    assign vec_done       = vec_done_i;

    logic any_engine_busy;
    assign any_engine_busy = gelu_busy_i | softmax_busy_i
                           | layernorm_busy_i | vec_busy_i;

    // ================================================================
    // Engine <-> SRAM wires
    // ================================================================
    // GELU
    logic        gelu_rd_en;
    logic [15:0] gelu_rd_addr;
    logic [7:0]  gelu_rd_data;
    logic        gelu_wr_en;
    logic [15:0] gelu_wr_addr;
    logic [7:0]  gelu_wr_data;

    // Softmax
    logic        sm_rd_en;
    logic [15:0] sm_rd_addr;
    logic [7:0]  sm_rd_data;
    logic        sm_wr_en;
    logic [15:0] sm_wr_addr;
    logic [7:0]  sm_wr_data;
    logic        sm_scr_wr_en;
    logic [15:0] sm_scr_wr_addr;
    logic [15:0] sm_scr_wr_data;
    logic        sm_scr_rd_en;
    logic [15:0] sm_scr_rd_addr;
    logic [15:0] sm_scr_rd_data;

    // LayerNorm
    logic        ln_rd0_en;
    logic [15:0] ln_rd0_addr;
    logic [7:0]  ln_rd0_data;
    logic        ln_rd1_en;
    logic [15:0] ln_rd1_addr;
    logic [7:0]  ln_rd1_data;
    logic        ln_wr_en;
    logic [15:0] ln_wr_addr;
    logic [7:0]  ln_wr_data;

    // Vec
    logic        vec_rd0_en;
    logic [15:0] vec_rd0_addr;
    logic [7:0]  vec_rd0_data;
    logic        vec_rd1_en;
    logic [15:0] vec_rd1_addr;
    logic [7:0]  vec_rd1_data;
    logic        vec_wr_en;
    logic [15:0] vec_wr_addr;
    logic [7:0]  vec_wr_data;

    // ================================================================
    // SRAM0: 8-bit x SRAM_DEPTH   Port A = read, Port B = write
    // Mux between engine and TB based on busy signals
    // ================================================================
    logic                    s0_a_en;
    logic [SRAM_ADDR_W-1:0] s0_a_addr;
    logic [7:0]              s0_a_dout;
    logic                    s0_b_en, s0_b_we;
    logic [SRAM_ADDR_W-1:0] s0_b_addr;
    logic [7:0]              s0_b_din;

    always_comb begin
        if (gelu_busy_i) begin
            s0_a_en   = gelu_rd_en;
            s0_a_addr = gelu_rd_addr[SRAM_ADDR_W-1:0];
            s0_b_en   = gelu_wr_en;
            s0_b_we   = gelu_wr_en;
            s0_b_addr = gelu_wr_addr[SRAM_ADDR_W-1:0];
            s0_b_din  = gelu_wr_data;
        end else if (softmax_busy_i) begin
            s0_a_en   = sm_rd_en;
            s0_a_addr = sm_rd_addr[SRAM_ADDR_W-1:0];
            s0_b_en   = sm_wr_en;
            s0_b_we   = sm_wr_en;
            s0_b_addr = sm_wr_addr[SRAM_ADDR_W-1:0];
            s0_b_din  = sm_wr_data;
        end else if (layernorm_busy_i) begin
            s0_a_en   = ln_rd0_en;
            s0_a_addr = ln_rd0_addr[SRAM_ADDR_W-1:0];
            s0_b_en   = ln_wr_en;
            s0_b_we   = ln_wr_en;
            s0_b_addr = ln_wr_addr[SRAM_ADDR_W-1:0];
            s0_b_din  = ln_wr_data;
        end else if (vec_busy_i) begin
            s0_a_en   = vec_rd0_en;
            s0_a_addr = vec_rd0_addr[SRAM_ADDR_W-1:0];
            s0_b_en   = vec_wr_en;
            s0_b_we   = vec_wr_en;
            s0_b_addr = vec_wr_addr[SRAM_ADDR_W-1:0];
            s0_b_din  = vec_wr_data;
        end else begin
            // TB access
            s0_a_en   = tb_sram0_rd_en;
            s0_a_addr = tb_sram0_rd_addr;
            s0_b_en   = tb_sram0_wr_en;
            s0_b_we   = tb_sram0_wr_en;
            s0_b_addr = tb_sram0_wr_addr;
            s0_b_din  = tb_sram0_wr_data;
        end
    end

    sram_dp #(.DEPTH(SRAM_DEPTH), .WIDTH(8)) u_sram0 (
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

    // Route SRAM0 read data to all engines and TB
    assign gelu_rd_data     = s0_a_dout;
    assign sm_rd_data       = s0_a_dout;
    assign ln_rd0_data      = s0_a_dout;
    assign vec_rd0_data     = s0_a_dout;
    assign tb_sram0_rd_data = s0_a_dout;

    // ================================================================
    // SRAM1: 8-bit x SRAM_DEPTH   Port A = read, Port B = write (TB only)
    // ================================================================
    logic                    s1_a_en;
    logic [SRAM_ADDR_W-1:0] s1_a_addr;
    logic [7:0]              s1_a_dout;

    always_comb begin
        if (layernorm_busy_i) begin
            s1_a_en   = ln_rd1_en;
            s1_a_addr = ln_rd1_addr[SRAM_ADDR_W-1:0];
        end else if (vec_busy_i) begin
            s1_a_en   = vec_rd1_en;
            s1_a_addr = vec_rd1_addr[SRAM_ADDR_W-1:0];
        end else begin
            s1_a_en   = tb_sram1_rd_en;
            s1_a_addr = tb_sram1_rd_addr;
        end
    end

    sram_dp #(.DEPTH(SRAM_DEPTH), .WIDTH(8)) u_sram1 (
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
    assign vec_rd1_data     = s1_a_dout;
    assign tb_sram1_rd_data = s1_a_dout;

    // ================================================================
    // Scratch SRAM: 16-bit x SRAM_DEPTH   (softmax intermediate values)
    // Port A = read, Port B = write
    // ================================================================
    logic                    scr_a_en;
    logic [SRAM_ADDR_W-1:0] scr_a_addr;
    logic [15:0]             scr_a_dout;
    logic                    scr_b_en, scr_b_we;
    logic [SRAM_ADDR_W-1:0] scr_b_addr;
    logic [15:0]             scr_b_din;

    always_comb begin
        if (softmax_busy_i) begin
            scr_a_en   = sm_scr_rd_en;
            scr_a_addr = sm_scr_rd_addr[SRAM_ADDR_W-1:0];
            scr_b_en   = sm_scr_wr_en;
            scr_b_we   = sm_scr_wr_en;
            scr_b_addr = sm_scr_wr_addr[SRAM_ADDR_W-1:0];
            scr_b_din  = sm_scr_wr_data;
        end else begin
            scr_a_en   = tb_scratch_rd_en;
            scr_a_addr = tb_scratch_rd_addr;
            scr_b_en   = tb_scratch_wr_en;
            scr_b_we   = tb_scratch_wr_en;
            scr_b_addr = tb_scratch_wr_addr;
            scr_b_din  = tb_scratch_wr_data;
        end
    end

    sram_dp #(.DEPTH(SRAM_DEPTH), .WIDTH(16)) u_scratch (
        .clk    (clk),
        .en_a   (scr_a_en),
        .we_a   (1'b0),
        .addr_a (scr_a_addr),
        .din_a  (16'd0),
        .dout_a (scr_a_dout),
        .en_b   (scr_b_en),
        .we_b   (scr_b_we),
        .addr_b (scr_b_addr),
        .din_b  (scr_b_din),
        .dout_b ()
    );

    assign sm_scr_rd_data      = scr_a_dout;
    assign tb_scratch_rd_data  = scr_a_dout;

    // ================================================================
    // GELU Engine
    // ================================================================
    gelu_engine u_gelu (
        .clk           (clk),
        .rst_n         (rst_n),
        .cmd_valid     (gelu_cmd_valid),
        .cmd_ready     (),
        .length        (gelu_length),
        .src_base      (gelu_src_base),
        .dst_base      (gelu_dst_base),
        .sram_rd_en    (gelu_rd_en),
        .sram_rd_addr  (gelu_rd_addr),
        .sram_rd_data  (gelu_rd_data),
        .sram_wr_en    (gelu_wr_en),
        .sram_wr_addr  (gelu_wr_addr),
        .sram_wr_data  (gelu_wr_data),
        .busy          (gelu_busy_i),
        .done          (gelu_done_i)
    );

    // ================================================================
    // Softmax Engine
    // ================================================================
    softmax_engine u_softmax (
        .clk             (clk),
        .rst_n           (rst_n),
        .cmd_valid       (softmax_cmd_valid),
        .cmd_ready       (),
        .length          (softmax_length),
        .src_base        (softmax_src_base),
        .dst_base        (softmax_dst_base),
        .scale_factor    (softmax_scale),
        .causal_mask_en  (softmax_causal_en),
        .causal_limit    (softmax_causal_limit),
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
    // LayerNorm Engine
    // ================================================================
    layernorm_engine u_layernorm (
        .clk           (clk),
        .rst_n         (rst_n),
        .cmd_valid     (layernorm_cmd_valid),
        .cmd_ready     (),
        .length        (layernorm_length),
        .src_base      (layernorm_src_base),
        .dst_base      (layernorm_dst_base),
        .gamma_base    (layernorm_gamma_base),
        .beta_base     (layernorm_beta_base),
        .sram_rd0_en   (ln_rd0_en),
        .sram_rd0_addr (ln_rd0_addr),
        .sram_rd0_data (ln_rd0_data),
        .sram_rd1_en   (ln_rd1_en),
        .sram_rd1_addr (ln_rd1_addr),
        .sram_rd1_data (ln_rd1_data),
        .sram_wr_en    (ln_wr_en),
        .sram_wr_addr  (ln_wr_addr),
        .sram_wr_data  (ln_wr_data),
        .busy          (layernorm_busy_i),
        .done          (layernorm_done_i)
    );

    // ================================================================
    // Vec Engine
    // ================================================================
    vec_engine u_vec (
        .clk           (clk),
        .rst_n         (rst_n),
        .cmd_valid     (vec_cmd_valid),
        .cmd_ready     (),
        .opcode        (vec_opcode),
        .length        (vec_length),
        .src0_base     (vec_src0_base),
        .src1_base     (vec_src1_base),
        .dst_base      (vec_dst_base),
        .scale         (vec_scale),
        .shift         (vec_shift),
        .sram_rd0_en   (vec_rd0_en),
        .sram_rd0_addr (vec_rd0_addr),
        .sram_rd0_data (vec_rd0_data),
        .sram_rd1_en   (vec_rd1_en),
        .sram_rd1_addr (vec_rd1_addr),
        .sram_rd1_data (vec_rd1_data),
        .sram_wr_en    (vec_wr_en),
        .sram_wr_addr  (vec_wr_addr),
        .sram_wr_data  (vec_wr_data),
        .busy          (vec_busy_i),
        .done          (vec_done_i)
    );

    // ================================================================
    // Systolic Array (direct drive from C++ testbench)
    // ================================================================
    logic signed [7:0]  sa_a_col_arr [16];
    logic signed [7:0]  sa_b_row_arr [16];
    logic signed [31:0] sa_acc_arr   [16][16];
    logic               sa_acc_valid_i;

    // Unpack flat packed inputs to arrays
    genvar gi;
    generate
        for (gi = 0; gi < 16; gi++) begin : gen_unpack
            assign sa_a_col_arr[gi] = sa_a_col_flat[gi*8 +: 8];
            assign sa_b_row_arr[gi] = sa_b_row_flat[gi*8 +: 8];
        end
    endgenerate

    systolic_array #(
        .M      (16),
        .N      (16),
        .DATA_W (8),
        .ACC_W  (32)
    ) u_systolic (
        .clk       (clk),
        .rst_n     (rst_n),
        .clear_acc (sa_clear_acc),
        .en        (sa_en),
        .a_col     (sa_a_col_arr),
        .b_row     (sa_b_row_arr),
        .acc_out   (sa_acc_arr),
        .acc_valid (sa_acc_valid_i)
    );

    // Address-based accumulator readout
    assign sa_acc_rd_data = sa_acc_arr[sa_acc_rd_row][sa_acc_rd_col];
    assign sa_acc_valid   = sa_acc_valid_i;

endmodule

`default_nettype wire
