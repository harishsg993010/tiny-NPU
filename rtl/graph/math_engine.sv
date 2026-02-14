// =============================================================================
// math_engine.sv - Element-wise LUT-based math engine
// Supports EXP, LOG, SQRT, RSQRT operations on INT8 data via lookup tables
// FSM: ME_IDLE -> ME_READ -> ME_SRAM_WAIT -> ME_LUT_ADDR -> ME_LUT_WAIT -> ME_WRITE -> loop/ME_DONE
// 5 cycles per element: SRAM read issue + SRAM wait + LUT addr + LUT wait + SRAM write
// =============================================================================
`default_nettype none

module math_engine
    import graph_isa_pkg::*;
#(
    parameter int SRAM0_AW = 16
) (
    input  wire                    clk,
    input  wire                    rst_n,

    // -- Command interface ------------------------------------------------
    input  wire                    cmd_valid,
    input  wire  [7:0]             cmd_opcode,
    input  wire  [15:0]            cmd_src_base,
    input  wire  [15:0]            cmd_dst_base,
    input  wire  [15:0]            cmd_length,

    // -- SRAM read port ---------------------------------------------------
    output logic                   sram_rd_en,
    output logic [SRAM0_AW-1:0]   sram_rd_addr,
    input  wire  [7:0]             sram_rd_data,

    // -- SRAM write port --------------------------------------------------
    output logic                   sram_wr_en,
    output logic [SRAM0_AW-1:0]   sram_wr_addr,
    output logic [7:0]             sram_wr_data,

    // -- Status -----------------------------------------------------------
    output logic                   busy,
    output logic                   done
);

    // =====================================================================
    // FSM states
    // =====================================================================
    typedef enum logic [2:0] {
        ME_IDLE      = 3'd0,
        ME_READ      = 3'd1,
        ME_SRAM_WAIT = 3'd2,
        ME_LUT_ADDR  = 3'd3,
        ME_LUT_WAIT  = 3'd4,
        ME_WRITE     = 3'd5,
        ME_DONE      = 3'd6
    } me_state_t;

    me_state_t r_state, w_state;

    // =====================================================================
    // Internal registers
    // =====================================================================
    logic [7:0]            r_opcode;
    logic [15:0]           r_src_base;
    logic [15:0]           r_dst_base;
    logic [15:0]           r_length;
    logic [15:0]           r_index;       // current element index

    logic [7:0]            r_rd_data;     // latched SRAM read data

    // =====================================================================
    // LUT instances - all 4 LUTs instantiated, output muxed by opcode
    // =====================================================================
    logic [7:0] lut_addr;
    logic [7:0] exp_out;
    logic [7:0] log_out;
    logic [7:0] sqrt_out;
    logic [7:0] rsqrt_out;
    logic [7:0] lut_result;

    graph_exp_lut u_exp_lut (
        .clk      (clk),
        .addr     (lut_addr),
        .data_out (exp_out)
    );

    graph_log_lut u_log_lut (
        .clk      (clk),
        .addr     (lut_addr),
        .data_out (log_out)
    );

    graph_sqrt_lut u_sqrt_lut (
        .clk      (clk),
        .addr     (lut_addr),
        .data_out (sqrt_out)
    );

    graph_rsqrt_lut u_rsqrt_lut (
        .clk      (clk),
        .addr     (lut_addr),
        .data_out (rsqrt_out)
    );

    // =====================================================================
    // LUT output mux based on opcode
    // =====================================================================
    always_comb begin
        case (r_opcode)
            OP_G_EXP:   lut_result = exp_out;
            OP_G_LOG:   lut_result = log_out;
            OP_G_SQRT:  lut_result = sqrt_out;
            OP_G_RSQRT: lut_result = rsqrt_out;
            default:     lut_result = 8'h00;
        endcase
    end

    // =====================================================================
    // FSM next-state logic
    // =====================================================================
    always_comb begin
        w_state = r_state;
        case (r_state)
            ME_IDLE: begin
                if (cmd_valid)
                    w_state = ME_READ;
            end
            ME_READ: begin
                // SRAM read issued this cycle; need 1 cycle for registered output
                w_state = ME_SRAM_WAIT;
            end
            ME_SRAM_WAIT: begin
                // SRAM registered output now valid
                w_state = ME_LUT_ADDR;
            end
            ME_LUT_ADDR: begin
                // Latch SRAM data and present to LUT; need 1 cycle for LUT output
                w_state = ME_LUT_WAIT;
            end
            ME_LUT_WAIT: begin
                // LUT registered output now valid
                w_state = ME_WRITE;
            end
            ME_WRITE: begin
                // Write LUT result to SRAM, advance to next element or finish
                if (r_index + 1 >= r_length)
                    w_state = ME_DONE;
                else
                    w_state = ME_READ;
            end
            ME_DONE: begin
                w_state = ME_IDLE;
            end
            default: w_state = ME_IDLE;
        endcase
    end

    // =====================================================================
    // FSM registered outputs
    // =====================================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            r_state    <= ME_IDLE;
            r_opcode   <= 8'h00;
            r_src_base <= 16'h0000;
            r_dst_base <= 16'h0000;
            r_length   <= 16'h0000;
            r_index    <= 16'h0000;
            r_rd_data  <= 8'h00;
            sram_rd_en   <= 1'b0;
            sram_rd_addr <= '0;
            sram_wr_en   <= 1'b0;
            sram_wr_addr <= '0;
            sram_wr_data <= 8'h00;
            lut_addr     <= 8'h00;
        end else begin
            r_state <= w_state;

            // Default: de-assert strobes
            sram_rd_en <= 1'b0;
            sram_wr_en <= 1'b0;

            case (r_state)
                ME_IDLE: begin
                    if (cmd_valid) begin
                        r_opcode   <= cmd_opcode;
                        r_src_base <= cmd_src_base;
                        r_dst_base <= cmd_dst_base;
                        r_length   <= cmd_length;
                        r_index    <= 16'h0000;
                    end
                end

                ME_READ: begin
                    // Issue SRAM read for current element
                    sram_rd_en   <= 1'b1;
                    sram_rd_addr <= r_src_base[SRAM0_AW-1:0] + r_index[SRAM0_AW-1:0];
                end

                ME_SRAM_WAIT: begin
                    // Wait for SRAM synchronous read (registered output)
                    // sram_rd_data will be valid at start of next state
                end

                ME_LUT_ADDR: begin
                    // SRAM data is now valid; latch and drive LUT address
                    r_rd_data <= sram_rd_data;
                    lut_addr  <= sram_rd_data;
                end

                ME_LUT_WAIT: begin
                    // Wait one cycle for LUT registered output
                end

                ME_WRITE: begin
                    // Write LUT result to destination SRAM
                    sram_wr_en   <= 1'b1;
                    sram_wr_addr <= r_dst_base[SRAM0_AW-1:0] + r_index[SRAM0_AW-1:0];
                    sram_wr_data <= lut_result;
                    r_index      <= r_index + 16'd1;
                end

                ME_DONE: begin
                    // Single-cycle done pulse, return to idle
                end

                default: ;
            endcase
        end
    end

    // =====================================================================
    // Status outputs
    // =====================================================================
    assign busy = (r_state != ME_IDLE);
    assign done = (r_state == ME_DONE);

endmodule

`default_nettype wire
