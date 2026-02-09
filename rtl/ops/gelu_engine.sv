// =============================================================================
// gelu_engine.sv - GELU activation engine wrapper
// Reads int8 from SRAM, applies gelu_lut, writes int8 back.
// One element per cycle. FSM: IDLE -> PROCESS -> DONE
// =============================================================================
import npu_pkg::*;
import fixed_pkg::*;

module gelu_engine (
    input  logic               clk,
    input  logic               rst_n,

    // Command interface
    input  logic               cmd_valid,
    output logic               cmd_ready,
    input  logic [15:0]        length,       // number of elements
    input  logic [15:0]        src_base,     // SRAM read base address
    input  logic [15:0]        dst_base,     // SRAM write base address

    // SRAM read port
    output logic               sram_rd_en,
    output logic [15:0]        sram_rd_addr,
    input  logic [DATA_W-1:0]  sram_rd_data,

    // SRAM write port
    output logic               sram_wr_en,
    output logic [15:0]        sram_wr_addr,
    output logic [DATA_W-1:0]  sram_wr_data,

    // Status
    output logic               busy,
    output logic               done
);

    // ----------------------------------------------------------------
    // FSM States
    // ----------------------------------------------------------------
    typedef enum logic [2:0] {
        S_IDLE,
        S_READ,       // Issue SRAM read address
        S_LUT_WAIT,   // Wait for SRAM data (1-cycle), feed to LUT
        S_LUT_READ,   // Wait for LUT output (1-cycle latency)
        S_WRITE,      // Write LUT result to output SRAM
        S_DONE
    } state_t;

    state_t state, state_nxt;

    // ----------------------------------------------------------------
    // Registered command parameters
    // ----------------------------------------------------------------
    logic [15:0] r_length, r_src_base, r_dst_base;
    logic [15:0] r_idx;

    // ----------------------------------------------------------------
    // GELU LUT signals
    // ----------------------------------------------------------------
    logic [7:0]  lut_addr;
    logic [7:0]  lut_data;

    // Pipeline register
    logic [7:0]  p_rd_data;

    // ----------------------------------------------------------------
    // GELU LUT instantiation
    // ----------------------------------------------------------------
    gelu_lut u_gelu_lut (
        .clk      (clk),
        .addr     (lut_addr),
        .data_out (lut_data)
    );

    // ----------------------------------------------------------------
    // FSM transition
    // ----------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n)
        if (!rst_n) state <= S_IDLE;
        else        state <= state_nxt;

    always_comb begin
        state_nxt = state;
        case (state)
            S_IDLE:     if (cmd_valid) state_nxt = S_READ;
            S_READ:     state_nxt = S_LUT_WAIT;    // SRAM read latency
            S_LUT_WAIT: state_nxt = S_LUT_READ;    // LUT read latency
            S_LUT_READ: state_nxt = S_WRITE;       // LUT output ready
            S_WRITE: begin
                if (r_idx == r_length - 16'd1)
                    state_nxt = S_DONE;
                else
                    state_nxt = S_READ;
            end
            S_DONE:     state_nxt = S_IDLE;
            default:    state_nxt = S_IDLE;
        endcase
    end

    // ----------------------------------------------------------------
    // Command capture
    // ----------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            r_length   <= '0;
            r_src_base <= '0;
            r_dst_base <= '0;
        end else if (state == S_IDLE && cmd_valid) begin
            r_length   <= length;
            r_src_base <= src_base;
            r_dst_base <= dst_base;
        end
    end

    // ----------------------------------------------------------------
    // Element index counter
    // ----------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n)
        if (!rst_n)
            r_idx <= '0;
        else if (state == S_IDLE && cmd_valid)
            r_idx <= '0;
        else if (state == S_WRITE && state_nxt == S_READ)
            r_idx <= r_idx + 16'd1;

    // ----------------------------------------------------------------
    // Latch SRAM read data
    // ----------------------------------------------------------------
    always_ff @(posedge clk) begin
        if (state == S_LUT_WAIT)
            p_rd_data <= sram_rd_data;
    end

    // ----------------------------------------------------------------
    // SRAM read control - issued during S_READ
    // ----------------------------------------------------------------
    assign sram_rd_en   = (state == S_READ);
    assign sram_rd_addr = r_src_base + r_idx;

    // ----------------------------------------------------------------
    // LUT address - driven during S_LUT_WAIT (data available from SRAM)
    // The LUT output will be ready in S_LUT_READ
    // ----------------------------------------------------------------
    assign lut_addr = (state == S_LUT_WAIT) ? sram_rd_data : p_rd_data;

    // ----------------------------------------------------------------
    // SRAM write - during S_WRITE, LUT output is stable
    // ----------------------------------------------------------------
    assign sram_wr_en   = (state == S_WRITE);
    assign sram_wr_addr = r_dst_base + r_idx;
    assign sram_wr_data = lut_data;

    // ----------------------------------------------------------------
    // Status and handshake
    // ----------------------------------------------------------------
    assign cmd_ready = (state == S_IDLE);
    assign busy      = (state != S_IDLE);
    assign done      = (state == S_DONE);

endmodule
