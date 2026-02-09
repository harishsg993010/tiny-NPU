// =============================================================================
// ucode_fetch.sv - Microcode Fetch Unit
// =============================================================================
// Fetches 128-bit microcode instructions from SRAM and presents them to the
// decode stage with backpressure support. Uses a 6-state FSM assuming
// single-cycle SRAM read latency.
// =============================================================================

module ucode_fetch
  import npu_pkg::*;
  import isa_pkg::*;
#(
    parameter int INSTR_W     = 128,
    parameter int SRAM_ADDR_W = 12
) (
    input  logic                    clk,
    input  logic                    rst_n,

    // --- Control interface (from ctrl registers) ---
    input  logic                    start,
    input  logic                    stop,
    input  logic [SRAM_ADDR_W-1:0] ucode_base_addr,
    input  logic [SRAM_ADDR_W-1:0] ucode_len,

    // --- UCODE SRAM read interface ---
    output logic                    rd_en,
    output logic [SRAM_ADDR_W-1:0] rd_addr,
    input  logic [INSTR_W-1:0]     rd_data,
    input  logic                    rd_valid,

    // --- To decode stage ---
    output logic                    instr_valid,
    input  logic                    instr_ready,
    output logic [INSTR_W-1:0]     instr_data,
    output logic [SRAM_ADDR_W-1:0] pc,

    // --- Status ---
    output logic                    done,
    output logic                    busy
);

    // -------------------------------------------------------------------------
    // FSM state encoding
    // -------------------------------------------------------------------------
    typedef enum logic [2:0] {
        S_IDLE      = 3'd0,
        S_FETCH     = 3'd1,
        S_WAIT_DATA = 3'd2,
        S_DISPATCH  = 3'd3,
        S_INCREMENT = 3'd4,
        S_DONE      = 3'd5
    } fetch_state_e;

    fetch_state_e state_q, state_d;

    // -------------------------------------------------------------------------
    // Internal registers
    // -------------------------------------------------------------------------
    logic [SRAM_ADDR_W-1:0] pc_q,        pc_d;
    logic [SRAM_ADDR_W-1:0] base_q,      base_d;
    logic [SRAM_ADDR_W-1:0] len_q,       len_d;
    logic [INSTR_W-1:0]     instr_buf_q, instr_buf_d;
    logic                    done_q,      done_d;

    // OP_END detection on fetched data
    logic fetched_is_end;
    assign fetched_is_end = (rd_data[7:0] == OP_END);

    // Next PC computation
    logic [SRAM_ADDR_W-1:0] pc_plus1;
    assign pc_plus1 = pc_q + {{(SRAM_ADDR_W-1){1'b0}}, 1'b1};

    // -------------------------------------------------------------------------
    // FSM next-state and output logic
    // -------------------------------------------------------------------------
    always_comb begin
        state_d     = state_q;
        pc_d        = pc_q;
        base_d      = base_q;
        len_d       = len_q;
        instr_buf_d = instr_buf_q;
        done_d      = 1'b0;

        rd_en       = 1'b0;
        rd_addr     = '0;
        instr_valid = 1'b0;
        instr_data  = instr_buf_q;

        case (state_q)
            S_IDLE: begin
                if (start) begin
                    pc_d    = '0;
                    base_d  = ucode_base_addr;
                    len_d   = ucode_len;
                    state_d = S_FETCH;
                end
            end

            S_FETCH: begin
                rd_en   = 1'b1;
                rd_addr = base_q + pc_q;
                state_d = S_WAIT_DATA;
            end

            S_WAIT_DATA: begin
                if (stop) begin
                    state_d = S_DONE;
                end else if (rd_valid) begin
                    instr_buf_d = rd_data;
                    if (fetched_is_end) begin
                        state_d = S_DONE;
                    end else begin
                        state_d = S_DISPATCH;
                    end
                end
            end

            S_DISPATCH: begin
                instr_valid = 1'b1;
                instr_data  = instr_buf_q;
                if (stop) begin
                    state_d = S_DONE;
                end else if (instr_ready) begin
                    state_d = S_INCREMENT;
                end
            end

            S_INCREMENT: begin
                pc_d = pc_plus1;
                if (stop) begin
                    state_d = S_DONE;
                end else if (pc_plus1 >= len_q) begin
                    state_d = S_DONE;
                end else begin
                    state_d = S_FETCH;
                end
            end

            S_DONE: begin
                done_d  = 1'b1;
                state_d = S_IDLE;
            end

            default: begin
                state_d = S_IDLE;
            end
        endcase

        // Global stop override from any active state
        if (stop && (state_q != S_IDLE) && (state_q != S_DONE)) begin
            state_d = S_DONE;
        end
    end

    // -------------------------------------------------------------------------
    // Sequential logic
    // -------------------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state_q     <= S_IDLE;
            pc_q        <= '0;
            base_q      <= '0;
            len_q       <= '0;
            instr_buf_q <= '0;
            done_q      <= 1'b0;
        end else begin
            state_q     <= state_d;
            pc_q        <= pc_d;
            base_q      <= base_d;
            len_q       <= len_d;
            instr_buf_q <= instr_buf_d;
            done_q      <= done_d;
        end
    end

    // -------------------------------------------------------------------------
    // Output assignments
    // -------------------------------------------------------------------------
    assign pc   = pc_q;
    assign done = done_q;
    assign busy = (state_q != S_IDLE);

endmodule
