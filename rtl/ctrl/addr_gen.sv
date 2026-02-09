// =============================================================================
// addr_gen.sv - Address Generator for Tiled Access Patterns
// =============================================================================
// Generates addresses for row-major (or column-major) tiled SRAM access.
// Iterates over a 2D tile producing:
//   addr = base_addr + row * stride_row + col * stride_col
// for row = 0..num_rows-1, col = 0..num_cols-1.
//
// Supports backpressure via addr_ready. Asserts tile_done when all addresses
// in the tile have been generated.
// =============================================================================

module addr_gen
  import npu_pkg::*;
  import isa_pkg::*;
#(
    parameter int ADDR_W = 16,
    parameter int TILE_M = 16,
    parameter int TILE_N = 16
) (
    input  logic              clk,
    input  logic              rst_n,

    // --- Configuration (latched on start) ---
    input  logic [ADDR_W-1:0] base_addr,
    input  logic [ADDR_W-1:0] stride_row,
    input  logic [ADDR_W-1:0] stride_col,
    input  logic [15:0]       num_rows,
    input  logic [15:0]       num_cols,
    input  logic              start,       // pulse: begin address generation

    // --- Downstream handshake ---
    output logic              addr_valid,
    input  logic              addr_ready,  // backpressure from consumer
    output logic [ADDR_W-1:0] addr,
    output logic [15:0]       row_idx,
    output logic [15:0]       col_idx,
    output logic              tile_done
);

    // -------------------------------------------------------------------------
    // FSM state encoding
    // -------------------------------------------------------------------------
    typedef enum logic [1:0] {
        A_IDLE    = 2'd0,
        A_GEN     = 2'd1,
        A_DONE    = 2'd2
    } addr_state_e;

    addr_state_e state_q, state_d;

    // -------------------------------------------------------------------------
    // Configuration registers (latched on start)
    // -------------------------------------------------------------------------
    logic [ADDR_W-1:0] base_q,       base_d;
    logic [ADDR_W-1:0] stride_row_q, stride_row_d;
    logic [ADDR_W-1:0] stride_col_q, stride_col_d;
    logic [15:0]       nrows_q,      nrows_d;
    logic [15:0]       ncols_q,      ncols_d;

    // -------------------------------------------------------------------------
    // Iteration counters
    // -------------------------------------------------------------------------
    logic [15:0] row_q, row_d;
    logic [15:0] col_q, col_d;

    // -------------------------------------------------------------------------
    // Address computation (combinational)
    // -------------------------------------------------------------------------
    logic [ADDR_W-1:0] computed_addr;
    assign computed_addr = base_q
                         + (row_q * stride_row_q)
                         + (col_q * stride_col_q);

    // -------------------------------------------------------------------------
    // Last element detection
    // -------------------------------------------------------------------------
    logic last_col;
    logic last_row;
    assign last_col = (col_q == (ncols_q - 16'd1));
    assign last_row = (row_q == (nrows_q - 16'd1));

    // -------------------------------------------------------------------------
    // Registered outputs
    // -------------------------------------------------------------------------
    logic              addr_valid_q, addr_valid_d;
    logic [ADDR_W-1:0] addr_q,       addr_d;
    logic [15:0]       row_idx_q,    row_idx_d;
    logic [15:0]       col_idx_q,    col_idx_d;
    logic              tile_done_q,  tile_done_d;

    // -------------------------------------------------------------------------
    // FSM next-state and output logic
    // -------------------------------------------------------------------------
    always_comb begin
        state_d      = state_q;
        base_d       = base_q;
        stride_row_d = stride_row_q;
        stride_col_d = stride_col_q;
        nrows_d      = nrows_q;
        ncols_d      = ncols_q;
        row_d        = row_q;
        col_d        = col_q;

        addr_valid_d = 1'b0;
        addr_d       = addr_q;
        row_idx_d    = row_idx_q;
        col_idx_d    = col_idx_q;
        tile_done_d  = 1'b0;

        case (state_q)
            // -----------------------------------------------------------------
            // IDLE: Wait for start pulse, latch configuration
            // -----------------------------------------------------------------
            A_IDLE: begin
                if (start) begin
                    base_d       = base_addr;
                    stride_row_d = stride_row;
                    stride_col_d = stride_col;
                    nrows_d      = num_rows;
                    ncols_d      = num_cols;
                    row_d        = '0;
                    col_d        = '0;

                    // Handle degenerate case: zero-size tile
                    if (num_rows == '0 || num_cols == '0) begin
                        state_d = A_DONE;
                    end else begin
                        state_d = A_GEN;
                    end
                end
            end

            // -----------------------------------------------------------------
            // GEN: Generate one address per cycle (when consumer is ready)
            // -----------------------------------------------------------------
            A_GEN: begin
                addr_valid_d = 1'b1;
                addr_d       = computed_addr;
                row_idx_d    = row_q;
                col_idx_d    = col_q;

                if (addr_ready) begin
                    if (last_col && last_row) begin
                        // Final address in the tile
                        state_d = A_DONE;
                    end else if (last_col) begin
                        // End of row: advance to next row, reset column
                        col_d = '0;
                        row_d = row_q + 16'd1;
                    end else begin
                        // Advance column
                        col_d = col_q + 16'd1;
                    end
                end
                // If !addr_ready, hold current address (backpressure)
            end

            // -----------------------------------------------------------------
            // DONE: Assert tile_done for one cycle, return to IDLE
            // -----------------------------------------------------------------
            A_DONE: begin
                tile_done_d = 1'b1;
                state_d     = A_IDLE;
            end

            default: begin
                state_d = A_IDLE;
            end
        endcase
    end

    // -------------------------------------------------------------------------
    // Sequential logic
    // -------------------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state_q      <= A_IDLE;
            base_q       <= '0;
            stride_row_q <= '0;
            stride_col_q <= '0;
            nrows_q      <= '0;
            ncols_q      <= '0;
            row_q        <= '0;
            col_q        <= '0;
            addr_valid_q <= 1'b0;
            addr_q       <= '0;
            row_idx_q    <= '0;
            col_idx_q    <= '0;
            tile_done_q  <= 1'b0;
        end else begin
            state_q      <= state_d;
            base_q       <= base_d;
            stride_row_q <= stride_row_d;
            stride_col_q <= stride_col_d;
            nrows_q      <= nrows_d;
            ncols_q      <= ncols_d;
            row_q        <= row_d;
            col_q        <= col_d;
            addr_valid_q <= addr_valid_d;
            addr_q       <= addr_d;
            row_idx_q    <= row_idx_d;
            col_idx_q    <= col_idx_d;
            tile_done_q  <= tile_done_d;
        end
    end

    // -------------------------------------------------------------------------
    // Output assignments
    // -------------------------------------------------------------------------
    assign addr_valid = addr_valid_q;
    assign addr       = addr_q;
    assign row_idx    = row_idx_q;
    assign col_idx    = col_idx_q;
    assign tile_done  = tile_done_q;

endmodule
