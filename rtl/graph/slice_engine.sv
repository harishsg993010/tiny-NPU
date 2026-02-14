// =============================================================================
// slice_engine.sv - Materialized N-D slice engine
// For each row: copy dst_row_len bytes from
//   src_base + row * src_row_len + start_offset
// to
//   dst_base + row * dst_row_len
// FSM: SL_IDLE -> SL_READ -> SL_WRITE -> (loop or SL_NEXT_ROW) -> SL_DONE
// =============================================================================
`default_nettype none

module slice_engine #(
    parameter int SRAM0_AW = 16
)(
    input  wire                     clk,
    input  wire                     rst_n,

    // Command interface
    input  wire                     cmd_valid,
    input  wire  [15:0]             cmd_src_base,
    input  wire  [15:0]             cmd_dst_base,
    input  wire  [15:0]             cmd_src_row_len,
    input  wire  [15:0]             cmd_dst_row_len,
    input  wire  [15:0]             cmd_start_offset,
    input  wire  [15:0]             cmd_num_rows,

    // SRAM0 read port
    output logic                    sram_rd_en,
    output logic [SRAM0_AW-1:0]    sram_rd_addr,
    input  wire  [7:0]             sram_rd_data,

    // SRAM0 write port
    output logic                    sram_wr_en,
    output logic [SRAM0_AW-1:0]    sram_wr_addr,
    output logic [7:0]             sram_wr_data,

    // Status
    output logic                    busy,
    output logic                    done
);

    import graph_isa_pkg::*;

    // =========================================================================
    // FSM states
    // =========================================================================
    typedef enum logic [2:0] {
        SL_IDLE,
        SL_READ,
        SL_WRITE,
        SL_NEXT_ROW,
        SL_DONE
    } sl_state_t;

    sl_state_t state, state_next;

    // =========================================================================
    // Registered command parameters
    // =========================================================================
    logic [15:0] r_src_base;
    logic [15:0] r_dst_base;
    logic [15:0] r_src_row_len;
    logic [15:0] r_dst_row_len;
    logic [15:0] r_start_offset;
    logic [15:0] r_num_rows;

    // =========================================================================
    // Working registers
    // =========================================================================
    logic [15:0] row_idx;       // current row (0..num_rows-1)
    logic [15:0] byte_idx;      // current byte within the slice (0..dst_row_len-1)

    // =========================================================================
    // State register
    // =========================================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            state <= SL_IDLE;
        else
            state <= state_next;
    end

    // =========================================================================
    // Next-state logic
    // =========================================================================
    always_comb begin
        state_next = state;
        case (state)
            SL_IDLE: begin
                if (cmd_valid)
                    state_next = SL_READ;
            end

            SL_READ: begin
                // Issue SRAM read; data available next cycle
                state_next = SL_WRITE;
            end

            SL_WRITE: begin
                // Write the read byte to dst; check if row is complete
                if (byte_idx >= r_dst_row_len - 16'd1)
                    state_next = SL_NEXT_ROW;
                else
                    state_next = SL_READ;
            end

            SL_NEXT_ROW: begin
                if (row_idx >= r_num_rows - 16'd1)
                    state_next = SL_DONE;
                else
                    state_next = SL_READ;
            end

            SL_DONE: begin
                state_next = SL_IDLE;
            end

            default: state_next = SL_IDLE;
        endcase
    end

    // =========================================================================
    // Output logic (combinational)
    // =========================================================================
    always_comb begin
        sram_rd_en   = 1'b0;
        sram_rd_addr = '0;
        sram_wr_en   = 1'b0;
        sram_wr_addr = '0;
        sram_wr_data = '0;
        busy         = (state != SL_IDLE);
        done         = (state == SL_DONE);

        case (state)
            SL_READ: begin
                // Read src_base + row_idx * src_row_len + start_offset + byte_idx
                sram_rd_en   = 1'b1;
                sram_rd_addr = r_src_base[SRAM0_AW-1:0] +
                               row_idx[SRAM0_AW-1:0] * r_src_row_len[SRAM0_AW-1:0] +
                               r_start_offset[SRAM0_AW-1:0] +
                               byte_idx[SRAM0_AW-1:0];
            end

            SL_WRITE: begin
                // Write to dst_base + row_idx * dst_row_len + byte_idx
                sram_wr_en   = 1'b1;
                sram_wr_addr = r_dst_base[SRAM0_AW-1:0] +
                               row_idx[SRAM0_AW-1:0] * r_dst_row_len[SRAM0_AW-1:0] +
                               byte_idx[SRAM0_AW-1:0];
                sram_wr_data = sram_rd_data;
            end

            default: ;
        endcase
    end

    // =========================================================================
    // Sequential logic
    // =========================================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            r_src_base    <= '0;
            r_dst_base    <= '0;
            r_src_row_len <= '0;
            r_dst_row_len <= '0;
            r_start_offset <= '0;
            r_num_rows    <= '0;
            row_idx       <= '0;
            byte_idx      <= '0;
        end else begin
            case (state)
                SL_IDLE: begin
                    if (cmd_valid) begin
                        r_src_base     <= cmd_src_base;
                        r_dst_base     <= cmd_dst_base;
                        r_src_row_len  <= cmd_src_row_len;
                        r_dst_row_len  <= cmd_dst_row_len;
                        r_start_offset <= cmd_start_offset;
                        r_num_rows     <= cmd_num_rows;
                        row_idx        <= '0;
                        byte_idx       <= '0;
                    end
                end

                SL_READ: begin
                    // SRAM read issued combinationally; data available next cycle
                end

                SL_WRITE: begin
                    // Advance byte counter
                    byte_idx <= byte_idx + 16'd1;
                end

                SL_NEXT_ROW: begin
                    row_idx  <= row_idx + 16'd1;
                    byte_idx <= '0;
                end

                default: ;
            endcase
        end
    end

endmodule

`default_nettype wire
