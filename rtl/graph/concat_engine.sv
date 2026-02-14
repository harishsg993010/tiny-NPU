// =============================================================================
// concat_engine.sv - Last-dimension concat engine
// For each row, copy src0_row_len bytes from src0, then src1_row_len bytes from
// src1 into a contiguous dst row of (src0_row_len + src1_row_len) bytes.
//
// For row r:
//   src0 region: src0_base + r * src0_row_len   [src0_row_len bytes]
//   src1 region: src1_base + r * src1_row_len   [src1_row_len bytes]
//   dst  region: dst_base  + r * (src0_row_len + src1_row_len)
//
// FSM: CT_IDLE -> CT_COPY_SRC0 -> CT_COPY_SRC1 -> CT_NEXT_ROW -> CT_DONE
//
// Each copy phase uses a 2-cycle read/write pattern per byte.
// =============================================================================
`default_nettype none

module concat_engine #(
    parameter int SRAM0_AW = 16
)(
    input  wire                     clk,
    input  wire                     rst_n,

    // Command interface
    input  wire                     cmd_valid,
    input  wire  [15:0]             cmd_src0_base,
    input  wire  [15:0]             cmd_src1_base,
    input  wire  [15:0]             cmd_dst_base,
    input  wire  [15:0]             cmd_src0_row_len,
    input  wire  [15:0]             cmd_src1_row_len,
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
        CT_IDLE,
        CT_COPY_SRC0,
        CT_COPY_SRC1,
        CT_NEXT_ROW,
        CT_DONE
    } ct_state_t;

    ct_state_t state, state_next;

    // =========================================================================
    // Registered command parameters
    // =========================================================================
    logic [15:0] r_src0_base;
    logic [15:0] r_src1_base;
    logic [15:0] r_dst_base;
    logic [15:0] r_src0_row_len;
    logic [15:0] r_src1_row_len;
    logic [15:0] r_num_rows;
    logic [15:0] r_dst_row_len;    // src0_row_len + src1_row_len (precomputed)

    // =========================================================================
    // Working registers
    // =========================================================================
    logic [15:0] row_idx;       // current row (0..num_rows-1)
    logic [15:0] byte_idx;      // current byte within the source segment
    logic        phase_wr;      // 0 = read cycle, 1 = write cycle

    // =========================================================================
    // State register
    // =========================================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            state <= CT_IDLE;
        else
            state <= state_next;
    end

    // =========================================================================
    // Next-state logic
    // =========================================================================
    always_comb begin
        state_next = state;
        case (state)
            CT_IDLE: begin
                if (cmd_valid)
                    state_next = CT_COPY_SRC0;
            end

            CT_COPY_SRC0: begin
                // 2-cycle per byte: read then write
                if (phase_wr && byte_idx >= r_src0_row_len - 16'd1) begin
                    // Finished copying src0 segment for this row
                    if (r_src1_row_len == 16'd0)
                        state_next = CT_NEXT_ROW;
                    else
                        state_next = CT_COPY_SRC1;
                end
                // If src0_row_len == 0, skip to src1
                if (r_src0_row_len == 16'd0) begin
                    if (r_src1_row_len == 16'd0)
                        state_next = CT_NEXT_ROW;
                    else
                        state_next = CT_COPY_SRC1;
                end
            end

            CT_COPY_SRC1: begin
                if (phase_wr && byte_idx >= r_src1_row_len - 16'd1) begin
                    state_next = CT_NEXT_ROW;
                end
                if (r_src1_row_len == 16'd0)
                    state_next = CT_NEXT_ROW;
            end

            CT_NEXT_ROW: begin
                if (row_idx >= r_num_rows - 16'd1)
                    state_next = CT_DONE;
                else
                    state_next = CT_COPY_SRC0;
            end

            CT_DONE: begin
                state_next = CT_IDLE;
            end

            default: state_next = CT_IDLE;
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
        busy         = (state != CT_IDLE);
        done         = (state == CT_DONE);

        case (state)
            CT_COPY_SRC0: begin
                if (r_src0_row_len != 16'd0) begin
                    if (!phase_wr) begin
                        // Read cycle: read from src0_base + row * src0_row_len + byte_idx
                        sram_rd_en   = 1'b1;
                        sram_rd_addr = r_src0_base[SRAM0_AW-1:0] +
                                       row_idx[SRAM0_AW-1:0] * r_src0_row_len[SRAM0_AW-1:0] +
                                       byte_idx[SRAM0_AW-1:0];
                    end else begin
                        // Write cycle: write to dst_base + row * dst_row_len + byte_idx
                        sram_wr_en   = 1'b1;
                        sram_wr_addr = r_dst_base[SRAM0_AW-1:0] +
                                       row_idx[SRAM0_AW-1:0] * r_dst_row_len[SRAM0_AW-1:0] +
                                       byte_idx[SRAM0_AW-1:0];
                        sram_wr_data = sram_rd_data;
                    end
                end
            end

            CT_COPY_SRC1: begin
                if (r_src1_row_len != 16'd0) begin
                    if (!phase_wr) begin
                        // Read cycle: read from src1_base + row * src1_row_len + byte_idx
                        sram_rd_en   = 1'b1;
                        sram_rd_addr = r_src1_base[SRAM0_AW-1:0] +
                                       row_idx[SRAM0_AW-1:0] * r_src1_row_len[SRAM0_AW-1:0] +
                                       byte_idx[SRAM0_AW-1:0];
                    end else begin
                        // Write cycle: write to dst_base + row * dst_row_len + src0_row_len + byte_idx
                        sram_wr_en   = 1'b1;
                        sram_wr_addr = r_dst_base[SRAM0_AW-1:0] +
                                       row_idx[SRAM0_AW-1:0] * r_dst_row_len[SRAM0_AW-1:0] +
                                       r_src0_row_len[SRAM0_AW-1:0] +
                                       byte_idx[SRAM0_AW-1:0];
                        sram_wr_data = sram_rd_data;
                    end
                end
            end

            default: ;
        endcase
    end

    // =========================================================================
    // Sequential logic
    // =========================================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            r_src0_base    <= '0;
            r_src1_base    <= '0;
            r_dst_base     <= '0;
            r_src0_row_len <= '0;
            r_src1_row_len <= '0;
            r_num_rows     <= '0;
            r_dst_row_len  <= '0;
            row_idx        <= '0;
            byte_idx       <= '0;
            phase_wr       <= 1'b0;
        end else begin
            case (state)
                CT_IDLE: begin
                    if (cmd_valid) begin
                        r_src0_base    <= cmd_src0_base;
                        r_src1_base    <= cmd_src1_base;
                        r_dst_base     <= cmd_dst_base;
                        r_src0_row_len <= cmd_src0_row_len;
                        r_src1_row_len <= cmd_src1_row_len;
                        r_num_rows     <= cmd_num_rows;
                        r_dst_row_len  <= cmd_src0_row_len + cmd_src1_row_len;
                        row_idx        <= '0;
                        byte_idx       <= '0;
                        phase_wr       <= 1'b0;
                    end
                end

                CT_COPY_SRC0: begin
                    if (r_src0_row_len != 16'd0) begin
                        if (!phase_wr) begin
                            // Read cycle done; next cycle is write
                            phase_wr <= 1'b1;
                        end else begin
                            // Write cycle done; advance byte or finish segment
                            phase_wr <= 1'b0;
                            if (byte_idx >= r_src0_row_len - 16'd1) begin
                                // Segment complete; reset byte_idx for src1
                                byte_idx <= '0;
                            end else begin
                                byte_idx <= byte_idx + 16'd1;
                            end
                        end
                    end
                end

                CT_COPY_SRC1: begin
                    if (r_src1_row_len != 16'd0) begin
                        if (!phase_wr) begin
                            phase_wr <= 1'b1;
                        end else begin
                            phase_wr <= 1'b0;
                            if (byte_idx >= r_src1_row_len - 16'd1) begin
                                byte_idx <= '0;
                            end else begin
                                byte_idx <= byte_idx + 16'd1;
                            end
                        end
                    end
                end

                CT_NEXT_ROW: begin
                    row_idx  <= row_idx + 16'd1;
                    byte_idx <= '0;
                    phase_wr <= 1'b0;
                end

                default: ;
            endcase
        end
    end

endmodule

`default_nettype wire
