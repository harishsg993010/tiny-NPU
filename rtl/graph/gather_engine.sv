// =============================================================================
// gather_engine.sv - Axis-0 row gather engine
// For each index in the index array, copy one row (row_size bytes) from src to
// dst.  Axis=0 only: index selects which row.  Out-of-bounds indices (>=
// num_rows) produce a zero-filled row in the destination.
// FSM: GA_IDLE -> GA_RD_IDX -> GA_LATCH_IDX -> GA_COPY_RD -> GA_COPY_WR ->
//      GA_NEXT_IDX -> GA_DONE
// =============================================================================
`default_nettype none

module gather_engine #(
    parameter int SRAM0_AW = 16
)(
    input  wire                     clk,
    input  wire                     rst_n,

    // Command interface
    input  wire                     cmd_valid,
    input  wire  [15:0]             cmd_src_base,
    input  wire  [15:0]             cmd_idx_base,
    input  wire  [15:0]             cmd_dst_base,
    input  wire  [15:0]             cmd_num_indices,
    input  wire  [15:0]             cmd_row_size,
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
        GA_IDLE,
        GA_RD_IDX,
        GA_LATCH_IDX,
        GA_COPY_RD,
        GA_COPY_WR,
        GA_NEXT_IDX,
        GA_DONE
    } ga_state_t;

    ga_state_t state, state_next;

    // =========================================================================
    // Registered command parameters
    // =========================================================================
    logic [15:0] r_src_base;
    logic [15:0] r_idx_base;
    logic [15:0] r_dst_base;
    logic [15:0] r_num_indices;
    logic [15:0] r_row_size;
    logic [15:0] r_num_rows;

    // =========================================================================
    // Working registers
    // =========================================================================
    logic [15:0] idx_cnt;       // current index counter (0..num_indices-1)
    logic [7:0]  index_val;     // latched index value read from SRAM
    logic        oob;           // index_val >= num_rows (out-of-bounds)
    logic [15:0] byte_cnt;      // current byte within the row being copied

    // =========================================================================
    // State register
    // =========================================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            state <= GA_IDLE;
        else
            state <= state_next;
    end

    // =========================================================================
    // Next-state logic
    // =========================================================================
    always_comb begin
        state_next = state;
        case (state)
            GA_IDLE: begin
                if (cmd_valid)
                    state_next = GA_RD_IDX;
            end

            GA_RD_IDX: begin
                // Issue SRAM read for index_array[idx_cnt]
                state_next = GA_LATCH_IDX;
            end

            GA_LATCH_IDX: begin
                // sram_rd_data now holds the index value; latch it
                if (r_row_size == 16'd0)
                    state_next = GA_NEXT_IDX;
                else
                    state_next = GA_COPY_RD;
            end

            GA_COPY_RD: begin
                // Issue SRAM read for src row byte (or skip if oob)
                state_next = GA_COPY_WR;
            end

            GA_COPY_WR: begin
                // Write byte to dst; check if row complete
                if (byte_cnt >= r_row_size - 16'd1)
                    state_next = GA_NEXT_IDX;
                else
                    state_next = GA_COPY_RD;
            end

            GA_NEXT_IDX: begin
                if (idx_cnt >= r_num_indices - 16'd1)
                    state_next = GA_DONE;
                else
                    state_next = GA_RD_IDX;
            end

            GA_DONE: begin
                state_next = GA_IDLE;
            end

            default: state_next = GA_IDLE;
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
        busy         = (state != GA_IDLE);
        done         = (state == GA_DONE);

        case (state)
            GA_RD_IDX: begin
                // Read index_array[idx_cnt] from SRAM
                sram_rd_en   = 1'b1;
                sram_rd_addr = r_idx_base[SRAM0_AW-1:0] + idx_cnt[SRAM0_AW-1:0];
            end

            GA_COPY_RD: begin
                if (!oob) begin
                    // Read src_base + index_val * row_size + byte_cnt
                    sram_rd_en   = 1'b1;
                    sram_rd_addr = r_src_base[SRAM0_AW-1:0] +
                                   {{(SRAM0_AW-8){1'b0}}, index_val} * r_row_size[SRAM0_AW-1:0] +
                                   byte_cnt[SRAM0_AW-1:0];
                end
                // If oob, no read needed -- we will write zero
            end

            GA_COPY_WR: begin
                sram_wr_en   = 1'b1;
                sram_wr_addr = r_dst_base[SRAM0_AW-1:0] +
                               idx_cnt[SRAM0_AW-1:0] * r_row_size[SRAM0_AW-1:0] +
                               byte_cnt[SRAM0_AW-1:0];
                sram_wr_data = oob ? 8'd0 : sram_rd_data;
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
            r_idx_base    <= '0;
            r_dst_base    <= '0;
            r_num_indices <= '0;
            r_row_size    <= '0;
            r_num_rows    <= '0;
            idx_cnt       <= '0;
            index_val     <= '0;
            oob           <= 1'b0;
            byte_cnt      <= '0;
        end else begin
            case (state)
                GA_IDLE: begin
                    if (cmd_valid) begin
                        r_src_base    <= cmd_src_base;
                        r_idx_base    <= cmd_idx_base;
                        r_dst_base    <= cmd_dst_base;
                        r_num_indices <= cmd_num_indices;
                        r_row_size    <= cmd_row_size;
                        r_num_rows    <= cmd_num_rows;
                        idx_cnt       <= '0;
                    end
                end

                GA_RD_IDX: begin
                    // SRAM read issued combinationally; data available next cycle
                end

                GA_LATCH_IDX: begin
                    // Capture the index value from SRAM read data
                    index_val <= sram_rd_data;
                    oob       <= ({8'd0, sram_rd_data} >= r_num_rows);
                    byte_cnt  <= '0;
                end

                GA_COPY_RD: begin
                    // SRAM read issued combinationally; data available next cycle
                end

                GA_COPY_WR: begin
                    // Write issued combinationally; advance byte counter
                    byte_cnt <= byte_cnt + 16'd1;
                end

                GA_NEXT_IDX: begin
                    idx_cnt <= idx_cnt + 16'd1;
                end

                default: ;
            endcase
        end
    end

endmodule

`default_nettype wire
