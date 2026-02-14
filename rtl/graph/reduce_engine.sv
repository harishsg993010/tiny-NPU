// =============================================================================
// reduce_engine.sv - Reduction engine for SUM/MAX/MEAN operations
// INT8 input, INT32 accumulator, INT8 output
// FSM: RE_IDLE → RE_OUTER_INIT → RE_READ → RE_ACCUM → RE_WRITE → RE_NEXT_OUTER → RE_DONE
// =============================================================================
`default_nettype none

module reduce_engine #(
    parameter int SRAM0_AW = 16
)(
    input  wire                     clk,
    input  wire                     rst_n,

    // Command interface
    input  wire                     cmd_valid,
    input  wire  [7:0]              cmd_opcode,
    input  wire  [15:0]             cmd_src_base,
    input  wire  [15:0]             cmd_dst_base,
    input  wire  [15:0]             cmd_reduce_dim,
    input  wire  [15:0]             cmd_outer_count,

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

    typedef enum logic [2:0] {
        RE_IDLE,
        RE_OUTER_INIT,
        RE_READ,
        RE_ACCUM,
        RE_WRITE,
        RE_NEXT_OUTER,
        RE_DONE
    } re_state_t;

    re_state_t state, state_next;

    // Registered command
    logic [7:0]  r_opcode;
    logic [15:0] r_src_base, r_dst_base;
    logic [15:0] r_reduce_dim, r_outer_count;

    // Loop counters
    logic [15:0] outer_idx;
    logic [15:0] inner_idx;

    // Accumulator (INT32 for SUM/MEAN, INT8 for MAX)
    logic signed [31:0] acc;

    // Pipeline: data valid after 1-cycle SRAM read latency
    logic rd_valid;

    // =========================================================================
    // State register
    // =========================================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            state <= RE_IDLE;
        else
            state <= state_next;
    end

    // =========================================================================
    // Next-state logic
    // =========================================================================
    always_comb begin
        state_next = state;
        case (state)
            RE_IDLE: begin
                if (cmd_valid)
                    state_next = RE_OUTER_INIT;
            end
            RE_OUTER_INIT: begin
                state_next = RE_READ;
            end
            RE_READ: begin
                state_next = RE_ACCUM;
            end
            RE_ACCUM: begin
                if (inner_idx >= r_reduce_dim - 16'd1)
                    state_next = RE_WRITE;
                else
                    state_next = RE_READ;
            end
            RE_WRITE: begin
                state_next = RE_NEXT_OUTER;
            end
            RE_NEXT_OUTER: begin
                if (outer_idx >= r_outer_count - 16'd1)
                    state_next = RE_DONE;
                else
                    state_next = RE_OUTER_INIT;
            end
            RE_DONE: begin
                state_next = RE_IDLE;
            end
            default: state_next = RE_IDLE;
        endcase
    end

    // =========================================================================
    // Output logic
    // =========================================================================
    always_comb begin
        sram_rd_en   = 1'b0;
        sram_rd_addr = '0;
        sram_wr_en   = 1'b0;
        sram_wr_addr = '0;
        sram_wr_data = '0;
        busy         = (state != RE_IDLE);
        done         = (state == RE_DONE);

        case (state)
            RE_READ: begin
                sram_rd_en   = 1'b1;
                sram_rd_addr = r_src_base[SRAM0_AW-1:0] +
                               outer_idx * r_reduce_dim + inner_idx;
            end
            RE_WRITE: begin
                sram_wr_en   = 1'b1;
                sram_wr_addr = r_dst_base[SRAM0_AW-1:0] + outer_idx;
                // Compute output based on opcode
                begin
                    automatic logic signed [31:0] result;
                    case (r_opcode)
                        OP_G_REDUCE_SUM: begin
                            // (acc + rounding) >> shift
                            // shift = ceil(log2(reduce_dim)) approximated
                            // For simplicity: just saturate to INT8
                            result = acc;
                            if (result > 32'sd127)
                                sram_wr_data = 8'sd127;
                            else if (result < -32'sd128)
                                sram_wr_data = -8'sd128;
                            else
                                sram_wr_data = result[7:0];
                        end
                        OP_G_REDUCE_MAX: begin
                            sram_wr_data = acc[7:0];
                        end
                        OP_G_REDUCE_MEAN: begin
                            // (acc + reduce_dim/2) / reduce_dim
                            result = (acc + {{16{1'b0}}, r_reduce_dim[15:1]}) /
                                     $signed({{16{1'b0}}, r_reduce_dim});
                            if (result > 32'sd127)
                                sram_wr_data = 8'sd127;
                            else if (result < -32'sd128)
                                sram_wr_data = -8'sd128;
                            else
                                sram_wr_data = result[7:0];
                        end
                        default: sram_wr_data = acc[7:0];
                    endcase
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
            r_opcode      <= '0;
            r_src_base    <= '0;
            r_dst_base    <= '0;
            r_reduce_dim  <= '0;
            r_outer_count <= '0;
            outer_idx     <= '0;
            inner_idx     <= '0;
            acc           <= '0;
            rd_valid      <= 1'b0;
        end else begin
            rd_valid <= 1'b0;

            case (state)
                RE_IDLE: begin
                    if (cmd_valid) begin
                        r_opcode      <= cmd_opcode;
                        r_src_base    <= cmd_src_base;
                        r_dst_base    <= cmd_dst_base;
                        r_reduce_dim  <= cmd_reduce_dim;
                        r_outer_count <= cmd_outer_count;
                        outer_idx     <= '0;
                    end
                end

                RE_OUTER_INIT: begin
                    inner_idx <= '0;
                    // Initialize accumulator based on opcode
                    case (r_opcode)
                        OP_G_REDUCE_MAX: acc <= -32'sd128; // min INT8
                        default:         acc <= 32'sd0;
                    endcase
                end

                RE_READ: begin
                    // SRAM read issued combinationally; data available next cycle
                    rd_valid <= 1'b1;
                end

                RE_ACCUM: begin
                    // sram_rd_data contains the read value (1-cycle latency)
                    begin
                        automatic logic signed [7:0] val = $signed(sram_rd_data);
                        case (r_opcode)
                            OP_G_REDUCE_SUM,
                            OP_G_REDUCE_MEAN: acc <= acc + {{24{val[7]}}, val};
                            OP_G_REDUCE_MAX: begin
                                if ($signed({{24{val[7]}}, val}) > acc)
                                    acc <= $signed({{24{val[7]}}, val});
                            end
                            default: acc <= acc + {{24{val[7]}}, val};
                        endcase
                    end
                    inner_idx <= inner_idx + 16'd1;
                end

                RE_NEXT_OUTER: begin
                    outer_idx <= outer_idx + 16'd1;
                end

                default: ;
            endcase
        end
    end

endmodule

`default_nettype wire
