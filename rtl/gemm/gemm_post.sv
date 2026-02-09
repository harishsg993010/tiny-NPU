// =============================================================================
// GEMM Post-Processing: Bias add, requantize, ReLU, write back
// =============================================================================
`default_nettype none

module gemm_post
    import npu_pkg::*;
    import isa_pkg::*;
#(
    parameter int ARRAY_M = 16,
    parameter int ARRAY_N = 16,
    parameter int DATA_W  = 8,
    parameter int ACC_W   = 32
)(
    input  wire                          clk,
    input  wire                          rst_n,

    // Command
    input  wire                          cmd_valid,
    output logic                         cmd_ready,
    input  wire  [7:0]                   flags,
    input  wire  [7:0]                   scale,
    input  wire  [7:0]                   shift,

    // ACC SRAM read
    output logic                         acc_rd_en,
    output logic [15:0]                  acc_rd_addr,
    input  wire  signed [ACC_W-1:0]      acc_rd_data,
    input  wire                          acc_rd_valid,

    // Bias data (optional)
    input  wire  signed [ACC_W-1:0]      bias_data,

    // Result write
    output logic                         res_wr_en,
    output logic [15:0]                  res_addr,
    output logic signed [DATA_W-1:0]     res_data,

    // Status
    output logic                         busy,
    output logic                         done
);

    typedef enum logic [2:0] {
        PP_IDLE,
        PP_READ,
        PP_WAIT,
        PP_PROCESS,
        PP_WRITE,
        PP_DONE
    } pp_state_t;

    pp_state_t pp_state, pp_state_next;

    logic [15:0] elem_cnt, total_elems;
    logic [15:0] base_addr, dst_addr;
    logic bias_en, requant_en, relu_en;
    logic signed [ACC_W-1:0] acc_val;
    logic signed [ACC_W-1:0] biased;
    logic signed [DATA_W-1:0] result;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            pp_state <= PP_IDLE;
        else
            pp_state <= pp_state_next;
    end

    always_comb begin
        pp_state_next = pp_state;
        cmd_ready    = 1'b0;
        acc_rd_en    = 1'b0;
        acc_rd_addr  = '0;
        res_wr_en    = 1'b0;
        res_addr     = '0;
        res_data     = '0;
        busy         = (pp_state != PP_IDLE);
        done         = 1'b0;

        case (pp_state)
            PP_IDLE: begin
                cmd_ready = 1'b1;
                busy = 1'b0;
                if (cmd_valid)
                    pp_state_next = PP_READ;
            end

            PP_READ: begin
                acc_rd_en   = 1'b1;
                acc_rd_addr = base_addr + elem_cnt;
                pp_state_next = PP_WAIT;
            end

            PP_WAIT: begin
                // 1-cycle SRAM read latency
                if (acc_rd_valid)
                    pp_state_next = PP_PROCESS;
            end

            PP_PROCESS: begin
                pp_state_next = PP_WRITE;
            end

            PP_WRITE: begin
                res_wr_en = 1'b1;
                res_addr  = dst_addr + elem_cnt;
                res_data  = result;
                if (elem_cnt >= total_elems - 1)
                    pp_state_next = PP_DONE;
                else
                    pp_state_next = PP_READ;
            end

            PP_DONE: begin
                done = 1'b1;
                pp_state_next = PP_IDLE;
            end

            default: pp_state_next = PP_IDLE;
        endcase
    end

    // Processing pipeline
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            elem_cnt     <= '0;
            total_elems  <= '0;
            base_addr    <= '0;
            dst_addr     <= '0;
            bias_en      <= 1'b0;
            requant_en   <= 1'b0;
            relu_en      <= 1'b0;
            acc_val      <= '0;
            biased       <= '0;
            result       <= '0;
        end else begin
            case (pp_state)
                PP_IDLE: begin
                    if (cmd_valid) begin
                        elem_cnt    <= '0;
                        total_elems <= 16'(ARRAY_M * ARRAY_N);
                        base_addr   <= '0;
                        dst_addr    <= '0;
                        bias_en     <= flags[FLAG_BIAS_EN];
                        requant_en  <= flags[FLAG_REQUANT];
                        relu_en     <= flags[FLAG_RELU];
                    end
                end

                PP_WAIT: begin
                    if (acc_rd_valid)
                        acc_val <= acc_rd_data;
                end

                PP_PROCESS: begin
                    // Bias add
                    biased = bias_en ? sat_add_i32(acc_val, bias_data) : acc_val;
                    // Requantize
                    if (requant_en)
                        result <= requantize(biased, scale, shift);
                    else
                        result <= saturate_i8(biased);
                    // ReLU
                    if (relu_en && result < 0)
                        result <= '0;
                end

                PP_WRITE: begin
                    elem_cnt <= elem_cnt + 1;
                end

                default: ;
            endcase
        end
    end

endmodule

`default_nettype wire
