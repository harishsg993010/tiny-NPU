// =============================================================================
// vec_engine.sv - Vector / element-wise ALU engine
// Operations: ADD, MUL, SCALE_SHIFT, CLAMP (one element per cycle)
// =============================================================================
import npu_pkg::*;
import fixed_pkg::*;

module vec_engine (
    input  logic                clk,
    input  logic                rst_n,

    // Command interface
    input  logic                cmd_valid,
    output logic                cmd_ready,
    input  logic [1:0]          opcode,       // 0=ADD 1=MUL 2=SCALE_SHIFT 3=CLAMP
    input  logic [15:0]         length,       // number of elements
    input  logic [15:0]         src0_base,
    input  logic [15:0]         src1_base,
    input  logic [15:0]         dst_base,
    input  logic [7:0]          scale,        // for SCALE_SHIFT
    input  logic [7:0]          shift,        // for SCALE_SHIFT

    // SRAM read port 0 (src0)
    output logic                sram_rd0_en,
    output logic [15:0]         sram_rd0_addr,
    input  logic [DATA_W-1:0]   sram_rd0_data,

    // SRAM read port 1 (src1)
    output logic                sram_rd1_en,
    output logic [15:0]         sram_rd1_addr,
    input  logic [DATA_W-1:0]   sram_rd1_data,

    // SRAM write port (dst)
    output logic                sram_wr_en,
    output logic [15:0]         sram_wr_addr,
    output logic [DATA_W-1:0]   sram_wr_data,

    // Status
    output logic                busy,
    output logic                done
);

    // Opcode encoding
    localparam logic [1:0] OP_ADD         = 2'd0;
    localparam logic [1:0] OP_MUL         = 2'd1;
    localparam logic [1:0] OP_SCALE_SHIFT = 2'd2;
    localparam logic [1:0] OP_CLAMP       = 2'd3;

    // FSM states
    typedef enum logic [2:0] {S_IDLE, S_READ, S_WAIT, S_PROCESS, S_DONE} state_t;
    state_t state, state_nxt;

    // Registers
    logic [1:0]  r_opcode;
    logic [15:0] r_length, r_src0_base, r_src1_base, r_dst_base;
    logic [7:0]  r_scale, r_shift;
    logic [15:0] r_idx;
    logic signed [DATA_W-1:0] p_src0, p_src1;

    // ----------------------------------------------------------------
    // FSM transition
    // ----------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n)
        if (!rst_n) state <= S_IDLE;
        else        state <= state_nxt;

    always_comb begin
        state_nxt = state;
        case (state)
            S_IDLE:    if (cmd_valid) state_nxt = S_READ;
            S_READ:    state_nxt = S_WAIT;
            S_WAIT:    state_nxt = S_PROCESS;
            S_PROCESS: state_nxt = (r_idx == r_length - 16'd1) ? S_DONE : S_READ;
            S_DONE:    state_nxt = S_IDLE;
            default:   state_nxt = S_IDLE;
        endcase
    end

    // ----------------------------------------------------------------
    // Command capture
    // ----------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            r_opcode    <= '0;
            r_length    <= '0;
            r_src0_base <= '0;
            r_src1_base <= '0;
            r_dst_base  <= '0;
            r_scale     <= '0;
            r_shift     <= '0;
        end else if (state == S_IDLE && cmd_valid) begin
            r_opcode    <= opcode;
            r_length    <= length;
            r_src0_base <= src0_base;
            r_src1_base <= src1_base;
            r_dst_base  <= dst_base;
            r_scale     <= scale;
            r_shift     <= shift;
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
        else if (state == S_PROCESS && state_nxt == S_READ)
            r_idx <= r_idx + 16'd1;

    // ----------------------------------------------------------------
    // Latch read data (available one cycle after read enable)
    // ----------------------------------------------------------------
    always_ff @(posedge clk) begin
        if (state == S_WAIT) begin
            p_src0 <= signed'(sram_rd0_data);
            p_src1 <= signed'(sram_rd1_data);
        end
    end

    // ----------------------------------------------------------------
    // SRAM read addresses - issued during S_READ
    // ----------------------------------------------------------------
    assign sram_rd0_en   = (state == S_READ);
    assign sram_rd0_addr = r_src0_base + r_idx;
    assign sram_rd1_en   = (state == S_READ) && (r_opcode == OP_ADD || r_opcode == OP_MUL);
    assign sram_rd1_addr = r_src1_base + r_idx;

    // ----------------------------------------------------------------
    // ALU - combinational result, written during S_PROCESS
    // ----------------------------------------------------------------
    logic signed [15:0] add_res;
    logic signed [15:0] mul_raw;
    logic signed [31:0] scale_raw;
    logic signed [15:0] scale_shifted;
    logic signed [15:0] mul_round;
    logic signed [7:0]  result;

    always_comb begin
        // Widen for intermediate precision
        add_res       = 16'(signed'(p_src0)) + 16'(signed'(p_src1));
        mul_raw       = 16'(p_src0) * 16'(p_src1);
        scale_raw     = 32'(signed'(p_src0)) * 32'(signed'({1'b0, r_scale}));
        scale_shifted = 16'(scale_raw >>> r_shift);

        // MUL requantize: >>7 with rounding
        mul_round = (mul_raw + 16'sd64) >>> 7;

        case (r_opcode)
            OP_ADD: begin
                if (add_res > 16'sd127)        result = 8'sd127;
                else if (add_res < -16'sd128)  result = -8'sd128;
                else                           result = add_res[7:0];
            end
            OP_MUL: begin
                if (mul_round > 16'sd127)       result = 8'sd127;
                else if (mul_round < -16'sd128) result = -8'sd128;
                else                            result = mul_round[7:0];
            end
            OP_SCALE_SHIFT: begin
                if (scale_shifted > 16'sd127)       result = 8'sd127;
                else if (scale_shifted < -16'sd128) result = -8'sd128;
                else                                result = scale_shifted[7:0];
            end
            OP_CLAMP: begin
                result = p_src0;  // identity clamp for int8
            end
            default: result = '0;
        endcase
    end

    // ----------------------------------------------------------------
    // SRAM write - during S_PROCESS
    // ----------------------------------------------------------------
    assign sram_wr_en   = (state == S_PROCESS);
    assign sram_wr_addr = r_dst_base + r_idx;
    assign sram_wr_data = result;

    // ----------------------------------------------------------------
    // Handshake and status
    // ----------------------------------------------------------------
    assign cmd_ready = (state == S_IDLE);
    assign busy      = (state != S_IDLE);
    assign done      = (state == S_DONE);

endmodule
