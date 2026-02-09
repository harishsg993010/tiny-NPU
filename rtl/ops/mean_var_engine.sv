// =============================================================================
// mean_var_engine.sv - Single-pass mean and variance computation for LayerNorm
// Accumulates sum and sum-of-squares in one streaming pass.
// mean = sum / length
// variance = sum_sq / length - mean^2
// Uses shift-based division for power-of-2 lengths; general reciprocal
// via iterative shift-subtract for non-power-of-2.
// =============================================================================
import npu_pkg::*;
import fixed_pkg::*;

module mean_var_engine (
    input  logic               clk,
    input  logic               rst_n,

    // Control
    input  logic               start,       // pulse: begin computation
    input  logic [15:0]        length,      // number of elements in the vector

    // Streaming input
    input  logic               din_valid,
    input  logic signed [7:0]  din,         // signed int8 input elements
    input  logic               din_last,    // last element flag

    // Outputs (available when result_valid pulses)
    output logic signed [15:0] mean_out,    // Q8.8 fixed-point mean
    output logic        [31:0] var_out,     // unsigned variance (Q16.16)
    output logic               result_valid
);

    // ----------------------------------------------------------------
    // Internal state
    // ----------------------------------------------------------------
    typedef enum logic [2:0] {
        S_IDLE,
        S_ACCUM,       // accumulating sum and sum_sq
        S_DIVIDE,      // computing mean = sum / length
        S_VARIANCE,    // computing var = sum_sq/length - mean^2
        S_DONE
    } state_t;

    state_t state, state_nxt;

    logic [15:0]        r_length;
    logic signed [31:0] r_sum;          // running sum of elements
    logic        [47:0] r_sum_sq;       // running sum of squares (unsigned, 48-bit to avoid overflow)
    logic signed [15:0] r_mean;         // Q8.8 mean result
    logic        [31:0] r_var;          // variance result

    // Division: iterative shift-subtract divider
    logic [5:0]         div_count;      // bit counter for division (up to 32 bits)
    logic signed [47:0] div_remainder;
    logic signed [31:0] div_quotient;
    logic        [47:0] sq_div_remainder;
    logic        [47:0] sq_div_quotient;
    logic               div_phase;      // 0 = dividing sum, 1 = dividing sum_sq

    // ----------------------------------------------------------------
    // FSM
    // ----------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n)
        if (!rst_n) state <= S_IDLE;
        else        state <= state_nxt;

    always_comb begin
        state_nxt = state;
        case (state)
            S_IDLE:      if (start) state_nxt = S_ACCUM;
            S_ACCUM:     if (din_valid && din_last) state_nxt = S_DIVIDE;
            S_DIVIDE:    if (div_count == 6'd0 && div_phase) state_nxt = S_VARIANCE;
            S_VARIANCE:  state_nxt = S_DONE;
            S_DONE:      state_nxt = S_IDLE;
            default:     state_nxt = S_IDLE;
        endcase
    end

    // ----------------------------------------------------------------
    // Capture length on start
    // ----------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            r_length <= '0;
        else if (start)
            r_length <= length;
    end

    // ----------------------------------------------------------------
    // Accumulation phase: sum and sum_sq
    // ----------------------------------------------------------------
    logic signed [15:0] din_ext;    // sign-extended input
    logic        [15:0] din_sq;     // din^2 (unsigned since square is non-negative)

    assign din_ext = 16'(signed'(din));
    assign din_sq  = 16'(din) * 16'(din);   // 8b*8b = 16b

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            r_sum    <= '0;
            r_sum_sq <= '0;
        end else if (start) begin
            r_sum    <= '0;
            r_sum_sq <= '0;
        end else if (state == S_ACCUM && din_valid) begin
            r_sum    <= r_sum + 32'(din_ext);
            r_sum_sq <= r_sum_sq + 48'(din_sq);
        end
    end

    // ----------------------------------------------------------------
    // Division phase: compute sum/length and sum_sq/length
    // Simple restoring divider, 32 cycles per division
    // First divides sum -> quotient is mean (shifted to Q8.8)
    // Then divides sum_sq -> quotient is E[x^2]
    // ----------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            div_count     <= '0;
            div_remainder <= '0;
            div_quotient  <= '0;
            sq_div_remainder <= '0;
            sq_div_quotient  <= '0;
            div_phase     <= 1'b0;
        end else if (state == S_ACCUM && din_valid && din_last) begin
            // Initialize division of sum by length
            // We want Q8.8 result, so shift sum left by 8 before dividing
            div_remainder <= 48'(signed'(r_sum + 32'(din_ext))) << 8;
            div_quotient  <= '0;
            div_count     <= 6'd32;
            div_phase     <= 1'b0;
            // Pre-compute for sum_sq division
            sq_div_remainder <= (r_sum_sq + 48'(din_sq)) << 8;
            sq_div_quotient  <= '0;
        end else if (state == S_DIVIDE && div_count > 6'd0) begin
            if (!div_phase) begin
                // Dividing sum for mean
                div_quotient <= div_quotient << 1;
                if (div_remainder >= 48'(r_length)) begin
                    div_remainder <= div_remainder - 48'(r_length);
                    div_quotient[0] <= 1'b1;
                end else begin
                    div_quotient[0] <= 1'b0;
                end
                div_count <= div_count - 6'd1;
            end else begin
                // Dividing sum_sq for E[x^2]
                sq_div_quotient <= sq_div_quotient << 1;
                if (sq_div_remainder >= 48'(r_length)) begin
                    sq_div_remainder <= sq_div_remainder - 48'(r_length);
                    sq_div_quotient[0] <= 1'b1;
                end else begin
                    sq_div_quotient[0] <= 1'b0;
                end
                div_count <= div_count - 6'd1;
            end
        end else if (state == S_DIVIDE && div_count == 6'd0 && !div_phase) begin
            // First division done, start second
            r_mean    <= div_quotient[15:0]; // Q8.8 mean
            div_phase <= 1'b1;
            div_count <= 6'd32;
        end
    end

    // ----------------------------------------------------------------
    // Variance computation: var = E[x^2] - (mean)^2
    // Both in Q8.8-ish fixed point; result is Q16.16 (unsigned)
    // ----------------------------------------------------------------
    logic [31:0] mean_sq;

    always_comb begin
        mean_sq = 32'(r_mean) * 32'(r_mean);  // Q8.8 * Q8.8 = Q16.16
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            r_var <= '0;
        else if (state == S_VARIANCE) begin
            // sq_div_quotient is E[x^2] in Q8.8 format (after <<8 division)
            // mean_sq is mean^2 in Q16.16 format
            // Align: shift sq_div_quotient left by 8 to get Q16.16
            if ((sq_div_quotient << 8) >= mean_sq)
                r_var <= (sq_div_quotient << 8) - mean_sq;
            else
                r_var <= '0;  // numerical guard against negative variance
        end
    end

    // ----------------------------------------------------------------
    // Outputs
    // ----------------------------------------------------------------
    assign mean_out     = r_mean;
    assign var_out      = r_var;
    assign result_valid = (state == S_DONE);

endmodule
