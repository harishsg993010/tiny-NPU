// =============================================================================
// reduce_sum.sv - Streaming reduction: accumulate sum of a vector
// Input is int16 (e.g., post-exp values), output is int32 with saturation
// =============================================================================
import npu_pkg::*;
import fixed_pkg::*;

module reduce_sum (
    input  logic                clk,
    input  logic                rst_n,

    // Control
    input  logic                start,      // pulse: reset accumulator to 0

    // Streaming input
    input  logic                din_valid,
    input  logic signed [15:0]  din,        // signed int16
    input  logic                din_last,   // last element flag

    // Output
    output logic signed [31:0]  sum_val,    // accumulated sum
    output logic                sum_valid   // pulses when result is ready
);

    logic signed [31:0] acc;
    logic               active;

    // Saturating accumulation
    logic signed [32:0] wide_sum;
    logic signed [31:0] sat_sum;

    always_comb begin
        wide_sum = 33'(acc) + 33'(signed'(din));
        if (wide_sum > 33'sd2147483647)
            sat_sum = 32'sd2147483647;
        else if (wide_sum < -33'sd2147483648)
            sat_sum = -32'sd2147483648;
        else
            sat_sum = wide_sum[31:0];
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            acc    <= 32'sd0;
            active <= 1'b0;
        end else if (start) begin
            acc    <= 32'sd0;
            active <= 1'b1;
        end else if (active && din_valid) begin
            acc <= sat_sum;
            if (din_last)
                active <= 1'b0;
        end
    end

    assign sum_val = acc;

    // Output valid one cycle after din_last
    logic last_seen;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            last_seen <= 1'b0;
        else
            last_seen <= active && din_valid && din_last;
    end

    assign sum_valid = last_seen;

endmodule
