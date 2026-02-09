// =============================================================================
// reduce_max.sv - Streaming reduction: find maximum value in a vector
// =============================================================================
import npu_pkg::*;
import fixed_pkg::*;

module reduce_max (
    input  logic               clk,
    input  logic               rst_n,

    // Control
    input  logic               start,      // pulse: reset running max to -128

    // Streaming input
    input  logic               din_valid,
    input  logic signed [7:0]  din,        // signed int8
    input  logic               din_last,   // last element flag

    // Output
    output logic signed [7:0]  max_val,    // signed int8 result
    output logic               max_valid   // pulses when result is ready
);

    logic signed [7:0] running_max;
    logic              active;        // currently processing a vector

    // ----------------------------------------------------------------
    // Running maximum accumulator
    // ----------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            running_max <= -8'sd128;
            active      <= 1'b0;
        end else if (start) begin
            running_max <= -8'sd128;
            active      <= 1'b1;
        end else if (active && din_valid) begin
            if (din > running_max)
                running_max <= din;
            if (din_last)
                active <= 1'b0;
        end
    end

    // ----------------------------------------------------------------
    // Output: max_val is always the running_max register
    // max_valid pulses for one cycle when din_last is received
    // ----------------------------------------------------------------
    assign max_val = running_max;

    logic last_seen;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            last_seen <= 1'b0;
        else
            last_seen <= active && din_valid && din_last;
    end

    // One cycle after din_last, the running_max has the correct final value
    // (because the last comparison happens on din_last cycle itself)
    // We must account for the case where the last element is the max:
    // The running_max is updated in the same cycle as din_last.
    // So the registered last_seen captures the result one cycle later,
    // at which point running_max already reflects the final comparison.
    assign max_valid = last_seen;

endmodule
