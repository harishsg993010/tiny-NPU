// =============================================================================
// Fixed-Point Arithmetic Package
// =============================================================================
`ifndef FIXED_PKG_SV
`define FIXED_PKG_SV

package fixed_pkg;

    // =========================================================================
    // Fixed-Point Multiply: INT8 * INT8 -> INT16
    // =========================================================================
    function automatic logic signed [15:0] mul_i8(
        input logic signed [7:0] a,
        input logic signed [7:0] b
    );
        return 16'(a) * 16'(b);
    endfunction

    // =========================================================================
    // Fixed-Point Multiply: INT16 * INT16 -> INT32
    // =========================================================================
    function automatic logic signed [31:0] mul_i16(
        input logic signed [15:0] a,
        input logic signed [15:0] b
    );
        return 32'(a) * 32'(b);
    endfunction

    // =========================================================================
    // Saturating Add INT16
    // =========================================================================
    function automatic logic signed [15:0] sat_add_i16(
        input logic signed [15:0] a,
        input logic signed [15:0] b
    );
        logic signed [16:0] sum;
        sum = 17'(a) + 17'(b);
        if (sum > 17'sd32767)
            return 16'sd32767;
        else if (sum < -17'sd32768)
            return -16'sd32768;
        else
            return sum[15:0];
    endfunction

    // =========================================================================
    // Saturating Add INT32
    // =========================================================================
    function automatic logic signed [31:0] sat_add_i32(
        input logic signed [31:0] a,
        input logic signed [31:0] b
    );
        logic signed [32:0] sum;
        sum = 33'(a) + 33'(b);
        if (sum > 33'sd2147483647)
            return 32'sd2147483647;
        else if (sum < -33'sd2147483648)
            return -32'sd2147483648;
        else
            return sum[31:0];
    endfunction

    // =========================================================================
    // Fixed-Point Multiply with Fractional Bits
    // result = (a * b) >> frac_bits, rounded
    // =========================================================================
    function automatic logic signed [31:0] fpmul(
        input logic signed [15:0] a,
        input logic signed [15:0] b,
        input int unsigned        frac_bits
    );
        logic signed [31:0] product;
        logic signed [31:0] rounded;
        product = 32'(a) * 32'(b);
        if (frac_bits > 0)
            rounded = product + (32'sd1 <<< (frac_bits - 1));
        else
            rounded = product;
        return rounded >>> frac_bits;
    endfunction

    // =========================================================================
    // Scale and Shift for Requantization
    // result = (val * scale) >> shift, with rounding
    // =========================================================================
    function automatic logic signed [15:0] scale_shift_i32_to_i16(
        input logic signed [31:0] val,
        input logic        [7:0]  scale,
        input logic        [7:0]  shift
    );
        logic signed [47:0] product;
        logic signed [47:0] rounded;
        logic signed [15:0] result;

        product = 48'(val) * 48'(signed'({1'b0, scale}));
        if (shift > 0)
            rounded = product + (48'sd1 <<< (shift - 1));
        else
            rounded = product;

        result = 16'(rounded >>> shift);
        // Saturate
        if ((rounded >>> shift) > 48'sd32767)
            return 16'sd32767;
        else if ((rounded >>> shift) < -48'sd32768)
            return -16'sd32768;
        else
            return result;
    endfunction

    // =========================================================================
    // Clamp to arbitrary signed range
    // =========================================================================
    function automatic logic signed [31:0] clamp_signed(
        input logic signed [31:0] val,
        input logic signed [31:0] lo,
        input logic signed [31:0] hi
    );
        if (val < lo)
            return lo;
        else if (val > hi)
            return hi;
        else
            return val;
    endfunction

endpackage

`endif
