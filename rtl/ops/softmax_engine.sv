// =============================================================================
// softmax_engine.sv - Full softmax over a row (3-pass architecture)
// Pass 1: Find max value (reduce_max)
// Pass 2: Subtract max, exp via LUT, accumulate sum (reduce_sum)
// Pass 3: Normalize by reciprocal of sum, requantize to int8
// =============================================================================
import npu_pkg::*;
import fixed_pkg::*;

module softmax_engine (
    input  logic               clk,
    input  logic               rst_n,

    // Command interface
    input  logic               cmd_valid,
    output logic               cmd_ready,
    input  logic [15:0]        length,          // number of elements in row
    input  logic [15:0]        src_base,        // SRAM base address for input scores
    input  logic [15:0]        dst_base,        // SRAM base address for output probs
    input  logic [15:0]        scale_factor,    // attention scale (Q8.8)
    input  logic               causal_mask_en,  // enable causal masking
    input  logic [15:0]        causal_limit,    // mask positions > limit to -128

    // SRAM read port (shared for all passes)
    output logic               sram_rd_en,
    output logic [15:0]        sram_rd_addr,
    input  logic [DATA_W-1:0]  sram_rd_data,

    // SRAM write port (shared for exp scratch and final output)
    output logic               sram_wr_en,
    output logic [15:0]        sram_wr_addr,
    output logic [DATA_W-1:0]  sram_wr_data,

    // Scratch SRAM write port (for intermediate exp values, 16-bit packed)
    output logic               scratch_wr_en,
    output logic [15:0]        scratch_wr_addr,
    output logic [15:0]        scratch_wr_data,

    // Scratch SRAM read port
    output logic               scratch_rd_en,
    output logic [15:0]        scratch_rd_addr,
    input  logic [15:0]        scratch_rd_data,

    // Status
    output logic               busy,
    output logic               done
);

    // ----------------------------------------------------------------
    // FSM States
    // ----------------------------------------------------------------
    typedef enum logic [3:0] {
        S_IDLE,
        S_P1_READ,          // Pass 1: drive SRAM read address
        S_P1_FEED,          // Pass 1: SRAM data valid, feed to reduce_max
        S_P1_WAIT,          // Pass 1: wait for max_valid
        S_P2_READ,          // Pass 2: drive SRAM read address
        S_P2_EXP,           // Pass 2: SRAM data valid, drive exp LUT address
        S_P2_FEED,          // Pass 2: exp data valid, feed to sum, write scratch
        S_P2_WAIT,          // Pass 2: wait for sum_valid
        S_P3_RECIP,         // Pass 3: drive recip LUT address
        S_P3_RECIP_WAIT,    // Pass 3: recip data valid, latch
        S_P3_READ,          // Pass 3: drive scratch read address
        S_P3_NORM,          // Pass 3: scratch data valid, normalize and write
        S_DONE
    } state_t;

    state_t state, state_nxt;

    // ----------------------------------------------------------------
    // Registered command parameters
    // ----------------------------------------------------------------
    logic [15:0] r_length, r_src_base, r_dst_base, r_scale_factor, r_causal_limit;
    logic        r_causal_mask_en;
    logic [15:0] r_idx;

    // ----------------------------------------------------------------
    // Sub-module wires
    // ----------------------------------------------------------------
    // reduce_max
    logic              rmax_start;
    logic              rmax_din_valid;
    logic signed [7:0] rmax_din;
    logic              rmax_din_last;
    logic signed [7:0] rmax_result;
    logic              rmax_valid;
    logic signed [7:0] r_max_val;  // latched max

    // exp_lut
    logic [7:0]        exp_addr;
    logic [15:0]       exp_data;

    // reduce_sum
    logic              rsum_start;
    logic              rsum_din_valid;
    logic signed [15:0] rsum_din;
    logic              rsum_din_last;
    logic signed [31:0] rsum_result;
    logic              rsum_valid;
    logic [15:0]       r_sum_val;  // latched sum (unsigned portion)

    // recip_lut
    logic [7:0]        recip_addr;
    logic [15:0]       recip_data;
    logic [15:0]       r_recip_val; // latched reciprocal

    // Pipeline registers
    logic signed [7:0] p_rd_data;   // latched SRAM read data
    logic [15:0]       p_exp_val;   // latched exp LUT output
    logic [15:0]       p_scratch;   // latched scratch read data

    // ----------------------------------------------------------------
    // Sub-module instantiations
    // ----------------------------------------------------------------
    reduce_max u_reduce_max (
        .clk       (clk),
        .rst_n     (rst_n),
        .start     (rmax_start),
        .din_valid (rmax_din_valid),
        .din       (rmax_din),
        .din_last  (rmax_din_last),
        .max_val   (rmax_result),
        .max_valid (rmax_valid)
    );

    exp_lut u_exp_lut (
        .clk      (clk),
        .addr     (exp_addr),
        .data_out (exp_data)
    );

    reduce_sum u_reduce_sum (
        .clk       (clk),
        .rst_n     (rst_n),
        .start     (rsum_start),
        .din_valid (rsum_din_valid),
        .din       (rsum_din),
        .din_last  (rsum_din_last),
        .sum_val   (rsum_result),
        .sum_valid (rsum_valid)
    );

    recip_lut u_recip_lut (
        .clk      (clk),
        .addr     (recip_addr),
        .data_out (recip_data)
    );

    // ----------------------------------------------------------------
    // FSM transition
    // ----------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n)
        if (!rst_n) state <= S_IDLE;
        else        state <= state_nxt;

    always_comb begin
        state_nxt = state;
        case (state)
            S_IDLE:     if (cmd_valid) state_nxt = S_P1_READ;
            S_P1_READ:  state_nxt = S_P1_FEED;
            S_P1_FEED:  state_nxt = (r_idx == r_length - 16'd1) ? S_P1_WAIT : S_P1_READ;
            S_P1_WAIT:  if (rmax_valid) state_nxt = S_P2_READ;
            S_P2_READ:  state_nxt = S_P2_EXP;
            S_P2_EXP:   state_nxt = S_P2_FEED;
            S_P2_FEED:  state_nxt = (r_idx == r_length - 16'd1) ? S_P2_WAIT : S_P2_READ;
            S_P2_WAIT:  if (rsum_valid) state_nxt = S_P3_RECIP;
            S_P3_RECIP:      state_nxt = S_P3_RECIP_WAIT;
            S_P3_RECIP_WAIT: state_nxt = S_P3_READ;
            S_P3_READ:  state_nxt = S_P3_NORM;
            S_P3_NORM:  state_nxt = (r_idx == r_length - 16'd1) ? S_DONE : S_P3_READ;
            S_DONE:     state_nxt = S_IDLE;
            default:    state_nxt = S_IDLE;
        endcase
    end

    // ----------------------------------------------------------------
    // Command capture
    // ----------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            r_length        <= '0;
            r_src_base      <= '0;
            r_dst_base      <= '0;
            r_scale_factor  <= '0;
            r_causal_mask_en<= 1'b0;
            r_causal_limit  <= '0;
        end else if (state == S_IDLE && cmd_valid) begin
            r_length        <= length;
            r_src_base      <= src_base;
            r_dst_base      <= dst_base;
            r_scale_factor  <= scale_factor;
            r_causal_mask_en<= causal_mask_en;
            r_causal_limit  <= causal_limit;
        end
    end

    // ----------------------------------------------------------------
    // Index counter (reused across passes)
    // ----------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            r_idx <= '0;
        else if (state == S_IDLE && cmd_valid)
            r_idx <= '0;
        else if (state == S_P1_FEED && state_nxt == S_P1_READ)
            r_idx <= r_idx + 16'd1;
        else if (state == S_P1_WAIT && rmax_valid)
            r_idx <= '0;  // reset for pass 2
        else if (state == S_P2_FEED && state_nxt == S_P2_READ)
            r_idx <= r_idx + 16'd1;
        else if (state == S_P2_WAIT && rsum_valid)
            r_idx <= '0;  // reset for pass 3
        else if (state == S_P3_NORM && state_nxt == S_P3_READ)
            r_idx <= r_idx + 16'd1;
    end

    // ----------------------------------------------------------------
    // Latch max value after pass 1
    // ----------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            r_max_val <= -8'sd128;
        else if (rmax_valid)
            r_max_val <= rmax_result;
    end

    // ----------------------------------------------------------------
    // Latch sum value after pass 2 (take lower 16 bits as Q8.8)
    // ----------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            r_sum_val <= '0;
        else if (rsum_valid)
            r_sum_val <= rsum_result[15:0];
    end

    // ----------------------------------------------------------------
    // Latch reciprocal value after pass 3 recip LUT read
    // ----------------------------------------------------------------
    always_ff @(posedge clk) begin
        if (state == S_P3_RECIP_WAIT)
            r_recip_val <= recip_data;
    end

    // ----------------------------------------------------------------
    // Pipeline register: latch SRAM read data
    // ----------------------------------------------------------------
    always_ff @(posedge clk) begin
        if (state == S_P1_FEED || state == S_P2_EXP)
            p_rd_data <= signed'(sram_rd_data);
    end

    // Latch exp LUT output (available 1 cycle after addr driven)
    always_ff @(posedge clk) begin
        p_exp_val <= exp_data;
    end

    // Latch scratch read data
    always_ff @(posedge clk) begin
        if (state == S_P3_READ)
            p_scratch <= scratch_rd_data;
    end

    // ----------------------------------------------------------------
    // SRAM read port control
    // ----------------------------------------------------------------
    assign sram_rd_en   = (state == S_P1_READ) || (state == S_P2_READ);
    assign sram_rd_addr = r_src_base + r_idx;

    // ----------------------------------------------------------------
    // Pass 1: reduce_max control
    // ----------------------------------------------------------------
    assign rmax_start    = (state == S_IDLE && cmd_valid);
    assign rmax_din_valid = (state == S_P1_FEED);

    // Apply causal mask: if position > causal_limit, feed -128
    logic signed [7:0] masked_input;
    always_comb begin
        if (r_causal_mask_en && r_idx > r_causal_limit)
            masked_input = -8'sd128;
        else
            masked_input = signed'(sram_rd_data);
    end

    assign rmax_din      = masked_input;
    assign rmax_din_last = (state == S_P1_FEED) && (r_idx == r_length - 16'd1);

    // ----------------------------------------------------------------
    // Pass 2: exp + sum
    // ----------------------------------------------------------------
    // Subtract max and apply scale, then feed to exp LUT
    logic signed [15:0] scaled_diff;
    logic signed [7:0]  exp_input;

    always_comb begin
        // (x - max): both are int8
        scaled_diff = 16'(signed'(masked_input)) - 16'(signed'(r_max_val));
        // Clamp to int8 range for exp LUT input
        if (scaled_diff < -16'sd128)
            exp_input = -8'sd128;
        else if (scaled_diff > 16'sd127)
            exp_input = 8'sd127;
        else
            exp_input = scaled_diff[7:0];
    end

    // Drive exp LUT address during pass 2 feed
    assign exp_addr = (state == S_P2_EXP) ? exp_input : 8'd0;

    // reduce_sum control
    assign rsum_start    = (state == S_P1_WAIT && rmax_valid);
    assign rsum_din_valid = (state == S_P2_FEED);
    assign rsum_din      = 16'(signed'(exp_data));  // exp output is unsigned Q8.8, treat as int16
    assign rsum_din_last = (state == S_P2_FEED) && (r_idx == r_length - 16'd1);

    // Write exp values to scratch SRAM during pass 2
    assign scratch_wr_en   = (state == S_P2_FEED);
    assign scratch_wr_addr = r_idx;
    assign scratch_wr_data = exp_data;  // 16-bit exp value

    // ----------------------------------------------------------------
    // Pass 3: normalize
    // ----------------------------------------------------------------
    // Drive reciprocal LUT with top 8 bits of sum
    assign recip_addr = r_sum_val[15:8];

    // Read scratch values during pass 3
    assign scratch_rd_en   = (state == S_P3_READ);
    assign scratch_rd_addr = r_idx;

    // Normalize: out = (exp_val * recip) >> 17, clamp to int8
    // exp_val is Q8.8, recip is Q0.16, product is Q8.24
    // >>17 maps [0.0, 1.0] to [0, ~128], clamped to [0, 127]
    logic [31:0] norm_product;
    logic [14:0] norm_trunc;
    logic signed [7:0] norm_result;

    always_comb begin
        norm_product = 32'(scratch_rd_data) * 32'(r_recip_val);
        norm_trunc = norm_product[31:17];  // >>17 for Q8.24 -> Q0.7
        // Clamp to [0, 127] since softmax outputs are non-negative
        if (norm_trunc > 15'd127)
            norm_result = 8'sd127;
        else
            norm_result = norm_trunc[7:0];
    end

    // SRAM write during pass 3 normalization
    assign sram_wr_en   = (state == S_P3_NORM);
    assign sram_wr_addr = r_dst_base + r_idx;
    assign sram_wr_data = norm_result;

    // ----------------------------------------------------------------
    // Status and handshake
    // ----------------------------------------------------------------
    assign cmd_ready = (state == S_IDLE);
    assign busy      = (state != S_IDLE);
    assign done      = (state == S_DONE);

endmodule
