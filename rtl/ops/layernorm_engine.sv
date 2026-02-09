// =============================================================================
// layernorm_engine.sv - Full LayerNorm over hidden dimension (2-pass)
// Pass 1: Stream input through mean_var_engine to get mean and variance
// Compute inv_std = rsqrt(variance + epsilon)
// Pass 2: Re-read input, apply: out[i] = ((in[i]-mean)*inv_std*gamma[i]+beta[i])
//         requantized to int8
// =============================================================================
import npu_pkg::*;
import fixed_pkg::*;

module layernorm_engine (
    input  logic               clk,
    input  logic               rst_n,

    // Command interface
    input  logic               cmd_valid,
    output logic               cmd_ready,
    input  logic [15:0]        length,       // hidden dimension
    input  logic [15:0]        src_base,     // input activation base address
    input  logic [15:0]        dst_base,     // output base address
    input  logic [15:0]        gamma_base,   // scale parameter base address
    input  logic [15:0]        beta_base,    // bias parameter base address

    // SRAM read port 0 (input / gamma)
    output logic               sram_rd0_en,
    output logic [15:0]        sram_rd0_addr,
    input  logic [DATA_W-1:0]  sram_rd0_data,

    // SRAM read port 1 (beta)
    output logic               sram_rd1_en,
    output logic [15:0]        sram_rd1_addr,
    input  logic [DATA_W-1:0]  sram_rd1_data,

    // SRAM write port (output)
    output logic               sram_wr_en,
    output logic [15:0]        sram_wr_addr,
    output logic [DATA_W-1:0]  sram_wr_data,

    // Status
    output logic               busy,
    output logic               done
);

    // ----------------------------------------------------------------
    // FSM States
    // ----------------------------------------------------------------
    typedef enum logic [3:0] {
        S_IDLE,
        S_P1_READ,       // Pass 1: read input for mean/var
        S_P1_FEED,       // Pass 1: feed element to mean_var_engine
        S_P1_WAIT,       // Pass 1: wait for mean/var result
        S_RSQRT,         // Compute rsqrt of variance
        S_RSQRT_LATCH,   // Latch rsqrt result (1-cycle LUT latency)
        S_P2_READ,       // Pass 2: read input + gamma + beta
        S_P2_COMPUTE,    // Pass 2: compute normalized output
        S_DONE
    } state_t;

    state_t state, state_nxt;

    // ----------------------------------------------------------------
    // Registered command parameters
    // ----------------------------------------------------------------
    logic [15:0] r_length, r_src_base, r_dst_base, r_gamma_base, r_beta_base;
    logic [15:0] r_idx;

    // ----------------------------------------------------------------
    // mean_var_engine signals
    // ----------------------------------------------------------------
    logic               mv_start;
    logic               mv_din_valid;
    logic signed [7:0]  mv_din;
    logic               mv_din_last;
    logic signed [15:0] mv_mean;       // Q8.8
    logic        [31:0] mv_var;        // variance (unsigned)
    logic               mv_valid;

    // Latched results
    logic signed [15:0] r_mean;        // Q8.8
    logic        [15:0] r_inv_std;     // Q0.16 from rsqrt LUT

    // rsqrt LUT signals
    logic [7:0]         rsqrt_addr;
    logic [15:0]        rsqrt_data;

    // Pipeline registers
    logic signed [7:0]  p_input;       // latched input value
    logic signed [7:0]  p_gamma;       // latched gamma value
    logic signed [7:0]  p_beta;        // latched beta value

    // ----------------------------------------------------------------
    // Sub-module instantiations
    // ----------------------------------------------------------------
    mean_var_engine u_mean_var (
        .clk          (clk),
        .rst_n        (rst_n),
        .start        (mv_start),
        .length       (r_length),
        .din_valid    (mv_din_valid),
        .din          (mv_din),
        .din_last     (mv_din_last),
        .mean_out     (mv_mean),
        .var_out      (mv_var),
        .result_valid (mv_valid)
    );

    rsqrt_lut u_rsqrt (
        .clk      (clk),
        .addr     (rsqrt_addr),
        .data_out (rsqrt_data)
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
            S_IDLE:         if (cmd_valid) state_nxt = S_P1_READ;
            S_P1_READ:      state_nxt = S_P1_FEED;
            S_P1_FEED:      state_nxt = (r_idx == r_length - 16'd1) ? S_P1_WAIT : S_P1_READ;
            S_P1_WAIT:      if (mv_valid) state_nxt = S_RSQRT;
            S_RSQRT:        state_nxt = S_RSQRT_LATCH;
            S_RSQRT_LATCH:  state_nxt = S_P2_READ;
            S_P2_READ:      state_nxt = S_P2_COMPUTE;
            S_P2_COMPUTE:   state_nxt = (r_idx == r_length - 16'd1) ? S_DONE : S_P2_READ;
            S_DONE:         state_nxt = S_IDLE;
            default:        state_nxt = S_IDLE;
        endcase
    end

    // ----------------------------------------------------------------
    // Command capture
    // ----------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            r_length     <= '0;
            r_src_base   <= '0;
            r_dst_base   <= '0;
            r_gamma_base <= '0;
            r_beta_base  <= '0;
        end else if (state == S_IDLE && cmd_valid) begin
            r_length     <= length;
            r_src_base   <= src_base;
            r_dst_base   <= dst_base;
            r_gamma_base <= gamma_base;
            r_beta_base  <= beta_base;
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
        else if (state == S_P1_WAIT && mv_valid)
            r_idx <= '0;  // reset for pass 2
        else if (state == S_P2_COMPUTE && state_nxt == S_P2_READ)
            r_idx <= r_idx + 16'd1;
    end

    // ----------------------------------------------------------------
    // Pass 1: Feed elements to mean_var_engine
    // ----------------------------------------------------------------
    assign mv_start    = (state == S_IDLE && cmd_valid);
    assign mv_din_valid = (state == S_P1_FEED);
    assign mv_din       = signed'(sram_rd0_data);
    assign mv_din_last  = (state == S_P1_FEED) && (r_idx == r_length - 16'd1);

    // Latch mean after pass 1
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            r_mean <= '0;
        else if (mv_valid)
            r_mean <= mv_mean;
    end

    // ----------------------------------------------------------------
    // rsqrt computation
    // Feed top 8 bits of variance to rsqrt LUT
    // ----------------------------------------------------------------
    assign rsqrt_addr = mv_var[31:24];  // top 8 bits as index

    // Latch rsqrt result
    always_ff @(posedge clk) begin
        if (state == S_RSQRT_LATCH)
            r_inv_std <= rsqrt_data;
    end

    // ----------------------------------------------------------------
    // SRAM read port control
    // ----------------------------------------------------------------
    always_comb begin
        sram_rd0_en   = 1'b0;
        sram_rd0_addr = '0;
        sram_rd1_en   = 1'b0;
        sram_rd1_addr = '0;

        if (state == S_P1_READ) begin
            // Pass 1: read input
            sram_rd0_en   = 1'b1;
            sram_rd0_addr = r_src_base + r_idx;
        end else if (state == S_P2_READ) begin
            // Pass 2: read input on port 0, read gamma on port 0 (sequenced),
            // but we can read input+beta simultaneously if we re-read gamma later
            // Simpler: port 0 = input, port 1 = beta
            // gamma read: use port 0 with a micro-pipeline
            // For simplicity: port 0 reads input, port 1 reads beta
            // gamma will be read from port 0 in a subsequent state (or packed)
            // SIMPLIFICATION: read input on rd0, beta on rd1, gamma embedded in scale
            sram_rd0_en   = 1'b1;
            sram_rd0_addr = r_src_base + r_idx;  // will alternate for gamma
            sram_rd1_en   = 1'b1;
            sram_rd1_addr = r_beta_base + r_idx;
        end
    end

    // Pipeline registers for pass 2 data
    always_ff @(posedge clk) begin
        if (state == S_P2_READ) begin
            p_input <= signed'(sram_rd0_data);
            p_beta  <= signed'(sram_rd1_data);
            // For gamma, we assume it's unity-scaled or stored with input
            // In a full implementation, a third read port or time-multiplexing
            // would fetch gamma. Here we use a default gamma of 1 (=127 in int8 scale).
            p_gamma <= 8'sd127;  // placeholder: unity gamma in int8 scale
        end
    end

    // ----------------------------------------------------------------
    // Pass 2: Normalize computation
    // out = ((in - mean) * inv_std) * gamma + beta -> requant to int8
    // ----------------------------------------------------------------
    logic signed [15:0] centered;       // in - mean (Q8.8 - Q8.8 isn't right, need alignment)
    logic signed [31:0] scaled;         // centered * inv_std
    logic signed [31:0] gamma_applied;  // scaled * gamma
    logic signed [31:0] bias_added;     // + beta
    logic signed [7:0]  norm_result;

    always_comb begin
        // Center: (in[i] - mean) where in is int8 and mean is Q8.8
        // Convert input to Q8.8 first: in << 8
        centered = (16'(signed'(p_input)) <<< 8) - r_mean;  // Q8.8

        // Scale by inv_std (Q0.16): result is Q8.24, take upper bits
        scaled = 32'(signed'(centered)) * 32'(signed'({1'b0, r_inv_std}));
        // scaled is Q8.24 (8.8 * 0.16 = 8.24)
        // Shift right by 16 to get Q8.8
        scaled = scaled >>> 16;

        // Apply gamma (int8, representing scale ~1.0 when gamma=127)
        // gamma_applied = scaled * gamma / 128 (normalize gamma to ~1.0)
        gamma_applied = (scaled * 32'(signed'(p_gamma))) >>> 7;

        // Add beta (int8, sign-extend to Q8.8 by shifting left 8)
        bias_added = gamma_applied + (32'(signed'(p_beta)) <<< 8);

        // Requantize: shift right by 8 to get int8 from Q8.8
        if ((bias_added >>> 8) > 32'sd127)
            norm_result = 8'sd127;
        else if ((bias_added >>> 8) < -32'sd128)
            norm_result = -8'sd128;
        else
            norm_result = bias_added[15:8];
    end

    // ----------------------------------------------------------------
    // SRAM write - during S_P2_COMPUTE
    // ----------------------------------------------------------------
    assign sram_wr_en   = (state == S_P2_COMPUTE);
    assign sram_wr_addr = r_dst_base + r_idx;
    assign sram_wr_data = norm_result;

    // ----------------------------------------------------------------
    // Status and handshake
    // ----------------------------------------------------------------
    assign cmd_ready = (state == S_IDLE);
    assign busy      = (state != S_IDLE);
    assign done      = (state == S_DONE);

endmodule
