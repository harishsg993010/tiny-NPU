// =============================================================================
// avgpool2d_engine.sv - 2D Average Pooling engine
// Computes AvgPool2D on INT8 NCHW data in SRAM
// FSM: AP_IDLE -> AP_READ -> AP_SRAM_WAIT -> AP_ACCUM -> (loop kh*kw) -> AP_WRITE -> (next pixel or AP_DONE)
// For each output pixel: accumulate kh*kw input elements, compute mean, write INT8 result
// =============================================================================
`default_nettype none

module avgpool2d_engine #(
    parameter int SRAM0_AW = 16
) (
    input  wire                     clk,
    input  wire                     rst_n,

    // -- Command interface ----------------------------------------------------
    input  wire                     cmd_valid,
    input  wire  [15:0]             cmd_src_base,   // SRAM base of input [N,C,H,W]
    input  wire  [15:0]             cmd_dst_base,   // SRAM base of output [N,C,out_h,out_w]
    input  wire  [15:0]             cmd_C,          // number of channels
    input  wire  [15:0]             cmd_H,          // input height
    input  wire  [15:0]             cmd_W,          // input width
    input  wire  [7:0]              cmd_kh,         // kernel height
    input  wire  [7:0]              cmd_kw,         // kernel width
    input  wire  [7:0]              cmd_sh,         // stride height
    input  wire  [7:0]              cmd_sw,         // stride width

    // -- SRAM0 read port ------------------------------------------------------
    output logic                    sram_rd_en,
    output logic [SRAM0_AW-1:0]    sram_rd_addr,
    input  wire  [7:0]              sram_rd_data,

    // -- SRAM0 write port -----------------------------------------------------
    output logic                    sram_wr_en,
    output logic [SRAM0_AW-1:0]    sram_wr_addr,
    output logic [7:0]             sram_wr_data,

    // -- Status ---------------------------------------------------------------
    output logic                    busy,
    output logic                    done
);

    // =========================================================================
    // FSM states
    // =========================================================================
    typedef enum logic [2:0] {
        AP_IDLE      = 3'd0,
        AP_READ      = 3'd1,
        AP_SRAM_WAIT = 3'd2,
        AP_ACCUM     = 3'd3,
        AP_WRITE     = 3'd4,
        AP_DONE      = 3'd5
    } ap_state_t;

    ap_state_t r_state, w_state;

    // =========================================================================
    // Internal registers - latched command parameters
    // =========================================================================
    logic [15:0]           r_src_base;
    logic [15:0]           r_dst_base;
    logic [15:0]           r_C;
    logic [15:0]           r_H;
    logic [15:0]           r_W;
    logic [7:0]            r_kh;
    logic [7:0]            r_kw;
    logic [7:0]            r_sh;
    logic [7:0]            r_sw;

    // Precomputed dimensions
    logic [15:0]           r_out_h;
    logic [15:0]           r_out_w;
    logic [15:0]           r_channel_stride;       // H * W
    logic [15:0]           r_out_channel_stride;   // out_h * out_w
    logic [15:0]           r_pool_area;            // kh * kw

    // Loop counters
    logic [15:0]           r_c;    // channel counter
    logic [15:0]           r_oh;   // output height counter
    logic [15:0]           r_ow;   // output width counter
    logic [15:0]           r_ky;   // kernel y counter
    logic [15:0]           r_kx;   // kernel x counter

    // Accumulator
    logic signed [31:0]    r_acc;

    // =========================================================================
    // FSM next-state logic
    // =========================================================================
    always_comb begin
        w_state = r_state;
        case (r_state)
            AP_IDLE: begin
                if (cmd_valid)
                    w_state = AP_READ;
            end
            AP_READ: begin
                // SRAM read issued this cycle; need 1 wait cycle for registered output
                w_state = AP_SRAM_WAIT;
            end
            AP_SRAM_WAIT: begin
                // SRAM registered output now valid at start of next state
                w_state = AP_ACCUM;
            end
            AP_ACCUM: begin
                // Accumulate current element, check if kernel traversal complete
                if (r_ky + 16'd1 >= {8'd0, r_kh} && r_kx + 16'd1 >= {8'd0, r_kw})
                    w_state = AP_WRITE;  // kernel done, go write mean
                else
                    w_state = AP_READ;   // more kernel elements to read
            end
            AP_WRITE: begin
                // Write result, advance to next output pixel or finish
                // Determine if all output pixels are done after advancing
                // (logic handled in registered block; here we check next-pixel vs done)
                if (r_ow + 16'd1 >= r_out_w && r_oh + 16'd1 >= r_out_h && r_c + 16'd1 >= r_C)
                    w_state = AP_DONE;
                else
                    w_state = AP_READ;
            end
            AP_DONE: begin
                w_state = AP_IDLE;
            end
            default: w_state = AP_IDLE;
        endcase
    end

    // =========================================================================
    // FSM registered outputs
    // =========================================================================
    // Mean computation wires
    logic signed [31:0] w_mean;
    logic signed [31:0] w_pool_area_signed;
    logic signed [31:0] w_rounding_bias;

    assign w_pool_area_signed = $signed({{16{1'b0}}, r_pool_area});
    assign w_rounding_bias    = w_pool_area_signed >>> 1; // pool_area / 2
    assign w_mean             = (r_acc + w_rounding_bias) / w_pool_area_signed;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            r_state             <= AP_IDLE;
            r_src_base          <= 16'h0000;
            r_dst_base          <= 16'h0000;
            r_C                 <= 16'h0000;
            r_H                 <= 16'h0000;
            r_W                 <= 16'h0000;
            r_kh                <= 8'h00;
            r_kw                <= 8'h00;
            r_sh                <= 8'h00;
            r_sw                <= 8'h00;
            r_out_h             <= 16'h0000;
            r_out_w             <= 16'h0000;
            r_channel_stride    <= 16'h0000;
            r_out_channel_stride <= 16'h0000;
            r_pool_area         <= 16'h0000;
            r_c                 <= 16'h0000;
            r_oh                <= 16'h0000;
            r_ow                <= 16'h0000;
            r_ky                <= 16'h0000;
            r_kx                <= 16'h0000;
            r_acc               <= 32'sh0000_0000;
            sram_rd_en          <= 1'b0;
            sram_rd_addr        <= '0;
            sram_wr_en          <= 1'b0;
            sram_wr_addr        <= '0;
            sram_wr_data        <= 8'h00;
        end else begin
            r_state <= w_state;

            // Default: de-assert strobes
            sram_rd_en <= 1'b0;
            sram_wr_en <= 1'b0;

            case (r_state)
                AP_IDLE: begin
                    if (cmd_valid) begin
                        // Latch command parameters
                        r_src_base          <= cmd_src_base;
                        r_dst_base          <= cmd_dst_base;
                        r_C                 <= cmd_C;
                        r_H                 <= cmd_H;
                        r_W                 <= cmd_W;
                        r_kh                <= cmd_kh;
                        r_kw                <= cmd_kw;
                        r_sh                <= cmd_sh;
                        r_sw                <= cmd_sw;

                        // Precompute dimensions
                        r_out_h             <= (cmd_H - {8'd0, cmd_kh}) / {8'd0, cmd_sh} + 16'd1;
                        r_out_w             <= (cmd_W - {8'd0, cmd_kw}) / {8'd0, cmd_sw} + 16'd1;
                        r_channel_stride    <= cmd_H * cmd_W;
                        r_out_channel_stride <= ((cmd_H - {8'd0, cmd_kh}) / {8'd0, cmd_sh} + 16'd1) *
                                                ((cmd_W - {8'd0, cmd_kw}) / {8'd0, cmd_sw} + 16'd1);
                        r_pool_area         <= {8'd0, cmd_kh} * {8'd0, cmd_kw};

                        // Initialize counters
                        r_c  <= 16'h0000;
                        r_oh <= 16'h0000;
                        r_ow <= 16'h0000;
                        r_ky <= 16'h0000;
                        r_kx <= 16'h0000;

                        // Initialize accumulator
                        r_acc <= 32'sh0000_0000;
                    end
                end

                AP_READ: begin
                    // Issue SRAM read for current kernel element
                    sram_rd_en   <= 1'b1;
                    sram_rd_addr <= r_src_base[SRAM0_AW-1:0]
                                  + r_c  * r_channel_stride
                                  + (r_oh * {8'd0, r_sh} + r_ky) * r_W
                                  + (r_ow * {8'd0, r_sw} + r_kx);
                end

                AP_SRAM_WAIT: begin
                    // Wait for SRAM synchronous read (registered output)
                    // sram_rd_data will be valid at start of next state (AP_ACCUM)
                end

                AP_ACCUM: begin
                    // Sign-extend INT8 to INT32 and accumulate
                    r_acc <= r_acc + $signed({{24{sram_rd_data[7]}}, sram_rd_data});

                    // Advance kernel position
                    if (r_kx + 16'd1 >= {8'd0, r_kw}) begin
                        r_kx <= 16'h0000;
                        r_ky <= r_ky + 16'd1;
                        // If ky+1 >= kh, kernel traversal is done -> AP_WRITE (handled in next-state)
                    end else begin
                        r_kx <= r_kx + 16'd1;
                    end
                end

                AP_WRITE: begin
                    // Compute mean with rounding and saturate to INT8
                    sram_wr_en   <= 1'b1;
                    sram_wr_addr <= r_dst_base[SRAM0_AW-1:0]
                                  + r_c * r_out_channel_stride
                                  + r_oh * r_out_w
                                  + r_ow;

                    // Saturate mean to INT8 range [-128, 127]
                    if (w_mean > 32'sh0000_007F)
                        sram_wr_data <= 8'h7F;       // +127
                    else if (w_mean < -32'sh0000_0080)
                        sram_wr_data <= 8'h80;       // -128
                    else
                        sram_wr_data <= w_mean[7:0];

                    // Reset accumulator and kernel counters for next output pixel
                    r_acc <= 32'sh0000_0000;
                    r_ky  <= 16'h0000;
                    r_kx  <= 16'h0000;

                    // Advance output position: ow -> oh -> c
                    if (r_ow + 16'd1 >= r_out_w) begin
                        r_ow <= 16'h0000;
                        if (r_oh + 16'd1 >= r_out_h) begin
                            r_oh <= 16'h0000;
                            r_c  <= r_c + 16'd1;
                            // If c+1 >= C, all done -> AP_DONE (handled in next-state)
                        end else begin
                            r_oh <= r_oh + 16'd1;
                        end
                    end else begin
                        r_ow <= r_ow + 16'd1;
                    end
                end

                AP_DONE: begin
                    // Single-cycle done pulse, return to idle
                end

                default: ;
            endcase
        end
    end

    // =========================================================================
    // Status outputs
    // =========================================================================
    assign busy = (r_state != AP_IDLE);
    assign done = (r_state == AP_DONE);

endmodule

`default_nettype wire
