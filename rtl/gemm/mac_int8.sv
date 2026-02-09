// =============================================================================
// INT8 Multiply-Accumulate Unit (2-stage pipeline)
// =============================================================================
`default_nettype none

module mac_int8 #(
    parameter int DATA_W = 8,
    parameter int ACC_W  = 32
)(
    input  wire                        clk,
    input  wire                        rst_n,
    input  wire                        clear_acc,
    input  wire                        en,
    input  wire  signed [DATA_W-1:0]   a_in,
    input  wire  signed [DATA_W-1:0]   b_in,
    output logic signed [ACC_W-1:0]    acc_out
);

    // Stage 1: multiply
    logic signed [2*DATA_W-1:0] product;
    logic                       en_d1;
    logic                       clear_d1;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            product  <= '0;
            en_d1    <= 1'b0;
            clear_d1 <= 1'b0;
        end else begin
            en_d1    <= en;
            clear_d1 <= clear_acc;
            if (en)
                product <= a_in * b_in;
            else
                product <= '0;
        end
    end

    // Stage 2: accumulate
    logic signed [ACC_W-1:0] acc_reg;
    logic signed [ACC_W:0]   acc_ext;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            acc_reg <= '0;
        end else if (clear_d1) begin
            acc_reg <= '0;
        end else if (en_d1) begin
            acc_ext = {acc_reg[ACC_W-1], acc_reg} + {{(ACC_W-2*DATA_W+1){product[2*DATA_W-1]}}, product};
            // Saturation: overflow when extended sign differs from result sign
            if (acc_ext[ACC_W] != acc_ext[ACC_W-1]) begin
                if (acc_ext[ACC_W])  // negative overflow
                    acc_reg <= {1'b1, {(ACC_W-1){1'b0}}};
                else  // positive overflow
                    acc_reg <= {1'b0, {(ACC_W-1){1'b1}}};
            end else begin
                acc_reg <= acc_ext[ACC_W-1:0];
            end
        end
    end

    assign acc_out = acc_reg;

endmodule

`default_nettype wire
