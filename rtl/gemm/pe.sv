// =============================================================================
// Processing Element for Systolic Array
// Passes data east (a) and south (b) with 1-cycle delay
// =============================================================================
`default_nettype none

module pe #(
    parameter int DATA_W = 8,
    parameter int ACC_W  = 32
)(
    input  wire                        clk,
    input  wire                        rst_n,
    input  wire                        clear_acc,
    input  wire                        en,
    // Data from west / north
    input  wire  signed [DATA_W-1:0]   a_in,
    input  wire  signed [DATA_W-1:0]   b_in,
    // Data to east / south (delayed 1 cycle)
    output logic signed [DATA_W-1:0]   a_out,
    output logic signed [DATA_W-1:0]   b_out,
    // Accumulated result
    output logic signed [ACC_W-1:0]    acc_out
);

    // Pass-through registers
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            a_out <= '0;
            b_out <= '0;
        end else if (en) begin
            a_out <= a_in;
            b_out <= b_in;
        end
    end

    // MAC unit
    mac_int8 #(
        .DATA_W (DATA_W),
        .ACC_W  (ACC_W)
    ) u_mac (
        .clk       (clk),
        .rst_n     (rst_n),
        .clear_acc (clear_acc),
        .en        (en),
        .a_in      (a_in),
        .b_in      (b_in),
        .acc_out   (acc_out)
    );

endmodule

`default_nettype wire
