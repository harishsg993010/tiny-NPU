// =============================================================================
// True Dual-Port SRAM - BRAM Inferred for Xilinx
// =============================================================================
`default_nettype none

module sram_dp #(
    parameter int DEPTH  = 4096,
    parameter int WIDTH  = 8,
    parameter int ADDR_W = $clog2(DEPTH)
)(
    input  wire                 clk,
    // Port A
    input  wire                 en_a,
    input  wire                 we_a,
    input  wire  [ADDR_W-1:0]  addr_a,
    input  wire  [WIDTH-1:0]   din_a,
    output logic [WIDTH-1:0]   dout_a,
    // Port B
    input  wire                 en_b,
    input  wire                 we_b,
    input  wire  [ADDR_W-1:0]  addr_b,
    input  wire  [WIDTH-1:0]   din_b,
    output logic [WIDTH-1:0]   dout_b
);

    (* ram_style = "block" *) logic [WIDTH-1:0] mem [0:DEPTH-1];

    // Port A: write-first
    always_ff @(posedge clk) begin
        if (en_a) begin
            if (we_a) begin
                mem[addr_a] <= din_a;
                dout_a      <= din_a;
            end else begin
                dout_a <= mem[addr_a];
            end
        end
    end

    // Port B: write-first
    always_ff @(posedge clk) begin
        if (en_b) begin
            if (we_b) begin
                mem[addr_b] <= din_b;
                dout_b      <= din_b;
            end else begin
                dout_b <= mem[addr_b];
            end
        end
    end

`ifndef SYNTHESIS
    // Assertions
    always @(posedge clk) begin
        if (en_a) begin
            assert (addr_a < DEPTH)
                else $error("SRAM Port A address out of bounds: %0d >= %0d", addr_a, DEPTH);
        end
        if (en_b) begin
            assert (addr_b < DEPTH)
                else $error("SRAM Port B address out of bounds: %0d >= %0d", addr_b, DEPTH);
        end
    end
`endif

endmodule

`default_nettype wire
