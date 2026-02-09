// =============================================================================
// Banked SRAM with optional Ping-Pong
// =============================================================================
`default_nettype none

module banked_sram #(
    parameter int NUM_BANKS  = 4,
    parameter int BANK_DEPTH = 4096,
    parameter int DATA_W     = 8,
    parameter int PINGPONG   = 1,
    parameter int ADDR_W     = $clog2(BANK_DEPTH),
    parameter int BSEL_W     = $clog2(NUM_BANKS)
)(
    input  wire                  clk,
    input  wire                  rst_n,
    // Read port
    input  wire                  rd_en,
    input  wire  [BSEL_W-1:0]   rd_bank_sel,
    input  wire  [ADDR_W-1:0]   rd_addr,
    input  wire                  rd_pingpong_sel,
    output logic [DATA_W-1:0]   rd_data,
    output logic                 rd_valid,
    // Write port
    input  wire                  wr_en,
    input  wire  [BSEL_W-1:0]   wr_bank_sel,
    input  wire  [ADDR_W-1:0]   wr_addr,
    input  wire                  wr_pingpong_sel,
    input  wire  [DATA_W-1:0]   wr_data
);

    localparam int TOTAL_BANKS = PINGPONG ? NUM_BANKS * 2 : NUM_BANKS;
    localparam int TSEL_W = $clog2(TOTAL_BANKS);

    // Per-bank signals
    logic                 bank_en_a [TOTAL_BANKS];
    logic                 bank_we_a [TOTAL_BANKS];
    logic [ADDR_W-1:0]   bank_addr_a [TOTAL_BANKS];
    logic [DATA_W-1:0]   bank_din_a [TOTAL_BANKS];
    logic [DATA_W-1:0]   bank_dout_a [TOTAL_BANKS];

    logic                 bank_en_b [TOTAL_BANKS];
    logic                 bank_we_b [TOTAL_BANKS];
    logic [ADDR_W-1:0]   bank_addr_b [TOTAL_BANKS];
    logic [DATA_W-1:0]   bank_din_b [TOTAL_BANKS];
    logic [DATA_W-1:0]   bank_dout_b [TOTAL_BANKS];

    // Instantiate SRAM banks
    genvar g;
    generate
        for (g = 0; g < TOTAL_BANKS; g++) begin : gen_banks
            sram_dp #(
                .DEPTH (BANK_DEPTH),
                .WIDTH (DATA_W)
            ) u_bank (
                .clk    (clk),
                .en_a   (bank_en_a[g]),
                .we_a   (bank_we_a[g]),
                .addr_a (bank_addr_a[g]),
                .din_a  (bank_din_a[g]),
                .dout_a (bank_dout_a[g]),
                .en_b   (bank_en_b[g]),
                .we_b   (bank_we_b[g]),
                .addr_b (bank_addr_b[g]),
                .din_b  (bank_din_b[g]),
                .dout_b (bank_dout_b[g])
            );
        end
    endgenerate

    // Bank index calculation
    logic [TSEL_W-1:0] rd_bank_idx;
    logic [TSEL_W-1:0] wr_bank_idx;

    always_comb begin
        if (PINGPONG) begin
            rd_bank_idx = {rd_pingpong_sel, rd_bank_sel};
            wr_bank_idx = {wr_pingpong_sel, wr_bank_sel};
        end else begin
            rd_bank_idx = {{(TSEL_W-BSEL_W){1'b0}}, rd_bank_sel};
            wr_bank_idx = {{(TSEL_W-BSEL_W){1'b0}}, wr_bank_sel};
        end
    end

    // Drive bank signals
    // Port A used for reads, Port B used for writes
    always_comb begin
        for (int i = 0; i < TOTAL_BANKS; i++) begin
            // Read (Port A)
            bank_en_a[i]   = rd_en && (rd_bank_idx == i[TSEL_W-1:0]);
            bank_we_a[i]   = 1'b0;
            bank_addr_a[i] = rd_addr;
            bank_din_a[i]  = {DATA_W{1'b0}};
            // Write (Port B)
            bank_en_b[i]   = wr_en && (wr_bank_idx == i[TSEL_W-1:0]);
            bank_we_b[i]   = wr_en && (wr_bank_idx == i[TSEL_W-1:0]);
            bank_addr_b[i] = wr_addr;
            bank_din_b[i]  = wr_data;
        end
    end

    // Read data mux (registered for timing)
    logic [TSEL_W-1:0] rd_bank_idx_d;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rd_valid     <= 1'b0;
            rd_bank_idx_d <= '0;
        end else begin
            rd_valid      <= rd_en;
            rd_bank_idx_d <= rd_bank_idx;
        end
    end

    always_comb begin
        rd_data = bank_dout_a[rd_bank_idx_d];
    end

endmodule

`default_nettype wire
