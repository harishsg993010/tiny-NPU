// =============================================================================
// MxN Systolic Array (default 16x16 = 256 MACs)
// Weight-stationary style: A flows east, B flows south
// =============================================================================
`default_nettype none

module systolic_array #(
    parameter int M      = 16,   // rows
    parameter int N      = 16,   // cols
    parameter int DATA_W = 8,
    parameter int ACC_W  = 32
)(
    input  wire                        clk,
    input  wire                        rst_n,
    input  wire                        clear_acc,
    input  wire                        en,
    // Input: one column of A per row, one row of B per column
    input  wire  signed [DATA_W-1:0]   a_col  [M],
    input  wire  signed [DATA_W-1:0]   b_row  [N],
    // Output: MxN accumulator grid
    output logic signed [ACC_W-1:0]    acc_out [M][N],
    output logic                       acc_valid
);

    // Internal wiring
    logic signed [DATA_W-1:0] a_wire [M][N+1];
    logic signed [DATA_W-1:0] b_wire [M+1][N];
    logic signed [ACC_W-1:0]  pe_acc [M][N];

    // Connect inputs
    genvar gi, gj;
    generate
        for (gi = 0; gi < M; gi++) begin : gen_a_input
            assign a_wire[gi][0] = a_col[gi];
        end
        for (gj = 0; gj < N; gj++) begin : gen_b_input
            assign b_wire[0][gj] = b_row[gj];
        end
    endgenerate

    // Instantiate MxN PE grid
    generate
        for (gi = 0; gi < M; gi++) begin : gen_row
            for (gj = 0; gj < N; gj++) begin : gen_col
                pe #(
                    .DATA_W (DATA_W),
                    .ACC_W  (ACC_W)
                ) u_pe (
                    .clk       (clk),
                    .rst_n     (rst_n),
                    .clear_acc (clear_acc),
                    .en        (en),
                    .a_in      (a_wire[gi][gj]),
                    .b_in      (b_wire[gi][gj]),
                    .a_out     (a_wire[gi][gj+1]),
                    .b_out     (b_wire[gi+1][gj]),
                    .acc_out   (pe_acc[gi][gj])
                );
            end
        end
    endgenerate

    // Output accumulator grid
    generate
        for (gi = 0; gi < M; gi++) begin : gen_acc_row
            for (gj = 0; gj < N; gj++) begin : gen_acc_col
                assign acc_out[gi][gj] = pe_acc[gi][gj];
            end
        end
    endgenerate

    // acc_valid: after data has propagated through entire array
    // Latency = K_depth (streaming) + M + N - 2 (wave propagation) + 2 (MAC pipeline)
    // We track with a shift register driven externally
    // For now, provide a simple counter-based valid
    localparam int DRAIN_LATENCY = M + N;
    localparam int CNT_W = $clog2(DRAIN_LATENCY+1);
    logic [CNT_W-1:0] valid_cnt;
    localparam [CNT_W-1:0] DRAIN_MAX = CNT_W'(DRAIN_LATENCY);

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_cnt <= '0;
            acc_valid <= 1'b0;
        end else if (clear_acc) begin
            valid_cnt <= '0;
            acc_valid <= 1'b0;
        end else if (en) begin
            if (valid_cnt < DRAIN_MAX)
                valid_cnt <= valid_cnt + 1;
            else
                acc_valid <= 1'b1;
        end
    end

endmodule

`default_nettype wire
