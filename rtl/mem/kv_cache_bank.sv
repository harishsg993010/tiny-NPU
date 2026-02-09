// =============================================================================
// KV Cache Bank - Head-dim-major layout
// K[layer][head][time][head_dim] with head_dim contiguous
// =============================================================================
`default_nettype none

module kv_cache_bank
    import npu_pkg::*;
#(
    parameter int MAX_LAYERS = 4,
    parameter int MAX_HEADS  = 4,
    parameter int MAX_SEQ    = 512,
    parameter int HEAD_DIM   = 16,
    parameter int DW         = 8
)(
    input  wire                          clk,
    input  wire                          rst_n,

    // Append interface (write one vector per cycle)
    input  wire                          append_valid,
    output logic                         append_ready,
    input  wire  [$clog2(MAX_LAYERS)-1:0] append_layer,
    input  wire  [$clog2(MAX_HEADS)-1:0]  append_head,
    input  wire  [$clog2(MAX_SEQ)-1:0]    append_time,
    input  wire                          append_is_v,
    input  wire  [HEAD_DIM*DW-1:0]       append_data,
    output logic                         append_done,

    // Read interface
    input  wire                          read_req_valid,
    output logic                         read_req_ready,
    input  wire  [$clog2(MAX_LAYERS)-1:0] read_layer,
    input  wire  [$clog2(MAX_HEADS)-1:0]  read_head,
    input  wire  [$clog2(MAX_SEQ)-1:0]    read_time_start,
    input  wire  [$clog2(MAX_SEQ)-1:0]    read_time_len,
    input  wire                          read_is_v,
    output logic                         read_data_valid,
    output logic [HEAD_DIM*DW-1:0]       read_data,
    output logic                         read_data_last
);

    // Total entries: 2 * MAX_LAYERS * MAX_HEADS * MAX_SEQ (K and V)
    localparam int TOTAL_ENTRIES = 2 * MAX_LAYERS * MAX_HEADS * MAX_SEQ;
    localparam int ADDR_W = $clog2(TOTAL_ENTRIES);
    localparam int VEC_W = HEAD_DIM * DW;

    // SRAM storage
    sram_dp #(
        .DEPTH (TOTAL_ENTRIES),
        .WIDTH (VEC_W)
    ) u_kv_mem (
        .clk    (clk),
        .en_a   (rd_en),
        .we_a   (1'b0),
        .addr_a (rd_addr),
        .din_a  ({VEC_W{1'b0}}),
        .dout_a (rd_dout),
        .en_b   (wr_en),
        .we_b   (wr_en),
        .addr_b (wr_addr),
        .din_b  (wr_din),
        .dout_b ()
    );

    logic                 rd_en;
    logic [ADDR_W-1:0]    rd_addr;
    logic [VEC_W-1:0]     rd_dout;
    logic                 wr_en;
    logic [ADDR_W-1:0]    wr_addr;
    logic [VEC_W-1:0]     wr_din;

    // Address calculation
    function automatic logic [ADDR_W-1:0] calc_addr(
        input logic [$clog2(MAX_LAYERS)-1:0] layer,
        input logic [$clog2(MAX_HEADS)-1:0]  head,
        input logic [$clog2(MAX_SEQ)-1:0]    t,
        input logic                          is_v
    );
        logic [ADDR_W-1:0] base;
        base = {is_v, layer, head, t};
        return base;
    endfunction

    // FSM
    typedef enum logic [2:0] {
        S_IDLE,
        S_APPEND,
        S_READ_START,
        S_READ_STREAM,
        S_DONE
    } state_t;

    state_t state, state_next;

    logic [$clog2(MAX_SEQ)-1:0] rd_time_cnt;
    logic [$clog2(MAX_SEQ)-1:0] rd_time_end;
    logic                        rd_pipe_valid;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            state <= S_IDLE;
        else
            state <= state_next;
    end

    always_comb begin
        state_next    = state;
        append_ready  = 1'b0;
        append_done   = 1'b0;
        read_req_ready = 1'b0;
        wr_en         = 1'b0;
        wr_addr       = '0;
        wr_din        = '0;
        rd_en         = 1'b0;
        rd_addr       = '0;

        case (state)
            S_IDLE: begin
                append_ready  = 1'b1;
                read_req_ready = 1'b1;
                if (append_valid) begin
                    state_next = S_APPEND;
                end else if (read_req_valid) begin
                    state_next = S_READ_START;
                end
            end

            S_APPEND: begin
                wr_en   = 1'b1;
                wr_addr = calc_addr(append_layer, append_head, append_time, append_is_v);
                wr_din  = append_data;
                append_done = 1'b1;
                state_next  = S_IDLE;
            end

            S_READ_START: begin
                rd_en   = 1'b1;
                rd_addr = calc_addr(read_layer, read_head, read_time_start, read_is_v);
                state_next = S_READ_STREAM;
            end

            S_READ_STREAM: begin
                if (rd_time_cnt < rd_time_end) begin
                    rd_en   = 1'b1;
                    rd_addr = calc_addr(read_layer, read_head,
                                       read_time_start + rd_time_cnt + 1, read_is_v);
                end
                if (rd_time_cnt >= rd_time_end) begin
                    state_next = S_IDLE;
                end
            end

            default: state_next = S_IDLE;
        endcase
    end

    // Read counter
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rd_time_cnt   <= '0;
            rd_time_end   <= '0;
            rd_pipe_valid <= 1'b0;
        end else begin
            rd_pipe_valid <= rd_en;
            if (state == S_IDLE && read_req_valid) begin
                rd_time_cnt <= '0;
                rd_time_end <= read_time_len;
            end else if (state == S_READ_STREAM) begin
                rd_time_cnt <= rd_time_cnt + 1;
            end
        end
    end

    // Output
    assign read_data_valid = rd_pipe_valid && (state == S_READ_STREAM || state == S_IDLE);
    assign read_data       = rd_dout;
    assign read_data_last  = rd_pipe_valid && (rd_time_cnt >= rd_time_end);

endmodule

`default_nettype wire
