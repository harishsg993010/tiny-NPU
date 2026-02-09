// ===========================================================================
//  AXI4 DMA Read Master
//
//  Accepts a command (base address + byte count), breaks it into legal AXI4
//  INCR bursts (respecting MAX_BURST_LEN and 4 KB boundary), issues AR
//  requests, receives R data, and forwards it downstream with backpressure.
//
//  Only one burst outstanding at a time (simple, area-efficient).
//  Fully synthesizable -- no delays.
// ===========================================================================
module axi_dma_rd
    import axi_types_pkg::*;
#(
    parameter int AXI_DATA_W     = 128,
    parameter int AXI_ADDR_W     = 32,
    parameter int MAX_BURST_LEN  = 16          // max beats per burst (1..256)
) (
    input  logic                      clk,
    input  logic                      rst_n,

    // -----------------------------------------------------------------------
    //  Command interface (from control logic)
    // -----------------------------------------------------------------------
    input  logic                      cmd_valid,
    output logic                      cmd_ready,
    input  logic [AXI_ADDR_W-1:0]    cmd_addr,      // byte-aligned start addr
    input  logic [23:0]              cmd_len,        // total bytes to read
    input  logic [3:0]               cmd_tag,

    // -----------------------------------------------------------------------
    //  AXI4 Master -- AR channel
    // -----------------------------------------------------------------------
    output logic [AXI_ID_W-1:0]      m_axi_arid,
    output logic [AXI_ADDR_W-1:0]    m_axi_araddr,
    output logic [7:0]               m_axi_arlen,    // beats - 1
    output logic [2:0]               m_axi_arsize,
    output logic [1:0]               m_axi_arburst,
    output logic                     m_axi_arvalid,
    input  logic                     m_axi_arready,

    // -----------------------------------------------------------------------
    //  AXI4 Master -- R channel
    // -----------------------------------------------------------------------
    input  logic [AXI_ID_W-1:0]      m_axi_rid,
    input  logic [AXI_DATA_W-1:0]    m_axi_rdata,
    input  logic [1:0]               m_axi_rresp,
    input  logic                     m_axi_rlast,
    input  logic                     m_axi_rvalid,
    output logic                     m_axi_rready,

    // -----------------------------------------------------------------------
    //  Data output (to internal consumers)
    // -----------------------------------------------------------------------
    output logic                     data_valid,
    input  logic                     data_ready,
    output logic [AXI_DATA_W-1:0]   data_out,
    output logic                     data_last,
    output logic [3:0]              data_tag,

    // -----------------------------------------------------------------------
    //  Status
    // -----------------------------------------------------------------------
    output logic                     busy,
    output logic                     done,
    output logic                     error
);

    // =======================================================================
    //  Derived constants
    // =======================================================================
    localparam int BYTES_PER_BEAT = AXI_DATA_W / 8;
    localparam int BEAT_SHIFT     = $clog2(BYTES_PER_BEAT);  // e.g. 4 for 128-bit

    // =======================================================================
    //  FSM
    // =======================================================================
    typedef enum logic [2:0] {
        S_IDLE,
        S_CALC_BURSTS,
        S_SEND_AR,
        S_RECV_DATA,
        S_DONE
    } state_t;

    state_t state, state_nxt;

    // =======================================================================
    //  Internal registers
    // =======================================================================
    logic [AXI_ADDR_W-1:0] cur_addr;        // current burst start address
    logic [23:0]           bytes_remaining;  // total bytes left to transfer
    logic [7:0]            burst_len;        // beats in current burst - 1
    logic [7:0]            beat_cnt;         // beats received so far in burst
    logic [23:0]           total_beats;      // total beats across all bursts
    logic [23:0]           beats_done;       // beats completed so far (global)
    logic [3:0]            tag_r;
    logic                  error_r;

    // =======================================================================
    //  Burst-length calculation
    //  Must not exceed:
    //    1) MAX_BURST_LEN beats
    //    2) Remaining beats for the transfer
    //    3) Beats until the next 4 KB boundary
    // =======================================================================
    logic [23:0] remaining_beats;
    logic [11:0] addr_in_4k;
    logic [8:0]  beats_to_4k;        // beats until 4 KB boundary (max 256)
    logic [8:0]  max_burst_beats;
    logic [8:0]  capped_beats;

    assign remaining_beats = bytes_remaining[23:BEAT_SHIFT]
                           + (|bytes_remaining[BEAT_SHIFT-1:0] ? 24'd1 : 24'd0);

    // Byte offset within the current 4 KB page
    assign addr_in_4k = cur_addr[11:0];

    // Number of beats until we hit the next 4 KB boundary
    // (4096 - addr_in_4k) / BYTES_PER_BEAT, rounded up
    wire [12:0] bytes_to_4k = 13'd4096 - {1'b0, addr_in_4k};
    assign beats_to_4k = bytes_to_4k[12:BEAT_SHIFT]
                        + (|bytes_to_4k[BEAT_SHIFT-1:0] ? 9'd1 : 9'd0);

    // Pick the minimum of the three constraints
    assign max_burst_beats = (remaining_beats < MAX_BURST_LEN[8:0])
                              ? remaining_beats[8:0]
                              : MAX_BURST_LEN[8:0];
    assign capped_beats    = (max_burst_beats < beats_to_4k)
                              ? max_burst_beats
                              : beats_to_4k;

    // =======================================================================
    //  FSM next-state and output logic
    // =======================================================================
    always_comb begin
        state_nxt = state;

        case (state)
            S_IDLE:
                if (cmd_valid)
                    state_nxt = S_CALC_BURSTS;

            S_CALC_BURSTS:
                if (bytes_remaining == 24'd0)
                    state_nxt = S_DONE;
                else
                    state_nxt = S_SEND_AR;

            S_SEND_AR:
                if (m_axi_arvalid && m_axi_arready)
                    state_nxt = S_RECV_DATA;

            S_RECV_DATA:
                if (m_axi_rvalid && m_axi_rready && m_axi_rlast) begin
                    if (bytes_remaining == 24'd0)
                        state_nxt = S_DONE;
                    else
                        state_nxt = S_CALC_BURSTS;
                end

            S_DONE:
                state_nxt = S_IDLE;

            default:
                state_nxt = S_IDLE;
        endcase
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            state <= S_IDLE;
        else
            state <= state_nxt;
    end

    // =======================================================================
    //  Datapath
    // =======================================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            cur_addr        <= '0;
            bytes_remaining <= 24'd0;
            burst_len       <= 8'd0;
            beat_cnt        <= 8'd0;
            total_beats     <= 24'd0;
            beats_done      <= 24'd0;
            tag_r           <= 4'd0;
            error_r         <= 1'b0;
        end else begin
            case (state)
                // ---------------------------------------------------------
                S_IDLE: begin
                    if (cmd_valid) begin
                        cur_addr        <= cmd_addr;
                        bytes_remaining <= cmd_len;
                        tag_r           <= cmd_tag;
                        error_r         <= 1'b0;
                        beats_done      <= 24'd0;
                        // Pre-compute total beats
                        total_beats     <= cmd_len[23:BEAT_SHIFT]
                                         + (|cmd_len[BEAT_SHIFT-1:0] ? 24'd1 : 24'd0);
                    end
                end

                // ---------------------------------------------------------
                S_CALC_BURSTS: begin
                    if (bytes_remaining != 24'd0) begin
                        burst_len <= capped_beats[7:0] - 8'd1;
                        beat_cnt  <= 8'd0;
                    end
                end

                // ---------------------------------------------------------
                S_SEND_AR: begin
                    // Nothing to update here; AR signals driven combinationally below
                end

                // ---------------------------------------------------------
                S_RECV_DATA: begin
                    if (m_axi_rvalid && m_axi_rready) begin
                        beat_cnt   <= beat_cnt + 8'd1;
                        beats_done <= beats_done + 24'd1;

                        // Advance address
                        cur_addr <= cur_addr + AXI_ADDR_W'(BYTES_PER_BEAT);

                        // Decrement remaining bytes
                        if (bytes_remaining >= BYTES_PER_BEAT[23:0])
                            bytes_remaining <= bytes_remaining - BYTES_PER_BEAT[23:0];
                        else
                            bytes_remaining <= 24'd0;

                        // Check for slave error
                        if (m_axi_rresp != AXI_RESP_OKAY)
                            error_r <= 1'b1;
                    end
                end

                // ---------------------------------------------------------
                S_DONE: begin
                    // Will transition to IDLE next cycle
                end

                default: ;
            endcase
        end
    end

    // =======================================================================
    //  AXI4 AR channel outputs
    // =======================================================================
    assign m_axi_arid    = tag_r;
    assign m_axi_araddr  = cur_addr;
    assign m_axi_arlen   = burst_len;
    assign m_axi_arsize  = axi_size_from_bytes(BYTES_PER_BEAT);
    assign m_axi_arburst = AXI_BURST_INCR;
    assign m_axi_arvalid = (state == S_SEND_AR);

    // =======================================================================
    //  R channel -> data output with backpressure
    //  We only accept R data when the downstream consumer is ready.
    // =======================================================================
    assign m_axi_rready = (state == S_RECV_DATA) && data_ready;

    assign data_valid   = (state == S_RECV_DATA) && m_axi_rvalid;
    assign data_out     = m_axi_rdata;
    assign data_tag     = tag_r;
    // data_last is asserted on the final beat of the entire transfer
    assign data_last    = (state == S_RECV_DATA) && m_axi_rvalid
                          && m_axi_rlast && (bytes_remaining <= BYTES_PER_BEAT[23:0]);

    // =======================================================================
    //  Command / status
    // =======================================================================
    assign cmd_ready = (state == S_IDLE);
    assign busy      = (state != S_IDLE) && (state != S_DONE);
    assign done      = (state == S_DONE);
    assign error     = error_r;

endmodule : axi_dma_rd
