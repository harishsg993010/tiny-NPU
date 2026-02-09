// ===========================================================================
//  AXI4 DMA Write Master
//
//  Accepts a command (base address + byte count), breaks it into legal AXI4
//  INCR bursts (respecting MAX_BURST_LEN and 4 KB boundary), issues AW
//  requests, streams W data with WLAST, and collects B responses.
//
//  One burst outstanding at a time (simple, area-efficient).
//  Fully synthesizable -- no delays.
// ===========================================================================
module axi_dma_wr
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
    input  logic [23:0]              cmd_len,        // total bytes to write
    input  logic [3:0]               cmd_tag,

    // -----------------------------------------------------------------------
    //  AXI4 Master -- AW channel
    // -----------------------------------------------------------------------
    output logic [AXI_ID_W-1:0]      m_axi_awid,
    output logic [AXI_ADDR_W-1:0]    m_axi_awaddr,
    output logic [7:0]               m_axi_awlen,    // beats - 1
    output logic [2:0]               m_axi_awsize,
    output logic [1:0]               m_axi_awburst,
    output logic                     m_axi_awvalid,
    input  logic                     m_axi_awready,

    // -----------------------------------------------------------------------
    //  AXI4 Master -- W channel
    // -----------------------------------------------------------------------
    output logic [AXI_DATA_W-1:0]    m_axi_wdata,
    output logic [AXI_DATA_W/8-1:0]  m_axi_wstrb,
    output logic                     m_axi_wlast,
    output logic                     m_axi_wvalid,
    input  logic                     m_axi_wready,

    // -----------------------------------------------------------------------
    //  AXI4 Master -- B channel
    // -----------------------------------------------------------------------
    input  logic [AXI_ID_W-1:0]      m_axi_bid,
    input  logic [1:0]               m_axi_bresp,
    input  logic                     m_axi_bvalid,
    output logic                     m_axi_bready,

    // -----------------------------------------------------------------------
    //  Data input (from internal producers)
    // -----------------------------------------------------------------------
    input  logic                     data_valid,
    output logic                     data_ready,
    input  logic [AXI_DATA_W-1:0]   data_in,
    input  logic                     data_last,

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
    localparam int BEAT_SHIFT     = $clog2(BYTES_PER_BEAT);

    // =======================================================================
    //  FSM
    // =======================================================================
    typedef enum logic [2:0] {
        S_IDLE,
        S_CALC_BURSTS,
        S_SEND_AW,
        S_SEND_WDATA,
        S_WAIT_BRESP,
        S_DONE
    } state_t;

    state_t state, state_nxt;

    // =======================================================================
    //  Internal registers
    // =======================================================================
    logic [AXI_ADDR_W-1:0] cur_addr;
    logic [23:0]           bytes_remaining;
    logic [7:0]            burst_len;        // current burst beats - 1
    logic [7:0]            beat_cnt;         // beats sent in current burst
    logic [3:0]            tag_r;
    logic                  error_r;

    // =======================================================================
    //  Burst-length calculation (identical to read DMA)
    // =======================================================================
    logic [23:0] remaining_beats;
    logic [11:0] addr_in_4k;
    logic [8:0]  beats_to_4k;
    logic [8:0]  max_burst_beats;
    logic [8:0]  capped_beats;

    assign remaining_beats = bytes_remaining[23:BEAT_SHIFT]
                           + (|bytes_remaining[BEAT_SHIFT-1:0] ? 24'd1 : 24'd0);

    assign addr_in_4k = cur_addr[11:0];

    wire [12:0] bytes_to_4k = 13'd4096 - {1'b0, addr_in_4k};
    assign beats_to_4k = bytes_to_4k[12:BEAT_SHIFT]
                        + (|bytes_to_4k[BEAT_SHIFT-1:0] ? 9'd1 : 9'd0);

    assign max_burst_beats = (remaining_beats < MAX_BURST_LEN[8:0])
                              ? remaining_beats[8:0]
                              : MAX_BURST_LEN[8:0];
    assign capped_beats    = (max_burst_beats < beats_to_4k)
                              ? max_burst_beats
                              : beats_to_4k;

    // =======================================================================
    //  Write-strobe generation
    //  Full strobes for all beats except possibly the very last beat of the
    //  entire transfer, where we mask off trailing bytes.
    // =======================================================================
    logic [AXI_DATA_W/8-1:0] full_strb;
    logic [AXI_DATA_W/8-1:0] last_strb;
    logic                     is_last_beat_of_xfer;

    assign full_strb = {(AXI_DATA_W/8){1'b1}};

    // On the final beat, only enable the strobes for valid trailing bytes
    // If the total length is an exact multiple of BYTES_PER_BEAT, all strobes
    // are active on the last beat too.
    logic [BEAT_SHIFT-1:0] tail_bytes_raw;
    assign tail_bytes_raw = bytes_remaining[BEAT_SHIFT-1:0];

    always_comb begin
        if (tail_bytes_raw == '0) begin
            // Exact multiple -- all lanes valid
            last_strb = full_strb;
        end else begin
            // Only the lower tail_bytes_raw lanes are valid
            last_strb = '0;
            for (int i = 0; i < BYTES_PER_BEAT; i++) begin
                if (i[BEAT_SHIFT-1:0] < tail_bytes_raw)
                    last_strb[i] = 1'b1;
            end
        end
    end

    assign is_last_beat_of_xfer = (beat_cnt == burst_len)
                                  && (bytes_remaining <= BYTES_PER_BEAT[23:0]);

    // =======================================================================
    //  FSM next-state logic
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
                    state_nxt = S_SEND_AW;

            S_SEND_AW:
                if (m_axi_awvalid && m_axi_awready)
                    state_nxt = S_SEND_WDATA;

            S_SEND_WDATA:
                if (m_axi_wvalid && m_axi_wready && m_axi_wlast)
                    state_nxt = S_WAIT_BRESP;

            S_WAIT_BRESP:
                if (m_axi_bvalid && m_axi_bready) begin
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
                S_SEND_AW: begin
                    // AW driven combinationally; nothing to update
                end

                // ---------------------------------------------------------
                S_SEND_WDATA: begin
                    if (m_axi_wvalid && m_axi_wready) begin
                        beat_cnt <= beat_cnt + 8'd1;

                        // Advance address
                        cur_addr <= cur_addr + AXI_ADDR_W'(BYTES_PER_BEAT);

                        // Decrement remaining
                        if (bytes_remaining >= BYTES_PER_BEAT[23:0])
                            bytes_remaining <= bytes_remaining - BYTES_PER_BEAT[23:0];
                        else
                            bytes_remaining <= 24'd0;
                    end
                end

                // ---------------------------------------------------------
                S_WAIT_BRESP: begin
                    if (m_axi_bvalid && m_axi_bready) begin
                        if (m_axi_bresp != AXI_RESP_OKAY)
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
    //  AXI4 AW channel outputs
    // =======================================================================
    assign m_axi_awid    = tag_r;
    assign m_axi_awaddr  = cur_addr;
    assign m_axi_awlen   = burst_len;
    assign m_axi_awsize  = axi_size_from_bytes(BYTES_PER_BEAT);
    assign m_axi_awburst = AXI_BURST_INCR;
    assign m_axi_awvalid = (state == S_SEND_AW);

    // =======================================================================
    //  AXI4 W channel outputs
    //  Forward upstream data directly to the W channel; stall if AXI not ready
    //  or if upstream has no data.
    // =======================================================================
    assign m_axi_wdata  = data_in;
    assign m_axi_wstrb  = is_last_beat_of_xfer ? last_strb : full_strb;
    assign m_axi_wlast  = (beat_cnt == burst_len);
    assign m_axi_wvalid = (state == S_SEND_WDATA) && data_valid;

    // Accept upstream data only when AXI W channel is also ready
    assign data_ready   = (state == S_SEND_WDATA) && m_axi_wready;

    // =======================================================================
    //  AXI4 B channel
    // =======================================================================
    assign m_axi_bready = (state == S_WAIT_BRESP);

    // =======================================================================
    //  Command / status
    // =======================================================================
    assign cmd_ready = (state == S_IDLE);
    assign busy      = (state != S_IDLE) && (state != S_DONE);
    assign done      = (state == S_DONE);
    assign error     = error_r;

endmodule : axi_dma_wr
