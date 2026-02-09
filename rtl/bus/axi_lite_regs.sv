// ===========================================================================
//  AXI4-Lite Slave Register Bank for NPU Control/Status
//
//  Implements full AXI4-Lite write and read protocol with:
//    - CTRL register: START bit auto-clears after 1 cycle (pulse),
//      SOFT_RESET bit holds asserted for 4 cycles then auto-clears.
//    - STATUS register: read-only, driven by external done/busy/error inputs.
//    - All other registers: standard read-write with reset defaults of 0.
//
//  Fully synthesizable -- no delays.
// ===========================================================================
module axi_lite_regs
    import npu_pkg::*;
    import axi_types_pkg::*;
(
    input  logic                      clk,
    input  logic                      rst_n,

    // -----------------------------------------------------------------------
    //  AXI4-Lite slave interface
    // -----------------------------------------------------------------------
    // Write-address channel
    input  logic [LITE_ADDR_W-1:0]    s_axil_awaddr,
    input  logic                      s_axil_awvalid,
    output logic                      s_axil_awready,

    // Write-data channel
    input  logic [LITE_DATA_W-1:0]    s_axil_wdata,
    input  logic [LITE_STRB_W-1:0]    s_axil_wstrb,
    input  logic                      s_axil_wvalid,
    output logic                      s_axil_wready,

    // Write-response channel
    output logic [1:0]                s_axil_bresp,
    output logic                      s_axil_bvalid,
    input  logic                      s_axil_bready,

    // Read-address channel
    input  logic [LITE_ADDR_W-1:0]    s_axil_araddr,
    input  logic                      s_axil_arvalid,
    output logic                      s_axil_arready,

    // Read-data channel
    output logic [LITE_DATA_W-1:0]    s_axil_rdata,
    output logic [1:0]                s_axil_rresp,
    output logic                      s_axil_rvalid,
    input  logic                      s_axil_rready,

    // -----------------------------------------------------------------------
    //  Status inputs (directly drive the read-only STATUS register)
    // -----------------------------------------------------------------------
    input  logic                      done_i,
    input  logic                      busy_i,
    input  logic                      error_i,

    // -----------------------------------------------------------------------
    //  Register value outputs
    // -----------------------------------------------------------------------
    output logic [31:0]               ctrl_o,
    output logic [31:0]               status_o,
    output logic [31:0]               ucode_base_o,
    output logic [31:0]               ucode_len_o,
    output logic [31:0]               ddr_base_act_o,
    output logic [31:0]               ddr_base_wgt_o,
    output logic [31:0]               ddr_base_kv_o,
    output logic [31:0]               ddr_base_out_o,
    output logic [31:0]               model_hidden_o,
    output logic [31:0]               model_heads_o,
    output logic [31:0]               model_head_dim_o,
    output logic [31:0]               seq_len_o,
    output logic [31:0]               token_idx_o,
    output logic [31:0]               debug_ctrl_o,

    // -----------------------------------------------------------------------
    //  Derived control outputs
    // -----------------------------------------------------------------------
    output logic                      start_pulse_o,
    output logic                      soft_reset_o
);

    // =======================================================================
    //  Internal register storage
    // =======================================================================
    logic [31:0] reg_ctrl;
    logic [31:0] reg_ucode_base;
    logic [31:0] reg_ucode_len;
    logic [31:0] reg_ddr_base_act;
    logic [31:0] reg_ddr_base_wgt;
    logic [31:0] reg_ddr_base_kv;
    logic [31:0] reg_ddr_base_out;
    logic [31:0] reg_model_hidden;
    logic [31:0] reg_model_heads;
    logic [31:0] reg_model_head_dim;
    logic [31:0] reg_seq_len;
    logic [31:0] reg_token_idx;
    logic [31:0] reg_debug_ctrl;

    // Soft-reset hold counter (holds reset for 4 cycles)
    logic [2:0]  soft_reset_cnt;

    // =======================================================================
    //  AXI4-Lite Write FSM
    // =======================================================================
    //  We accept AW and W simultaneously (both ready asserted together).
    //  When both handshakes complete, we perform the register write and
    //  present the B response.
    // =======================================================================
    logic        aw_latched;
    logic [31:0] aw_addr_r;
    logic        w_latched;
    logic [31:0] w_data_r;
    logic [3:0]  w_strb_r;

    // AW ready: accept when we have no pending write and B channel is clear
    assign s_axil_awready = !aw_latched && !s_axil_bvalid;
    // W ready: accept when we have no pending write data and B channel is clear
    assign s_axil_wready  = !w_latched  && !s_axil_bvalid;

    // Write response is always OKAY (no decode errors for valid addresses)
    assign s_axil_bresp = AXI_RESP_OKAY;

    // Latch AW and W channels; perform write when both are captured
    logic wr_en;
    logic [31:0] wr_addr;
    logic [31:0] wr_data;
    logic [3:0]  wr_strb;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            aw_latched <= 1'b0;
            aw_addr_r  <= 32'd0;
            w_latched  <= 1'b0;
            w_data_r   <= 32'd0;
            w_strb_r   <= 4'd0;
        end else begin
            // wr_en clearing has priority over new latch-set to prevent
            // latches getting stuck when AW+W arrive simultaneously
            if (wr_en) begin
                aw_latched <= 1'b0;
                w_latched  <= 1'b0;
            end else begin
                if (s_axil_awvalid && s_axil_awready)
                    aw_latched <= 1'b1;
                if (s_axil_wvalid && s_axil_wready)
                    w_latched <= 1'b1;
            end

            if (s_axil_awvalid && s_axil_awready)
                aw_addr_r <= s_axil_awaddr;

            if (s_axil_wvalid && s_axil_wready) begin
                w_data_r <= s_axil_wdata;
                w_strb_r <= s_axil_wstrb;
            end
        end
    end

    // Determine when both AW and W data are available (possibly in same cycle)
    logic aw_fire, w_fire;
    assign aw_fire = (s_axil_awvalid && s_axil_awready) || aw_latched;
    assign w_fire  = (s_axil_wvalid  && s_axil_wready)  || w_latched;
    assign wr_en   = aw_fire && w_fire && !s_axil_bvalid;

    // Select between newly arriving data and latched data
    assign wr_addr = aw_latched ? aw_addr_r : s_axil_awaddr;
    assign wr_data = w_latched  ? w_data_r  : s_axil_wdata;
    assign wr_strb = w_latched  ? w_strb_r  : s_axil_wstrb;

    // BVALID management
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            s_axil_bvalid <= 1'b0;
        else if (wr_en)
            s_axil_bvalid <= 1'b1;
        else if (s_axil_bvalid && s_axil_bready)
            s_axil_bvalid <= 1'b0;
    end

    // =======================================================================
    //  Byte-lane write helper: apply strobes to produce new register value
    // =======================================================================
    function automatic logic [31:0] strb_write(
        input logic [31:0] old_val,
        input logic [31:0] new_val,
        input logic [3:0]  strb
    );
        logic [31:0] result;
        result[ 7: 0] = strb[0] ? new_val[ 7: 0] : old_val[ 7: 0];
        result[15: 8] = strb[1] ? new_val[15: 8] : old_val[15: 8];
        result[23:16] = strb[2] ? new_val[23:16] : old_val[23:16];
        result[31:24] = strb[3] ? new_val[31:24] : old_val[31:24];
        return result;
    endfunction

    // =======================================================================
    //  Register write logic
    // =======================================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            reg_ctrl           <= 32'd0;
            reg_ucode_base     <= 32'd0;
            reg_ucode_len      <= 32'd0;
            reg_ddr_base_act   <= 32'd0;
            reg_ddr_base_wgt   <= 32'd0;
            reg_ddr_base_kv    <= 32'd0;
            reg_ddr_base_out   <= 32'd0;
            reg_model_hidden   <= 32'd0;
            reg_model_heads    <= 32'd0;
            reg_model_head_dim <= 32'd0;
            reg_seq_len        <= 32'd0;
            reg_token_idx      <= 32'd0;
            reg_debug_ctrl     <= 32'd0;
            soft_reset_cnt     <= 3'd0;
        end else begin
            // ----------------------------------------------------------
            //  START bit auto-clear: pulse for exactly 1 cycle
            // ----------------------------------------------------------
            if (reg_ctrl[CTRL_START])
                reg_ctrl[CTRL_START] <= 1'b0;

            // ----------------------------------------------------------
            //  SOFT_RESET auto-clear after 4 cycles
            // ----------------------------------------------------------
            if (reg_ctrl[CTRL_SOFT_RESET]) begin
                if (soft_reset_cnt == 3'd3) begin
                    reg_ctrl[CTRL_SOFT_RESET] <= 1'b0;
                    soft_reset_cnt            <= 3'd0;
                end else begin
                    soft_reset_cnt <= soft_reset_cnt + 3'd1;
                end
            end

            // ----------------------------------------------------------
            //  Register write decode
            // ----------------------------------------------------------
            if (wr_en) begin
                case (wr_addr[7:0])   // decode lower 8 bits of address
                    REG_CTRL[7:0]: begin
                        reg_ctrl <= strb_write(reg_ctrl, wr_data, wr_strb);
                        // Reset the soft-reset counter whenever CTRL is written
                        if (wr_strb[0] && wr_data[CTRL_SOFT_RESET])
                            soft_reset_cnt <= 3'd0;
                    end
                    // REG_STATUS is read-only -- writes are silently ignored
                    REG_UCODE_BASE[7:0]:     reg_ucode_base     <= strb_write(reg_ucode_base,     wr_data, wr_strb);
                    REG_UCODE_LEN[7:0]:      reg_ucode_len      <= strb_write(reg_ucode_len,      wr_data, wr_strb);
                    REG_DDR_BASE_ACT[7:0]:   reg_ddr_base_act   <= strb_write(reg_ddr_base_act,   wr_data, wr_strb);
                    REG_DDR_BASE_WGT[7:0]:   reg_ddr_base_wgt   <= strb_write(reg_ddr_base_wgt,   wr_data, wr_strb);
                    REG_DDR_BASE_KV[7:0]:    reg_ddr_base_kv    <= strb_write(reg_ddr_base_kv,    wr_data, wr_strb);
                    REG_DDR_BASE_OUT[7:0]:   reg_ddr_base_out   <= strb_write(reg_ddr_base_out,   wr_data, wr_strb);
                    REG_MODEL_HIDDEN[7:0]:   reg_model_hidden   <= strb_write(reg_model_hidden,   wr_data, wr_strb);
                    REG_MODEL_HEADS[7:0]:    reg_model_heads     <= strb_write(reg_model_heads,    wr_data, wr_strb);
                    REG_MODEL_HEAD_DIM[7:0]: reg_model_head_dim <= strb_write(reg_model_head_dim, wr_data, wr_strb);
                    REG_SEQ_LEN[7:0]:        reg_seq_len        <= strb_write(reg_seq_len,        wr_data, wr_strb);
                    REG_TOKEN_IDX[7:0]:      reg_token_idx      <= strb_write(reg_token_idx,      wr_data, wr_strb);
                    REG_DEBUG_CTRL[7:0]:     reg_debug_ctrl     <= strb_write(reg_debug_ctrl,     wr_data, wr_strb);
                    default: ;  // unmapped -- silently ignore
                endcase
            end
        end
    end

    // =======================================================================
    //  AXI4-Lite Read FSM
    // =======================================================================
    //  Accept AR, decode address, drive R data.
    // =======================================================================
    logic        rd_pending;
    logic [31:0] rd_addr_r;

    assign s_axil_arready = !rd_pending && !s_axil_rvalid;
    assign s_axil_rresp   = AXI_RESP_OKAY;

    // Assemble the read-only STATUS register from external inputs
    logic [31:0] status_reg;
    assign status_reg = {29'd0, error_i, busy_i, done_i};

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s_axil_rvalid <= 1'b0;
            s_axil_rdata  <= 32'd0;
            rd_pending    <= 1'b0;
            rd_addr_r     <= 32'd0;
        end else begin
            // Latch read address
            if (s_axil_arvalid && s_axil_arready) begin
                rd_pending <= 1'b1;
                rd_addr_r  <= s_axil_araddr;
            end

            // Drive read data one cycle after accepting AR
            if (rd_pending && !s_axil_rvalid) begin
                s_axil_rvalid <= 1'b1;
                rd_pending    <= 1'b0;

                case (rd_addr_r[7:0])
                    REG_CTRL[7:0]:           s_axil_rdata <= reg_ctrl;
                    REG_STATUS[7:0]:         s_axil_rdata <= status_reg;
                    REG_UCODE_BASE[7:0]:     s_axil_rdata <= reg_ucode_base;
                    REG_UCODE_LEN[7:0]:      s_axil_rdata <= reg_ucode_len;
                    REG_DDR_BASE_ACT[7:0]:   s_axil_rdata <= reg_ddr_base_act;
                    REG_DDR_BASE_WGT[7:0]:   s_axil_rdata <= reg_ddr_base_wgt;
                    REG_DDR_BASE_KV[7:0]:    s_axil_rdata <= reg_ddr_base_kv;
                    REG_DDR_BASE_OUT[7:0]:   s_axil_rdata <= reg_ddr_base_out;
                    REG_MODEL_HIDDEN[7:0]:   s_axil_rdata <= reg_model_hidden;
                    REG_MODEL_HEADS[7:0]:    s_axil_rdata <= reg_model_heads;
                    REG_MODEL_HEAD_DIM[7:0]: s_axil_rdata <= reg_model_head_dim;
                    REG_SEQ_LEN[7:0]:        s_axil_rdata <= reg_seq_len;
                    REG_TOKEN_IDX[7:0]:      s_axil_rdata <= reg_token_idx;
                    REG_DEBUG_CTRL[7:0]:     s_axil_rdata <= reg_debug_ctrl;
                    default:                 s_axil_rdata <= 32'hDEAD_BEEF;
                endcase
            end

            // De-assert RVALID when master accepts the data
            if (s_axil_rvalid && s_axil_rready)
                s_axil_rvalid <= 1'b0;
        end
    end

    // =======================================================================
    //  Output assignments
    // =======================================================================
    assign ctrl_o           = reg_ctrl;
    assign status_o         = status_reg;
    assign ucode_base_o     = reg_ucode_base;
    assign ucode_len_o      = reg_ucode_len;
    assign ddr_base_act_o   = reg_ddr_base_act;
    assign ddr_base_wgt_o   = reg_ddr_base_wgt;
    assign ddr_base_kv_o    = reg_ddr_base_kv;
    assign ddr_base_out_o   = reg_ddr_base_out;
    assign model_hidden_o   = reg_model_hidden;
    assign model_heads_o    = reg_model_heads;
    assign model_head_dim_o = reg_model_head_dim;
    assign seq_len_o        = reg_seq_len;
    assign token_idx_o      = reg_token_idx;
    assign debug_ctrl_o     = reg_debug_ctrl;

    // START is a single-cycle pulse derived from the register bit
    assign start_pulse_o = reg_ctrl[CTRL_START];

    // SOFT_RESET stays high while the counter is active
    assign soft_reset_o  = reg_ctrl[CTRL_SOFT_RESET];

endmodule : axi_lite_regs
