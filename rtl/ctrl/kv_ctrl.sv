// =============================================================================
// KV Cache Controller - Bridges 8-bit SRAM0 to 128-bit kv_cache_bank
// Replaces the C++ software shim with a hardware FSM.
// =============================================================================
`default_nettype none

module kv_ctrl
    import isa_pkg::*;
#(
    parameter int MAX_LAYERS = 4,
    parameter int MAX_HEADS  = 4,
    parameter int MAX_SEQ    = 512,
    parameter int HEAD_DIM   = 16,
    parameter int DW         = 8
)(
    input  wire                          clk,
    input  wire                          rst_n,

    // Command interface (from decoder)
    input  wire                          cmd_valid,
    input  wire  [7:0]                   cmd_opcode,
    input  wire  [15:0]                  cmd_src0,
    input  wire  [15:0]                  cmd_dst,
    input  wire  [15:0]                  cmd_m,       // layer_id
    input  wire  [15:0]                  cmd_n,       // vec_len
    input  wire  [15:0]                  cmd_k,       // time_index (append) or time_len (read)
    input  wire  [7:0]                   cmd_flags,
    input  wire  [15:0]                  cmd_imm,     // head_id

    // SRAM0 read port (port A)
    output logic                         sram_rd_en,
    output logic [15:0]                  sram_rd_addr,
    input  wire  [7:0]                   sram_rd_data,

    // SRAM0 write port (port B)
    output logic                         sram_wr_en,
    output logic [15:0]                  sram_wr_addr,
    output logic [7:0]                   sram_wr_data,

    // KV cache bank - append interface
    output logic                         append_valid,
    input  wire                          append_ready,
    output logic [$clog2(MAX_LAYERS)-1:0] append_layer,
    output logic [$clog2(MAX_HEADS)-1:0]  append_head,
    output logic [$clog2(MAX_SEQ)-1:0]    append_time,
    output logic                         append_is_v,
    output logic [HEAD_DIM*DW-1:0]       append_data,
    input  wire                          append_done,

    // KV cache bank - read interface
    output logic                         read_req_valid,
    input  wire                          read_req_ready,
    output logic [$clog2(MAX_LAYERS)-1:0] read_layer,
    output logic [$clog2(MAX_HEADS)-1:0]  read_head,
    output logic [$clog2(MAX_SEQ)-1:0]    read_time_start,
    output logic [$clog2(MAX_SEQ)-1:0]    read_time_len,
    output logic                         read_is_v,
    input  wire                          read_data_valid,
    input  wire  [HEAD_DIM*DW-1:0]       read_data,
    input  wire                          read_data_last,

    // Status
    output logic                         busy,
    output logic                         done
);

    localparam int VEC_W = HEAD_DIM * DW;

    typedef enum logic [2:0] {
        KV_IDLE,
        KV_APPEND_RD,
        KV_APPEND_WR,
        KV_READ_REQ,
        KV_READ_WAIT,
        KV_READ_WR,
        KV_READ_NEXT,
        KV_DONE
    } kv_state_t;

    kv_state_t state, state_next;

    // Latched command fields
    logic [7:0]  op_r;
    logic [15:0] src_r, dst_r, m_r, n_r, k_r, imm_r;
    logic [7:0]  flags_r;

    // Byte counter for SRAM0 reads/writes
    logic [15:0] byte_cnt;

    // 128-bit vector register for packing/unpacking
    logic [VEC_W-1:0] vec_reg;

    // Time counter for multi-vector reads
    logic [15:0] time_cnt;

    // SRAM0 read pipeline: data arrives 1 cycle after addr
    logic        rd_pipe_valid;

    // State register
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            state <= KV_IDLE;
        else
            state <= state_next;
    end

    // Latch command on cmd_valid
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            op_r    <= '0;
            src_r   <= '0;
            dst_r   <= '0;
            m_r     <= '0;
            n_r     <= '0;
            k_r     <= '0;
            flags_r <= '0;
            imm_r   <= '0;
        end else if (cmd_valid && state == KV_IDLE) begin
            op_r    <= cmd_opcode;
            src_r   <= cmd_src0;
            dst_r   <= cmd_dst;
            m_r     <= cmd_m;
            n_r     <= cmd_n;
            k_r     <= cmd_k;
            flags_r <= cmd_flags;
            imm_r   <= cmd_imm;
        end
    end

    // Byte counter
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            byte_cnt <= '0;
        end else begin
            case (state_next)
                KV_APPEND_RD: begin
                    if (state == KV_IDLE || state == KV_APPEND_RD)
                        byte_cnt <= (state == KV_IDLE) ? 16'd0 : byte_cnt + 16'd1;
                end
                KV_READ_WR: begin
                    if (state == KV_READ_WAIT)
                        byte_cnt <= 16'd0;
                    else
                        byte_cnt <= byte_cnt + 16'd1;
                end
                default: byte_cnt <= '0;
            endcase
        end
    end

    // Pipeline to match SRAM0 1-cycle read latency.
    // We issue addr in cycle N, SRAM registers dout at posedge N+1,
    // and we capture sram_rd_data in our always_ff at posedge N+1.
    logic [15:0] pack_idx;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rd_pipe_valid <= 1'b0;
            pack_idx      <= '0;
        end else begin
            rd_pipe_valid <= sram_rd_en && (state == KV_APPEND_RD);
            if (sram_rd_en && (state == KV_APPEND_RD))
                pack_idx <= byte_cnt;
        end
    end

    // Pack bytes into vec_reg during APPEND read phase.
    // rd_pipe_valid is high 1 cycle after the read was issued, which is
    // when sram_rd_data holds the result.
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            vec_reg <= '0;
        end else if (rd_pipe_valid) begin
            vec_reg[pack_idx*8 +: 8] <= sram_rd_data;
        end else if (state == KV_READ_WAIT && read_data_valid) begin
            vec_reg <= read_data;
        end
    end

    // Time counter for multi-vector reads
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            time_cnt <= '0;
        end else if (state == KV_IDLE && cmd_valid) begin
            time_cnt <= '0;
        end else if (state == KV_READ_NEXT) begin
            time_cnt <= time_cnt + 16'd1;
        end
    end

    // Next state logic
    always_comb begin
        state_next = state;

        case (state)
            KV_IDLE: begin
                if (cmd_valid) begin
                    if (cmd_opcode == OP_KV_APPEND)
                        state_next = KV_APPEND_RD;
                    else if (cmd_opcode == OP_KV_READ) begin
                        if (cmd_k == 16'd0)
                            state_next = KV_DONE;  // time_len=0 means no vectors
                        else
                            state_next = KV_READ_REQ;
                    end
                end
            end

            KV_APPEND_RD: begin
                // Issue SRAM0 reads for vec_len bytes
                // byte_cnt counts 0..n_r-1 for issuing addresses
                // Need one extra cycle for last byte's read latency
                if (byte_cnt >= n_r) begin
                    // All addresses issued, wait for last byte
                    state_next = KV_APPEND_WR;
                end
            end

            KV_APPEND_WR: begin
                if (append_done)
                    state_next = KV_DONE;
            end

            KV_READ_REQ: begin
                if (read_req_ready)
                    state_next = KV_READ_WAIT;
            end

            KV_READ_WAIT: begin
                if (read_data_valid)
                    state_next = KV_READ_WR;
            end

            KV_READ_WR: begin
                if (byte_cnt >= n_r - 16'd1)
                    state_next = KV_READ_NEXT;
            end

            KV_READ_NEXT: begin
                if (time_cnt + 16'd1 >= k_r)
                    state_next = KV_DONE;
                else
                    state_next = KV_READ_REQ;
            end

            KV_DONE: begin
                state_next = KV_IDLE;
            end

            default: state_next = KV_IDLE;
        endcase
    end

    // Output logic
    always_comb begin
        sram_rd_en   = 1'b0;
        sram_rd_addr = '0;
        sram_wr_en   = 1'b0;
        sram_wr_addr = '0;
        sram_wr_data = '0;
        append_valid = 1'b0;
        read_req_valid = 1'b0;
        done         = 1'b0;

        case (state)
            KV_APPEND_RD: begin
                // Issue SRAM0 read for byte[byte_cnt]
                if (byte_cnt < n_r) begin
                    sram_rd_en   = 1'b1;
                    sram_rd_addr = src_r + byte_cnt;
                end
            end

            KV_APPEND_WR: begin
                append_valid = 1'b1;
            end

            KV_READ_REQ: begin
                read_req_valid = 1'b1;
            end

            KV_READ_WR: begin
                sram_wr_en   = 1'b1;
                sram_wr_addr = dst_r + time_cnt * n_r + byte_cnt;
                sram_wr_data = vec_reg[byte_cnt*8 +: 8];
            end

            KV_DONE: begin
                done = 1'b1;
            end

            default: ;
        endcase
    end

    // Append port wiring
    assign append_layer = m_r[$clog2(MAX_LAYERS)-1:0];
    assign append_head  = imm_r[$clog2(MAX_HEADS)-1:0];
    assign append_time  = k_r[$clog2(MAX_SEQ)-1:0];
    assign append_is_v  = flags_r[0];
    assign append_data  = vec_reg;

    // Read port wiring - single-vector reads (time_len=0 to bank)
    assign read_layer      = m_r[$clog2(MAX_LAYERS)-1:0];
    assign read_head       = imm_r[$clog2(MAX_HEADS)-1:0];
    assign read_time_start = time_cnt[$clog2(MAX_SEQ)-1:0];
    assign read_time_len   = {$clog2(MAX_SEQ){1'b0}};  // single vector
    assign read_is_v       = flags_r[0];

    // Status
    assign busy = (state != KV_IDLE);

endmodule

`default_nettype wire
