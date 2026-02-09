// =============================================================================
// ucode_decode.sv - Microcode Decode and Dispatch Unit
// =============================================================================
// Decodes 128-bit instruction words into structured fields and routes commands
// to the appropriate engine. Implements backpressure via the scoreboard and
// handles barrier/end instructions as special cases.
// =============================================================================

module ucode_decode
  import npu_pkg::*;
  import isa_pkg::*;
#(
    parameter int INSTR_W = 128
) (
    input  logic                clk,
    input  logic                rst_n,

    // --- From fetch stage ---
    input  logic                instr_valid,
    input  logic [INSTR_W-1:0] instr_data,
    output logic                instr_ready,

    // --- From scoreboard ---
    input  logic [5:0]          can_issue,      // one bit per engine
    input  logic                all_idle,        // all engines done (for barrier)

    // --- GEMM engine command ---
    output logic                gemm_cmd_valid,
    output logic [15:0]         gemm_cmd_src0,
    output logic [15:0]         gemm_cmd_src1,
    output logic [15:0]         gemm_cmd_dst,
    output logic [15:0]         gemm_cmd_m,
    output logic [15:0]         gemm_cmd_n,
    output logic [15:0]         gemm_cmd_k,
    output logic [7:0]          gemm_cmd_flags,

    // --- Softmax engine command ---
    output logic                softmax_cmd_valid,
    output logic [15:0]         softmax_cmd_src0,
    output logic [15:0]         softmax_cmd_dst,
    output logic [15:0]         softmax_cmd_len,
    output logic [7:0]          softmax_cmd_flags,

    // --- LayerNorm engine command ---
    output logic                layernorm_cmd_valid,
    output logic [15:0]         layernorm_cmd_src0,
    output logic [15:0]         layernorm_cmd_dst,
    output logic [15:0]         layernorm_cmd_len,
    output logic [7:0]          layernorm_cmd_flags,

    // --- GELU engine command ---
    output logic                gelu_cmd_valid,
    output logic [15:0]         gelu_cmd_src0,
    output logic [15:0]         gelu_cmd_dst,
    output logic [15:0]         gelu_cmd_len,
    output logic [7:0]          gelu_cmd_flags,

    // --- Vec engine command ---
    output logic                vec_cmd_valid,
    output logic [15:0]         vec_cmd_src0,
    output logic [15:0]         vec_cmd_src1,
    output logic [15:0]         vec_cmd_dst,
    output logic [15:0]         vec_cmd_len,
    output logic [7:0]          vec_cmd_flags,
    output logic [15:0]         vec_cmd_imm,

    // --- DMA read command ---
    output logic                dma_rd_cmd_valid,
    output logic [15:0]         dma_rd_cmd_src,
    output logic [15:0]         dma_rd_cmd_dst,
    output logic [15:0]         dma_rd_cmd_len,
    output logic [7:0]          dma_rd_cmd_flags,

    // --- DMA write command ---
    output logic                dma_wr_cmd_valid,
    output logic [15:0]         dma_wr_cmd_src,
    output logic [15:0]         dma_wr_cmd_dst,
    output logic [15:0]         dma_wr_cmd_len,
    output logic [7:0]          dma_wr_cmd_flags,

    // --- KV cache command ---
    output logic                kv_cmd_valid,
    output logic [7:0]          kv_cmd_opcode,
    output logic [15:0]         kv_cmd_src0,
    output logic [15:0]         kv_cmd_dst,
    output logic [15:0]         kv_cmd_len,
    output logic [7:0]          kv_cmd_flags,
    output logic [15:0]         kv_cmd_imm,

    // --- Barrier ---
    output logic                barrier_trigger,

    // --- To scoreboard ---
    output logic                issue_valid,
    output logic [2:0]          issue_engine_id,

    // --- Decoded instruction (for debug / downstream) ---
    output ucode_instr_t        decoded_instr,

    // --- Status ---
    output logic                program_end
);

    // -------------------------------------------------------------------------
    // Decode state
    // -------------------------------------------------------------------------
    typedef enum logic [1:0] {
        D_IDLE    = 2'd0,
        D_DECODE  = 2'd1,
        D_BARRIER = 2'd2
    } decode_state_e;

    decode_state_e state_q, state_d;

    // -------------------------------------------------------------------------
    // Instruction field extraction (combinational)
    // -------------------------------------------------------------------------
    logic [7:0]   dec_opcode;
    logic [7:0]   dec_flags;
    logic [15:0]  dec_dst_base;
    logic [15:0]  dec_src0_base;
    logic [15:0]  dec_src1_base;
    logic [15:0]  dec_m;
    logic [15:0]  dec_n;
    logic [15:0]  dec_k;
    logic [15:0]  dec_imm;

    always_comb begin
        dec_opcode    = instr_data[7:0];
        dec_flags     = instr_data[15:8];
        dec_dst_base  = instr_data[31:16];
        dec_src0_base = instr_data[47:32];
        dec_src1_base = instr_data[63:48];
        dec_m         = instr_data[79:64];
        dec_n         = instr_data[95:80];
        dec_k         = instr_data[111:96];
        dec_imm       = instr_data[127:112];
    end

    // -------------------------------------------------------------------------
    // Target engine selection (combinational)
    // -------------------------------------------------------------------------
    logic [2:0]  target_engine;
    logic        target_valid;       // opcode maps to a real engine
    logic        target_is_barrier;
    logic        target_is_end;
    logic        target_is_nop;
    logic        engine_free;        // selected engine can accept

    always_comb begin
        target_engine     = '0;
        target_valid      = 1'b0;
        target_is_barrier = 1'b0;
        target_is_end     = 1'b0;
        target_is_nop     = 1'b0;

        case (dec_opcode)
            OP_NOP: begin
                target_is_nop = 1'b1;
            end
            OP_DMA_LOAD: begin
                target_engine = ENG_DMA;
                target_valid  = 1'b1;
            end
            OP_DMA_STORE: begin
                target_engine = ENG_DMA;
                target_valid  = 1'b1;
            end
            OP_GEMM: begin
                target_engine = ENG_GEMM;
                target_valid  = 1'b1;
            end
            OP_VEC: begin
                target_engine = ENG_VEC;
                target_valid  = 1'b1;
            end
            OP_SOFTMAX: begin
                target_engine = ENG_SOFTMAX;
                target_valid  = 1'b1;
            end
            OP_LAYERNORM: begin
                target_engine = ENG_LAYERNORM;
                target_valid  = 1'b1;
            end
            OP_GELU: begin
                target_engine = ENG_GELU;
                target_valid  = 1'b1;
            end
            OP_KV_APPEND: begin
                target_engine = ENG_DMA;
                target_valid  = 1'b1;
            end
            OP_KV_READ: begin
                target_engine = ENG_DMA;
                target_valid  = 1'b1;
            end
            OP_BARRIER: begin
                target_is_barrier = 1'b1;
            end
            OP_END: begin
                target_is_end = 1'b1;
            end
            default: begin
                target_is_nop = 1'b1;
            end
        endcase

        // Check if the target engine is free
        engine_free = can_issue[target_engine];
    end

    // -------------------------------------------------------------------------
    // Registered output flops (one-cycle decode latency)
    // -------------------------------------------------------------------------
    logic                cmd_valid_q;
    logic [7:0]          cmd_opcode_q;
    logic [7:0]          cmd_flags_q;
    logic [15:0]         cmd_dst_q;
    logic [15:0]         cmd_src0_q;
    logic [15:0]         cmd_src1_q;
    logic [15:0]         cmd_m_q;
    logic [15:0]         cmd_n_q;
    logic [15:0]         cmd_k_q;
    logic [15:0]         cmd_imm_q;
    logic [2:0]          cmd_engine_q;
    logic                issue_valid_q;
    logic                barrier_trigger_q;
    logic                program_end_q;

    // Next values
    logic                cmd_valid_d;
    logic [7:0]          cmd_opcode_d;
    logic [7:0]          cmd_flags_d;
    logic [15:0]         cmd_dst_d;
    logic [15:0]         cmd_src0_d;
    logic [15:0]         cmd_src1_d;
    logic [15:0]         cmd_m_d;
    logic [15:0]         cmd_n_d;
    logic [15:0]         cmd_k_d;
    logic [15:0]         cmd_imm_d;
    logic [2:0]          cmd_engine_d;
    logic                issue_valid_d;
    logic                barrier_trigger_d;
    logic                program_end_d;

    // -------------------------------------------------------------------------
    // FSM + dispatch logic
    // -------------------------------------------------------------------------
    always_comb begin
        state_d           = state_q;
        instr_ready       = 1'b0;

        cmd_valid_d       = 1'b0;
        cmd_opcode_d      = cmd_opcode_q;
        cmd_flags_d       = cmd_flags_q;
        cmd_dst_d         = cmd_dst_q;
        cmd_src0_d        = cmd_src0_q;
        cmd_src1_d        = cmd_src1_q;
        cmd_m_d           = cmd_m_q;
        cmd_n_d           = cmd_n_q;
        cmd_k_d           = cmd_k_q;
        cmd_imm_d         = cmd_imm_q;
        cmd_engine_d      = cmd_engine_q;
        issue_valid_d     = 1'b0;
        barrier_trigger_d = 1'b0;
        program_end_d     = 1'b0;

        case (state_q)
            // -----------------------------------------------------------------
            D_IDLE: begin
                if (instr_valid) begin
                    if (target_is_end) begin
                        // OP_END: consume instruction, signal end
                        instr_ready   = 1'b1;
                        program_end_d = 1'b1;
                        state_d       = D_IDLE;
                    end else if (target_is_nop) begin
                        // OP_NOP: consume and discard
                        instr_ready = 1'b1;
                        state_d     = D_IDLE;
                    end else if (target_is_barrier) begin
                        // OP_BARRIER: enter barrier wait state
                        instr_ready       = 1'b1;
                        barrier_trigger_d = 1'b1;
                        state_d           = D_BARRIER;
                    end else if (target_valid && engine_free) begin
                        // Normal engine command: decode and dispatch
                        instr_ready   = 1'b1;
                        cmd_valid_d   = 1'b1;
                        cmd_opcode_d  = dec_opcode;
                        cmd_flags_d   = dec_flags;
                        cmd_dst_d     = dec_dst_base;
                        cmd_src0_d    = dec_src0_base;
                        cmd_src1_d    = dec_src1_base;
                        cmd_m_d       = dec_m;
                        cmd_n_d       = dec_n;
                        cmd_k_d       = dec_k;
                        cmd_imm_d     = dec_imm;
                        cmd_engine_d  = target_engine;
                        issue_valid_d = 1'b1;
                        state_d       = D_DECODE;
                    end
                    // else: engine busy, stall (instr_ready stays 0)
                end
            end

            // -----------------------------------------------------------------
            D_DECODE: begin
                // One-cycle decode latency: command outputs are registered,
                // go back to IDLE to accept next instruction
                state_d = D_IDLE;
            end

            // -----------------------------------------------------------------
            D_BARRIER: begin
                // Wait until all engines are idle
                if (all_idle) begin
                    state_d = D_IDLE;
                end
                // instr_ready stays 0: stall fetch during barrier
            end

            default: begin
                state_d = D_IDLE;
            end
        endcase
    end

    // -------------------------------------------------------------------------
    // Sequential logic
    // -------------------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state_q           <= D_IDLE;
            cmd_valid_q       <= 1'b0;
            cmd_opcode_q      <= '0;
            cmd_flags_q       <= '0;
            cmd_dst_q         <= '0;
            cmd_src0_q        <= '0;
            cmd_src1_q        <= '0;
            cmd_m_q           <= '0;
            cmd_n_q           <= '0;
            cmd_k_q           <= '0;
            cmd_imm_q         <= '0;
            cmd_engine_q      <= '0;
            issue_valid_q     <= 1'b0;
            barrier_trigger_q <= 1'b0;
            program_end_q     <= 1'b0;
        end else begin
            state_q           <= state_d;
            cmd_valid_q       <= cmd_valid_d;
            cmd_opcode_q      <= cmd_opcode_d;
            cmd_flags_q       <= cmd_flags_d;
            cmd_dst_q         <= cmd_dst_d;
            cmd_src0_q        <= cmd_src0_d;
            cmd_src1_q        <= cmd_src1_d;
            cmd_m_q           <= cmd_m_d;
            cmd_n_q           <= cmd_n_d;
            cmd_k_q           <= cmd_k_d;
            cmd_imm_q         <= cmd_imm_d;
            cmd_engine_q      <= cmd_engine_d;
            issue_valid_q     <= issue_valid_d;
            barrier_trigger_q <= barrier_trigger_d;
            program_end_q     <= program_end_d;
        end
    end

    // -------------------------------------------------------------------------
    // Output routing: demux registered command to appropriate engine
    // -------------------------------------------------------------------------

    // -- GEMM --
    assign gemm_cmd_valid = cmd_valid_q && (cmd_opcode_q == OP_GEMM);
    assign gemm_cmd_src0  = cmd_src0_q;
    assign gemm_cmd_src1  = cmd_src1_q;
    assign gemm_cmd_dst   = cmd_dst_q;
    assign gemm_cmd_m     = cmd_m_q;
    assign gemm_cmd_n     = cmd_n_q;
    assign gemm_cmd_k     = cmd_k_q;
    assign gemm_cmd_flags = cmd_flags_q;

    // -- Softmax --
    assign softmax_cmd_valid = cmd_valid_q && (cmd_opcode_q == OP_SOFTMAX);
    assign softmax_cmd_src0  = cmd_src0_q;
    assign softmax_cmd_dst   = cmd_dst_q;
    assign softmax_cmd_len   = cmd_n_q;    // N used as vector length
    assign softmax_cmd_flags = cmd_flags_q;

    // -- LayerNorm --
    assign layernorm_cmd_valid = cmd_valid_q && (cmd_opcode_q == OP_LAYERNORM);
    assign layernorm_cmd_src0  = cmd_src0_q;
    assign layernorm_cmd_dst   = cmd_dst_q;
    assign layernorm_cmd_len   = cmd_n_q;
    assign layernorm_cmd_flags = cmd_flags_q;

    // -- GELU --
    assign gelu_cmd_valid = cmd_valid_q && (cmd_opcode_q == OP_GELU);
    assign gelu_cmd_src0  = cmd_src0_q;
    assign gelu_cmd_dst   = cmd_dst_q;
    assign gelu_cmd_len   = cmd_n_q;
    assign gelu_cmd_flags = cmd_flags_q;

    // -- Vec --
    assign vec_cmd_valid = cmd_valid_q && (cmd_opcode_q == OP_VEC);
    assign vec_cmd_src0  = cmd_src0_q;
    assign vec_cmd_src1  = cmd_src1_q;
    assign vec_cmd_dst   = cmd_dst_q;
    assign vec_cmd_len   = cmd_n_q;
    assign vec_cmd_flags = cmd_flags_q;
    assign vec_cmd_imm   = cmd_imm_q;

    // -- DMA Read (LOAD) --
    assign dma_rd_cmd_valid = cmd_valid_q && (cmd_opcode_q == OP_DMA_LOAD);
    assign dma_rd_cmd_src   = cmd_src0_q;
    assign dma_rd_cmd_dst   = cmd_dst_q;
    assign dma_rd_cmd_len   = cmd_n_q;
    assign dma_rd_cmd_flags = cmd_flags_q;

    // -- DMA Write (STORE) --
    assign dma_wr_cmd_valid = cmd_valid_q && (cmd_opcode_q == OP_DMA_STORE);
    assign dma_wr_cmd_src   = cmd_src0_q;
    assign dma_wr_cmd_dst   = cmd_dst_q;
    assign dma_wr_cmd_len   = cmd_n_q;
    assign dma_wr_cmd_flags = cmd_flags_q;

    // -- KV Cache (APPEND and READ share the same port) --
    assign kv_cmd_valid  = cmd_valid_q && ((cmd_opcode_q == OP_KV_APPEND) ||
                                           (cmd_opcode_q == OP_KV_READ));
    assign kv_cmd_opcode = cmd_opcode_q;
    assign kv_cmd_src0   = cmd_src0_q;
    assign kv_cmd_dst    = cmd_dst_q;
    assign kv_cmd_len    = cmd_n_q;
    assign kv_cmd_flags  = cmd_flags_q;
    assign kv_cmd_imm    = cmd_imm_q;

    // -- Barrier --
    assign barrier_trigger = barrier_trigger_q;

    // -- Scoreboard issue --
    assign issue_valid     = issue_valid_q;
    assign issue_engine_id = cmd_engine_q;

    // -- Program end --
    assign program_end = program_end_q;

    // -- Decoded instruction struct (for debug / downstream consumers) --
    always_comb begin
        decoded_instr.opcode    = cmd_opcode_q;
        decoded_instr.flags     = cmd_flags_q;
        decoded_instr.dst_base  = cmd_dst_q;
        decoded_instr.src0_base = cmd_src0_q;
        decoded_instr.src1_base = cmd_src1_q;
        decoded_instr.M         = cmd_m_q;
        decoded_instr.N         = cmd_n_q;
        decoded_instr.K         = cmd_k_q;
        decoded_instr.imm       = cmd_imm_q;
    end

endmodule
