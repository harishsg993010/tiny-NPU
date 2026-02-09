// =============================================================================
// barrier.sv - Barrier Synchronization Unit
// =============================================================================
// Implements barrier synchronization for the microcode engine. When triggered
// by an OP_BARRIER instruction, it stalls the decode pipeline until all
// in-flight engine operations have completed (all_idle asserted by the
// scoreboard). This ensures data consistency between phases of computation.
// =============================================================================

module barrier
  import npu_pkg::*;
  import isa_pkg::*;
(
    input  logic clk,
    input  logic rst_n,

    // --- From decode ---
    input  logic trigger,   // pulse: OP_BARRIER decoded

    // --- From scoreboard ---
    input  logic all_idle,  // all engines have finished

    // --- To decode / fetch ---
    output logic stall,     // prevents further instruction issue
    output logic done       // pulse: barrier satisfied
);

    // -------------------------------------------------------------------------
    // FSM state encoding
    // -------------------------------------------------------------------------
    typedef enum logic [1:0] {
        B_IDLE = 2'd0,
        B_WAIT = 2'd1,
        B_DONE = 2'd2
    } barrier_state_e;

    barrier_state_e state_q, state_d;

    // -------------------------------------------------------------------------
    // Registered outputs
    // -------------------------------------------------------------------------
    logic stall_q, stall_d;
    logic done_q,  done_d;

    // -------------------------------------------------------------------------
    // FSM next-state logic
    // -------------------------------------------------------------------------
    always_comb begin
        state_d = state_q;
        stall_d = stall_q;
        done_d  = 1'b0;

        case (state_q)
            // -----------------------------------------------------------------
            // IDLE: Wait for a barrier trigger pulse
            // -----------------------------------------------------------------
            B_IDLE: begin
                stall_d = 1'b0;
                if (trigger) begin
                    // Check if all engines are already idle
                    if (all_idle) begin
                        // Nothing in flight: barrier is immediately satisfied
                        done_d  = 1'b1;
                        state_d = B_DONE;
                    end else begin
                        // Engines still running: enter wait state
                        stall_d = 1'b1;
                        state_d = B_WAIT;
                    end
                end
            end

            // -----------------------------------------------------------------
            // WAIT: Assert stall, wait for all engines to complete
            // -----------------------------------------------------------------
            B_WAIT: begin
                stall_d = 1'b1;
                if (all_idle) begin
                    stall_d = 1'b0;
                    done_d  = 1'b1;
                    state_d = B_DONE;
                end
            end

            // -----------------------------------------------------------------
            // DONE: Deassert stall, pulse done, return to IDLE
            // -----------------------------------------------------------------
            B_DONE: begin
                stall_d = 1'b0;
                state_d = B_IDLE;
            end

            default: begin
                state_d = B_IDLE;
                stall_d = 1'b0;
            end
        endcase
    end

    // -------------------------------------------------------------------------
    // Sequential logic
    // -------------------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state_q <= B_IDLE;
            stall_q <= 1'b0;
            done_q  <= 1'b0;
        end else begin
            state_q <= state_d;
            stall_q <= stall_d;
            done_q  <= done_d;
        end
    end

    // -------------------------------------------------------------------------
    // Output assignments
    // -------------------------------------------------------------------------
    assign stall = stall_q;
    assign done  = done_q;

endmodule
