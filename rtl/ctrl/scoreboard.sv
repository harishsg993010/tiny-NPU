// =============================================================================
// scoreboard.sv - Engine Dependency Tracker
// =============================================================================
// Tracks busy/idle state of each compute engine. Prevents the decode stage
// from issuing a new command to an engine that is still executing. Provides
// an all_idle signal for barrier synchronisation.
// =============================================================================

module scoreboard
  import npu_pkg::*;
  import isa_pkg::*;
#(
    parameter int NUM_ENGINES = 6
) (
    input  logic                      clk,
    input  logic                      rst_n,

    // --- From decode (issue notification) ---
    input  logic                      issue_valid,
    input  logic [2:0]                issue_engine_id,

    // --- From engines (completion notification) ---
    input  logic [NUM_ENGINES-1:0]    engine_done,

    // --- To decode ---
    output logic [NUM_ENGINES-1:0]    engine_busy,
    output logic [NUM_ENGINES-1:0]    can_issue,
    output logic                      all_idle
);

    // -------------------------------------------------------------------------
    // Busy-bit register per engine
    // -------------------------------------------------------------------------
    logic [NUM_ENGINES-1:0] busy_q, busy_d;

    // -------------------------------------------------------------------------
    // One-hot decode of the issued engine ID
    // -------------------------------------------------------------------------
    logic [NUM_ENGINES-1:0] issue_onehot;

    always_comb begin
        issue_onehot = '0;
        if (issue_valid && (issue_engine_id < NUM_ENGINES[2:0])) begin
            issue_onehot[issue_engine_id] = 1'b1;
        end
    end

    // -------------------------------------------------------------------------
    // Next-state logic for busy bits
    // -------------------------------------------------------------------------
    // Priority: engine_done clears the busy bit; issue_valid sets it.
    // If both happen on the same engine in the same cycle, done takes priority
    // (the bit is cleared this cycle, and the new issue will set it next cycle
    // via the normal issue path -- but since we are in the same cycle, we
    // clear first, then set, giving net "busy" which is the safe behaviour).
    //
    // Actually, the safest approach: done clears, issue sets. If simultaneous,
    // the done takes priority (clear wins) and the new command will be issued
    // on the next cycle because can_issue will momentarily show free.
    // However, that creates a race. The correct design:
    //   - done clears the bit
    //   - issue sets the bit
    //   - simultaneous on same engine: done wins (clear), decode will see
    //     can_issue=1 on the *next* cycle and re-issue then.
    //
    // But the decode already checks can_issue before asserting issue_valid,
    // so simultaneous done+issue on the same engine means the decode saw
    // busy=1 last cycle (and could not have issued). Therefore this case
    // should not happen in practice. We handle it defensively: done wins.
    // -------------------------------------------------------------------------

    always_comb begin
        busy_d = busy_q;

        // Clear bits for engines that are done
        busy_d = busy_d & ~engine_done;

        // Set bits for newly issued engines
        busy_d = busy_d | issue_onehot;

        // Defensive: if same engine has both done and issue in same cycle,
        // done takes priority (clear the bit). The issue will effectively
        // be ignored this cycle; decode should not create this situation.
        for (int i = 0; i < NUM_ENGINES; i++) begin
            if (engine_done[i] && issue_onehot[i]) begin
                busy_d[i] = 1'b0;
            end
        end
    end

    // -------------------------------------------------------------------------
    // Sequential logic
    // -------------------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            busy_q <= '0;
        end else begin
            busy_q <= busy_d;
        end
    end

    // -------------------------------------------------------------------------
    // Output assignments
    // -------------------------------------------------------------------------
    assign engine_busy = busy_q;
    assign can_issue   = ~busy_q;
    assign all_idle    = (busy_q == '0);

endmodule
