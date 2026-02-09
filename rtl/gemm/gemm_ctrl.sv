// =============================================================================
// GEMM Controller - Tiled data flow for systolic array
// Loads A and B matrices from SRAM into internal buffers, then streams
// staggered data into the systolic array and stores requantized results.
// Supports M/N/K tiling for dimensions exceeding the 16x16 systolic array.
// K-tiling uses an external ACC SRAM to accumulate int32 partial sums
// across K-tiles before final requantization.
// =============================================================================
`default_nettype none

module gemm_ctrl
    import npu_pkg::*;
    import isa_pkg::*;
#(
    parameter int ARRAY_M     = 16,
    parameter int ARRAY_N     = 16,
    parameter int DATA_W      = 8,
    parameter int ACC_W       = 32,
    parameter int SRAM_ADDR_W = 16
)(
    input  wire                          clk,
    input  wire                          rst_n,

    // Command interface
    input  wire                          cmd_valid,
    input  wire  [15:0]                  cmd_src0,
    input  wire  [15:0]                  cmd_src1,
    input  wire  [15:0]                  cmd_dst,
    input  wire  [15:0]                  cmd_M,
    input  wire  [15:0]                  cmd_N,
    input  wire  [15:0]                  cmd_K,
    input  wire  [7:0]                   cmd_flags,
    input  wire  [15:0]                  cmd_imm,

    // SRAM read port (one byte per cycle, 1-cycle read latency)
    output logic                         sram_rd_en,
    output logic [SRAM_ADDR_W-1:0]       sram_rd_addr,
    input  wire  [DATA_W-1:0]            sram_rd_data,

    // SRAM write port (one byte per cycle, no latency)
    output logic                         sram_wr_en,
    output logic [SRAM_ADDR_W-1:0]       sram_wr_addr,
    output logic [DATA_W-1:0]            sram_wr_data,

    // ACC SRAM interface (for K-tiling accumulation)
    output logic                         acc_rd_en,
    output logic [7:0]                   acc_rd_addr,
    input  wire  signed [ACC_W-1:0]      acc_rd_data,
    output logic                         acc_wr_en,
    output logic [7:0]                   acc_wr_addr,
    output logic signed [ACC_W-1:0]      acc_wr_data,

    // Systolic array interface
    output logic                         sa_clear,
    output logic                         sa_en,
    output logic signed [DATA_W-1:0]     sa_a_col [ARRAY_M],
    output logic signed [DATA_W-1:0]     sa_b_row [ARRAY_N],
    input  wire  signed [ACC_W-1:0]      sa_acc   [ARRAY_M][ARRAY_N],

    // Status
    output logic                         busy,
    output logic                         done
);

    // =========================================================================
    // State machine
    // =========================================================================
    typedef enum logic [3:0] {
        ST_IDLE,
        ST_SETUP,
        ST_TILE_SETUP,
        ST_LOAD_A,
        ST_LOAD_B,
        ST_CLEAR,
        ST_STREAM,
        ST_STORE,
        ST_TILE_NEXT,
        ST_DONE
    } state_t;

    state_t state, state_next;

    // =========================================================================
    // Registered command fields
    // =========================================================================
    logic [15:0] r_src0, r_src1, r_dst;
    logic [15:0] r_M, r_N, r_K;
    logic        r_transpose_b;
    logic        r_do_requant;
    logic [7:0]  r_scale, r_shift;

    // Effective dimensions for current tile (clamped to array size)
    logic [4:0]  m_eff, n_eff, k_eff;

    // =========================================================================
    // Tile counters
    // =========================================================================
    logic [3:0] m_tile_r, n_tile_r, k_tile_r;     // current tile indices
    logic [3:0] m_tiles_r, n_tiles_r, k_tiles_r;  // total tiles per dimension

    // Tile base addresses (precomputed in ST_TILE_SETUP)
    logic [15:0] a_tile_base, b_tile_base, c_tile_base;

    // Store phase for 2-phase ACC read-modify-write
    logic store_phase;

    // =========================================================================
    // Internal buffers: A[ARRAY_M][ARRAY_N], B[ARRAY_N][ARRAY_N]
    // =========================================================================
    logic signed [DATA_W-1:0] buf_a [ARRAY_M][ARRAY_N];
    logic signed [DATA_W-1:0] buf_b [ARRAY_N][ARRAY_N];

    // =========================================================================
    // Counters
    // =========================================================================
    logic [4:0]  load_row, load_col;  // row/col for LOAD_A and LOAD_B
    logic        load_phase;          // 0=issue read, 1=capture data
    logic [5:0]  stream_cnt;          // streaming cycle counter
    logic [5:0]  stream_total;        // k_eff + m_eff + n_eff - 1
    logic [4:0]  store_row, store_col;

    // =========================================================================
    // State register
    // =========================================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            state <= ST_IDLE;
        else
            state <= state_next;
    end

    // =========================================================================
    // Helper: does this GEMM need multi-K-tile accumulation?
    // =========================================================================
    wire multi_k = (k_tiles_r > 4'd1);
    wire last_k_tile = (k_tile_r == k_tiles_r - 4'd1);
    wire first_k_tile = (k_tile_r == 4'd0);

    // =========================================================================
    // Next-state logic
    // =========================================================================
    always_comb begin
        state_next = state;

        case (state)
            ST_IDLE: begin
                if (cmd_valid)
                    state_next = ST_SETUP;
            end

            ST_SETUP: begin
                state_next = ST_TILE_SETUP;
            end

            ST_TILE_SETUP: begin
                state_next = ST_LOAD_A;
            end

            ST_LOAD_A: begin
                if (load_phase && load_row == m_eff - 5'd1 && load_col == k_eff - 5'd1)
                    state_next = ST_LOAD_B;
            end

            ST_LOAD_B: begin
                if (load_phase && load_row == k_eff - 5'd1 && load_col == n_eff - 5'd1)
                    state_next = ST_CLEAR;
            end

            ST_CLEAR: begin
                state_next = ST_STREAM;
            end

            ST_STREAM: begin
                if (stream_cnt == stream_total)
                    state_next = ST_STORE;
            end

            ST_STORE: begin
                // Determine when store is complete based on K-tiling mode
                if (!multi_k) begin
                    // Single K-tile: 1 cycle per element
                    if (store_row == m_eff - 5'd1 && store_col == n_eff - 5'd1)
                        state_next = ST_TILE_NEXT;
                end else begin
                    // Multi-K-tile: 2 cycles per element (phase 0: read ACC, phase 1: write)
                    if (first_k_tile) begin
                        // First K-tile: just write to ACC SRAM, 1 cycle per element
                        if (store_row == m_eff - 5'd1 && store_col == n_eff - 5'd1)
                            state_next = ST_TILE_NEXT;
                    end else begin
                        // Middle/last K-tiles: 2 phases (read ACC, then accumulate+write)
                        if (store_phase && store_row == m_eff - 5'd1 && store_col == n_eff - 5'd1)
                            state_next = ST_TILE_NEXT;
                    end
                end
            end

            ST_TILE_NEXT: begin
                // Advance tile counters: inner=k, middle=n, outer=m
                if (k_tile_r < k_tiles_r - 4'd1)
                    state_next = ST_TILE_SETUP;
                else if (n_tile_r < n_tiles_r - 4'd1)
                    state_next = ST_TILE_SETUP;
                else if (m_tile_r < m_tiles_r - 4'd1)
                    state_next = ST_TILE_SETUP;
                else
                    state_next = ST_DONE;
            end

            ST_DONE: begin
                state_next = ST_IDLE;
            end

            default: state_next = ST_IDLE;
        endcase
    end

    // =========================================================================
    // ACC SRAM address for current store element
    // =========================================================================
    logic [7:0] acc_elem_addr;
    assign acc_elem_addr = {3'b0, store_row} * {3'b0, n_eff} + {3'b0, store_col};

    // =========================================================================
    // Output logic (active-low reset defaults)
    // =========================================================================
    always_comb begin
        // Defaults
        sram_rd_en   = 1'b0;
        sram_rd_addr = '0;
        sram_wr_en   = 1'b0;
        sram_wr_addr = '0;
        sram_wr_data = '0;
        sa_clear     = 1'b0;
        sa_en        = 1'b0;
        busy         = (state != ST_IDLE);
        done         = 1'b0;

        acc_rd_en    = 1'b0;
        acc_rd_addr  = '0;
        acc_wr_en    = 1'b0;
        acc_wr_addr  = '0;
        acc_wr_data  = '0;

        for (int i = 0; i < ARRAY_M; i++) sa_a_col[i] = '0;
        for (int j = 0; j < ARRAY_N; j++) sa_b_row[j] = '0;

        case (state)
            ST_IDLE: begin
                busy = 1'b0;
            end

            ST_LOAD_A: begin
                if (!load_phase) begin
                    sram_rd_en   = 1'b1;
                    sram_rd_addr = a_tile_base + {11'b0, load_row} * r_K + {11'b0, load_col};
                end
            end

            ST_LOAD_B: begin
                if (!load_phase) begin
                    sram_rd_en = 1'b1;
                    if (r_transpose_b)
                        sram_rd_addr = b_tile_base + {11'b0, load_col} * r_K + {11'b0, load_row};
                    else
                        sram_rd_addr = b_tile_base + {11'b0, load_row} * r_N + {11'b0, load_col};
                end
            end

            ST_CLEAR: begin
                sa_clear = 1'b1;
            end

            ST_STREAM: begin
                sa_en = 1'b1;
                for (int i = 0; i < ARRAY_M; i++) begin
                    automatic int idx_a = int'(stream_cnt) - i;
                    if (idx_a >= 0 && idx_a < int'(k_eff) && i < int'(m_eff))
                        sa_a_col[i] = buf_a[i][idx_a[3:0]];
                    else
                        sa_a_col[i] = '0;
                end
                for (int j = 0; j < ARRAY_N; j++) begin
                    automatic int idx_b = int'(stream_cnt) - j;
                    if (idx_b >= 0 && idx_b < int'(k_eff) && j < int'(n_eff))
                        sa_b_row[j] = buf_b[idx_b[3:0]][j];
                    else
                        sa_b_row[j] = '0;
                end
            end

            ST_STORE: begin
                if (!multi_k) begin
                    // === Single K-tile: requantize directly to output SRAM ===
                    sram_wr_en   = 1'b1;
                    sram_wr_addr = c_tile_base + {11'b0, store_row} * r_N + {11'b0, store_col};
                    if (r_do_requant)
                        sram_wr_data = requantize(sa_acc[store_row[3:0]][store_col[3:0]],
                                                  r_scale, r_shift);
                    else
                        sram_wr_data = saturate_i8(sa_acc[store_row[3:0]][store_col[3:0]]);
                end else if (first_k_tile) begin
                    // === First K-tile of multi-K: write partial sum to ACC SRAM ===
                    acc_wr_en   = 1'b1;
                    acc_wr_addr = acc_elem_addr;
                    acc_wr_data = sa_acc[store_row[3:0]][store_col[3:0]];
                end else begin
                    // === Middle or last K-tile: read-modify-write ACC SRAM ===
                    if (!store_phase) begin
                        // Phase 0: issue ACC SRAM read
                        acc_rd_en   = 1'b1;
                        acc_rd_addr = acc_elem_addr;
                    end else begin
                        // Phase 1: accumulate and write back
                        if (last_k_tile) begin
                            // Last K-tile: accumulate, requantize, write to output SRAM
                            sram_wr_en   = 1'b1;
                            sram_wr_addr = c_tile_base + {11'b0, store_row} * r_N + {11'b0, store_col};
                            if (r_do_requant)
                                sram_wr_data = requantize(
                                    acc_rd_data + sa_acc[store_row[3:0]][store_col[3:0]],
                                    r_scale, r_shift);
                            else
                                sram_wr_data = saturate_i8(
                                    acc_rd_data + sa_acc[store_row[3:0]][store_col[3:0]]);
                        end else begin
                            // Middle K-tile: accumulate and write back to ACC SRAM
                            acc_wr_en   = 1'b1;
                            acc_wr_addr = acc_elem_addr;
                            acc_wr_data = acc_rd_data + sa_acc[store_row[3:0]][store_col[3:0]];
                        end
                    end
                end
            end

            ST_DONE: begin
                done = 1'b1;
            end

            default: ;
        endcase
    end

    // =========================================================================
    // Sequential: command capture, tile setup, counters, buffer loading
    // =========================================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            r_src0        <= '0;
            r_src1        <= '0;
            r_dst         <= '0;
            r_M           <= '0;
            r_N           <= '0;
            r_K           <= '0;
            r_transpose_b <= 1'b0;
            r_do_requant  <= 1'b0;
            r_scale       <= '0;
            r_shift       <= '0;
            m_eff         <= '0;
            n_eff         <= '0;
            k_eff         <= '0;
            m_tile_r      <= '0;
            n_tile_r      <= '0;
            k_tile_r      <= '0;
            m_tiles_r     <= '0;
            n_tiles_r     <= '0;
            k_tiles_r     <= '0;
            a_tile_base   <= '0;
            b_tile_base   <= '0;
            c_tile_base   <= '0;
            store_phase   <= 1'b0;
            load_row      <= '0;
            load_col      <= '0;
            load_phase    <= 1'b0;
            stream_cnt    <= '0;
            stream_total  <= '0;
            store_row     <= '0;
            store_col     <= '0;
            for (int i = 0; i < ARRAY_M; i++)
                for (int j = 0; j < ARRAY_N; j++)
                    buf_a[i][j] <= '0;
            for (int i = 0; i < ARRAY_N; i++)
                for (int j = 0; j < ARRAY_N; j++)
                    buf_b[i][j] <= '0;
        end else begin
            case (state)
                ST_IDLE: begin
                    if (cmd_valid) begin
                        r_src0        <= cmd_src0;
                        r_src1        <= cmd_src1;
                        r_dst         <= cmd_dst;
                        r_M           <= cmd_M;
                        r_N           <= cmd_N;
                        r_K           <= cmd_K;
                        r_transpose_b <= cmd_flags[FLAG_TRANSPOSE_B];
                        r_do_requant  <= cmd_flags[FLAG_REQUANT];
                        r_scale       <= cmd_imm[7:0];
                        r_shift       <= cmd_imm[15:8];
                    end
                end

                ST_SETUP: begin
                    // Compute tile counts
                    m_tiles_r <= 4'((r_M + 16'd15) >> 4);
                    n_tiles_r <= 4'((r_N + 16'd15) >> 4);
                    k_tiles_r <= 4'((r_K + 16'd15) >> 4);
                    // Reset tile counters
                    m_tile_r  <= '0;
                    n_tile_r  <= '0;
                    k_tile_r  <= '0;
                end

                ST_TILE_SETUP: begin
                    // Compute per-tile effective dimensions
                    begin
                        automatic logic [15:0] m_remain = r_M - {12'b0, m_tile_r} * 16'd16;
                        automatic logic [15:0] n_remain = r_N - {12'b0, n_tile_r} * 16'd16;
                        automatic logic [15:0] k_remain = r_K - {12'b0, k_tile_r} * 16'd16;
                        m_eff <= (m_remain < 16) ? m_remain[4:0] : 5'd16;
                        n_eff <= (n_remain < 16) ? n_remain[4:0] : 5'd16;
                        k_eff <= (k_remain < 16) ? k_remain[4:0] : 5'd16;
                    end

                    // Compute tile base addresses
                    // a_tile_base = src0 + m_tile*16*K + k_tile*16
                    a_tile_base <= r_src0 + {8'b0, m_tile_r, 4'b0} * r_K + {8'b0, k_tile_r, 4'b0};

                    if (r_transpose_b)
                        // B stored as [N][K]: b_tile_base = src1 + n_tile*16*K + k_tile*16
                        b_tile_base <= r_src1 + {8'b0, n_tile_r, 4'b0} * r_K + {8'b0, k_tile_r, 4'b0};
                    else
                        // B stored as [K][N]: b_tile_base = src1 + k_tile*16*N + n_tile*16
                        b_tile_base <= r_src1 + {8'b0, k_tile_r, 4'b0} * r_N + {8'b0, n_tile_r, 4'b0};

                    // c_tile_base = dst + m_tile*16*N + n_tile*16
                    c_tile_base <= r_dst + {8'b0, m_tile_r, 4'b0} * r_N + {8'b0, n_tile_r, 4'b0};

                    // Reset load/store counters
                    load_row    <= '0;
                    load_col    <= '0;
                    load_phase  <= 1'b0;
                    stream_cnt  <= '0;
                    store_row   <= '0;
                    store_col   <= '0;
                    store_phase <= 1'b0;
                end

                ST_LOAD_A: begin
                    if (!load_phase) begin
                        load_phase <= 1'b1;
                    end else begin
                        buf_a[load_row][load_col] <= signed'(sram_rd_data);
                        load_phase <= 1'b0;
                        if (load_col == k_eff - 5'd1) begin
                            load_col <= '0;
                            if (load_row == m_eff - 5'd1) begin
                                load_row   <= '0;
                                load_col   <= '0;
                            end else begin
                                load_row <= load_row + 5'd1;
                            end
                        end else begin
                            load_col <= load_col + 5'd1;
                        end
                    end
                end

                ST_LOAD_B: begin
                    if (!load_phase) begin
                        load_phase <= 1'b1;
                    end else begin
                        buf_b[load_row][load_col] <= signed'(sram_rd_data);
                        load_phase <= 1'b0;
                        if (load_col == n_eff - 5'd1) begin
                            load_col <= '0;
                            if (load_row == k_eff - 5'd1) begin
                                load_row <= '0;
                                load_col <= '0;
                            end else begin
                                load_row <= load_row + 5'd1;
                            end
                        end else begin
                            load_col <= load_col + 5'd1;
                        end
                    end
                end

                ST_CLEAR: begin
                    stream_cnt   <= '0;
                    stream_total <= 6'(k_eff) + 6'(m_eff) + 6'(n_eff) - 6'd1;
                end

                ST_STREAM: begin
                    stream_cnt <= stream_cnt + 6'd1;
                end

                ST_STORE: begin
                    if (!multi_k) begin
                        // Single K-tile: advance every cycle
                        if (store_col == n_eff - 5'd1) begin
                            store_col <= '0;
                            store_row <= store_row + 5'd1;
                        end else begin
                            store_col <= store_col + 5'd1;
                        end
                    end else if (first_k_tile) begin
                        // First K-tile of multi-K: write ACC, 1 cycle per element
                        if (store_col == n_eff - 5'd1) begin
                            store_col <= '0;
                            store_row <= store_row + 5'd1;
                        end else begin
                            store_col <= store_col + 5'd1;
                        end
                    end else begin
                        // Middle/last K-tiles: 2-phase read-modify-write
                        if (!store_phase) begin
                            // Phase 0 -> phase 1 (read issued, wait for data)
                            store_phase <= 1'b1;
                        end else begin
                            // Phase 1 -> phase 0, advance counters
                            store_phase <= 1'b0;
                            if (store_col == n_eff - 5'd1) begin
                                store_col <= '0;
                                store_row <= store_row + 5'd1;
                            end else begin
                                store_col <= store_col + 5'd1;
                            end
                        end
                    end
                end

                ST_TILE_NEXT: begin
                    // Advance tile counters: inner=k, middle=n, outer=m
                    if (k_tile_r < k_tiles_r - 4'd1) begin
                        k_tile_r <= k_tile_r + 4'd1;
                    end else begin
                        k_tile_r <= '0;
                        if (n_tile_r < n_tiles_r - 4'd1) begin
                            n_tile_r <= n_tile_r + 4'd1;
                        end else begin
                            n_tile_r <= '0;
                            if (m_tile_r < m_tiles_r - 4'd1) begin
                                m_tile_r <= m_tile_r + 4'd1;
                            end
                        end
                    end
                end

                default: ;
            endcase
        end
    end

endmodule

`default_nettype wire
