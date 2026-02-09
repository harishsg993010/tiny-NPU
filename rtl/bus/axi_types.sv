// ===========================================================================
//  AXI Type Definitions Package
//  Defines AXI4 and AXI4-Lite parameters, channel structs, burst types,
//  response codes, and a helper function for AxSIZE encoding.
//  Fully synthesizable -- no delays, no unsynthesizable constructs.
// ===========================================================================
package axi_types_pkg;

    // =======================================================================
    //  AXI4 Full Parameters
    // =======================================================================
    localparam int AXI_ADDR_W  = 32;
    localparam int AXI_DATA_W  = 128;
    localparam int AXI_STRB_W  = AXI_DATA_W / 8;   // 16
    localparam int AXI_ID_W    = 4;
    localparam int AXI_LEN_W   = 8;
    localparam int AXI_SIZE_W  = 3;
    localparam int AXI_BURST_W = 2;

    // =======================================================================
    //  AXI4-Lite Parameters
    // =======================================================================
    localparam int LITE_ADDR_W = 32;
    localparam int LITE_DATA_W = 32;
    localparam int LITE_STRB_W = LITE_DATA_W / 8;   // 4

    // =======================================================================
    //  AXI4 Burst Types
    // =======================================================================
    localparam logic [1:0] AXI_BURST_FIXED = 2'b00;
    localparam logic [1:0] AXI_BURST_INCR  = 2'b01;
    localparam logic [1:0] AXI_BURST_WRAP  = 2'b10;

    // =======================================================================
    //  AXI4 Response Codes
    // =======================================================================
    localparam logic [1:0] AXI_RESP_OKAY   = 2'b00;
    localparam logic [1:0] AXI_RESP_SLVERR = 2'b10;

    // =======================================================================
    //  AXI4 Channel Structs
    //  Note: arready / rready / awready / wready / bready are kept as
    //        separate signals on the port list (they flow in the opposite
    //        direction), so they are NOT included in these structs.
    // =======================================================================

    // -- Address-Read (AR) channel ------------------------------------------
    typedef struct packed {
        logic [AXI_ID_W-1:0]    arid;
        logic [AXI_ADDR_W-1:0]  araddr;
        logic [AXI_LEN_W-1:0]   arlen;
        logic [AXI_SIZE_W-1:0]  arsize;
        logic [AXI_BURST_W-1:0] arburst;
        logic                    arvalid;
    } axi4_ar_chan_t;

    // -- Read-Data (R) channel ----------------------------------------------
    typedef struct packed {
        logic [AXI_ID_W-1:0]   rid;
        logic [AXI_DATA_W-1:0] rdata;
        logic [1:0]            rresp;
        logic                  rlast;
        logic                  rvalid;
    } axi4_r_chan_t;

    // -- Address-Write (AW) channel -----------------------------------------
    typedef struct packed {
        logic [AXI_ID_W-1:0]    awid;
        logic [AXI_ADDR_W-1:0]  awaddr;
        logic [AXI_LEN_W-1:0]   awlen;
        logic [AXI_SIZE_W-1:0]  awsize;
        logic [AXI_BURST_W-1:0] awburst;
        logic                    awvalid;
    } axi4_aw_chan_t;

    // -- Write-Data (W) channel ---------------------------------------------
    typedef struct packed {
        logic [AXI_DATA_W-1:0] wdata;
        logic [AXI_STRB_W-1:0] wstrb;
        logic                  wlast;
        logic                  wvalid;
    } axi4_w_chan_t;

    // -- Write-Response (B) channel -----------------------------------------
    typedef struct packed {
        logic [AXI_ID_W-1:0] bid;
        logic [1:0]          bresp;
        logic                bvalid;
    } axi4_b_chan_t;

    // =======================================================================
    //  Helper Function: AXI size encoding from byte-width
    //  Returns AxSIZE value for a given data bus width in bytes.
    //  E.g. 16 bytes (128-bit bus) -> 3'b100
    // =======================================================================
    function automatic logic [2:0] axi_size_from_bytes(input int unsigned byte_width);
        case (byte_width)
            1:       return 3'b000;   //   1 byte
            2:       return 3'b001;   //   2 bytes
            4:       return 3'b010;   //   4 bytes
            8:       return 3'b011;   //   8 bytes
            16:      return 3'b100;   //  16 bytes
            32:      return 3'b101;   //  32 bytes
            64:      return 3'b110;   //  64 bytes
            128:     return 3'b111;   // 128 bytes
            default: return 3'b010;   // default to 4 bytes
        endcase
    endfunction

endpackage : axi_types_pkg
