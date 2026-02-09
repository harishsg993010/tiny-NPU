// =============================================================================
// gelu_lut.sv - GELU activation function via 256-entry ROM lookup table
// Input:  x[7:0] signed int8
// Output: gelu_out[7:0] signed int8
// Approximation: GELU(x) ~ x * sigmoid(1.702 * x)
// Scale: int8 maps to float via /32, so int8 range [-128,127] -> [-4.0, 3.97]
// LUT[i] = round(GELU(signed_i / 32.0) * 32.0) clamped to [-128, 127]
// Pipeline: 1-cycle registered output
// =============================================================================
import npu_pkg::*;
import fixed_pkg::*;

module gelu_lut (
    input  logic       clk,
    input  logic [7:0] addr,       // treated as signed int8 index
    output logic [7:0] data_out    // signed int8 GELU result
);

    // 256-entry ROM - inferred as block RAM
    (* rom_style = "block" *)
    logic [7:0] rom [0:255];

    // ----------------------------------------------------------------
    // ROM initialization
    // GELU(x) ~ x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3)))
    // Simplified: GELU(x) ~ x * sigmoid(1.702 * x)
    // Scale: x_float = signed_i / 32.0
    // LUT[i] = clamp(round(GELU(x_float) * 32.0), -128, 127)
    //
    // Key values (signed int8 index -> GELU output):
    //   0 -> 0           (GELU(0) = 0)
    //  32 -> 27          (GELU(1.0) ~ 0.841 -> 0.841*32 = 26.9)
    //  64 -> 62          (GELU(2.0) ~ 1.955 -> 62.6)
    //  96 -> 95          (GELU(3.0) ~ 2.996 -> 95.9)
    // -32 -> -5          (GELU(-1.0) ~ -0.159 -> -5.1)
    // -64 -> -1          (GELU(-2.0) ~ -0.045 -> -1.4)
    // -128-> 0           (GELU(-4.0) ~ 0.0 -> 0)
    //
    // Placeholder values - regenerate with make_lut.py for full precision
    // ----------------------------------------------------------------
    initial begin
        // Positive indices 0..127 (signed 0 to +127)
        // x_float = i/32.0, range [0, 3.97]
        rom[  0] = 8'd0;     // GELU(0) = 0
        rom[  1] = 8'd1;     // GELU(0.03125) ~ 0.016
        rom[  2] = 8'd1;
        rom[  3] = 8'd2;
        rom[  4] = 8'd2;     // GELU(0.125) ~ 0.066
        rom[  5] = 8'd3;
        rom[  6] = 8'd3;
        rom[  7] = 8'd4;
        rom[  8] = 8'd5;     // GELU(0.25) ~ 0.140
        rom[  9] = 8'd5;
        rom[ 10] = 8'd6;
        rom[ 11] = 8'd7;
        rom[ 12] = 8'd7;
        rom[ 13] = 8'd8;
        rom[ 14] = 8'd9;
        rom[ 15] = 8'd10;
        rom[ 16] = 8'd11;    // GELU(0.5) ~ 0.346
        rom[ 17] = 8'd12;
        rom[ 18] = 8'd13;
        rom[ 19] = 8'd14;
        rom[ 20] = 8'd15;
        rom[ 21] = 8'd16;
        rom[ 22] = 8'd17;
        rom[ 23] = 8'd18;
        rom[ 24] = 8'd19;    // GELU(0.75) ~ 0.558
        rom[ 25] = 8'd20;
        rom[ 26] = 8'd21;
        rom[ 27] = 8'd22;
        rom[ 28] = 8'd23;
        rom[ 29] = 8'd24;
        rom[ 30] = 8'd25;
        rom[ 31] = 8'd26;
        rom[ 32] = 8'd27;    // GELU(1.0) ~ 0.841
        rom[ 33] = 8'd28;
        rom[ 34] = 8'd29;
        rom[ 35] = 8'd30;
        rom[ 36] = 8'd32;
        rom[ 37] = 8'd33;
        rom[ 38] = 8'd34;
        rom[ 39] = 8'd35;
        rom[ 40] = 8'd36;
        rom[ 41] = 8'd37;
        rom[ 42] = 8'd39;
        rom[ 43] = 8'd40;
        rom[ 44] = 8'd41;
        rom[ 45] = 8'd42;
        rom[ 46] = 8'd44;
        rom[ 47] = 8'd45;
        rom[ 48] = 8'd46;    // GELU(1.5) ~ 1.399
        rom[ 49] = 8'd48;
        rom[ 50] = 8'd49;
        rom[ 51] = 8'd50;
        rom[ 52] = 8'd52;
        rom[ 53] = 8'd53;
        rom[ 54] = 8'd54;
        rom[ 55] = 8'd56;
        rom[ 56] = 8'd57;
        rom[ 57] = 8'd58;
        rom[ 58] = 8'd60;
        rom[ 59] = 8'd61;
        rom[ 60] = 8'd62;
        rom[ 61] = 8'd63;
        rom[ 62] = 8'd63;
        rom[ 63] = 8'd63;
        rom[ 64] = 8'd62;    // GELU(2.0) ~ 1.955
        rom[ 65] = 8'd64;
        rom[ 66] = 8'd65;
        rom[ 67] = 8'd66;
        rom[ 68] = 8'd67;
        rom[ 69] = 8'd69;
        rom[ 70] = 8'd70;
        rom[ 71] = 8'd71;
        rom[ 72] = 8'd72;
        rom[ 73] = 8'd73;
        rom[ 74] = 8'd74;
        rom[ 75] = 8'd75;
        rom[ 76] = 8'd76;
        rom[ 77] = 8'd77;
        rom[ 78] = 8'd79;
        rom[ 79] = 8'd80;
        rom[ 80] = 8'd81;
        rom[ 81] = 8'd82;
        rom[ 82] = 8'd83;
        rom[ 83] = 8'd84;
        rom[ 84] = 8'd85;
        rom[ 85] = 8'd86;
        rom[ 86] = 8'd87;
        rom[ 87] = 8'd88;
        rom[ 88] = 8'd89;
        rom[ 89] = 8'd90;
        rom[ 90] = 8'd91;
        rom[ 91] = 8'd92;
        rom[ 92] = 8'd93;
        rom[ 93] = 8'd94;
        rom[ 94] = 8'd95;
        rom[ 95] = 8'd95;
        rom[ 96] = 8'd96;    // GELU(3.0) ~ 2.996
        rom[ 97] = 8'd97;
        rom[ 98] = 8'd98;
        rom[ 99] = 8'd99;
        rom[100] = 8'd100;
        rom[101] = 8'd101;
        rom[102] = 8'd102;
        rom[103] = 8'd103;
        rom[104] = 8'd104;
        rom[105] = 8'd105;
        rom[106] = 8'd106;
        rom[107] = 8'd107;
        rom[108] = 8'd108;
        rom[109] = 8'd109;
        rom[110] = 8'd110;
        rom[111] = 8'd111;
        rom[112] = 8'd112;
        rom[113] = 8'd113;
        rom[114] = 8'd114;
        rom[115] = 8'd115;
        rom[116] = 8'd116;
        rom[117] = 8'd117;
        rom[118] = 8'd118;
        rom[119] = 8'd119;
        rom[120] = 8'd120;
        rom[121] = 8'd121;
        rom[122] = 8'd122;
        rom[123] = 8'd123;
        rom[124] = 8'd124;
        rom[125] = 8'd125;
        rom[126] = 8'd126;
        rom[127] = 8'd127;

        // Negative indices 128..255 (signed -128 to -1)
        // x_float = (i-256)/32.0, range [-4.0, -0.03125]
        // GELU is approximately 0 for very negative inputs
        rom[128] = 8'd0;     // GELU(-4.0) ~ 0.0
        rom[129] = 8'd0;
        rom[130] = 8'd0;
        rom[131] = 8'd0;
        rom[132] = 8'd0;
        rom[133] = 8'd0;
        rom[134] = 8'd0;
        rom[135] = 8'd0;
        rom[136] = 8'd0;
        rom[137] = 8'd0;
        rom[138] = 8'd0;
        rom[139] = 8'd0;
        rom[140] = 8'd0;
        rom[141] = 8'd0;
        rom[142] = 8'd0;
        rom[143] = 8'd0;
        rom[144] = 8'd0;     // GELU(-3.5) ~ -0.001
        rom[145] = 8'd0;
        rom[146] = 8'd0;
        rom[147] = 8'd0;
        rom[148] = 8'd0;
        rom[149] = 8'd0;
        rom[150] = 8'd0;
        rom[151] = 8'd0;
        rom[152] = 8'd0;
        rom[153] = 8'd0;
        rom[154] = 8'd0;
        rom[155] = 8'd0;
        rom[156] = 8'd0;
        rom[157] = 8'd0;
        rom[158] = 8'd0;
        rom[159] = 8'd0;
        rom[160] = 8'd0;     // GELU(-3.0) ~ -0.004
        rom[161] = 8'd0;
        rom[162] = 8'd0;
        rom[163] = 8'd0;
        rom[164] = 8'd0;
        rom[165] = 8'd0;
        rom[166] = 8'd0;
        rom[167] = 8'd0;
        rom[168] = 8'd0;
        rom[169] = 8'd0;
        rom[170] = 8'd0;
        rom[171] = 8'd0;
        rom[172] = 8'd0;
        rom[173] = 8'd0;
        rom[174] = 8'd0;
        rom[175] = 8'd0;
        rom[176] = 8'd0;
        rom[177] = 8'd0;
        rom[178] = 8'd0;
        rom[179] = 8'd0;
        rom[180] = 8'd0;
        rom[181] = 8'd0;
        rom[182] = 8'd0;
        rom[183] = 8'd0;
        rom[184] = 8'd0;
        rom[185] = 8'd0;
        rom[186] = 8'd0;
        rom[187] = 8'd0;
        rom[188] = 8'd0;
        rom[189] = 8'd0;
        rom[190] = 8'd0;
        rom[191] = 8'd0;
        rom[192] = 8'hFF;    // GELU(-2.0) ~ -0.045 -> round(-1.4) = -1
        rom[193] = 8'hFF;
        rom[194] = 8'hFF;
        rom[195] = 8'hFF;
        rom[196] = 8'hFF;
        rom[197] = 8'hFF;
        rom[198] = 8'hFF;
        rom[199] = 8'hFE;    // -2
        rom[200] = 8'hFE;
        rom[201] = 8'hFE;
        rom[202] = 8'hFE;
        rom[203] = 8'hFE;
        rom[204] = 8'hFD;    // -3
        rom[205] = 8'hFD;
        rom[206] = 8'hFD;
        rom[207] = 8'hFD;
        rom[208] = 8'hFC;    // -4
        rom[209] = 8'hFC;
        rom[210] = 8'hFC;
        rom[211] = 8'hFB;    // -5
        rom[212] = 8'hFB;
        rom[213] = 8'hFB;
        rom[214] = 8'hFB;
        rom[215] = 8'hFB;
        rom[216] = 8'hFB;
        rom[217] = 8'hFA;    // -6
        rom[218] = 8'hFA;
        rom[219] = 8'hFA;
        rom[220] = 8'hFA;
        rom[221] = 8'hFA;
        rom[222] = 8'hFA;
        rom[223] = 8'hFA;
        rom[224] = 8'hFB;    // GELU(-1.0) ~ -0.159 -> round(-5.1) = -5
        rom[225] = 8'hFB;
        rom[226] = 8'hFB;
        rom[227] = 8'hFB;
        rom[228] = 8'hFC;
        rom[229] = 8'hFC;
        rom[230] = 8'hFC;
        rom[231] = 8'hFC;
        rom[232] = 8'hFC;
        rom[233] = 8'hFD;
        rom[234] = 8'hFD;
        rom[235] = 8'hFD;
        rom[236] = 8'hFD;
        rom[237] = 8'hFE;
        rom[238] = 8'hFE;
        rom[239] = 8'hFE;
        rom[240] = 8'hFE;    // GELU(-0.5) ~ -0.154 -> round(-4.9) = -5
        rom[241] = 8'hFE;
        rom[242] = 8'hFF;
        rom[243] = 8'hFF;
        rom[244] = 8'hFF;
        rom[245] = 8'hFF;
        rom[246] = 8'hFF;
        rom[247] = 8'hFF;
        rom[248] = 8'hFF;    // GELU(-0.25) ~ -0.077
        rom[249] = 8'hFF;
        rom[250] = 8'hFF;
        rom[251] = 8'h00;
        rom[252] = 8'h00;
        rom[253] = 8'h00;
        rom[254] = 8'h00;
        rom[255] = 8'h00;    // GELU(-0.03125) ~ -0.015 -> rounds to 0
    end

    // ----------------------------------------------------------------
    // Registered read (1-cycle latency)
    // ----------------------------------------------------------------
    always_ff @(posedge clk) begin
        data_out <= rom[addr];
    end

endmodule
