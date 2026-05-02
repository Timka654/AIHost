namespace AIHost.Compute;

/// <summary>
/// GGUF K-quant dequantization shaders — corrected block layouts and formulas.
///
/// Block layouts (QK_K = 256 elements):
///   Q2_K:  scales[16] | qs[64]  | d(f16) | dmin(f16)         = 84 bytes
///   Q3_K:  hmask[32]  | qs[64]  | scales[12] | d(f16)         = 110 bytes
///   Q4_K:  d(f16) | dmin(f16) | scales[12] | qs[128]          = 144 bytes
///   Q5_K:  d(f16) | dmin(f16) | scales[12] | qh[32] | qs[128] = 176 bytes
///   Q6_K:  ql[128] | qh[64] | scales[16] | d(f16)             = 210 bytes
/// </summary>
public static class QuantizationFormats
{
    /// <summary>
    /// Q2_K: 256 elements, 84 bytes/block
    ///   scales[16] low-nibble = scale factor, high-nibble = min factor
    ///   qs[64]: 2 bits per element
    ///   d / dmin: fp16 super-scales
    /// </summary>
    public const string DequantizeQ2K_Correct = @"
#version 450
layout(local_size_x = 256) in;
layout(set = 0, binding = 0) readonly buffer InputBuf { uint data[]; } inBuf;
layout(set = 0, binding = 1) buffer OutputBuf { float data[]; } outBuf;

float f16tof32(uint h) {
    uint s = (h >> 15u) & 1u; uint e = (h >> 10u) & 31u; uint m = h & 1023u;
    if (e == 0u) { if (m == 0u) return 0.0; e = 1u; while ((m & 1024u) == 0u) { m <<= 1u; e -= 1u; } m &= 1023u; }
    else if (e == 31u) { return m != 0u ? uintBitsToFloat(0x7FC00000u) : (s != 0u ? uintBitsToFloat(0xFF800000u) : uintBitsToFloat(0x7F800000u)); }
    e = e - 15u + 127u; return uintBitsToFloat((s << 31u) | (e << 23u) | (m << 13u));
}
uint readU8(uint off) { return (inBuf.data[off / 4u] >> ((off % 4u) * 8u)) & 0xFFu; }
uint readU16(uint off) { return readU8(off) | (readU8(off + 1u) << 8u); }

void main() {
    uint gid = gl_GlobalInvocationID.x;
    uint blk = gid / 256u;
    uint e   = gid % 256u;
    uint off = blk * 84u;           // 84 bytes per block

    // Scale (1 per 16 elements, scales[] at offset 0, lower nibble=scale, upper=min)
    uint sub  = e / 16u;
    uint sc   = readU8(off + sub);
    float sc_lo = float(sc & 0x0Fu);
    float sc_hi = float((sc >> 4u) & 0x0Fu);
    float d    = f16tof32(readU16(off + 80u));
    float dmin = f16tof32(readU16(off + 82u));

    // ggml dequantize_row_q2_K: qs[chunk*32+l] with shift=(local/32)*2
    // qs[0..31] store bits 0-1,2-3,4-5,6-7 for element groups of 32
    uint chunk   = e / 128u;
    uint local   = e % 128u;
    uint qs_idx  = chunk * 32u + (local % 32u);
    uint shift   = (local / 32u) * 2u;
    uint qv = (readU8(off + 16u + qs_idx) >> shift) & 0x3u;

    outBuf.data[gid] = d * sc_lo * float(qv) - dmin * sc_hi;
}
";

    /// <summary>
    /// Q3_K: 256 elements, 110 bytes/block
    ///   hmask[32]: high bit of each 3-bit value
    ///   qs[64]: lower 2 bits of each value
    ///   scales[12]: int8 scale per 16-element sub-block (6 unique values via CUDA mapping)
    ///   d: fp16 super-scale
    /// </summary>
    public const string DequantizeQ3K_Correct = @"
#version 450
layout(local_size_x = 256) in;
layout(set = 0, binding = 0) readonly buffer InputBuf { uint data[]; } inBuf;
layout(set = 0, binding = 1) buffer OutputBuf { float data[]; } outBuf;

float f16tof32(uint h) {
    uint s = (h >> 15u) & 1u; uint e = (h >> 10u) & 31u; uint m = h & 1023u;
    if (e == 0u) { if (m == 0u) return 0.0; e = 1u; while ((m & 1024u) == 0u) { m <<= 1u; e -= 1u; } m &= 1023u; }
    else if (e == 31u) { return m != 0u ? uintBitsToFloat(0x7FC00000u) : (s != 0u ? uintBitsToFloat(0xFF800000u) : uintBitsToFloat(0x7F800000u)); }
    e = e - 15u + 127u; return uintBitsToFloat((s << 31u) | (e << 23u) | (m << 13u));
}
uint readU8(uint off) { return (inBuf.data[off / 4u] >> ((off % 4u) * 8u)) & 0xFFu; }
uint readU16(uint off) { return readU8(off) | (readU8(off + 1u) << 8u); }

void main() {
    uint gid = gl_GlobalInvocationID.x;
    uint blk = gid / 256u;
    uint e   = gid % 256u;
    uint off = blk * 110u;          // 110 bytes per block

    // ggml dequantize_row_q3_K: same grouped layout as Q2_K for qs and hmask
    uint chunk  = e / 128u;
    uint local  = e % 128u;
    uint qs_idx = chunk * 32u + (local % 32u);
    uint shift  = (local / 32u) * 2u;

    // Lower 2 bits from qs at offset 32
    uint low2 = (readU8(off + 32u + qs_idx) >> shift) & 3u;

    // High 1 bit from hmask at offset 0: hmask[local%32] bit (chunk*4 + local/32)
    uint hm_idx = local % 32u;
    uint hm_bit = chunk * 4u + (local / 32u);
    uint high1  = (readU8(off + hm_idx) >> hm_bit) & 1u;

    // 3-bit signed value: high1=1 → no subtract, high1=0 → subtract 4
    int qval = int(low2 | (high1 << 2u)) - 4;

    // Scale decoding: ggml applies aux[] transform to scales[12] at offset 96.
    // For scale index is = e/16, aux_group = is/4, k = is%4:
    //   aux[0][k] = (raw[k]   & 0xF) | (((raw[8+k]>>0)&3)<<4)
    //   aux[1][k] = (raw[k+4] & 0xF) | (((raw[8+k]>>2)&3)<<4)
    //   aux[2][k] = (raw[k]  >>4)    | (((raw[8+k]>>4)&3)<<4)
    //   aux[3][k] = (raw[k+4]>>4)    | (((raw[8+k]>>6)&3)<<4)
    uint is_sc  = e / 16u;
    uint k      = is_sc % 4u;
    uint ag     = is_sc / 4u;
    uint sc_off = off + 96u;
    uint tmp_b  = readU8(sc_off + 8u + k);
    uint rk, scale_byte;
    if      (ag == 0u) { rk = readU8(sc_off + k);     scale_byte = (rk & 0xFu) | (((tmp_b >> 0u) & 3u) << 4u); }
    else if (ag == 1u) { rk = readU8(sc_off + k + 4u); scale_byte = (rk & 0xFu) | (((tmp_b >> 2u) & 3u) << 4u); }
    else if (ag == 2u) { rk = readU8(sc_off + k);     scale_byte = ((rk >> 4u) & 0xFu) | (((tmp_b >> 4u) & 3u) << 4u); }
    else               { rk = readU8(sc_off + k + 4u); scale_byte = ((rk >> 4u) & 0xFu) | (((tmp_b >> 6u) & 3u) << 4u); }

    float d = f16tof32(readU16(off + 108u));
    float dl = d * float(int(scale_byte) - 32);

    outBuf.data[gid] = dl * float(qval);
}
";

    /// <summary>
    /// Q4_K: 256 elements, 144 bytes/block
    ///   d(f16) | dmin(f16) | scales[12] | qs[128]
    ///   scales[12] encodes 8 (6-bit scale, 6-bit min) pairs via K_SCALE format
    /// </summary>
    public const string DequantizeQ4K_Correct = @"
#version 450
layout(local_size_x = 256) in;
layout(set = 0, binding = 0) readonly buffer InputBuf { uint data[]; } inBuf;
layout(set = 0, binding = 1) buffer OutputBuf { float data[]; } outBuf;

float f16tof32(uint h) {
    uint s = (h >> 15u) & 1u; uint e = (h >> 10u) & 31u; uint m = h & 1023u;
    if (e == 0u) { if (m == 0u) return 0.0; e = 1u; while ((m & 1024u) == 0u) { m <<= 1u; e -= 1u; } m &= 1023u; }
    else if (e == 31u) { return m != 0u ? uintBitsToFloat(0x7FC00000u) : (s != 0u ? uintBitsToFloat(0xFF800000u) : uintBitsToFloat(0x7F800000u)); }
    e = e - 15u + 127u; return uintBitsToFloat((s << 31u) | (e << 23u) | (m << 13u));
}
uint readU8(uint off) { return (inBuf.data[off / 4u] >> ((off % 4u) * 8u)) & 0xFFu; }
uint readU16(uint off) { return readU8(off) | (readU8(off + 1u) << 8u); }

// K_SCALE: extract 6-bit scale and min for sub-block j (0..7) from scales[12] at sc_off
void get_scale_min_k4(uint j, uint sc_off, out float sc_out, out float min_out) {
    if (j < 4u) {
        sc_out  = float(readU8(sc_off + j) & 63u);
        min_out = float(readU8(sc_off + j + 4u) & 63u);
    } else {
        sc_out  = float((readU8(sc_off + j + 4u) & 0xFu) | ((readU8(sc_off + j - 4u) >> 6u) << 4u));
        min_out = float((readU8(sc_off + j + 4u) >> 4u)  | ((readU8(sc_off + j)       >> 6u) << 4u));
    }
}

void main() {
    uint gid = gl_GlobalInvocationID.x;
    uint blk = gid / 256u;
    uint e   = gid % 256u;   // element within block (0..255)
    uint off = blk * 144u;   // 144 bytes per block

    float d    = f16tof32(readU16(off + 0u));  // d at offset 0
    float dmin = f16tof32(readU16(off + 2u));  // dmin at offset 2

    // llama.cpp layout: block = 4 groups of 64 elements.
    // Each group uses qs[group*32 .. group*32+31] (32 bytes = 64 nibbles).
    //   elements [0 ..31]: lower nibbles, scale is=group*2
    //   elements [32..63]: upper nibbles, scale is=group*2+1
    uint group        = e / 64u;            // 0..3
    uint within_group = e % 64u;            // 0..63
    uint is_upper     = (within_group >= 32u) ? 1u : 0u;
    uint wg_lower     = within_group % 32u; // 0..31
    uint scale_index  = group * 2u + is_upper; // 0..7

    float sc, mn;
    get_scale_min_k4(scale_index, off + 4u, sc, mn);

    // qs[128] starts at byte 16; each group uses 32 bytes starting at group*32
    uint qs_off = off + 16u + group * 32u + wg_lower;
    uint qs_byte = readU8(qs_off);
    uint q = (qs_byte >> (is_upper * 4u)) & 0xFu;

    outBuf.data[gid] = d * sc * float(q) - dmin * mn;
}
";

    /// <summary>
    /// Q5_K: 256 elements, 176 bytes/block
    ///   d(f16) | dmin(f16) | scales[12] | qh[32] | qs[128]
    /// </summary>
    public const string DequantizeQ5K_Correct = @"
#version 450
layout(local_size_x = 256) in;
layout(set = 0, binding = 0) readonly buffer InputBuf { uint data[]; } inBuf;
layout(set = 0, binding = 1) buffer OutputBuf { float data[]; } outBuf;

float f16tof32(uint h) {
    uint s = (h >> 15u) & 1u; uint e = (h >> 10u) & 31u; uint m = h & 1023u;
    if (e == 0u) { if (m == 0u) return 0.0; e = 1u; while ((m & 1024u) == 0u) { m <<= 1u; e -= 1u; } m &= 1023u; }
    else if (e == 31u) { return m != 0u ? uintBitsToFloat(0x7FC00000u) : (s != 0u ? uintBitsToFloat(0xFF800000u) : uintBitsToFloat(0x7F800000u)); }
    e = e - 15u + 127u; return uintBitsToFloat((s << 31u) | (e << 23u) | (m << 13u));
}
uint readU8(uint off) { return (inBuf.data[off / 4u] >> ((off % 4u) * 8u)) & 0xFFu; }
uint readU16(uint off) { return readU8(off) | (readU8(off + 1u) << 8u); }

void get_scale_min_k4(uint j, uint sc_off, out float sc_out, out float min_out) {
    if (j < 4u) {
        sc_out  = float(readU8(sc_off + j) & 63u);
        min_out = float(readU8(sc_off + j + 4u) & 63u);
    } else {
        sc_out  = float((readU8(sc_off + j + 4u) & 0xFu) | ((readU8(sc_off + j - 4u) >> 6u) << 4u));
        min_out = float((readU8(sc_off + j + 4u) >> 4u)  | ((readU8(sc_off + j)       >> 6u) << 4u));
    }
}

void main() {
    uint gid = gl_GlobalInvocationID.x;
    uint blk = gid / 256u;
    uint e   = gid % 256u;
    uint off = blk * 176u;          // 176 bytes per block

    float d    = f16tof32(readU16(off + 0u));
    float dmin = f16tof32(readU16(off + 2u));

    // Scale: 8 sub-blocks of 32 elements (same index as before)
    uint sub = e / 32u;
    float sc, mn;
    get_scale_min_k4(sub, off + 4u, sc, mn);

    // ggml dequantize_row_q5_K: j groups of 64, within each 32 lower + 32 upper nibbles
    // ql[j*32+l] stores lower nibble for element j*64+l, upper nibble for j*64+32+l
    uint j        = e / 64u;
    uint local    = e % 64u;
    uint is_upper = (local >= 32u) ? 1u : 0u;
    uint l        = local % 32u;
    uint ql_idx   = j * 32u + l;

    // Lower 4 bits from qs[128] at offset 48
    uint ql = (readU8(off + 48u + ql_idx) >> (is_upper * 4u)) & 0xFu;

    // High 1 bit from qh[32] at offset 16: qh[ql_idx] bit (j*2+is_upper)
    uint qh_bit = j * 2u + is_upper;
    uint qh = (readU8(off + 16u + ql_idx) >> qh_bit) & 1u;

    uint q = ql | (qh << 4u);      // 5-bit value 0..31

    outBuf.data[gid] = d * sc * float(q) - dmin * mn;
}
";

    /// <summary>
    /// Q6_K: 256 elements, 210 bytes/block
    ///   ql[128] | qh[64] | scales[16] | d(f16)
    ///   scales are int8, d is fp16
    /// </summary>
    public const string DequantizeQ6K_Correct = @"
#version 450
layout(local_size_x = 256) in;
layout(set = 0, binding = 0) readonly buffer InputBuf { uint data[]; } inBuf;
layout(set = 0, binding = 1) buffer OutputBuf { float data[]; } outBuf;

float f16tof32(uint h) {
    uint s = (h >> 15u) & 1u; uint e = (h >> 10u) & 31u; uint m = h & 1023u;
    if (e == 0u) { if (m == 0u) return 0.0; e = 1u; while ((m & 1024u) == 0u) { m <<= 1u; e -= 1u; } m &= 1023u; }
    else if (e == 31u) { return m != 0u ? uintBitsToFloat(0x7FC00000u) : (s != 0u ? uintBitsToFloat(0xFF800000u) : uintBitsToFloat(0x7F800000u)); }
    e = e - 15u + 127u; return uintBitsToFloat((s << 31u) | (e << 23u) | (m << 13u));
}
uint readU8(uint off) { return (inBuf.data[off / 4u] >> ((off % 4u) * 8u)) & 0xFFu; }
int readI8(uint off) { uint v = readU8(off); int sv = int(v); return (v >= 128u) ? (sv - 256) : sv; }
uint readU16(uint off) { return readU8(off) | (readU8(off + 1u) << 8u); }

void main() {
    uint gid = gl_GlobalInvocationID.x;
    uint blk = gid / 256u;
    uint e   = gid % 256u;   // element within block (0..255)
    uint off = blk * 210u;   // 210 bytes per block

    // Q6_K layout (llama.cpp convention):
    // Block = 2 halves of 128 elements.
    // Each half: ql[64 bytes] stores 4x32 4-bit values; qh[32 bytes] stores 4x32 2-bit values.
    // Elements within a half are interleaved into 4 quarters of 32 each.
    uint blk_h    = e / 128u;
    uint w        = e % 128u;
    uint quarter  = w / 32u;    // 0..3
    uint wq       = w % 32u;    // 0..31 within quarter

    // ql offset: quarters 0,2 use ql[0..31], quarters 1,3 use ql[32..63] within the half
    uint ql_extra = ((quarter == 1u || quarter == 3u) ? 32u : 0u);
    uint ql_off   = off + blk_h * 64u + ql_extra + wq;
    uint is_upper = (quarter == 2u || quarter == 3u) ? 1u : 0u;
    uint lower_4  = (readU8(ql_off) >> (is_upper * 4u)) & 0xFu;

    // qh offset: all quarters use qh[wq] within the half; bit positions 0,2,4,6
    // ggml dequantize_row_q6_K: q1->bits0-1, q2->bits2-3, q3->bits4-5, q4->bits6-7
    // quarter 0 = q1 (y[l+0]), quarter 1 = q2 (y[l+32]), quarter 2 = q3 (y[l+64]), quarter 3 = q4 (y[l+96])
    uint qh_off   = off + 128u + blk_h * 32u + wq;
    uint qh_shift = quarter * 2u;  // 0, 2, 4, 6 for quarters 0..3
    uint upper_2  = (readU8(qh_off) >> qh_shift) & 0x3u;

    uint q = lower_4 | (upper_2 << 4u);  // 6-bit value

    int  sc = readI8(off + 192u + (e / 16u));  // 16 elements per scale
    float d = f16tof32(readU16(off + 208u));    // super-block scale

    outBuf.data[gid] = d * float(sc) * (float(q) - 32.0);
}
";
}



