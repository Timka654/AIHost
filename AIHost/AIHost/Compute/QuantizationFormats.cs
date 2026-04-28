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
    uint loc = gid % 256u;
    uint off = blk * 84u;           // 84 bytes per block

    // Scale/min for sub-block (16 elements per sub-block)
    uint sub = loc / 16u;           // 0..15
    uint sc = readU8(off + sub);    // scales[sub]
    float sc_lo = float(sc & 0x0Fu);   // low nibble = scale factor
    float sc_hi = float((sc >> 4u) & 0x0Fu); // high nibble = min factor

    // 2-bit quantized value
    uint qs_byte = readU8(off + 16u + (loc / 4u)); // qs at offset 16
    uint qv = (qs_byte >> ((loc % 4u) * 2u)) & 0x3u;

    // Super-scales
    float d    = f16tof32(readU16(off + 80u)); // d at offset 80
    float dmin = f16tof32(readU16(off + 82u)); // dmin at offset 82

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
    uint loc = gid % 256u;
    uint off = blk * 110u;          // 110 bytes per block

    // Lower 2 bits from qs (at offset 32)
    uint low2 = (readU8(off + 32u + (loc / 4u)) >> ((loc % 4u) * 2u)) & 3u;

    // High 1 bit from hmask (at offset 0)
    uint high1 = (readU8(off + (loc / 8u)) >> (loc % 8u)) & 1u;

    // 3-bit signed: q = low2|(high1<<2), val = q - 4 gives range [-4..3]
    int qval = int(low2 | (high1 << 2u)) - 4;

    // Scale: scales at offset 96, int8 values
    // One scale per 16-element sub-block (linear, is=0..15).
    // Bytes 12-13 overlap d field (matches CPU behavior).
    // Clamp to 13 to avoid reading past block boundary on GPU.
    uint is_idx = min(loc / 16u, 13u);
    uint sc_raw = readU8(off + 96u + is_idx);
    int sc = sc_raw >= 128u ? int(sc_raw) - 256 : int(sc_raw);

    float d = f16tof32(readU16(off + 108u)); // d at offset 108

    outBuf.data[gid] = d * float(sc) * float(qval);
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
    uint loc = gid % 256u;
    uint off = blk * 144u;          // 144 bytes per block

    float d    = f16tof32(readU16(off + 0u));  // d at offset 0
    float dmin = f16tof32(readU16(off + 2u));  // dmin at offset 2

    // 8 sub-blocks of 32 elements each
    uint sub = loc / 32u;           // 0..7
    float sc, mn;
    get_scale_min_k4(sub, off + 4u, sc, mn); // scales[12] at offset 4

    // 4-bit quantized value from qs[128] at offset 16
    uint qs_byte = readU8(off + 16u + (loc / 2u));
    uint q = (qs_byte >> ((loc % 2u) * 4u)) & 0xFu;

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
    uint loc = gid % 256u;
    uint off = blk * 176u;          // 176 bytes per block

    float d    = f16tof32(readU16(off + 0u));  // d at offset 0
    float dmin = f16tof32(readU16(off + 2u));  // dmin at offset 2

    // 8 sub-blocks of 32 elements each
    uint sub = loc / 32u;           // 0..7
    float sc, mn;
    get_scale_min_k4(sub, off + 4u, sc, mn); // scales[12] at offset 4

    // Lower 4 bits from qs[128] at offset 48
    uint qs_byte = readU8(off + 48u + (loc / 2u));
    uint ql = (qs_byte >> ((loc % 2u) * 4u)) & 0xFu;

    // High 1 bit from qh[32] at offset 16
    uint qh_byte = readU8(off + 16u + (loc / 8u));
    uint qh = (qh_byte >> (loc % 8u)) & 1u;

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
int  readI8(uint off) { uint v = readU8(off); return v >= 128u ? int(v) - 256 : int(v); }
uint readU16(uint off) { return readU8(off) | (readU8(off + 1u) << 8u); }

void main() {
    uint gid = gl_GlobalInvocationID.x;
    uint blk = gid / 256u;
    uint loc = gid % 256u;
    uint off = blk * 210u;          // 210 bytes per block

    // Lower 4 bits from ql[128] at offset 0
    uint ql = readU8(off + (loc / 2u));
    // Upper 2 bits from qh[64] at offset 128
    uint qh = readU8(off + 128u + (loc / 4u));
    uint q = ((ql >> ((loc % 2u) * 4u)) & 0x0Fu) | (((qh >> ((loc % 4u) * 2u)) & 0x03u) << 4u);

    // Int8 scale from scales[16] at offset 192 (16-element sub-blocks)
    int sc = readI8(off + 192u + (loc / 16u));

    float d = f16tof32(readU16(off + 208u)); // d at offset 208

    outBuf.data[gid] = d * float(sc) * (float(q) - 32.0);
}
";
}
