#version 450
layout(local_size_x = 256) in;
layout(set = 0, binding = 0) readonly buffer InputBuf { uint data[]; } inBuf;
layout(set = 0, binding = 1) buffer OutputBuf { float data[]; } outBuf;
layout(set = 0, binding = 2) readonly buffer OffsetBuf { uint elementOffset; } offBuf;

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
    uint gid = gl_GlobalInvocationID.x + offBuf.elementOffset;
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