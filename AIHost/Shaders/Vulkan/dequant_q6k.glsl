#version 450
layout(local_size_x = 256) in;
layout(set = 0, binding = 0) readonly buffer InputBuf { uint data[]; } inBuf;
layout(set = 0, binding = 1) buffer OutputBuf { float data[]; } outBuf;
layout(set = 0, binding = 2) readonly buffer OffsetBuf { uint elementOffset; } offBuf;

// Fix: Subnormal F16→F32 — return directly, don't fall through
float f16tof32(uint h) {
    uint s = (h >> 15u) & 1u;
    uint eu = (h >> 10u) & 31u;
    uint m = h & 1023u;

    if (eu == 0u) {
        if (m == 0u) return 0.0;
        int e2 = 1; while ((m & 1024u) == 0u) { m <<= 1u; e2 -= 1; }
        m = (m & 1023u) << 13u;
        uint exp = uint(e2 - 15 + 127);  // final F32 exponent
        return uintBitsToFloat((s << 31u) | (exp << 23u) | m);
    }
    if (eu == 31u) {
        return m != 0u ? uintBitsToFloat(0x7FC00000u)
                       : (s != 0u ? uintBitsToFloat(0xFF800000u) : uintBitsToFloat(0x7F800000u));
    }
    eu = eu - 15u + 127u;
    return uintBitsToFloat((s << 31u) | (eu << 23u) | (m << 13u));
}
uint readU8(uint off) { return (inBuf.data[off / 4u] >> ((off % 4u) * 8u)) & 0xFFu; }
uint readU16(uint off) { return readU8(off) | (readU8(off + 1u) << 8u); }

void main() {
    uint gid = gl_GlobalInvocationID.x + offBuf.elementOffset;
    uint blk = gid / 256u;
    uint e   = gid % 256u;
    uint off = blk * 210u;

    // Q6_K interleaved half layout
    uint blk_h    = e / 128u;
    uint w        = e % 128u;
    uint quarter  = w / 32u;
    uint wq       = w % 32u;

    uint ql_extra = ((quarter == 1u || quarter == 3u) ? 32u : 0u);
    uint ql_off   = off + blk_h * 64u + ql_extra + wq;
    uint is_upper = (quarter == 2u || quarter == 3u) ? 1u : 0u;
    uint lower_4  = (readU8(ql_off) >> (is_upper * 4u)) & 0xFu;

    uint qh_off   = off + 128u + blk_h * 32u + wq;
    uint qh_shift = quarter * 2u;
    uint upper_2  = (readU8(qh_off) >> qh_shift) & 0x3u;

    uint q = lower_4 | (upper_2 << 4u);

    // Inline signed byte: sc at offset 192
    uint scv = readU8(off + 192u + (e / 16u));
    int  sc = (scv >= 128u) ? (int(scv) - 256) : int(scv);

    float d = f16tof32(readU16(off + 208u));

    outBuf.data[gid] = d * float(sc) * (float(q) - 32.0);
}
