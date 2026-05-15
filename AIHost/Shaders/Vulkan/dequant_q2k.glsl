#version 450
layout(local_size_x = 256) in;
layout(set = 0, binding = 0) readonly buffer InputBuf { uint data[]; } inBuf;
layout(set = 0, binding = 1) buffer OutputBuf { float data[]; } outBuf;
layout(set = 0, binding = 2) readonly buffer OffsetBuf { uint elementOffset; } offBuf;

// Fix: Subnormal F16→F32 — return directly, don't fall through to -15+127
float f16tof32(uint h) {
    uint s = (h >> 15u) & 1u; uint eu = (h >> 10u) & 31u; uint m = h & 1023u;
    if (eu == 0u) {
        if (m == 0u) return 0.0;
        int e2 = 1; while ((m & 1024u) == 0u) { m <<= 1u; e2 -= 1; }
        m = (m & 1023u) << 13u;
        uint exp = uint(e2 - 15 + 127);
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