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
    uint off = blk * 110u;

    uint chunk  = e / 128u;
    uint local  = e % 128u;
    uint qs_idx = chunk * 32u + (local % 32u);
    uint shift  = (local / 32u) * 2u;

    uint low2 = (readU8(off + 32u + qs_idx) >> shift) & 3u;

    uint hm_idx = local % 32u;
    uint hm_bit = chunk * 4u + (local / 32u);
    uint high1  = (readU8(off + hm_idx) >> hm_bit) & 1u;

    int qval = int(low2 | (high1 << 2u)) - 4;

    uint is_sc  = e / 16u;
    uint k      = is_sc % 4u;
    uint ag     = is_sc / 4u;
    uint sc_off = off + 96u;
    uint tmp_b  = readU8(sc_off + 8u + k);
    uint rk;
    uint scale_byte;
    if (ag == 0u) {
        rk = readU8(sc_off + k);
        scale_byte = (rk & 0xFu) | (((tmp_b >> 0u) & 3u) << 4u);
    } else if (ag == 1u) {
        rk = readU8(sc_off + k + 4u);
        scale_byte = (rk & 0xFu) | (((tmp_b >> 2u) & 3u) << 4u);
    } else if (ag == 2u) {
        rk = readU8(sc_off + k);
        scale_byte = ((rk >> 4u) & 0xFu) | (((tmp_b >> 4u) & 3u) << 4u);
    } else {
        rk = readU8(sc_off + k + 4u);
        scale_byte = ((rk >> 4u) & 0xFu) | (((tmp_b >> 6u) & 3u) << 4u);
    }

    float d = f16tof32(readU16(off + 108u));
    float dl = d * float(int(scale_byte) - 32);

    outBuf.data[gid] = dl * float(qval);
}