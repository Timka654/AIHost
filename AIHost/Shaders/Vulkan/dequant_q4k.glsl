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
    uint gid = gl_GlobalInvocationID.x + offBuf.elementOffset;
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