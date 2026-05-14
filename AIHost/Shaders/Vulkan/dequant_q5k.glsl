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
    uint gid = gl_GlobalInvocationID.x + offBuf.elementOffset;
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