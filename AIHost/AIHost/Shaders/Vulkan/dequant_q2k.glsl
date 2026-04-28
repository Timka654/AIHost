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
    uint off = blk * 84u;

    uint sub = loc / 16u;
    uint sc = readU8(off + sub);
    float sc_lo = float(sc & 0x0Fu);
    float sc_hi = float((sc >> 4u) & 0x0Fu);

    uint qs_byte = readU8(off + 16u + (loc / 4u));
    uint qv = (qs_byte >> ((loc % 4u) * 2u)) & 0x3u;

    float d    = f16tof32(readU16(off + 80u));
    float dmin = f16tof32(readU16(off + 82u));

    outBuf.data[gid] = d * sc_lo * float(qv) - dmin * sc_hi;
}
