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
    uint off = blk * 210u;

    uint ql = readU8(off + (loc / 2u));
    uint qh = readU8(off + 128u + (loc / 4u));
    uint q = ((ql >> ((loc % 2u) * 4u)) & 0x0Fu) | (((qh >> ((loc % 4u) * 2u)) & 0x03u) << 4u);

    int sc = readI8(off + 192u + (loc / 16u));

    float d = f16tof32(readU16(off + 208u));

    outBuf.data[gid] = d * float(sc) * (float(q) - 32.0);
}
