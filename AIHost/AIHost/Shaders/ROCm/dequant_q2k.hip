extern "C" __global__ void dequant_q2k(const unsigned int* __restrict__ inBuf, float* __restrict__ outBuf, unsigned int total_elements) {
    unsigned int gid = blockDim.x * blockIdx.x + threadIdx.x;
    if (gid >= total_elements) return;
    
    unsigned int blk = gid / 256;
    unsigned int loc = gid % 256;
    unsigned int off = blk * 84;

    auto readU8 = [&](unsigned int offset) -> unsigned int {
        return (inBuf[offset / 4] >> ((offset % 4) * 8)) & 0xFF;
    };
    
    auto readU16 = [&](unsigned int offset) -> unsigned int {
        return readU8(offset) | (readU8(offset + 1) << 8);
    };
    
    auto f16tof32 = [](unsigned int h) -> float {
        unsigned int s = (h >> 15) & 1; unsigned int e = (h >> 10) & 31; unsigned int m = h & 1023;
        if (e == 0) { if (m == 0) return 0.0f; e = 1; while ((m & 1024) == 0) { m <<= 1; e -= 1; } m &= 1023; }
        else if (e == 31) { return m != 0 ? __uint_as_float(0x7FC00000) : (s != 0 ? __uint_as_float(0xFF800000) : __uint_as_float(0x7F800000)); }
        e = e - 15 + 127; return __uint_as_float((s << 31) | (e << 23) | (m << 13));
    };

    unsigned int sub = loc / 16;
    unsigned int sc = readU8(off + sub);
    float sc_lo = float(sc & 0x0F);
    float sc_hi = float((sc >> 4) & 0x0F);

    unsigned int qs_byte = readU8(off + 16 + (loc / 4));
    unsigned int qv = (qs_byte >> ((loc % 4) * 2)) & 0x3;

    float d    = f16tof32(readU16(off + 80));
    float dmin = f16tof32(readU16(off + 82));

    outBuf[gid] = d * sc_lo * float(qv) - dmin * sc_hi;
}
