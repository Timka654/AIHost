extern "C" __global__ void dequant_q3k(const unsigned int* __restrict__ inBuf, float* __restrict__ outBuf, unsigned int total_elements) {
    unsigned int gid = blockDim.x * blockIdx.x + threadIdx.x;
    if (gid >= total_elements) return;
    
    unsigned int blk = gid / 256;
    unsigned int loc = gid % 256;
    unsigned int off = blk * 110;

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

    unsigned int low2 = (readU8(off + 32 + (loc / 4)) >> ((loc % 4) * 2)) & 3;
    unsigned int high1 = (readU8(off + (loc / 8)) >> (loc % 8)) & 1;
    int qval = int(low2 | (high1 << 2)) - 4;

    unsigned int ip   = loc / 128;
    unsigned int ib16 = (loc % 128) / 16;
    unsigned int is_idx = 2 * ip + ib16 / 2;
    unsigned int sc_raw = readU8(off + 96 + is_idx);
    int sc = sc_raw >= 128 ? int(sc_raw) - 256 : int(sc_raw);

    float d = f16tof32(readU16(off + 108));

    outBuf[gid] = d * float(sc) * float(qval);
}
