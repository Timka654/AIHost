extern "C" __global__ void dequant_q4k(const unsigned int* __restrict__ inBuf, float* __restrict__ outBuf, unsigned int total_elements) {
    unsigned int gid = blockDim.x * blockIdx.x + threadIdx.x;
    if (gid >= total_elements) return;
    
    unsigned int blk = gid / 256;
    unsigned int loc = gid % 256;
    unsigned int off = blk * 144;

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
    
    auto get_scale_min_k4 = [&](unsigned int j, unsigned int sc_off, float& sc_out, float& min_out) {
        if (j < 4) {
            sc_out  = float(readU8(sc_off + j) & 63);
            min_out = float(readU8(sc_off + j + 4) & 63);
        } else {
            sc_out  = float((readU8(sc_off + j + 4) & 0x0F) | ((readU8(sc_off + j - 4) >> 6) << 4));
            min_out = float((readU8(sc_off + j + 4) >> 4)  | ((readU8(sc_off + j)       >> 6) << 4));
        }
    };

    float d    = f16tof32(readU16(off + 0));
    float dmin = f16tof32(readU16(off + 2));

    unsigned int sub = loc / 32;
    float sc, mn;
    get_scale_min_k4(sub, off + 4, sc, mn);

    unsigned int qs_byte = readU8(off + 16 + (loc / 2));
    unsigned int q = (qs_byte >> ((loc % 2) * 4)) & 0x0F;

    outBuf[gid] = d * sc * float(q) - dmin * mn;
}
