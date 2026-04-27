namespace AIHost.Compute;

/// <summary>
/// Реализация GGUF K-quants деквантизации на GPU
/// </summary>
public static class QuantizationFormats
{
    /// <summary>
    /// Q6_K: 256 элементов, 210 байт на блок
    /// </summary>
    public const string DequantizeQ6K_Correct = @"
#version 450
layout(local_size_x = 256) in;
layout(set = 0, binding = 0) readonly buffer InputBuf { uint data[]; } inBuf;
layout(set = 0, binding = 1) buffer OutputBuf { float data[]; } outBuf;

float f16tof32(uint h) {
    uint s = (h >> 15u) & 1u;
    uint e = (h >> 10u) & 31u;
    uint m = h & 1023u;
    if (e == 0u) {
        if (m == 0u) return s != 0u ? -0.0 : 0.0;
        e = 1u;
        while ((m & 1024u) == 0u) { m <<= 1u; e -= 1u; }
        m &= 1023u;
    } else if (e == 31u) {
        return m != 0u ? uintBitsToFloat(0x7FC00000u) : 
               (s != 0u ? uintBitsToFloat(0xFF800000u) : uintBitsToFloat(0x7F800000u));
    }
    e = e - 15u + 127u;
    return uintBitsToFloat((s << 31u) | (e << 23u) | (m << 13u));
}

uint readU8(uint off) {
    return (inBuf.data[off / 4u] >> ((off % 4u) * 8u)) & 0xFFu;
}

int readI8(uint off) {
    uint v = readU8(off);
    return int(v) - (int(v & 0x80u) << 1);
}

uint readU16(uint off) {
    return readU8(off) | (readU8(off + 1u) << 8u);
}

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
";

    /// <summary>
    /// Q2_K: 256 элементов, 82 байта на блок
    /// </summary>
    public const string DequantizeQ2K_Correct = @"
#version 450
layout(local_size_x = 256) in;
layout(set = 0, binding = 0) readonly buffer InputBuf { uint data[]; } inBuf;
layout(set = 0, binding = 1) buffer OutputBuf { float data[]; } outBuf;

float f16tof32(uint h) {
    uint s = (h >> 15u) & 1u;
    uint e = (h >> 10u) & 31u;
    uint m = h & 1023u;
    if (e == 0u) {
        if (m == 0u) return s != 0u ? -0.0 : 0.0;
        e = 1u;
        while ((m & 1024u) == 0u) { m <<= 1u; e -= 1u; }
        m &= 1023u;
    } else if (e == 31u) {
        return m != 0u ? uintBitsToFloat(0x7FC00000u) : 
               (s != 0u ? uintBitsToFloat(0xFF800000u) : uintBitsToFloat(0x7F800000u));
    }
    e = e - 15u + 127u;
    return uintBitsToFloat((s << 31u) | (e << 23u) | (m << 13u));
}

uint readU8(uint off) {
    return (inBuf.data[off / 4u] >> ((off % 4u) * 8u)) & 0xFFu;
}

uint readU16(uint off) {
    return readU8(off) | (readU8(off + 1u) << 8u);
}

void main() {
    uint gid = gl_GlobalInvocationID.x;
    uint blk = gid / 256u;
    uint loc = gid % 256u;
    uint off = blk * 82u;
    
    uint q = readU8(off + 32u + (loc / 4u));
    uint qv = (q >> ((loc % 4u) * 2u)) & 0x03u;
    float d = f16tof32(readU16(off + 96u));
    
    outBuf.data[gid] = d * float(qv);
}
";

    /// <summary>
    /// Q3_K: 256 элементов, ~110 байт на блок
    /// </summary>
    public const string DequantizeQ3K_Correct = @"
#version 450
layout(local_size_x = 256) in;
layout(set = 0, binding = 0) readonly buffer InputBuf { uint data[]; } inBuf;
layout(set = 0, binding = 1) buffer OutputBuf { float data[]; } outBuf;

float f16tof32(uint h) {
    uint s = (h >> 15u) & 1u;
    uint e = (h >> 10u) & 31u;
    uint m = h & 1023u;
    if (e == 0u) {
        if (m == 0u) return s != 0u ? -0.0 : 0.0;
        e = 1u;
        while ((m & 1024u) == 0u) { m <<= 1u; e -= 1u; }
        m &= 1023u;
    } else if (e == 31u) {
        return m != 0u ? uintBitsToFloat(0x7FC00000u) : 
               (s != 0u ? uintBitsToFloat(0xFF800000u) : uintBitsToFloat(0x7F800000u));
    }
    e = e - 15u + 127u;
    return uintBitsToFloat((s << 31u) | (e << 23u) | (m << 13u));
}

uint readU8(uint off) {
    return (inBuf.data[off / 4u] >> ((off % 4u) * 8u)) & 0xFFu;
}

uint readU16(uint off) {
    return readU8(off) | (readU8(off + 1u) << 8u);
}

void main() {
    uint gid = gl_GlobalInvocationID.x;
    uint blk = gid / 256u;
    uint loc = gid % 256u;
    uint off = blk * 110u;
    
    uint bi = loc * 3u / 8u;
    uint bo = (loc * 3u) % 8u;
    uint q = readU8(off + 32u + bi);
    if (bo > 5u) q |= readU8(off + 33u + bi) << 8u;
    uint qv = (q >> bo) & 0x07u;
    
    float d = f16tof32(readU16(off + 128u));
    outBuf.data[gid] = d * (float(qv) - 4.0);
}
";

    /// <summary>
    /// Q4_K: 256 элементов, 144 байта на блок
    /// </summary>
    public const string DequantizeQ4K_Correct = @"
#version 450
layout(local_size_x = 256) in;
layout(set = 0, binding = 0) readonly buffer InputBuf { uint data[]; } inBuf;
layout(set = 0, binding = 1) buffer OutputBuf { float data[]; } outBuf;

float f16tof32(uint h) {
    uint s = (h >> 15u) & 1u;
    uint e = (h >> 10u) & 31u;
    uint m = h & 1023u;
    if (e == 0u) {
        if (m == 0u) return s != 0u ? -0.0 : 0.0;
        e = 1u;
        while ((m & 1024u) == 0u) { m <<= 1u; e -= 1u; }
        m &= 1023u;
    } else if (e == 31u) {
        return m != 0u ? uintBitsToFloat(0x7FC00000u) : 
               (s != 0u ? uintBitsToFloat(0xFF800000u) : uintBitsToFloat(0x7F800000u));
    }
    e = e - 15u + 127u;
    return uintBitsToFloat((s << 31u) | (e << 23u) | (m << 13u));
}

uint readU8(uint off) {
    return (inBuf.data[off / 4u] >> ((off % 4u) * 8u)) & 0xFFu;
}

int readI8(uint off) {
    uint v = readU8(off);
    return int(v) - (int(v & 0x80u) << 1);
}

uint readU16(uint off) {
    return readU8(off) | (readU8(off + 1u) << 8u);
}

void main() {
    uint gid = gl_GlobalInvocationID.x;
    uint blk = gid / 256u;
    uint loc = gid % 256u;
    uint off = blk * 144u;
    
    uint ql = readU8(off + (loc / 2u));
    uint q = (ql >> ((loc % 2u) * 4u)) & 0x0Fu;
    
    int sc = readI8(off + 128u + (loc / 16u));
    float d = f16tof32(readU16(off + 142u));
    
    outBuf.data[gid] = d * float(sc) * (float(q) - 8.0);
}
";

    /// <summary>
    /// Q5_K: 256 элементов, 176 байт на блок
    /// </summary>
    public const string DequantizeQ5K_Correct = @"
#version 450
layout(local_size_x = 256) in;
layout(set = 0, binding = 0) readonly buffer InputBuf { uint data[]; } inBuf;
layout(set = 0, binding = 1) buffer OutputBuf { float data[]; } outBuf;

float f16tof32(uint h) {
    uint s = (h >> 15u) & 1u;
    uint e = (h >> 10u) & 31u;
    uint m = h & 1023u;
    if (e == 0u) {
        if (m == 0u) return s != 0u ? -0.0 : 0.0;
        e = 1u;
        while ((m & 1024u) == 0u) { m <<= 1u; e -= 1u; }
        m &= 1023u;
    } else if (e == 31u) {
        return m != 0u ? uintBitsToFloat(0x7FC00000u) : 
               (s != 0u ? uintBitsToFloat(0xFF800000u) : uintBitsToFloat(0x7F800000u));
    }
    e = e - 15u + 127u;
    return uintBitsToFloat((s << 31u) | (e << 23u) | (m << 13u));
}

uint readU8(uint off) {
    return (inBuf.data[off / 4u] >> ((off % 4u) * 8u)) & 0xFFu;
}

int readI8(uint off) {
    uint v = readU8(off);
    return int(v) - (int(v & 0x80u) << 1);
}

uint readU16(uint off) {
    return readU8(off) | (readU8(off + 1u) << 8u);
}

void main() {
    uint gid = gl_GlobalInvocationID.x;
    uint blk = gid / 256u;
    uint loc = gid % 256u;
    uint off = blk * 176u;
    
    uint ql = readU8(off + (loc / 2u));
    uint qh = readU8(off + 128u + (loc / 8u));
    uint q = ((ql >> ((loc % 2u) * 4u)) & 0x0Fu) | (((qh >> (loc % 8u)) & 0x01u) << 4u);
    
    int sc = readI8(off + 160u + (loc / 16u));
    float d = f16tof32(readU16(off + 174u));
    
    outBuf.data[gid] = d * float(sc) * (float(q) - 16.0);
}
";
}
