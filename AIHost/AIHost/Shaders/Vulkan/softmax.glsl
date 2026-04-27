#version 450
layout(local_size_x = 256) in;

layout(set = 0, binding = 0) buffer InOutBuf { float data[]; } buf;
layout(set = 0, binding = 1) readonly buffer Params { uint size; } params;

shared float sMax[256];
shared float sSum[256];

void main() {
    uint gid = gl_GlobalInvocationID.x;
    uint lid = gl_LocalInvocationID.x;
    
    float val = (gid < params.size) ? buf.data[gid] : -1e38;
    sMax[lid] = val;
    barrier();
    
    for (uint s = 128u; s > 0u; s >>= 1u) {
        if (lid < s) sMax[lid] = max(sMax[lid], sMax[lid + s]);
        barrier();
    }
    float maxVal = sMax[0];
    
    if (gid < params.size) {
        val = exp(buf.data[gid] - maxVal);
        buf.data[gid] = val;
    } else {
        val = 0.0;
    }
    
    sSum[lid] = val;
    barrier();
    
    for (uint s = 128u; s > 0u; s >>= 1u) {
        if (lid < s) sSum[lid] += sSum[lid + s];
        barrier();
    }
    float sumVal = sSum[0];
    
    if (gid < params.size && sumVal > 0.0) {
        buf.data[gid] /= sumVal;
    }
}
