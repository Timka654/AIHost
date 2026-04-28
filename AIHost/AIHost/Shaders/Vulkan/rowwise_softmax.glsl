#version 450
layout(local_size_x = 256) in;

layout(set = 0, binding = 0) buffer InOutBuf { float data[]; } buf;
layout(set = 0, binding = 1) readonly buffer Params { uint rows; uint cols; } params;

shared float sharedMem[256];

void main() {
    uint row = gl_WorkGroupID.x;
    uint tid = gl_LocalInvocationID.x;
    
    if (row >= params.rows) return;
    
    uint rowOffset = row * params.cols;
    
    // Find max
    float localMax = -1e38;
    for (uint i = tid; i < params.cols; i += gl_WorkGroupSize.x) {
        localMax = max(localMax, buf.data[rowOffset + i]);
    }
    sharedMem[tid] = localMax;
    barrier();
    
    // Reduce max
    for (uint s = gl_WorkGroupSize.x / 2u; s > 0u; s >>= 1u) {
        if (tid < s) {
            sharedMem[tid] = max(sharedMem[tid], sharedMem[tid + s]);
        }
        barrier();
    }
    float rowMax = sharedMem[0];
    barrier();
    
    // Compute exp and sum
    float localSum = 0.0;
    for (uint i = tid; i < params.cols; i += gl_WorkGroupSize.x) {
        float val = exp(buf.data[rowOffset + i] - rowMax);
        buf.data[rowOffset + i] = val;
        localSum += val;
    }
    sharedMem[tid] = localSum;
    barrier();
    
    // Reduce sum
    for (uint s = gl_WorkGroupSize.x / 2u; s > 0u; s >>= 1u) {
        if (tid < s) {
            sharedMem[tid] += sharedMem[tid + s];
        }
        barrier();
    }
    float rowSum = sharedMem[0];
    barrier();
    
    // Normalize
    for (uint i = tid; i < params.cols; i += gl_WorkGroupSize.x) {
        buf.data[rowOffset + i] /= rowSum;
    }
}
