#version 450
layout(local_size_x = 256) in;

layout(set = 0, binding = 0) buffer InOutBuf { float data[]; } buf;
layout(set = 0, binding = 1) readonly buffer WeightBuf { float data[]; } weight;
layout(set = 0, binding = 2) readonly buffer Params { 
    uint rows; 
    uint cols; 
    float eps; 
} params;

shared float sharedMem[256];

void main() {
    uint row = gl_WorkGroupID.x;
    uint tid = gl_LocalInvocationID.x;
    
    if (row >= params.rows) return;
    
    uint rowOffset = row * params.cols;
    
    // Compute sum of squares
    float localSumSq = 0.0;
    for (uint i = tid; i < params.cols; i += 256u) {
        float x = buf.data[rowOffset + i];
        localSumSq += x * x;
    }
    
    sharedMem[tid] = localSumSq;
    barrier();
    
    // Reduce sum of squares
    for (uint s = 128u; s > 0u; s >>= 1u) {
        if (tid < s) sharedMem[tid] += sharedMem[tid + s];
        barrier();
    }
    
    float rms = sqrt(sharedMem[0] / float(params.cols) + params.eps);
    barrier();
    
    // Apply normalization
    for (uint i = tid; i < params.cols; i += 256u) {
        buf.data[rowOffset + i] = (buf.data[rowOffset + i] / rms) * weight.data[i];
    }
}
