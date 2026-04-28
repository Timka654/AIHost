#version 450
layout(local_size_x = 256) in;

layout(set = 0, binding = 0) buffer InOutBuf { float data[]; } buf;
layout(set = 0, binding = 1) readonly buffer ParamsBuffer {
    uint seq_len;
    uint head_dim;
    uint position;
    float theta;
} params;

void main() {
    uint pair_idx = gl_GlobalInvocationID.x;
    if (pair_idx >= params.head_dim / 2u) return;
    
    float freq = 1.0 / pow(params.theta, float(pair_idx * 2u) / float(params.head_dim));
    float angle = float(params.position) * freq;
    
    float cos_val = cos(angle);
    float sin_val = sin(angle);
    
    uint idx1 = pair_idx * 2u;
    uint idx2 = pair_idx * 2u + 1u;
    
    float x1 = buf.data[idx1];
    float x2 = buf.data[idx2];
    
    buf.data[idx1] = x1 * cos_val - x2 * sin_val;
    buf.data[idx2] = x1 * sin_val + x2 * cos_val;
}
