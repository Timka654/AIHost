// Rotary Position Embedding (RoPE) for ROCm/HIP

extern "C" __global__ void rope(
    float* __restrict__ buf,
    unsigned int seq_len,
    unsigned int head_dim,
    unsigned int position,
    float theta)
{
    unsigned int pair_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (pair_idx >= head_dim / 2)
        return;
    
    float freq = 1.0f / powf(theta, float(pair_idx * 2) / float(head_dim));
    float angle = float(position) * freq;
    
    float cos_val = cosf(angle);
    float sin_val = sinf(angle);
    
    unsigned int idx1 = pair_idx * 2;
    unsigned int idx2 = pair_idx * 2 + 1;
    
    float x1 = buf[idx1];
    float x2 = buf[idx2];
    
    buf[idx1] = x1 * cos_val - x2 * sin_val;
    buf[idx2] = x1 * sin_val + x2 * cos_val;
}
