// RMS Layer Normalization for ROCm/HIP
// out[row,i] = x[row,i] / sqrt(mean(x[row]^2) + eps) * weight[i]

extern "C" __global__ void layernorm(
    float* __restrict__ buf,
    const float* __restrict__ weight,
    unsigned int rows,
    unsigned int cols,
    float eps)
{
    unsigned int row = blockIdx.x;
    unsigned int tid = threadIdx.x;
    
    if (row >= rows)
        return;
    
    __shared__ float sharedMem[256];
    unsigned int rowOffset = row * cols;
    
    // Compute sum of squares
    float localSumSq = 0.0f;
    for (unsigned int i = tid; i < cols; i += 256)
    {
        float x = buf[rowOffset + i];
        localSumSq += x * x;
    }
    
    sharedMem[tid] = localSumSq;
    __syncthreads();
    
    // Reduce sum of squares
    for (unsigned int s = 128; s > 0; s >>= 1)
    {
        if (tid < s)
            sharedMem[tid] += sharedMem[tid + s];
        __syncthreads();
    }
    
    float rms = sqrtf(sharedMem[0] / float(cols) + eps);
    __syncthreads();
    
    // Apply normalization
    for (unsigned int i = tid; i < cols; i += 256)
    {
        buf[rowOffset + i] = (buf[rowOffset + i] / rms) * weight[i];
    }
}
