// SiLU (Swish) activation kernel for ROCm/HIP
// SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))

extern "C" __global__ void silu(
    float* __restrict__ buf,
    unsigned int size)
{
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (gid >= size)
        return;

    float x = buf[gid];
    float sigmoid = 1.0f / (1.0f + expf(-x));
    buf[gid] = x * sigmoid;
}
