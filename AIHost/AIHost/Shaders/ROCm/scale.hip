// Scale kernel for ROCm/HIP
// buf[i] *= scale

extern "C" __global__ void scale(
    float* __restrict__ buf,
    unsigned int size,
    float scale)
{
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (gid >= size)
        return;
    
    buf[gid] *= scale;
}
