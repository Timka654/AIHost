// Element-wise multiplication kernel for ROCm/HIP
// C[i] = A[i] * B[i]

extern "C" __global__ void elementwise_mul(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    unsigned int size)
{
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (gid >= size)
        return;

    C[gid] = A[gid] * B[gid];
}
